# Device Memory 与 State Store

[`DeviceBufferPool`](../../serving/device_pool.py) 统一拥有一个 `ChipWorker` 上的 DeviceTensor allocation；[`DeviceStateStore`](../../serving/device_state_store.py) 在该 pool 上用 current/next 双缓冲维护逐层可变 state。

## 模块定位

Device pool 负责分配、exact-match 复用、H2D/D2H、释放和统计，但不关闭 `ChipWorker`。DeviceRuntime 拥有 pool，并在关闭 pool 后再关闭 worker。

State store 只管理 `STATE` category 的 persistent leases。它消费 [`LayerStateSchema`](../../serving/state.py)，向 runner/runtime 提供 Kernel input/output 名到 DeviceTensor 的映射，不参与普通 intermediate 或 staging 复用。

## Allocation Category

| Category | 分配模式 | 典型生命周期 |
|---|---|---|
| `FIXED_WEIGHT` | Persistent | 权重首次 materialize 至 runtime 关闭 |
| `STATE` | Persistent | state prepare 至 state store 关闭 |
| `ACTIVE_UPLOAD` | Reusable | 当前 step 内 Host aux/input 上传 |
| `INTERMEDIATE` | Reusable | 相邻 Kernel 之间 |
| `SCRATCH` | Reusable | 单次 Kernel dispatch |
| `STAGING_ROUTED` | Reusable | prefill routed-expert dispatch，可选跨 step 保留 allocation |
| `STAGING_SELECTED` | Reusable | decode selected-expert dispatch，跨 decode step 复用 allocation |

Persistent allocation 不进入 free list；reusable allocation 必须先 `release()` 归还 free list，之后才能再次 acquire 或 `free()`。

## DeviceLease

`DeviceLease` 是 pool 为一个 allocation 生成的不透明凭据，保存 tensor、category、shape、dtype、nbytes、reuse key、pool token 和 allocation id。

Pool token 阻止 lease 跨 pool 使用；allocation id 和对象 identity 用于检测重复 release、重复 free 或已释放 lease。调用方不应自行构造或修改 lease。

## 分配与复用

### Persistent

`allocate_persistent()` 只接受 `FIXED_WEIGHT` 或 `STATE`。每次调用创建独立 DeviceTensor；可选 Host `init` 直接传给 `worker.alloc_tensor()`，并计入 H2D bytes。

### Reusable

`acquire()` 的 free-list key 是：

```text
(category, reuse_key, shape, dtype)
```

四项完全相同时才复用。命中 free list 且提供新 `init` 时，pool 通过 `copy_to()` 覆盖旧内容；未命中时创建新 allocation。Category 或 slot 相同但 shape/dtype 不同不会 alias。

`release()` 只改变 in-use 状态并加入 free list，不释放显存。`free()` 才调用 `worker.free_tensor()`；对仍在使用的 reusable lease 直接 free 会失败。

## Host 与 Device 拷贝

`copy_to()` 和 `copy_from()` 都要求 exact shape 和 dtype：

- Host 输入必须位于 CPU；非连续输入在 H2D 前转为 contiguous。
- 调用方提供的 D2H 输出必须已经 contiguous。
- Worker copy 使用 lease 的 `nbytes` 和 pool 的 `worker_id`。
- H2D/D2H 逻辑字节数累计到 pool stats。

## Pool Stats

`DevicePoolStats` 是即时快照：

| 字段 | 含义 |
|---|---|
| `allocation_count` | 当前仍由 pool 记录的 allocation 数 |
| `in_use_count` | 当前未归还 free list 的 allocation 数 |
| `alloc_count` / `free_count` | 生命周期累计实际 alloc/free 次数 |
| `reuse_count` | free-list 命中次数 |
| `h2d_bytes` / `d2h_bytes` | 生命周期累计拷贝字节 |
| `current_bytes` | 当前尚未真正 free 的总字节 |
| `active_bytes` | 当前 in-use allocation 字节 |
| `peak_bytes` | 历史峰值 current bytes |
| `category_bytes` | 当前 allocation 按 category 的字节分布，包含 free-list idle buffers |

因此 `active_bytes` 与 `current_bytes` 不同：已 release 但仍留在 free list 的 staging/intermediate 计入 current 和 category bytes，不计入 active bytes。

## State 双缓冲

`DeviceStateStore.prepare(schemas)` 为每个 `StateTensorSpec` 分配两个独立 persistent buffers：

1. `current` 使用 `spec.create_tensor()` 初始化。
2. `next` 不初始化，作为下一次 Kernel output 目标。
3. `inputs(layer_id)` 按 `input_name` 返回 current tensors。
4. `outputs(layer_id)` 按 `output_name` 返回 next tensors。
5. `commit(layer_id, outputs)` 验证所有 output 都是预绑定 next tensor，再交换 current 与 next leases。

Commit 先验证全部 outputs，再统一 swap；缺少任何 state output 或绑定了错误 tensor 时不发生部分交换。多次 commit 只在相同两块设备内存之间交替，不重新分配或复制 state。

## 数据位置与生命周期

| 资源 | 位置 | 所有者 | 释放 |
|---|---|---|---|
| Pool allocation records/free lists | Host metadata | `DeviceBufferPool` | `free()` / `close()` |
| DeviceTensor | NPU | `DeviceBufferPool` | worker `free_tensor()` |
| State current/next leases | Host metadata + NPU tensors | `DeviceStateStore` | store `close()` |
| State schema | Host dataclass | `DeepSeekV4StatePlan` | Python 生命周期 |

State store `close()` 只 free 自己的 state leases，不影响同一 pool 中的 fixed weights、scratch 或 staging。关闭后可重新 `prepare()`；重复 close 是安全的。

Pool `close()` 会强制 free 所有 active 和 idle allocations、清空 free lists，并拒绝后续分配；重复 close 不会再次调用 worker。

## 约束与错误处理

- Shape 各维必须非负，reuse key 必须可 hash。
- DeviceTensor 必须暴露匹配的 shape、dtype、nbytes 和整数 data pointer。
- Persistent 和 reusable category 不能混用分配接口。
- State schemas 的 layer id、state name、input name 和 output name 在各自范围内必须唯一。
- State store 只能 prepare 一次，除非先 close。
- 未 prepare、未知 layer 或错误 next-buffer identity 会被拒绝。
- Prepare 中途失败时按反序 free 已分配 state leases。

## 验证方法

### Device pool

```bash
pytest -q tests/serving/test_device_pool.py
```

覆盖 persistent 初始化、exact-key 复用、copy round trip、cross-pool lease、release/free 状态机、统计和幂等 close。

### State store

```bash
pytest -q tests/serving/test_device_state_store.py
```

覆盖 current/next 初始化、atomic commit、重复 swap、真实 ratio schemas、重复名称拒绝、close 和 reprepare。

这些测试使用 fake ChipWorker 验证所有权和生命周期，不执行模型 Kernel。

## 代码索引

| 内容 | 位置 |
|---|---|
| Device pool | [`serving/device_pool.py`](../../serving/device_pool.py) |
| State store | [`serving/device_state_store.py`](../../serving/device_state_store.py) |
| State schemas | [`serving/state.py`](../../serving/state.py) |
| Pool 测试 | [`tests/serving/test_device_pool.py`](../../tests/serving/test_device_pool.py) |
| State 测试 | [`tests/serving/test_device_state_store.py`](../../tests/serving/test_device_state_store.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`State Plan`](06_state_plan.md)
- [`Runtime Values`](08_runtime_values.md)
- [`Device Runtime`](10_device_runtime.md)
