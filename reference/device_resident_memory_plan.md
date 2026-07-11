# Device-resident 显存预算与生命周期

## 1. 结论

基于当前 checkpoint、模型配置和 kernel `TensorSpec` 重新计算后，第一版 WorkerBackend 可以采用：

- 除 routed experts 外，所有最终 runtime-layout 权重常驻 NPU；shared experts 也常驻。
- 43 层 mutable state/cache 在 NPU 上双缓冲常驻。
- hidden、kernel output 和 scratch 不回 Host，按生命周期在 device pool 中复用。
- routed experts 不进入固定 cache：prefill 按层上传完整 pack，decode 按层上传 6 个 selected experts。
- Host 只读取 selected expert indices 等控制数据以及最终 public output。

完整 43 层、包含 head、`max_seq_len=4096` 时，静态估算的最坏峰值约为：

```text
27.33 GiB
```

64 GB 按十进制换算为 59.60 GiB，仍有约 32.27 GiB 余量。该预算足以容纳 runtime、compiled
program、allocator 对齐和碎片；实现后仍须用实际设备指标验证。

接口与文件设计以 `device_resident_implementation_plan.md` 为准，本文只约束容量、对象归属和释放
时机。

## 2. 计算口径

计算来自代码和实际 checkpoint index，而不是模型说明文档：

- checkpoint index：`/data/wuzhifeng/dsv4_ckpt/bf16_lowvram_cache/weight_index.json`
- 模型配置：`models/config.py` 的 `FLASH_CONFIG`
- state/cache shape：`serving/state.py` 的 `DeepSeekV4StatePlan`
- kernel 参数、scratch 和 output shape：
  - `models/embedding.py`
  - `models/block.py`
  - `models/split_block.py`
  - `models/head.py`

关键参数：

```text
dim                  = 4096
moe_inter_dim        = 2048
n_layers             = 43
n_routed_experts     = 256
n_activated_experts  = 6
hc_mult              = 4
head_dim             = 512
index_head_dim       = 128
window_size          = 128
max_seq_len          = 4096
vocab_size           = 129280
```

所有权重按 kernel 最终接收的 runtime dtype 和 layout 计费，不按 checkpoint 原始 dtype 计费。例如
`head.weight` 的 kernel-facing tensor 是 FP32，因此按 FP32 计算。

## 3. 固定 resident 权重

固定 cache 包含 1,199 个 runtime layout：

| 分类 | Entries | Runtime bytes | GiB |
|---|---:|---:|---:|
| Embedding | 1 | 1,059,061,760 | 0.986 |
| Head、final norm、head HC | 5 | 2,119,180,356 | 1.973 |
| 43 层 HC | 258 | 135,275,592 | 0.126 |
| Attention common | 387 | 9,198,604,032 | 8.567 |
| Gate 与 FFN norm | 129 | 99,878,912 | 0.093 |
| Shared experts | 129 | 2,164,260,864 | 2.016 |
| Compressor 与 indexer | 290 | 977,225,984 | 0.910 |
| 合计 | 1,199 | 15,753,487,500 | 14.672 |

固定 cache 不设 LRU，也不设置容量参数。`RuntimeWeightKey` 是 Host runtime-layout cache 和 Worker
device cache 的共同语义 key；同一个 layout 第一次 materialize 时上传，之后始终复用同一个
device handle。

以下权重不能进入固定 cache：

- `routed_w1_t`
- `routed_w2_t`
- `routed_w3_t`
- `selected_w1_t`
- `selected_w2_t`
- `selected_w3_t`

43 层全量 routed experts 为：

```text
3 × 43 × 256 × 4096 × 2048 × BF16
= 516 GiB
```

因此 routed experts 必须由 `HostStagingTensor` 明确标记，交给 WorkerBackend 的 staging 路径，不能
靠 tensor 名称猜测，也不能落入固定 device cache。

## 4. State/cache 常驻预算

当前 43 层 mutable state 与共享 RoPE table 的单份规模为：

| State 类型 | Bytes |
|---|---:|
| KV cache | 5,636,096 |
| Ratio-4 attention compressed cache | 22,020,096 |
| Ratio-4 attention KV state | 688,128 |
| Ratio-4 attention score state | 688,128 |
| Ratio-4 indexer KV cache | 5,505,024 |
| Ratio-4 indexer KV state | 172,032 |
| Ratio-4 indexer score state | 172,032 |
| Ratio-128 compressed cache | 655,360 |
| Ratio-128 KV state | 5,242,880 |
| Ratio-128 score state | 5,242,880 |
| Normal/compressed/zero RoPE tables | 2,097,408 |
| 合计 | 48,120,064 |

当前 kernel 使用独立 state input/output。第一版不改 kernel，而由 `WorkerStateStore` 根据
`DeepSeekV4StatePlan.layer_state_schema()` 为每层分配 current/next 两套 device buffer；成功执行后交换
handle，异常时不 commit。

该 store 已完成独立实现及真实 ChipWorker 双缓冲验证；尚未接入 WorkerBackend 的 kernel dispatch。

```text
2 × mutable state + RoPE tables
= 94,142,720 bytes
= 89.78 MiB
= 0.088 GiB
```

`DeepSeekV4StatePlan` 只描述 shape/dtype 和生成 Host aux，不拥有可变 runtime state；实际 state 生命周期
分别由 `DirectStateStore` 和 `WorkerStateStore` 管理。

## 5. Prefill 峰值

### 5.1 Routed expert staging

单层完整 256 experts：

```text
3 × 256 × 4096 × 2048 × BF16
= 12,884,901,888 bytes
= 12 GiB
```

这是 resident 执行的最大动态分配。`DeviceBufferPool` 已实现三个 4 GiB 语义 staging slot，同一 step
内按层覆盖复用，不做逐层大块 alloc/free。默认在 prefill step 结束后释放；只有显式
`--keep-prefill-routed-staging` 才跨 step 保留。

### 5.2 S=4096 工作集

| 动态分类 | Bytes | MiB/GiB |
|---|---:|---:|
| Routed expert pack | 12,884,901,888 | 12.000 GiB |
| Hidden input/output ping-pong | 268,435,456 | 256.0 MiB |
| Kernel scratch/output | 343,408,640 | 327.5 MiB |
| RoPE、top-k、标量 aux | 3,702,796 | 3.53 MiB |
| 动态合计 | 13,500,448,780 | 12.573 GiB |

固定 common weights 和全局 state pool 已在其他分类计入，不能再按每个 TensorSpec 重复计算。

### 5.3 总峰值

```text
14.672 GiB fixed non-routed weights
+ 0.088 GiB state/cache double buffers
+ 12.573 GiB max prefill dynamic working set
= 27.33 GiB
```

## 6. Decode 峰值

每层 selected expert pack：

```text
3 × 6 × 4096 × 2048 × BF16
= 301,989,888 bytes
= 288 MiB
= 0.28125 GiB
```

decode pre-MoE 的最大 scratch 约 2.6 MiB。因此 warm decode 常态峰值约为：

```text
14.672 GiB fixed weights
+ 0.088 GiB state/cache
+ 0.281 GiB selected experts
+ small intermediate/scratch
= about 15.04 GiB
```

selected staging 由三个 96 MiB slot 组成，在每层被覆盖并跨 decode step 复用。

该路径已经在真实 ChipWorker 上完成单层连续 3-step 验证：每层仅 indices 和最终 public output D2H，
selected pack 使用 `STAGING_SELECTED` 空闲 buffer 跨 step 复用。

## 7. DeviceBufferPool 生命周期

分配必须带明确分类：

| 分类 | 生命周期 | 复用方式 |
|---|---|---|
| `FIXED_WEIGHT` | backend lifetime | 按 `RuntimeWeightKey` 永久命中 |
| `STATE` | backend lifetime | current/next handle 交换 |
| `STAGING_ROUTED` | prefill step，默认结束释放 | 3 个固定 slot |
| `STAGING_SELECTED` | backend lifetime | 3 个固定 slot |
| `INTERMEDIATE` | active step/dispatch | 按 dtype、shape、用途复用 |
| `SCRATCH` | active dispatch | kernel 完成后回池 |
| `ACTIVE_UPLOAD` | active step | Host aux/临时输入的短期缓存 |

“中间 tensor 常驻 NPU”表示数据在 kernel dispatch 之间不回 Host，不表示永久保存 43 层历史输出。
hidden 应使用最大 shape ping-pong buffer；无后续消费者的 output/scratch 在安全点归还池。

新分配的 output/state 必须保持与当前 `TensorSpec` materialize 的零初始化语义一致。只有证明某个 kernel
会完整覆盖全部元素时，才允许跳过初始化。

## 8. Host 与 Device 的边界

### 8.1 固定权重

```text
checkpoint
  → WeightLoader Host runtime-layout cache
  → RuntimeWeight
  → WorkerBackend first-miss H2D
  → fixed DeviceTensor cache
```

Host layout 保留到 `WeightLoader.close()`，既服务 DirectBackend，也作为 device 恢复来源。

### 8.2 Routed experts

```text
expert cache/checkpoint
  → WeightLoader builds HostStagingTensor
  → WorkerBackend copies into a staging slot
  → kernel dispatch
  → slot reused/released by step policy
```

### 8.3 Selected expert indices

decode pre-MoE 产生 `[1, 1, 6]` 的 INT32 indices，仅 24 bytes。Host 必须读取它们才能选择 routed
expert 权重，因此第一版每层允许一次小 D2H；weights、ffn intermediate 和 hidden 均保持在 NPU。

### 8.4 最终输出

第一版只在 public API 边界把最终 logits/hidden 拷回 Host。未来可把 sampling/argmax 下沉到 NPU，
只返回 token id，但不属于首版范围。

## 9. 64 GB 安全余量

```text
64,000,000,000 bytes = 59.60 GiB
59.60 GiB - 27.33 GiB = 32.27 GiB
```

该余量用于：

- CANN/PyPTO runtime。
- compiled program 和 runtime workspace。
- allocator 对齐、碎片和异步执行期间的短时重叠。
- H2D/D2H runtime 资源。
- L2 swimlane 资源。

第一版验收应将 30 GiB 作为 S=4096 prefill 的静态目标上界；如果实测明显超过，需要先按分类定位，
而不是给固定权重增加 LRU。

## 10. 性能基线和收益口径

当前 DirectBackend 的最近一次 5 层、`seq_len=1`、无 head、3 decode step profile：

```text
warm decode average             1678 ms / 5 layers
embedding                       144 ms
block average                   302 ms / layer
pre materialize                 17.7 ms / layer
pre kernel                      104.5 ms / layer
selected expert load/build      84.7 ms / layer
post kernel                     94.1 ms / layer
```

Device resident 主要消除固定权重、state、hidden 和普通 intermediate 的 Host materialize/H2D/D2H，不能
消除 selected expert 的 Host load/build 与 H2D。基于当前分解，完整 warm decode 的合理收益预估为
10%–20%，乐观上限约 25%；最终以同机交替 A/B 为准。

必须分别记录：

```text
device.fixed_weight.bytes/hit/miss/upload_ms
device.state.bytes
device.intermediate.bytes
device.scratch.bytes
device.routed_staging.bytes
device.routed_upload_ms
device.h2d.bytes
device.d2h.bytes
device.current_bytes
device.peak_bytes
```

## 11. 释放顺序

`WorkerBackend.close()` 必须幂等，并按所有权逆序释放：

```text
停止新 dispatch
→ 释放 active upload 和 routed/selected staging
→ 释放 scratch、intermediate、hidden
→ WorkerStateStore 释放 state/cache 双缓冲
→ 释放 fixed-weight DeviceTensor
→ 清理 compiled cache
→ ChipWorker.close()
```

Runner 仍负责在 backend 关闭后关闭 `WeightLoader`。任何不由 WorkerBackend 分配的 borrowed
`DeviceTensor` 都不得由它释放。

## 12. 验收标准

- fixed resident bytes 约 14.672 GiB，state 双缓冲约 0.088 GiB。
- decode 常态约 15.04 GiB。
- S=4096 prefill 静态峰值目标不超过 30 GiB。
- 默认 prefill 结束后 12 GiB staging 被释放，显存明显回落。
- `close()` 后所有 owned device allocations 清零，重复 close 不报错。
- Direct/Worker 在 embedding、单层、5 层、43 层及多 decode step 上数值一致。
- 性能测试区分 cold upload、first compile、warm decode，并固定 `--enable-l2-swimlane` 配置。
