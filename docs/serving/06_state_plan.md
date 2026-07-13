# State Plan 与 Host 辅助输入 Cache

[`DeepSeekV4StatePlan`](../../serving/state.py) 描述每层可变 state 的 shape、dtype 和 Kernel 绑定名称，并缓存 RoPE、topk indices 与控制标量等不可变 Host 辅助输入。

## 模块定位

State plan 是 Host 侧的声明和输入构造组件，不持有 DeviceTensor。Runner 根据它选择 Block 变体并构造每层 aux values；[`DeviceStateStore`](../../serving/device_state_store.py) 根据它生成的 `LayerStateSchema` 分配和交换 NPU current/next buffers。

该模块同时保存完整 normal/compressed RoPE tables，并对当前 prefill `seq_len` 或 decode `start_pos` 的切片结果进行 Host cache。它不负责 H2D；aux tensor 交给 [`DeviceRuntime`](../../serving/device_runtime.py) materialize。

## 核心类型

| 类型 | 职责 |
|---|---|
| `LayerSpec` | 保存 `layer_id`、compression ratio 和是否 hash route |
| `StateTensorSpec` | 定义 state 语义名、Kernel input/output 名、shape、dtype 和初值 |
| `LayerStateSchema` | 组合一层的 `LayerSpec` 与全部 state tensors |
| `DeepSeekV4StatePlan` | 构造全部层 schema、RoPE tables 和 prefill/decode aux bundles |

`StateTensorSpec.create_tensor()` 使用 `torch.full()` 按初值创建 Host tensor，供 device state store 初始化 persistent state。

## 固定运行时约束

State plan 当前只接受：

| 配置 | 要求 |
|---|---|
| `batch_size` | `1` |
| `max_seq_len` | `4096` |
| `config.window_size` | `128` |
| Compression ratio | `0`、`4` 或 `128` |

`LayerSpec.ratio` 直接取 `config.compress_ratios[layer_id]`，`hash_route` 在 `layer_id < config.n_hash_layers` 时为真。默认配置中前 3 层使用 hash route；ratio 序列决定 SWA、CSA 或 HCA state schema。

## 逐层 State Schema

所有层都有一个 sliding-window KV cache：

| State | Input / output | Shape | Dtype / 初值 |
|---|---|---|---|
| `kv_cache` | `kv_cache` / `kv_cache_out` | `[1, 128, head_dim]` | BF16 / `0` |

Ratio 128 层额外包含：

| State | Shape | Dtype / 初值 |
|---|---|---|
| `comp_kv_state` | `[1, 128, head_dim]` | FP32 / `0` |
| `comp_score_state` | `[1, 128, head_dim]` | FP32 / `-finfo(float32).max` |
| `comp_cache` | `[1, 4096/128, head_dim]` | BF16 / `0` |

Ratio 4 层额外包含 attention compressor 和 indexer 两组 state：

| State | Shape | Dtype / 初值 |
|---|---|---|
| `attn_comp_kv_state` | `[1, 8, 2*head_dim]` | FP32 / `0` |
| `attn_comp_score_state` | `[1, 8, 2*head_dim]` | FP32 / 最小有限值 |
| `attn_comp_cache` | `[1, 4096/4, head_dim]` | BF16 / `0` |
| `idx_kv_cache` | `[1, 4096/4, index_head_dim]` | BF16 / `0` |
| `idx_comp_kv_state` | `[1, 8, 2*index_head_dim]` | FP32 / `0` |
| `idx_comp_score_state` | `[1, 8, 2*index_head_dim]` | FP32 / 最小有限值 |

`idx_kv_cache` 的 Kernel input 名是 `idx_kv_cache_in`，其余 state 通常以语义名作为 input，并以 `_out` 后缀作为 output。

## RoPE Tables

构造 state plan 时一次生成两组长度为 4096 的 FP32 cos/sin tables：

| Profile | 使用者 | Base / original length |
|---|---|---|
| Normal | ratio 0 主 attention | `rope_theta` / `0` |
| Compressed | ratio 4、128 主 attention 与 compressor | `compress_rope_theta` / `original_seq_len` |

`materialize_rope_range()` 为主 attention 提取连续位置；prefill compressor 按 ratio 下采样；decode 只在 `(start_pos + 1) % ratio == 0` 时取 compressor 位置，否则复用预创建的全零 RoPE pair。

Ratio 4 和 ratio 128 的主 attention 都使用 compressed profile，但各自的 compressor slice 和控制标量按 ratio 区分。

## Top-k Indices

`build_window_topk_idxs()` 构造固定窗口的 `int32` indices：prefill 为每个 token 建立因果窗口，decode 在 window 内滚动并按 cache slot 顺序排列。

`build_compress_topk_idxs()` 构造 compressed cache indices：

- Prefill 仅允许访问当前 token 之前已经形成的压缩块。
- Decode 使用 `(start_pos + 1) // ratio` 决定可见块数。
- `offset` 把 compressed indices 放到 window cache 后的逻辑地址空间。
- 未使用位置填 `-1`，输出最后一维固定 padding 到 `topk_max`。

## Prefill Aux

`build_prefill_aux(layer_id, seq_len)` 根据 ratio 返回：

| Ratio | 主要字段 |
|---|---|
| `0` | `topk_idxs`、`cos`、`sin` |
| `4` | window topk、main RoPE、attention/indexer compressor RoPE、block count、`idx_offset` |
| `128` | window+compressed topk、main/compressor RoPE、block count |

同一 prefill `seq_len` 内，所有层共享 window topk；相同 ratio 的层共享完整 aux bundle。`seq_len` 改变时清空 prefill aux cache，再按需构建。

## Decode Aux

`build_decode_aux(layer_id, start_pos)` 为单 token 返回 common `cache_pos` 和 main RoPE，并按 ratio 增加：

| Ratio | 额外字段 |
|---|---|
| `0` | window `topk_idxs` |
| `4` | window topk、compressor slot/cache slot、should-compress、index offset 与 compressor RoPE |
| `128` | window+compressed topk、compressor slot/cache slot、should-compress 与 compressor RoPE |

同一 `start_pos` 内共享 window topk、RoPE slices、控制标量和相同 ratio bundle；位置改变时清空 decode aux cache。

## 数据位置与复用

| 数据 | 位置 | 生命周期 |
|---|---|---|
| Normal/compressed full RoPE tables | Host | state plan 生命周期 |
| Zero compressor RoPE | Host | state plan 生命周期 |
| Prefill aux cache | Host | 当前 `seq_len`，按 ratio 和子对象复用 |
| Decode aux cache | Host | 当前 `start_pos`，按 ratio 和子对象复用 |
| State schemas | Host dataclass | state plan 生命周期 |
| Mutable state data | NPU | 不由本模块持有；由 device state store 管理 |

Aux builder 返回新的外层 `dict`，但其中缓存 tensor 保持同一对象。Runner 在同一个 step 跨层复用这些 Host tensor 时，runtime 的 active-upload 机制可避免同一 tensor 重复 H2D。

## 约束与错误处理

- Prefill `seq_len` 必须在 `[1, 4096]`。
- Decode `start_pos` 必须在 `[1, 4095]`。
- Decode 风格 topk 只接受 `seq_len=1`。
- RoPE dim 必须为正偶数，ratio 必须为正数。
- Unsupported compression ratio 在 schema 或 aux 构造时直接报错。
- `topk_max` 小于实际需要的 compressed block 数时拒绝截断。

## 验证方法

### Host 侧验证

```bash
pytest -q tests/serving/test_state.py
```

测试覆盖逐层 spec、state shape/dtype/初值、Kernel input/output 名、topk 与官方/Kernel helper 对齐、RoPE profile、compressor 边界、各 ratio aux、跨层对象复用和固定 shape contract。

### Device state 集成

```bash
pytest -q tests/serving/test_device_state_store.py tests/serving/test_runner.py
```

前者验证 schema 被转换为 current/next persistent buffers，后者验证 prefill/decode 绑定正确的 state inputs 与 outputs。

## 代码索引

| 内容 | 位置 |
|---|---|
| State plan | [`serving/state.py`](../../serving/state.py) |
| State store | [`serving/device_state_store.py`](../../serving/device_state_store.py) |
| Runner 调用方 | [`serving/runner.py`](../../serving/runner.py) |
| State 测试 | [`tests/serving/test_state.py`](../../tests/serving/test_state.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`整模 Runner`](05_runner.md)
- [`Device Memory`](09_device_memory.md)
