# DeepSeek V4 Flash State Manager Plan

本文记录 `serving/state.py` 的职责和实现方案。`state.py` 是整网 host runtime 的状态管理
层，只管理 cache、compressor state、topk、RoPE 和 decode 位置辅助输入，不负责权重加载、
kernel 选择、PyPTO 编译或运行。

## 边界

`serving/state.py` 放在 host 侧，不能依赖 PyPTO kernel 文件。它可以依赖 PyTorch 和
`models.config.FLASH_CONFIG`，但不直接 import `models/block.py`、`models/rope.py` 或
`models/sparse_attn.py`。这样后续 `serving/runner.py` 可以在纯 host 逻辑中稳定构造输入，
不会被 kernel module 的 PyPTO import 绑定。

当前 kernel 编译时已经固定以下运行约束：

```text
B = 1
window_size = 128
max_seq_len = 4096
TOPK_HCA = 4096 // 128 = 32
INDEX_SCORE_LEN = 4096 // 4 = 1024
```

因此 `state.py` 第一版应显式校验 `batch_size == 1`、`max_seq_len == 4096`。如果后续需要
支持其他最大长度，必须先重新参数化并验证对应 PyPTO kernel。

## Layer 形态

每个正常 block 的状态形态由官方配置决定：

```text
layer 0,1       ratio=0,   hash_route=True
layer 2         ratio=4,   hash_route=True
layer 3,5,...   ratio=128, hash_route=False
layer 4,6,...   ratio=4,   hash_route=False
```

`compress_ratios[43]` 属于 MTP，不在当前整网范围内。

## 持久 State

所有层都有 SWA window KV cache：

```text
kv_cache [1, 128, 512] bf16
```

ratio=128 层额外维护 HCA compressor state：

```text
comp_cache       [1, 32, 512] bf16
comp_kv_state    [1, 128, 512] fp32
comp_score_state [1, 128, 512] fp32
```

ratio=4 层额外维护 CSA attention compressor state：

```text
attn_comp_cache       [1, 1024, 512] bf16
attn_comp_kv_state    [1, 8, 1024] fp32
attn_comp_score_state [1, 8, 1024] fp32
```

ratio=4 层还维护 CSA indexer state：

```text
idx_kv_cache          [1, 1024, 128] bf16
idx_comp_kv_state     [1, 8, 256] fp32
idx_comp_score_state  [1, 8, 256] fp32
```

这些 tensor 由 `DeepSeekV4State` 初始化，并在每次 block kernel 返回后通过
`update_layer_state()` 替换为对应 `*_out`。

## Prefill 输入

prefill 固定 `start_pos = 0`，输入 `seq_len > 0`。

ratio=0：

```text
topk_idxs = build_window_topk_idxs(seq_len, start_pos=0, topk_max=128)
cos/sin   = materialize_rope_range(profile=normal, start_pos=0, seq_len)
```

ratio=128：

```text
window_topk   = build_window_topk_idxs(seq_len, 0, 128)
compress_topk = build_compress_topk_idxs(128, seq_len, 0, offset=seq_len, topk_max=32)
topk_idxs     = cat([window_topk, compress_topk], dim=-1)

comp_block_count = seq_len // 128
comp_cos/sin     = freqs[:cutoff:128] with compress RoPE profile
```

如果 `seq_len < 128`，`comp_block_count = 0`，但 kernel 仍需要 `[1, 32]` 的 RoPE 占位，
因此 `comp_cos/sin` 取 compress profile 的首行。

ratio=4：

```text
window_topk_idxs       = build_window_topk_idxs(seq_len, 0, 128)
attn_comp_block_count  = seq_len // 4
idx_comp_block_count   = seq_len // 4
attn_comp_cos/sin      = freqs[:cutoff:4] with compress RoPE profile
idx_comp_cos/sin       = same as attn_comp_cos/sin
idx_offset             = seq_len
```

如果 `seq_len < 4`，`attn_comp_cos/sin` 和 `idx_comp_cos/sin` 同样使用首行占位。

注意：当前 `models/block.py` 中 ratio=4 的主 attention RoPE 也使用 compress RoPE profile；
ratio=128 的主 attention RoPE 使用 normal profile。`state.py` 必须保持这个行为。

## Decode 输入

decode 固定 `seq_len = 1`，`start_pos > 0` 由 runner 显式维护。

公共输入：

```text
cache_pos = start_pos % 128
cos/sin   = materialize_rope_range(layer-specific main profile, start_pos, 1)
```

ratio=0：

```text
kv_cache
topk_idxs = build_window_topk_idxs(1, start_pos, 128)
cache_pos
```

ratio=128：

```text
kv_cache
comp_cache
comp_kv_state
comp_score_state
topk_idxs = cat([window_topk, compress_topk(offset=128)], dim=-1)

comp_slot            = start_pos % 128
comp_cache_slot      = start_pos // 128
comp_should_compress = int((start_pos + 1) % 128 == 0)
```

ratio=4：

```text
kv_cache
attn_comp_cache
attn_comp_kv_state
attn_comp_score_state
idx_kv_cache_in
idx_comp_kv_state
idx_comp_score_state
window_topk_idxs
idx_offset = 128

comp_slot            = start_pos % 4
comp_cache_slot      = start_pos // 4
comp_should_compress = int((start_pos + 1) % 4 == 0)
```

decode compressor RoPE：

```text
if comp_should_compress:
    rope_pos = start_pos + 1 - ratio
    comp_cos/sin = compress_freqs[rope_pos:rope_pos + 1]
else:
    comp_cos/sin = zeros([1, 32], fp32)
```

ratio=4 的 `attn_comp_cos/sin` 和 `idx_comp_cos/sin` 使用同一份计算结果。

## API

建议实现：

```python
class DeepSeekV4State:
    def __init__(config=FLASH_CONFIG, max_seq_len=4096, batch_size=1, device="cpu")
    def layer_spec(layer_id) -> LayerSpec
    def layer_state(layer_id) -> LayerState
    def build_prefill_inputs(layer_id, seq_len) -> dict[str, torch.Tensor]
    def build_decode_inputs(layer_id, start_pos) -> dict[str, torch.Tensor]
    def update_layer_state(layer_id, outputs) -> None
```

辅助函数：

```python
build_window_topk_idxs(seq_len, start_pos=0, topk_max=128, batch_size=1)
build_compress_topk_idxs(ratio, seq_len, start_pos, offset, topk_max, batch_size=1)
build_deepseek_v4_rope_tables(config, compress_ratio, max_seq_len, rope_dim, device)
materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)
materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, ratio)
materialize_decode_compressor_rope(freqs_cos, freqs_sin, start_pos, ratio)
```

`update_layer_state()` 只更新 state tensor，不修改 `start_pos`。生成位置应由
`serving/runner.py` 或 `serving/generate.py` 管理，避免 state manager 隐式推进位置导致
prefill/decode 边界不清晰。

## 测试

新增 `tests/test_state.py`，覆盖：

- 43 层 layer type 和 `compress_ratios` 对齐。
- SWA/HCA/CSA state 初始化 shape/dtype。
- prefill `seq_len=1, 3, 4, 127, 128, 129` 的 topk、block_count、RoPE shape。
- decode `start_pos=1, 3, 4, 127, 128, 129` 的 slot、cache_slot、should_compress。
- `build_window_topk_idxs` 和 `build_compress_topk_idxs` 与 `models.sparse_attn` 中 helper 一致。
- `update_layer_state()` 能按 ratio 正确替换对应持久 state。
