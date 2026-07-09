# Selected-Expert Decode Serving 实现方案

本文描述 `serving/` 路径下为了支持 decode selected-expert 需要做的修改。
`models/` 侧的 kernel、golden 和 standalone runner 已在
`reference/selected_expert_decode_models_impl.md` 中描述。

## 目标

prefill 继续使用当前 packed routed expert 路径，不改变已有行为：

```text
load full routed pack -> block_*_prefill_fwd -> out
```

decode 改成 selected-expert 路径。每层每个 decode token 先运行 block 的
pre-moe 阶段得到 gate 结果，再只加载当前 token 选中的 `TOPK=6` 个 routed
expert 权重，最后运行 post-moe 阶段得到 block 输出：

```text
1. split_block.*_selected_decode_pre_moe_fwd
   -> update attention / compressor / indexer state
   -> output ffn_normed, indices, weights, ffn_hc_post, ffn_hc_comb, attn_hc_out

2. host reads indices[0, 0, :]
   -> load selected expert weights in top-k order

3. split_block.selected_decode_post_moe_fwd
   -> selected MoE + ffn hc_post
   -> output block out
```

`post_moe` kernel 对所有 decode block 形态通用。它只依赖：

```text
ffn_normed
weights
selected_w*_t
shared_w*_t
attn_hc_out
ffn_hc_post
ffn_hc_comb
```

因此 `swa/csa/hca`、`hash/topk` 都可以复用同一个 post-moe kernel。

## 修改文件

### `serving/weight_loader.py`

新增 selected expert 权重结构：

```python
@dataclass(frozen=True)
class MoESelectedExpertWeights:
    selected_w1_t: torch.Tensor  # [TOPK, HIDDEN, MOE_INTER_DIM]
    selected_w2_t: torch.Tensor  # [TOPK, MOE_INTER_DIM, HIDDEN]
    selected_w3_t: torch.Tensor  # [TOPK, HIDDEN, MOE_INTER_DIM]
```

新增接口：

```python
get_layer_moe_selected_experts(
    layer_id: int,
    expert_ids: torch.Tensor | list[int],
    *,
    device: str | torch.device | None = None,
) -> MoESelectedExpertWeights
```

实现要求：

- `expert_ids` 长度必须是 `TOPK=6`。
- `selected_w*_t[k]` 必须对应 `expert_ids[k]`。
- 每个 expert id 必须在 `[0, n_routed_experts)` 范围内。
- 通过 `get_moe_routed_expert(...)` 读取单个 expert 的 `w1/w2/w3`，优先使用
  `expert_cache_dir/layer_NNN_experts.safetensors`，缺失时回退官方 checkpoint 在线转换。
- 不需要去重。官方 top-k 理论上不会重复；即使重复，按 top-k 顺序复制也能保持语义正确。
- 保留 `get_layer_moe_routed_pack(...)`，因为 prefill 仍然使用 packed routed expert；
  但 full pack 也从同一条 per-expert 加载路径组装，不再使用单独的 routed pack cache。

### `serving/runner.py`

新增 import：

```python
from models import split_block as split_block_kernels
```

`prefill(...)` 路径保持不变，继续调用当前 `_run_block(..., decode=False)` 的 packed
prefill 实现。

`decode(...)` 中每层从单个 packed decode block 调用改为 selected decode 两段式：

```text
pre_outputs = run selected_decode_pre_moe
state.update_layer_state(layer_id, pre_outputs)

selected = weight_loader.get_layer_moe_selected_experts(
    layer_id,
    pre_outputs["indices"][0, 0],
)

post_outputs = run selected_decode_post_moe
hidden = post_outputs["out"]
```

建议把 `_run_block(...)` 拆出两个内部路径：

```python
_run_block_packed_prefill(...)
_run_block_selected_decode(...)
```

也可以保留 `_run_block(...)`，但内部在 `decode=True` 时直接跳到 selected decode
实现。关键是避免 prefill 的参数组装被 selected decode 改动影响。

#### Pre-MoE Kernel 选择

新增 `_selected_decode_pre_case(...)`：

```text
ratio=0,   hash_route=True  -> swa_hash_selected_decode_pre_moe_fwd
ratio=4,   hash_route=True  -> csa_hash_selected_decode_pre_moe_fwd
ratio=128, hash_route=False -> hca_topk_selected_decode_pre_moe_fwd
ratio=4,   hash_route=False -> csa_topk_selected_decode_pre_moe_fwd
```

每个 case 对应 `models/split_block.py` 中的 build specs：

```text
build_swa_hash_selected_decode_pre_moe_specs
build_csa_hash_selected_decode_pre_moe_specs
build_hca_topk_selected_decode_pre_moe_specs
build_csa_topk_selected_decode_pre_moe_specs
```

#### Post-MoE Kernel 选择

所有 selected decode block 形态统一使用：

```text
selected_decode_post_moe_fwd
build_selected_decode_post_moe_specs
```

runner 侧的 case name 可以使用通用名称，例如：

```text
block_selected_decode_post_moe_fwd
```

这样 profile 输出不会误解为只属于 SWA hash。

#### Values 组装

pre-moe values 需要包含：

```text
x
state.build_decode_inputs(...)
layer HC weights
layer attention weights
ffn_norm_w
gate weights
hash route input_ids 或 topk route gate_bias
ratio=128 compressor weights
ratio=4 attention compressor weights
ratio=4 indexer weights
```

pre-moe values 不加载：

```text
shared_w*_t
routed_w*_t
selected_w*_t
```

post-moe values 需要包含：

```text
ffn_normed      <- pre_outputs
weights         <- pre_outputs
attn_hc_out     <- pre_outputs
ffn_hc_post     <- pre_outputs
ffn_hc_comb     <- pre_outputs
selected_w*_t   <- weight_loader.get_layer_moe_selected_experts(...)
shared_w*_t     <- weight_loader.get_layer_moe_shared(...)
```

post-moe 不需要 gate、attention、compressor 或 indexer 权重。

#### State 更新

pre-moe kernel 会输出原 packed decode block 需要更新的 state/cache：

```text
kv_cache_out
comp_*_out
attn_comp_*_out
idx_*_out
```

因此 `serving/state.py::update_layer_state(...)` 可以继续复用。更新时机必须在
pre-moe 后立即执行：

```python
self.state.update_layer_state(layer_id, pre_outputs)
```

post-moe 不更新 state，只产生当前层的 `out`。

### `serving/profiler.py`

建议扩展 profile 名称解析，识别 selected decode pre/post kernel：

```text
*_selected_decode_pre_moe_fwd
*_selected_decode_post_moe_fwd
```

runner 中建议新增 timing 字段：

```text
layer.selected_decode.pre_values
layer.selected_decode.pre_materialize
layer.selected_decode.pre_kernel
layer.selected_decode.state_update
layer.selected_decode.selected_experts
layer.selected_decode.post_values
layer.selected_decode.post_materialize
layer.selected_decode.post_kernel
```

这样可以直接观察 selected-expert decode 是否减少了 routed pack 加载和拷贝成本。

### `serving/generate.py`

`generate.py` 只调用：

```python
runner.prefill(...)
runner.decode(...)
```

decode 内部切换为 selected-expert 后，对 generation loop 是透明的。
CLI 需要透传 `--expert-cache-dir` 给 runner，使 prefill full pack 和 decode selected
experts 都能使用同一份离线 expert cache。

### `serving/state.py`

预计不需要修改。selected pre-moe kernel 的 state 输出名和当前 packed decode
block 一致，现有 `update_layer_state(...)` 可以继续使用。

## 修改顺序

1. 修改 `serving/weight_loader.py`，新增 `MoESelectedExpertWeights` 和
   `get_layer_moe_selected_experts(...)`。
2. 增加或更新本地测试，验证 selected expert 权重按 `indices` 顺序加载。
3. 修改 `serving/runner.py`，先接入 selected decode 单层路径。
4. 在 Ascend 上验证 `--decode-steps 1 --max-layers 1`。
5. 验证 `--decode-steps 1 --max-layers 5`，覆盖四种 block 形态：

```text
swa_hash
csa_hash
hca_topk
csa_topk
```

6. 验证完整 43 层 decode。
7. 更新 `serving/profiler.py` 和 `reference/runner_impl.md`，记录 selected decode
   已成为主 decode 路径。

## 风险点

### Selected Expert 顺序

`selected_w*_t[k]` 必须严格对应 `indices[0, 0, k]`。如果 host 侧对 expert id
排序、去重或重新排列，会破坏 `weights[0, 0, k]` 与 expert 输出的对应关系。

### State 更新时机

attention/compressor/indexer state 必须在 pre-moe 后更新。post-moe 只做 FFN MoE
和 HC post，不会再产生 state 输出。

### Prefill 不应受影响

prefill 仍然需要整层 packed routed experts。修改时不要删除或改变
`get_layer_moe_routed_pack(...)`、packed block prefill case、prefill values 组装逻辑。

### Profile 命名

post-moe kernel 已统一为 `selected_decode_post_moe_fwd`。runner/profile 层也应使用
通用名称记录，避免后续性能分析误判。
