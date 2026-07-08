# Selected-Expert Decode Models 实现方案

本文只描述 `models/` 路径下为了支持 decode selected-expert 需要做的修改。
`serving/` 中如何先运行 gate、加载被选中的专家权重、再调用 selected decode
kernel 不在本文范围内。

## 目标

decode 场景固定 `B=1`、`S=1`。每层每步 MoE gate 只会产生 `TOPK=6`
个 routed expert：

```text
indices [1, 1, 6]
weights [1, 1, 6]
```

selected-expert decode kernel 不再接收整层 256 个 routed experts：

```text
routed_w1_t [256, 4096, 2048]
routed_w2_t [256, 2048, 4096]
routed_w3_t [256, 4096, 2048]
```

而是接收当前 token 实际选中的 6 个 expert 权重：

```text
selected_w1_t [6, 4096, 2048]
selected_w2_t [6, 2048, 4096]
selected_w3_t [6, 4096, 2048]
```

kernel 内约定：

```text
selected_w*_t[k] 对应 gate 输出 indices[0, 0, k] 的 expert 权重
```

因此 selected decode kernel 本身不需要接收 `selected_eids`，也不需要在
kernel 内做 `expert_id -> local selected index` 查找。`selected_eids` 可以只在
host/golden/test 中用于校验传入权重是否和 gate 结果一致。

## 文件修改范围

### `models/moe.py`

新增 decode-only selected MoE helper、kernel 和 golden。为了配合 block 的
pre-moe/post-moe 拆分，MoE 侧也需要明确拆成两部分：

```text
gate 阶段：x -> indices / weights
selected expert 阶段：x + weights + selected_w*_t -> out
```

`gate_hash_fwd` / `gate_topk_fwd` 已经实现第一部分，不需要新增新的 gate kernel。
selected-expert 需要新增的是第二部分，也就是“不再运行 gate，只消费已计算好的
weights 和 selected expert 权重”的 kernel/helper。

建议新增常量：

```python
S_DECODE = 1
```

新增内部 helper：

```python
_run_selected_decode_routed_experts(
    x: [1, 1, HIDDEN] bf16,
    weights: [1, 1, TOPK] fp32,
    selected_w1_t: [TOPK, HIDDEN, MOE_INTER_DIM] bf16,
    selected_w2_t: [TOPK, MOE_INTER_DIM, HIDDEN] bf16,
    selected_w3_t: [TOPK, HIDDEN, MOE_INTER_DIM] bf16,
    route_y: [1, 1, TOPK, HIDDEN] bf16,
)
```

计算逻辑：

```text
for k in range(TOPK):
    route_weight = weights[0, 0, k]
    gate = x @ selected_w1_t[k]
    up   = x @ selected_w3_t[k]
    hidden = silu_clamp(gate, up) * route_weight
    route_y[0, 0, k] = hidden @ selected_w2_t[k]
```

这里和当前 packed MoE 一样，在 `w2` 前乘 route weight，保持和
`official/model.py::Expert.forward(x, weight)` 语义一致。

新增 selected expert 阶段 helper：

```python
moe_selected_decode_experts_fwd(
    x,
    weights,
    selected_w1_t,
    selected_w2_t,
    selected_w3_t,
    shared_w1_t,
    shared_w2_t,
    shared_w3_t,
    out,
)
```

这个 helper 对 hash route 和 topk route 通用，因为两种 route 在 selected expert
阶段的输入都已经统一为：

```text
x       [1, 1, HIDDEN]
weights [1, 1, TOPK]
selected_w*_t[k] 按 gate 输出 indices[0, 0, k] 的顺序排列
```

为了 standalone 验证，可以再提供两个 test wrapper：

```python
moe_hash_selected_decode_test(...)
moe_topk_selected_decode_test(...)
```

这些 test wrapper 可以内部调用 `gate_hash_fwd` / `gate_topk_fwd` 生成
`indices` 和 `weights`，再调用 `moe_selected_decode_experts_fwd`。但 block 的
post-moe kernel 不应再调用 gate，它只消费 pre-moe 阶段输出的 `weights`。

block pre-moe 阶段直接复用现有 gate kernel：

```python
gate_hash_fwd(...)
gate_topk_fwd(...)
```

新增 golden：

```python
golden_moe_selected_decode_experts_forward(tensors)
```

golden 逻辑和 selected expert 阶段保持一致：

1. 读取已经存在的 `weights`。
2. 对 `k=0..5` 直接使用 `selected_w*_t[k]` 计算 expert。
3. 加上 shared expert 输出。
4. 写入 `out`。

为了验证与官方 packed MoE 语义一致，测试中先用当前 gate golden 算出
`indices` 和 `weights`，再构造 dependent inputs：

```text
selected_w*_t[k] = routed_w*_t[indices[0, 0, k]]
```

然后比较：

```text
golden_moe_selected_decode_experts_forward == golden_moe_forward
```

### `models/block.py`

新增 selected-expert decode block kernel。现有 packed decode block 不修改，作为
已经验证通过的基线保留。

selected-expert decode 不能继续使用“一个 kernel 完成整个 block”的接口。原因是
host 侧必须先拿到 MoE gate 的 `indices`，才能加载当前 token 选中的 6 个 expert
权重。因此每种 decode block 形态都需要拆成两段：

```text
*_selected_decode_pre_moe_fwd
*_selected_decode_post_moe_fwd
```

第一段 pre-moe kernel 不接收 routed expert 权重，执行到 FFN MoE gate 为止：

```text
attention hc_pre
attention
attention out
attention hc_post
ffn hc_pre
ffn_norm
gate
```

pre-moe kernel 输出：

```text
ffn_normed:  [1, 1, HIDDEN] bf16
indices:     [1, 1, TOPK] int32
weights:     [1, 1, TOPK] fp32
ffn_hc_post: [1, 1, HC_PAD] fp32
ffn_hc_comb: [1, 1, HC_MULT * HC_MULT] fp32
```

同时它还输出/更新该 block 形态原本 decode kernel 需要更新的 attention、
compressor、indexer cache/state。host 拿到 `indices[0, 0, :]` 后，按 top-k 顺序
加载 selected expert 权重。

第二段 post-moe kernel 接收 pre-moe 输出和 selected expert 权重：

```text
ffn_normed
weights
selected_w1_t [TOPK, HIDDEN, MOE_INTER_DIM]
selected_w2_t [TOPK, MOE_INTER_DIM, HIDDEN]
selected_w3_t [TOPK, HIDDEN, MOE_INTER_DIM]
shared_w1_t
shared_w2_t
shared_w3_t
ffn_hc_post
ffn_hc_comb
```

post-moe kernel 执行：

```text
selected MoE
ffn hc_post
```

并输出完整 block 的 `out`。

需要覆盖当前 decode 中会出现的四种 block 形态：

```text
swa_hash_selected_decode_pre_moe_fwd
swa_hash_selected_decode_post_moe_fwd
csa_hash_selected_decode_pre_moe_fwd
csa_hash_selected_decode_post_moe_fwd
hca_topk_selected_decode_pre_moe_fwd
hca_topk_selected_decode_post_moe_fwd
csa_topk_selected_decode_pre_moe_fwd
csa_topk_selected_decode_post_moe_fwd
```

这些 kernel 与现有 decode block 的 attention、compressor、indexer、HC、norm
逻辑保持一致，只把 FFN MoE 所在位置拆开。pre-moe 阶段仍调用当前
`gate_hash_fwd` / `gate_topk_fwd`，post-moe 阶段把原来的完整 MoE 调用替换为
selected expert 阶段：

```python
moe_hash_fwd(...) -> moe_selected_decode_experts_fwd(...)
moe_topk_fwd(...) -> moe_selected_decode_experts_fwd(...)
```

接口中 routed expert 权重从：

```text
routed_w1_t [256, HIDDEN, MOE_INTER_DIM]
routed_w2_t [256, MOE_INTER_DIM, HIDDEN]
routed_w3_t [256, HIDDEN, MOE_INTER_DIM]
```

替换为：

```text
selected_w1_t [TOPK, HIDDEN, MOE_INTER_DIM]
selected_w2_t [TOPK, MOE_INTER_DIM, HIDDEN]
selected_w3_t [TOPK, HIDDEN, MOE_INTER_DIM]
```

pre-moe 阶段需要返回中间输出，因此会比现有 packed decode block 多出
`ffn_normed`、`indices`、`weights`、`ffn_hc_post`、`ffn_hc_comb` 等业务输出。
post-moe 阶段只消费这些中间输出和 selected expert 权重，不再重复 attention 或
gate 计算。

新增 selected block golden：

```python
golden_selected_decode_pre_moe_forward(...)
golden_selected_decode_post_moe_forward(...)
```

pre-moe golden 复用当前 block golden 的 attention、HC、norm 和 gate 逻辑，并
写出 `indices` / `weights`。post-moe golden 调用
`golden_moe_selected_decode_experts_forward`，再执行 FFN `hc_post`。

测试策略分两层：

1. selected pre-moe/post-moe kernel vs 对应 selected golden。
2. 在 CPU pytest 中先用 packed block golden 算 gate 结果，再按 `indices` 构造
   `selected_w*_t`，验证 selected block golden 和 packed block golden 输出一致。

### `models/expert.py`

原则上不需要修改。

`expert_routed_fwd` 已经是单 expert 权重接口，可以作为 selected MoE 的参考。
但 selected decode kernel 中建议先在 `models/moe.py` 内直接展开 routed expert
计算，原因是：

- 当前 selected 权重是 `[TOPK, ...]` 的 3D tensor；
- 需要验证 `selected_w*_t[k]` 作为 2D view 传给已有 linear helper 是否稳定；
- 直接展开可以减少 inline view 传参的不确定性，和当前 `_run_route_major_routed_experts`
  的写法保持一致。

如果后续验证 3D -> 2D view 传参稳定，可以再把重复计算收敛回
`expert_routed_fwd`。

### `models/gate.py`

不需要修改。

selected-expert decode 仍然复用当前 gate 计算逻辑。`indices` 决定 host 侧选择哪
6 个 expert 权重，`weights` 进入 selected MoE kernel 参与 expert 输出缩放。

### `models/linear.py`

大概率不需要修改。

selected expert 的 linear shape 与当前 routed expert 完全一致：

```text
w1: 4096 -> 2048
w3: 4096 -> 2048
w2: 2048 -> 4096
```

当前已有对应的 linear 逻辑。如果在实现时选择复用 `expert_routed_fwd` 或已有
linear helper，并发现 3D selected 权重切片传入 2D linear helper 编译不稳定，再
补充专用 selected linear helper。

### `models/golden.py`

通常不需要修改。

如果 selected block 测试需要比较 top-k index 或 selected expert id，可复用已有
`topk_indices_by_score` 等 compare helper。只有当新增输出需要特殊比较时才扩展。

### `tests/`

需要新增或扩展：

```text
tests/test_moe.py
tests/test_block.py
```

`tests/test_moe.py` 覆盖：

- `golden_moe_selected_decode_experts_forward` 在 hash route 下与 packed golden 一致。
- `golden_moe_selected_decode_experts_forward` 在 topk route 下与 packed golden 一致。
- selected 权重由 `indices[0, 0, k]` 从完整 routed pack 中取出。
- `selected_w*_t[k]` 的顺序和 `indices[..., k]` 严格一致。

`tests/test_block.py` 覆盖：

- 四种 selected decode block golden 与对应 packed decode block golden 一致。
- `S=1` 固定 decode 场景。
- hash route 和 topk route 都覆盖。

PyPTO kernel 的设备端验证仍通过各文件自带 runner 在 Ascend 上执行，不依赖
pytest。

## 修改顺序

推荐按下面顺序实现，避免一次性修改完整 block 后难以定位问题。

### 1. `models/moe.py` 的 selected expert 阶段 golden

先只实现：

```python
golden_moe_selected_decode_experts_forward
```

并在 `tests/test_moe.py` 中验证：

```text
selected golden == packed golden
```

此时不写 PyPTO kernel，只确认 selected-expert 的数学语义和官方 MoE 对齐。

### 2. `models/moe.py` 的 selected expert 阶段 kernel

实现：

```python
moe_selected_decode_experts_fwd
moe_hash_selected_decode_test
moe_topk_selected_decode_test
```

`moe_selected_decode_experts_fwd` 不运行 gate，只消费 `weights`。两个 test wrapper
可以为了 standalone 验证而内部运行 gate。

先用 standalone runner 验证：

```bash
python models/moe.py -p a2a3 -d {} --case selected-decode
```

如果现有 CLI 没有 `--case`，可以沿用当前文件风格，默认同时跑 packed 和
selected 用例，或新增只影响 `models/moe.py` 的 case 参数。

### 3. `models/block.py` 的 selected decode pre/post golden

新增 selected decode pre-moe 和 post-moe golden，先不写 block kernel。

在 pytest 中构造：

```text
packed routed_w*_t
pre-moe golden 输出 gate indices / weights
selected_w*_t[k] = routed_w*_t[indices[0, 0, k]]
```

再把 pre-moe 输出喂给 post-moe golden，验证完整 selected block golden 与 packed
block golden 一致。这样可以提前发现 block 内 gate、HC、MoE combine 的语义偏差。

### 4. `models/block.py` 的 selected decode pre/post kernel

按 block 形态逐个新增：

1. `swa_hash_selected_decode_pre_moe_fwd` / `swa_hash_selected_decode_post_moe_fwd`
2. `csa_hash_selected_decode_pre_moe_fwd` / `csa_hash_selected_decode_post_moe_fwd`
3. `hca_topk_selected_decode_pre_moe_fwd` / `hca_topk_selected_decode_post_moe_fwd`
4. `csa_topk_selected_decode_pre_moe_fwd` / `csa_topk_selected_decode_post_moe_fwd`

每完成一个 block 形态的 pre/post 两段，就同步 Ascend 验证对应 case。不要同时
修改四个形态后再统一验证。

### 5. 性能采集

selected decode block 全部通过后，使用现有 profile/swimlane 流程对比：

```text
packed decode block
selected decode block
```

需要关注两类数据：

- device kernel time 是否下降；
- block 输入 tensor 总量是否显著下降。

selected-expert 的主要收益来自 routed expert 权重从约 12 GiB 降到约 288 MiB。
如果只看单个 device kernel swimlane，收益可能没有 runner 端到端明显，因为
swimlane 不包含完整 host -> NPU 权重传输成本。

## 风险点

1. selected 权重顺序必须和 gate 输出顺序一致。

   kernel 默认 `selected_w*_t[k]` 就是 `indices[..., k]` 对应的 expert。如果 host
   侧去重或排序 expert id，kernel 接口就必须额外传入 `selected_eids` 并做查找。
   decode 最简单稳定的方案是不去重、不排序，直接按 top-k 顺序传 6 组权重。

2. hash route 和 topk route 都需要支持。

   hash route 的 `indices` 来自 `tid2eid[input_ids]`，topk route 的 `indices`
   来自 gate top-k。两者 selected 权重构造方式相同，但 gate 输入不同。

3. selected-expert 只优先支持 decode。

   prefill 中多个 token 的 selected expert 并集可能接近 256，收益不稳定，而且
   kernel 接口需要动态 selected expert 数。当前不建议改 prefill。

4. standalone runner 的 TensorSpec 无法天然表达“selected 权重依赖 gate 输出”。

   因此 `models/moe.py` 和 `models/block.py` 的 PyPTO standalone 用例可以只验证
   selected kernel 与 selected golden 一致；与 packed/官方语义一致的验证放在
   pytest 中完成。

5. 现有 packed path 必须保留。

   selected-expert 是 decode 优化路径，不应破坏已经跑通的 packed-expert 全量推理。
