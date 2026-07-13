# MoE Gate

## 模块定位

Gate 是 Mixture-of-Experts（MoE）的 routed-expert 选择组件。它接收 Block 中经过 FFN
Hyper-Connection pre 和 `ffn_norm` 后的 hidden state，为每个 token 从 256 个 routed
experts 中选择 6 个，并输出对应的 FP32 routing weights。

```text
FFN normalized input x [1,S,4096], BF16
+ gate projection weight [4096,256], BF16
  -> logits [1,S,256], FP32
  -> sqrt(softplus(logits)) scores [1,S,256], FP32
  ├─ hash layers 0..2
  │    + input_ids + tid2eid
  │    -> predetermined expert indices [1,S,6]
  └─ score-routed layers 3..42
       + expert bias [256]
       -> biased-score Top-K indices [1,S,6]
  -> gather unbiased scores
  -> normalize selected scores and multiply route_scale=1.5
  -> weights [1,S,6], FP32
```

Gate 只决定 routed experts 及其权重，不执行 expert FFN，也不选择 shared expert。当前
模型始终额外执行一个 shared expert；routed/shared expert 的计算与合并属于后续 MoE
组件。

根据 [`models/config.py`](../../models/config.py)，0-based layer 0、1、2 使用 hash
routing，layer 3 到 42 使用 score-based Top-K routing。该选择只由 layer id 与
`n_hash_layers=3` 决定，和当前层使用 SWA、CSA 或 HCA Attention 无直接关系。

## 官方模型中的 Gate

[`official/model.py`](../../official/model.py) 的 `Gate` 包含一组公共 projection
weight，以及按 routing 类型二选一的参数：

| 官方参数 | Shape | Dtype/属性 | 使用路径 |
|---|---:|---|---|
| `weight` | `[256,4096]` | 可学习参数 | 所有层 |
| `tid2eid` | `[129280,6]` | INT32、`requires_grad=False` | Hash layers |
| `bias` | `[256]` | FP32、可学习参数 | Score-routed layers |

官方先计算 256 个 expert logits。`score_func` 支持 `softmax`、`sigmoid` 和
`sqrtsoftplus`；当前模型配置固定使用：

$$
s_e = \sqrt{\operatorname{softplus}(l_e)}
$$

Hash routing 的 expert indices 完全由当前 token id 查表得到：

$$
I = tid2eid[input\_id]
$$

Score-based routing 则使用 `scores + bias` 选择 Top-K：

$$
I = \operatorname{TopK}(s+b, 6)
$$

Bias 只改变 expert selection，不改变 routing weight。两条路径都从未加 bias 的原始
score 中 gather 被选 expert 的值。因为当前 `score_func != softmax`，官方对 6 个
selected scores 归一化，再乘 route scale：

$$
w_k = 1.5 \cdot \frac{s_{I_k}}{\sum_{j=0}^{5}s_{I_j}}
$$

因此每个 token 的 routing weights 之和为 1.5（忽略浮点舍入）。官方 `Gate.forward`
返回 `(weights, indices)`，随后 `MoE.forward()` 按 expert id 调度 routed expert，并把
各 route output 与 shared-expert output 相加。

## PyPTO kernel 实现

[`models/gate.py`](../../models/gate.py) 为两种 routing 提供独立 inline kernel：

| 符号 | 类型 | 职责 |
|---|---|---|
| `_sqrt_softplus_scores` | `@pl.jit.inline` | 对 FP32 logits 计算稳定形式的 sqrt-softplus |
| `gate_hash_fwd` | `@pl.jit.inline` | 按 `input_ids -> tid2eid` 选择 6 个 experts 并计算 weights |
| `gate_topk_fwd` | `@pl.jit.inline` | 按 biased score 重复 argmax 选择 6 个 experts 并计算 weights |
| `gate_hash_test` | `@pl.jit` | Hash routing standalone 验收 wrapper |
| `gate_topk_test` | `@pl.jit` | Top-K routing standalone 验收 wrapper |
| `golden_gate_forward` | PyTorch golden | 两种路径共用的 Gate 参考实现 |
| `golden_gate_hash` | PyTorch wrapper | Hash golden 入口 |
| `golden_gate_topk` | PyTorch wrapper | Top-K golden 入口 |
| `build_gate_hash_specs` | Host spec builder | 构造 hash routing tensors |
| `build_gate_topk_specs` | Host spec builder | 构造 score-based routing tensors |

两个 kernel 都直接调用 [`models/linear.py`](../../models/linear.py) 中的
`linear_4096_to_256_fp32`。完整模型的上层 kernel 会直接复用 `gate_hash_fwd` 或
`gate_topk_fwd`；调用关系分别位于 [`models/moe.py`](../../models/moe.py) 和
[`models/split_block.py`](../../models/split_block.py)。本文不展开上层 MoE dispatch
或 Block 拆分流程。

当前 PyPTO Gate 接口返回 `(indices, weights)`，顺序与官方 Python API 的
`(weights, indices)` 相反；调用方均按当前 kernel 接口绑定，不依赖 tuple 名义顺序。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `linear(x.float(), weight.float())` | `linear_4096_to_256_fp32` | 直接调用：BF16 operands、FP32 accumulation/output |
| `F.softplus(scores).sqrt()` | `_sqrt_softplus_scores` | 语义等价：数值稳定形式 |
| Hash `tid2eid[input_ids]` | `gate_hash_fwd` table gather | 融合内联 |
| Score route `scores + bias` | `gate_topk_fwd::score_work` | 融合内联 |
| `scores.topk(6)` | 6 次 `row_argmax` + selected mask | 语义等价 |
| 从 unbiased `original_scores` gather | `picked_scores` | 语义等价：bias 不进入 weight |
| Selected-score normalization | `row_sum` + per-route divide | 融合内联 |
| `weights *= route_scale` | 编译期常量 `ROUTE_SCALE=1.5` | 语义等价 |
| `score_func="softmax"` | 无 | 不支持或未执行：当前配置固定 sqrtsoftplus |
| `score_func="sigmoid"` | 无 | 不支持或未执行：当前配置固定 sqrtsoftplus |
| 官方 `(weights, indices)` return order | 当前 `(indices, weights)` | 接口差异：tensor 语义一致 |
| 完整模型中的 Gate 计算 | `gate_hash_fwd` / `gate_topk_fwd` | 上层模型 kernel 直接调用 |

## 数据接口

### 公共接口

两条路径共用：

```text
x:          [1,S,4096], BF16
gate_w_t:   [4096,256], BF16
indices:    [1,S,6],    INT32
weights:    [1,S,6],    FP32
```

Batch 固定为 1，`S` 是动态 token 维。`x` 是 FFN normalized hidden state，不是
Attention 输入。`gate_w_t` 是 checkpoint `[256,4096]` weight 的转置 runtime
layout。Kernel-local `logits` 和 `scores` 均为 `[1,S,256]` FP32 scratch。

`indices` 中每个 expert id 必须位于 `[0,256)`。Indices 的第 `k` 项与 weights 的第
`k` 项组成同一 route。

### Hash routing

```text
tid2eid:   [129280,6], INT32
input_ids: [1,S],      INT64
```

`input_ids` 必须位于 `[0,129280)`，`tid2eid` 每项必须是有效 routed expert id。当前
kernel 不接收 `gate_bias`。虽然 expert indices 已由查表确定，Gate 仍计算全部 256 个
scores，以便为选中的 6 个 experts 生成 input-dependent routing weights。

### Score-based Top-K routing

```text
gate_bias: [256], FP32
```

该路径不接收 `input_ids` 或 `tid2eid`。Bias 只参与 selection；输出 weights 使用
unbiased score。当前 Top-K 固定为 6，并通过 `TOPK_PAD=8` 为内部临时 tensor 提供
满足最小访问宽度的 padding。

### State 与 runtime 边界

Gate 自身没有持久 state、cache 或跨 step scratch。它只生成当前输入对应的
`indices` 和 `weights`，不拥有后续 expert dispatch、expert weight 或 output
aggregation 的生命周期。

## 实现方式

### Projection 与 sqrt-softplus

`linear_4096_to_256_fp32` 使用 BF16 activation/weight 和 FP32 matmul accumulation，
直接输出 FP32 logits，不经过中间 BF16 rounding。

`_sqrt_softplus_scores` 每次最多处理 16 个 tokens，并使用以下稳定公式避免直接计算
大正数的 exponential：

$$
\operatorname{softplus}(z) = \max(z,0) + \log(1 + e^{-|z|})
$$

随后在 FP32 中取平方根。Dynamic sequence tail 使用 `valid_shape`，并在逐行 assemble
前通过 `fillpad` 物化无效行。

### Hash routing

Kernel 对每个 token 读取 `tid2eid[input_id,0:6]`，按 table 顺序写出 indices，并从
256-wide score row gather 对应的 6 个 unbiased scores。内部 `picked_scores` shape
为 `[8,8]`，未使用列为零；`row_sum` 得到 6 项之和，随后逐项除以该分母并乘 1.5。

Hash routing 不验证 table 中是否有重复 expert id，也不按 score 对 table 结果排序；
`tid2eid` 的顺序就是 route 维顺序。

### Score-based Top-K routing

Kernel 把 `scores + gate_bias` 放入 `[8,256]` work tensor 的第一行，其余行初始化为
FP32 最小有限值。随后重复 6 次：

1. `row_argmax` 取得当前最高 biased score 的 expert id；
2. 从原始 unbiased score row gather routing score；
3. 写出 expert id 和 picked score；
4. 把已选位置替换为 `NEG_INF`，避免重复选择。

最后仅对 6 个 unbiased picked scores 归一化并乘 route scale。由于 sqrt-softplus
scores 为正，正常有限输入下分母为正。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、256 routed experts、Top-K 6 和 vocabulary 129280，
  不是任意尺寸的 Gate API；
- 当前只实现 `sqrtsoftplus` score function；官方 `softmax` 和 `sigmoid` 分支不执行；
- 当前 route scale 固定为 1.5，且始终归一化 selected scores；
- Hash routing 固定用于 layer 0、1、2，其余 40 层使用 biased Top-K；
- Hash table 不要求按 score 排序，也不会被 Gate 修改；
- Top-K bias 只影响 indices，不影响 weights；
- 当前 Gate kernel 不执行 expert FFN 或 routed/shared output aggregation；
- Gate 无持久 state，两种 routing kernel 均使用动态 sequence 维；
- 当前 kernel 接口的 output tuple 顺序与官方 Python API 不同；
- Hash routing 的 `input_ids` 和 `tid2eid` 条目范围由调用方保证。

## Golden 参考实现

`models/gate.py::golden_gate_forward` 从 BF16 `x`、BF16 transposed gate weight 和路径
专用 input snapshot 开始。它使用 FP32 `torch.matmul` 计算 logits，再执行
`torch.sqrt(F.softplus(logits))`。

Hash golden 用 `tid2eid[input_ids]` 直接生成 indices；Top-K golden 对
`scores + gate_bias` 调用 `torch.topk(6)`。两者都从 unbiased scores gather weights，
按 selected 维归一化并乘 `ROUTE_SCALE`，最后写出 INT32 indices 与 FP32 weights。

`golden_gate_hash` 和 `golden_gate_topk` 只是固定 `hash_route` 参数的 wrapper。Golden
不执行 routed/shared experts 或 output aggregation。

## 精度验收标准

两类 standalone case 使用相同标准：

| 输出 | 验收方式 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---|---:|---:|---:|
| `indices` | 精确 expert id 与顺序 | `1e-5` | `1e-5` | 0 |
| `weights` | `ratio_allclose` | `1e-4` | `1/128` | `0.001` |

`indices` 使用默认 `torch.allclose`。因为输入为 INT32，任何至少 1 的 expert-id 差异
都会超出 `1e-5` 容差，因此等价于逐元素精确比较。

Weights 的逐元素容差为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的 weight 元素超出该条件。Actual weights 中出现任何 NaN 或 Inf
都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上同时验证 hash 与 score-based Top-K routing：

```bash
python models/gate.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

使用非 16 对齐 sequence length 验证 dynamic tail：

```bash
python models/gate.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13
```

当前入口每次都会依次执行 `gate-hash` 与 `gate-topk`，没有单独的 case 选择参数。如需
仅检查编译，可增加 `--compile-only`；如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`，该参数会直接传入 PyPTO `RunConfig`。

Host-side Gate golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_gate.py
```

## 集成验证范围

### 独立 kernel 验收

`models/gate.py::main()` 分别编译和执行 `gate_hash_test` 与 `gate_topk_test`，比较
INT32 indices 和 FP32 weights。

[`test_gate.py`](../../tests/models/test_gate.py) 使用缩小 hidden/expert/vocabulary
shape，在 sequence length 1、3、13 上逐元素比较当前 golden 与官方 `Gate` 的 hash
和 biased Top-K 路径。该 host test 不执行 PyPTO NPU kernel，不能替代 standalone
实机验收。

### MoE 与 Block 集成

- [`test_moe.py`](../../tests/models/test_moe.py) 覆盖 Gate 进入完整 MoE 后的 routing
  语义；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 FFN norm、Gate、MoE 和
  Hyper-Connection 的 Block 集成；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖拆分 Block 路径中的
  Gate output 及其与完整 Block 的对应关系。

这些集成测试验证 Gate output 进入上层模型后的语义，但不能替代 `models/gate.py`
的 standalone 实机验收。
