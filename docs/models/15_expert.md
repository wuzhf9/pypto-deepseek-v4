# MoE Expert

## 模块定位

Expert 是 Mixture-of-Experts（MoE）中的 SwiGLU Feed-Forward Network。每个 Expert
包含两条 4096 → 2048 projection、一次带 clamp 的 SwiGLU，以及一条 2048 → 4096
projection。

```text
x [1,S,4096], BF16
+ w1_t [4096,2048], BF16
+ w3_t [4096,2048], BF16
  -> gate/up [1,S,2048], BF16
  -> gate upper clamp + up symmetric clamp
  -> SiLU(gate) * up, FP32
  ├─ shared expert: no routing weight
  └─ routed expert: * route weight [1,S,1], FP32
  -> hidden [1,S,2048], BF16
+ w2_t [2048,4096], BF16
  -> out [1,S,4096], BF16
```

当前每个模型层包含 256 个 routed experts 和 1 个 shared expert。Gate 为每个 token
选择 6 个 routed experts；shared expert 不经过 Gate，始终执行。本文描述单个 Expert
的数学计算和权重接口。Expert dispatch、6 条 routed output 的聚合以及 shared output
相加由 [`models/moe.py`](../../models/moe.py) 负责。

## 官方模型中的 Expert

[`official/model.py`](../../official/model.py) 的 `Expert` 包含三个无 bias 的 `Linear`：

| 官方参数 | Projection | Checkpoint shape | 当前 runtime transposed shape |
|---|---|---:|---:|
| `w1.weight` | Gate projection | `[2048,4096]` | `[4096,2048]` |
| `w3.weight` | Up projection | `[2048,4096]` | `[4096,2048]` |
| `w2.weight` | Down projection | `[4096,2048]` | `[2048,4096]` |

令 `L=swiglu_limit=10.0`。对 BF16 输入 `x`，当前配置下的官方计算为：

$$
g = \operatorname{FP32}(W_1x), \qquad
u = \operatorname{FP32}(W_3x)
$$

$$
\hat{g} = \min(g,L), \qquad
\hat{u} = \operatorname{clamp}(u,-L,L)
$$

$$
h = \operatorname{SiLU}(\hat{g})\odot\hat{u}
$$

Shared expert 调用 `Expert.forward(x, weights=None)`，直接把 `h` 转回输入 dtype 后送入
`w2`。Routed expert 额外接收每个 token 的标量 routing weight：

$$
h_{routed}=w_{route}\odot h
$$

Routing weight 在 `h` 转回 BF16 之前相乘，并通过最后一维 1 broadcast 到全部 2048
个 intermediate channels；随后 `w2` 产生 expert output。Gate 只做上界 clamp，不做
下界 clamp；Up 同时做上下界 clamp。

官方 `MoE.forward()` 按 expert id 聚集 token，向 routed `Expert` 传入匹配的 routing
weight，累加 routed outputs，最后加上 shared-expert output。官方还支持 FP4 expert
weight 和 Expert Parallel；这些不是当前 BF16、单卡 kernel 的执行方式。

## PyPTO kernel 实现

[`models/expert.py`](../../models/expert.py) 提供 shared 和单 routed expert 两套接口：

| 符号 | 类型 | 职责 |
|---|---|---|
| `expert_shared_fwd` | `@pl.jit.inline` | 执行无 routing weight 的 shared expert |
| `expert_routed_fwd` | `@pl.jit.inline` | 执行带 `[1,S,1]` FP32 routing weight 的单 routed expert |
| `expert_shared_test` | `@pl.jit` | Shared expert standalone 验收 wrapper |
| `expert_routed_test` | `@pl.jit` | Routed expert standalone 验收 wrapper |
| `_golden_expert` | PyTorch golden | 两种 Expert 共用的参考实现 |
| `golden_expert_shared` | PyTorch wrapper | Shared expert golden 入口 |
| `golden_expert_routed` | PyTorch wrapper | Routed expert golden 入口 |
| `build_expert_shared_specs` | Host spec builder | 构造 shared case tensors |
| `build_expert_routed_specs` | Host spec builder | 构造 routed case tensors |

两条 inline kernel 都调用 [`models/linear.py`](../../models/linear.py) 中的
`linear_4096_to_2048` 和 `linear_2048_to_4096`。Gate/up projection、SwiGLU、routing
weight 和 down projection 之间的 BF16 rounding boundary 与官方 BF16 路径对齐。

完整模型直接调用 `expert_shared_fwd` 计算 shared expert。Routed expert 的同等数学
计算融合在 [`models/moe.py`](../../models/moe.py) 的 MoE kernel 中；因此
`expert_routed_fwd` 是可独立使用和验收的单 Expert 实现，但当前完整模型主干不直接
调用该符号。本文不展开 MoE 内部的 expert dispatch 和 output aggregation。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `w1(x)` | `linear_4096_to_2048(x, w1_t)` | 直接调用；runtime weight 已转置 |
| `w3(x)` | `linear_4096_to_2048(x, w3_t)` | 直接调用；runtime weight 已转置 |
| Gate projection 转回输入 dtype | `gate` BF16 tensor | 语义等价的 rounding boundary |
| Up projection 转回输入 dtype | `up` BF16 tensor | 语义等价的 rounding boundary |
| `clamp(gate, max=L)` | `minimum(gate, limit)` | 融合内联；不做下界 clamp |
| `clamp(up, -L, L)` | `minimum(maximum(up, -limit), limit)` | 融合内联 |
| `F.silu(gate) * up` | `gate * sigmoid(gate) * up` | FP32 语义等价实现 |
| Routed `weights * hidden` | `row_expand_mul` / route scalar multiply | 在 hidden BF16 rounding 前融合 |
| Hidden 转回输入 dtype | `cast(..., BF16, mode="rint")` | 直接对应 |
| `w2(hidden)` | `linear_2048_to_4096(hidden, w2_t)` | 直接调用；runtime weight 已转置 |
| Shared `Expert.forward(..., None)` | `expert_shared_fwd` | 完整 MoE 直接调用 |
| 单 routed `Expert.forward(..., weight)` | `expert_routed_fwd` | 独立 kernel 可用；完整 MoE 未直接调用 |
| 完整模型中的 routed Expert 计算 | `models/moe.py` 中的融合实现 | 融合内联、语义等价；未直接调用 `expert_routed_fwd` |
| FP4 expert runtime | 无 | 不支持或未执行；host 侧准备 BF16 runtime weights |
| Expert Parallel / `all_reduce` | 无 | 不支持或未执行；当前模型为单卡逻辑 |

## 数据接口

### 单 Expert 公共接口

```text
x:    [1,S,4096],    BF16
w1_t: [4096,2048],   BF16
w2_t: [2048,4096],   BF16
w3_t: [4096,2048],   BF16
out:  [1,S,4096],    BF16
```

Batch 固定为 1，`S` 是动态 token 维。`x` 是经过 FFN Hyper-Connection pre 和
`ffn_norm` 后的 hidden state。`w1_t`、`w2_t` 和 `w3_t` 均为 checkpoint weight 的
转置 runtime layout。三个 projection 都没有 bias。

单 routed expert 额外接收：

```text
weights: [1,S,1], FP32
```

`weights[b,s,0]` 属于同一行 `x[b,s,:]`，并 broadcast 到该 token 的全部
intermediate channels。Shared expert 不接收 routing weight，也不使用隐式全 1
tensor。

Standalone kernel 内部生成以下动态 scratch：

```text
gate:   [1,S,2048], BF16
up:     [1,S,2048], BF16
hidden: [1,S,2048], BF16
```

SwiGLU tile 在 FP32 中计算，不作为接口 output。Expert 自身没有 KV cache、跨层 state
或跨 decode-step state，也不管理 routed expert 的 dispatch、权重 staging 或 cache
生命周期。

## 实现方式

### Projection 与 BF16 rounding boundary

Standalone 两条 kernel 先分别执行 `w1` 和 `w3` projection。Linear 使用 BF16
activation/weight 和 FP32 matmul accumulation，projection output 随后以 BF16 写入
`gate` 和 `up`。SwiGLU 读取这两个 BF16 snapshot 并转成 FP32，因此不会直接消费
projection 的未舍入 FP32 accumulator。

SwiGLU 与 routing weight 完成后，`hidden` 再以 round-to-nearest BF16 写回，最后才
执行 `w2`。这一边界意味着 routed weight 不能在不重新分析舍入语义的情况下移到
`w2` output 之后。

### Clamp 与 SwiGLU

SwiGLU 按 16-token × 32-channel tile 处理。Gate 只执行 `min(gate,10)`；负值保持
不变。Up 执行 `max(up,-10)` 后再 `min(...,10)`。Kernel 在 FP32 中显式计算：

$$
\operatorname{SiLU}(z)=z\cdot\frac{1}{e^{-z}+1}
$$

Routed path 使用 `row_expand_mul` 把 `[T,1]` FP32 weight 扩展到 `[T,32]` tile；shared
path 跳过该操作。两者随后都把 hidden tile 以 `rint` 转成 BF16。

Sequence tail 通过 `valid_shape` 限定有效 token 行，并用 `fillpad` 填充 tile 中无效
位置。有效行再 assemble 回动态 `[S,2048]` hidden tensor，因此 `S` 不要求 16 对齐。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、intermediate 2048 和 BF16 activation/weight；
- 当前 `swiglu_limit` 固定为 10.0，不提供 kernel runtime 参数；
- Gate 只做上界 clamp，Up 做 `[-10,10]` 对称 clamp；
- 当前 routed expert 的 routing weight 固定为 FP32，且必须在 hidden BF16 rounding 前
  相乘；
- `expert_routed_fwd` 描述单 routed expert；完整模型在 MoE kernel 内融合相同数学
  计算，不直接调用该符号；
- 当前不执行 FP4/FP8 expert kernel、激活量化或量化 GEMM；checkpoint 量化权重由
  host 加载路径转换为 BF16 runtime layout；
- 当前不实现 Expert Parallel、跨 rank expert shard 或 routed output `all_reduce`；
- Expert kernel 不检查 routing weight 的范围或归一化结果，这些由 Gate 和调用方保证；
- Expert 不负责 route selection、token dispatch、route aggregation 或 shared/routed
  output 合并。

## Golden 参考实现

`models/expert.py::_golden_expert` 直接从 BF16 `x` 和三组 BF16 transposed weights
开始。它用 FP32 `torch.matmul` 分别计算 gate/up projection，立即转成 BF16 snapshot
后再升回 FP32，然后执行与 kernel 相同的 clamp 和 `F.silu(gate) * up`。

Routed golden 在 FP32 hidden 上乘 `[1,S,1]` weight；shared golden 跳过该步骤。两者
都把 hidden 转为 BF16，再以 FP32 matmul 执行 down projection，并把 output 转为 BF16。
因此 golden 明确覆盖两次关键 BF16 rounding：gate/up projection output，以及
SwiGLU 后的 intermediate hidden。

`golden_expert_shared` 和 `golden_expert_routed` 只固定 `_golden_expert` 的 routed
分支，不执行 Gate、routed expert dispatch、route aggregation 或 shared/routed
合并。

## 精度验收标准

Shared 和 routed standalone case 对 `out` 使用相同标准：

| 输出 | 验收方式 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---|---:|---:|---:|
| `out` | `ratio_allclose` | `1e-4` | `1/128` | `0.001` |

逐元素容差为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的 BF16 output 元素超出该条件。Actual output 中出现任何 NaN 或 Inf
都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上同时验证 shared 和单 routed expert：

```bash
python models/expert.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

使用非 16 对齐 sequence length 验证 dynamic tail：

```bash
python models/expert.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13
```

当前入口每次都会依次执行 `expert-shared` 与 `expert-routed`，没有单独的 case 选择
参数。如需仅检查编译，可增加 `--compile-only`；如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`，该参数会直接传入 PyPTO `RunConfig`。

Host-side Expert golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_expert.py
```

## 集成验证范围

### 独立 Expert 验收

`models/expert.py::main()` 分别编译和执行 `expert_shared_test` 与
`expert_routed_test`，只比较单 Expert 的 BF16 `out`。

[`test_expert.py`](../../tests/models/test_expert.py) 使用缩小的 hidden/intermediate
shape 和 `swiglu_limit=1.5`，在 sequence length 1、3、13 上分别验证 shared 和
routed golden 与官方 `Expert` 完全一致。该 host test 不执行 PyPTO NPU kernel，
不能替代 standalone 实机验收。

### MoE 与 Block 集成

- [`test_moe.py`](../../tests/models/test_moe.py) 覆盖 Gate、routed/shared Expert 和
  output aggregation 的完整 MoE 语义；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 FFN Hyper-Connection、
  `ffn_norm`、Gate、Expert 和 MoE output 的 Block 集成；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖拆分 Block 路径中的
  Expert 计算及其与完整 Block 的对应关系。

这些集成测试验证 Expert 计算进入上层模型后的语义，但不能替代
`models/expert.py` 的 standalone 实机验收。
