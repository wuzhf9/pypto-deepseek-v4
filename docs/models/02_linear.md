# Linear

## 模块定位

Linear 是 DeepSeek V4 Flash 中使用最广泛的基础计算之一，负责 Attention、Indexer、
Compressor、MoE Gate 和 Expert 的投影。对于输入 $x$ 和官方权重 $W$，无 bias 的
线性计算为：

$$
y = xW^T
$$

当前 PyPTO 接口接收预先转置的 `weight_t = Wᵀ`，因此 kernel 内直接计算：

$$
y = x \cdot \text{weight\_t}
$$

[`models/linear.py`](../../models/linear.py) 不是任意 shape 的通用 Linear API，而是
根据 [`models/config.py`](../../models/config.py) 中的模型维度，为主干推理路径
提供 11 个固定输入/输出尺寸的 BF16 matmul kernel。

## 官方模型中的 Linear

[`official/model.py`](../../official/model.py) 提供以下通用实现：

- `linear()`：根据 weight dtype 分派到 BF16 `F.linear`、FP8 GEMM 或 FP4 GEMM；
- `Linear`：保存完整 weight 的基础 module；
- `ColumnParallelLinear`：按输出维度切分 tensor-parallel weight；
- `RowParallelLinear`：按输入维度切分 weight，并对局部输出执行 all-reduce。

官方 `linear()` 接受可选 bias 参数，但当前模型调用要求 `bias is None`。主模型中的
Linear 主要分布如下：

| 官方模块 | Weight/计算 | 逻辑尺寸 |
|---|---|---:|
| `Attention.wq_a` | hidden 到 Q LoRA | 4096 → 1024 |
| `Attention.wq_b` | Q LoRA 到所有 query heads | 1024 → 32768 |
| `Attention.wkv` | hidden 到共享 KV | 4096 → 512 |
| `Attention.wo_a` | grouped attention output projection | 每组 4096 → 1024，共 8 组 |
| `Attention.wo_b` | flattened O LoRA 到 hidden | 8192 → 4096 |
| `Indexer.wq_b` | Q LoRA 到 index heads | 1024 → 8192 |
| `Indexer.weights_proj` | hidden 到 index-head weights | 4096 → 64 |
| Attention `Compressor.wkv/wgate`，ratio 4 | hidden 到 overlap projection | 4096 → 1024 |
| Attention `Compressor.wkv/wgate`，ratio 128 | hidden 到 projection | 4096 → 512 |
| Indexer `Compressor.wkv/wgate`，ratio 4 | hidden 到 overlap projection | 4096 → 256 |
| `Gate.weight` | hidden 到 routed-expert scores | 4096 → 256 |
| `Expert.w1/w3` | hidden 到 MoE intermediate | 4096 → 2048 |
| `Expert.w2` | MoE intermediate 到 hidden | 2048 → 4096 |
| `MTPBlock.e_proj/h_proj` | hidden 到 hidden | 4096 → 4096 |

`Block.hc_pre`、`ParallelHead.hc_head` 和 language-model head 也包含线性计算，但
它们具有 HC padding、仅取最后 token、超大 vocabulary 输出等专用接口，不属于
`models/linear.py` 的固定 kernel family。

## PyPTO kernel 实现

`models/linear.py` 提供以下 `@pl.jit.inline` kernel。每个 kernel 都有同名 `_test`
顶层 wrapper 和对应的 `build_*_specs`，用于独立编译及 golden 验收。

| Inline kernel | 输出 dtype | 验证 wrapper | Tensor spec builder |
|---|---|---|---|
| `linear_4096_to_64` | BF16 | `linear_4096_to_64_test` | `build_4096_to_64_specs` |
| `linear_4096_to_512` | BF16 | `linear_4096_to_512_test` | `build_4096_to_512_specs` |
| `linear_4096_to_512_fp32` | FP32 | `linear_4096_to_512_fp32_test` | `build_4096_to_512_fp32_specs` |
| `linear_4096_to_256_fp32` | FP32 | `linear_4096_to_256_fp32_test` | `build_4096_to_256_fp32_specs` |
| `linear_4096_to_1024` | BF16 | `linear_4096_to_1024_test` | `build_4096_to_1024_specs` |
| `linear_4096_to_1024_fp32` | FP32 | `linear_4096_to_1024_fp32_test` | `build_4096_to_1024_fp32_specs` |
| `linear_4096_to_2048` | BF16 | `linear_4096_to_2048_test` | `build_4096_to_2048_specs` |
| `linear_1024_to_8192` | BF16 | `linear_1024_to_8192_test` | `build_1024_to_8192_specs` |
| `linear_1024_to_32768` | BF16 | `linear_1024_to_32768_test` | `build_1024_to_32768_specs` |
| `linear_2048_to_4096` | BF16 | `linear_2048_to_4096_test` | `build_2048_to_4096_specs` |
| `linear_8192_to_4096` | BF16 | `linear_8192_to_4096_test` | `build_8192_to_4096_specs` |

带 `_fp32` 后缀的三个 kernel 仍使用 BF16 输入和 BF16 weight，只是不在写回前将
FP32 accumulator 转换为 BF16。它们用于 Compressor 和 Gate 后续需要 FP32
pooling、softmax 或路由分数的路径。

## 官方模块到当前实现的映射

| 官方计算 | PyPTO 实现 | 关系 | 集成位置 |
|---|---|---|---|
| `Attention.wq_a` | `linear_4096_to_1024` | 直接调用 | `models/attention_qkv.py` |
| `Attention.wq_b` | `linear_1024_to_32768` | 直接调用 | `models/attention_qkv.py` |
| `Attention.wkv` | `linear_4096_to_512` | 直接调用 | `models/attention_qkv.py` |
| `Attention.wo_a` | grouped 8 × (4096 → 1024) einsum | 语义等价/专用实现 | `models/attention_out.py` |
| `Attention.wo_b` | `linear_8192_to_4096` | 直接调用 | `models/attention_out.py` |
| `Indexer.wq_b` | `linear_1024_to_8192` | 直接调用 | `models/indexer.py` |
| `Indexer.weights_proj` | `linear_4096_to_64` | 直接调用 | `models/indexer.py` |
| Ratio-4 Attention `Compressor.wkv/wgate` | `linear_4096_to_1024_fp32` | 直接调用 | `models/compressor_ratio4.py` |
| Ratio-128 Attention `Compressor.wkv/wgate` | `linear_4096_to_512_fp32` | 直接调用 | `models/compressor_ratio128.py` |
| Ratio-4 Indexer `Compressor.wkv/wgate` | `linear_4096_to_256_fp32` | 直接调用 | `models/compressor_ratio4.py` |
| `Gate.weight` | `linear_4096_to_256_fp32` | 直接调用 | `models/gate.py` |
| Shared `Expert.w1/w3` | `linear_4096_to_2048` | 直接调用 | `models/expert.py::expert_shared_fwd` |
| Shared `Expert.w2` | `linear_2048_to_4096` | 直接调用 | `models/expert.py::expert_shared_fwd` |
| Routed `Expert.w1/w2/w3` 独立算子路径 | `linear_4096_to_2048`、`linear_2048_to_4096` | 直接调用，完整模型未使用 | `models/expert.py::expert_routed_fwd` |
| Routed `Expert.w1/w2/w3` 完整 MoE 路径 | 相同尺寸的 route-major matmul | 融合内联 | `models/moe.py` |
| `MTPBlock.e_proj/h_proj` | 无 4096 → 4096 kernel | 不支持/未执行 | 当前 Runner 不执行 MTP |
| `Block.hc_pre` linear | HC 专用 matmul | 语义等价/专用实现 | `models/hc.py` |
| `ParallelHead.hc_head` linear | HC head 专用 matmul | 语义等价/专用实现 | `models/head.py::hc_head_fwd` |
| Language-model head | last-token 4096 → 129280 matmul | 语义等价/专用实现 | `models/head.py::lm_head_fwd` |

`models/moe.py` 从 `models.linear` 复用 tiling 常量，但 routed-expert 主路径并不调用
独立 Linear 函数。它根据 expert id 或 selected-expert 顺序选择 weight，并把三次
matmul、SwiGLU 和 route weight 组合在同一个 MoE kernel 中。

## 数据接口

所有独立 Linear kernel 使用相同的基本接口：

```text
x:        [1, S, K], BF16
weight_t: [K, N],    BF16
out:      [1, S, N], BF16 或 FP32
```

其中：

- Batch 固定为 1；
- `S` 是动态 sequence/token 维度；
- `K` 和 `N` 必须是该 kernel 名称规定的固定模型尺寸；
- 所有 matmul 使用 FP32 accumulator；
- BF16 输出使用 round-to-nearest 模式转换，`_fp32` 变体直接写回 FP32；
- 接口不接受 bias、量化 scale、持久 state 或 cache。

官方 checkpoint 中的 Linear weight 逻辑布局是 `[N, K]`。Serving 层通过
[`serving/weight_loader.py`](../../serving/weight_loader.py) 将 weight 读取或反量化
为 BF16，再生成连续的 `[K, N]` `linear_t` runtime layout。固定非 routed-expert
layout 会由 runtime weight cache 管理，并由
[`serving/runner.py`](../../serving/runner.py) 绑定到完整模型 kernel。

独立 Linear kernel 不拥有跨调用中间 tensor。`out` 由调用方提供，输入展平 view、
输出 tile 和 FP32 accumulator 均属于单次调用内部 scratch。Routed-expert packed
weight 和 selected-expert staging 属于 MoE/serving 路径，不属于独立 Linear 接口。

## Kernel 实现方式

11 个 kernel 共享以下基础 tiling：

```text
T_TILE = 16
K_TILE = 128
O_TILE = 32
OUT_GROUP = 2
```

`1024 → 8192` 和 `1024 → 32768` 两个大输出 kernel 使用单独的输出 tiling：

```text
ATTN_Q_OUT_TILE = 64
ATTN_Q_OUT_GROUP = 2
```

公共计算流程为：

1. 将输入和输出分别 reshape 为 `[S, K]` 和 `[S, N]`；
2. 每个 token block 最多处理 16 个 token；
3. 将 reduction 维度按 128 切分；
4. 使用第一个 K block 创建 FP32 accumulator；
5. 通过 pipeline 累加其余 K block；
6. 按输出 block 并行计算并写回；
7. BF16 变体在写回前执行 round-to-nearest cast，FP32 变体保留 accumulator dtype。

动态 `S` 的尾块通过 `valid_shape` 处理，因此 sequence length 不要求是 16 的整数倍。
模型配置在 import 时检查 K/N 与 tile 的整除关系。Linear kernel 本身不执行
activation、normalization、routing 或 bias addition。

## 实现差异与限制

当前实现与官方通用 Linear 路径的主要差异如下：

- 官方 `Linear` 接受构造时决定的任意合法维度；当前 PyPTO kernel 仅覆盖主干模型
  实际使用的 11 种固定尺寸；
- 官方 weight 使用 `[N, K]` 并通过 `F.linear` 隐式转置；当前 kernel 接收 host
  预先生成的连续 `[K, N]` layout；
- 官方 `linear()` 可以分派到 FP4/FP8 GEMM；当前 kernel 只接收 BF16 weight，量化
  checkpoint 在 serving 加载阶段反量化；
- 官方包含 tensor-parallel `ColumnParallelLinear` 和 `RowParallelLinear`；当前
  PyPTO 路径是单卡实现，不在 Linear kernel 内执行 shard 或 all-reduce；
- 当前接口不支持 bias；官方主模型同样要求 Linear 调用不带 bias；
- Grouped `wo_a`、HC projection、language-model head 和 routed-expert matmul 使用
  专用或融合实现；
- 4096 → 4096 的 MTP projection 没有独立 kernel，且 MTP 不属于当前 Runner 路径。

## Golden 参考实现

`models/linear.py::golden_linear` 是 11 个尺寸共用的 PyTorch 参考实现：

```python
x = tensors["x"].float()
weight_t = tensors["weight_t"].float()
tensors["out"][:] = torch.matmul(x, weight_t).to(tensors["out"].dtype)
```

Golden 与 kernel 使用相同的 BF16 输入 snapshot 和 `[K, N]` BF16 weight。Matmul
在 FP32 中计算，并按照当前 case 的输出 spec 转为 BF16 或保留 FP32。独立 Linear
没有需要忽略的输出、mask 或有效区域；动态 sequence tail 也参与完整比较。

## 精度验收标准

11 个独立 kernel 使用相同的验收标准：

| 项目 | 标准 |
|---|---:|
| Absolute tolerance | `1e-4` |
| Relative tolerance | `1 / 128`，约为 `0.0078125` |
| 允许超出容差的元素比例 | `0` |
| NaN/Inf | 不允许 |

每个输出元素必须满足：

```text
abs(actual - expected) <= 1e-4 + (1 / 128) * abs(expected)
```

该标准同时用于 BF16 输出和三个 FP32 输出变体，且不允许忽略任何超出容差的元素。

## 验收方法

`models/linear.py` 的命令行入口会依次编译并验证全部 11 个 case。Ascend A2/A3
实机命令为：

```bash
python models/linear.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

使用非 tile 对齐的 sequence length 验证动态尾块：

```bash
python models/linear.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

当前没有 `tests/models/test_linear.py`。11 个固定尺寸的直接编译、NPU 执行和
golden 比较统一由 `models/linear.py::main()` 完成。

### Host 侧官方语义覆盖

Linear 所在的上层数据流由以下测试覆盖：

- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py)、
  [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 和
  [`test_attention_hca.py`](../../tests/models/test_attention_hca.py)：Attention
  Q/KV/output projection；
- [`test_indexer.py`](../../tests/models/test_indexer.py)：Indexer query 和
  weight projection；
- [`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py) 和
  [`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py)：
  FP32-output Compressor projection；
- [`test_gate.py`](../../tests/models/test_gate.py)：routed-expert score projection；
- [`test_expert.py`](../../tests/models/test_expert.py) 和
  [`test_moe.py`](../../tests/models/test_moe.py)：shared/routed expert projection；
- [`test_block.py`](../../tests/models/test_block.py) 和
  [`test_split_block.py`](../../tests/models/test_split_block.py)：完整层和
  selected-expert decode 中的组合路径。

这些 host 测试比较组合模块的 PyTorch golden 与 `official/model.py` 语义，不编译
或执行独立 NPU Linear kernel，因此不能替代 `models/linear.py::main()` 的验收。

### Serving 权重布局和完整模型集成

[`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 Linear weight
的加载、反量化、转置和 runtime layout。完整模型验证进一步覆盖固定 layout 的
device residency、逐层绑定以及 routed-expert packed/selected weight 路径，但不能
替代单一 Linear case 的数值误差定位。
