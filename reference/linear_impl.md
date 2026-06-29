# DeepSeek V4 Flash Linear Shape Plan

本文档记录 `../deepseek_v4_flash/inference/model.py` 中需要 PyPTO 实现的
Linear 计算形状。当前目标不实现 MTP，因此 `MTPBlock.e_proj/h_proj` 不列入实现范围。

## 权重布局

官方 PyTorch 权重使用 `F.linear` 语义：

```text
input:  [..., in_features]
weight: [out_features, in_features]
output: [..., out_features]
```

PyPTO linear kernel 统一接收加载阶段预转置后的权重：

```text
x:        [1, S_DYN, in_features]
weight_t: [in_features, out_features]
out:      [1, S_DYN, out_features]
```

也就是说，权重加载逻辑负责把官方 checkpoint 中的 `[out, in]` 转成
`[in, out]`，kernel 内部只做：

```text
out = x @ weight_t
```

这样可以对齐 `../pypto-serving/pypto-lib/models/deepseek/v4/*qkv_proj_rope.py`
中 `wq_a: [D, Q_LORA]` 的写法，避免在 kernel 内使用 `b_trans=True` 或
`pl.transpose`。当前不实现 bias；`model.py` 中普通 `Linear` 调用均未启用
bias。MoE gate 的 `gate.bias` 是 router 逻辑的一部分，应在 MoE/gate 模块中单独处理。

## Shape 列表

| 官方 Shape `[out, in]` | Kernel `weight_t` Shape `[in, out]` | 上层模块 | `model.py` 来源 | 用途 |
| --- | --- | --- | --- | --- |
| `[64, 4096]` | `[4096, 64]` | `Indexer` | `weights_proj`, line 394 | indexer 对每个 head 的 score 权重投影 |
| `[256, 4096]` | `[4096, 256]` | `Gate` | `Gate.weight`, line 557; `linear(...)`, line 565 | MoE router logits |
| `[256, 4096]` | `[4096, 256]` | `Indexer.Compressor` | `Compressor.wkv/wgate`, lines 297-298 via line 398 | ratio=4 indexer compressor，`head_dim=128`、`coff=2` |
| `[512, 4096]` | `[4096, 512]` | `Attention` | `wkv`, line 460 | attention window KV projection |
| `[512, 4096]` | `[4096, 512]` | `Compressor` | `wkv/wgate`, lines 297-298 | ratio=128 compressor，`head_dim=512`、`coff=1` |
| `[1024, 4096]` | `[4096, 1024]` | `Attention` | `wq_a`, line 457 | low-rank Q 第一段投影 |
| `[1024, 4096]` | `[4096, 1024]` | `Compressor` | `wkv/wgate`, lines 297-298 | ratio=4 overlap compressor，`head_dim=512`、`coff=2` |
| `[8192, 1024]` | `[1024, 8192]` | `Indexer` | `wq_b`, line 393 | indexer Q 从 q_lora 到 `index_n_heads * index_head_dim` |
| `[32768, 1024]` | `[1024, 32768]` | `Attention` | `wq_b`, line 459 | attention Q 从 q_lora 到 `n_heads * head_dim` |
| `[8192, 4096]` | `[4096, 8192]` | `Attention` | `wo_a`, line 462; grouped use at lines 539-542 | grouped attention output 第一段投影 |
| `[4096, 8192]` | `[8192, 4096]` | `Attention` | `wo_b`, line 463 | attention output 第二段投影 |
| `[2048, 4096]` | `[4096, 2048]` | `Expert` / shared expert | `w1`, line 591; `w3`, line 593 | MoE SwiGLU gate/up projection |
| `[4096, 2048]` | `[2048, 4096]` | `Expert` / shared expert | `w2`, line 592 | MoE down projection |
| `[24, 16384]` | `[16384, 24]` | `Block` HC pre | `hc_attn_fn/hc_ffn_fn`, lines 666-667; `F.linear`, line 678 | HC pre-mixing logits |
| `[4, 16384]` | `[16384, 4]` | `ParallelHead` HC head | transformer `hc_head_fn`, line 797; `F.linear`, line 732 | final HC head mixing logits |
| `[129280, 4096]` | `[4096, 129280]` | `ParallelHead` LM head | `weight`, line 713; `F.linear`, line 716 | final logits projection |

## 实现原则

- 不为每个业务名复制一份 kernel；按 shape family 在 `models/linear.py` 中提供入口。
- `linear.py` 中的所有入口都接收 `weight_t: [in, out]`，不接收官方 `[out, in]`
  权重。
- 权重加载代码负责完成一次性转置，并保证转置后的 tensor 为连续 BF16 布局。
- kernel 内不使用 `b_trans=True`、`pl.transpose` 或临时 transpose tile。
- 不在 `linear.py` 中实现 fp8/fp4、activation quant、scale 或 bias。
- 对于只处理最后 token 的 head 类路径，可以在对应模块中提供更窄的 wrapper，但底层
  linear 计算仍保持 `x @ weight_t` 语义。

## Kernel 计算结构

Linear 本质是：

```text
C[M, N] = A[M, K] @ B[K, N]
```

其中：

```text
A: x_flat   [tokens, in_features]
B: weight_t [in_features, out_features]
C: out_flat [tokens, out_features]
```

当前实现采用 `M + N` 切分：token 维按 `T_TILE` 分块，输出 channel 维按
`O_TILE` 或更大的输出分片分块，K 维在同一个 PyPTO scope 内循环累加。

核心结构对齐 pypto-serving 的 q projection：

```text
for token_block:
  valid_tok = min(T_TILE, tokens - token_block_start)

  for output_block_group in spmd:
    for output_block in group:
      acc = x_tile[:, 0:K_TILE] @ weight_t[0:K_TILE, out_block]
      for k_block:
        acc += x_tile[:, k_block] @ weight_t[k_block, out_block]

      acc_bf16 = cast(acc, BF16)
      for row in range(valid_tok):
        out_flat[token_block_start + row, output_block] = acc_bf16[row, :]
```

这样 matmul 仍然使用固定 `T_TILE=16` 的 tile，满足 Ascend AIC matmul 的行对齐
要求；但写回外部 dynamic output 时只写 `valid_tok` 行，避免尾块越界。

对于输出维度较小的 shape，可以先把多个输出分片累积到局部 FP32 buffer，再按
`valid_tok` 逐行写回。对于 `[1024, 32768]` 这类输出维度很大的 shape，不需要创建
完整 `[T_TILE, 32768]` 的局部 buffer，可以直接对每个输出分片完成 FP32 累加、cast
BF16，然后逐有效行写回。

## Matmul 切分方式

matmul 有 `M`、`K`、`N` 三个轴可以切分，非空组合共 7 种：

```text
1. M
2. K
3. N
4. M + K
5. M + N
6. K + N
7. M + K + N
```

当前目标是单卡正确性优先，不做性能优化，因此采用最直接的 `M + N` 切分：

```text
for m_block:
  for n_block:
    acc = 0
    for k_block:
      acc += A[m_block, k_block] @ B[k_block, n_block]
    C[m_block, n_block] = acc
```

这样只有 `K` 在同一个 PyPTO scope 内循环累加，不需要跨 core 或跨 kernel 的 partial
reduce，验证和调试都更简单。

## 已实现 Shape

`models/linear.py` 当前实现：

| Kernel | `x` Shape | `weight_t` Shape | `out` Shape |
| --- | --- | --- | --- |
| `linear_4096_to_512` | `[1, S_DYN, 4096]` | `[4096, 512]` | `[1, S_DYN, 512]` |
| `linear_4096_to_1024` | `[1, S_DYN, 4096]` | `[4096, 1024]` | `[1, S_DYN, 1024]` |
| `linear_1024_to_32768` | `[1, S_DYN, 1024]` | `[1024, 32768]` | `[1, S_DYN, 32768]` |

执行 `python models/linear.py ...` 会顺序验证所有已实现 shape；新增 shape 后应加入同一个
验证列表，避免只验证单个 kernel。

后续新增 shape 时应继续使用同一权重布局和 FP32 累加语义；如果某个输出维度过大，
可以把局部 buffer 改为输出分片 buffer，但不要回到 kernel 内转置权重的路径。

## 实现经验总结

### 权重转置应固定在加载阶段完成

PyPTO linear kernel 统一接收 `weight_t: [in, out]`。早期尝试过在 kernel 内使用
`b_trans=True` 或手动 transpose 官方 `[out, in]` 权重，但这会让每个 shape 的
切分和编译行为更难预测。当前约定是加载阶段一次性把权重转为连续 BF16
`[in, out]`，kernel 只实现 `x @ weight_t`。

### AIC matmul 的 token tile 不能随意改成 1

`S=1` 失败后曾尝试把 `[1024, 32768]` kernel 的 token tile 改成 1，从而避免尾块
多写行。但 Ascend AIC matmul 对左矩阵行数有 16 对齐要求，远端 C++ 编译会报：

```text
srcRow must be aligned to 16
dstRow must be aligned to 16
```

因此正确做法不是把 matmul 的 `T_TILE` 改成 1，而是保持 `T_TILE=16` 做计算，在最终
写回 dynamic output 时按 `valid_tok` 裁剪。

### `valid_shape` 只保护输入 tile，不会自动裁剪 `assemble`

`pl.slice(x_flat, [T_TILE, K_TILE], ..., valid_shape=[valid_tok, K_TILE])` 可以处理尾块
输入不足 16 行的问题，但它不会影响后续 `pl.assemble` 的写回范围。

如果直接把 `[16, O_TILE]` 或 `[16, ATTN_Q_OUT_TILE]` 写回到 `out_flat[t0:, o0:]`，当
`valid_tok < 16` 时会向真实输出写出有效 token 范围。`[1024, 32768]` 的输出维度很大，
`S=1` 时多写约 15 行、每行 32768 个 BF16 元素，容易稳定触发运行时错误 `507018`。
`[4096, 512]` 和 `[4096, 1024]` 也有相同语义风险，只是越界范围小，未必每次触发。

所有带动态 `S_DYN` 的 linear kernel 都应使用同一写回模式：

```text
acc_bf16 = cast(acc, BF16)
for row in range(valid_tok):
  out_row = acc_bf16[row:row+1, :]
  assemble(out_flat, out_row, [t0 + row, o0])
```

如果某个 kernel 先写入局部 FP32 buffer，也要在最终写回外部 output 时逐有效行写回，
不能整块 `[T_TILE, out_tile]` 写回。

### `[1024, 32768]` 需要按输出分片写回

`[1024, 32768]` 的 N 轴远大于其他已实现 shape。直接使用较大的输出 tile 会增加本地
buffer 压力；当前可运行写法使用：

```text
T_TILE = 16
K_TILE = 128
ATTN_Q_OUT_TILE = 64
ATTN_Q_OUT_GROUP = 2
```

计算结构是先按输出分片并行，K 轴在分片内循环累加，每个 `[16, 64]` 分片完成后立即
按 `valid_tok` 逐行写回。这个写法避免创建完整 `[16, 32768]` 的局部输出 buffer，也避免
尾块越界写回。

### 验证必须覆盖非对齐序列长度

只验证默认序列长度不能发现尾块写回问题。新增或修改 linear kernel 后至少要跑：

```bash
python models/linear.py -p a2a3 -d {} -s 1
python models/linear.py -p a2a3 -d {} -s 13
python models/linear.py -p a2a3 -d {} -s 128
```

其中 `S=1` 覆盖最短 decode 形态，`S=13` 覆盖非 16 对齐尾块，`S=128` 覆盖多个完整
token block。执行脚本会顺序验证所有已实现 shape，后续新增 shape 应加入同一个 cases
列表。
