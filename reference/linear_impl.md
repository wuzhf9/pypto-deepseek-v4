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
`O_TILE` 分块，K 维在同一个 PyPTO scope 内循环累加。

核心结构对齐 pypto-serving 的 q projection：

```text
for token_block:
  local_fp32 = create_tensor([T_TILE, out_features], FP32)

  for output_block_group in spmd:
    for output_block in group:
      acc = x_tile[:, 0:K_TILE] @ weight_t[0:K_TILE, out_block]
      for k_block:
        acc += x_tile[:, k_block] @ weight_t[k_block, out_block]
      local_fp32[:, out_block] = acc

  with pl.at(...):
    out_flat[token_block, :] = cast(local_fp32, BF16)
```

这样 matmul 结果先落到局部 FP32 buffer，再统一写回外部 output。这个写法与
`pypto-serving` 的 `qr_fp32[:, q_a_col0:q_a_col0+Q_LORA_TILE] = q_acc`
保持一致，也避免了直接从每个 matmul block 写外部 output 的运行时问题。

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

执行 `python models/linear.py ...` 会顺序验证所有已实现 shape；新增 shape 后应加入同一个
验证列表，避免只验证单个 kernel。

后续新增 shape 时应继续使用同一权重布局和局部 FP32 buffer 写法；如果某个输出维度过大，
可以把 `local_fp32` 改为输出分片 buffer，但不要回到 kernel 内转置权重的路径。
