# Attention SWA Precision Notes

本文档记录 `models/attention_swa.py` 精度定位过程中的结论，避免后续继续定位时重复走弯路。

## 已确认结论

- `attention_out_fwd` 本身不是当前最终 `out` 严格阈值失败的主要原因。
  - `models/attention_out.py` standalone 下，`o_inv`、`proj`、`out` 均严格通过。
  - `attention_swa.py --compare-proj-out-to actual-upstream` 下，使用设备实际 `o_inv/proj` 重算参考时 prefill/decode 均通过。
- `sparse_attn_swa_fwd` 本身只引入很少的额外非逐位误差。
  - `--exact-upstream --compare-attn-to actual-upstream` 下，prefill 的 `attn_o` 非逐位一致从 official 对比时约 `910/262144` 降到 `26/262144`。
  - decode 的 `attn_o` 非逐位一致降到 `1/32768`。
- 大部分最终误差来自 q 路径中极少量 BF16 matmul 差异的传播和放大。
  - prefill 最早在 `q_a = x @ wq_a_t` 出现非逐位一致：`4/8192`。
  - decode 最早报错在 `q_proj`：`3/32768`。
  - KV 路径在当前固定 seed case 中未表现为主要误差源。
- `--exact-upstream --compare-q-path-to actual-upstream` 已确认 q 路径中的非 matmul 部分不是主要误差来源。
  - prefill 中用实际 `q_a` 重算 `qr` 后，`qr` 严格通过，说明 `rmsnorm_1024` 与 golden 一致。
  - prefill 中用实际 `qr` 重算 `q_proj` 后，`q_proj` 仍有 `47/262144` 个逐位差异，最大差异 `0.00390625`，误差来自 `linear_1024_to_32768` 的 matmul 结果本身。
  - prefill 中用实际 `q_proj` 重算 head RMS scale 和 RoPE 后，`q` 严格通过，说明 `q *= rsqrt(mean(q^2)+eps)` 和 RoPE 没有继续引入差异。
  - decode 中用实际 `qr` 重算 `q_proj` 后仍有 `3/32768` 个逐位差异，后续用实际 `q_proj` 重算 `q` 严格通过，结论与 prefill 一致。
- 将 `sparse_attn` 从 `sum(exp * v) / denom` 改为先计算 `qk_probs = exp / denom` 再 `qk_probs @ v` 后，standalone 仍通过，整段 prefill `out` bad points 从 `877` 降到 `872`，改善很小。
- 尝试手动 TOPK 逐项累加和 `PV_K_TILE=8` 小块累加均受 PyPTO tile 对齐约束影响无法编译；`PV_K_TILE=16` 可编译但没有改善整段误差。
- 针对 q 路径尝试改变 linear K/N 切分没有减少误差。
  - `linear_4096_to_1024` 的 K tile 从 `128` 改为 `256` 后，`q_a` 仍为 `4/8192` 个逐位差异。
  - `linear_1024_to_32768` 尝试过 `K=64,N=64`、`K=128,N=32`、`K=256,N=32`、`K=1024,N=16`，`q_proj` 仍保持 prefill `47/262144`、decode `3/32768` 的逐位差异。
  - `K=128,N=32` 是最接近 `linear_4096_to_512` 和 `linear_4096_to_1024` 的切轴方式：输出轴每次 matmul 处理 `O_TILE=32` 列，`spmd` group 覆盖 2 个输出 tile，但不创建 `[T_TILE,32768]` 全量临时 buffer。该写法能编译运行，但误差数量和最大误差与基线完全一致。
  - 进一步将 `linear_1024_to_32768` 改成与前两个 linear kernel 一样的完整 FP32 临时输出结构：每个 token block 创建 `[T_TILE,32768]` 的 `out_tile_fp32`，`spmd` 内写入 FP32 临时 buffer，然后在 `CORE_GROUP` 中统一 cast 到 BF16 并写回。该写法没有触发内存超限，但 `q_proj` 仍保持 prefill `47/262144`、decode `3/32768` 的逐位差异。
  - 将完整 FP32 临时输出结构中的输出轴调度从 `pl.spmd` 改成普通 `pl.range` 会使 `matmul` 落在 orchestration 层并编译失败；显式用 `with pl.at(level=pl.Level.CORE_GROUP)` 包住每个输出 group 后可以编译运行，但 `q_proj` 仍保持 prefill `47/262144`、decode `3/32768` 的逐位差异。因此误差也不是 `pl.spmd` 并行调度导致的。
  - 尝试改变 `linear_1024_to_32768` 输出 cast 的 rounding mode 也没有改善。`mode="round"` 在 decode 下仍为 `3/32768`，prefill 变为 `48/262144`，比 `mode="rint"` 的 `47/262144` 略差；`mode="trunc"` 在 decode 下使 `q_proj` 约 `50%` 元素不一致，明显不符合 PyTorch BF16 cast 语义。因此当前保留 `mode="rint"`。
  - `K=256,N=64` 会因为右矩阵 buffer 超过平台 64KB 限制无法编译。
  - 因此当前 q 路径差异更像是 PyPTO/Ascend matmul 与 PyTorch CPU float matmul 在 bf16 输出舍入上的实现差异，而不是我们显式 K 分块顺序造成的逻辑错误。

## 当前诊断开关

- `--exact-upstream`
  - 上游中间量使用 `atol=0, rtol=0, max_error_ratio=0` 比较。
  - `proj/out` 会被忽略，便于定位上游最早非逐位差异。
- `--compare-attn-to actual-upstream`
  - 使用设备实际 `q` 和 `kv/kv_cache_out` 在 CPU 侧重算 sparse golden，再比较设备 `attn_o`。
- `--compare-proj-out-to actual-upstream`
  - 使用设备实际 `o_inv/proj` 在 CPU 侧重算 `proj/out`。
- `--model-level-output`
  - 中间关键张量保持严格，`proj/out` 使用模型级输出阈值。
- `--compare-q-path-to actual-upstream`
  - 使用设备实际 `q_a/qr/q_proj` 在 CPU 侧逐级重算 `qr/q_proj/q`，用于继续定位 q 路径误差。

## 下一步方向

继续从 q 路径定位：

1. 运行 `--exact-upstream --compare-q-path-to actual-upstream`。
2. 如果 `qr` 在 actual `q_a` 下通过，说明 RMSNorm 不是主要问题。
3. 如果 `q_proj` 在 actual `qr` 下仍不逐位一致，重点检查 `linear_1024_to_32768`。
4. 如果 `q_proj` 通过但 `q` 不通过，重点检查 head RMS scale 或 RoPE。
5. `q_a` 没有上游可替代输入，若仍不逐位一致，重点检查 `linear_4096_to_1024` 的切分和累加顺序。

截至当前定位，`qr` 和 `q` 的 actual-upstream 对比已经通过，`linear_4096_to_1024` 与 `linear_1024_to_32768` 的多种切分方式也没有改善逐位差异。后续如果目标是整段严格通过，更应优先评估模型级阈值是否合理，或者引入与设备 matmul 更一致的 golden，而不是继续在 q 路径上反复调整 tile。
