# DeepSeek V4 Flash PyPTO Kernel 目录设计

本文定义当前仓库实现 DeepSeek V4 Flash bf16 PyPTO kernel 时采用的目录结构和模块边界。

设计目标：

- 对齐 `../deepseek_v4_flash` 下 bf16 推理路径的模型结构和计算逻辑。
- 只实现单卡逻辑，不实现 TP/EP 多卡并行。
- 只实现 bf16 计算路径，不实现 fp8/fp4 kernel。
- 不实现 paged attention、chunked prefill、packed prefill、dynamic batching 等额外推理特性。
- 优先保证权重加载后能够在 Ascend NPU 上推理出逻辑正确的句子。

## 实现边界

DeepSeek V4 Flash PyPTO 实现应直接放在当前仓库中，作为自包含实现维护。

不依赖：

- `../pypto-serving`
- `../pypto-serving/pypto-lib`
- `../pypto-serving/pypto-lib/models/deepseek/v4`

如果需要借鉴其中某个功能，应在当前仓库内重新实现简化版本，而不是把
`pypto-serving` 的复杂模块作为运行时依赖。

实现可以继续参考：

- `../deepseek_v4_flash` 的 bf16 模型计算逻辑。
- `../pypto-serving` 的 Qwen3 权重加载、host wrapper、runner 组织方式。

但这些都只作为参考，不作为代码依赖。

## 顶层目录结构

建议在当前仓库中创建如下结构。

`tree` 命令视角：

```text
.
├── reference/
│   ├── goal.md
│   ├── dsv4_flash_arch.md
│   ├── dsv4_pypto_kernel_design.md
│   └── pypto_serving_qwen3.md
├── pypto_lib/
│   └── models/
│       └── deepseek_v4_flash_bf16/
│           ├── __init__.py
│           ├── config.py
│           ├── common.py
│           ├── rope.py
│           ├── rmsnorm.py
│           ├── linear.py
│           ├── hc.py
│           ├── sparse_attn.py
│           ├── compressor_common.py
│           ├── compressor_ratio4.py
│           ├── compressor_ratio128.py
│           ├── indexer.py
│           ├── attention_common.py
│           ├── attention_swa.py
│           ├── attention_csa.py
│           ├── attention_hca.py
│           ├── moe.py
│           ├── block.py
│           ├── head.py
│           ├── prefill_fwd.py
│           ├── decode_fwd.py
│           ├── dispatch.py
│           ├── weight_layout.py
│           └── golden.py
├── python/
│   └── dsv4_flash_bf16/
│       ├── __init__.py
│       ├── model_loader.py
│       ├── executor.py
│       ├── runner.py
│       ├── tokenizer.py
│       └── generate.py
└── tests/
    └── dsv4_flash_bf16/
        ├── test_hc.py
        ├── test_compressor_ratio4.py
        ├── test_compressor_ratio128.py
        ├── test_attention_swa.py
        ├── test_attention_hca.py
        ├── test_attention_csa.py
        ├── test_moe.py
        └── test_prefill_decode.py
```

其中：

- `prefill_fwd.py` 和 `decode_fwd.py` 是最终对外编译和运行的顶层 PyPTO 程序。
- 其余文件提供可复用的 inline kernel 或 host 侧辅助逻辑。
- `golden.py` 只用于测试和对齐，不参与 NPU runtime。
- `python/dsv4_flash_bf16/` 是当前仓库内的权重加载、PyPTO 编译、runner 和简单 generation 逻辑。
- `tests/dsv4_flash_bf16/` 用于逐模块 golden 对齐和端到端 smoke test。

## 文件职责

### `config.py`

保存 DeepSeek V4 Flash 的静态模型参数和 kernel 编译常量。

应包含：

- `dim = 4096`
- `n_layers = 43`
- `n_heads = 64`
- `head_dim = 512`
- `rope_head_dim = 64`
- `q_lora_rank = 1024`
- `o_lora_rank = 1024`
- `o_groups = 8`
- `window_size = 128`
- `moe_inter_dim = 2048`
- `n_routed_experts = 256`
- `n_shared_experts = 1`
- `n_activated_experts = 6`
- `n_hash_layers = 3`
- `score_func = "sqrtsoftplus"`
- `route_scale = 1.5`
- `swiglu_limit = 10.0`
- `hc_mult = 4`
- `hc_sinkhorn_iters = 20`
- `hc_eps = 1e-6`
- `compress_ratios`
- RoPE/YaRN 参数
- 固定 batch/seq 编译参数

注意：

- 固定 `batch = 1`。
- 固定 `max_seq_len`，用于编译和验证。
- 不放 fp8/fp4 dtype 配置，不实现低精度路径。

### `common.py`

保存跨模块共享的小工具、常量和 shape helper。

可包含：

- 维度别名
- tile 常量
- shape 检查 helper
- dtype helper
- 常用 `pl.dynamic` 定义

不要把具体计算逻辑全部塞入 `common.py`，否则模块边界会失控。

### `linear.py`

实现 bf16 linear 的基础 PyPTO helper。

目标语义对齐：

```python
F.linear(x, weight)
```

注意权重布局需要和 `weight_layout.py` 保持一致。

建议 kernel 内使用的权重布局：

```text
weight_t: [in_features, out_features]
```

这样 PyPTO matmul 可直接：

```text
x [T, in] @ weight_t [in, out] -> out [T, out]
```

该文件只处理 bf16/fp32 accumulation，不包含：

- activation quant
- fp8 gemm
- fp4 gemm
- int8 dequant

### `rmsnorm.py`

实现 RMSNorm。

语义：

```text
y = x.float()
y = y * rsqrt(mean(y * y) + eps)
out = weight.float() * y
out -> bf16
```

需要支持：

- hidden RMSNorm: `[T, dim]`
- q-lora RMSNorm: `[T, q_lora_rank]`
- KV RMSNorm: `[T, head_dim]`
- per-head q RMS normalize: `[T, n_heads, head_dim]`

### `rope.py`

实现 RoPE 和 inverse RoPE。

需要支持：

- q rope tail:

  ```text
  q[..., -rope_head_dim:]
  ```

- kv rope tail:

  ```text
  kv[..., -rope_head_dim:]
  ```

- attention output inverse rope:

  ```text
  o[..., -rope_head_dim:]
  ```

输入建议使用预先生成好的 cos/sin 表，而不是在 PyPTO kernel 内生成。

需要分别支持：

- base RoPE: `ratio = 0` 的 SWA 路径
- compressed RoPE/YaRN: `ratio = 4/128` 的 CSA/HCA 路径

### `hc.py`

实现 Hyper-Connections 相关逻辑。

应包含：

- `hc_pre`
- `hc_post`
- `hc_head`
- `hc_split_sinkhorn`

语义对齐 `../deepseek_v4_flash/inference/model.py`：

```text
hc_pre:
  x_hc [T, hc_mult, dim]
  -> flatten [T, hc_mult * dim]
  -> mixes = linear(flatten, hc_fn) * rms_scale
  -> pre, post, comb = hc_split_sinkhorn(mixes)
  -> x_single = sum(pre_i * x_hc_i)

hc_post:
  x_single [T, dim]
  residual [T, hc_mult, dim]
  -> post * x_single + comb * residual

hc_head:
  x_hc [T, hc_mult, dim]
  -> sigmoid-based reduce
  -> x_single [T, dim]
```

`hc_pre/hc_post/hc_head` 是整个模型正确性的基础，应单独提供 golden 对齐测试。

### `sparse_attn.py`

实现 bf16 sparse attention。

语义对齐 `low_vram_kernels.py`：

```text
for each token:
  idxs = topk_idxs[token]
  selected = kv_cache[idxs >= 0]
  scores = q @ selected.T * softmax_scale
  scores = concat(scores, attn_sink)
  probs = softmax(scores)
  out = probs_without_sink @ selected
```

实现约束：

- `batch = 1`
- `seq` 为固定编译长度
- `topk` 为静态 shape
- 无 paged KV
- KV cache 为 contiguous tensor

不实现 TileLang 级别的性能优化。

### `compressor_common.py`

保存 ratio=4 和 ratio=128 compressor 共享的基础计算。

建议包含：

- `x -> wkv`
- `x -> wgate`
- `score + ape`
- softmax weighted sum
- compressor RMSNorm
- compressor RoPE
- compressed KV cache 写入 helper

不包含 ratio 分支和完整 state 更新逻辑。ratio=4 和 ratio=128 的 state shape、prefill 分块和 decode boundary 行为不同，应分别放在独立文件中。

不包含：

- `act_quant`
- `rotate_activation`
- `fp4_act_quant`

### `compressor_ratio4.py`

实现 `compress_ratio = 4` 的 overlap compressor。

该路径服务：

- CSA attention 的 compressed KV。
- Indexer 内部的 compressed KV。

ratio=4 的 state 包含 overlap window 和 current window：

```text
coff = 2
kv_state:    [batch, 2 * ratio, 2 * head_dim]
score_state: [batch, 2 * ratio, 2 * head_dim]
```

prefill 逻辑：

- `x -> wkv`
- `x -> wgate`
- 加 `ape`
- 按 4-token block 分组。
- 对完整 block 做 softmax weighted sum。
- 使用 overlap transform 拼接相邻 block 信息。
- remainder 写入 state，等待 decode 时补齐。
- RMSNorm。
- RoPE。
- 写 compressed KV cache。

decode 逻辑：

- 将当前 token 的 kv/score 写入 current window。
- 当 `(start_pos + 1) % 4 == 0` 时生成一个 compressed KV。
- boundary 上从 previous/overlap window 和 current window 拼接压缩输入。
- 写 compressed KV cache。
- 将 current window shift 成下一轮 previous/overlap window。

不包含：

- learned topk 计算；该逻辑属于 `indexer.py`。
- activation quant。
- FP4 前 Hadamard rotate。

### `compressor_ratio128.py`

实现 `compress_ratio = 128` 的普通 block compressor。

该路径服务 HCA attention。

ratio=128 没有 overlap，也没有 learned indexer：

```text
coff = 1
kv_state:    [batch, ratio, head_dim]
score_state: [batch, ratio, head_dim]
```

prefill 逻辑：

- `x -> wkv`
- `x -> wgate`
- 加 `ape`
- 按 128-token block 分组。
- 对完整 block 做 softmax weighted sum。
- remainder 写入 state，等待 decode 时补齐。
- RMSNorm。
- RoPE。
- 写 compressed KV cache。

decode 逻辑：

- 将当前 token 的 kv/score 写入 `start_pos % 128`。
- 当 `(start_pos + 1) % 128 == 0` 时压缩整个 state。
- 写 compressed KV cache。

规则 compressed topk 不在本文件实现，由 `attention_common.py` 提供。

不包含：

- `rotate_activation`
- `fp4_act_quant`
- activation quant

### `indexer.py`

实现 `ratio = 4` 的 learned compressed KV indexer。

语义：

```text
qr -> indexer.wq_b -> q_index
q_index -> RoPE
compressor_ratio4(x) -> indexer compressed kv cache
weights = weights_proj(x)
index_score = q_index @ compressed_kv
index_score = relu(index_score) * weights
sum over index heads
topk -> compressed topk idxs
```

该模块只服务 CSA 路径。

不包含：

- `rotate_activation`
- `fp4_act_quant`

### `attention_common.py`

保存 SWA/CSA/HCA 三条 attention 路径共享的基础计算。

建议包含：

- `q_proj_rope_bf16`
- `kv_proj_rope_bf16`
- `window_topk`
- `compress_topk_rule`
- `attention_output_proj`

其中 `q_proj_rope_bf16` 对齐：

```text
x -> wq_a -> q_norm -> wq_b
q -> [T, n_heads, head_dim]
q -> per-head RMS normalize
q rope tail -> RoPE
```

`kv_proj_rope_bf16` 对齐：

```text
x -> wkv -> kv_norm
kv rope tail -> RoPE
```

`attention_output_proj` 对齐：

```text
o -> inverse RoPE
o -> group by o_groups
o -> wo_a
o -> wo_b
```

### `attention_swa.py`

实现 `compress_ratio = 0` 的 attention 路径。

对应 DeepSeek V4 Flash 中 SWA 路径。

应提供：

```text
attention_swa_prefill(...)
attention_swa_decode(...)
```

功能：

- HC 前后的逻辑不放在这里，由 `block.py` 负责。
- 本文件只处理 `Attention.forward()` 内部逻辑。
- 只使用 sliding window KV。
- 不调用 compressor。
- 不调用 indexer。
- 使用 base RoPE。

prefill：

- 对整个 prompt 计算 q/kv。
- 写 window KV cache。
- 构造 window topk。
- sparse attention。

decode：

- 对当前 token 计算 q/kv。
- 写 `kv_cache[start_pos % window_size]`。
- 构造当前 token 的 window topk。
- sparse attention。

### `attention_csa.py`

实现 `compress_ratio = 4` 的 attention 路径。

对应 DeepSeek V4 Flash 中 CSA 路径。

应提供：

```text
attention_csa_prefill(...)
attention_csa_decode(...)
```

功能：

- sliding window KV
- `compressor_ratio4.py`
- learned indexer
- indexer compressor, also using ratio=4 overlap compressor logic
- compressed topk 拼接到 window topk
- compressed KV 拼接到 sparse attention 的 KV pool
- 使用 compressed RoPE/YaRN

prefill：

- 批量计算 q/kv。
- 写 window KV cache。
- 运行 attention compressor 生成 compressed KV。
- 运行 indexer compressor。
- 运行 indexer 得到 compressed topk。
- 拼接 topk 后 sparse attention。

decode：

- 单 token q/kv。
- 写 window KV cache。
- 增量更新 compressor state。
- 到 ratio 边界时写一个 compressed KV。
- 增量更新 indexer compressor state。
- 计算当前 token compressed topk。
- sparse attention。

### `attention_hca.py`

实现 `compress_ratio = 128` 的 attention 路径。

对应 DeepSeek V4 Flash 中 HCA 路径。

应提供：

```text
attention_hca_prefill(...)
attention_hca_decode(...)
```

功能：

- sliding window KV
- `compressor_ratio128.py`
- 规则 compressed topk
- 无 learned indexer
- 使用 compressed RoPE/YaRN

prefill：

- 批量计算 q/kv。
- 写 window KV cache。
- 运行 ratio=128 compressor。
- 通过规则函数生成 compressed topk。
- sparse attention。

decode：

- 单 token q/kv。
- 写 window KV cache。
- 增量更新 compressor state。
- 到 ratio 边界时写 compressed KV。
- 通过规则函数生成当前 token compressed topk。
- sparse attention。

### `moe.py`

实现单卡 bf16 MoE。

职责：

- Gate
- hash routing
- score routing
- routed experts
- shared expert
- SwiGLU
- route weight 加权求和

语义：

```text
scores = x @ gate_w.T
scores = sqrt(softplus(scores))
if hash layer:
  indices = tid2eid[input_ids]
else:
  indices = topk(scores + gate_bias)
weights = original_scores.gather(indices)
weights = weights / sum(weights)
weights = weights * route_scale

y = 0
for selected expert:
  y += weight * expert(x)
y += shared_expert(x)
```

expert:

```text
gate = w1(x)
up = w3(x)
if swiglu_limit > 0:
  clamp gate/up
hidden = silu(gate) * up
out = w2(hidden)
```

范围不包含：

- EP dispatch
- all-to-all
- routed expert shard
- int8 expert path
- recv buffer
- combine kernel

MoE 直接对 token 和 selected experts 做循环，优先保持计算逻辑清晰。

### `block.py`

实现一个完整 Transformer block 的 orchestration。

应提供：

```text
block_prefill(...)
block_decode(...)
```

逻辑：

```text
residual = x_hc
x, post, comb = hc_pre(attn)
x = attn_norm(x)
x = attention_{swa,csa,hca}_{prefill,decode}(x)
x_hc = hc_post(x, residual, post, comb)

residual = x_hc
x, post, comb = hc_pre(ffn)
x = ffn_norm(x)
x = moe(x, input_ids, layer_id)
x_hc = hc_post(x, residual, post, comb)
```

`block.py` 根据该层 `compress_ratio` 调用不同 attention 模块：

```text
ratio == 0   -> attention_swa
ratio == 4   -> attention_csa
ratio == 128 -> attention_hca
```

### `head.py`

实现最终输出头。

职责：

- `hc_head`
- final RMSNorm
- LM head linear

语义：

```text
x_single = hc_head(x_hc)
x_norm = rms_norm(x_single, norm_w)
logits = x_norm[last_token] @ lm_head.T
```

输出最后一个 token 的 logits。

### `prefill_fwd.py`

实现完整 prompt prefill 顶层 PyPTO 程序。

职责：

- 输入 prompt token embeddings 或 token ids + embedding weight。
- 初始化 HC hidden：

  ```text
  h = embed(input_ids)
  h_hc = repeat(h, hc_mult)
  ```

- 依次运行所有 layer 的 `block_prefill`。
- 写入 attention window KV cache。
- 写入 compressed KV cache。
- 写入 compressor/indexer state。
- 调用 `head.py` 输出最后 token logits。

实现约束：

- `batch = 1`
- `seq_len` 固定或半固定
- 不做 chunked prefill
- 不做 packed prefill
- 不做 paged KV

### `decode_fwd.py`

实现单 token decode 顶层 PyPTO 程序。

职责：

- 输入当前 token embedding 或 token id。
- 输入 `start_pos`。
- 从 prefill/decode 累积的 caches 中读取历史 KV。
- 对当前 token 依次运行所有 layer 的 `block_decode`。
- 更新 window KV cache。
- 更新 compressed KV cache。
- 更新 compressor/indexer state。
- 输出 logits。

实现约束：

- `batch = 1`
- 每次只 decode 1 个 token
- cache 使用 contiguous tensor
- 不实现 paged attention

### `dispatch.py`

提供 host-level wrapper，用于 PyPTO 编译和运行。

作用类似 `../pypto-serving/examples/model/qwen3_14b/runner/qwen3_l3_dispatch.py`。

应包含：

```text
@pl.jit.host
def dsv4_prefill_host(...):
    return prefill_fwd(...)

@pl.jit.host
def dsv4_decode_host(...):
    return decode_fwd(...)
```

注意：这里的 `dispatch.py` 不是 MoE EP dispatch，不做专家路由通信。

### `weight_layout.py`

定义 Python 侧权重整理规则。

职责：

- 将 HF / bf16 low-vram 权重名映射到 PyPTO kernel 参数。
- 将 linear weight 转置为 PyPTO matmul 友好的布局。
- 将 fp8/fp4 原始权重反量化为 bf16。
- 为每层按 PyPTO kernel 签名组织权重。
- 生成 RoPE cos/sin 表。
- 生成 cache/state tensor shape。

该文件属于 Python runtime/loader 辅助，不一定必须放在 kernel 目录中；也可以放在 runner 侧。

保留这个文件以明确权重布局约定，避免 kernel 和 loader 各自猜 shape。

### `golden.py`

提供 PyTorch golden reference，用于逐模块验证。

应尽量直接复用或对齐：

- `low_vram_attention.py`
- `low_vram_moe.py`
- `low_vram_executor.py`
- `low_vram_kernels.py`

建议包含：

- `golden_hc_pre`
- `golden_hc_post`
- `golden_hc_head`
- `golden_compressor_ratio4_prefill`
- `golden_compressor_ratio4_decode`
- `golden_compressor_ratio128_prefill`
- `golden_compressor_ratio128_decode`
- `golden_attention_swa_prefill`
- `golden_attention_swa_decode`
- `golden_attention_hca_prefill`
- `golden_attention_hca_decode`
- `golden_attention_csa_prefill`
- `golden_attention_csa_decode`
- `golden_moe`
- `golden_block`
- `golden_prefill`
- `golden_decode`

## Attention 是否拆成 SWA/CSA/HCA

建议拆。

原因：

1. 三条路径对应固定 `compress_ratio`，是 layer 静态属性。
2. 三条路径的参数和 state 差异很大。
3. SWA 不应该被迫传 compressor/indexer 参数。
4. CSA 的 overlap compressor 和 learned indexer 复杂度最高，单独隔离更利于调试。
5. HCA 没有 indexer，规则 topk 更简单，单独实现可以保持接口清晰。

拆分后每个 attention 文件内部提供 prefill 和 decode 两个入口。

## 是否每个模块都拆 prefill/decode

只对 state 行为不同的模块拆。

建议拆：

- `attention_swa.py`
- `attention_csa.py`
- `attention_hca.py`
- `compressor_ratio4.py`
- `compressor_ratio128.py`
- `block.py`
- `prefill_fwd.py`
- `decode_fwd.py`

不一定需要拆：

- `linear.py`
- `rmsnorm.py`
- `rope.py`
- `hc.py`
- `sparse_attn.py`
- `moe.py`
- `head.py`

原因：

- prefill 和 decode 的主要差异来自 cache/state 更新。
- 纯函数模块不需要拆。
- MoE 对 prefill/decode 的计算语义一致，只是 token 数不同。

## 实现依赖顺序

建议按依赖关系实现：

1. `config.py`
2. `linear.py`
3. `rmsnorm.py`
4. `rope.py`
5. `hc.py`
6. `sparse_attn.py`
7. `attention_common.py`
8. `attention_swa.py`
9. `moe.py`
10. `head.py`
11. `block.py`
12. `compressor_common.py`
13. `compressor_ratio128.py`
14. `attention_hca.py`
15. `compressor_ratio4.py`
16. `indexer.py`
17. `attention_csa.py`
18. `prefill_fwd.py`
19. `decode_fwd.py`
20. `dispatch.py`
21. `weight_layout.py`

### 基础数学模块

- `linear`
- `rmsnorm`
- `rope`
- `hc_pre/hc_post/hc_head`

目标：模块级 golden 对齐。

### SWA Attention

- `q_proj_rope_bf16`
- `kv_proj_rope_bf16`
- `window_topk`
- `sparse_attn`
- `output_proj`
- `attention_swa_prefill`
- `attention_swa_decode`

目标：跑通 `compress_ratio = 0` 的 attention。

### MoE

- Gate
- hash routing
- score routing
- routed experts
- shared expert

目标：单卡 bf16 MoE 与 low-vram reference 对齐。

### HCA

- `compressor_ratio128`
- rule compressed topk
- HCA prefill/decode

目标：跑通 `compress_ratio = 128` attention。

### CSA

- `compressor_ratio4`
- indexer compressor
- learned indexer topk
- CSA prefill/decode

目标：跑通 `compress_ratio = 4` attention。

### Full Model

- `block_prefill`
- `block_decode`
- `prefill_fwd`
- `decode_fwd`
- host dispatch
- weight layout
- simple generation loop

目标：加载真实权重，在 Ascend NPU 上生成逻辑正确句子。

## 范围外功能

本目标不包含：

- 多卡 TP
- 多卡 EP
- expert all-to-all
- `dispatch_ep`
- paged KV cache
- block table
- slot mapping
- packed prefill
- chunked prefill
- continuous batching
- fp8/fp4 activation quant
- fp8/fp4 GEMM
- INT8 expert dispatch/dequant
- 性能优化导向的 kernel fusion

这些能力不属于 `reference/goal.md` 定义的目标。
