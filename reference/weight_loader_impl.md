# DeepSeek V4 Flash Weight Loader Plan

本文记录 `serving/weight_loader.py` 需要完成的权重加载、反量化、dtype 转换、布局转换和
打包操作。目标是把官方 checkpoint 权重转换成当前 PyPTO kernel 需要的输入 tensor。

## 索引与名称规范化

`weight_loader.py` 不应只依赖原始 HF `model.safetensors.index.json` 的 `weight_map`，而应
复用或兼容 `../deepseek_v4_flash/inference/convert_bf16_index.py` 生成的
`weight_index.json`，因为它记录了反量化所需信息：

```text
raw_name
file
dtype
shape
kind
scale
scale_file
scale_raw_name
```

加载时需要支持 HF 原始名称到 inference 名称的规范化：

```text
model.embed_tokens.weight                  -> embed.weight
model.layers.N.self_attn.q_a_proj.weight   -> layers.N.attn.wq_a.weight
model.layers.N.self_attn.q_b_proj.weight   -> layers.N.attn.wq_b.weight
model.layers.N.mlp.gate_proj.weight        -> layers.N.ffn.experts.X.w1.weight
model.layers.N.mlp.down_proj.weight        -> layers.N.ffn.experts.X.w2.weight
model.layers.N.mlp.up_proj.weight          -> layers.N.ffn.experts.X.w3.weight
lm_head.weight                             -> head.weight
```

当前整网不实现 MTP，因此 `mtp.0.*` 权重不加载。正常 block 只加载
`layers.0.*` 到 `layers.42.*`。

## 反量化到 BF16

所有 linear 权重进入 PyPTO kernel 前都必须是普通 floating tensor，不保留 fp8/fp4 存储。
反量化逻辑应和 `../deepseek_v4_flash/inference/convert_bf16_index.py` 对齐。

### FP8 权重

`weight_index.json` 中 `kind == "fp8_weight"` 的权重：

```text
weight: float8_e4m3fn
scale:  fp32
```

加载操作：

```text
1. 读取 .weight 和对应 .scale。
2. 按 128x128 block scale 反量化。
3. 输出 torch.bfloat16。
```

### FP4 Expert 权重

`weight_index.json` 中 `kind == "fp4_packed_weight"` 的权重：

```text
weight: packed int8，每 byte 两个 E2M1 FP4
scale:  fp32
```

加载操作：

```text
1. 读取 packed .weight 和对应 .scale。
2. 拆 low/high nibble。
3. 用 E2M1 表恢复 fp32 值。
4. 按 32 元素 scale 反量化。
5. 输出 torch.bfloat16。
```

### Plain Tensor

`kind == "plain_tensor"` 或 integer tensor：

```text
BF16/F32 直接读取，按 kernel 需要转换 dtype。
I32/I64 直接读取，按 kernel 需要转换 dtype。
```

`get_linear_weight(name)` 应返回已反量化的 floating tensor。

## 通用 Dtype 规则

按当前 kernel 接口，加载后 dtype 规则如下：

```text
embed.weight                      -> bf16
普通 linear weight                -> bf16
RMSNorm weight                    -> bf16
q_norm / kv_norm weight           -> bf16
compressor norm weight            -> bf16
HC fn / scale / base              -> fp32
attn_sink                         -> fp32
compressor ape                    -> fp32
gate.weight                       -> bf16
gate.bias                         -> fp32
gate.tid2eid                      -> int32
head.weight                       -> fp32
```

反量化后的 fp8/fp4 linear 权重也转成 bf16。非 linear 的 fp32 参数不降成 bf16，避免和
官方 bf16 low-vram 计算逻辑产生额外差异。

## 全局权重

### Embedding

```text
checkpoint: embed.weight [VOCAB, HIDDEN]
kernel:     weight       [VOCAB, HIDDEN] bf16
operation:  保持原布局，转 bf16，contiguous
```

### Final Norm

```text
checkpoint: norm.weight [HIDDEN]
kernel:     norm_w      [HIDDEN] bf16
operation:  保持原布局，转 bf16，contiguous
```

### Final HC Head

```text
checkpoint:
  hc_head_fn    [HC_MULT, HC_DIM]
  hc_head_scale [1]
  hc_head_base  [HC_MULT]

kernel:
  hc_fn_t  [HC_DIM, HC_PAD] fp32
  hc_scale [1] fp32
  hc_base  [HC_PAD] fp32
```

转换：

```python
hc_fn_t = torch.zeros(HC_DIM, HC_PAD, dtype=torch.float32)
hc_fn_t[:, :HC_MULT] = hc_head_fn.t().contiguous().float()

hc_base = torch.zeros(HC_PAD, dtype=torch.float32)
hc_base[:HC_MULT] = checkpoint_hc_head_base.float()
```

`hc_scale` 保持 `[1] fp32`。

### LM Head

```text
checkpoint: head.weight [VOCAB, HIDDEN]
kernel:     head_w      [VOCAB, HIDDEN] fp32
operation:  保持官方布局，转 fp32，contiguous
```

LM head 是当前唯一明确保留官方 `[out, in]` 布局的 linear 权重。`models/head.py` 内部
使用 `b_trans=True` 对齐官方 `F.linear`。

## Block HC 权重

每层有 attention 和 FFN 两套 HC 参数。

```text
checkpoint:
  layers.N.hc_attn_fn    [MIX_HC, HC_DIM]
  layers.N.hc_attn_scale [3]
  layers.N.hc_attn_base  [MIX_HC]
  layers.N.hc_ffn_fn     [MIX_HC, HC_DIM]
  layers.N.hc_ffn_scale  [3]
  layers.N.hc_ffn_base   [MIX_HC]

kernel:
  attn_hc_fn_t [HC_DIM, MIX_HC] fp32
  ffn_hc_fn_t  [HC_DIM, MIX_HC] fp32
```

转换：

```python
attn_hc_fn_t = layers.N.hc_attn_fn.float().t().contiguous()
ffn_hc_fn_t  = layers.N.hc_ffn_fn.float().t().contiguous()
```

`scale/base` 保持原 shape，转 fp32。

## Attention 权重

所有 attention linear 权重都从官方 `[out, in]` 转成 kernel `[in, out]`。

```text
wq_a.weight [1024, 4096]  -> wq_a_t [4096, 1024] bf16
wq_b.weight [32768, 1024] -> wq_b_t [1024, 32768] bf16
wkv.weight  [512, 4096]   -> wkv_t  [4096, 512] bf16
wo_a.weight [8192, 4096]  -> wo_a_t [4096, 8192] bf16
wo_b.weight [4096, 8192]  -> wo_b_t [8192, 4096] bf16
```

Norm 和 sink：

```text
q_norm.weight  [1024] bf16
kv_norm.weight [512] bf16
attn_sink      [64]   fp32
```

## Compressor Ratio 128

用于 HCA attention 层。

```text
checkpoint:
  layers.N.attn.compressor.wkv.weight   [512, 4096]
  layers.N.attn.compressor.wgate.weight [512, 4096]
  layers.N.attn.compressor.ape          [128, 512]
  layers.N.attn.compressor.norm.weight  [512]

kernel:
  comp_wkv_t   [4096, 512] bf16
  comp_wgate_t [4096, 512] bf16
  comp_ape     [128, 512] fp32
  comp_norm_w  [512] bf16
```

## Compressor Ratio 4: Attention Path

用于 CSA attention 的 attention compressor。

```text
checkpoint:
  layers.N.attn.compressor.wkv.weight   [1024, 4096]
  layers.N.attn.compressor.wgate.weight [1024, 4096]
  layers.N.attn.compressor.ape          [4, 1024]
  layers.N.attn.compressor.norm.weight  [512]

kernel:
  attn_comp_wkv_t   [4096, 1024] bf16
  attn_comp_wgate_t [4096, 1024] bf16
  attn_comp_ape     [4, 1024] fp32
  attn_comp_norm_w  [512] bf16
```

## Indexer 权重

用于 CSA attention 中的 compressor topk indexer。

```text
checkpoint:
  layers.N.attn.indexer.wq_b.weight             [8192, 1024]
  layers.N.attn.indexer.weights_proj.weight     [64, 4096]
  layers.N.attn.indexer.compressor.wkv.weight   [256, 4096]
  layers.N.attn.indexer.compressor.wgate.weight [256, 4096]
  layers.N.attn.indexer.compressor.ape          [4, 256]
  layers.N.attn.indexer.compressor.norm.weight  [128]

kernel:
  idx_wq_b_t        [1024, 8192] bf16
  idx_weights_proj_t[4096, 64] bf16
  idx_comp_wkv_t    [4096, 256] bf16
  idx_comp_wgate_t  [4096, 256] bf16
  idx_comp_ape      [4, 256] fp32
  idx_comp_norm_w   [128] bf16
```

## MoE Gate

所有层都有 gate weight；hash 层使用 `tid2eid`，非 hash 层使用 `bias`。

```text
checkpoint:
  layers.N.ffn.gate.weight [256, 4096]

kernel:
  gate_w_t [4096, 256] bf16
```

hash 层：

```text
checkpoint: layers.N.ffn.gate.tid2eid [VOCAB, TOPK]
kernel:     tid2eid                   [VOCAB, TOPK] int32
```

topk 层：

```text
checkpoint: layers.N.ffn.gate.bias [256]
kernel:     gate_bias              [256] fp32
```

## MoE Shared Expert

```text
checkpoint:
  layers.N.ffn.shared_experts.w1.weight [2048, 4096]
  layers.N.ffn.shared_experts.w2.weight [4096, 2048]
  layers.N.ffn.shared_experts.w3.weight [2048, 4096]

kernel:
  shared_w1_t [4096, 2048] bf16
  shared_w2_t [2048, 4096] bf16
  shared_w3_t [4096, 2048] bf16
```

## MoE Routed Experts: Packed 主路径

当前主方案使用 packed routed experts，对齐 `models/moe.py` 顶层接口。

每个 expert 原始权重：

```text
experts.E.w1.weight [2048, 4096]
experts.E.w2.weight [4096, 2048]
experts.E.w3.weight [2048, 4096]
```

打包后：

```text
routed_w1_t [N_EXPERTS, 4096, 2048] bf16
routed_w2_t [N_EXPERTS, 2048, 4096] bf16
routed_w3_t [N_EXPERTS, 4096, 2048] bf16
```

转换：

```python
routed_w1_t = torch.empty(N_EXPERTS, 4096, 2048, dtype=torch.bfloat16)
routed_w2_t = torch.empty(N_EXPERTS, 2048, 4096, dtype=torch.bfloat16)
routed_w3_t = torch.empty(N_EXPERTS, 4096, 2048, dtype=torch.bfloat16)

for e in range(N_EXPERTS):
    routed_w1_t[e].copy_(get_linear_weight(f"layers.{N}.ffn.experts.{e}.w1.weight").t().contiguous())
    routed_w2_t[e].copy_(get_linear_weight(f"layers.{N}.ffn.experts.{e}.w2.weight").t().contiguous())
    routed_w3_t[e].copy_(get_linear_weight(f"layers.{N}.ffn.experts.{e}.w3.weight").t().contiguous())
```

打包时要避免长时间同时保留原始官方布局权重和 packed 权重。当前 decode 已切换到
selected-expert；prefill 仍需要 full routed pack，但 full pack 由 per-layer expert cache
逐专家组装，不再维护单独的 routed pack cache 路径。

## Expert Offline Cache

在线读取 routed experts 的主要耗时来自 fp4 -> bf16 反量化和转置。为了同时支持 prefill
full routed pack 和 decode selected-expert，离线 cache 采用每层一个 safetensors 文件、
每个 expert 独立存储的 bf16 布局。

cache 目录结构：

```text
bf16_expert_cache/
├── manifest.json
├── layer_000_experts.safetensors
├── layer_001_experts.safetensors
└── ...
```

每个 `layer_NNN_experts.safetensors` 包含：

```text
expert_000.w1_t [4096, 2048] bf16
expert_000.w2_t [2048, 4096] bf16
expert_000.w3_t [4096, 2048] bf16
...
```

生成命令：

```bash
python serving/convert_expert_cache.py \
  --checkpoint ~/dsv4_ckpt \
  --output ~/dsv4_bf16_expert_cache \
  --layers 0 \
  --overwrite \
  --profile
```

runner 使用命令：

```bash
python serving/runner.py \
  --checkpoint ~/dsv4_ckpt \
  --expert-cache-dir ~/dsv4_bf16_expert_cache \
  -p a2a3 -d {} -s 13 --max-layers 1 --no-head --profile
```

`DeepSeekV4WeightLoader.get_moe_routed_expert()` 会优先读取 expert cache；如果对应层或
expert 缺失，则回退到官方 checkpoint 在线 fp4 反量化和转置。
`get_layer_moe_routed_pack()` 和 `get_layer_moe_selected_experts()` 都复用同一条
per-expert 加载路径。`release_prefix(...)` 不删除 cache 文件，只清理内存中的 tensor cache。

## MoE Routed Experts: Per-Expert 备选接口

为 selected-expert 预留：

```text
get_moe_routed_expert(layer_id, expert_id)
  -> w1_t [4096, 2048] bf16
  -> w2_t [2048, 4096] bf16
  -> w3_t [4096, 2048] bf16
```

这个接口和 packed 主路径使用同一套反量化与转置逻辑。

## Rope Tables 和 TopK

以下不是 checkpoint 权重，不由 `weight_loader.py` 从 safetensors 读取，但 runner 调用 block
时需要提供：

```text
cos / sin
comp_cos / comp_sin
idx_comp_cos / idx_comp_sin
topk_idxs / window_topk_idxs
comp_block_count / idx_comp_block_count
cache_pos / comp_slot / comp_cache_slot / comp_should_compress
idx_offset
```

这些应由 `serving/runner.py` 或 `serving/state.py` 根据当前 `seq_len/start_pos/ratio` 生成。

## 建议 API

基础接口：

```python
has_tensor(name)
get_tensor(name, *, dtype=None, cache=True)
get_linear_weight(name, *, cache=True)
get_linear_t(name, *, dtype=torch.bfloat16, cache=True)
release(name=None)
release_prefix(prefix)
```

全局接口：

```python
get_embedding_weight()
get_head_weights()
```

层级接口：

```python
get_layer_hc(layer_id)
get_layer_attention_common(layer_id)
get_layer_compressor_ratio128(layer_id)
get_layer_compressor_ratio4_attention(layer_id)
get_layer_indexer(layer_id)
get_layer_moe_gate(layer_id, *, hash_route: bool)
get_layer_moe_shared(layer_id)
get_layer_moe_routed_pack(layer_id)
get_moe_routed_expert(layer_id, expert_id)
```

`get_layer_*` 可以返回 dataclass 或普通 dict。字段名应直接对齐 block kernel 参数名，例如：

```text
attn_hc_fn_t
attn_hc_scale
wq_a_t
q_norm_w
gate_w_t
routed_w1_t
shared_w1_t
```

这样 `serving/runner.py` 可以按 kernel 签名直接组装参数。

## 校验要求

实现 `weight_loader.py` 后需要增加测试覆盖：

- tiny safetensors checkpoint：
  - plain bf16/fp32 tensor
  - fp8 linear 权重 + scale
  - fp4 packed expert 权重 + scale
  - integer `tid2eid`
- `get_linear_t()` 输出 shape、dtype、contiguous。
- HC head padding 是否正确。
- block HC 转置是否正确。
- lm head 是否保持 `[VOCAB, HIDDEN]` 且为 fp32。
- packed routed experts 是否按 expert id 正确 stack。
- `release()` / `release_prefix()` 是否清理缓存。
