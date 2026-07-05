# DeepSeek V4 Flash PyPTO Whole-Model Plan

本文记录当前仓库整网推理的 host 侧实现方案。目标是在单卡 Ascend NPU 上用 PyPTO
加载 DeepSeek V4 Flash bf16 权重并推理出逻辑正确的句子，不实现多卡并行、MTP、
fp4/fp8 量化、paged attention 或其他性能特性。

## 整体策略

`models/` 目录只放 PyPTO kernel。整网不实现一个包含 43 层的大 kernel，而是由
`serving/` 目录下的 host runner 按层调度已有 kernel：

```text
input_ids
  -> embedding
  -> expand hc copies
  -> for layer_id in 0..42:
       load layer weights
       run one block kernel
       update layer state
       release layer weights
  -> final head
  -> logits
```

这个策略对齐 `../deepseek_v4_flash/inference/low_vram_executor.py` 的 low-vram 思路：
权重按需加载，层后释放；attention cache/state 按层持久保存；hidden activation 只保留
当前层输出。单卡 64GB 显存无法常驻完整模型权重，因此 layer-by-layer 是整网主线。

## 目录边界

建议新增：

```text
serving/
├── weight_loader.py
├── state.py
├── runner.py
└── generate.py
```

职责划分：

- `serving/weight_loader.py`
  - 从 safetensors/index 加载 checkpoint tensor。
  - 完成 bf16/fp32 dtype 转换。
  - 完成当前 kernel 需要的权重转置和 padding。
  - 提供 layer-by-layer 权重获取和释放接口。
- `serving/state.py`
  - 初始化每层 attention cache/state。
  - 管理 prefill 后 decode 需要复用的 state tensor。
  - 根据 `compress_ratios[layer_id]` 创建 SWA/HCA/CSA 对应状态。
- `serving/runner.py`
  - host 侧单层调度。
  - 根据 layer_id 选择 block kernel。
  - 负责把当前层权重、state、rope table 和 hidden 传给 PyPTO kernel。
- `serving/generate.py`
  - tokenizer、prompt 编码、prefill、decode loop。
  - 根据 logits 做 argmax 或采样。

不新增 `models/model.py` 或 `models/transformer.py`。如果未来确实要实现新的 PyPTO kernel，
再放入 `models/`；当前整网调度属于 host runtime 逻辑。

## 已有 Kernel 覆盖范围

43 个正常 block 只需要当前 `models/block.py` 中的 8 个入口：

```text
block_swa_hash_prefill_fwd
block_swa_hash_decode_fwd
block_csa_hash_prefill_fwd
block_csa_hash_decode_fwd
block_hca_topk_prefill_fwd
block_hca_topk_decode_fwd
block_csa_topk_prefill_fwd
block_csa_topk_decode_fwd
```

同一种形态的不同 layer 只替换权重和 state，不需要新增 per-layer kernel。

layer 形态由官方配置决定：

```text
layer 0,1       ratio=0,   hash_route=True
layer 2         ratio=4,   hash_route=True
layer 3,5,...   ratio=128, hash_route=False
layer 4,6,...   ratio=4,   hash_route=False
```

官方配置中正常 block 为 `n_layers=43`。`compress_ratios[43]` 属于 MTP，不在当前整网
范围内。

## Prefill 流程

prefill 输入：

```text
input_ids: [B, S] INT64
start_pos = 0
```

host runner 流程：

```text
1. load global embedding weight
2. embedding_fwd(input_ids, embed_weight) -> h [B, S, HIDDEN]
3. expand hc copies -> h [B, S, HC_MULT, HIDDEN]
4. for layer_id in 0..42:
     ratio = compress_ratios[layer_id]
     hash_route = layer_id < n_hash_layers
     load current layer weights
     materialize rope tables needed by this layer
     run matching block prefill kernel
     persist cache/state outputs for decode
     release current layer weights
5. load final norm, hc head, lm head
6. head_fwd(h, final_norm, hc_head, lm_head) -> logits [B, VOCAB]
```

block kernel 选择：

```text
ratio=0,   hash_route=True  -> block_swa_hash_prefill_fwd
ratio=4,   hash_route=True  -> block_csa_hash_prefill_fwd
ratio=128, hash_route=False -> block_hca_topk_prefill_fwd
ratio=4,   hash_route=False -> block_csa_topk_prefill_fwd
```

## Decode 流程

decode 输入：

```text
input_ids: [B, 1] INT64
start_pos: 当前 token 位置，prefill 长度之后递增
```

host runner 流程：

```text
1. embedding_fwd(input_ids) -> h [B, 1, HIDDEN]
2. expand hc copies -> h [B, 1, HC_MULT, HIDDEN]
3. for layer_id in 0..42:
     load current layer weights
     reference current layer cache/state
     run matching block decode kernel
     update cache/state
     release current layer weights
4. head_fwd(h) -> logits [B, VOCAB]
5. select next token
```

decode 的 `start_pos` 必须由 host 侧显式维护，不能从 cache position 反推。ratio=4/128
的 compressor 边界行为依赖 `start_pos % ratio`，SWA ring cache 依赖
`start_pos % window_size`。

## State 设计

整网 runner 为每层维护 attention state。第一版可以在 host 侧用 Python object 或 dict
描述：

```text
states[layer_id] = {
  "kv_cache": ...,
  "compressor": {
    "kv_cache": ...,
    "kv_state": ...,
    "score_state": ...,
  } | None,
  "indexer": {
    "kv_cache": ...,
    "compressor": {
      "kv_cache": ...,
      "kv_state": ...,
      "score_state": ...,
    },
  } | None,
}
```

shape 按当前 `models/block.py`、`attention_*`、`compressor_*` 和 `indexer.py` 的接口
保持一致。`max_seq_len` 第一版建议使用当前验证目标，例如 4096，而不是官方
`max_position_embeddings=1048576`。

以 `max_seq_len=4096`、`B=1` 估算，state 内存相对权重较小，可以先按层持久保存。
整网显存压力主要来自当前层权重，尤其是 MoE routed experts。

## 权重加载和布局

`weight_loader` 必须统一遵循当前 kernel 约定：

- 普通 linear 权重从官方 `[out, in]` 转成 `[in, out]` 后传入 kernel。
- HC pre/head 权重转置后传入：
  - block HC: `[MIX_HC, HC_DIM] -> [HC_DIM, MIX_HC]`
  - final HC head: `[HC_MULT, HC_DIM] -> [HC_DIM, HC_PAD]`
- LM head 是明确例外，保持官方 `[VOCAB, HIDDEN]` 布局，在 `head.py` 内用
  `b_trans=True` 对齐 `F.linear`。
- RMSNorm 权重保持官方一维 bf16。
- compressor `ape`、gate bias、tid2eid、attn sink 等非 linear 权重保持官方语义布局。

第一版 `weight_loader` 只做必要转换，不做 fp4/fp8 量化、不做 rotate_activation，也不做
多卡切分。

建议 `weight_loader` 提供基础 tensor 接口和 layer-level convenience 接口：

```text
get_tensor(name)
get_linear_t(name)
get_layer_attention(layer_id)
get_layer_hc(layer_id)
get_layer_moe_gate(layer_id)
get_layer_moe_shared(layer_id)
get_layer_moe_routed_pack(layer_id)
get_moe_routed_expert(layer_id, expert_id)
release(name)
release_prefix(prefix)
```

其中 `get_layer_moe_routed_pack(layer_id)` 是当前主方案使用的 packed-expert 接口；
`get_moe_routed_expert(layer_id, expert_id)` 为后续 selected-expert 备选方案保留。

## MoE 主方案：Packed Expert

当前主方案是 layer-by-layer + packed-expert。

每层 MoE routed expert 在加载阶段打包成当前 `models/moe.py` 需要的布局：

```text
routed_w1_t [N_EXPERTS, HIDDEN, MOE_INTER_DIM]
routed_w2_t [N_EXPERTS, MOE_INTER_DIM, HIDDEN]
routed_w3_t [N_EXPERTS, HIDDEN, MOE_INTER_DIM]
```

对应转换：

```python
routed_w1_t[e] = experts[e].w1.weight.t().contiguous()
routed_w2_t[e] = experts[e].w2.weight.t().contiguous()
routed_w3_t[e] = experts[e].w3.weight.t().contiguous()
```

单层 256 个 routed experts 的 bf16 权重规模约为：

```text
per expert:
  w1 [2048, 4096]
  w2 [4096, 2048]
  w3 [2048, 4096]
  total = 25,165,824 params = 48 MiB bf16

256 experts:
  48 MiB * 256 = 12 GiB bf16
```

如果加载和转置过程中官方布局权重与 packed 权重同时存在，峰值可能接近 24 GiB。再加上
attention、shared expert、gate、activation、state 和 PyPTO runtime buffer，packed-expert
在 64GB 单卡上存在显存风险，但它能最大化复用当前已经验证过的 `models/block.py` 和
`models/moe.py`，因此作为第一版整网主线。

## MoE 备选方案：Selected Expert

如果 packed-expert 在单层内出现显存或编译/runtime buffer 问题，切换到
selected-expert 路径。

selected-expert 的思路对齐 `../deepseek_v4_flash/inference/low_vram_moe.py`：

```text
1. run gate -> indices, weights
2. host 侧收集当前 token 实际命中的 expert_id
3. 只加载命中的 routed experts
4. 对每个命中 expert 运行 expert kernel
5. combine routed output + shared expert output
6. 释放已加载 expert 权重
```

这个路径能显著降低 routed expert 权重显存，但需要调整当前 MoE 执行方式。可选实现：

- 新增整网专用 MoE kernel，把 gate、selected expert、combine 拆开。
- 复用 `models/gate.py` 和 `models/expert.py`，由 host 侧循环命中 experts。

`weight_loader` 应从一开始保留 per-expert 获取接口，避免后续从 packed-expert 切换时重写
基础加载逻辑。

## PyPTO Runtime 复用与权重换入

整网 runner 需要在同一个 Python 进程中重复运行已编译 kernel，并在每层之间切换权重。
`../pypto-serving` 已经验证了以下基础能力：

- 长生命周期 `DistributedWorker` 可以反复运行已编译的 PyPTO program。
- 静态权重可以通过 worker-resident tensor 上传一次并跨多次 kernel 调用复用。
- KV cache 可以通过 `alloc_tensor` 常驻 worker device memory，并在 runner close 时
  `free_tensor` 释放。
- 启动阶段可以把原始权重转成 kernel-ready 布局，然后释放原始权重，降低 CPU 侧内存。

可以参考的实现点：

- `examples/model/qwen3_14b/runner/npu_executor.py`
  - `_compile_model()` 编译 kernel 并准备 runtime artifact。
  - `_kernel_weight()` 将二维权重转置成 kernel-ready BF16。
  - `_release_layer_weights()` 在 kernel-ready 权重生成后释放原始层权重。
- `examples/model/qwen3_14b/runner/npu_runner.py`
  - `_shared_l3_worker()` 创建并复用一个长生命周期 `DistributedWorker`。
  - `_StaticDeviceTensor` 和 `_l3_static_tensors` 将静态权重缓存到 worker device memory。
  - `_materialize_static_tensors()` 在 serving 前上传静态权重。
- `python/runtime/worker.py`
  - `alloc_tensor(init=...)` 支持分配 worker-resident tensor 并上传 host 数据。
  - `free_tensor(...)` 支持释放 worker-resident tensor。

但 `pypto-serving` 没有直接覆盖当前整网最关键的 low-vram 权重换入模式：

- 每层循环中频繁 `alloc_tensor -> run kernel -> free_tensor` 大权重。
- 预分配一组单层 weight buffer，并在每层之间反复 `copy_to` 覆盖内容。
- 单层 packed MoE 约 12 GiB routed expert 权重的上传、运行和释放。
- selected-expert 的 PyPTO host 调度。

因此这一部分必须单独实验验证。优先级如下：

1. **频繁 alloc/free 实验**
   - 编译一个简单 kernel。
   - 在同一 Python 进程和同一个 worker 中循环 43 次。
   - 每次上传一组接近单层权重规模的 tensor，运行 kernel，然后释放 tensor。
   - 观察是否有 runtime 错误、显存泄漏或碎片问题。
2. **固定 buffer pool 实验**
   - 启动时按单层最大权重集合分配一组 worker-resident buffer。
   - 每层只把新权重 copy 到已有 buffer，再用同一批 buffer handle 运行 kernel。
   - 如果频繁 alloc/free 不稳定，这应作为首选备选方案。
3. **packed MoE 峰值实验**
   - 单独验证一层 packed routed experts 能否在 64GB 单卡上完成上传和运行。
   - 如果 packed MoE 失败，再切换到 selected-expert 方案。

当前主方案仍是 layer-by-layer + packed-expert。权重换入方式优先尝试频繁
`alloc_tensor/free_tensor`；如果不稳定，则改成固定单层 weight buffer pool。这个调整不
改变已有 block kernel，也不改变 packed-expert 的计算语义。

## 第一版实现顺序

1. 实现 `serving/weight_loader.py`：
   - 读取 safetensors index。
   - 提供基础 tensor 和 transposed linear 获取接口。
   - 支持 packed routed experts。
   - 保留 per-expert 获取接口。
2. 实现 `serving/state.py`：
   - 根据 `compress_ratios[layer_id]` 创建每层 cache/state。
   - 支持 `max_seq_len=4096`。
3. 实现 `serving/runner.py`：
   - 单层 dispatch。
   - 根据 layer type 调用对应 block kernel。
   - 跑完释放当前层权重。
4. 实现整网 prefill：
   - embedding -> 43 层 -> head。
   - 先只输出 logits，不做 decode loop。
5. 实现 `serving/generate.py`：
   - 复用 prefill state。
   - 每步 decode 一个 token。
   - 先使用 greedy argmax。
6. 如 packed-expert 超出单卡显存或 PyPTO runtime buffer 不稳定，再切换 MoE 为
   selected-expert 备选路径。

## 需要继续细化的问题

- 频繁 `alloc_tensor/free_tensor` 和固定 weight buffer pool 两种权重换入方式的实测稳定性。
- packed-expert 单层在 Ascend 64GB 上的实际峰值显存和 runtime buffer 开销。
- selected-expert MoE 是否需要新增 kernel，还是可以复用 `gate.py` / `expert.py` 并由 host
  侧循环。
- state tensor 是全部常驻 NPU，还是 layer-by-layer 在 host/NPU 间搬运。
- 首个端到端验证用多长 prompt、生成多少 token，以及如何和
  `../deepseek_v4_flash` bf16 low-vram 输出对齐。
