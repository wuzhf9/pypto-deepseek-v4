# PyPTO Serving Qwen3-14B 推理流程参考

本文总结 `../pypto-serving` 中 Qwen3-14B NPU 路径的完整推理流程，以及它如何使用
`../pypto-serving/pypto-lib/models/qwen3/14b` 下的 PyPTO kernel 完成权重加载、kernel 编译、prefill 和 decode。

## 相关代码入口

- 示例入口：`../pypto-serving/examples/model/qwen3_14b/npu_generate.py`
- NPU executor：`../pypto-serving/examples/model/qwen3_14b/runner/npu_executor.py`
- NPU runner：`../pypto-serving/examples/model/qwen3_14b/runner/npu_runner.py`
- Host wrapper：`../pypto-serving/examples/model/qwen3_14b/runner/qwen3_l3_dispatch.py`
- Engine 生成循环：`../pypto-serving/python/core/engine.py`
- 模型加载器：`../pypto-serving/python/core/model_loader.py`
- PyPTO kernel 目录：`../pypto-serving/pypto-lib/models/qwen3/14b`

## 总体流程

Qwen3-14B NPU 推理分成五个阶段：

1. 从 Hugging Face 模型目录加载 config、tokenizer 和 safetensors 权重。
2. `Qwen314BPyptoExecutor` 动态加载 PyPTO kernel，并编译 prefill/decode 两个 distributed program。
3. 将 HF 权重转换成 kernel 需要的布局，上传静态权重和 RoPE 表到 PyPTO L3 worker。
4. 对 prompt 执行 prefill，写入 paged KV cache，并返回第一个 next-token logits。
5. 进入 autoregressive decode 循环，每步查 embedding、更新 KV cache、跑 fused decode、采样下一个 token。

## 1. 模型目录加载

`python/core/model_loader.py` 负责读取 Hugging Face 风格模型目录。

加载逻辑：

- 必须存在 `config.json`。
- tokenizer 通过 `TransformersTokenizerAdapter.from_pretrained(model_dir)` 加载。
- 权重从 safetensors 读取：
  - 如果存在 `model.safetensors.index.json`，按 index 中的 `weight_map` 收集 shard 文件。
  - 否则读取目录下所有 `*.safetensors`。

加载到 `RuntimeModel` 的主要权重：

- `model.embed_tokens.weight`
- `model.norm.weight` 或 `model.final_layernorm.weight`
- 可选 `lm_head.weight`；如果缺失则复用 embedding 权重
- 每层 `model.layers.{i}`：
  - `input_layernorm.weight`
  - `self_attn.q_proj.weight`
  - `self_attn.k_proj.weight`
  - `self_attn.v_proj.weight`
  - `self_attn.o_proj.weight`
  - 可选 `self_attn.q_norm.weight`
  - 可选 `self_attn.k_norm.weight`
  - `post_attention_layernorm.weight`
  - `mlp.gate_proj.weight`
  - `mlp.up_proj.weight`
  - `mlp.down_proj.weight`

这些权重先按 `RuntimeConfig.weight_dtype` cast 到 runtime device，后续再由 NPU executor 转换成 PyPTO kernel 需要的 CPU shared-memory 布局。

## 2. PyPTO kernel 加载与编译

`Qwen314BPyptoExecutor._compile_model()` 是 PyPTO kernel 接入的核心。

它会先定位 `pypto-lib/models/qwen3/14b` 目录，然后动态加载：

- `prefill_fwd.py`
- `decode_layer.py`

加载后将 PyPTO kernel 函数挂到 host wrapper：

```python
qwen3_l3_dispatch.prefill_fwd = qwen3_prefill_fwd.prefill_fwd
qwen3_l3_dispatch.decode_fwd = qwen3_decode_layer.decode_fwd
```

实际编译的是 host wrapper：

- `qwen3_l3_dispatch.qwen3_prefill_host`
- `qwen3_l3_dispatch.qwen3_decode_host`

host wrapper 本身用 `@pl.jit.host` 定义，作用是提供稳定的 HOST 级参数签名，并转调实际 PyPTO kernel。

编译前会做形状校验：

- `hidden_size = 5120`
- `intermediate_size = 17408`
- `num_attention_heads = 40`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `page_size = 128`
- decode kernel 固定 `BATCH = 16`
- decode kernel 固定 `NUM_LAYERS = 40`
- vocab pad 到 512 的倍数后必须等于 kernel 中的 `VOCAB = 152064`

编译输出封装为 `_CompiledKernels`，包含：

- 编译后的 prefill callable
- 编译后的 decode callable
- final norm 权重
- RoPE cos/sin 表
- padded lm head 权重
- 堆叠后的所有层权重
- prefill/decode 的共享输入输出 buffer

## 3. 权重转换和静态张量准备

HF 权重不能直接喂给 PyPTO kernel。`npu_executor.py` 会把每层权重转换成 kernel-ready layout。

线性层转换：

```python
weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu().share_memory_()
```

norm 权重转换：

```python
weight.view(1, -1).float().cpu()
```

每层会被整理成 `_KernelLayerWeights`：

- `input_rms_weight`
- `wq`
- `wk`
- `wv`
- `q_norm_weight`
- `k_norm_weight`
- `wo`
- `post_rms_weight`
- `w_gate`
- `w_up`
- `w_down`

然后 `_stack_decode_weights()` 将所有层沿第 0 维拼接，形成 fused all-layer kernel 使用的张量，例如：

- `decode_input_rms_weight`: `[num_layers, hidden]`
- `decode_wq`: `[num_layers * hidden, hidden]`
- `decode_wk`: `[num_layers * hidden, kv_hidden]`
- `decode_wv`: `[num_layers * hidden, kv_hidden]`
- `decode_w_gate`: `[num_layers * hidden, intermediate]`
- `decode_w_up`: `[num_layers * hidden, intermediate]`
- `decode_w_down`: `[num_layers * intermediate, hidden]`

这些名字带 `decode_`，但 prefill 和 decode 都复用同一批堆叠权重。

静态张量包括：

- final norm weight
- RoPE cos/sin
- padded lm head weight
- 所有堆叠层权重

`Qwen314BModelRunner` 初始化时会把这些张量放入 shared memory，并在第一次创建 `DistributedWorker` 后上传到 worker-resident tensor。后续 prefill/decode dispatch 复用这些静态 device tensor，避免每步重复上传。

## 4. KV cache 组织

该路径使用 paged KV cache。prefill 和 decode 使用同一个 device-resident KV pool。

runner 给 kernel 传入：

- `k_cache`
- `v_cache`
- `block_table`
- `slot_mapping`

`block_table` 描述每个 request 的逻辑 block 到物理 page id 的映射。

`slot_mapping` 描述当前 token 或 prompt chunk 中每个 token 应写入的物理 KV slot：

```text
physical_slot = page_id * page_size + page_offset
```

prefill 会为 prompt chunk 中每个 token 写一个 slot。

decode 每步只处理当前 token，因此每个 batch row 只有一个 `slot_mapping`，对应 `seq_len - 1` 的位置。

## 5. Prefill 流程

Engine 的 batch generate 逻辑：

1. tokenizer 编码 prompt。
2. 为每个 request 申请 KV allocation。
3. 通过 `lookup_embeddings()` 查 prompt token embedding。
4. 构造 `PrefillBatch`：
   - `token_ids`
   - `input_embeddings`
   - `seq_lens`
   - `kv_allocations`
5. 调用 `executor.run_prefill()`。

`Qwen314BModelRunner.run_prefill()` 做的事情：

1. `_prepare_prefill_inputs()` 将变长 prompt 打包成 token-major hidden buffer。
2. 填充 `seq_lens`、`chunk_lens`、`chunk_offsets`。
3. 根据 KV allocation 写 `block_table`。
4. 为 prompt chunk 中每个 token 计算 `slot_mapping`。
5. 取 shared paged KV cache 的 `key_pages` 和 `value_pages`。
6. 调用 `_run_distributed_program(compiled.prefill, ...)`。
7. 返回未 padded vocab 范围内的 logits。

`prefill_fwd.py` 内部流程：

1. `prefill_fwd()` 绑定动态维度，包括 user batch、packed token 数、block table、KV cache rows。
2. 对 `num_layers_actual` 循环。
3. 每层调用 `prefill_layer()`，完成：
   - input RMSNorm
   - Q/K/V projection
   - Q/K RMSNorm
   - RoPE
   - KV cache 写入
   - causal attention
   - output projection + residual
   - post-attention RMSNorm
   - SwiGLU MLP
   - down projection + residual
4. 所有层结束后，收集每个 batch 的最后一个 chunk token hidden。
5. 调用 `rms_lm_head()` 做 final RMSNorm + LM head matmul。
6. 输出 `[user_batch, vocab]` logits。

## 6. Decode 流程

prefill logits 先经过 sampler 采样得到第一个 generated token。

之后每步 decode：

1. 将上一轮采样出的 token append 到 request 输出。
2. 检查 EOS、stop string、max_new_tokens。
3. 为活跃 request 扩展一个 KV slot。
4. 对当前 token 查 embedding。
5. 构造 `DecodeBatch`：
   - `token_ids`
   - `hidden_states`
   - `seq_lens`
   - `kv_allocations`
6. 调用 `executor.run_decode()`。
7. 对 decode logits 采样下一 token。

`Qwen314BModelRunner.run_decode()` 做的事情：

1. `_prepare_decode_inputs()` 为活跃 request 准备：
   - 当前 token hidden
   - `seq_lens`
   - `block_table`
   - 当前 token 的 `slot_mapping`
2. 因为 decode kernel 固定 `BATCH = max_batch_size`，调用 `_pad_decode_inputs()` 将活跃 batch padding 到 kernel batch。
3. padding 行复制 row 0 的 hidden、seq_lens、block_table、slot_mapping。
4. 这样 padding 行即使写 KV，也只会重写 row 0 的相同位置，属于幂等写，不会破坏其他 request。
5. 调用 `_run_distributed_program(compiled.decode, ...)`。
6. 返回实际活跃 batch 范围内、未 padded vocab 范围内的 logits。

`decode_layer.decode_fwd()` 内部流程：

1. 将输入 hidden copy 到 `cur`。
2. 固定循环 `_FWD_NLAYERS = 40` 层。
3. 每层调用 `_decode_layer()`，完成：
   - input RMSNorm
   - Q/K/V projection
   - Q/K RMSNorm
   - RoPE
   - 写当前 token KV
   - 根据 `block_table` 读取历史 paged KV
   - grouped decode attention
   - output projection + residual
   - post-attention RMSNorm
   - SwiGLU MLP
   - down projection + residual
4. 所有层结束后调用 `rms_lm_head()`。
5. 输出 `[BATCH, VOCAB]` logits。

## 7. PyPTO dispatch 方式

runner 通过 `_run_distributed_program()` 统一执行 prefill/decode。

核心机制：

- 第一次执行时创建 `DistributedWorker([compiled.prefill.compiled, compiled.decode.compiled])`。
- 静态张量通过 `_StaticDeviceTensor` 标记。
- `_coerce_l3_arg()` 会把静态 CPU shared-memory tensor 上传成 worker-resident tensor，并缓存。
- 动态输入 buffer、KV cache、logits buffer 作为每次 dispatch 的参数传入。
- 最终调用：

```python
worker.run(callable_spec.compiled, *l3_args)
```

prefill 和 decode 都通过同一个 L3 worker 执行，且共享同一组 worker-resident 静态权重和同一个 paged KV cache。

## 8. 实际使用到的 qwen3/14b PyPTO 文件

当前 `../pypto-serving` 的 Qwen3-14B NPU serving/generate 路径实际使用这些文件：

| 文件 | 作用 |
| --- | --- |
| `config.py` | Qwen3-14B 固定模型形状、动态维度、tiling 常量、vocab、layer 数等。 |
| `prefill_fwd.py` | all-layer prefill kernel；处理 prompt chunk、写 KV cache、输出 next-token logits。 |
| `decode_layer.py` | all-layer paged decode kernel；每步处理一个 token，读取/写入 paged KV cache，输出 logits。 |
| `rms_lm_head.py` | final RMSNorm + LM head projection，被 prefill 和 decode 共同调用。 |

同目录中未被当前 serving/NPU 推理链路引用的文件：

| 文件 | 备注 |
| --- | --- |
| `qwen3_14b_l3_generate.py` | 定义 unified generation kernel builder，但当前仓库只有定义，没有调用点。 |
| `qwen3_14b_decode_ssn_draft.py` | 当前 serving/NPU 路径没有引用。 |

## 9. 后续复用该流程时的注意点

- PyPTO kernel 当前强绑定 Qwen3-14B 形状，不是通用 Qwen3 loader。
- decode kernel 固定 `BATCH = 16`、`NUM_LAYERS = 40`、`VOCAB = 152064`。
- runtime `max_batch_size` 必须匹配 decode kernel 的固定 batch。
- runtime `page_size` 必须是 128。
- prefill 支持动态 user batch 和 packed prompt tokens。
- decode 是固定 batch kernel，活跃 batch 不足时由 runner 做 row 0 replicate padding。
- prefill 和 decode 使用同一个 paged KV pool，因此 prefill 写入的 prompt KV 可被 decode 直接读取，不需要额外 bridge。
- `lm_head.weight` 缺失时会复用 embedding 权重；否则单独加载并 pad 到 kernel vocab。
- 权重转换和堆叠发生在 executor 编译阶段，原始 layer 权重随后会被释放，避免重复占用内存。
