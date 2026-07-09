# DeepSeek V4 Flash Runner Implementation Plan

本文记录 `serving/runner.py` 的实现方案。runner 是 host 侧调度层，不新增模型计算逻辑，
也不放在 `models/` 目录下。目标是在单卡 Ascend NPU 上按层加载权重、调用已经实现的
PyPTO kernel、维护 state，并最终完成 `input_ids -> logits` 的整网推理。

## Runner 职责

`serving/runner.py` 负责把已有模块串成整网：

```text
input_ids
  -> embedding_fwd
  -> expand hidden to HC copies
  -> for layer_id in 0..42:
       select block kernel
       load current layer weights
       build current layer auxiliary inputs
       run block kernel
       update persistent layer state
       release current layer weights
  -> head_fwd
  -> logits
```

runner 不负责以下内容：

- 不实现新的 attention、MoE、compressor、indexer 或 head 计算。
- 不实现多卡并行、EP、paged attention、chunked prefill、MTP。
- 不在 `models/` 下新增整网 transformer kernel。
- 不做 fp4/fp8 runtime 量化，也不做 fp4 前的 `rotate_activation`。

当前 `models/block.py`、`models/embedding.py`、`models/head.py` 已经覆盖整网需要的
PyPTO 入口；`serving/weight_loader.py` 负责权重读取和布局转换；`serving/state.py`
负责 cache/state、topk index 和 RoPE 输入。

## Runtime 选择

直接实现 `DeepSeekV4Runner`。runner 内部可以保留 debug/backend 选项，但不先实现独立的
`SmokeRunner`，避免 block kernel 选择、参数组装、输出映射和 state 更新逻辑出现两套实现。

当前主路径使用 direct backend。worker-resident backend 曾用于验证 state/cache 和 hidden
常驻 NPU 的可行性，但 profile 显示主要瓶颈仍在 block kernel runtime，worker 路径的收益
不足以作为后续主线：

```text
DeepSeekV4Runner
  - 负责整网 prefill/decode 流程
  - 负责权重加载、state 管理、kernel dispatch
  - 使用 direct backend 做 smoke/debug 和完整推理
```

direct backend 对齐 `models/golden.py::run_jit` 的直接 `compiled(*args, config=...)`
调用方式，已经验证能够跑通完整推理。后续优化优先转向 PyPTO kernel 内部计算路径，而不是
继续扩展 `--backend worker`。

## 对外接口

建议第一版 runner 提供以下接口：

```python
class DeepSeekV4Runner:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device_id: int = 0,
        max_seq_len: int = 4096,
        backend: Literal["direct", "worker"] = "direct",
    ) -> None: ...

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def decode(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor: ...

    def close(self) -> None: ...
```

输入约束：

- `input_ids` shape 为 `[1, S]` 或 `[1, 1]`，当前 kernel 固定 `B=1`。
- `prefill` 固定 `start_pos=0`。
- `decode` 必须由调用方显式传入 `start_pos`，不能从 cache 位置反推。
- 第一版 `max_seq_len=4096`，和当前 kernel/state 的 TOPK 上限一致。

输出：

```text
prefill(input_ids) -> logits [1, VOCAB]
decode(input_ids, start_pos) -> logits [1, VOCAB]
```

`serving/generate.py` 后续可以在 runner 之上实现 tokenizer、greedy argmax 或采样。

## Kernel 编译与 Dispatch

runner 启动时准备以下 kernel：

```text
embedding_fwd
block_swa_hash_prefill_fwd
block_swa_hash_decode_fwd
block_csa_hash_prefill_fwd
block_csa_hash_decode_fwd
block_hca_topk_prefill_fwd
block_hca_topk_decode_fwd
block_csa_topk_prefill_fwd
block_csa_topk_decode_fwd
head_fwd
```

block 选择规则来自官方 `compress_ratios` 和 `n_hash_layers`：

```text
ratio=0,   hash_route=True  -> block_swa_hash_*_fwd
ratio=4,   hash_route=True  -> block_csa_hash_*_fwd
ratio=128, hash_route=False -> block_hca_topk_*_fwd
ratio=4,   hash_route=False -> block_csa_topk_*_fwd
```

正常 block 只有 `layer_id=0..42`。`compress_ratios[43]` 属于 MTP，不在当前 runner 范围内。

## Dynamic Shape 编译策略

`pypto-serving` 的 Qwen3 prefill 已经验证了 `pl.dynamic(...)` 可以用于 batch/token 相关
shape，并通过一次编译复用不同实际长度的输入。可参考：

```text
../pypto-serving/pypto-lib/models/qwen3/14b/prefill_fwd.py
../pypto-serving/examples/model/qwen3_14b/runner/npu_executor.py
../pypto-serving/examples/model/qwen3_14b/runner/npu_runner.py
```

Qwen3 的做法是：

```text
1. 编译时使用最大容量 dummy tensor，例如 batch * max_seq。
2. kernel signature 中 batch/token 相关维度使用 pl.dynamic(...)。
3. kernel 内对实际动态维度调用 bind_dynamic(...)。
4. 运行时 host 侧传入实际长度切片，例如 hidden[:total_tokens]。
```

当前 runner 采用同一原则。以 `max_seq_len=4096` 为第一版容量上限：

```text
compile:
  使用 max_seq_len=4096 的最大 shape 编译每个 block kernel

prefill runtime:
  hidden[:, :S, ...]
  topk[:, :S, ...]
  cos[:S]
  sin[:S]
  ratio=4/128 compressor rope 使用实际 compressed block 数

decode runtime:
  S 固定为 1
  start_pos 由 host 显式传入 state helper，决定 cache slot、topk 和 compressor 边界
```

因此需要验证的不是 PyPTO 是否支持 dynamic shape，而是当前这组 block kernel 在一次最大容量
编译后，能否稳定复用不同实际 `S` 的输入切片。验证时应覆盖：

```text
prefill S = 1, 3, 4, 5, 127, 128, 129, 4096 范围内的代表值
decode start_pos 覆盖 window/ratio 边界
四种 block 形态：swa_hash、csa_hash、hca_topk、csa_topk
```

## Prefill 调度

prefill 流程：

```text
1. validate input_ids shape [1, S], S > 0, S <= max_seq_len
2. load embedding weight
3. run embedding_fwd -> hidden [1, S, 4096]
4. expand hidden -> block input [1, S, 4, 4096]
5. for layer_id in 0..42:
     spec = state.layer_spec(layer_id)
     aux = state.build_prefill_inputs(layer_id, S)
     weights = weight_loader layer-level getters
     scratch = allocate scratch tensors required by block specs
     run selected block prefill kernel
     state.update_layer_state(layer_id, block outputs)
     hidden = block out
     release current layer weights
6. load final head weights
7. run head_fwd(hidden, head weights) -> logits [1, VOCAB]
8. return logits
```

prefill 后，`DeepSeekV4State` 内保存每层 decode 需要的 SWA/HCA/CSA cache 和 compressor/indexer
state。runner 不从输出 tensor 名称中推断语义，只调用 `state.update_layer_state(...)`。

## Decode 调度

decode 流程：

```text
1. validate input_ids shape [1, 1]
2. validate 0 < start_pos < max_seq_len
3. run embedding_fwd -> hidden [1, 1, 4096]
4. expand hidden -> block input [1, 1, 4, 4096]
5. for layer_id in 0..42:
     spec = state.layer_spec(layer_id)
     aux = state.build_decode_inputs(layer_id, start_pos)
     weights = weight_loader layer-level getters
     scratch = allocate scratch tensors required by block specs
     run selected block decode kernel
     state.update_layer_state(layer_id, block outputs)
     hidden = block out
     release current layer weights
6. run head_fwd(hidden, head weights) -> logits [1, VOCAB]
7. return logits
```

`start_pos` 必须由上层 generation loop 维护。原因是：

- SWA ring cache 使用 `start_pos % window_size`。
- ratio=4/128 compressor 使用 `(start_pos + 1) % ratio == 0` 判断是否压缩。
- CSA/HCA compressed topk 的 offset 与当前 decode 位置有关。

## 权重组织

runner 不直接拼 checkpoint 名称，而是只使用 `DeepSeekV4WeightLoader` 的 layer-level 接口：

```text
global:
  get_embedding_weight()
  get_head_weights()

per layer:
  get_layer_hc(layer_id)
  get_layer_attention_common(layer_id)
  get_layer_moe_gate(layer_id)
  get_layer_moe_shared(layer_id)
  get_layer_moe_routed_pack(layer_id)

ratio=128:
  get_layer_compressor_ratio128(layer_id)

ratio=4:
  get_layer_compressor_ratio4_attention(layer_id)
  get_layer_indexer(layer_id)
```

所有普通 linear 权重已经由 loader 转成 `[in, out]` 布局。LM head 是例外，保持官方
`[VOCAB, HIDDEN]`，由 `head.py` 内部用 `b_trans=True` 对齐 `F.linear`。

第一版主方案继续使用 packed-expert：

```text
routed_w1_t [256, 4096, 2048]
routed_w2_t [256, 2048, 4096]
routed_w3_t [256, 4096, 2048]
```

如果 packed experts 在单卡 64GB 上出现显存或 runtime buffer 问题，再切换到
selected-expert 备选方案。runner 的权重加载路径需要保留 `get_moe_routed_expert(...)`
可用性，避免后续重写 loader。

## Tensor 生命周期

第一版采用 correctness-first 策略，先通过 direct backend 验证参数组装和计算链路：

```text
CPU:
  - weight_loader 按需加载当前层权重
  - hidden 每层输出后回到 host，下一层再作为 host tensor 传入
  - embedding/head 输入输出 buffer 可以使用 host shared tensor

NPU:
  - direct backend 下 state 作为普通 host tensor 参数参与每次 kernel 调用
  - 当前层权重和 scratch 只在当前 kernel 调用期间有效
  - 当前层运行结束后释放当前层权重和 scratch
```

state/cache 和 hidden 常驻 NPU 已通过实验验证可行，但小规模 profile 没有体现出足够收益。
当前优化重点改为拆分 block kernel runtime 的内部耗时，定位 attention、MoE、HC、linear、
sparse attention 等子路径的真实占比。

## Scratch Tensor 处理

当前 block kernel 仍有一批 HC padding/scratch tensor 需要由外部传入，主要用于非 16 对齐
tail 和 PyPTO buffer 对齐。runner 不手写 shape 推导逻辑，建议复用各 kernel 的
`build_*_specs(...)` 或抽出 shared spec helper，按实际 `seq_len/start_pos` 创建 scratch。

原则：

- 业务输出 tensor 必须交给 `state.update_layer_state(...)` 或作为下一层 hidden。
- 中间 scratch tensor 不进入 state。
- scratch 只在单次 kernel 调用期间有效。
- 如果后续 kernel 已经把某些 dynamic scratch 迁入内部，runner 对应删除外部参数即可。

## 输出映射

block 输出统一按语义映射，而不是按 tuple 位置硬编码。runner 应在每次 kernel 调用后构造：

```python
outputs = {
    "kv_cache_out": ...,
    "comp_kv_state_out": ...,
    "comp_score_state_out": ...,
    "comp_cache_out": ...,
    "attn_comp_kv_state_out": ...,
    "attn_comp_score_state_out": ...,
    "attn_comp_cache_out": ...,
    "idx_kv_cache_out": ...,
    "idx_comp_kv_state_out": ...,
    "idx_comp_score_state_out": ...,
    "out": ...,
}
```

只包含当前 block 形态实际返回的 key。然后：

```text
state.update_layer_state(layer_id, outputs)
hidden = outputs["out"]
```

这样可以避免 SWA/HCA/CSA 的输出数量不同导致 runner 代码 fragile。

## 验证顺序

runner 实现后按以下顺序验证：

1. **single kernel smoke**
   - 使用 `DeepSeekV4Runner` 的 debug/backend 选项调用 `embedding_fwd`，确认输入输出 shape/dtype。
   - 调用一个 `block_swa_hash_prefill_fwd`，确认参数组装正确。
   - 调用 `head_fwd`，确认 logits shape 为 `[1, VOCAB]`。
2. **single layer prefill**
   - 对 layer 0 运行 embedding -> block -> head。
   - 和同一层 golden 或 `../deepseek_v4_flash` 中间结果对齐。
3. **multi layer short prefill**
   - 先跑前 1、2、4 层，确认 state 更新和权重释放稳定。
   - 再跑 43 层 prefill，先只看 logits 是否有限且 shape 正确。
4. **decode smoke**
   - 使用 prefill 后的 state，运行一个 decode token。
   - 覆盖 `start_pos` 触发 ratio=4 和 ratio=128 compressor 边界的场景。
5. **text generation**
   - 在 `serving/generate.py` 中加入 tokenizer 和 greedy argmax。
   - 与 `../deepseek_v4_flash` bf16 low-vram 输出做短 prompt 行为对比。

## 风险点

- 单层 packed routed experts 约 12 GiB bf16；加载、转置、上传和 runtime buffer 的峰值需要
  在 Ascend 上实测。
- 频繁传入或上传大权重可能遇到 runtime 临时 buffer 峰值、碎片或泄漏；必要时改成固定
  weight buffer pool。
- 按 Qwen3 模式用最大容量编译一次后，当前 block kernel 是否能稳定复用不同实际 `S` 的
  输入切片需要实测；如果某个 block 形态触发重复编译或 runtime 错误，需要单独调整该
  kernel 的 dynamic shape 写法。
- host/NPU 每层往返仍然存在，但当前 profile 显示它不是首要瓶颈。
- worker-resident state/cache 和 hidden 路径已实验并回退，不作为后续主线。
- 如果 packed-expert 不可行，切换 selected-expert 时需要新增或调整 MoE 调度，但不影响
  attention、compressor、indexer、block 的已验证逻辑。

## 后续细化项

- 给 block runner 抽一个稳定的参数组装 helper，避免每个 block 形态重复拼长参数列表。
- 实测 packed-expert 单层峰值显存。
- 拆分 block kernel runtime，定位最耗时的子模块和切分方式。

## 当前实现进展

当前已经新增 `serving/runner.py` 和 `serving/generate.py`，实现了第一版端到端生成链路。

已实现内容：

- `DeepSeekV4Runner.prefill(input_ids)`
  - `embedding_fwd -> block loop -> optional head_fwd`。
  - 支持 `max_layers=43` 完整层数。
- `DeepSeekV4Runner.decode(input_ids, start_pos=...)`
  - 复用同一个 runner 实例中的 prefill state/cache。
  - `start_pos` 由 CLI 或调用方显式传入。
- `direct` backend
  - 对齐 `models.golden.run_jit` 的直接 `compiled(*args, config=...)` 调用方式。
  - 当前第一版默认使用 `backend="direct"`。
- block dispatch
  - 根据 `LayerSpec.ratio/hash_route` 选择四种 block 形态：
    `swa_hash`、`csa_hash`、`hca_topk`、`csa_topk`。
  - prefill/decode 分别选择对应 kernel。
- 参数组装
  - 复用各 block 的 `build_*_specs(...)` 作为参数顺序来源。
  - 通过 `DeepSeekV4WeightLoader` 加载当前层权重。
  - 通过 `DeepSeekV4State` 构造 topk、RoPE、cache slot、compressor 边界等辅助输入。
  - block 输出按语义 key 传给 `state.update_layer_state(...)`。
- CLI smoke 入口
  - `python serving/runner.py ...`
  - `--decode-steps N` 会先执行 prefill，再从 `start_pos=seq_len` 开始连续 decode `N` 步。
  - 带 head 时使用上一步 logits 的 argmax 作为下一步 decode token；`--no-head` 时使用随机 token 做 kernel 串联验证。
  - `--profile` 打印 host 侧 timing，用于定位权重加载、tensor materialize、kernel compile/run 和 state 更新开销。
  - `--verbose-layer-log` 打印逐层 start/done 和 finite 检查；默认关闭，避免长生成输出过多。
  - `--expert-cache-dir PATH` 从离线 bf16 expert cache 读取 MoE routed expert 权重，缺失 expert 回退在线转换。
- 生成入口
  - `python serving/generate.py ...`
  - 按 `../deepseek_v4_flash` 的流程调用官方 `encode_messages(...)`，再调用 tokenizer `encode(...)`。
  - 支持 prefill、greedy decode、temperature sampling、EOS 停止和 `--max-new-tokens`。
  - `--verbose-layer-log` 可透传到 runner，用于定位某一层的 shape、dtype 和 finite 状态。
  - 复用 `DeepSeekV4Runner` 和 `DeepSeekV4State`，不维护第二套模型执行逻辑。

当前未启用 `worker` backend。state/cache 与 hidden 常驻 NPU 的实验路径已经回退，原因是
profile 显示 block kernel runtime 占每层耗时约 90%，worker 路径确定性优化空间不足以作为
后续主线。下一步优化应聚焦 kernel runtime。

Ascend 端已经完成的关键验证：

- `max_layers=43` prefill 能跑通。
- `max_layers=43` prefill + head 能跑通。
- `max_layers=43` prefill + decode 能跑通。
- `serving/generate.py` 使用真实 tokenizer、真实权重和完整 43 层，能够生成逻辑正确的中文回复。

已验证的完整生成命令：

```bash
python serving/generate.py \
  --checkpoint ~/dsv4_ckpt \
  --encoding-path ~/dsv4_ckpt/encoding \
  --expert-cache-dir ~/dsv4_bf16_expert_cache \
  --prompt '你好' \
  --max-new-tokens 20 \
  -p a2a3 -d {}
```

该命令输出：

```text
AI: 你好！很高兴见到你。有什么我可以帮你的吗？无论是聊天、解答问题，还是提供
prompt_tokens: 5
generated_tokens: 20
elapsed_s: 1576.100
output_tps: 0.013
```

## 当前验证方式

本地已完成的轻量检查：

```bash
python -m py_compile serving/runner.py
python -m py_compile serving/generate.py
python -m py_compile models/golden.py
pytest -q tests
```

本地没有完整 PyPTO native extension 环境，直接 import/run PyPTO 会触发
`pypto.pypto_core.DataType` 缺失，因此功能验证需要在 Ascend 环境进行。

远程 smoke 可以继续保留从小到大的验证顺序，用于修改 runner 或 block 参数组装后快速回归。

最小 prefill：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 1 --max-layers 1 --no-head
```

这条命令只验证：

```text
embedding_fwd -> block_swa_hash_prefill_fwd(layer 0)
```

如果通过，再验证带 head：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 1 --max-layers 1
```

然后逐步扩大：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 3 --max-layers 1 --no-head
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 1 --max-layers 2 --no-head
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 1 --max-layers 3 --no-head
```

`max_layers=1/2/3` 分别覆盖：

```text
layer 0: swa_hash
layer 1: swa_hash
layer 2: csa_hash
```

再继续验证 `max_layers=4/5`，覆盖 `hca_topk` 和 `csa_topk`。

decode 验证从 prefill 后接单步 decode 开始：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 13 --max-layers 1 --decode-steps 1
```

扩大覆盖时优先使用：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 13 --max-layers 5 --decode-steps 1
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 13 --max-layers 43 --decode-steps 1
```

完整生成验证使用：

```bash
python serving/generate.py \
  --checkpoint ~/dsv4_ckpt \
  --encoding-path ~/dsv4_ckpt/encoding \
  --expert-cache-dir ~/dsv4_bf16_expert_cache \
  --prompt '你好' \
  --max-new-tokens 20 \
  -p a2a3 -d {}
```

性能采样时增加 `--profile`，优先使用短生成场景：

```bash
python serving/generate.py \
  --checkpoint ~/dsv4_ckpt \
  --encoding-path ~/dsv4_ckpt/encoding \
  --expert-cache-dir ~/dsv4_bf16_expert_cache \
  --prompt '你好' \
  --max-new-tokens 2 \
  -p a2a3 -d {} \
  --profile
```

如果已经生成离线 expert cache，可以增加：

```bash
--expert-cache-dir ~/dsv4_bf16_expert_cache
```

`--profile` 会输出如下维度：

```text
embedding.weight / embedding.materialize / embedding.kernel / embedding.total
layer.values.aux / hc / attn / gate / shared / routed_pack / ffn_norm / compressor/indexer
layer.materialize / layer.kernel / layer.state_update / layer.release / layer.total
layer.weight_loader
head.weight / head.materialize / head.kernel / head.total
prefill.total / decode.total
```

其中 `*.kernel` 会拆出 `compile_ms`、`run_ms` 和 `cache_hit`，用于区分首次编译、编译缓存命中
和实际 runtime 执行开销。
`layer.weight_loader` 是聚合统计，格式为 `name=ms/count`，包括 raw load、scale load、
fp4/fp8 dequant、transpose 和 copy 等权重加载细分耗时。

当前已实现 safetensors file handle cache。`DeepSeekV4WeightLoader` 会按 shard 文件路径复用
`safe_open(...)` reader，避免每个 tensor 读取都重新打开 safetensors 文件。`release_prefix(...)`
只清理 tensor cache，不关闭 file handle；`release()` 无参数或 `close()` 会释放 tensor cache
并关闭所有 file handle。

当前已实现 routed expert 离线 bf16 expert cache。`DeepSeekV4WeightLoader.get_moe_routed_expert(...)`
会优先读取 `--expert-cache-dir` 中的 `layer_NNN_experts.safetensors`，缺失时回退到
官方 checkpoint 在线 fp4 反量化和转置。prefill 的 full routed pack 和 decode 的 selected
experts 都复用这一条 per-expert 加载路径。

## 优化计划

1. 降低验证噪声。
   - 已实现 `--verbose-layer-log`，runner 逐层日志默认关闭。
   - 默认只保留关键的 prompt token 数、生成 token 数、总耗时和最终文本输出。
   - 该优化不改变计算路径，只减少长生成时的日志量。
2. 重新做短场景 profile。
   - 使用 `prompt='你好' --max-new-tokens 2 --profile`。
   - 重点区分权重读取、权重 materialize、kernel compile、kernel run、state update 和 head 开销。
   - 不先假设瓶颈，后续优化按 profile 数据排序。
3. 继续优化权重加载路径。
   - 已完成 safetensors file handle cache。
   - 已完成 routed expert 离线 bf16 expert cache。
   - 根据 profile 决定是否增加非 routed 权重的离线 bf16/runtime-layout cache。
   - 候选对象包括频繁发生 fp8/fp4 反量化、转置或 dtype 转换的 attention、compressor、indexer、gate、shared expert 权重。
4. 聚焦 block kernel runtime。
   - 以 `block_*_prefill/decode` 为入口增加或临时插入子模块 timing/诊断输出。
   - 优先拆分 MoE、attention、HC pre/post、sparse attention、linear 路径的耗时。
   - 根据占比决定是否调整 kernel 切分、拆分大 kernel、减少无效专家计算或重写局部算子。
5. selected-expert decode 路径作为备选。
   - 当前 packed-expert 已证明能够跑通并满足内存需求。
   - 如果 profile 显示 decode 的 packed routed expert 权重加载仍是绝对瓶颈，再评估 selected-expert。
   - selected-expert 会引入新的 MoE kernel 和权重接口，不作为当前主路径。
