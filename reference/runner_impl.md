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

`DistributedWorker` 由 PyPTO 框架提供，不是 `pypto-serving` 自己实现的功能。虽然当前没有
多卡并行需求，但 `DistributedWorker` 仍然适合作为单卡长生命周期 runtime 管理器使用：

```text
DeepSeekV4Runner
  - 负责整网 prefill/decode 流程
  - 负责权重加载、state 管理、kernel dispatch
  - 第一版使用 direct backend 做 smoke/debug
  - 后续使用单设备 DistributedWorker 做正式整网运行
```

单设备 `DistributedWorker` 的作用是复用已编译 kernel、管理 worker-resident tensor，并让
state cache 常驻 NPU。direct backend 可以对齐 `models/golden.py::run_jit` 的直接
`compiled(*args, config=...)` 调用方式，用于单 kernel 或短链路 debug；它不是另一套 runner。

`DistributedWorker` 对 host tensor 有一个重要约束：传给 worker 的 host tensor 必须是 worker
fork 前已经分配的 shared-memory tensor。按层新加载的普通权重 tensor 不能直接作为 worker
参数传入。因此第一版先使用 direct backend 验证 runner 参数组装和计算链路；worker backend
需要先实现 shared host weight buffer pool，再启用 state 常驻 NPU。

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
  - worker backend 启用后，attention/compressor/indexer state 常驻 worker-resident DeviceTensor
  - 当前层权重和 scratch 只在当前 kernel 调用期间有效
  - 当前层运行结束后释放当前层权重和 scratch
```

worker backend 中的 state 常驻 NPU 对齐 `pypto-serving` 的 KV cache 常驻 NPU 模式。我们的
state 包括：

```text
kv_cache
comp_cache
comp_kv_state
comp_score_state
attn_comp_cache
attn_comp_kv_state
attn_comp_score_state
idx_kv_cache
idx_comp_kv_state
idx_comp_score_state
```

hidden 不在第一版中跨层常驻 NPU。这样可以避免先验证“上一层 block 输出的 DeviceTensor 能否
直接作为下一层 block 输入复用”这一额外问题。代价是每层 hidden 都有 host/NPU 往返，但当前
目标是逻辑正确，不追求性能。

后续整网能够正确跑通后，如果需要优化性能，再把 hidden 改为常驻 NPU：

- hidden 常驻 NPU，在层间直接传递。
- 按最大单层形状预分配 weight buffer pool，每层覆盖内容而不是频繁 alloc/free。

这些优化不改变 kernel 计算语义。

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
- host/NPU 每层往返会很慢，但当前目标是逻辑正确，不追求性能。
- hidden 常驻 NPU 是后续性能优化项，不作为第一版 runner 的正确性依赖。
- 如果 packed-expert 不可行，切换 selected-expert 时需要新增或调整 MoE 调度，但不影响
  attention、compressor、indexer、block 的已验证逻辑。

## 后续细化项

- 确认 PyPTO 单设备 `DistributedWorker` 的最小创建、编译、运行、DeviceTensor state 管理和 close API。
- 给 block runner 抽一个稳定的参数组装 helper，避免每个 block 形态重复拼长参数列表。
- 实测 packed-expert 单层峰值显存。
- 实测每层权重 host 传入、worker-resident 权重上传释放和固定 weight buffer pool 三种策略。

## 当前实现进展

当前已经新增 `serving/runner.py`，实现了第一版 `DeepSeekV4Runner`。

已实现内容：

- `DeepSeekV4Runner.prefill(input_ids)`
  - `embedding_fwd -> block loop -> optional head_fwd`
  - 默认 `max_layers=1`，用于先验证 layer 0 prefill smoke。
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
  - `--routed-pack-cache-dir PATH` 从离线 bf16 routed pack cache 读取 MoE routed expert packed 权重，缺失层回退在线转换。

当前未启用 `worker` backend。原因是 PyPTO `DistributedWorker` 要求传入 worker 的 host
tensor 必须是 worker fork 前已经分配的 shared-memory tensor；而当前 layer-by-layer 权重
是在运行过程中按层加载的普通 CPU tensor。worker backend 需要先实现 shared host weight
buffer pool，再启用 state 常驻 NPU。

## 当前验证方式

本地已完成的轻量检查：

```bash
python -m py_compile serving/runner.py
python -m py_compile models/golden.py
```

本地没有完整 PyPTO native extension 环境，直接 import/run PyPTO 会触发
`pypto.pypto_core.DataType` 缺失，因此功能验证需要在 Ascend 环境进行。

推荐第一条远程 smoke 命令：

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

性能采样时增加 `--profile`：

```bash
python serving/runner.py --checkpoint ../deepseek_v4_flash -p a2a3 -d 0 -s 13 --max-layers 5 --decode-steps 1 --profile
```

如果已经生成离线 routed pack cache，可以增加：

```bash
--routed-pack-cache-dir ~/dsv4_bf16_routed_pack_cache
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

## 下一步计划

1. 在 Ascend 上验证第一版 direct runner。
   - 先跑 layer 0 prefill smoke。
   - 再跑带 head 的 logits smoke。
   - 然后扩大到前 3、5 层，覆盖所有 block 形态。
2. 根据远程报错修正 runner 参数组装。
   - 重点检查 spec shape、权重布局、state output key 和 `ffn_norm_w` 命名。
3. 验证 decode。
   - 先用 `max_layers=1` 验证 `swa_hash_decode`。
   - 再用 `max_layers=5` 覆盖 `csa_hash_decode`、`hca_topk_decode`、`csa_topk_decode`。
   - 最后验证 `max_layers=43 --decode-steps 1`，确认完整 prefill state 能被 decode 消费。
4. 设计并实现 shared host weight buffer pool。
   - 让 worker fork 前创建固定 shared-memory host buffers。
   - 每层加载权重后 copy 到已有 buffer。
   - 再启用 `DistributedWorker` backend 和 state 常驻 NPU。
5. direct runner 跑通足够层数后，再实现 `serving/generate.py`。
   - tokenizer、prefill、decode loop、greedy argmax。
   - 与 `../deepseek_v4_flash` bf16 low-vram 路径做短 prompt 行为对比。
