# Performance Optimization Plan

本文记录下一阶段的性能优化方案。当前目标已经从正确性验证切换为端到端生成速度优化，
但仍保持单卡 Ascend NPU、bf16 计算、完整权重加载和完整推理流程不变。

## 当前结论

当前完整推理已经能够在 Ascend NPU 上加载 DeepSeek V4 Flash 权重并生成逻辑正确的句子。
根据已有 profile，逐层运行时间主要集中在 block kernel runtime，约占每层耗时的 90%。

已经验证过的方向：

- `worker` backend、state/cache 常驻 NPU、hidden 常驻 NPU等 host/runtime 调度优化可以跑通，
  但在当前 profile 下不是主要瓶颈。
- safetensors file handle cache 和 routed expert 离线 bf16 pack cache 已经降低了部分权重
  加载成本。
- 后续主线应聚焦 PyPTO kernel runtime，而不是继续优先优化 host 调度路径。

因此优化策略不是平均地优化所有底层 kernel，而是先定位端到端最热的 block 形态和子路径，
再针对热点 kernel 做自底向上的实现优化。

## 基准场景

性能优化必须使用固定基准，避免不同 prompt、不同 decode 长度和首次编译开销导致结论不稳定。

推荐基准：

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

必要时增加短 runner 场景：

```bash
python serving/runner.py \
  --checkpoint ~/dsv4_ckpt \
  --expert-cache-dir ~/dsv4_bf16_expert_cache \
  -p a2a3 -d {} \
  -s 1 --decode-steps 1 --profile
```

需要覆盖的 shape：

- prefill: `S=1`、`S=13`、`S=128`
- decode: `start_pos=1`、`start_pos=127`、`start_pos=128`、`start_pos=129`
- 长生成抽样：`max-new-tokens=30` 或更长，用于确认优化不破坏连续 decode 状态

每个候选优化至少跑 3 次，记录中位数。首次编译时间和 cache hit 后 runtime 要分开看。

## Profile 维度

现有 runner 的 `--profile` 会输出：

```text
embedding.weight / embedding.materialize / embedding.kernel / embedding.total
layer.values.aux / hc / attn / gate / shared / routed_pack / ffn_norm / compressor/indexer
layer.materialize / layer.kernel / layer.state_update / layer.release / layer.total
layer.weight_loader
head.weight / head.materialize / head.kernel / head.total
prefill.total / decode.total
```

其中 `*.kernel` 包含：

```text
compile_ms
run_ms
cache_hit
```

优化判断只以 `cache_hit=True` 后的 `run_ms` 为主。`compile_ms` 只用于确认是否存在动态 shape
导致的重复编译问题。

## 热点定位顺序

第一步按 block 形态统计：

```text
swa_hash
csa_hash
hca_topk
csa_topk
```

分别统计 prefill 和 decode 的单层耗时，确认最热的 block 形态。

第二步在最热 block 中临时拆分或插入 timing，定位子路径：

```text
hc_pre
attention_qkv
sparse_attn
attention_out
hc_post
gate
moe shared expert
moe routed expert
ffn_norm / residual
compressor / indexer state update
```

这些拆分可以作为临时诊断代码，不一定长期保留。最终优化应该回到端到端 block runtime
是否下降，而不是只看单个独立 kernel 的 benchmark。

## 优化优先级

### 1. MoE 路径

MoE 通常是最值得优先检查的部分，因为每层 routed expert pack 很大，并且计算包含 gate、
topk、shared expert、routed expert 和 combine。

候选方向：

- 检查 packed routed expert 是否存在 decode `S=1` 下的大量无效循环。
- 评估 selected-expert decode 路径，避免每步使用完整 packed expert 权重参与调度。
- 优化 route-major 循环和 expert weight 访问模式。
- 确认 topk padding、indice 比较和 expert combine 没有额外 GM 往返。

selected-expert 会引入新的权重接口和 kernel 形态，只作为 profile 证明 MoE 是绝对瓶颈后的
备选方案。

### 2. Attention 路径

重点看：

- `attention_qkv`
- `sparse_attn_swa/csa/hca`
- `attention_out`
- `compressor_ratio4/128`
- `indexer`

候选方向：

- 为 decode `S=1` 写专门路径，减少动态 S、padding 和无效 tile。
- 调整大 linear 的 tile 和切分轴，尤其是 `1024 -> 32768`、`4096 -> 1024`、
  `8192 -> 4096` 等路径。
- sparse attention 评估 online softmax 或更少中间写回的实现。
- 减少 RoPE、q head rms scale、KV cache 更新之间的重复读写。

### 3. HC 路径

`hc.py` 已经记录了 PyPTO 小 shape 和非 16 对齐 tail 的约束。后续优化 HC 时必须保留这些
经验：

- 序列维使用 `S_PAD = ceil_div(S, 16) * 16` 避免动态 tail valid shape 问题。
- `MIX_HC=24` pad 到 `MIX_PAD=32`。
- `HC_MULT=4` pad 到 `HC_PAD=8`，避免 fp32 行宽 16 bytes 的对齐问题。
- `comb_logits` 作为 GM scratch 是当前稳定实现的一部分。

HC 优化优先看是否能减少 scratch GM 写回和重复 padding 拷贝，而不是先修改数学逻辑。

### 4. 小算子

`rmsnorm`、`rope`、`embedding`、`head` 等小算子只有在 profile 显示占比明显时再优化。
这些 kernel 对正确性敏感，但通常不是端到端瓶颈。

## 实施流程

每个优化项按以下流程执行：

1. 记录 baseline。
   - 固定命令、固定 prompt、固定 token 数。
   - 记录 `layer.kernel run_ms`、`layer.total` 和端到端输出 TPS。
2. 定位热点。
   - 先按 block 形态排序。
   - 再对最热 block 临时拆分子路径。
3. 设计最小修改。
   - 优先只改一个 kernel 或一个子路径。
   - 保持 golden 与官方 `model.py` 语义一致。
4. 验证正确性。
   - 本地 `pytest -q tests`。
   - Ascend 上运行对应独立 kernel 或 block。
   - 至少跑一个完整 `generate.py` 短生成。
5. 验证性能。
   - 同一基准跑 3 次取中位数。
   - 只把 cache hit 后的 runtime 作为主要指标。
6. 记录结果。
   - 如果收益明确，保留修改并记录 profile 数据。
   - 如果收益不稳定或只改善独立 kernel，不改善端到端，回退该优化。

## 判定标准

单个优化应该满足：

- 逻辑与官方 bf16 路径一致。
- 现有测试通过。
- 完整生成仍输出逻辑正确句子。
- 对目标场景的端到端 runtime 有稳定收益。

收益判断优先级：

```text
完整 generate 总耗时
decode 每 token 平均耗时
layer.kernel run_ms
独立 kernel benchmark
```

独立 kernel benchmark 只能作为参考，不能替代完整路径验证。

## 风险点

- PyPTO 对动态 shape、窄 tile、非 16 对齐 tail 和小宽度 fp32 tile 比较敏感。
- 某些优化可能降低单 kernel 时间，但增加 block 内 scratch、materialize 或 state update 成本。
- 调整 matmul 切分可能改变 bf16 累加误差，需要重新验证长序列生成。
- selected-expert 虽然可能降低 decode 计算量，但会改变权重加载和 MoE kernel 接口，实施成本较高。

## 当前下一步

下一步先实现 block 内部的临时 timing/拆分诊断，不直接修改计算逻辑。

目标是回答两个问题：

1. 四种 block 形态中哪一种最耗时。
2. 最热 block 中 MoE、attention、HC、sparse attention 哪个子路径占比最高。

拿到这个排序后，再开始对最高占比的 kernel 做自底向上的优化。
