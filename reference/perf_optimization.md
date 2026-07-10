# Performance Optimization Plan

本文只基于当前 `models/`、`serving/` 和测试代码进行静态分析，不依赖仓库内其他设计文档的结论。
流量数字根据当前 `TensorSpec`、tensor shape 和 dtype 直接估算；具体收益仍需在 Ascend NPU
上通过同步后的 runtime profile 验证。

## 核心判断

当前最值得优先处理的不是继续微调小算子 tile，而是以下三类结构性问题：

1. 权重和状态保存在 host，direct backend 每次 kernel 调用都接收 host tensor。
2. decode state 使用 input/output 分离接口，为更新一个 slot 复制和返回完整 cache。
3. selected-expert decode 虽然只加载 6 个 routed experts，但 kernel 仍串行执行专家，且
   `S=1` 继续使用以 `T_TILE=16` 为主的通用计算路径。

按照当前 tensor spec 静态求和，单个 decode token 的 kernel 输入签名规模大致为：

| 路径 | 输入规模 |
|---|---:|
| Embedding | 1010 MiB |
| LM Head | 2022 MiB |
| 43 层 decode pre-MoE | 约 9.8 GiB |
| 43 层 decode post-MoE | 约 14.1 GiB |
| 合计 | 约 26.9 GiB/token |

这些数字不代表 PyPTO runtime 必然逐字节重复传输全部输入，但当前代码没有 device-resident
tensor 管理，因此它们准确反映了现有调用边界暴露的数据规模。

## 优先级总览

| 优先级 | 优化方向 | 当前问题 | 预期收益 |
|---|---|---|---|
| P0 | 常用权重 NPU 常驻 | 每个 token 重复走 host tensor 调用 | 极高 |
| P0 | BF16 runtime-layout 缓存 | 普通层权重重复读取、反量化和转置 | 极高 |
| P0 | State 原地更新并常驻 NPU | 每个 token 复制并返回完整 cache | 高 |
| P0 | 重写 selected-expert decode MoE | 6 个专家串行，`S=1` 使用 16 行 tile | 高 |
| P1 | 重写 indexer top-k | 最坏执行 512 次长度 1024 的 argmax | 高 |
| P1 | 优化 CSA sparse attention | 固定处理 640 个槽位并重复计算 QK | 中高 |
| P1 | 专用 `S=1` kernel | HC、RMSNorm、RoPE、Linear 存在 padding 浪费 | 中高 |
| P1 | 优化 LM head | 2 GiB FP32 权重和完整 vocab logits | 中高 |
| P2 | Prefill expert-major dispatch | 完整加载 256 专家并逐 route 执行 GEMV | 高，主要降低 TTFT |
| P2 | Continuous batching | 固定 `B=1`，Cube M 维利用率低 | 吞吐收益极高，改造较大 |

## P0：权重缓存和 NPU 常驻

### 当前数据路径

`DeepSeekV4Runner` 使用：

- `DeepSeekV4WeightLoader(default_device="cpu")`
- `DeepSeekV4State(device="cpu")`
- `_DirectBackend`

direct backend 将 runner 组装的 host tensor 直接作为 compiled kernel 参数。当前 weight loader 已
区分临时 `_base_cache` 和最终 `_layout_cache`：`get_linear_t()`、HC transpose、head padding 以及
identity runtime tensor 会复用最终 layout。每层结束时 runner 只释放 base cache，layout 跨 token
保留。

代码复核确认，同一个规范化 checkpoint tensor 在生产路径中只对应一种 runtime layout。当前
`_base_cache` 只在首次 transpose 时保存原布局中间 tensor，随后本层结束即释放，没有实际 hit。
远程单层 profile 中该临时 base cache 为269,484,032 bytes。

### 优化方案

#### 1. 增加 runtime-layout host cache

删除 `_base_cache`，只保留最终提供给 kernel 的固定 `_layout_cache`：

- 已反量化为 BF16。
- 已转置为 `*_t` 布局。
- 已完成必要 padding。
- contiguous。
- 优先使用 pinned memory，方便异步 H2D。

cache key 至少包含：

```text
parameter name
target dtype
runtime layout/version
padding profile
```

普通 layout cache 不使用 LRU，不设置 `max_layout_cache_bytes`。它惰性缓存 runner 实际访问到的
全部非 routed 不可变权重，包含 shared experts；仅 routed experts、selected expert pack 和 full
routed pack 被排除，并继续使用独立 expert cache/LRU。

根据实际 checkpoint index shape 和最终 runtime dtype，完整43层固定 host layout 工作集为：

```text
包含 shared experts 和 head：15,753,487,500 bytes（14.672 GiB）
包含 shared experts、--no-head：13,634,307,144 bytes（12.698 GiB）
```

当前远程服务器可用 host 内存约1,871.7 GiB，单 runner 完整工作集约占0.784%，因此普通权重
无需容量淘汰。多个 runner 会线性复制该工作集；若未来实例数显著增加，应优先共享不可变
layout，而不是重新增加普通权重 LRU。

#### 2. 离线转换所有非 routed 权重

当前只有 routed experts 有离线 BF16 cache。attention、compressor、indexer、gate、HC、shared
expert、embedding 和 head 仍可能在运行时执行 dtype 转换或 transpose。

建议增加离线转换工具，将这些权重直接保存为 kernel 需要的最终布局。这样运行时只做顺序读取
和必要的 H2D，不再做反量化和转置。

#### 3. 实现 device-resident backend

优先常驻以下权重：

- embedding
- LM head
- 43 层 attention/common weights
- compressor/indexer weights
- HC weights
- gate weights
- shared experts

根据当前 tensor shape 粗略估算，这部分总量约十几 GiB，适合优先放入 64GB NPU。真正无法
完整常驻的是每层 256 个 routed experts。

device backend 应支持：

- 按 name 注册不可变 device tensor。
- kernel 调用复用 device handle，而不是重复传 host tensor。
- state 和 scratch buffer 池化。
- pre-MoE 与 post-MoE 之间保留 device 中间 tensor。
- runner close 时统一释放资源。

#### 4. Routed expert device LRU

一个 routed expert 的三组 BF16 权重约为：

```text
3 × 4096 × 2048 × 2 bytes ≈ 48 MiB
```

可以使用 `(layer_id, expert_id)` 作为 device LRU key。常驻 common weights 后，剩余显存可缓存
数百个 layer-specific experts。需要记录实际 routing 分布和跨 token 命中率，再决定容量和淘汰
策略。

## P0：消除 selected-expert 的双重 host 拷贝

当前离线 expert cache 加载流程是：

```text
safetensors tensor
→ clone().contiguous()
→ 独立 expert tensor
→ copy_ 到 selected_w{1,2,3}_t[slot]
→ 作为 kernel 参数上传
```

每层 6 个专家约 288 MiB 有效权重，但 host 侧至少产生 clone 和 selected pack 两轮大块拷贝。

建议：

1. 为 selected experts 准备固定大小的 pinned staging buffer。
2. 从 safetensors slice 直接 copy 到最终 slot，避免先构造独立 expert tensor。
3. staging buffer 双缓冲，避免每层重新分配 3 个大 tensor。
4. 对 host BF16 expert 增加有容量限制的 LRU。
5. 如果 device expert LRU 命中，直接把 device tensor 绑定到 kernel，不再经过 host pack。
6. 如果 kernel 必须接收连续 `[TOPK, ...]` pack，可在 device 上 gather/copy 命中 expert，避免
   host 二次拼包。

权重预取可优先覆盖：

- 当前 pre-MoE kernel 运行时准备 shared expert。
- 当前层 post-MoE 运行时准备下一层 common weights。
- device cache miss 时使用异步 H2D 和双 buffer。

当前层 selected expert IDs 依赖 pre-MoE gate 结果，无法在 pre-MoE 完成前精确预取本层 routed
weights，但其他不可变权重仍可并行准备。

## P0：State 常驻并原地更新

### 当前问题

当前 decode state 使用 input/output 分离参数。例如 sliding-window KV cache 为了更新一行，
先复制完整的 `128 × 512` cache，再覆盖目标 slot。

ratio=128 compressor 每个 token 还会复制：

- 128 行 KV state。
- 128 行 score state。
- 整个 compressed cache。

ratio=4/CSA 路径会输入和输出：

- `attn_comp_cache`，约 1 MiB/layer。
- `idx_kv_cache`，约 0.25 MiB/layer。
- window KV cache。
- 两套 compressor state。

这些 state 由 host `DeepSeekV4State` 持有，kernel 完成后又整体返回 host 并替换旧 tensor。

### 优化方案

1. 所有 layer state 常驻 NPU。
2. kernel 接口改为 in-place slot update。
3. host 只维护 `start_pos`、slot 和 should-compress 等小标量。
4. 对不发生压缩的 token，不返回 unchanged cache。
5. pre-MoE 的 state update 在 device 上完成，不经过 host tensor 替换。
6. `ffn_normed`、`attn_hc_out`、`ffn_hc_post`、`ffn_hc_comb` 等 pre/post 中间值保留在
   device scratch 中。
7. post-MoE 只接收 device 中间 handle 和 selected expert device handle。

理想情况下，每层 pre-MoE 后只有 6 个 expert IDs 需要暴露给 host；device expert cache 命中时，
可以进一步避免这个同步点。

## P0：重写 selected-expert decode MoE

### 当前问题

`_run_selected_experts_decode()` 使用：

```python
for k in pl.range(TOPK):
    for t in pl.range(tokens):
        ...
```

decode 固定 `tokens=1`，因此 6 个专家完全串行。每个专家还创建：

```text
gate_tile_fp32: [16, 2048]
up_tile_fp32:   [16, 2048]
hidden_tile:    [16, 2048]
```

实际只有一行有效。即使 Cube GEMM 的 M 维仍必须使用 16 行基本 tile，SwiGLU、cast、clamp、
sigmoid、scratch 初始化等向量逻辑也不应处理 16 行。

每个 routed expert 输出还先写入：

```text
route_y: [1, 1, 6, 4096]
```

随后 `_combine_route_major()` 再读回 6 行并与 shared expert 相加。

### 推荐 kernel 结构

```text
parallel k=0..5:
    gate/up projection
    SwiGLU
    down projection
    partial[k, 4096]

parallel shared expert:
    shared_partial[4096]

single reduction:
    sum(partial[0:6]) + shared_partial

fused HC post
```

具体候选：

- expert 维使用 `pl.spmd` 或 3D batched matmul。
- routed 和 shared expert 并行执行。
- 将 w1/w3 离线拼成 `[4096, 4096]`，一次 projection 后切分 gate/up。
- routing weight 融合进 down projection 输出。
- partial tensor 保持片上或最小 GM scratch，避免长期存在 `route_y`。
- reduction 与 `hc_post_fwd` 融合，避免先生成完整 `moe_out`。
- 为 decode 固定 `S=1`，不再绑定通用动态 token 维。

单层 MoE 的理论计算量约为：

```text
7 experts × 3 × 4096 × 2048 ≈ 176M MAC/token/layer
```

它是当前 block 中最大的计算部分之一，应在权重/状态搬运优化后优先处理。

## P1：重写 Indexer top-k

`indexer_decode_fwd()` 最多生成 1024 个 score，然后使用重复 argmax 选择 top-512：

```python
for k in pl.range(512):
    best_pos = row_argmax(score_work)
    mask selected position
    write score_work back
```

最坏情况下相当于 512 次扫描 1024 元素，并在每次选择后重新生成 mask、读取和写回整行。这是
明确的算法级热点。

优先评估：

1. PyPTO/Ascend 原生 top-k primitive。
2. 分块 top-k 后多级 merge。
3. bitonic/radix selection。
4. 针对 `K=N/2` 的 partition/threshold 算法。
5. score 和 index 成对保存在局部 buffer，减少 GM 往返。
6. 对 `cache_len < INDEX_TOPK` 的早期 token，只执行实际有效次数。

Gate top-k 也使用重复 argmax，但规模只有 `6 × 256`，优先级远低于 indexer 的
`512 × 1024`。

## P1：优化 CSA sparse attention

### 只处理实际有效 top-k

CSA 使用固定：

```text
128 window + 512 indexed = 640 slots
```

短上下文中大量 index 为 `-1`，但 kernel 仍执行全部 40 个 16-position chunk。SWA/HCA 也
分别固定处理 128/160 个槽位。

建议向 sparse attention 传入实际 valid count，或按长度选择不同 kernel bucket：

```text
16 / 32 / 64 / 128 / 256 / 512 / 640
```

prefill 中每个 token 的可见长度不同，可按 token block 分组，避免短 prompt 仍支付完整静态
top-k 成本。

### 消除 CSA 重复 QK/softmax

当前 CSA 将 512 维 value 输出切成两个 256 block。外层先遍历 output block，内层再遍历 40
个 top-k chunk，因此相同的：

- 512 维 K gather。
- QK matmul。
- online softmax。

会执行两次。

候选改法：

- 一次计算 QK/online softmax，同时维护两个 256 维 value accumulator。
- 如果片上空间允许，直接使用 512 维 output accumulator。
- 或先保存一份概率，再分别完成两个 PV block；需要比较概率落 GM 与重复 QK 的代价。

### 消除 kv_pool

CSA decode 当前构造：

```text
kv_pool: [1, 1152, 512]
```

它只是 window cache 与 compressed cache 的物理拼接，随后 sparse attention 又按 top-k gather。
可以根据 index 范围直接从两个源 cache gather，完全移除 `kv_pool` copy 和 scratch。

HCA 的 window/compressed pool 也可采用同样方式。

## P1：专用 `S=1` decode kernel

当前很多 kernel 是 prefill/decode 共用实现：

| 模块 | 当前 token tile |
|---|---:|
| Linear | 16 |
| Routed/shared expert | 16 |
| HC pre | pad 到 16 |
| RMSNorm | 8 |
| RoPE | 通用 token tile |

### HC

单个 decode token 会构造 16-token padding，随后对 padding 行执行 HC linear、split、Sinkhorn
和 mix。每层有两次 HC pre，因此适合增加固定 `S=1` 版本：

- 只对一行执行向量逻辑。
- Cube linear 可以保留最小硬件 tile，但只保留第一行后续计算。
- 不生成 `x_pad/mixes/pre/comb_logits/x_mixed_pad/post_pad/comb_pad` 等大量外部 scratch。
- `hc_pre + rmsnorm` 评估融合。
- `moe reduction + hc_post` 评估融合。

### RMSNorm

增加单行实现，避免 `T_TILE=8` 的 padding vector 运算和逐行 assemble。可进一步与前后 linear
融合，减少 hidden tensor GM 往返。

### RoPE

当前 RoPE kernel 会在 head 循环内重复构造：

```text
dup_idx
swap_idx
sign
```

这些都是编译期常量。应移到循环外，或直接用偶数/奇数 lane 访问替代 gather index 构造。
decode 固定一行时可以使用专用 vector 实现。

### Linear

单请求 decode 的 GEMM M 维仍可能受 Cube 最小 tile 约束，因此单纯把 `T_TILE` 从 16 改为 1
未必能显著降低 matmul 时间。更有价值的是：

- 减少无效向量后处理。
- 合并共享同一个输入的 projection。
- 减少中间 tensor 和 GM writeback。
- 用 continuous batching 填满 M 维。

CSA attention 中多个 projection 共享相同 hidden 输入，可以评估将 wq_a、wkv、compressor
wkv/wgate、indexer weights projection 等拼接成更大的 projection，再按语义切分结果。需要注意
不同输出要求 BF16 或 FP32，融合后应保持对应累加和 cast 语义。

## P1：优化 LM head 和 embedding

### 权重常驻

当前 tensor spec 中：

```text
embedding: [129280, 4096] BF16 ≈ 1010 MiB
head:      [129280, 4096] FP32 ≈ 2020 MiB
```

这两个权重每个 token 都会参与独立 kernel 调用，应优先常驻 NPU。

### Head BF16 权重实验

当前 head 明确使用 FP32 权重和 FP32 matmul。可以增加可选路径：

```text
BF16 weight + BF16 input + FP32 accumulate
```

它可以将 head 权重容量和搬运量减半，并更适合 Cube。该修改可能改变 argmax 边界附近的数值，
必须使用真实生成和 logits/top-token 一致率验证，不能直接替换默认路径。

### Greedy argmax 融合

greedy decode 不需要把 129280 个 logits 返回 host。可以：

1. 每个 vocab block 计算局部最大值和 index。
2. kernel 内做全局归约。
3. 只返回 token ID，必要时附带最大 logit。

sampling 模式可以返回 top-k logits/indices，而不是完整 vocab tensor。

### Prefill head 只计算最后一个 token

当前 LM head 最终只使用最后一个 hidden，但 HC head 和 final RMSNorm 会先处理整个 prompt。
这些计算在 token 维彼此独立，因此可以先 slice 最后一个 token，再执行：

```text
HC head → RMSNorm → vocab projection
```

该优化数学上等价，能直接降低长 prompt 的 head 开销。

## P2：Prefill expert-major dispatch

当前 prefill 为每层构造完整 256-expert pack。仅 SWA block 的 `S=1` 输入规格就约 12.5 GiB，
其中绝大多数是 routed expert 权重。

kernel 内部仍按：

```text
top-k slot
→ token
→ 单个 expert GEMV
```

逐个执行，没有把命中同一 expert 的 token 聚合成 batch。

建议重构为：

```text
gate
→ 按 expert 对 token/route 分桶
→ 计算 active expert set
→ 只加载 active expert weights
→ 每个 active expert 执行 batched GEMM
→ scatter/reduce 到 token 输出
```

收益来源：

- 短 prompt 不再加载全部 256 experts。
- 同一 expert 权重只加载一次。
- 多个 token 组成较大的 GEMM M 维。
- 减少 route-major 中间输出和重复 weight read。

对长 prompt，即使 active expert 覆盖接近 256，expert-major batching 仍能改善计算利用率。

## P2：Continuous batching

当前所有 kernel 固定 `B=1`，单请求 decode 的有效 M 通常为 1，而主要 Linear/MoE kernel 使用
16 行 tile。若目标是服务吞吐而非单请求最低延迟，continuous batching 是最重要的架构级优化。

需要改造：

- 支持多个 request 的独立 `start_pos` 和 cache slot。
- batch-aware KV/compressor/indexer state。
- 将不同请求的 decode token 拼成 M≈16 或更大。
- 对 selected experts 做跨请求 expert dispatch。
- 同一个 expert 命中的多个请求共享权重并执行 batched GEMM。
- batch 内请求完成后动态补位。

该方案同时提高 Cube 利用率和 expert cache 命中价值，但涉及 kernel shape、state 管理和 serving
调度的系统性修改，不适合作为第一步。

## 编译和 scratch 管理

四种 selected decode pre-MoE 的 tensor shape/dtype 不随 `start_pos` 变化，因此 decode kernel
可以稳定复用编译结果。编译缓存不是 steady-state decode 的首要问题。

仍可进行以下低风险优化：

- `_run_config()` 不必每次 kernel 调用重新构造。
- `_materialize_specs()` 当前会为缺失的 scratch/output tensor 反复创建 host tensor；应建立
  按 `(kernel, shape, dtype)` 复用的 scratch pool。
- prefill 的编译 key 包含精确序列 shape，可使用长度 bucket 控制编译数量，但要比较 padding
  计算开销。
- 首次编译耗时必须与 steady-state runtime 分开统计。

## Profile 要求

当前 runner profile 能区分 pre/post values、materialize、compile/run 和 state update，但 block
内部大量子模块是 inline 的，无法直接判断 kernel 内热点。

每个优化阶段建议至少记录：

### Host/runtime

- checkpoint raw read bytes/time
- dequant bytes/time
- transpose/layout conversion bytes/time
- pinned buffer copy bytes/time
- H2D/D2H bytes/time
- device cache hit/miss/eviction
- selected expert IDs 和跨 token 命中率
- scratch allocation count/time

### Kernel

- HC pre/post
- attention QKV
- sparse attention gather/QK/PV
- attention output projection
- compressor/indexer
- indexer top-k
- gate
- routed expert gate/up/down
- shared expert
- route reduction
- head HC/RMSNorm/vocab projection

如果编译器/runtime 没有可靠的 inline region profile，可临时拆成独立 kernel 做 A/B 测量，但
最终验收必须回到完整 block 和端到端生成。

## 推荐实施顺序

### 阶段 1：消除重复 host 计算和分配

1. 缓存转置后的 BF16 runtime-layout 非 routed 权重。
2. 为 selected expert pack 增加 pinned 双缓冲。
3. 消除 expert cache 的 clone + selected pack 双重 copy。
4. 复用 `RunConfig`、scratch 和 output buffer。
5. 增加 bytes、cache hit 和 H2D/D2H profile。

### 阶段 2：建立 device-resident runtime

1. embedding/head 常驻 NPU。
2. 所有 common layer weights 和 shared experts 常驻 NPU。
3. state 常驻 NPU 并原地 slot update。
4. pre/post 中间值保持 device resident。
5. routed expert device LRU。

### 阶段 3：优化 decode kernel

1. selected experts 并行执行。
2. 融合 routed/shared reduction 与 HC post。
3. 增加 `S=1` HC/RMSNorm/RoPE 路径。
4. 重写 indexer top-k。
5. 消除 CSA 重复 QK 和 kv_pool。
6. 根据实际 valid count 缩短 sparse attention 循环。

### 阶段 4：优化 TTFT 和吞吐

1. Prefill expert-major dispatch。
2. Prefill head 只处理最后一个 token。
3. Prefill length bucket 和 compile cache 管理。
4. Continuous batching。
5. 在需要时评估 speculative/MTP 路径。

## 验证标准

每个优化都应同时满足：

1. 本地 golden/unit tests 通过。
2. Ascend 独立 kernel 或 block 数值验证通过。
3. 完整 43 层 prefill + decode 通过。
4. 固定 prompt 的生成文本和 token 序列符合预期。
5. 分开记录首次编译和 cache-hit runtime。
6. 至少运行 3 次并比较中位数。
7. 同时报告：
   - TTFT
   - 单 token decode latency
   - output tokens/s
   - H2D/D2H bytes
   - 峰值 host memory
   - 峰值 device memory
   - expert cache hit rate

优化优先级应由端到端数据调整，但在没有进一步 NPU profile 之前，最合理的起点仍然是：

```text
runtime-layout cache
→ device-resident common weights/state
→ selected-expert decode kernel
→ indexer top-k / CSA sparse attention
→ 其他小算子 tile
```
