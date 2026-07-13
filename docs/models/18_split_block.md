# Split Decode Block

## 模块定位

Split Decode Block 是完整 Transformer Block 的 decode 专用执行形态。它把单 token
decode 拆成 pre-MoE 和 post-MoE 两个 NPU kernel，中间只把 Gate 生成的 6 个 expert
indices 读取到 host：

```text
hidden [1,1,4,4096], BF16
  -> pre-MoE kernel on NPU
       Attention HC pre -> RMSNorm -> decode Attention -> HC post
       FFN HC pre -> RMSNorm -> Gate
       -> attention state outputs
       -> ffn_normed / weights / FFN HC context stay on NPU
       -> indices [1,1,6], INT32 copied to host
  -> load only 6 selected routed-expert weights
  -> post-MoE kernel on NPU
       selected routed experts + shared expert -> FFN HC post
       -> out [1,1,4,4096], BF16
```

拆分的直接目的，是在 decode 时先得到 Gate 选择结果，再只加载当前 token 实际使用的
6 个 routed experts，避免为每层、每个 decode step 加载全部 256 个 routed-expert
weights。这是当前 decode 性能设计的核心边界，不是模型数学结构的变化。

每个 Expert 的 `w1/w2/w3` 均包含 `4096×2048` 个 BF16 元素，总计 48 MiB。因此当前
kernel-facing routed weight payload 为：

| Decode 权重形态 | Expert 数量 | 每层 BF16 payload |
|---|---:|---:|
| Full packed routed weights | 256 | 12 GiB |
| Selected routed weights | 6 | 288 MiB |

在当前 routed weights 不作为 fixed weights 常驻 device 的前提下，拆分路径将每层
decode 的 routed-expert host load/H2D staging payload 缩小到 `6/256`，约为原来的
`1/42.7`。代价是增加一次 pre/post kernel 边界、一次 24-byte indices D2H control
读取，以及 selected weight gathering。

## 官方模型中的 Block decode

[`official/model.py`](../../official/model.py) 没有 Split Block 类。官方 `Block.forward`
在一次 Python 调用中按以下顺序执行完整 Block：

```text
Attention HC pre
  -> Attention RMSNorm
  -> Attention
  -> Attention HC post
FFN HC pre
  -> FFN RMSNorm
  -> MoE Gate
  -> selected routed experts + shared expert
  -> FFN HC post
```

官方 `MoE.forward()` 先运行 Gate，再按 `indices` 只执行被选中的 routed experts；模型
语义本身并不要求计算全部 256 个 experts。官方 Python module 可以直接从常驻的
`ModuleList` 访问对应 Expert，而当前 PyPTO runtime 的 routed weights 存在 host，必须
在 kernel dispatch 前显式确定本次要上传的 tensor。

因此当前实现把官方连续数据流切在 Gate 与 routed Expert 之间。Pre-MoE kernel 对应
官方 Block 从 Attention HC pre 到 Gate output；post-MoE kernel 对应 selected
routed/shared Expert、route aggregation 和 FFN HC post。两段合并后的数学输出与完整
decode Block 对齐。

官方还支持 Expert Parallel 和跨 rank routed output aggregation；当前 Split Block
固定为单卡逻辑，不执行分布式通信。

## PyPTO kernel 实现

[`models/split_block.py`](../../models/split_block.py) 提供四种 pre-MoE kernel 和一个
公共 post-MoE kernel：

| 符号 | 类型 | Attention / routing | 当前层范围 |
|---|---|---|---|
| `swa_hash_selected_decode_pre_moe_fwd` | `@pl.jit` | SWA + Hash Gate | layer 0–1 |
| `csa_hash_selected_decode_pre_moe_fwd` | `@pl.jit` | CSA + Hash Gate | layer 2 |
| `hca_topk_selected_decode_pre_moe_fwd` | `@pl.jit` | HCA + Top-K Gate | odd layers 3–41 |
| `csa_topk_selected_decode_pre_moe_fwd` | `@pl.jit` | CSA + Top-K Gate | even layers 4–42 |
| `selected_decode_post_moe_fwd` | `@pl.jit` | Selected MoE + FFN HC post | 四种 pre 路径共用 |

对应的 PyTorch golden：

| 符号 | 职责 |
|---|---|
| `_golden_selected_decode_pre_moe` | 四种 pre 路径共用 Block-level 参考流程 |
| `golden_*_selected_decode_pre_moe` | 固定 Attention 和 routing 组合的 pre wrapper |
| `golden_selected_decode_post_moe` | 公共 selected MoE + FFN HC post golden |
| `golden_*_selected_decode_post_moe` | 公共 post golden 的命名 wrapper |

`build_*_selected_decode_pre_moe_specs` 从
[`models/block.py`](../../models/block.py) 对应 full decode spec 中选择 pre-MoE 所需
tensors，并把跨边界 intermediates 声明为 outputs。`build_selected_decode_post_moe_specs`
构造公共 post-MoE 接口。所有 builders 都固定 `S=1`。

当前 [`serving/runner.py`](../../serving/runner.py) 的 decode 主干始终使用 Split Block。
`models/block.py` 中的 full decode kernels 仍然存在并可独立验收，但当前 Runner 不调用
它们；prefill 继续使用 full Block kernel。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| 完整 `Block.forward` decode | pre-MoE + host selection + post-MoE | 语义等价；增加 runtime split boundary |
| Attention HC pre/norm/Attention/HC post | 四种 pre-MoE kernels | 直接调用现有 inline kernels |
| FFN HC pre + FFN RMSNorm | 四种 pre-MoE kernels | 直接调用 `hc_pre_fwd` / `rmsnorm_4096` |
| `self.ffn.gate(...)` | `gate_hash_fwd` / `gate_topk_fwd` | Pre-MoE kernel 直接调用 |
| Gate `(weights, indices)` | Pre outputs `indices/weights` | Tensor 语义一致；当前 output 顺序不同于官方 tuple |
| 按 indices 选择 Expert modules | Runner + weight loader selected slice copy | Runtime 边界；只 materialize 6 组 weights |
| Routed/shared Expert + aggregation | `moe_selected_decode_experts_fwd` | Post-MoE kernel 直接调用 |
| FFN `hc_post` | `hc_post_fwd` | Post-MoE kernel 直接调用 |
| Attention state mutation | Pre-MoE state outputs + `commit_state` | 语义等价；state 由 device runtime 持有 |
| Full decode kernel | `models/block.py::*_decode_fwd` | 存在但当前 serving decode 未使用 |
| Expert Parallel / `all_reduce` | 无 | 不支持或未执行；当前单卡逻辑 |

## 数据接口

### Dispatch matrix

当前 dispatch 由 [`models/config.py`](../../models/config.py) 的 compression ratio 和
`layer_id < n_hash_layers` 共同决定：

| Layers | Compression ratio | Attention | Routing | Pre-MoE kernel |
|---|---:|---|---|---|
| 0–1 | 0 | Sliding Window Attention（SWA） | Hash | `swa_hash_selected_decode_pre_moe_fwd` |
| 2 | 4 | Compressed Sparse Attention（CSA） | Hash | `csa_hash_selected_decode_pre_moe_fwd` |
| Odd 3–41 | 128 | Heavily Compressed Attention（HCA） | Top-K | `hca_topk_selected_decode_pre_moe_fwd` |
| Even 4–42 | 4 | Compressed Sparse Attention（CSA） | Top-K | `csa_topk_selected_decode_pre_moe_fwd` |

不存在当前配置可达的 SWA+Top-K 或 HCA+Hash 路径。四种 pre paths 都接收
`x [1,1,4,4096]` BF16、对应 Attention/HC/normalization/Gate fixed weights、当前层
attention state、position-dependent auxiliary inputs 和 HC scratch。

### Pre-MoE 公共边界 outputs

除 Attention 类型专用 state outputs 外，四种 pre kernel 都输出：

```text
attn_hc_out: [1,1,4,4096], BF16
ffn_normed:  [1,1,4096],   BF16
indices:     [1,1,6],      INT32
weights:     [1,1,6],      FP32
ffn_hc_post: [1,1,8],      FP32
ffn_hc_comb: [1,1,16],     FP32
```

这些 outputs 的用途不同：

- `indices` 是唯一必须读取到 host 的 control tensor，用于选择 6 组 expert weights；
- `weights` 保持在 device，在 post-MoE 内乘到对应 routed Expert；
- `ffn_normed` 是 selected MoE input；
- `attn_hc_out` 是 FFN HC post 的 residual；
- `ffn_hc_post/comb` 是 pre-MoE 已生成、等待 post-MoE 消费的 HC context。

因为 `S=1`、Top-K=6，indices 的逻辑 D2H payload 只有 6 个 INT32，即 24 bytes。其余
跨 kernel intermediates 不回传 host。

### Attention state outputs

Pre-MoE kernel 完成 Attention 计算并生成当前 decode position 的新 state：

| Attention | State outputs | Shape / dtype |
|---|---|---|
| SWA | `kv_cache_out` | `[1,128,512]`, BF16 |
| HCA | `kv_cache_out` | `[1,128,512]`, BF16 |
| HCA | `comp_kv_state_out`, `comp_score_state_out` | `[1,128,512]`, FP32 |
| HCA | `comp_cache_out` | `[1,32,512]`, BF16 |
| CSA | `kv_cache_out` | `[1,128,512]`, BF16 |
| CSA | `attn_comp_kv_state_out`, `attn_comp_score_state_out` | `[1,8,1024]`, FP32 |
| CSA | `attn_comp_cache_out` | `[1,1024,512]`, BF16 |
| CSA | `idx_kv_cache_out` | `[1,1024,128]`, BF16 |
| CSA | `idx_comp_kv_state_out`, `idx_comp_score_state_out` | `[1,8,256]`, FP32 |

State 的位置、compression slot、RoPE 和 sparse indices inputs 由
[`serving/state.py`](../../serving/state.py) 按 `start_pos` 构造。具体 Attention state
语义见 [`11_attention_swa.md`](11_attention_swa.md)、
[`12_attention_hca.md`](12_attention_hca.md) 和
[`13_attention_csa.md`](13_attention_csa.md)。

### Post-MoE 接口

公共 post kernel 接收 pre outputs 和当前 route slots 对应的 Expert weights：

```text
ffn_normed:    [1,1,4096], BF16
weights:       [1,1,6],    FP32
selected_w1_t: [6,4096,2048], BF16
selected_w2_t: [6,2048,4096], BF16
selected_w3_t: [6,4096,2048], BF16
shared_w1_t:   [4096,2048], BF16
shared_w2_t:   [2048,4096], BF16
shared_w3_t:   [4096,2048], BF16
attn_hc_out:   [1,1,4,4096], BF16
ffn_hc_post:   [1,1,8],      FP32
ffn_hc_comb:   [1,1,16],     FP32
out:           [1,1,4,4096], BF16
```

Selected weights 的第一维按 Gate route slot 排列，而不是按全局 expert id 排序；第
`k` 组 weights 必须对应 `weights[:,:,k]`。Shared Expert weights 是 fixed resident
weights，不参与 selected gathering。

### State 与 runtime ownership

Pre-MoE kernel 执行后，Runner 先调用 device runtime 的 `commit_state()` 提交所有
Attention state outputs，再读取 indices。State 即使在 host expert selection 期间也
保持 device resident，并供下一 decode step 复用。

`runtime.read_control(indices)` 只复制 indices 并释放该 control intermediate。
`ffn_normed`、routing weights、`attn_hc_out` 和 FFN HC context 保持为 device tensors，
直接绑定到 post kernel。Selected expert weights 以 `DECODE_SELECTED` staging 上传；
post kernel 完成后 staging allocation 归还 buffer pool，可被后续层或 decode step
复用。

Split Block 本身没有新增跨 step 模型 state。它改变的是同一层 decode 内的 kernel
边界和 expert weight 生命周期。

## 实现方式

### Pre-MoE kernel

四种 pre kernels 复用相同 Block 前半流程：

1. 保存输入 4-stream hidden 作为 Attention residual；
2. 执行 Attention `hc_pre_fwd` 和 `rmsnorm_4096`；
3. 按 dispatch 类型执行 SWA、CSA 或 HCA decode，并写出对应 state outputs；
4. 执行 Attention `hc_post_fwd`，生成 `attn_hc_out`；
5. 执行 FFN `hc_pre_fwd` 和 `rmsnorm_4096`，生成 `ffn_normed` 与 FFN HC context；
6. 执行 Hash 或 Top-K Gate，输出 indices 和 weights；
7. 在 Gate 之后结束 kernel，不执行 routed/shared Experts。

所有 pre builders 固定 `S=1`，但 HC pre 仍沿用 16-token padded scratch。Attention
state 在 pre kernel 内完成更新，避免把 Attention 计算或 state mutation 延后到 host
selection 之后。

### Host selected-expert bridge

[`serving/runner.py`](../../serving/runner.py) 在两次 dispatch 之间执行：

1. 从 device 读取 `[1,1,6]` indices；
2. 调用 [`serving/weight_loader.py`](../../serving/weight_loader.py) 的
   `get_layer_moe_selected_experts()`；
3. 优先通过 [`serving/expert_cache.py`](../../serving/expert_cache.py) 对 packed BF16
   cache 做 6 个 lazy slices；若该层没有 packed cache，则从 checkpoint 逐 expert
   构造相同 transposed BF16 layout；
4. 保持 route-slot 顺序，把三组 `[6,...]` tensors 作为 selected staging 传给 post。

该 bridge 不读取 FP32 routing weights 或 hidden intermediates。它引入同步点，但避免
读取、构造和上传 256-expert full pack；在当前权重尺寸下，减少的 routed weight
payload 远大于 24-byte control 回传。

### Post-MoE kernel

`selected_decode_post_moe_fwd` 调用
[`models/moe.py`](../../models/moe.py) 的 `moe_selected_decode_experts_fwd`，按 route
slot 执行 6 个 routed Experts、1 个 shared Expert 和 route aggregation，生成
`moe_out [1,1,4096]` BF16。

随后 kernel 调用 [`models/hc.py`](../../models/hc.py) 的 `hc_post_fwd`，使用 pre
阶段保存的 `attn_hc_out`、`ffn_hc_post` 和 `ffn_hc_comb` 完成 FFN residual mixing，
输出下一层需要的 4-stream hidden。Post kernel 与 Attention 类型、Hash/Top-K routing
类型无关，因此四条 pre 路径共用同一实现。

### Prefill 与 decode 的边界差异

Prefill 在 Gate 之前无法预先知道每个 prompt token 的 selected experts，且一个
sequence 内可能覆盖大量 expert ids。当前 prefill 因此继续调用
[`models/block.py`](../../models/block.py) 的 full Block kernel，并为每层提供完整
256-expert packed weights。

Decode 每次只有一个 token，Gate output 只有 6 个 route slots。当前 Runner 固定使用
Split Block，把 routed weights 的选择推迟到 Gate 执行之后，从而将 decode 的
expert-weight host load/H2D 范围限制为 6 个 experts。

## 实现差异与限制

- Split Block 是当前 runtime 的 decode-only 执行边界，固定 `B=1`、`S=1` 和
  `start_pos>0`；prefill 不使用该路径；
- 拆分不会改变官方 Block 数学语义，目标是避免 decode 为每层加载全部 256 个 routed
  experts，降低 host load、H2D payload 和 device staging 需求；
- 每层增加一次 kernel dispatch boundary 和一次 indices D2H synchronization；
- 只有 indices 回到 host；routing weights、hidden intermediates、HC context 和
  Attention state 保持在 device；
- 当前支持四种配置可达组合：SWA+Hash、CSA+Hash、HCA+Top-K、CSA+Top-K；
- Post-MoE kernel 不接收 indices，要求 selected weights 已严格按 6 个 route slots
  排列；
- Full decode kernels 仍存在于 `models/block.py`，但当前 Runner 不调用；
- Attention state 在 pre-MoE 完成并提交，post-MoE 不修改 Attention state；
- Split Block 不新增跨 step state，selected weights 是当前层、当前 decode step 的
  transient staging；
- 当前不实现多卡 Expert Parallel 或 routed output `all_reduce`；
- 性能收益取决于当前 routed weights 的非 resident 策略；12 GiB 与 288 MiB 是由
  kernel-facing BF16 shapes 推导的理论 payload，不等同于端到端 latency 比例。

## Golden 参考实现

`models/split_block.py::_golden_selected_decode_pre_moe` 要求 `start_pos>0` 且 `S=1`。
它复用 [`models/block.py`](../../models/block.py) 的 host helpers，依次执行 Attention HC
pre、RMSNorm、对应 decode Attention、Attention HC post、FFN HC pre 和 FFN RMSNorm，
最后调用 `golden_gate_forward` 写出 indices/weights。它同时写出 Attention state、
`attn_hc_out`、`ffn_normed` 和 FFN HC context。

四个 public pre golden 只固定 `attention_kind` 和 `hash_route`。Post wrappers 最终都
调用 `golden_selected_decode_post_moe`；该函数先调用
`golden_moe_selected_decode_experts_forward`，再用 Block HC post helper 生成最终
4-stream output。

Golden 使用 kernel-facing selected weights，不读取真实 expert cache，也不模拟
indices D2H、H2D staging 或 device state commit。Host integration test 负责从 full
packed weights 按 Gate indices 构造 selected tensors，并与 full decode golden 比较。

## 精度验收标准

不同 outputs 按其数值路径分为三组：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `indices` | `1e-5` | `1e-5` | 0 |
| Attention cache/state outputs | `1e-4` | `1/128` | `0.001` |
| `weights`, `ffn_hc_post`, `ffn_hc_comb` | `1e-4` | `1/128` | `0.001` |
| `attn_hc_out`, `ffn_normed`, final `out` | `1e-3` | `2/128` | `0.005` |

`indices` 使用默认 `torch.allclose`；由于 dtype 为 INT32，`1e-5` 容差等价于 expert id
和 route-slot 顺序逐元素一致。其余 outputs 使用 `ratio_allclose`：state/weight/HC
context 最多允许 0.1% 元素超出容差，主要 BF16 hidden outputs 最多允许 0.5%。Actual
output 中出现任何 NaN 或 Inf 都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上以 `start_pos=3` 验证四种 pre paths 和公共 post path；该位置
同时覆盖 ratio-4 compressor 的 update boundary：

```bash
python models/split_block.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 3 \
  --case all
```

单独验证 ratio-128 HCA compressor boundary：

```bash
python models/split_block.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 127 \
  --case hca-topk-pre
```

`--case pre` 只运行四种 pre cases；`--case post` 按四个配置标签运行公共 post kernel。
也可以用 `swa-hash-pre`、`csa-hash-pre`、`hca-topk-pre`、`csa-topk-pre` 及对应
`*-post` 名称选择单项。增加 `--compile-only` 可仅检查编译；增加
`--enable-l2-swimlane` 会把相应选项传入 PyPTO `RunConfig`。

Host-side split/full decode 等价验证可运行：

```bash
pytest -q tests/models/test_split_block.py
```

## 集成验证范围

### Standalone Split Block 验收

`models/split_block.py::main()` 验证四种 pre-MoE kernel 和公共 post-MoE kernel。
Standalone pre cases 从已有 Attention state snapshot 开始；它们不执行真实 host
selected-expert bridge。Post case 接收 spec builder 直接生成的 6 组 selected weights。

[`test_split_block.py`](../../tests/models/test_split_block.py) 使用缩小模型分别覆盖
SWA+Hash、CSA+Hash、HCA+Top-K 和 CSA+Top-K。测试先运行 full packed decode golden，
再运行 pre golden、按 indices 从 full pack 提取 selected weights、运行 post golden，
并逐元素比较 Gate outputs、FFN intermediates、Attention state 和最终 Block output。
该 host test 不执行 PyPTO NPU kernel，不能替代 standalone 实机验收。

### Serving orchestration 与 lifecycle

- [`test_runner.py`](../../tests/serving/test_runner.py) 覆盖 Runner 只通过 runtime 读取
  indices，并把其余 pre outputs 原样绑定到 post kernel；
- [`test_device_runtime.py`](../../tests/serving/test_device_runtime.py) 覆盖 control D2H、
  pre/post intermediates device-resident 和 selected staging 跨 decode step 的 allocation
  复用；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 selected expert
  数量、id 范围、route-slot 顺序和 BF16 transposed layouts；
- [`test_expert_cache.py`](../../tests/serving/test_expert_cache.py) 覆盖 packed cache 的
  lazy selected slices、重复 expert id 和输出 tensor 校验。

这些 serving tests 验证拆分边界和权重生命周期，不替代 Split Block 数学 kernel 的
精度验收。
