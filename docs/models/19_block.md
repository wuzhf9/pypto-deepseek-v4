# Transformer Block

## 模块定位

Transformer Block 是 DeepSeek V4 Flash 主模型的重复层。每个 Block 接收并输出 4 份
Hyper-Connection（HC）residual streams，在内部依次执行一个 Attention 子层和一个
Mixture-of-Experts（MoE）FFN 子层：

```text
x [1,S,4,4096], BF16
  -> save attention residual
  -> Attention HC pre
  -> Attention RMSNorm
  -> Attention: SWA / CSA / HCA
  -> Attention HC post
       -> attn_hc_out [1,S,4,4096], BF16
  -> save FFN residual
  -> FFN HC pre
  -> FFN RMSNorm
  -> MoE: Gate + routed experts + shared expert
  -> FFN HC post
  -> out [1,S,4,4096], BF16
```

Attention 和 FFN 各自拥有独立 HC 参数与 residual；第二个 residual 是 Attention HC
post 的结果，不是原始 Block input。Attention state 在 Attention 子层内更新，MoE 不
修改 Attention state。

[`models/block.py`](../../models/block.py) 实现完整 Block kernel，包括 prefill 和 full
decode 两种形态。当前 serving 的 prefill 每层直接运行一个完整 Block kernel；decode
为了只加载 6 个 selected expert weights，改用
[`18_split_block.md`](18_split_block.md) 描述的 pre/post-MoE Split Block。

## 官方模型中的 Block

[`official/model.py`](../../official/model.py) 的 `Block` 包含：

| 官方字段 | 类型/数量 | 职责 |
|---|---|---|
| `attn` | `Attention` | 按 layer compression ratio 执行 Attention |
| `ffn` | `MoE` | Gate、256 routed experts 和 1 shared expert |
| `attn_norm` | `RMSNorm(4096)` | Attention HC pre output normalization |
| `ffn_norm` | `RMSNorm(4096)` | FFN HC pre output normalization |
| `hc_attn_*` | `fn/base/scale` | Attention 子层 HC mixing 参数 |
| `hc_ffn_*` | `fn/base/scale` | FFN 子层独立 HC mixing 参数 |

官方 `Block.forward(x,start_pos,input_ids)` 的严格执行顺序为：

1. 保存 `x` 为 Attention residual；
2. `hc_pre(x,hc_attn_*)` 归约 4 streams，并生成 Attention `post/comb`；
3. `attn_norm` 后调用 `attn(x,start_pos)`；
4. `hc_post(attn_out,attention_residual,post,comb)` 恢复 4 streams；
5. 保存该结果为 FFN residual；
6. `hc_pre(x,hc_ffn_*)` 后执行 `ffn_norm`；
7. `ffn(x,input_ids)` 执行 MoE；
8. `hc_post(moe_out,ffn_residual,post,comb)` 生成 Block output。

官方用 `start_pos=0` 表示 prompt/prefill，用正数 position 执行单 token decode；同一个
`Block.forward` 不区分不同顶层函数。Attention 类型由 layer compression ratio
决定，MoE routing 类型由 layer id 是否位于 hash layers 决定。

官方 `MTPBlock` 继承 `Block` 并增加 MTP-specific projection 和 head。当前 Runner 只
执行 43 个主模型 Blocks，不执行官方 MTP layer。

## PyPTO kernel 实现

[`models/block.py`](../../models/block.py) 把当前配置可达的四种 Block 组合分别实现为
prefill 和 full decode kernel：

| 配置 | Prefill kernel | Full decode kernel | Spec builders |
|---|---|---|---|
| SWA + Hash | `block_swa_hash_prefill_fwd` | `block_swa_hash_decode_fwd` | `build_swa_hash_{prefill,decode}_specs` |
| CSA + Hash | `block_csa_hash_prefill_fwd` | `block_csa_hash_decode_fwd` | `build_csa_hash_{prefill,decode}_specs` |
| HCA + Top-K | `block_hca_topk_prefill_fwd` | `block_hca_topk_decode_fwd` | `build_hca_topk_{prefill,decode}_specs` |
| CSA + Top-K | `block_csa_topk_prefill_fwd` | `block_csa_topk_decode_fwd` | `build_csa_topk_{prefill,decode}_specs` |

8 个 kernel 都是 `@pl.jit` 顶层入口，不是其他 kernel 内复用的 inline wrapper。它们
直接组合以下 inline kernels：

- [`models/hc.py`](../../models/hc.py) 的 `hc_pre_fwd` / `hc_post_fwd`；
- [`models/rmsnorm.py`](../../models/rmsnorm.py) 的 `rmsnorm_4096`；
- 对应的 SWA、CSA 或 HCA Attention prefill/decode kernel；
- [`models/moe.py`](../../models/moe.py) 的 `moe_hash_fwd` 或 `moe_topk_fwd`。

`golden_block_forward` 是 8 个 case 共用的 PyTorch Block golden。8 个
`golden_block_*_{prefill,decode}` wrappers 只固定 `start_pos`、Attention 类型和
Hash/Top-K routing 类型。

当前 [`serving/runner.py`](../../serving/runner.py) 只直接 dispatch 四个 prefill
kernels。Full decode kernels 存在并有 standalone/host 验收，但当前 serving decode
不调用。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `Block.forward` | 8 个 fixed-combination Block kernels | 语义等价；按 Attention/routing/prefill/decode specialization |
| Attention `hc_pre` | `hc_pre_fwd` | 直接调用；额外 padded scratch |
| `attn_norm` | `rmsnorm_4096` | 直接调用 |
| `self.attn(x,start_pos)` | 对应 Attention prefill/decode inline kernel | 直接调用；显式 state tensors |
| Attention `hc_post` | `hc_post_fwd` | 直接调用 |
| FFN `hc_pre` | 第二次 `hc_pre_fwd` | 直接调用；独立参数和 scratch |
| `ffn_norm` | 第二次 `rmsnorm_4096` | 直接调用 |
| `self.ffn(x,input_ids)` | `moe_hash_fwd` / `moe_topk_fwd` | 直接调用；full 256-expert weight layout |
| FFN `hc_post` | 第二次 `hc_post_fwd` | 直接调用 |
| Official module-owned Attention cache | Explicit state inputs/outputs | 接口差异；语义等价的 state mutation |
| Official unified prefill/decode method | Separate prefill/decode kernel symbols | Specialization；tensor semantics 一致 |
| Current serving prefill | Full Block prefill kernels | 直接调用 |
| Current serving decode | `models/split_block.py` | Full decode kernels 存在但当前主干未使用 |
| `MTPBlock` | 无 | 不支持或未执行 |
| Tensor/Expert Parallel communication | 无 | 不支持或未执行；当前单卡逻辑 |

## 数据接口

### Dispatch matrix

[`models/config.py`](../../models/config.py) 的 43 个主模型层映射为：

| Layers | Compression ratio | Attention | MoE routing | Block kernel family |
|---|---:|---|---|---|
| 0–1 | 0 | Sliding Window Attention（SWA） | Hash | `block_swa_hash_*` |
| 2 | 4 | Compressed Sparse Attention（CSA） | Hash | `block_csa_hash_*` |
| Odd 3–41 | 128 | Heavily Compressed Attention（HCA） | Top-K | `block_hca_topk_*` |
| Even 4–42 | 4 | Compressed Sparse Attention（CSA） | Top-K | `block_csa_topk_*` |

不存在当前配置可达的 SWA+Top-K 或 HCA+Hash family。Prefill/decode 只改变 sequence、
position auxiliary inputs 和 state interface，不改变某层的 Attention/routing family。

### 公共 Block interface

所有 kernels 的主要模型 input/output 为：

```text
x:   [1,S,4,4096], BF16
out: [1,S,4,4096], BF16
```

Batch 和 HC streams 固定为 1 与 4。Prefill `S` 是动态 prompt length；full decode spec
固定 `S=1` 并要求 `start_pos>0`。完整模型的 `prefill + decode` 位置范围不超过 4096。

Block 还接收两组 HC FP32 weights、两个 BF16 RMSNorm weights、当前 Attention family
的 weights/auxiliary tensors，以及以下完整 MoE weights：

```text
gate_w_t:    [4096,256], BF16
routed_w1_t: [256,4096,2048], BF16
routed_w2_t: [256,2048,4096], BF16
routed_w3_t: [256,4096,2048], BF16
shared_w1_t: [4096,2048], BF16
shared_w2_t: [2048,4096], BF16
shared_w3_t: [4096,2048], BF16
```

Hash families 额外接收 `tid2eid [129280,6]` INT32 和 `input_ids [1,S]` INT64；Top-K
families 接收 `gate_bias [256]` FP32，不接收 `input_ids/tid2eid`。

Full Block 的 routed-expert interface 始终要求一层全部 256 个 BF16 experts，总计
12 GiB。Prefill serving 以 `PREFILL_ROUTED` staging 提供该 pack；Gate/shared/HC/
Attention/normalization weights 是 fixed `RuntimeWeight`。

### Internal intermediates 与 scratch

每个 kernel 内部按当前 `S` 创建两套子层 intermediates：

```text
attn_hc_x_mixed / attn_normed / attn_out: [1,S,4096], BF16
attn_hc_post: [1,S,8], FP32
attn_hc_comb: [1,S,16], FP32
attn_hc_out:  [1,S,4,4096], BF16

ffn_hc_x_mixed / ffn_normed / moe_out: [1,S,4096], BF16
ffn_hc_post: [1,S,8], FP32
ffn_hc_comb: [1,S,16], FP32
```

调用方还为 Attention/FFN 两次 HC pre 分别提供
`S_PAD=ceil_div(S,16)*16` 的 padded scratch。CSA/HCA 根据其 compressor/indexer
路径额外接收 `kv_pool` 等 scratch。它们是当前 dispatch 的临时 device buffers，不是
Block state。

### Attention state interface

所有 families 都输出 `kv_cache_out [1,128,512]` BF16。HCA 额外输出 ratio-128
compressor states/cache；CSA 额外输出 ratio-4 Attention compressor 和 Indexer
states/cache。具体 schema 与 shape 见
[`11_attention_swa.md`](11_attention_swa.md)、
[`12_attention_hca.md`](12_attention_hca.md) 和
[`13_attention_csa.md`](13_attention_csa.md)。

Prefill kernels 不接收旧 Attention state，从整个 prompt 构造 state outputs。Full
decode kernels 接收上一 position 的 state 和 `cache_pos/comp_slot` 等 scalar inputs，
并输出更新后的 state。Block kernel 自身不跨 dispatch 保存 state。

### Serving ownership

[`serving/state.py`](../../serving/state.py) 定义每层 state schema 和 position-dependent
auxiliary tensors；device runtime 为所有层预分配并持有 state。每个 prefill Block
kernel 完成后，Runner 调用 `commit_state()` 提交 state outputs，再把 `out` 直接作为
下一层 device input。

Prefill 每层从 packed expert cache 获得 full routed pack 并上传 staging。该 staging
不是 fixed weight，默认在 prefill step 结束时释放 allocation。Full decode standalone
spec 同样要求完整 routed pack；当前 serving decode 改用 Split Block 和 selected
staging，不走该接口。

## 实现方式

### 固定顺序的两段子层

四种 Block families 都遵循同一 composition：

1. Attention HC pre 产生 BF16 mixed hidden 和 FP32 `post/comb`；
2. Attention RMSNorm；
3. 对应 family 的 Attention，并产生 state outputs；
4. Attention HC post 将 Attention output 与 Block input residual 合成 4 streams；
5. FFN HC pre 产生第二组 mixed hidden 与 `post/comb`；
6. FFN RMSNorm；
7. Hash 或 Top-K full-expert MoE；
8. FFN HC post 将 MoE output 与 Attention HC output residual 合成最终 Block output。

这两个 residual 保存点不能互换：Attention HC post 消费原始 Block input，FFN HC
post 消费 Attention HC post 的 output。两个 RMSNorm 都位于对应 HC pre 之后、子层
计算之前。

### Attention/routing specialization

每个 top-level kernel 静态绑定一种 Attention 和一种 routing，因此 kernel 内没有按
layer id 的 runtime branch。Host 根据 `LayerSpec(ratio,hash_route)` 选择 family：ratio
0/4/128 分别选择 SWA/CSA/HCA，前三层选择 Hash Gate，其余层选择 Top-K Gate。

Attention、HC、RMSNorm 和 MoE 的 tile、rounding、mask、cache update 和 expert
aggregation 规则由对应 inline kernel 保持。Block 只组织 tensor boundary 与执行顺序；
具体数值细节见各子模块文档。

### Prefill 与 full decode

Prefill kernel 在 `start_pos=0` 上处理动态 `S`，从 prompt 生成 Attention state，并在
同一 dispatch 内运行完整 MoE。Non-aligned `S` 由 HC 和各 Attention kernel 的 padding/
valid-shape 机制处理。

Full decode kernel 固定单 token，读取已有 Attention state，在 `start_pos` 对应位置
更新 state 后运行完整 MoE。它在数学上对应官方单次 decode Block，但需要绑定完整
256-expert pack，因此当前 serving 不使用该执行形态。

### Device-resident layer chaining

Block `out` 是 device intermediate，Runner 不在层间导出到 host。Fixed weights 首次
materialize 后保持 device resident；Attention state 同样常驻 device。Prefill routed
pack 是例外，通过 bounded staging 逐层提供。完整模型只在必要的 public output/debug
或 Split Block control 边界发生 D2H。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、HC streams 4、43 个主模型层和 BF16 主计算路径；
- Prefill 支持动态 `S`，full decode 固定 `S=1` 且要求 `start_pos>0`；
- 当前仅实现配置可达的四种 Attention/routing 组合；
- 一个 full Block kernel 同时包含 Attention、MoE 和两次 HC，接口较大，并要求调用方
  提供所有 fixed weights、state/auxiliary tensors 和 scratch；
- Full Block MoE interface 始终接收全部 256 个 routed experts，即使每个 token 只执行
  6 个；
- 当前 Runner 只使用 full prefill kernels；full decode kernels 存在但主干未使用，
  serving decode 使用 Split Block 降低 expert weight load/H2D；
- Attention state 通过显式 input/output tensors 表达，不保存在 kernel object 内；
- 当前不执行官方 `rotate_activation`、FP4/FP8 runtime、Tensor Parallel、Expert
  Parallel 或集合通信；
- 当前不执行 MTP layer，本文只描述 43 个 main-model Blocks；
- Block 不包含 embedding 或 language-model head，它们是 Runner 中独立 dispatch；
- 子模块偶现精度误差的仓库级说明统一记录在
  [`README.md`](README.md)，本文不记录单次验收状态。

## Golden 参考实现

`models/block.py::golden_block_forward` 从 kernel-facing input snapshot 开始，按真实
Block 顺序调用当前仓库的 HC、RMSNorm、Attention 和 MoE PyTorch goldens。它要求
`start_pos>=0`；当 `start_pos>0` 时要求 `S=1`。最终写出 BF16 `out`，Attention golden
同时写出当前 family 的 state outputs。

8 个 public golden wrappers 固定以下参数：

| Wrapper suffix | `start_pos` | `attention_kind` | `hash_route` |
|---|---:|---|---|
| `swa_hash_prefill` | 0 | `swa` | true |
| `swa_hash_decode` | CLI value | `swa` | true |
| `csa_hash_prefill` | 0 | `csa` | true |
| `csa_hash_decode` | CLI value | `csa` | true |
| `hca_topk_prefill` | 0 | `hca` | false |
| `hca_topk_decode` | CLI value | `hca` | false |
| `csa_topk_prefill` | 0 | `csa` | false |
| `csa_topk_decode` | CLI value | `csa` | false |

Golden 不模拟 device-resident storage、packed-cache I/O、H2D staging 或 Runner state
commit；standalone spec 直接提供 kernel-facing tensors。

## 精度验收标准

Standalone Block 只比较最终 Block output 和 Attention state outputs：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `out` | `1e-3` | `2/128` | `0.005` |
| 所有 Attention cache/state outputs | `1e-4` | `1/128` | `0.001` |

所有 outputs 使用 `ratio_allclose`。最终 BF16 hidden 最多允许 0.5% 元素超出容差；
Attention state/cache 最多允许 0.1%。Actual output 中出现任何 NaN 或 Inf 都会直接
判为不合法。

## 验收方法

在 Ascend A2/A3 实机上验证四种 prefill 和四种 full decode kernels：

```bash
python models/block.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8 \
  --decode-start-pos 3 \
  --case all
```

使用非 16 对齐 prompt length 验证四种 prefill dynamic-tail 路径：

```bash
python models/block.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13 \
  --case prefill
```

使用 ratio-128 compression boundary 单独验证 HCA full decode：

```bash
python models/block.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 127 \
  --case hca-topk-decode
```

`--case prefill` / `--case decode` 分别选择四个同模式 cases，也可以使用完整 case name
选择单项。增加 `--compile-only` 可仅检查编译；增加 `--enable-l2-swimlane` 会把相应
选项传入 PyPTO `RunConfig`。

Host-side Block golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_block.py
```

## 集成验证范围

### Standalone Block 验收

`models/block.py::main()` 覆盖四种配置组合的 prefill/full decode kernels。每个 case
从独立的 weight、state 和 auxiliary snapshot 开始，不验证跨层 Runner orchestration
或真实 expert-cache I/O。

[`test_block.py`](../../tests/models/test_block.py) 使用缩小模型，在四种组合上分别验证
prefill 和 decode golden 与官方 `Block`，并比较最终 output 和 Attention state。该
测试把官方外部 sparse attention、HC Sinkhorn、linear/quantized primitives 替换为
当前仓库的 host references；因此它验证 Block composition、参数/layout 和 state
边界，不是对这些外部 primitive 的独立交叉实现比较，也不执行 PyPTO NPU kernel。

### Split decode 与 serving

- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖当前 serving decode
  的 pre/post-MoE 拆分与 full decode golden 等价；
- [`test_state.py`](../../tests/serving/test_state.py) 覆盖 43 层 dispatch spec、Attention
  state schema、prefill/decode auxiliary tensors 和 compression boundaries；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 覆盖 state
  device allocation、input/output mapping 和 commit；
- [`test_runner.py`](../../tests/serving/test_runner.py) 覆盖 prefill full Block 与 decode
  Split Block 的 orchestration、state bindings 和 device intermediate chaining；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 Block 各类 fixed
  weights、full routed pack 和 selected routed weights 的 runtime layouts。

这些 integration tests 验证 Block 在完整 runtime 中的 dispatch 与生命周期，不能
替代各 top-level Block kernel 的实机精度验收。
