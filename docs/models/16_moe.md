# Mixture-of-Experts

## 模块定位

Mixture-of-Experts（MoE）是每个 Transformer Block 的 FFN 子层。它接收经过 FFN
Hyper-Connection pre 和 `ffn_norm` 的 hidden state，通过 Gate 为每个 token 选择 6
个 routed experts，同时始终执行 1 个 shared expert，最后把 7 条 Expert output 相加。

```text
FFN normalized input x [1,S,4096], BF16
  -> Gate
       -> indices [1,S,6], INT32
       -> weights [1,S,6], FP32
  ├─ expert-major route packing
  │    -> 16-row routed Expert tiles
  │    -> packed_y [28416,4096], BF16 scratch
  └─ 1 shared Expert
       -> shared_y [1,S,4096], BF16
  -> inverse-map FP32 route aggregation
  -> out [1,S,4096], BF16
```

根据 [`models/config.py`](../../models/config.py)，当前 43 个主模型层都使用 MoE：每层
有 256 个 routed experts、每个 token 激活 6 个、shared expert 数量固定为 1。0-based
layer 0、1、2 使用 hash routing，layer 3 到 42 使用 score-based Top-K routing。

本文描述 [`models/moe.py`](../../models/moe.py) 的 MoE 组合、routed Expert 调度、
prefill/decode 接口差异以及 serving 权重边界。Gate 和单 Expert 的内部算法分别见
[`14_gate.md`](14_gate.md) 与 [`15_expert.md`](15_expert.md)。

## 官方模型中的 MoE

[`official/model.py`](../../official/model.py) 的 `MoE` 包含以下子模块和配置：

| 官方字段 | 当前配置 | 职责 |
|---|---:|---|
| `gate` | 1 | 生成 routed expert indices 和 routing weights |
| `experts` | 256 | Routed `Expert` 列表 |
| `shared_experts` | 1 | 始终执行的 shared `Expert` |
| `n_activated_experts` | 6 | 每个 token 执行的 routed Expert 数量 |
| `n_local_experts` | `256 / world_size` | 当前 rank 拥有的 routed experts |

官方先把 `x` 展平为 token-major `[B*S,4096]`，调用 Gate 得到 `(weights,
indices)`，再统计每个 expert 被选择的次数。对当前 rank 拥有且至少被选择一次的每个
expert，官方使用 `torch.where(indices == expert_id)` 找到 `(token, route slot)`，执行：

$$
y_t \mathrel{+}= Expert_{I_{t,k}}(x_t,w_{t,k})
$$

其中 routing weight 在 routed Expert 的 SwiGLU hidden 转回 BF16 之前相乘。Routed
accumulator `y` 为 FP32。多 rank 时，官方随后对 `y` 执行 `all_reduce`；再执行无
routing weight 的 shared expert：

$$
out_t = \operatorname{BF16}\left(y_t+Expert_{shared}(x_t)\right)
$$

官方按 expert id 递增顺序调度和累加 routed output，并支持 Expert Parallel 以及 FP4
routed-expert weight。当前实现使用单卡 BF16 路径，不执行这些分布式和量化 kernel。

## PyPTO kernel 实现

[`models/moe.py`](../../models/moe.py) 提供 full-expert 与 selected-expert 两种执行
形态：

| 符号 | 类型 | 职责 |
|---|---|---|
| `_pack_routes_by_expert` | `@pl.jit.inline` | 统计 expert count，并构造 16 行对齐的 expert-major route 元数据 |
| `_run_expert_major_routed_experts` | `@pl.jit.inline` | 每个 expert 以最多 16 条 route 为一组执行多行 routed Expert 计算 |
| `_run_selected_experts_decode` | `@pl.jit.inline` | 使用按 route slot 排列的 6 组 selected weights 计算 routed outputs |
| `_combine_expert_major` | `@pl.jit.inline` | 根据 inverse route map 合并 prefill packed outputs 与 shared output |
| `_combine_route_major` | `@pl.jit.inline` | 合并 selected-decode 的 6 条 route outputs 与 shared output |
| `moe_hash_fwd` | `@pl.jit.inline` | Hash Gate + full routed experts + shared expert + aggregation |
| `moe_topk_fwd` | `@pl.jit.inline` | Top-K Gate + full routed experts + shared expert + aggregation |
| `moe_selected_decode_experts_fwd` | `@pl.jit.inline` | 已知 Gate weights 后执行 selected routed experts、shared expert 与聚合 |
| `moe_hash_test` | `@pl.jit` | Hash full-expert standalone wrapper |
| `moe_topk_test` | `@pl.jit` | Top-K full-expert standalone wrapper |
| `moe_selected_decode_experts_test` | `@pl.jit` | Selected decode standalone wrapper |
| `golden_moe_forward` | PyTorch golden | Hash/Top-K full-expert 共用参考实现 |
| `golden_moe_selected_decode_experts_forward` | PyTorch golden | Selected decode 参考实现 |
| `build_moe_hash_specs` | Host spec builder | 构造 hash full-expert tensors |
| `build_moe_topk_specs` | Host spec builder | 构造 Top-K full-expert tensors |
| `build_moe_selected_decode_specs` | Host spec builder | 构造固定 `S=1` 的 selected decode tensors |

`moe_hash_fwd` 和 `moe_topk_fwd` 分别直接调用
[`models/gate.py`](../../models/gate.py) 的 `gate_hash_fwd` / `gate_topk_fwd`，并直接
调用 [`models/expert.py`](../../models/expert.py) 的 `expert_shared_fwd`。Routed Expert
数学计算融合在两个 routed execution kernel 内，没有逐 route 调用
`expert_routed_fwd`。

完整 prefill Block 直接复用 full-expert MoE inline kernel；完整 decode Block 通过
[`models/split_block.py`](../../models/split_block.py) 把 Gate 与 selected-expert MoE
拆成 pre/post 两段，post-MoE kernel 直接调用 `moe_selected_decode_experts_fwd`。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `self.gate(x, input_ids)` | `gate_hash_fwd` / `gate_topk_fwd` | Full-expert MoE 直接调用 |
| 按 `indices` dispatch token | `_pack_routes_by_expert` | 融合内联；仅打包 token id、route weight 和 inverse map |
| `expert(x[idx], weights[idx,top,None])` | `_run_expert_major_routed_experts` 内的 `w1/w3 → SwiGLU → weight → w2` | 融合内联、语义等价 |
| Decode 只执行 selected experts | `_run_selected_experts_decode` | 语义等价；weight 已按 route slot 排列 |
| `self.shared_experts(x)` | `expert_shared_fwd` | 直接调用 |
| FP32 routed accumulator | `_combine_expert_major` / `_combine_route_major` | 语义等价；shared output 作为初始 accumulator |
| Output 转回输入 dtype | 两个 combine kernel 的 BF16 `rint` | 直接对应 |
| 官方 expert-major accumulation | Prefill expert-major compute + inverse-map aggregation | 调度语义等价；最终加法仍按 token 的 route slot 顺序 |
| Expert Parallel expert shard | 无 | 不支持或未执行；当前固定单卡 256 experts |
| Routed output `all_reduce` | 无 | 不支持或未执行；当前单卡无需通信 |
| FP4 routed Expert runtime | 无 | 不支持或未执行；host 侧提供 BF16 runtime weights |

## 数据接口

### 公共输入与输出

三种 MoE 接口共用：

```text
x:             [1,S,4096], BF16
shared_w1_t:   [4096,2048], BF16
shared_w2_t:   [2048,4096], BF16
shared_w3_t:   [4096,2048], BF16
out:           [1,S,4096], BF16
```

Batch 固定为 1。Full-expert hash/Top-K 路径的 `S` 是动态 sequence 维，并限制在
`1..4096`；selected decode 的 standalone spec 和完整 runtime 固定 `S=1`。`x` 是 FFN
normalized hidden state，MoE output 尚未经过 FFN Hyper-Connection post。

所有 Expert weights 均使用 checkpoint weight 的转置 BF16 runtime layout。Shared
expert 不接收 routing weight。

### Full-expert routed weights

Hash 与 Top-K full-expert 路径都接收一层全部 256 个 routed experts：

```text
routed_w1_t: [256,4096,2048], BF16
routed_w2_t: [256,2048,4096], BF16
routed_w3_t: [256,4096,2048], BF16
```

第一维直接以 `indices[b,s,k]` 作为 expert id 索引。Hash 路径额外接收：

```text
gate_w_t: [4096,256], BF16
tid2eid:  [129280,6], INT32
input_ids:[1,S],      INT64
```

Top-K 路径额外接收：

```text
gate_w_t:  [4096,256], BF16
gate_bias: [256],      FP32
```

Gate 在 full-expert MoE 内生成以下 scratch：

```text
indices:          [1,S,6],       INT32
weights:          [1,S,6],       FP32
expert_counts:    [256,1],       INT32
expert_bases:     [256,1],       INT32
packed_token_ids: [28416,8],     INT32
packed_weights:   [1,28416],     FP32
route_to_packed:  [24576,8],     INT32
packed_y:         [28416,4096],  BF16
shared_y:         [1,S,4096],    BF16
```

`MAX_ROUTES=4096*6=24576`。`packed_y` 为每个 expert 预留 16 行对齐的连续区间，额外
`256*(16-1)` 行覆盖所有 expert 都产生尾块时的最坏 padding。实际计算只遍历
`expert_counts` 指定的 route；`route_to_packed` 把原始 `(token,route slot)` 映射回 packed
row。Scratch 使用静态最大 shape，但不是跨层或跨 step 的持久 state。

### Selected decode weights

Selected decode 接收已经由上层 Gate 生成的 FP32 weights，不在当前 kernel 内再次执行
Gate：

```text
x:             [1,1,4096], BF16
weights:       [1,1,6],    FP32
selected_w1_t: [6,4096,2048], BF16
selected_w2_t: [6,2048,4096], BF16
selected_w3_t: [6,4096,2048], BF16
```

Selected weight 第一维不是全局 expert id，而是 Gate route slot。第 `k` 组
`selected_w*_t` 必须对应 `weights[:,:,k]`。该接口不接收 indices；indices 只用于
kernel 外选择和排列 6 组 weights。若 hash table 产生重复 expert id，对应 weight
slice 也会按 route slot 重复保留。

Selected kernel 内部仍创建 `[1,1,6,4096]` BF16 `route_y` 和 `[1,1,4096]` BF16
`shared_y`。

### State、cache 与 runtime 边界

MoE 数学 kernel 没有跨层或跨 step 的持久 state。当前 serving runtime 对不同权重
采用两类生命周期：

- Gate 与 shared-expert weights 是固定 `RuntimeWeight`，首次 materialize 后作为
  fixed weights 常驻 device；
- Routed-expert weights 不进入 fixed-weight 集合，通过 `HostStagingTensor` 上传到
  bounded device staging buffer。

Prefill 通过 [`serving/weight_loader.py`](../../serving/weight_loader.py) 的
`get_layer_moe_routed_pack()` 读取一层完整 packed BF16 expert weights。优先从
[`serving/expert_cache.py`](../../serving/expert_cache.py) 的逐层 safetensors cache
取得三组 packed tensors；cache 不包含该层时，weight loader 从 checkpoint 逐 expert
构造相同 layout。三组 tensor 以 `PREFILL_ROUTED` staging 进入 device。

Decode 的上层 pre-MoE kernel 输出 `indices` 和 `weights`。Runner 仅将 INT32 indices
读取到 host，调用 `get_layer_moe_selected_experts()` 按 6 个 route slots 从 packed
cache 复制相应 weight slices；FP32 weights 和 FFN hidden 保持在 device。三组 selected
tensors 以 `DECODE_SELECTED` staging 进入 post-MoE kernel，kernel 完成后归还 buffer
pool，后续 decode step 可以复用相同 staging allocation。

默认情况下，prefill routed staging allocation 在 prefill step 结束时释放。
`--keep-prefill-routed-staging` 会保留这些 device buffer 供后续 allocation 复用，但
不会把 routed expert weights 转换为 fixed `RuntimeWeight`。

## 实现方式

### Full-expert expert-major 执行

`moe_hash_fwd` / `moe_topk_fwd` 先执行对应 Gate，生成 route-slot 顺序的 indices 和
weights。`_pack_routes_by_expert` 在一个 metadata task 中执行两遍 route 扫描：第一遍统计
256 个 expert 的 route count 并生成 16 行对齐的 prefix base，第二遍写入 packed token id、
route weight 和 inverse map。它不会复制 `[route,4096]` activation。

`_run_expert_major_routed_experts` 再按 expert 和 16-row tile 调度：

1. 根据 packed token id 把最多 16 条原始 `x` row gather 到一个 BF16 tile；
2. 对该 tile 复用同一 expert 权重，用 BF16 activation/weight、FP32 accumulation 同时
   执行 `w1` 和 `w3` projection；
3. 把 gate/up projection 转为 BF16 snapshot，再在 FP32 中执行 clamp 和 SwiGLU；
4. 在 hidden BF16 rounding 之前按 row 乘对应 `packed_weights`；
5. 执行 `w2` FP32 accumulation，把 BF16 output 写入该 expert 的连续 `packed_y` rows。

该实现融合了单 routed Expert 的完整数学计算。每次只计算 Gate 选出的 6 条 routes，
但 full-expert weight tensor 仍包含全部 256 个 experts。同一个 expert 在 tile 中命中多条
route 时，三组权重由多行 matmul 共享读取。

### Selected-expert decode

`_run_selected_experts_decode` 不读取 expert id。它按 `k=0..5` 直接访问第 `k` 组
selected weights，并保留与 full-expert routed kernel 相同的 BF16/FP32 计算和舍入
顺序。这样 decode kernel 只接收当前 token 实际使用的 6 个 experts，而不接收完整
256-expert weight pack。

`moe_selected_decode_experts_fwd` 不包含 Gate；它接收 pre-MoE 阶段保留的 routing
weights，执行 selected routed experts、shared expert 和最终聚合。Indices 已在 host
weight selection 后消费，不进入 post-MoE kernel。

### Shared Expert 与 route aggregation

三种路径都直接调用 `expert_shared_fwd` 生成 BF16 `shared_y`。Prefill 使用
`_combine_expert_major`，逐 token 通过 `route_to_packed` 读取 6 条 BF16 packed rows；
selected decode 使用 `_combine_route_major` 读取 `[1,1,6,4096]` `route_y`。两者都先把
shared output 转成 FP32 accumulator，按 route slot 0 到 5 累加，最后以 `rint` 转成 BF16
`out`。

因此 routing weight 已经在每条 routed Expert 内部生效，aggregation 阶段不再乘
weight。`route_y`、`shared_y` 和最终 `out` 的舍入边界与单 Expert 的内部 hidden
rounding 是不同层次的边界。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、intermediate 2048、256 routed experts、Top-K 6
  和 1 个 shared expert；
- Full-expert 路径支持动态 `S=1..4096`，selected decode 的 spec 和完整 runtime 固定 `S=1`；
- Hash routing 固定用于 layer 0、1、2，其他 40 层使用 biased Top-K routing；
- Full-expert MoE 只计算每个 token 被选中的 6 条 routes，但接口仍要求提供一层全部
  256 个 routed-expert weights；
- Selected decode 要求 6 组 weights 已按 Gate route slot 排列，kernel 不验证其全局
  expert id；
- Prefill 按 expert-major tile 执行 routed Expert，但最终仍按 token 的 route slot 顺序聚合；
  官方按 expert id 顺序累加，两者数学等价但 FP32 浮点加法顺序不同；
- 当前不实现 Expert Parallel、expert shard、跨 rank 通信或 `all_reduce`；
- 当前不执行 FP4/FP8 Expert kernel；routed/shared Expert runtime weights 均为 BF16；
- MoE 不包含持久模型 state；serving 中的 packed cache 和 staging 是权重生命周期，
  不是 kernel state；
- `moe_selected_decode_experts_fwd` 不执行 Gate，只接受已生成的 routing weights；
- 当前 kernel 假设 indices、route-slot weight 对应关系和 expert-id 范围均由 Gate、
  weight loader 与调用方保证。

## Golden 参考实现

`models/moe.py::golden_moe_forward` 覆盖 hash 与 Top-K full-expert 路径。它从 BF16
`x`、Gate tensors、256-expert packed BF16 weights 和 shared-expert BF16 weights
开始，先用 PyTorch 复现 Gate，再对每个 `(token,route slot)` 调用内部
`_expert_forward_golden` 生成 BF16 `route_y`。

为匹配官方 `MoE.forward()`，full-expert golden 按 expert id、token、route slot 顺序
把匹配的 route output 加入 FP32 `routed_acc`，然后加 shared BF16 output 并转回 BF16。
`golden_moe_hash` 和 `golden_moe_topk` 只是固定 `hash_route` 参数的 wrapper。

`golden_moe_selected_decode_experts_forward` 从 `S=1` 的 BF16 `x`、FP32 routing
weights 和 6 组 selected BF16 weights 开始，不重复执行 Gate。它按 route slot 生成
BF16 route outputs，在 FP32 中沿 route 维求和，再加 shared output 并转为 BF16。

Golden 不读取 serving expert cache，也不模拟 H2D、device buffer pool 或 host control
回传；standalone tensor spec 已提供 kernel-facing layout。

## 精度验收标准

Hash、Top-K 和 selected-decode 三类 standalone case 都只比较 BF16 `out`：

| 输出 | 验收方式 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---|---:|---:|---:|
| `out` | `ratio_allclose` | `1e-3` | `2/128` | `0.005` |

逐元素容差为：

```text
abs(actual - expected) <= 1e-3 + (2/128) * abs(expected)
```

允许最多 0.5% 的 output 元素超出该条件。Actual output 中出现任何 NaN 或 Inf 都会
直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上依次验证 hash、Top-K 和 selected-decode：

```bash
python models/moe.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8 \
  --case all
```

`--case all` 中的 selected-decode case 始终使用 `S=1`，不受 `--seq-len` 影响。使用
非 16 对齐 sequence length 验证两个 full-expert dynamic-tail 路径：

```bash
python models/moe.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13 \
  --case hash

python models/moe.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13 \
  --case topk
```

也可通过 `--case selected-decode` 单独验证 decode MoE。以上命令可增加
`--compile-only` 仅检查编译，或增加 `--enable-l2-swimlane` 把相应选项传入 PyPTO
`RunConfig`。

Host-side MoE golden 与官方模型以及 selected/full 等价关系可运行：

```bash
pytest -q tests/models/test_moe.py
```

## 集成验证范围

### 独立 MoE 验收

`models/moe.py::main()` 分别提供 `hash`、`topk` 和 `selected-decode` 三类 case。
Full-expert cases 覆盖 Gate、routed experts、shared expert 和 aggregation；selected
case 从已知 routing weights 开始，不覆盖 Gate。

[`test_moe.py`](../../tests/models/test_moe.py) 使用缩小的 hidden、expert 数量和 Top-K
配置，在 sequence length 1、3、13 上逐元素比较 hash/Top-K golden 与官方 `MoE`；
同时对 hash 和 Top-K 两类 `S=1` 输入比较 selected golden 与 full-expert golden。该
host test 不执行 PyPTO NPU kernel，不能替代 standalone 实机验收。

### Block 与 selected decode 集成

- [`test_block.py`](../../tests/models/test_block.py) 覆盖 full-expert MoE 进入完整 Block
  后与 FFN norm、Hyper-Connection 的组合；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 decode pre-MoE
  Gate output、selected weight 绑定、post-MoE 与完整 Block 的对应关系；
- [`test_runner.py`](../../tests/serving/test_runner.py) 覆盖 Runner 通过 runtime 读取
  indices，以及 prefill full pack / decode selected weights 的绑定。

### Serving 权重生命周期

- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 Gate/shared
  fixed runtime layouts、full routed pack、selected slices、expert id 顺序与范围校验；
- [`test_expert_cache.py`](../../tests/serving/test_expert_cache.py) 覆盖 packed BF16 cache
  metadata、整层读取和 selected slice copy；
- [`test_device_runtime.py`](../../tests/serving/test_device_runtime.py) 覆盖 fixed weights、
  `PREFILL_ROUTED` / `DECODE_SELECTED` staging allocation 的复用与释放。

这些 serving tests 验证权重和 buffer 生命周期，不替代 MoE 数学 kernel 的精度验收。
