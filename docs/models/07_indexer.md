# Indexer

## 模块定位

Indexer 是 Compressed Sparse Attention（CSA）中的有状态候选位置选择组件。它为
每个 query token 计算已生成 compressed block 的相关性分数，并返回最多 512 个
compressed KV 位置；CSA 将这些位置与 128 个 sliding-window 位置拼接，作为 sparse
attention 的索引输入。

```text
Attention normalized input x [1,S,4096]
Attention low-rank query qr [1,S,1024]
  ├─ Indexer query projection + RoPE [1,S,64,128]
  ├─ per-head score weight [1,S,64]
  └─ Ratio-4 Indexer Compressor -> index KV cache [1,1024,128]
             ↓
      per-head query/KV score
             ↓ ReLU × head weight, sum heads
      compressed Top-K indices [1,S,512]
             ↓
      CSA window indices + compressed indices
```

Indexer 只存在于 compression ratio 为 4 的 CSA layer。当前配置来自
[`models/config.py`](../../models/config.py)：64 个 index heads、每个 head 128 维、
RoPE width 64、compression ratio 4、最大 Top-K 512。Ratio-0 SWA 不需要 compressed
position，ratio-128 HCA 使用规则化 compressed index，不创建 Indexer。

## 官方模型中的 Indexer

[`official/model.py`](../../official/model.py) 的 `Indexer` 包含三部分：

1. `wq_b` 将 Attention 已计算的 low-rank query `qr` 从 1024 维投影到
   `64 × 128 = 8192` 维；
2. `weights_proj` 将 hidden state 投影到 64 个 head weight；
3. 内部 `Compressor(..., compress_ratio=4, head_dim=128, rotate=True)` 构造用于
   scoring 的 128 维 compressed KV cache。

官方主要参数为：

| 参数 | Shape | 作用 |
|---|---:|---|
| `wq_b.weight` | `[8192,1024]` | 生成 64 个 128 维 index query head |
| `weights_proj.weight` | `[64,4096]` | 生成每个 token 的 64 个 head weight |
| `compressor.wkv.weight` | `[256,4096]` | Ratio-4 overlap KV projection |
| `compressor.wgate.weight` | `[256,4096]` | Ratio-4 overlap gate projection |
| `compressor.ape` | `[4,256]` | Block-relative FP32 gate score |
| `compressor.norm.weight` | `[128]` | Compressed index KV 的 RMSNorm weight |

`Indexer.forward()` 的 scoring 公式为：

$$
r_{s,h,t} = \operatorname{ReLU}(q_{s,h} \cdot k_t)
$$

$$
score_{s,t} = \sum_{h=0}^{63} r_{s,h,t} \cdot w_{s,h}
$$

其中 $k_t$ 是第 $t$ 个 compressed block 的 index KV，head weight 为：

$$
w = weights\_proj(x) \cdot 128^{-1/2} \cdot 64^{-1/2}
$$

官方在 query RoPE 后执行 `rotate_activation` 和 FP4 activation simulation，内部
Compressor 的 compressed KV 也执行相同处理。多 rank 路径会在 head-weighted score
上执行 `all_reduce`，再选择 Top-K。

Prefill 时，每个 token 只能看到已经完成的 4-token block。对于 zero-based token
位置 $s$，可见 compressed block 数为 `floor((s + 1) / 4)`；尚不可见的位置被 mask。
Decode 时，可见长度为 `floor((start_pos + 1) / 4)`，到 4-token boundary 时包含刚刚
生成的新 compressed block。

## PyPTO kernel 实现

[`models/indexer.py`](../../models/indexer.py) 提供两条 inline kernel：

| 符号 | 类型 | 职责 |
|---|---|---|
| `indexer_prefill_fwd` | `@pl.jit.inline` | 更新 prefill index cache/state，并逐 token 生成 Top-K |
| `indexer_decode_fwd` | `@pl.jit.inline` | 更新单 token state/cache，并生成当前 token Top-K |
| `indexer_prefill_test` | `@pl.jit` | Prefill standalone 验收 wrapper |
| `indexer_decode_test` | `@pl.jit` | Decode standalone 验收 wrapper |
| `build_indexer_prefill_specs` | Host spec builder | 构造动态 prefill 输入、辅助 tensor 和输出 |
| `build_indexer_decode_specs` | Host spec builder | 构造单 token decode state、控制量和输出 |
| `golden_indexer_forward` | PyTorch golden | 共用的 prefill/decode Indexer 参考实现 |

Indexer 直接调用
[`models/compressor_ratio4.py`](../../models/compressor_ratio4.py) 中的 Indexer
Compressor inline kernel。Ratio-4 overlap pooling 和 state shift 详见
[Ratio-4 Compressor](06_compressor_ratio4.md)。

完整模型由 [`models/attention_csa.py`](../../models/attention_csa.py) 调用 Indexer，
然后将 `topk_idxs` 与 sliding-window indices 合并。Indexer standalone wrapper 不
包含 CSA sparse attention 本身。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `Indexer.forward` prefill | `indexer_prefill_fwd` | 语义等价：固定单卡 shape |
| `Indexer.forward` decode | `indexer_decode_fwd` | 语义等价：固定单 token shape |
| `wq_b(qr)` | `linear_1024_to_8192` | 直接调用：inline kernel |
| Query `unflatten(..., 64,128)` | `pl.reshape` | 语义等价 |
| Query `apply_rotary_emb` | `rope_4d_128_fwd` | 直接调用：inline kernel |
| Query `rotate_activation` / `fp4_act_quant` | 无 | 不支持或未执行：BF16 runtime 不做 Hadamard rotation/FP4 simulation |
| `self.compressor(x, start_pos)` | `compressor_ratio4_indexer_*_fwd` | 直接调用：inline kernel |
| `weights_proj(x)` | `linear_4096_to_64` | 直接调用：inline kernel |
| `einsum("bshd,btd->bsht")` | 32-row cache tile matmul | 语义等价：FP32 accumulation |
| `relu(score) * weights` 后按 head 求和 | Score tile 内部计算 | 融合内联 |
| 多 rank `all_reduce(index_score)` | 无 | 不支持或未执行：当前为单卡逻辑 |
| Prefill visibility mask | `visible_len=(t+1)//4` | 融合内联 |
| `score.topk(...)` | 重复 `row_argmax` 与 selected-position mask | 语义等价 |
| CSA 调用 Indexer | `attention_csa_*_fwd` | 直接调用 |
| CSA Block 调用 | `block_csa_*_fwd` / selected decode pre-MoE | 融合内联：经 CSA 调用 |

## 数据接口

### 公共输入和权重

Prefill 与 decode 共用：

```text
x:                  [1,S,4096],   BF16
qr:                 [1,S,1024],   BF16
wq_b_t:             [1024,8192],  BF16
weights_proj_t:     [4096,64],    BF16
cos/sin:            [S,32],       FP32
offset:             [1],          INT32
comp_wkv_t:         [4096,256],   BF16
comp_wgate_t:       [4096,256],   BF16
comp_ape:           [4,256],      FP32
comp_norm_w:        [128],        BF16
```

`qr` 不是 Indexer 自己生成的输入。它来自主 Attention 的
`wq_a -> q_norm` low-rank query 路径，并同时供主 query projection 与 Indexer 使用。
所有 `*_t` linear weight 都是 checkpoint weight 的转置 runtime layout。

Prefill 中 `offset=S`，因为 CSA prefill KV pool 先放置 `S` 行当前 prompt KV，再追加
Attention compressed KV。Decode 中 `offset=window_size=128`，因为 decode KV pool
由 128 行 window cache 后接 compressed cache。Indexer 返回的非负位置已经加上
offset，可以直接作为 CSA KV pool index。

### Prefill 接口

```text
comp_cos/sin:             [C,32],       FP32
comp_block_count:         [1],          INT32
topk_idxs:                [1,S,512],    INT32
index_kv_cache:           [1,1024,128], BF16
comp_kv_state_out:        [1,8,256],    FP32
comp_score_state_out:     [1,8,256],    FP32
```

其中 `blocks=floor(S/4)`、`C=max(1,blocks)`。当没有完整 block 时，Compressor RoPE
保留一行占位，`index_kv_cache` 初始化为零，所有 `topk_idxs` 为 `-1`。对于有完整
block 的 token，最多写入 `min(512,visible_len)` 个有效 index，其余位置保持 `-1`。

### Decode 接口

```text
x:                        [1,1,4096],   BF16
qr:                       [1,1,1024],   BF16
comp_kv_state:            [1,8,256],    FP32
comp_score_state:         [1,8,256],    FP32
index_kv_cache_in:        [1,1024,128], BF16
comp_slot:                [1],          INT32
comp_cache_slot:          [1],          INT32
comp_should_compress:     [1],          INT32
comp_cos/sin:             [1,32],       FP32
topk_idxs:                [1,1,512],    INT32
index_kv_cache:           [1,1024,128], BF16
comp_kv_state_out:        [1,8,256],    FP32
comp_score_state_out:     [1,8,256],    FP32
```

控制量满足：

```text
comp_slot = start_pos % 4
comp_cache_slot = start_pos // 4
comp_should_compress = int((start_pos + 1) % 4 == 0)
```

Decode 只支持 `S=1`。有效 cache 长度为
`comp_cache_slot + comp_should_compress`；boundary step 会先把新 compressed KV 写入
cache，再把它纳入当前 token 的 ranking。非 boundary step 的 Compressor cos/sin
是零占位，index cache 内容不变，但 overlap staging state 仍会更新。

### State 所有权

完整模型中的三组 Indexer 持久 state 为：

| State | Shape | Dtype | 初始值 |
|---|---:|---|---|
| `idx_kv_cache` | `[1,1024,128]` | BF16 | 0 |
| `idx_comp_kv_state` | `[1,8,256]` | FP32 | 0 |
| `idx_comp_score_state` | `[1,8,256]` | FP32 | FP32 最小有限值 |

[`serving/state.py`](../../serving/state.py) 声明这些逻辑 state，
[`serving/device_state_store.py`](../../serving/device_state_store.py) 为每组 state 分配
current/next 两个持久 NPU buffer。Indexer 读取 current、写入 next；完整 ratio-4
Block 执行结束后由 Runner 提交并交换 state。

`q_proj`、`q_rope`、head `weights`、`index_score` 和 Compressor 的临时
`comp_compressed` 都是单次 kernel 内部 scratch。`topk_idxs` 是交给 CSA 的当前
step 中间 tensor，不跨 prefill/decode step 保存。

## 实现方式

### Query 与 head weight

Indexer 复用 Attention 的 BF16 `qr`，通过 `linear_1024_to_8192` 生成 query，并
reshape 为 `[1,S,64,128]`。`rope_4d_128_fwd` 保留每个 index head 的前 64 维，只
旋转最后 64 维；ratio-4 layer 的主 query 和 Indexer query 都使用 compressed RoPE
profile。

`linear_4096_to_64` 从 `x` 生成每个 token 的 64 个 BF16 head weight。Kernel 将其
转为 FP32，乘以 `128^-0.5 * 64^-0.5`，再 round-to-nearest 转回 BF16；scoring 时
重新转为 FP32。

### Index KV 更新

Indexer 在 scoring 前调用 Ratio-4 Indexer Compressor：

- prefill 压缩所有完整 4-token block，并初始化 1024-row index cache；
- decode 每步更新 8-row overlap state；
- 只有 4-token boundary 生成新的 128 维 BF16 index KV；
- 新 KV 写入 `index_kv_cache[comp_cache_slot]` 后，当前 token 即可看到该 slot。

Indexer 不直接读取 Compressor 的临时 `compressed` 输出；后续 scoring 统一从更新
后的 `index_kv_cache` 读取。

### Score 计算

Index cache 以 32 行为一个 tile。每个 tile 的主要计算为：

1. 取 query tile `[64,128]` 和 KV tile `[32,128]`；
2. 以 FP32 accumulation 计算 `[32,64]` 的 per-head dot product；
3. 对 dot product 执行 ReLU；
4. 逐 head 乘当前 token 的 64 个 FP32 weight；
5. 沿 head 维求和，得到 32 个 compressed position score；
6. 将 tile 尾部和不可见 cache row 填为 `NEG_INF`。

Prefill 对每个 token 独立计算 `visible_len=min((t+1)//4, blocks)`。Decode 使用
`cache_len=comp_cache_slot+comp_should_compress`，只处理已经生成的 cache prefix。
`index_score` 固定为 `[1,S,1024]` FP32 scratch，不作为公共输出。

### Top-K 与 index offset

每个 score row 先复制到 `[8,1024]` work tensor 的第一行。Kernel 最多迭代 512
次 `row_argmax`：每次写出最佳位置，再将已选位置替换为 `NEG_INF`。有效选择数为
`min(512,visible_len)` 或 `min(512,cache_len)`，其余输出保持 `-1`。

Prefill/decode 的本地 cache index 分别加 `S`/128 offset。CSA 随后将这 512 个位置
接在 128 个 sliding-window index 后形成 `[1,S,640]` sparse-attention index；
`-1` 表示无有效 compressed candidate。

### Host 辅助输入复用

`DeepSeekV4StatePlan` 为同一个 prefill `seq_len` 或 decode `start_pos` 缓存 main
RoPE、Compressor RoPE、offset、slot、cache slot 和 boundary flag。Indexer 与
Attention Compressor 共用相同的 Compressor RoPE slice；相同 step 的不同 ratio-4
layer 也复用这些 immutable host tensor。

## 实现差异与限制

当前实现与官方 Indexer 的主要差异如下：

- 当前仅支持 ratio 4、`B=1`、64 heads、head dim 128、Top-K 512 和最大 4096
  positions，不是通用 Indexer API；
- 当前只在 CSA layer 使用 Indexer；HCA 不执行 learned Indexer scoring；
- 官方对 query 和 compressed index KV 执行 `rotate_activation`；当前不执行
  Hadamard rotation；
- 官方模拟 FP4 query/KV activation；当前 query、index cache 和中间模型接口保持
  BF16；
- 当前 score matmul、ReLU-weighted sum 和 visibility scratch 使用 FP32；
- 当前为单卡逻辑，不执行官方多 rank score `all_reduce`；
- 官方 `topk` 输出最后维度为 `min(index_topk,end_pos/ratio)`；当前接口固定为 512，
  不足位置以 `-1` padding；
- 当前不输出 `index_score`，只输出 Top-K indices 和更新后的 state/cache；
- Prefill offset 固定为当前 `seq_len`，decode offset 固定为 window size 128；该约束
  与 CSA KV pool layout 绑定；
- Decode 入口要求 `start_pos>0` 且 sequence length 为 1；
- 1024-row index cache 对应当前完整模型固定的 4096 position 上限。

## Golden 参考实现

`models/indexer.py::golden_indexer_forward` 从 BF16 `qr`、`x` 和 transposed weight
snapshot 开始。Query projection 使用 FP32 `torch.matmul` 后转为 BF16，再通过
`_apply_rope_golden` 旋转最后 64 维。Ratio-4 Indexer Compressor 由
`golden_compressor_ratio4_indexer_forward` 更新 state/cache。

Head weight 使用 FP32 matmul，按当前实现的 scale 转为 BF16 后再转回 FP32。
Golden 以 FP32 `einsum` 生成 per-head score，执行 ReLU、head weighting 和求和；
prefill visibility mask 使用 `NEG_INF`，最后通过 `torch.topk` 选择位置并应用
offset 和 `-1` padding。

Host 测试与官方模型比较时关闭 `rotate_activation` 和 FP4 activation simulation，
因此验证目标是本仓库定义的单卡 BF16 Indexer 语义。Golden 不将 `index_score`
暴露为输出，Top-K 是 scoring 路径的直接验收边界。

## 精度验收标准

Standalone Indexer 的输出分为 index 与 tensor 两类：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `topk_idxs` | `1e-5` | `1e-5` | `0`；INT32 实际要求完全相同 |
| `index_kv_cache` | `1e-4` | `1/128` | `0.001` |
| `comp_kv_state_out` | `1e-4` | `1/128` | `0.001` |
| `comp_score_state_out` | `1e-4` | `1/128` | `0.001` |

三个浮点输出的逐元素容差为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的浮点元素超出该条件，数量阈值按 comparator 对元素总数取整；Actual
浮点输出出现任何 NaN 或 Inf 都会直接判为不合法。`topk_idxs` 未使用浮点
ratio comparator，而是使用 `run_jit` 默认 `torch.allclose`；由于两侧均为 INT32，
且当前 index 范围内的绝对与相对容差之和远小于 1，因此等价于逐元素精确比较。

## 验收方法

在 Ascend A2/A3 实机上执行默认 prefill 和 boundary decode：

```bash
python models/indexer.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 16 \
  --decode-start-pos 3 \
  --case all
```

使用一个完整 block 加 remainder 验证 prefill visibility 和 padding：

```bash
python models/indexer.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --case prefill
```

分别验证不生成和生成新 index KV 的 decode step：

```bash
python models/indexer.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 2 \
  --case decode

python models/indexer.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 3 \
  --case decode
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

`models/indexer.py::main()` 分别编译和执行 prefill/decode wrapper，比较 fixed-width
Top-K、index cache 和两组 overlap Compressor state。该入口不执行 CSA sparse
attention，适合独立定位 query、score、Top-K 或 index-cache 更新误差。

### Host Indexer 语义覆盖

[`test_indexer.py`](../../tests/models/test_indexer.py) 覆盖：

- prefill 长度 3、4、7、8、13、16 和 32；
- decode position 1、2、3 和 7；
- 无可见 block、boundary 前后、多个 block、visibility mask 和 `-1` padding；
- Top-K、index KV cache 和两组 Compressor state 与官方 `Indexer` 的对应结果。

这些 host 测试验证官方到当前 BF16 语义映射，不编译或执行 NPU kernel，不能替代
standalone 实机验收。

### Compressor、CSA 与 Block 集成

- [`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py) 独立覆盖
  Indexer Compressor 的 overlap state/cache；
- [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 覆盖 Indexer、
  Attention Compressor、sliding-window indices、sparse attention 和 output
  projection 的 CSA 组合；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 Indexer 所在的 CSA
  hash/Top-K MoE Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 验证 selected-expert
  decode 拆分前后 Indexer state/cache 与完整 CSA decode Block 一致。

这些组合测试扩大完整数据流覆盖，但不能替代 standalone Indexer 对 Top-K 和 state
输出的独立误差定位。

### Serving state 与权重生命周期

- [`test_state.py`](../../tests/serving/test_state.py) 验证 ratio-4 Indexer state、
  offset、slot、boundary flag 和共享 RoPE auxiliary tensor；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证 `wq_b`、
  `weights_proj` 和内部 Compressor weight 的 checkpoint mapping 与 runtime layout；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 验证
  Indexer 三组 state 的 device allocation、双 buffer 和初始值。

完整模型由 `DeepSeekV4Runner` 通过 `get_layer_indexer()` 加载六组 fixed weight，并
把 Indexer state/cache 保持在 NPU。Prefill 在完整 CSA Block 内调用 Indexer；decode
在 selected-expert pre-MoE kernel 内调用 Indexer，Runner 随后提交 state，并把
`topk_idxs` 作为同一 kernel 内的中间 tensor 交给 CSA sparse attention。
