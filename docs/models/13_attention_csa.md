# Compressed Sparse Attention

## 模块定位

Compressed Sparse Attention（CSA）是当前模型中 `compress_ratio == 4` 的完整
Attention 执行路径。它保留最近 128 个 token 的 sliding window，并用 Ratio-4
Attention Compressor 为更早上下文生成 512 维 compressed KV；Indexer 再根据当前
query 从最多 1024 个 compressed positions 中选择最多 512 个候选。

```text
Attention normalized input x [1,S,4096], BF16
+ fixed Attention / Compressor / Indexer weights
+ compressed-profile position inputs
+ window indices [1,S,128]
  -> Attention QKV
       -> q [1,S,64,512]
       -> qr [1,S,1024] -> Indexer
       -> shared window kv [1,S,512]
  -> Ratio-4 Indexer Compressor -> index KV cache [1,1024,128]
       -> Indexer score / Top-K -> compressed indices [1,S,512]
  -> Ratio-4 Attention Compressor -> compressed KV [1,C,512]
  -> KV pool: [window kv/cache, Attention compressed kv/cache]
  -> concatenate window + Indexer indices [1,S,640]
  -> 640-candidate sparse attention + per-head sink
  -> inverse RoPE + grouped output projection
  -> out [1,S,4096], BF16
```

CSA 使用两套彼此独立的 Ratio-4 Compressor：Attention Compressor 生成真正参与
sparse attention value aggregation 的 512 维 compressed KV；Indexer Compressor
生成仅用于候选打分的 128 维 index KV。Indexer 返回的位置最终索引 Attention
compressed KV pool，而不是把 index KV 当作 attention value。

根据 [`models/config.py`](../../models/config.py) 的当前 43 层主模型配置，ratio 4
出现在 0-based layer 2、4、6，依次到 42，共 21 层。Layer 2 与 hash-routing MoE
组合，其余 20 层与 Top-K MoE routing 组合；两类 Block 使用相同 CSA Attention，
差异只在后续 MoE routing。

## 官方模型中的 Compressed Sparse Attention

[`official/model.py`](../../official/model.py) 在 `compress_ratio == 4` 时为
`Attention` 同时创建：

- `Compressor(args, compress_ratio=4, head_dim=512)`，生成 Attention compressed KV；
- `Indexer(args, compress_ratio=4)`，内部再创建
  `Compressor(..., head_dim=128, rotate=True)`；
- 128-row window cache 与 1024-row Attention compressed cache；
- 独立的 1024-row Indexer KV cache；
- compressed RoPE profile。

官方数据流先生成主 `q`、low-rank boundary `qr` 和 shared window `kv`。Indexer
消费 `x` 与 `qr`，更新自己的 compressed KV cache，计算 query/cache relevance score，
并返回最多 512 个 compressed positions。Attention 将它们与
`get_window_topk_idxs()` 的 window positions 拼接，再运行 sparse attention。

对于 zero-based token position $t$，可见 compressed block 数为：

$$
n_{visible}(t) = \left\lfloor\frac{t+1}{4}\right\rfloor
$$

Indexer 只在这些可见 positions 中取 `min(512,n_visible)` 项。Prefill 返回的 Indexer
位置增加 offset `S`；decode 增加 offset 128，使它们直接指向对应 Attention KV pool
的 compressed segment。未达到 4-token boundary 的位置不会看到尚未完成的 block。

到 `(start_pos+1) % 4 == 0` 的 decode boundary 时，两套 Compressor 都先生成新的
compressed KV。Indexer 将新 128 维 index KV 纳入当前 query ranking；被选中的
position 随后读取同 slot 的新 512 维 Attention compressed KV。

Ratio-4 overlap pooling 与六组 Compressor state 详见
[Ratio-4 Compressor](06_compressor_ratio4.md)，Indexer scoring 与 Top-K 规则详见
[Indexer](07_indexer.md)。QKV、sparse aggregation 和输出投影分别参见
[Attention QKV](08_attention_qkv.md)、[Sparse Attention](09_sparse_attn.md) 与
[Attention Out](10_attention_out.md)。

## PyPTO kernel 实现

[`models/attention_csa.py`](../../models/attention_csa.py) 提供一项 index 拼接 helper
和 prefill/decode 组合 kernel：

| 符号 | 类型 | 职责 |
|---|---|---|
| `build_csa_prefill_topk` | `@pl.jit.inline` | 拼接 128-wide window 与 512-wide Indexer indices |
| `attention_csa_prefill_fwd` | `@pl.jit.inline` | 组合 CSA prefill 的 QKV、两套 Compressor、Indexer、KV pool、sparse attention 和 output projection |
| `attention_csa_decode_fwd` | `@pl.jit.inline` | 组合单 token CSA decode 与全部 state update |
| `attention_csa_prefill_test` | `@pl.jit` | Prefill standalone 验收 wrapper |
| `attention_csa_decode_test` | `@pl.jit` | Decode standalone 验收 wrapper |
| `golden_attention_csa_forward` | PyTorch golden | 共用的 prefill/decode 完整 CSA 参考实现 |
| `build_csa_prefill_specs` | Host spec builder | 构造指定 `seq_len` 的 CSA prefill tensors |
| `build_csa_decode_specs` | Host spec builder | 构造指定 `start_pos` 的单 token decode tensors |

`build_csa_prefill_topk` 虽保留 prefill 名称，但 prefill 和 decode kernel 都直接调用
它；该 helper 只做固定宽度拼接，不含阶段相关逻辑。

组合 kernel 直接调用：

- `attention_qkv_fwd`；
- `update_prefill_window_cache` / `update_decode_window_cache`；
- `indexer_prefill_fwd` / `indexer_decode_fwd`；
- `compressor_ratio4_attention_prefill_fwd` /
  `compressor_ratio4_attention_decode_fwd`；
- HCA 文件中复用的 `build_prefill_kv_pool` / `build_decode_kv_pool`；
- `sparse_attn_csa_fwd`；
- `attention_out_fwd`。

完整 Block 进一步提供 `block_csa_hash_*_fwd` 和 `block_csa_topk_*_fwd`。Selected
decode 对应 `csa_hash_selected_decode_pre_moe_fwd` 与
`csa_topk_selected_decode_pre_moe_fwd`，两者都在 pre-MoE 段内完成 CSA 及全部
Attention state update。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `compress_ratio == 4` Attention dispatch | ratio-4 CSA Block cases | 语义等价：当前主模型 21 层 |
| Q/KV projection 与 forward RoPE | `attention_qkv_fwd` | 直接调用：inline kernel |
| KV non-RoPE `act_quant` | 无 | 不支持或未执行：BF16 runtime 不量化 activation |
| Window cache update | `update_*_window_cache` | 直接调用：inline kernel |
| `Attention.compressor` | `compressor_ratio4_attention_*_fwd` | 直接调用：inline kernel |
| `Attention.indexer` | `indexer_*_fwd` | 直接调用：inline kernel |
| `Indexer.compressor` | Indexer 内的 ratio-4 compressor | 直接调用：inline kernel |
| `rotate_activation` / FP4 simulation | 无 | 不支持或未执行：当前 BF16 路径不做 Hadamard rotation/量化 |
| Window + Indexer index concatenation | `build_csa_prefill_topk` | 语义等价：prefill/decode 共用 |
| `torch.cat([kv, kv_compress], dim=1)` | `build_prefill_kv_pool` | 语义等价 |
| Decode 读取组合 Attention cache | `build_decode_kv_pool` | 语义等价：从 window/Attention compressed state 组装 |
| `sparse_attn(..., topk width 640)` | `sparse_attn_csa_fwd` | 直接调用：inline kernel |
| inverse RoPE 与 `wo_a`/`wo_b` | `attention_out_fwd` | 直接调用：inline kernel |
| 官方 module-owned mutable state | device `current`/`next` state pairs | 语义等价：成功执行后交换 buffer |
| Indexer score multi-rank `all_reduce` | 无 | 不支持或未执行：当前为单卡逻辑 |
| Tensor Parallel Attention | 无 | 不支持或未执行：当前计算完整 heads/groups |

## 数据接口

### 主 Attention 输入和权重

Prefill 与 decode 共用：

```text
x:             [1,S,4096],     BF16
wq_a_t:        [4096,1024],    BF16
q_norm_w:      [1024],         BF16
wq_b_t:        [1024,32768],   BF16
wkv_t:         [4096,512],     BF16
kv_norm_w:     [512],          BF16
attn_sink:     [64],           FP32
window_topk_idxs: [1,S,128],   INT32
wo_a_t:        [4096,8192],    BF16
wo_b_t:        [8192,4096],    BF16
cos/sin:       [S,32],         FP32
out:           [1,S,4096],     BF16
```

主 Attention 使用 compressed RoPE profile。QKV kernel 输出的 `qr [1,S,1024]`
同时进入主 query projection 和 Indexer；它是 CSA 特有的跨组件中间边界。

### Attention Compressor 输入

```text
attn_comp_wkv_t:       [4096,1024], BF16
attn_comp_wgate_t:     [4096,1024], BF16
attn_comp_ape:         [4,1024],    FP32
attn_comp_norm_w:      [512],       BF16
attn_comp_cos/sin:     [C,32] / [1,32], FP32
attn_comp_block_count: [1],         INT32  # 仅 prefill
```

### Indexer 与 Indexer Compressor 输入

```text
idx_wq_b_t:            [1024,8192], BF16
idx_weights_proj_t:    [4096,64],   BF16
idx_offset:            [1],         INT32
idx_comp_wkv_t:        [4096,256],  BF16
idx_comp_wgate_t:      [4096,256],  BF16
idx_comp_ape:          [4,256],     FP32
idx_comp_norm_w:       [128],       BF16
idx_comp_cos/sin:      [C,32] / [1,32], FP32
idx_comp_block_count:  [1],         INT32  # 仅 prefill
```

Prefill 中 `idx_offset=S`，decode 中 `idx_offset=128`。Host runtime 让 Attention 与
Indexer Compressor 共用数值相同的 compressed RoPE slice 和 block/boundary 控制，
但 kernel 接口保留各自参数名。

### Prefill 接口

Prefill 满足 `start_pos=0`，并使用或写出：

```text
blocks = floor(S/4)
C = max(1, blocks)
kv_pool:                  [1,S+C,512], BF16 caller-provided workspace
kv_cache_out:             [1,128,512], BF16
attn_comp_kv_state_out:   [1,8,1024],  FP32
attn_comp_score_state_out:[1,8,1024],  FP32
attn_comp_cache_out:      [1,1024,512],BF16
idx_kv_cache_out:         [1,1024,128],BF16
idx_comp_kv_state_out:    [1,8,256],   FP32
idx_comp_score_state_out: [1,8,256],   FP32
```

当 `blocks=0` 时，两套 Compressor 都保留一行 shape placeholder，但 Indexer 不返回
有效 compressed position，sparse indices 不会引用 Attention KV pool 的 placeholder。
Prefill 不读取旧 state，而是从 prompt 重建 window cache 和六组 Compressor/Indexer
state。

### Decode 接口

Decode 固定 `S=1`，额外输入和输出为：

```text
kv_cache:                 [1,128,512], BF16
attn_comp_kv_state:       [1,8,1024],  FP32
attn_comp_score_state:    [1,8,1024],  FP32
attn_comp_cache:          [1,1024,512],BF16
idx_kv_cache_in:          [1,1024,128],BF16
idx_comp_kv_state:        [1,8,256],   FP32
idx_comp_score_state:     [1,8,256],   FP32
cache_pos:                [1],         INT32
comp_slot:                [1],         INT32
comp_cache_slot:          [1],         INT32
comp_should_compress:     [1],         INT32

kv_cache_out:             [1,128,512], BF16
attn_comp_kv_state_out:   [1,8,1024],  FP32
attn_comp_score_state_out:[1,8,1024],  FP32
attn_comp_cache_out:      [1,1024,512],BF16
idx_kv_cache_out:         [1,1024,128],BF16
idx_comp_kv_state_out:    [1,8,256],   FP32
idx_comp_score_state_out: [1,8,256],   FP32
out:                      [1,1,4096],  BF16
```

控制量为：

```text
cache_pos = start_pos % 128
comp_slot = start_pos % 4
comp_cache_slot = start_pos // 4
comp_should_compress = int((start_pos + 1) % 4 == 0)
```

Decode Attention KV pool 固定为 `[1,1152,512]`：前 128 行是更新后的 window cache，
后 1024 行是更新后的 Attention compressed cache。Indexer cache 不进入该 KV pool；
它只用于生成 512-wide compressed indices。

### State 所有权和生命周期

[`serving/state.py`](../../serving/state.py) 为每个 CSA layer 声明七组 state：

| State | Input name | Output name | Shape | Dtype | 初始值 |
|---|---|---|---:|---|---|
| `kv_cache` | `kv_cache` | `kv_cache_out` | `[1,128,512]` | BF16 | 0 |
| `attn_comp_kv_state` | `attn_comp_kv_state` | `attn_comp_kv_state_out` | `[1,8,1024]` | FP32 | 0 |
| `attn_comp_score_state` | `attn_comp_score_state` | `attn_comp_score_state_out` | `[1,8,1024]` | FP32 | FP32 最小有限值 |
| `attn_comp_cache` | `attn_comp_cache` | `attn_comp_cache_out` | `[1,1024,512]` | BF16 | 0 |
| `idx_kv_cache` | `idx_kv_cache_in` | `idx_kv_cache_out` | `[1,1024,128]` | BF16 | 0 |
| `idx_comp_kv_state` | `idx_comp_kv_state` | `idx_comp_kv_state_out` | `[1,8,256]` | FP32 | 0 |
| `idx_comp_score_state` | `idx_comp_score_state` | `idx_comp_score_state_out` | `[1,8,256]` | FP32 | FP32 最小有限值 |

[`serving/device_state_store.py`](../../serving/device_state_store.py) 为每组 state 分配
current/next 两个持久 NPU buffer。Prefill 写 next outputs；decode 读取 current 并写
next。完整 Block 或 selected decode pre-MoE kernel 成功返回后，Runner 一次性校验并
交换七组 buffer，保证 Attention KV 与 Indexer KV 使用同一个 step 的一致 state。

Standalone prefill 从新 output state 开始；standalone decode 使用独立构造的随机
current state/cache，不是同一次 standalone prefill 的延续。Host-side
`test_attention_csa.py` 则从官方 prefill state 构造 decode input，用于验证组合语义。
Serving 中七组 state 在 prefill 后跨 decode steps 连续复用并始终驻留 NPU。

## 实现方式

### Prefill 路径

`attention_csa_prefill_fwd` 按以下顺序执行：

1. `attention_qkv_fwd` 生成 compressed-profile `qr`、`q` 和 shared KV；
2. 更新 128-row window cache output；
3. `indexer_prefill_fwd` 更新 128 维 index KV cache/state，并为每个 token 产生最多
   512 个已加 `idx_offset=S` 的 compressed positions；
4. `build_csa_prefill_topk` 将 window 与 Indexer indices 拼成 `[1,S,640]`；
5. Attention Ratio-4 Compressor 生成 512 维 compressed KV 及其 state/cache；
6. 构造 `[prompt KV, Attention compressed KV]` pool；
7. 执行 CSA sparse attention 和 Attention Out。

对于 token `t`，Indexer 只对 `floor((t+1)/4)` 个可见 blocks 排序，并返回最多 512
个 position；其余 index slots 为 `-1`。Window 部分选择
`[max(0,t-127),...,t]`。两部分各自补齐到 128/512，最终固定宽度为 640。

Attention 与 Indexer Compressor 都以 `blocks=floor(S/4)` 生成相同数量的 compressed
rows，但 weight、output width、state/cache 独立。两套 cache 的 slot 编号一一对应，
使 Indexer 选出的本地 slot 可以加 offset 后直接索引 Attention compressed KV。

### Decode 路径

`attention_csa_decode_fwd` 的顺序为：

1. 生成当前 token 的 `qr`、`q` 和 window KV；
2. 更新 window current→next cache；
3. `indexer_decode_fwd` 更新 Indexer Compressor state/cache，并基于更新后的 index
   KV cache 产生当前 token 的 512-wide Top-K；
4. 拼接 window 与 Indexer indices；
5. 更新 Attention Compressor state/cache；
6. 构造 `[updated window cache, updated Attention compressed cache]`；
7. 执行 640-candidate sparse attention 和 Attention Out。

Boundary step 中两套 Compressor 同时写入 `comp_cache_slot`。Indexer 的 ranking 在
Attention Compressor 执行前产生，但两套 Compressor 使用相同输入、block boundary
与 slot 编号，因此 ranking 中的新 slot 会在随后构造 Attention KV pool 前写入。
非 boundary step 只更新 overlap staging state，两套 compressed cache 保持不变。

Decode Indexer 的可见 cache 长度为
`comp_cache_slot + comp_should_compress = floor((start_pos+1)/4)`。它从该前缀选择最多
512 个 slots，再加 offset 128；window indices 使用与
[Sliding Window Attention](11_attention_swa.md) 相同的 ring-slot 语义。

### KV pool、Top-K 与 online softmax

Prefill KV pool 的有效结构是 `[S prompt rows, floor(S/4) compressed rows]`；不足一个
block 时的 extra row 只是 shape placeholder。Decode pool 固定包含 128 + 1024 rows，
但每个 query 只通过 indices 读取其中最多 128 + 512 项。

`sparse_attn_csa_fwd` 将 640 个 candidates 按 16-row chunks 执行 online softmax，
并把 `attn_sink` 作为无 value 的额外 logit。该实现细节与精度边界参见
[Sparse Attention](09_sparse_attn.md)。

### Host auxiliary cache

`DeepSeekV4StatePlan` 为 CSA 构造并缓存：

- compressed-profile 主 Attention cos/sin；
- 128-wide window indices；
- prefill 的两组 block count、`idx_offset=S` 和 Ratio-4 Compressor RoPE；
- decode 的 window/cache slots、boundary flag、`idx_offset=128` 和 boundary RoPE。

Attention Compressor 与 Indexer Compressor 共用同一份 host cos/sin slice；同一
prefill length 或 decode position 的 immutable auxiliary inputs 跨 ratio-4 layers
复用。Mutable state/cache 不属于 host auxiliary cache。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、64 main/index heads、main head dim 512、index head
  dim 128、window 128、ratio 4 和最大位置 4096，不是通用 CSA API；
- 当前主模型有 21 个 CSA layers，其中 layer 2 与 hash MoE、其余层与 Top-K MoE
  组合；MoE routing 不参与 CSA Indexer ranking；
- Attention/Indexer compressed cache 都固定为 1024 rows，Indexer 最多返回 512 个
  positions，总 sparse width 固定为 640；
- 当前使用 compressed RoPE profile，与 HCA 的主 Attention profile 相同；
- 官方 Indexer query/compressed KV 执行 Hadamard rotation 与 FP4 simulation，当前
  BF16 runtime 不执行；
- 官方 ordinary/Attention compressed KV 执行 activation quantization，当前不执行；
- 官方将 window 与 Attention compressed KV 放在同一 cache；当前拆成独立 state，
  并在 kernel 中显式构造 KV pool；
- 当前使用 current/next buffers，不原地修改 state input；
- 当前不实现 Tensor Parallel 或 Indexer score `all_reduce`；
- Prefill 必须从 `start_pos=0` 开始，decode 必须满足 `start_pos>0` 且 `S=1`；
- Kernel 不包含 Attention 前后的 Hyper-Connection 或 Attention RMSNorm。

## Golden 参考实现

`models/attention_csa.py::golden_attention_csa_forward` 从 BF16 activation/weights、
FP32 sink/Compressor parameters、INT32 auxiliary inputs 和可选 current state snapshot
开始，按组合 kernel 的 BF16 boundary 执行：

1. Q/KV projection、normalization 与 compressed-profile RoPE；
2. window ring-cache construction/update；
3. `golden_indexer_prefill/decode` 的 Indexer Compressor、score、visibility 和 Top-K；
4. window/Indexer index concatenation；
5. `golden_compressor_ratio4_attention_forward` 的 overlap pooling 与 state/cache；
6. prefill/decode Attention KV pool construction；
7. `golden_sparse_attn` 的 FP32 score、sink softmax 和 value aggregation；
8. inverse RoPE、grouped `wo_a` 与 `wo_b` projection。

Golden 写出 `kv_cache_out`、六组 Compressor/Indexer state outputs 和最终 `out`。
Prefill `kv_pool` 是 caller-provided workspace，不作为 comparator output。Decode golden
校验 `S=1`、window cache position，以及两套 Compressor 共用的 slot/cache-slot/
boundary 公式。官方 activation quantization、FP4 simulation 和 Hadamard rotation 在
当前 BF16 golden 中明确移除。

`golden_attention_csa_prefill` 和 `golden_attention_csa_decode` 是共用 golden 的两类
wrapper。

## 精度验收标准

State/cache 与最终输出使用两组标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `kv_cache_out` | `1e-4` | `1/128` | `0.001` |
| `attn_comp_kv_state_out` | `1e-4` | `1/128` | `0.001` |
| `attn_comp_score_state_out` | `1e-4` | `1/128` | `0.001` |
| `attn_comp_cache_out` | `1e-4` | `1/128` | `0.001` |
| `idx_kv_cache_out` | `1e-4` | `1/128` | `0.001` |
| `idx_comp_kv_state_out` | `1e-4` | `1/128` | `0.001` |
| `idx_comp_score_state_out` | `1e-4` | `1/128` | `0.001` |
| `out` | `1e-3` | `2/128` | `0.005` |

逐元素容差分别为：

```text
state/cache:
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)

out:
abs(actual - expected) <= 1e-3 + (2/128) * abs(expected)
```

State/cache 允许最多 0.1% 元素超出容差，`out` 允许最多 0.5%。数量阈值按 comparator
对元素总数取整；actual output 中出现任何 NaN 或 Inf 都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上验证多个完整 Ratio-4 blocks 和 decode compression boundary：

```bash
python models/attention_csa.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 16 \
  --decode-start-pos 3 \
  --case all
```

验证一个完整 block 加 remainder，以及不触发 compression 的 decode step：

```bash
python models/attention_csa.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --decode-start-pos 4 \
  --case all
```

使用跨 window 长度验证 prompt ring-cache layout：

```bash
python models/attention_csa.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 129 \
  --case prefill
```

可通过 `--case prefill` 或 `--case decode` 单独选择路径。如需仅检查编译，可增加
`--compile-only`；如需启用 L2 swimlane，可增加 `--enable-l2-swimlane`，该参数会
直接传入 PyPTO `RunConfig`。

Host-side 完整 CSA golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_attention_csa.py
```

## 集成验证范围

### 独立 kernel 验收

`models/attention_csa.py::main()` 分别编译和执行 `attention_csa_prefill_test` 与
`attention_csa_decode_test`，比较 window cache、六组 Compressor/Indexer outputs
和最终 `out`。两个 case 独立构造 state，decode case 不是同一次 standalone
prefill 的延续。

[`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 使用缩小 shape，把
官方 rotation/activation quantization 替换为 identity，并以当前 PyTorch
sparse-attention golden 替换外部 callable。它在多个 prefill lengths 和 decode
positions 上逐元素比较完整 CSA golden、官方 `Attention.forward()` output、window
cache、两套 Compressor state/cache 与 Indexer cache。

### 组件与 Block 集成

- [`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py) 独立覆盖
  两套 Ratio-4 Compressor 的 overlap pooling、remainder、boundary 和 state machine；
- [`test_indexer.py`](../../tests/models/test_indexer.py) 独立覆盖 Indexer query、score、
  visibility、Top-K 和 offset；
- [`test_sparse_attn.py`](../../tests/models/test_sparse_attn.py) 覆盖 640-wide interface
  的 synthetic indices 以及 window helper；
- [`test_attention_out.py`](../../tests/models/test_attention_out.py) 覆盖 inverse RoPE
  与 output weight layout；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 CSA + hash/Top-K MoE 的完整
  prefill/decode Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖两种 CSA
  selected-expert decode 拆分与完整 Block 的 state/output 对应关系。

### Serving state 与权重生命周期

- [`test_state.py`](../../tests/serving/test_state.py) 覆盖 ratio-4 layer selection、七组
  state schema、compressed RoPE、window indices、slot/boundary/offset 和 host aux cache；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 覆盖七组
  state 的 current/next device allocation、初始值校验和重复 commit buffer 复用；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 CSA Attention、
  两套 Compressor 与 Indexer fixed weights 的 checkpoint mapping、runtime shape 和
  layout。

完整模型 prefill 在 CSA Block kernel 后提交 window 与六组 Compressor/Indexer state。
Decode 的 selected-expert 路径在 pre-MoE kernel 返回后立即提交七组 state，再执行
routed/shared expert 的 post-MoE kernel。Fixed weights、state/cache 与跨组件中间
tensor 都保持 device resident，不在 decode steps 之间回传 host。
