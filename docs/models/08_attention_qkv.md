# Attention QKV

## 模块定位

Attention QKV 是所有 Attention 变体共用的输入投影组件。它接收 Block 中经过
Attention Hyper-Connection pre 和 `attn_norm` 后的 `attn_normed`，生成 64 个 query
heads 和一份 shared KV。当前模型不生成彼此独立的 K、V tensor；同一个 512 维
shared KV 同时参与 sparse-attention score 和 value aggregation。

```text
attn_normed x [1,S,4096], BF16
+ fixed weights {wq_a_t, q_norm_w, wq_b_t, wkv_t, kv_norm_w}
+ position inputs {cos, sin}
  ├─ Query low-rank path
  │    4096 -> 1024 -> learned RMSNorm -> qr [1,S,1024]
  │    1024 -> 32768 -> [1,S,64,512]
  │    per-head RMS rescale -> RoPE -> q [1,S,64,512]
  └─ Shared KV path
       4096 -> 512 -> learned RMSNorm -> RoPE -> kv [1,S,512]
```

`q` 和 `kv` 直接进入 SWA、CSA 或 HCA sparse attention。`qr` 是 query low-rank
中间边界：CSA 的 Indexer 会继续消费它；SWA/HCA 只用它生成主 query，随后不再使用。

该组件自身无持久 state。Prefill 和 decode 使用同一动态 sequence kernel；调用方
负责把 `kv` 写入 window cache，并为当前 step 提供正确位置区间的 cos/sin。
静态尺寸来自 [`models/config.py`](../../models/config.py)。

## 官方模型中的 Attention QKV

[`official/model.py`](../../official/model.py) 的 `Attention` 在 `forward()` 前半段完成
Q/KV 投影：

```python
qr = q = self.q_norm(self.wq_a(x))
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
apply_rotary_emb(q[..., -rd:], freqs_cis)

kv = self.kv_norm(self.wkv(x))
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```

相关参数为：

| 官方参数 | Shape（单卡逻辑） | 作用 |
|---|---:|---|
| `wq_a.weight` | `[1024,4096]` | Query low-rank down projection |
| `q_norm.weight` | `[1024]` | Low-rank query learned RMSNorm |
| `wq_b.weight` | `[32768,1024]` | 生成 `64 × 512` query channels |
| `wkv.weight` | `[512,4096]` | 生成 shared KV |
| `kv_norm.weight` | `[512]` | Shared KV learned RMSNorm |

`wq_b` 是 `ColumnParallelLinear`：官方多 rank 路径沿 query output/head 维切分，每个
rank 只生成 `n_local_heads`。`wq_a` 和 `wkv` 不沿该维切分。当前仓库只实现单卡
完整 64-head 语义。

`q_norm` 与 `kv_norm` 都具有 learned weight。`wq_b` 后的 per-head RMS rescale 没有
learned weight，只把每个 512 维 query head 按自身均方根缩放。RoPE 仅作用于 query
和 shared KV 的最后 64 维。

官方在 KV RoPE 后还会对前 448 个 non-RoPE channel 执行 activation quantization；
该操作不属于当前 BF16 runtime 路径。

## PyPTO kernel 实现

[`models/attention_qkv.py`](../../models/attention_qkv.py) 提供：

| 符号 | 类型 | 职责 |
|---|---|---|
| `attention_qkv_fwd` | `@pl.jit.inline` | 完成 Q/KV projection、normalization 和 forward RoPE |
| `attention_qkv_fwd_test` | `@pl.jit` | Standalone 编译和精度验收 wrapper |
| `golden_attention_qkv` | PyTorch golden | 对齐当前 BF16 rounding boundary 的参考计算 |
| `build_attention_qkv_specs` | Host spec builder | 构造指定 `seq_len/start_pos` 的输入与输出 |

Inline kernel 复用三类 primitive：

- [`models/linear.py`](../../models/linear.py) 中的
  `linear_4096_to_1024`、`linear_1024_to_32768` 和 `linear_4096_to_512`；
- [`models/rmsnorm.py`](../../models/rmsnorm.py) 中的 `rmsnorm_1024` 和
  `rmsnorm_512`；
- [`models/rope.py`](../../models/rope.py) 中的 `rope_4d_512_fwd` 和
  `rope_3d_512_fwd`。

对应 primitive 的通用数学定义与独立验收参见 [Linear](02_linear.md)、
[RMSNorm](01_rmsnorm.md) 和 [RoPE](03_rope.md)。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `wq_a(x)` | `linear_4096_to_1024` | 直接调用：inline kernel |
| `q_norm(...)` | `rmsnorm_1024` | 直接调用：inline kernel |
| `qr = q` low-rank boundary | `qr` output | 语义等价 |
| `wq_b(qr)` | `linear_1024_to_32768` | 直接调用：inline kernel |
| `unflatten(..., n_heads, head_dim)` | `pl.reshape(..., 64,512)` | 语义等价 |
| Query per-head RMS rescale | `q_scale` stage | 融合内联 |
| Query `apply_rotary_emb` | `rope_4d_512_fwd` | 直接调用：inline kernel |
| `wkv(x)` | `linear_4096_to_512` | 直接调用：inline kernel |
| `kv_norm(...)` | `rmsnorm_512` | 直接调用：inline kernel |
| Shared KV `apply_rotary_emb` | `rope_3d_512_fwd` | 直接调用：inline kernel |
| KV non-RoPE `act_quant` | 无 | 不支持或未执行：BF16 runtime 不量化 activation |
| `wq_b` Tensor Parallel output shard | 无 | 不支持或未执行：当前输出完整 64 heads |
| SWA QKV path | `attention_swa_*_fwd` | 直接调用 |
| CSA QKV path | `attention_csa_*_fwd` | 直接调用 |
| HCA QKV path | `attention_hca_*_fwd` | 直接调用 |
| CSA Indexer 复用 `qr` | `indexer_*_fwd` | 直接消费 QKV 输出 |

## 数据接口

公共接口为：

```text
x:          [1,S,4096],     BF16
wq_a_t:     [4096,1024],    BF16
q_norm_w:   [1024],         BF16
wq_b_t:     [1024,32768],   BF16
wkv_t:      [4096,512],     BF16
kv_norm_w:  [512],          BF16
cos:        [S,32],         FP32
sin:        [S,32],         FP32
qr:         [1,S,1024],     BF16
q:          [1,S,64,512],   BF16
kv:         [1,S,512],      BF16
```

其中：

- Batch 固定为 1，`S` 是动态 token 维；
- `x` 是当前 Attention sublayer 的 `attn_normed`，不是 HC layout 的四维 tensor；
- `*_t` 是 checkpoint linear weight 的转置 runtime layout；
- 所有 runtime weight 为 BF16；量化 checkpoint 在 host 加载阶段转换为 BF16；
- cos/sin 已经按当前 token 的实际 position 切片，宽度 32 对应 64 个 RoPE channel；
- `qr`、`q`、`kv` 都是当前调用输出，不是持久 state；
- `q_a`、`q_proj`、`kv_proj`、`kv_normed` 和 q-scale partial tensor 是 kernel-local
  scratch。

Prefill 使用 `[1,S,...]` 输出；decode 使用 `S=1`。Kernel 不接收 `start_pos`，位置
只通过 cos/sin 内容体现，因此同一 compiled shape 可以用于不同绝对位置。

完整模型通过 `DeepSeekV4WeightLoader.get_layer_attention_common()` 加载五组 QKV
weight/layout，并由 device runtime 作为 fixed weight 复用。

## 实现方式

### Query low-rank path

Query 路径按以下 BF16 boundary 执行：

1. `x @ wq_a_t` 使用 FP32 matmul accumulation，round-to-nearest 转为 BF16；
2. `rmsnorm_1024` 在 FP32 中计算均方、inverse RMS 和 learned weight，再转为 BF16
   `qr`；
3. `qr @ wq_b_t` 使用 FP32 accumulation，转为 BF16 `q_proj`；
4. reshape 为 `[1,S,64,512]`；
5. 对每个 token、每个 head 独立执行无 learned weight 的 RMS rescale；
6. 对最后 64 维执行 forward RoPE，输出 BF16 `q`。

Per-head rescale 使用最多 8 个 token 的 tile、每次 128 个 channel，共四个 channel
block，并沿 64 heads 执行 SPMD。均方、`sqrt/recip` 和缩放在 FP32 中完成，结果以
round-to-nearest 转回 BF16 后再进入 RoPE。非 8 对齐的 sequence tail 通过
`valid_shape` 处理。

### Shared KV path

Shared KV 路径为：

1. `x @ wkv_t` 以 FP32 accumulation 生成 BF16 `kv_proj`；
2. `rmsnorm_512` 使用 learned `kv_norm_w` 在 FP32 中归一化并转回 BF16；
3. `rope_3d_512_fwd` 复制前 448 维，只旋转最后 64 维，输出 BF16 `kv`。

该路径没有 head 维，因为当前模型使用一份 shared KV。QKV kernel 不更新 window
cache；SWA/CSA/HCA caller 在拿到 `kv` 后分别执行 prefill 或 decode cache update。

### RoPE profile 与位置区间

Kernel 只消费 cos/sin，不在内部选择 RoPE profile：

- Ratio 0 SWA 由 `DeepSeekV4StatePlan` 提供 normal profile；
- Ratio 4 CSA 和 ratio 128 HCA 提供 compressed profile；
- Prefill 提供从 position 0 开始的连续 `S` 行；
- Decode 提供 `[start_pos:start_pos+1]`。

Standalone `build_attention_qkv_specs()` 使用 normal profile，并通过 `start_pos` 切片
table。Compressed profile 的组合语义由 CSA/HCA 集成路径覆盖。

## 实现差异与限制

当前实现与官方 Attention QKV 路径的主要差异如下：

- 当前只支持 `B=1`、hidden 4096、Q-LoRA rank 1024、64 query heads、head dim 512
  和 RoPE width 64，不是任意 shape 的 QKV API；
- 当前是 shared-KV 接口，不提供独立 K 和 V 输出；
- 官方 `wq_b` 可按 query heads 执行 Tensor Parallel；当前为单卡完整 64-head
  输出，不包含 rank shard 或集合通信；
- 当前所有 weight 和激活接口为 BF16，linear、normalization 和 RoPE 的敏感计算在
  FP32 中完成；
- 官方在 KV RoPE 后量化 non-RoPE channel；当前不执行 activation quantization；
- Kernel 不拥有或更新 KV cache，cache lifecycle 属于 SWA/CSA/HCA caller；
- Kernel 不接收 compression ratio 或 `start_pos`，profile 和绝对位置由 cos/sin
  决定；
- Prefill/decode 共用一个实现，decode 由调用方约束为 `S=1`；
- 完整 runtime 的 position 上限为 4096。

## Golden 参考实现

`models/attention_qkv.py::golden_attention_qkv` 从 BF16 `x` 和 BF16 transposed
weight snapshot 开始，使用 FP32 `torch.matmul`，并在与 kernel 相同的阶段转为
BF16。

Query golden 依次执行 4096→1024 projection、learned RMSNorm、1024→32768
projection、64×512 reshape、无 weight per-head RMS rescale 和 forward RoPE。KV
golden 执行 4096→512 projection、learned RMSNorm 和 forward RoPE。

`_apply_rope_golden` 只替换最后 64 维。Golden 最终写出 `qr`、`q`、`kv` 三个 BF16
output，不包含 cache update、activation quantization 或 sparse attention。

## 精度验收标准

三个 standalone 输出使用相同标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `qr` | `1e-4` | `1/128` | `0.001` |
| `q` | `1e-4` | `1/128` | `0.001` |
| `kv` | `1e-4` | `1/128` | `0.001` |

逐元素容差为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的元素超出该条件，数量阈值按 comparator 对元素总数取整。Actual
output 中出现任何 NaN 或 Inf 都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上验证默认 token tile 和非零 position：

```bash
python models/attention_qkv.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8 \
  --start-pos 7
```

使用非 tile 对齐 sequence length 验证 tail：

```bash
python models/attention_qkv.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --start-pos 0
```

使用单 token 和较后位置验证 decode-like shape：

```bash
python models/attention_qkv.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 1 \
  --start-pos 127
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

`models/attention_qkv.py::main()` 直接编译和执行 `attention_qkv_fwd_test`，比较
`qr`、`q` 和 `kv`。当前没有单独的 `tests/models/test_attention_qkv.py`；standalone
入口是该组件独立 NPU 数值验收边界。

### Attention host 语义覆盖

- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py) 覆盖 normal RoPE、
  shared KV window cache 和 SWA prefill/decode；
- [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 覆盖 compressed
  RoPE、`qr` 到 Indexer 的传递、两套 Compressor 和 CSA prefill/decode；
- [`test_attention_hca.py`](../../tests/models/test_attention_hca.py) 覆盖 compressed
  RoPE、shared KV、Ratio-128 Compressor 和 HCA prefill/decode。

这些 host 测试通过完整 Attention golden 与官方 `Attention` 比较，间接覆盖 QKV
组合语义，但不直接执行 standalone NPU QKV kernel，不能替代组件独立验收。

### Block 与 serving 集成

- [`test_block.py`](../../tests/models/test_block.py) 覆盖 SWA/CSA/HCA 与
  Hyper-Connection、MoE 组成的完整 Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 selected-expert
  decode 的 QKV、cache 与完整 Block 对应关系；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证 QKV weight
  的 checkpoint mapping 和转置 runtime layout；
- [`test_state.py`](../../tests/serving/test_state.py) 验证三类 Attention 路径为 QKV
  提供的 normal/compressed RoPE profile 和 prefill/decode position slice。

完整模型中，QKV fixed weights、输出和后续 cache 都保持在 NPU。`q` 进入 sparse
attention，`kv` 由 caller 写入 device-resident window cache；CSA 还会把 `qr` 直接
传给 Indexer。QKV kernel 自身不负责 state commit。
