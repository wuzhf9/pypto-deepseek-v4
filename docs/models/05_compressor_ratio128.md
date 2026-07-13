# Ratio-128 Compressor

## 模块定位

Ratio-128 Compressor 是 Heavily Compressed Attention（HCA）的有状态输入压缩
组件。它把每个连续 128-token block 的 512 维 KV projection 压缩为一个 512 维
vector，使 HCA 在保留 128-token sliding window 的同时，以较小的 compressed KV
cache 表示更早的 token。

这里的 `x` 不是 Block 的原始 hidden state，而是经过 Attention Hyper-Connection
pre 和 `attn_norm` 后的 `attn_normed`。它是 Compressor 唯一的模型激活输入；其余
输入由 fixed weight、位置辅助 tensor、decode 控制量和 serving-owned state/cache
组成。

Prefill 从当前 prompt 重新构造 Compressor state：

```text
attn_normed x [1,S,4096], BF16
+ fixed weights {comp_wkv_t, comp_wgate_t, comp_ape, comp_norm_w}
+ auxiliary {comp_cos, comp_sin, comp_block_count}
  -> KV projection + gate projection [1,S,512], FP32
  -> 每 128 token 按 channel 执行 gated softmax pooling
  -> RMSNorm
  -> 最后 64 个 channel 执行 compressed-profile RoPE
  ├-> compressed KV [1,floor(S/128),512], BF16
  │     -> HCA sparse-attention KV pool
  └-> next state/cache
        {comp_kv_state, comp_score_state, comp_cache}
```

Decode 还会读取上一 step 的 device-resident state，并写入独立的 next buffer：

```text
attn_normed x [1,1,4096], BF16
+ fixed weights {comp_wkv_t, comp_wgate_t, comp_ape, comp_norm_w}
+ auxiliary {comp_cos, comp_sin, comp_slot, comp_cache_slot,
             comp_should_compress}
+ current {comp_kv_state, comp_score_state, comp_cache}
  -> 更新当前 128-token staging slot
  -> boundary 时执行 pooling + RMSNorm + RoPE
  -> next {comp_kv_state, comp_score_state, comp_cache}
```

非 boundary decode step 只更新 staging state 并保持 compressed cache；boundary step
才产生一个新的 compressed KV。Current/next state 由 serving runtime 持有和交换，
不是 kernel-local scratch。

当前完整模型只在 [`models/config.py`](../../models/config.py) 中 compression ratio 为
128 的主模型层选择该路径，即 0-based layer 3、5、7，依次到 41，共 20 层。这些层
位于三个 hash-routing layer 之后，与 Top-K MoE routing 组合。

## 官方模型中的 Ratio-128 Compressor

[`official/model.py`](../../official/model.py) 使用通用 `Compressor` 类同时表达 ratio
4 和 ratio 128。`Attention.__init__()` 在当前层 `compress_ratio != 0` 时创建
Compressor；ratio 128 使用 `head_dim=512`、`overlap=False`、`rotate=False`。

Ratio-128 Compressor 包含四组可学习参数：

| 官方参数 | Shape | 参数 dtype/语义 |
|---|---:|---|
| `wkv.weight` | `[512,4096]` | 官方 module 中为 FP32；checkpoint 存储 BF16，投影为待聚合 KV |
| `wgate.weight` | `[512,4096]` | 官方 module 中为 FP32；checkpoint 存储 BF16，生成 gate score |
| `ape` | `[128,512]` | block 内相对位置对应的 FP32 additive score |
| `norm.weight` | `[512]` | 官方 module 中为 FP32；checkpoint 存储 BF16，供 pooling 后 RMSNorm 使用 |

对于 block 内第 $i$ 个 token 和第 $d$ 个 channel，官方 pooling 可以写为：

$$
s_{i,d} = (x_i W_{gate})_d + APE_{i,d}
$$

$$
\alpha_{i,d} = \operatorname{softmax}_{i}(s_{i,d})
$$

$$
p_d = \sum_{i=0}^{127}\alpha_{i,d}(x_i W_{kv})_d
$$

softmax 沿 128-token 维执行，每个输出 channel 拥有独立的 token 权重。Pooling 后
先转回输入 dtype 并执行 RMSNorm，再只对最后 64 个 channel 应用 compressed RoPE。

官方 prefill 压缩所有完整 block，并把不足 128 个 token 的尾部投影保存到
`kv_state` 和 `score_state`。Decode 每步写入 `start_pos % 128` 对应的 state slot；
只有 `(start_pos + 1) % 128 == 0` 时才聚合一个完整 block，并写入
`kv_cache[:, start_pos // 128]`。

## PyPTO kernel 实现

[`models/compressor_ratio128.py`](../../models/compressor_ratio128.py) 将 prefill 和
decode 拆为两条 kernel 路径：

| 符号 | 类型 | 职责 |
|---|---|---|
| `compressor_ratio128_prefill_fwd` | `@pl.jit.inline` | 压缩完整 prefill block、保存 remainder、初始化 compressed cache |
| `compressor_ratio128_decode_fwd` | `@pl.jit.inline` | 更新一个 decode slot，并在 128-token 边界生成 compressed KV |
| `compressor_ratio128_prefill_test` | `@pl.jit` | Prefill 独立验收 wrapper |
| `compressor_ratio128_decode_test` | `@pl.jit` | Decode 独立验收 wrapper |
| `build_prefill_specs` | Host spec builder | 构造 prefill 输入、辅助 tensor 和 state output |
| `build_decode_specs` | Host spec builder | 构造单 token decode 输入、控制量和 state input/output |
| `golden_compressor_ratio128_forward` | PyTorch golden | 共用的 prefill/decode 参考状态机 |

完整 HCA kernel 在
[`models/attention_hca.py`](../../models/attention_hca.py) 中直接调用两个 inline
kernel。独立 wrapper 只用于单组件编译、执行和精度验收。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `Compressor(..., compress_ratio=128)` | `compressor_ratio128_*_fwd` | 语义等价：固定 shape 的 prefill/decode 实现 |
| `wkv(x.float())` | `linear_4096_to_512_fp32` | 直接调用：inline kernel |
| `wgate(x.float())` | `linear_4096_to_512_fp32` | 直接调用：inline kernel |
| `score + ape` 与 channel-wise softmax pooling | Prefill/decode softmax-pool stage | 融合内联 |
| `Compressor.norm` | Compressor 内部 512 维 RMSNorm | 融合内联 |
| `apply_rotary_emb` | Compressor 内部最后 64 维 RoPE | 融合内联 |
| `kv_state` / `score_state` | `comp_kv_state` / `comp_score_state` | 语义等价的 device state |
| `Attention.kv_cache[:, window_size:]` | `comp_cache` | 语义等价的独立 device state |
| `act_quant(..., inplace=True)` | 无 | 不支持或未执行：BF16 runtime 不执行 activation quantization |
| Ratio-4 overlap transform | `compressor_ratio4.py` | 本模块不支持；由独立实现覆盖 |
| HCA Attention 内调用 Compressor | `attention_hca_*_fwd` | 直接调用 |
| HCA Block prefill | `block_hca_topk_prefill_fwd` | 融合内联：经 HCA Attention 调用 |
| HCA selected decode pre-MoE | `hca_topk_selected_decode_pre_moe_fwd` | 融合内联：经 HCA Attention 调用 |

当前实现没有独立 PyPTO RMSNorm 或 RoPE kernel 调用边界，这两段计算直接嵌入
Compressor。数学定义和 table profile 分别参见 [RMSNorm](01_rmsnorm.md) 与
[RoPE](03_rope.md)。

## 数据接口

### 权重和公共输入

Prefill 与 decode 共用以下模型输入：

```text
x:        [1,S,4096], BF16
wkv_t:    [4096,512], BF16
wgate_t:  [4096,512], BF16
ape:      [128,512],  FP32
norm_w:   [512],      BF16
```

`wkv_t` 和 `wgate_t` 是 checkpoint linear weight 的转置 runtime layout。完整模型
通过 `DeepSeekV4WeightLoader.get_layer_compressor_ratio128()` 加载这四组参数，并将
其作为 fixed weight 常驻 NPU。

### Prefill 接口

```text
cos/sin:               [C,32],      FP32
block_count:           [1],         INT32
compressed:            [1,C,512],   BF16
kv_state_out:          [1,128,512], FP32
score_state_out:       [1,128,512], FP32
compressed_cache_out:  [1,32,512],  BF16
```

其中 `actual_blocks=floor(S/128)`，`C=max(1,actual_blocks)`。`block_count` 保存真实
block 数；当 `S<128` 时仍提供一行 cos/sin 和一行 `compressed` 以满足 kernel
shape，此时该行是占位输出，不表示有效 compressed KV。32 个 cache slot 来自当前
完整模型的 `max_seq_len=4096`。

Prefill 不接收旧 compressor state。它根据当前 prompt 重新生成三组 state output：
完整 block 写入 compressed cache，remainder 写入 staging state，其余 slot 初始化
为 0 或负无穷语义值。

### Decode 接口

```text
x:                     [1,1,4096],  BF16
kv_state:              [1,128,512], FP32
score_state:           [1,128,512], FP32
compressed_cache:      [1,32,512],  BF16
slot:                   [1],         INT32
cache_slot:             [1],         INT32
should_compress:        [1],         INT32
cos/sin:                [1,32],      FP32
compressed:             [1,1,512],   BF16
kv_state_out:           [1,128,512], FP32
score_state_out:        [1,128,512], FP32
compressed_cache_out:   [1,32,512],  BF16
```

控制量由当前绝对位置确定：

```text
slot = start_pos % 128
cache_slot = start_pos // 128
should_compress = int((start_pos + 1) % 128 == 0)
```

Decode runtime 只支持 `S=1`。不在 block 边界时，cos/sin 是零占位 tensor，
`compressed` 不包含新的有效 KV，compressed cache 保持不变；到达边界时 cos/sin
对应该 block 第一个 token 的位置 `start_pos + 1 - 128`。

### State 所有权

[`serving/state.py`](../../serving/state.py) 为每个 ratio-128 layer 声明三组逻辑
state：

| State | Shape | Dtype | 初始值 |
|---|---:|---|---|
| `comp_kv_state` | `[1,128,512]` | FP32 | 0 |
| `comp_score_state` | `[1,128,512]` | FP32 | `-torch.finfo(float32).max` |
| `comp_cache` | `[1,32,512]` | BF16 | 0 |

这些 state 由 serving runtime 持有，不属于 kernel-local scratch。
[`serving/device_state_store.py`](../../serving/device_state_store.py) 为每组 state 分配
current/next 两个持久 NPU buffer；kernel 读取 current、写入 next，Runner 在该层
执行结束后调用 `commit_state()` 交换二者。

内部 `kv_proj`、`score_proj`、`pooled` 和 `normed` 是单次 kernel 调用内创建的
scratch tensor，不跨 step 保存。

## 实现方式

### Prefill

Prefill 的主要阶段为：

1. 使用两个 `linear_4096_to_512_fp32` 分别生成 FP32 `kv_proj` 和
   `score_proj`；
2. 计算 `cutoff=floor(S/128)*128` 和 `remainder=S-cutoff`；
3. 将 remainder 的 projection 写入 state 的前 `remainder` 行，并给 score 加上
   `ape[:remainder]`；其余 KV state 清零，score state 写入 `NEG_INF`；
4. 将整个 32-row compressed cache 初始化为零；
5. 对每个完整 block，将 `score_proj + ape` 转置为按 channel 的 128 项序列，执行
   max-subtraction、`exp`、sum 和归一化，再对 `kv_proj` 加权求和；
6. 将 pooled FP32 结果 round-to-nearest 转为 BF16；
7. 在 FP32 中完成 512 维 RMSNorm，并将结果转回 BF16；
8. 复制前 448 个 channel，对最后 64 个 channel 执行 compressed-profile RoPE；
9. 将有效 `compressed` 行写入 compressed cache 的起始 slot。

Projection 使用 BF16 input/weight 和 FP32 matmul accumulation。Pooling 以 64 个
channel 为一个 `HEAD_CHUNK` 并行；RMSNorm 使用 8-row × 128-channel tile；RoPE
使用最多 16 行，并以 64 channel 复制不旋转的 prefix。最后一个 block 之外的
remainder 不参与当前 prefill pooling，而是留给后续 decode 补齐。

### Decode

Decode 每次处理一个 token：

1. 计算该 token 的 FP32 KV 和 gate projection；
2. 将 current state 复制到 next state，并用新 projection 更新 `slot`；gate score
   在写入前加上 `ape[slot]`；
3. 将 current compressed cache 复制到 next cache；
4. 若 `should_compress=0`，不执行 pooling、RMSNorm 和 RoPE，cache 内容不变；
5. 若 `should_compress=1`，对更新后的 128-row state 执行与 prefill 相同的
   channel-wise pooling、BF16 boundary、RMSNorm 和 RoPE；
6. 将新 compressed KV 写入 `compressed_cache_out[cache_slot]`。

完成一个 block 后 staging state 不单独清空。下一个 block 的 128 个 decode token
按 slot 0 到 127 逐行覆盖旧值；只有所有 slot 都更新后才再次触发 pooling，因此
不会把上一 block 的残留行用于新的有效 compressed KV。

### Host 辅助输入与跨层复用

`DeepSeekV4StatePlan` 根据 `seq_len` 或 `start_pos` 生成 `block_count`、slot、cache
slot、boundary flag 及 compressed RoPE slice。相同 step 的所有 ratio-128 layer
共用 host-side immutable auxiliary tensor cache。主 Attention RoPE 和 Compressor
RoPE 都使用 compressed profile，但 Compressor 只抽取每个完整 128-token block 的
起始位置。

## 实现差异与限制

当前实现与官方通用 Compressor 的主要差异如下：

- 当前文件只支持 ratio 128、`B=1`、hidden size 4096、head size 512 和 RoPE
  width 64，不是任意 ratio/head shape 的通用实现；
- Ratio 128 不使用 overlap；官方 ratio-4 overlap 行为由另一个固定实现负责；
- 官方 `wkv`/`wgate` module 以 FP32 parameter 执行，但 checkpoint 权重原始存储为
  BF16；当前 kernel 直接接收 BF16 runtime weight，并使用 FP32 matmul accumulation；
- 官方在 RoPE 后执行 activation quantization；当前 BF16 runtime 不执行
  `act_quant`，compressed cache 保持 BF16；
- 当前使用独立 `comp_cache`，而官方把 window KV 与 compressed KV 放在同一个
  Attention cache 的两个 slice；HCA 构造 KV pool 时再拼接两者；
- 官方 state buffer 由 `Compressor` module 持有；当前 state schema 与
  device-resident lifecycle 由 serving runtime 持有；
- 当前 kernel 以 FP32 最小有限值 `NEG_INF` 表达无效 score，而官方 state 使用
  IEEE `-inf`；两者在 softmax 中表达相同的 masked slot 语义；
- Prefill 的动态输出维至少保留一行，因此 `S<128` 时存在不代表有效数据的占位
  `compressed` row；有效行数始终由 `block_count` 决定；
- Decode 入口要求 `start_pos>0` 且 sequence length 为 1；`start_pos=0` 属于
  prefill 路径；
- 当前 compressed cache 只有 32 行，对应完整模型固定的 4096 position 上限。

## Golden 参考实现

`models/compressor_ratio128.py::golden_compressor_ratio128_forward` 是 prefill 和
decode 共用的 PyTorch 状态机。它从调用前的 BF16 `x` 和 BF16 transposed weight
snapshot 开始，以 FP32 `torch.matmul` 生成 projection，然后按官方公式完成
channel-wise softmax pooling。

Golden 在 pooling 后显式转为 BF16，再以 FP32 计算 RMSNorm，并将 normed tensor
转回 BF16。最后通过 `_apply_rope_golden` 对末尾 64 维执行 RoPE，输出 BF16
`compressed` 和 `compressed_cache_out`；projection staging state 保持 FP32。

Prefill golden 根据 `floor(S/128)` 计算有效 block，并校验 `block_count`。Decode
golden 会校验 `slot`、`cache_slot` 和 `should_compress` 是否与 `start_pos` 公式一致。
不触发压缩时只更新 staging state，`compressed` 保持零值，cache 保持原值。

Host 侧 [`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py)
将该 golden 与官方 `Compressor` 比较时关闭官方 activation quantization，以验证本
仓库 BF16 边界下的 projection、pooling、state、RMSNorm、RoPE 和 cache 语义。

## 精度验收标准

Standalone prefill/decode 的四个输出使用同一标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `compressed` | `1e-4` | `1/128` | `0.001` |
| `kv_state_out` | `1e-4` | `1/128` | `0.001` |
| `score_state_out` | `1e-4` | `1/128` | `0.001` |
| `compressed_cache_out` | `1e-4` | `1/128` | `0.001` |

每个元素的容差条件为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的元素超出该条件，实际数量阈值按 comparator 对 tensor 元素总数
取整。Actual output 中出现任何 NaN 或 Inf 都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上同时验证两个完整 prefill block 和 decode 边界：

```bash
python models/compressor_ratio128.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 256 \
  --decode-start-pos 127 \
  --case all
```

验证一个完整 block 加一个 remainder token：

```bash
python models/compressor_ratio128.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 129 \
  --case prefill
```

分别验证不足一个 block 的 prefill 和不触发压缩的 decode step：

```bash
python models/compressor_ratio128.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 127 \
  --case prefill

python models/compressor_ratio128.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 126 \
  --case decode
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

`models/compressor_ratio128.py::main()` 分别编译并执行 prefill/decode wrapper，比较
`compressed`、两组 FP32 state 和 BF16 compressed cache。该入口适合定位组件内部
的 projection、pooling、normalization、RoPE 或 state update 误差。

### Host 语义覆盖

[`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py) 覆盖：

- prefill 长度 64、128 和 300；
- decode 位置 126 和 127，即 block 边界前后；
- 从 126-token prefill 连续 decode 并跨越 ratio-128 边界；
- golden 与官方 `Compressor` 的 state、cache 和有效 compressed 输出；
- decode spec 拒绝 `start_pos=0`。

这些 host 测试验证状态机和官方语义，不编译或执行 NPU kernel，不能替代 standalone
实机验收。

### HCA 与 Block 集成

- [`test_attention_hca.py`](../../tests/models/test_attention_hca.py) 覆盖 Compressor
  与 HCA QKV、window cache、sparse attention 和 output projection 的组合，并验证
  prefill/decode 及连续跨 boundary 行为；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 ratio-128 HCA 与
  Hyper-Connection、RMSNorm 和 Top-K MoE 组成的完整 Block golden；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 验证 selected-expert
  decode 拆分前后与完整 HCA decode Block 的 compressor state/cache 一致性。

这些组合测试扩大了数据流覆盖范围，但不能替代 standalone kernel 对 Compressor
各输出的独立误差定位。

### Serving state 与权重生命周期

- [`test_state.py`](../../tests/serving/test_state.py) 验证 ratio-128 layer schema、
  prefill/decode 辅助输入、compressed RoPE profile、slot 和 boundary 公式；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证官方 checkpoint
  中 ratio-128 Compressor weight 的 runtime shape 和 layout；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 验证三组
  state 的 device allocation、双 buffer 和 score 初始值。

完整模型通过 `DeepSeekV4Runner` 在 ratio-128 layer 绑定 fixed weights、host auxiliary
inputs 及 current/next state。Prefill 使用完整 HCA Block kernel；decode 的
selected-expert pre-MoE kernel 内执行 HCA 和 Compressor，随后 Runner 提交 state 并
继续 selected expert 与 post-MoE 路径。权重、state、cache 和跨组件中间 tensor 在
执行期间保持在 NPU。
