# Ratio-4 Compressor

## 模块定位

Ratio-4 Compressor 是 Compressed Sparse Attention（CSA）的有状态压缩组件。每个
CSA layer 同时使用两套 ratio-4 Compressor：

- Attention Compressor 将 `attn_normed` 压缩为 512 维 KV，供 sparse attention
  读取；
- Indexer Compressor 将同一 `attn_normed` 压缩为 128 维 index KV，供 Indexer
  计算 compressed position score 和 Top-K 位置。

`attn_normed` 是 Block 内 Attention Hyper-Connection pre 和 `attn_norm` 的输出，
也是两套 Compressor 唯一的模型激活输入。Indexer scoring 还会接收 low-rank query
`qr`，但 `qr` 不进入 Indexer Compressor，因此不属于本文 Compressor 接口。

Prefill 中两套 Compressor 共用 `x`、block count 和 compressed RoPE slice，但使用
各自独立的 fixed weight 与 state/cache：

```text
attn_normed x [1,S,4096], BF16
+ shared-value auxiliary {attn_comp_cos/sin, idx_comp_cos/sin,
                          attn_comp_block_count, idx_comp_block_count}
  ├─ Attention fixed weights {attn_comp_wkv_t, attn_comp_wgate_t,
  │                           attn_comp_ape, attn_comp_norm_w}
  │    -> Attention Compressor
  │    ├-> compressed KV [1,floor(S/4),512]
  │    │     -> CSA sparse-attention KV pool
  │    └-> next {attn_comp_kv_state, attn_comp_score_state,
  │              attn_comp_cache}
  └─ Indexer fixed weights {idx_comp_wkv_t, idx_comp_wgate_t,
                            idx_comp_ape, idx_comp_norm_w}
       -> Indexer Compressor
       ├-> index KV [1,floor(S/4),128]
       │     -> idx_kv_cache -> Indexer score/Top-K
       └-> next {idx_comp_kv_state, idx_comp_score_state,
                 idx_kv_cache}
```

Decode 会额外读取六组 current state/cache，并由两套 Compressor 同步更新：

```text
attn_normed x [1,1,4096], BF16
+ Attention/Indexer fixed-weight groups
+ auxiliary {attn_comp_cos/sin, idx_comp_cos/sin,
             comp_slot, comp_cache_slot, comp_should_compress}
+ current {attn_comp_kv_state, attn_comp_score_state, attn_comp_cache,
           idx_comp_kv_state, idx_comp_score_state, idx_kv_cache}
  -> 更新两套 8-row overlap staging state
  -> boundary 时分别生成 512 维 Attention KV 和 128 维 index KV
  -> next {两套 staging state + attn_comp_cache + idx_kv_cache}
```

非 boundary decode step 仍会写入当前 overlap slot，但两套 compressed cache 保持
不变。Current/next state 由 serving runtime 持有和交换，不属于 kernel-local
scratch。

当前完整模型在 [`models/config.py`](../../models/config.py) 中 compression ratio 为
4 的 0-based layer 2、4、6，依次到 42 选择 CSA，共 21 层。Layer 2 与 hash-routing
MoE 组合，其余 ratio-4 layer 与 Top-K MoE routing 组合。

## 官方模型中的 Ratio-4 Compressor

[`official/model.py`](../../official/model.py) 使用通用 `Compressor` 类实现 ratio 4
和 ratio 128。`compress_ratio == 4` 时 `overlap=True`，并令
`coff = 1 + overlap = 2`，因此 projection、APE 和 decode staging state 都包含两组
channel：

| 官方实例 | `head_dim` | Projection width | `rotate` | 使用位置 |
|---|---:|---:|---|---|
| `Attention.compressor` | 512 | 1024 | `False` | 构造 CSA Attention 的 compressed KV |
| `Indexer.compressor` | 128 | 256 | `True` | 构造 Indexer scoring 使用的 compressed KV |

两套实例分别包含 `wkv`、`wgate`、`ape` 和 `norm` 参数：

| 参数 | Attention shape | Indexer shape | 官方 dtype/语义 |
|---|---:|---:|---|
| `wkv.weight` | `[1024,4096]` | `[256,4096]` | Module 中为 FP32，checkpoint 存储 BF16 |
| `wgate.weight` | `[1024,4096]` | `[256,4096]` | Module 中为 FP32，checkpoint 存储 BF16 |
| `ape` | `[4,1024]` | `[4,256]` | FP32 block-relative additive score |
| `norm.weight` | `[512]` | `[128]` | Module 中为 FP32，checkpoint 存储 BF16 |

### Overlap pooling

每个 4-token block 的 projection 最后一维由两组等宽 channel 组成。对于第 $j$ 个
block：

- 前半组从前一个 block 的四个 token 取得；
- 后半组从当前 block 的四个 token 取得；
- 两组在 8-token 维上按 channel 执行 gated softmax pooling；
- 第一个 block 没有前序 block，因此前四行以负无穷 score mask。

官方 `overlap_transform()` 将 `[B,blocks,4,2D]` 转换为
`[B,blocks,8,D]`，从而每消费 4 个新 token 生成一个 compressed KV，但除第一个
输出外，每个 compressed KV 的感受范围跨越相邻两个 4-token block。

Pooling 结果转回输入 dtype 后执行 RMSNorm，并只对最后 64 个 channel 应用
compressed-profile RoPE。Attention Compressor 随后执行 activation quantization；
Indexer Compressor 还会执行 `rotate_activation` 和 FP4 activation quantization。

### Prefill 与 decode state

官方 ratio-4 state shape 为 `[B,8,2D]`：

- 前四行保存上一个完整 block 的 projection；
- 后四行保存当前尚在收集的 block；
- `score_state` 在写入 projection 时已经加上对应的 `ape[slot]`。

Prefill 压缩所有完整 block，将最后一个完整 block 保存到前四行，并将 remainder
保存到后四行。Decode 每步写入 `4 + start_pos % 4`；到
`(start_pos + 1) % 4 == 0` 时聚合八行 overlap state，然后把刚完成的后四行移动到
前四行，作为下一个 compressed block 的历史窗口。

## PyPTO kernel 实现

[`models/compressor_ratio4.py`](../../models/compressor_ratio4.py) 为 Attention 和
Indexer 各提供 prefill/decode 两条 inline kernel：

| 目标 | Prefill inline kernel | Decode inline kernel |
|---|---|---|
| Attention | `compressor_ratio4_attention_prefill_fwd` | `compressor_ratio4_attention_decode_fwd` |
| Indexer | `compressor_ratio4_indexer_prefill_fwd` | `compressor_ratio4_indexer_decode_fwd` |

四个 inline kernel 分别具有同名的 `*_test` 顶层 `@pl.jit` wrapper。Standalone
spec builder 也按目标和阶段拆分：

| 目标 | Prefill builder | Decode builder |
|---|---|---|
| Attention | `build_attention_prefill_specs` | `build_attention_decode_specs` |
| Indexer | `build_indexer_prefill_specs` | `build_indexer_decode_specs` |

`golden_compressor_ratio4_forward` 实现共用的 overlap 状态机，
`golden_compressor_ratio4_attention_forward` 和
`golden_compressor_ratio4_indexer_forward` 只负责传入各自的 head/projection
尺寸。Prefill/decode golden wrapper 用于 standalone 验收入口。

在完整模型中：

- [`models/attention_csa.py`](../../models/attention_csa.py) 直接调用 Attention
  Compressor；
- [`models/indexer.py`](../../models/indexer.py) 在 Indexer 内直接调用 Indexer
  Compressor；
- CSA 再把 Indexer 返回的 compressed Top-K 与 sliding-window indices 组合。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `Compressor(..., compress_ratio=4, head_dim=512)` | `compressor_ratio4_attention_*_fwd` | 语义等价：固定 Attention shape |
| `Compressor(..., compress_ratio=4, head_dim=128, rotate=True)` | `compressor_ratio4_indexer_*_fwd` | 语义等价：不含 rotate/quant 的 BF16 路径 |
| `wkv(x.float())` / `wgate(x.float())`，Attention | `linear_4096_to_1024_fp32` | 直接调用：inline kernel |
| `wkv(x.float())` / `wgate(x.float())`，Indexer | `linear_4096_to_256_fp32` | 直接调用：inline kernel |
| `overlap_transform` | 8-row pool tile 与 state shift | 融合内联 |
| Channel-wise gated softmax pooling | Prefill/decode softmax-pool stage | 融合内联 |
| `Compressor.norm` | 内部 512/128 维 RMSNorm | 融合内联 |
| `apply_rotary_emb` | 内部最后 64 维 RoPE | 融合内联 |
| `Attention.compressor.kv_state/score_state` | `attn_comp_kv_state/score_state` | 语义等价的 device state |
| `Indexer.compressor.kv_state/score_state` | `idx_comp_kv_state/score_state` | 语义等价的 device state |
| Attention compressed cache slice | `attn_comp_cache` | 语义等价的独立 device state |
| `Indexer.kv_cache` | `idx_kv_cache` | 语义等价的独立 device state |
| Attention `act_quant` | 无 | 不支持或未执行：BF16 runtime 不量化 activation |
| Indexer `rotate_activation` / `fp4_act_quant` | 无 | 不支持或未执行：不做 Hadamard rotation 和 FP4 activation quantization |
| `Attention.forward` 的 ratio-4 分支 | `attention_csa_*_fwd` | 直接调用 Compressor inline kernel |
| CSA Block prefill/decode | `block_csa_*_fwd` / selected decode pre-MoE | 融合内联：经 CSA/Indexer 调用 |

RMSNorm 和 RoPE 数学定义分别参见 [RMSNorm](01_rmsnorm.md) 与
[RoPE](03_rope.md)。当前实现没有在 Compressor 内调用独立顶层 RMSNorm 或 RoPE
wrapper，而是将对应数学计算嵌入更大的 kernel。

## 数据接口

### 公共约束

两套 Compressor 都接收：

```text
x:            [1,S,4096], BF16
cos/sin:      [C,32],     FP32   # prefill
cos/sin:      [1,32],     FP32   # decode
block_count:  [1],        INT32  # 仅 prefill
```

其中 prefill `actual_blocks=floor(S/4)`、`C=max(1,actual_blocks)`。当 `S<4` 时，
cos/sin 与 `compressed` 仍保留一行占位 shape；有效 compressed 行数由
`block_count` 决定。Decode 只支持 `S=1`，并使用：

```text
slot = start_pos % 4
cache_slot = start_pos // 4
should_compress = int((start_pos + 1) % 4 == 0)
```

非 boundary step 使用零 cos/sin 占位，不产生新的有效 compressed KV。

### Attention Compressor

```text
wkv_t/wgate_t:         [4096,1024], BF16
ape:                    [4,1024],    FP32
norm_w:                 [512],       BF16
compressed:             [1,C,512],   BF16
kv_state(_out):         [1,8,1024],  FP32
score_state(_out):      [1,8,1024],  FP32
compressed_cache(_out): [1,1024,512], BF16
```

Prefill 只输出新 state/cache；decode 接收 current state/cache 并写出 next
state/cache。1024 个 cache row 对应当前 4096 position 上限除以 ratio 4。

### Indexer Compressor

```text
wkv_t/wgate_t:         [4096,256],   BF16
ape:                    [4,256],      FP32
norm_w:                 [128],        BF16
compressed:             [1,C,128],    BF16
kv_state(_out):         [1,8,256],    FP32
score_state(_out):      [1,8,256],    FP32
compressed_cache(_out): [1,1024,128], BF16
```

Standalone wrapper 使用 `compressed_cache` 名称；完整 Indexer 将同一 buffer 暴露为
`idx_kv_cache`，并直接用它计算 index score。Indexer Compressor 的
`compressed` 在完整 Indexer kernel 中只是内部 tensor，跨组件边界的是
`idx_kv_cache`、两组 state 和最终 `topk_idxs`。

### Serving state 所有权

[`serving/state.py`](../../serving/state.py) 为每个 ratio-4 layer 声明六组
Compressor 相关逻辑 state：

| State | Shape | Dtype | 初始值 |
|---|---:|---|---|
| `attn_comp_kv_state` | `[1,8,1024]` | FP32 | 0 |
| `attn_comp_score_state` | `[1,8,1024]` | FP32 | FP32 最小有限值 |
| `attn_comp_cache` | `[1,1024,512]` | BF16 | 0 |
| `idx_comp_kv_state` | `[1,8,256]` | FP32 | 0 |
| `idx_comp_score_state` | `[1,8,256]` | FP32 | FP32 最小有限值 |
| `idx_kv_cache` | `[1,1024,128]` | BF16 | 0 |

这些 state 由 serving runtime 持有，不属于 kernel-local scratch。
[`serving/device_state_store.py`](../../serving/device_state_store.py) 为每组 state 分配
current/next 两个持久 NPU buffer。Ratio-4 Block 执行结束后，Runner 通过
`commit_state()` 一次性提交 window cache 和上述六组 state。

两套 projection、pooled 和 normed tensor 都是单次调用内的 scratch，不跨 step
保存。

## 实现方式

### Prefill overlap state

Attention 与 Indexer 的 prefill 数据流相同，仅宽度不同：

1. 分别用两个 FP32-output linear kernel 生成 KV 和 gate projection；
2. 计算 `cutoff=floor(S/4)*4` 和 `remainder=S-cutoff`；
3. 若存在完整 block，把最后一个完整 block 的完整 `2D` projection 写入 state 前
   四行；
4. 把 remainder projection 写入 state 后四行的开头，并清零/mask 其余行；
5. 初始化对应的 1024-row compressed cache；
6. 对每个完整 block 构造 `[8,D]` pool tile：前四行取前一个 block projection 的
   前半 `D` channel，后四行取当前 block projection 的后半 `D` channel；
7. 每个 channel 独立在八行上执行 max-subtraction、exp 和 softmax weighted sum；
8. 将 pooled FP32 结果 round-to-nearest 转为 BF16，执行 FP32 RMSNorm，再转回
   BF16；
9. 复制未旋转 prefix，对最后 64 维执行 compressed-profile RoPE；
10. 把有效输出写入 compressed cache 起始行。

第一个 block 的前四行以 `NEG_INF` mask，因此只使用当前 block projection 的后半
channel。后续 block 同时使用前一个 block 的前半 channel 和当前 block 的后半
channel。

### Decode overlap state

Decode 每步处理一个 token：

1. 生成该 token 的 `2D` KV/gate projection；
2. 将 projection 写入 state row `4 + slot`，score 在写入前加 `ape[slot]`；
3. 非 boundary step 复制其他 state/cache，不执行 pooling；
4. Boundary step 从前四行 projection 的前半 channel 和后四行 projection 的后半
   channel 构造 `[8,D]` pool；
5. 完成 pooling、BF16 boundary、RMSNorm 和 RoPE，并写入
   `compressed_cache_out[cache_slot]`；
6. 将刚完成 block 的后四行复制到前四行，保留为下一个 overlap window。

Attention 与 Indexer 使用同一组 host control scalars，因此在同一 decode position
同时更新或同时保持各自的 compressed cache。到达 boundary 时，Compressor RoPE
位置为该当前 4-token block 的起始位置 `start_pos + 1 - 4`。

### Tiling 与数值边界

两套 projection 都使用 BF16 input/weight 和 FP32 matmul accumulation：Attention
调用 `linear_4096_to_1024_fp32`，Indexer 调用
`linear_4096_to_256_fp32`。Pooling 的 channel tile 为 64；RMSNorm 使用最多 8 行、
每次 128 channel；RoPE 使用最多 16 行并只旋转最后 64 维。

Attention 输出前 448 维不旋转；Indexer 输出前 64 维不旋转。无效 score 使用
FP32 最小有限值 `NEG_INF`，与官方 `-inf` state 表达相同的 softmax mask 语义。

### Host 辅助输入复用

`DeepSeekV4StatePlan` 为相同 `seq_len` 或 `start_pos` 的 ratio-4 layer 缓存
`block_count`、slot、cache slot、boundary flag 和 compressed RoPE slice。Attention
Compressor 与 Indexer Compressor 共用同一份 cos/sin host tensor；ratio 4 和 ratio
128 也共用 compressed RoPE profile，但按各自 ratio 提取位置。

## 实现差异与限制

当前实现与官方通用 ratio-4 Compressor 的主要差异如下：

- 当前只支持 `B=1`、ratio 4、hidden size 4096、Attention head 512 和 Indexer
  head 128，不是任意 shape 的 Compressor API；
- 官方把 Attention window KV 和 compressed KV 放在一个 cache 的两个 slice；当前
  将 `kv_cache` 与 `attn_comp_cache` 作为独立 device state，并在 CSA 中构造 KV
  pool；
- 官方 Indexer Compressor 使用 `rotate=True`；当前模型明确不执行
  `rotate_activation`，Indexer query 和 compressed KV 都不做 Hadamard rotation；
- 官方在 Indexer query/KV 上模拟 FP4 activation，并在 Attention compressed KV
  上执行 activation quantization；当前 BF16 runtime 不执行这些量化操作；
- 官方 `wkv`、`wgate` 和 `norm` module parameter 为 FP32，但 checkpoint 存储为
  BF16；当前接口直接使用 BF16 runtime weight，并在 projection/RMSNorm 中执行
  FP32 计算；
- 当前 state/cache 生命周期由 serving runtime 管理，并使用 current/next device
  buffer；官方 state 由各 PyTorch module 自身持有；
- Prefill 动态 compressed 维至少保留一行，`S<4` 时该行只是占位数据；
- Decode 入口要求 `start_pos>0` 且 sequence length 为 1；
- 两套 compressed cache 均固定为 1024 行，对应 4096 position 上限；
- 当前为单卡逻辑，不执行官方 Indexer score 的跨 rank `all_reduce`。

## Golden 参考实现

`models/compressor_ratio4.py::golden_compressor_ratio4_forward` 从 BF16 `x` 和 BF16
transposed weight snapshot 开始，以 FP32 `torch.matmul` 生成 projection。它显式
构造 `[B,blocks,8,D]` overlap pool，将前一 block 的前半 projection 和当前 block
的后半 projection 放入同一 softmax 维度。

Pooling 后 golden 先转为 BF16，再以 FP32 计算 RMSNorm 并转回 BF16；最后通过
`_apply_rope_golden` 旋转最后 64 维。Prefill golden 校验 `block_count` 并构造新的
state/cache；decode golden 校验 slot、cache slot 和 boundary flag，只有 boundary
step 才生成新 compressed KV。

Attention 和 Indexer golden 使用同一状态机，只传入不同的 `head_dim`、`proj_dim`
和 module name。Host 测试与官方模型比较时，官方 kernel stub 关闭 activation
quantization，并将 Indexer 的 `rotate_activation` 替换为 identity，因此比较目标是
本仓库定义的 BF16、无 Hadamard rotation 语义。

## 精度验收标准

四个 standalone case 的四组输出采用相同标准：

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

在 Ascend A2/A3 实机上执行 Attention/Indexer、prefill/decode 四个 case：

```bash
python models/compressor_ratio4.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 16 \
  --decode-start-pos 3 \
  --target all \
  --case all
```

使用一个完整 block 加 remainder 验证 prefill overlap state：

```bash
python models/compressor_ratio4.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --target all \
  --case prefill
```

分别验证不触发和触发压缩的 decode step：

```bash
python models/compressor_ratio4.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 2 \
  --target all \
  --case decode

python models/compressor_ratio4.py \
  --platform a2a3 \
  --device 0 \
  --decode-start-pos 3 \
  --target all \
  --case decode
```

`--target attention` 或 `--target indexer` 可单独选择其中一套 Compressor。如需仅
检查编译，可增加 `--compile-only`；如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`。

## 集成验证范围

### 独立 kernel 验收

`models/compressor_ratio4.py::main()` 可执行 Attention/Indexer 的 prefill/decode 四
种组合，并分别比较 compressed output、两组 FP32 state 和 BF16 cache。该入口适合
定位 overlap pooling、normalization、RoPE 或 state shift 的误差。

### Host Compressor 语义覆盖

[`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py) 分别对
Attention 和 Indexer 覆盖：

- prefill 长度 3、4、6、7、8、13、16 和 32；
- decode position 1、2、3 和 7；
- 无完整 block、恰好 boundary、多个 block 和 remainder；
- golden 与官方 Compressor 的 state、cache 和有效 compressed output。

这些 host 测试验证 overlap 状态机和官方 BF16 语义，不编译或执行 NPU kernel，
不能替代 standalone 实机验收。

### Indexer、CSA 与 Block 集成

- [`test_indexer.py`](../../tests/models/test_indexer.py) 覆盖 Indexer Compressor、
  index query、score 和 Top-K 的组合；
- [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 覆盖两套
  Compressor 与 sliding-window KV、Indexer、sparse attention 和 output projection
  的完整 CSA 组合；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 CSA 与 Hyper-Connection、
  RMSNorm、hash/Top-K MoE 组成的 Block golden；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 验证 selected-expert
  decode 拆分前后与完整 CSA decode Block 的 compressor state/cache 一致性。

这些组合测试验证上层数据流，但不能替代 standalone kernel 对两套 Compressor
输出的独立误差定位。

### Serving state 与权重生命周期

- [`test_state.py`](../../tests/serving/test_state.py) 验证六组 ratio-4 state、共享
  auxiliary tensor、compressed RoPE profile、slot 和 boundary 公式；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证 Attention
  Compressor 与 Indexer/Indexer Compressor checkpoint weight 的 runtime layout；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 验证
  ratio-4 state 的 device allocation、双 buffer 和 score 初始值。

完整模型由 `DeepSeekV4Runner` 为 ratio-4 layer 绑定两套 fixed weights 和六组
device-resident state。Prefill 在完整 CSA Block 内执行；decode 在 selected-expert
pre-MoE kernel 内执行 CSA、Indexer 与两套 Compressor，Runner 随后提交 state 并
继续 selected expert 和 post-MoE 路径。权重、state、cache 与跨组件中间 tensor 在
执行期间保持在 NPU。
