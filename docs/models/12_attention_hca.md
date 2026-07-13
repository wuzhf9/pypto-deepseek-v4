# Heavily Compressed Attention

## 模块定位

Heavily Compressed Attention（HCA）是当前模型中 `compress_ratio == 128` 的完整
Attention 执行路径。它同时保留最近 128 个 token 的 sliding window，并把每个完整
128-token block 压缩成一个 512 维 shared KV，使更早的上下文以最多 32 个 compressed
slots 参与 sparse attention。

```text
Attention normalized input x [1,S,4096], BF16
+ fixed Attention and Ratio-128 Compressor weights
+ compressed-profile position inputs
+ window + compressed indices [1,S,160]
  -> Attention QKV
       -> q [1,S,64,512]
       -> shared window kv [1,S,512]
  -> Ratio-128 Compressor
       -> compressed kv [1,C,512]
       -> compressor staging state/cache
  -> KV pool: [window kv/cache, compressed kv/cache]
  -> 160-candidate sparse attention + per-head sink
  -> inverse RoPE + grouped output projection
  -> out [1,S,4096], BF16
```

HCA 不使用 Indexer，也不对 compressed candidates 做 learned ranking。每个 token
直接使用所有已经完成且可见的 ratio-128 blocks；固定 32-wide compressed index
区域的剩余位置以 `-1` 填充。

根据 [`models/config.py`](../../models/config.py) 的当前 43 层主模型配置，ratio 128
出现在 0-based layer 3、5、7，依次到 41，共 20 层。这些层均位于三个 hash-routing
layers 之后，与 Top-K MoE routing 组合。`block_hca_topk_*` 名称中的 `topk` 指 MoE
routing，不表示 HCA Attention 内部执行 Indexer Top-K。

## 官方模型中的 Heavily Compressed Attention

[`official/model.py`](../../official/model.py) 通过通用 `Attention` 和 `Compressor`
表达 ratio-128 路径。`Attention.__init__()` 在该层创建：

- 一组普通 Q/KV 与 output projection weights；
- `Compressor(args, compress_ratio=128, head_dim=512)`；
- `[window_size + max_seq_len/128, head_dim]` 的组合 KV cache；
- compressed RoPE profile；
- 不创建 `Indexer`。

官方先用 `get_window_topk_idxs()` 生成 128-token window indices，再用
`get_compress_topk_idxs()` 生成所有可见 compressed block indices，并按最后一维
拼接。Prefill 中 compressed index offset 为当前 `seqlen`；decode 中 offset 为
`window_size=128`。

对于 zero-based query position $t$，可见 compressed block 数为：

$$
n_{comp}(t) = \left\lfloor\frac{t+1}{128}\right\rfloor
$$

因此 position 127 会在当前 step 首次包含 block 0。Decode 到达
`(start_pos+1) % 128 == 0` 的 boundary 时，Compressor 先生成新的 compressed KV，
当前 query 随后即可通过 compressed index 读取该 slot。

Prefill sparse attention 读取：

```text
[current prompt KV, current prompt compressed KV]
```

Decode 则先更新 window cache 和 Compressor state/cache，再读取：

```text
[128-row window cache, 32-row compressed cache]
```

Ratio-128 Compressor 的 gated pooling、RMSNorm、compressed RoPE 和 staging state
语义详见 [Ratio-128 Compressor](05_compressor_ratio128.md)。QKV、sparse aggregation
和输出投影分别参见 [Attention QKV](08_attention_qkv.md)、
[Sparse Attention](09_sparse_attn.md) 和 [Attention Out](10_attention_out.md)。

## PyPTO kernel 实现

[`models/attention_hca.py`](../../models/attention_hca.py) 提供两类 KV-pool helper 与
prefill/decode 组合 kernel：

| 符号 | 类型 | 职责 |
|---|---|---|
| `build_prefill_kv_pool` | `@pl.jit.inline` | 拼接 prompt KV 与当前 prefill compressed KV |
| `build_decode_kv_pool` | `@pl.jit.inline` | 拼接更新后的 window cache 与 32-row compressed cache |
| `attention_hca_prefill_fwd` | `@pl.jit.inline` | 组合 HCA prefill 的 QKV、两类 cache、KV pool、sparse attention 和 output projection |
| `attention_hca_decode_fwd` | `@pl.jit.inline` | 组合单 token HCA decode 与全部 state update |
| `attention_hca_prefill_test` | `@pl.jit` | Prefill standalone 验收 wrapper |
| `attention_hca_decode_test` | `@pl.jit` | Decode standalone 验收 wrapper |
| `golden_attention_hca_forward` | PyTorch golden | 共用的 prefill/decode 完整 HCA 参考实现 |
| `build_hca_prefill_specs` | Host spec builder | 构造指定 `seq_len` 的 HCA prefill tensors |
| `build_hca_decode_specs` | Host spec builder | 构造指定 `start_pos` 的单 token decode tensors |

组合 kernel 直接调用以下 inline 实现：

- `attention_qkv_fwd`；
- `update_prefill_window_cache` / `update_decode_window_cache`；
- `compressor_ratio128_prefill_fwd` / `compressor_ratio128_decode_fwd`；
- `sparse_attn_hca_fwd`；
- `attention_out_fwd`。

完整 Block 将其继续融合进 `block_hca_topk_{prefill,decode}_fwd`。完整 runtime 的
selected-expert decode 使用 `hca_topk_selected_decode_pre_moe_fwd` 在 pre-MoE 段内
执行 HCA 与全部 Attention state update。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `compress_ratio == 128` Attention dispatch | ratio-128 HCA Block case | 语义等价：当前主模型 20 层 |
| Q/KV projection 与 forward RoPE | `attention_qkv_fwd` | 直接调用：inline kernel |
| KV non-RoPE `act_quant` | 无 | 不支持或未执行：BF16 runtime 不量化 activation |
| Window cache update | `update_*_window_cache` | 直接调用：inline kernel |
| `Compressor(..., ratio=128)` | `compressor_ratio128_*_fwd` | 直接调用：inline kernel |
| 官方组合 cache 的 compressed slice | 独立 `comp_cache` state | 语义等价：构造 KV pool 时拼接 |
| `get_window_topk_idxs()` | host `build_window_topk_idxs()` | 语义等价：固定 128-wide |
| `get_compress_topk_idxs(128,...)` | host `build_compress_topk_idxs()` | 语义等价：固定 32-wide、以 `-1` 补齐 |
| `torch.cat([kv, kv_compress], dim=1)` | `build_prefill_kv_pool` | 语义等价 |
| Decode 读取组合 `kv_cache` | `build_decode_kv_pool` | 语义等价：从两组 device state 组装 |
| `sparse_attn(..., topk width 160)` | `sparse_attn_hca_fwd` | 直接调用：inline kernel |
| inverse RoPE 与 `wo_a`/`wo_b` | `attention_out_fwd` | 直接调用：inline kernel |
| 官方 mutable state/cache | device `current`/`next` state pairs | 语义等价：成功执行后交换 buffer |
| Ratio-4 Indexer Top-K | 无 | 不适用：HCA 使用规则化 compressed candidates |
| Tensor Parallel Attention | 无 | 不支持或未执行：当前为单卡完整 heads/groups |

## 数据接口

### Attention 公共输入和权重

Prefill 与 decode 共用：

```text
x:             [1,S,4096],     BF16
wq_a_t:        [4096,1024],    BF16
q_norm_w:      [1024],         BF16
wq_b_t:        [1024,32768],   BF16
wkv_t:         [4096,512],     BF16
kv_norm_w:     [512],          BF16
attn_sink:     [64],           FP32
topk_idxs:     [1,S,160],      INT32
wo_a_t:        [4096,8192],    BF16
wo_b_t:        [8192,4096],    BF16
cos/sin:       [S,32],         FP32
comp_wkv_t:    [4096,512],     BF16
comp_wgate_t:  [4096,512],     BF16
comp_ape:      [128,512],      FP32
comp_norm_w:   [512],          BF16
out:           [1,S,4096],     BF16
```

主 Attention 和 Compressor 都使用 compressed RoPE profile。主 `cos/sin` 对应当前
query positions；`comp_cos/sin` 对应每个完成的 128-token block 的起始位置。
`*_t` 是 checkpoint linear weight 的转置 runtime layout。

Kernel-local `qr`、`q`、`kv`、临时 `compressed`、`kv_pool` 和 `attn_o` 不跨 step
保存。`qr` 在 HCA 中只用于生成主 query，不会像 CSA 一样传给 Indexer。

### Prefill 接口

Prefill 满足 `start_pos=0`，并额外接收或写出：

```text
comp_cos/sin:          [C,32],       FP32
comp_block_count:      [1],          INT32
kv_pool:               [1,S+C,512],  BF16 caller-provided workspace
kv_cache_out:          [1,128,512],  BF16
comp_kv_state_out:     [1,128,512],  FP32
comp_score_state_out:  [1,128,512],  FP32
comp_cache_out:        [1,32,512],   BF16
```

其中 `blocks=floor(S/128)`、`C=max(1,blocks)`。当 `blocks=0` 时，`comp_cos/sin` 和
临时 `compressed` 保留一行 shape 占位，KV pool 也存在一行 compressed placeholder；
`topk_idxs` 不引用该行，因此它没有模型语义。`comp_block_count` 是有效 compressed
行数的唯一判定依据。

Prefill 不读取旧 state，而是从 prompt 重新生成 window cache、Compressor staging
state 和 compressed cache。Sparse attention 的 window indices 指向 KV pool 前 `S`
行；compressed indices 从 offset `S` 开始。

### Decode 接口

Decode 固定 `S=1`，额外输入和输出为：

```text
kv_cache:             [1,128,512], BF16
comp_kv_state:        [1,128,512], FP32
comp_score_state:     [1,128,512], FP32
comp_cache:           [1,32,512],  BF16
cache_pos:            [1],         INT32
comp_slot:            [1],         INT32
comp_cache_slot:      [1],         INT32
comp_should_compress: [1],         INT32
comp_cos/sin:         [1,32],      FP32

kv_cache_out:         [1,128,512], BF16
comp_kv_state_out:    [1,128,512], FP32
comp_score_state_out: [1,128,512], FP32
comp_cache_out:       [1,32,512],  BF16
out:                  [1,1,4096],  BF16
```

控制量满足：

```text
cache_pos = start_pos % 128
comp_slot = start_pos % 128
comp_cache_slot = start_pos // 128
comp_should_compress = int((start_pos + 1) % 128 == 0)
```

Decode KV pool 固定为 `[1,160,512]`：前 128 行是更新后的 window cache，后 32 行是
更新后的 compressed cache。Window indices 是 ring cache physical slots；compressed
indices 从 offset 128 开始。

### State 所有权和生命周期

[`serving/state.py`](../../serving/state.py) 为每个 HCA layer 声明四组 state：

| State | Input name | Output name | Shape | Dtype | 初始值 |
|---|---|---|---:|---|---|
| `kv_cache` | `kv_cache` | `kv_cache_out` | `[1,128,512]` | BF16 | 0 |
| `comp_kv_state` | `comp_kv_state` | `comp_kv_state_out` | `[1,128,512]` | FP32 | 0 |
| `comp_score_state` | `comp_score_state` | `comp_score_state_out` | `[1,128,512]` | FP32 | FP32 最小有限值 |
| `comp_cache` | `comp_cache` | `comp_cache_out` | `[1,32,512]` | BF16 | 0 |

[`serving/device_state_store.py`](../../serving/device_state_store.py) 为每组逻辑 state
分配 current/next 两个持久 NPU buffer。Prefill 只写 next outputs；decode 读取 current
并写 next。完整 Block 或 selected decode pre-MoE kernel 成功返回后，Runner 调用
`commit_state()` 校验并交换四组 buffer，使下一 step 读取本次更新结果。

Prefill standalone 从空 output state 开始；decode standalone 使用独立构造的随机
current state/cache，因此不是同一次 standalone prefill 的延续。Serving state 则在
prefill 后连续跨 decode steps 复用，并始终驻留 NPU。

## 实现方式

### Prefill 路径

`attention_hca_prefill_fwd` 按以下顺序执行：

1. `attention_qkv_fwd` 生成 compressed-profile query 和 shared KV；
2. `update_prefill_window_cache` 把 prompt 最近 128 个 KV 写入 ring cache output；
3. `compressor_ratio128_prefill_fwd` 压缩所有完整 128-token blocks，把 remainder
   写入 FP32 staging state，并初始化 32-row compressed cache；
4. `build_prefill_kv_pool` 依次复制 prompt KV 和 compressed rows；
5. `sparse_attn_hca_fwd` 按 128 个 window slots 与 32 个 compressed slots 聚合；
6. `attention_out_fwd` 执行 inverse RoPE 和两级 output projection。

Prefill 对 token `t` 的 window 部分选择 `[max(0,t-127),...,t]`；compressed 部分选择
`[0,...,floor((t+1)/128)-1] + S`。两部分都补齐到固定宽度。Window candidate 保留
局部细节，compressed candidate 提供 block-level 表示；二者可能覆盖相同原始 token
范围，但对应不同的 KV representation。

`build_prefill_kv_pool` 使用动态 `S`、`C`，在 CORE_GROUP scope 中把两段 512-channel
rows 写入调用方提供的 buffer。该 buffer 只服务当前 HCA kernel，不进入持久 state。

### Decode 路径

`attention_hca_decode_fwd` 的顺序为：

1. 为当前 token 生成 `q` 和 `kv`；
2. 更新 window current→next cache，并写入 `cache_pos`；
3. 更新 Compressor current→next staging state；
4. 非 boundary step 复制 compressed cache；boundary step 生成新 compressed KV 并
   写入 `comp_cache_slot`；
5. 以 16-row tile 并行构造 `[updated window cache, updated compressed cache]`；
6. 用 160-wide indices 执行 sparse attention，再执行 Attention Out。

在 boundary step，`comp_should_compress=1`，新 compressed slot 在构造 KV pool 前已经
写入，因此当前 token 可以立即读取它。非 boundary step 的 `comp_cos/sin` 是零占位，
Compressor 只更新一个 staging row，compressed cache 保持不变。

Compressed index 的有效前缀长度为 `floor((start_pos+1)/128)`；其余 32-wide slots
为 `-1`。Window index 与 [Sliding Window Attention](11_attention_swa.md) 使用相同的
128-row ring-slot 语义。

### Host auxiliary cache

`DeepSeekV4StatePlan` 为 HCA 构造并缓存：

- compressed-profile 主 Attention cos/sin；
- fixed-width window + compressed indices；
- prefill `comp_block_count` 与每个完整 block 的 Compressor RoPE；
- decode cache/Compressor slots、boundary flag 和 boundary RoPE。

相同 prefill length 或 decode position 的不可变 HCA auxiliary inputs 跨 ratio-128
layers 复用。Ratio-4 与 ratio-128 主 Attention 还共享同一 compressed RoPE profile
slice；mutable state/cache 不属于 host auxiliary cache。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、64 query heads、shared KV dim 512、window size
  128、compression ratio 128 和最大位置 4096，不是通用 compressed Attention API；
- 当前主模型固定有 20 个 HCA layers，并与 Top-K MoE routing 组合；MoE routing 不
  参与 HCA candidate 选择；
- Compressed candidate 上限固定为 `4096/128=32`，总 sparse width 固定为 160；
- HCA 不执行 learned Indexer Top-K，而是包含全部可见 compressed blocks；
- 当前使用 compressed RoPE profile，ratio 4 与 ratio 128 的主 Attention profile
  相同；
- 官方在 ordinary/compressed KV 的 non-RoPE channels 上执行 activation quantization，
  当前 BF16 runtime 不执行该操作；
- 官方将 window 与 compressed KV 放在同一 cache；当前拆成独立 state，并在 kernel
  内显式构造 KV pool；
- 当前使用 current/next state buffers，不原地修改 state input；
- 当前不实现 Tensor Parallel head/group shard 或集合通信；
- Prefill 必须从 `start_pos=0` 开始，decode 必须满足 `start_pos>0` 且 `S=1`；
- Kernel 不包含 Attention 前后的 Hyper-Connection 或 Attention RMSNorm。

## Golden 参考实现

`models/attention_hca.py::golden_attention_hca_forward` 从调用前的 BF16 activation、
BF16 transposed weights、FP32 sink/Compressor inputs、INT32 indices 和可选 current
state snapshot 开始，按组合 kernel 的 BF16 boundary 执行：

1. Q/KV projection、normalization 和 compressed-profile forward RoPE；
2. window ring-cache construction/update；
3. `golden_compressor_ratio128_forward` 的 projection、gated pooling、state/cache、
   RMSNorm 和 compressed RoPE；
4. prefill 或 decode KV pool construction；
5. `golden_sparse_attn` 的 FP32 score、sink softmax 和 value aggregation；
6. inverse RoPE、grouped `wo_a` 与 `wo_b` projection。

Golden 写出 `kv_cache_out`、`comp_kv_state_out`、`comp_score_state_out`、
`comp_cache_out` 和最终 `out`。Prefill spec 中的 `kv_pool` 只用于承载中间拼接结果，
不作为数值 comparator output。Decode golden 会校验 `S=1`、`cache_pos` 以及
Compressor slot/boundary 公式；官方 activation quantization 在当前 BF16 golden 中
明确移除。

`golden_attention_hca_prefill` 和 `golden_attention_hca_decode` 是共用 golden 的两类
wrapper。

## 精度验收标准

State/cache 与最终输出使用两组标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `kv_cache_out` | `1e-4` | `1/128` | `0.001` |
| `comp_kv_state_out` | `1e-4` | `1/128` | `0.001` |
| `comp_score_state_out` | `1e-4` | `1/128` | `0.001` |
| `comp_cache_out` | `1e-4` | `1/128` | `0.001` |
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

在 Ascend A2/A3 实机上同时验证两个完整 compressed blocks、window wrap 和 decode
compression boundary：

```bash
python models/attention_hca.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 256 \
  --decode-start-pos 127 \
  --case all
```

验证一个完整 block 加 remainder，以及不触发 compression 的 decode step：

```bash
python models/attention_hca.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 129 \
  --decode-start-pos 126 \
  --case all
```

可通过 `--case prefill` 或 `--case decode` 单独选择路径。如需仅检查编译，可增加
`--compile-only`；如需启用 L2 swimlane，可增加 `--enable-l2-swimlane`，该参数会
直接传入 PyPTO `RunConfig`。

Host-side 完整 HCA golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_attention_hca.py
```

## 集成验证范围

### 独立 kernel 验收

`models/attention_hca.py::main()` 分别编译和执行 `attention_hca_prefill_test` 与
`attention_hca_decode_test`，比较四组 state/cache output 和最终 `out`。两个 case
独立构造 state，decode case 不是同一次 standalone prefill 的延续。

[`test_attention_hca.py`](../../tests/models/test_attention_hca.py) 使用缩小 shape，关闭
官方 activation quantization，并用当前 PyTorch sparse-attention golden 替换外部
callable；它逐元素比较完整 HCA golden 与官方 `Attention.forward()` 的
prefill/decode output、window cache、Compressor state/cache，并覆盖连续 decode
跨 ratio-128 boundary。

### 组件与 Block 集成

- [`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py) 独立覆盖
  Ratio-128 Compressor 的 pooling、remainder、boundary 和 state machine；
- [`test_sparse_attn.py`](../../tests/models/test_sparse_attn.py) 覆盖 HCA fixed-width
  window/compressed indices 与官方 helper 的一致性；
- [`test_attention_out.py`](../../tests/models/test_attention_out.py) 覆盖 inverse RoPE
  与 output weight layout；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 HCA + Top-K MoE 的完整
  prefill/decode Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 HCA selected-expert
  decode pre-MoE/post-MoE 拆分及与完整 Block 的 state/output 对应关系。

### Serving state 与权重生命周期

- [`test_state.py`](../../tests/serving/test_state.py) 覆盖 ratio-128 layer selection、
  四组 state schema、compressed RoPE、160-wide indices、slot/boundary 控制量和 host
  auxiliary cache；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 覆盖四组
  state 的 current/next device allocation、初始值校验和重复 commit buffer 复用；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 HCA Attention
  与 Ratio-128 Compressor fixed weights 的 checkpoint mapping、runtime shape 和 layout。

完整模型 prefill 在 HCA Block kernel 后提交 window 与 Compressor state。Decode 的
selected-expert 路径在 pre-MoE kernel 返回后立即提交四组 state，再执行 routed/shared
expert 的 post-MoE kernel。Fixed weights、state/cache 与跨组件中间 tensor 都保持
device resident，不在 decode steps 之间回传 host。
