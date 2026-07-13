# Sliding Window Attention

## 模块定位

Sliding Window Attention（SWA）是当前模型中 `compress_ratio == 0` 的完整 Attention
执行路径。它组合 Q/KV projection、normal RoPE、128-token sliding-window sparse
attention、inverse RoPE 和 grouped output projection，并维护供后续 decode step 使用的
shared KV ring cache。

```text
Attention normalized input x [1,S,4096], BF16
+ fixed Attention weights
+ normal-profile position inputs {cos, sin}
+ causal window indices [1,S,128]
  -> Attention QKV
       -> q [1,S,64,512]
       -> shared kv [1,S,512]
  -> update 128-row window KV cache
  -> 128-candidate sparse attention + per-head sink
  -> inverse RoPE + grouped output projection
  -> out [1,S,4096], BF16
```

SWA 不使用 Compressor 或 Indexer。根据
[`models/config.py`](../../models/config.py) 的当前 43 层主模型配置，ratio 0 只出现
在 layer 0 和 layer 1；这两层同时位于前三个 hash-routing layers，但 Attention 类型
和 MoE routing 类型是两个独立配置维度，SWA 的数学语义本身不依赖 hash routing。

该路径位于 Block 的 Attention Hyper-Connection pre 与 post 之间。输入 `x` 已完成
Attention RMSNorm，输出 `out` 仍是 4096 维 sublayer result；Hyper-Connection 和 MoE
不属于本模块。

## 官方模型中的 Sliding Window Attention

[`official/model.py`](../../official/model.py) 的 `Attention` 在
`compress_ratio == 0` 时执行纯 sliding-window 路径：

- 不实例化 `Compressor` 或 `Indexer`；
- KV cache 大小为 `[max_batch_size, window_size, head_dim]`；
- 使用 base `rope_theta`，并将 `original_seq_len` 设为 0，不启用 compressed YaRN
  profile；
- 使用 `get_window_topk_idxs()` 构造 causal window 或 decode ring-slot indices；
- sparse attention 只读取 window KV，不拼接 compressed KV。

Prefill（`start_pos == 0`）中，官方先生成当前 prompt 的 `q` 和 shared `kv`，再把最近
最多 128 个 KV 写入 cache。Sparse attention 仍直接读取按 prompt 顺序排列的当前
`kv`，而不是 ring cache，因此每个 token 的 causal window index 都位于 `[0,S)`。

Decode（`start_pos > 0`）中，官方将当前 token KV 写入：

```text
kv_cache[:, start_pos % 128] = kv.squeeze(1)
```

随后 sparse attention 读取完整 128-row cache。`get_window_topk_idxs()` 返回物理 cache
slot：cache 尚未填满时使用 `[0,start_pos]` 并以 `-1` 补齐；填满后按“最旧 token 到
当前 token”的顺序返回环形 slot。

官方完整计算还包括 [Attention QKV](08_attention_qkv.md)、
[Sparse Attention](09_sparse_attn.md) 和 [Attention Out](10_attention_out.md) 中描述的
projection、normalization、RoPE、attention sink 与两级输出投影。

## PyPTO kernel 实现

[`models/attention_swa.py`](../../models/attention_swa.py) 提供 prefill/decode 两条组合
kernel 和两类 cache update helper：

| 符号 | 类型 | 职责 |
|---|---|---|
| `update_prefill_window_cache` | `@pl.jit.inline` | 将 prompt 最近 128 个 shared KV 按 ring slot 写入 cache |
| `update_decode_window_cache` | `@pl.jit.inline` | 复制 current cache，并在 `cache_pos` 写入当前 token KV |
| `attention_swa_prefill_fwd` | `@pl.jit.inline` | 组合 SWA prefill QKV、cache update、sparse attention 和 output projection |
| `attention_swa_decode_fwd` | `@pl.jit.inline` | 组合单 token decode、cache update、sparse attention 和 output projection |
| `attention_swa_prefill_test` | `@pl.jit` | Prefill standalone 验收 wrapper |
| `attention_swa_decode_test` | `@pl.jit` | Decode standalone 验收 wrapper |
| `golden_attention_swa_forward` | PyTorch golden | 共用的 prefill/decode 完整 SWA 参考实现 |
| `build_swa_prefill_specs` | Host spec builder | 构造指定 `seq_len` 的 prefill tensors |
| `build_swa_decode_specs` | Host spec builder | 构造指定 `start_pos` 的单 token decode tensors |

组合 kernel 直接调用：

- `models/attention_qkv.py::attention_qkv_fwd`；
- `models/sparse_attn.py::sparse_attn_swa_fwd`；
- `models/attention_out.py::attention_out_fwd`。

当前完整 Block 进一步把 SWA inline kernel 融合进
`block_swa_hash_{prefill,decode}_fwd`；selected-expert decode 使用
`swa_hash_selected_decode_pre_moe_fwd` 在 pre-MoE 段内调用 SWA decode。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `compress_ratio == 0` Attention dispatch | ratio-0 SWA Block case | 语义等价：当前主模型 layer 0、1 |
| Q/KV projection、RMSNorm、forward RoPE | `attention_qkv_fwd` | 直接调用：inline kernel |
| KV non-RoPE `act_quant` | 无 | 不支持或未执行：BF16 runtime 不量化 activation |
| `get_window_topk_idxs()` | host `build_window_topk_idxs()` | 语义等价：固定宽度 128、以 `-1` 补齐 |
| Prefill cache write | `update_prefill_window_cache` | 语义等价 |
| Decode `kv_cache[:, start_pos % win] = kv` | `update_decode_window_cache` | 语义等价：current/next buffer 接口 |
| `sparse_attn(q, kv, sink, topk, scale)` | `sparse_attn_swa_fwd` | 直接调用：inline kernel |
| inverse RoPE 和 `wo_a`/`wo_b` | `attention_out_fwd` | 直接调用：inline kernel |
| 官方单个 mutable `kv_cache` buffer | device `current`/`next` state pair | 语义等价：成功执行后交换 buffer |
| Compressor / compressed KV | 无 | 不适用：ratio 0 路径不执行 |
| Indexer / compressed Top-K | 无 | 不适用：ratio 0 路径不执行 |
| Tensor Parallel Attention | 无 | 不支持或未执行：当前为单卡完整 heads/groups |

## 数据接口

### 公共输入和权重

Prefill 与 decode 共用：

```text
x:          [1,S,4096],     BF16
wq_a_t:     [4096,1024],    BF16
q_norm_w:   [1024],         BF16
wq_b_t:     [1024,32768],   BF16
wkv_t:      [4096,512],     BF16
kv_norm_w:  [512],          BF16
attn_sink:  [64],           FP32
topk_idxs:  [1,S,128],      INT32
wo_a_t:     [4096,8192],    BF16
wo_b_t:     [8192,4096],    BF16
cos/sin:    [S,32],         FP32
out:        [1,S,4096],     BF16
kv_cache_out: [1,128,512],  BF16
```

`*_t` 是 checkpoint linear weight 的转置 runtime layout。`cos/sin` 使用 normal RoPE
profile，并且已经对应当前 token 的实际 position。`topk_idxs` 中非负值是当前 sparse
attention KV input 的物理行号，`-1` 表示无效候选。

Kernel 内部的 `qr [1,S,1024]`、`q [1,S,64,512]`、`kv [1,S,512]` 和
`attn_o [1,S,64,512]` 都是单次调用 scratch。SWA 不消费 `qr` 的跨组件输出，因为
ratio-0 layer 没有 Indexer。

### Prefill 接口

Prefill 满足 `start_pos=0`，`S` 是动态 prompt length。它没有 `kv_cache` input，只把
`kv_cache_out` 绑定到 runtime 的 next state buffer。Sparse attention 使用当前
prompt `kv [1,S,512]`，而 cache output 只为后续 decode 准备。

对于 `S <= 128`，有效 cache slot `0:S` 保存 prompt KV，其余 slot 不参与后续
attention，内容没有语义要求。对于 `S > 128`，cache 保存最后 128 个 prompt KV，
并保证这 128 个保留位置中的绝对位置 `p` 位于物理 slot `p % 128`。

### Decode 接口

Decode 固定 `S=1`，额外输入为：

```text
kv_cache: [1,128,512], BF16
cache_pos: [1],        INT32
```

其中 `cache_pos = start_pos % 128`。Kernel 先把 current `kv_cache` 完整复制到
`kv_cache_out`，再用当前 token KV 覆盖 `cache_pos`，因此当前 token 在同一步 sparse
attention 中立即可见。`topk_idxs [1,1,128]` 也基于更新后的 physical slot layout。

### State 所有权和生命周期

[`serving/state.py`](../../serving/state.py) 为每个 layer 声明一个 window state：

| State | Input name | Output name | Shape | Dtype | 初始值 |
|---|---|---|---:|---|---:|
| `kv_cache` | `kv_cache` | `kv_cache_out` | `[1,128,512]` | BF16 | 0 |

[`serving/device_state_store.py`](../../serving/device_state_store.py) 为该逻辑 state 分配
两个持久 NPU buffer：`current` 作为 decode input，`next` 绑定 `kv_cache_out`。Prefill
或 decode kernel 成功返回后，Runner 调用 `commit_state()` 交换两者；下一 decode
step 读取刚刚生成的 cache，而旧 current buffer 成为下一次输出目标。两个 buffer
跨 step 复用，不执行逐步 host round trip。

Prefill 独立验收的 `kv_cache_out` 从零初始化；decode 独立验收则使用随机
`kv_cache`，用于单独验证复制、slot overwrite 和 attention 读取。Serving 中的
cache 是每层独立的 device-resident state，生命周期由 runtime 而不是 standalone
spec builder 管理。

## 实现方式

### Prefill 路径

`attention_swa_prefill_fwd` 按以下顺序执行：

1. `attention_qkv_fwd` 生成 normal-RoPE query 和 shared KV；
2. `update_prefill_window_cache` 更新后续 decode 所需的 128-row cache；
3. `sparse_attn_swa_fwd` 直接以当前 prompt KV 为 KV input，按每个 token 的 causal
   sliding-window indices 聚合最多 128 个位置；
4. `attention_out_fwd` 对 sparse result 执行 inverse RoPE 和两级输出投影。

当 `S <= 128` 时，cache update 顺序复制 `kv[t] -> cache[t]`。当 `S > 128` 时，设
`cutoff = S % 128`，将最后 128 个 KV 分成 `128-cutoff` 与 `cutoff` 两段，分别写入
`cache[cutoff:128]` 和 `cache[0:cutoff]`。这与逐 token 执行
`cache[position % 128] = kv` 后的 ring layout 相同。

Prefill window indices 对 zero-based token `t` 选择：

```text
[max(0, t-127), ..., t]
```

固定宽度不足 128 的尾部以 `-1` 填充。因果性完全由这些 indices 表达，Sparse
Attention 内部不额外生成 causal mask。

### Decode 路径

`attention_swa_decode_fwd` 要求单 token 输入，顺序为：

1. 生成当前 token 的 `q` 和 `kv`；
2. 复制 current cache 到 next cache；
3. 将当前 KV 写入 `cache_pos=start_pos % 128`；
4. 以更新后的 next cache 和物理 slot indices 执行 sparse attention；
5. 执行 Attention Out，返回 `kv_cache_out` 和 4096 维 `out`。

当 `start_pos < 127` 时，有效 slots 为 `[0,...,start_pos]`。当
`start_pos >= 127` 时，设 `pos=start_pos % 128`，indices 为：

```text
[pos+1, ..., 127, 0, ..., pos]
```

因此最多 128 个候选始终按时间从旧到新排列，最后一个 slot 对应当前 token。Slot
顺序不会改变 attention 数学结果，但保证 host helper 与官方索引语义一致。

### Host auxiliary cache

`DeepSeekV4StatePlan` 在 host 侧一次生成最大长度 4096 的 normal/compressed RoPE
table。SWA 只使用 normal table，并按当前 `seq_len` 或 `start_pos` 缓存：

- prefill 的 `[S,32]` cos/sin 和 `[1,S,128]` window indices；
- decode 的 `[1,32]` cos/sin、`cache_pos` 和 `[1,1,128]` window indices。

相同 prefill length 或 decode position 的不可变 window indices 会在不同 Attention
类型和 layer 之间复用；mutable KV cache 不属于 host auxiliary cache。

## 实现差异与限制

- 当前只支持 `B=1`、hidden 4096、64 query heads、shared KV dim 512、window size
  128 和 output groups 8，不是任意 shape/window 的 SWA API；
- 当前主模型只有 layer 0、1 使用 SWA；MTP layer 不在当前 Runner 执行范围内；
- 当前完整 Block 只存在 SWA + hash-routing 的配置组合，但 hash routing 属于 MoE，
  不是 SWA kernel 的输入或数学限制；
- 当前使用 normal RoPE profile，最大运行位置为 4096；
- 官方在 shared KV 的 non-RoPE channels 上执行 activation quantization simulation，
  当前 BF16 路径不执行该操作；
- 当前不实现 Tensor Parallel 的 head/group shard 或集合通信；
- Prefill 必须从 `start_pos=0` 开始，decode 必须满足 `start_pos>0` 且 `S=1`；
- Sparse Attention 候选宽度固定为 128，padding sentinel 固定为 `-1`；
- Cache update 使用 current/next device buffers，不原地修改输入 buffer；
- Kernel 不包含 Attention 前后的 Hyper-Connection 或 Attention RMSNorm。

## Golden 参考实现

`models/attention_swa.py::golden_attention_swa_forward` 从 BF16 `x`、BF16 transposed
weights、FP32 `attn_sink`、INT32 window indices、FP32 cos/sin 和可选 decode cache
snapshot 开始，按 kernel 相同的 BF16 boundary 执行完整 SWA：

1. Q/KV projection、learned RMSNorm、query per-head RMS rescale 和 forward RoPE；
2. prefill ring-cache construction，或 decode cache clone + slot overwrite；
3. `golden_sparse_attn` 的 FP32 score、sink softmax 和 value accumulation；
4. inverse RoPE、grouped `wo_a` 和 `wo_b` projection。

Golden 写出 `kv_cache_out` 和 `out`。Decode golden 额外检查 `S=1` 且
`cache_pos == start_pos % 128`；prefill sparse attention 与官方相同，读取 prompt KV
而不是 cache output。官方 KV activation quantization 在当前 BF16 golden 中明确移除。

`golden_attention_swa_prefill` 和 `golden_attention_swa_decode` 是该共用实现的
prefill/decode wrapper。

## 精度验收标准

两类 standalone 输出使用不同标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `kv_cache_out` | `1e-4` | `1/128` | `0.001` |
| `out` | `1e-3` | `2/128` | `0.005` |

逐元素容差分别为：

```text
kv_cache_out:
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)

out:
abs(actual - expected) <= 1e-3 + (2/128) * abs(expected)
```

`kv_cache_out` 允许最多 0.1% 元素超出容差，`out` 允许最多 0.5%。数量阈值按
comparator 对元素总数取整；actual output 中出现任何 NaN 或 Inf 都会直接判为
不合法。

## 验收方法

在 Ascend A2/A3 实机上同时验证跨 window prefill 和已填满 cache 的 decode：

```bash
python models/attention_swa.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 129 \
  --decode-start-pos 129 \
  --case all
```

使用短序列验证 `-1` padding、部分填充 cache 和较早 decode position：

```bash
python models/attention_swa.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13 \
  --decode-start-pos 13 \
  --case all
```

可通过 `--case prefill` 或 `--case decode` 单独选择路径。如需仅检查编译，可增加
`--compile-only`；如需启用 L2 swimlane，可增加 `--enable-l2-swimlane`，该参数会
直接传入 PyPTO `RunConfig`。

Host-side 完整 Attention golden 与官方模型的比较可运行：

```bash
pytest -q tests/models/test_attention_swa.py
```

## 集成验证范围

### 独立 kernel 验收

`models/attention_swa.py::main()` 分别编译和执行 `attention_swa_prefill_test` 与
`attention_swa_decode_test`，比较 `kv_cache_out` 和 `out`。两个 case 独立构造
state，decode case 不是同一次 standalone prefill 的延续。

[`test_attention_swa.py`](../../tests/models/test_attention_swa.py) 使用缩小 shape，将
外部 sparse-attention callable 替换为当前 PyTorch golden，并将 activation
quantization 设为 identity；它在多个 prefill length 和 decode position 上逐元素
精确比较当前完整 SWA golden、官方 `Attention.forward()` 输出和 window cache。

### 组件与 Block 集成

- [`test_sparse_attn.py`](../../tests/models/test_sparse_attn.py) 覆盖 window index helper
  和 128-candidate sparse-attention 输入语义；
- [`test_attention_out.py`](../../tests/models/test_attention_out.py) 覆盖 inverse RoPE
  与 output weight layout；
- [`test_block.py`](../../tests/models/test_block.py) 覆盖 SWA + hash MoE 的完整
  prefill/decode Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 SWA selected-expert
  decode 的 pre-MoE/post-MoE 拆分及与完整 Block 的对应关系。

### Serving state 集成

- [`test_state.py`](../../tests/serving/test_state.py) 覆盖 ratio-0 layer selection、
  `[1,128,512]` state schema、normal RoPE、window indices、`cache_pos` 和 host aux cache；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 覆盖
  current/next device buffer 分配、初始化、原子校验和重复 commit 后的 buffer 复用；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 Attention fixed
  weights 的 checkpoint mapping、runtime dtype 和转置 layout。

完整模型 prefill 在每层 Block kernel 后提交 `kv_cache_out`。Decode 的 selected-expert
路径在 pre-MoE kernel 返回后立即提交 cache，再执行 routed/shared expert 的 post-MoE
kernel。KV cache、fixed weights 和中间 tensor 都保持 device resident，不需要在
decode step 之间回传 host。
