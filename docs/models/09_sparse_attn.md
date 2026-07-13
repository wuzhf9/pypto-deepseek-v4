# Sparse Attention

## 模块定位

Sparse Attention 是 Sliding Window Attention（SWA）、CSA 和 HCA 三类 Attention
共用的核心聚合组件。它接收已经完成 projection、normalization 和 RoPE 的 64-head
query，从调用方准备的 shared KV pool 中按位置索引收集候选项，计算 scaled
dot-product attention，并输出与 query 同 shape 的 attention result。

```text
q [1,S,64,512]
+ shared KV pool [1,K,512]
+ topk_idxs [1,S,T]
+ per-head attn_sink [64]
  -> gather selected shared KV rows
  -> q·kv × 512^-0.5 + invalid-position mask
  -> softmax(selected logits + one sink logit per head)
  -> weighted sum of selected shared KV
  -> out [1,S,64,512]
```

三类接口的固定候选宽度为：

| Attention 类型 | 候选组成 | `T` |
|---|---|---:|
| SWA (Sliding Window Attention)     | 128 个 sliding-window slots | 128 |
| CSA（Compressed Sparse Attention） | 128 个 window slots + 512 个 Indexer slots | 640 |
| HCA（Heavily Compressed Attention） | 128 个 window slots + 32 个 ratio-128 compressed slots | 160 |

静态模型尺寸来自 [`models/config.py`](../../models/config.py)。该组件不生成 query、
KV、compressed KV 或索引，也不执行 Attention 输出投影；这些边界分别由
[Attention QKV](08_attention_qkv.md)、Compressor、[Indexer](07_indexer.md) 和后续
Attention output 组件负责。

## 官方模型中的 Sparse Attention

[`official/model.py`](../../official/model.py) 从外部 `kernel` 包导入 `sparse_attn`。
`Attention.forward()` 在准备好 `q`、KV pool、`attn_sink` 和 `topk_idxs` 后，以
`head_dim**-0.5` 为 scale 调用该算子：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

其数学语义可写为：

$$
l_{s,h,j} = q_{s,h} \cdot kv_{i_j} / \sqrt{512}
$$

$$
p_{s,h} = \operatorname{softmax}([l_{s,h,0}, \ldots, l_{s,h,n-1}, a_h])
$$

$$
o_{s,h} = \sum_{j=0}^{n-1} p_{s,h,j} kv_{i_j}
$$

其中 $i_j$ 是 `topk_idxs` 中的有效 KV pool 行号，$a_h$ 是该 head 的可学习
`attn_sink[h]`。Sink 只参与 softmax 的最大值和分母，没有对应的 value，因此可以
吸收一部分概率质量，但不直接向输出累加向量。

官方根据 `compress_ratio` 构造候选位置：

- ratio 0：只使用 `get_window_topk_idxs()`；
- ratio 4：拼接 window indices 与 Indexer 选出的 compressed indices；
- ratio 128：拼接 window indices 与 `get_compress_topk_idxs()` 生成的所有可见
  compressed indices。

Prefill 时 sparse attention 读取当前 prompt KV，必要时在其后拼接当前 prompt
生成的 compressed KV。Decode 时先更新 ring window cache 和 compressed cache，再
从两类 cache 组成的 KV pool 中读取当前 token 可见的位置。

## PyPTO kernel 实现

[`models/sparse_attn.py`](../../models/sparse_attn.py) 提供三套固定宽度 inline kernel：

| 符号 | 类型 | 职责 |
|---|---|---|
| `sparse_attn_swa_fwd` | `@pl.jit.inline` | 处理 128 个 sliding-window candidates |
| `sparse_attn_csa_fwd` | `@pl.jit.inline` | 处理 128 + 512 个 CSA candidates |
| `sparse_attn_hca_fwd` | `@pl.jit.inline` | 处理 128 + 32 个 HCA candidates |
| `sparse_attn_*_test` | `@pl.jit` | 三类 standalone 编译与数值验收 wrapper |
| `golden_sparse_attn` | PyTorch golden | 三类 kernel 共用的稀疏 attention 参考实现 |
| `build_window_topk_idxs` | Host helper | 构造并以 `-1` 补齐固定宽度 window indices |
| `build_compress_topk_idxs` | Host helper | 构造 HCA 规则化 compressed indices |
| `build_csa_synthetic_topk_idxs` | Test helper | 不运行 Indexer 时构造确定性的 CSA 验收输入 |
| `build_*_specs` | Host spec builder | 构造三类 prefill/decode standalone tensors |

`build_csa_synthetic_topk_idxs()` 只服务于 sparse-attention 独立验收：它按可见性选择
compressed cache 的前缀，不能代表真实 Indexer ranking。完整 CSA 路径使用
`models/indexer.py` 的 score Top-K 输出。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| 外部 `kernel.sparse_attn` | `sparse_attn_{swa,csa,hca}_fwd` | 语义等价：按固定候选宽度专用化 |
| `q @ selected_kv.T` | BF16 operands、FP32 matmul accumulation | 语义等价 |
| `head_dim**-0.5` | 编译期常量 `SOFTMAX_SCALE` | 语义等价：固定 head dim 512 |
| `-1` padding | `NEG_INF` score bias，gather value 初始化为零 | 语义等价 |
| `attn_sink` | softmax max/denominator 中的额外 logit | 语义等价 |
| selected KV 同时作为 K/V | `sparse_kv` / `sparse_value` | 语义等价：shared KV |
| SWA window candidates | `sparse_attn_swa_fwd` | 直接调用 |
| CSA window + Indexer candidates | `sparse_attn_csa_fwd` | 直接调用 |
| HCA window + ratio-128 candidates | `sparse_attn_hca_fwd` | 直接调用 |
| 官方动态候选宽度 | 固定 128/640/160，尾部补 `-1` | 语义等价：静态 kernel interface |
| Tensor Parallel local heads | 完整 64 heads | 不支持或未执行：当前为单卡逻辑 |

## 数据接口

三套 kernel 共用以下逻辑接口：

```text
q:          [1,S,64,512], BF16
kv:         [1,K,512],    BF16
attn_sink:  [64],         FP32
topk_idxs:  [1,S,T],      INT32
out:        [1,S,64,512], BF16
```

其中 `S` 和 `K` 是动态维，`T` 由 kernel 变体固定为 128、640 或 160。每个非负
index 都是 `kv` 第二维的绝对行号；`-1` 表示 padding/masked candidate。调用方必须
保证所有非负 index 小于 `K`。Kernel 不排序、不去重索引；重复位置会作为多个
softmax candidate 分别参与计算。

三类 KV pool 与索引空间如下：

| 路径 | Prefill KV pool | Decode KV pool | Compressed index offset |
|---|---|---|---:|
| SWA | 当前 prompt KV，`K=S` | 128-row window ring cache | 不适用 |
| CSA | `[prompt KV, ratio-4 compressed KV]` | `[window cache, 1024-row compressed cache]` | Prefill 为 `S`，decode 为 128 |
| HCA | `[prompt KV, ratio-128 compressed KV]` | `[window cache, 32-row compressed cache]` | Prefill 为 `S`，decode 为 128 |

Prefill 中 window index 是当前 prompt KV 的普通行号，并带 causal/sliding-window
可见性。Decode 中 window index 是 ring cache 的物理 slot，顺序由当前
`start_pos % 128` 决定。Sparse-attention kernel 本身不接收 `start_pos`；所有位置、
可见性和 pool layout 语义已经编码在 `topk_idxs` 中。

### State 所有权

Sparse Attention 自身没有持久 state，只读取调用方准备好的 device tensor：

- window KV cache 由 SWA/CSA/HCA Attention kernel 更新；
- ratio-4/ratio-128 compressed cache 及 Compressor staging state 由对应 Compressor
  更新；
- CSA index cache 由 Indexer 更新；
- host-side RoPE、window/HCA topk 和控制量由
  [`serving/state.py`](../../serving/state.py) 构造和缓存；
- device-resident state buffer 的分配、current/next 交换由 serving runtime 负责。

`sparse_kv`、score、softmax denominator 和 output numerator 都是单次 kernel 内部
scratch，不跨 layer 或 decode step 保存。完整 Attention 的 cache 会跨 decode step
复用，但这不表示 sparse-attention kernel 自己拥有 cache。

## 实现方式

### SWA 与 HCA

SWA 和 HCA 对每个 query token 先构造完整候选 KV scratch。Gather 阶段将 scratch
清零，只复制满足 `0 <= index < K` 的 KV 行；`-1` position 生成接近 FP32 最小值的
score bias。随后每 16 个 query heads 为一组：

1. 以 FP32 accumulation 计算 query 与全部候选 KV 的 dot product；
2. 乘固定 scale `512^-0.5` 并叠加 invalid-position bias；
3. 取 selected logits 与 per-head sink logit 的共同最大值；
4. 计算 selected exponential sum，并把 sink exponential 加入分母；
5. 按 32 个 value channels 分块累加 FP32 numerator；
6. 除以共同 denominator，以 round-to-nearest 转为 BF16 `out`。

两者的计算结构相同，区别是候选 scratch 分别为 128 和 160 行。

### CSA 分块在线 softmax

CSA 的固定候选宽度为 640。为避免一次驻留完整候选 KV，kernel 将 candidate 维切成
40 个 16-row blocks，并在每个 256-channel output block 上执行在线 softmax：

1. 以 `attn_sink` 初始化 running max，denominator 初始化为 sink contribution，
   numerator 初始化为零；
2. gather 当前 16 个 candidate 的完整 512 维 key 和当前 256 维 value；
3. 计算当前 chunk logits 及 mask，得到新的 running max；
4. 用 `exp(old_max-new_max)` 重标定已有 denominator 和 numerator；
5. 累加当前 chunk 的 exponential sum 与 weighted value；
6. 遍历 40 个 chunks 后归一化并写出 BF16 output。

该在线更新与对 640 个 selected logits 和一个 sink logit 执行整体 softmax 数学等价。
由于输出分成两个 256-channel blocks，score/softmax 流程会为每个 output block 各执行
一次。

### 固定宽度 Top-K helper

`build_window_topk_idxs()` 对 prefill 生成 causal 128-token sliding window；对 decode
生成 window ring cache 的有效物理 slot 顺序。`build_compress_topk_idxs()` 对 HCA
生成当前 token 已完成的 ratio-128 block，并把 pool offset 加到本地 compressed
slot 上。两者都将官方可变宽度矩阵补齐为 kernel 需要的固定宽度，多余项为 `-1`。

完整 serving 路径在 `serving/state.py` 中维护等价的 host helper，并按当前 prefill
长度或 decode position 缓存辅助 tensor。`models/sparse_attn.py` 内 helper 主要用于
standalone 和模型集成 spec；它们不是 kernel 内部运行时计算。

## 实现差异与限制

- 当前只支持 `B=1`、64 heads、head dim 512 和 shared KV，不是通用 sparse-attention
  API；
- 最大序列位置为 4096，因此 HCA compressed candidate 上限固定为 `4096/128=32`；
- CSA compressed Top-K 固定为 512，完整 compressed cache 固定为 1024 rows；
- 当前将外部官方 sparse-attention callable 专用化为 SWA、CSA、HCA 三套 kernel，
  不在运行时接收 compression ratio 或任意 Top-K width；
- `-1` 是唯一支持的 padding sentinel；调用方必须保证其他 index 位于 KV pool 范围
  内；
- Kernel 不负责 causal mask，因果性和 compressed-block visibility 完全由输入索引
  保证；
- Kernel 不生成或更新 KV cache、compressed cache、Indexer cache 与 topk state；
- Prefill 与 decode 共用各自 Attention 类型的同一个 sparse kernel，decode 由调用方
  约束为 `S=1`；
- 当前为单卡完整 64-head 逻辑，不实现 Tensor Parallel head shard 或集合通信；
- 主要输入输出为 BF16，dot product、softmax state 和 value accumulation 使用 FP32。

## Golden 参考实现

`models/sparse_attn.py::golden_sparse_attn` 逐 batch、逐 token 处理输入。它过滤
`topk_idxs >= 0` 的位置，按索引读取 shared KV，以 FP32 `einsum` 计算 64-head score，
乘 `softmax_scale`，再在末尾拼接 `[64,1]` 的 FP32 `attn_sink`。

Golden 对 selected logits 和 sink 一起执行 `torch.softmax`，移除 sink probability
后，以 selected probability 对同一份 KV 做 weighted sum，最后转回 query dtype。
当没有任何有效 index 时，输出保持为零。它不构造 cache 或执行 QKV/output
projection。

CSA standalone 使用 `build_csa_synthetic_topk_idxs()` 生成可重复的有效候选，但
golden 的 attention 数学计算与真实 Indexer 索引完全相同。真实 ranking 与 pool
offset 由 CSA/Indexer 集成测试覆盖。

## 精度验收标准

三类 standalone kernel 的 `out` 使用相同标准：

| 输出 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---:|---:|---:|
| `out` | `1e-4` | `1/128` | `0.001` |

逐元素容差为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的元素超出该条件，数量阈值按 comparator 对元素总数取整。Actual
output 中出现任何 NaN 或 Inf 都会直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上同时验证 SWA、CSA、HCA 的 prefill 和 decode：

```bash
python models/sparse_attn.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 256 \
  --decode-start-pos 255
```

使用非 tile 对齐长度和较早 decode position 验证 padding、causal visibility 与 tail：

```bash
python models/sparse_attn.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13 \
  --decode-start-pos 13
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

Host-side index helper 与官方逻辑的精确比较可运行：

```bash
pytest -q tests/models/test_sparse_attn.py
```

## 集成验证范围

### 独立 kernel 验收

`models/sparse_attn.py::main()` 顺序执行六个 case：SWA/CSA/HCA 各自的 prefill 和
decode。每个 case 独立构造 `q`、KV pool、sink 和固定宽度 indices，并与同一个
PyTorch golden 比较 `out`。

[`test_sparse_attn.py`](../../tests/models/test_sparse_attn.py) 不执行 NPU kernel；它
覆盖 window/HCA index helper 与 `official/model.py` 的一致性，以及 synthetic CSA
indices 的固定宽度、offset、visibility 和 `-1` padding。

### Attention 组合语义

- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py) 覆盖 window cache、
  prefill/decode pool 和 SWA sparse aggregation；
- [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 覆盖 ratio-4
  Compressor、真实 Indexer ranking、640-wide indices 和 CSA sparse aggregation；
- [`test_attention_hca.py`](../../tests/models/test_attention_hca.py) 覆盖 ratio-128
  Compressor、规则化 compressed indices 和 HCA sparse aggregation。

这些 host 测试将官方 `kernel.sparse_attn` 替换为当前 PyTorch golden，用于比较完整
Attention 的组合语义，不直接执行 PyPTO sparse kernel，不能替代 standalone NPU
验收。

### Block 与 serving 集成

- [`test_block.py`](../../tests/models/test_block.py) 覆盖三类 Attention 与
  Hyper-Connection、MoE 组成的完整 prefill/decode Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 selected-expert
  decode pre-MoE 路径中的 sparse attention、cache input/output 和后续 state commit；
- [`test_state.py`](../../tests/serving/test_state.py) 覆盖 runtime 的 window/HCA topk
  host cache、prefill/decode offset、position 边界与 ratio-specific aux bundle；
- [`test_device_state_store.py`](../../tests/serving/test_device_state_store.py) 覆盖 KV、
  Compressor 和 Indexer state 在 NPU current/next buffer 间的生命周期。

完整模型中，fixed weights、KV/cache state 与中间 tensor 均保持 device resident。
Sparse Attention 消费当前 kernel 内已经更新并组装好的 KV pool；kernel 返回后，
Attention output projection 继续消费 `out`，持久 state 在完整 Block 成功执行后由
serving runtime 提交。
