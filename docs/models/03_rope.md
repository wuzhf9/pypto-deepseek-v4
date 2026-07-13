# RoPE

## 模块定位

Rotary Position Embedding（RoPE）通过对相邻通道组成的二维向量执行位置相关旋转，
将 token 位置信息注入 query、KV 和 compressed KV。DeepSeek V4 Flash 的 attention
head 宽度为 512，indexer head 宽度为 128，但两者都只旋转最后 64 个通道；其余
通道保持不变。

对于一对输入通道 $(x_0, x_1)$ 和当前位置的角度 $(\cos\theta, \sin\theta)$，前向
旋转为：

$$
y_0 = x_0\cos\theta - x_1\sin\theta
$$

$$
y_1 = x_0\sin\theta + x_1\cos\theta
$$

inverse RoPE 使用共轭旋转，用于 Attention 输出投影前撤销 query 空间中的旋转。
当前模型配置和派生尺寸来自
[`models/config.py`](../../models/config.py)。

## 官方模型中的 RoPE

[`official/model.py`](../../official/model.py) 将 RoPE 分为 table 生成和 tensor 旋转
两部分：

- `precompute_freqs_cis()` 生成 `[sequence, rope_dim / 2]` 的 complex frequency
  table；
- `apply_rotary_emb()` 将最后 64 个实数通道视为 32 个 complex value，并与
  `freqs_cis` 相乘；
- `inverse=True` 时使用 `freqs_cis.conj()` 完成逆向旋转。

官方 Attention 中的调用位置包括：

| 官方位置 | Tensor | 操作 |
|---|---|---|
| `Attention.forward` | query `[B,S,64,512]` 的最后 64 维 | Forward RoPE |
| `Attention.forward` | shared KV `[B,S,512]` 的最后 64 维 | Forward RoPE |
| `Attention.forward` | attention output `[B,S,64,512]` 的最后 64 维 | Inverse RoPE |
| `Indexer.forward` | index query `[B,S,64,128]` 的最后 64 维 | Forward RoPE |
| `Compressor.forward` | attention compressed KV `[B,C,512]` 的最后 64 维 | Forward RoPE |
| `Indexer.Compressor.forward` | index compressed KV `[B,C,128]` 的最后 64 维 | Forward RoPE |

### RoPE profile

当前配置提供 normal 和 compressed 两套 profile：

| Profile | 使用层 | Base theta | Original sequence length | YaRN factor |
|---|---|---:|---:|---:|
| Normal | Compression ratio 0（SWA） | 10000 | 0 | 不启用插值 |
| Compressed | Compression ratio 4/128（CSA/HCA） | 160000 | 65536 | 16 |

两种 compressed ratio 共用同一套 compressed profile。`original_seq_len=0` 时不会
执行 YaRN frequency interpolation；compressed profile 使用 `beta_fast=32`、
`beta_slow=1` 计算平滑修正范围。

## PyPTO kernel 实现

[`models/rope.py`](../../models/rope.py) 将官方 complex table 表示为两个独立的
FP32 `cos`/`sin` tensor，并提供 host helper 与五个 `@pl.jit.inline` kernel。

### Host helper

| Helper | 职责 |
|---|---|
| `rope_profile_for_compress` | 根据 `compress: bool` 选择 normal/compressed profile |
| `precompute_freqs_cos_sin` | 生成 `[sequence, rope_dim/2]` FP32 cos/sin table |
| `build_deepseek_v4_rope_tables` | 使用当前模型配置构建完整 table |
| `materialize_rope_range` | 提取主 Attention 的连续 `[start_pos:start_pos+seq_len]` 区间 |
| `materialize_compressor_rope` | 提取 compressor prefill 的 `[:cutoff:ratio]` 位置 |

### Inline kernel

| Inline kernel | 输入 shape | 方向 | 验证 wrapper | Tensor spec builder |
|---|---|---|---|---|
| `rope_3d_512_fwd` | `[1,S,512]` | Forward | `rope_3d_512_fwd_test` | `build_rope_3d_512_specs` |
| `rope_3d_128_fwd` | `[1,S,128]` | Forward | `rope_3d_128_fwd_test` | `build_rope_3d_128_specs` |
| `rope_4d_512_fwd` | `[1,S,64,512]` | Forward | `rope_4d_512_fwd_test` | `build_rope_4d_512_specs` |
| `rope_4d_512_inv` | `[1,S,64,512]` | Inverse | `rope_4d_512_inv_test` | `build_rope_4d_512_specs` |
| `rope_4d_128_fwd` | `[1,S,64,128]` | Forward | `rope_4d_128_fwd_test` | `build_rope_4d_128_specs` |

所有 kernel 接收完整 head tensor，在内部复制不参与旋转的 prefix，只对最后 64 个
通道执行 FP32 旋转，然后以 round-to-nearest 转回 BF16。

## 官方模块到当前实现的映射

| 官方计算 | PyPTO 实现 | 关系 | 集成位置 |
|---|---|---|---|
| `precompute_freqs_cis` | `precompute_freqs_cos_sin` | 语义等价 | `models/rope.py`、`serving/state.py` |
| Normal/compressed Attention table 选择 | `rope_profile_for_compress` | 语义等价 | `models/rope.py`、`serving/state.py` |
| Attention query forward RoPE | `rope_4d_512_fwd` | 直接调用 | `models/attention_qkv.py` |
| Attention shared KV forward RoPE | `rope_3d_512_fwd` | 直接调用 | `models/attention_qkv.py` |
| Attention output inverse RoPE | `rope_4d_512_inv` | 直接调用 | `models/attention_out.py` |
| Indexer query forward RoPE | `rope_4d_128_fwd` | 直接调用 | `models/indexer.py` |
| Standalone 128 维 KV forward RoPE | `rope_3d_128_fwd` | 存在但完整模型未直接调用 | `models/rope.py::main` |
| Ratio-4 Attention compressor RoPE | 512 维等价计算 | 融合内联 | `models/compressor_ratio4.py` |
| Ratio-4 Indexer compressor RoPE | 128 维等价计算 | 融合内联 | `models/compressor_ratio4.py` |
| Ratio-128 Attention compressor RoPE | 512 维等价计算 | 融合内联 | `models/compressor_ratio128.py` |

`rope_3d_128_fwd` 保留独立实现和验收入口，可以表达 Indexer compressed KV 的
RoPE；当前完整模型将该计算与 compressor pooling、RMSNorm 和 cache 更新融合，
因此不直接调用它。

## 数据接口

独立 kernel 的公共输入形式为：

```text
x:   [1, S, ... , D], BF16
cos: [S, 32],         FP32
sin: [S, 32],         FP32
out: 与 x shape 相同, BF16
```

其中：

- Batch 固定为 1；
- `S` 是动态 sequence/token 维度；
- 可选 head 维固定为 64；
- `D` 固定为 512 或 128；
- `rope_dim=64`，因此 cos/sin table 的最后维度为 32；
- 512 维 head 的前 448 维保持不变，128 维 head 的前 64 维保持不变；
- kernel 不接受 complex tensor、weight、bias、持久 state 或 cache。

`cos` 和 `sin` 必须已经对应当前 token 的实际位置。普通 prefill 使用从位置 0 开始
的连续区间；decode 使用 `[start_pos:start_pos+1]`；compressor prefill 使用每个完整
compression block 的起始位置；compressor decode 只在形成完整 block 时提供对应
位置，否则提供占位 table 并由 `should_compress` 阻止写入 compressed KV。

独立 RoPE kernel 不拥有跨调用 scratch。`out` 由调用方提供，pair index、交换 index、
符号、interleaved cos/sin 和 FP32 rotated tile 都在单次调用内生成。

## Kernel 实现方式

当前公共 tiling 为：

```text
ROPE_T_TILE = 16
ROPE_PREFIX_TILE = 64
```

Forward kernel 的主要步骤为：

1. 将输入 reshape 为按 token 展平的二维表示；
2. 每个 block 最多处理 16 个 token；
3. 以 64 通道为单位复制不参与旋转的 prefix；
4. 为最后 64 个通道构造 pair、swap 和 sign index；
5. 将 32 列 cos/sin gather 成与 64 个实数通道对应的 interleaved 表示；
6. 在 FP32 中执行二维旋转；
7. 转为 BF16 并与原 prefix 组装为完整输出。

4D kernel 进一步按 64 个 heads 执行 SPMD 并行。Inverse kernel 复用相同的 pair
布局，仅改变 sin 项的符号以实现共轭旋转。动态 sequence tail 使用 `valid_shape`，
因此 `S` 不要求是 16 的整数倍。

### Host table 与 step cache

[`serving/state.py`](../../serving/state.py) 在 `DeepSeekV4StatePlan` 构造时生成长度
4096 的 normal/compressed 两套 host table。Runtime 主模型当前固定
`max_seq_len=4096`，尽管模型配置元数据中的最大位置长度更大。

StatePlan 按 prefill `seq_len` 或 decode `start_pos` 缓存不可变辅助输入：

- 相同 step、相同 profile 的主 RoPE slice 跨层复用；
- ratio 4 和 ratio 128 的主 Attention 共用 compressed profile slice；
- ratio-4 Attention compressor 与 Indexer compressor 共用同一 cos/sin slice；
- compressor prefill slice 按 ratio 分别缓存；
- decode 只在 `(start_pos + 1) % ratio == 0` 时物化有效 compressor RoPE。

`serving/state.py` 保留 runtime 侧 helper，以支持短序列占位行、decode compressor
边界和 per-step cache；`tests/serving/test_state.py` 用于保证其 table/profile 语义与
kernel helper 和官方公式一致。

## 实现差异与限制

当前实现与官方 RoPE 路径的主要差异如下：

- 官方使用 complex `freqs_cis` 和 complex multiplication；当前接口使用两个 FP32
  cos/sin tensor 和显式实数 pair rotation；
- 官方 `apply_rotary_emb` 接收已经切出的最后 64 维 view；当前独立 kernel 接收
  完整 512/128 维 head，并复制不旋转的 prefix；
- 当前 kernel 只覆盖模型实际使用的 3D/4D、512/128 固定 shape；
- 只有 4D 512 提供 inverse kernel，因为当前模型只在 Attention output 上执行
  inverse RoPE；
- Compressor 的 512/128 维 RoPE 已融合到对应 compressor kernel；
- 当前完整推理 runtime 固定最大 sequence length 为 4096；
- `models/rope.py::materialize_compressor_rope` 返回官方 `[:cutoff:ratio]` 切片；
  runtime state 在不足一个 compression block 时额外保留一行占位数据，以满足固定
  kernel shape，但不会将其视为有效 compressed KV。

## Golden 参考实现

`models/rope.py` 提供两层 PyTorch golden：

- `_apply_rope_tail_golden`：将最后 64 维拆成相邻 pair，在 FP32 中执行 forward 或
  inverse 旋转，再转换为输入 dtype；
- `_apply_rope_golden`：复制完整输入，只替换最后 64 维；
- `golden_rope_fwd` 和 `golden_rope_inv`：写入 standalone kernel 的 `out`。

Forward pair 计算为：

```text
y0 = x0 * cos - x1 * sin
y1 = x0 * sin + x1 * cos
```

Inverse pair 计算为：

```text
y0 = x0 * cos + x1 * sin
y1 = -x0 * sin + x1 * cos
```

Golden 使用与 kernel 相同的 BF16 输入 snapshot 和 FP32 cos/sin table，最终输出为
BF16。Host 测试还会将手写 golden 与官方 complex path 做逐元素精确比较。

## 精度验收标准

五个独立 kernel 使用同一验收标准：

| 项目 | 标准 |
|---|---:|
| Absolute tolerance | `1e-4` |
| Relative tolerance | `5e-3` |
| 允许超出容差的元素比例 | `0` |
| NaN/Inf | 不允许 |

每个输出元素必须满足：

```text
abs(actual - expected) <= 1e-4 + 5e-3 * abs(expected)
```

Host 侧 table、slice 和手写 rotation 与官方实现的比较使用 `rtol=0, atol=0`。这组
精确比较用于验证公式和 table 生成，不替代 NPU kernel 的容差验收。

## 验收方法

`models/rope.py` 的命令行入口会依次验证五个 kernel。Ascend A2/A3 实机命令为：

```bash
python models/rope.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8 \
  --start-pos 7
```

使用非 tile 对齐 sequence length 并从 prefill 起点验证：

```bash
python models/rope.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --start-pos 0
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

五个固定 shape 的直接编译、NPU 执行和 golden 比较由
`models/rope.py::main()` 完成。`rope_3d_128_fwd` 仅在这一独立入口中直接执行。

### Host table 与官方公式覆盖

[`test_rope_golden.py`](../../tests/models/test_rope_golden.py) 覆盖：

- normal、ratio-4 和 ratio-128 profile table 与官方 `freqs_cis`；
- compressor prefill slice；
- 3D/4D、512/128、forward/inverse 手写 rotation 与官方 complex path；
- 不参与旋转的 prefix 保持不变。

### 上层模型路径覆盖

- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py)、
  [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 和
  [`test_attention_hca.py`](../../tests/models/test_attention_hca.py)：Attention
  query/KV forward RoPE 与 output inverse RoPE；
- [`test_attention_out.py`](../../tests/models/test_attention_out.py)：inverse RoPE
  与 output projection 的组合路径；
- [`test_indexer.py`](../../tests/models/test_indexer.py)：128 维 index query RoPE；
- [`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py) 和
  [`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py)：
  融合的 compressed KV RoPE；
- [`test_state.py`](../../tests/serving/test_state.py)：normal/compressed profile、
  prefill/decode slice、compression boundary 和跨层 host cache 复用。

这些 host 测试验证官方语义、辅助输入和组合数据流，不编译或执行独立 NPU RoPE
kernel，因此不能替代 `models/rope.py::main()` 的验收。

### 完整模型集成

完整模型的 SWA、CSA 和 HCA prefill/decode 路径覆盖两套 profile、主 Attention
RoPE、compressor RoPE 和 inverse RoPE。完整模型验证可以检查 position、cache
边界及逐层绑定，但不能替代独立 kernel 对 rotation 数值误差的定位。
