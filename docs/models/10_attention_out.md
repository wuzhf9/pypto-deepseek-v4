# Attention Out

## 模块定位

Attention Out 是三类 Attention 共用的输出变换组件。它接收
[Sparse Attention](09_sparse_attn.md) 生成的 64-head shared-KV attention result，先对
每个 head 的最后 64 维执行 inverse Rotary Position Embedding（RoPE），再通过两级
output projection 将 `[1,S,64,512]` 映射回模型 hidden space `[1,S,4096]`。

```text
sparse attention result o [1,S,64,512], BF16
+ position inputs {cos, sin}
+ fixed weights {wo_a_t, wo_b_t}
  -> inverse RoPE on the last 64 channels of every head
  -> reshape 64 heads into 8 output groups, 8 heads per group
  -> 8 × grouped projection: 4096 -> 1024
  -> flatten 8 group outputs: 8 × 1024 -> 8192
  -> output projection: 8192 -> 4096
  -> attention out [1,S,4096], BF16
```

该组件位于 Attention 子层的末端：上游由 [Attention QKV](08_attention_qkv.md) 和
Sparse Attention 生成 attention result，下游由 Block 的 Attention
Hyper-Connection post 消费 4096 维输出。Sliding Window Attention（SWA）、
Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA）的
prefill/decode 路径均直接调用同一个实现。

静态尺寸来自 [`models/config.py`](../../models/config.py)：64 heads、head dim 512、
8 个 output groups、每组 8 heads、output LoRA rank 1024、hidden size 4096。

## 官方模型中的 Attention Out

[`official/model.py`](../../official/model.py) 的 `Attention` 定义两级输出投影：

| 官方参数 | 单卡逻辑 shape | 类型 | 作用 |
|---|---:|---|---|
| `wo_a.weight` | `[8192,4096]` | `ColumnParallelLinear` | 每个 output group 独立执行 4096→1024 投影 |
| `wo_b.weight` | `[4096,8192]` | `RowParallelLinear` | 将拼接后的 8192 维中间量回投影到 hidden 4096 |

`Attention.forward()` 在 sparse attention 后执行：

```python
apply_rotary_emb(o[..., -rd:], freqs_cis, True)
o = o.view(bsz, seqlen, self.n_local_groups, -1)
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
x = self.wo_b(o.flatten(2))
```

Inverse RoPE 使用 query/KV 路径相同位置的 frequency table 和共轭旋转，只作用于
每个 512 维 head 的最后 64 维。64 个 heads 随后按连续顺序重排为 8 组，每组输入
宽度为 `8 × 512 = 4096`；每组拥有独立的 `wo_a` 权重，输出 1024 维。8 组输出拼接
为 8192 维，再由 `wo_b` 映射回模型 hidden size。

官方多 rank 路径沿 `wo_a` output groups 切分，并沿 `wo_b` input 维切分；
`RowParallelLinear` 会对各 rank 的 FP32 partial output 执行 `all_reduce`。当前仓库只
实现单卡完整 8-group 路径。

官方注释指出 checkpoint 中 `wo_a` 可以是 FP8，但该处为简化而使用 BF16 计算；
当前 runtime 同样在 host 权重准备阶段将量化权重转换为 BF16，不执行量化 GEMM。

## PyPTO kernel 实现

[`models/attention_out.py`](../../models/attention_out.py) 提供：

| 符号 | 类型 | 职责 |
|---|---|---|
| `attention_out_fwd` | `@pl.jit.inline` | 完成 inverse RoPE、grouped `wo_a` 和 `wo_b` projection |
| `attention_out_fwd_test` | `@pl.jit` | Standalone 编译与数值验收 wrapper |
| `golden_attention_out` | PyTorch golden | 对齐 inverse RoPE、weight layout 和 BF16 rounding boundary |
| `build_tensor_specs` | Host spec builder | 构造指定 `seq_len/start_pos` 的输入、权重、RoPE 和输出 |

Inline kernel 直接复用：

- [`models/rope.py`](../../models/rope.py) 的 `rope_4d_512_inv`；
- [`models/linear.py`](../../models/linear.py) 的 `linear_8192_to_4096`。

`wo_a` 的 8-group 计算由 `attention_out_fwd` 自己实现，因为它不是一个普通的单矩阵
4096→8192 projection：每组只读取对应的 8 个 attention heads，并使用该组独立的
4096→1024 weight。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| output `apply_rotary_emb(..., inverse=True)` | `rope_4d_512_inv` | 直接调用：inline kernel |
| `o.view(B,S,n_groups,-1)` | `pl.reshape` 为 8 个连续 4096-channel groups | 语义等价 |
| `wo_a.weight.view(8,1024,4096)` | `wo_a_t` 的 8 个 4096×1024 column ranges | 语义等价：transposed runtime layout |
| `einsum("bsgd,grd->bsgr")` | `attention_out_wo_a` grouped matmul | 融合内联 |
| `o.flatten(2)` | `proj` reshape 为 `[1,S,8192]` | 语义等价 |
| `self.wo_b(...)` | `linear_8192_to_4096` | 直接调用：inline kernel |
| `wo_a` Column Parallel group shard | 无 | 不支持或未执行：当前计算完整 8 groups |
| `wo_b` Row Parallel partial sum / `all_reduce` | 无 | 不支持或未执行：当前单卡直接生成完整 output |
| FP8/量化 output projection | 无 | 不支持或未执行：runtime 使用 BF16 weight |
| SWA/CSA/HCA output path | `attention_{swa,csa,hca}_*_fwd` | 直接调用 `attention_out_fwd` |

## 数据接口

公共 kernel 接口为：

```text
o:         [1,S,64,512], BF16
wo_a_t:    [4096,8192],  BF16
wo_b_t:    [8192,4096],  BF16
cos:       [S,32],       FP32
sin:       [S,32],       FP32
out:       [1,S,4096],   BF16
```

其中：

- Batch 固定为 1，`S` 是动态 token 维；
- `o` 是 Sparse Attention 的输出，不是 Attention QKV 生成的 query；
- `wo_a_t` 和 `wo_b_t` 是 checkpoint weight 的转置 runtime layout；
- `wo_a_t[:, g*1024:(g+1)*1024]` 对应第 `g` 个 output group 的独立权重；
- `cos/sin` 已按当前 token 的实际 position 切片，32 列对应 64 个 RoPE channels；
- `out` 是进入 Attention Hyper-Connection post 的 4096 维 sublayer output。

Kernel 内部创建两个临时 tensor：

```text
o_inv: [1,S,64,512], BF16
proj:  [1,S,8192],   BF16
```

`o_inv` 保存 inverse RoPE 结果；`proj` 保存 8 个 grouped projection 的拼接结果。
两者均为单次 kernel scratch，不是跨 layer 或 decode step 保存的 state。

Prefill 使用动态 `S`；decode 由调用方约束为 `S=1`。Kernel 不接收 `start_pos` 或
compression ratio，绝对位置和 normal/compressed RoPE profile 完全由 cos/sin 内容
决定：SWA 使用 normal profile，CSA/HCA 使用 compressed profile。
Standalone `build_tensor_specs()` 使用 normal profile 构造 cos/sin；compressed profile
下的相同 inverse kernel 由 CSA/HCA 组合路径覆盖。

完整模型通过 `DeepSeekV4WeightLoader.get_layer_attention_common()` 加载并转置
`wo_a`、`wo_b`。两组 fixed weights 在加载后保持 device resident，并由每层的
prefill/decode kernel 复用。

## 实现方式

### Inverse RoPE

`attention_out_fwd` 首先调用 `rope_4d_512_inv`。该 primitive 复制每个 head 的前
448 个 channels，对最后 64 个 channels 使用当前 cos/sin 做共轭旋转；旋转计算在
FP32 中完成，再以 round-to-nearest 转为 BF16 `o_inv`。完整算法和 table profile
参见 [RoPE](03_rope.md)。

### Grouped `wo_a` projection

`o_inv` 展平为 `[S,32768]`，但 `wo_a` 不把全部 32768 channels 作为一个输入。
Kernel 对第 `g` 组只读取：

```text
o_inv[:, g*4096:(g+1)*4096]
wo_a_t[:, g*1024:(g+1)*1024]
```

并写入 `proj[:, g*1024:(g+1)*1024]`。因此数学形式为：

$$
proj_{s,g} = o\_inv_{s,g} W^{(a)}_g
$$

每个 token tile 最多处理 16 行。Input/K 维以 128 channels 分块，共 32 个 K blocks；
output 维以 32 channels 分块。SPMD task 同时编码 8 个语义 output groups 和组内
output tile；每个 pipeline iteration 处理两个相邻的 32-channel output tiles。
源码常量 `OUT_GROUP=2` 表示这种 output-tile batching，不是模型配置中的 8 个
`O_GROUPS`。

Matmul 使用 BF16 operands 和 FP32 accumulation。每个 4096→1024 grouped result
完成后以 round-to-nearest 转为 BF16，并组装到 `proj`；非 16 对齐的 sequence tail
通过 `valid_shape` 处理。

### `wo_b` projection

`proj` reshape 为 `[1,S,8192]` 后，直接调用 `linear_8192_to_4096`。该 linear 同样
以 16-token、128-input-channel、32-output-channel tiling 执行，8192 维 input 共
64 个 K blocks，并按两个 output tiles pipeline。FP32 accumulation 完成后以
round-to-nearest 转为 BF16 `out`。

因此当前实现保留三个明确的 BF16 boundary：inverse RoPE 输出、`wo_a` grouped
projection 输出以及 `wo_b` 最终输出。这些 boundary 与 standalone golden 一致。

## 实现差异与限制

- 当前只支持 `B=1`、64 heads、head dim 512、8 output groups、LoRA rank 1024 和
  hidden size 4096，不是任意 group/shape 的 Attention output API；
- 64 heads 必须按连续顺序组成 8 个 8-head groups，不能在运行时改变 group mapping；
- 当前使用 checkpoint weight 的 `[in,out]` 转置 runtime layout；
- 当前只接受 BF16 `o` 和 BF16 weights，inverse RoPE、matmul accumulation 使用
  FP32，不执行 FP8/FP4 activation quantization 或量化 GEMM；
- 当前不实现官方 Tensor Parallel 的 group shard、row shard 或 `all_reduce`；
- Kernel 不拥有 KV cache、compressed cache、topk index 或其他持久 state；
- Prefill 和 decode 共用同一个动态 sequence kernel，decode 由调用方约束为 `S=1`；
- Kernel 不选择 RoPE profile，也不校验 cos/sin 对应的绝对位置；
- 完整 runtime 的 position 上限为 4096。

## Golden 参考实现

`models/attention_out.py::golden_attention_out` 从 BF16 `o`、BF16 transposed weights
和 FP32 cos/sin snapshot 开始：

1. `_apply_rope_golden(..., inverse=True)` 对最后 64 维执行 FP32 inverse rotation，
   再转回 BF16；
2. 将 `o` reshape 为 `[1,S,8,4096]`；
3. 将 `wo_a_t` 转回 `[8192,4096]`，再 reshape 为 `[8,1024,4096]`；
4. 使用 FP32 `torch.einsum("bsgd,grd->bsgr")`，将结果转为 BF16；
5. 展平为 `[1,S,8192]`，使用 FP32 `torch.matmul` 乘 `wo_b_t`，将结果转为 BF16。

Golden 只写出最终 `[1,S,4096]` 的 `out`。`o_inv` 和 `proj` 是实现中的中间 boundary，
不作为 standalone comparator output。

## 精度验收标准

Standalone kernel 的 `out` 使用：

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

在 Ascend A2/A3 实机上验证默认 token tile 和非零 position：

```bash
python models/attention_out.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8 \
  --start-pos 7
```

使用非 16 对齐的 sequence length 验证 token tail：

```bash
python models/attention_out.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7 \
  --start-pos 3
```

使用单 token 和较后位置验证 decode-like shape：

```bash
python models/attention_out.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 1 \
  --start-pos 127
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

Host-side golden weight layout 可运行：

```bash
pytest -q tests/models/test_attention_out.py
```

## 集成验证范围

### 独立 kernel 验收

`models/attention_out.py::main()` 直接编译和执行 `attention_out_fwd_test`，比较最终
`out`。[`test_attention_out.py`](../../tests/models/test_attention_out.py) 不执行 NPU
kernel；它使用原始 `[out,in]` checkpoint-style weights 重建官方 grouped projection，
并逐元素精确验证 `golden_attention_out` 对 transposed runtime layout 的解释。

### Attention 组合语义

- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py) 覆盖 normal inverse
  RoPE、SWA sparse result 与两级 output projection；
- [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 覆盖 compressed
  inverse RoPE、CSA sparse result 与两级 output projection；
- [`test_attention_hca.py`](../../tests/models/test_attention_hca.py) 覆盖 compressed
  inverse RoPE、HCA sparse result 与两级 output projection。

这些 host 测试通过完整 Attention golden 与官方 `Attention.forward()` 比较组合语义，
不直接执行 standalone NPU Attention Out kernel，不能替代组件独立验收。

### Block 与 serving 集成

- [`test_block.py`](../../tests/models/test_block.py) 覆盖 Attention Out 与
  Hyper-Connection、MoE 组成的完整 Block；
- [`test_split_block.py`](../../tests/models/test_split_block.py) 覆盖 selected-expert
  decode pre-MoE 段中的 Attention Out 及后续 Hyper-Connection；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证 `wo_a`、`wo_b`
  checkpoint mapping、BF16 materialization 和转置 runtime layout。

完整模型中，`wo_a_t`、`wo_b_t` 和中间 tensor 保持 device resident。Attention Out
返回的 4096 维 tensor 在同一个 Block kernel 中直接交给 Attention Hyper-Connection
post；该组件不参与 state commit。
