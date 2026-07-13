# Weight Loader 与 Host Runtime Layout

[`DeepSeekV4WeightLoader`](../../serving/weight_loader.py) 在 Host 上按需读取 checkpoint、反量化、转换为 Kernel-facing layout，并把固定权重描述为 `RuntimeWeight`、routed expert 权重描述为 `HostStagingTensor`。

## 模块定位

Weight loader 位于 checkpoint/expert cache 与 runner/runtime 之间。它只生成 Host tensor 和 runtime value descriptors，不创建 DeviceTensor，也不决定设备 buffer 的生命周期。

Runner 按 embedding、head 和逐层组件请求结构化权重组；[`DeviceRuntime`](../../serving/device_runtime.py) 根据返回值类型决定固定 NPU 常驻或有界 staging。

## Checkpoint 索引

默认从 checkpoint 根目录读取 `model.safetensors.index.json`。Loader 支持标准 `weight_map` 和包含完整 entry 的内部 index 字典，并把 Hugging Face 名称规范为 inference-side 名称，例如：

| Checkpoint 名称 | Runtime 名称 |
|---|---|
| `model.embed_tokens.weight` | `embed.weight` |
| `model.layers.2.self_attn.q_a_proj.weight` | `layers.2.attn.wq_a.weight` |
| `model.layers.2.mlp.experts.7.gate_proj.weight` | `layers.2.ffn.experts.7.w1.weight` |
| `model.layers.3.mlp.gate.e_score_correction_bias` | `layers.3.ffn.gate.bias` |
| `lm_head.weight` | `head.weight` |

MTP 0 entries 在 index 规范化时被排除。每个 normalized entry 保存分片文件、原始 tensor 名和可选 kind/scale 信息。

[`validate_checkpoint_directory()`](../../serving/checkpoint.py) 在入口层检查 checkpoint 目录、`tokenizer.json` 和索引文件；loader 自身要求索引存在，并在真正读取 tensor 时打开索引引用的 safetensors 分片。

## 反量化与 dtype

Loader 根据 index `kind` 或原始 tensor dtype 识别：

| Kind | 原始表示 | 转换 |
|---|---|---|
| Plain tensor | floating point | 转到目标 device/dtype |
| Integer tensor | `int32` / `int64` | 保持整数语义，再按请求 dtype 转换 |
| FP8 weight | `float8_e4m3fn` + scale | 按 `[128,128]` block scale 转为 BF16 |
| FP4 packed weight | `int8` packed E2M1 + scale | 每 byte 解包低/高 nibble，按 32 元素 scale 转为 BF16 |

量化权重没有匹配 scale entry 时直接报错。普通 linear weight 最终为浮点 tensor；除 LM head 外，linear layout 通常执行 `.t().contiguous()` 变为 `[in, out]`。

## 结构化权重接口

Runner 使用以下主要接口：

| 接口 | 返回内容 |
|---|---|
| `get_embedding_weight()` | embedding fixed `RuntimeWeight` |
| `get_head_weights()` | HC head、final norm 和 LM head |
| `get_layer_hc()` | attention/FFN HC function、scale 和 base |
| `get_layer_attention_common()` | attention norm、Q/KV/O 投影和 sink |
| `get_layer_compressor_ratio*()` | compressor 投影、APE 和 norm |
| `get_layer_indexer()` | indexer 与其 compressor 权重 |
| `get_layer_moe_gate()` | gate weight，以及 hash `tid2eid` 或 top-k bias |
| `get_layer_moe_shared()` | shared expert w1/w2/w3 fixed weights |
| `get_layer_moe_routed_pack()` | prefill 全 routed-expert staging |
| `get_layer_moe_selected_experts()` | decode selected-expert staging |

Head HC function 和 base padding 到宽度 `16`；LM head 保持 checkpoint 的 FP32 `[vocab, dim]` layout，不走普通 linear transpose。

## Host Runtime Layout Cache

固定权重通过 `_get_runtime_weight()` 进入无容量限制、无 LRU 的 `_layout_cache`。Cache key 是：

```text
(RuntimeWeightKey(name, dtype, layout, layout_version, padding_profile), target_device)
```

这保证同一 checkpoint tensor 的 identity、transposed、不同 dtype 或不同 padding layout 不会互相覆盖。非 identity layout 必须提供显式 builder，避免把 raw tensor 错误缓存为已转换 layout。

Cache 命中返回同一个 `RuntimeWeight` 对象；`layout_cache_bytes` 统计所有 Host cached tensor 的逻辑字节数。普通权重和 shared experts 保留到显式 `release()`、`release_prefix()` 或 `close()`，不会因容量压力自动淘汰。

Safetensors file handle 是另一层复用：按分片绝对路径保持打开到 loader 整体释放。释放某个 layout 不会关闭文件 handle。

## Routed Expert 例外

单个 routed expert 的 w1/w2/w3 使用 `cache=False` 构造转置 Host tensor，不进入固定 layout cache。

### Prefill

`get_layer_moe_routed_pack()` 优先通过 `ExpertCacheReader.load_routed_pack()` 读取一层完整 packed cache；cache miss 时逐个加载全部 `n_routed_experts`。结果包装为三个 `PREFILL_ROUTED` `HostStagingTensor`，slot 分别为 `w1_t`、`w2_t`、`w3_t`。

### Decode

`get_layer_moe_selected_experts()` 将 NPU 路由结果规范为恰好 `n_activated_experts` 个 id。它优先从 packed cache lazy slices 复制；cache miss 时逐 expert 从 checkpoint 构造。结果包装为三个 `DECODE_SELECTED` staging tensors，保留 id 顺序和重复项。

Routed experts 不进入 `_layout_cache`；shared experts 仍作为固定 `RuntimeWeight` 缓存并最终常驻 NPU。

## 数据位置与生命周期

| 数据/资源 | 位置 | 所有者 | 生命周期与释放 |
|---|---|---|---|
| Index | Host Python dict | weight loader | loader 生命周期 |
| Safetensors handles | Host | weight loader | 首次访问分片至整体 release/close |
| Fixed Host layout | Host tensor in `RuntimeWeight` | weight loader | 首次请求至显式释放 |
| Single routed expert layout | Host tensor | 当前调用 | 不缓存，组包/复制后释放引用 |
| Prefill routed pack | Host `HostStagingTensor` | runner values | 当前层 prefill materialize |
| Decode selected pack | Host `HostStagingTensor` | runner values | 当前层 decode post-MoE materialize |
| Expert cache handles | Host 文件映射 | `ExpertCacheReader` | loader 整体 release/close |

Loader 默认目标 device 为 CPU，Serving runtime 也要求 Host descriptors 携带 CPU tensor。传入其他 target 只改变 loader tensor 构造位置，不等同于 DeviceRuntime 的 NPU ownership。

## 释放接口

| 接口 | 行为 |
|---|---|
| `release(name)` | 删除指定参数名的所有 dtype/layout/device cache entries |
| `release_prefix(prefix)` | 删除参数名前缀匹配的 layout entries |
| `release()` | 清空全部 layout cache、关闭 expert reader 和 checkpoint handles |
| `close()` | 等价于整体 `release()` |

释放时同步更新 `layout_cache_bytes`。`close()` 不关闭 `DeviceRuntime`；runner 分别管理二者。

## 性能与可观测性

启用 `profile=True` 时按名称累计 count 与 elapsed milliseconds，包括：

- `raw_load.<kind>`、`scale_load.<kind>`、`dequant.<kind>`、`to_device.<kind>`；
- `dtype_cast.<dtype>`、`transpose.<layout>`、`copy_linear_t`；
- `cache.layout.hit`、`cache.layout.miss`；
- `expert_cache.routed_pack`、`expert_cache.selected_slice_copy`；
- `selected_experts.build`。

`reset_profile_stats()` 只清空统计，不清空 layout 或文件 handle；`profile_summary()` 保留事件首次出现顺序。

## 约束与错误处理

- Layer id 与 expert id 必须位于模型配置范围。
- Selected expert id 数量必须恰好等于 `n_activated_experts`。
- Layout builder 结果必须与 key 的 dtype 和目标 device 一致。
- Quantized weight shape 和 scale shape 必须满足各自 block 规则。
- Hash route 优先读取 `tid2eid`，不存在时兼容 checkpoint 中的 `tie2eid` 名称；top-k route读取 gate bias。
- Packed expert cache 声明损坏时不回退 checkpoint；仅未配置或 manifest 缺层时 fallback。

## 验证方法

### Host 侧验证

```bash
pytest -q tests/serving/test_weight_loader.py
```

测试覆盖名称规范化、FP8/FP4 反量化、各组件 layout、固定 cache identity、key 隔离、无 LRU 行为、显式释放、file handle 复用、expert fallback、lazy selected slices 和 prefill full pack。

### Official checkpoint 接口验证

```bash
pytest -q tests/serving/test_weight_loader.py --official-checkpoint /path/to/checkpoint
```

配置 official checkpoint 后，测试读取代表性的 HC、attention、compressor、indexer、gate 和 expert 权重，验证 shape、dtype 与 contiguous。该方法仅验证 Host 权重接口，不执行 NPU Kernel。

## 代码索引

| 内容 | 位置 |
|---|---|
| Weight loader | [`serving/weight_loader.py`](../../serving/weight_loader.py) |
| Checkpoint 校验 | [`serving/checkpoint.py`](../../serving/checkpoint.py) |
| Expert reader | [`serving/expert_cache.py`](../../serving/expert_cache.py) |
| Runtime values | [`serving/runtime_types.py`](../../serving/runtime_types.py) |
| Loader 测试 | [`tests/serving/test_weight_loader.py`](../../tests/serving/test_weight_loader.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`Expert Cache`](04_expert_cache.md)
- [`Runtime Values`](08_runtime_values.md)
- [`Device Runtime`](10_device_runtime.md)
