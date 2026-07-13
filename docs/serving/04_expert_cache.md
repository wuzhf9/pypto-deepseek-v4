# Packed Expert Cache Reader

[`serving/expert_cache.py`](../../serving/expert_cache.py) 定义 packed BF16 routed-expert cache 的稳定磁盘格式，并为 weight loader 提供 prefill 整包读取和 decode 选中 expert 切片读取。

## 模块定位

`ExpertCacheReader` 位于磁盘 cache 与 [`DeepSeekV4WeightLoader`](../../serving/weight_loader.py) 之间。它负责格式和 shape 校验、safetensors handle 复用以及 tensor 读取，不负责 checkpoint fallback、Host staging 封装或 H2D。

上游导出格式由 [`export_expert_cache.py`](../../export_expert_cache.py) 生成。某层未出现在 manifest 中时，reader 返回“未缓存”，由 weight loader 回退到原始 checkpoint；manifest 已声明但文件缺失或损坏时则直接报错。

## 代码组成

| 类型/函数 | 职责 |
|---|---|
| `ExpertCacheManifest` | 保存已校验的版本、模型维度和逐层文件映射 |
| `LayerExpertCacheInfo` | 保存某层已校验的路径和 key 集合 |
| `layer_expert_cache_filename()` | 生成 `layer_NNN_experts.safetensors` 文件名 |
| `ExpertCacheReader.inspect_layer()` | 延迟打开并校验一层 cache |
| `copy_selected_into()` | 按请求顺序复制 decode selected expert slices |
| `load_routed_pack()` | 物化 prefill 使用的完整逐层 packed tensor |
| `close()` | 关闭缓存的 safetensors handles 并清空层信息 |

## 磁盘接口

当前格式常量为：

| 项目 | 值 |
|---|---|
| Format | `dsv4_bf16_layer_experts` |
| Version | `2` |
| Manifest | `manifest.json` |
| 层文件 | `layer_{layer_id:03d}_experts.safetensors` |
| Tensor keys | `routed_w1_t`、`routed_w2_t`、`routed_w3_t` |

Reader 在构造时立即读取 manifest，并严格比较 `n_layers`、`n_routed_experts`、`dim`、`moe_inter_dim` 和 `dtype=bfloat16`。逐层文件在第一次访问该层时才打开和校验。

每层 tensor 约束为：

| Key | Shape | Safetensors dtype |
|---|---|---|
| `routed_w1_t` | `[n_routed_experts, dim, moe_inter_dim]` | `BF16` |
| `routed_w2_t` | `[n_routed_experts, moe_inter_dim, dim]` | `BF16` |
| `routed_w3_t` | `[n_routed_experts, dim, moe_inter_dim]` | `BF16` |

Key 集合必须完全相等：缺少 packed key 或混入逐 expert key 都会被拒绝。

## 读取流程

### Prefill 整包读取

`load_routed_pack(layer_id, device)` 调用 `inspect_layer()` 后，通过 `get_tensor()` 读取三个完整 packed tensor。目标是 CPU 时显式 clone 并 contiguous；其他 device 则执行 `.to(device).contiguous()`。

当前 Serving weight loader 以 CPU 为目标读取，再把结果包装为 `PREFILL_ROUTED` `HostStagingTensor`，因此 cache reader 本身不拥有 NPU staging。

### Decode selected-slice 读取

`copy_selected_into()` 接收 expert id 序列和三个调用方预分配的连续 BF16 输出 tensor：

1. 校验所有 expert id。
2. 确认该层已缓存并校验文件。
3. 在写入前校验三个输出 tensor 的 shape、dtype 和 contiguous。
4. 使用 `get_slice()` 按 expert id 取出 w1、w2、w3 slice。
5. 再次校验每个 slice 后，按请求顺序复制到输出 slot。

Expert id 的顺序和重复项都会保留。例如 `[2, 0, 2]` 产生三个输出 slot，而不是去重或排序。

## 数据位置与生命周期

| 数据/资源 | 位置 | 所有者 | 生命周期与释放 |
|---|---|---|---|
| Manifest dataclass | Host | `ExpertCacheReader` | reader 构造至对象释放 |
| `safe_open` handle | Host 文件映射 | `ExpertCacheReader` | 首次访问该路径至 `close()` |
| `LayerExpertCacheInfo` | Host | `ExpertCacheReader` | 首次校验该层至 `close()` |
| Prefill packed tensor | Host | 调用方 weight loader | 每层 prefill 值构造阶段 |
| Decode selected 输出 | Host | 调用方 weight loader | 每层 decode selected staging 构造阶段 |

`inspect_layer()` 的 metadata 缓存避免重复读取 key 和 shape；`_get_handle()` 按解析后的绝对路径复用一个 `safe_open` handle。`close()` 可重复调用，并清空 handle 与层信息缓存。

## 未缓存与错误的区别

| 情况 | Reader 行为 | Weight loader 行为 |
|---|---|---|
| `directory=None` | 返回 `None` / `False` | 回退 checkpoint |
| Manifest 未列出某层 | 返回 `None` / `False` | 仅该层回退 checkpoint |
| 配置了目录但缺少 manifest | 构造时抛出 `FileNotFoundError` | 不执行 fallback |
| Manifest 声明文件但文件不存在 | 访问层时抛出 `FileNotFoundError` | 不执行 fallback |
| Format/version/config 不匹配 | 抛出 `ValueError` | 不执行 fallback |
| Key、shape 或 dtype 错误 | 抛出 `ValueError`/`TypeError` | 不执行 fallback |

只有“未配置”或“manifest 未声明该层”表示合法 cache miss。已声明数据的缺失和损坏不会被静默掩盖。

## 性能与可观测性

Reader 通过注入的 `profile_callback(name, start)` 报告两类事件：

| 事件 | 范围 |
|---|---|
| `expert_cache.routed_pack` | 三个完整 packed tensor 的物化 |
| `expert_cache.selected_slice_copy` | selected slices 读取与复制 |

Handle 和层 metadata 会复用，但每次 prefill 仍会物化完整 packed tensor，每次 decode 仍会把 selected slices 复制到新的 Host staging tensor。

## 约束与错误处理

- Layer id 必须位于 `[0, n_layers)`，expert id 必须位于 `[0, n_routed_experts)`。
- Selected 输出第一维必须等于请求 expert id 数量。
- 三个 selected 输出必须在写入前全部通过校验，避免部分输出已被修改。
- Reader 不校验导出 manifest 的 `source_checkpoint`；运行时兼容性由格式版本和模型维度字段确定。

## 验证方法

### Host 侧验证

```bash
pytest -q tests/serving/test_expert_cache.py
```

测试覆盖 disabled reader、manifest 严格校验、缺层与缺文件差异、handle 复用、重复 selected ids、完整 pack 独立存储、写入前校验和错误 key/shape/dtype。

### 运行时集成验证

```bash
pytest -q tests/serving/test_weight_loader.py tests/cli/test_export_expert_cache.py
```

前者验证 cache hit 与 checkpoint fallback 如何转换为 staging，后者验证导出产物能被相同格式约束回读。Reader 本身不调度 NPU Kernel。

## 代码索引

| 内容 | 位置 |
|---|---|
| Reader 实现 | [`serving/expert_cache.py`](../../serving/expert_cache.py) |
| 导出入口 | [`export_expert_cache.py`](../../export_expert_cache.py) |
| Weight loader 调用方 | [`serving/weight_loader.py`](../../serving/weight_loader.py) |
| Reader 测试 | [`tests/serving/test_expert_cache.py`](../../tests/serving/test_expert_cache.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`Expert Cache 导出`](03_export_expert_cache.md)
- [`Weight Loader`](07_weight_loader.md)
