# Packed Expert Cache 导出

[`export_expert_cache.py`](../../export_expert_cache.py) 将 checkpoint 中逐专家的 routed MoE 权重转换为逐层 packed BF16 safetensors，供 prefill 整层读取和 decode selected-slice 读取。

## 功能定位

该入口是纯 Host 离线流程，不创建 `ChipWorker`。它复用 [`DeepSeekV4WeightLoader`](../../serving/weight_loader.py) 的 checkpoint 名称规范化、反量化和转置逻辑，输出 [`ExpertCacheReader`](../../serving/expert_cache.py) 能直接校验和读取的最终格式。

导出粒度固定为完整层：每个目标层包含全部 routed experts，CLI 不支持只导出部分 expert。

## 前置条件

`--checkpoint` 必须包含统一校验要求的 tokenizer、权重索引和索引引用的分片。`--output` 可以不存在或是兼容的既有 expert cache 目录。

输出目录若非空但没有 `manifest.json`，导出器拒绝写入，避免把 cache 文件混入未知目录。

## 基本用法

导出全部层：

```bash
python export_expert_cache.py \
  --checkpoint /path/to/checkpoint \
  --output /path/to/expert-cache
```

只导出部分层：

```bash
python export_expert_cache.py \
  --checkpoint /path/to/checkpoint \
  --output /path/to/expert-cache \
  --layers 0,2-4,8
```

## 命令行参数

| 参数 | 类型/默认值 | 作用与约束 |
|---|---|---|
| `--checkpoint` | `str`，必填 | 原始 checkpoint 目录 |
| `--output` | `str`，必填 | expert cache 输出目录，会执行 `expanduser()` 并按需创建 |
| `--layers` | `str`，默认 `None` | 逗号分隔的层号或闭区间；省略时导出全部 43 层 |
| `--overwrite` | flag，默认关闭 | 重建并替换已存在的目标层文件 |
| `--profile` | flag，默认关闭 | 输出每层 weight loader 子项耗时 |

`--layers` 会去重并按升序处理。例如 `3,1-2,2` 规范为 `[1, 2, 3]`。层号必须位于 `[0, n_layers)`，区间结束值不能小于开始值。

## 执行流程

```text
校验 checkpoint 和输出目录
        │
        ├── 读取或创建 manifest
        ├── 解析并排序目标 layer ids
        ├── 创建 Host DeepSeekV4WeightLoader
        │
        └── 对每个目标层
                ├── 若文件存在且未 --overwrite：校验并跳过
                ├── 逐个读取全部 routed experts
                ├── 转换并拷贝到 w1/w2/w3 packed tensor
                ├── 校验 key、shape、dtype、contiguous
                ├── 写入临时 safetensors 并回读校验
                ├── os.replace() 原子替换目标文件
                └── 原子更新 manifest
```

每处理完一个 expert，入口调用 `release_prefix()` 释放对应 Host layout，避免导出完整层时把所有逐专家中间 layout 保留在 loader cache 中。

## 磁盘产物

```text
<expert-cache>/
├── manifest.json
├── layer_000_experts.safetensors
├── layer_001_experts.safetensors
└── ...
```

每层文件包含且仅包含：

| Key | Shape | Dtype | 含义 |
|---|---|---|---|
| `routed_w1_t` | `[n_routed_experts, dim, moe_inter_dim]` | BF16 | 全专家 w1 转置 layout |
| `routed_w2_t` | `[n_routed_experts, moe_inter_dim, dim]` | BF16 | 全专家 w2 转置 layout |
| `routed_w3_t` | `[n_routed_experts, dim, moe_inter_dim]` | BF16 | 全专家 w3 转置 layout |

Safetensors metadata 保存 cache `format` 和字符串形式的 `version`。Manifest 保存相同格式版本、源 checkpoint 绝对路径、模型层数、专家数、维度、`bfloat16` dtype、tensor shape 描述和已完成层列表。

## 增量与原子写入

未指定 `--overwrite` 时，已存在的目标层不会直接信任：导出器先打开文件，检查 key 集合、shape、BF16 dtype 和 metadata，验证通过后才跳过，并补写 manifest 层记录。

写新文件时先使用进程号命名的隐藏临时文件。临时文件保存并通过回读校验后，再通过 `os.replace()` 替换目标路径；校验或写入失败时清理临时文件，既有目标文件保持不变。Manifest 使用独立的临时 JSON 和 `os.replace()` 更新。

## 资源与性能选项

每层构造三个完整 packed tensor，因此 Host 峰值至少包含该层的 w1、w2、w3 输出以及当前正在加载的 expert。导出完成或异常退出时，`finally` 关闭 weight loader、safetensors handle 和 Host layout cache。

`--profile` 在每层开始前清空 loader 统计，完成后输出读取、反量化、dtype cast、transpose 等实际发生的子项；`build` 和 `save` wall time 始终包含在逐层输出中。

## 错误处理

- 输出目录非空但没有 manifest 时拒绝写入。
- 既有 manifest 的 format、version、source checkpoint 或模型配置不匹配时拒绝续写。
- 层范围非法、目标文件 key/shape/dtype/metadata 不匹配时抛出异常。
- packed tensor 必须连续且为 BF16。
- CLI 不接受旧的 `--weight-index` 或部分专家参数。

## 验证方法

### CLI 与 Host 集成测试

```bash
pytest -q tests/cli/test_export_expert_cache.py
```

该测试覆盖层范围解析、expert 维度顺序、manifest、safetensors round trip、跳过既有层、错误版本拒绝和原子写入失败保护。

### 产物验收

对小范围层执行导出后，通过同一入口再次运行且不指定 `--overwrite`。验收时确认既有层通过校验并打印 `skip existing`，而不是被重写。运行时读取语义由 [`Expert Cache`](04_expert_cache.md) 的测试覆盖。

该流程不使用 NPU，因此不需要 `ChipWorker` 硬件验证。

## 代码索引

| 内容 | 位置 |
|---|---|
| 导出入口 | [`export_expert_cache.py`](../../export_expert_cache.py) |
| Cache 格式与 reader | [`serving/expert_cache.py`](../../serving/expert_cache.py) |
| Weight loader | [`serving/weight_loader.py`](../../serving/weight_loader.py) |
| CLI 测试 | [`tests/cli/test_export_expert_cache.py`](../../tests/cli/test_export_expert_cache.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`Expert Cache`](04_expert_cache.md)
- [`Weight Loader`](07_weight_loader.md)
