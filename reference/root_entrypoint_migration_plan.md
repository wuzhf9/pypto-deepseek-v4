# 根目录入口程序迁移方案

## 1. 目标与原则

当前仓库的三个用户入口位于 `serving/`：

```text
serving/generate.py
serving/run_model.py
serving/convert_expert_cache.py
```

本次重构将它们直接移动到仓库根目录：

```text
generate.py
smoke_model.py
export_expert_cache.py
```

目标是让仓库根目录成为唯一 CLI 入口位置，并把 `serving/` 收敛为可复用的运行时、checkpoint、权重、
state 和 expert cache 实现。

重构遵循以下原则：

- 三个入口只保留根目录版本；
- 不保留 `serving.generate`、`serving.run_model` 或 `serving.convert_expert_cache` 兼容模块；
- 不增加根目录 wrapper + serving 实现的双层入口；
- 不改变 CLI 参数语义、checkpoint 约束、模型数值或资源生命周期；
- 不改变 `serving.runner`、`serving.device_runtime`、`serving.weight_loader` 等内部模块路径；
- 三个入口继续使用绝对 import；
- 当前历史 reference 文档不做全局机械改写，只更新仍作为当前执行依据的命令。

## 2. 最终目录结构

```text
./
├── generate.py
├── smoke_model.py
├── export_expert_cache.py
├── models/
├── official/
│   ├── encoding_dsv4.py
│   └── model.py
├── serving/
│   ├── checkpoint.py
│   ├── device_pool.py
│   ├── device_runtime.py
│   ├── device_state_store.py
│   ├── expert_cache.py
│   ├── profiler.py
│   ├── runner.py
│   ├── runtime_types.py
│   ├── state.py
│   └── weight_loader.py
└── tests/
    ├── conftest.py
    ├── models/
    ├── serving/
    └── cli/
```

迁移后的命令为：

```bash
python generate.py ...
python smoke_model.py ...
python export_expert_cache.py ...
```

不再支持：

```bash
python serving/generate.py ...
python serving/run_model.py ...
python serving/convert_expert_cache.py ...
python -m serving.generate ...
python -m serving.run_model ...
python -m serving.convert_expert_cache ...
```

## 3. 文件变更总览

### 3.1 移动文件

| 当前文件 | 最终文件 | 修改方式 |
|---|---|---|
| `serving/generate.py` | `generate.py` | 原样迁移主体，保持绝对 import |
| `serving/run_model.py` | `smoke_model.py` | 迁移并使用准确的 smoke 入口名称 |
| `serving/convert_expert_cache.py` | `export_expert_cache.py` | 迁移并使用明确的磁盘导出语义 |

迁移完成后删除三个旧文件，不保留 re-export。

### 3.2 修改测试

| 文件 | 修改内容 |
|---|---|
| `tests/test_generate.py` | 移动为 `tests/cli/test_generate.py`，并改为 `import generate` |
| `tests/test_run_model.py` | 移动为 `tests/cli/test_smoke_model.py`，并改为 `import smoke_model` |
| `tests/test_convert_expert_cache.py` | 移动为 `tests/cli/test_export_expert_cache.py`，并改为 `import export_expert_cache as exporter` |

### 3.3 可能更新的活跃文档

| 文件 | 修改内容 |
|---|---|
| `reference/packed_bf16_expert_cache_implementation_plan.md` | 将最终有效的 exporter/smoke_model 示例切换为根目录命令 |
| `reference/prefill_routed_expert_prefetch_plan.md` | 如存在当前有效的运行命令，改为根目录入口 |
| 本文 | 作为入口迁移的最新实施依据 |

其他文档若描述的是历史设计阶段，保留当时的 `serving/...` 路径，不进行机械替换。

## 4. 各文件具体修改

### 4.1 根目录 `generate.py`

从 `serving/generate.py` 迁移全部内容，包括：

- prompt/`--prompt-file` 解析；
- official encoding 绑定；
- tokenizer 加载；
- checkpoint 目录校验；
- `DeviceRuntime` 和 `DeepSeekV4Runner` 组合；
- prefill/decode generation loop；
- completion 格式化和 stats 输出；
- `parse_args()`、`main()` 和 `if __name__ == "__main__"`。

保持以下 import 不变：

```python
from models.config import FLASH_CONFIG
from official.encoding_dsv4 import ...
from serving.checkpoint import validate_checkpoint_directory
from serving.device_runtime import DeviceRuntime
from serving.state import DEFAULT_MAX_SEQ_LEN
```

继续延迟 import `serving.runner.DeepSeekV4Runner`，保持 tokenizer/helper 单元测试不强制加载 PyPTO kernel。

不修改以下行为：

- `--checkpoint` 必须包含 `tokenizer.json` 和 `model.safetensors.index.json`；
- 不恢复 `--tokenizer-path`、`--weight-index`、`--encoding-path` 或 `--backend`；
- official encoding 固定从仓库 `official.encoding_dsv4` 加载；
- runtime 初始化失败时立即关闭 runtime；
- runner 生命周期由 `try/finally` 关闭。

### 4.2 根目录 `smoke_model.py`

从 `serving/run_model.py` 迁移全部内容，包括：

- smoke CLI；
- checkpoint 校验；
- 随机输入构造；
- DeviceRuntime 创建和异常清理；
- prefill/decode finite 检查；
- head/no-head 下一 token 选择；
- `main()` 和脚本入口。

保持以下行为：

- checkpoint 在 DeviceRuntime 创建前校验；
- `--enable-l2-swimlane` 和 `--keep-prefill-routed-staging` 继续透传；
- `--max-layers`、`--no-head`、`--decode-steps`、`--profile` 等参数不变；
- 不恢复 `--weight-index` 或 `--backend`；
- Runner 继续通过 `runtime=` 注入。

### 4.3 根目录 `export_expert_cache.py`

从 `serving/convert_expert_cache.py` 迁移全部内容，包括：

- layer range 解析；
- routed expert packed tensor 构建；
- manifest 创建和校验；
- safetensors 原子写入；
- existing layer skip/overwrite；
- profile 输出；
- CLI 和脚本入口。

保持以下 import 指向 `serving` 内部库：

```python
from serving.checkpoint import validate_checkpoint_directory
from serving.expert_cache import ...
from serving.weight_loader import DeepSeekV4WeightLoader, tensor_nbytes
```

保持以下行为：

- `--checkpoint` 同时要求 tokenizer 和 canonical model index；
- exporter 固定由 checkpoint 根目录解析权重；
- 不恢复 `--weight-index`；
- cache format/version、packed keys 和 manifest schema 不变；
- 不触碰已有 cache 数据格式。

### 4.4 `tests/cli/test_generate.py`

只切换被测模块 import：

```python
import generate
```

保留所有测试内容：

- official encoding 默认绑定；
- prompt 优先级和文件读取；
- tokenizer/checkpoint 组合；
- generation loop；
- Runner/runtime 参数透传；
- 初始化失败资源清理；
- 已删除 CLI 参数的拒绝测试。

monkeypatch 目标继续使用根模块属性，例如：

```python
monkeypatch.setattr(generate, "DeviceRuntime", ...)
```

### 4.5 `tests/cli/test_smoke_model.py`

import 改为：

```python
import smoke_model
```

保留 runtime 构造、Runner 注入、checkpoint 校验前置和异常关闭测试。所有 monkeypatch 目标改为根模块。

### 4.6 `tests/cli/test_export_expert_cache.py`

import 改为：

```python
import export_expert_cache as exporter
```

保留 packed layer、manifest、atomic write、skip/overwrite、CLI 参数拒绝和 checkpoint canonical layout 测试。

## 5. 不需要修改的文件

以下文件不应因为入口移动而改变：

- `serving/checkpoint.py`；
- `serving/device_runtime.py`；
- `serving/runner.py`；
- `serving/weight_loader.py`；
- `serving/expert_cache.py`；
- `serving/state.py`；
- `models/*`；
- `official/*`。

这些模块已经使用绝对 import，入口移动不会改变其职责或依赖方向。

## 6. 修改顺序

### Stage 1：迁移 generate 入口

1. 将 `serving/generate.py` 移动到根目录 `generate.py`；
2. 将测试移动为 `tests/cli/test_generate.py` 并修改 import；
3. 运行 `tests/cli/test_generate.py`；
4. 运行 `python generate.py --help`；
5. 搜索 `serving.generate` 和 `serving/generate.py` 的生产代码/测试残留。

### Stage 2：迁移并重命名 smoke_model 入口

6. 将 `serving/run_model.py` 移动并重命名为根目录 `smoke_model.py`；
7. 将 `tests/test_run_model.py` 移动为 `tests/cli/test_smoke_model.py`，并修改 import；
8. 运行 `tests/cli/test_smoke_model.py`；
9. 运行 `python smoke_model.py --help`；
10. 搜索 `serving.run_model` 和 `serving/run_model.py` 的生产代码/测试残留。

### Stage 3：迁移并重命名 expert cache exporter 入口

11. 将 `serving/convert_expert_cache.py` 移动并重命名为根目录 `export_expert_cache.py`；
12. 将 `tests/test_convert_expert_cache.py` 移动为 `tests/cli/test_export_expert_cache.py`，并修改 import；
13. 运行 exporter 单元测试；
14. 运行 `python export_expert_cache.py --help`；
15. 搜索 `serving.convert_expert_cache` 和 `serving/convert_expert_cache.py` 残留。

### Stage 4：删除旧入口和更新命令

16. 确认 Git 将三个文件识别为 rename；
17. 确认 `serving/` 中不再存在三个旧文件；
18. 更新当前有效的 CLI 示例；
19. 不创建 `serving` 兼容 wrapper；
20. 执行全仓残留检查：

```bash
rg -n \
  "serving/(generate|run_model|convert_expert_cache)\.py|serving\.(generate|run_model|convert_expert_cache)" \
  . --glob '!reference/*.md'
```

生产代码和测试搜索结果必须为空。

### Stage 5：完整本地与远端验证

21. 运行语法检查；
22. 运行三个入口定向测试；
23. 运行完整测试；
24. 同步根目录三个入口到远端 `~/dsv4/`；
25. 删除远端 `~/dsv4/serving/` 中三个旧入口；
26. 验证根目录 `smoke_model.py`；
27. 验证根目录 `generate.py`；
28. 只读检查根目录 exporter CLI；
29. 确认远端任务资源释放和 exit=0。

## 7. 本地验证方案

### 7.1 语法检查

```bash
python -m compileall -q \
  generate.py \
  smoke_model.py \
  export_expert_cache.py \
  tests/cli/test_generate.py \
  tests/cli/test_smoke_model.py \
  tests/cli/test_export_expert_cache.py
```

### 7.2 定向测试

```bash
pytest -q \
  tests/cli/test_generate.py \
  tests/cli/test_smoke_model.py \
  tests/cli/test_export_expert_cache.py
```

### 7.3 完整测试

```bash
pytest -q tests
```

### 7.4 CLI 检查

```bash
python generate.py --help
python smoke_model.py --help
python export_expert_cache.py --help
```

验收要求：

- 三个命令从仓库根目录直接可运行；
- 不要求额外设置项目根目录 `PYTHONPATH`；
- help 参数与迁移前一致；
- generate 不出现 `--tokenizer-path`、`--weight-index`、`--encoding-path` 或 `--backend`；
- smoke_model/exporter 不出现 `--weight-index`。

## 8. Ascend 远端验证方案

### 8.1 同步

将以下文件同步到远端根目录 `~/dsv4/`：

```text
generate.py
smoke_model.py
export_expert_cache.py
```

删除远端旧入口：

```text
~/dsv4/serving/generate.py
~/dsv4/serving/run_model.py
~/dsv4/serving/convert_expert_cache.py
```

### 8.2 smoke_model smoke

```bash
python smoke_model.py \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  -p a2a3 -d {} --max-layers 1 --no-head -s 1 --decode-steps 1
```

验收：prefill/decode shape、dtype、finite 正常，NPU lock 释放，任务 exit=0。

### 8.3 generate smoke

```bash
python generate.py \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  --prompt hello --max-new-tokens 1 \
  -p a2a3 -d {}
```

验收：能够加载顶层 tokenizer、仓内 official encoding、模型权重和 head，并输出一个 token，任务 exit=0。

### 8.4 exporter CLI

```bash
python export_expert_cache.py --help
```

只做 CLI 检查，不重新转换或改写 516 GiB packed cache。

## 9. 风险与处理

### 9.1 外部命令 breaking change

所有仍使用 `python serving/...` 或 `python -m serving...` 的脚本会失败。这是本次有意的 breaking change，
不增加兼容入口；调用方必须切换到根目录命令。

### 9.2 测试模块名变化

测试从 package import 改为根模块 import。由于 pytest 从仓库根目录运行，根目录已在 `sys.path`，无需修改
`tests/conftest.py`。

### 9.3 远端残留旧入口

如果只同步新文件而不删除远端旧文件，两个路径都可运行并可能产生版本漂移。远端验证前必须删除三个旧
入口，并且所有后续命令只使用根目录版本。

### 9.4 文档历史路径

历史方案保留旧路径是为了记录当时结构；只更新当前仍会复制执行的命令，避免大范围文档修改掩盖本次代码
迁移。

## 10. 验收标准

全部满足后才视为迁移完成：

- 根目录存在 `generate.py`、`smoke_model.py` 和 `export_expert_cache.py`；
- `serving/` 不再存在同名文件；
- 不存在兼容 wrapper 或 re-export；
- 三个测试文件直接 import 根模块；
- 三个根目录入口 `--help` 正常；
- CLI 参数和 checkpoint 约束不变；
- 定向测试和完整测试通过；
- 非历史生产代码/测试不存在旧路径引用；
- 远端只使用根目录入口；
- smoke_model 和 generate NPU smoke 均 exit=0；
- exporter 只读 CLI 检查通过，未改写 packed cache。
