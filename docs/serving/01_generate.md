# 文本生成入口

[`generate.py`](../../generate.py) 是完整的文本生成入口，负责 prompt 编码、43 层 prefill、逐 token decode、采样、EOS 处理、文本解码和生成统计。

## 功能定位

该入口面向真实文本输入，始终使用 `FLASH_CONFIG.n_layers` 执行全部模型层并运行 head。它创建一个 [`DeviceRuntime`](../../serving/device_runtime.py)，注入 [`DeepSeekV4Runner`](../../serving/runner.py)，但不在入口内实现模型计算或设备内存管理。

与其他入口相比：

- [`smoke_model.py`](../../smoke_model.py) 使用随机 token，可裁剪层数或跳过 head，主要用于运行时验收。
- [`export_expert_cache.py`](../../export_expert_cache.py) 只在 Host 侧准备磁盘 expert cache，不执行推理。

## 前置条件

`--checkpoint` 指向的目录必须通过 [`validate_checkpoint_directory()`](../../serving/checkpoint.py) 校验，并能由 `AutoTokenizer.from_pretrained()` 和 weight loader 读取。`--expert-cache-dir` 可省略；省略时 routed expert 权重回退到原始 checkpoint。

Prompt encoding 直接使用仓库内的 [`official/encoding_dsv4.py`](../../official/encoding_dsv4.py)，支持 `chat` 和 `thinking` 两种模式。

## 基本用法

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --prompt "你好" \
  --max-new-tokens 10
```

也可以通过 `--prompt-file` 读取 UTF-8 文件。入口不从 stdin 读取 prompt。

## 命令行参数

| 参数 | 类型/默认值 | 作用与约束 |
|---|---|---|
| `--checkpoint` | `str`，必填 | checkpoint 目录 |
| `--expert-cache-dir` | `str`，默认 `None` | 可选 packed BF16 expert cache 目录 |
| `--prompt` | `str`，默认 `None` | 直接提供用户 prompt |
| `--prompt-file` | `Path`，默认 `None` | 原样读取 UTF-8 prompt 文件 |
| `--thinking-mode` | `chat` / `thinking`，默认 `chat` | 传给官方 `encode_messages()` 和可选 completion parser |
| `--max-new-tokens` | `int`，默认 `10` | 最大生成 token 数，必须非负 |
| `--temperature` | `float`，默认 `0.0` | `<=0` 使用 greedy argmax；`>0` 使用随机采样 |
| `-p`, `--platform` | `str`，默认 `a2a3` | 传给 PyPTO `RunConfig` |
| `-d`, `--device` | `int`，默认 `0` | 单个 `ChipWorker` 使用的设备编号 |
| `--seed` | `int`，默认 `33377335` | 设置 PyTorch 随机种子 |
| `--include-eos` | flag，默认关闭 | EOS 命中时将 EOS id 保留在生成 token 列表中 |
| `--parse-eos` | flag，默认关闭 | 保留 EOS 解码结果并用官方 parser 提取回复正文 |
| `--skip-special-tokens` | flag，默认关闭 | 未启用 `--parse-eos` 时传给 tokenizer decode |
| `--enable-l2-swimlane` | flag，默认关闭 | 设置 `RunConfig.enable_l2_swimlane` |
| `--keep-prefill-routed-staging` | flag，默认关闭 | 跨 step 保留 prefill routed staging 的设备分配 |
| `--profile` | flag，默认关闭 | 输出 Serving 分段 profile |
| `--verbose-layer-log` | flag，默认关闭 | 输出逐层执行和有限值诊断 |

入口不暴露层数、head 或最大序列长度参数。runner 固定使用全部 43 层、head 和 `DEFAULT_MAX_SEQ_LEN=4096`。

## 执行流程

```text
校验 checkpoint
    │
    ├── 加载 tokenizer 与官方 encoding helpers
    ├── 解析 --prompt / --prompt-file
    ├── 将 user message 编码为 prompt ids
    ├── 创建 DeviceRuntime 与 DeepSeekV4Runner
    │
    ├── runner.prefill([1, prompt_len])
    ├── 从 logits 选择第一个 token
    ├── runner.decode([1, 1], start_pos=prompt_len)
    ├── 重复采样与 decode，直到数量上限或 EOS
    │
    ├── runner.close()
    ├── tokenizer.decode()
    └── 输出用户文本、回复和统计
```

`generate_ids()` 先用完整 prompt 调用一次 prefill。prefill logits 直接用于选择第一个生成 token；只有仍需继续生成时，才把刚生成的 token 作为 `[1, 1]` 输入调用 decode。每次 decode 的 `start_pos` 从 `prompt_len` 开始递增。

### Token 选择

- `temperature <= 0`：对 logits 执行 `argmax`。
- `temperature > 0`：先除以 temperature，计算 FP32 softmax，再使用 exponential race 形式采样。
- 命中 `eos_id`：默认终止且不把 EOS 加入返回列表；`--include-eos` 或 `--parse-eos` 会保留 EOS。

## 输入与输出

Prompt ids 在 Host 上转换为连续 `torch.int64` tensor，接受 `[S]` 或 `[1, S]`，最终规范为 `[1, S]`。Prompt 不能为空，且：

```text
prompt_tokens + max_new_tokens <= 4096
```

`--prompt` 和 `--prompt-file` 至少指定一个。两者同时指定时不报错，向 stderr 输出 warning，并以 `--prompt` 为准。

程序输出：

- `User`：原始 prompt；
- `AI`：tokenizer 解码并可选经过 EOS parser 处理的回复；
- `prompt_tokens` 与 `generated_tokens`；
- `elapsed_s`：仅生成循环耗时，不包含 tokenizer 和 runner 初始化；
- `output_tps = generated_tokens / elapsed_s`。

## 资源与性能选项

入口负责创建 runtime，runner 负责在 `close()` 中依次关闭 runtime 和 weight loader。runner 初始化失败时，入口直接关闭已创建的 runtime；生成过程中发生异常时，`finally` 仍会调用 `runner.close()`。

`--profile`、`--verbose-layer-log`、`--enable-l2-swimlane` 和 `--keep-prefill-routed-staging` 的运行时语义见 [`Serving 总览`](README.md)。

## 错误处理

- 缺少 `--checkpoint` 或两个 prompt 参数均缺失时，argparse 终止。
- prompt 文件不存在或无法按 UTF-8 读取时，文件读取异常向上传递。
- `max_new_tokens < 0`、prompt 为空或上下文总长度超过 `4096` 时，生成前抛出 `ValueError`。
- 输入 token shape 不是 `[S]` 或 `[1, S]` 时拒绝执行。
- checkpoint、tokenizer、runtime 或 runner 初始化失败时不进入生成循环。

## 验证方法

### CLI 单元测试

```bash
pytest -q tests/cli/test_generate.py
```

该测试覆盖 prompt 来源、官方 encoding helper、参数拒绝、greedy 选择、EOS、长度校验、prefill/decode 顺序、固定 runner 配置、统计输出和异常清理。

### NPU 验证

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --prompt "你好" \
  --max-new-tokens 2
```

验收时确认能够完成 tokenizer、43 层 prefill、head、采样和文本解码，并输出生成统计；首 token 未命中 EOS 时，该命令还会执行一个 decode。模块数值精度按 [`docs/models`](../models/README.md) 的方法独立验收。

## 代码索引

| 内容 | 位置 |
|---|---|
| 入口实现 | [`generate.py`](../../generate.py) |
| 官方 prompt encoding | [`official/encoding_dsv4.py`](../../official/encoding_dsv4.py) |
| 整模 runner | [`serving/runner.py`](../../serving/runner.py) |
| CLI 测试 | [`tests/cli/test_generate.py`](../../tests/cli/test_generate.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`冒烟验证入口`](02_smoke_model.md)
- [`Expert cache 导出`](03_export_expert_cache.md)
- [`整模 Runner`](05_runner.md)
