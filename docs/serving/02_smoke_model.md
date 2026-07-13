# 整模冒烟验证入口

[`smoke_model.py`](../../smoke_model.py) 使用随机 token 执行可裁剪层数的 prefill 和 decode，用于验证 Serving runtime、runner 与模型 Kernel 的整合路径。

## 功能定位

该入口不加载 tokenizer，也不生成自然语言文本。它允许控制序列长度、模型层数、head 和 decode step，适合用较小范围定位整模集成问题。

与 [`generate.py`](../../generate.py) 不同，smoke 路径可以设置 `max_layers=0` 或 `--no-head`；与模型模块单测不同，它通过真实 `DeviceRuntime` 串联多层 Kernel 和设备状态。

## 前置条件

`--checkpoint` 必须通过统一 checkpoint 校验。`--expert-cache-dir` 可选；执行包含 MoE 的层时，未提供 cache 会回退到原始 checkpoint routed expert 权重。

## 基本用法

执行一层 prefill 和一个 decode step：

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1
```

## 命令行参数

| 参数 | 类型/默认值 | 作用与约束 |
|---|---|---|
| `--checkpoint` | `str`，必填 | checkpoint 目录 |
| `-p`, `--platform` | `str`，默认 `a2a3` | PyPTO platform |
| `-d`, `--device` | `int`，默认 `0` | 单卡设备编号 |
| `-s`, `--seq-len` | `int`，默认 `1` | 随机 prefill token 数，必须为正 |
| `--max-layers` | `int`，默认 `1` | 执行层数，runner 要求位于 `[0, 43]` |
| `--enable-l2-swimlane` | flag，默认关闭 | 传给 PyPTO `RunConfig` |
| `--keep-prefill-routed-staging` | flag，默认关闭 | 保留 prefill routed staging 设备分配 |
| `--no-head` | flag，默认关闭 | 返回 hidden tensor，不运行 head |
| `--decode-steps` | `int`，默认 `0` | prefill 后执行的 decode 次数，必须非负 |
| `--profile` | flag，默认关闭 | 输出 Serving profile |
| `--verbose-layer-log` | flag，默认关闭 | 输出逐层执行信息和有限值诊断 |
| `--expert-cache-dir` | `str`，默认 `None` | 可选 packed BF16 expert cache |
| `--seed` | `int`，默认 `0` | 随机输入和无 head decode 输入的 PyTorch seed |

## 执行流程

1. 校验 checkpoint、`decode_steps` 和总上下文长度。
2. 在 Host 上生成 shape `[1, seq_len]` 的随机 `int64` token。
3. 创建 `DeviceRuntime`，再以依赖注入方式创建 `DeepSeekV4Runner`。
4. 调用一次 `runner.prefill()`，检查输出全部有限并打印摘要。
5. 构造第一个 `[1, 1]` decode 输入。
6. 按 `decode_steps` 循环调用 `runner.decode()`，每步检查有限值。
7. 无论正常返回还是异常，都调用 `runner.close()`。

### Decode 输入来源

| 模式 | 下一步 token |
|---|---|
| 默认运行 head | 对 logits 执行 `argmax`，转换为 Host `[1, 1]` `int64` tensor |
| `--no-head` | hidden tensor 不能直接表示 token，因此重新生成随机 `[1, 1]` token |

## 输入与输出

随机 token 范围是 `[0, FLASH_CONFIG.vocab_size)`。Prefill 输入固定为 `B=1`，decode 输入固定为 `[1, 1]`。

每个阶段输出一条日志，包含输入 shape、输出 shape、dtype 和 `finite`。任一输出含 NaN 或 Inf 时返回状态码 `1`；所有阶段有限时返回 `0`。

运行 head 时输出 logits；`--no-head` 时输出最后一层或 embedding 产生的 hidden tensor。`max_layers=0` 仍会执行 embedding，并根据 `--no-head` 决定是否执行 head。

## 资源与性能选项

入口拥有 `DeviceRuntime` 的创建责任，runner 获得 runtime 后负责最终关闭。若 runner 构造失败，入口关闭 runtime；若 prefill/decode 失败，`finally` 关闭 runner。

`--profile` 用于分解 layer values、materialize、compile 和 kernel run 时间；`--verbose-layer-log` 会额外把逐层输出复制回 Host 检查有限值，因此不应把该模式的时间当作无诊断开销的性能结果。

## 错误处理

- `decode_steps < 0` 时拒绝执行。
- `seq_len + decode_steps > 4096` 时拒绝执行。
- `seq_len <= 0` 最终由 runner 的 prefill 输入校验拒绝。
- `max_layers` 不在 `[0, 43]` 时由 runner 构造函数拒绝。
- 缺少 checkpoint 必需文件、expert cache manifest 不兼容或权重缺失时向上传递异常。

## 验证方法

### CLI 单元测试

```bash
pytest -q tests/cli/test_smoke_model.py
```

该测试验证 runtime 在入口外创建并注入 runner、L2 配置传递、checkpoint 必填和构造失败时关闭 runtime。

### NPU 验证

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1
```

验收时确认 prefill 和 decode 均完成，输出 shape、dtype 符合 `run_head` 模式，并且所有阶段打印 `finite=True`。扩大 `--max-layers` 或 `--seq-len` 用于增加覆盖范围，不在稳定文档中记录某次执行结果。

## 代码索引

| 内容 | 位置 |
|---|---|
| 入口实现 | [`smoke_model.py`](../../smoke_model.py) |
| 整模 runner | [`serving/runner.py`](../../serving/runner.py) |
| CLI 测试 | [`tests/cli/test_smoke_model.py`](../../tests/cli/test_smoke_model.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`文本生成入口`](01_generate.md)
- [`整模 Runner`](05_runner.md)
- [`Device Runtime`](10_device_runtime.md)
