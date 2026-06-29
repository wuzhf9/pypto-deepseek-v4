# 项目目标

当前仓库的目标是使用 PyPTO 实现 DeepSeek V4 Flash 的权重加载和完整推理功能，并能够在 Ascend NPU 上推理出逻辑正确的句子。

## 核心目标

1. 正确加载 DeepSeek V4 Flash 模型权重。
2. 将权重转换为 PyPTO kernel 可使用的 bf16 计算路径。
3. 使用 PyPTO 实现 DeepSeek V4 Flash 的完整前向推理流程。
4. 在 Ascend NPU 上完成端到端推理，并生成逻辑正确的句子。

## 主要参考

模型结构和计算逻辑参考：

- `../deepseek_v4_flash/inference/model.py`
- `../deepseek_v4_flash/inference/kernel.py`
- `../deepseek_v4_flash/inference/low_vram_attention.py`
- `../deepseek_v4_flash/inference/low_vram_moe.py`
- `../deepseek_v4_flash/inference/low_vram_executor.py`
- `reference/dsv4_flash_arch.md`

PyPTO 权重加载、kernel 编译和运行流程参考：

- `../pypto-serving/examples/model/qwen3_14b/runner/npu_executor.py`
- `../pypto-serving/examples/model/qwen3_14b/runner/npu_runner.py`
- `../pypto-serving/examples/model/qwen3_14b/runner/qwen3_l3_dispatch.py`
- `../pypto-serving/python/core/model_loader.py`
- `reference/pypto_serving_qwen3.md`

## bf16 实现约束

Ascend NPU 当前不具备 fp8/fp4 计算路径，因此 PyPTO 实现应参考 `../deepseek_v4_flash` 的 bf16 版本。

需要保持的行为：

- 模型结构和数学计算逻辑与 bf16 版本一致。
- FP8/FP4 checkpoint 权重加载后转换或反量化为 bf16。
- PyPTO kernel 使用 bf16 权重和 bf16 activation 进行计算。

需要移除或不实现的低精度路径：

- `act_quant`
- `fp4_act_quant`
- `rotate_activation`
- `fp8_gemm`
- `fp4_gemm`
- 其他仅服务于 fp8/fp4 量化推理的模拟或 kernel 路径

## 非目标

当前阶段不追求推理性能。

不需要优先实现：

- paged attention
- dynamic batching
- 高吞吐 serving
- 多请求调度
- KV cache 复杂分页管理
- fp8/fp4 kernel
- 与官方低精度实现完全一致的数值误差特性

## 实现优先级

优先级从高到低：

1. 端到端正确性：能加载真实权重并生成逻辑正确的句子。
2. 计算逻辑完整性：DeepSeek V4 Flash 的 HC、Attention、compressed KV、Indexer、MoE、Head 等结构应与 bf16 版本一致。
3. PyPTO 接入完整性：权重整理、host wrapper、kernel 编译、runtime runner 和推理循环可稳定运行。
4. 代码可读性和可验证性：实现应便于逐模块对齐 bf16 参考版本。
5. 性能优化：当前阶段暂不考虑。

## 当前判断

后续实现时应将 `../deepseek_v4_flash` 的 bf16 版本视为数学参考实现，将 `../pypto-serving` 的 Qwen3 路径视为 PyPTO 工程参考实现。

最终验收标准是：在 Ascend NPU 上通过 PyPTO 加载 DeepSeek V4 Flash 权重，执行完整推理流程，并输出逻辑正确的句子。
