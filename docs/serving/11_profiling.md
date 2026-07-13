# Serving Profiling

[`ProfileRecorder`](../../serving/profiler.py) 为 runner 提供轻量 wall-time 事件，并把 DeviceRuntime 最近一次 compile/run 统计和 weight loader 聚合统计统一输出为 `[PROFILE]` 日志。

## 模块定位

Profiler 只在启用时测量和打印，不保存全局 trace、不汇总跨层百分比，也不控制 Kernel。Runner 决定事件边界；runtime 和 weight loader 暴露底层字段。

`generate.py` 与 `smoke_model.py` 的 `--profile` 会启用 runner profiler 和 loader 内部统计。`export_expert_cache.py --profile` 不创建 `ProfileRecorder`，而是直接打印 `[PACKED_CACHE_PROFILE]` loader summary。

## 核心接口

| 接口 | 输出内容 |
|---|---|
| `timer(name, **fields)` | context manager 整体 Python wall time |
| `runtime_timer(name, runtime, **fields)` | wall time + 最近一次 compile/run/cache-hit |
| `record_weight_loader(name, loader, **fields)` | layout cache bytes + loader 各子项 count/time |
| `block_profile_fields()` | layer、mode、ratio、route、block shape 和 kernel fields |
| `block_shape_from_kernel()` | 从 Kernel 名去掉 `block_`、`_fwd` 和 prefill/decode suffix |

即使被测代码抛出异常，`timer` 和 `runtime_timer` 仍在 `finally` 中打印已经经过的 wall time。

## Runner 事件层级

### 整体与首尾 Kernel

| 事件 | 范围 |
|---|---|
| `prefill.total` | 完整 prefill public 调用 |
| `decode.total` | 一个 token 的完整 decode public 调用 |
| `embedding.total` | embedding weight、materialize 和 Kernel |
| `embedding.weight` | embedding Host weight 获取 |
| `embedding.materialize` | Device bindings 构造 |
| `embedding.kernel` | compile + worker run |
| `head.total` / `head.weight` / `head.materialize` / `head.kernel` | 对应 head 阶段 |

### Prefill Layer

| 事件 | 范围 |
|---|---|
| `layer.total` | 一层完整 prefill |
| `layer.values` | aux 和全部权重 value 构造 |
| `layer.weight_loader` | 本层 loader 子项汇总 |
| `layer.materialize` | values 到 DeviceTensor bindings |
| `layer.kernel` | Block compile + run |
| `layer.state_update` | state commit |

`layer.values` 内还按 `aux`、`hc`、`attn`、`gate`、`ffn_norm`、`shared`、`routed_pack` 以及适用的 compressor/indexer 分项。

### Selected Decode Layer

Decode 分别输出：

- `layer.selected_decode.pre_values`
- `layer.selected_decode.pre_weight_loader`
- `layer.selected_decode.pre_materialize`
- `layer.selected_decode.pre_kernel`
- `layer.selected_decode.state_update`
- `layer.selected_decode.post_values`
- `layer.selected_decode.post_weight_loader`
- `layer.selected_decode.post_materialize`
- `layer.selected_decode.post_kernel`

Post values 包含 indices D2H、selected expert 构造和 shared expert 获取，因此分析 decode 权重路径时应查看 post weight-loader 事件，而不是只看 pre-MoE。

## 公共字段

Layer 事件携带：

| 字段 | 含义 |
|---|---|
| `layer` | 层号 |
| `mode` | `prefill` 或 `decode` |
| `ratio` | `0`、`4` 或 `128` |
| `hash_route` | 是否 hash routing |
| `block_shape` | 规范化 Block family |
| `kernel` | 实际 Kernel case name |

Runtime timer 追加：

| 字段 | 含义 |
|---|---|
| `compile_ms` | `case.fn.compile()` 查找/编译阶段时间 |
| `run_ms` | `worker.run()` wall time |
| `cache_hit` | 本次 compile cache 是否命中 |

事件开头的 wall time 包含 context manager、compile、run 和 Python 调度开销，通常不等于 `run_ms`。

## Weight Loader 子项

`record_weight_loader()` 输出当前 `layout_cache_bytes`，并附加自上次 `reset_profile_stats()` 后的 loader 事件：

- raw/scale load；
- FP8/FP4 dequant；
- dtype cast 与 transpose；
- Host layout cache hit/miss；
- expert cache full-pack 或 selected-slice copy；
- selected-experts build。

格式为 `<name>=<elapsed_ms>ms/<count>`。Elapsed 是该事件累计时间，count 是发生次数，不是平均时间。

## 其他性能输出

| 输出 | 来源 | 含义 |
|---|---|---|
| `[stats] elapsed_s/output_tps` | `generate.py` | 整个 token 生成循环，不含初始化 |
| `[PACKED_CACHE] build/save` | `export_expert_cache.py` | 单层 Host 构造与磁盘保存 wall time |
| `[PACKED_CACHE_PROFILE]` | exporter + weight loader | 单层 loader 子项累计 |
| `runtime.pool_stats` | `DeviceRuntime` API | allocation/reuse/H2D/D2H/显存统计，不自动打印 |
| `--verbose-layer-log` | runner | 逐层信息和有限值检查，会引入额外 D2H |

这些统计范围不同。特别是 `output_tps` 不能用于分解 layer 或 Kernel，`run_ms` 也不包含 Host weight load。

## 使用方法

### Smoke profile

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1 \
  --profile
```

### 完整生成 profile

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --prompt "你好" \
  --max-new-tokens 2 \
  --profile
```

首次运行可能同时包含 Host layout miss、fixed-weight H2D 和 compile miss。分析 warm decode 时应至少区分这些首次成本，并确认相关 `cache_hit` 与 loader hit/miss 字段。

## 约束与解释原则

- 所有计时基于 `time.perf_counter()` 的 Python wall time，不是设备侧硬件 counter。
- `compile_ms` 在 cache hit 时仍包含字典查找等极小 wall time，不固定为零。
- Loader 统计只在 `profile=True` 时记录；关闭 profile 时 cache 行为仍存在。
- `layout_cache_bytes` 是 Host runtime layout bytes，不是 NPU 显存。
- ProfileRecorder 当前不自动打印 pool stats。
- 不启用 `verbose_layer_log` 的结果更接近无逐层 D2H 诊断开销的路径。

## 验证方法

### Host 侧事件边界

```bash
pytest -q tests/serving/test_runner.py tests/serving/test_weight_loader.py tests/serving/test_device_runtime.py
```

这些测试验证 profiler 所依赖的 runner 调用边界、loader 计数与 cache hit/miss、runtime compile cache 和最近一次 run 字段的数据来源。

### NPU profile 验证

使用上述 smoke profile 命令。验收时确认：

- embedding、layer prefill、selected decode pre/post 和可选 head 事件完整；
- runtime timer 包含 `compile_ms`、`run_ms`、`cache_hit`；
- weight-loader 事件包含 `layout_cache_bytes` 和实际发生的子项；
- 日志能够按 layer、ratio、route 和 kernel 区分 Block。

本文只规定事件解释和验证方法，不保存某次采集的耗时或性能结论。

## 代码索引

| 内容 | 位置 |
|---|---|
| ProfileRecorder | [`serving/profiler.py`](../../serving/profiler.py) |
| Runner instrumentation | [`serving/runner.py`](../../serving/runner.py) |
| Weight loader counters | [`serving/weight_loader.py`](../../serving/weight_loader.py) |
| Runtime fields | [`serving/device_runtime.py`](../../serving/device_runtime.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`整模 Runner`](05_runner.md)
- [`Weight Loader`](07_weight_loader.md)
- [`Device Runtime`](10_device_runtime.md)
