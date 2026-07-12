# Prefill Routed Expert Prefetch 优化方案

## 1. 结论

Stage 7 已基于最终 packed cache 的 43 层实测 profile 重新评估。每层 Host routed pack 已从旧格式的
约 3.5–3.8 秒降至约 365–366 毫秒，完整 43 层只剩约 15.7 秒 Host pack 时间。

Depth=1 Host-only prefetch 的新理论上限：

```text
S=1 cold       最多节省 15.400 s，prefill 降低 13.12%
S=1024 cold    最多节省 15.392 s，prefill 降低  9.01%
```

考虑下一层 12 GiB clone 与当前层 12 GiB H2D 的 Host 内存带宽竞争，现实收益预计约为：

```text
S=1       4%–9%
S=1024    2%–6%
```

同时 Host anonymous peak 增加一份 12 GiB pack，完整模型最坏 RSS 会从约 549 GiB 上升到约 561 GiB。
因此 Stage 7 的最终决策是：**当前不默认实现 Host prefetch**。它仍是一个可行但收益有限的可选优化，只有
在业务明确需要继续降低约 5% TTFT、服务器 Host 内存有足够余量，并接受专门的多轮 A/B 时才值得实施。

下文 V1/V2 名称保留为 packed cache 完成前的历史方案；以 8.8 节的 Stage 7 数据和第 12 节最终决策为准。

## 2. 当前串行路径

当前 `DeepSeekV4Runner._run_prefill_block()` 每层严格执行：

```text
get_layer_moe_routed_pack(layer L)
  → 从三个 packed mmap tensor 各 clone 一次
  → 得到 w1_t/w2_t/w3_t Host pack
  → WorkerBackend.materialize()
  → 同步 H2D 到三块 staging buffer
  → ChipWorker.run()
  → 处理下一层
```

对应代码边界：

- `serving/runner.py::_prefill_block_values()` 同步调用 `get_layer_moe_routed_pack()`；
- `serving/backends/worker_backend.py::_materialize_staging()` 调用 pool acquire/copy；
- `serving/backends/device_pool.py::copy_to()` 同步调用 `ChipWorker.copy_to()`；
- `serving/backends/worker_backend.py::run()` 同步调用 `ChipWorker.run()`。

当前只有一组 routed staging。不同层会覆盖复用相同的三个 slot，但 Host load、H2D 和 kernel 之间没有
流水并行。

## 3. 数据量

每层完整 routed expert pack：

```text
w1_t = 256 × 4096 × 2048 × BF16 = 4 GiB
w2_t = 256 × 2048 × 4096 × BF16 = 4 GiB
w3_t = 256 × 4096 × 2048 × BF16 = 4 GiB
合计                                  = 12 GiB
```

完整 43 层：

```text
12 GiB × 43 = 516 GiB
```

这不表示同时占用 516 GiB 内存；当前三块 device staging 按层覆盖，设备侧同时只保留约 12 GiB。
516 GiB 表示完整 prefill 期间累计构建和传输的数据量。

## 4. V1：Host Routed Pack Prefetch

### 4.1 目标流水

使用 depth=1 的后台预取：

```text
主线程：  H2D(L) ── kernel(L) ── H2D(L+1) ── kernel(L+1)
后台：              load/build(L+1)          load/build(L+2)
```

稳态单层耗时从近似：

```text
host_load_build + H2D + kernel
```

变为：

```text
max(host_load_build, H2D + kernel)
```

可掩盖的内容包括：

- safetensors expert cache 读取；
- 256 个 expert tensor 的 clone/materialize；
- 把 expert tensor 复制到完整 `w1_t/w2_t/w3_t` Host pack；
- contiguous/layout 整理。

V1 不改变 kernel、TensorSpec、DeviceTensor 或 state 语义。

### 4.2 建议新增文件

```text
serving/routed_expert_prefetcher.py
tests/test_routed_expert_prefetcher.py
```

`RoutedExpertPrefetcher` 建议职责：

- 只维护一个后台线程；
- depth 固定为 1，同时最多持有下一层 pack；
- 返回 `Future[MoERoutedPackWeights]` 或等价的窄接口；
- 后台异常在主线程 `result()` 时原样抛出；
- close 时取消尚未开始的任务，并等待正在执行的任务结束；
- 不接触 ChipWorker、DeviceTensor、state 或 kernel dispatch。

不要使用无界 executor 或一次提交剩余 42 层，否则会快速耗尽 Host 内存。

### 4.3 WeightLoader 调整

当前 WeightLoader 维护共享 safetensors handles、layout cache 和 profile counters，不能直接假设这些对象
支持无锁并发访问。

建议优先采用以下边界：

- prefetcher 为 routed expert cache 使用独立 file handles；
- Host pack 构建复用 WeightLoader 的 shape/dtype 校验和 expert cache path 规则；
- 不让后台线程访问普通 fixed layout cache；
- profile counters 按 prefetch task 单独记录，再由主线程汇总。

备选方案是在 WeightLoader 的 file-handle/cache 路径增加锁，但不得让大 tensor clone/copy 全程持锁，
否则会阻塞主线程普通权重访问并抵消预取收益。

### 4.4 Runner 调度

伪代码：

```python
future = prefetcher.submit(0)
for layer_id in range(max_layers):
    routed = future.result()
    if layer_id + 1 < max_layers:
        future = prefetcher.submit(layer_id + 1)

    values = build_prefill_values(layer_id, routed=routed)
    bindings = backend.materialize(specs, values)

    del values, routed
    outputs = backend.run(case, specs, bindings)
```

`_prefill_block_values()` 需要拆分为：

- 构建 aux、普通 fixed weights、shared experts 和 state bindings；
- 接收已构建的 routed pack，而不是内部同步加载 routed pack。

Runner 仍只处理 backend-neutral 的模型编排，不 import WorkerBackend 或 DeviceTensor。

### 4.5 Host 内存生命周期

单个 Host routed pack 约 12 GiB。后台预取下一层时，必须避免当前层 pack 因局部变量继续存活：

1. `backend.materialize()` 同步 H2D 返回后，`WorkerKernelBindings` 中只保留 device tensors/leases；
2. Runner 在 `backend.run()` 前删除当前 `values` 和 routed pack 引用；
3. 后台最多持有下一层一个 12 GiB pack；
4. 异常路径和 close 必须释放 future result。

否则当前层和下一层 Host pack 可能同时占用约 24 GiB，且 CPU allocator 还可能保留已释放内存。

## 5. V2：异步 H2D 与 Kernel 双缓冲（暂不实施）

本节只保留未来设计背景，不属于当前实现计划。当前 PyPTO 没有公开的异步 copy/event 接口；在框架
能力落地并独立验证前，不修改 DeviceBufferPool、Backend protocol 或 WorkerBackend 来实现 V2。

### 5.1 目标流水

```text
Device staging A：kernel(L)
Device staging B：async H2D(L+1)
完成后等待 event 并交换 A/B
```

需要两组 staging：

```text
A: w1_t / w2_t / w3_t = 12 GiB
B: w1_t / w2_t / w3_t = 12 GiB
```

理想稳态单层耗时接近：

```text
max(host_load_build, H2D, kernel)
```

而不是三者相加。

### 5.2 当前 runtime 阻塞

对远端当前 PyPTO 源码的只读检查确认：

- `ChipWorker.run()` 明确为同步执行；
- `ChipWorker.copy_to()` 调用返回时 copy 已完成；
- 当前公开接口没有 `copy_to_async`、copy event 或 stream wait；
- Host tensor 只要求活到同步 `copy_to()` 返回，没有异步 ownership contract。

因此不能只用 Python 后台线程并发调用当前同一个 ChipWorker 的 `copy_to()` 和 `run()`。这既没有公开的
并发保证，也无法表达“kernel 使用 A、copy 写 B、下层运行前等待 B”的设备事件依赖。

V2 需要 PyPTO 提供类似能力：

```python
ticket = worker.copy_to_async(dst, src, nbytes, stream=copy_stream)
worker.wait(ticket)
worker.run(compiled, ...)
```

或者允许向同一 ChipWorker 提交带 copy/compute/event 依赖的 runtime DAG。

不要通过创建第二个独立 ChipWorker 规避接口限制：DeviceTensor 属于创建它的 worker 地址空间，跨 worker
传递会破坏 ownership；复制全部 fixed weights 到第二个 worker 也会显著增加显存并改变当前 backend
模型。

### 5.3 Pinned Host Buffer

真正异步 H2D 需要 Host pack 使用 pinned/registered memory，并明确保持到 copy event 完成。当前
expert-cache 路径返回普通 contiguous CPU tensor；V2 需要：

- 预先分配两组 pinned Host pack；
- expert tensor 直接写入 inactive Host pack；
- async copy ticket 完成前不得覆盖或释放该 pack；
- pinned allocation 同样使用固定深度，不允许按层累计。

### 5.4 Backend 接口边界

异步 staging 必须留在 Backend protocol 内，不把 DeviceTensor 或 ChipWorker 暴露给 Runner。建议新增
backend-neutral 的 staging ticket 能力，例如：

```python
ticket = backend.prefetch_staging(host_staging_values)
bindings = backend.materialize(specs, values, staging_ticket=ticket)
```

WorkerBackend 内部负责：

- 选择 inactive device slot；
- 提交 async copy；
- 保存 Host buffer ownership；
- 在 materialize/run 前等待 copy event；
- kernel 完成后把旧 active slot 归还；
- 异常和 close 时等待/取消未完成传输并释放两组 slot。

Runner 只能持有 opaque ticket，不得判断具体 backend 或读取 device handle。

## 6. Device 内存预算

当前 S=4096 静态峰值：

```text
fixed non-routed weights              14.672 GiB
state/cache double buffers             0.088 GiB
single routed staging + dynamic       12.573 GiB
合计                                  27.33 GiB
```

V2 增加第二组 routed staging：

```text
27.33 GiB + 12.00 GiB = 39.33 GiB
```

64 GB 十进制设备容量约为 59.60 GiB，理论余量约 20.27 GiB，供 CANN/PyPTO runtime、compiled program、
workspace、allocator 碎片和异步重叠使用。静态预算可行，但必须在真实 S=4096 prefill 上测量；不能仅凭
静态值判定安全，尤其需要关注已有偶发 507018。

## 7. 与 `--keep-prefill-routed-staging` 的关系

当前参数只控制 prefill step 结束后是否保留一组 12 GiB routed staging allocation。它不会：

- 创建 A/B 双缓冲；
- 预取下一层 Host pack；
- 减少每层 12 GiB H2D；
- 让 H2D 与 kernel 并行。

新的优化不应改变该参数的既有语义。当前 V1 使用独立参数：

```text
--prefetch-routed-experts
```

不新增 async/double-buffer 选项。未来只有在 PyPTO 提供并验证 async copy/event 后，才重新评估 V2。

## 8. Profile 与收益验收

### 8.1 当前 profile 口径

最终 packed cache 已完成无 L2 swimlane 的完整 43 层基线采集。当前指标：

- `layer.values.routed_pack`：完整 Host routed pack load/build 时间；
- `expert_cache.routed_pack`：三个完整 packed tensor 的 clone 时间；
- `layer.materialize`：当前层所有 device materialize 时间；
- `layer.kernel`：分别报告 compile、run 和 cache hit；
- `prefill.total`：完整 prefill wall time。

当前 `layer.materialize` 还包括 fixed weight 首次上传、aux/input upload 和 staging acquire，因此它是 routed
H2D 的保守时间上界，不是纯 `STAGING_ROUTED` copy 时间。S=1/S=1024 的 materialize 平均值接近，且
每层 routed pack 固定为 12 GiB，因此当前数据已足够估算 V1 可利用的流水窗口；实现 V1 时再补充按
allocation category 细分的 copy profile。

8.2–8.7 节保留 packed cache 完成前的旧基线、旧收益模型与未来异步 H2D 背景，不再作为当前实现决策；
当前结论以 8.8 节为准。

后续目标指标：

```text
routed.expert_cache_load
routed.host_pack_build
routed.wait_prefetch
routed.h2d
layer.kernel
```

### 8.2 43 层无-swimlane 实测基线

共同配置：

```text
--max-layers 43
--no-head
--decode-steps 0
--profile
不开启 --enable-l2-swimlane
```

| 指标 | S=1 | S=1024 |
|---|---:|---:|
| prefill total | 222.737 s | 290.728 s |
| routed Host pack 总时间 | 150.291 s | 163.133 s |
| routed Host pack 平均/层 | 3.495 s | 3.794 s |
| routed Host pack min–max | 2.893–4.794 s | 2.851–8.879 s |
| `expert_cache.load` 总时间 | 74.839 s | 81.438 s |
| `expert_cache.load` 平均/层 | 1.740 s | 1.894 s |
| materialize/H2D 上界总时间 | 19.301 s | 19.628 s |
| materialize/H2D 上界平均/层 | 448.8 ms | 456.5 ms |
| materialize/H2D 上界 min–max | 317.1–933.0 ms | 322.2–975.2 ms |
| cold kernel elapsed 总时间 | 35.954 s | 90.648 s |
| cache-hit kernel run 平均 | 5.421 ms | 1,268.4 ms |

Host routed pack 约占当前总 prefill：

```text
S=1:    150.291 / 222.737 = 67.47%
S=1024: 163.133 / 290.728 = 56.11%
```

`expert_cache.load` 约占 Host pack 的一半；剩余时间主要来自 clone、contiguous 以及把 256 个专家复制到
完整 `w1_t/w2_t/w3_t` pack。V1 因此不能只预读 safetensors，必须让完整 Host pack build 在后台完成。

S=1024 的 cache-hit kernel 进一步拆分：

| Kernel | 平均 run |
|---|---:|
| SWA hash | 985 ms |
| HCA top-k | 962 ms |
| CSA top-k | 1,589 ms |

S=1 的 cache-hit kernel 只有约 5.4 ms；S=1024 则提供约 0.96–1.59 秒的真实计算窗口。

### 8.3 H2D 近似

只按 routed pack 的 12 GiB 和 S=1024 materialize 平均 456.5 ms 粗略计算：

```text
12 GiB / 0.4565 s ≈ 26.3 GiB/s
```

由于 materialize 还包含其他上传和管理开销，该值只能作为保守等效吞吐，不能视为精确链路带宽。
V1 不减少每层 H2D bytes；它只允许下一层 Host load/build 与当前层 materialize 和 kernel 重叠。

### 8.4 V1 理论收益模型

depth=1 pipeline 中，第一层 Host pack 无法预取。理想情况下，每个层间边界可节省：

```text
saving[L] = min(
    host_load_build[L + 1],
    materialize[L] + kernel[L]
)
```

完整理论节省为 `L=0..41` 的求和。该模型假设后台 load/build 不降低主线程 H2D/kernel 性能，因此是收益
上界；实际实现还会受到 Host 内存带宽、page cache、文件系统和 CPU copy 竞争影响。

S=1 冷启动实测理论值：

```text
baseline      222.737 s
ideal V1      192.548 s
saving         30.189 s
reduction       13.55%
speedup          1.157x
```

S=1 全部 kernel 已编译后的估算：

```text
baseline      ~187.0 s
ideal V1      ~167.8 s
saving         ~19.2 s
reduction       10.27%
speedup          1.114x
```

S=1 的 warm kernel 几乎不能掩盖 Host load，主要可利用窗口是每层约 0.45 秒 materialize/H2D。

S=1024 冷启动实测理论值：

```text
baseline      290.728 s
ideal V1      213.602 s
saving         77.127 s
reduction       26.53%
speedup          1.361x
```

S=1024 全部 kernel 已编译后的估算：

```text
baseline      254.673 s
ideal V1      182.394 s
saving         72.279 s
reduction       28.38%
speedup          1.396x
```

长 prompt 的 materialize + kernel 窗口约为 1.4–2.0 秒/层，能够覆盖明显更多的下一层 Host pack 构建，
因此 V1 的主要目标应是降低长 prompt TTFT，而不是 S=1 prefill。

### 8.5 实际收益目标

考虑 Host load/build 与 H2D 对 Host 内存带宽的竞争，以及后台线程、future、file-handle 管理开销，第一版
不以理论上界作为硬验收线。建议目标：

```text
S=1:    5%–10% prefill wall-time reduction
S=1024: 15%–25% prefill wall-time reduction
```

如果 S=1024 在相同配置的多轮 A/B 中稳定达到约 20%，即可认为 V1 达到较好收益。

S=1024 首轮 profile 在 layer 1 复现 507018，设备成功清理并释放锁；相同命令复测完成全部 43 层，输出
`(1, 1024, 4, 4096)`、BF16、finite、exit=0。该行为与既有偶发问题一致，性能 A/B 必须保留多轮复测，
不能用单次失败或单次成功代表稳定性。

### 8.6 V1 验收项

V1 核心验收：

- 第 1 层之后 `routed.wait_prefetch` 显著低于原同步 Host load/build；
- 完整 43 层 prefill wall time 明显下降；
- H2D bytes 不变，仍约 12 GiB/层；
- Host pack 并发深度始终不超过 1 个 future；
- Host 内存峰值受控，无后台线程或 file-handle 泄漏；
- 输出与当前 WorkerBackend 一致。

### 8.7 V2 未来参考指标

以下指标仅作为未来重新启动 V2 时的参考，不纳入当前验收：

- timeline 证明 H2D(L+1) 与 kernel(L) 实际重叠；
- 不能只以 Python 线程重叠作为证据；
- 完整 prefill H2D bytes 不变，但 H2D exposed time 显著下降；
- device peak 接近预算且不超过 64 GB；
- S=1、S=13、S=1024、S=4096 和完整 43 层均通过；
- 异常路径、event wait、slot swap 和 close 不泄漏 allocation。

### 8.8 Stage 7：最终 packed cache 重新评估

数据来源：

```text
S=1     task_20260712_030300_63703418429
S=1024  task_20260712_031200_72525616419
43 layers, no-head, decode-steps=0, profile, no L2 swimlane
```

实测分解：

| 指标 | S=1 | S=1024 |
|---|---:|---:|
| prefill total | 117.360 s | 170.903 s |
| Host routed pack 总计 | 15.746 s | 15.714 s |
| Host routed pack 平均/层 | 366.182 ms | 365.437 ms |
| Host routed pack min–max | 309.071–442.611 ms | 314.262–430.283 ms |
| materialize 总计 | 27.393 s | 29.670 s |
| materialize 平均/层 | 637.042 ms | 690.009 ms |
| materialize min–max | 409.249–1,351.169 ms | 590.568–1,455.770 ms |
| kernel elapsed 总计 | 36.293 s | 92.586 s |
| kernel run 总计 | 19.376 s | 73.574 s |
| compile 总计 | 16.912 s | 19.004 s |

沿用 depth=1 公式：

```text
saving[L] = min(host_pack[L + 1], materialize[L] + kernel[L])
```

结果中 42/42 个后续 Host pack 都能在理想模型中被当前层 materialize 单独完全覆盖，不需要额外依赖 kernel
窗口。第一层 Host pack 不可隐藏，因此最大节省恰好接近 layer 1–42 Host pack 之和：

| 场景 | Baseline | 最大节省 | Ideal | 降低 | 加速比 |
|---|---:|---:|---:|---:|---:|
| S=1 cold | 117.360 s | 15.400 s | 101.960 s | 13.12% | 1.151x |
| S=1 去 compile | 100.448 s | 15.400 s | 85.049 s | 15.33% | 1.181x |
| S=1024 cold | 170.903 s | 15.392 s | 155.512 s | 9.01% | 1.099x |
| S=1024 去 compile | 151.899 s | 15.392 s | 136.508 s | 10.13% | 1.113x |

该上限明显低于旧 per-expert cache 阶段 S=1024 的 26%–28%，不能继续使用旧目标作为实现依据。

#### Host 带宽竞争敏感性

当前 12 GiB packed clone 的等效 Host 带宽约为 32.8 GiB/s；materialize 的 12 GiB routed H2D 上界对应
约 17–19 GiB/s。两者重叠会同时读取 Host 内存，且 clone 还写入 12 GiB anonymous output，因此不能假设
后台 clone 对 materialize 完全无影响。

假设 prefetch 完全隐藏 Host pack，但使 materialize 总时间增加一定比例，净收益为：

| materialize 变慢 | S=1 净降低 | S=1024 净降低 |
|---:|---:|---:|
| 10% | 10.84% | 7.31% |
| 20% | 8.56% | 5.61% |
| 30% | 6.27% | 3.92% |
| 40% | 3.99% | 2.22% |
| 50% | 1.71% | 0.53% |

S=1/S=1024 的理论 break-even materialize slowdown 分别约为 57.5%/53.1%。因此实现大概率不会负收益，
但在敏感性分析假设的 20%–40% 带宽干扰范围内，S=1024 只剩约 2%–6% 收益。

#### 内存与工程成本

- 当前层完成同步 H2D 前仍持有 12 GiB Host pack；同时预取下一层必然把 anonymous peak 增加到约 24 GiB；
- 43 层 file-backed pages、固定 Host layout 和 runtime 不变，实测约 549 GiB RSS 上再增加约 12 GiB；
- 需要独立 Reader/file handles、bounded future、异常传播、Runner values 拆分以及 close 测试；
- H2D bytes 仍为 516 GiB，device staging 和 kernel 时间不减少；
- decode selected expert 无法跨层提前确定，不受该优化影响。

综合收益、内存和实现复杂度，Host prefetch 从“推荐实现”降级为“有明确 TTFT 需求时再实现”。

## 9. 测试计划

### 9.1 本地 Fake 测试

- depth=1，只允许当前 future 和下一层 pack；
- 第 `L+1` 层 load 与第 `L` 层 fake kernel 时间重叠；
- future 异常在主线程传播；
- 中途 kernel 失败时 future 被正确回收；
- close 幂等并终止线程；
- 43 层结果顺序不乱；
- Host pack 不进入 fixed cache；

当前不增加 V2 fake event 或 A/B device staging 测试。

### 9.2 远端验证

按以下顺序执行：

1. 1 层 prefill，确认无额外后台任务残留；
2. 5 层 prefill，采集同步与 prefetch 时间线；
3. 43 层 S=1 和 S=13，验证完整调度；
4. 43 层 S=1024，多轮复测偶发 507018；
5. 43 层 S=4096，采集 Host/device 峰值和完整 timeline；
6. 带 head 的完整 generate smoke，确认 prefill 后 decode 不受影响。

功能验证先不开启 L2 swimlane；性能复测再固定相同 swimlane 配置做 A/B。

## 10. 实施顺序

### Phase 1：旧格式 profile（历史基线已完成）

1. 已采集未预取的 5 层、43 层 S=1 和 43 层 S=1024 基线；
2. 已确认 Host routed pack 是主要 exposed time，S=1024 V1 理论上限约 26%–28%；
3. V1 实现时补充 Host pack build、prefetch wait 和按 category H2D 指标。

### Phase 2：Host prefetch（Stage 7 决定暂不启动）

以下仅作为未来重启时的实施清单：

4. 新增 bounded `RoutedExpertPrefetcher`；
5. 为 routed expert path 建立线程安全或独立 file-handle 边界；
6. 拆分 Runner prefill values 构建并接入 depth=1 pipeline；
7. 补齐异常、close、顺序和内存生命周期测试；
8. 远端运行 5 层和完整 43 层性能/精度验证。

### 后续非当前范围：V2 重新启动条件

当前计划在 Phase 2 完成后结束。只有同时满足以下条件，才另建方案重新启动 V2：

1. PyPTO 正式提供 `copy_to_async` 或等价的 copy/compute DAG；
2. runtime 提供明确的 event wait、Host buffer ownership 和异常清理 contract；
3. 独立 smoke 证明同一 ChipWorker 上 H2D 与 kernel 在设备 timeline 中真实重叠；
4. 双 routed staging 的约 39.33 GiB 静态峰值通过真实 S=4096 显存验证。

在此之前，不修改 DeviceBufferPool 增加 A/B routed slot，不扩展 Backend staging ticket，也不使用 Python
线程并发调用同步 `copy_to()`/`run()`。

## 11. 风险与边界

- Prefetch 只能隐藏延迟，不能减少 516 GiB 累计 Host pack/H2D 数据量；
- 如果 Host load/build 明显快于 kernel，V1 收益会受限；
- 如果 H2D 是主要 exposed time，V1 收益会受限；当前只记录该结论，不继续实施 V2；
- 后台线程不能访问非线程安全的共享 safetensors handle/cache；
- Host 和 device 双缓冲必须有严格上限，禁止按层累计；
- V2 不能在没有设备 timeline/event 证据时宣称实现了 copy/compute overlap；
- decode selected experts 依赖当前层 pre-MoE indices，不适用相同的跨层提前预取；
- 不应为该优化破坏 Runner/Backend 解耦或把 DeviceTensor 暴露到 WeightLoader。

## 12. 推荐决策

当前不实现 Host routed pack prefetch，Stage 7 以完成收益重估和记录决策结束。原因：

1. S=1024 cold 理论上限只有 9.01%，现实预计约 2%–6%；
2. S=1 理论上限 13.12%，现实预计约 4%–9%；
3. 增加约 12 GiB Host anonymous peak，完整模型最坏 RSS 接近 561 GiB；
4. 需要新的线程、Reader ownership、future 异常与 Runner 调度边界，复杂度与收益不再匹配。

如果未来业务确认 5% 左右 TTFT 也具有足够价值，可把本方案作为独立优化重新启动，并必须先做 5 层
memory-bandwidth A/B；只有 materialize slowdown 小于约 30% 且完整 wall time 符合预期，才扩大到 43 层。

异步 H2D 仍受限于 PyPTO 接口。当前同步 ChipWorker API 下，禁止用普通 Python 线程并发调用
`copy_to()` 和 `run()` 来伪造设备 copy/compute overlap。
