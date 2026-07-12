# Prefill/Decode 性能开销分析

## 1. 分析口径

本文基于最终 packed BF16 expert cache 实现（commit `a2b6ae3`）的完整 43 层实测 profile，配置为：

- Worker backend；
- `--max-layers 43 --no-head --profile`；
- 不开启 L2 swimlane；
- prefill 分别使用 `S=1` 和 `S=1024`；
- decode 使用 `S=1` prefill 后连续执行 3 个 decode step；
- 第一个 decode step 作为 cold decode，后两个 step 作为 warm decode。

对应远端任务：

- S=1 prefill：`task_20260712_030300_63703418429`；
- S=1024 prefill：`task_20260712_031200_72525616419`；
- 完整 decode A/B：`task_20260712_033628_92489621739`。

43 层的 `layer.total` 与层内子计时基本闭合，累计差值只有约 13～15 ms。因此，
`step.total - embedding.total - sum(layer.total)` 可以明确定位到最后一层之后的
`backend.export_output()` 和 `backend.end_step()`，本文统一记为 `Export + end_step`。当前 profile
没有继续拆分这两个调用，不能把该项全部归因于 D2H、设备同步或资源释放中的某一项。

## 2. Prefill 开销

| 阶段 | S=1 | 占比 | S=1024 | 占比 |
|---|---:|---:|---:|---:|
| Embedding | 2.629 s | 2.24% | 2.646 s | 1.55% |
| Host values/权重准备 | 30.939 s | 26.36% | 30.016 s | 17.56% |
| 其中 routed expert pack | 15.746 s | 13.42% | 15.714 s | 9.19% |
| Materialize/H2D | 27.393 s | 23.34% | 29.670 s | 17.36% |
| Kernel 总计 | 36.293 s | 30.92% | 92.586 s | 54.17% |
| ├ Kernel compile | 16.912 s | 14.41% | 19.004 s | 11.12% |
| └ Kernel run | 19.376 s | 16.51% | 73.574 s | 43.05% |
| Export + end_step | 20.090 s | 17.12% | 15.971 s | 9.35% |
| 总时间 | **117.360 s** | 100% | **170.903 s** | 100% |

### 2.1 S=1024

长序列 prefill 的最大开销已经是 kernel：

- kernel 总计占 54.17%；
- 排除首次 compile 后，kernel run 仍占 43.05%；
- Host values 和 materialize 分别占 17.56% 和 17.36%；
- routed expert Host pack 只占端到端时间的 9.19%。

因此，长序列 prefill 的首要瓶颈是模型计算。只优化 routed Host pack 的理论收益上限已经有限，和
Stage 7 中“不默认实现 Host prefetch”的结论一致。

### 2.2 S=1

S=1 冷启动中，kernel 总计以 30.92% 成为最大的聚合项，但其中包含 16.912 秒首次编译。排除一次性
compile 后，主要开销顺序变为：

1. Host values/权重准备：30.939 s；
2. Materialize/H2D：27.393 s；
3. Kernel run：19.376 s。

Host values 中 routed expert pack 为 15.746 秒，其余约 15.194 秒主要来自普通权重首次读取、layout
准备及相关 Host values 构建。因而短序列 warm prefill 更偏向 Host 数据准备和 H2D 瓶颈，而不是 kernel
计算瓶颈。

### 2.3 Prefill 收尾时间

`Export + end_step` 在 S=1/S=1024 下分别为 20.090 秒和 15.971 秒。Prefill 默认会在
`WorkerBackend.end_step()` 中释放三块 routed staging DeviceTensor，合计约 12 GiB，因此 staging 释放是
该区间的重要候选开销；但在补充独立计时前，不能断言全部时间都由释放产生。

`--keep-prefill-routed-staging` 可以避免在 prefill→decode 关键路径立即释放这三块 buffer，但会继续占用
约 12 GiB NPU 内存，并把实际释放成本推迟到 backend close。该参数主要改变延迟所在的位置，并不天然
消除完整进程生命周期中的释放成本。

## 3. Decode 开销

### 3.1 Cold decode

第一个 decode step 总计 41.850 秒，其中 Pre-MoE kernel 为 31.546 秒，占 75.38%。该阶段包含
selected-decode kernel 首次编译和 runtime 初始化，不代表稳态单 token 性能。

### 3.2 Warm decode

第 2、3 个 decode step 平均耗时约 7.020 秒/token：

| 阶段 | 平均时间 | 占比 |
|---|---:|---:|
| Embedding | 1.9 ms | 0.03% |
| Pre-MoE values | 20.5 ms | 0.29% |
| Pre-MoE materialize | 642.2 ms | 9.15% |
| Pre-MoE kernel | 156.6 ms | 2.23% |
| Selected expert Host 构建 | 1.935 s | 27.57% |
| └ 其中 selected slice/copy | 1.909 s | 27.20% |
| Post-MoE materialize/H2D | **2.150 s** | **30.62%** |
| Post-MoE kernel | 119.5 ms | 1.70% |
| Export + end_step | 1.978 s | 28.18% |
| 层内计时差值等其他小项 | 约 14 ms | 约 0.20% |
| 总时间 | **7.020 s** | 100% |

Warm decode 的最大已命名开销是 Post-MoE selected 权重 materialize/H2D，占 30.62%。其后依次是：

1. `Export + end_step`：28.18%；
2. selected expert Host slice/copy：27.20%；
3. Pre-MoE materialize：9.15%；
4. Pre/Post 两段 kernel 合计：约 3.93%。

这说明 warm decode 已经不是 kernel 计算瓶颈，而是 selected expert 权重的 Host 切片、H2D 和最后的
导出/同步/收尾路径占主导。

## 4. 当前最高占比与优化优先级

### Prefill

- S=1024：最高占比是 kernel run（43.05%），kernel 总计占 54.17%；
- S=1 cold：最高聚合项是 kernel，但包含一次性编译；
- S=1 排除编译：最高占比是 Host values/权重准备，其次是 materialize/H2D；
- prefill 最后的 16～20 秒需要进一步拆分 export 和 staging release，避免误判。

### Decode

- cold decode：首次 kernel compile/runtime 初始化占主导；
- warm decode：最大已命名项是 Post-MoE materialize/H2D（30.62%）；
- selected Host slice/copy 和 Post-MoE H2D 合计约占 57.8%；
- 两段 kernel 合计不足 4%，暂时不是主要优化目标。

### 建议顺序

1. 给 `export_output()` 和 `end_step()` 增加独立 profile，继续拆分 prefill 的 16～20 秒和 decode 的约
   2 秒收尾时间；
2. decode 优先优化 selected expert Host slice/copy 与 Post-MoE H2D；
3. 长序列 prefill 优先分析 kernel，而不是继续投入纯 Host routed prefetch；
4. 短序列 prefill 再评估普通权重首次准备、H2D 和 staging 生命周期；
5. 稳态 decode kernel 优化优先级暂时低于数据准备和搬运。
