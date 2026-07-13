# 整模 Runner

[`DeepSeekV4Runner`](../../serving/runner.py) 负责把 embedding、逐层 Block、head、状态计划、权重加载和设备 runtime 组织成完整 prefill/decode 调用。

## 模块定位

Runner 只拥有整模编排，不实现模型数学、DeviceTensor 分配或 checkpoint 格式。入口脚本创建 [`DeviceRuntime`](../../serving/device_runtime.py) 并注入 runner；runner 创建 [`DeepSeekV4WeightLoader`](../../serving/weight_loader.py) 和 [`DeepSeekV4StatePlan`](../../serving/state.py)，再调用 [`models/`](../models/README.md) 提供的 spec builders 与 Kernel entrypoints。

Runner 内部传递的 hidden、state 和 Block 输出保持为 runtime-owned DeviceTensor-compatible value，只在 public `prefill()` / `decode()` 返回边界复制到 Host。

## 代码组成

| 接口 | 职责 |
|---|---|
| `__init__()` | 创建 loader/state plan，并要求 runtime 准备目标层的 state |
| `prefill()` | 执行 embedding、完整 prefill Blocks、可选 head 和 Host export |
| `decode()` | 执行 embedding、逐层 selected-expert decode、可选 head 和 Host export |
| `_run_embedding()` | 绑定 embedding specs、输入和固定 embedding weight |
| `_run_prefill_block()` | 每层执行一个完整 prefill Block 并提交 state |
| `_run_decode_block()` | 每层执行 pre-MoE、读取 indices、执行 post-MoE |
| `_block_pre_moe_values()` | 汇总 HC、attention、gate、compressor/indexer 和 aux values |
| `_block_case()` | 选择完整 Block 变体 |
| `_selected_decode_pre_case()` | 选择 decode pre-MoE 变体 |
| `close()` | 关闭 runtime 和 weight loader |

## 构造接口

| 参数 | 默认值 | 作用 |
|---|---|---|
| `checkpoint_path` | 必填 | weight loader 的 checkpoint 根目录 |
| `runtime` | 必填 keyword | 已创建、尚未关闭的 `DeviceRuntime` |
| `config` | `FLASH_CONFIG` | 模型维度和逐层配置 |
| `max_seq_len` | `4096` | state plan 长度；当前实现只接受 `4096` |
| `max_layers` | `1` | 执行层数；`None` 表示全部配置层 |
| `run_head` | `True` | public 输出前是否运行模型 head |
| `profile` | `False` | 启用 `ProfileRecorder` 和 loader 子项统计 |
| `verbose_layer_log` | `False` | 输出逐层日志并把层输出复制回 Host 检查 |
| `expert_cache_dir` | `None` | 可选 packed expert cache 目录 |

`max_layers` 必须在 `[0, config.n_layers]`。Runner 初始化时只为前 `max_layers` 层调用 `runtime.prepare_state()`。

## Block 变体选择

`LayerSpec.ratio` 与 `hash_route` 共同决定 Block：

| Ratio | Route | Prefill Kernel family | Decode pre-MoE family |
|---|---|---|---|
| `0` | hash | `block_swa_hash_prefill_fwd` | `swa_hash_selected_decode_pre_moe_fwd` |
| `4` | hash | `block_csa_hash_prefill_fwd` | `csa_hash_selected_decode_pre_moe_fwd` |
| `128` | top-k | `block_hca_topk_prefill_fwd` | `hca_topk_selected_decode_pre_moe_fwd` |
| `4` | top-k | `block_csa_topk_prefill_fwd` | `csa_topk_selected_decode_pre_moe_fwd` |

所有 decode 变体共用 `selected_decode_post_moe_fwd`。完整 decode Block family 仍可由 `_block_case(..., decode=True)` 选择，但当前 public `decode()` 实际调用的是 split pre/post 路径。

## Prefill 流程

```text
校验并规范 input_ids [1, S]
        │
        ├── begin_step(PREFILL, seq_len=S, start_pos=0)
        ├── embedding
        ├── 对 layer_id = 0 .. max_layers-1
        │       ├── StatePlan.build_prefill_aux()
        │       ├── 读取固定 attention/HC/gate/shared 权重
        │       ├── 构造全部 routed-expert pack
        │       ├── 绑定 state output buffers
        │       ├── materialize + 运行完整 prefill Block
        │       └── commit_state(layer_id)
        ├── 可选 head
        ├── export_output() 到 Host
        └── end_step()
```

Prefill 不把 current state 作为输入，只绑定每层 next state outputs；Block 完成后提交 next buffers。层间 hidden 保持在 NPU，由下一个 Block 消费。

## Decode 流程

每层 decode 拆为两个 Kernel：

```text
hidden + fixed weights + aux + current/next state
        │
        ├── selected decode pre-MoE
        ├── commit_state(layer_id)
        ├── read_control(indices)                 NPU -> Host
        ├── load selected routed experts          Host
        ├── materialize selected expert staging   Host -> NPU
        └── common selected decode post-MoE
                │
                └── next layer hidden
```

Pre-MoE 输出 `indices`、`weights`、`ffn_normed` 和 post-MoE 所需 HC 中间量。Runner 只把 `indices` 复制到 Host；其他中间量继续作为 runtime-owned NPU tensor 传给 post-MoE。Shared expert weights 在 post-MoE 阶段从固定权重路径取得。

## 数据位置与生命周期

| 数据/资源 | 表示与位置 | 生命周期 |
|---|---|---|
| `input_ids` | Host `torch.int64` | 当前 public step |
| Fixed weights | Host `RuntimeWeight` 描述符，materialize 后为 NPU fixed weight | loader/runtime 生命周期 |
| Routed expert weights | Host `HostStagingTensor`，materialize 到 NPU staging | 当前层 dispatch；分配可复用 |
| Hidden / Block outputs | runtime-owned NPU intermediate | 相邻 Kernel 之间，最后消费者后归还 |
| Layer state | NPU current/next persistent buffers | runner/runtime 生命周期 |
| Decode indices | pre-MoE NPU output，随后复制到 Host | 当前层 decode 路由阶段 |
| Public output | Host contiguous `torch.Tensor` | 返回给入口调用方 |

## Step 与异常清理

输入校验发生在 `begin_step()` 之前。成功 begin 后，prefill/decode 主体处于 `try/finally` 中，无论 embedding、Block、head 或 Host export 是否失败都会调用 `end_step()`。如果 `begin_step()` 自身失败，则不会调用 `end_step()`。

`close()` 先关闭 runtime，再关闭 weight loader。入口脚本必须保证 runner 最终关闭；runner 的 public 方法不会自动关闭整个 runtime。

## 约束与错误处理

- Prefill 输入必须为 `[1, S]`，且 `1 <= S <= max_seq_len`。
- Decode 输入必须为 `[1, 1]`，且 `0 < start_pos < max_seq_len`。
- 两种输入都会转换为 CPU contiguous `int64`。
- 不支持 `LayerSpec` 之外的 ratio/route 组合。
- Hash route 必须提供 `tid2eid`，top-k route 必须提供 `gate_bias`。
- `run_head=False` 时 public API 返回 hidden，而不是 logits。

## 性能与可观测性

Runner 将每层拆为 values、weight loader、materialize、kernel 和 state update 事件。Decode 进一步区分 pre-MoE 与 post-MoE。`verbose_layer_log` 会执行 `export_debug_tensor()`，引入额外 D2H 和有限值检查。

Runner 不预加载所有固定权重；某权重在层值构造阶段首次请求，并在 runtime materialize 时首次常驻 NPU。因此首次 prefill 同时包含 Host layout 构造、H2D 和 Kernel 编译，warm decode 的组成不同。

## 验证方法

### Host 侧编排验证

```bash
pytest -q tests/serving/test_runner.py
```

测试使用 opaque device values 验证 embedding 绑定、只在 public 边界 export、step begin/end、异常清理、selected indices 读回，以及 prefill/decode state bindings。

### NPU 集成验证

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1
```

验收时确认 prefill 与 decode 均完成、输出有限。使用更大的 `--max-layers` 覆盖不同 `LayerSpec` 变体；不在本文记录某次验收状态。

## 代码索引

| 内容 | 位置 |
|---|---|
| Runner 实现 | [`serving/runner.py`](../../serving/runner.py) |
| State plan | [`serving/state.py`](../../serving/state.py) |
| Weight loader | [`serving/weight_loader.py`](../../serving/weight_loader.py) |
| Runtime | [`serving/device_runtime.py`](../../serving/device_runtime.py) |
| 编排测试 | [`tests/serving/test_runner.py`](../../tests/serving/test_runner.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`State Plan`](06_state_plan.md)
- [`Weight Loader`](07_weight_loader.md)
- [`Runtime Values`](08_runtime_values.md)
- [`Device Runtime`](10_device_runtime.md)
