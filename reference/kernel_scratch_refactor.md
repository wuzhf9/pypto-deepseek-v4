# PyPTO Kernel Scratch 内迁重构方案

本文记录当前仓库中 PyPTO kernel scratch tensor 的重构规则和建议顺序。目标是在不改变
DeepSeek V4 Flash bf16 计算逻辑的前提下，减少外部 kernel 接口中的纯中间 buffer。

## 已验证经验

`models/head.py` 已验证两类行为：

- 可以在 kernel 内部创建并传给子 kernel：
  - 直接来自已有输入动态维的 tensor，例如 `tokens = pl.tensor.dim(x, 1)` 后创建
    `[B, tokens, HIDDEN]`。
  - 固定 shape scratch，例如 `[T_TILE, VOCAB]`。
- 不应直接在 kernel 内部创建后传给子 kernel：
  - 经过表达式计算得到的新 padded dynamic shape，例如
    `padded_tokens = ((tokens + T_TILE - 1) // T_TILE) * T_TILE` 后创建
    `[B, padded_tokens, ...]`。

上述 padded dynamic 实验在远端 Ascend 编译阶段失败，错误为：

```text
@pl.jit: missing inferred tensor metadata for parameter 'x_pad'
```

因此当前重构只迁移直接动态维 scratch 和固定 shape scratch。`S_PAD_DYN`、`C_DYN` 与
`S_DYN` 组合出来的 `K_DYN` 等表达式动态维先保留外部传入。

## 判断规则

可以迁入 kernel 内部：

- shape 直接来自已有输入动态维：
  - `tokens = pl.tensor.dim(x, 1)`
  - `pl.create_tensor([B, tokens, ...], dtype=...)`
- shape 直接来自另一个输入动态维：
  - `compressed_len = pl.tensor.dim(cos, 0)`
  - `pl.create_tensor([B, compressed_len, ...], dtype=...)`
- 固定 shape scratch：
  - 例如 `[T_TILE, VOCAB]`、`[T_TILE, HIDDEN]`

暂不迁入：

- padded dynamic shape：
  - 例如 `ceil(S / tile) * tile`
  - 对应 `S_PAD_DYN`
- 表达式组合 dynamic shape：
  - 例如 prefill `kv_pool` 的 `K_DYN = S_DYN + C_DYN`
- cache/state 业务输出：
  - 后续整网推理需要继续返回或更新
- 独立 kernel 的最终输出：
  - 例如 `rmsnorm out`、`linear out`、`rope out`
  - 这些文件作为独立验证入口时仍应保留外部输出

## 各文件重构清单

| 文件 | 可迁入内部的 tensor | 保留外部/输出 |
|---|---|---|
| `models/attention_qkv.py` | `q_a`, `q_proj`, `kv_proj`, `kv_normed` | `qr`, `q`, `kv` 先保留为语义输出 |
| `models/attention_out.py` | `o_inv`, `proj` | `out` |
| `models/gate.py` | `logits`, `scores` | `indices`, `weights` |
| `models/expert.py` | `gate`, `up`, `hidden` | `out` |
| `models/compressor_ratio128.py` | `kv_proj`, `score_proj`, `pooled`, `normed` | `compressed`, `kv_state_out`, `score_state_out`, `compressed_cache_out` |
| `models/compressor_ratio4.py` | `kv_proj`, `score_proj`, `pooled`, `normed` | `compressed`, state/cache 输出 |
| `models/indexer.py` | `q_proj`, `q_rope`, `weights`, `comp_kv_proj`, `comp_score_proj`, `comp_pooled`, `comp_normed`, `index_score` | `topk_idxs`, `index_kv_cache`, state 输出 |
| `models/attention_swa.py` | qkv scratch、`qr/q/kv`、`attn_o`, `o_inv`, `proj`, `attn_out` | `kv_cache_out`, `out` |
| `models/attention_hca.py` | qkv scratch、compressor scratch、`compressed`, `attn_o`, `o_inv`, `proj`, decode `kv_pool` | prefill `kv_pool` 暂保留，state/cache/out 保留 |
| `models/attention_csa.py` | qkv scratch、indexer scratch、compressor scratch、`idx_topk_idxs`, `csa_topk_idxs`, `attn_o`, `o_inv`, `proj`, decode `kv_pool` | prefill `kv_pool` 暂保留，state/cache/out 保留 |
| `models/moe.py` | `logits`, `scores`, `indices`, `weights`, `route_y`, `shared_gate`, `shared_up`, `shared_hidden`, `shared_y` | `out` |
| `models/hc.py` | 单独模块先不改；在 `block.py` 内部创建 `x_mixed/post/comb` 这类直接 `S_DYN` 输出 | `x_pad/mixes/pre/comb_logits/x_mixed_pad/post_pad/comb_pad` 保留外部 |
| `models/block.py` | 所有 `[B, S_DYN, ...]` 中间值：normed、attention out、moe out、HC direct 输出等 | HC padded scratch、prefill `kv_pool`、所有 cache/state/out |

## 推荐顺序

采用自底向上的方式重构：

1. `models/attention_qkv.py`、`models/attention_out.py`、`models/gate.py`、`models/expert.py`
2. `models/compressor_ratio128.py`、`models/compressor_ratio4.py`
3. `models/indexer.py`
4. `models/moe.py`
5. `models/attention_swa.py`
6. `models/attention_hca.py`
7. `models/attention_csa.py`
8. `models/block.py`

这样可以先稳定叶子 kernel 的接口，再逐步减少组合 kernel 和 block kernel 的参数数量。不要先从
`models/block.py` 开始，否则子 kernel 旧接口、golden 输出和 dynamic metadata 问题会混在一起，
定位成本较高。

## 单文件重构流程

后续每次只重构一个文件，按以下顺序执行，避免接口、golden 和测试状态不一致。

### 1. 确认待迁入 tensor

先在目标文件中列出当前 kernel 参数、`TensorSpec`、golden 写回和测试断言：

```bash
rg "<tensor_name>|<kernel_name>|TensorSpec" -n models tests
```

只迁移纯 scratch tensor。判断依据：

- 该 tensor 只在当前 kernel 内作为中间结果使用。
- 该 tensor 不是整网推理需要返回或更新的 cache/state。
- 该 tensor 不是独立 kernel 的最终业务输出。
- shape 属于已经验证可内部创建的类型：
  - 直接来自输入动态维，例如 `tokens = pl.tensor.dim(x, 1)`。
  - 固定 shape。

如果上层文件还没有重构，外层接口中的旧 scratch 参数可以先保留，但子 kernel 调用点必须同步改成新接口。
这样自底向上重构时，上层文件可能暂时还有冗余参数，但不会因为子 kernel 签名变化而不可用。

### 2. 修改 kernel 接口和内部创建逻辑

在 `@pl.jit.inline` kernel 中删除 scratch 参数，并在 `bind_dynamic` 后用直接动态维创建：

```python
tokens = pl.tensor.dim(x, 1)
scratch = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
```

注意事项：

- 不再对已删除的 scratch 参数调用 `bind_dynamic`。
- scratch 创建应尽量靠近第一次使用的位置。
- 如果 scratch 需要传给子 kernel，它的 shape 必须和子 kernel 注解完全一致。
- 不要把 padded dynamic shape 迁入内部，例如 `ceil(S / tile) * tile` 计算出的 shape。

### 3. 修改 test runner 和 TensorSpec

目标文件自己的 `*_test` runner 要同步删除 scratch 参数。

`build_*_specs` 中删除对应 `TensorSpec`，包括：

- 非 `is_output` 的 scratch 输入。
- 仅用于比较中间结果的 scratch 输出。

保留：

- 真实业务输出。
- cache/state 输出。
- 当前文件仍然需要作为独立验证入口比较的输出。

### 4. 修改 golden

golden 只写当前公开接口中的输出，不保留旧接口兼容写法。

需要删除的典型写法：

```python
if "scratch" in tensors:
    tensors["scratch"][:] = scratch
```

如果该 tensor 已经迁入 kernel 内部，就不要在 golden 中继续写回，也不要为了兼容旧测试保留条件分支。
这可以保证 golden、runner 和 kernel 接口语义一致。

### 5. 修改直接调用点

用 `rg` 查找所有调用目标 kernel 的位置：

```bash
rg "<kernel_name>\\(" -n models tests
```

需要同步修改：

- 当前文件自己的 runner。
- 直接调用该子 kernel 的组合 kernel。

如果组合 kernel 的外部接口仍然保留旧 scratch 参数，可以先不删除这些参数和 specs；等轮到该组合文件重构时再统一清理。
但组合 kernel 内部调用子 kernel 时必须使用新签名。

### 6. 修改 pytest 用例

测试用例也必须跟随公开接口收敛：

- 不再构造已迁入内部的 scratch tensor。
- 不再断言已迁入内部的中间结果。
- 只断言业务输出、cache/state 输出，或当前文件仍公开的语义输出。

如果测试中原本用中间结果验证某段逻辑，可以保留独立的纯 PyTorch helper 计算 expected，但不要把中间 tensor 放回
`tensors` 字典中让 golden 写入。

### 7. 本地验证

每次单文件重构后至少执行：

```bash
python -m compileall <changed model files> <changed test files>
pytest -q <related test files>
```

如果该文件有直接上层调用点也被改动，例如重构 `expert.py` 时同步改了 `moe.py`，`compileall` 需要包含上层文件。

### 8. 远端 Ascend 验证

本地通过后再同步相关代码到 Ascend 服务器验证。只同步本次改动需要的代码文件，文档和无关测试不需要同步。

默认验证：

```bash
ssh ascend_server 'source set_env.sh && cd dsv4 && task-submit --device auto --run "python models/<file>.py -p a2a3 -d {}"'
```

如果 kernel 带 `S_DYN`，继续验证：

```bash
ssh ascend_server 'source set_env.sh && cd dsv4 && task-submit --device auto --run "python models/<file>.py -p a2a3 -d {} -s 13"'
ssh ascend_server 'source set_env.sh && cd dsv4 && task-submit --device auto --run "python models/<file>.py -p a2a3 -d {} -s 1"'
```

如果 runner 有 `--case`、`--decode-start-pos` 等参数，按该文件已有默认和边界场景补充验证。

### 9. 最终检查

重构完成后做两类扫描：

```bash
rg 'if ".*" in tensors' -n models/<file>.py
rg '"<removed_tensor_name>"' -n models tests
```

确认：

- 目标文件没有保留已删除 scratch 的条件写回。
- 相关测试没有继续构造或断言已迁入内部的 scratch。
- 直接调用点都已经使用新签名。

## 验证策略

每个文件重构后都需要单独验证：

- 本地执行 `python -m compileall <changed files>`。
- 本地执行相关 pytest 用例。
- 远端 Ascend 按该文件 runner 验证默认 case。
- 对带 `S_DYN` 的 kernel，至少再验证：
  - `-s 1`
  - 一个非 tile 对齐长度，例如 `-s 13`

对于 `K_DYN = S_DYN + C_DYN`、padded dynamic shape 等尚未验证的情况，应先做最小实验，不要直接
在复杂组合 kernel 中迁移。
