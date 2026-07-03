# DeepSeek V4 Flash MoE PyPTO 实现方案

本文记录当前仓库实现 MoE 的方案。目标是对齐 `official/model.py` 中
`MoE.forward` 的单卡 bf16 计算逻辑，不实现 EP 多卡、fp4/fp8 量化、远端通信或性能优化。

## 官方计算逻辑

`official/model.py` 中 `MoE.forward` 的核心流程是：

```python
shape = x.size()
x = x.view(-1, self.dim)
weights, indices = self.gate(x, input_ids.flatten())
y = torch.zeros_like(x, dtype=torch.float32)
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top, None])
if world_size > 1:
    dist.all_reduce(y)
y += self.shared_experts(x)
return y.type_as(x).view(shape)
```

单卡实现中 `world_size = 1`，所有 routed experts 都在本地。官方 PyTorch 写法通过
`torch.where(indices == i)` 为每个 expert 动态生成变长 token 列表；PyPTO 中不适合直接
构造这种 ragged 列表，因此采用显式 route-major 中间结果。

## Route-Major 方案

Route-major 将每个 `(token, topk)` 看作一条 route：

```text
route = (b, s, k)
expert id = indices[b, s, k]
route weight = weights[b, s, k]
input row = x[b, s]
```

完整数据流为：

```text
x [B, S, H]
  |
  | gate_hash_fwd / gate_topk_fwd
  v
indices [B, S, TOPK], weights [B, S, TOPK]

x + indices + weights + packed routed expert weights
  |
  | routed route-major expert
  v
route_y [B, S, TOPK, H]

x + shared expert weights
  |
  | expert_shared_fwd
  v
shared_y [B, S, H]

route_y + shared_y
  |
  | combine
  v
out [B, S, H]
```

其中：

```text
B = 1
S = S_DYN
H = 4096
MOE_INTER_DIM = 2048
TOPK = 6
N_EXPERTS = 256
```

该方案不使用 `RECV_MAX`，也不构造：

```text
recv_x [N_EXPERTS, RECV_MAX, H]
recv_y [N_EXPERTS, RECV_MAX, H]
recv_token [N_EXPERTS, RECV_MAX]
recv_count [N_EXPERTS, 1]
```

因此不会出现某个 expert 路由过多导致 recv buffer overflow 的问题。

## 文件边界

MoE 直接实现为单文件：

```text
models/moe.py
```

不再新增独立的 `moe_dispatch.py` 和 `moe_combine.py`。在 route-major 方案下，
`indices` 和 `weights` 本身已经描述了所有 route，单独 dispatch 只会增加无意义的中间
接口；combine 也只是 `route_y` 沿 TOPK 维度求和并加 shared expert 输出，适合放在
`moe.py` 内部。

`models/expert.py` 中已有的 kernel 继续作为 MoE 的核心 building block：

```text
expert_routed_fwd
expert_shared_fwd
```

`expert_routed_fwd` 已经和官方 `Expert.forward(x, weights=...)` 对齐，会在 `w2` 之前
将 routing weight 乘到 intermediate 上：

```text
hidden = silu(w1(x)) * w3(x)
hidden = weights * hidden
out = w2(hidden.to(dtype))
```

这点和官方实现一致，避免把 `weights` 延后到 `w2` 之后引入额外 rounding 差异。

## 权重布局

官方 checkpoint 中 routed expert 权重按 expert 分散存储：

```text
experts.0.w1.weight [2048, 4096]
experts.0.w2.weight [4096, 2048]
experts.0.w3.weight [2048, 4096]
...
experts.255.*
```

PyPTO MoE kernel 使用 packed routed expert 权重。加载阶段负责转置并堆叠：

```text
routed_w1_t [N_EXPERTS, 4096, 2048]
routed_w2_t [N_EXPERTS, 2048, 4096]
routed_w3_t [N_EXPERTS, 4096, 2048]
```

即：

```python
routed_w1_t[e] = experts[e].w1.weight.t().contiguous()
routed_w2_t[e] = experts[e].w2.weight.t().contiguous()
routed_w3_t[e] = experts[e].w3.weight.t().contiguous()
```

shared expert 使用单 expert 转置权重：

```text
shared_w1_t [4096, 2048]
shared_w2_t [2048, 4096]
shared_w3_t [4096, 2048]
```

gate 权重沿用当前 `gate.py` 的约定：

```text
gate_w_t [4096, 256]
gate_bias [256]
tid2eid [vocab_size, TOPK]
```

## Kernel 接口

`models/moe.py` 提供两个顶层 kernel，分别对应 `Gate.hash=True` 和普通 topk routing。

### `moe_hash_fwd`

```python
moe_hash_fwd(
    x:             [B, S_DYN, H] BF16,
    gate_w_t:      [H, N_EXPERTS] BF16,
    tid2eid:       [VOCAB, TOPK] INT32,
    input_ids:     [B, S_DYN] INT64,
    routed_w1_t:   [N_EXPERTS, H, MOE_INTER_DIM] BF16,
    routed_w2_t:   [N_EXPERTS, MOE_INTER_DIM, H] BF16,
    routed_w3_t:   [N_EXPERTS, H, MOE_INTER_DIM] BF16,
    shared_w1_t:   [H, MOE_INTER_DIM] BF16,
    shared_w2_t:   [MOE_INTER_DIM, H] BF16,
    shared_w3_t:   [H, MOE_INTER_DIM] BF16,
    logits:        [B, S_DYN, N_EXPERTS] FP32,
    scores:        [B, S_DYN, N_EXPERTS] FP32,
    indices:       [B, S_DYN, TOPK] INT32,
    weights:       [B, S_DYN, TOPK] FP32,
    route_y:       [B, S_DYN, TOPK, H] BF16,
    shared_gate:   [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_up:     [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_hidden: [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_y:      [B, S_DYN, H] BF16,
    out:           [B, S_DYN, H] BF16,
)
```

### `moe_topk_fwd`

```python
moe_topk_fwd(
    x:             [B, S_DYN, H] BF16,
    gate_w_t:      [H, N_EXPERTS] BF16,
    gate_bias:     [N_EXPERTS] FP32,
    routed_w1_t:   [N_EXPERTS, H, MOE_INTER_DIM] BF16,
    routed_w2_t:   [N_EXPERTS, MOE_INTER_DIM, H] BF16,
    routed_w3_t:   [N_EXPERTS, H, MOE_INTER_DIM] BF16,
    shared_w1_t:   [H, MOE_INTER_DIM] BF16,
    shared_w2_t:   [MOE_INTER_DIM, H] BF16,
    shared_w3_t:   [H, MOE_INTER_DIM] BF16,
    logits:        [B, S_DYN, N_EXPERTS] FP32,
    scores:        [B, S_DYN, N_EXPERTS] FP32,
    indices:       [B, S_DYN, TOPK] INT32,
    weights:       [B, S_DYN, TOPK] FP32,
    route_y:       [B, S_DYN, TOPK, H] BF16,
    shared_gate:   [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_up:     [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_hidden: [B, S_DYN, MOE_INTER_DIM] BF16,
    shared_y:      [B, S_DYN, H] BF16,
    out:           [B, S_DYN, H] BF16,
)
```

`logits/scores/indices/weights` 作为显式 buffer 传入，和现有 `gate.py` 的接口保持一致，
便于单独验证 gate 输出。`route_y/shared_y/out` 是最终 MoE 验证关注的输出；其余中间
buffer 在 runner 中可按需要使用 `ignore_output` 或 `is_output=False`。

## 计算逻辑

### Gate

hash routing 调用：

```python
indices, weights = gate_hash_fwd(
    x, gate_w_t, tid2eid, input_ids, logits, scores, indices, weights
)
```

topk routing 调用：

```python
indices, weights = gate_topk_fwd(
    x, gate_w_t, gate_bias, logits, scores, indices, weights
)
```

对应官方：

```python
weights, indices = self.gate(x, input_ids.flatten())
```

### Routed Expert

route-major routed expert 逻辑为：

```python
for k in range(TOPK):
    for e in range(N_EXPERTS):
        route_weight[b, s, 0] = weights[b, s, k] if indices[b, s, k] == e else 0
        route_out = expert_routed_fwd(
            x,
            route_weight,
            routed_w1_t[e],
            routed_w2_t[e],
            routed_w3_t[e],
            route_gate,
            route_up,
            route_hidden,
            route_out,
        )
        route_y[b, s, k, :] += route_out[b, s, :]
```

因为每个 `(b, s, k)` 只会对应一个 expert，其他 expert 的 `route_weight` 为 0，所以
`route_y[:, :, k, :]` 最终只保留目标 expert 的输出。

上述逻辑对应官方：

```python
idx, top = torch.where(indices == i)
y[idx] += expert(x[idx], weights[idx, top, None])
```

区别是官方按 expert 动态 gather 有效 token；PyPTO 方案用固定 expert loop 和 mask weight
展开成静态计算。

需要先验证 PyPTO 是否支持从 packed 权重中取单 expert 二维 view 并传给
`expert_routed_fwd`：

```python
routed_w1_t[e]  # [H, MOE_INTER_DIM]
routed_w2_t[e]  # [MOE_INTER_DIM, H]
routed_w3_t[e]  # [H, MOE_INTER_DIM]
```

`pypto-serving` 中存在类似的静态 view 传参模式：`moe_ep.py` 会把
`[N_RANKS, N_LOCAL, MOE_INTER, D]` 的 routed weight 通过 `routed_w1[r]` 传给接收
`[N_LOCAL, MOE_INTER, D]` 的 inline kernel。这个用法说明静态下标降一维 view 作为
inline kernel 参数是可行的。我们的场景是 `[N_EXPERTS, H, MOE_INTER_DIM]` 通过
`routed_w1_t[e]` 传给接收 `[H, MOE_INTER_DIM]` 的 `expert_routed_fwd`，语义上是同一类
问题。

仍然需要实际验证的点是后续算子形态不同：`pypto-serving` 的 expert kernel 接收 3D
packed 权重后继续在内部切成 `[1, tile, tile]` 做 3D rhs matmul；当前仓库的
`expert_routed_fwd` 会把 2D view 传入已有 `linear_4096_to_2048` 和
`linear_2048_to_4096` helper。也就是说，view 传参本身大概率可行，但这种 view 进入现有
2D linear helper 后是否稳定，需要通过最小 `moe.py` kernel 验证。

优先实现路径：

```python
expert_routed_fwd(
    x,
    route_weight,
    routed_w1_t[e],
    routed_w2_t[e],
    routed_w3_t[e],
    route_gate,
    route_up,
    route_hidden,
    route_out,
)
```

如果这种 3D -> 2D view 不能稳定进入现有 linear helper，则在 `moe.py` 中按
`expert.py` 的写法展开 routed expert 计算，并让 linear helper 支持 packed weight + 静态
expert id。

### Shared Expert

shared expert 直接调用：

```python
shared_y = expert_shared_fwd(
    x,
    shared_w1_t,
    shared_w2_t,
    shared_w3_t,
    shared_gate,
    shared_up,
    shared_hidden,
    shared_y,
)
```

对应官方：

```python
y += self.shared_experts(x)
```

### Combine

combine 在 `moe.py` 内部完成：

```python
out[b, s, :] = shared_y[b, s, :]
for k in range(TOPK):
    out[b, s, :] += route_y[b, s, k, :]
out = out.to(BF16)
```

这里不再乘 `weights`，因为 `expert_routed_fwd` 已经在 `w2` 前完成 weight 乘法。

## Golden

`golden_moe_forward(tensors, *, hash_route: bool)` 直接按官方 `MoE.forward` 语义编写，
不拆成 dispatch/combine golden。

核心逻辑：

```python
x = tensors["x"]
shape = x.shape
x_flat = x.view(-1, HIDDEN)

if hash_route:
    golden_gate_hash(tensors)
else:
    golden_gate_topk(tensors)

indices = tensors["indices"].view(-1, TOPK).long()
weights = tensors["weights"].view(-1, TOPK).float()

route_y = torch.zeros(B, S, TOPK, HIDDEN, dtype=torch.bfloat16)
for t in range(S):
    for k in range(TOPK):
        e = int(indices[t, k])
        w = weights[t, k].view(1, 1, 1)
        route_y[:, t:t + 1, k, :] = expert_forward(
            x[:, t:t + 1, :],
            routed_w1_t[e],
            routed_w2_t[e],
            routed_w3_t[e],
            w,
        )

shared_y = expert_forward(
    x,
    shared_w1_t,
    shared_w2_t,
    shared_w3_t,
    weights=None,
)

out = shared_y.float() + route_y.float().sum(dim=2)
tensors["out"][:] = out.to(torch.bfloat16).view(shape)
```

测试文件还需要直接调用 `official/model.py` 的 `MoE`，通过 monkeypatch 去掉 fp4/fp8 相关
路径，验证 `golden_moe_forward` 与官方模块一致。

## Runner

`models/moe.py` 的 main 默认同时运行：

```text
moe-hash
moe-topk
```

参数风格与其他文件保持一致：

```text
-p / --platform
-d / --device
-s / --seq-len
--compile-only
```

默认 `seq_len = 8`，远程验证时至少覆盖：

```text
-s 1
-s 13
```

精度阈值先沿用 expert/gate 的组合经验：

```python
ratio_allclose(atol=6e-3, rtol=2.0 / 128, max_error_ratio=0.005)
```

如果定位发现主要误差来自 routed expert 多次 matmul 累积，可单独放宽 `out`，但
`indices` 必须保持 exact 语义验证。

## Tests

新增 `tests/test_moe.py`，覆盖：

- `golden_moe_forward(hash_route=True)` 与 `official/model.py::MoE.forward` 一致。
- `golden_moe_forward(hash_route=False)` 与 `official/model.py::MoE.forward` 一致。
- `seq_len = 1, 3, 13`。
- hash route 中 `tid2eid` 每个 token 的 TOPK expert 互不重复。
- topk route 中 gate 选择的 TOPK expert 互不重复。
- route weight 在 `w2` 前乘，与 `expert_routed_fwd` 和官方 `Expert.forward` 一致。

`tests/test_moe.py` 只验证 Python golden 与官方语义；PyPTO kernel 的实际编译和运行仍通过
`models/moe.py` 的 runner 在 Ascend 服务器上验证。
