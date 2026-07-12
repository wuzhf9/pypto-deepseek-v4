# Packed BF16 Expert Cache 优化方案

> 本文保留格式迁移时的方案与 A/B 推导。当前生产代码已经收敛为唯一 packed cache 格式，不再提供旧格式
> reader、converter 或兼容接口；manifest 的 `version: 2` 仅作为磁盘 schema 标识。

## 1. 结论

当前 BF16 expert cache 每层保存 256 个专家、每个专家三组独立 tensor，共 768 个 safetensors key。
Prefill 为了构建 kernel 要求的完整 `[256, ...]` 权重，会先逐 expert clone，再执行第二遍 copy 拼成三个
总计 12 GiB 的 Host pack。

建议将 expert cache 升级为按层预打包的 format v2，每层只保存三个连续 BF16 tensor：

```text
routed_w1_t: [256, 4096, 2048]
routed_w2_t: [256, 2048, 4096]
routed_w3_t: [256, 4096, 2048]
```

目标是：

- prefill 直接把三个 packed tensor 作为完整 Host staging，删除逐 expert clone 和二次 pack copy；
- decode 使用 safetensors `get_slice()`，只读取选中的 6 个 expert slice；
- 不为 decode clone 或读取完整 12 GiB packed tensor；
- format v1 暂时作为兼容回退，验证完成后再决定是否移除。

该方案直接消除重复 Host 工作，优先级高于仅掩盖现有低效路径的 Host prefetch。完成并重新 profile 后，
再决定是否继续实现 `prefill_routed_expert_prefetch_plan.md` 中的 V1 prefetch。

## 2. 当前 format v1 路径

`serving/convert_expert_cache.py` 当前每层写入：

```text
expert_000.w1_t
expert_000.w2_t
expert_000.w3_t
...
expert_255.w1_t
expert_255.w2_t
expert_255.w3_t
```

Prefill 的 `DeepSeekV4WeightLoader.get_layer_moe_routed_pack()` 当前执行：

```text
分配 routed_w1_t/routed_w2_t/routed_w3_t 空 Host pack，共 12 GiB
  → 循环 256 experts
  → 每个 expert 的三个 tensor 执行 get_tensor()
  → CPU 路径执行 clone().contiguous()
  → copy_ 到完整 Host pack 对应 expert slot
  → 完整 pack H2D
```

因此大致包含两遍 12 GiB Host 内存流量：

```text
safetensors mmap → per-expert clone
per-expert clone → full routed pack
```

完整 43 层、S=1024、无 L2 swimlane 的 profile：

```text
routed Host pack 总时间     163.133 s
expert_cache.load 总时间     81.438 s
其余 Host pack build         81.695 s
```

约一半 Host pack 时间位于第二遍拼装及相关 allocation、validation 和 Python 循环管理。

## 3. Format v2 文件布局

### 3.1 Manifest

建议升级：

```text
format  = dsv4_bf16_layer_experts
version = 2
```

manifest layout：

```json
{
  "layout": {
    "routed_w1_t": ["n_routed_experts", "dim", "moe_inter_dim"],
    "routed_w2_t": ["n_routed_experts", "moe_inter_dim", "dim"],
    "routed_w3_t": ["n_routed_experts", "dim", "moe_inter_dim"]
  }
}
```

每层仍使用一个文件：

```text
layer_000_experts.safetensors
...
layer_042_experts.safetensors
```

但每个文件只包含三个 key，不再包含 768 个 per-expert key。

### 3.2 连续性

expert 维必须位于第一维。于是：

```python
packed_w1[expert_id]
packed_w2[expert_id]
packed_w3[expert_id]
```

分别是单个 expert 的连续 tensor view，满足 decode selected pack 的逐 slice copy 要求。

## 4. Prefill 加载路径

Prefill 需要 kernel 当前签名要求的完整 256-expert pack：

```python
handle = self._get_file_handle(path)
w1 = handle.get_tensor("routed_w1_t")
w2 = handle.get_tensor("routed_w2_t")
w3 = handle.get_tensor("routed_w3_t")
```

然后直接构造：

```python
MoERoutedPackWeights(
    routed_w1_t=HostStagingTensor(w1, StagingKind.PREFILL_ROUTED, "w1_t"),
    routed_w2_t=HostStagingTensor(w2, StagingKind.PREFILL_ROUTED, "w2_t"),
    routed_w3_t=HostStagingTensor(w3, StagingKind.PREFILL_ROUTED, "w3_t"),
)
```

### 4.1 第一阶段：packed clone

为降低首次实现风险，可以先对三个完整 packed tensor 各执行一次 clone：

```text
mmap packed tensor → one full packed clone → H2D
```

这仍保留一遍 12 GiB Host copy，但已经删除：

- 768 次独立 `get_tensor()`；
- 768 个 per-expert clone；
- 第二遍逐 expert pack copy；
- 256 次 Python loop 和 per-expert validation。

按现有 profile 的 `routed pack - expert_cache.load` 粗略估算，理论可消除约 81.7 秒/43 层的额外 build，
但 packed clone 本身仍需实测，不能直接把该值当作最终收益。

### 4.2 第二阶段：mmap direct H2D

验证 ChipWorker 能从 safetensors mmap tensor 地址正确执行同步 H2D 后，删除完整 clone：

```text
safetensors mmap tensor → synchronous ChipWorker.copy_to()
```

当前 `copy_to()` 是同步接口，因此只需保证 mmap tensor 和 safe_open handle 存活到调用返回，不涉及异步
Host ownership。

direct mmap 可能把文件 page fault 时间转移到 `layer.materialize`，但仍可消除匿名 12 GiB Host pack 和
额外 Host memory copy。需要分别观察 wall time、RSS、page cache 和 materialize 时间，不能只看原来的
`layer.values.routed_pack` 指标。

## 5. Decode 只读取六个专家

Packed cache 不要求 decode 读取完整 256 experts。当前 safetensors 0.7.0 提供 `safe_open.get_slice()`：

```python
w1_slice = handle.get_slice("routed_w1_t")
w2_slice = handle.get_slice("routed_w2_t")
w3_slice = handle.get_slice("routed_w3_t")

for slot, expert_id in enumerate(selected_ids):
    selected_w1_t[slot].copy_(w1_slice[expert_id])
    selected_w2_t[slot].copy_(w2_slice[expert_id])
    selected_w3_t[slot].copy_(w3_slice[expert_id])
```

`get_slice()` 返回 lazy slice view；只有被索引和复制的 expert 页面需要实际访问。

Decode 数据量保持：

```text
单 expert 三组 BF16 权重：48 MiB
6 experts：                 288 MiB
selected Host pack：         288 MiB
selected H2D：               288 MiB
```

打开 12 GiB 文件或建立完整文件的虚拟地址映射，不等于立即占用 12 GiB RSS。物理页面只在被访问时进入
内存，并可由操作系统回收。

### 5.1 禁止整包 clone

当前 `_materialize_cached_tensor()` 在 CPU 路径执行：

```python
return tensor.clone().contiguous()
```

format v2 decode 不能先调用：

```python
handle.get_tensor("routed_w1_t").clone()
```

否则会真的读取并分配完整 packed tensor。必须新增独立的 packed slice accessor，避免复用当前整 tensor
materialize helper。

### 5.2 建议接口

```python
class PackedExpertCache:
    def get_prefill_pack(self, layer_id: int) -> MoERoutedPackWeights:
        """Return three full packed tensors for prefill."""

    def copy_selected_into(
        self,
        layer_id: int,
        expert_ids: list[int],
        *,
        out_w1: torch.Tensor,
        out_w2: torch.Tensor,
        out_w3: torch.Tensor,
    ) -> None:
        """Touch and copy only selected expert slices."""
```

该 helper 可以作为 WeightLoader 私有组件或私有方法组，不应把 safetensors handle 暴露给 Runner 或
WorkerBackend。

## 6. File Handle 与内存生命周期

WeightLoader 当前缓存 `safe_open` handle 到 `close()`，可以继续使用该生命周期：

- prefill mmap tensor 存活到同步 `backend.materialize()` H2D 返回；
- decode slice 只需存活到复制进 selected Host pack；
- selected Host pack 继续按当前路径同步 H2D；
- `WeightLoader.close()` 关闭所有 handles；
- WorkerBackend 不持有 Host mmap tensor 的 backend-lifetime 引用。

43 个 12 GiB 文件的 handle/mmap 可能占用较大的虚拟地址空间，但不会自动形成同等 RSS；format v1 当前
同样会缓存各层 file handle。验收时仍需观察虚拟内存、RSS 和 page cache，确认没有异常常驻。

## 7. 磁盘布局权衡

Format v2 文件内部大致为：

```text
all experts w1
all experts w2
all experts w3
```

Decode 六专家会从三个大区域各读取 6 个约 16 MiB 的连续块，共 18 个大 slice。虽然同一 expert 的
w1/w2/w3 不再物理相邻，但每次仍是 16 MiB 级连续访问，不是小块随机 I/O。

如果实测 decode 变慢，可按以下顺序优化：

1. 按 expert ID 排序读取，再恢复 selected slot 顺序；
2. 对下一段执行文件 readahead；
3. 保留当前 selected Host/device staging 复用；
4. 观察高频 expert 页面是否自然命中 OS page cache。

不建议同时永久保存 format v1 和 v2；两套完整 cache 会使约 516 GiB 磁盘占用翻倍。迁移期间可只转换
少量层验证，完整验证后再统一切换 cache 目录。

## 8. Format v1 兼容

第一版 loader 按文件 key 或 manifest version 判断：

```text
format v2：
  prefill → full packed tensor
  decode  → get_slice()[expert_id]

format v1：
  prefill → current 256-expert loop
  decode  → current per-expert key
```

不要在运行时把 format v1 动态转换成 12 GiB format v2 并缓存；这会重新引入大 Host build 和不可控
内存生命周期。格式转换必须离线完成。

如果当前没有仓外 cache format 兼容要求，v1 回退只保留到 v2 精度、性能和转换稳定性验证完成；随后可
单独删除旧格式代码，避免长期维护两套路径。

## 9. 文件修改范围

### 9.1 `serving/convert_packed_expert_cache.py`（新增）

新增独立 format v2 converter；现有 `serving/convert_expert_cache.py` 保持不变，继续生成 format v1。
验证期间不在同一个脚本中增加 format/version 分支。

- manifest version 升级为 2；
- 每层输出三个 packed tensor；
- 保持 layer/expert 参数用于小范围转换验证；
- partial expert conversion 不得伪装成完整 v2 pack；
- 输出每层 shape、dtype、bytes、build/save time；
- 支持明确 overwrite，不原地破坏已有 cache。

转换 12 GiB 单层时需要受控 Host 内存。若 `save_file()` 要求三个完整 tensor 同时存在，转换器峰值至少
约 12 GiB，加当前 expert 临时 tensor 和写文件开销；不得并行转换多个层。

### 9.2 `serving/weight_loader.py`

- 读取/校验 expert cache manifest；
- 增加 format v2 packed key 检测；
- prefill 返回完整 packed Host staging；
- decode 通过 `get_slice()` 复制 selected experts；
- v2 路径禁止调用整包 `_materialize_cached_tensor()`；
- profile 拆分 packed open、packed full access、selected slice copy；
- close 继续统一关闭 file handles。

### 9.3 `serving/runtime_types.py`

无需新增类型。继续使用：

```text
HostStagingTensor + StagingKind.PREFILL_ROUTED
HostStagingTensor + StagingKind.DECODE_SELECTED
```

### 9.4 WorkerBackend 与 Runner

原则上无需修改：

- Runner 继续调用 `get_layer_moe_routed_pack()` 和 `get_layer_moe_selected_experts()`；
- WorkerBackend 继续消费相同 HostStagingTensor；
- prefill H2D 仍为约 12 GiB/层；
- decode H2D 仍为约 288 MiB/层。

如果为了释放 mmap 引用需要调整局部变量生命周期，只允许 backend-neutral 的引用释放，不得把
safetensors、mmap 或 packed format 判断放入 Runner。

### 9.5 测试文件

建议修改/新增：

```text
tests/test_weight_loader.py
tests/test_convert_packed_expert_cache.py
tests/test_packed_expert_cache.py
```

## 10. 测试计划

### 10.1 本地测试

- v2 manifest、key、shape、dtype 和 byte size；
- v2 full pack 与 v1 逐 expert pack 逐元素一致；
- `get_slice()[expert_id]` 与 v1 expert tensor 一致；
- selected expert ID 顺序和输出 slot 顺序保持一致；
- 非连续/错误 dtype/缺失 key 明确失败；
- partial v2 cache 明确失败，不静默补零；
- v1 fallback 正常；
- close 幂等并释放 handle；
- decode 路径不调用完整 packed `clone()`；
- selected load 只触碰选中 expert 的 accessor。

### 10.2 单层远端验证

先只转换一层，并验证：

1. format v1/v2 prefill 输出一致；
2. 固定六 expert IDs 的 selected tensor 一致；
3. 单层 prefill + 多 decode step 输出一致；
4. prefill H2D bytes 仍约 12 GiB；
5. decode H2D bytes 仍约 288 MiB；
6. decode 前后 Host RSS 增量接近 selected pack，而不是 12 GiB；
7. mmap direct H2D 能正确读取完整 tensor。

### 10.3 完整远端验证

- 5 层覆盖四种 layer variant；
- 43 层 S=1、S=13、S=1024；
- S=1024 对偶发 507018 做多轮复测；
- 带 head generate prefill + 连续 decode；
- 对比 v1/v2 prefill total、routed pack、materialize 和 RSS；
- 对比 v1/v2 decode selected load/build，确保没有明显回退；
- 观察 file handles、虚拟内存和 page cache 生命周期。

功能验证先不开启 L2 swimlane；性能 A/B 使用相同固定配置。

## 11. 性能预期

### 11.1 Packed clone

如果 v2 仍保留一次完整 packed clone，主要收益来自删除第二遍 per-expert pack copy 和 768-key 管理。
按 S=1024 profile 粗略上界：

```text
current routed pack          163.133 s
current expert_cache.load     81.438 s
removable build difference    81.695 s
```

相对 S=1024 cold prefill 290.728 秒约为 28.1%；相对估算 warm prefill 254.673 秒约为 32.1%。实际 packed
clone、page fault、文件布局和 allocator 成本会降低收益，必须以实测为准。

### 11.2 mmap direct H2D

理论上还能删除完整 packed clone 和 12 GiB anonymous Host pack，但磁盘/page cache 读取不会消失，可能
转移到 materialize。验收看完整 wall time 和 RSS，不以某个计时项从 values 移到 materialize 作为收益。

### 11.3 与 prefetch 组合

Packed cache 后，如果剩余 Host 阶段接近约 1.9 秒/层，S=1024 当前约 1.4–2.0 秒/层的 materialize +
kernel 窗口可能覆盖大部分剩余 Host load。推荐先完成 packed cache，再用新 profile 重新计算 V1 prefetch
收益；不要直接沿用旧 format v1 的 26%–28% 理论上限。

## 12. 实施顺序

### Phase 1：单层 format v2 converter

1. 新增 `convert_packed_expert_cache.py` 并定义 manifest v2；
2. 使用新 converter 转换单层三个 packed tensor；
3. 校验与 v1 逐元素一致；
4. 记录转换 Host peak 和文件大小。

### Phase 2：Decode lazy slice

5. WeightLoader 增加 v2 detection；
6. 先实现 `get_slice()` selected expert 路径；
7. 验证只读取 6 experts、RSS 和 288 MiB H2D；
8. 对比连续 decode 性能与 v1。

优先验证 decode 可以避免完整读取，再接入 prefill，防止 packed 格式先造成 decode 回退。

### Phase 3：Prefill packed clone

9. 接入三个完整 packed tensor；
10. 第一版保留整 packed clone；
11. 运行 1/5/43 层精度和 profile；
12. 确认收益来自删除 per-expert pack build。

### Phase 4：Prefill mmap direct H2D

13. 验证 safetensors mmap tensor 可直接传给同步 H2D；
14. 删除完整 clone；
15. 验证 handle/reference 生命周期；
16. 对比 wall time、RSS、page cache 和 materialize。

### Phase 5：重新评估 prefetch

17. 用 format v2 新基线更新 Host/H2D/kernel 数据；
18. 重新计算 depth=1 V1 prefetch 理论上限；
19. 只有剩余 Host exposed time 仍足够大时才继续实现 prefetch。

## 13. 风险与边界

- Packed cache 不减少 prefill 的 12 GiB H2D；只删除重复 Host build/copy；
- decode 必须使用 `get_slice()`，禁止完整 packed clone；
- mmap 建立的虚拟地址范围不等同于 RSS，但仍需实测系统限制；
- direct mmap H2D 可能把 page fault 时间转移到 materialize；
- format v2 转换不能并行多层导致 Host OOM；
- 不长期保存 v1/v2 两份完整 cache 造成磁盘翻倍；
- 不把 safetensors handle、mmap tensor 或 cache version 暴露给 Runner/Backend；
- selected decode 的 288 MiB Host/device staging 生命周期保持不变；
- format v2 完整验证前保留 v1 回退；移除旧格式应作为后续独立步骤。
- 现有 v1 converter 在 v2 完整验证前保持不变；删除它需要后续独立方案和明确确认。

## 14. 推荐决策

先实现 format v2 的 decode lazy slice，证明 packed 文件不会让 decode 读取完整 256 experts；再实现 prefill
packed clone 和 mmap direct H2D。完成后重新 profile，再决定是否继续 Host prefetch。

该顺序优先保护已经通过精度和性能验证的 decode 路径，同时让 prefill 优化直接消除重复 Host work，而
不是先增加并发复杂度去掩盖旧格式的低效打包。
