# <模块名称>

<!-- 使用面向用户的一句话说明模块作用；完成后删除所有 HTML 注释。 -->

## 模块定位

<!-- 说明模块职责、所在数据流和必要的数学定义。 -->

## 官方模型中的 <模块名称>

<!-- 从 official/model.py 枚举相关 module、parameter、函数或内联表达式。 -->

## PyPTO 实现

<!-- 列出当前 kernel、wrapper、spec builder 和变体。 -->

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `<official symbol>` | `<current symbol>` | 直接调用/融合内联/数学等价/存在但未使用/不支持或未执行 |

## 数据接口

```text
input:  <shape>, <dtype>
weight: <shape>, <dtype>
output: <shape>, <dtype>
```

<!-- 补充动态维度、固定约束、state/cache 和内部 scratch 的边界。 -->

## 实现方式

<!-- 描述数据流、tiling、累加精度、rounding、融合或状态更新。 -->

## 实现差异与限制

<!-- 明确当前实现与官方通用路径的差异和未支持范围。 -->

## Golden 参考实现

<!-- 标明 golden 函数、输入 snapshot、输出 dtype 和特殊比较区域。 -->

## 精度验收标准

| 项目 | 标准 |
|---|---:|
| Absolute tolerance | `<atol>` |
| Relative tolerance | `<rtol>` |
| 允许超出容差的元素比例 | `<ratio>` |
| NaN/Inf | 不允许/按代码说明 |

## 验收方法

```bash
<current validation command>
```

<!-- 只记录可复现命令、必要参数和前置条件，不记录某次运行的状态或结果。 -->

## 集成验证范围

<!-- 分开列出 standalone kernel 验证、host integration tests 和完整模型验证。 -->
