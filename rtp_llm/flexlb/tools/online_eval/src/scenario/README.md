# 场景编译与执行

本包将数据配置和注册的 Python program 编译成有类型、预算与资源引用的执行计划，并执行基础 action、检查和清理。

- 组件职责与生命周期见[框架结构](../../docs/architecture/framework.md)。
- 配置字段、program 注册和 action 边界见[新增 case](../../docs/development/adding-cases.md)。
- 入口、退出码和产物见[运行测试](../../docs/development/running.md)及[结果与指标](../../docs/development/results.md)。

通用 action 位于 `actions/`；case 专属能力位于 `../cases/<case>/`，通过 program 声明，不能由 YAML 导入代码。
