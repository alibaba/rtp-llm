# Mock 测试文档导航

[上级入口](../README.md)按四种功能选择 runbook。本页说明文档边界。

## 当前文档

### 开发机

1. [编译与运行底座](development/build-and-runtime.md)：三类测试共用的依赖、构建产物、进程模型和启动参数。
2. [压测](development/stress.md)：标准负载、运行、聚合和 A/B。
3. [功能测试](development/functional.md)：功能合同的选择、执行和判定。
4. [场景测试](development/scenario.md)：持续负载、干预、恢复和证据完整性。

### Whale

1. [CI 与部署](whale/README.md)：完整 CPU bundle、独立引擎镜像及验收链路。
2. [配置与环境变量](whale/configuration.md)：两种拓扑和运行时配置。

### 参考

- [参数参考](reference/parameters.md)
- [入口清单与迁移](reference/entrypoints.md)
- [结果与指标](reference/results.md)
- [框架结构](reference/architecture.md)
- [新增 case](reference/adding-cases.md)
- `reference/cases/`：少数需要单独解释的场景合同。
- `reference/concepts/`：P/D 生命周期与流量源等底层语义。

阶段设计、验证快照和旧操作说明不进入源码树；需要时从 Git 历史读取。代码与当前 runbook 是唯一现行依据。
