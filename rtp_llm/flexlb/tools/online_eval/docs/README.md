# Mock 测试文档

文档只说明现行规则、架构和运行方式。维护约束见 [文档规则](../AGENTS.md)；入口命令从
`rtp_llm/flexlb` 执行，另有工作目录说明的命令除外。实验结果保存在运行归档中。

## 架构与契约

- [框架结构](architecture/framework.md)：组件职责、配置和执行边界。
- [报告装配契约](architecture/reporting.md)：展示词汇、配对、旧归档适配和 bundle 发现。
- [请求与资源生命周期](architecture/request-lifecycle.md)：P/D 分配、Fetch、完成与释放。
- [流量与播放模型](architecture/traffic.md)：真实/合成输入、精确长度、节奏与身份策略。

## 开发与运行

- [编译与运行底座](development/build-and-runtime.md)
- [功能测试](development/functional.md)与[场景测试](development/scenario.md)
- [压测](development/stress.md)与[性能门禁](development/performance-gate.md)
- [播放调节与复现](development/playback-controls.md)
- [合成保真度](development/synthetic-fidelity.md)
- [新增 case](development/adding-cases.md)
- [命令入口](development/entrypoints.md)、[参数](development/parameters.md)、[结果与指标](development/results.md)

## Whale 运行环境

- [CI 与部署](whale/README.md)
- [配置与环境变量](whale/configuration.md)
- [配置与监控对齐](whale/production-alignment.md)

数据存放、元数据与归档见 [数据规则](../data/README.md)，配置入口见 [配置目录](../config/README.md)。
