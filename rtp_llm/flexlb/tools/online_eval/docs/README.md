# 文档导航

文档只说明现行、可复用的契约。命令默认从 `rtp_llm/flexlb` 执行；维护规则见 [AGENTS.md](../AGENTS.md)。

| 要做什么 | 入口 |
|---|---|
| 准备运行环境 | [编译与运行底座](development/build-and-runtime.md) |
| 选例、预览和执行 | [运行测试](development/running.md) · [命令入口](development/entrypoints.md) · [参数](development/parameters.md) |
| 收取报告、判断结果和对照 | [结果与指标](development/results.md) |
| 新增或调整 case | [新增 case](development/adding-cases.md) |
| 调整播放、核对输入 | [播放与复现](development/playback-controls.md) · [合成保真度](development/synthetic-fidelity.md) |
| 在 Whale 部署与验收 | [Whale 部署](whale/README.md) · [配置](whale/configuration.md) · [生产对齐](whale/production-alignment.md) |

理解实现时阅读[框架结构](architecture/framework.md)、[请求生命周期](architecture/request-lifecycle.md)、[流量模型](architecture/traffic.md)和[报告契约](architecture/reporting.md)。固定输入及配置分别见[数据规则](../data/README.md)和[配置目录](../config/README.md)。
