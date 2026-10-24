# Whale Mock 部署入口

本目录提供 `embedded`、`single_pod`、`per_engine` 三种 test-only 部署模式，使用 `MOCK_DEPLOYMENT_MODE` 统一选择。固定入口通过 `deployment.py` 分派到 Master＋Mock supervisor 或独立 Mock launcher。组件代码、Dockerfile、默认配置和测试保留在此；CI、Whale 配置、环境变量和验收流程统一见[Mock 测试 Whale runbook](../online_eval/docs/whale/README.md)。

## Mock 控制端口 API

- [Mock ViT 图片链路](docs/VIT_API.md)
- [输出长度热更新](docs/OUTPUT_LENGTH_API.md)
- [Prefill 执行时间公式热更新](docs/PREFILL_FORMULA_API.md)

历史实现说明通过 Git 历史追溯；旧操作步骤不再保留在源码树中。

`engine-config.json` 是显式 synthetic 测试行为，`performance.json` 只提供校准。
real 导出、容量导入、支持范围与配置回读见[引擎配置](../online_eval/docs/architecture/engine-configuration.md)。
