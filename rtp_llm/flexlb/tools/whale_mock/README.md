# Whale 寄生 Mock Bundle

本目录实现同一 Pod 内 Master + 多个逻辑 Mock Engine 的 test-only bundle。组件代码、Dockerfile、默认配置和测试保留在此；CI、Whale 配置、环境变量和验收流程统一见[Mock 测试 Whale runbook](../online_eval/docs/whale/README.md)。

## Mock 控制端口 API

- [输出长度热更新](docs/OUTPUT_LENGTH_API.md)
- [Prefill 执行时间公式热更新](docs/PREFILL_FORMULA_API.md)

历史实现说明通过 Git 历史追溯；旧操作步骤不再保留在源码树中。
