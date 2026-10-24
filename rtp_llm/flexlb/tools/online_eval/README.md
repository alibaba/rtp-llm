# FlexLB Mock 测试

从[运行测试](docs/development/running.md)开始；前置依赖见[编译与运行](docs/development/build-and-runtime.md)，产物与判定见[结果与指标](docs/development/results.md)。Whale 使用独立的[部署入口](docs/whale/README.md)。

| 目录 | 职责 |
|---|---|
| `config/` | 场景、指标集合、报告视图及运行配置 |
| `src/cases/` | 各 case 的输入、流程、门禁、指标与报告 |
| `src/scenario/` | 用例构建、注册与合同，通用编译、执行及收尾 |
| `src/runtime/`、`src/adapters/` | 通用进程与采样机制，以及服务协议和环境适配 |
| `src/monitoring/` | 持续监控、采集质量评估与指标归档 |
| `src/analysis/`、`src/reporting/` | 可复用统计、输入保真度分析、图表和 bundle |
| `src/traffic/`、`src/artifacts/` | 输入生成、播放与制品归档 |
| `scripts/`、`tests/` | 命令入口、内部管线与回归测试 |
| `data/`、`run/` | 固定输入与忽略的运行输出 |

从本目录运行测试：`PYTHONPATH=src:. python3 -m pytest -q tests`。
更多说明见[文档导航](docs/README.md)、[配置目录](config/README.md)和[数据规则](data/README.md)。

合入评审及实验解释需同时阅读[已知问题与验收边界](docs/development/known-issues.md)。
