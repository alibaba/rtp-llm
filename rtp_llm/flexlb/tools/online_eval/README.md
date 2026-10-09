# FlexLB Mock 测试

从[运行测试](docs/development/running.md)开始；前置依赖见[编译与运行](docs/development/build-and-runtime.md)，产物与判定见[结果与指标](docs/development/results.md)。Whale 使用独立的[部署入口](docs/whale/README.md)。

| 目录 | 职责 |
|---|---|
| `config/` | 场景、指标集合、报告视图及运行配置 |
| `src/cases/` | case 配置接口、注册表及各 case 的流程、门禁、指标与报告 |
| `src/scenario/`、`src/runtime/` | 通用编译执行、基础 action、进程与协议操作 |
| `src/workload/`、`src/monitoring/` | 持续负载的证据生命周期、采集与归档 |
| `src/analysis/`、`src/reporting/` | 可复用统计、输入保真度分析、图表和 bundle |
| `src/traffic/`、`src/artifacts/` | 输入生成、播放与制品归档 |
| `scripts/`、`tests/` | 命令入口、内部管线与回归测试 |
| `data/`、`run/` | 固定输入与忽略的运行输出 |

从本目录运行测试：`PYTHONPATH=src:. python3 -m pytest -q tests`。
更多说明见[文档导航](docs/README.md)、[配置目录](config/README.md)和[数据规则](data/README.md)。
