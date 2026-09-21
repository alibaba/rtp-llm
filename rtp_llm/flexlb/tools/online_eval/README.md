# FlexLB Mock 测试

这里是 Mock 测试的唯一入口。本文档只讨论代码内的编译、启动、运行、结果和配置；机器连接、资源租约和代码同步由外部执行环境负责。

## 选择测试类型

| 目标 | 运行位置 | 文档 |
|---|---|---|
| 测吞吐、延迟、容量或做 A/B | 开发机 | [压测](docs/development/stress.md) |
| 验证协议、状态转换和确定性边界 | 开发机 | [功能测试](docs/development/functional.md) |
| 验证持续负载、故障、扩缩容和恢复 | 开发机 | [场景测试](docs/development/scenario.md) |
| 构建镜像并部署 Mock 到 Whale | Whale | [Whale 部署](docs/whale/README.md) |

开发机上的三类测试共享同一套[编译与运行底座](docs/development/build-and-runtime.md)。运行前先读它，再读对应的测试文档。参数含义集中在[参数参考](docs/reference/parameters.md)，结果判定集中在[结果与指标](docs/reference/results.md)。

## 文档状态

- `docs/development/` 和 `docs/whale/` 是当前 runbook。
- `docs/reference/` 是当前概念、参数和特定 case 契约。
- `docs/archive/` 只用于追溯，不是操作依据。
- `stress/run_online_eval.sh` 等脚本保留兼容性；使用者不需要阅读脚本来理解流程。

完整目录和旧文档去向见[文档导航](docs/README.md)与[迁移映射](docs/migration-map.md)。
