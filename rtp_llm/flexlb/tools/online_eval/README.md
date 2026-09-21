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

## 目录边界

- `online_eval/`：流量、监控、证据和报告等可复用 Python 组件。
- `flexlb_test_framework/`：功能/场景测试执行器；`case_programs/` 保存步骤和断言。
- `scenarios/`：只保存声明式 YAML，不放 Python 实现。
- `stress/`：压测入口及压测专用聚合、比较和 HTML 资产。
- `tests/`：代码测试；运行结果统一写到被忽略的 `run/` 或显式输出目录。
- `docs/development/`、`docs/whale/`：当前 runbook；`docs/reference/`：稳定契约。

仓库不保存阶段报告、实验结果、生成图或历史文档副本；需要追溯时使用 Git 历史。
完整导航见[文档入口](docs/README.md)。
