# Case 框架

执行关系：YAML 配置 → Python case → 编译计划 → 公共执行器 → Java Master / Mock。
入口见 [工具导航](../README.md)，定义见 [框架设计](../docs/framework-design.md)。

| 位置 | 职责 |
|---|---|
| `case_config.py`、`case_programs/` | 配置校验、参数注入、业务流程和断言 |
| `scenario/` | 编译、action、执行、证据与资源清理 |
| `instance_runner.py`、`instance_plan.py`、`resource_plan.py` | 实例选择、并行计划、端口与资源预算 |
| `harness.py`、`engine_ops.py` | 环境、进程与 RPC 操作 |
| `grade.py`、`debug_client.py` | 判定档位与 Java debug 客户端 |
| `ha.py` | HA 流量生成和观测窗口 |

新增业务流程写入 `case_programs/`，配置放在顶层 `scenarios/`。
