# 运行底座

本包负责实例与端口规划、Master / Mock 生命周期、进程回收、HTTP/gRPC 和 Java 客户端。case 流程与判定位于 `../cases/`。

| 模块 | 职责 |
|---|---|
| `instance_runner.py`、`instance_plan.py`、`resource_plan.py` | 选例、并行计划与资源预算 |
| `environment_config.py`、`environment.py` | 环境输入、复用身份、启动和清理 |
| `process.py`、`network.py` | 进程句柄、定向信号、HTTP 和端口探测 |
| `paths.py`、`java_runtime.py`、`zk_helper.py` | 制品、Java 21 与 ZooKeeper helper |
| `java_client.py`、`engine_ops.py`、`proto_utils.py` | 发流客户端、RPC 与 protobuf |

运行前置见[编译与运行](../../docs/development/build-and-runtime.md)，资源和执行契约见[框架结构](../../docs/architecture/framework.md)。
