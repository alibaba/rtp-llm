# 运行底座

本包提供 deadline、进程控制、网络探测、通用端口区间检查和有界采样机制，供场景执行、服务适配与监控组件使用。

| 模块 | 职责 |
|---|---|
| `deadline.py` | 截止时间与主线程中断保护 |
| `process.py`、`cleanup.py` | 子进程控制、进程回收与独立清理操作聚合 |
| `network.py`、`startup.py` | HTTP、端口探测及受 deadline 约束的通用启动等待 |
| `resource_plan.py` | 整数资源边界、端口区间展开与冲突检查 |
| `observation.py` | 采样预算、双时钟锚点计算与轮询 |
| `paths.py`、`java_runtime.py` | 当前仓库制品位置与 Java 21 执行环境支持 |

客户端协议和 Java mock 资源布局位于 `../adapters/`；资源作用域与最终状态裁决位于 `../scenario/`；取证扩展合同位于 `../case_api/`；启动记录及证据格式读写位于 `../artifacts/`。本包只依赖标准库及包内机制，不导入这些调用方。

运行前置见[编译与运行](../../docs/development/build-and-runtime.md)，职责与执行契约见[框架结构](../../docs/architecture/framework.md)。
