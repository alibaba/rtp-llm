# Overview

FlexLB 为 RTP-LLM 提供多角色路由、请求排队、KV cache 感知选点和主从转发。
工程采用 Java 21、Spring Boot 2.7.18、WebFlux 和 gRPC。本目录描述当前实现，
开发约束见 [AGENTS.md](../../AGENTS.md)。

## 模块与入口

| 模块               | 职责                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| flexlb-api         | Schedule、Cancel、GetRequestState、HTTP 诊断、转发及应用上下线        |
| flexlb-common      | 不可变 Request、WorkerStatus 快照、配置、协议模型与指标抽象           |
| flexlb-grpc        | 引擎与 KVCM 客户端、连接管理及 protobuf stub                          |
| flexlb-sync        | RequestScheduler、请求生命周期、endpoint 容量、路由预测和 worker 同步 |
| flexlb-cache       | LOCAL_SYNC、KVCM、LOCAL_STANDBY 匹配、元数据更新和理论命中统计        |
| flexlb-mock-engine | 可配置的模拟引擎及调度集成、故障和性能测试                            |

## 请求主链路

```text
FlexlbServiceImpl.Schedule
  → 解析请求 ID、角色、优先级与期限，捕获当前配置快照
  → follower 按单跳规则转发；需要本地处理时调用 RequestScheduler.submit
  → DirectRequestScheduler 或 QueuedRequestScheduler
  → RequestWorkerSelector 捕获所选 worker 的 generation pin
  → 提交 Prefill / Decode 预留与 RequestContext 中的发布能力
  → DeliveryDispatcher：BATCH 发 EnqueueBatch，NON_BATCH 返回路由
  → WorkerStatus 对账、取消或超时驱动精确资源清理
```

Encoder 请求使用独立阶段和 EncoderEndpoint。QUEUE 模式由 EncoderQueueCoordinator
等待编码容量；Frontend 向 Encoder 发送实际工作，FlexLB 不发送编码 EnqueueBatch。
VIT-only 路由直接返回选择结果，不登记 Generation 生命周期。

## 状态和并发边界

- `Request` 是不可变输入。`RequestData` 保存配置快照、输入解析任务及遥测数据；
  登记后不能替换请求、调度元数据或 future。
- `RequestContext` 决定一个请求的状态转换和可用操作，返回 `ContextResult` 中的
  被动事实及精确能力。`AbstractRequestScheduler` 执行这些决定。
- `RequestRepository` 只维护活跃、终态和身份索引；执行器、计时器、停服次序由
  `SchedulerRuntime` 管理。
- endpoint 的容量统计、保留的请求关联和引擎实际执行状态分别记录。完成响应不意味着
  已释放所有资源；完整状态中暂时缺失 Decode 任务，也不等于收到了 Finished。
- `PrefillState`、`DecodeState` 使用精确预留与 generation 防止旧回调清理新请求。
  WorkerStatus 更新先在状态锁内生成被动变更，再在锁外通知请求调度器。
- 内部 worker 身份为 `ip:httpPort@engineIndex`，单引擎也使用 `@0`。响应和拨号使用
  物理地址；多引擎响应另外携带有 presence 的 `engine_index`。
- 请求身份为原始字符串 ID 与 `RequestPhase`。Encoder 和 Generation 可共用业务 ID，
  各自取消、过期和查询；原始 int64 ID 独立保留供协议响应回显。

## 相关文档

| 文档                                                                        | 内容                             |
| --------------------------------------------------------------------------- | -------------------------------- |
| [01-routing-and-balancing](01-routing-and-balancing.md)                     | 角色、group 链、候选预测和选点   |
| [02-queue-scheduling](02-queue-scheduling.md)                               | 调度所有者、派发、取消和抢占     |
| [03-resource-management](03-resource-management.md)                         | endpoint 代际与 P/D/Encoder 资源 |
| [04-worker-sync-and-cache](04-worker-sync-and-cache.md)                     | WorkerStatus、协议兼容和缓存匹配 |
| [05-lifecycle-and-consistency](05-lifecycle-and-consistency.md)             | 上下线、优雅 drain 和主选举      |
| [06-configuration-and-observability](06-configuration-and-observability.md) | 配置、诊断、日志和监控           |
