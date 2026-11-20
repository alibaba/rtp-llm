# Overview

FlexLB 是面向 AI 模型推理负载（RTP-LLM）的高性能智能负载均衡器：提供多种负载均衡策略、
请求排队调度、KV cache 感知路由与主从高可用。基于 Spring Boot 2.7.18 响应式架构（WebFlux），
运行于 Java 21。

本目录是**稳态架构文档**（描述当前代码长什么样）。架构变了要回来更新本目录。
流程约束（构建、测试、提交规范）见根目录 [AGENTS.md](../../AGENTS.md)。

## 文档索引

| 文档                                                                        | 内容                                                                                          |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [01-routing-and-balancing](01-routing-and-balancing.md)                     | Router / LoadBalancer 模式、角色多阶段路由与 group 链、回滚机制、四种策略算法                 |
| [02-queue-scheduling](02-queue-scheduling.md)                               | RequestAdmissionService / QueuedRequestScheduler、请求生命周期与取消                          |
| [03-resource-management](03-resource-management.md)                         | EndpointRegistry、worker 可用性与本地资源预留                                                 |
| [04-worker-sync-and-cache](04-worker-sync-and-cache.md)                     | Worker 状态 gRPC 同步、本地乐观预测、KV cache 三种匹配源（LOCAL_SYNC / KVCM / LOCAL_STANDBY） |
| [05-lifecycle-and-consistency](05-lifecycle-and-consistency.md)             | 生命周期 Hook 与 /hook/\* 端点、ZooKeeper LeaderSelector 主选举、slave 转发                   |
| [06-configuration-and-observability](06-configuration-and-observability.md) | 全量配置字段与默认值、HTTP 端点、日志架构、指标体系                                           |

## 技术栈

| 类别     | 技术                                                                                |
| -------- | ----------------------------------------------------------------------------------- |
| 语言     | Java 21                                                                             |
| 框架     | Spring Boot 2.7.18（WebFlux 响应式，HTTP 端点返回 `Mono`/`Flux`）                   |
| 响应式   | Project Reactor 2024.0.10                                                           |
| RPC      | gRPC 1.65.0（worker 状态同步 + KVCM meta service）                                  |
| 协调     | Apache Curator 5.4.0（ZooKeeper LeaderSelector 主选举）                             |
| 本地缓存 | Caffeine                                                                            |
| 网络     | Netty 4.1.127.Final                                                                 |
| 观测     | FlexMonitor 抽象（默认 NoOp，internal profile 下 KMonitor）、OpenTelemetry W3C 传播 |
| 测试     | JUnit 5 + Mockito 5.20.0                                                            |

## 模块划分

七个独立 Maven 业务模块，源码根包与模块一一对应。

| Maven 模块 / Java 根包       | 职责                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| `flexlb-api` / `api`         | FlexLB 应用入口、装配、对外服务及调度协议。                |
| `flexlb-config` / `config`   | 配置定义、来源、解析、合并与校验。                         |
| `flexlb-balance` / `balance` | 调度、队列、路由选择、请求生命周期及清理。                 |
| `flexlb-worker` / `worker`   | Worker 身份与状态、Endpoint 账本、状态同步及 Engine 通信。 |
| `flexlb-cache` / `cache`     | 缓存匹配、索引、反馈和 KVCM 集成。                         |
| `flexlb-master` / `master`   | 主节点选举、主地址查询和变更通知。                         |
| `flexlb-util` / `util`       | 公共组件、连接、服务发现、常量及通用定义。                 |

```text
org.flexlb/
├── api/                  应用入口与装配
│   ├── server/           FlexLB 对外服务、转发与拦截器
│   ├── config/           应用装配配置
│   ├── protocol/         生成的调度协议
│   └── log/、metrics/    应用日志控制与运行指标
├── config/               总配置与 ConfigService
│   ├── scheduler/        调度、队列、抢占与请求超时
│   ├── routing/          选路策略
│   ├── worker/           Worker 管理参数
│   ├── cache/            缓存匹配与 KVCM 参数
│   ├── monitoring/       指标、日志与追踪
│   ├── master/           主节点选举参数
│   ├── service/          模型服务、角色、地址和分组
│   └── loading/          加载、合并和校验
│       └── source/、parser/
├── balance/
│   ├── schedule/         RequestScheduler、请求登记、生命周期与清理
│   ├── queue/            排序队列与容量等待
│   ├── strategy/         角色选择与路由组合
│   ├── request/          路由使用的外部查询
│   └── requestlog/、trace/、metrics/
├── worker/               Worker 公共身份
│   ├── endpoint/         Worker 请求记录与资源
│   ├── rpc/              调用 Worker 远程接口及生成协议
│   ├── status/           获取、更新状态及同步指标
│   └── cache/            Worker 缓存状态与更新信息
├── cache/
│   └── match/、localsync/、localstandby/、kvcm/、hash/、metrics/
├── master/               主节点选举与变更通知
└── util/
    └── discovery/、network/、formula/、log/、metrics/
```

生产依赖只向下：`api → balance → cache → worker → master → config → util`；
各模块可直接依赖其所需的更底层模块，底层不依赖上层。Worker 消费请求、路由、缓存及
指标的窄接口；balance/cache 提供实现，api 完成装配。接口回调不要求反向 Maven 依赖。
服务发现只依赖通用地址与发现配置，不引用 Worker 或请求状态。

Engine 协议由 `flexlb-worker` 生成，调度协议由 `flexlb-api` 生成，KVCM 协议由
`flexlb-cache` 生成；wire package、字段号及 RPC 方法保持兼容。
`flexlb-mock-engine` 是随 reactor 验证的测试工具，不属于上述七个业务模块。

## 请求主链路

Schedule 经一致性转发和入口校验后提交 RequestAdmissionService。
DIRECT 立即执行选址；QUEUE 按 FIFO 或 PRIORITY 从全局队列逐个选择请求。
RequestRouteSelector 组合角色，策略在 Endpoint 中登记容量；后续角色失败时完整回滚。
Prefill 按 compute tokens 和配置上限选址，可用缓存优先；Decode 保留 KV 容量检查。
完整路由提交后立即返回给前端，前端向选中的节点发送请求。
QUEUE 没有合适节点时等待容量变化。完成、取消和下线按精确请求及 Endpoint 代际清理资源。

## 核心不变量

- **路由读、同步写**：`EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS`（静态单例，四个角色各一个
  `ConcurrentHashMap<ip:httpPort@index, WorkerStatus>`）由后台同步线程写、路由线程读；
  `WorkerStatus` 内所有计数为 Atomic 字段，原子性是**字段级**而非快照级。
- **内部逻辑地址、外部物理地址**：worker 身份内部一律用逻辑 `ip:httpPort@index`
  （map key、cache 匹配、回滚、feedback，N=1 也是 `@0`）；对外（schedule 响应、gRPC 拨号）
  一律用物理地址，线上协议不出现 `@index`，仅 N>1 时 schedule 响应附加 wire
  `engine_index`。
- **选中即记账，失败必回滚**：策略 `select()` 成功即 `putLocalTask()` 预扣队列时间与 KV
  token；多阶段路由部分失败必须经 `rollBackRoutingFailure()` → `removeLocalTask()` 撤销。
- **本地预测 + 引擎对账**：`localTaskMap` 记录 IN_TRANSIT 任务；引擎状态返回后
  `updateTaskStates()` 对账（IN_TRANSIT→CONFIRMED→RUNNING→FINISHED / LOST），
  `updateKvCacheTokens()` 回填在途任务的 KV 修正。
- **策略必须注册**：4 个策略 bean 在构造函数中自注册进 `LoadBalanceStrategyFactory`；
  `DefaultRouter` 用 `@DependsOn` 保证注册先于路由器构造。
- **并发抢占用 CAS**：ShortestTTFT 系策略通过 `lastSelectedTime` 的 `compareAndSet`
  防止多个调度线程基于同一快照选中同一 worker。
- **master 优先，可降级**：启用一致性时 slave 把请求转发给 master；master 未知或不可达时
  **降级为本地路由**（不是拒绝）。
- **KVCM 与 LOCAL_SYNC 互斥**：`kvcmEnabled` 时不再调度本地 cache 状态轮询，cache 匹配走
  KVCM，Local Standby 作为兜底（自动/手动 failover）。
