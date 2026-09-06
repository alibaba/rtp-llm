# Overview

FlexLB 是面向 AI 模型推理负载（RTP-LLM）的高性能智能负载均衡器：提供多种负载均衡策略、
请求排队调度、KV cache 感知路由与主从高可用。基于 Spring Boot 2.7.18 响应式架构（WebFlux），
运行于 Java 21。

本目录是**稳态架构文档**（描述当前代码长什么样）。架构变了要回来更新本目录。
流程约束（构建、测试、提交规范）见根目录 [AGENTS.md](../../AGENTS.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-routing-and-balancing](01-routing-and-balancing.md) | Router / LoadBalancer 模式、角色多阶段路由与 group 链、回滚机制、四种策略算法 |
| [02-queue-scheduling](02-queue-scheduling.md) | PriorityScheduler / WorkerBatcher / RouteService、请求生命周期与取消 |
| [03-resource-management](03-resource-management.md) | EndpointRegistry、worker 可用性与本地资源预留 |
| [04-worker-sync-and-cache](04-worker-sync-and-cache.md) | Worker 状态 gRPC 同步、本地乐观预测、KV cache 三种匹配源（LOCAL_SYNC / KVCM / LOCAL_STANDBY） |
| [05-lifecycle-and-consistency](05-lifecycle-and-consistency.md) | 生命周期 Hook 与 /hook/* 端点、ZooKeeper LeaderSelector 主选举、slave 转发 |
| [06-configuration-and-observability](06-configuration-and-observability.md) | 全量配置字段与默认值、HTTP 端点、日志架构、指标体系 |

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Java 21 |
| 框架 | Spring Boot 2.7.18（WebFlux 响应式，HTTP 端点返回 `Mono`/`Flux`） |
| 响应式 | Project Reactor 2024.0.10 |
| RPC | gRPC 1.65.0（worker 状态同步 + KVCM meta service） |
| 协调 | Apache Curator 5.4.0（ZooKeeper LeaderSelector 主选举） |
| 本地缓存 | Caffeine |
| 网络 | Netty 4.1.127.Final |
| 观测 | FlexMonitor 抽象（默认 NoOp，internal profile 下 KMonitor）、OpenTelemetry W3C 传播 |
| 测试 | JUnit 5 + Mockito 5.20.0 |

## 模块划分

多模块 Maven 工程：

| 模块 | 职责 |
|---|---|
| **flexlb-api** | Web 层：`HttpLoadBalanceServer`（调度）、`FlexlbControlServer`（控制）、`HealthCheckServer`、`AppStateHookServer`（生命周期 hook）；主端口 7001 |
| **flexlb-common** | 共享模型（`WorkerStatus`、`Request`/`Response`、`RoleType`、`BalanceContext`）、`FlexlbConfig`/`ConfigService`、`FlexMonitor` 指标抽象、`BlockCacheKeyCalculator` |
| **flexlb-grpc** | 引擎 gRPC 客户端（worker status / cache status）+ KVCM meta-service 客户端（`KvcmGrpcClient`、leader 解析、健康探测） |
| **flexlb-sync** | 核心：路由与策略、排队调度、动态资源管理、worker 状态同步、主选举、生命周期 Hook、监控上报 |
| **flexlb-cache** | KV cache 匹配：LOCAL_SYNC 两级索引（`KvCacheManager`/`GlobalCacheIndex`/`EngineLocalView`）、KVCM provider、Local Standby 兜底索引、failover 编排、block hash 计算 |

flexlb-sync 内部包结构（`org.flexlb`）：

```
balance/
├── scheduler/      DefaultRouter / PriorityScheduler / WorkerBatcher / Dispatcher
├── endpoint/       EndpointRegistry / PrefillEndpoint / DecodeEndpoint / 本地预留账本
├── strategy/       LoadBalanceStrategy + Prefill / Decode / Random 策略
└── resource/       PrefillResourceMeasure / DecodeResourceMeasure / ResourceMeasureFactory
consistency/        LBStatusConsistencyService / ZookeeperMasterElectService
sync/
├── synchronizer/   MasterEngineSynchronizer（20ms 定时）/ AbstractEngineStatusSynchronizer（线程池）
├── runner/         EngineSyncRunner / GrpcWorkerStatusRunner / GrpcCacheStatusCheckRunner
├── schedule/       ExpirationCleaner（worker/任务过期清理）
└── status/         EngineWorkerStatus / ModelWorkerStatus / WorkerBlockHashConfigResolver
service/
├── grace/          GracefulOnlineService / GracefulShutdownService / ActiveRequestCounter / strategy/ 各 Hooker
├── grpc/           EngineGrpcService
├── monitor/        EngineHealthReporter / BatchSchedulerReporter / PrioritySchedulerReporter / FlexlbLogManager
└── RouteService
```

## 请求主链路

```
gRPC FlexlbService.Schedule (FlexlbGrpcServer, HTTP 端口 + 100)
    ↓ ActiveRequestCounter.acquire()（在途请求计数）
    ↓ 若启用一致性且本机非 master → 转发给 master（master 不可达则降级本地路由）
    ↓ RequestBlockHashService.prepareBlockCacheKeys()（block hash 计算，独立线程池）
RouteService.route()
    ├─ scheduler.type=DIRECT：当前调用链执行 DefaultRouter.route()
    └─ scheduler.type=QUEUE：PriorityScheduler.submit()
           ├─ ordering.type=FIFO：普通路由后提交到 Prefill WorkerBatcher
           └─ ordering.type=PRIORITY：PriorityAdmissionScheduler 做优先级准入/抢占
    ↓
DefaultRouter：按非空角色 map 固定顺序路由 PDFUSION → DECODE → PREFILL → VIT，
               首个成功角色的 worker group 约束后续角色的候选（group 亲和链）
    ↓
LoadBalancer 策略选 worker（读 EngineWorkerStatus 共享状态 + CacheAwareService 匹配）
    ↓ 选中即 WorkerStatus.putLocalTask()：本地乐观记账（队列时间 / KV token 预扣）
后阶段失败 → DefaultRouter.rollBackRoutingFailure() 逐个 LoadBalancer.rollBack() 撤销记账
```

## 核心不变量

- **路由读、同步写**：`EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS`（静态单例，四个角色各一个
  `ConcurrentHashMap<ip:port, WorkerStatus>`）由后台同步线程写、路由线程读；`WorkerStatus`
  内所有计数为 Atomic 字段，原子性是**字段级**而非快照级。
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
