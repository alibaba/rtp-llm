# Scheduling and Request Lifecycle

`FLEXLB_CONFIG.scheduler` 是带 `type` 的联合配置：

- `DIRECT`：`RouteService` 在调用链中执行 `DefaultRouter.route()`，返回已完成的
  `CompletableFuture<Response>`。
- `QUEUE`（默认）：请求交给 `PriorityScheduler`，由调度器持有请求生命周期、
  endpoint 预留和对外发布权。

QUEUE 模式下，`ordering.type` 和 `dispatcher.type` 是两个正交维度：

- ordering：`FIFO`（默认）或 `PRIORITY`；PRIORITY 由 `PriorityAdmissionScheduler`
  进行优先级准入、状态快照和可选抢占。
- dispatcher：`BATCH`（默认）或 `NON_BATCH`；前者通过引擎 enqueue RPC
  发布 batch，后者将路由决策返回调用方，由调用方向引擎发请求。

主要代码：`RouteService`、`PriorityScheduler`、`PriorityAdmissionScheduler`、
`WorkerBatcher`、`DefaultBatchDispatcher` 和 `RouteDecisionDelivery`。

## 提交与准入

`RouteService.route()` 先将当前不可变的 `FlexlbConfig` 快照绑定到
`BalanceContext`，再按 scheduler 类型分流。QUEUE 路径的关键边界是：

1. `request_id` 是请求代际标识；活跃或已终态的重复 ID 会被拒绝。
2. `QueueCapacityConfig.maxOutstandingRequestsGlobal`（默认 100000）精确限制
   Master 当前持有的请求数，包括还未注册进 inflight map 的准入中请求。
3. 调度器在可能向引擎或调用方发布前装配唯一的绝对过期事件。
4. PRIORITY ordering 进入优先级 plan/commit；FIFO ordering 先调用
   `DefaultRouter`，提交 endpoint 预留后才把请求放入目标 Prefill 的
   `WorkerBatcher`。

路由、预留、inflight 注册和发布都属于同一 request generation。失败或
取消只能通过调度器的单一 reducer 收敛，避免重复回滚和重复完成 future。

## WorkerBatcher 与发布

`WorkerBatcher` 是每个 Prefill endpoint 的决策组组织者。BATCH dispatcher 使用
`FixedWindowBatcherAlgorithm`，按 `maxRequests`、`maxCollectionWaitMs`、预测执行时间
与 endpoint 容量触发发送；NON_BATCH dispatcher 使用
`ImmediateNonBatchAlgorithm`，一个决策组只包含一个请求。

在任何模式下，调度器都在对外可见前提交 endpoint 账本和
`RequestLifecycle`。BATCH 路径记录引擎 ACK/执行状态；NON_BATCH 路径记录
路由决策的交付与调用方确认。

## 取消、过期与状态查询

- `cancelRequest(requestId, expectedBatchId, reason)` 由 scheduler 作为生命周期和资源的
  唯一拥有者执行；`expectedBatchId` 防止旧取消请求命中重用 ID 的新代际。
- 若请求可能已到达引擎，本地资源在引擎终态或取消 fence 收敛前不会被
  乐观释放。
- `getRequestState()` 同时查询活跃 inflight 和最近终态快照；gRPC 转发也带
  单跳 fence，避免跟随者间循环代理。
- `queueTimeoutMs`（默认 3600000）给 QUEUE 所有权提供上界；
  `RequestLifecycleConfig` 另外约束 stale inflight 和已交付未确认请求。

## 默认配置

- scheduler：`QUEUE` + `FIFO`，`queueTimeoutMs=3600000`，
  `maxOutstandingRequestsGlobal=100000`。
- dispatcher：`BATCH`，`maxRequests=8`，`maxCollectionWaitMs=300`，
  `maxWaitingRequestsPerPrefillWorker=1024`，`enqueueRpcTimeoutMs=5000`。
- lifecycle：`staleInflightTimeoutMs=300000`，
  `deliveredNotAcceptedTimeoutMs=30000`，
  `maxDeliveredNotAcceptedRequestsGlobal=200`。
