# Scheduling and Request Lifecycle

同一个业务 `request_id` 在 FlexLB 内按 `RequestPhase` 分为 `ENCODER` 与 `GENERATION`
两条生命周期。Encoder 单独决策按 `scheduler.type` 走 DIRECT 或独立 QUEUE：登记请求后选点，选中端点接管请求时进入
`DISPATCHING`，确认并提交路由成功响应时进入 `ACKNOWLEDGED`，随后由 Encoder 的 WorkerStatus 活跃任务、
完成任务或失活超时推进状态。Frontend 负责把请求发给选中的 Encoder，FlexLB 不做
Encoder 批量派发，也不调用 Encoder Cancel RPC。Encoder QUEUE 只在模型配置包含 Encoder role 时创建，
按照 FIFO 或 PRIORITY 顺序决策，并用 `dispatcher.maxInflightPerEncoderWorker` 限制每个 Encoder 的
`running + waiting + 本地待观察请求` 数量。WorkerStatus 更新、请求结束和 endpoint 退役会唤醒等待的
Encoder 请求；队列超时和取消按 Encoder 阶段独立清理。Generation 沿用原调度流程及独立队列。

`Cancel` 和 `GetRequestState` 可传 `phase` 查找对应阶段；省略或传默认值时定位
Generation，以兼容旧客户端。常见 EPD 调用先单独决策 Encoder，处理完成后再以同一个
业务 ID 决策 Prefill、Prefill + Decode 或 PDFusion。混合阶段角色同时请求不在当前
验证流程内，也没有专门的角色组合白名单。

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
2. `scheduler.maxQueuedRequests`（默认 100000）限制 Generation 全局队列中的请求数，
   包括等待放置、正在规划和等待资源的请求。`GlobalQueueCoordinator.offer` 在队列锁下检查
   当前深度。PRIORITY 模式下，低于 `scheduler.ordering.highPriorityThreshold`（默认 50）的
   新请求直接返回 `QUEUE_PRIORITY_THROTTLED`（429），错误信息为“低优先级限流”。
   达到阈值的新请求从队列中选择优先级严格低于自己的可抢占请求，先选择优先级最低的
   一档；如果这一档有多个请求，就抢占最晚入队的一个，保留更早入队的请求。
   被抢占的请求也返回 429。没有可抢占请求时，新请求返回 `QUEUE_FULL`（503），
   错误信息为“队列满”；FIFO 模式满队列时也返回该错误。调度器已选中、正在选 worker
   或处理选路结果的请求仍占队列名额，处理结束前不参与队列抢占；因资源不足继续排队的
   请求可以参与。成功路由或请求结束后释放队列位置；容量与阈值配置快照更新后对后续
   入队生效。
   Encoder 使用独立队列。
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

## Decode 抢占指令交付

`EvictionManager` 在原路由选定的逻辑 Decode 容量不足时规划抢占；
`engineCancellation.mode` 选择 `RPC`（默认）或 `RETURN`。RPC 由
`DecodePreemptionCoordinator` 发起 Cancel 并等待权威终态；RETURN 只原子占有精确的
victim 代际和新请求预留，随后通过 `QueueRouteAdmission` 发布普通路由结果。
已进入 Decode 运行阶段的较低优先级请求在 `DECODE_ENGINE_OWNED` 允许且目标引擎支持取消时，
可被已进入全局队列的更高优先级请求通过该流程抢占。运行中的 Decode 请求不占用全局队列
位置；取消它本身不会释放 `scheduler.maxQueuedRequests` 的队列位置。

RETURN 要求 NON_BATCH。客户端将指令透传给目标 Decode，由 Decode 完成所有老请求取消及资源释放后执行新请求。
Master 不发起此次抢占的 Cancel RPC，也不等待取消完成才返回。

## 默认配置

- scheduler：`QUEUE` + `FIFO`，`queueTimeoutMs=3600000`，
  `maxQueuedRequests=100000`。
- dispatcher：`BATCH`，`maxRequests=8`，`maxCollectionWaitMs=300`，
  `maxWaitingRequestsPerPrefillWorker=1024`，`enqueueRpcTimeoutMs=5000`。
- lifecycle：`staleInflightTimeoutMs=300000`，
  `deliveredNotAcceptedTimeoutMs=30000`，
  `maxDeliveredNotAcceptedRequestsGlobal=200`。
