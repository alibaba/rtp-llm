# Scheduling and Request Lifecycle

`FLEXLB_CONFIG.scheduler` 是带 `type` 的联合配置：

- `DIRECT`：`RouteService` 在调用链中执行 `DefaultRouter.route()`，返回已完成的
  `CompletableFuture<Response>`。
- `QUEUE`（默认）：请求交给 `PriorityScheduler`，由调度器持有请求生命周期、
  endpoint 预留和对外发布权。

QUEUE 模式下，`ordering.type`、`globalDecision.type`、`decision.type` 和
`dispatcher.type` 分别控制排序、worker 选择前的全局规划、worker 内决策组和交付方式。其中两个
decision type 是启动期拓扑：运行期更新不会替换策略或调度循环。

- ordering：`FIFO`（默认）或 `PRIORITY`；PRIORITY 由 `PriorityAdmissionScheduler`
  进行优先级准入、状态快照和可选抢占。
- globalDecision：`SINGLE`（默认）保持逐请求 worker 选择；`FIXED_WINDOW` 在
  `GlobalQueueCoordinator` 中先收集最多 `maxRequests` 条请求，再按 AutoTPM 优先级层调用
  `CostBasedBatchedPrefillStrategy` 统一选择 Prefill worker。`maxPlanEvaluations`（默认 4096）
  限制一次全局规划的 completion search 及每个局部优化阶段。首版只支持 prefill
  `candidateChoice.type=BEST_ONLY`。
- decision：控制选定 Prefill worker 内的 `WorkerBatcher` 使用 SINGLE 或 FIXED_WINDOW。
- dispatcher：`BATCH`（默认）或 `NON_BATCH`；前者通过引擎 enqueue RPC
  发布 batch，后者将路由决策返回调用方，由调用方向引擎发请求。

全局 `FIXED_WINDOW` 必须配 worker `scheduler.decision.type=SINGLE`，以便凑批只发生在 worker
选择前；反之，worker `FIXED_WINDOW` 要配 `globalDecision.type=SINGLE`。validator 在启动和更新合并后
拒绝两层同时 fixed-window。FlexLB 的全局规划、逐请求提交和发送顺序均不保证引擎执行顺序。

主要代码：`RouteService`、`PriorityScheduler`、`PriorityAdmissionScheduler`、
`WorkerBatcher`、`DefaultBatchDispatcher` 和 `RouteDecisionDelivery`。

## 提交与准入

`RouteService.route()` 先将当前不可变的 `FlexlbConfig` 快照绑定到
`BalanceContext`，再按 scheduler 类型分流。QUEUE 路径的关键边界是：

1. `request_id` 是请求代际标识；活跃或已终态的重复 ID 会被拒绝。
2. `QueueCapacityConfig.maxOutstandingRequestsGlobal`（默认 100000）精确限制
   Master 当前持有的请求数，包括还未注册进 inflight map 的准入中请求。
3. 调度器在可能向引擎或调用方发布前装配唯一的绝对过期事件。
4. 启动期 `globalDecision=SINGLE` 时沿用逐请求 `DefaultRouter.routeForQueue()`；启动期
   `globalDecision=FIXED_WINDOW` 时先形成全局窗口，PRIORITY 按优先级层分别联合规划，FIFO
   对窗口整体联合规划。所有结果仍按原队列顺序逐请求提交 endpoint 预留后进入目标 Prefill 的
   `WorkerBatcher`。PRIORITY 每次只规划最高 eligible 层；该层逐条提交后，同一已收集窗口中的
   下一层再读取真实容量重新规划，因此低层不会沿用高层提交前的虚拟状态。

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

## 全局联合 Prefill 规划

`PrefillStrategy` 是 cost-based Prefill 的抽象基类，统一候选发现、过滤、cache/TTFT 评估和
结果 materialize，并向 `DefaultRouter` 提供统一的 `select()` / `selectBatch()` 契约。Spring 的启动期
binding 在全局窗口模式注册 `CostBasedBatchedPrefillStrategy`，否则注册
`CostBasedPrefillStrategy`；全局窗口调用联合规划逻辑。联合规划对不可变候选快照执行机会损失优先的贪心分配，
若贪心结果不完整，先用有界搜索恢复尽可能完整的可行计划，随后进行有界的单请求迁移和双请求
交换。cache affinity 复用现有配置，并以 shortest-TTFT 批量基线为固定上限。

联合计划计算落点、批内虚拟工作以及同一 worker 的 KV/request 容量预算，不提前消费真实
delivery capacity。每条请求仍保留独立的
`AdmissionMutation`、generation pin、精确发布事务、取消和 blocker 语义。PRIORITY 不允许跨层
联合优化；提交前的更高优先级重检、priority rescue 与抢占规则保持原样。

当一次联合规划包含两条或更多请求时，策略以同一个 `globalPlanning.decisionId` 把计划摘要写入每条
可建模请求的 PV `routingDecisions[].globalPlanning`：其中有 greedy 与最终放置数、completion search
是否调用/耗尽预算/修复数、TTFT 和 cache-affinity 阶段的 evaluation、move、swap 及目标差值；该请求的
`requestChanges` 只列出真正改变过它的阶段。`selectionReason=GLOBAL_BATCH` 只表示该落点由联合计划
确定，`prefillPolicy.candidateChoice=BEST_ONLY` 仍单独表达策略约束；它不等价于普通单请求策略的
`CACHE_LEADER` 或 `OVER_CAP` reason。若联合计划没有可行落点，reason 为
`GLOBAL_BATCH_NO_AVAILABLE_CANDIDATE`；无法建模 engine work 的 LRU fallback 保持
`UNMODELED_PENDING_LRU`，不伪造全局计划证据。

`WorkerBatcher` 随后写入的 `decisionGroup` 仍是 worker-local 的交付组，不能用其 size 推断全局
收集或联合规划大小；请使用 `globalPlanning.requestCount` 和 `globalPlanning.decisionId`。单请求进入 batched
strategy 时复用普通选择流程，因而保留普通 reason，且没有 `globalPlanning`。debug 级 application log
会同步输出 `global_prefill_batch_plan id=<globalPlanning.decisionId>` 的阶段摘要，便于先按 id 定位同一计划。

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

RETURN 要求 NON_BATCH。客户端将指令透传给目标 Decode，由 Decode 完成所有老请求取消及资源释放后执行新请求。
Master 不发起此次抢占的 Cancel RPC，也不等待取消完成才返回。

## 默认配置

- scheduler：`QUEUE` + `FIFO`，`queueTimeoutMs=3600000`，
  `globalDecision=SINGLE`，`maxOutstandingRequestsGlobal=100000`。
- dispatcher：`BATCH`，`maxRequests=8`，`maxCollectionWaitMs=300`，
  `maxWaitingRequestsPerPrefillWorker=1024`，`enqueueRpcTimeoutMs=5000`。
- lifecycle：`staleInflightTimeoutMs=300000`，
  `deliveredNotAcceptedTimeoutMs=30000`，
  `maxDeliveredNotAcceptedRequestsGlobal=200`。
