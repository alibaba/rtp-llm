# Scheduling and Request Lifecycle

同一个业务 `request_id` 在 FlexLB 内按 `RequestPhase` 分为 `ENCODER` 与 `GENERATION`
两条生命周期。Encoder 单独决策进入独立的 Encoder QUEUE：登记请求后选点，选中端点接管请求时进入
`DISPATCHING`，确认并提交路由成功响应时进入 `ACKNOWLEDGED`，随后由 Encoder 的 WorkerStatus 活跃任务、
完成任务或失活超时推进状态。Frontend 负责把请求发给选中的 Encoder，FlexLB 不做
Encoder 批量派发，也不调用 Encoder Cancel RPC。Encoder QUEUE 只在模型配置包含 Encoder role 时创建，
按照 FIFO 或 PRIORITY 顺序决策，并用 `dispatcher.maxInflightPerEncoderWorker` 限制每个 Encoder 的
`running + waiting + 本地待观察请求` 数量。WorkerStatus 更新、请求结束和 endpoint 退役会唤醒等待的
Encoder 请求；队列超时和取消按 Encoder 阶段独立清理。Generation 沿用原调度流程及独立队列。

`Cancel` 和 `GetRequestState` 通过 `role` 选择请求记录：`ENCODER` 定位 Encoder，
`PREFILL`、`DECODE`、`PDFUSION` 以及省略或传默认值时都定位 Generation。常见 EPD 调用先单独决策 Encoder，处理完成后再以同一个
业务 ID 决策 Prefill、Prefill + Decode 或 PDFusion。混合阶段角色同时请求不在当前
验证流程内，也没有专门的角色组合白名单。

`FLEXLB_CONFIG.scheduler.type` 取值为 `QUEUE`（默认）：请求交给 `PriorityScheduler`，
由调度器持有请求生命周期、endpoint 预留和对外发布权。

`ordering.type` 和 `dispatcher.type` 是两个正交维度：

- ordering：`FIFO`（默认）或 `PRIORITY`；PRIORITY 由 `PriorityAdmissionScheduler`
  进行优先级准入、状态快照和可选抢占。
- dispatcher：`BATCH`（默认）或 `NON_BATCH`；前者通过引擎 enqueue RPC
  发布 batch，后者将路由决策返回调用方，由调用方向引擎发请求。

主要代码：`RouteService`、`PriorityScheduler`、`PriorityAdmissionScheduler`、
`PrefillQueue`、`DefaultBatchDispatcher` 和 `RouteDecisionDelivery`。

## 提交与准入

### NON_BATCH_V2

`dispatcher.type = NON_BATCH_V2` 时，Generation 使用 `balance.schedulev2` 的三段队列：

```text
MainQueue -> 每台机器一个 PrefillQueue -> 共用 DecodeQueue -> publish(Prefill 地址, Decode 地址)
  调度线程         每台机器独立线程          调度线程             调用方再向引擎发请求
     |                                        |
     +-> 并行查缓存、预测自身耗时              +-> 并行准备请求需求
```

- 三段都按 `RequestOrder` 排序：PRIORITY 先按优先级降序，同优先级按进入主队列的顺序号升序；
  FIFO 只按顺序号。底层用有序集合，抢占时倒序遍历，从最低优先级中最晚入队的请求开始找。
- `maxQueuedRequests` 统计三段里所有尚未发布的请求，包括已经 reserve、正在等待 Decode permit 的请求。
  进入 Prefill 队列时还占一个共享子队列名额，上限为
  `min(maxQueuedRequests, Prefill 机器数 × maxInflightPerPrefillWorker × prefillSubqueueLimitMultiplier)`。
  两种计数都保留到 publish，失败或取消时也归还。Prefill 下线退回主队列时，只归还子队列名额。
- 总排队数满时，新来的高优先级请求只从主队列挤掉更低优先级的请求。
  选中 Prefill 后若共享子队列名额满了，则只在这台 Prefill 的等待队列里挤掉更低优先级的请求。
  两处都要求新请求达到高优先级阈值，并且用阶段 CAS 保证已经出队的请求不会被误踢。
- Decode 没有单独的队长上限，沿用总排队数和子队列名额。高优先级请求按排序先执行，
  不通过队长触发踢掉 Decode 等待请求。容量不足时的 RETURN 抢占仍由 Decode 线程执行。
- P/D 共用 `dispatcher.dispatchStarvationThresholdMs`，默认 500ms，每一轮读取最新配置。
  Prefill 只有 KV 不够时允许较小请求先走，在途数量满时整台机器等待；
  Decode 放不下队头时允许同 group 后面的请求先走。队头等待达到阈值后恢复严格顺序，0 表示始终严格顺序。
  Prefill 等待从进入这台机器的队列起算，Decode 等待从 Prefill dispatch 起算。
- 主队列和 Decode 各有一个调度线程，以及独立的 `ProcessWorkers` 线程池。
  两个池各自的线程数默认是 JVM 可见 CPU 数，沿用启动参数 `-Dflexlb.queue.planner.threads=N`。
  调度线程按队列顺序提交准备任务，处理已完成材料，不等前面尚未准备完成的请求。
  Decode 队头已经等到饥饿阈值时，同 group 后面的结果即使先完成，也不能先 reserve。
- Prefill worker 对完整候选集合查缓存、预测请求自身耗时，返回 `PrefillInputs`，不选定节点。
  主调度线程累加最新队列中排在请求之前的预测耗时及在途耗时，再按 TTFT、缓存亲和、同分轮询规则选择节点。
  选定后紧接着分配子队列名额、必要时抢占并入队，再处理下一份材料。最终决策不调用缓存 IO 或预测器。
  队列线程把请求从 waiting 转入 dispatched 的过渡期间，TTFT 只计算一次这条请求。
- Decode worker 只准备容量需求和评分公式，返回 `DecodeInputs`。Decode 调度线程读取最新节点负载，
  计算评分、选择节点并立即 reserve，再处理下一个请求；空机器放行、RETURN 抢占、permit、发布和归还也在该线程执行。
- 正在准备及刚完成、尚未被调度线程接收的任务合计不超过池的线程数。若高优先级队头没有 worker 名额，
  可以丢掉一个较低顺序请求的已完成材料腾出名额；请求本身仍排队，稍后重新准备。
  准备材料不持有 endpoint hold；请求取消或服务关闭后，迟到的材料直接丢弃。
- 缓存匹配结果按 `dispatcher.cacheMatchResultTtlMs` 复用，默认 30000ms，必须为正数；关联的 Prefill 耗时一起缓存。
  容量不足时复用材料，只重试最终决策；每轮读取最新 TTL，使用前检查 Prefill 候选集合。
  TTL 到期、节点增删或开始下线时重新准备；请求离开本阶段、失败或取消时清理缓存。
  Decode 材料不含缓存匹配结果，不使用该 TTL，只随请求阶段清理。
- 发布前重新获取所选 Prefill 的 hold，并确认节点未开始下线、注册表仍指向同一对象。
  等待 Decode 或已有 Decode 预占时发现 Prefill 不可用，由 Decode 线程归还预占，再退回主队列。
  退回时清掉旧 Prefill 记录和材料，归还子队列名额，但保留总排队名额、原顺序号和请求截止时间。
- `DecodeReservations` 登记 V2 尚在管理的 Decode 预占，从创建前一直保护到归还或引擎确认。
  旧孤儿清理同时检查老请求表和这个并发表，避免按旧 TTL 清掉 V2 的正常预占。
- Decode 空机预占到发布之间的短窗口允许尾部请求交给刚摘除、仍处于优雅下线阶段的节点，依赖引擎继续服务。
  请求超时以定时取消实际执行为准，不在 publish 时追加严格的绝对截止时间校验。
- V2 队列可观测性由 `V2QueueReporter` 独立上报，每秒按请求阶段统计 `schedule_v2.queue.depth`：
  `main`、逐 endpoint 的 `prefill`、`decode_waiting`、`decode_reserved`，已发布请求不计入等待。
  共享子队列名额保留到发布，所以名额占用不等于 Prefill 排队长度。空队列持续报零，移除节点后补零。
- `schedule_v2.queue.blocked.qps` 按阶段、endpoint、优先级、固定原因统计进入阻塞的请求；
  同一次排队连续相同原因去重，原因或 endpoint 变化、退回重排后可再计一次。
  `blocked_attempt.qps` 记录所有实际失败检查，包括重复重试；未执行检查的顺序跳过不计数。
  两者都不是最终失败率。`rejected.qps` 在请求完成回调中按失败前阶段和错误码名称计一次。
  截止时间触发的取消单独标为 `REQUEST_TIMEOUT`，不与复用同一错误码的其他失败混淆。
- Prefill 阻塞区分在途满、KV 不足；Decode 区分请求数满、KV 不足和同时不足，
  并区分等待预占与等待许可。Decode 原因使用失败后的即时容量视图，若容量已恢复则记状态变化。
- `QueueProgress` 把阶段和累计时间放在同一次 CAS 中更新，退回重排时累加，失败与取消也记录最后一段。
  完成响应前写入 PV：`globalQueueMs` 是主队列累计时间，`prefillQueueMs` 和 `decodeQueueMs` 是分项，
  `subQueueMs` 是 Prefill 与 Decode 的总等待时间；`schedule_v2.queue.wait_ms` 提供相同口径的监控。
- 每台 Prefill 队列有自己的线程和 `LoopSignal`。新请求、记录移除时唤醒自己的线程，
  空闲时阻塞等待。机器下线或同地址换成新的 endpoint 时，旧线程退回等待请求并停止；新队列启动新线程。
- 每个循环最外层捕获 `Throwable`，记录异常后继续下一轮，worker 的异常作为结果交回调度线程。
  主队列和 Decode waiting 每轮最多处理 256 个候选；因 group 阻塞而直接跳过的请求不占次数。
  Decode 的 published 和 reserved 每轮遍历全部记录。

以下描述为原有 BATCH / NON_BATCH 链路。

`RouteService.route()` 先将当前不可变的 `FlexlbConfig` 快照绑定到
`BalanceContext`，再提交给 QUEUE 调度器。QUEUE 路径的关键边界是：

1. `request_id` 是请求代际标识；活跃或已终态的重复 ID 会被拒绝。
2. `scheduler.maxQueuedRequests`（默认 100000）在 QUEUE + NON_BATCH 下是 Generation 的总排队上限，
   包括 GlobalQueue 和所有 EngineSubQueue 中尚未交付的请求；其他 QUEUE 模式仍只限制 GlobalQueue 的深度。
   QUEUE + NON_BATCH 下，
   同一形态的 EngineSubQueue 合计等待上限是
   `min(scheduler.maxQueuedRequests, 当前形态的逻辑端点数 × dispatcher.maxInflightPerPrefillWorker × dispatcher.prefillSubqueueLimitMultiplier)`
   （倍数默认 4）。端点数直接读取 `EndpointRegistry.getEndpointCount(当前形态)`，
   不逐台遍历；全局入口按总排队请求数判断是否达到配置的总上限；
   从全局队列进入 Prefill 子队列不重复占总名额。队列满时，PRIORITY 模式下低于
   `scheduler.ordering.highPriorityThreshold`（默认 50）的
   新请求直接返回 `QUEUE_FULL_LOW_PRIORITY_REJECTED`（8433）。
   达到阈值的新请求只从全局队列中选择优先级严格低于自己的可抢占请求，先选择优先级最低的
   一档；如果这一档有多个请求，就抢占最晚入队的一个，保留更早入队的请求。
   被挤出队列的请求按自己的优先级返回错误码：低于阈值返回 `QUEUE_FULL_LOW_PRIORITY_REJECTED`（8433），
   达到阈值返回 `QUEUE_FULL_HIGH_PRIORITY_REJECTED`（8434）。没有可抢占请求时，高优新请求返回
   `QUEUE_FULL_HIGH_PRIORITY_REJECTED`（8434）；FIFO 模式满队列时返回 `QUEUE_FULL`（8502）。
   调度器已选中、正在选 worker
   或处理选路结果的请求仍占队列名额，处理结束前不参与队列抢占；因资源不足继续排队的
   请求可以参与。请求移交 Prefill 后退出全局队列，但保留总排队名额直到路由交付；
   请求取消或失败时也释放名额。容量与阈值配置快照更新后对后续
   入队生效。
   Encoder 使用独立队列。
3. 调度器在可能向引擎或调用方发布前装配唯一的绝对过期事件。
4. PRIORITY ordering 进入优先级 plan/commit；FIFO ordering 先调用
   `DefaultRouter`，提交 endpoint 预留后才把请求放入目标 Prefill 的
   `PrefillQueue`。

路由、预留、inflight 注册和发布都属于同一 request generation。失败或
取消只能通过调度器的单一 reducer 收敛，避免重复回滚和重复完成 future。

## PrefillQueue 与发布

`PrefillQueue` 是每个 Prefill endpoint 的决策组组织者。BATCH dispatcher 使用
`FixedWindowBatcherAlgorithm`，按 `maxRequests`、`maxCollectionWaitMs`、预测执行时间
与 endpoint 容量触发发送；NON_BATCH dispatcher 按 `scheduler.decision.type` 使用
`SINGLE` 或 `FIXED_WINDOW` 组织决策组，逐请求返回路由结果。
Prefill 选路在目标 group 的已注册节点中计算缓存命中与选路代价；
队首的投递等待和 Prefill 当前可用 KV 不限制选路候选。
Decode 预留登记选中的请求，不在预留时检查引擎并发或 KV 额度；
投递许可在交付时按实际容量获取。
QUEUE + NON_BATCH 完成 Decode 排队预留和 Prefill 子队列入队后，请求按子队列顺序等待交付；
本地等待项不占路由交付名额。交付准备时检查 Prefill 请求上限和可用 KV，
以及选中 Decode 的并发与 KV 额度。容量不足时子队列等待对应容量释放，
不因其他机器释放容量而迁移。
若准备后 WorkerStatus 突然报告此前未知的活动请求，本轮可能短暂超过单机在途上限。
同一形态的 Prefill 子队列共享上述等待预算；PREFILL 与 PDFUSION 是互相独立的路由形态，
端点数、待交付名额和容量唤醒分别计算。共享名额满导致入队失败时请求留在全局队列，
等待任一子队列释放名额后重新从完整候选池选路。优先级模式下可先尝试替换
选中 Prefill 本机尚未交付的低优请求，不跨 Prefill 查找受害者。
总排队上限和共享子队列上限是两个独立边界：同一高优请求可能先在全局入口抢占，
再在选中 Prefill 的子队列抢占另一个低优请求。
QUEUE + BATCH 完成 Decode 排队预留和 Prefill 子队列入队后，按批次上限准备交付。
Prefill 待交付队列中的请求被高优新请求替换时，同样按被替换请求自己的优先级返回 8433 或 8434。

Worker 下线停止子队列时，尚未完成 dispatch 交接的请求释放旧 Prefill 路由预留和 Decode 预留，
回到原 global 队列条目重新选路，保留请求 future、优先级、排队顺序和截止时间。
若入队发布尚未结束，回队操作等待该次发布结束；期间发生取消或超时则按取消或超时结束请求。
已完成 dispatch 交接的请求继续按交付结果和引擎状态清理。调度器整体关闭时结束待调度请求。

Decode 退役通知或投递准备、交接检查发现所选 Decode 已退役时，尚未完成 dispatch 交接的请求
释放旧 Prefill 和 Decode 预留，回到原 global 队列条目重新决策两端。回队保留 future、优先级、
排队顺序和截止时间，并恢复总排队计数；入队发布尚未结束时等待发布结束再回队。
取消和超时按原请求生命周期处理，已完成 dispatch 交接的请求按交付结果和引擎状态清理。

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
