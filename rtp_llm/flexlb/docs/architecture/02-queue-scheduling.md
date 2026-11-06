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

- `DIRECT`：`DirectRequestScheduler` 在启动后接收请求，通过请求执行池选点并提交路由。
- `QUEUE`（默认）：`QueuedRequestScheduler` 决定入队、下一轮调度时间、候选请求、组批与重试。

两个实现均继承 `AbstractRequestScheduler`。基类直接负责请求注册、准入、取消、超时和资源清理；
没有独立的 `RequestLifecycle` 对象，也不需要将基类绑定到另一个调度器。
自定义调度器继承基类，实现 `RequestScheduler` 的调度事件、启动/停止和排空契约。
何时调度、选择哪些请求、是否重试由具体实现决定，基类不创建调度循环。

QUEUE 模式下，`ordering.type` 和 `dispatcher.type` 是两个正交维度：

- ordering：`FIFO`（默认）或 `PRIORITY`；`PlacementQueue` 保存全局候选及顺序，
  `PrefillRequestQueue` 保存已选中 Prefill 的成员及顺序。
- dispatcher：`BATCH`（默认）或 `NON_BATCH`；前者通过引擎 enqueue RPC 发布批次，
  后者将路由决策返回调用方，由调用方向引擎发请求。

## 提交与准入

`FlexlbServiceImpl` 调用调度器继承的 `submit(RequestContext)`。基类登记请求、安装期限，
随后直接调用本实例的 `onRequest`。`RequestData.getRequestScheduler()` 指向该实际调度器。
校验、登记和交给调度策略在同一提交流程内完成，不再通过独立的 `register` 返回 Future，
然后由另一个入口反查注册结果及请求归属。超时错误直接取自 Context 的请求配置。
生产入口测试覆盖完整提交；只验证请求状态的单元测试显式隔离调度策略，不保留测试专用登记入口。

`RequestRepository` 保存活动请求和最近终态索引，`RequestContext` 保存单个请求的原子事实。
准入直接接收精确 Context，不再传递 ID 和 Future 后重新查找；仍验证目录中的对象身份，
避免旧回调操作重用 ID 的新请求。不可变的 `PlacementOperation` 只标识一次选路操作；尚未结束的资源准备和正在完成的状态统一记录在 `RequestContext` 的 `PlacementFacts` 中，取消与停服需要等待其清理完成。Scheduler 通过 Context 的精确身份检查认领完成工作，旧操作和重复完成不会再次清理资源。

路由、预留和发布使用同一个请求身份。`RequestRouteSelector` 选择端点，
`AbstractRequestScheduler.publishPlacement()` 负责单个请求的资源预留、入队或直接发布及放置收尾。`DeliveryDispatcher` 准备和发送批次，并把投递结果交给实际调度器处理。
资源释放及响应完成的公共流程由基类执行。取消、超时、投递失败和外部响应完成
都进入请求结束流程，由该流程统一调用 `removeRequest(context)` 清除队列成员。
各入口不再分别移除队列，也不发出通用的“请求状态变化”通知。

具体实现通过 `shouldRetryRouteFailure` 决定是否撤回失败路由；基类完成全部资源释放后，
检查绝对截止时间，再将可重试请求交回 `onRequest(context)`。首次入队与撤回后恢复共用
一个入口，QUEUE 在同一把队列锁内移除旧成员、恢复原顺序，保留原优先级和截止时间。
没有第二个重试决策回调，也不再传递状态快照让调度器重新推断动作。

## 组批与发布

`RequestTimeEstimation.projectCandidate` 一次计算排名耗时和当前准入结果。
排名耗时保留在 `state/projectedTtftMsValue`，容量阻塞单独反映在
`admissionState/admissionDetail`，选择策略可以保留候选排序并识别阻塞原因。
调用方不再单独应用准入策略。结果是当前线程复用的只读视图，必须在下一次投影前消费完毕。
`ProjectionQueueSnapshot` 保存冻结的投影输入；`PrefillEndpoint.QueueSnapshot`
保存本轮调度取得的实际路由队列，两者不共用类型。

路由选择原因只记录在 `RequestData` 的 `RoutingDecision` 中；指标从该决策读取，
PV 日志输出同一份决策，不维护或输出第二份选择原因映射。

`QueuedRequestScheduler` 根据 `GroupPlanner` 的决策选取 Worker 队列成员，
按容量变化、请求到达和下一期限触发调度。`BuiltinGroupPlanner` 根据 BATCH / NON_BATCH
配置提供默认组批规则，执行和预测使用一致的规则。

对外发布前，Dispatcher 提交 Endpoint 账本和 Context 的投递事实。
BATCH 路径记录引擎 ACK/执行状态；NON_BATCH 路径记录路由交付与调用方确认。
异步投递、取消和资源释放仍有真实竞争，因此保留精确句柄及这些边界上的同步保护。

## 取消、过期与状态查询

- `cancelRequest(requestId, expectedBatchId, reason)` 由 scheduler 作为生命周期和资源的
  唯一拥有者执行；`expectedBatchId` 防止旧取消请求命中重用 ID 的新代际。
- 若请求可能已到达引擎，本地资源在引擎终态或取消 fence 收敛前不会被
  乐观释放。
- 取消事实被接受后，Scheduler 直接收敛已有抢占操作并结束请求，
  不再进入通用事件续接流程重新判断是否选路或确认投递。结束流程直接区分尚无路由与已有路由，
  统一移除调度队列成员；尚无路由时完成本地清理，已有路由时释放实际取得的资源。
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
  `maxOutstandingRequestsGlobal=100000`。
- dispatcher：`BATCH`，`maxRequests=8`，`maxCollectionWaitMs=300`，
  `maxWaitingRequestsPerPrefillWorker=1024`，`enqueueRpcTimeoutMs=5000`。
- lifecycle：`staleInflightTimeoutMs=300000`，
  `deliveredNotAcceptedTimeoutMs=30000`，
  `maxDeliveredNotAcceptedRequestsGlobal=200`。
