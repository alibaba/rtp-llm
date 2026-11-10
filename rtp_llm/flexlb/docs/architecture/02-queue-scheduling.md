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
然后由另一个入口反查注册结果及请求归属。超时错误由实际调度器的 `schedulingTimeoutError()` 决定：
Direct 返回 `BATCH_SLO_EXPIRED`，Queue 返回 `RESOURCE_EXHAUSTED`。注册时将确定的错误码交给 Context，
发送前过期、确认时失活和定时器超时都沿用该值。`RequestData` 只保存入口给定的 priority、deadline
和原始输入，不根据配置补建元数据或丢弃 payload；缺少调度元数据的请求返回 `INVALID_REQUEST`。
Dispatcher 校验投递输入，并在注册成功后释放不使用的 payload。Repository 只登记已验证的请求，
其 pending delivery 统计包含所有模式；对外 queue length 由 scheduler 提供，Direct 为零。
等待诊断由 Queue scheduler 决定，与错误码无关。Encoder 选择和提交都使用所属 scheduler 的同一容量接口。
PrefillEndpoint 接收装配阶段提供的不可变队列规则，启用队列时只校验规则一致性，不再从全局配置构造策略。
生产入口测试覆盖完整提交；只验证请求状态的单元测试显式隔离调度策略，不保留测试专用登记入口。

`RequestRepository` 保存活动请求和最近终态索引，`RequestContext` 保存单个请求的原子事实。
准入直接接收精确 Context，不再传递 ID 和 Future 后重新查找；仍验证目录中的对象身份，
避免旧回调操作重用 ID 的新请求。不可变的 `PlacementOperation` 只标识一次选路操作；Context 直接保存当前 `PlacementOperation`，防止重复完成的标志只在 Context 内部使用。取消与停服需要等待当前尝试清理完成。Scheduler 通过 Context 的精确身份检查认领完成工作，旧操作和重复完成不会再次清理资源。

路由、登记和发布使用同一个尝试身份。`RequestRouteSelector` 选择并登记 P/D；角色失败时在返回前整体回滚。
`AbstractRequestScheduler.publishPlacement()` 复用登记，负责入队或直接发布及放置收尾。
提交期间候选由提交任务持有；Endpoint 接受后才调用 Context.publishRoute。Endpoint 操作不持有 Context 锁，
期间发生取消会拒绝发布，由提交任务释放登记后结束尝试。入队通知在 Context 接收发布后发出。
Context 不保存 PREPARED/NOT_PUBLISHED/ROLLBACK_COMPLETED 中间结果；只有本地清理异常时，
才保留尚未释放的路由供现有异步清理重试，这种记录不可派发。
Context 中的精确 route 引用表示已接收的路由，不再维护另一个 published 标志。
为清理失败保留 route 时，同时记录 pendingEnd；结束原因阻止派发，清理不能被当成正常发布。

### 选点、发布与重试的协调

QUEUE 的 planner 在同一次任务中完成选择登记、入队/发布和尝试收尾。
选择结果和登记句柄只保存在当前任务的局部变量中；不存在 PlacementPlan 或结果交接队列。
调度线程只挑选全局候选、管理可用执行名额、唤醒条件和 Prefill 队列调度；
全局队列锁不包围选点、Endpoint 提交、取消 RPC 或响应回调。

| 路径                 | 修改的状态与可见时点                                              | 协调与清理责任                                                                                             |
| -------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 选择成功             | Strategy 在各 Endpoint 锁内核对评分版本并登记；下一次选择立即可见 | 当前 planner 继续完成发布，原登记不经另一条交接队列                                                        |
| 一个 role 受阻       | Selector 返回前撤销本轮部分登记                                   | 普通等待前 planner 结束 PlacementOperation，再按 role/group 登记等待；容量序号防止遗漏等待前发生的变化     |
| 局部版本冲突         | Strategy 只刷新冲突 Endpoint 的动态评分                           | 局部预算耗尽后返回 RETRY；本轮登记清理完再恢复全局候选                                                     |
| 提交前节点失效       | Endpoint 拒绝旧身份，未交付登记全部释放                           | 当前任务结束后重选；不把节点失效变成永久绑定该节点的容量等待                                               |
| 取消/超时与发布竞争  | Context 记录取消/结束原因，拒绝晚到发布                           | 当前任务完成本地清理前仍占 planner 名额；不能提前启动同请求的新尝试                                        |
| 优先级抢占           | 在全局队列锁内领取抢占名额，锁外发起局部撤回或远端取消            | 仅保存原 PlacementOperation、Decode Endpoint 和完成 Future；结果直接交给 Selector 续选，不另存回收登记副本 |
| 停服期间抢占结果晚到 | 停止接纳新的任务；原抢占完成回调负责晚到登记                      | planner 池已停止时，完成回调直接执行关闭分支，释放登记；不再尝试发布                                       |

抢占回调的订阅发生在原任务交回 planner 名额之后，因此已经完成的 Future
也不会重入尚未退出的选点任务。抢占额度的领取、释放与等待者唤醒均受全局队列锁保护。
抢占失败只完成一次原 PlacementOperation；无结果缓存、完成标志或额外 Outcome 状态机。
普通失败重选与抢占续选共用最终结果决策，均先结束本轮资源责任，再恢复队列资格。
独立请求可以并行发布；同一 Endpoint 内的容量检查和修改仍由原状态锁串行化。
发布不再占调度线程，但仍占一个有界 planner 名额，避免用无界任务积压替代原交接队列。
普通全局等待按角色/group 接收资源变化，不保留部分登记、不永久绑定失败节点；冲突预算耗尽重新排队计算。`DeliveryDispatcher` 准备和发送批次，并把投递结果交给实际调度器处理。

`PlacementCapacityWaitQueue` 在入队时将失败节点归入 role/group 等待域。
`PlacementAvailability` 只记录 group 和 role 的变化版本，不维护单节点等待版本；
同组其他候选的容量变化也能触发重选。事件仍携带具体节点，供已发布路由处理节点失效。
选点前读取 sequence，入队时核对等待域版本；退休历史清理先推进版本下界，避免丢失期间发生的唤醒。
资源释放及响应完成的公共流程由基类执行。取消、超时、投递失败和外部响应完成
都进入请求结束流程，由该流程统一调用 `removeRequest(context)` 清除队列成员。
各入口不再分别移除队列，也不发出通用的“请求状态变化”通知。

具体实现通过 `shouldRetryRouteFailure` 决定是否撤回失败路由；基类完成全部资源释放后，
检查绝对截止时间，再将可重试请求交回 `onRequest(context)`。首次入队与撤回后恢复共用
一个入口，QUEUE 在同一把队列锁内移除旧成员、恢复原顺序，保留原优先级和截止时间。
没有第二个重试决策回调，也不再传递状态快照让调度器重新推断动作。

未发送路由的撤回直接释放精确 P/D 登记并结束当前 PlacementOperation，
不等待发送准备任务的完成回调。Decode Endpoint 删除登记时同时清除该登记的临时派发额度；
旧任务随后提交或归还许可，只能操作旧登记，不能影响同请求重选后的新登记。
Prefill 队列提交和请求投递提交也核对精确路由，拒绝已撤回任务的晚到提交。
Delivery 仅归还自己持有的成员许可，不保存准备完成标志、释放错误或完成回调列表。
停服仍由 DeliveryExecution 等待实际发送任务和 RPC 响应结束，不依赖路由撤回的等待关系。

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

对外发布前，Scheduler 在同一段 Context 临界区内校验请求、转入 Decode 派发容量并记录投递事实。
最后一次提交也检查所选 Endpoint 是否仍可用；失败后的重选或结束只由 Scheduler 处理，
Dispatcher 的批量和单请求路径不再重复推断节点失效并触发第二次失败处理。
Dispatcher 直接调用 Scheduler 的提交入口；Delivery.Member 只保存本次成员和临时许可，
不再经由 Dispatcher 回调 Scheduler。Context 不生成待写回的 DeliveryFacts，只有资源提交
完成后才记录本次投递，没有 prepare/publish 两个临时结果交接入口。
BATCH 路径记录引擎 ACK/执行状态；NON_BATCH 路径记录路由交付与调用方确认。
异步投递、取消和资源释放仍有真实竞争，因此保留精确句柄及这些边界上的同步保护。

BATCH 在发送任务内先构造完整 payload，再由同一个 `commitDelivery` 临界区核对请求、
转移 Decode 派发额度并直接记录 `SENDING`。没有对外可见的 BATCH `NOT_STARTED` 事实，
也不再有独立的 `claim` / `beginSend` 交接。编码期间取消或过期仍处于本地登记阶段；
编码失败不生成投递事实，临时许可和原登记分别由发送任务与 Scheduler 清理。
完整编码后发生的取消与远端发送结果仍使用已有精确投递身份和取消证明协议。

`RouteResult` 同时保存角色组合与精确路由身份，供队列和回调引用；没有单独的执行包装层。
`Delivery` 保存本次批次与成员许可；请求字节解析、payload 编码和响应校验统一属于
`BatchEnqueueHelper`，请求数据不再依赖投递资源对象解析 protobuf。

## 取消、过期与状态查询

- `cancelRequest(requestId, expectedBatchId, reason)` 由 Scheduler 协调请求取消和资源释放；
  Context 记录取消事实，对应 Endpoint 修改资源账本。
  `expectedBatchId` 防止旧取消请求命中重用 ID 的新代际。
- 若请求可能已到达引擎，本地资源在引擎终态或取消 fence 收敛前不会被
  乐观释放。
- 取消事实被接受后，Scheduler 直接收敛已有抢占操作并结束请求，
  不再进入通用事件续接流程重新判断是否选路或确认投递。结束流程直接区分尚无路由与已有路由，
  统一移除调度队列成员；尚无路由时完成本地清理，已有路由时释放实际取得的资源。
- `getRequestState()` 同时查询活跃 inflight 和最近终态快照；gRPC 转发也带
  单跳 fence，避免跟随者间循环代理。
- `queueTimeoutMs`（默认 3600000）给 QUEUE 所有权提供上界；
  `RequestLifecycleConfig` 另外约束 stale inflight 和已交付未确认请求。

### 请求状态与清理归属

| 事实或操作                         | 唯一修改者 / 协调者                    | 对外可见内容                                                             |
| ---------------------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| 当前选点尝试                       | RequestContext；Scheduler 领取和结束   | Snapshot 中的 PlacementOperation 身份，不暴露完成中的内部标志            |
| 选点期间的取消、超时、退役结束原因 | RequestContext.pendingEnd              | 清理完成后处理同一份原因，不再另存 pendingRetirement / inactivityExpired |
| 请求投递事实                       | RequestContext；Scheduler 协调资源提交 | 提交完成后的 DeliveryFacts，没有中间事实对象在各类间转交                 |
| Endpoint 容量与队列记录            | 对应 Endpoint                          | 精确登记和容量结果；Scheduler 不直接写资源计数                           |
| 资源清理完成                       | Context.recordResourcesReleased        | 仅接收精确路由的资源释放结果，不能改变通知状态                           |
| 停止、期限等通知完成               | Context.recordCloseNotifications       | 仅记录通知结果，不能声称资源已释放                                       |

未发送路由在选点尝试期间退役，若最终整体回滚，则撤销这条退役结束原因并允许重选；
取消、超时和其它已接受的结束原因继续保留。资源释放完成前，当前尝试仍然存在，不能开始下一次选择。
抢占恢复直接读取已有送达事实和远端证据，不再先设置 confirmationPending，
再回到抢占协调器读取该标志。未完成的抢占仍阻止对外确认。

### 一次关闭工作与响应顺序

`closeRequest` 在冻结结束事实前登记一份关闭工作；inactivity 不再先进入普通工作线程，
再登记另一份关闭任务。同一请求的通知、资源释放和归档由既有关闭通道串行执行，
重复清理通知仍可合并；携带已保留响应的工作必须执行，不能被合并丢弃。

纯本地取消先释放登记再提交响应；已知远端终态的响应仍直接交给原响应完成器，
可以先于异步资源清理完成。inactivity 的响应等待已接纳工作开始执行，避免计时线程
被资源操作阻塞。尚未完成 placement 或抢占时，沿用原来的结束裁决与等待约束。
普通调用的队列移除错误仍同步报告给调用者，不妨碍已登记工作继续清理。

关闭任务只提交一次。启动拒绝在原工作计数释放前报告失败并处理尚未交付的响应，
不重新登记或立即重投同一任务。未释放资源继续保留真实结束状态，后续明确的清理通知
可以继续推进；关停仍等待已接纳工作及其同请求续交，不伪造关闭或归档成功。

## Decode 抢占指令交付

`QueuedRequestScheduler` 在选点登记容量不足时调用 `DecodePreemption`；
`DefaultDecodePreemption` 调用 policy 规划 victim，并负责取消和资源回收。
选择阶段失败时已释放其他角色；抢占取得的 incoming Decode 登记继续交给 Selector 选择其他角色。
失败或取消释放该登记；成功发布复用它。提交失败统一结束本次尝试，再等待或重选，
不在提交阶段保留旧路由、补挂抢占额度。派发许可不足仍由 Worker 队列的派发策略处理。
`engineCancellation.mode` 选择 `RPC`（默认）或 `RETURN`。RPC 由
`DefaultDecodePreemption` 发起 Cancel 并等待权威终态；RETURN 只原子占有精确的
victim 代际和新请求预留，随后通过 Scheduler 的 `publishPlacement` 发布路由结果。

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
