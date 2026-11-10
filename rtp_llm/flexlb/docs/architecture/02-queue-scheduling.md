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

两个实现均继承 `AbstractRequestScheduler`。基类负责请求注册、取消、超时、响应完成和资源清理；
没有独立的 `RequestLifecycle` 对象，也不需要将基类绑定到另一个调度器。
自定义调度器继承基类时需实现自己的路由流程，以及 `RequestScheduler` 的调度事件、启动/停止和排空契约。
只调整调度触发时机的扩展可以继承 Direct 或 Queued 实现，复用对应的路由发布策略。
何时调度、选择哪些请求、是否重试由具体实现决定，基类不创建调度循环。

QUEUE 模式下，`ordering.type` 和 `dispatcher.type` 是两个正交维度：

- ordering：`FIFO`（默认）或 `PRIORITY`；`SchedulingQueue` 保存全局候选及顺序，
  `PrefillRequestQueue` 保存已选中 Prefill 的成员及顺序。
- dispatcher：`BATCH`（默认）或 `NON_BATCH`；前者通过引擎 enqueue RPC 发布批次，
  后者将路由决策返回调用方，由调用方向引擎发请求。

## 提交与准入

交付链路由五个类组成：`DeliveryCoordinator` 管理准备、出队与交接；
`DeliveryDispatcher` 独立完成 BATCH、NON_BATCH 和 DIRECT 的原子提交、发送及结果上报，不反向调用 Coordinator；
`EngineDelivery` 负责 protobuf 编解码、发送、响应校验及取消重试；
`DeliveryExecution` 管理执行器、批次编号、发送额度及回调排空；
`Delivery` 保存一组交付成员和临时资源句柄。Dispatcher 在初始化时创建并复用。
NON_BATCH 将队列中选出的请求逐条返回路由，DIRECT 在选址后立即返回路由，
BATCH 经 EngineDelivery 发送 EnqueueBatch。三条路径共用 Dispatcher 的原子提交入口。
Scheduler 在选址收尾后消费 DIRECT 的路由结果；异步结果经
`DeliveryDispatcher.report` 调用 `onDeliveryResult`。QueuedRequestScheduler 直接判断入队或 DIRECT，
不再把入队回调传给 Coordinator，也不额外包装一次路由交接结果。
预测通过 `recordDeliveryPrediction` 在 RPC 前登记，准备失败通过
`recordPreparationFailure` 进入撤回或失败处理，取消确认通过 `recordRemoteCleanup`
推进资源清理；这些事件不作为投递完成上报。Dispatcher 在请求锁内提交资源，在锁外上报
提交失败；Dispatcher 处理发送失败，上层不再重复上报准备失败。抢占仍在同一请求锁内记录抢占结果
与清理确认，保留原子的状态更新。

RequestContext 分别记录投递完成、预测、准备失败和远端清理确认，不再通过混合事件分发修改状态。
Scheduler 共用精确请求校验、请求锁内记录和锁外推进；只有预测与返回路由需要查询并补记 Decode 确认。

`FlexlbServiceImpl` 调用调度器继承的 `submit(RequestContext)`。基类登记请求、安装期限，
随后直接调用本实例的 `scheduleRequest`。`RequestData.getRequestScheduler()` 指向该实际调度器。
校验、登记和交给调度策略在同一提交流程内完成，不再通过独立的 `register` 返回 Future，
然后由另一个入口反查注册结果及请求归属。超时错误由实际调度器的 `schedulingTimeoutError()` 决定：
Direct 返回 `BATCH_SLO_EXPIRED`，Queue 返回 `QUEUE_TIMEOUT`（8503 / GatewayTimeout）。注册时将确定的错误码交给 Context，
发送前过期、确认时失活和定时器超时都沿用该值。`RequestData` 只保存入口给定的 priority、deadline
和原始输入，不根据配置补建元数据或丢弃 payload；缺少调度元数据的请求返回 `INVALID_REQUEST`。
Dispatcher 校验投递输入，并在注册成功后释放不使用的 payload。Repository 只登记已验证的请求，
其 pending delivery 统计包含所有模式；对外 queue length 由 scheduler 提供，Direct 为零。
等待诊断由 Queue scheduler 决定，与错误码无关。Encoder 选择和提交都使用所属 scheduler 的同一容量接口。
PrefillEndpoint 接收装配阶段提供的不可变队列规则，启用队列时只校验规则一致性，不再从全局配置构造策略。
生产入口测试覆盖完整提交；只验证请求状态的单元测试显式隔离调度策略，不保留测试专用登记入口。

`RequestRepository` 保存活动请求和最近终态索引，`RequestContext` 保存单个请求的原子事实。
准入直接接收精确 Context，不再传递 ID 和 Future 后重新查找；仍验证目录中的对象身份，
避免旧回调操作重用 ID 的新请求。不可变的 `RoutingContext` 只标识一次操作；Context 用
`RoutingPhase` 明确区分选路、选路收尾、移除旧路由和移除收尾，无当前操作时为 `NONE`；
对应 `SELECTING`、`FINISHING_SELECTION`、`REMOVING_ROUTE`、`FINISHING_ROUTE_REMOVAL`。
`RoutingCompletion.FINISHED` 只表示普通选路操作结束，不表示路由已发布；撤回操作只能恢复、撤回成功或失败。
交付准入直接检查路由、结束原因、取消、抢占和选路事实，不另存一份路由可执行状态。
回滚失败保留的路由同时保留 `pendingEnd`，结束流程开始后由 `outcome` 阻止投递；精确资源清理仍可继续。
取消与停服需要等待当前尝试清理完成。Scheduler 通过 Context 的精确身份检查认领完成工作，旧操作和重复完成不会再次清理资源。

路由、登记和发布使用同一个尝试身份。`RequestRouteSelector` 选择并登记 P/D；角色失败时在返回前整体回滚。
Direct 和 Queued 分别实现 `selectAndPublish(request, selection)`，各自负责选路、发布和异常收尾。
Direct 立即提交 Prefill 交付、发布路由响应，并将选中端点的容量拒绝转换为请求拒绝；
Queued 将路由发布到 Prefill 队列，保留容量阻塞结果供等待或重试。两者共用 Encoder 提交和资源回滚能力，
不通过可空的队列参数切换流程。基类不包含具体选路、入队或立即交付流程。
BLOCKED 且开始异步抢占时，保留同一个选路操作及其诊断；先释放 planner 名额，再启动续接。
Queued 的 `resumeRoute` 处理抢占失败、晚到预留释放和选路收尾；旧路由移除、重新入队及队列等待诊断也由 Queued 负责。
提交期间候选由提交任务持有；Endpoint 接受后才调用 Context.publishRoute。Endpoint 操作不持有 Context 锁，
期间发生取消会拒绝发布，由提交任务释放登记后结束尝试。入队通知在 Context 接收发布、选路操作结束后发出；
通知失败不撤销交接，即使通知期间请求已经取消并归档，也不会再按未发布资源回滚。
Direct/Encoder 在投递提交成功时才接收路由，路由与投递写进同一个 Context 快照；提交前失败无需清空已发布路由。
Endpoint 的入队返回值是成员提交结果，Scheduler 不通过反查队列猜测是否提交。
队列实现须在发布成员之前完成校验，拒绝或校验异常不改变成员及其版本；通知失败不撤销已提交成员。
响应执行 Registration 接收准确响应 Future、锁检查对象以及 Scheduler 绑定的响应选择回调；
业务选择规则仍由请求流程裁决，执行器只在完成响应时调用绑定回调。创建票据必须提供回调，不能默认跳过仲裁。
响应准备失败由原来尚未提交的 Registration 发布失败结果，不另建备用登记；任务已提交或登记已结束时不再重发。
响应仍在原调用方构造，以保留构造失败的同步异常语义；构造完成后通过 Registration.prepare
绑定完成类型、响应及异常。交给收尾队列的对象已经携带完整响应，submitClose/runClose 不再
分别传递 candidate 和 Response。Registration.publish 统一接管同步/异步执行及未提交计数释放，
执行器拒绝由原票据结束；收尾队列拒绝时调用票据的 failIfUnsubmitted，不再由调用方另做 close。
准备阶段的异常恢复与收尾交接在代码结构上分开，不再通过 handedOff 布尔值推断当前责任。
请求清理每轮依次同步期限、发起尚未完成的停止操作、尝试资源释放、完成结束通知；结束通知不再在准备阶段提前执行。
每次唤醒最多三次尝试，失败仍保留后续唤醒和已成功释放的记录。提交收尾与执行收尾共用响应/释放顺序规则，
分别读取各自时点的状态，避免排队期间的变化被忽略。
Encoder 登记接收请求侧计算好的未命中 token 数与并发上限，Endpoint 不解析请求输入长度或缓存参数。
Context 不保存 PREPARED/NOT_PUBLISHED/ROLLBACK_COMPLETED 中间结果；只有本地清理异常时，
才保留尚未释放的路由供现有异步清理重试，这种记录不可派发。
Context 中的精确 route 引用表示已接收的路由，不再维护另一个 published 标志。
为清理失败保留 route 时，同时记录 pendingEnd；结束原因阻止派发，清理不能被当成正常发布。

### 选点、发布与重试的协调

QUEUE 的 planner 在同一次任务中完成选择登记、入队/发布和尝试收尾。
选择结果和登记句柄只保存在当前任务的局部变量中；不存在 PlacementPlan 或结果交接队列。
调度线程挑选全局候选、管理选路名额，并调用 `PrefillDeliveryScheduler.scheduleAt`；
Prefill 投递组件独立维护就绪索引、定时检查、投递任务名额及容量订阅，向共享驱动返回下次检查时间。
每个 Endpoint 的任务记录由 `PrefillDeliveryScheduler` 在同一把锁下维护：
`IDLE` 表示没有待处理检查，`PENDING` 表示已入待调度队列，`SCHEDULED` 表示任务已领取但未结束，
`SCHEDULED_PENDING` 表示任务期间又收到检查通知。通知和到期定时器共用入队入口；
领取任务同时出队并占用名额，任务结束保留期间到来的通知，按结果继续调度或登记期限。
每次领取都实际检查队列；容量等待先订阅再检查当前容量，不比较旧等待版本。
提交或执行失败关闭队列并清理请求；停止后不再领取任务，等待已领取任务退出后清理订阅。
全局调度器只转发 Prefill 事件，不直接修改任务记录或定时状态；
这些循环步骤只由 `flexlb-scheduling` 后台线程执行。请求、Endpoint 和容量事件只更新对应状态并唤醒线程，
`RequestScheduler` 不提供外部直接执行一轮调度的接口；关闭时等待后台线程及已接纳的 planner 任务退出。
全局队列锁不包围选点、Endpoint 提交、取消 RPC 或响应回调。

| 路径                 | 修改的状态与可见时点                                              | 协调与清理责任                                                                                         |
| -------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 选择成功             | Strategy 在各 Endpoint 锁内核对评分版本并登记；下一次选择立即可见 | 当前 planner 继续完成发布，原登记不经另一条交接队列                                                    |
| 一个 role 受阻       | Selector 返回前撤销本轮部分登记                                   | 普通等待前 planner 结束 RoutingContext，再按 role/group 登记等待；容量序号防止遗漏等待前发生的变化     |
| 局部版本冲突         | Strategy 只刷新冲突 Endpoint 的动态评分                           | 局部预算耗尽后返回 RETRY；本轮登记清理完再恢复全局候选                                                 |
| 提交前节点失效       | Endpoint 拒绝旧身份，未交付登记全部释放                           | 当前任务结束后重选；不把节点失效变成永久绑定该节点的容量等待                                           |
| 取消/超时与发布竞争  | Context 记录取消/结束原因，拒绝晚到发布                           | 当前任务完成本地清理前仍占 planner 名额；不能提前启动同请求的新尝试                                    |
| 优先级抢占           | 在全局队列锁内领取抢占名额，锁外发起局部撤回或远端取消            | 仅保存原 RoutingContext、Decode Endpoint 和完成 Future；结果直接交给 Selector 续选，不另存回收登记副本 |
| 停服期间抢占结果晚到 | 停止接纳新的任务；原抢占完成回调负责晚到登记                      | planner 池已停止时，完成回调直接执行关闭分支，释放登记；不再尝试发布                                   |

抢占回调的订阅发生在原任务交回 planner 名额之后，因此已经完成的 Future
也不会重入尚未退出的选点任务。抢占额度的领取、释放与等待者唤醒均受全局队列锁保护。
抢占失败只完成一次原 RoutingContext；无结果缓存、完成标志或额外 Outcome 状态机。

等待本地抢占名额的请求也可以由 Worker 容量变化唤醒；两种条件对应同一个请求，阻塞计数去重。
名额释放时，仍在处理本轮选址结果的请求保留唤醒登记，直到交回 planner 名额后再标记重试，
避免重试标记被扫描提前消耗或被随后的容量等待覆盖。
普通规划和抢占续接通过 `finishSchedulingTask` 归还各自名额，共用重试与唤醒规则；
抢占完成通知在队列锁外完成。抢占恢复直接调用 Selector 的带准确预留的 select 入口，
不再创建仅用于转调的选择函数。
普通规划提交被拒绝后，名额归还在 finally 中执行，自定义队列移除失败也不跳过该步骤。
`startOrWaitForDecodePreemption` 返回的是结果订阅动作；返回前抢占可能已经启动。

`PrefillQueueScheduler.scheduleOnce` 执行单个队列的一轮投递并返回等待条件；`NO_WAIT` 不表示没有执行副作用。
Prefill 按路由接收顺序追加，后来的高优先级请求不插队；优先级仍用于全局调度与抢占选择。
每次投递只提交当前队首的连续前缀，队首等待 Decode 容量时不旁路发送后面的请求。
取消、超时和抢占按准确路由移除成员，剩余成员顺序不变；重新选路后按新的接收顺序追加。
队列快照和预测保持此顺序，虚拟新请求追加到末尾。Dispatcher 提交时再次校验准确连续前缀，
防止过期快照越过队首。固定顺序可能产生队首阻塞，不承诺与旁路策略相同的吞吐或尾延迟。
`PrefillEndpoint.acceptQueuedRoute` 按 RequestContext → Endpoint 的锁顺序完成入队和路由发布，
消费者取得队列快照时只能看到已经接收的路由；取消先发生则发布被拒绝，临时队列项被移除。
通知及未交出资源的回滚在锁外执行。发送前仍检查请求是否取消，不把“曾经发布成功”当作永久有效。
过期队列成员通过 `removeExpiredRequest` 移除并取消；准备失败沿 `commitBoundary` 上报。
普通选路与撤路都通过 `finishRoutingRequest` 处理结束原因优先级和请求后续动作，
Queue 仍负责在此之前移除旧队列关联、释放准确旧路由及完成旧交付。
普通失败重选与抢占续选共用最终结果决策，均先结束本轮资源责任，再恢复队列资格。

`DeliveryCoordinator.cancel` 委托 `EngineDelivery` 执行取消并返回引擎结果及对应的 delivery 清理回执，
不解释抢占阶段。`PreemptionCompletion` 负责抢占取消超时、晚到结果处理、
请求抢占状态更新和 Decode 资源对账，完成后通过明确的请求后续动作入口推进响应与清理。
`stop` 的 Future 只通知本轮停止尝试结束；是否释放资源由 Context 中的引擎回执和本地资源记录决定，
不再另外返回一套停止结果分类。
同步状态更新直接在调用线程执行，不再向任务池登记请求工作。
独立请求可以并行发布；同一 Endpoint 内的容量检查和修改仍由原状态锁串行化。
发布不再占调度线程，但仍占一个有界 planner 名额，避免用无界任务积压替代原交接队列。
普通全局等待按角色/group 接收资源变化，不保留部分登记、不永久绑定失败节点；冲突预算耗尽重新排队计算。`DeliveryCoordinator` 准备交付成员，交给 `DeliveryDispatcher` 执行对应的交付路径，再把交付结果交给实际调度器处理。

`CapacityWaitQueue` 在入队时将失败节点归入 role/group 等待域。
`CapacityNotifier.notifyChanged(endpoint)` 通知 worker 容量或可用性发生变化。
它只记录 group 和 role 的变化版本，不维护单节点等待版本；
同组其他候选的容量变化也能触发重选。事件仍携带具体节点，供已发布路由处理节点失效。
选点前读取 sequence，入队时核对等待域版本；退休历史清理先推进版本下界，避免丢失期间发生的唤醒。
资源释放及响应完成的公共流程由基类执行。取消、超时、投递失败和外部响应完成
都进入请求结束流程，由该流程统一调用 `removeRequest(context)` 清除队列成员。
各入口不再分别移除队列，也不发出通用的“请求状态变化”通知。

具体实现通过 `shouldRetryRouteFailure` 决定是否撤回失败路由；基类完成全部资源释放后，
检查绝对截止时间，再将可重试请求交回 `scheduleRequest(context)`。首次入队与撤回后恢复共用
一个入口，QUEUE 在同一把队列锁内移除旧成员、恢复原顺序，保留原优先级和截止时间。
没有第二个重试决策回调，也不再传递状态快照让调度器重新推断动作。

未发送路由的撤回直接释放精确 P/D 登记并结束当前 RoutingContext，
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

Scheduler 选定请求后，由 Dispatcher 完成交付提交。Dispatcher 直接查询 RequestRepository，
在同一段 Context 临界区内校验请求及端点有效性、转交 Decode 派发许可并记录投递事实。
取消不能插入许可转交与状态记录之间；网络发送、容量通知和结果反馈均在请求锁外执行。
`DeliveryDispatcher.commit` 共用提交临界区：DIRECT 传入精确选址上下文，队列路径使用已经发布的路由。
Scheduler 不提供交付提交入口。Delivery.Member 只保存本次成员和临时许可。
提交未开始时，Dispatcher 反馈 `DeliveryNotStarted`，由 Scheduler 核对路由或选址身份后处理
截止时间、端点失效重选和请求结束；Dispatcher 不直接执行这些生命周期决策。
Context 只在资源提交完成后记录本次投递，不暴露中间的 DeliveryFacts。
BATCH 路径记录引擎 ACK/执行状态；NON_BATCH 路径记录路由交付与调用方确认。
异步投递、取消和资源释放仍有真实竞争，因此保留精确句柄及这些边界上的同步保护。

BATCH 在发送任务内先构造完整 payload，再由 Dispatcher 的同一个提交临界区核对请求、
转移 Decode 派发额度并直接记录 `SENDING`。没有对外可见的 BATCH `NOT_STARTED` 事实，
也不再有独立的 `claim` / `beginSend` 交接。编码期间取消或过期仍处于本地登记阶段；
编码失败不生成投递事实，临时许可和原登记分别由发送任务与 Scheduler 清理。
完整编码后发生的取消与远端发送结果仍使用已有精确投递身份和取消证明协议。

`RouteResult` 同时保存角色组合与精确路由身份，供队列和回调引用；没有单独的执行包装层。
`Delivery` 保存本次批次与成员许可；请求字节解析、payload 编码和响应校验统一属于
`EngineDelivery`，请求数据不再依赖投递资源对象解析 protobuf。

队列交付按下面的责任边界推进：

1. `PrefillQueueScheduler.processSelection` 只提交选中的连续队头、预测信息和交付策略；
   Coordinator 在内部创建 `Delivery`，准备预测与临时许可，上游不创建或清理交付对象。
2. Coordinator 将不可变的 `PrefillEndpoint.DeliverySelection` 传给 `dequeueForDelivery`。
   Endpoint 在自己的锁内检查精确路由、队头顺序、当前时间下的截止期限、容量和节点可用性，
   一次完成出队及工作量登记，返回出队前的工作快照。失败返回具体拒绝原因，不修改队列或容量。
   该入口只修改选中的有效前缀；未选中的成员不参与其校验。
   队列有效性统一在此检查；出队被拒绝时，Coordinator 归还已准备的临时许可，不提交交付。
3. Coordinator 接收出队前的工作快照，将交付交给 Dispatcher。队列长度在监控上报时读取，
   不再通过交付对象传递出队时的长度副本。
   交接前的异常由 Coordinator 释放临时资源；交接后由 Dispatcher 负责，同步路由发布、异步发送或执行器拒绝各自完成清理。
   Coordinator 不查询执行许可的内部状态，也不重复清理已交出的成员。
   Coordinator 处理已交付前缀之后的准备失败成员，上游不读取 `Delivery` 的中间失败字段。
   准备失败的精确队列移除和结果通知统一由 Coordinator 完成；回调异常被记录，已提交前缀继续执行。
   Endpoint 不接收 `Delivery`，不设置其流程标志；结果本身不持有需要关闭的资源。
4. BATCH 任务编码后，`DeliveryDispatcher.commit` 在请求锁内检查取消、期限与精确路由，
   转移 Decode 许可并记录 `SENDING`。出队成功不代表请求已发送；出队后取消仍会阻止未开始的发送。
   发送前失败释放临时许可并上报准备失败，已调用 RPC 的不确定结果继续沿原有反馈和取消协议处理。

## 取消、过期与状态查询

- `cancelRequest(requestId, expectedBatchId, reason)` 由 Scheduler 协调请求取消和资源释放；
  Context 记录取消事实，对应 Endpoint 修改资源账本。
  `expectedBatchId` 防止旧取消请求命中重用 ID 的新代际。
- 若请求可能已到达引擎，本地资源在引擎终态或取消 fence 收敛前不会被
  乐观释放。
- 取消事实被接受后，Scheduler 直接结束请求，取消入口不推进抢占协议，
  不再进入通用事件续接流程重新判断是否选路或确认投递。结束流程直接区分尚无路由与已有路由，
  统一移除调度队列成员；尚无路由时完成本地清理，已有路由时释放实际取得的资源。
- `getRequestState()` 同时查询活跃 inflight 和最近终态快照；gRPC 转发也带
  单跳 fence，避免跟随者间循环代理。
- `queueTimeoutMs`（默认 3600000）给 QUEUE 所有权提供上界；
  `RequestLifecycleConfig` 另外约束 stale inflight 和已交付未确认请求。

### 请求状态与清理归属

| 事实或操作                         | 唯一修改者 / 协调者                    | 对外可见内容                                                             |
| ---------------------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| 当前选点尝试                       | RequestContext；Scheduler 领取和结束   | Snapshot 中的 RoutingContext 身份，不暴露完成中的内部标志                |
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

`closeRequest` 冻结结束事实后，根据请求类型直接执行或提交清理操作。
同一请求的通知、资源释放和归档由 `RequestCloseQueue` 串行执行，
重复清理通知仍可合并；携带已保留响应的工作必须执行，不能被合并丢弃。

纯本地取消先释放登记再提交响应；已知远端终态的响应仍直接交给原响应完成器，
可以先于异步资源清理完成。inactivity 的响应等待已接纳工作开始执行，避免计时线程
被资源操作阻塞。尚未完成 placement 或抢占时，沿用原来的结束裁决与等待约束。
普通调用的队列移除错误仍同步报告给调用者，不妨碍已接纳的清理操作继续执行。

启动拒绝由清理队列报告给对应操作，处理尚未交付的响应；未释放资源继续保留真实结束状态，
后续明确的清理通知可以继续推进。Context 只保存业务事实，不管理任务登记或线程。

`RequestTaskPool` 只执行异步提交的任务，等待范围仅覆盖这些任务及其在排空前提交的续接。
取消执行由 Dispatcher 管理；Dispatcher 等待停止流程和正在应用的取消回调，
已失去请求关联的远端 RPC 不阻止本地关闭。停服先停止调度和投递来源，推进已有请求结束，
等待取消、清理及响应处理，再关闭执行器；执行器停止接纳后不再接收后续提交。

## Decode 抢占指令交付

`QueuedRequestScheduler` 在选点登记容量不足时调用 `DecodePreemption`；
`DefaultDecodePreemption` 调用 policy 规划 victim，并负责取消和资源回收。
选择阶段失败时已释放其他角色；抢占取得的 incoming Decode 登记继续交给 Selector 选择其他角色。
失败或取消释放该登记；成功发布复用它。提交失败统一结束本次尝试，再等待或重选，
不在提交阶段保留旧路由、补挂抢占额度。派发许可不足仍由 Worker 队列的派发策略处理。
`engineCancellation.mode` 选择 `RPC`（默认）或 `RETURN`。RPC 由
`DefaultDecodePreemption` 发起 Cancel 并等待 Endpoint 确认被选中预留的资源已全部释放；RETURN 只原子占有精确的
victim 代际和新请求预留，随后通过 Scheduler 的 `selectAndPublish` 续接并发布路由结果。

RPC 等待通过 Endpoint 的容量变更通知唤醒，再按抢占编号和精确预留查询资源账本，
不等待被抢占请求的完成 Future。先订阅再查询，避免订阅前已释放资源造成漏唤醒；
结束等待时移除监听。提交新请求预留时仍检查请求是否有效、等待是否超时及 Endpoint 当前状态。

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

### 队列决策与选点预测的单向依赖

PrefillQueueScheduler 的等待 Decision 仅负责派发等待、资源订阅和诊断。
超时诊断由 QueuedRequestScheduler 通过 PrefillDeliveryScheduler 直接读取对应队列的等待信息，
Endpoint 不保存调度诊断回调。
observeWait 不再修改 Endpoint 的预测版本；容量回调只唤醒调度。
选点预测使用队列/工作量快照；Prefill 与 Decode 分别登记本请求的影响。
全局执行器和队首 Decode 容量只影响派发，不能借此拒绝另一个请求的选点。
已选中但尚未入 worker 队列的请求仍参与预测和容量统计。

真正的 QUEUE 超时返回 `QUEUE_TIMEOUT`（8503 / GatewayTimeout），不使用容量拒绝码 `RESOURCE_EXHAUSTED`；
DIRECT 保留 `BATCH_SLO_EXPIRED`。超时保留 deadline 原因和实际等待阶段。未取得完整路由时，附加当前尝试最后一次选点证据；
新尝试和成功发布会清除旧证据，取消后冻结证据。已进入 Prefill 队列时，附加带采样时间、
队首 requestId/enqueueSeq 的队列观察，明确它是队首事实，不能解释为本请求的 Decode 不足。
最后一次观察不代表超时瞬间资源仍然不足。候选消失、容量不足、优先级阻塞和版本冲突分别返回，
不能从“没有成功路由”推断整个集群资源耗尽。
