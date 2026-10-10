# 附录：FlexLB 枚举现状审查快照

基线：本地 `2b2338ce9a86fbbee6b77203b1d57950dbea3346` 的生产 Java 源码（`flexlb-api/common/grpc/mock-engine/sync`），共 83 处 enum 定义，不含测试与生成代码。以下“当前”只表示该历史基线，后续局部改动见 [第一批证据](evidence/enum-implementation-round1.md) 和 [第二批证据](evidence/enum-implementation-round2.md)。本附录用于迁移时核对原枚举的所有者、合法边与分类，不是目标状态机；目标以[最终方案](final-design.md)为准。

## 判定规则

- **状态**：同一精确对象跨调用保存且会修改的事实。表中的“所有者”指唯一写入该字段的对象实例，不是 enum 类；必须有初态、带事件与身份的合法边、终态和非法边。
- **分类**：配置、身份、策略或原因，在对象创建或配置解析时确定；同一对象内不转移。改变配置/身份应建立新的配置快照或对象。
- **事件**：一次观测或命令，值不可变；事件驱动所有者的状态，不把事件种类本身当状态。
- **结果**：一次方法/RPC 的返回值，值不可变；下一次调用返回别的值不构成该结果的转移。
- **投影**：从权威事实计算的只读分类；只能由源账本更新后的新快照改变，不增加第二个可写字段。

除了下表明确列出的边，**不同值之间的转移一律非法**。重复事件可以幂等地保持原值，但必须先校验 request generation、endpoint generation、attempt/reservation/claim 等精确身份；“允许同值”不代表允许旧事件覆盖新对象。终态不能复活。非法边应在拥有者入口拒绝或判为 stale；不能靠调用方约定，也不能用 enum ordinal 推断顺序。跨所有者不能直接赋值：先由资源账本提交事实，再以精确身份通知请求所有者。

## 当前持久状态：唯一写入者与合法边

下表描述**当前实现**，不是建议保留所有字段。`∅` 表示凭证不存在或尚未创建；“结束”表示对象退出作用域，而非新 enum 值。未列出的事件和逆向边非法。

| 当前枚举 | 精确所有者 / 初态 | 合法边及触发条件 | 重构判断 |
| --- | --- | --- | --- |
| `RequestState.Phase` | `RequestSlot` / `QUEUED`，`RequestState` 仅为快照 | `QUEUED→DISPATCHING/CANCEL_REQUESTED/TIMED_OUT/FAILED`；`DISPATCHING→ACKNOWLEDGED/CANCEL_REQUESTED/TIMED_OUT/FAILED/COMPLETED`；`ACKNOWLEDGED→CANCEL_REQUESTED/TIMED_OUT/FAILED/COMPLETED`；`CANCEL_REQUESTED→CANCELLED/TIMED_OUT/FAILED/COMPLETED`；四个终态无出边。`CANCELLED` 前由 `RequestSlot` 补 `CANCEL_REQUESTED` | 对外值保持兼容；内部迁移到 `RequestStage` 前证明旧快照可准确派生 |
| `RequestSlot.SlotPhase` | `RequestSlot` / `ACTIVE` | `ACTIVE→TERMINALIZING→TERMINAL_RECORD`；只有清理门确认后才能进入终态记录 | 目标由 `RequestStage.FINALIZING/FINISHED` 与唯一清理 runner 承担，不能仅替换常量 |
| `RequestSlot.DecisionStage` | `RequestSlot` / `UNDECIDED` | 预测投递时 `UNDECIDED→WAITING_ENGINE`；匹配的 Prefill ACTIVE 可到 `PREFILL_RUNNING`；Prefill 完成可到 `WAITING_DECODE` 或 `ACCEPTED`；匹配的 Decode 接收/终态可直接到 `ACCEPTED`。早到 Engine 事实可跳过预测阶段 | 与可见性期限及 endpoint 事实核对后删除副本；不能把 ACK 充作 RUNNING |
| `RequestSlot.EngineOwnership` | `RequestSlot` / `DECODE_PENDING` | 匹配的 Decode 证据 `DECODE_PENDING→DECODE_OWNED`；无独立 Decode 的 PDFUSION 不人为推进 | 目标从精确 Decode 账本派生；禁止倒退或凭裸 requestId 推进 |
| `RequestSlot.CleanupProgress.Phase` | 单次 `CleanupProgress` / `PENDING` | `PENDING→RUNNING`；并发清理信号 `RUNNING→RUN_AGAIN`；一轮后 `RUNNING→WAITING` 或 `RUN_AGAIN→PENDING→RUNNING` | 删除前先实现单一执行者及不丢信号的重检；`RUN_AGAIN` 不是业务生命周期 |
| `RequestSlot.PublicationKind` | `RequestSlot` / `∅` | `∅→DELIVERY` 或 `∅→TERMINAL`，一次选择后不改写；发布 permit 在锁外完成 Future | 目标由一次性发布凭证与已选结果承担；`future.isDone()` 单独不足以仲裁 |
| `DeliveryClaimKind` | `RequestSlot` / `NONE`；`DeliveryClaim` 中的 kind 创建后不变 | 精确投递 claim 建立时 `NONE→ROUTE_DECISION/BATCH_ENQUEUE`；两种非 NONE 值间不转换 | 请求字段目标从 claim/冻结模式派生；对外快照保留兼容值 |
| `RouteAdmission.Ownership` | 一个 `RouteAdmission` / `PROVISIONAL` | 成功交接 `PROVISIONAL→COMMITTED`；放弃并回滚 `PROVISIONAL→CLOSED` | 保留精确凭证语义；`COMMITTED→CLOSED` 在当前对象上非法，后续资源归下游 |
| `PrefillAdmissionResources.MemberOwnership` | 每个 `Member` / `ADMISSION_OWNED` | 成功移交 `→ENDPOINT_OWNED`；身份失效 `→OWNERSHIP_LOST` | 局部交接状态；不得与请求生命周期合并 |
| `DecodeEndpoint.EngineDispatchPermit.Resolution` | 一个 permit / `ACQUIRED` | 转给 Engine `→ENGINE_LIFECYCLE_OWNED`；未发送释放 `→RELEASED`；owner 丢失 `→INVALIDATED`；代际退休 `→ENDPOINT_RETIRED` | 四个结果都无出边；重复 Engine 转交仅作幂等查询，不得再次占容量 |
| `DecodeState.ClaimOwner` | 一个精确抢占 claim / 创建时为 `SHADOW_IN_FLIGHT` 或 `ENGINE_CONFIRMED` | Engine 确认证据使 `SHADOW_IN_FLIGHT→ENGINE_CONFIRMED` | claim 所有权，不是请求状态；禁止凭普通 Cancel ACK 直接释放 Engine owner |
| `PreemptionCancelPhase` | **当前双写**：`PreemptionRegistration`（由 `RequestSlot` 锁保护）及 `DecodeState.PreemptionClaim`（由 admission 锁保护），都从 `CLAIMED` 起 | `CLAIMED→CANCEL_IN_FLIGHT`；其后 `→CANCEL_REQUESTED/NOT_FOUND_STALE/CANCEL_UNKNOWN`；`CANCEL_REQUESTED→CANCEL_UNKNOWN` | 当前没有唯一所有者，是首批重构重点；目标由一次精确抢占尝试拥有协议推进，endpoint 只保留资源所有权/结算证据。迁移前需证明跨锁通知与失败补偿；`CANCEL_REQUESTED` 不是 terminal，`NOT_FOUND_STALE` 不等于 fenced |
| `DecodeTaskPhase` | `DecodeState` 的精确 reservation；另可派生快照 | 本地未派发为 `MASTER_QUEUED_NOT_DISPATCHED`；可能发送后为 `ENGINE_MAY_HAVE_SEEN`；精确 Engine KV/运行证据进到 `ACCEPTED_NOT_RUNNING/RUNNING`；终态退出账本 | 保留资源风险分层；禁止 `ENGINE_MAY_HAVE_SEEN→MASTER_QUEUED_NOT_DISPATCHED` 来伪造未发送 |
| `PriorityPreemptionProgress` | Engine 上报的同一任务观测，由 `WorkerStatus`/endpoint 单调合并 / `NONE` | `NONE→CANCELING/CANCELED`，`CANCELING→CANCELED` | 协议观测；禁止降级，且 `CANCELED` 仍需按精确资源事实结算 |
| `PrefillState.LeaseState` | 一个 route/batch lease / `OPEN` | `OPEN→OWNED`（交接），`OPEN→CLOSED`（放弃），`OWNED→CLOSED`（结算/退休） | lease 自己归 Prefill 账本；禁止 `CLOSED→OPEN` 或把 `OWNED` 当本地可回滚 |
| `PrefillState.QueueMembership` | 一个 `RequestEntry` / `WAITING` 或明确构建的 `UNINDEXED` | `WAITING→UNINDEXED`（出队/移除），`WAITING→STOP_DETACHED`（停止时摘索引）；后者只能由精确停止清理处理 | 队列索引事实；不能用请求 Phase 代替 |
| `EndpointGenerationLifecycle.RetirementPhase` | 一个 endpoint generation / `ACCEPTING_HANDOFFS` | `→RETIRING→RETIRED`；先关新交接，待活动 handoff 与清理结束 | 禁止旧 generation 复活为 accepting |
| `EndpointRegistry.RegistryPhase` | 一个 registry / `OPEN` | `OPEN→CLOSING→CLOSED`，由 registry close 持锁执行 | 禁止关闭后发布新 endpoint |
| `BatchDeliveryStrategy.BatchTransaction.Phase` | 一个 `BatchTransaction` / `PREPARED` | `PREPARED→COMMITTED→SUBMITTED→INFLIGHT`；准备、提交、发送失败时分别可到 `TERMINAL`；`SUBMITTED→TERMINAL` 可由未接管或未发送路径完成 | 当前 `INFLIGHT` 无到 `TERMINAL` 的赋值，代表交出责任；不新增 BatchState 镜像 |
| `DefaultBatchDispatcher.PermitPhase` | 一个 `PermitReservation` / `PREPARED` | CAS：`PREPARED→SUBMITTED→RELEASED`，或 `PREPARED→RELEASED` | 保留一次释放保护；重复 close/finally 不再释放 |
| `ExpirationTimer.DeadlineState` | 一个 `DeadlineRegistration` / `PREPARED` | 安装后 `→ARMED`；回调抢先 `PREPARED→FIRED_BEFORE_INSTALL→CONSUMED`；正常触发 `ARMED→CONSUMED`；未消费态可 `→CANCELED` | 只有改为无回调期限索引后才删抢跑状态；`CONSUMED/CANCELED` 无出边 |
| `ExpirationTimer.CloseState` | 一个 timer / `OPEN` | `OPEN→CLOSING→CLOSED` | close 串行等待，不允许新注册 |
| `RequestCompletionPublisher.PublisherPhase` | 一个 publisher / `OPEN` | `OPEN→CLOSING→CLOSED` | 关闭时 drain，不能再收新发布任务 |
| `WorkerBatcher.RuntimeState` | 一个 worker generation 的 batcher / `NEW` | `NEW→STARTING→RUNNING→STOPPING→STOPPED`；启动失败或未启动停止可直接到 `STOPPED`；运行线程退出也可到 `STOPPED` | 与 endpoint generation 分开；旧 batcher 不复活 |

`RequestState.Phase` 的当前代码不允许 `QUEUED→CANCELLED` 直接赋值；实际实现会先补 `CANCEL_REQUESTED`。早期 `types.md` 曾列出直接边，现已移除，不能为了统一图形去放宽代码。`BatchTransaction.INFLIGHT` 也不能被解释成 Engine 已完成。

最终方案将投递结果的目标语义分为 `DELIVERED`、`NOT_SENT`、`PREFILL_REJECTED`、`UNKNOWN`；历史基线的 `DeliveryResult.Status` 是五值。改生产类型前应逐个核对调用方、错误码和监控。

### 必须显式拒绝的关键反例

| 非法变化 | 原因 |
| --- | --- |
| 请求任一终态 `→` 活动态；`RequestStage.FINALIZING/FINISHED→SCHEDULING/DELIVERY/TRACKING` | 同一 request generation 不复活；重试必须有新身份 |
| `DISPATCHED→WAITING`，仅因为 Enqueue 超时或 ACK 丢失 | 远端可能已收到，不能把未知结果当 NOT_SENT 并再次发送 |
| `DECODE_OWNED→DECODE_PENDING`；`ENGINE_MAY_HAVE_SEEN→MASTER_QUEUED_NOT_DISPATCHED` | 已取得的 Engine 所有权/可能送达证据不可倒退 |
| `CANCEL_REQUESTED/NOT_FOUND_STALE→已释放资源`，只凭 Cancel ACK 或普通 NOT_FOUND | 取消受理、未找到与精确 fencing/权威终态的证据强度不同 |
| `PermitPhase.RELEASED→SUBMITTED`；`DeadlineState.CONSUMED/CANCELED→ARMED` | 已释放的 permit 或已消费的 deadline 不能再次被认领 |
| `BatchTransaction.INFLIGHT→PREPARED`；批次结束直接使请求 `FINISHED` | 批次交接和推理资源结算属于不同所有者 |

## 其余生产枚举：类型、所有者及“不转移”约束

以下覆盖其余生产 Java 枚举。斜杠前是定义位置或所属对象；同行列出的每个类型共享该行的类别与所有权约束。`Kind` 均写出外层类型，避免同名混淆。它们没有合法的**同一值对象内**转移；需要改变事实时创建新事件/结果/快照或调用上表中的状态所有者。

| 所有者 / 定义位置 | 枚举 | 类别与边界 |
| --- | --- | --- |
| `FlexlbGrpcForwarder` | `ForwardOperation`, `ForwardBlockReason` | 命令分类、阻断结果；由转发调用创建 |
| `FlexlbServiceImpl` | `ScheduleOrigin` | 一次请求入口路径的观测分类 |
| `ArithmeticFormulaAst` | `Function` | 解析后的公式函数身份 |
| `DecisionPolicyConfig`, `DispatcherConfig`, `QueueOrderingConfig`, `SchedulerConfig`, `RoutingConfig` | 前四者各自的 `Type`，以及 `RoutingConfig.EstimatorType` | schema 3 配置选择；配置加载者校验合法组合，运行请求不原地换模式 |
| `VictimStage` | `VictimStage` | 抢占目标配置分类，不是 victim 的推进状态 |
| `ZkMasterEvent` | `ZkMasterEvent` | 选主与服务事件；选主服务处理后更新自己的状态 |
| `SchedulingMetadata` | `PrioritySource` | 不可变优先级来源 |
| `AdmissionRejectReason`, `StrategyErrorType` | 各自同名枚举 | 一次拒绝/错误的协议原因；保持既有错误码 |
| `WorkerStatus` | `PollKind` | 一次状态/缓存轮询的种类 |
| `RoleType`, `BackendServiceProtocolEnum` | 各自同名枚举 | worker 角色和传输协议身份；协议映射在边界校验 |
| `BalanceStatusEnum`, `StatusEnum` | 各自同名枚举 | API/异常状态码，不是调度状态 |
| `FlexMetricType`, `FlexPriorityType`, `LogLevel` | 各自同名枚举 | 指标种类、优先级、日志级别配置 |
| `TaskPhase` | `TaskPhase` | Engine 上报的 wire 值；每次观测不可变，endpoint 按精确身份归并；JSON 值不能因内部改名变化 |
| `AbstractGrpcClient` | `ServiceType` | RPC 客户端服务分类 |
| `JavaMockEngineCluster`, `MasterTargetRouter`, `MockLruBlockCache` | `CancelFaultKind`, `ErrorKind`, `AllocationFailure` | Mock 故障注入/单次错误分类；不进入生产请求状态 |
| `PlacementResult`, `CapacityBoundary`, `DeliveryResult` | 各自的 `Status` | 单次 placement/capacity/delivery 结果；`DeliveryResult` 的 NOT_SENT 与不确定远端结果必须分开 |
| `DecodeEndpoint` | `ReleaseReason`, `ReservationReleaseResult`, `EngineDispatchPermitAcquireStatus`, `EngineDispatchPermitTransferStatus`, `DispatchOutcome`, `PreemptionBeginResult`, `PreemptionDecision` | 操作原因/返回/命令；`ReleaseReason` 目标改为四个明确操作，其他值不进入请求状态 |
| `DecodeEndpoint.PreemptionUpdate`, `DecodeEndpoint.WorkerStatusFact` | 各自的 `Kind` | 不可变事件种类；先校验 reservation 再应用 |
| `PrefillState` | `CapacityStatus`, `WorkerStatusFact.Kind` | 单次容量结果、Engine 事实事件 |
| `DecodePreemptionCoordinator`, `EngineCancelChannel`, `EvictionPlanner` | `ClaimDisposition`, `CancelAck`, `VictimOwnership` | 一次 claim 处理结果、RPC 回答、规划分类；普通 NOT_FOUND 与 REQUEST_FENCED 不合并 |
| `PrefillTimePredictor` | `LearningResult` | 一次学习更新结果 |
| `RouteProjection`, `RouteProjection.Candidate` | `AfterProbeAdmission`，以及 `Candidate.InitialHeadDisposition`、`Candidate.State` | 不可变规划/工作投影；不反写容量账本 |
| `WorkSnapshot` | `Phase` | 工作量投影分类；不能作为 Engine 原始状态 |
| `CancelReason` | `CancelReason` | 请求首次取消原因；`∅→CLIENT_CANCELLED/DEADLINE_EXCEEDED` 只允许一次，后续事件不得覆盖 |
| `GlobalQueueCoordinator`, `PlacementAvailability` | `Outcome`, `ChangeKind` | 一次队列规划结果、容量/拓扑事件 |
| `RequestCompletionPublisher` | `ResponseCompletion` | 一次 Future 完成动作种类 |
| `RequestSlot.RequestEffect`, `WorkerTerminalSource`, `DeferredTerminal` | `Status`, `WorkerTerminalSource`, `Kind` | 单次处理结果、事实来源、延迟终态事件；不随请求状态推进而改写 |
| `ScheduledRequest` | `DecodeMode` | 请求捕获的 Decode 策略选择；重试不改模式 |
| `DecodeSelector` | `Availability` | 一次候选评估的三值结果 |
| `LBConsistencyConfig` | `MasterElectType` | 选主配置，当前仅 `ZOOKEEPER` |

`DeliveryResult.Status` 的目标 `DeliveryOutcome` 仅可在保持错误映射和监控分类后合并 `TIMED_OUT/UNCERTAIN` 为 UNKNOWN。`RequestSlot.DecisionStage` 和 `EngineOwnership` 消失前，必须验证 DIRECT/QUEUE、BATCH/NON_BATCH、PD/PDFUSION 的 Engine 事实来源及旧 generation 过滤。枚举数量减少是消除重复可写事实的结果，不是把状态改成 boolean、字符串或一个全局通用 enum。
