# RequestSlot 接口收敛与逐函数清单

## 源码组织规则

每个功能块按“入口 → 身份/条件判断 → 状态更新 → 锁外执行”阅读。只被这一功能使用的 private 方法放在该块中；跨功能共用的收尾、响应仲裁和锁断言分别集中。访问权限由封装需要决定，不再用 public/private 把一个功能拆到文件两端。

| 类 | 功能块 |
| --- | --- |
| RequestSlot | 身份状态、准入、派发、派发失败清理、Worker/退出、取消/失败/关闭、三类期限、Timer 关闭、抢占、共同收尾、响应、锁断言 |
| ExpirationTimer | 调度期限、可见性期限、沉默期限、句柄注册/取消、保留期维护、关闭、异常汇总 |
| RequestCompletionPublisher | 发布许可、响应入口、队列执行、关闭、异常汇总 |
| RequestTerminalCleanup | 共同收尾、派发失败两侧结算、独立执行/异常汇总 |

本轮组织调整保留了每个函数的声明和函数体，仅移动位置并补充功能块注释；已逐方法核对相同内容，并重新编译主代码和测试。

## 实际暴露的接口：先看这里

Slot 管两个生命周期：**给请求返回什么结果**，以及 **这个请求的资源何时结清**。成功 ACK 可以先返回，Engine 继续执行；失败也可以先返回，Slot 继续保留未完成的清理义务。

调用路径是 `RequestRegistry → RequestSlot → 已有执行组件`。业务调用者不需要、也不应该自己调用锁内步骤。

目前类本体有 116 个方法：69 个 private，47 个非 private。47 个中，23 个是业务事件入口，其余 24 个是查询、初始化与组件协作接口。这个数字包含内部组件合同，不能把它说成只有几个公开方法。源码按功能块排列，同一功能的入口与 private 实现相邻；下表标明各接口的调用方。

| 谁调用 / 发生什么事 | 可以调用哪些入口 | 一次调用负责什么 |
| --- | --- | --- |
| Registry：安排请求 | `tryBeginAdmissionHandle`、`commitRoute`；句柄 `close/terminate` | 取得准入责任、绑定并发布调度请求、完整结算本次准入 |
| Registry / 派发策略：发送请求 | `prepareDispatch`、`claimDelivery`、`setDeliveryPrediction`、`publishRoute`、`failDeliveryPreparation`；claim 的 `complete` | 原子准备/交接、设置观察期限、接收派发结果或发布路由结果 |
| Registry：Worker 事实 | `processPrefillStatus`、`processDecodeStatus`、`recordPrefillRetirement`、`recordDecodeRetirement` | 核对来源，更新执行与资源状态，推进响应或收尾 |
| Registry：主动结束请求 | `cancelRequest`、`recordSchedulingFailure`、`claimShutdownAction`、`terminateLocallyAndPublishResponse` | 接收不同结束原因，经共同条件选择结果与清理责任 |
| Timer / Registry：到期检查 | `onSchedulingDeadline`、`onInactivityDeadline`、`onDecisionVisibilityDeadline`、`expireInactiveRequest` | 分别处理调度期限、沉默期限、可见性诊断；显式检查不消费定时器句柄 |
| Registry / PreemptionRegistration：抢占 | `tryInstallPreemption`、`updatePreemption`、`releasePreemption`、`completePreemption` | 绑定精确协议身份，接收协议进展，释放等待条件 |

下面三组不是业务流程中的下一步，调用者不能自行拼装它们代替事件入口：

| 专用调用方 | 协作接口 |
| --- | --- |
| Registry（13） | `requestId`、`future`、`createdAtMs`、`ownsFuture`、`snapshot`、`activeItem`、`ownsActiveItem`、`activeItemForReservation`、`isOpen`、`isLiveGeneration`、`isRemovableTerminalRecord`、`detachGeneration`、`configureInactivityTimeout` |
| Timer（6） | `inactivityDeadlineAtMs`、`decisionDeadlineAtMs`、`installInactivityDeadline`、`installRequestDeadline`、`installDecisionDeadline`、`detachDeadlinesForTimerClose` |
| Publisher / Cleanup（5） | `completeExternal`、`selectPublication`、`submitTerminalResponse`、`requireCleanupOwner`、`commitTerminalRecord` |

`decodeOwnsRequest`、`isTerminalRecord`、`requestInactive`、`needsDecisionConfirmation`、`hasCancellationFirstCause`、`requireCancellationFirstCause` 已不再作为组件接口暴露。必要的内部判断采用 private `...Locked`；状态单测在测试工具中持锁检查，不要求生产代码开放这些方法。

## 已实施的结构

RequestSlot 是单请求的状态仲裁者：确认精确身份、处理业务事实、选择一次响应、合并资源进度、决定何时能结束跟踪。响应已确定不等于资源已释放；所有清理未完的路径仍须经过同一个关闭门。

| 边界 | 当前实现 |
| --- | --- |
| 事件入口、查询、组件回调 | 自行管理 Slot 锁；调用者无需先加锁 |
| 锁内决策 | 全部为 private，名称以 Locked 结尾；不得在这里完成 Future |
| 准入结束 | AdmissionHandle 的 close/terminate 共用 finishAdmission → settleAdmissionLocked；删除原来的多层完成摘要 |
| 请求结束 | selectDeliveryFailureLocked 选择失败响应；decideRequestEndLocked 解释结束事实；claimTerminalActionLocked 认领责任；tryCloseAfterCleanupLocked 只推进既定失败清理；commitTerminalRecord 提交最终记录 |
| 超时 | onSchedulingDeadline、onInactivityDeadline、onDecisionVisibilityDeadline 含义分开，不再依赖 expire 重载 |
| 执行 | RequestTerminalCleanup 负责 Endpoint 清账；ExpirationTimer 负责定时器排期与取消；RequestCompletionPublisher 负责 ACK 指标、异步与同步发布 |
| 派发事务 | prepareDispatch / claimDelivery 在一个临界区内校验资格并调用由 Registry 提供的具体事务；Slot 不再依赖 BatchTransaction / RouteTransaction 类型 |
| 返回语义 | RequestEffect 的 STALE 表示失效，DEFERRED 表示保存等待，APPLIED 表示已处理；可执行动作单独存放，不再用 READY 同时表达“处理结果”和“存在动作” |

事务准备、Endpoint 所有权交接和抢占协调仍有必须与 Slot 状态校验原子完成的操作，保持既有 Slot → Endpoint 锁顺序。它们与可能触发回调的普通清账、Future 发布不同，不能机械地全部搬到锁外。

## 一个流程现在怎么读

```text
AdmissionHandle.close() / terminate(failure)
  → finishAdmission
      锁内：关闭本次句柄 → settleAdmissionLocked 一次性结算暂存事实
      锁外：执行选中动作，或推进已有失败清理
      独立尝试：安排沉默检查 → 归还准入门闩 → 抛出汇总异常
```

```text
派发失败
  锁内：selectDeliveryFailureLocked，响应只选一次
  锁外：提交响应；cleanUpRequest 委托两侧清账
  锁内：单调合并清理结果 → tryCloseAfterCleanupLocked
  具备关闭条件：认领动作 → 锁外执行 → commitTerminalRecord
```

晚到的 pending 清账结果不能覆盖先到的完成事实；清账进行中或失败时，Worker/取消/超时不能提前删除跟踪记录。

## 合并与移除

- 删除 completeAdmissionHandle、claimAdmissionHandleTermination、finishAdmissionHandle、promoteAdmissionCancellation、processAdmissionResultLocked 和 AdmissionHandleCompletion。
- finishExternalResponse/Failure/Cancellation 合入共同处理；三种 Future 完成操作仍由 RequestFuture 提供标准入口。selectResponse/Failure/Cancellation 合为 selectPublication。
- assertInvariant/invariantHolds 合一；timeoutErrorType、isCurrentGeneration 直接读字段。
- ownsActiveGenerationLocked 和 ownsResourceTrackingLocked 保留，分别回答业务生命周期与资源跟踪是否仍有效，不能混用。
- Timer 和 Cleanup 的执行细节回归已有组件；各类内部按功能组织，入口、决策和执行步骤相邻；没有引入新的通用事件框架。

## 逐函数阅读约定

签名列出输入和返回类型，说明补充返回值语义与副作用。private Locked 方法由内部调用者持锁；synchronized 方法自行持锁。锁外入口可能在内部短暂加锁。

`TerminalAction` 是一次执行责任，不是已释放证明；`SelectedPublication` 是已选结果，不是已完成 Future。`null` 仍用于“没有取得句柄/动作”，boolean 仍用于明确的资格判断或 Future 标准结果，不再混入内部事件处理的三态语义。

当前 RequestSlot 本体有 116 个手写方法及一个构造器；本轮前为 143 个方法。下表覆盖当前文件全部 139 个手写方法/构造器（不展开 record 自动生成成员）。

## AdmissionHandle

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `AdmissionHandle(RequestSlot owner)` [L58](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L58) | 构造 | 输入 Slot，保存非空 owner；返回句柄，不单独取得准入资格。 |
| `public void terminate(Response failure)` [L63](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L63) | 锁外入口 | 输入失败 Response；CAS 保证句柄只结束一次，再调用 Slot.terminateAdmission。void；重复调用忽略，不代表资源已经释放。 |
| `public void close()` [L73](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L73) | 锁外入口 | 无输入；CAS 后调用 Slot.finishAdmission。void；重复调用忽略。 |

## DeliveryClaim

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `private DeliveryClaim(RequestSlot slot, ScheduledRequest item, DeliveryClaimKind kind, long correlationId)` [L89](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L89) | 构造 | 输入 Slot、请求、派发种类和关联 ID；保存精确身份，返回 claim。 |
| `public void complete(DeliveryResult result)` [L96](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L96) | 锁外入口 | 输入 DeliveryResult；委托 Slot.completeDelivery，void。Slot 内校验身份与回调次数；仅用于批量派发结果。 |

## CleanupProgress

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `CleanupProgress(ScheduledRequest item, DeliveryResult.Status source)` [L162](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L162) | 构造 | 输入请求及失败来源；依据是否存在对应 Endpoint/reservation 初始化两侧结清标记，保存来源；不复制请求。 |
| `boolean ready()` [L168](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L168) | 持 Slot 锁读取 | 无输入；返回 phase 为 WAITING 且 Prefill、Decode 均结清。这个 true 尚不包含准入、抢占条件。 |

## RequestSlot

### 构造

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `RequestSlot( RequestCompletionPublisher completionPublisher, long requestId, ExpirationTimer expirationTimer, RequestTerminalCleanup terminalCleanup, Runnable admissionFinished)` [L173](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L173) | 构造 | 输入 Publisher、请求 ID、Timer、Cleanup 和准入门闩释放回调；初始化请求状态及 Future，不分配 Endpoint 资源。指标执行依赖已从 Slot 移除。 |

### 请求身份与状态：只读查询、生命周期条件和状态提交

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `long requestId()` [L191](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L191) | 只读常量 | 无输入；返回该代请求的 ID，不修改状态。 |
| `RequestFuture future()` [L195](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L195) | 只读常量 | 无输入；返回绑定的 RequestFuture，同一个实例。 |
| `long createdAtMs()` [L199](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L199) | 只读常量 | 无输入；返回创建时间毫秒值。 |
| `boolean ownsFuture(CompletableFuture<?> expected)` [L203](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L203) | 只读常量 | 输入 Future；返回是否与 Slot 持有的是同一对象，非 equals 比较。 |
| `synchronized RequestState snapshot()` [L207](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L207) | 自行持锁 | 无输入；返回请求 ID、公开阶段、派发类型/批次、时间、说明的快照。不包含清理进度和资源是否释放。 |
| `synchronized ScheduledRequest activeItem()` [L214](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L214) | 自行持锁 | 无输入；返回仍可参与请求决策的 item，否则 null。已失败但仍待清理时也返回 null。 |
| `synchronized boolean ownsActiveItem(ScheduledRequest expected)` [L218](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L218) | 自行持锁 | 输入请求对象；返回活跃代且 item 为同一实例。 |
| `synchronized ScheduledRequest activeItemForReservation(long reservationToken)` [L222](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L222) | 自行持锁 | 输入 reservationToken；匹配当前活跃请求时返回 item，否则 null。 |
| `synchronized boolean isOpen()` [L231](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L231) | 自行持锁 | 无输入；返回 Slot 是否还能接受准入，不检查 Registry 是否仍保留该代。 |
| `synchronized boolean isLiveGeneration()` [L238](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L238) | 自行持锁 | 无输入；当前代且尚未转为纯终态记录就为 true；包括正在收尾。 |
| `synchronized boolean isRemovableTerminalRecord(long updatedBeforeMs)` [L242](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L242) | 自行持锁 | 输入保留时间截止值；只有已终态、无 item、当前代且更新早于截止才返回 true。自身不移除 Registry 记录。 |
| `synchronized void detachGeneration()` [L250](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L250) | 自行持锁 | 无输入；Registry 移除当前代时置 currentGeneration=false。void；重复摘除抛异常。 |
| `private boolean ownsActiveGenerationLocked()` [L258](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L258) | private，要求持锁 | 无输入；返回当前代、Slot ACTIVE、公开状态未结束是否同时满足。 |
| `private boolean ownsResourceTrackingLocked()` [L265](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L265) | private，要求持锁 | 无输入；返回当前代且 Slot ACTIVE。允许请求结果已失败但清理未结束。 |
| `private boolean decodeOwnsRequestLocked()` [L270](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L270) | private，要求持锁 | 无输入；返回 Slot 是否观察到 Decode 拥有请求。它不是查询 Endpoint 当前是否还有账单。 |
| `private boolean isTerminalRecordLocked()` [L274](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L274) | private，要求持锁 | 无输入；返回是否已成为只保留查询信息的终态记录。 |
| `private void ensureTransitionAllowedLocked(RequestState.Phase next)` [L278](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L278) | private，要求持锁 | 输入目标 RequestState.Phase；校验状态转换是否合法。void；非法抛异常，不修改状态。 |
| `private RequestState transitionLocked( RequestState.Phase next, String message)` [L286](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L286) | private，要求持锁 | 输入目标阶段、说明；校验后更新 state/detail/updatedAtMs 并检查不变量。返回快照；阶段相同直接返回，不更新说明。 |
| `private RequestState commitTerminalStateLocked(TerminalOutcome outcome)` [L300](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L300) | private，要求持锁 | 输入最终意图；已结束则保留现有状态，否则必要时经过 CANCEL_REQUESTED 后转换。返回 RequestState；不清理 item/定时器。 |
| `private void assertInvariantLocked()` [L310](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L310) | private，要求持锁 | 无输入；统一检查准入/抢占互斥、暂存事实拥有者、最终记录不持有请求资源。void，违规抛异常。已合并 assertInvariantLocked → invariantHolds。 |

### 准入：开始 → 发布调度结果 → 结束并结算暂存事实

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized AdmissionHandle tryBeginAdmissionHandle()` [L351](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L351) | 自行持锁 | 无显式输入；确认开放且无 item/准入/抢占，安装新的准入句柄。返回句柄或 null；Registry 的全局计数由外部管理。 |
| `PlacementResult.Status commitRoute(ScheduledRequest exact, BooleanSupplier publication)` [L365](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L365) | 自锁→锁外 | 输入 ScheduledRequest 和返回 boolean 的发布回调；先绑定 item，在锁外发布，失败时撤销绑定。返回 SUCCESS/CLOSED/BLOCKED；回调异常保留回滚异常后抛出。 |
| `private boolean tryBindItemForPublicationLocked(ScheduledRequest candidate)` [L392](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L392) | private，要求持锁 | 输入 ScheduledRequest；检查请求 ID、Future、准入 owner、阶段，再绑定 item。返回是否绑定成功；未发布队列也未派发。 |
| `private void rollbackItemPublicationLocked(ScheduledRequest exact)` [L408](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L408) | private，要求持锁 | 输入原 ScheduledRequest；确认仍持有同一准入和 item 后清空绑定。void；身份已变抛异常。 |
| `private void finishAdmission(AdmissionHandle exact, Response failureResponse)` [L420](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L420) | 自锁→锁外 | 输入本次句柄和可空失败响应；关闭准入，在一次锁内结算中处理暂存事实，锁外执行动作或继续失败清理。随后独立尝试安排沉默检查、归还准入门闩，汇总异常后抛出。void。 |
| `private RequestEffect settleAdmissionLocked(Response failure)` [L446](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L446) | private，要求持锁 | 输入可空准入失败；消费暂存取消、沉默超时、Worker/退休事实，保留首次取消原因并按优先级选定动作。返回非空 RequestEffect；不生成 AdmissionHandleCompletion 中间摘要。 |
| `private void retainAdmissionTerminalLocked(DeferredTerminal candidate)` [L498](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L498) | private，要求持锁 | 输入结束事件；保留一个候选，优先 Decode 退出，其次权威 Worker 事实。void；不是保存全部事件历史。 |
| `private void retainAdmissionPrefillRetirementLocked(PendingPrefillRetirement candidate)` [L506](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L506) | private，要求持锁 | 输入 Prefill 退出记录；保存一次，相同身份可重复，不同身份抛异常。void。 |

### 派发：准备 → 交接 → 预测 → 接收结果 / 发布路由

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized <T> CapacityBoundary.Attempt<T> prepareDispatch(ScheduledRequest exact, java.util.function.Supplier<CapacityBoundary.Attempt<T>> prepare)` [L520](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L520) | 自行持锁 | 输入精确请求和返回 Attempt<T> 的准备事务；同一锁内核验准入资格并执行事务。返回原事务结果；身份失效返回 OWNERSHIP_LOST。回调不得发布响应或调用用户代码。 |
| `synchronized DeliveryClaim claimDelivery(ScheduledRequest exact, DeliveryClaimKind kind, long correlationId, BooleanSupplier transferToEndpoint)` [L527](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L527) | 自行持锁 | 输入精确请求、派发种类、关联 ID 和交接事务；同一锁内核对身份、转移 Endpoint 所有权并记录 DISPATCHING。返回 DeliveryClaim；资格失效返回 null，非法身份或交接失败抛异常。 |
| `private boolean ownsPreparedDeliveryLocked(ScheduledRequest exact)` [L547](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L547) | private，要求持锁 | 输入精确请求；返回是否 QUEUED、开放、未派发且无抢占，作为准备和交接的资格条件。 |
| `private boolean ownsDeliveryClaimLocked(DeliveryClaim claim)` [L553](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L553) | private，要求持锁 | 输入 claim；先检查属于本 Slot，再调用请求/类型/批次身份判断。返回 boolean；外来或 null claim 抛异常，不只是返回 false。 |
| `private boolean ownsDeliveryClaimLocked( ScheduledRequest expected, DeliveryClaimKind kind, long expectedBatchId)` [L559](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L559) | private，要求持锁 | 输入请求对象、派发种类、批次；返回是否匹配当前活跃派发。它不检查 completeCalled。 |
| `void setDeliveryPrediction(DeliveryClaim claim, WorkSnapshot work, long predictedMs)` [L570](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L570) | 自锁→锁外 | 输入 claim、前序工作快照、预计耗时；更新观察期限并重挂定时器。void；不确认派发，不完成 Future。 |
| `void publishRoute(DeliveryClaim claim, WorkSnapshot work, long predictedMs)` [L580](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L580) | 自锁→锁外 | 输入路由 claim、工作快照、耗时；同一锁内更新预测并确认路由，锁外发布响应。void；批量 claim 非法，重复预测会拒绝。 |
| `private void completeDelivery(DeliveryClaim claim, DeliveryResult result)` [L594](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L594) | 自锁→锁外 | 输入批量 claim 和 DeliveryResult；一次性接收批量回调。明确失败立即选定失败并清理；成功确认 ACK；不确定结果等待 Engine 证据。void；重复有效回调抛异常，旧重复回调忽略。 |
| `private RequestEffect acknowledgeDeliveryLocked(long expectedBatchId, PreemptionRegistration signal)` [L638](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L638) | private，要求持锁 | 输入批次身份、可空抢占通知对象；无取消/抢占阻塞时置 ACKNOWLEDGED 并预留响应许可；否则暂存确认。返回 RequestEffect；尚未写入 Future。 |
| `void failDeliveryPreparation(ScheduledRequest exact, Throwable cause)` [L666](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L666) | 自锁→锁外 | 输入精确请求和准备异常；符合未派发条件才以 NOT_SENT 选定失败、发布并清理。void；资格已失效直接退出。 |
| `private static String detailOf(Throwable cause)` [L680](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L680) | 纯格式化 | 输入可空 Throwable；返回错误消息、类名或默认派发失败说明。 |

### 派发失败清理：选定响应 → 两侧结算 → 单调合并进度

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `private SelectedPublication selectDeliveryFailureLocked(ScheduledRequest exact, DeliveryResult.Status source, String detail)` [L692](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L692) | private，要求持锁 | 输入原请求、失败来源、说明；建立 CleanupProgress，提交 FAILED/已有取消状态，认领响应。返回 SelectedPublication 或 null；不在这里发布 Future、执行 Endpoint 清理。 |
| `private void cleanUpRequest(ScheduledRequest exact, DeliveryResult.Status source)` [L718](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L718) | 锁外，阶段间短暂持锁 | 输入精确请求、失败来源；锁内认领清理轮次，锁外委托 Cleanup 分别结算两侧，锁内单调合并结清进度，再继续被请求的下一轮或认领结束动作。void；失败义务仍保留。 |
| `private void resumeCleanup()` [L783](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L783) | 自锁→锁外 | 无显式输入；从 Slot 取 item 与清理来源，调用 cleanUpRequest。没有待清理进度则退出；是多个解阻事件共用的恢复入口。 |
| `private TerminalAction tryCloseAfterCleanupLocked()` [L795](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L795) | private，要求持锁 | 无输入；读取既定结果和资源进度，经唯一关闭门尝试认领动作。返回 TerminalAction 或 null；不重新决定响应、不发布结果。替代 finishRequest(null)。 |

### Worker 事实与 Endpoint 退出：接收、核验、推进

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `void processPrefillStatus(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact)` [L803](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L803) | 自锁→锁外 | 输入 Prefill Endpoint、角色、Worker 事实；锁内处理事实，锁外处理定时器/响应/收尾。释放抢占等待时继续清理。void；这是 Prefill 状态业务入口。 |
| `private EngineObservation applyPrefillStatusLocked(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact, long nowMs)` [L823](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L823) | private，要求持锁 | 输入 Endpoint、角色、事实、观察时间；过滤旧身份，更新活跃时间、预测阶段或清理事实，必要时处理抢占。返回 EngineObservation；其中动作尚未执行。 |
| `private boolean ownsPrefillFactLocked(PrefillEndpoint source, ScheduledRequest expected)` [L868](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L868) | private，要求持锁 | 输入 Prefill Endpoint、请求；返回是否属于仍跟踪的精确请求及 Endpoint。 |
| `void processDecodeStatus(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact)` [L873](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L873) | 自锁→锁外 | 输入 Decode Endpoint、Worker 事实；锁内更新所有权/结束证据，锁外执行结果。void；这是 Decode 状态业务入口。 |
| `private EngineObservation applyDecodeStatusLocked(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact, long nowMs)` [L882](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L882) | private，要求持锁 | 输入 Endpoint、事实、观察时间；过滤旧 reservation，记录活跃/接受/结束证据。返回 EngineObservation；失败已选定时只推进清理。 |
| `private boolean ownsDecodeFactLocked( DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation)` [L902](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L902) | private，要求持锁 | 输入 Decode Endpoint、reservation；返回 Endpoint 和 reservation 是否与跟踪身份一致。 |
| `private DecodeAcceptance markDecodeAcceptedLocked()` [L913](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L913) | private，要求持锁 | 无输入；活跃时写 DECODE_OWNED/ACCEPTED，解除疑似丢失并摘下观察定时器。返回 DecodeAcceptance，非活跃时 DEFERRED；没有 ACK 发布。 |
| `private void executeEngineObservationEffects(EngineObservation observation)` [L926](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L926) | 锁外 | 输入 EngineObservation；锁外取消旧决策定时器，独立尝试安排新定时器及执行动作，汇总异常。void；已删除重复 work 参数。 |
| `void recordPrefillRetirement(PrefillEndpoint source, ScheduledRequest exact)` [L933](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L933) | 自锁→锁外 | 输入退出的 Prefill Endpoint 和精确请求；已有失败则记录 Prefill 结清，否则按所有权条件决定结束或暂存。void。 |
| `private TerminalAction claimPrefillRetirementLocked(PendingPrefillRetirement pending)` [L949](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L949) | private，要求持锁 | 输入可空退出记录；未交付且无 Decode/抢占 owner 才能结束；准入未完则暂存。返回 TerminalAction 或 null；不执行清理。 |
| `void recordDecodeRetirement(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact)` [L964](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L964) | 自锁→锁外 | 输入退出的 Decode Endpoint 和 reservation；登记退出证据并执行结果。void；不凭 Prefill 退出推断 Decode 已释放。 |
| `private RequestEffect applyDecodeRetirementLocked( DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation, String detail)` [L973](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L973) | private，要求持锁 | 输入 Decode Endpoint、reservation、说明；已有失败记录结清，否则准入期间暂存、抢占期间完成登记，再选择结束。返回 RequestEffect。 |

### 取消、调度失败与关闭：确定原因，再进入共同收尾

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `public RequestState cancelRequest(long expectedBatchId, CancelReason reason)` [L1008](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1008) | 自锁→锁外 | 按批次身份记录第一次取消；可本地结束时执行收尾。返回锁内取得的 RequestState；身份不匹配返回 null。返回快照不保证已反映随后执行的收尾，不等于资源已释放。 |
| `private boolean recordCancellationLocked(CancelReason reason, String message)` [L1027](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1027) | private，要求持锁 | 输入取消原因/说明；只写第一原因，准入中先暂存；关闭准入并写 CANCEL_REQUESTED。返回是否首次接受，不做 Endpoint 清理。 |
| `private TerminalAction tryTerminateCancellationLocked()` [L1046](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1046) | private，要求持锁 | 无显式输入，读取取消原因、准入/抢占/派发状态；能本地结束则返回已认领的 TerminalAction，否则 null；动作尚未执行。 |
| `private boolean hasCancellationFirstCauseLocked()` [L1056](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1056) | private，要求持锁 | 无输入；返回全局 cancellationReason 是否存在，不包含尚未提升的准入取消原因。 |
| `private CancelReason requireCancellationFirstCauseLocked()` [L1060](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1060) | private，要求持锁 | 无输入；返回全局第一取消原因，不存在则抛异常。 |
| `private StrategyErrorType cancellationErrorTypeLocked(CancelReason reason)` [L1068](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1068) | private，要求持锁 | 输入取消原因；deadline 超时映射为配置错误，其余为 REQUEST_CANCELLED。返回错误码。 |
| `void recordSchedulingFailure(StrategyErrorType error, String detail)` [L1074](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1074) | 自锁→锁外 | 输入调度错误码/说明；生成普通失败事件，经过准入和抢占等待规则，再执行选出的动作。void；不像 selectDeliveryFailureLocked 那样立即选定派发失败。 |
| `TerminalAction claimShutdownAction()` [L1082](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1082) | 自锁 | 无显式输入；仅对可本地撤回的请求生成关闭失败。返回 TerminalAction 或 null；名字是 prepare，但已认领 Slot 收尾，调用方必须执行动作。 |
| `private boolean canClaimLocalTerminalLocked()` [L1091](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1091) | private，要求持锁 | 无输入；检查未派发/未 ACK、Future 未完成、无准入/抢占、无 Decode 所有权等条件。返回能否本地结束。 |

### 调度期限：安装与到期

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized boolean installRequestDeadline(RequestDeadline exact)` [L1104](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1104) | 自行持锁 | 输入请求定时器句柄；活跃且仍开放时安装并返回 true，不适用为 false，重复安装抛异常。 |
| `void onSchedulingDeadline(RequestDeadline exact)` [L1117](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1117) | 自锁→锁外 | 输入精确 RequestDeadline；丢弃过期句柄。准入中记录超时取消，未派发时尝试本地结束。void；可改变状态并执行清理，已派发请求不按这条期限撤销。 |

### 沉默期限：计划、安装、消费与显式检查

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized void configureInactivityTimeout(long timeoutMs)` [L1142](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1142) | 自行持锁 | 输入正数毫秒；设置沉默回收阈值。void；非正数非法。 |
| `synchronized OptionalLong inactivityDeadlineAtMs()` [L1149](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1149) | 自行持锁 | 无输入；计算应挂载的沉默期限。返回 OptionalLong；已挂定时器、已决定过期或不再跟踪时为空，不是简单 getter。 |
| `synchronized boolean installInactivityDeadline(InactivityDeadline exact)` [L1157](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1157) | 自行持锁 | 输入新定时器句柄；仍需要期限时安装并返回 true，否则 false。 |
| `void onInactivityDeadline(InactivityDeadline exact, long nowMs)` [L1165](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1165) | 自锁→锁外 | 输入精确 InactivityDeadline 和当前时间；消费句柄、判断沉默期限，继续失败清理或执行正常过期收尾。void；旧定时器直接忽略。 |
| `private boolean consumeInactivityDeadlineLocked(InactivityDeadline exact)` [L1180](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1180) | private，要求持锁 | 输入触发的定时器句柄；验证同一身份后摘除，返回是否接受该触发；不等于已经过期。 |
| `void expireInactiveRequest(long nowMs)` [L1189](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1189) | 自锁→锁外 | 输入当前时间；执行同一沉默检查，但不校验/消费某个定时器句柄。void；供显式过期检查调用。 |
| `private TerminalAction decideInactivityLocked(long nowMs)` [L1203](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1203) | private，要求持锁 | 输入当前时间；未到期返回 null；已有失败则锁存 expired，返回 null 等待锁外清理；正常请求记录超时取消，准入中暂存，否则返回 TerminalAction。 |
| `private boolean requestInactiveLocked(long nowMs)` [L1221](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1221) | private，要求持锁 | 输入当前时间；与最后匹配 Worker 状态时间比较，返回是否达到沉默阈值。不包含是否仍归本 Slot 跟踪的判断。 |

### 可见性期限：预测、Engine 证据、疑似丢失诊断

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `private DecisionDeadline updateDeliveryPredictionLocked(WorkSnapshot precedingWork, long unstartedWorkMs, long nowMs)` [L1228](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1228) | private，要求持锁 | 输入前序工作快照、尚未开始的耗时、当前时间；一次性计算 Engine/Decode 观察期限，并融合已有 Decode 接受事实。返回已失效 DecisionDeadline 或 null；未实际取消它。 |
| `private void advanceDecisionLocked(DecisionStage next, OptionalLong nextDeadline)` [L1268](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1268) | private，要求持锁 | 输入决策观察阶段、可选截止时间；一并更新阶段、期限并清除 decisionExpired。void；不是请求结果转换。 |
| `synchronized OptionalLong decisionDeadlineAtMs()` [L1274](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1274) | 自行持锁 | 无输入；返回需要挂载的观察截止时间，可为空；已安装时为空。 |
| `synchronized boolean installDecisionDeadline(DecisionDeadline exact)` [L1279](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1279) | 自行持锁 | 输入决策定时器；需仍活跃、未安装且截止值匹配才安装成功，返回 boolean。 |
| `synchronized DecisionExpiry onDecisionVisibilityDeadline(DecisionDeadline exact)` [L1288](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1288) | 自行持锁 | 输入精确 DecisionDeadline；过滤旧句柄，记录观察期到期与疑似丢失。返回 DecisionExpiry 或 null；没有直接结束请求。 |
| `private boolean needsDecisionConfirmationLocked()` [L1309](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1309) | private，要求持锁 | 无输入；活跃请求的 Engine/Decode 观察期已到、又无取消时返回 true。不代表已证明请求丢失。 |
| `private void markAwaitingConfirmationLocked(String message)` [L1317](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1317) | private，要求持锁 | 输入说明；符合未确认条件时更新 SUSPECTED_LOST 诊断及更新时间。void；不立即失败或释放资源。 |
| `private void reconcileDecisionEvidenceLocked()` [L1331](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1331) | private，要求持锁 | 无输入；出现有效 Engine 证据后清除 SUSPECTED_LOST 说明。void；不完成 Future。 |
| `private DecisionDeadline detachObsoleteDecisionDeadlineLocked()` [L1344](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1344) | private，要求持锁 | 无输入；若现有定时器已不符合当前决策期限，摘下并返回，否则 null。未调用定时器取消。 |
| `private DecisionDeadline detachDecisionDeadlineLocked()` [L1353](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1353) | private，要求持锁 | 无输入；取走并清空 DecisionDeadline，可能 null；不校验、不取消。 |
| `private static long addWork(long precedingMs, long unstartedMs)` [L1359](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1359) | 纯计算 | 输入两个耗时；返回饱和相加结果，溢出按 Long.MAX_VALUE 处理。 |
| `private static long deadlineAfter(long startedAtMs, long durationMs)` [L1363](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1363) | 纯计算 | 输入开始时间、正数时长；返回饱和计算的截止时间。输入非法抛异常。 |

### Timer 关闭：摘除本请求的全部定时器

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized ExpirationTimer.DetachedDeadlines detachDeadlinesForTimerClose()` [L1373](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1373) | 自行持锁 | 无输入；同时摘下三类定时器。返回 DetachedDeadlines，由外部取消；本函数不执行取消。 |

### 抢占协议：注册、进展、释放与完成

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `synchronized PreemptionRegistration tryInstallPreemption( long reservationToken, long attemptToken, String detail)` [L1386](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1386) | 自行持锁 | 输入 reservationToken、attemptToken、说明；校验准入/取消/派发和精确 reservation，安装抢占登记。返回登记或 null；不发 Cancel RPC。 |
| `private PreemptionRegistration exactPreemptionLocked( PreemptionRegistration claim)` [L1410](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1410) | private，要求持锁 | 输入登记；核验 requestId 和同一实例。返回精确登记或 null；参数类型已经是 PreemptionRegistration，内部 instanceof 不增加类型信息。 |
| `boolean updatePreemption(PreemptionRegistration claim, PreemptionCancelPhase next)` [L1419](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1419) | 自锁→锁外 | 输入抢占登记和新协议阶段；调用锁内阶段处理，再执行动作。boolean 表示非 STALE，不表示请求或清理已经完成。 |
| `private RequestEffect applyPreemptionPhaseLocked( PreemptionRegistration claim, PreemptionCancelPhase next)` [L1430](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1430) | private，要求持锁 | 输入精确登记、协议阶段；推进协议，必要时写取消状态、处理暂存结果。返回 RequestEffect；动作尚未执行。 |
| `boolean releasePreemption(PreemptionRegistration claim)` [L1458](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1458) | 自锁→锁外 | 输入精确抢占登记；合法释放后继续清理或执行被阻塞的动作。boolean 表示释放被接受；过期登记返回 false。 |
| `private RequestEffect applyPreemptionReleaseLocked( PreemptionRegistration claim)` [L1476](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1476) | private，要求持锁 | 输入登记；合法时摘除抢占 owner，恢复被阻塞的决定。返回 DEFERRED/STALE/APPLIED；没有直接执行清理。 |
| `boolean completePreemption(PreemptionRegistration claim, String detail)` [L1491](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1491) | 自锁→锁外 | 输入抢占登记、说明；接收协调器已完成 fencing 的通知，更新状态并执行动作。boolean 表示事件被接受。 |
| `private RequestEffect applyPreemptionCompletedLocked( PreemptionRegistration claim, String detail)` [L1500](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1500) | private，要求持锁 | 输入登记、说明；在协调器已完成 fencing 后标记协议结束，推进失败清理或选择被抢占结果。返回 RequestEffect。 |
| `private RequestEffect applyPrefillActivityLocked(PrefillEndpoint source, ScheduledRequest expected)` [L1530](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1530) | private，要求持锁 | 输入 Prefill Endpoint 和精确请求；不匹配返回 STALE；正常活动无抢占待处理返回 APPLIED；若待解抢占则调用精确 Decode 协调事务，未能解开返回 DEFERRED，解开后 APPLIED。 |
| `private RequestEffect applyPriorityCancellationLocked(PrefillEndpoint source, ScheduledRequest expected)` [L1543](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1543) | private，要求持锁 | 输入 Prefill Endpoint、请求；验证并结算精确抢占，标记资源进度或选被抢占结果。返回 RequestEffect；Endpoint 结算失败不冒充成功。 |
| `private void retainPreemptionTerminalLocked(PreemptionRegistration exact, DeferredTerminal candidate)` [L1569](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1569) | private，要求持锁 | 输入抢占登记、候选结束事件；保留第一个事件，允许权威 Worker 事实替换非权威事件。void。 |
| `private RequestEffect processPendingEventsUnderPreemptionLocked( PreemptionRegistration exact, boolean transportUnknown, PreemptionRegistration signal)` [L1577](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1577) | private，要求持锁 | 输入登记、是否传输不确定、可空通知对象；先协调结束事件，否则协调暂存 ACK。返回 RequestEffect，协调失败为 DEFERRED；决定协议何时脱离 Slot。 |
| `private boolean detachPreemptionOwnerLocked(PreemptionRegistration exact)` [L1614](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1614) | private，要求持锁 | 输入精确登记；匹配才清空 Slot.preemption 并返回 true。既不释放 Endpoint，也不发送结束通知。 |

### 共同收尾：消费结束事实 → 认领清理责任 → 提交最终记录

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `private RequestEffect processRequestEndLocked(ScheduledRequest expected, DeferredTerminal event)` [L1627](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1627) | private，要求持锁 | 输入请求、结束事件；已有失败时只合并清理事实，否则在准入/抢占期间暂存或选择结束动作。返回 RequestEffect；非自带锁的外部入口。 |
| `private TerminalAction decideRequestEndLocked(DeferredTerminal event)` [L1662](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1662) | private，要求持锁 | 输入可空 DeferredTerminal；把事件映射为结果/Response，并调用共同关闭判断。已有失败时使用 Slot.state/detail。返回 TerminalAction 或 null；null 事件暗含“推进已有失败清理”。 |
| `private TerminalAction claimTerminalActionLocked(DeferredTerminal event, TerminalOutcome transition, Response response, boolean requestPublication)` [L1720](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1720) | private，要求持锁 | 输入事件、结束意图、可空响应、是否申请发布；判断能否关闭，置 TERMINALIZING，把抢占/定时器移到动作中。返回 TerminalAction 或 null；还没释放资源、提交终态记录或完成 Future。 |
| `synchronized void requireCleanupOwner(TerminalAction action)` [L1771](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1771) | 自行持锁 | 输入已认领 TerminalAction；自行持锁核验当前收尾阶段和请求身份，不匹配抛异常，匹配返回 void。执行器据此验证动作，不读取 Slot 私有字段。 |
| `synchronized TerminationResult commitTerminalRecord(TerminalAction action)` [L1777](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1777) | 自行持锁 | 输入执行过收尾的精确动作；提交最终状态、清空引用，置 TERMINAL_RECORD。返回 TerminationResult，包含快照/转换异常/发布许可；自身不删除 Registry 条目。 |
| `private void execute(RequestEffect effect)` [L1827](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1827) | 锁外 | 输入非空 RequestEffect；没有动作直接返回；有动作则锁外执行清账或 ACK 发布，并独立尝试协议信号通知。void；汇总异常，不用 finally 串接业务。 |

### 响应：本地结束、结果仲裁与发布交接

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `boolean terminateLocallyAndPublishResponse(Response response)` [L1843](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1843) | 锁外，内部短暂持锁 | 输入 Response；通过本地结束判断和清理取得发布许可，再异步提交响应。true 表示已提交，false 表示未能取得许可；不保证 Future 已完成。 |
| `boolean completeExternal(ResponseCompletion completion, Response response, Throwable error, boolean interrupt)` [L1850](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1850) | 入口自行管理锁 | 输入 Future 完成种类及对应响应/异常/中断标记；统一映射本地结束结果，清理后同步完成 Future。boolean 为 Future 完成结果；不能本地结束时 false。 |
| `private static TerminalOutcome responseOutcome(Response response)` [L1866](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1866) | 入口自行管理锁 | 输入可空 Response；生成成功或失败 TerminalOutcome，使用响应错误说明或默认说明。纯映射，多入口复用。 |
| `private PublicationPermit terminateLocallyAndAcquirePublication(TerminalOutcome transition)` [L1872](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1872) | 自锁→锁外 | 输入结束意图；仅在可本地撤回时认领动作并同步执行收尾。返回 PublicationPermit 或 null；不替调用方完成 Future。 |
| `private PublicationPermit requirePublicationPermitLocked( PublicationKind kind)` [L1881](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1881) | private，要求持锁 | 输入发布种类；向发布器保留一次发布许可并验证归属。返回 permit，关闭/身份错误时抛异常；不是选择最终响应。 |
| `SelectedPublication selectPublication(PublicationPermit permit, ResponseCompletion completion, Response response, Throwable failure, boolean mayInterruptIfRunning)` [L1895](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1895) | 锁外，内部短暂持锁 | 输入 permit、完成形式及各自负载；核验许可、一次性 claim，在锁内仲裁响应。返回可在锁外执行的 SelectedPublication；异常会归还许可。 |
| `private boolean claimPublicationResultLocked(PublicationKind kind)` [L1914](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1914) | private，要求持锁 | 输入发布种类；判断候选响应能否获胜，DELIVERY 成功时记录 winner。返回 boolean；TERMINAL 验证此前已选赢家，仍未完成 Future。 |
| `void submitTerminalResponse(PublicationPermit permit, Response response)` [L1931](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1931) | 锁外 | 输入发布许可、Response；选定一次响应并交发布器。void；不是清理函数。 |

### 锁边界断言

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `private void requireSlotLock(String operation)` [L1937](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1937) | 线程检查 | 输入操作名；当前线程未持有 Slot 监视器则抛异常，否则无输出。 |
| `private void requireOutsideSlotLock(String operation)` [L1944](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1944) | 线程检查 | 输入操作名；若当前线程仍持有 Slot 锁则抛异常。 |

## RequestEffect

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `RequestEffect { … }（紧凑构造器）` [L1995](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L1995) | 构造校验 | 输入事件处理状态和可选动作/信号；只有 APPLIED 可携带动作，收尾和派发动作不能同时存在；非法组合抛异常。APPLIED 允许没有动作。 |
| `static RequestEffect terminal(TerminalAction action, PreemptionRegistration signal)` [L2003](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2003) | 纯工厂 | 输入可空收尾动作和信号；动作为空返回 DEFERRED，否则返回带动作的 APPLIED。 |
| `static RequestEffect delivery(DeliveryPublication delivery, PreemptionRegistration signal)` [L2007](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2007) | 纯工厂 | 输入非空派发发布动作及信号；返回带动作的 APPLIED。 |

## WorkerTerminalSource

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `WorkerTerminalSource(boolean decodeTerminalAlreadyApplied)` [L2029](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2029) | 枚举构造 | 输入是否已更新 Decode 账单；保存常量，Prefill 为 false，Decode 为 true。 |
| `boolean decodeTerminalAlreadyApplied()` [L2033](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2033) | 只读 | 无输入；返回该来源对应的 Decode 账单是否已更新。 |

## DeferredTerminal

| 签名：参数与返回类型 | 锁边界 | 职责、输出与副作用 |
| --- | --- | --- |
| `DeferredTerminal { … }（紧凑构造器）` [L2063](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2063) | 构造校验 | 输入事件种类和错误/Worker 等负载；校验种类及 errorType、workerSource 的组合，非法则抛异常。不是资源已释放的证明。 |
| `static DeferredTerminal failure( StrategyErrorType errorType, String detail)` [L2079](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2079) | 纯工厂 | 输入错误类型、说明；返回 FAILURE 事件。 |
| `static DeferredTerminal inactivityExpired(String detail)` [L2085](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2085) | 纯工厂 | 输入说明；返回 INACTIVITY_EXPIRED 事件。 |
| `static DeferredTerminal timeout(String detail)` [L2089](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2089) | 纯工厂 | 输入说明；返回 TIMEOUT 事件。 |
| `static DeferredTerminal worker( WorkerTerminalSource source, boolean successful, long errorCode)` [L2094](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2094) | 纯工厂 | 输入非空 Worker 来源、成功标记、错误码；返回 WORKER 事件。 |
| `static DeferredTerminal priority(String detail)` [L2103](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2103) | 纯工厂 | 输入说明；返回 PRIORITY 事件。 |
| `static DeferredTerminal decodeGenerationRetired(String detail)` [L2108](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2108) | 纯工厂 | 输入说明；返回 DECODE_GENERATION_RETIRED 事件。 |
| `boolean authoritativeWorker()` [L2114](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2114) | 只读 | 无输入；WORKER 或 DECODE_GENERATION_RETIRED 返回 true。名字不完整：Endpoint 退出也算 true。 |
| `boolean endpointAlreadyRetired()` [L2119](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2119) | 只读 | 无输入；仅 DECODE_GENERATION_RETIRED 返回 true。 |
| `boolean decodeTerminalAlreadyApplied()` [L2123](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestSlot.java#L2123) | 只读 | 无输入；仅 WORKER 且来自 Decode Endpoint 返回 true，单纯退出事件不返回 true。 |

## 验证与剩余边界

- `./mvnw -q -pl flexlb-sync -am test`：1160 项，0 失败，0 错误，0 跳过。
- 原有 4 项子 JVM 死锁复现保持原断言，新增 Worker 普通收尾与 Endpoint 扫描、Timer 关闭与 Slot 竞争两个场景。
- 新增的自行加锁查询，其生产调用原先已经位于同一个 Slot 临界区；它们不会在查询时再取得 Endpoint/Timer/Publisher 锁。
- `retainForSchedulerCleanup` 仍只查并发目录，不获取 Slot 锁。这是保留 Slot → Endpoint 顺序时防止反向依赖的关键。
- `fetchAttachTimeoutMs` 未在本轮改动。

方法总量仍包含准入、观察期限和抢占的内部规则。本轮没有为了压低数字把这些规则改为一个通用事件 switch，也没有把必须原子完成的 Endpoint 事务移到锁外。
