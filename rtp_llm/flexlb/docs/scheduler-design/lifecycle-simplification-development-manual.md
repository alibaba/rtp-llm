# FlexLB 请求、资源与异步流程简化：开发交接手册

## 新交接：QueuedRequestScheduler 复杂度审核（2026-11-09）

此节保留为历史审核记录。2026-11-10 已与另一份流程审查合并，当前实施依据为
[QueuedRequestScheduler 最终交接](queued-request-scheduler-handoff.md)。其中修正了下文
Q1 的二选一等待建议，并明确 Q4 的预算调整不是已确定的实施要求。

本节以本次读取的工作区实现为准：QueuedRequestScheduler 共 1884 行。
这是源码审核及待实施方案，未修改生产代码、未新增或运行本节 UT。下文历史验收数字不能
作为本节发现或方案的验证结果。旧章节中的公共 resumeRoute 等归属描述也不能代替当前代码；
当前 executeQueuedRouting、resumeRoute 和撤路 completeRouting 都在 QueuedRequestScheduler 内。

必须保留：自定义 scheduler、选址策略、SchedulingQueue、PrefillRequestQueue 和 GroupPlanner；
策略内部候选缓存、局部更新和准确预留；原始优先级、FIFO、绝对超时、有限旁路及精确资源身份。
目标是减少独立协议和重复恢复，不是把 1884 行搬进几个文件。

### 实际流程与责任交接

| 步骤            | 当前入口与执行位置                                                      | 产生的事实、失败处理                                                                                                                     |
| --------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 接收            | scheduleRequest → enqueue；调用线程、scheduler lock                     | 创建 GlobalQueueEntry，登记 queuedEntries 和可替换 orderedQueue；重入队恢复原 sequence。Future 完成、关闭和重复身份分别检查。            |
| 选择待处理请求  | run → schedule → scheduleAt → planningCandidates                        | 容量域唤醒请求进入重试索引；受 scanBudget 限制扫描；选中的 entry 加入 inFlight。此时线程池尚未接纳。                                     |
| 执行选址        | submitPlan → place → planRoute → executeQueuedRouting；planner 线程     | 准备输入，校验期限，开始同一个 RoutingContext；策略选择并登记资源。线程池拒绝仍需归还已占用的名额。                                      |
| 普通结果        | enqueueRoute 或 commitEncoderRoute → finishUnpublishedRoute → 通知/交付 | 无 Encoder 时先物理入 Prefill 队列，再发布 Context 路由；资源交接后通知失败不能回滚。Encoder 路径返回 RouteDelivered 后交给 Dispatcher。 |
| 容量不足        | finishDecision → park → finishPlanning                                  | 根据选择前的 capacity sequence 登记容量等待；若期间发生容量变化，则重试。完成本轮处理后才释放 inFlight。                                 |
| 抢占            | tryPriorityRescue → RoutingPlan.continuation → resumeAfterPreemption    | pendingPreemptions 保留独立异步工作的完成责任；先释放 planner 名额，再安装续接。恢复仍使用原 RoutingContext；晚到预留必须精确释放。      |
| Worker 队列调度 | onEndpointChanged → ready/timed 队列 → runQueueScheduling               | 每个 PrefillQueueScheduler 单任务执行，捕获有限队列前缀、预测分组并调用 Dispatcher.submit。                                              |
| 批量等待/旁路   | decideQueueScheduling → decideGroup → observeWait → waitCurrent         | 区分收集窗口、容量不足、输入版本变化；订阅后复查以防漏唤醒。Decode 队头阻塞时允许受限制的同级旁路，提交前再查队头容量。                  |
| 撤路/重排       | onRouteFailure → withdrawFailedRoute → completeRouting                  | 仅未发送等允许情形撤回；移除队列、关闭旧 delivery、释放资源、完成路由操作，再按原期限入队。                                              |
| 关闭            | stop / close / awaitStopped / drainOnClose                              | 停止接纳、等待调度调用及 planner、等待抢占续接，解除容量订阅；不能用队列空代替全部工作结束。                                             |

### Q1：一次等待被拆成两个登记，已有重复计数，并存在漏唤醒交错

证据位置：blockedRequestCount（308）、tryPriorityRescue（1333）、park（1248）、
finishPreemption（1447）、planningCandidates（1055），以及 CapacityWaitQueue.park/isWaiting。
行号仅用于本轮定位，后续按方法名查找。

当前名额已满时，tryPriorityRescue 把 entry 加入 preemptionQuotaWaiters，返回 null。
executeQueuedRouting 因没有 continuation 正常结束路由操作，随后 finishDecision 仍将 BLOCKED
结果交给 park，加入 waitingRequests。因此“等待本地抢占名额”的请求同时成为“等待 Worker
容量”的请求。blockedRequestCount 直接相加，两份集合交集中的请求被计算两次。

更严重的源码推演交错（尚未由新 UT 复现）：

1. 请求 B 看见抢占名额已满，登记 quota waiter，退出锁；尚未调用 park。
2. 请求 A 的抢占结束，finishPreemption 移除 B 的容量等待（此时尚不存在），标记 B 重试，清空 quota waiters。
3. B 的原 planner 接着执行 park。名额释放没有改变 Worker capacity sequence，因此可以成功挂起。
4. B 结束 planner；下一轮即使扫描到重试标记，也因 waitingRequests.isWaiting(B) 为真而跳过。
   如果扫描更早发生，还可能先因 B 在 inFlight 中消耗该标记。
5. 没有后续相关容量事件时，这次名额释放没有产生有效重试，B 可能等到其他事件或 deadline。

根因是内部恢复结果用 null 同时表达“不使用抢占”和“需要等名额”，等待原因被丢失，外层
只能按原始 Worker blocker 再挂起。finishPreemption 的跨集合 remove 就是补偿这一设计的代码。

具体修改方向：让内部 RoutingPlan 区分完成选址、异步续接、等待抢占名额，不修改策略公共
RoutingResult 的容量语义。tryPriorityRescue 名额不足时只返回明确的等待决定；不要提前登记
另一个集合。处理该决定时，在同一 scheduler lock 内复查 pendingPreemptions.size：有空位就
在释放本轮 inFlight 时标记重试，否则登记 quota waiter。Worker BLOCKED 才调用容量 park。
名额等待者只占一个等待分支。若要求容量事件也能提前唤醒 quota waiter，可沿 CapacityKey
建立定向索引；它只负责触发重试，不把同一请求再登记成一个独立容量等待任务。
唤醒先保留资格，直到对应 planner 释放后再发布可扫描标记，不能在仍 inFlight 时提前消费。

删除项：quota waiter 的容量 park、finishPreemption 对其逐个 waitingRequests.remove 的补偿、
重复计数。保留 pendingPreemptions：它表示还会返回结果的异步任务，不等于队列成员。
不要为了统一状态把取消请求的未完成抢占直接删掉，否则晚到资源无人释放。

先补 UT：用屏障固定“名额判断后、容量 park 前”的交错；无 Worker 容量通知仍应重试。
另测一个 waiter 的 blocked count 为 1、通知发生于 planner 释放前不丢资格、取消与旧续接不影响
替代 entry。保留 GlobalQueueProgressTest.returningPreemptionSlotRetriesWaiterWithoutWorkerCapacityEdge；
现有测试等待请求走到一般等待状态，未明确固定上述中间交错。

### Q2：路由完成的公共规则被子类复制，扩展边界仍然过大

证据：QueuedRequestScheduler.completeRouting（654）与 AbstractRequestScheduler.completeRouting（797）。
两者都负责 beginRoutingCompletion、Context 完成、pendingEnd 与发布失败优先级、closeRequest、
processPendingRequestActions 及异常记录。Queue 版本额外负责旧队列移除、旧 delivery、资源释放和重新入队。
这意味着修改一次公共“取消与路由失败谁优先”的规则，需要同时修改两个流程。

不要直接把 86 行撤路方法搬到 helper。目标是公共完成过程只保留一份，Queue 只提供旧路由
移除和重新排队行为。建议在现有基类完成流程中保留统一的：

```java
// 流程示意；尚非已实现 API。
if (!ctx.beginRoutingCompletion(routing)) return;
// 必须在 completeRouting 清掉旧 route 之前完成移除/释放；失败转为 FAILED。
var completion = detachOldRouteIfRequired(routing, result, failure);
var after = ctx.completeRouting(routing, completion.result(), nowMs);
// 这里只保留一份 pendingEnd / failure 优先级。
finishRoutingRequest(ctx, after, completion);
```

实现时优先复用已有结果类型，不能为了这个示意再建立长期状态对象。Queue 的差异只在
detach 和 WITHDRAWN 后按原序号/原期限重新入队。Dispatcher 的旧 delivery 关闭和完成时点、
撤路资源释放异常均纳入公共协调；不把这些责任转嫁给自定义 queue 或 selector。
也不要求所有自定义 scheduler 支持撤路；默认能力拒绝，支持者提供准确行为。

先补 UT：取消/SHUTDOWN/发布失败同时存在时两种完成路径具有相同优先级；撤路释放失败不能
重新入队；迟到旧 delivery 不能完成新路由。继续使用 QueuedPublicationNotificationFailureTest、
QueuedBatchDeliveryTest 中物理发布后通知失败、发送开始后不得撤路的既有断言。

### Q3：绕过队头的一次调用参数被保存成调度器成员

证据：decideQueueScheduling（约 520）设置 bypass 和 blockedHead，finally 清空；
PrefillQueueScheduler implements BooleanSupplier，其 getAsBoolean（1721）隐式读取这两个字段。
decideGroup 将整个 queueScheduler 当回调传给 processSelection。
DeliveryDispatcher.submit/commit 在提交过程内检查这个回调，当前用途只属于本轮旁路判断。

根因是把“一次选择仍允许提交”的条件放进长期对象，调用者必须知道先写字段再调用、最后清空
的隐含协议；正常选择与旁路选择还通过字段是否为空区分。不是必要的业务状态。

具体修改：正常调用传固定的 true 条件；旁路调用把本轮 head 和 boundary 捕获为局部值，传入
`() -> !queueScheduler.capacityAvailable(head, boundary)`。停止检查继续保留在 processSelection；
提交前复查继续保留在 Dispatcher。删 bypass、blockedHead、implements BooleanSupplier、getAsBoolean
和设置/清空字段的 try/finally。只在旁路路径创建闭包，不能对每个 endpoint 重算预测。
保留旁路次数/优先级的队列规则及 PrefixPolicy 扩展，不把这些策略硬编码成一个标准 FIFO。

先补/复用 UT：recoveringHeadAbortsBypassBeforeCommit、failedBypassCommitReleasesTemporaryResourcesAndRetainsAllowance、
onlySamePriorityRequestsOnAnotherDecodeMayBypass，以及自定义 ALL_SELECTED 行为。

### Q4：一个执行器对应三条接纳路径，并发配置含义不清

证据：scheduleAt 使用 plannerCount - inFlight.size；schedulePrefillQueues 独立使用
queueSchedulingInFlight < plannerCount；resumeAfterPreemption 直接 planners.execute。
三者共用 fixedThreadPool(plannerCount)。因此 plannerCount 不是所有已提交调度工作的统一上限；
它同时影响线程数、全局任务额度、Worker 队列任务额度和抢占数量。

这不证明线程会超额，也不能仅凭代码断言吞吐退化：实际线程数仍受线程池限制。
确切问题是多套准入与公平性规则没有共同说明。慢选址占满线程时，已经选好 Worker 的队列任务
仍排在同一执行器后面；调度周期内先处理 readyPrefillQueues 不等于线程池内优先执行它们。

实施前先固定契约：CPU 执行额度统一管理，保留每个 endpoint 至多一个调度任务、每个 entry 至多
一个选址任务；异步抢占等待不是 CPU 在运行，单独限额有实际意义。可在同一个 scheduleAt 中给
已就绪队列任务与请求选址分配总额度，续接也进入这套接纳；不要新建另一份全局任务集合来镜像
Executor。必须规定两类工作的进展保证，不能用固定“永远先批量”导致全局请求饥饿。
若业务明确要求慢选址与交付准备隔离，应采用独立执行预算并给出测量依据，而不是冒称可以
无损合并。这一项属于设计待决，当前不承诺具体删行数。

先补 UT：阻塞全部选址线程后观察已就绪队列的执行顺序；续接、取消和关闭同时发生时准确
归还额度；ready 队列持续非空时新请求仍有进展。性能验证区分排队时延、执行时间、任务分配，
不把线程数未变当作性能不变。

### Q5：一次 endpoint 变化退回全局请求扫描

证据：onEndpointChanged 把不可用 endpoint 放入集合，withdrawUnavailableRoutes（903）随后
遍历 repository.snapshotActive，对每个 route 遍历本轮失效 endpoints；queuedRequestCount（193）
也遍历 repository。前者最坏为 O(活动请求数 × 本轮失效 endpoint 数)，且在调度驱动线程执行。

这是明确的算法成本，不等于已测出性能瓶颈。不能简单添加 endpoint→request 副本索引，因为
发布、撤回、Encoder/Decode/PREFILL 角色、退休代际都要维护它，可能增加更多一致性代码。
先核对 Endpoint 已有准确资源登记和 Prefill 队列索引能否枚举受影响 route；可用则复用快照并按
route 身份去重，不可用时再比较新增反向索引的成本。queuedRequestCount 的口径包含已选路但
未交付的请求，不能直接替换成 orderedQueue.size。

先补 UT：一个 Worker 失效只处理引用它的准确 route，多角色共用同一路由只撤回一次，旧
endpoint generation 不影响替代资源；再用请求量、失效节点数的增长曲线验证成本。

### 接手实施顺序与验收

1. 优先为 Q1 写确定性交错 UT，再统一等待登记；同时修复唯一请求计数。不得用 sleep 代替屏障。
2. Q3 是边界明确的小型删除，可独立做，旁路提交前校验不能删。
3. Q2 按一条完整撤路路径合并公共规则，测试之后同步清除旧完成流程；仅抽方法不算完成。
4. Q4、Q5 先确定调度公平性和既有索引能力，再据测量实施，不将审核假设直接当已证实缺陷。

每轮保存工作区基线，分别统计所有受影响生产代码、测试和文档；净减量必须计入新增辅助类。
列明真正删掉的状态、配对操作和失败分支。测试沿上述真实入口验证，不只反射私有字段。
实现完成后执行相关回归、成员顺序与全 reactor Spotless；本节当前只是交接，不宣称验证通过。

---

日期：2026-11-09。状态：同步与抢占续接已共用选路操作，资源交接改用操作内的明确结果；整体复杂度治理目标仍未全部达到。

### 当前实现：同一选路操作跨过抢占等待和发布

上一轮将 BLOCKED 的选路先结束、抢占时再建操作，增加了操作创建、异常收尾和诊断回填；该实现已撤回。
当前 `planRoute` 只开始一次 RoutingContext，抢占和后续角色选择沿用它。普通调用仍使用
`selectAndPublish(request, selection, queue)`，自定义选择函数和队列机制保持可替换。

本轮删除的机制：

- 抢占时第二次 tryBeginRouteSelection、诊断复制与恢复，以及未启动第二次操作时的专用结束分支。
- Queue 内对预留结果的检查、取消后精确预留的释放、finishRouting/failRouting 配对和 PendingPreemption 记录。
  Queue 只管是否抢占、并发名额、续接线程与最终等待/重试；公共 resumeRoute 负责资源与操作收尾。
- Direct/Encoder 在投递前的 Context 路由发布。路由和投递在同一个请求锁内形成同一个快照；
  提交前的工作量计算或代际校验失败时，路由仍属本次操作，直接释放，不先交给请求再撤回。
- 配套的 RequestContext.discardRoute 以及路由重试时释放资源、清空 Context 再结束操作的恢复路径。
- 清理时通过 Context.snapshot().route 反查资源是否已经交接的分支。Direct 在提交前完成工作量和可见期限计算，
  接收成功只返回已有 RouteDelivered；资源交接结果不需要独立记录，交接后也无需补偿可见期限计算失败。
  Queue 在物理入队、请求接收成功后返回，通知放在操作完成之后；通知期间取消并归档也不会重新触发临时资源回滚。

必须保留的调度顺序：先释放本轮 planner 名额，再启动抢占续接；重试标记与对应名额释放仍在同一队列锁内。
不能为减少代码先标记重试，队列扫描可能在请求仍不可调度时消耗该标记并丢失唤醒。
不可变 RoutingPlan 只承载已完成决策或待启动的续接工作，替换原有 PendingPreemption；不承载新的资源账本或状态枚举。
普通返回的对象替换了原 finally 捕获闭包，抢占政策函数在 Scheduler 实例上复用；没有新增普通请求 Future、执行器跳转或全量候选重算。

新增回归验证 Direct 路由与投递在同一 Context 快照中可见；退役重试仍验证两个 Endpoint 的资源余额和有限重试。
抢占测试恢复为同一个操作身份，维持等待期间不得结束、完成后只结束一次和晚到预留准确释放的断言。
架构测试禁止 Queue 自行释放 Decode 预留、结束选路操作或读取抢占结果中的 reservation。

最终版本全 reactor 测试：4,267 用例、零失败、零错误、3 跳过，日志为
`/tmp/flexlb-routing-improvement-final-acceptance.log`；可见期限计算前移后的 109 个核心回归另见
`/tmp/flexlb-routing-improvement-precommit-tests.log`。
全 reactor 格式检查见 `/tmp/flexlb-routing-improvement-acceptance-spotless.log`。发布阶段微基准在 `/tmp/flexlb-context-publication-bench`。
微基准使用真实 Context、RouteResult 和 Delivery.Member，不调用 Mockito，只测 Context 发布；不能代表端到端吞吐或 P99。
三个交替 JVM fork、每个版本十五轮有效样本的中位数：发布分配从 288 降为 176 B/次，CPU 从 128 降为 105 ns/次。
对应结果为 `/tmp/flexlb-context-publication-bench/summary-direct.json`，输入构造在计时之外；这只验证两次快照合为一次的成本。
前期隔离验证目录为 `/tmp/flexlb-routing-improvement-check`，用于避开并行执行设施迁移造成的短暂编译不一致；
交付以当前工作区的验证结果为准，不混用隔离版本和当前版本的测试结论。

仍需单独治理的范围：完整 RouteResult 形成前的回滚异常持续追踪（包括 Selector 部分角色及抢占晚到预留），
以及请求结束、远端取消和执行设施之间的职责。
本轮没有宣称删除所有恢复路径，也不把此前 100–200 行的估计当作验收结果。

## 1. 给接手 agent 的任务

### 当前结论的纠正

前几轮实施主要删除冗余分支、修复局部边界，未改变核心组织方式。后文的逐项验收和 4,277 用例报告只证明被测版本的行为，不证明根本复杂度已收敛；原“手册范围内全部完成”的结论撤回。

当前需要首先解决的根本问题是：调度算法扩展对象同时充当请求服务、资源事务协调者和运行设施的关闭者。RequestScheduler 接口虽然表述调度决策，API、ApplicationLifecycle、RequestData 和 RequestRepository 实际都依赖 AbstractRequestScheduler；外部扩展必须继承携带完整生命周期协议的基类。这条依赖主干尚未改变。

后续架构改造必须以一条纵向路径证明边界变化：自定义 scheduler 发起异步选路，期间请求取消或系统关闭，选路稍后取得资源。请求公共运行部分必须负责准确撤回、释放、响应和归档；扩展实现仅决定下一步调度，不持有 publication Registration、WorkRegistration 或清理步骤。该路径必须实际移除扩展侧 tryBeginRouting/finishRouting/completeRouting 的手动配对责任，以及为该交接存在的拒绝恢复；不能仅把旧基类方法搬到新类后继续转调。

目标分为三个决策边界：scheduler 负责调度与队列政策；资源组件独占容量及准确登记；请求公共运行部分独占请求与一次尝试从接纳到结束的推进。是否采用串行事件处理、操作 Future 或其他机制，要由上述竞争路径的完整实现和成本验证决定，不能未经证明先新增统一事件框架。

### 基于策略内预留重新确定的开发方案

状态：替换上一版 Runtime / SelectionSession 主线；同步入口、异步预留处置及操作内明确交接已实施。协调位置是现有基类的公共请求流程，不让 Selector 反向依赖请求发布。先改变一次资源操作的责任范围，不先移动基类或引入统一运行框架。

#### 1. 已确认事实与新的问题定位

策略内的评分、选点、立即登记及局部冲突刷新必须作为一个闭环保留。CostBasedPrefillStrategy 登记冲突时只刷新对应候选，复用其他评分；廉价重新比较与重新执行全量预测、缓存匹配是不同成本。Endpoint 在同一资源锁内校验版本并登记，其他 planner 在回调成功返回前即可观察预留。

RequestRouteSelector 已经组合角色登记，并用 selections / transferred / finally 回滚部分成功。因此不新增 SelectionSession。开工前它返回带预留的 RouteResult 后，调度实现还要配对 publishRoute、finishSelection、finishRouting。本轮把同步与异步配对收进公共流程，删除 Queue 对 reservation 的访问以及清理时的 Context 路由反查。

#### 2. 目标：一次选路资源操作持有责任直到交接确定

现有 AbstractRequestScheduler 的公共请求流程协调“选点登记 → 发布或回滚 → 返回最终结果”，通过选择函数调用内置 Selector 或自定义选择实现。Selector 只组合角色预留并撤销部分成功，不决定请求响应、重试与关闭。策略继续在原调用栈中立即登记。成功返回给调度实现时，资源已交接给请求/投递；普通失败返回时，临时登记已清理。清理失败有明确未完成责任，不能变成可重试 BLOCKED。

Scheduler 只决定调用时机、候选政策、等待、出入队和是否发起抢占；不再拿到一个尚待处理的预留，再手工配对路由开始与结束。

拟定调用形状只表达完整操作：

```java
// 在 Scheduler 现有 planner 线程同步执行；队列、策略仍可替换。
RoutingResult<RouteResult, CapacityKey> selectAndPublish(
        RequestContext request,
        Supplier<RoutingResult<RouteResult, CapacityKey>> selection,
        QueueExecutionSettings queue);
```

签名是否继续使用 Context 须核对必要信息后决定；本批不为参数整洁新增 RequestTicket、RoutingInput 或 SelectedRoles 全套包装。策略接口继续使用现有评分输入和精确登记回调，不返回“稍后由公共层再预留”的裸候选。

#### 3. 唯一责任交接点

角色登记成功至正式交接前，由同一次 route 调用负责清理；正式交接之后，由请求结束流程负责。不能用“Context 现在是否还保留 route”代替“本次资源是否已经交接过”。

必须使用操作内明确的交接结果，优先复用现有 transferred 局部变量及准确 route 身份；不新增全局资源表或第二份请求状态。交接成功与请求后来完成是不同事实：归档清除 route 后，原选路调用不能重新认领并回滚这批资源。

交接结果不能由 RoutingResult.SUCCESS 推导。发布失败但请求结束流程已经接管未清资源时，调度结果仍是失败，原选路调用也必须放弃重复清理；回滚失败而无人接管时则不能结束该操作。必须同时说明“业务结果”和“剩余资源由谁清理”，不能用一个成功标志混同两者。优先把准确交接事实限制在操作内部，不扩散成新的公共状态机。

发布边界须逐模式证明：

- QUEUE：物理入队与请求接收之间，消费者继续遵守现有未发布 route 不可派发的约束。取消导致请求拒绝接收时，由当前 route 操作移除准确成员并回滚；接收成功后，唤醒失败不撤销交接。
- DIRECT / NON_BATCH：沿现有发送边界交接；发送可能已发生时禁止当作未发布回滚。保持本地清理与响应的原有时序。
- BATCH：选路入队不是 RPC 完成；派发凭据、实际 RPC 与远端证据维持现有独立生命周期。本批不把它们并入选路成功。
- 所有模式：禁止用可能“已经接收但抛异常”的回调作为唯一交接接口。接收前的检查与异常、接收后的通知分开；确有中间阶段时，准确尝试身份必须保留，不能假称能删除全部状态。

只有上述交接契约成立后，才能删除跨层的资源归属反查。若实现仍需相同反查，说明本批尚未完成，不能只换方法名。

#### 4. 异步抢占不是同步路径的特殊裸资源出口

保留策略内局部刷新。异步抢占可能跨越较长时间，不能把 N+k 的局部刷新结论机械推广到抢占前后的全局环境。

Scheduler 可以决定是否抢占，但抢占完成得到的准确预留由选路协调流程接收和处置；取消、关闭或后续角色失败时也由它清理。不能继续要求 scheduler 收到 reservation 后再拼接 select、publish、finish。可复用当前抢占 Future 的续接，不把普通成功路径统一包装成 Future，不增加 Runtime 投递队列。

已实施 `planRoute` / `resumeRoute`：前者返回最终决策或待启动续接，后者接收准确抢占结果并处理晚到资源。Queue 只负责释放 planner 名额后启动续接，预留检查和释放调用已从 Queue 删除。

#### 5. 分批改哪些代码、实际删什么

1. 固定当前完整工作区作行为和性能基线，记录哈希。当前 HEAD 不能代表工作区；并行修改时使用独立快照和构建目录。
2. 在现有基类内合并选择与发布的独立清理作用域，贯通 QUEUE；保留 Selector 的部分角色回滚边界。不能再新建一个只转调旧协议的服务。本轮已实施。
3. 迁移 QueuedRequestScheduler.place：删除其选择/发布之间的资源持有、operation 置空交接、finishSelection finally 和重复回滚。保留队列政策、planner 名额、冲突预算和等待条件。
4. 迁移 DirectRequestScheduler 和外部 scheduler，同步覆盖 DIRECT、Encoder、BATCH / NON_BATCH。扩展实现不继承资源清理协议，但仍可控制自身队列和线程。
5. 迁移抢占续接，删除 scheduler 对 reclaimedReservation 的接管和清理责任。
6. 最后删除失去用途的 publishRoute/finishSelection/finishRouting 外部协议和资源归属反查。准确尝试身份、关闭等待、远端证据按实际用途保留。之后再评估请求服务与算法对象的接入分离，不能把拆基类作为先行成果。

每批必须有删除清单。若完整选路方法继续把步骤交给外层调度实现配对，或者公共选路作用域之外仍有同一临时预留的兜底回滚，则完整资源责任目标尚未达到。不能把同步入口完成计为异步资源责任也已完成。

#### 6. 决定方案成立的测试

- P 成功、D 失败：P 只产生一次实际释放，错误不会被清理异常覆盖。
- 策略正在登记时取消，登记随后成功：不发布，准确资源清理完成，保持原取消响应时序。
- 入队后请求拒绝接收：准确成员被移除，消费者没有派发，其他请求不受影响。
- 已交接请求立即完成并归档，选路调用随后退出：不能因为 Context.route 已清空而再次取得清理责任。
- 唤醒失败：已经完成的资源交接不被撤销。
- 抢占结果晚于取消/关闭：预留由原协调流程清理，不泄漏、不重新入队。
- 两个 planner 使用同一评分版本：Endpoint 原子登记保证容量；失败方局部刷新，策略行为不变。
- BLOCKED、CONFLICT、UNAVAILABLE 按原政策处理，不擅自改选次优或延长八次重试预算。

#### 7. 性能是结构约束和实测门禁

正常路径新增执行器投递为 0，维持现有 planner 并行度和 Endpoint 锁粒度；请求锁内不评分、不预测、不等待网络。复用现有 selections、RoleSelection、候选缓存与投影，不新增每资源 Future、全局登记表或整批候选复制。

测量必须分开记录昂贵评分与廉价比较。无全局失效时，N 个可评估候选和 k 次单候选冲突，昂贵评分应为 N+k；允许重新比较缓存评分，不允许全量重新预测或额外整批缓存匹配。全局依赖真正变化时按正确失效范围刷新。

保留现有性能测试硬门槛：队列诊断捕获每次分配小于 1,024 B；超时诊断读取小于 10,000 ns 且可测时为 0 B；非空投影每次分配不超过 96 B。深度 512 捕获的 2 ms / 256 KiB 默认兜底上限不能当作允许退化幅度。

使用现有 tools/routing-hotspots/run.py 协议对照评分、快照和投影；它不包含 RPC 和完整调度线程，不能代表 P99 或 TTFT。同时补完整 route 操作测量，覆盖无冲突、单节点冲突、P/D 部分成功、取消和异步抢占：统计 CPU/request、bytes/request、评分/预测/捕获次数和线程切换。

完整链路维持相同 JDK、配置、引擎脚本和负载，覆盖 10/100 节点、0/32/512 队列深度、1/64 planner 并行扫描以及 DIRECT/QUEUE、BATCH/NON_BATCH 的适用组合。控制到达率测调度 P50/P95/P99，单独爬升测最大稳定吞吐，同时核对错误率、资源余额和最终清理；不得改变线程、组批或早退政策掩盖开销。

先测基线再制定与业务 SLO 对应的预算。没有既有预算时报告配对差值和置信区间；检出退化则失败，样本不足则增加测量，不能擅自允许下降某个百分比。发布微基准已有配对结果，整套吞吐、P99 和全量性能门禁仍未通过验收。

沿请求、资源和异步任务生命周期，消除重复状态、语义不完整的交接和补偿回路。必须保留自定义 scheduler、选择策略、入队与出队机制。目标不是拆类、减少接口数量或统一所有调度算法。

用户已授权按本手册实施并持续完成验证。本文不构成提交、推送或发布授权。

实施顺序：基线 → 入队与撤回（问题 1、6）→ 已接纳请求的执行与关闭（问题 2）→ 计时器（问题 3）→ 资源释放（问题 4）→ 抢占（问题 5）。每阶段完成生产调用迁移、扩展实现验证、旧机制删除和检查，再进入下一阶段。不得先创建全部抽象，最后联调。

本文中的接口效果是行为要求，不是必须照抄的 Java 类型设计。复用现有类型优先；新增状态必须说明其不可替代的信息。删除候选不是已经证明冗余的字段，须满足对应删除条件。

## 2. 开工前必须完成

1. 读 `AGENTS.md` 以及 `skills/flexlb-development/SKILL.md`、`skills/workflow-first-design/SKILL.md`、`skills/evidence-first-implementation/SKILL.md`，路径均相对 FlexLB 根目录。
2. 查看 `git status --short`、暂存与工作区差异。手册编写时 Endpoint、Dispatcher、策略、调度器、预测及测试已有大量未提交修改；不得覆盖、恢复或整体暂存别人的工作。
3. 阅读当前架构文档 `docs/architecture/01-routing-and-balancing.md` 至 `05-lifecycle-and-consistency.md`，以及同目录 `flexlb-complexity-review.md`。历史提案与实际实现冲突时，以当前代码和已确认业务契约为依据。
4. 重新定位本文符号和调用方，不依赖旧行号。记录当前 HEAD 和本阶段涉及的既有差异；不输出敏感配置。
5. 先运行本阶段现有测试建立基线。区分已有失败、环境失败与本次引入的失败；不能通过放宽断言使检查变绿。

### 2.1 基线表

每个阶段在本手册末尾的执行记录中填写，不能只写类图：

| 项目     | 必须回答                                     |
| -------- | -------------------------------------------- |
| 正常路径 | 从哪个生产入口进入，到哪个可观察结果结束？   |
| 写入事实 | 字段/记录表达什么，唯一修改者是谁？          |
| 提交点   | 哪一步之后副作用已经发生，不能按未发生处理？ |
| 交接     | 调用返回、任务接纳、网络发送分别意味着什么？ |
| 部分成功 | 哪一步成功、哪一步失败，谁继续负责？         |
| 关闭     | 尚未提交的后续工作和晚到回调由谁等待？       |
| 扩展     | 自定义实现可决定什么，必须保证什么？         |
| 删除     | 哪些状态、入口、反查、重试将失去用途？       |

## 3. 不得改变的约束

- 请求、一次尝试、reservation、Endpoint generation 必须准确匹配；旧回调不得修改新尝试。
- 发送结果未知不等于未发送；取消请求不等于远端停止；响应完成不等于资源结清。
- 保留当前优先级、容量、亲和性、等待、重试预算、超时和同步错误传播语义。发现这些契约不清时先从实现及测试确定，不能为简化机制静默改变。
- 不在请求锁内执行用户回调、网络等待或可能等待 Endpoint 锁的昂贵操作。
- 接纳关闭与已接纳工作结束是不同边界；队列为空、activeCount 为零不是结束证明。
- 保留热路径必要的派生统计和快照；不要通过全量扫描实现“状态只存一份”。

### 3.1 扩展验收

保留已有基类扩展能力，不顺带承诺任意只实现 `RequestScheduler` 的对象都能接入 API 和应用关闭：当前完整接入范围需要核对 RequestData、RequestRepository、FlexlbServiceImpl、ApplicationLifecycle 的实际类型依赖。

必须在外部包扩展测试中验证：自定义 scheduler 决定等待/重试，自定义策略改变候选选择，自定义队列改变成员顺序与出队规则，且不复制发布、资源清理或关闭补偿代码。不得强迫扩展使用内置队列扫描、组批或固定线程模型。未定义的新算法能力不属于本次自动扩展范围。

### 3.2 类与接口的职责、可见信息和依赖边界（强制验收）

实现必须保证每个类和接口只负责、只看到完成自身职责所必需的信息。测试通过和状态减少不能替代这项要求。以下是职责边界，不要求每行新建一个类；现有类承担多个角色时，先核对其必要性，不能通过拆类后传递完整 Context 伪装隔离。

| 角色              | 必要输入与责任                                                             | 不应掌握或操作                                                      |
| ----------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 调度决策          | 调度所需请求信息、容量事实、准确操作结果；决定选择、等待、重试、出队和组批 | 线程池内部计数、响应发布 Registration、资源账本字段和计时器安装细节 |
| 选择策略          | 候选、评分输入、容量观察及必要的精确登记能力                               | 请求结束协议、Future 完成、关闭执行器、队列内部结构                 |
| 队列实现          | 准确成员身份、排序字段及其出入队规则                                       | Context 的结束状态、远端取消、资源释放、响应发布                    |
| Endpoint 资源管理 | 准确 reservation/generation、容量和远端资源证据；登记与结清                | scheduler 的重试选择、请求响应内容、全局请求归档策略                |
| 请求状态转换      | 准确身份、业务事件和状态；裁决竞争结果                                     | 网络调用、执行器接纳协议、Timer Handle 和其他组件可变账本           |
| 请求流程协调      | 推进跨步骤操作、消费结果、处理部分成功与最终结束                           | 队列排序算法、Endpoint 私有记录和执行器私有状态的直接读写           |
| 异步执行设施      | 工作接纳、执行、完成及关闭；必要的任务身份                                 | 请求为何取消、候选选择、抢占业务策略                                |
| 计时设施          | deadline/token、任务安装与取消、到期通知                                   | 请求为何选择某个 deadline、响应内容、资源释放策略                   |

具体实施要求：

1. 修改类或接口前，写清一句职责、必要输入、输出/副作用和允许依赖。沿调用方核对，而不是仅凭类名判断。
2. 参数优先使用已有的准确身份、只读视图和操作结果。若传入完整 Context、Scheduler、Endpoint 或服务集合，逐一说明为什么需要其能力；不能只为方便而让接收方遍历对象图找数据。
3. 可变状态由负责该事实的组件修改。调用者通过有明确效果的方法请求操作，不拿内部 map、计数器或可变记录自行维护不变量。
4. 接口不泄露私有执行步骤：调用方不应被要求按 `reserve → mark → publish → ack → repair` 的隐含顺序配合内部实现。确有跨阶段业务操作时，用准确操作身份表达阶段与责任，不把步骤简单藏进万能接口。
5. 回调参数也必须最小化。禁止用 Supplier、Runnable 或函数接口绕过依赖边界，让底层回调读取上层全部状态；检查闭包捕获的对象及执行时的锁和线程。
6. 跨组件流程必须有协调者，但协调者只编排公共操作。不能把删除的状态、失败分支和组件私有知识全部搬进一个新的 Manager/Runtime。
7. 不为每个参数新建 DTO，不给每个类机械配接口，不引入只有转发作用的层。只有生命周期、权限或独立替换需求成立时才建立边界。
8. 错误结果只暴露调用方需要据此决策的含义，诊断原因可附带；不能迫使调用方识别线程池、资源 map 或通知设施的私有异常才能正确清理。

六项改造中的边界反例：队列不得为处理取消而读取 Context；发布执行器不得判断请求应成功还是取消；Timer 不得决定业务超时策略；资源句柄不得反调 scheduler 决定重试；抢占账本不得直接完成前端响应；撤回工具不得内置默认重新入队算法。

每阶段合入前，检查本次变更的构造依赖、方法签名、字段访问、回调捕获和失败分支。评审者必须能仅依据本类职责解释这些依赖的必要性。若一个局部改动要求多个无关类理解新增状态，退回检查事实归属与接口效果，不能以“后续再整理”通过验收。

## 4. 阶段 A：入队与撤回（问题 1、6）

### 初始审查基线的回路与入口

`AbstractRequestScheduler.enqueueRoute` 调 Endpoint 入队；异常后通过 `isQueued(route)` 反查物理成员，再 `Context.publishRoute`、signal，并可能重抛原异常。

`completePlacement` 撤回队列成员、关闭 Delivery、释放资源、回写 Context，然后检查终态/期限并调用 `scheduleRequest` 重入调度。重新入队失败又转请求结束。

阅读：`PrefillRequestQueue`、`OrderedPrefillRequestQueue`、`PrefillEndpoint`、`RoutingContext`、`RequestContext`、`RequestRouteSelector`、`DirectRequestScheduler`、`QueuedRequestScheduler`；原 `PlacementOperation` 及其发布/收尾入口已迁移为路由语义，按当前 `publishRoute/finishRouting/completeRouting` 调用追踪。初始基线中的命名仅用于解释删除来源。

### 目标契约

入队结果明确说明准确 route 是否提交。通知执行失败不改变已提交的成员事实。Context 与物理队列不必持有一个大锁，但中间阶段必须有明确负责者：取消方记录取消，由正在完成入队的操作负责清除本次成员，不能双方都认为对方清理。

撤回只提供旧尝试的确定结果；下一步重试、等待或结束由原 scheduler 的明确决策入口负责。现有 `scheduleRequest` 如果已经是合法扩展入口可保留，但调用条件和结果应显式，不能为了消除一次调用而再加等价包装层。

### 开发步骤

1. 枚举自定义队列实现。明确 `add/remove` 的提交与异常契约。不能一边允许任意修改后抛异常，一边删除唯一的结果查证方式。
2. 优先把成员修改与通知分开。若扩展确需部分提交，则返回包含提交结果的准确操作结果；异常仍不能伪装未提交。复用现有结果类型，禁止新增多个同义布尔标志。
3. Endpoint 封装一次成员修改的确定结果；迁移 scheduler 的调用与取消清理。
4. 收敛撤回操作：只有旧尝试已清理，才允许 scheduler 启动下一尝试。清理失败不能当普通选路失败重试。
5. 迁移 Direct、Queued 及外部扩展，保持原策略。删除旧入口、外层反查和隐含策略分支；不留永久双路径。

### 必测交错与删除门槛

- 入队完成、通知失败：成员与预留仍匹配，不重复入队、不错误回滚；后续推进有明确负责者。
- 物理入队之后、Context 发布之前取消：准确旧 route 被移除，最终无泄漏。
- 撤回一半时请求超时：不得再次入队；已释放部分不被重复计费。
- 旧 route 晚到回调与新 route 并存：不能释放新资源。
- 自定义 scheduler 选择等待而非立即重试：公共层不得自行重入调度。

起点测试：`PrefillRequestQueueContractTest`、`QueuedDeliveryWithdrawalRaceTest`、`RequestRouteTest`、`RequestAdmissionExpirationRaceTest`、`ExternalRequestSchedulerTest`、`CustomRequestSchedulerContractTest`。先读实际断言再选用。

退出条件：生产路径不再依赖“抛异常后反查是否提交”恢复操作结果；撤回结果和重试决策职责明确，扩展验证通过。若扩展契约确实要求反查，记录证据并修订删除目标，不能强删。

## 5. 阶段 B：已接纳请求的执行与关闭（问题 2）

### 初始审查基线的回路与入口

`confirmDelivery/publishResponse/publicationFailed/retryPublication/submitClose/runClose` 在 Context、响应 Registration、RequestTaskPool 之间交接。RequestTaskPool 同时追踪提交者、任务、stop、Lane 及关闭后的同请求续接。DeliveryExecution 另有 RPC 回调与容量排空。

阅读全部提交与关闭调用，包括 API `ApplicationLifecycle`、Repository、ResponseCompletionExecutor、DeliveryDispatcher、DeliveryExecution、ExpirationTimer，以及 `RequestFuture` 的外部 complete/cancel 路径。

### 目标契约

请求一旦接纳，公共运行部分负责必要响应和收尾直到明确完成。scheduler 决定业务结果与推进时机。内部工作不因“新请求接纳已关闭”而失去执行资格。不同 scheduler 可有不同执行模型，不强制改成单线程循环。

### 开发步骤

1. 列全工作生产者：同步调用、RPC 回调、计时器、队列/资源通知、扩展 scheduler。标明谁还能产生后续任务。
2. 定义结束条件，覆盖正在提交但尚未入执行器的工作。选择最少的追踪机制；不能盲删所有计数，也不能另造一套全局任务注册表。
3. 按依赖关闭：停止新请求 → 停止/结算调度来源 → 排空已接纳请求及必要回调 → 关闭其使用的执行器。具体计时器停止点由当前取消和超时契约决定，不机械照搬顺序。
4. 明确可重入响应回调不参与自身阻塞等待。响应完成和资源收尾可独立推进，保持现有可观察时序。
5. 在上述保证成立后，逐个移除因关闭竞争存在的备用 Registration 和续接特例。

必须区分关闭拒绝和真正执行失败。新边界只消除前者，不能宣称能保证线程创建失败等任意故障下继续服务。非正常失败必须保留明确错误和清理责任。

### 验收

用屏障控制停机与同步提交、晚到 RPC、响应重入、后续清理交错。断言响应最多一次、必要工作不丢、无提前关闭、无自等待死锁、资源最终结清。保留“响应可先于远端清理”的行为。

起点测试：`StopExecutorSafetyTest`、`RequestDrainTest`、`QueuedSchedulerStopRaceTest`、`RequestTaskPoolConcurrencyTest`、`ResponseCompletionExecutorTest`、`RequestPublicationFailureOwnershipTest`、`RequestLifecycleSubmissionTest`。

删除候选：仅用于关闭竞争的 `retryPublication`、ThreadLocal 特许续接、重复 owner 追踪。每项逐一证明覆盖范围；不要把 capacity permit、RPC pending 与本地任务计数误当同一事实。

退出条件：每个剩余计数对应不同未完成工作，关闭顺序可解释；不存在“已接纳但无人负责”的窗口，已失效的补偿入口实际删除。

## 6. 阶段 C：计时器（问题 3）

初始基线：Context 保存 Plan，Timer 保存 Slot/Handle；多个业务路径调用 `needsUpdate/update/recover` 修补安装。Plan 与 Handle 并非同义状态，不能机械合并。

阅读 `ExpirationTimer`、`ScheduledTaskTimer`、`TaskTimer`、Context 的 deadline 推导和 scheduler 的所有 update 调用。

开发步骤：

1. 为每种 deadline 写明业务含义、刷新条件、取消条件及当前失败行为。
2. 选一个固定的版本交付入口，保证 Context 已提交的版本有人负责安装。必须覆盖交付前取消、归档和关闭。
3. Timer 在接纳后负责版本替换、旧 handle 取消及安装失败处理。先确定失败是否可以重试；若改变失败后的请求行为，说明契约差异，不擅自采用新语义。
4. 逐个移除确认发送、发布失败、关闭通知中的重复修补。保留准确 token 和版本判断。
5. 删除失效的安装重试分支，不新增全局扫描作为默认兜底。

验收：旧版本后到不能覆盖新版本；安装与取消交错无孤儿；安装失败有确定后续；旧 token 不能终止当前状态；归档后无引用泄漏。

起点测试：`TaskTimerTest`、`RequestDeadlineTokenRefreshTest`、`RequestConfirmationTimeoutTest`、`RequestInactivityTest`、`RequestCleanupArchiveRaceTest`。

退出条件：安装责任集中，业务响应失败不再顺带修计时器。不能只把三次重试移动位置。

## 7. 阶段 D：资源释放（问题 4）

初始基线的 `releaseRouteResources` 读取 CloseFacts，释放 Endpoint 资源，再 `recordResourcesReleased` 回写。Encoder 和 Prefill 的结果被合并；一方失败时下一轮可能重复调用另一方。原 `cleanupRequest` 最多三轮推进通知和清理；最终顺序与保留判定见验收表。

阅读 RoleSelection、RouteResult、各 Endpoint、DecodeResources、ContextResult.CloseFacts、Context 的结束条件和全部 release 调用。

开发步骤：

1. 逐资源列清准确身份、取得点、释放证据、实际释放点、通知点。区分本地释放完成与远端允许释放的证据。
2. 复用准确 reservation/route 建立稳定释放结果；不要先统一成一个万能 Lease。资源记录被删除后，结果仍须能供原请求读取，不能通过 requestId 查新记录。
3. 先迁移一种资源的正常结束、回滚、取消和 retirement 全路径。重复调用可以幂等，但成功步骤不应依赖外层不精确进度反复执行。
4. 请求收尾消费资源结果，删除对应可独立写入的 Context 镜像。若必须保留聚合视图，应由事实派生。
5. 迁移其余资源；分析并移除固定 cleanup pass 中已经失效的再次尝试。通知与释放是不同事实，不能一并吞掉。

验收：Encoder 成功/Prefill 失败只继续未完成部分；Decode 远端未知不提前释放；旧 generation 结果不影响新记录；本地回滚不制造新的远端取消；响应不被不必要地延迟。

起点测试：`RequestResourceAccountingTest`、`RequestAdmissionResourceLeakTest`、`RequestCleanupPassTest`、`RequestTerminalSettlementTest`、`ReviewLifecycleRegressionTest`。

退出条件：每项资源只有一个可信释放结果；旧镜像已删除或成为不可独立写的派生结果。若 CloseFacts 某字段承载独有事实，保留并记录证据，禁止为兑现数量目标强删。

## 8. 阶段 E：抢占（问题 5）

当前 `reconcilePreemption` 读 Context → 更新 Decode 账本 → 回写 Context → 发布容量 → 完成 attempt 通知。锁外执行有必要，不能用扩大请求锁消灭交接。

阅读 `PreemptionRegistration`、Context 的抢占转换、DecodeEndpoint 的 claims/attempts、DefaultDecodePreemption 及取消回调。先确认每个字段的语义，不把业务结束和账本提交混为一谈。

开发步骤：

1. 写出同一 attempt 的资源提交点和所有通知接收方。
2. 使同一 attempt 的重复结算返回稳定结果，不重复扣减或释放；复用现有记录优先。
3. 账本提交后仅推进未完成通知；通知失败不得重放整项资源变更。
4. 检查 Context 的确认字段：只删除账本结果的镜像，保留独有取消/终态事实。
5. 对每处 `Failures.recover` 分别决定删除、保留或替换依据，不全局修改工具影响无关路径。

验收：账本提交后通知失败；worker 完成与抢占确认竞争；取消失败后恢复发送；重复通知；Endpoint 替换；旧 attempt 晚到。断言容量不重复变化、发起者不会永久等待、请求不重复发送。

起点测试：`RequestReconciliationFailureOwnershipTest`、`RequestAcknowledgementPreemptionRaceTest`、`RequestPreemptionRegressionTest`、`RequestNotificationOwnershipRaceTest`。

退出条件：账本提交与通知完成分别有明确事实来源；无需重新推断已提交动作是否发生，没有新建另一套抢占状态机。

## 9. 阻止补丁累积的实施规则

- 每次新增 boolean、map、计数、token 或重试分支，记录其独有事实、修改者、释放点和不能派生的原因。
- 新旧路径迁移以一个阶段为期限。临时适配必须写退出条件，阶段验收前清除；不可永久双写。
- 测试失败先分类：违反业务契约、依赖旧实现细节、发现设计漏项、环境问题。设计漏项先修本阶段契约及受影响调用，不在报错点直接叠加 fallback。
- 禁止以固定 sleep、增加循环次数、吞异常、扩大锁、全量扫描代替责任交接。
- 一项操作若部分成功，不得按整体失败重新执行；先识别真正未完成步骤。
- 发现候选删除项实际必要，保留它并给出反例；不能为了符合本手册改坏语义。
- 禁止把扩展算法强塞进公共层，也禁止把公共清理责任丢给每个扩展实现。

## 10. 验证、性能与交付

以下命令从 FlexLB 根目录运行，测试名单先依据当前实际文件确认：

```bash
./mvnw -pl flexlb-sync -am test -Dtest=PrefillRequestQueueContractTest,ExternalRequestSchedulerTest,CustomRequestSchedulerContractTest -Dsurefire.failIfNoSpecifiedTests=false
./mvnw test
python3 tools/order_java_members.py --check
./mvnw spotless:check -Pspotless-check
```

第一条仅是阶段 A 的起始验证，不能替代该阶段全部相关测试。每阶段跑受影响测试及扩展契约测试；最终跑 reactor 测试和全量 Spotless。提交时还必须按仓库规范运行 pre-commit；不绕过，不夹带用户改动。环境或无关基线失败如实记录。

并发测试使用锁存器/屏障指定交错。断言可观察结果、资源余额、准确身份和最终清理，不只断言私有方法调用次数。性能比较使用同一场景与环境，关注热路径扫描、分配、锁等待、吞吐和尾延迟；沿用仓库已有预算，未给预算时报告数据和不确定性，不虚构“无退化”。

每阶段交付记录：

| 项目     | 填写内容                                                                                 |
| -------- | ---------------------------------------------------------------------------------------- |
| 基线     | HEAD、涉及的既有改动、已确认行为                                                         |
| 完成路径 | 生产入口至正常/失败出口、扩展实现                                                        |
| 删除收益 | 删除的字段、登记、确认、重试、兼容入口及代码位置                                         |
| 新增机制 | 独有事实与必要性；没有则写无                                                             |
| 职责边界 | 变更类的职责、必要输入与依赖；删除的越界访问、完整对象传递及隐含调用顺序；仍保留者的理由 |
| 反例验证 | 具体交错、测试名、结果                                                                   |
| 检查     | 实际命令、通过/失败/未运行及原因                                                         |
| 剩余风险 | 未验证契约、环境限制、保留候选的理由                                                     |

实现变化后更新现有 `docs/architecture/` 相应说明，不另造与现状竞争的架构文档。最终核对：没有旧路径调用，没有同义双写，没有被测试夹具藏住的兼容入口，自定义实现不需要复制生命周期补偿。职责边界审核必须通过：每个类/接口只获得必要信息、只修改自己负责的事实，没有通过闭包、万能 Context 或转发层隐藏越界依赖；命名、类说明与实际效果一致，无失效包装、无用字段和重复实现。

## 11. 执行记录

### 2026-11-09：首批实现

基线 HEAD：`92ed25b137`，工作区包含已有修改，实施期间策略与预测相关文件继续变化。本批没有提交，也没有覆盖这些修改。

- 入队：Scheduler 使用 Endpoint 返回的提交结果，移除异常后反查 `isQueued` 的分支。队列接口明确校验失败不发布成员；内置实现把优先级校验前移，避免先插入后发现非法优先级，且仅在插入成功后递增序列。
- 响应：删除 `Registration.reserve` 的三次注册重试及失败报告回调参数；真实 register 的拒绝条件是不可逆关闭，不存在下一次立即恢复的条件。
- 职责：Registration 不再持有 RequestContext 或沿 Context/Data 查找 Future，只接收准确 RequestFuture 与锁检查对象。响应候选所属请求仍由 scheduler 使用，不让执行器访问业务状态。
- 测试：增加非法成员不改变队列/版本/快照、真实关闭后持续拒绝注册的测试；把模拟“关闭后自动恢复”的旧测试改为确认单次正常发布。关闭重入测试显式提供它实际完成的 Future。
- 本批相关 11 个测试类共 236 个用例，在相关测试运行及修复后的单独复验中全部通过。随后完整 `./mvnw test -q` 退出码为 0，包含 sync、api、mock-engine 等全部 reactor 模块。最终全量 `./mvnw spotless:check -Pspotless-check -q` 通过，修改文件的成员顺序检查通过。未执行隔离前后性能对比，不宣称性能无退化。

设计修正：`RequestScheduler.scheduleRequest` 已明确支持撤回后重新调度，回调本身是合法扩展契约，不能仅因通用流程调用它就判为越界。尚未证明需要替换该入口。

范围限制：这只是入队与响应设施的局部收敛，不代表阶段 A/B 整体完成，更不代表六项改造完成。尚未改变 RequestTaskPool 关闭协议、计时器版本交付、资源结清来源或抢占协调；不能宣称已经实现大规模减量。

### 2026-11-09：请求结束时的计时器同步

`notifyRequestEnd` 改为执行 Timer 的幂等 update，以成功返回作为本次同步完成的依据；失败保留通知未完成状态，交由已有关闭流程继续推进。删除该路径的安装前探测与安装后反查，避免一次失败后因查询认为无需安装而提前宣布通知完成。普通调度路径的 no-op 探测仍保留，未将其批量改成重复 update。

验证：ExpirationTimerTest、ExpirationTimerInstallationRecoveryTest、RequestTerminalSettlementTest、RequestPublicationFailureOwnershipTest、DeliveryClaimTest、RequestCleanupPassTest 相关测试通过；全量 Spotless 和修改文件成员顺序检查通过。本轮未重复完整 reactor 测试。

被否定的实现：取消抛异常后直接再次调用 Future.cancel，会违反现有并发取消至多发起一次的契约，相关实现已撤回。此处应先确认取消异常的可恢复性与责任，不能直接删除 canceled 保护。关闭与资源结清的整体收敛仍未完成。

### 2026-11-09：Encoder 登记输入边界

EncoderEndpoint.commit 现在接收明确的 uncachedTokens 与请求数上限，不再通过 route.ctx().data().getRequest() 读取输入长度和 Encoder 缓存命中参数。请求协调入口计算未命中 token 数，Endpoint 校验非负并管理精确 route 的容量记录。全部生产调用和测试调用已迁移，没有保留旧签名包装。

EncoderEndpointTest、EncoderSchedulingIntegrationTest、QueuedPublicationNotificationFailureTest、ExternalRequestSchedulerTest 通过；全量 Spotless 与修改类成员顺序检查通过。此项收窄数据依赖，并未删除资源结清进度，也不代表资源生命周期整体改造完成。

### 2026-11-09：删除响应备用登记

删除 retryPublication 中新建 fallback Registration 及其作用域关闭。响应构造或 deadline 安装失败时，原登记仍为 RESERVED，直接使用原登记提交失败响应；SUBMITTED/RUNNING/COMPLETED/CLOSED 不再作为重新登记的依据。提交拒绝仍由原执行器完成 Future 的异常结果，晚到任务仍受原登记阶段限制。

恢复入口改名为 publishUnsubmittedResponse，明确只完成原先未提交的响应。删除 hasCompletionOwner 聚合查询，以 isUnsubmitted 表达真正的提交前提，没有新增持久状态。准备失败参数化测试断言只登记一次，且资源继续保留至 worker 完成证据到达。

验证：ResponseCompletionExecutorTest、RequestPublicationFailureOwnershipTest、RequestCompletionPublicationRaceTest、RequestCancellationFailureOrderingTest、StopExecutorSafetyTest、RequestTerminalSettlementTest、ExternalRequestSchedulerTest、CustomRequestSchedulerContractTest 全部通过；全量 Spotless、修改生产文件成员顺序和 diff 空白检查通过。本轮未重新执行整个 reactor 测试。已接纳任务的整体关闭协议和资源结清来源仍未完成迁移。

### 2026-11-09：完整流程收敛与最终验收

以下记录取代前面各批次的“尚未完成”范围说明；最终测试结果单独填写，不把历史测试当作本轮验证。

| 阶段          | 本轮处理与保留判定                                                                                                                                                                                                                                                                                   | 主要验收证据                                                                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A：入队与撤回 | 入队以 Endpoint 明确结果为准，校验在成员发布前完成。当前工作区由实际 route 发布确定成员交接，RoutingCompletion.FINISHED（初始迁移名 PlacementCompletion）只结束尝试，不再以 PUBLISHED 枚举假定物理发布；撤回后仍由原 scheduler 的 scheduleRequest 决策，不增加新的调度回调层。清理失败不能重入调度。 | 队列替换、通知失败、撤回与取消竞争、外部 scheduler 和自定义队列测试。                                                                                        |
| B：执行与关闭 | 已删除响应重复登记、备用登记及其失败交接。保留不同工作来源的排空条件；任务池新增自身关闭重入检查，且在改变接纳状态前拒绝。                                                                                                                                                                           | RequestTaskPoolConcurrencyTest、StopExecutorSafetyTest、RequestDrainTest、QueuedSchedulerStopRaceTest、RequestLifecycleSubmissionTest。                      |
| C：计时器     | Scheduler 不再查询 needsUpdate；版本去重与安装判定只由 Timer.update 管理。确认发送先同步 deadline，再构造响应；删除响应失败入口中的计时器修复。旧 token 和安装失败状态保留。                                                                                                                         | ExpirationTimerTest、ExpirationTimerInstallationRecoveryTest、DeliveryClaimTest、RequestDeadlineTokenRefreshTest、RequestInactivityTest。                    |
| D：资源释放   | 先执行停止/deadline 与已经就绪的通知，再释放资源，最后立即完成依赖本轮释放才就绪的抢占通知。删除 notificationReady 触发的成功路径重复清理，三次尝试只用于实际失败。保留精确资源身份和本地结清观察，不建立新的万能资源句柄。                                                                          | RequestResourceAccountingTest、RequestCleanupPassTest、RequestTerminalSettlementTest、EncoderEndpointTest、ReviewLifecycleRegressionTest。                   |
| E：抢占       | 保留锁外账本提交、Context 确认和锁外通知。当前实现已经先确认提交，再独立执行容量及恢复通知，通知重试不会重放整项账本修改；不再新增另一份提交结果缓存。依赖资源释放的收尾通知在本轮内完成，已就绪的通知保持原有时序。                                                                                 | RequestReconciliationFailureOwnershipTest、RequestAcknowledgementPreemptionRaceTest、RequestPreemptionRegressionTest、RequestNotificationOwnershipRaceTest。 |

#### 必须保留的事实与反例

| 状态或机制                                 | 表达的独有事实                                   | 删除或合并会破坏什么                                                                                                     |
| ------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Context 中的资源结清观察                   | 一次准确 route 的本地清理已完成，可供归档使用    | Endpoint 当前 map 中没有记录，可能是退休、替换或从未登记，不能据此为另一次尝试生成释放证据。历史观察不是第二份容量账本。 |
| Delivery 发送和远端释放证据                | 请求可能已在远端执行，是否允许解除本地占用       | 仅以 Future 完成或本地排队任务结束释放容量，会把发送结果未知当作未执行。                                                 |
| notificationsAttempted / notificationsDone | stop 已交接与所有结束通知完成是两个阶段          | 抢占还在等待资源结算时，不能重复发 stop，也不能提前归档。                                                                |
| pendingWork / pendingStops                 | 本地效果与远端停止流程各自尚未完成               | 本地排空不应等待远端取消，但最终关闭必须等待二者。                                                                       |
| WorkRegistration 的提交者和任务责任        | 正在提交、尚未进入执行器的调用方仍可产生后续效果 | 仅靠线程池队列/activeCount 会在提交窗口提前关闭。                                                                        |
| 当前请求的 ThreadLocal                     | 执行重入检查以及关闭期间原任务的准确续接资格     | 允许任意请求续接会重新打开接纳；拒绝全部续接会丢失已接纳工作。替换成新许可对象会增加另一套交接。                         |
| deadline Plan / Slot                       | 期望时间与实际安装的任务不同                     | 安装失败、旧版本后到、回调先于安装返回都需要准确的安装记录。                                                             |
| 抢占 Context 与 Decode 记录                | 取消决策/请求结束与容量事务各自的结果            | 合并需要扩大跨组件锁或把调度策略放进资源层，不符合扩展和锁边界要求。                                                     |

不能为了减少字段再保存一套“某资源是否调用过 release”的标志。精确资源记录本身保证重复调用无重复容量副作用；Encoder 的旧 route、替代记录和重复释放测试新增容量通知次数断言。资源部分失败时，已完成的 Decode 结清观察使后续尝试跳过其实际释放。

本轮集成期间还修复了测试夹具的永久冲突重试：SelectionTestSupport 对同 ID 不同 route 立即拒绝，不再在 CONFLICT 分支无限循环。Response 已增加诊断附加信息，StrategyErrorTypeTest 使用精确完整消息验证分类文本和附加信息均被保留，不再断言旧的无诊断文本。

收尾顺序的反例已经验证：不能把所有通知统一推迟到资源释放之后。DirectAdmissionContractTest 要求本地已完成请求在晚到地址发布前已经 CLOSED；RequestTerminalSettlementTest 要求 deadline 更新失败时仍记录通知尝试并保留未完成责任。最终实现保持已就绪通知的原有时序，只在释放使抢占通知就绪时直接完成该通知；没有新增标志或扩大重试预算。上述回归测试保持原有行为断言。

#### 最终验收结果与版本边界

- 验证目录：`/tmp/flexlb-handbook-acceptance-20261109-01/rtp_llm/flexlb`，以 HEAD `92ed25b137` 加本次工作区快照建立；独立输出目录避免并行构建删除类文件或混用版本。
- 完整 `./mvnw clean test -q` 已执行。生产逻辑、sync、API 等通过；mock-engine 中错误码矩阵有 8 个旧文案断言失败，已修正为同时验证 deadline 到期和具体 Prefill KV 压力，保留错误码、拒绝原因、PV 和引擎未接收断言。
- 随后执行 `./mvnw -pl flexlb-mock-engine -am test -Dtest=SchedulingErrorCodeMatrixTest -Dsurefire.failIfNoSpecifiedTests=false -q` 退出码为 0。合并全量执行与本次修复后复验的 Surefire 报告：4,277 个用例，0 failures，0 errors，3 skipped。不能将首次全量执行的退出码记为 0。
- 快照全量 Spotless、修改核心文件成员顺序及 diff 空白检查通过；主工作区最终全量 Spotless 也通过。故障注入的持续负载场景报告引擎接收与完成一致、零泄漏；未执行隔离前后性能对比，不宣称性能无退化。
- 验证期间主工作区继续发生其他修改，最终不能再声称整棵最新工作区与快照逐字节相同。已核对本次 cleanupRequest、prepareRequestEnd、completeRequestEndNotifications、publishFailure、publishUnsubmittedResponse、confirmDelivery：除并行的 VictimResolution → PreemptionRequestResult 类型改名外，方法实现一致；RequestTaskPool、ResponseCompletionExecutor、ExpirationTimer、EncoderEndpoint 与快照逐字节一致。其他并行修改不纳入上述全量验收结论。

本次收敛没有新增持久状态或新框架；没有提交或推送。需要保留的机制及反例已逐项列明，不把保留审计描述成删除成果，也不把其他并行修改归为本 agent 的实现成果。
