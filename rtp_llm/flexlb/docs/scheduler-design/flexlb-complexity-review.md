# FlexLB 复杂度 Review：沿实际流程检查扩展边界

- 日期：2026-11-09。
- 不可突破的前提：支持自定义 Scheduler、选择策略及出入队机制。
- 目标：找出公共实现和自定义实现共同变复杂的具体原因；不以搬代码、拆类、取消扩展或全局串行化代替减量。
- 基线：重新阅读时的当前工作区，包含用户已有未提交改动。引用方法名定位，不将上一轮源码或 HEAD 当作当前实现。
- 阅读边界：第 1–20 节记录修改前的流程取证及方案；交错推演不代表运行测试结果。第 21 节记录随后实际实现的代码及验证，不能把整份方案视为已完成。

## 1. 阅读路线与事实范围

以一个需要 Prefill 和 Decode 的生成请求为主线，走 QUEUE + BATCH 路径，再核对 DIRECT、容量阻塞、取消、退役、路由撤回、本地优先级替换和远端抢占。

具体来源：

| 编号 | 源码                                                                                                                   | 本次追踪的入口                                                                                                             |
| ---- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| S1   | [FlexlbServiceImpl](../../flexlb-api/src/main/java/org/flexlb/httpserver/FlexlbServiceImpl.java)                       | scheduleLocally、routeLocally、取消绑定                                                                                    |
| S2   | [AbstractRequestScheduler](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/AbstractRequestScheduler.java) | submit、claimPlacement、publishPlacement、commitDelivery、completePlacement、advanceRequest、关闭与清理                    |
| S3   | [QueuedRequestScheduler](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/QueuedRequestScheduler.java)     | enqueue、planningCandidates、place、finishDecision、decideQueueScheduling、decideGroup、resumeAfterPreemption              |
| S4   | [RequestRouteSelector](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestRouteSelector.java)         | select、registerDecode、registerPrefill、rollback                                                                          |
| S5   | [DecodeEndpoint](../../flexlb-sync/src/main/java/org/flexlb/balance/endpoint/DecodeEndpoint.java)                      | tryRegisterSelection、acquireDispatchPermit、dispatch、资源对账与抢占事务                                                  |
| S6   | [PrefillEndpoint](../../flexlb-sync/src/main/java/org/flexlb/balance/endpoint/PrefillEndpoint.java)                    | tryRegisterSelection、enqueueRoute、captureQueue、commitLocked、releaseRequest                                             |
| S7   | [DeliveryDispatcher](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DeliveryDispatcher.java)             | submit、prepare、commit、deliverCommitted、dispatchBatch、report、StopTask                                                 |
| S8   | [RequestContext](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestContext.java)                     | beginPlacement、publishRoute、completePlacement、recordDeliveryResult、recordWorkerRequestEvent、beginClose、completeClose |
| S9   | [DefaultDecodePreemption](../../flexlb-sync/src/main/java/org/flexlb/balance/eviction/DefaultDecodePreemption.java)    | replaceQueuedDecodeReservations、preempt、handleAcknowledgements、Attempt                                                  |
| S10  | [RequestTimeEstimation](../../flexlb-sync/src/main/java/org/flexlb/balance/projection/RequestTimeEstimation.java)      | 虚拟请求插入、GroupPlanner 重放、预测分组推进                                                                              |

以下用 R 表示一个请求，P/D 表示选中的具体 Prefill/Decode Endpoint 实例，O 表示一次 PlacementOperation，L 表示整条 RouteResult，M 表示一次 Delivery.Member。

这些身份不能混同。request_id 相同不代表同一次尝试，Endpoint 地址相同不代表同一代实例。

## 2. 从入口到选路：每一步实际发生什么

### 2.1 接入、注册与第一次入全局队列

| 步骤 | 调用及线程/锁                                                                    | 实际写入或副作用                                                                              | 此时成功意味着什么                         | 下一步失败由谁处理                                   |
| ---- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| A1   | S1 scheduleLocally → routeLocally → submit；请求调用线程                         | 准备取消监听绑定回调，进入公共请求入口                                                        | 尚未选路                                   | API 处理接入异常，已注册请求交给公共结束流程         |
| A2   | S2 submit → RequestRepository.register；registrationLock → Context → RequestData | 将 Scheduler、RequestFuture、请求身份、期限与 inactivity 参数绑定；登记 activeByPhase         | 请求已被接纳，尚无 Worker 资源             | 公共请求生命周期                                     |
| A3   | submit 安装缓存观测/输入释放回调、ExpirationTimer，再执行 onRegistered           | API 在登记后、scheduleRequest 前绑定取消                                                      | 取消可以命中已注册请求                     | 取消走原 Scheduler，不依赖下一轮调度                 |
| A4   | S3 scheduleRequest → enqueue；Scheduler 锁                                       | GlobalQueueEntry 加入 orderedQueue 与 queuedEntries；首次发放 sequence，恢复时保留原 sequence | 请求进入全局选路候选范围                   | removeRequest 清理队列及等待索引；在途规划另行收尾   |
| A5   | scheduleAt → planningCandidates；Scheduler 锁                                    | 容量就绪者变为 ready retry；扫描游标前进；候选加入 inFlight                                   | 仅取得规划执行名额；没有从全局队列删除请求 | submitPlan 被拒绝时 finishPlanning；取消仍可删除成员 |
| A6   | place 在线程池执行 → claimPlacement；Context 锁                                  | Context.placement = O                                                                         | 当前请求有一个活跃选路/交接操作            | place 的 finally 必须结束 O；取消不能当作 O 已退出   |

关键区别：A5 后“已被扫描”“仍在全局队列”“正在规划”“等待容量”是不同事实。

[OrderedRequestQueue](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/OrderedRequestQueue.java) 的 scanForPlanningCandidates 不等于出队；它移动扫描游标并移除 readyRetries 中的条目，真正的成员删除发生在后续 removeRequest。

因此，orderedQueue、inFlight、waitingRequests 不能仅因都包含 R 就判为重复。复杂度问题在于：自定义 PlacementQueue 必须遵守这套扫描、恢复与重试协议，接口所表达的范围已经超过普通队列容器。

### 2.2 角色选择与本地资源登记

S4 select 使用局部 selections 收集本轮角色登记；按配置中的 requiredRoles 顺序执行，多角色 group 由既有选择约束。以下 D、P 步骤是在需要两者的路径中分别展开，并非声称所有配置都按同一顺序选择。

| 步骤 | 调用及锁                                                               | 写入或副作用                                                                                                    | 此时仍未发生的事                                 |
| ---- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| B1   | Decode 策略读取候选快照并评分                                          | 通常只是本次策略的候选与成本缓存                                                                                | 没有请求级新路由发布                             |
| B2   | 策略调用 register 回调 → registerDecode → D.tryRegisterSelection；D 锁 | 校验 generation、admissionVersion、Worker 观察与 topology；检查容量；建立 MASTER_QUEUED_NOT_DISPATCHED 资源记录 | 不是 Engine 实际接纳；没有发送许可，也没有发送   |
| B3   | registerDecode → addDecode                                             | 将精确 D 预留加入局部 selections                                                                                | Context.route 仍可能为 null                      |
| B4   | Prefill 策略获取 projectionInputs、工作量、cache-hit 等并评分          | 生成 Selection，传给登记回调                                                                                    | 评分结果仍需通过登记校验                         |
| B5   | registerPrefill → P.tryRegisterSelection；P 锁                         | 校验当前预测输入及版本；QUEUE 建立 SELECTED 记录，DIRECT 建立 RESERVED 记录；本地工作量开始影响后续选点         | QUEUE 尚未加入 Worker 队列；Context 还没有接受 L |
| B6   | new RouteResult(context, selections) → Prefill RequestRecord.bind      | P 的资源记录绑定精确 L                                                                                          | 对象构造已经有资源关联副作用，但还不是路由发布   |
| B7   | select 验证 Endpoint 并返回 SUCCESS                                    | 本轮资源的清理责任从 Selector 的局部 selections 交给外层                                                        | SUCCESS 不表示已入队、已发送或已响应             |

局部失败的处理有两层：

1. B2/B5 取得单个资源后，若构造 RoleSelection 或加入 selections 失败，registerDecode/registerPrefill 立即释放刚取得的资源。
2. 任一后续角色失败、取消或最终 Endpoint 校验失败，select 的 finally 回滚本轮已经登记的角色。

这是避免多角色部分成功泄漏所必需的流程。不能为了接口简洁删除“选中即登记”，否则连续选点看不到已选工作，会改变负载分布。

### 2.3 一个具体边界问题：登记版本绑定了预测缓存的物理身份

S6 tryRegisterSelection 不只校验业务资源版本，还要求：

    progress.ownershipVersion 与 expected 相同
    projectionSource 存在且属于当前 progress
    projectionSource.materialized == expected
    predictor.evaluator() == expected.evaluator()

所以 PrefillSelectionStrategy 的调用者必须提供当前完整 RequestTimeEstimation.Inputs 对象，单有正确 Endpoint 和资源版本并不够。

反例推演：自定义策略按轮转选择 P；资源容量没有变化，但预测 evaluator 或当前物化输入发生变化。登记仍可返回 CONFLICT，策略必须重新取得预测输入并重走登记。

对默认 TTFT 策略，这可能是必要的评分有效性检查；对不依赖该评分模型的策略，其有效性依赖未必相同。当前将预测缓存身份也作为共同登记条件，使所有策略被绑定到同一个有效性范围。

这不是资源超额缺陷，而是一个可定位的复杂度来源：预测实现细节进入资源提交和扩展契约，导致自定义策略需要理解缓存、evaluator、对象身份和冲突重算。

## 3. 从选中到路由发布：两次登记及一次结束，不是一个 SUCCESS

### 3.1 QUEUE 路由交接

| 步骤 | 操作                                                  | 状态变化                                                             | 失败责任                                             |
| ---- | ----------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------- |
| C1   | publishPlacement 验证 O 和 Endpoint                   | 尚未必发布                                                           | 外层持有本轮 L 的资源责任                            |
| C2   | enqueueRoute → P.enqueueRoute；P 锁                   | P 的 SELECTED 记录变 WAITING，L 进入自定义 PrefillRequestQueue       | 未被 Context 接受的资源仍需回滚                      |
| C3   | Context.publishRoute(O, L)；Context 锁                | Context.route = L，记录 P/D 仍待释放                                 | 如果取消先发生，拒绝这次交接；O 的结束路径负责释放   |
| C4   | signal(INPUT)                                         | 唤醒 Worker 队列调度；通知可能触发另一线程读取 L                     | 通知失败不能被解释为此前没有入队或没有绑定           |
| C5   | publishPlacement 的 finally → finishPlacement(L, O)   | 未发布路由释放；已发布路由保留；完成 O                               | 回滚失败时 retainRouteForCleanup，不能当作资源已归还 |
| C6   | Queued.finishDecision(SUCCESS) → removeRequest(entry) | 删除全局成员、等待关联及 queuedEntries；finishPlanning 释放 inFlight | Worker 资源和队列成为后续工作来源                    |

当前重新读取的 enqueueRoute 直接接收 Endpoint 的布尔结果；上一轮出现过的“异常后再次 isQueued 探测”不能继续作为当前实现证据。

C2 与 C3 之间的短暂状态差异并不是自动构成错误。Context 中的 O 以及 Worker 调度入口对 placement/route/pendingEnd 的检查，正是用来识别交接未结束的窗口。

但这是扩展需要面对的真实成本：只知道“调用入队成功”还不足以解释这个请求是否已被公共生命周期接受。

### 3.2 一个具体语义混用：PUBLISHED 不一定有路由

普通选路返回 BLOCKED 或 REJECTED 时，place 的 finally 仍通过 finishSelection 结束 O。

finishPlacement(O) 对非撤回操作使用 PublicationResult.PUBLISHED。Context.completePlacement 主要据此清除 placement，并不要求此时存在 route。

因此，同一个 PUBLISHED 实际覆盖：

- 真正发布了 L 后结束 O。
- 本次没有发布路由，只是选路操作退出。

这使扩展作者必须记住：PublicationResult 并不总能按字面解释，真实发布事实还要另读 Context.route。

代码量的来源不是这个枚举有几个值，而是“操作结束”与“业务发布结果”语义混合后，调用方需要额外对象检查和隐含约定来补全含义。

### 3.3 DIRECT 的真实差别

[DirectRequestScheduler](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DirectRequestScheduler.java) 同样 claimPlacement、selectAndPublish，但直接路径会：

1. 准备 M 并取得 D 的临时 dispatch permit。
2. P.commitDirect 将 RESERVED 记录变为已提交工作。
3. Context.publishRoute 绑定 L。
4. 计算前序工作。
5. commitDelivery 转移许可并记录请求投递开始。
6. 生成 RouteDelivered，公共反馈路径确认路由响应。
7. finishPlacement 保证 O 在响应处理前结束。

此处响应交付的是路由地址，不是 Master 发出的 BATCH RPC。P/D 资源继续保留到 Engine 完成等后续依据到达。

因此 DIRECT 与 BATCH 的结束与释放条件有真实差异。减量不能靠把两种成功语义硬合并。

## 4. 容量不足的完整分支：等待不是把请求简单放回去

在 place 进入时记录 observedCapacity = availability.sequence()，这是防止选路过程中错过容量变化的依据。

| 返回结果 | Queued 后续                           | 资源状态                                            |
| -------- | ------------------------------------- | --------------------------------------------------- |
| RETRY    | finishPlanning 将请求标为 ready retry | 本轮登记须已清理，或已转入失败清理                  |
| BLOCKED  | 可能尝试抢占；否则 park               | Selector 已回滚本轮部分登记，不跨轮保留任意部分路由 |
| REJECTED | 发布失败响应，删除全局成员            | 公共生命周期结束                                    |
| SUCCESS  | 删除全局成员                          | 已交给 Worker 队列或直接投递                        |
| CLOSED   | 删除全局成员                          | 取消/结束或关闭路径继续收尾                         |

[PlacementCapacityWaitQueue](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/PlacementCapacityWaitQueue.java) 的实际等待协议：

1. 把 blocker 归入 role/group 等待域。
2. 如果该域容量变化序号已晚于 observedCapacity，park 返回 false，立即再规划，避免丢唤醒。
3. 否则登记 membership 与域内有序条目。
4. 容量变化使域 available。
5. resumeReady 每次把一个条目变成该域 retry，回到 orderedQueue 的 ready retry 索引。
6. 成功或取消可把机会继续传给后继；本次仍受阻则重新 park，消耗当前机会。

这不是资源预留。被唤醒的请求仍要重新选择，不能保证固定节点可用。

这里确有多组索引，但分别表达成员、规划执行、容量等待、机会传递。不能凭“集合多”删除。可审查的边界是：这些机制属于某个 Scheduler 的实现，还是所有替代 PlacementQueue 都被要求承接同一种扫描/恢复协议。

## 5. Worker 出队到物理发送：至少三个成员集合

### 5.1 候选与组批

S3 runQueueScheduling 先判断上一等待是否仍成立，必要时移除容量订阅，再调用 decideQueueScheduling：

1. P.captureQueue 捕获有限数量的有序候选和版本。
2. 检查队首的 Placement 交接、过期及 KV 可行性。
3. 检查部分后续候选是否过期。
4. GroupPlanner.select 选择一组，dispatchReason 决定是否立刻提交。
5. 若仍在收集窗口，记录 nextCheck，不取得发送资源。
6. 重新检查实际容量约束，然后调用 Dispatcher.submit。
7. 若阻塞资源是 Decode Engine，Scheduler 再询问队列能否提供绕过队首的候选。

所以替换队列实现并不等于接管全部出队机制：触发绕过的资源条件在 Scheduler，替代候选规则在队列，组批在 Planner，部分接纳规则在 Dispatcher 参数。

### 5.2 临时资源准备和 Endpoint 提交

| 步骤 | 操作                                                              | 新增状态/资源                                                                                                       | 失败后的处理                                            |
| ---- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| D1   | Dispatcher.submit 校验候选成员及 selectionStillAllowed            | 尚无新的发送事实                                                                                                    | 返回 OWNERSHIP_LOST 等边界，成员仍可能留队              |
| D2   | prepare 按候选顺序逐个处理                                        | BATCH 首成员取得执行许可；每成员取得 D dispatch permit；检查消息大小                                                | 遇到首个失败/阻塞或消息上限停止，形成实际 prepared 前缀 |
| D3   | 根据 PrefixPolicy 判断是否接受部分准备                            | ALL_SELECTED 在此要求完整；ORDERED_PREFIX 可接受前缀                                                                | 未提交时 Delivery.close 归还临时许可，不删除原请求资源  |
| D4   | commit 计算实际准备集合的批次预测，写 selectionTime、bypassedHead | Delivery 仍未必 committed                                                                                           | 失败不等于请求已发送                                    |
| D5   | P.commitDelivery / commitLocked；P 锁                             | 再核对成员、容量、绕过条件；创建 BatchRecord；成员从队列删除并变 COMMITTED；调用 recordDispatch；Delivery.committed | 此后不能按“仍留在等待队列”处理                          |
| D6   | dispatchCommitted → 执行许可.submit                               | 发送任务被接纳                                                                                                      | 提交拒绝/准备失败必须向各请求报告，不能只释放临时许可   |

D5 是本地 Worker 队列和资源提交，不是 Context 的发送开始，也不是 RPC 已调用。

### 5.3 实际发送前，成员还会再次变化

S7 deliverCommitted：

1. 读取各请求 Context，去掉不再 preparedDelivery、路由不匹配或 Endpoint 不可用的成员。
2. 根据剩余成员构建完整 protobuf payload。
3. 逐个调用所属 Scheduler.commitDelivery(M)。
4. commitDelivery 在 Context 锁内检查取消、期限、路由和 Endpoint；M.initialize；D permit.dispatch 将记录变为 ENGINE_MAY_HAVE_SEEN；Context.recordDeliveryStarted。
5. 未能开始的成员从列表和 payload 中移除。
6. 集合变化时重新预测实际批次。
7. 上报 Prediction，再调用 batchEnqueueAsync。

因此要明确区分：

    Planner 选中的集合
      → Dispatcher 准备并在 Endpoint 提交的集合
      → Context 接受开始、最终放进 RPC 的集合

取消可以合法地令最后一个集合变小。ALL_SELECTED 只在资源准备阶段检查完整性，并不保证随后所有成员都一定通过开始发送检查。不能将它解释为远端原子提交保证。

### 5.4 对自定义出队机制的具体影响

PrefillRequestQueue 当前的提交反馈是 recordDispatch(skippedHead)，发生于 D5；没有直接传入最终 RPC 集合。

如果某个公平性机制按“取得本地执行名额”计费，D5 可以是它需要的点。

如果它按“实际开始发送”或“Engine 接受”计费，反馈点和反馈内容不同。它不能仅通过一个 skippedHead 参数准确获取后两类事实。

因此具体缺口是：**扩展出队策略的状态更新时点及输入，没有和公共执行过程中的真实阶段一一对应。**

不能把这当作所有策略都需要更多事件。应先明确已支持的计费/公平性语义，再提供相应完整事实，避免各扩展反向读取 Delivery、Context 或猜测 remove 的原因。

## 6. RPC 返回、响应与 Engine 完成：再区分三个结束点

### 6.1 发送结果

dispatchBatch 用 invoked 区分：

- 调用 RPC 前失败：NOT_SENT。
- 已进入 RPC 调用但异常或结果未知：UNCERTAIN。
- 回复到达：BatchEnqueueHelper 校验 batch 和成员，产生逐请求结果。

report 将 TransportResult 送回请求原 Scheduler。

Context 的 DeliveryFacts 将结果归为 NOT_SENT、REJECTED、DELIVERED、UNKNOWN。UNKNOWN 不能自动重发或释放；它可能与 Decode 已确认事实共同支持投递确认。

### 6.2 响应确认

onDeliveryResult 接受精确 M 的事件后调用 advanceRequest：

1. 优先处理可撤回的未发送路由失败。
2. 协调现有抢占。
3. 根据取消/待结束原因选择清理，或者确认投递。
4. 同步新的期限计划。

confirmDelivery 在 Context 锁内核对确认资格、inactivity 和抢占，登记响应执行任务并确认；锁外 publishResponse 选择响应并执行 Future 完成。

Future 完成会运行用户回调，因此必须与 Context 锁分开。响应任务是否退出，也不同于 Future.isDone。

### 6.3 Engine 完成与资源归档

P/D 状态对账先修改各自本地资源事实，解锁后发布 WorkerRequestEvent。

Context.recordWorkerRequestEvent 校验路由、预留和角色；P 完成不必意味着整个 P/D 请求结束，Decode 完成等事实才可能形成请求结束。

随后：

    beginClose 冻结请求结果
      → 通知 Dispatcher 停止/等待必要远端事实
      → releaseReason 判断当前是否允许释放
      → releaseRouteResources 逐角色释放并记录进度
      → completeClose 形成最终 RequestState
      → archive 删除活跃请求并保留终态
      → 删除 Timer 注册

completeClose 不仅要求资源释放，还要求通知完成、Delivery 可释放、没有 Placement，并且抢占已经结束。

因此“已返回成功”“远端执行结束”“全部清理归档”不能合并成一个完成状态。这里有必需的分布式事实，不能将其全部归咎为代码设计差。

## 7. 具体失败与竞争推演

### 7.1 D 已登记，P 选择失败

1. D 记录已经影响后续选点。
2. P 返回 BLOCKED。
3. Selector finally 回滚 D 及本轮其他已取得角色。
4. 结束 O。
5. Queued park；DIRECT 按即时失败策略返回。

不能为减少回滚代码而让 D 在后续轮次一直保留，否则新增跨轮部分绑定协议，会改变容量占用和选点策略。

可减少的是重复的本轮资源收尾实现，不是必要的回滚行为。

### 7.2 P 已进入 Worker 队列，Context 尚未接受时取消

1. C2 完成，P 记录 WAITING，队列包含 L。
2. 取消记录 cancellation/pendingEnd。
3. C3 的 publishRoute 拒绝交接。
4. finishPlacement 发现 L 未被 Context 接受，释放整条 L 的资源。
5. O 结束后，原先因 Placement 暂缓的请求结束得以继续。
6. Worker 若已读取候选，也必须通过 Context/路由精确校验拒绝执行。

这条竞争说明：只让自定义队列负责 remove 不能完成取消；公共资源与请求状态的交接必须有确定负责者。

### 7.3 本地队列已提交，但某个请求尚未开始发送时取消

1. D5 已创建批次、删除队列成员并反馈 recordDispatch。
2. 取消使某成员不再满足 preparedDelivery。
3. deliverCommitted 过滤或 commitDelivery 拒绝该成员。
4. payload 和预测按剩余集合收缩。
5. 被取消请求按公共清理路径释放资源及相关批次成员。

这里不能把出队反馈等同于发送反馈；也不能将失败成员简单重新 add 到原队列，否则可能绕过期限、请求结果和资源清理。

### 7.4 RPC 结果未知，而 Worker 状态先到

1. 发送已跨出调用边界，记录 SENDING/UNKNOWN。
2. Decode 对账接受精确预留，记录已被 Engine 观察。
3. Delivery 确认规则结合 UNKNOWN 与 decodeAccepted 判断是否可确认。
4. 若同时取消，仍需按远端清理依据推进，而不能视作未发送重放。
5. 不论 Worker 反馈与 transport 回调先后，旧 M 都不能修改新尝试。

这是远端不确定性带来的必需状态；所谓统一失败重试会破坏正确性。

### 7.5 未发送路由的 Endpoint 下线后重选

1. 默认 Queued.shouldRetryRouteFailure 只对 ENDPOINT_UNAVAILABLE 选择重试。
2. Context.failUnsentRoute 检查是否确实允许撤回；已发送、未知或已被 Engine 观察的路由不能按同样方式重放。
3. beginRouteWithdrawal 用 O 取得旧 L 的撤回权。
4. completePlacement(WITHDRAWN) 释放旧路由资源；失败转请求失败，不能继续重选。
5. Context 清除旧 route、delivery、routeFailure 等尝试事实。
6. 检查期限，再调用原 Scheduler.scheduleRequest；保留原排序序号和截止时间。

这里必须保留“自定义 Scheduler 决定重试策略，公共机制验证能否安全重试”的分工。可以删除的是扩展反复实现撤回与资源清理，不是重试选择权。

## 8. 抢占的两条真实路径及成本来源

### 8.1 未发送排队 victim 的本地替换

S9 replaceQueuedDecodeReservations：

1. 对每个 victim 找原 Scheduler。
2. owner.beginPreemption 核对优先级、精确 D 预留、路由和可撤回状态，在 victim Context 登记撤回 O。
3. D.replaceQueuedRequests 在 D 锁内检查整个 victim 集合并创建 incoming 的精确 D 登记。
4. 若替换成功，对每个 victim completePlacement(WITHDRAWN)；否则 RESTORED。
5. victim 的 WITHDRAWN 路径释放 P 等实际资源，并回调原 Scheduler.scheduleRequest。
6. 某个撤回收尾失败时，incoming D 登记也要释放，不能继续接纳。

真实困难在于：D 的替换原子性不覆盖 P 队列和 victim Context。这里必须有跨资源协调，不能仅让队列删除 victim。

当前协调还要理解每个 victim 的 Placement 结果、后续重排及 incoming 回滚；它与普通撤回路径共享操作类型，形成复杂调用关系。

### 8.2 Engine victim 的远端取消

1. Preemption 为各 victim 调用 owner.tryClaim，取得 PreemptionRegistration；Context 保留其阶段与待结束事实。
2. D.beginPreemption 在 Endpoint 内创建 victim claim、incoming 临时登记和 attempt 记录，防止释放的容量被第三方抢走。
3. Scheduler.onPreemption 推进取消阶段。
4. Dispatcher 发送 Cancel，分别记录 ACK、NOT_FOUND 或 UNKNOWN。
5. Worker 状态和 Cancel 回复都可能触发 Scheduler/Context 对账。
6. onPreemption 或 reconcilePreemption 在 Context 核对精确尝试，锁外修改 D 账本，再回 Context 提交结果。
7. victim 的通知和资源条件满足后，完成 VictimResolution。
8. Preemption 等待 resolution，D.commitPreemption 要求 remainingVictims 已为空，才能将 incoming 临时登记变成可排队记录。
9. Queued.resumeAfterPreemption 处理晚到 incoming 登记；请求已取消就释放，否则将该登记交回 Selector 补齐其他角色并发布。

ACK 与 victim 真正结束不能合并。未知 Cancel 也不能当作未发生；Attempt.abort 必须只撤销安全可撤销部分，将已经发出的部分留给后续对账。

### 8.3 从实际步骤得到的结论

多份状态不全重复：Context 表达请求能否继续，D 表达容量账本，Dispatcher 表达 RPC 事实，Preemption 协调 incoming 与多个 victims。

可审查的具体重复责任是：一次 victim 处理同时需要多个入口解释“阶段可以如何前进”和“余下清理由谁接手”；incoming 又由 Queued 保留 O、等待结果并处理晚到资源。

不能以“状态多”判定可删。应逐项找只有交接作用、没有独立业务事实的标记，并在收敛同一操作的推进责任时删除其对应状态和恢复，而不是再新增一个大 Manager。

## 9. 结束与关闭为什么也会复杂

closeRequest 先在 Context 锁内登记请求工作和响应执行资格，再 beginClose；锁外交给 submitClose。

submitClose 根据实际执行方式决定本地同步/异步清理；inactivity 不阻塞 Timer 线程。某些远端结果可先响应，本地取消通常先清理再响应。

cleanupRequest 最多进行三轮“通知结束—检查释放条件—释放资源—重新检查”。原因之一是释放事实可能推进抢占完成，进而使结束通知变为可执行；晚到事件还会再次进入 advanceRequest。

这说明结束是等待多个事实到齐的推进过程，而非一次 close 调用。

已有 [RequestTaskPool](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestTaskPool.java) 为同一请求的清理安排串行执行与通知合并；[ResponseCompletionExecutor](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/ResponseCompletionExecutor.java) 则隔离用户完成回调及排空。

不能用单个队列为空、Future 已完成或瞬时线程空闲替代这些保证。可减少的是让 Scheduler 扩展接手执行设施内部责任、分散的交接协议和重复的调用顺序知识。

## 10. 基于流程证据重新归纳的实际原因

### 10.1 共同执行依据依赖默认算法实现细节

证据：B5 的资源登记依赖当前物化预测输入精确身份和 evaluator，策略 Selection 同时承载资源必需事实、评分和诊断。

后果：替代选择算法仍需学习默认预测有效性及输出结构；缓存/模型变化也进入共同登记冲突处理。

需要澄清的边界：资源一致性检查、策略评分有效性、工作量估计分别基于什么事实，不能默认所有策略依赖相同评分模型。

### 10.2 操作身份、操作退出与业务结果混用

证据：O 同时用于选路、撤回和本地抢占；finishPlacement 的 PUBLISHED 可能没有发布路由；是否真的发布再看 Context.route。

后果：调用方必须同时理解枚举、Context、route、operation=null 的责任转移以及 finally 约定。扩展实现复制的不是业务策略，而是这些隐含顺序。

需要减少的是语义混用及重复收尾；不应机械地为每个阶段新建一个长期对象。

### 10.3 选择结果和执行结果之间没有足够明确的扩展反馈

证据：D1–D6 与物理发送形成三个不同成员集合，队列主要得到 skippedHead 反馈。

后果：有状态的自定义出队机制容易通过读取内部状态、推断 remove 原因或修改公共代码补齐计账依据。

需要明确的是哪个阶段更新哪种策略状态、公共层向它报告哪些真实成员与失败事实。不同阶段不能压成一个含糊的 dispatched。

### 10.4 默认执行算法被分摊成多个扩展点，但整体替换边界不足

证据：全局候选扫描与容量机会传递分属 orderedQueue、waitingRequests 和 Scheduler；Worker 绕过、组批、部分提交分属 Queue、Scheduler、Planner 和 Dispatcher。

后果：局部替换实现能工作，但改变一条完整规则可能需要多处新增条件。每增加一个 hook，又要增加版本匹配、重新探测与恢复。

需要判断哪些规则必须共同变化，形成完整决策与提交反馈，而不是取消任何一个已要求的扩展能力。

### 10.5 请求运行入口与调度实现绑定

证据：Repository、RequestData、API 与应用关闭实际关联 AbstractRequestScheduler；反馈经请求原 Scheduler 进入基类。

后果：公共资源与生命周期能力自然继续进入基类；扩展需要继承整组能力。重试回调关联原 Scheduler 是必要的，但不要求它承担所有执行设施细节。

此项是巨类集中形成的位置原因，前四项才解释为什么需要这么多协议代码。

## 11. 哪些先前结论不应继续使用

- 不再把接口、回调、阶段或集合的数量本身视为设计错误。
- 不再把所有本地登记和远端状态当成重复数据。
- 不主张把策略全部改成纯评分，再在不知道策略取舍的外层统一重试。
- 不主张把完整流程变成不可替换的统一 Scheduler。
- 当前已删除的预测反向读取阻塞决策，不作为现状证据。
- 当前 enqueueRoute 不再沿用此前异常后 isQueued 探测的证据。
- 不把 ALL_SELECTED 误解为最终 RPC 成员的原子性保证。
- 不将静态推演报告为已复现缺陷；本报告主要评价协议和扩展成本。

## 12. 减量应具体删除什么，以及必须保留什么

| 从流程中确认的问题                | 可争取删除的内容                                        | 不可删除的语义                               |
| --------------------------------- | ------------------------------------------------------- | -------------------------------------------- |
| 预测物化对象成为共同登记依据      | 自定义算法被迫适配默认预测缓存及无关重算                | 评分依赖改变时的重新比较、本地影响及时可见   |
| O 退出与发布结果混用              | 依赖额外对象检查的解释、重复 finally 和双重结束调用约定 | 一次尝试的精确身份，取消后晚到资源必清理     |
| 多次成员收缩缺少明确反馈          | 扩展推断队列删除原因、反向读取内部集合的辅助代码        | 取消可阻止未开始发送的请求，发送未知不得重放 |
| 出队规则分散                      | 为同一业务规则在多层增加的特判和回调                    | 自定义入队、出队、组批、绕过与公平性自主权   |
| incoming/victim 推进责任往返      | 只用于交接、没有独立业务事实的标记和恢复路径            | 已发取消与资源释放分离、incoming 容量保护    |
| 生命周期能力经实际 Scheduler 暴露 | 自定义实现中的公共清理与关闭适配代码                    | 原 Scheduler 决策与安全执行机制的明确交接    |

以上不是立即可删的逐字段清单。没有证明状态失去独立含义以前，不能删除；没有验证性能以前，不能用全局串行化消除竞争。

## 13. 推荐的落地方案：先消除协议，再收缩类

以下是推荐实施方案，尚未实现或验证。方法名用于明确操作语义，不表示需要逐字新增全部类型。优先修改和复用现有对象，不建设通用事件总线、通用事务框架或第二套请求状态机。

### 13.1 目标流程与责任

| 环节               | 谁作可替换决策                              | 公共实现完成的操作                                  | 返回后仍由谁负责                                       |
| ------------------ | ------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------ |
| 调度触发与候选组织 | 自定义 RequestScheduler                     | 接纳请求、通知可运行条件、提供受保护的执行入口      | Scheduler 负责自己的调度工作；公共部分负责请求生命周期 |
| 选择 P/D 等角色    | 自定义选择策略                              | 校验准确版本、登记本地影响、管理本轮取得的全部资源  | 发布前由本次尝试负责，发布后转交请求                   |
| 路由发布           | Scheduler 选择目标路径与时机                | 入队/直接提交、Context 接受、失败回滚、结束本次操作 | 公共实现保留未完成清理，不交给调用方猜测               |
| Worker 出队        | 自定义出队策略，默认实现可复用 GroupPlanner | 准备实际许可、校验决策、提交精确成员、物理投递      | 公共执行负责许可和发送；策略只更新自己的决策状态       |
| 失败后动作         | Scheduler/策略决定等待、重试、拒绝或抢占    | 报告已发生事实，先完成规定的撤回与清理              | 决策者不能绕过安全重放条件                             |
| 响应与结束         | 公共生命周期按已接受的业务结果推进          | 响应仲裁、远端依据、资源释放、归档                  | 不由每个 Scheduler 重新实现                            |

目标不是调用层次最少，而是：每个扩展实现不再复制公共操作的收尾，公共执行也不再读取默认 Scheduler 的算法私有状态。

## 14. 选路与发布：让现有 PlacementOperation 接住资源责任

### 14.1 将“选路尝试结束”从“发布结果”中分离

保留 PlacementOperation 的精确身份，但不再用 PUBLISHED 表示未发布的普通退出。

- 普通尝试退出只表示：不会再发布新的路由；未交接的资源已释放，或清理失败已由公共部分保留。
- 发布事实只在 Context 接受精确 RouteResult 的位置建立。
- 旧路由撤回分别表达“撤回成功”和“恢复原路由”，不借普通选路退出推断。

将当前 finishPlacement 的职责收成现有操作的确定结束入口。结束入口幂等，但正常调用路径只由一个负责人调用一次；幂等用于拒绝晚到或重复通知，不用来要求扩展到处补一次 finish。

### 14.2 每轮资源只登记一次责任，交接一次

将 Selector 当前局部 selections 及回滚责任归到本轮 PlacementOperation 的作用域内；不要保留局部列表再新增一份全局跟踪集合。

1. 单角色资源取得后，立即登记到本次操作。登记到操作前发生异常，仍由该取得方法立即释放，保持现有局部保证。
2. Selector 返回的角色组合只是本次操作持有的选择，不把清理责任隐含转给任意调用方。
3. Context 接受整条路由后，在同一个明确交接点把已取得资源交给请求生命周期。
4. 未交接就退出，操作释放全部实际取得资源。
5. 释放失败，保留精确资源与失败，进入现有公共清理路径，不返回可安全重试的普通结果。

RouteResult 构造器不再直接调用 Prefill 记录的 bind；由上述公共发布/关联步骤显式完成绑定。保持 Prefill 记录与完整路由的精确关联，取消的是构造返回值时隐藏修改活账本的副作用。

### 14.3 对扩展提供不夹带策略的完整执行入口

以现有 selectAndPublish 为基础收敛，不先新建一层通用执行框架。概念用法是：

    result = placements.tryPlace(request, selector, destination);
    // Scheduler 根据 result 决定等待、拒绝、重试或抢占。

其中 destination 表达已有的直接路由或 Worker 入队交接目标，不用 null 同时暗示执行方式和失败策略。

入口负责创建 O、调用 Selector、发布、回滚、完成 O；返回的 BLOCKED 保持 BLOCKED，不在公共入口按直接模式转换为拒绝。DIRECT 现有的即时失败策略放回 DirectRequestScheduler，QUEUED 现有等待与抢占策略保留在 QueuedRequestScheduler。

同步普通选路继续同步返回结果，不为统一异步接口而把所有请求包成新 Future。

### 14.4 抢占续选仍然受管理，不把收尾再次暴露给扩展

抢占路径需要跨异步边界保留操作责任，不强制在 finally 提前结束原 O，也不未经验证地改变取消等待行为。

把 Queued 当前 resumeAfterPreemption 中“晚到登记检查、已取消时释放、接入 Selector、发布、finally 完成”的机械部分收进同一公共操作实现。Scheduler 仍决定是否发起抢占、目标以及失败后的策略。

精确 incoming 登记在消费前由公共抢占操作持有：成功关联后交接，取消/下线/续接拒绝时释放。借用现有 RequestTaskPool 登记已接纳的操作与续接，覆盖完成前的关闭等待；不能因为规划线程已经退出就释放这份责任。

具体删除项：普通选路与续选调用方的 operation=null 交接标记、重复 finishSelection/finally、默认模式失败解释，以及本轮资源的重复回滚包装。保留原始失败和清理失败的区分，但将异常合并收在公共操作边界。

## 15. Prefill 登记：区分资源版本、评分依据和预测缓存

### 15.1 不以缓存对象身份作为业务提交条件

调整 PrefillEndpoint.tryRegisterSelection 的输入：接收明确的资源版本及工作量，不再通过 projectionSource.materialized == expected 判断提交资格。

保留以下独立校验：

- Endpoint 精确代际、仍可接纳及请求资源身份。
- 用于本地影响和容量判断的资源版本。
- 当策略评分依赖预测时，其评分依据的预测版本。

使用小型不可变版本值表达这些事实，不能让“是否命中同一个物化缓存对象”影响正确性。物化缓存被回收或重建、但对应事实版本未变时，不应仅因此冲突。

### 15.2 保留默认 TTFT 的重新比较行为

默认 TTFT 选择必须携带它依赖的预测版本；登记时依据已变化，仍返回冲突，由默认策略刷新受影响候选并重新比较。不能改成只检查容量还有余量后继续接受原胜者。

不依赖 TTFT 的策略不携带该评分依赖，公共实现仍为本地记账取得所需工作量估计。若工作量估计需要刷新，先在锁外刷新该候选的估计，再按资源版本提交；模型更新是否使候选偏好失效由实际评分依赖决定，不能一律重新执行轮转选择。

版本应随对应事实改变，而不是简单把现有对象地址换成一个每次构造都递增的编号。资源读写仍在原 Endpoint 锁边界内；预测仍在锁外。

### 15.3 公共结果不要求模仿默认评分模型

资源登记所需的候选身份、版本、工作量与 cache-hit 等执行事实保留。亲和原因、缓存查询结果和默认评分解释转为可选观测信息，不参与通用的成功判定或资源回滚。

不强制将 selectAndRegister 拆成失去原子登记语义的“纯评分接口”。先保持策略入口与冲突重新比较能力，删除预测缓存物理身份依赖及无关字段适配；再按实际重复量复用 P/D 登记机制，不建立一个改变角色策略含义的统一重试循环。

具体删除项：Endpoint 对 projectionSource.materialized 的引用相等校验、策略为满足该条件传递整个预测物化对象的依赖、替代算法必须适配的默认评分解释，以及纯缓存重建造成的额外冲突。保留业务版本和评分过期时的冲突。

## 16. 出入队：给完整决策与实际执行建立明确反馈

### 16.1 明确两个独立可替换范围

- 队列实现负责成员组织、入队接纳、删除、恢复、快照及自身状态。
- 出队策略负责从快照中决定候选、组批、绕过、何时再检查以及是否允许部分准备。

二者都允许自定义，不能把出队策略固定成默认行为。可新增一个窄的 DispatchPolicy 扩展点；默认实现组合现有 GroupPlanner 和默认绕过规则。独立类型只有在同时删除散落的默认决策分支后才成立，不新增一层转发。

现有 PrefillRequestQueue.Factory 和 PlacementQueue 替换入口保留迁移路径。全局队列继续由自定义 Scheduler 决定如何使用，不要求所有 Scheduler 采用默认 scanBudget/ready retry 机制。

### 16.2 出队决策与提交校验

一次决定只保存会影响公共执行的信息：精确候选、对应队列/策略版本、全选或前缀准备要求、必要的决策有效条件、下一次检查时间。没有成员的决定可以表达等待。

处理顺序：

1. 在 Endpoint 锁内捕获成员与策略所需的不可变状态。
2. 锁外调用策略选择，允许昂贵预测，不修改活队列或取得许可。
3. 公共执行准备实际资源。
4. 锁外重新读取需要的容量事实，由策略判定业务决定是否仍成立。例如默认绕过仍需核对队首是否继续受阻。
5. Endpoint 锁内校验精确成员、队列/策略版本和资源提交条件，再提交实际成员。

保留必要的提交前核对，不宣称可以跨多个 Endpoint 获得全局原子快照。与当前相同的关键竞争测试必须继续成立，包括队首恢复时放弃绕过。

将 QueuedRequestScheduler 中触发绕过、临时 bypass/blockedHead 状态及组批业务分支收进默认策略。Dispatcher 不再接受捕获默认 PrefillQueueScheduler 活状态的 BooleanSupplier；通过明确的决策及策略有效性检查实现同样保证。

### 16.3 三个阶段必须准确反馈，不能都叫 dispatched

| 反馈         | 建立事实的位置                      | 内容                                         | 扩展用途                             |
| ------------ | ----------------------------------- | -------------------------------------------- | ------------------------------------ |
| 本地队列提交 | PrefillEndpoint.commitLocked        | 本次决定、实际提交成员、被准备失败移除的成员 | 按本地执行名额计账，更新默认绕过规则 |
| 请求开始发送 | commitDelivery 成功后、Context 锁外 | 精确 M 与批次，实际开始的成员                | 按开始发送计账，区分提交后取消       |
| 远端结果     | report 接受精确回执后               | 实际成员的 DELIVERED/REJECTED/UNKNOWN 等事实 | 仅供确实依赖远端接受情况的策略使用   |

这是有限的操作反馈，不为每个请求创建新的订阅、事件队列或全局追踪表。默认策略只处理它需要的本地提交反馈，其他反馈可为空实现。

把 recordDispatch(skippedHead) 改为完整的本地提交反馈，skippedHead 是其中默认策略可使用的事实，不能代替成员结果。队列 remove 继续只表示删除成员，不承担推断取消、失败或发送原因的职责。

本地队列与策略状态的短操作沿用明确的锁约束；扩展代码不得在该回调内做网络、等待或昂贵计算。开始发送和远端反馈不在 Context 锁内执行。

策略反馈抛异常时不能伪装成提交没发生，也不能重放已提交资源。停止该策略继续作新决定，走现有 stopAndDrain/失败处理保留已接纳工作的公共清理；不无限重试可能已部分更新的回调。相关故障需要单独测试。

### 16.4 预测沿同一出队决策重放

默认 DispatchPolicy 的不可变快照和选择逻辑同时用于真实候选与虚拟 incoming。现有 GroupPlanner 继续作为默认策略可复用的纯组批能力。

不要在 RequestTimeEstimation 中继续增加某个自定义公平算法的分支。需要预测的扩展提供可重放的策略状态；投影推进只更新该快照副本，不修改活策略状态。

未提供可重放信息的扩展，预测明确报告未知；配置及消费策略必须接受该含义，不能偷偷按 FIFO 模拟。默认路径的预测结果和阻塞语义必须保持。此处不承诺任意算法都可精确预测。

具体删除项：Scheduler 和默认队列中重复分散的绕过判断、用于跨组件传递决定的 bypass/blockedHead 临时字段、推断 remove 原因的扩展代码，以及预测器中需要复制业务出队规则的特判。

## 17. 抢占与生命周期：收敛机械推进，不收回策略自主权

### 17.1 抢占保留两条真实机制

本地排队 victim 替换继续使用现有精确路由撤回与 D 原子替换；Engine victim 继续使用 Cancel 和远端依据，不强行合并状态。

自定义策略选择 victim 和抢占时机，DefaultDecodePreemption/替代协调实现负责一次 incoming 与 victims 的完整推进。公共请求执行能力提供“取得 victim 处理资格、应用精确反馈、完成/恢复”的操作，不让抢占算法操作 Scheduler 的线程池和关闭设施。

在没有逐项证明之前，不删除 victim claim、incoming 临时预留、远端 UNKNOWN 或 Context 待结束事实。可以先删除的是 Scheduler 中重复的异步续选资源收尾，使用第 14 节的公共操作接住晚到登记。

后续再把一次 victim 的 Context—D 对账固定在一个公共处理入口：该入口负责精确身份、锁外账本修改、再次提交和容量通知。onPreemption 与 reconcilePreemption 共用这个事务步骤，但保留两者不同的输入事实与策略含义。若不能减少实际重复检查或恢复，不为抽方法增加另一层结果对象。

### 17.2 最后调整 Scheduler 与请求运行能力的关联

前述协议收敛后，再把公共请求执行能力与 RequestScheduler 关联分开：

- RequestData/Repository 保留原 RequestScheduler，供重排、移除待调度请求、重试决策和调度工作停止使用。
- 请求提交、取消、状态反馈和公共资源清理由稳定的公共运行入口提供，API 不要求实际 Scheduler 必须是 AbstractRequestScheduler。
- 原 AbstractRequestScheduler 可作为薄的兼容基类委托这些完整操作，不再承载整个执行实现；独立实现 RequestScheduler 也应能接入完整 API 与关闭。
- ApplicationLifecycle 组织共享组件的停止与排空；每个 Scheduler 只报告并排空自己实际接纳的工作。

这一步不是把现有 1991 行整体复制进 RequestLifecycle。先完成第 14–16 节的实际协议删除，再按剩余操作责任复用现有 Repository、Dispatcher、Timer 和执行器；公共入口可以是薄协调层，不能再暴露整套内部设施。

具体删除项：API/Repository 对实际调度基类的类型依赖、通过请求取得 Scheduler.taskPool 的访问，以及每种扩展重复的公共生命周期适配。不得同时保留两套独立请求 owner 或清理状态来源。

## 18. 迁移顺序、验证及删除账单

| 顺序 | 实施范围                                            | 必须贯通的路径                                              | 验收与删除                                                         |
| ---- | --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ |
| 1    | PlacementOperation、Selector、现有 selectAndPublish | 普通 P/D 成功、部分失败、发布中取消、回滚失败、Direct/Queue | 删除假 PUBLISHED 退出和调用方重复收尾；失败策略留在 Scheduler      |
| 2    | Prefill 版本与策略结果                              | 默认 TTFT、自定义轮转/不同评分、模型变化、资源竞争          | 删除物化对象身份依赖；默认重新比较结果保持，缓存重建不制造无关冲突 |
| 3    | Worker 出队决策及有限反馈                           | 默认 FIFO/优先级、绕过、GroupPlanner、部分准备、提交后取消  | 删除散落默认分支；自定义策略只实现对应规则，不修改 Dispatcher      |
| 4    | 预测使用同一决策快照                                | 默认差分、扩展快照重放、未知预测                            | 默认结果保持；删除扩展算法的预测特判和重复业务规则                 |
| 5    | 抢占续选复用受管理操作                              | 本地 victim 撤回、远端 ACK/UNKNOWN、晚到 incoming、关闭     | 删除 Scheduler 异步资源收尾重复；不得丢失任何未完成资源责任        |
| 6    | 接入与生命周期类型解耦                              | 外部包独立 Scheduler、API 提交取消、Worker 反馈、应用关闭   | 删除实际基类依赖；兼容基类仅委托，无第二套状态                     |

每步先打通真实纵向路径，再迁移调用方和删除旧入口，不先铺满空接口。必要的兼容适配只转到同一实现，不能维护两份资源账本或双重反馈。

验证至少包括：

1. 默认实现现有成功、阻塞、重试、取消、回滚失败和关闭测试继续成立。
2. 外部包自定义 Scheduler 能等待、拒绝和主动发起下一次调度，且不写资源 finally。
3. 自定义策略不依赖 TTFT 时，无需伪造其缓存身份；默认 TTFT 在评分过期时仍正确重新比较。
4. 一个按本地提交计账、一个按开始发送计账的出队实现，在部分准备和提交后取消时分别得到准确成员。
5. 队列提交成功后通知/策略反馈失败，不能重发或丢失公共清理。
6. 远端 UNKNOWN、抢占取消晚到、incoming 释放和关闭交错不影响新尝试。
7. 运行现有预测差分及实际路由/调度性能基准，比较扫描、预测重算、分配量、锁等待、吞吐与尾延迟；不为代码短降低既有门槛。

每步合并前给出实际删除的字段、方法、适配与恢复分支，并核对公共实现加扩展实现的总成本。方案阶段不承诺未经实现验证的代码减少比例。

## 19. 开发与审核应改变的具体做法

1. **每个公共操作核对成功的实际含义。** 区分资源登记、队列提交、请求发送开始、RPC 调用、ACK、响应、清理归档。方法名和返回结果不能让调用方靠猜。
2. **每个扩展点核对输入依赖和输出责任。** 例如轮转策略为什么需要当前预测物化对象；公平队列按哪个提交点更新状态；不是只检查“有接口即可替换”。
3. **一次规则变化追到最终效果。** 改出队选择后，继续核对临时许可、实际成员、payload、预测、取消、回滚和策略反馈。
4. **保护事实，不保护偶然机制。** 精确身份与 NOT_SENT/UNKNOWN 必须保留；默认游标和枚举混合语义未必应成为所有扩展的永久契约。
5. **重构必须交付删除账单。** 列出不再需要的状态、跨组件检查、恢复分支、扩展适配和测试夹具；只移动代码不算减量。
6. **以真正不同的扩展检验局部性。** 外部包自定义 Scheduler、不同评分依据、不同出入队规则，应能复用公共保证，不依赖内部包访问或复制事务。
7. **性能与正确性一起验收。** 保留局部刷新、有限扫描及并行计算的有效约束，不能为了代码短改变等待、回退或公平性策略。

## 20. 测试证据与后续验证范围

已阅读或对照的相关契约包括：

- [外部包自定义 Scheduler](../../flexlb-sync/src/test/java/org/flexlb/extension/ExternalRequestSchedulerTest.java)：复用选路与投递、自主触发、取消。
- [自定义调度和替代队列](../../flexlb-sync/src/test/java/org/flexlb/balance/scheduler/CustomRequestSchedulerContractTest.java)：扫描预算、精确恢复、期限与装配替换。
- [DirectAdmissionContractTest](../../flexlb-sync/src/test/java/org/flexlb/balance/scheduler/DirectAdmissionContractTest.java)：Prefill 提交后取消、未发布回滚失败保留、下线重选、通知失败后仍保留已发布资源。
- [QueuedBatchDeliveryTest](../../flexlb-sync/src/test/java/org/flexlb/balance/scheduler/QueuedBatchDeliveryTest.java)：绕过竞争、容量恢复及临时许可释放。
- [DeliverySettlementTest](../../flexlb-sync/src/test/java/org/flexlb/balance/scheduler/DeliverySettlementTest.java)：未知发送结果和远端清理依据。
- [RequestSchedulerContractTest](../../flexlb-sync/src/test/java/org/flexlb/balance/scheduler/RequestSchedulerContractTest.java)：取消 Future 后仍存在的选路、回调与关闭排空。

本次没有运行这些测试。正式实施前，应针对上述具体步骤选择交错验证，尤其是 B5 有效性范围、C2/C3 交接、D5 后取消、部分准备与最终发送反馈。

缓存、主从一致性以及模型算法本身未作同等深度全面审核；不把本报告结论未经验证推广为所有模块的问题。

## 21. 本轮实际代码：版本登记与明确状态

本轮只落实前两处具体边界，不宣称整个减量方案已经完成。

### 21.1 资源登记按事实版本校验，保留策略内局部重算

RequestTimeEstimation.Inputs 改为携带 SelectionVersion，内容是 Endpoint generation、ownershipVersion 和 schedulingInputVersion。原五参数构造器保留给合成投影输入，合成输入没有可登记的 Endpoint 身份。

PrefillEndpoint.tryRegisterSelection 比较这些版本及 evaluator，不再要求 projectionSource 存在，也不再要求 materialized 与 expected 是同一个对象。相同事实版本的重新物化输入可以登记；旧资源版本、旧调度输入和不同代际仍被拒绝。

CostBasedPrefillStrategy 和 DecodeSelector 的候选缓存、策略内登记循环及冲突后的局部刷新保留：首轮计算所有候选；本轮单点冲突只重算该 Endpoint，与其余已缓存评分重新比较；不重新查询全量 cache 匹配。当前仍保留 evaluator 校验，并未实现对所有自定义评分模型的可选依赖协议。

新增测试验证“相同版本、不同物化对象可登记”和“资源版本不变但调度输入变化仍冲突”。既有 conflictRecalculatesOnlyTheChangedPrefillAndReusesCacheLookup 验证冲突候选读取两次、另一候选一次、cache 查询一次；bestPrefillRegistrationBlockDoesNotFallBackToAnotherCandidate 保护默认最优节点阻塞语义。

### 21.2 操作阶段、路由用途与操作结果分开

- PlacementPhase：NONE、SELECTING、FINISHING_SELECTION、WITHDRAWING、FINISHING_WITHDRAWAL；替代 Context 内的 completingPlacement 布尔值及其隐含含义。
- RouteState：NONE、ASSIGNED、CLEANUP_ONLY；发布建立 ASSIGNED，未发布回滚失败或请求进入清理后为 CLEANUP_ONLY，资源归档/清除后为 NONE。
- PlacementCompletion：FINISHED、RESTORED、WITHDRAWN、FAILED；FINISHED 只结束普通选路操作，不声称发布。Context 校验普通选路与撤回操作不能互用结果。

Snapshot 同时公开这些事实；preparedDelivery 要求 RouteState.ASSIGNED。保留清理路由仍可匹配精确清理依据，但不能因 route 非空就参与发送。

生产调用方与现有测试已迁移 PublicationResult → PlacementCompletion、PUBLISHED → FINISHED。自定义扩展若直接使用这组低层类型，需要进行同样的源码迁移；此处未保留另一套同义枚举。

新增测试覆盖未发布选路结束、收尾阶段、回滚失败路由仅供清理，以及普通选路不能使用撤回完成结果。

### 21.3 验证与未通过项

- 当前源码复制到隔离临时目录，补齐原有相对路径 proto 源，避免共享 target 被并行构建清理。定向运行 PlacementOperationTest、PrefillResourceSnapshotTest、DirectAdmissionContractTest、CustomRequestSchedulerContractTest、ExternalRequestSchedulerTest 及上述两项 Prefill 策略方法：133 个用例，0 失败、0 错误。
- 工作区较宽回归运行 379 个用例，0 断言失败、1 错误：RequestTerminalSettlementTest.nextPlacementAttemptCannotReuseEarlierCapacityAsItsTimeoutCause 的响应执行器 mock 报 frontend publication is closed；未修改该测试或响应执行器来消除错误。
- 本次六个核心文件的 Spotless 检查、成员排序检查及 git diff --check 通过。
- 已从 FlexLB 根目录执行完整 Maven reactor 的 Spotless 检查；执行时被另有工作区改动的 flexlb-common/Response.java 格式阻断，整个 reactor 尚未通过，不能以定向检查冒充全量完成。
- 未运行性能基准；局部重算的调用次数得到契约测试验证，但不据此宣称已有吞吐或尾延迟实测结论。

## 22. onDeliveryResult 的调用为什么分散（当前工作区复核）

本节只评审流程，不修改 Java 实现。当前生产代码有 8 个调用表达式，集中在 AbstractRequestScheduler 和 DeliveryDispatcher；测试调用不计入。工作区仍在演进，以下以方法名和事件类型定位。

| 调用方                        | 事件                                 | 真实含义                                              |
| ----------------------------- | ------------------------------------ | ----------------------------------------------------- |
| executeRouting                | RouteDelivered                       | DIRECT 路由交付就绪，结束路由操作后推进响应发布       |
| commitDelivery 的异常分支     | TransportResult(NOT_SENT)            | 已登记 delivery 的 BATCH 提交失败，尚未调用 RPC       |
| commitDelivery 的另一异常分支 | PreparationFailure(member, ...)      | 已登记的路由交付准备失败                              |
| preparationFailed             | PreparationFailure(null, route, ...) | 没有 delivery 的路由准备失败                          |
| deliverBatch                  | Prediction                           | 提交实际批次成员后设置预计可见时间，还未收到 RPC 结果 |
| report                        | TransportResult                      | RPC 结果或明确未发送/发送不确定的结果                 |
| publishRoute                  | RouteDelivered                       | 队列模式返回路由，推进响应发布                        |
| StopSession.applyReply        | CleanupAcknowledgement               | 取消回复提供远端清理确认，不是发送结果                |

正常 BATCH 的同一个 member 会经历 Prediction → TransportResult，因此调用两次是当前接口设计的必然结果，不等于发送两次。若进入取消确认，后面还会有 CleanupAcknowledgement。DIRECT 和队列 ROUTE_DECISION 的两个 RouteDelivered 来源则是不同执行路径，不应理解成一个请求必然收到两次成功通知。RouteDelivered 的命名也偏早：这里是触发后续响应发布，并不证明前端已收到响应。

真正的问题是 DeliveryEvidence 把准备、预测、交付结果、取消意图和远端清理放在同一个协议里。RequestContext.recordDeliveryResult 又通过 instanceof 拆回不同流程；PreparationFailure 甚至允许 delivery=null，迫使外层倒推请求身份。统一入口没有统一业务含义。

另一个可见问题是异常责任重叠：commitDelivery 在 transferred 后发生异常时先通知，再抛出；deliverBatch 的成员循环 catch 又调用 preparationFailed，publishRoutes 的赋值未返回时也会进入 started==null 的准备失败通知。两个通知通常靠请求身份、已有 delivery 和路由撤回状态过滤，不能据此断言资源一定重复释放；但调用方无法知道内部是否已经处理，重复防御确实存在。

清理还有旁路：StopSession.applyReply 若成功把 CancelResult 交给抢占操作，就不调用 onDeliveryResult；PreemptionCompletion.apply 内直接 recordDeliveryResult(cleanup)。这是为了与抢占状态在同一请求锁内处理，不能只把外部调用次数压成一个就认为边界清楚。

建议具体收敛如下（尚未实现）：

1. 将 Prediction 合并进提交 delivery 的参数：实际成员和预测确定后，以 commitDelivery(member, mode, batchId, visibilityDeadlineMs) 一次记录发送身份和预计可见时间。批次部分提交失败时，需在发送前按最终成员修正预测；保持批次特征聚合与批次预测的复杂度，不重新评估所有 Endpoint。预测不能等 RPC 回来才记录，否则发送期间的可见性超时语义改变。
2. onDeliveryResult 只接受 BATCH 的实际发送结果，发送前失败和异步回调都经 DeliveryDispatcher.report 汇总。RouteDelivered 改走明确的路由响应发布操作；这不是给事件总线换名字，而是让 DIRECT 和出队路径在各自提交之后调用同一发布行为。
3. commitDelivery 不再同时通知并抛异常。采用明确的提交结果区分“未接纳”“已提交”“提交失败”；如果失败已经发生资源转移，结果必须携带确切 member 以及需要清理的资源，外层接管一次处理。不能简单删除内部 catch 而遗失已转移资源。
4. 无 delivery 的准备失败直接进入已有 onRouteFailure 路由撤回流程，不再构造 nullable delivery 的事件。
5. 远端清理确认使用独立的记录操作；普通取消和抢占共用同一状态变更逻辑。抢占仍需在请求锁内原子应用抢占结果与清理确认，统一锁外推进后续动作，不强行改成两个独立回调。

验收应覆盖：正常 BATCH 只有一次发送结果通知；RPC 前失败只处理一次；RPC 超时后的清理确认不覆盖发送结果；旧 member 回调不污染新路由；部分成员提交失败后预测与实际批次一致；DIRECT 和队列路由发布仍保持先结束路由操作、后推进响应的顺序。自定义 scheduler、策略和出入队机制保留，共享的是交接与结果协议。

## 23. 投递结果单入口实现与验证

本次先新增 DeliveryReportingBoundaryTest，在修改前确认两个用例失败，再修改生产代码。
生产代码现在只有 DeliveryDispatcher.report 中的一处 onDeliveryResult 调用。BATCH 的
TransportResult 和 DIRECT/排队返回路由的 RouteDelivered 均从这里上报；接收入口拒绝
Prediction、PreparationFailure、CleanupAcknowledgement 等非完成事件。

预测仍在 RPC 前通过 recordDeliveryPrediction 登记，保留部分成员提交失败后的批次重新预测。
本次没有把预测合并进 commitDelivery 的参数，避免同时改变预测时点与锁范围。
准备失败使用 recordPreparationFailure；取消清理确认使用 recordRemoteCleanup。
这些入口共享私有的状态记录和锁外推进逻辑，抢占的原子状态更新保持不变。

资源提交仍由 Scheduler 在请求锁内验证和执行；Dispatcher 接管提交异常，根据是否登记了
精确 member 报告未发送结果或准备失败。外层不再重复发准备失败通知。DIRECT 仍在结束路由
操作后交给 Dispatcher 发布结果。调用方身份和参数校验失败必须不改变原请求状态，不能按
业务投递失败结束请求；原有的 foreign scheduler、错误 permit、非法 batch ID 三个 UT
发现过这一回归，现已修复，保留原断言。

新增行为 UT 检查预测先于 RPC 且只上报一次发送结果、RPC 调用抛异常时只报告 UNCERTAIN、
资源提交失败时只通知一次准备失败且只释放一次 permit、不调用 RPC。原有模拟器与测试
改为从对应的预测、准备失败、清理入口注入事件，不再向投递完成入口注入其他阶段的事件。
字节码边界 UT 扫描 org.flexlb.balance 的生产类，要求唯一调用点位于 DeliveryDispatcher.report。

隔离源码副本上的 564 个定向回归用例通过，覆盖 DIRECT、BATCH、路由发布、请求生命周期、
取消与响应发布竞争、超时和抢占等。工作区完整 Maven reactor Spotless 检查通过。
隔离构建用于避免其他构建清理共享 target，不是线上运行或性能基准。

最后收尾后重跑 252 个相关用例全部通过，其中 DeliveryClaimTest 的 106 个用例覆盖
发送身份、取消确认和晚到回调。两轮去重合计 696 个用例，最终报告无失败、无错误。
本次核心类与新增 UT 的成员顺序检查、git diff --check 通过；最终完整 reactor
Spotless 再次通过。没有提交或暂存工作区改动。

## 24. AbstractRequestScheduler 全方法复核：private 数量背后的流程复杂度

本节基于本轮读取的 1921 行源码；工作区仍有其他修改，以方法名定位。只评审，不修改 Java。
用仓库 Java AST 统计外层类的方法，不计构造器和内部 record：68 个，其中 public 24 个 / 417 行，
protected 9 个 / 143 行，包级 17 个 / 533 行，private 18 个 / 586 行。行数从方法声明到结束括号，
包含签名换行与方法内部空行，不是可执行语句数，也不是预估可删除行数。

### 24.1 结论与实际主流程

private 不是主要问题：只有 18 个，很多是合理的独立操作。真正的问题是请求结束与响应发布的
协调协议分布在 public、包级、private 三组里；仅盯 private 会漏掉 81 行 completeRouting、
82 行 tryCompleteFromExternal、67 行 closeRequest 和 72 行 commitDelivery。

正常 BATCH 返回路径：Dispatcher.report → onDeliveryResult → recordDeliveryProgress →
processPendingRequestActions → confirmDelivery → publishResponse → Registration.submit →
响应线程再次 selectResponse → 完成 Future。Worker 完成或取消随后仍需独立清理。

结束路径：cancel / deadline / worker / delivery failure → closeRequest → submitClose →
RequestCloseQueue → runClose → cleanupRequest → prepareRequestEnd / releaseRouteResources /
completeRequestEndNotifications → member.complete → repository.archive → timer.remove。
关闭期间等待远端证明时，回调重新唤醒这条路径；一次调用不一定能清理完。

这两条链存在必要的锁外执行、线程交接和晚到回调。应消除重复决策及重复异常恢复，不能简单
把异步结束写成顺序释放后返回，也不能通过新建 LifecycleManager 把整条链搬走。

### 24.2 四项需要优先缩减的逻辑

**A. 响应执行票据的处理重复出现在多个业务入口。**

confirmDelivery、closeRequest、tryCompleteFromExternal 都创建 Registration，再创建
ResponseCandidate，然后分别处理未提交、提交异常、关闭票据。submitClose 和 runClose 又
分别接收这组参数，再次承担 publishFailure / close。publishResponse、publishFailure、
publishUnsubmittedResponse 不是三个业务阶段，而是这套多处管理协议产生的辅助函数。

Registration 已经有 RESERVED/SUBMITTED/RUNNING/COMPLETED/CLOSED 的原子转换、执行器拒绝
收尾和计数释放。调用方仍必须检查 isUnsubmitted 并尝试异常补发，说明票据只封装了执行阶段，
没有完整封装“业务已预留响应，但构造或交接尚未成功”的失败责任。

建议在现有 Registration 内完善绑定后的响应执行与未交接失败处理；业务方负责决定 ACK、
取消、终态和竞争规则，执行器负责接纳、运行、拒绝与计数释放。不能把业务仲裁移进线程池。
成功交接给 close lane 后，lane 是唯一票据关闭责任方；未交接的票据由当前调用方结束。
目标删除 publishFailure / publishUnsubmittedResponse 及多个重复的 fallback/finally，
不是新建一个类照搬这些分支。必须保留先 register 再冻结结果、锁外用户回调、异步 ACK 在
执行时竞争、某些失败提前选定 candidate 的差异，不能把 selectResponse 全部前移。

**B. 收尾执行策略被反复推导，基础设施故障与业务清理缠在一起。**

submitClose 用 outcome/RETIREMENT/BATCH 推导 local，再决定 inline、提前响应、先移除队列；
runClose 又用同一组条件推导 releaseBeforeResponse。两处读取时间不同，不可直接删掉后者：
排队期间状态可能改变。应先把“执行线程”“响应与资源清理的顺序”作为两个明确规则，消除
同义谓词 local/releaseBeforeResponse，保留入队时和执行时必要的校验。

cleanupRequest → prepareRequestEnd → completeRequestEndNotifications 后，cleanupRequest
还会再次调用 completeRequestEndNotifications。第二次调用有时能观察资源释放期间完成的
抢占，因此不能仅凭方法名相同判断为无意义。改为显式的“启动停止操作 → 尝试资源释放 →
完成结束通知”，完成通知放在这些操作之后一次执行；以通知竞争 UT 证明不丢失唤醒。

三次循环不是可随手删除的死代码：RequestTerminalSettlementTest 明确断言一次唤醒最多三次
尝试，并测试 timer 故障后资源仅释放一次。当前 timer 安装失败会挡住 notificationsDone，
进而保留请求；需要先明确 timer 安装失败是独立基础设施错误还是归档前置条件，再把重试缩小
到失败的步骤。保留原有重试上限和后续唤醒，不能为了缩短代码直接放弃重试或提前归档。

**C. 事件入口拆开了，内部仍使用混合事件再分发。**

上一轮保留 report 唯一入口是有效改动，但 onDeliveryResult、recordDeliveryPrediction、
recordPreparationFailure、recordRemoteCleanup 最终仍进入 53 行 recordDeliveryProgress。
其中不仅记结果，还做未发送路由失败判断、读取 Decode 资源、生成 ACTIVE 事件、更新状态、
推进后续动作。所以外部入口清楚了，内部复杂度没有等比例减少。

建议预测入口只处理预测及与预测开始有关的 Decode 确认；清理确认只合并清理证明；没有
delivery 的准备失败直接按精确 route 做撤回或结束。投递完成仅处理 RouteDelivered /
TransportResult。共用身份校验和锁外推进，不再把四种输入重新塞进同一个总分发函数。
这样删除的是 instanceof 分支及 nullable claim 的反推，而不是把原函数复制四份。
当前锁外读取 Decode 确认后、锁内重新验证 member/route 的两次检查保护真实竞争，不能删。

**D. 一个路由方法同时接受多种运行方式，通过 null 推断调用模式。**

executeRouting 同时负责选择、异步抢占续作、DIRECT/QUEUE 交付选择、资源清理、重试结果
解释和 DIRECT 错误映射。queue==null、mayPublish==null、recover==null 分别传递运行方式，
routeAccepted/delivered/continuation 又分别表达不同交接结果。它们不都是重复变量：
已交给队列和已准备返回路由确实不同；只用一个 success 会丢失失败后的资源责任。

建议保留公共的精确 RoutingContext、资源未交接回滚和“结束 routing 后再发布响应”协议；
DIRECT 的 BLOCKED→REJECTED 错误策略在 DirectRequestScheduler 的决策边界处理，QUEUE 的
续作与等待由 QueuedRequestScheduler 明确提供。用已有模式配置/明确调用参数表达交付方式，
不要依赖 queue 是否为 null 推断。只有重复的结果解释和二次错误封装真的删除，才算减量；
增加一个策略接口再原封不动搬分支不能算完成。自定义 scheduler、选择策略、队列扩展必须保留。

### 24.3 每个 private 方法的判断

| 方法                            | 行数 | 判断与具体处理方向                                                                         |
| ------------------------------- | ---: | ------------------------------------------------------------------------------------------ |
| recordDeliveryProgress          |   53 | 过多职责；按上面的事件语义删除混合分发，不复制四份通用流程                                 |
| onRouteFailure                  |   11 | 保留；组合请求归属、重试策略和原子标记，是合理业务操作                                     |
| withdrawFailedRoute             |    6 | 保留；获取精确撤回操作后完成，阻止重复撤回，不必为了少一个方法内联                         |
| processPendingRequestActions    |   42 | 保留统一推进点；明确撤回→抢占协调→关闭/确认→期限同步的优先级；精简各入口的重复推进         |
| confirmDelivery                 |   70 | 明显可缩；混合失活判定、确认状态、输入释放、计时、响应票据和异常补发。优先消除响应票据模板 |
| recordDeadline                  |   34 | 大部分必要；精确 token 和到期检查不能去掉。诊断文本应只在真正接受超时时构造                |
| publishResponse                 |   35 | 业务响应仲裁有必要；票据执行/失败处理应完整归属已有 Registration，监控不能影响响应         |
| publishFailure                  |    6 | 响应票据协议完善后删除；当前不是独立业务步骤                                               |
| publishUnsubmittedResponse      |   12 | 同上；未交接票据的失败完成不应由多个业务方法分别调用补救                                   |
| executeRouting                  |   91 | 过载；保留一次路由的资源交接骨架，删除模式推断和重复结果解释，见 D                         |
| commitEncoderRoute              |   33 | 保留 Encoder 特有容量与输入语义；与 Prefill 强行统一会制造更多 optional 字段               |
| needsFollowUpActions            |    7 | 保留；过滤无须推进的高频 Worker 更新，比盲目调度每个事件更省成本                           |
| submitClose                     |   61 | 重点缩减；混合线程选择、移除队列、提前响应、lane 提交、拒绝与票据回收，见 A/B              |
| runClose                        |   35 | 执行单个 close 任务有必要；删除重复票据失败兜底，统一响应/释放顺序规则                     |
| cleanupRequest                  |   34 | 资源清理与归档必要；按失败步骤重试，不能机械删三次循环                                     |
| prepareRequestEnd               |   20 | 名称掩盖 timer 更新、stop 发起和结束通知三个动作；应消除内部提前调用结束通知               |
| completeRequestEndNotifications |   24 | 抢占结束通知必要；与 stop 通知是不同事实，不能把 notificationsDone 当“所有远端已清理”      |
| response                        |   12 | 保留；短小明确的终态响应映射，无须再拆                                                     |

### 24.4 其余全部方法复核

同名重载按参数区分，以下包含剩余 50 个外层方法。

| 方法                                             | 行数 | 判断                                                                                                                                                    |
| ------------------------------------------------ | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| submit(ctx)                                      |    3 | 合理的默认回调重载                                                                                                                                      |
| submit(ctx,onRegistered)                         |  105 | 入口流程有必要；校验、登记、安装定时器、策略交接和响应观察集中。先削减响应失败模板，不能只拆校验函数凑短                                                |
| cancel(id,batchId,reason)                        |    3 | 合理的默认 Generation 重载                                                                                                                              |
| cancel(id,batchId,reason,phase)                  |   20 | 保留活动/归档查询和所属 scheduler 校验                                                                                                                  |
| cancel(ctx,batchId,reason)                       |   47 | 取消原因冻结和锁外清理必要；期限更新责任与公共推进点应统一                                                                                              |
| recordRequestFailure                             |    3 | 保留单一异常累积点                                                                                                                                      |
| publish                                          |    4 | 外部 Future 仲裁适配，名称可以更准确，但不是长度问题                                                                                                    |
| isAdmissionOpen                                  |   11 | 保留精确请求接纳判断；不能与路由可发布条件混同                                                                                                          |
| currentRoutingContext                            |   12 | 保留精确路由操作查询与校验                                                                                                                              |
| selectAndPublish(ctx,selection,queue)            |    9 | 扩展入口有必要；queue null 的模式表达需要澄清                                                                                                           |
| finishRouting                                    |    7 | 正常选择与恢复旧路由不同；保留语义，不合成含糊布尔值                                                                                                    |
| completeRouting(routingContext,result)           |   12 | 精确 owner 转发；可审视静态入口与实例入口双重契约，但不是优先减量点                                                                                     |
| onDeliveryResult                                 |    8 | 保留唯一投递结果消费入口；内部混合分发仍需改进                                                                                                          |
| recordRemoteCleanup                              |    3 | 清理语义入口必要；目前只是通用事件包装，见 C                                                                                                            |
| failRouting                                      |   10 | 失败响应合法性和所属 scheduler 转发必要                                                                                                                 |
| onRequestStateChanged                            |    8 | 保留状态推进入口；避免和其他入口重复做同一轮推进                                                                                                        |
| isCurrentRequest                                 |    3 | 精确身份判断，保留                                                                                                                                      |
| onWorkerRequestEvent                             |   19 | 保留精确事件记录、是否需要推进的过滤和锁外推进                                                                                                          |
| onEndpointRetired                                |   23 | 未发送可重路由、已交付需清理的差异真实存在，保留                                                                                                        |
| onRouteFailure(route,reason,detail)              |    4 | 可读的业务入口，保留                                                                                                                                    |
| rejectRoute                                      |   22 | 队列拒绝有重试与结束两个结果；保留策略边界，减少重复错误文本构造即可                                                                                    |
| onDeadline                                       |   24 | 旧 token 与提前触发保护必要；用户回调不能占住 timer 线程                                                                                                |
| shutdown                                         |   49 | 停接纳、停止生产者、等回调、清资源、关执行器顺序有必要；不是“都 close 一遍”可替代                                                                       |
| checkShutdownThread                              |    8 | 保留，防止响应/清理线程等待自己                                                                                                                         |
| planRoute                                        |   11 | QUEUE 续作扩展必要，不能与同步 DIRECT 无条件合并                                                                                                        |
| tryBeginRouteSelection                           |   11 | 身份与原子创建路由操作必要                                                                                                                              |
| resumeRoute                                      |   46 | 晚到抢占结果不能泄漏 reservation；与 executeRouting 的取消检查有不同时间边界，不可盲删                                                                  |
| selectAndPublish(routingContext,selection,queue) |    6 | 已有操作上的扩展入口，保留                                                                                                                              |
| schedulingTimeoutError                           |    1 | 策略扩展点，保留                                                                                                                                        |
| encoderRequestLimit                              |    3 | Encoder 容量策略扩展点，保留                                                                                                                            |
| describeWait                                     |   54 | 业务诊断应保留，但字符串拼接占主流程阅读成本；只有统一诊断字段与重复消息后才算减少逻辑                                                                  |
| checkRequestMode                                 |   10 | 调度器与请求配置匹配检查，保留                                                                                                                          |
| removeRequest                                    |    1 | 自定义出队/取消清理必须保留的扩展点                                                                                                                     |
| recordDeliveryPrediction                         |    3 | 预测入口必要；可消除进入通用事件总分发，见 C                                                                                                            |
| recordPreparationFailure                         |    3 | 准备失败入口必要；无 member 的失败应走精确 route 失败流程                                                                                               |
| throwIfRequestFailed                             |    3 | shutdown 的失败出口，保留                                                                                                                               |
| finishUnpublishedRoute                           |   24 | 未交接资源的回滚与路由操作完成有先后关系；回滚失败必须保留清理记录                                                                                      |
| releaseRouteResources                            |   33 | 统一资源释放有价值；Encoder/Prefill 与 Decode 的独立成功记录不能合并成一个 released                                                                     |
| enqueueRoute                                     |   29 | 物理入队后记录发布的交接必要；队列拒绝诊断不应成为另一套路由决策                                                                                        |
| notifyQueuedRoute                                |   15 | 发布后唤醒与监控，不得回滚已接管请求；保留独立操作                                                                                                      |
| commitDirectRoute                                |   71 | 获取临时 permit、Prefill 提交、预测、请求提交及清理混合；资源作用域不可删除，响应错误构造可统一                                                         |
| commitDelivery(member,kind,batchId)              |    4 | 当前是 Scheduler→Dispatcher→Scheduler 四参数方法的回跳；应让生产 sender 直接调用 Dispatcher，保留实际请求锁内提交方法和归属校验，删除这层回跳及测试依赖 |
| preparationFailed                                |    7 | 只是由 route 找 scheduler 后构造失败；事件语义明确后可由 Dispatcher 的准备失败出口直接处理                                                              |
| completeRouting(routingContext,result,failure)   |   81 | 同时做撤回、资源释放、完成状态、重入调度及失败收尾；将“选择完成”和“撤回完成”明确为不同执行分支，统一最终推进，不复制清理                                |
| closeRequest                                     |   67 | 结束判定与响应票据交接混合，优先按 A/B 减量                                                                                                             |
| scheduleClose                                    |   12 | 无响应的清理唤醒与携带响应的 close 不同；可保留明确入口，不能丢 close lane 合并语义                                                                     |
| tryCompleteFromExternal                          |   82 | 外部 Future 不能伪造成功、不能覆盖已交付结果；约束保留，响应执行/回收模板应消除                                                                         |
| closeOutstandingAndTerminalize                   |   19 | 最终关闭兜底；不能把它当成前面丢清理责任的通用补偿                                                                                                      |
| closeExecutors                                   |    8 | 生产者、stop 回调和响应执行器的关闭顺序必要                                                                                                             |
| commitDelivery(member,kind,batchId,selecting)    |   72 | 请求锁内的容量交接必须集中；把 DIRECT 与已有 route 两种合法前置条件说清，不能将整个操作移出请求锁                                                       |

### 24.5 建议实施顺序与验收

1. 先保存改动前源码快照，按生产代码、测试、文档分别计算净变化，不能拿包含其他工作区改动的 HEAD diff 充当节省量。
2. 先做小而可证的删除：去掉三参数 commitDelivery 回跳；统一同义的收尾顺序判断；保留精确身份 UT。不能只删除测试中的归属保护断言。
3. 再完善现有 Registration 的失败与交接协议，删除两个响应补发 helper 以及业务入口重复的票据收尾。以 ResponseCompletionExecutorTest、RequestPublicationFailureOwnershipTest、RequestCompletionPublicationRaceTest、RequestCancellationFailureOrderingTest 验证拒绝、重入、晚到 ACK 和取消竞争。
4. 最后处理 cleanup 的步骤和重试边界，使用 RequestTerminalSettlementTest、RequestCleanupPassTest、RequestNotificationOwnershipRaceTest、RequestReconciliationFailureOwnershipTest 保持部分成功单调、每步最多一次成功释放和后续唤醒。
5. 每一轮同时比较类内与全项目生产代码净行数、状态数量、交接次数和异常恢复入口数。不能承诺尚未实现的减量数字，也不以 private 数量或每个方法低于某行数作为验收标准。

本轮只做源码和已有测试契约审核，没有修改生产代码，也没有将“读过已有 UT”说成本轮运行过 UT。

## 25. 三项内部流程调整：先 UT，再实现

本轮先保存完整源码副本作为独立基线，随后增加响应交接、旧事件隔离和跨唤醒重试用例。
修改生产代码之前运行 ResponseCompletionExecutorTest、RequestDeliveryClaimStateTest 和
RequestTerminalSettlementTest，64 个用例通过。测试先明确以下边界：提交后 close 不能
提前释放响应计数；响应动作失败只能释放一次；预测和清理不能伪造发送完成；旧 member
不能修改当前请求；timer 失败超过三次时仍由后续唤醒继续，资源只成功释放一次。

### 已实施的变化

1. Registration 必须绑定 Scheduler 提供的响应仲裁回调，完成响应时再调用它，保留晚到
   ACK 与终态的竞争规则。未提交失败与补发资格判断由 Registration 承担，删除 Scheduler
   的 publishFailure / publishUnsubmittedResponse。原始外部 Future 完成保留同步返回语义。
   没有新增响应管理类，也没有把业务选择规则搬进执行器。reserveResponse 只负责将候选
   与执行票据绑定；已交接票据的计数仍由实际执行者释放。
2. 删除 prepareRequestEnd 的混合职责。每轮先同步期限、发起仍需发起的 stop，再尝试释放，
   最后完成结束通知；不再在资源释放之前提前完成一轮通知。停止操作按已有 attempted
   事实去重，成功释放状态保留。submitClose/runClose 使用同一个 releaseBeforeResponse
   规则，仍分别读取当时状态。保留每轮最多三次尝试及失败记录，没有擅自移除 timer 安装
   对结束通知的约束，也没有把远端取消变成同步等待。
3. 删除 Scheduler.recordDeliveryProgress 和 DeliveryFacts.record 的混合事件分发。
   RequestContext 分别记录预测、远端清理、准备失败和发送完成。Scheduler 共用的仅是
   精确身份校验、锁内状态记录与锁外推进；预测及路由交付共用 Decode 确认读取，并保留
   锁外读取后的锁内身份复核。抢占清理改为直接调用 recordRemoteCleanup，原子区不变。

新增响应恢复测试还验证了：未提交失败在请求锁外仲裁；提交后的成功响应不能被晚到失败
替换；失败候选失去仲裁时不写 Future，但必须释放执行计数。一个手工构建未绑定票据的
旧测试夹具暴露了绕过仲裁的问题，现按生产协议绑定候选，并删除默认放行的三参数构造器；
保留“终态压过排队 ACK”的原断言。

### 代码量与范围

相对本轮独立基线，格式化后生产文件净变化如下（包含空行与注释）：

| 文件                       | 修改前 | 修改后 | 净变化 |
| -------------------------- | -----: | -----: | -----: |
| AbstractRequestScheduler   |   1921 |   1897 |    -24 |
| RequestContext             |   1120 |   1158 |    +38 |
| ContextResult              |    694 |    658 |    -36 |
| ResponseCompletionExecutor |    303 |    330 |    +27 |
| PreemptionCompletion       |    303 |    303 |      0 |
| 合计                       |   4341 |   4346 |     +5 |

这轮消除了混合分发和重复补发入口，但没有实现全项目生产代码净减量。不能把
AbstractRequestScheduler 减少的 24 行单独宣传成总体节省。新的类型明确入口及票据仲裁
绑定也有代码成本；响应票据的成功路径交接、清理预算和异步唤醒机制仍然存在，不能说
所有生命周期复杂度已经消失。测试代码本轮增加了行为覆盖，并迁移了旧事件注入入口。

### 验证结果

隔离副本中，相关测试套件去重合计 666 个用例通过，0 失败、0 错误；最后移除无仲裁回调
的构造器后，重跑其中 143 个响应执行、响应竞争、收尾、状态与架构用例通过。
工作区完整 Maven reactor Spotless 检查、核心类及新增测试的成员顺序检查、git diff --check
通过。没有运行性能基准，没有暂存或提交。独立基线位于临时目录
`/var/folders/hp/trr4w7bx56bctwg22ymmsph00000gn/T/scheduler-simplify-6viomflg/baseline`。

## 26. 继续收敛响应任务交接，而非只调整入口

本轮重新保存源码基线，避免把期间其他路由重构计入本轮收益。当前改动只涉及
AbstractRequestScheduler、ResponseCompletionExecutor 及对应测试和架构说明。

之前的 submitClose(ctx, publication, candidate, response) 与 runClose(ctx, publication,
candidate, response) 需要调用方一直携带响应执行资格、业务候选和响应值，并在多层捕获异常、
补发与关闭。现在响应构造完成后调用 publication.prepare(completion, response, failure)，
随后只交接这个已经包含响应动作的对象：submitClose(ctx, publication) → runClose(ctx,
publication) → publication.publish(false)。候选仲裁回调在预留时绑定，不再向下传递候选。

具体删除项：

- 删除 Scheduler.publishResponse，响应执行与关闭收敛到 Registration.publish。
- 删除 Registration.completeIfUnsubmitted 通用补发接口及外部 Future 完成路径的重复补发。
- 删除 closeRequest.handedOff：准备阶段的 try/catch 在交给 submitClose 之前结束，责任边界
  由调用结构表达。submitClose 接管后，调用方不再尝试修复已交接任务。
- 删除 submitClose/runClose 的 candidate、response 参数和多层 finally close；提前响应、
  排队响应均调用同一票据的 publish，队列拒绝统一调用 failIfUnsubmitted。
- runClose 仍保证响应发布失败不阻断资源清理，资源清理失败也不阻断已经确定的响应。

Registration 增加一个 preparedResponse 动作，复用原有 RESERVED/SUBMITTED/RUNNING 等
阶段，不新增请求生命周期状态或全局任务集合。响应构造保留在原调用线程，防止把
closeRequest 的同步构造异常悄悄改为异步；真正的 Future 回调仍在请求锁外执行。
异步 ACK 在执行时参加仲裁，外部 Future 完成保留同步 boolean 返回值。Registration.prepare
只能调用一次，不能覆盖已经准备的响应。

先增加收尾通知合并不得丢响应、失败补发被执行器拒绝仍须完成 Future 并释放计数的 UT，
与现有响应竞争及失败用例一起在修改前运行通过。随后增加完整响应任务的行为测试：
仅在实际执行时仲裁、准备后不可换成另一种响应、同步仲裁失败返回 false、拒绝后无需调用方
额外 close。旧反射调用 publishResponse 的测试改用真实的 prepare/publish 入口，保留
终态必须压过排队 ACK 的断言。

以本轮独立基线、统一格式化后的实际行数计算：

| 生产文件                   | 修改前 | 修改后 |  净变化 |
| -------------------------- | -----: | -----: | ------: |
| AbstractRequestScheduler   |   1482 |   1403 |     -79 |
| ResponseCompletionExecutor |    330 |    346 |     +16 |
| 合计                       |   1812 |   1749 | **-63** |

本轮没有将代码迁移到新的生产文件。这里的基线是开始本轮时的工作区，不是 HEAD，也不是
上一轮的 1897 行 Scheduler；中间其他修改不计入本轮减量。基线保存在
`/var/folders/hp/trr4w7bx56bctwg22ymmsph00000gn/T/response-task-8acp40g7/baseline`。

验证结果：隔离源码副本中运行 26 个测试类、690 个测试，失败、错误及跳过均为 0。
覆盖响应竞争、取消、交付、资源清理、超时、抢占、路由和自定义 Scheduler 契约。
工作区全 Maven reactor 的 `./mvnw spotless:check -Pspotless-check` 通过，
成员顺序检查及 `git diff --check` 通过。生产代码中 `onDeliveryResult` 的调用仍只有
DeliveryDispatcher.report 一处。本轮未进行性能基准测试。
