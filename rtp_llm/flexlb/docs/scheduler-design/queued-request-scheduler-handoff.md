# QueuedRequestScheduler 开发实施方案

状态：实施中。队列扫描、统一执行额度、通知精简和 Decode ResourceWait 已接入；
RequestContext 生命周期、EndpointRegistry 关闭及 AbstractRequestScheduler 的选路/取消主干已重写。
未提交 P/D 选择的抢占、角色策略生成计划尚未完成，不能将当前代码视为整份方案验收通过。
本文是本次队列调度、容量通知和 P/D 抢占改造的单一实施依据。
接口为目标接口，代码为说明原子边界和控制流的伪代码；不代表当前实现已经具备这些行为。

## 1. 要实现的行为

- 有通知就尽快检查；没有通知时，排队请求按固定周期重新检查本地容量。
- 多线程选路，每次最多提交“并发上限减去未结束执行任务数”个任务。
- 队列按归一化优先级、原始入队序号排序；恢复入队不改变序号或请求截止时间。
- 保留角色/分组的容量等待与单个重试限制；普通通知广播检查，不维护逐请求通知索引。
- 未发送的 Prefill、Decode 选择都允许被抢占；已发送 Decode 按现有取消协议处理。
- A 抢占低优先级 B 后，B 不能立刻抢回目标资源；更高优先级 C 可以使用该资源。
- 资源不专属于 A；不保证并发选路的完成顺序严格等于队列顺序。
- A 等待不占线程，最多等到原请求的绝对截止时间。取消或超时必须解除目标 Endpoint 限制。
- 资源结果到达晚于取消时，取得方或接收方必须回滚精确登记；不能只结束 Future。

不新增已发送 Prefill、Encoder、VIT 抢占能力。PDFUSION 的本地未提交资源按其 Prefill
Endpoint 契约覆盖；其已提交请求不能直接套用独立 Decode 的取消/释放规则。

## 2. 最终类组织与删除清单

### 2.1 各类只负责一个明确范围

| 类                                    | 最终责任                                                         |
| ------------------------------------- | ---------------------------------------------------------------- |
| QueuedRequestScheduler                | 入队、通知/定时检查、并发额度、任务投递、结果处理和关闭          |
| SchedulingQueue / OrderedRequestQueue | 排队成员、固定排序、有限扫描、精确删除                           |
| CapacityWaitQueue                     | 按角色/分组暂停容量不足请求；每域一个重试，成功后继续放行        |
| RequestContext                        | 请求身份、截止时间、当前选路编号、取消/结束和提交事实            |
| RequestRouteSelector                  | 组合各角色选择；部分失败回滚；传递计划；采用抢占取得的角色登记   |
| 各角色具体 Strategy                   | 正常选路；容量不足时生成本角色可执行的抢占计划                   |
| DecodePreemptionPlanner               | Decode slot/KV、候选组合和优先级代价的具体计算；无接口、无副作用 |
| RequestPreemption                     | 唯一抢占执行扩展接口                                             |
| DefaultRequestPreemption              | 执行计划，管理等待安装、目标检查、取消/到期和结果处理完成        |
| WorkerEndpoint 及 P/D 实现            | 本角色资源账本、优先级接纳检查、精确等待登记、资源获取           |
| CapacityNotifier                      | 锁外广播 Endpoint 条件变化；仅管理订阅，不保存容量或等待历史     |
| ExpirationTimer                       | 原有请求截止时间的权威触发入口                                   |

### 2.2 删除什么，原来的工作由谁承担

| 删除项                                                                           | 原作用                               | 替代方式                                                       |
| -------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------- |
| CapacityWaitQueue 的通知历史依赖                                                 | 避免等待登记前后丢失变化             | 保留容量域暂停、单个重试和成功后的继续放行；广播与周期触发检查 |
| GlobalQueueEntry 公共类                                                          | 请求包装、链表节点、排队身份混合     | 对外使用 RequestContext；链表节点私有，任务身份独立            |
| SchedulingRequests 整类                                                          | 重复排队登记、执行/抢占混合工作跟踪  | 队列独占成员关系；Scheduler 跟踪执行；抢占组件跟踪等待         |
| preemptionQuotaWaiters                                                           | 哪些请求等本地抢占名额               | 名额释放触发队列检查；无需第二集合                             |
| OrderedRequestQueue.readyRetries、markRequestReadyForRetry                       | 单个请求重新进入扫描                 | 统一请求扫描和有限游标                                         |
| CapacityNotifier.sequence/lastChanged/forgottenThrough                           | 处理等待登记与容量通知交错           | 广播触发等待域检查；固定周期兜底                               |
| removeClosedWorker/discardHistory                                                | 清理上述通知历史                     | 无历史可清理；仍保留 Endpoint 真正下线清理                     |
| Queue.notifications 原子序号和 park/unpark                                       | 防止休眠窗口丢通知                   | Condition 唤醒与固定周期兜底                                   |
| RequestScheduler.onEndpointChanged、Direct 空实现                                | 为 Queue 暴露通知入口                | Queue 自行订阅 CapacityNotifier                                |
| DecodePreemptionPolicy 接口及 Bean                                               | 可替换的计划生成层                   | 角色具体 Strategy 生成计划，Decode 使用具体 Planner            |
| DefaultDecodePreemption/DecodePreemption 旧类型                                  | Decode 执行入口                      | 替换为统一 RequestPreemption 执行接口与默认实现                |
| RequestResult.preemptionContinuation、reservation、controlFailure 的抢占控制用途 | 把执行闭包和 Decode 资源塞入通用结果 | 计划是不可变数据，资源 Future 由等待记录管理                   |
| 父类 startOrWaitForDecodePreemption、allowDecodePreemption 参数                  | 父类驱动特定抢占流程                 | Queue 根据策略返回计划启动抢占                                 |

保留 CapacityNotifier 的原因：多个 Endpoint 生产变化，Queue/抢占消费变化，生命周期需要
订阅和注销；它只留下 add/remove/notify 三类操作。禁止重新加入请求、域版本、历史和 ready 状态。
保留 RequestPreemption 执行接口的原因：当前 Spring 存在实际替换入口；只保留这一处执行扩展，
不再建立抽象基类或每角色执行接口。删除计划策略接口是本次明确的扩展契约调整，迁移说明需列出。

WorkerRequestEvent、EngineRequestCanceller 和请求清理队列服务于请求执行事实、远端取消及清理，
不是容量唤醒机制，不能按 notify 关键词删除。保留这些责任不表示其所有包装都无需审核。

## 3. 最少状态及唯一来源

| 事实                             | 唯一来源                                   | 何时删除                                     |
| -------------------------------- | ------------------------------------------ | -------------------------------------------- |
| 哪些请求在排队、排在何处         | SchedulingQueue                            | 成功、拒绝、取消或关闭                       |
| 哪些任务已接纳但未执行/清理完    | Queue.executing                            | 任务 finally 完成结果处理和回滚后            |
| A现在是否仍需要目标资源          | RequestContext.currentResourceWait精确关联 | 取得资源、放弃、取消、超时或目标失效时清除   |
| 哪些抢占工作尚未清理完           | 抢占组件Pending持有同一个等待对象          | 安装、检查、结果采用/回滚全部结束后          |
| 哪些等待限制 Endpoint 接纳优先级 | Endpoint 中同一 ResourceWait 对象的索引    | 已取得资源、取消、超时、放弃目标或下线       |
| 本次请求/选路是否仍有效          | RequestContext                             | 按已有请求生命周期更新                       |
| 本次检查扫描到哪里               | 队列游标                                   | 本次扫完后等待；通知唤醒或周期到达后重新检查 |

A、Endpoint索引、Pending引用同一个ResourceWait；A的关联决定等待是否仍有效。
Endpoint索引只用于找到等待，不作为A仍在等待的证明。
Pending 引用 Endpoint 的 ResourceWait，不复制 requestId、priority、deadline 或 Future；
Pending 只补充上层精确 selection 和必要的检查/交接工作信息。Endpoint 不引用 balance 包。
执行记录不复制队列节点。取消可以移出队列，但不能删除仍可能返回资源的执行/等待记录。

## 4. 调度时机与执行额度

### 4.1 事件行为

| 触发                                             | 动作                                               |
| ------------------------------------------------ | -------------------------------------------------- |
| 新请求/恢复请求入队                              | 请求扫描并唤醒                                     |
| Worker 状态/容量/可用性变化                      | 请求扫描；对应目标的抢占等待请求检查               |
| 抢占撤销释放、已执行请求释放、有效优先级限制解除 | 同上                                               |
| 选路任务结束                                     | 归还执行空位，继续未扫描部分；不自行重试刚失败请求 |
| 固定检查周期到达                                 | 请求扫描并检查仍未结束的抢占等待                   |
| 抢占 Future 完成                                 | 安排采用或回滚结果，经过同一执行额度               |
| 取消/超时/关闭                                   | 结束自身等待、安排清理，保证旧工作有执行者         |

普通选路失败的临时资源回滚不广播普通重试通知，避免“失败→回滚→通知→再次失败”。
该释放仍更新真实账本，其他请求由固定周期发现。抢占主动撤销 B 属于实际调度进展，
撤销完成后发布一次通知。通知函数不需要 externalProgress 布尔参数或新增通知类别状态机，
区别由明确的调用位置表达。回滚是否通知不得改变回滚自身的资源正确性。

### 4.2 固定周期与扫描

增加 SchedulerConfig.capacityCheckIntervalMs，初始实现默认 100ms，正数校验。
100ms 是本方案的初始延迟/重试成本取值，性能验收须测量并可调，不声称已验证最优。
周期使用单调时钟；请求截止时间仍取 SchedulingMetadata.expiresAtMs，二者不可互相替代。
通知不推迟原定周期。等待采用有界 awaitNanos；即使没有排队请求，超时只检查空队列，
不执行选路。这里统一使用有界等待，避免锁外抢占待办通知与等待交错造成永久休眠。

一次扫描受候选数和检查条数限制，未扫完接着推进，不能每次都只看队首。
扫描中收到通知，不额外记住“还要再扫一次”，继续当前扫描；本次没发现的变化由固定周期兜底。
线程正在await时收到signal会提前醒来；通知本身不排队、不保存，不增加扫描请求标记。
持续通知时仍按完整有界扫描向后推进，不能让队尾饥饿。

每次投递的空位 = plannerCount - executing.size()。
普通选择、抢占短检查、取得资源后的接续都使用相同额度；远端等待不占额度。
另以 maxPendingPreemptions 限制抢占等待数量，初版沿用 plannerCount，安装过程中也占名额。

## 5. Queue 和 Scheduler 的接口与成员

### 5.1 SchedulingQueue / OrderedRequestQueue

```java
interface SchedulingQueue {
    void add(RequestContext request);
    void restore(RequestContext request);
    boolean contains(RequestContext request);
    boolean remove(RequestContext request);
    void beginScan();
    void retry(RequestContext request); // 从被允许重试的成员位置恢复检查，不改变顺序
    List<RequestContext> scan(int limit, int scanBudget, Predicate<RequestContext> eligible);
    boolean hasUnscannedRequests();
    List<RequestContext> drain();
    // size/priorityCounts/order 等真实指标与排序能力保留。
}
```

OrderedRequestQueue 的私有 Node 只含 request、previous、next；队列维护 Context→Node 查找，
取代原 SchedulingRequests.registered。固定序号权威值放在请求已有调度信息中，首次入队赋值，
之后不允许随意修改。队列不在 Node 中再存一份独立可变序号。
原 routingGroup 是入队时确定的选路输入，移到既有调度元数据，保持冻结时机；不在重试时悄悄重算。

异步任务删除队列成员必须经过 Scheduler 的精确任务/selection 核对，不能直接调用 queue.remove。
同一 Context 在旧任务执行期间可以恢复排队，但不能启动新选路；旧任务退出仅移除自己的执行记录，
不能删除恢复后的排队成员。恢复路径先令旧 selection 失效，成功/失败提交队列变化时仍核对身份。

### 5.2 QueuedRequestScheduler 成员

```java
RequestRouteSelector router;
RequestPreemption preemption;
SchedulingQueue queue;
CapacityWaitQueue waitingRequests;
CapacityNotifier notifier;
ExecutorService planners;
int plannerCount;
double scanBudgetMultiplier;
long checkIntervalNanos;

ReentrantLock lock;
Condition changed;
Map<RequestContext, SchedulingTask> executing; // 精确任务对象；包含executor排队时间
boolean started;
boolean stopping;
long nextCheckNanos;
Thread schedulerThread;
ThreadLocal<Boolean> plannerThread; // 防止关闭时等待自己
```

保留 waitingRequests（CapacityWaitQueue）控制容量不足后的重试；它不保存资源余额。
没有 quotaWaiters、registered、第二个抢占完成 Future 或资源账本副本。
SchedulingTask 是私有的实际执行任务，不是新的跨组件请求包装。一次请求同时最多执行一个短任务；
等待安装过程和旧选择的退出由同一任务完成，资源结果接续不能与它并发执行。

```java
scheduleRequest(request) {
    检查模式和请求有效性;
    恢复场景先结束旧精确等待并使旧selection失效;
    lock();
    try { queue.addOrRestore(request); changed.signal(); }
    finally { unlock(); }
}

onCapacityChanged(endpoint) {
    preemption.requestCheck(endpoint); // 锁外，仅标记精确目标等待
    lock();
    try { waitingRequests.capacityChanged(); changed.signal(); }
    finally { unlock(); }
}

run() {
    while (true) {
        // 锁外处理固定周期待办、获取抢占快照；通知交错由有界等待兜底。
        if (nowNanos >= nextCheckNanos) {
            preemption.requestPeriodicCheck();
            更新nextCheckNanos到下一个固定周期;
        }
        waitingWork = preemption.readyWork();
        lock();
        try {
            if (本轮到达固定检查时间) { waitingRequests.capacityChanged(); queue.beginScan(); }
            slots = plannerCount - executing.size();
            waitingRequests.resumeReady(slots, 当前未执行的请求, queue::retry);
            // 普通候选排除仍在等待容量以及正在等待抢占的请求。
            tasks = chooseTasks(waitingWork, queue, slots, scanBudget);
            对tasks先登记executing;
            if (stopping && 无执行/等待/安装/结果清理) return;
            if (tasks为空) {
                if (slots > 0 && queue.hasUnscannedRequests()) continue;
                // 扫完或名额已满才等待；await原子释放锁，返回前重新取得锁。
                changed.awaitNanos(max(0, nextCheckNanos - nowNanos));
                continue; // 回到循环，重新读抢占待办、执行额度和请求状态
            }
        } finally { unlock(); }
        for (task : tasks) submit(task);
    }
}

submit(task) {
    try { planners.execute(task); }
    catch (failure) { 处理拒绝及已取得资源; finishTask(task); }
}
finishTask(task) {
    lock();
    try {
        executing.remove(task.request, task);
        if (queue.hasUnscannedRequests() || waitingRequests.hasReadyRequests() || stopping) changed.signal();
        // 本次已扫完时，普通任务结束不唤醒重复检查；等容量通知或周期。
        // 抢占结果完成有自己的signal，仍需及时消费或回滚。
    }
    finally { unlock(); }
}
```

队列首次add/restore必须让新成员进入可扫描范围。Condition允许虚假唤醒，醒来只重新检查，
不假定资源已到位；偶发额外检查不影响额度和身份约束。取消/关闭同样在锁下signal。

待办快照不是执行许可；任务执行前必须核对自己的精确等待/selection。
chooseTasks 只选已有数据并占执行名额，不在 Scheduler 锁内调用 Context、Endpoint、抢占组件或网络。
普通任务中 beginRouteSelection 是最后的防重复边界，已存在抢占等待时不能新开选择。
任务完成后的成功移除/恢复必须精确匹配任务身份；取消入口是有权操作当前请求的同步入口。

```java
selectRoute(task) {
    exact = tryBeginRouteSelection(task.request);
    if (exact == null) return;
    boolean handedToWait = false;
    try {
        result = executeRouteSelection(exact, () -> router.select(...));
        handedToWait = handleRoutingResult(task, exact, result);
    } finally {
        if (!handedToWait) finishSelectionIfUncommitted(exact);
        finishTask(task);
    }
}
handleRoutingResult(task, exact, result) {
    SUCCESS: 精确移除排队成员及容量等待，允许同域下一个请求检查;
    REJECTED: 发布失败并精确移除排队成员及容量等待;
    CLOSED: 按请求当前事实清理，不删除替代登记;
    RETRY: 保留队列，本次结束，固定周期重新检查;
    BLOCKED:
        保存诊断;
        if (result.preemptionPlan != null && preemption.start(exact, result.preemptionPlan)) {
            移除普通容量等待;
            return true;
        }
        waitingRequests.park(task.request, result.blocker); // Scheduler 锁内
    return false;
}
```

这里的 boolean 仅说明本次 selection 是否交给异步等待，不能同时表示请求成功或需要重试。
抢占接纳失败、计划失效和无额度都作为普通阻塞处理；不另登记 quota 等待。

## 6. 策略生成计划与结果类型

### 6.1 正常选择优先，生成计划不产生副作用

```text
角色Strategy先按现有策略选择/登记允许的候选
→ 成功返回RoleSelection
→ 容量不足且允许抢占：从资源快照生成PreemptionPlan
→ 无可行计划：普通BLOCKED

RequestRouteSelector遇到BLOCKED
→ 回滚此前其他角色的临时登记
→ 返回同一份计划给Queue
```

不能因为某个候选可抢占就跳过策略原本允许的其他正常候选；既有绑定/降级规则不变。
策略不发取消、不登记等待、不触发 Future。计划只是瞬时建议，执行处复核真实状态。

### 6.2 类型与接口

```java
// 单一计划值类型，替换DecodePreemptionPlan。
record PreemptionPlan(WorkerEndpoint endpoint, RoleType role, String group,
                      List<Target> targets) {}
// Target保存精确登记及计划时允许的动作阶段；已有资源视图可表达则直接复用。
// 必须能区分未提交撤销与已发送取消，不能只存requestId或地址。

DecodeStrategy.buildPreemptionPlan(request, snapshot, group);
CostBasedPrefillStrategy.buildPreemptionPlan(request, role, snapshot, group);
```

DecodeStrategy 组合使用具体 DecodePreemptionPlanner，保留 slot/KV 缺口、候选组合及优先级代价算法；
删除 DecodePreemptionPolicy 接口。Prefill 先使用具体策略私有函数，不预设对称的 Planner 接口/类。
原 PriorityDecodePreemptionPolicy.candidates 的 ownership 参数未参与判断、实际只筛 requiresEngineCancel，
迁移不能照旧注释宣称支持本地抢占；删除无效参数，实装未提交 P/D。

RoutingResult 增加可选 preemptionPlan，仅 BLOCKED 使用；RequestResult 传同一引用。
RoleSelection 始终表示已取得的单角色登记，不承载计划或等待。
计划诊断使用已有 diagnostics，不复制容量、状态、代价多套平行 record。
RequestRouteSelector 增加采用任一 P/D RoleSelection 的接续入口，替换 Decode 专属参数。
已取得角色登记与本次请求/Endpoint代际/group不匹配时回滚，不盲目组合其他角色。

## 7. 抢占执行组件

### 7.1 接口和成员

```java
interface RequestPreemption extends AutoCloseable {
    boolean start(SelectionAttempt exact, PreemptionPlan plan);
    void endResourceWait(SelectionAttempt exact, Throwable reason);
    void requestCheck(WorkerEndpoint endpoint);
    void requestPeriodicCheck();
    List<WorkItem> readyWork();
    void execute(WorkItem exactWork);
    void resultHandled(SelectionAttempt exact);
    void stopAdmission();
    boolean awaitIdle(long deadlineNanos);
}
```

WorkItem 是内部待执行短任务的只读描述，复用现有任务接口可表达时不另建公共类型。
它由 Scheduler 占执行额度后调用；不得在 readyWork 中做网络调用、资源登记或调用上层。
执行接口是唯一替换点，默认类不继承额外基类。装配由 SchedulerConfiguration 完成。

```java
// DefaultRequestPreemption
Map<SelectionAttempt, Pending> pending;
Object lock;
boolean stopped;
int maxPending; // 由配置装配，安装中记录也计入
RequestRepository requests;
RequestSchedulerReporter reporter;

class Pending {
    SelectionAttempt exact;
    ResourceWait wait;       // 安装期间为空；引用Endpoint中同一对象
    boolean checkRequested; // 检查中到达的通知仍保留下一次检查资格
    // 执行中由Scheduler执行任务记录表达；不再复制deciding计数。
}
```

pending 在进入 Endpoint 安装前先登记，取消/关闭能看见安装过程；安装完成后再复核请求有效性。
取消先到则安装者立即移除 Endpoint 等待。安装不能因 wait 暂为空而被误判为全部关闭。
Pending 直到结果采用/回滚完才删除；没有第二个完成 Future。
安装wait时注册一个完成回调，回调只唤醒Scheduler，不直接执行接续选路或抢占检查。
readyWork区分“资源检查”和“已完成结果处理”两种实际任务；同一Pending不得重复交付相同任务。
检查任务取走checkRequested后，新通知仍可再次置位；任务结束不清除后来到达的标记。
结果处理只在原选择/检查任务退出后执行，由executing中的精确任务排他保证。

### 7.2 开始抢占

```java
start(exact, plan) {
    锁内检查停止/重复/等待额度并登记Pending安装记录;
    try {
        锁外复核A有效、目标精确代际/group;
        wait = roleStrategy.createResourceWait(exact, plan.endpoint);
        // create仅构造对象，不写Endpoint；先在A上绑定，再安装索引。
        if (!A.bindResourceWait(exact, wait)) { 结束安装记录; return false; }
        发布Pending.wait引用;
        plan.endpoint.registerResourceWait(wait); // 锁内核对A仍关联wait才安装
        复核安装期间取消/超时/关闭;
        if (已经失效) { endpoint.removeResourceWait(wait); return 已接纳需收尾; }
        endpoint.tryAcquire(wait); // 等待已安装，再检查真实容量，成功则完全不取消B
        if (wait.result已完成) return true;
        for (B : plan.targets) {
            复核精确路由、登记、优先级和允许动作;
            未提交：请求B所属Scheduler撤销本次选择并恢复排队;
            已发送且允许：调用B已有取消入口;
            资源已足够则停止继续撤销/取消;
        }
        requestCheck(plan.endpoint); // 安装后主动检查，补通知窗口
        return true;
    } catch (failure) {
        结束已安装等待和安装记录，回滚已经取得的资源;
        throw failure;
    }
}
```

createResourceWait 的需求由具体角色 Strategy 根据现有字段构造；绑定A之后再交给相应Endpoint登记，
公共 Scheduler 不强转 P/D 计算 computeTokens、hardKv 或 expectedKv。
没有可行计划、额度不足或尚未发生副作用的失效返回 false；接纳后则必须完成等待结果和清理。
如果部分 B 的取消已发出而后续失败，结束 A 的等待，但不撤回或伪造 B 的取消结果。

计划中的“未发送 B”若已提交，不自动升级成远端取消。结束失效计划，真实容量仍不够则下一次
正常选路重新生成计划。重复命中同一 B 的实际撤销/取消入口需幂等，不能凭预计释放提前扣账。
通知和固定周期只检查既有目标，不能反复选择更多 B，不能扩大原计划。

### 7.3 等待检查与结果接续

```java
execute(checkWork) {
    if (A失效或到期或目标下线) { endResourceWait(exact, reason); return; }
    endpoint.tryAcquire(wait); // 短任务；无资源就返回，不阻塞线程
}

handleResourceResult(exact, wait) { // Scheduler占执行额度后执行
    try {
        registration = 读取已完成Future;
        if (A/exact/目标已失效) rollback(registration);
        else router采用对应RoleSelection继续选路并提交;
    } finally {
        结束未提交的exact并完成回滚;
        preemption.resultHandled(exact);
        finishTask(exactTask);
    }
}
```

Future 只表示已登记资源或等待失败；结果交付后仍必须检查请求有效性。
目标下线而 A 仍有效，结束旧等待/selection 后恢复排队；A 取消/超时则按原原因结束，
不能都变成 DISPATCH_FAILED。其他角色失败时回滚已取得登记，不复活旧等待。

## 8. Endpoint 等待记录与优先级限制

### 8.1 成员和接口

```java
// WorkerEndpoint，stateLock保护。只由支持抢占的P/D实现创建记录。
Map<Long, ResourceWait> resourceWaits;

abstract static class ResourceWait {
    long waitId;
    WorkerRequest request; // 指向A的窄接口；不引用balance.RequestContext
    WorkerEndpoint endpoint; // 固定实例/代际
    String requestId;
    long selectionId;
    int priority;
    long deadlineMs;
    CompletableFuture<RequestRegistration> result;
    // P实现保存computeTokens；D实现保存已有hardKv/expectedKv/AdmissionCapacity。
}

registerResourceWait(...); // 各角色明确需求参数；锁内精确登记
removeResourceWait(ResourceWait exact, Throwable cause);
tryAcquire(ResourceWait exact);
closeResourceWaits();
priorityAllowedLocked(int incomingPriority, long now);
```

除 result 以外元数据不可变；Future 对外只暴露只读 CompletionStage。
角色需求使用具体内部实现，不再定义与 Capacity 同字段的资源对象。
不保存额外 priorityFloor 缓存，直接遍历有界等待记录取得最高有效优先级。
所有实际接纳路径都调用同一优先级检查；仅在 Selector 快照上检查不能阻止在途 B 抢回。
Prefill 正常 tryRegisterSelection 补归一化 priority 参数；Decode 已有该参数。

### 8.2 A/B/C 的准确行为

```java
isEffective(wait, now) {
    return wait.request.currentResourceWait() == wait
        && !wait.request.isResponseDone()
        && !wait.result.isDone()
        && now < wait.deadlineMs
        && wait.endpoint仍是当前可接纳代际;
}

priorityAllowedLocked(incomingPriority, now) {
    floor = resourceWaits中isEffective(wait, now)为真的最高优先级;
    return incomingPriority >= floor;
}
```

A=60 等待时，B=20 不能登记，C=90 可以登记；C 登记不删除 A 的等待。
限制只在目标 Endpoint 生效，其他 Endpoint 继续并行。相同优先级不承诺登记先后。
过期记录即使清理回调延迟也不参与 floor；周期/请求超时继续负责完成结果和释放引用。

```java
tryAcquire(exact) {
    lock();
    try {
        if (!sameWait(exact) || !isEffective(exact, now())) return;
        检查代际/group/接纳状态/优先级/真实容量;
        registration = 可用时实际登记;
        if (registration != null) resourceWaits.remove(exact.waitId, exact);
    } finally { unlock(); }
    if (registration != null && !exact.result.complete(registration))
        rollbackSelection(registration);
}
removeResourceWait(exact, cause) {
    lock();
    try { removed = resourceWaits.remove(exact.waitId, exact); }
    finally { unlock(); }
    exact.result.completeExceptionally(cause);
    if (removed) notifyChanged(this); // 限制解除后普通低优先级可能恢复可行
}
```

资源登记与等待删除在同一把 Endpoint 锁内；Future完成和通知在锁外。
取消若与交付竞争失败，接收方仍依据Context失效回滚；不能将Future完成顺序当作请求提交顺序。

### 8.3 A持有当前等待，多个请求可以等待同一个Endpoint

```java
// WorkerRequest新增只读接口；实现不得获取Context锁，供Endpoint锁内读取。
ResourceWait currentResourceWait();
boolean isResponseDone(); // 读取既有响应Future的isDone，不保存副本布尔状态

// RequestContext
private volatile ResourceWait currentResourceWait;

synchronized boolean bindResourceWait(SelectionAttempt exact, ResourceWait wait) {
    if (!本次选路仍有效 || 已取消或结束 || currentResourceWait != null) return false;
    currentResourceWait = wait;
    return true;
}
synchronized boolean clearResourceWait(ResourceWait exact) {
    if (currentResourceWait != exact) return false;
    currentResourceWait = null;
    return true;
}
```

bind/clear只能由请求流程和抢占协调调用，不向Endpoint开放写Context能力。
请求取消/超时在记录结束事实的同一次Context锁内清除关联，并把旧wait交给锁外收尾。
放弃目标和重选同样先清关联，再结束Future、删除Endpoint索引。
目标下线、获取成功通过抢占结果处理清除A的精确关联；此间Endpoint已删索引或Future已结束，
不会继续产生优先级限制。旧回调clear旧wait失败即退出，绝不能清掉A的新等待。
清关联不等于新任务可开始：旧selection和executing/Pending必须完成交接/回滚才允许重新选路。

多个请求独立持有自己的wait：

```text
A.currentResourceWait → waitA → E
C.currentResourceWait → waitC → E
E.resourceWaits索引waitA和waitC；没有“当前唯一等待者”字段
```

A=60、C=90有效时，优先级下限90；C完成或放弃后下限60；A也结束后限制解除。
遍历时现场计算最高有效优先级，不维护需同步清理的priorityFloor。
A和C命中同一个B时，实际撤销/取消入口按精确登记幂等；实际容量只由账本记录一次。
每个请求最多一个当前目标，不限制每个Endpoint只有一个等待者。

### 8.4 登记、解除关联和物理清理的顺序

| 事件                         | 谁负责                             | 顺序                                                             |
| ---------------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| 开始抢占                     | Preemption协调，A和E各管自己的数据 | 创建wait→A绑定→E安装索引→复核有效→处理B                          |
| 安装时A取消                  | Context取消与安装者                | A先清关联；E安装检查拒绝，或安装者复查后精确删除                 |
| 取得资源                     | E获取，Preemption交接              | E锁内扣资源并删索引→锁外Future交付→清A关联→采用或回滚            |
| A放弃/取消/超时              | A所属Scheduler及Preemption         | A先清精确关联→锁外结束等待Future→E删精确索引                     |
| 目标关闭                     | E及Preemption                      | E停接纳并清本代际索引→锁外Future失败→清A关联→按请求状态重排/结束 |
| 系统关闭                     | Scheduler及Preemption              | 停新接纳→逐个清A精确关联并结束等待→等安装和资源结果清理完        |
| B释放不够、C先用、短检查结束 | 保留A等待                          | 不改变A关联，不结束其Future                                      |

E计算优先级时跳过失效记录，并可在同一次锁内移除失效索引；锁内不完成Future、不调用Context写接口。
周期检查负责失效等待的Future结束、上层Pending收尾和引用释放；E关闭负责清空索引。
如果漏掉物理删除，A的关联已清便立即不再限制接纳；如果关联清理也漏了，响应Future结束或截止时间
仍使记录失效。若A尚未结束且放弃动作完全没有记录，无法凭空知道其意图，必须由放弃入口清关联。

取消可能发生在Endpoint检查有效之后、登记完成之前。这不要求同时持Context和Endpoint两把锁；
允许产生短暂本地登记，但锁外交付前和路由提交前都要核对A/selection，失效登记必须回滚。
这保证取消后不提交路由，并避免反向锁顺序。所有生产读取currentResourceWait均走volatile只读访问。

## 9. 未发送 P/D 抢占与发送边界

纯排队未登记资源的 B 不需要被取消，排序即可。
已登记未提交的 B，撤销和不可回滚提交必须在同一身份边界互斥：

```text
撤销先完成：旧selection失效 → 本角色登记回滚 → 其他角色回滚 → B按原序号/截止时间恢复队列
提交先完成：本地撤销失败 → 不自动取消；由新计划依据允许阶段判断
```

Prefill REGISTERED 可以作为本地资源阶段，但还须核对路由提交事实；
Decode confirmed=false 只表示没收到Worker确认，不证明未发送。
对外路由一旦发布、外部已可执行，也不能因为本地没有gRPC写出就视为未发送。
实际实施沿 AbstractRequestScheduler.commitRoute/响应发布定位精确不可撤销点，并用屏障UT锁定。

B所属Scheduler负责撤销整个选择并恢复排队；Endpoint只操作自己的登记。
使用已有请求事件入口表达“未提交选择被撤销”，不伪装成Worker失败或用户取消。
不同时持有多个Endpoint锁；先使旧selection不能提交，再逐角色回滚，全部清理后恢复队列。
已发送Decode取消仍依据现有ACK/状态报告及释放规则，不能在取消发出时释放本地容量。

## 10. 公共请求流程和通知生产者的修改

### 10.1 AbstractRequestScheduler

保留 tryBeginRouteSelection、executeRouteSelection、finishSelectionWithoutCommit 等实际公共操作。
删除抢占钩子及 allowDecodePreemption。目标契约：

- 成功路径提交路由；失败中已经取得的资源由公共路径回滚。
- BLOCKED回到Queue时保留exact selection，Queue当场决定交给等待还是结束它。
- Direct遇BLOCKED必须当场结束exact；不能产生悬空选路。
- 调用方finally保证未交给等待的selection结束，异常也覆盖。
- RequestContext只新增一个volatile currentResourceWait关联；没有抢占Map、优先级缓存或第二个资源Future。

此处接口行为变化先补UT，不能在父类已经结束selection后又让抢占使用同一编号。
RequestResult归档等其他职责不在本次顺带重构。

### 10.2 CapacityNotifier及生产点

```java
final class CapacityNotifier {
    Set<Consumer<WorkerEndpoint>> listeners;
    void addListener(...);
    void removeListener(...);
    void notifyChanged(WorkerEndpoint exact) {
        // 调用者已释放Endpoint锁；逐监听器隔离异常。
        for (listener : listeners) safeNotify(listener, exact);
    }
}
```

删除WorkerStatus/address重载、CapacityKey回调和所有版本历史。
CapacityKey继续用于阻塞诊断和目标描述；删除waitingDomain及无调用的anyGroup。

| 生产者                   | 保留的发布点                                       | 删除/调整                          |
| ------------------------ | -------------------------------------------------- | ---------------------------------- |
| PrefillEndpoint          | Worker更新、真实释放、抢占撤销、接纳限制解除、关闭 | 普通临时选路回滚不唤醒自身重试     |
| DecodeEndpoint           | 同上，含真实取消/上报造成的容量变化                | 不将ACK或预计释放作为可用容量      |
| EncoderEndpoint          | 状态变化、真实释放、关闭                           | 仅使用简化通知，无抢占等待         |
| EndpointRegistry         | 新Endpoint可用、下线/替换                          | 删除通知历史回收；保留精确代际清理 |
| WorkerStatusSynchronizer | 沿现有Endpoint状态提交后锁外发布                   | 不新增调度RPC轮询，不维护请求映射  |

通知范围沿现有实例装配，不扩大跨模型广播。某次通知失败不影响账本事实，周期扫描负责重查。

## 11. 取消、超时与关闭

取消/超时在Context锁内同时记录结束事实和清除currentResourceWait，使选路不能再提交；
锁外结束这份旧等待并清理Endpoint索引、队列登记。
取消不会中断资源账本的原子操作；取得资源的一方必须在交付失败时回滚，接收方必须在请求失效时回滚。
不能把Future.cancel或orTimeout当完整清理。请求的原ExpirationTimer保持权威，
资源检查和登记同时核对deadline防止迟到执行；固定检查周期不能延长请求等待上限。

PreemptionConfig.timeoutMs现有含义是取消ACK后的等待配置，不直接用作A资源等待截止时间。
保留实际取消协议配置；若字段已无实际使用，单独完成引用和兼容核对后删除，不静默改义。

关闭顺序：

1. 停止接收新请求、新普通选路和新抢占；禁止继续发出新取消。
2. 结束全部等待，覆盖安装中记录；排空队列并记录请求关闭。
3. 已接纳任务和资源结果仍有执行者，执行采用失败回滚和请求清理。
4. 等待抢占组件无安装/等待/结果交接，且Scheduler.executing为空。
5. 退出driver、shutdown planners并等待终止，注销订阅；共享任务池和定时器最后关闭。

stop只发起停止，close由外部生命周期线程等待；工作线程不能等待自己。
异常回调不允许静默丢任务。必要的同步回滚仅处理本地短操作，不把网络等待放进CallerRuns。

## 12. 配置、实施步骤和验收

### 12.1 配置和装配

- SchedulerConfig增加capacityCheckIntervalMs=100，校验正数；保留scanBudgetMultiplier和plannerCount。
- 初版maxPendingPreemptions沿用plannerCount，不增加独立可调参数，后续有测量再决定。
- PreemptionStage补未提交Prefill、未提交Decode；保留已发送Decode阶段。
- SchedulerConfiguration替换旧执行Bean、删除Policy Bean、具体Planner注入DecodeStrategy。
- 原自定义Policy扩展点迁移到具体角色策略/保留的执行接口，明确兼容变化；不保留空转适配链。
- api生命周期装配按第11节调整；cache/master/util无需增加通知状态。

### 12.2 按纵向行为实施

1. 建立现有UT基线，先补下面的缺失契约测试，不以编译失败作为有效红测。
2. 完成队列接口和精确任务身份迁移，删GlobalQueueEntry/SchedulingRequests，确保取消旧任务安全。
3. 完成同一调度入口的通知、固定周期和执行额度；保留 CapacityWaitQueue 的容量重试限制，删除通知历史版本。
4. 角色策略生成计划，迁移具体Decode算法；验证生成过程无副作用。
5. 完成未提交P/D撤销与提交互斥，回滚整条选择后恢复排队。
6. 完成Endpoint优先级等待、执行组件、Future交接和取消/超时/关闭；串通完整抢占流程。
7. 清理旧类型、字段、回调、Bean、测试夹具及架构文档；不保留双机制。

### 12.3 必须通过的行为测试

| 场景             | 断言                                                               |
| ---------------- | ------------------------------------------------------------------ |
| 通知与固定周期   | 通知快速启动；无通知仍重查；周期不被通知推迟                       |
| 静态无容量       | 不因任务结束/临时回滚形成忙循环，检查次数受周期与真实通知约束      |
| 大队列           | 有限扫描持续向后推进，不能永远只查队首                             |
| 并发额度         | executor排队、正常选择、检查、接续总执行数不超过上限               |
| 提交失败         | 任务计数/执行记录对称清理，已得资源不丢失                          |
| 固定顺序         | 恢复序号和截止时间不变；旧任务不能删除新登记                       |
| 策略产计划       | 无取消RPC、账本变动、等待登记等副作用；原slot/KV代价契约保持       |
| 未发送P/D        | 撤销与提交屏障竞争，最多一个成功，完整回滚后恢复B                  |
| 已发送Decode     | ACK不等于释放；真实资源未释放不能登记A                             |
| A/B/C            | B低优先级被挡，包括旧在途登记；C高优先级可用；C成功不结束A等待     |
| 等待移除         | 取得、取消、超时、放弃目标、下线均清精确关联；旧wait不能删新wait   |
| 故意漏物理删除   | A已清关联时B立即可登记；响应结束/截止到时也不再受残留索引限制      |
| 多等待者         | C=90退出后降到A=60；C成功不结束A，多个计划对同一B不重复释放        |
| 取消交错         | 登记前、交付前、交付后提交前，资源余额与响应次数正确               |
| 目标换代         | 同地址新实例不接受旧等待，旧结果只清旧资源                         |
| 其他角色失败     | 已得P/D登记回滚一次，旧等待不复活                                  |
| 关闭             | 安装中、检查中、等待中、已交付未处理均收尾，无悬挂Future或资源泄漏 |
| Direct及替换实现 | Direct不遗留selection；队列及抢占执行替换契约仍成立                |

采用可控时钟、屏障和锁存器，不用长sleep掩盖竞态。保留仍有效的旧业务断言。
性能验收覆盖大队列、混合分组、高频通知、持续无容量和密集抢占，记录吞吐、P95/P99、CPU、
扫描量、选择/回滚次数及Endpoint锁等待。通知变粗会增加重查，未测量不宣称性能等价。

完成代码后执行成员排序、受影响UT、全reactor编译和完整Spotless；提交前执行仓库pre-commit。
本文件不记录未经运行的测试通过数，也不将设计目标标注为已经实施。
