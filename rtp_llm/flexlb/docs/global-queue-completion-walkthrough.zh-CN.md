# GlobalQueueCoordinator 完成队列改动：逐行阅读

历史版本说明：本文对应下面指定的 commit。后续版本已将主循环改为每轮处理一个结果后
补充选路任务，不再使用本文第二区块中的互斥分支；当前规则见 [调度说明](global-queue-coordinator.md)。

对应实现 commit：`280f2e0a4e`，相对于 `4904d00594`。下列行号指该实现版本的
`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/GlobalQueueCoordinator.java`。
逐条解释所有新增的可执行语句；空行和仅用于结束代码块的大括号不单独展开。
最后解释删除的旧机制，以及测试契约的变化。`plan()`、`commit()`、抢占和容量等待规则沿用原实现。

**先区分三个阶段**

```text
尚未获得选路名额 → 正在选路 → 选路完成、等待准入 → 成功或容量等待
                     └──────── inFlight ────────┘
```

“选路”只是选择 worker；“准入”才真正检查并占用资源。两个阶段之间，容量可以变化。
本次只改变选路任务及其结果的处理顺序，不把准入搬到多个选路线程里执行。

**一、字段：只增加两份状态（65–69 行）**

```java
private final Set<GlobalQueueEntry> inFlight =
        Collections.newSetFromMap(new IdentityHashMap<>());
private final ArrayDeque<Plan> completedPlans = new ArrayDeque<>();
```

| 行 | 解释 |
| --- | --- |
| 65–66 | 注释明确：已经派出去的请求，要一直占着名额，直到结果处理和资源清理结束；取消也不立即释放名额。 |
| 67 | `inFlight` 记录已经分配选路名额、尚未处理完结果的请求。不是 worker 上正在推理的请求。 |
| 68 | 用对象身份构造集合。只有同一个 `GlobalQueueEntry` 对象才算重复，不依赖其内容的 `equals()`。集合用来防止重复派发，并计算剩余名额。 |
| 69 | `completedPlans` 存已经完成的选路结果。结果从尾部放入、从头部取出，按结果入队的顺序处理。它不是请求到达顺序队列，也不是 BATCH 队列。 |

原来的 `orderedQueue` 仍负责优先级/FIFO；`waitingRequests` 仍负责容量等待。
`inFlight` 和 `completedPlans` 都由原有 `lock` 保护，因此这里不需要换成并发容器。
新增的 `Collections`、`IdentityHashMap`、`Set` import 只是服务于这两行字段。
删除 `MIN_PLANNING_FRONTIER_SIZE`，因为不再每轮固定取一整批，而是按剩余名额取。

**二、主循环：先处理已完成的结果，否则补充选路任务（153–193 行）**

以下保留实际控制流，在每个有意义的语句旁解释：

```java
private void runDecisionLoop() {
    try {
        while (!closed.get()) {                 // 155：没关闭就继续处理事件。
            Plan completed;                     // 156：本轮拿到的已完成结果。
            List<GlobalQueueEntry> candidates = List.of(); // 157：本轮新获得名额的请求，初始为空。
            lock.lock();                        // 158：接下来读写共享队列和名额，需要加锁。
            try {
                if (closed.get()) {             // 160：拿锁前可能刚发生关闭，再检查一次。
                    break;                      // 161：直接离开循环，进入统一清理。
                }
                completed = completedPlans.pollFirst(); // 163：取已经存在的结果；没有就返回 null，不等 R1。
                if (completed == null) {        // 164：没有结果待处理时，再寻找可启动的任务。
                    int slots = plannerCount - inFlight.size(); // 165：允许在途数减去当前在途数。
                    if (slots > 0) {            // 166：有空闲名额才取请求。
                        candidates = planningCandidates(slots); // 167：按优先级/FIFO 扫描，最多取 slots 个。
                        inFlight.addAll(candidates); // 168：先登记占用名额，避免请求被重复派发。
                    }
                    if (candidates.isEmpty()) { // 170：本轮既没有结果，也没拿到新任务。
                        if (slots == 0 || (!orderedQueue.hasUnscannedRequests()
                                && !waitingRequests.hasReadyRequests())) { // 171–172：名额满了，或暂时没有待扫描/可唤醒的工作。
                            awaitChanged();     // 173：睡眠等待通知；Condition.await 会释放这把锁。
                        }
                        continue;               // 175：被唤醒后重新判断；若只是扫描预算耗尽，则直接继续扫描。
                    }
                }
            } finally {
                lock.unlock();                  // 179：离开索引操作区，包括 continue/break 分支。
            }
            if (completed != null) {            // 181：这一轮拿到了选路结果。
                processCompletedPlan(completed); // 182：锁外做准入、等待登记和清理。
            } else {
                candidates.forEach(this::submitPlan); // 184：锁外把刚获得名额的请求交给选路线程池。
            }
        }
    } finally {                                 // 187：正常关闭或异常退出都会走这里。
        closed.set(true);                       // 188：后续晚到的结果必须走清理，不再入完成队列。
        availability.removeListener(availabilityListener); // 189：停止接收容量通知。
        planners.shutdown();                    // 190：停止接收新任务，允许已提交任务结束。
        drainOnClose();                         // 191：清理队列和缓冲结果，通知剩余请求关闭失败。
    }
}
```

第 160 行不是重复装饰：如果关闭发生在外层判断之后、获取锁之前，且关闭通知已经发出，
没有这次检查就可能在第 173 行睡过去，再也等不到唤醒。

第 171–175 行保留“有限扫描”。例如队列前面有很多等待中的请求，本轮扫描预算用完还没找到
可选请求，但后面还有未扫描项，此时应该继续向后扫，不能把它误认为没有工作而睡眠。

第 163 行是解除慢选路阻塞的核心。旧代码取的是“R1 的 future”，新代码取的是
“已经放进完成队列的第一个结果”。R1 没完成时，它根本不在这个队列中。

本实现先处理已经积累的完成结果，再补名额；不会为了凑齐一批结果等待慢请求。
若批量取请求期间 R0 才到达，已经分配的名额也不会收回，R0 竞争下一次空闲名额。

**三、submitPlan：启动计算，把结果统一送回来（195–209 行）**

```java
private void submitPlan(GlobalQueueEntry entry) {
    try {
        planners.execute(() -> {                // 197：把计算任务交给已有的固定线程池。
            Plan result;                        // 198：线程最终必须交回一个结果。
            try {
                result = plan(entry);           // 200：执行旧的选路函数；路由算法没有改。
            } catch (Throwable failure) {
                result = Plan.failure(entry, failure, availability.sequence()); // 202：把异常转成失败结果。
            }
            publishPlan(result);                // 204：成功、阻塞或异常，都统一交回协调器。
        });
    } catch (Throwable failure) {                // 206：这是“任务没提交进去”的失败，例如线程池正在关闭。
        publishPlan(Plan.failure(entry, failure, availability.sequence())); // 207：也必须交回结果，不能丢失已占名额的请求。
    }
}
```

内外两个 catch 对应不同阶段：内层是任务已经运行后失败，外层是任务提交失败。
原来的 `CompletableFuture.supplyAsync()`、`SubmittedPlan` 和后面的 `join()` 不再需要。
这里没有取消请求用于最终响应的 `CompletableFuture<Response>`，删除的只是规划任务那层 future。

**四、publishPlan：生产者交结果，关闭后自己释放资源（211–224 行）**

```java
private void publishPlan(Plan plan) {
    lock.lock();                                // 212：多个 planner 都可能同时交回结果。
    try {
        if (!closed.get()) {                    // 214：协调器还在接收工作。
            completedPlans.addLast(plan);       // 215：将结果放到完成队列尾部。
            changed.signal();                   // 216：唤醒可能正在等待的协调线程。
            return;                             // 217：结果的清理责任交给协调线程；finally 仍然执行。
        }
    } finally {
        lock.unlock();                          // 220：无论正常入队还是已经关闭，都释放锁。
    }
    closePlan(plan);                            // 223：关闭后的晚到结果不入队，由当前 planner 清理。
}
```

不能简单地在关闭后 `return`：Plan 可能持有 endpoint pin、AdmissionMutation 等需要关闭的对象。
检查关闭状态和入队在同一把锁下；退出清理也在这把锁下取走缓冲结果，因此结果不会在队列清理后偷偷遗留。
资源关闭放在锁外，避免清理过程拖住所有入队、取消和容量通知。

这里的“完成顺序”精确地指结果获得锁并进入 `completedPlans` 的顺序，不是各线程 CPU
算完的纳秒时间戳顺序。它保证不主动等待更早的请求，不保证同时完成线程之间的绝对先后。

**五、processCompletedPlan：准入结果只影响本请求（226–254 行）**

```java
private void processCompletedPlan(Plan plan) {
    boolean retry = false;                      // 227：默认不立即重试。
    try {
        if (!closed.get()) {                    // 229：协调器没关闭才尝试准入。
            Outcome outcome = commit(plan);     // 230：复用旧准入逻辑，包括真实资源检查和抢占。
            retry = outcome == Outcome.REPLAN   // 231：计划过时，只重试这个请求。
                    || (outcome == Outcome.BLOCKED && !park(plan)); // 232：容量不足则登记等待；登记发现新容量通知时，改为重试。
        }
    } catch (Throwable failure) {
        removeRequest(plan.entry);              // 235：异常请求从队列移除。
        completeDecisionResponse(plan.entry, error(
                StrategyErrorType.DISPATCH_FAILED,
                "Placement failed: " + failure.getMessage())); // 236–238：通过原生命周期组件返回失败。
        Logger.error("Global queue commit failed: request_id={}",
                plan.entry.context.getRequestId(), failure); // 239–240：记录请求和异常。
    } finally {
        closePlan(plan);                        // 243：先关闭计划持有的资源和本次操作所有权。
        lock.lock();                            // 244：下面修改在途集合和待重试索引。
        try {
            inFlight.remove(plan.entry);        // 246：直到这里才释放名额。
            if (retry && isQueued(plan.entry)) { // 247：需要重试，并且没有被取消/完成/移除。
                orderedQueue.markRequestReadyForRetry(plan.entry); // 248：只把自己放回可调度索引，保留原序号和优先级。
            }
        } finally {
            lock.unlock();                      // 251：释放索引锁，回主循环处理下一件事。
        }
    }
}
```

第 231–232 行展开后的意思：

| 准入结果 | 本次处理 |
| --- | --- |
| DONE | 已经成功、终止或取消，不重试。成功移除队列由旧 commit 完成。 |
| REPLAN | 当前选路结果过时，关闭本计划后重新给本请求排队。 |
| BLOCKED，park 成功 | 请求进入容量等待，不立即重试。 |
| BLOCKED，park 拒绝等待 | 选路期间发生过相关容量变化，不能错过通知；重新给本请求排队。 |

`park()` 返回 true 也可能表示请求已经离开队列、无需再等待。

第 243 行必须在第 246–248 行之前。否则重新取出的同一个请求可能看到上一次
`AdmissionMutation` 还没释放，错误地认为它不能继续处理。这里仍然只允许一个协调线程执行 commit，
没有把资源提交变成多线程并发。

这里的 close 只清理 Plan 仍然拥有的临时对象；成功准入后已转交给请求生命周期/worker 的资源，
不会因为 closePlan 而被当作失败请求释放。

**六、planningCandidates：按空闲名额取请求（264–276 行）**

```java
private List<GlobalQueueEntry> planningCandidates(int slots) {
    waitingRequests.resumeReady(slots, orderedQueue::markRequestReadyForRetry); // 266：把获得重试机会的等待请求放回可调度索引。
    return orderedQueue.scanForPlanningCandidates(
            slots, calculateScanBudget(slots), entry -> { // 267–268：最多取 slots 个，扫描项数另有预算。
                if (entry.future.isDone()) {     // 269：最终响应已经结束，不应再选路。
                    removeRequestUnderLock(entry); // 270：清理队列和等待索引。
                    return false;               // 271：不选取它。
                }
                return !entry.removed           // 273：尚未移除。
                        && !inFlight.contains(entry) // 273：尚未分配选路任务，防止重复派发。
                        && !waitingRequests.isWaiting(entry); // 274：没有停在容量等待里。
            });
}
```

原来的 `awaitPlanningCandidates()` 自带等待循环，现在等待统一放到主循环，避免“请求来了”
和“选路结果来了”各自拥有一套等待流程。单独的 `isEligible()` 合并进这个筛选函数。
`resumeReady()` 内部的同域单个重试规则没有改变；请求从等待中恢复，也要与其他请求按优先级/FIFO竞争名额。

**七、关闭：清理两种位置上的结果（487–534 行）**

`drainOnClose()` 新增的语句：

| 行 | 代码 | 解释 |
| --- | --- | --- |
| 489 | `List<Plan> completed;` | 暂存准备释放的已完成结果。 |
| 494 | `completed = List.copyOf(completedPlans);` | 锁内取出已经交回、尚未处理的全部结果。 |
| 495 | `completedPlans.clear();` | 清空共享完成队列。 |
| 496 | `inFlight.clear();` | 协调器已退出，不再分配名额，清空索引。未返回的任务仍负责自己的资源清理。 |
| 500 | `completed.forEach(GlobalQueueCoordinator::closePlan);` | 解锁后逐一关闭结果。 |

原有的 `orderedQueue.drain()`、`waitingRequests.clear()`、给剩余请求返回关闭失败，继续保留。

`close()` 仍调用 `planners.shutdown()`，但删除了“协调线程结束后再调用 shutdownNow()”。
现在协调线程不再等每一个慢选路任务结束；如果强制丢弃已提交但尚未执行的任务，就破坏了
“每个已提交任务都会交结果或执行晚到清理”的约定。因此让这些数量受限的任务自然结束。
已有的有时限 `awaitTermination()` 仍保留。超时后仍在运行的选路并没有被强制杀死；它回来后由
`publishPlan()` 的关闭分支清理。这也是慢 planner 可能延长资源释放时间的边界。

**八、删除的旧代码：为什么现在不需要**

旧主循环中的这一段是原来等待慢请求的地方：

```java
for (int planIndex = 0; planIndex < plans.size(); planIndex++) {
    Plan plan = plans.awaitNext();
    // 后续请求即使算完，也必须等 awaitNext 返回当前顺序的结果。
}
```

旧 `PlanningPipeline.awaitNext()`：

```java
fill();                                      // 提交一些任务。
SubmittedPlan next = submitted.removeFirst(); // 取最早提交的任务，不是最早完成的任务。
return awaitPlan(next.future(), next.entry()); // 最终 join 它，哪怕它很慢。
```

因此删除整套 `PlanningPipeline`，包括：

| 旧成员/语句 | 旧作用 | 新实现如何处理 |
| --- | --- | --- |
| `entries`、`nextToSubmit` | 记住这批请求及提交位置 | 每次按当前空闲名额从有序索引取请求。 |
| `submitted`、`SubmittedPlan` | 把请求及其 future 按提交顺序保存 | 已派发请求记在 inFlight；完成结果记在 completedPlans。 |
| `maxInFlight`、`fill()` | 限制这批同时计算几个 | `plannerCount - inFlight.size()` 直接控制名额。 |
| `size()`、外层按下标循环 | 必须逐个处理这一批 | 统一事件循环持续处理结果和补充任务。 |
| `awaitNext()`、`awaitPlan()`、`future.join()` | 等最早提交的任务算完 | 从完成队列取已经算好的结果。 |
| `closeSubmitted()` | 同步等后续任务结束，再关闭其计划 | 新到达/唤醒的请求不再作废已有任务。关闭时缓冲结果和晚到结果各自清理。 |
| `supplyAsync()` | 为每个选路任务创建额外 future | `execute()` 运行后直接交结果，提交失败也交失败结果。 |

其他删除项：

| 旧代码 | 删除原因 |
| --- | --- |
| `hasEarlierRequestsToScan(plan.entry)` | 不再因前面有新请求/唤醒请求，就拒绝处理已经完成的结果。 |
| `waitingRequests.hasEarlierReadyRequest()` | 原来服务于上述全局结果顺序检查；等待队列自身的排序仍保留。 |
| `orderedQueue.hasEarlierRequestsToScan()` | 同上；候选扫描依然按 FIFO/优先级取。 |
| `restartFrontier` | 不再整批重启。 |
| `rescanUncommittedCandidates(frontier)` | 一个计划过时只重排自己，不把其他计划全部放回。 |
| `planningCandidateLimit()` | 原来每轮用固定 plannerCount；现在使用剩余名额 slots。 |
| `awaitPlanningCandidates()` 的循环和等待 | 合并到主循环，同时处理请求到达、容量通知、结果完成。 |

原先捕获扫描错误后继续循环的分支也没有保留：索引扫描等意外异常现在会让协调线程退出，
进入统一关闭/清理并由原有 uncaught-exception handler 记录。单个选路任务的错误、提交失败、
单个准入错误仍在对应函数内处理，不会因此关闭整个协调器。

**九、用一个完整例子把代码连起来**

假设 plannerCount=2：

```text
① R1、R2 获得名额：inFlight={R1,R2}，剩余名额=0。
② R1 卡在选路；R2 完成，publishPlan 把 R2 结果放进 completedPlans。
③ 主循环取 R2，commit 成功，关闭结果，inFlight 移除 R2。
④ 剩余名额=1。按队列顺序取 R3，R3 开始选路，R1 继续占着自己的名额。
⑤ 此时更早的 R0 被叫醒：它回到可调度索引，不打断 R1/R3。
⑥ R3 完成并释放名额。下一次取请求时优先取 R0。
⑦ R1 终于完成，照常处理；不要求此前的 R2、R3 重新计算。
```

若 R1 被取消，队列会移除它，但名额暂时不释放。等 R1 的计算返回，结果会因请求已结束而
被关闭，随后释放名额。否则反复取消慢请求可以绕过并行上限，造成任务越积越多。

**十、测试改动不是“旧顺序断言全部还成立”**

1. `GlobalQueueProgressTest`：把“早到请求唤醒后必须先提交”的旧测试，换成“唤醒请求使用空闲名额，
   不作废正在执行的计划”；加入慢请求不阻挡其他结果/补充任务、取消仍保留在途名额、关闭后晚到结果释放、
   优先级/FIFO决定下一个名额的测试。连同原有容量和抢占场景，共 8 个测试。
2. `RequestSchedulerTest`：原来只等两个后继完成初始尝试，就隐含认为队头也完成了。
   完成顺序放松后不再成立；改成等三个请求各自完成初始尝试。容量恢复次数和顺序断言继续保留。
3. `TransientCapacityQueueContractTest`：BATCH 下 Prefill 忙也可能先接收路由并占用 Decode，
   所以把用于验证全局等待顺序的两个场景改成 Decode 容量饱和。释放容量前确认请求已经尝试并停止活动。
   继续断言同域等待请求的 FIFO/优先级，以及单个位置不会超卖，不再拿它证明所有在途结果严格有序。
4. `BaselineParityE2ETest`：严格 FIFO 的旧基线限定为单 planner、关闭优先级的情形。
   默认多 planner 不再承诺最终严格 FIFO。这是明确的业务顺序语义变化，不是完全等价重构。
5. 删除两个已经没有调用者的“有没有更早请求”辅助方法及其直接断言；有序扫描和等待队列排序测试仍保留。

本次没有改真实容量准入、抢占算法、BATCH/NON_BATCH 发送算法，也没有新增配置开关。
性能是否达标须看独立的 750P/750D 测量，不能从“代码更短”或功能测试通过推断。
