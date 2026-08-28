# FlexLB 资源本地有序调度落地设计

> 日期：2026-08-30
> 范围：FlexLB QUEUE 模式的临时容量等待、恢复与选机
> 目标：保留请求到达时的立即选机性能，同时避免全局惊群、重复决策、低优请求插队和资源泄漏。

## 1. 最终决策

采用“立即尝试 + 资源本地有序等待”，不再使用扫描全局 pending 请求的中央重试模型。

```text
请求到达
  -> 注册唯一 RequestSlot / future / absolute deadline / placement sequence
  -> 立即执行一次完整 placement
       -> 成功：一次性提交 P/D ownership
       -> 静态不可能：终态拒绝
       -> 暂时不可用：登记到 blocker 对应的 role + group wait lane，立即返回 future

真实容量释放
  -> 只激活对应 wait lane
  -> 按 priority、sequence 取队首重新执行完整 placement
       -> 成功：继续处理下一位，直到容量用完
       -> 仍被当前 lane 阻塞：停止，等待下一次真实容量变化
       -> 被另一资源阻塞：移动到新的 lane
```

一句话边界：**等待层只决定“谁可以再试”，现有 Selector 决定“选哪台机器”，endpoint reservation 决定“是否真正拿到资源”。**

## 2. 不做什么

- 不改变 `ConfiguredLoadBalanceSelector` 及 cache-first、short-TTFT、random 等配置策略。
- 不引入 top-K、候选裁剪、跨请求打分或新的全局 placement planner。
- 不在全局层构造 batch；`WorkerBatcher` 的 fixed-window 决策继续独占机器本地凑批语义。
- 不增加 `WAITING_P`、`WAITING_D`、`RETRYING`、`PREPARED` 等请求生命周期状态。
- 不让请求线程 sleep、轮询、等待 condition 或等待资源回调。
- 不因普通 WorkerStatus 心跳唤醒 pending backlog。
- 不追溯撤销已经进入短提交事务的低优请求。

DIRECT 模式保持原流程。QUEUE 的 FIFO、PRIORITY、BATCH、SINGLE、NON_BATCH 继续使用现有配置和 delivery 实现。

## 3. 三个互相独立的概念

### 3.1 Placement order

请求在 `RequestScheduler.submit()` 注册成功后获得不可变的全局 `sequence`：

```text
FIFO:      sequence 升序
PRIORITY:  priority 降序，sequence 升序
```

重试、跨 lane 移动、WorkerStatus 更新和抢占都不能修改该顺序。sequence 只用于 Master 内同一进程、同一请求生命周期的稳定排序。

### 3.2 Capacity signal

信号只表示“这个资源域的状态发生了可能有利的变化”，不携带资源，也不承诺某个请求一定成功。

资源域使用已有的：

```text
PlacementKey = role + group
```

允许发送 placement capacity signal 的事件：

- Prefill 队列真实移除一个或多个请求；
- Decode 精确 reservation / permit 确认释放；
- WorkerStatus 证明 Decode 可用容量相对上一版本真实增加；
- 新 endpoint generation 发布。

不能发送 placement capacity signal 的事件：

- 内容不变的 WorkerStatus 心跳；
- predictor、窗口计时器或普通 scheduling input 更新；
- 开始抢占但资源尚未确认释放；
- FixedWindow 等待条件变化；
- 单纯配置重新读取。

信号必须带单调 version，用于关闭“发现无容量”和“登记等待”之间的竞态。重复信号可以合并。

### 3.3 Exact ownership

真正的资源获取仍然是 endpoint 上的非阻塞操作：

```text
tryReserve(...) -> exact ReservationHandle | null
offerForPlacement(...) -> true | false
```

信号不能直接授予资源。每次被激活的请求仍执行完整 Selector，并通过 generation pin、reservation token 和 endpoint lock/CAS 校验精确 ownership。

Decode 有一条必须显式区分的模式边界：启用 PRIORITY/抢占的硬 placement 路径，queued reservation 立即计入 placement concurrency，保证一次容量释放不会在异步 delivery permit 建立前超发；配置明确采用 dispatch-time Decode 容量的非抢占路径继续使用 soft queued hold，其 Selector 本来也不会因为瞬时 Decode concurrency 进入 placement wait lane。

## 4. 请求线程模型

请求线程只执行一次立即尝试，不等待容量：

```java
PlacementResult result = attemptPlacement(request);
if (result.committed()) {
    return future;
}
waitRegistry.await(request, result.blocker(), observedVersion);
return future;
```

第一次尝试保留在调用线程中，避免无压力时多一次线程切换。Selector 和 endpoint 获取都必须是有限 CPU 操作，不允许远程 I/O 或资源等待。

后续尝试由有界 placement executor 执行。每个 lane 同时最多一个 active request；不同资源域可以有限并行，不能为每个 endpoint 创建线程。

## 5. Wait lane 的最小模型

每个 `PlacementKey` 只需要：

```text
ordered waiting entries
one active entry
last consumed capacity version
scheduled flag
```

每个请求最多拥有一个等待登记，并且只能位于：

- 一个 lane 的 ordered set；或
- 该 lane 的 active 槽位；或
- 已经关闭，不再由等待层持有。

不再需要：

- candidates / attempted 双集合；
- retry round；
- rescanRequested / restartRound；
- limited/pool-wide bypass；
- 一次信号扫描 256 个请求的语义预算。

placement executor 的并发上限和 cooperative yield 只用于保护 Master CPU，不改变排序，也不代表一次释放固定放行多少请求。

## 6. 一次容量变化放行多少请求

没有固定 N，也不读取 FixedWindow 的 batch 上限。

一个 lane 被激活后：

1. 取当前有序队首；
2. 完整选机并尝试精确提交；
3. 成功则继续下一位；
4. 队首仍被同一资源阻塞则立即停止；
5. 队首改为被另一资源阻塞，则移动它并继续当前 lane；
6. lane 为空则结束。

因此实际放行数量由当时的 Selector、Prefill queue seat、Decode exact reservation 和请求资源需求共同决定。资源足够时一次事件可以连续恢复多个请求；资源不足时第一个边界失败就停止，不扫描整个 backlog。

## 7. 公平性与大请求饥饿

同一 `role + group` 的竞争请求严格遵守配置顺序：

- PRIORITY：高优先级先；
- 同优先级：全局 sequence 先；
- FIFO：全局 sequence 先。

队首暂时无法满足时，不允许低顺序的小请求继续消耗同一受限资源。容量会为队首逐步积累，避免“大请求永远被小请求吃空”。

前提是静态不可能满足的请求必须由现有静态容量检查终态拒绝，不能进入 wait lane 永久挡住后续请求。

严格顺序只作用于真实共享的资源域。互不竞争的 group 可以并行；强制跨独立 group 的绝对全局 FIFO 会制造无意义的 head-of-line blocking，不采用。

### 7.1 新请求不能绕过已有等待者

新请求仍可以立即运行 Selector，但在取得精确 reservation 前必须执行一次轻量 order gate：

```text
selected P/D resources
  -> 查询这些 role + group 是否存在顺序更高的 waiting/active request
       -> 有：本次 admission 关闭并登记到该 lane
       -> 无：进入短资源提交事务
```

提交事务开始后不再追溯插队。随后到达的高优请求优先获得下一次机会，但不会撤销一个已经进入 endpoint reservation + publication 短临界区的请求。

## 8. 多角色 placement 与提交

Selector 可以按现有 `requiredRoles` 顺序选择 P、D 和其他角色；等待层不改变这个顺序，也不缓存上一次部分选择。

一次 attempt 的语义是：

```text
读取当前完整视图
  -> 使用现有 Selector 选择所有 required roles
  -> order gate
  -> 获取 exact Decode reservation（如需要）
  -> 将 RequestSlot 与 Prefill queue publication 一次提交
```

任何一步失败：

- 关闭所有 generation pins；
- 回滚本次 exact Decode reservation；
- 回滚未成功发布的 RequestSlot item；
- 返回一个精确 blocker；
- 原 future、deadline、priority、sequence 保持不变。

不允许部分结果跨 attempt 复用。P 成功、D 失败后，下次仍完整重选 P/D，以满足机器状态新鲜度要求。

硬 Decode placement reservation 和后续 engine-dispatch permit 是同一精确 ownership 的两个阶段，不能出现“reservation 已发布但 placement concurrency 未占用”的窗口。soft queued hold 只属于明确延迟 Decode 容量判断的配置，不能与抢占 admission 混用。

## 9. 提交与更高优请求并发

“Selector 选中”不等于拿到资源。

- 如果低优请求只完成 Selector，高优请求先取得 endpoint lock/reservation：低优请求的 `tryReserve` 失败并进入 wait lane。
- 如果低优请求已经取得 exact reservation 并进入短提交事务：高优请求不能撤销该 provisional publication；它等待下一次机会或走现有已提交请求抢占协议。

这条边界避免引入 revocable provisional lease 和额外状态机。当前 Decode `queuedPhase` ownership 和 Prefill queue lock 继续作为提交期保护。

抢占开始不是容量释放。只有 victim ownership 被精确解除，并且容量没有同时转移给 incoming 请求时，才能发出新的 placement capacity signal。

## 10. Timeout、取消与资源释放

`ExpirationTimer + RequestSlot` 继续是唯一 deadline 所有者。wait lane 不创建定时器，也不复制 deadline。

### 10.1 等待期间超时

```text
absolute deadline 到达
  -> RequestSlot 原子终结并完成原 future
  -> future completion 关闭 wait registration
  -> lane 删除该 entry
```

等待请求没有 endpoint ownership，因此没有 P/D 资源要释放。

### 10.2 attempt 与超时并发

- timeout 先关闭 RequestSlot：`beginAdmission/commitRoute` 失败，本地 admission close 回滚所有 provisional ownership；
- commit 先完成：timeout 从 RequestSlot 取得精确 `BatchItem/ReservationHandle`，按既有队列删除、Decode release、必要的 cancel/fence 流程清理。

所有释放必须使用精确身份：request id、endpoint generation、reservation token、exact BatchItem。旧 timeout 不能释放新 generation 的资源。

资源确认释放后才发布 capacity signal。future 完成回调只删除等待登记，不直接猜测或释放 endpoint 资源。

## 11. FixedWindow 边界

全局等待层不感知 prefix、collection window、token、KV、预测时间和 batch size。

请求 placement 成功后进入 Selector 选中的 `WorkerBatcher`；随后完全由其 fixed-window 决策决定：

- 窗口何时开启/关闭；
- 哪些请求能够组成同一 batch；
- batch 的 count/token/KV/time 约束；
- delivery 时机。

Prefill queue 真正移除请求后只发一次 role/group capacity edge。等待层连续填充可用 queue seat，但不会指定 endpoint、强行凑 batch 或使用释放资源的 endpoint hint，因而不会覆盖 cache-first/short-TTFT 的选择结果。

## 12. 关闭和异常边界

- scheduler 关闭：停止接受登记，摘除 availability listener，关闭所有 waiting/active entry，并以 scheduler shutdown error 完成尚未完成的 future；
- placement attempt 抛异常：只终结该请求并回滚其 admission，不停止整个 lane executor；
- listener 抛异常：隔离记录，不能阻断 endpoint 的资源释放；
- 重复 signal：version 合并；
- signal 与 park 并发：如果 blocker version 已前进，登记后立即调度一次；
- future completion 与 active attempt 并发：active attempt 在 Selector 前和 commit 时都检查 canonical RequestSlot，任何已取得资源由 admission close 精确回滚。

## 13. 代码落点

| 文件/类 | 修改职责 |
| --- | --- |
| `RequestScheduler` | 分配稳定 sequence；执行首次立即尝试；连接 future 与 wait registration；提交前 order gate |
| `PlacementWaitRegistry` | 重写为轻量 resource-local wait registry；移除全局候选扫描和 retry round |
| `QueueRouteAdmission` | 暴露本次 admission 的 P/D `PlacementKey`；保持 exact reservation 与全量回滚所有权 |
| `PlacementAvailability` | 保留 versioned edge；合并重复通知；只承载真实容量变化 |
| `WorkerBatcher` | 仅真实 Prefill queue seat 释放时通知 placement；scheduling input 更新只唤醒本地 batcher |
| `DecodeEndpoint` | 精确 reservation/permit 释放及真实容量改善后通知；不在抢占开始时通知 |
| `RequestLifecycleCoordinator/RequestSlot/ExpirationTimer` | 不增加状态；复用现有唯一 timeout、publication 与 cleanup 协议 |
| Selector / FixedWindow / delivery strategy | 不修改策略语义 |

## 14. 实施顺序

1. 重写等待层数据结构和单 lane 激活协议，保留现有外部行为入口；
2. 在 `RequestScheduler` 分配稳定 order，接入提交前 predecessor gate；
3. 让 `QueueRouteAdmission` 提供精确竞争资源键；
4. 删除 WorkerStatus/scheduling-input 的伪 placement signal，补齐精确资源释放信号；
5. 清理等待层不再需要的 pool-wide/limited scan 语义；
6. 运行 FlexLB UT；
7. 在 `luoli_gpu` 中运行 750P/750D 压测，重点验证 Master 决策 CPU、attempts/request、candidate evaluations、恢复延迟、Context Wall TPS 和 batch 质量。

## 15. 验收不变量

必须同时满足：

1. 无容量时保留原 `BalanceContext`、future、absolute deadline、priority 和 sequence；
2. 普通 WorkerStatus 心跳产生 0 次 pending placement；
3. 一个资源事件只激活相关 lane，不复制或广播 backlog；
4. 每个 lane 同时最多一个 active request；
5. 同一竞争域按配置 priority/FIFO 顺序；
6. 每个 attempt 使用当前配置 Selector 完整重选 required roles；
7. 成功后不因普通状态更新重新选机；
8. 失败、超时、取消、抢占和 shutdown 均无 P/D pin、reservation、queue item、wait entry 泄漏；
9. FixedWindow 的窗口和 batch 质量语义不变；
10. 压力越大，Master 只处理真实容量对应的有效尝试，不形成重试正反馈。
11. PRIORITY/抢占模式的一份 Decode queued reservation 只占一个 placement slot；释放一个 slot 最多产生一个成功和一次容量边界探测。
