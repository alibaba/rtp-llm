# FlexLB QUEUE 全局决策点设计

> 日期：2026-08-30
> 范围：单模型 QUEUE 模式的排序、选机、TTL、准入和交付

## 1. 目标和边界

QUEUE 模式只有一个全局有序入口。请求在这个入口完成一次完整的 P/D
选机和精确资源提交，然后交给已选中的 Prefill endpoint 做本地成组和交付。
这样可以去掉旧的“每个 worker 各自选机 + 全局等待表再次选机”的两阶段路由。

四个配置维度保持正交：

| 维度 | 作用 | QUEUE 实现 |
| --- | --- | --- |
| `scheduler.type` | 是否进入队列 | `QUEUE` 使用全局队列，`DIRECT` 保持直达路径 |
| `ordering` | 队列顺序 | `FIFO`；或 1--100 优先级降序、同优先级 FIFO |
| `decision` | endpoint 内如何成组 | `SINGLE`；或 `FIXED_WINDOW` |
| `dispatcher` | 成组后的交付方式 | `NON_BATCH` 单请求交付；或 `BATCH` `EnqueueBatch` |

全局队列负责 ordering 和 placement decision；endpoint `WorkerBatcher` 只负责
已选 endpoint 上的 decision window、KV/Token 边界和 dispatcher handoff。
`decision` 不会改变全局顺序，`dispatcher` 也不会重新选机。

## 2. 运行结构

```text
并发 Ingress
    -> RequestRegistry.register
    -> GlobalQueueCoordinator.offer

GlobalQueueCoordinator
    -> intrusive FIFO bucket / 101 个 priority bucket
    -> 有界 Planner Pool：对全量 live endpoint 计算候选、cache、TTFT/TTL 投影
    -> 顺序 Commit：校验 generation、预留 P/D、发布 ScheduledRequest

选中的 PrefillEndpoint
    -> WorkerBatcher：SINGLE 或 FIXED_WINDOW
    -> RouteDeliveryStrategy 或 BatchDeliveryStrategy
```

Ingress 不执行 RPC、不等待容量、不扫描队列。Global queue 只有一个短的
ordered commit 点；候选计算在 planner pool 中并行，提交仍按队列顺序线性化。
Planning frontier 的请求预算来自全 fleet 当前 delivery credits 与
`scheduler.capacity.maxOutstandingRequestsGlobal` 的较小值；它不代表 decision
group。实际同时计算的 route plan 不超过 planner pool 大小。线程数来自
`InternalRuntimeSettings.queuePlannerThreads`（默认值为 JVM 可见 CPU 核数，可用
`flexlb.queue.planner.threads` 显式覆盖）。PRIORITY 在每次 commit 前重新确认没有
更高优先级请求；需要重新读取容量或排序 frontier 时，尚未提交的 plan 会先关闭。
若某请求在 commit 时只与一个具体
engine 的本地容量冲突，就把它按 endpoint park；后续不使用该 endpoint 的
请求可以先提交，同一 endpoint 的请求仍然等待容量事件。selector 层没有具体
endpoint 的 pool-wide miss 不能安全绕过队头。每次 route decision
都会读取该 role 的完整 live endpoint snapshot；look-ahead 只限制同时准备的
请求数，不限制机器候选数，因此不会因 cursor 窗口漏掉全局最优机器。

FIFO/PRIORITY 的顺序仍是默认提交顺序；endpoint-local capacity conflict 是唯一
明确允许的绕行条件，不通过隐藏 fallback 破坏顺序。优先级抢占只在
`EvictionManager` 的现有协议中作为一次明确的 priority rescue，不形成多层
重试链。

## 3. 队列和生命周期

FIFO 使用 intrusive FIFO bucket；PRIORITY 使用 101 个 intrusive FIFO bucket
和 `BitSet`（有效优先级为 1--100）。入队、取最高优先级、同优先级 FIFO，以及
已知 entry 的取消/完成删除都是 O(1)。任意位置删除会立即 unlink，不保留
tombstone，也不扫描大队列。

`RequestRegistry` 是唯一的 request lifecycle owner，继续负责：

- absolute deadline 和 TTL timer；
- generation、future、admission mutation；
- cancel、terminal response 和 cleanup；
- 已发布请求的精确 queue item / Decode reservation 释放。

Global queue 的 entry 只保存 context、future、priority 和 queue-local 状态，
不增加 `WAITING_P`、`WAITING_D`、`RETRYING`、`PREPARED` 等生命周期状态。

## 4. 一次统一 placement decision

对队头（以及 bounded look-ahead 中的后续请求）执行：

1. 读取当前 worker、cache 和 delivery projection；
2. 用现有 `DefaultRouter.routeForQueue` 完整选择 required roles；
3. 在 `QueueRouteAdmission.tryPublish` 中校验 generation、P queue seat、
   Decode reservation 和全局 acceptance cap；
4. 成功后一次性发布 `ScheduledRequest` 到选中的 Prefill endpoint；
5. 关闭 admission mutation 和 generation pin 的临时所有权。

任何失败都关闭本次 pins/reservation，不把部分 P/D 选择带到下一次尝试。
成功的 placement 不会因普通 WorkerStatus 心跳再次选机。

## 5. TTL、拒绝和容量事件

TTL 的拒绝分为两类：

- Ingress 时已经过 absolute deadline：`RequestRegistry` 立即返回
  `BATCH_SLO_EXPIRED`；
- 每次进入 placement decision 时都会先检查 absolute deadline；timer 即使延迟，
  已过期请求也不会扫描 fleet 或发布到 endpoint。等待容量期间到达 deadline
  时由 lifecycle 以 `DEADLINE_EXCEEDED` 终止。

过期的 projection 不用于提前拒绝。路由 projection 可以参与候选排序和
TTFT 估计，但不能因为旧的 estimated wait 错误拒绝请求；真正的容量和
generation 校验只在 commit 点确认。

`PlacementAvailability` 只发送可能改变真实 placement 结果的 versioned 事件：
endpoint generation 发布/替换/移除、Prefill publication credit 真实释放、Decode
exact reservation/permit 真实释放。exact endpoint 事件同时推进 group/role 的
version，但 listener 只接收一次物理事件；exact waiter 不会因同组其他机器释放
而惊群。普通心跳、预测器变化、窗口计时器和“开始抢占”不会唤醒全局队列。
park 前在 coordinator 锁内复查事件 sequence，关闭 check/park 的 lost-wakeup
窗口。

## 6. decision 和 dispatcher

Global queue 发布的是已经固定 P/D endpoint 的 `ScheduledRequest`，不在全局层
构造 batch，也不执行 collection wait。endpoint `WorkerBatcher` 是唯一的 group
owner，随后按配置执行：

- `SINGLE`：立即处理队头；
- `FIXED_WINDOW`：按 collection wait、最大请求数、token/KV 和预测边界成组；
- `NON_BATCH`：每个请求走 `RouteDeliveryStrategy`；
- `BATCH`：一个已提交 group 走一次 `EnqueueBatch`，由
  `BatchDeliveryStrategy`/`DefaultBatchDispatcher` 异步完成。

每个 endpoint queue 的窗口和 delivery capacity 是独立的本地运行时状态，
不会重新调用 selector，也不会修改 global ordering。全局 speculative planning
最多保持 `queuePlannerThreads` 个计算在途，并在每个顺序 commit 后滚动补充；
commit 紧邻 exact reservation。只有 placement version 明确证明 selection stale
时才重新规划；稳定的 exact capacity miss 直接 park，不用固定次数 retry。

## 7. 关闭和异常

关闭顺序为：停止 global queue 接受新 entry，移除 availability listener，
停止 planner/decision 线程并完成尚未发布的 future；随后由
`RequestRegistry` 等待 admission mutation 静默、终止 outstanding request，
最后关闭 endpoint runtime 和 publisher。

planner、availability listener、telemetry 和 dispatcher 的异常彼此隔离；单个
请求的 route failure 只终止该请求，不让队列线程重试同一坏状态或停止整个
调度器。

## 8. 代码边界

- `RequestScheduler`：QUEUE facade，只做配置检查、lifecycle register 和入队。
- `GlobalQueueCoordinator`：全局 ordering、bounded planning、按 endpoint
  conflict 的 commit/bypass 和 blocked wakeup。
- `OrderedRequestQueue`、`BlockedRequestIndex`：intrusive ordering index 与 exact
  endpoint waiter/selector frontier。
- `DefaultRouter`：现有候选策略和 required-role 选择，不持有队列状态。
- `QueueRouteAdmission`：P/D exact ownership 和一次性 publication。
- `WorkerBatcher`：已选 endpoint 的本地 decision window 和 delivery handoff。
- `RequestRegistry`、`DecodeEndpoint`、`PrefillState`：精确生命周期和资源释放。

旧的全局等待表、嵌套 placement request、predecessor 扫描和多阶段 retry
状态不再存在，也不应以兼容入口或生成物形式重新引入。
