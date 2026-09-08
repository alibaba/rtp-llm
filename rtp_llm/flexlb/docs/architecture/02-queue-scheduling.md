# Scheduling and Request Lifecycle

FLEXLB_CONFIG.scheduler 是带 type 的配置：

- DIRECT：RouteService 在调用线程执行 DefaultRouter.routeDirect()；没有队列和 endpoint
  batcher 工作线程。
- QUEUE（默认）：RequestScheduler 是唯一公共入口，RequestRegistry 拥有请求生命周期，
  GlobalQueueCoordinator 决定全局顺序与选址，WorkerBatcher 只处理已经选定 endpoint 的
  分组及交付。

QUEUE 下有三个彼此独立的配置维度：

| 维度 | 配置 | 责任 |
|---|---|---|
| 排序 | scheduler.ordering：FIFO 或 PRIORITY | GlobalQueueCoordinator 的全局队列顺序；PRIORITY 可配置 preemption |
| 决策分组 | scheduler.decision：SINGLE 或 FIXED_WINDOW | 每个 Prefill WorkerBatcher 如何形成一份交付决策 |
| 交付 | dispatcher：BATCH 或 NON_BATCH | EnqueueBatch，或向调用方交付单个 route decision |

## 准入与全局决策

RequestScheduler.submit() 在 ingress 线程只做同步注册和入队，不在此处扫描 worker：

1. 读取当前配置，拒绝非 QUEUE 配置或在启动后才切换到 QUEUE 的实例。
2. RequestRegistry.register() 以 request_id 建立唯一 RequestSlot，检查重复 id 和绝对 deadline，
   并取得 maxOutstandingRequestsGlobal 限制的全局准入 permit。容量满时，PRIORITY 可以用
   符合配置的可逆 victim 转移 permit；否则返回 QUEUE_FULL。
3. 将同一个 future 和 context 写入 GlobalQueueCoordinator。future 被取消或完成会 O(1) 移除
   对应队列节点，不能留下阻塞后缀的历史节点。

GlobalQueueCoordinator 使用一个决策线程和受 internalRuntime.queuePlannerThreads 限制的规划池：

- FIFO 以入队序列排序；PRIORITY 以 priority 降序、入队序列升序、request id 为最终稳定 tie-break。
- 决策线程从全局队列提取有限 planning frontier，在锁外调用 DefaultRouter.routeForQueue()。
  路由和 RPC 不在队列锁内执行。
- 计划必须按队列顺序提交。若高优先级请求在计划期间抵达，当前 frontier 会丢弃并重新捕获。
- 无法提交的请求按精确 PlacementKey 停放；仅当对应 endpoint 容量事件变化时才重新参与决策。
  不重试不相关的 blocked request，且不同 endpoint/group 可以独立前进。
- 计划在 endpoint 提交时再次验证 generation、容量和 deadline。陈旧计划关闭自己的 pins 并重规划，
  不会把旧快照发布到 endpoint。

## Endpoint 运行时与交付

每个已发布 Prefill generation 有一个 WorkerBatcher。它不是 route selector，而是该 endpoint 的
活动请求索引、分组和交付所有者：

- SINGLE 每份决策只包含一个请求；FIXED_WINDOW 在 maxRequests、maxCollectionWaitMs 或可选
  maxPredictedExecutionMs 条件满足时形成决策组。
- BATCH 由 BatchDeliveryStrategy 和 DefaultBatchDispatcher 准备 EnqueueBatch，并在引擎 ACK
  后把 batch delivery 状态交给 RequestRegistry。
- NON_BATCH 由 RouteDeliveryStrategy 交付 route decision；调用方确认或 lifecycle 超时决定后续
  资源归属。
- dispatcher 的 per-Prefill in-flight 限制区分 batch 和 request；队列长度硬上限位于
  scheduler.capacity.maxWaitingRequestsPerPrefillWorker。

在 QUEUE 路径中，Prefill/Decode 的 reservation、delivery claim 与完成 future 都由
RequestRegistry 的精确 RequestSlot reducer 连接。batcher 不拥有全局 admission、request id
去重、取消或终态回收。

## 取消、过期和状态查询

RequestRegistry 是以下状态变化的唯一 reducer：

- client cancel、deadline、排队超时、delivery ACK/拒绝、引擎接受/运行/终态；
- priority preemption 与可能的引擎取消 fence；
- stale inflight 和已交付未确认请求的清理；
- 关闭时停止新准入、等待已跨过 admission 边界的 mutation，然后终止剩余 request generation。

每个 request generation 都有绝对 deadline；队列决策点与 endpoint 发布点都会再次检查，防止
延迟定时器将已过期请求交付给引擎。取消携带 expectedBatchId 作为 generation fence，避免旧
取消命中复用 request id 的新请求。

FlexLB gRPC 还提供 GetRequestState 和 Cancel。启用一致性时，follower 对这两个调用同样尝试
单跳转发到 master；没有 master 地址时才本地处理。见
[05-lifecycle-and-consistency](05-lifecycle-and-consistency.md)。

## 默认配置

- scheduler：QUEUE、FIFO、queueTimeoutMs=3,600,000；全局准入为 100,000。
- decision：FIXED_WINDOW，maxRequests=8，maxCollectionWaitMs=300。
- dispatcher：BATCH，enqueueRpcTimeoutMs=5,000；未设置时 per-Prefill in-flight 限制不额外收紧。
- lifecycle：staleInflightTimeoutMs=300,000，deliveredNotAcceptedTimeoutMs=30,000，
  maxDeliveredNotAcceptedRequestsGlobal=200。
- 每个 Prefill endpoint 等待队列上限：1,024。
