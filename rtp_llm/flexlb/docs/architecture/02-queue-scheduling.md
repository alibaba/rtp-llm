# Queue Scheduling

FlexLB 支持两种路由模式，由 `FLEXLB_CONFIG.enableQueueing`（默认 **false**）控制：

- **直连模式**：`RouteService.route()` 在当前线程 `Mono.fromCallable(() -> router.route(ctx))`
  同步路由，无队列、无重试。
- **队列模式**：请求进入全局阻塞队列，由固定工作线程池异步消费，支持重试 / 取消 / 超时。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/QueueManager.java`、
`RequestScheduler.java`、`service/RouteService.java`。

## QueueManager

单个全局队列：`BlockingDeque<BalanceContext>` = `LinkedBlockingDeque`，容量
`maxQueueSize`（默认 1,000,000）。`AtomicLong sequenceGenerator` 为每个请求分配
`sequenceId`（仅用于快照观测，不用于取消）。

### tryRouteAsync(ctx)

1. 创建 `CompletableFuture<Response>` 挂到 ctx；设置 `enqueueTime`、`sequenceId`。
2. 非阻塞 `offerLast(ctx)`；失败（队列满）→ 上报 rejected 指标，**立即**返回
   `Response.error(QUEUE_FULL)`，不完成 future。
3. 成功 → 返回 `Mono.fromFuture(future)`：
   - `.timeout(request.generateTimeout)`（默认 3,600,000 ms）——**调用侧的主动超时**；
   - `.onErrorResume(handleQueueException)`：`TimeoutException` → 从队列移除 + `QUEUE_TIMEOUT`；
     `CancellationException` → 移除 + `REQUEST_CANCELLED`；`InterruptedException` → 移除 +
     `QUEUE_TIMEOUT`；其他 → `NO_AVAILABLE_WORKER`。

**没有独立的超时清扫线程**：超时由 Reactor `.timeout()` 主动触发 + 工作线程出队时懒检查双保险。

### takeRequest(isBlock, blockTimeoutMs)

循环 `poll()`：设置 `dequeueTime`；若 `isCancelled()` → `future.completeExceptionally
(CancellationException)` 跳过；若排队等待已超过 `generateTimeout` →
`completeExceptionally(TimeoutException)` 跳过；否则上报排队等待时间并返回。

### 快照与指标

- `snapshotQueue()`：遍历队列写 JSON 文件到 `/tmp/flexlb-queue-snapshots/`（最多保留 10 个），
  暴露于 `GET /rtp_llm/queue_snapshot`，返回 `{filePath, timestamp, count}`。
- `@Scheduled(fixedRate=1000)` 上报队列长度（`RoutingQueueReporter`）。

## RequestScheduler

- `@PostConstruct`：`Executors.newFixedThreadPool(scheduleWorkerSize)`（默认 = CPU 核数），
  daemon 线程名 `routing-queue-worker`，提交 `scheduleWorkerSize` 个 `workerLoop`。
- **workerLoop**：`running` 期间循环——
  1. `dynamicWorkerManager.tryAcquirePermit(500ms)`（资源许可门控，见
     [03-resource-management](03-resource-management.md)）；拿不到就 continue；
  2. `queueManager.takeRequest(true, 500)`；空则 continue（finally 释放许可）；
  3. `processRequest(ctx)`。
- **processRequest 重试**：`while (!future.isDone())` 循环调 `router.route(ctx)`：
  - 成功或不可重试错误 → `future.complete(response)`；
  - 可重试（`StrategyErrorType.isCanRetry()`：`NO_AVAILABLE_WORKER` 及 4 个 `NO_*_WORKER`）→
    `incrementRetryCount()` + `Thread.sleep(routingRetryIntervalMs)`（默认 10ms，固定间隔无
    退避）后重试；`maxRetryCount` 默认 0 = 不限次（实际上界是 generateTimeout 触发 future 完成，
    循环条件自然退出）。
  - **重试发生在工作线程内部**，请求不回队列（不存在 `offerToHead` 之类的头部重排机制）。
- **停机**：`@PreDestroy` → `running=false` → `shutdown()` → `awaitTermination(10s)` →
  必要时 `shutdownNow()`。

## RouteService

- `route(ctx)`：注入 `FlexlbConfig` 后按 `enableQueueing` 分流（见上）；两条路径都
  `doOnSuccess(ctx::setResponse)`。
- `cancel(ctx)`：**按 BalanceContext 引用取消**（不是按 sequenceId）——`tryCancel()`
  原子置位，并在首次取消时回滚已路由结果、完成队列 future。触发点唯一是
  客户端订阅取消时 `RouteService.route()` 的 `.doOnCancel()`。

## 队列模式完整生命周期

1. `POST /rtp_llm/schedule` → `HttpLoadBalanceServer.scheduleRequest`（`ActiveRequestCounter`
   计数，master 转发判断）→ `prepareBlockCacheKeys(ctx).then(routeService.route(ctx))`。
2. `tryRouteAsync`：建 future、`offerLast`（满 → `QUEUE_FULL`）、返回带 timeout 的 Mono。
3. `routing-queue-worker` 线程：许可门控 → 出队（懒检查取消/超时）→ `processRequest`。
4. `DefaultRouter.route()` 多角色路由（失败回滚见
   [01-routing-and-balancing](01-routing-and-balancing.md)），`NO_*_WORKER` 类错误按
   10ms 间隔原地重试。
5. `future.complete()` → 调用方 Mono 链恢复 → HTTP 响应。
6. 客户端断连：`RouteService.route()` 的 `.doOnCancel` → `RouteService.cancel(ctx)`；仍在队列则被 `handleCanceled`
   移除，已被工作线程取出则在出队检查时丢弃。

## BalanceContext 队列相关字段

| 字段 / 方法 | 定义 |
|---|---|
| `future` | `CompletableFuture<Response>`（注意：泛型是 Response，不是 BalanceContext） |
| `cancelled` / `cancel()` / `isCancelled()` | `AtomicBoolean`，CAS 置位 |
| `retryCount` / `incrementRetryCount()` | `AtomicInteger` |
| `enqueueTime` / `dequeueTime` / `sequenceId` | long；sequenceId 仅用于快照观测 |

## 错误码

`flexlb-common/.../dao/loadbalance/StrategyErrorType.java`：

| 错误 | 码 | canRetry | 产生位置 |
|---|---|---|---|
| `QUEUE_FULL` | 8502 | 否 | `tryRouteAsync` offer 失败 |
| `QUEUE_TIMEOUT` | 8503 | 否 | Reactor timeout / 出队懒检查 / 中断 |
| `REQUEST_CANCELLED` | 8504 | 否 | 取消路径 |
| `NO_AVAILABLE_WORKER` | 8400 | 是 | 校验失败 / 策略无候选 |
| `NO_PREFILL/DECODE/PDFUSION/VIT_WORKER` | 8402–8405 | 是 | 对应角色路由失败 |
| `INVALID_REQUEST` | 8406 | 否 | 请求为空 |
| `CONNECT_FAILED` / `CONNECT_TIMEOUT` | 8202/8203 | 否 | 转发 master 失败 |

## 相关配置（默认值）

`enableQueueing=false`、`maxQueueSize=1000000`、`maxRetryCount=0`（不限）、
`routingRetryIntervalMs=10`、`scheduleWorkerSize=CPU核数`；队列等待上限即请求自带的
`generateTimeout`（默认 1 小时），无独立队列超时配置；工作线程 poll / 许可等待 500ms 为硬编码。
