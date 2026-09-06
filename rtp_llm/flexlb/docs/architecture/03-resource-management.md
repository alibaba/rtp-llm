# Resource Management

当前实现没有全局的动态许可信号量。容量约束分为三层：

1. `PriorityScheduler` 用 `maxOutstandingRequestsGlobal` 限制 Master 拥有的 QUEUE 请求总数。
2. `ResourceMeasure` 在路由时判断单个 worker 是否可选。
3. `PrefillEndpoint` / `DecodeEndpoint` 用本地预留账本保护尚未反映到引擎
   `WorkerStatus` 中的调度决策。

主要代码：`balance/resource/`、`balance/endpoint/`、`DefaultRouter` 和
`PriorityScheduler`。

## EndpointRegistry

`EndpointRegistry` 按角色和 `ip:port` 维护 Prefill、Decode、P/D Fusion 与 VIT endpoint。
同一地址的 `WorkerStatus` 代际变化时，registry 原子替换 endpoint 并关闭旧实例；
过期 worker 只能用当时观察到的 `WorkerStatus` 对象条件删除，避免删掉同地址的
新代际。

## Prefill 可用性与队列

`PrefillResourceMeasure` 要求 endpoint 存活，且引擎观测到的有效待处理请求低于
`router.roles.prefill.availability.maxPendingRequests`（默认 64）。上下线切换使用
`router.availabilityHysteresisPercent`（默认 15）做滞回，避免在阈值附近抖动。

QUEUE scheduler 选中 Prefill 后，请求进入该 endpoint 的 `WorkerBatcher`。
`BatchDispatcherConfig.maxWaitingRequestsPerPrefillWorker`（默认 1024）是 batcher 等待队列的
硬上限，与路由层的 worker 可用性阈值是不同责任。

## Decode 容量与本地预留

`DecodeResourceMeasure` 要求 endpoint 存活，并同时检查：

- 引擎可见负载未超过可选的
  `router.roles.decode.availability.maxEngineRequests`。
- 真实 KV 使用率未超过 `maxKvUsagePercent`（默认 90）；同样使用
  `availabilityHysteresisPercent` 做滞回。

`DecodeEndpoint.reserve()` 在对外发布前建立 shadow 预留，记录并发槽位、硬 KV
和预期 KV。已在 Prefill 队列中的预留继续保护 KV，但不计入引擎面的
concurrency，避免引擎空闲时被本地排队预留假性压满。

WorkerStatus 到达后，`DecodeEndpoint.calibrate()` 用引擎已确认状态对账。对于取消或
发布结果不确定的请求，Engine fence 在权威终态到达前持有相应账本，防止
KV 或并发容量被提前释放并二次分配。

## 策略层使用

`ResourceMeasureFactory` 按指标注册 Prefill / Decode measure。`DefaultRouter` 及其策略先
排除不可用 endpoint，再根据等待时间、KV 负载、cache 亲和性和配置的候选集
选择 worker。QUEUE 路径成功后，endpoint 账本由 `PriorityScheduler` 负责提交、
回滚和终态释放。

## 观测

`BatchSchedulerReporter` 上报 endpoint 队列深度、等待时间、batch 大小、inflight、
Decode 负载和本地 KV 预留。`PrioritySchedulerReporter` 上报优先级准入、抢占、
取消和分层 Decode 负载。观测失败不得改变调度结果或资源所有权。
