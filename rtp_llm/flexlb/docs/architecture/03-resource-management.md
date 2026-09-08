# Resource Management

FlexLB 的容量由三类所有者共同表达：RequestRegistry 的全局准入、EndpointRegistry 的
generation 生命周期，以及 Prefill / Decode endpoint 的精确本地账本。

## Worker generation 与 EndpointRegistry

WorkerDirectory 将服务发现到的逻辑 worker 保存为每角色的 WorkerStatus generation。首次获得
可提交的 worker status 前，该 generation 不可路由。EndpointRegistry 随后创建对应的
PrefillEndpoint、DecodeEndpoint 或通用 WorkerEndpoint，并只公开这一份已初始化的 endpoint
generation。

逻辑地址是 ip:httpPort@engineIndex，N=1 也使用 @0。Registry 按角色和逻辑地址维护 endpoint；
状态失效或连续 transport failure 会先从路由目录 detach 旧 generation，再等待
所有已取得的 GenerationPin 完成，然后清理 endpoint 与 LOCAL_SYNC cache 元数据。

选择器看到的 routing directory 只是非拥有快照。真正提交前必须 capture 精确 pin；因此地址相同
的 replacement 无法继承旧 request 的 reservation，退役也不会打断已线性化的 handoff。

## Prefill 容量

PrefillEndpoint 通过 PrefillState 管理本 generation 的活动请求与 route projection：

- DIRECT 路径用 registerDirectRequest() 建立精确 direct registration。
- QUEUE 路径由 WorkerBatcher 将已提交的 request 放入 active index，并按 decision policy 形成
  delivery group。
- projection 将已拥有工作、引擎观测、cache 命中和本请求预估 prefill 工作组合为候选的 TTFT /
  drain time；这是 CostBasedPrefillStrategy 的输入。
- scheduler.capacity.maxWaitingRequestsPerPrefillWorker 是每个 batcher active queue 的硬界限。
  dispatcher 的 in-flight delivery 限制是独立边界，不应混作 worker 状态中的 pending 值。

Prefill 容量变化通过 EndpointEventProjector / PlacementAvailability 发布给全局队列。被某一
endpoint 精确容量阻塞的请求仅在该事件出现时重新规划。

## Decode 容量与防重分配

DecodeEndpoint 的 admission lock 管理以 request id、endpoint generation 和 reservation token
三元组标识的 shadow reservation。其 layered admission view 同时保留：

- 引擎已观测的 KV、并发和任务状态；
- master 已排队但引擎未必见到的 request；
- 已发往引擎、等待接受证明的 request；
- 已确认接受或运行的 request，以及因取消/超时仍受 engine fence 保护的 request。

选址时 CostBasedDecodeStrategy 读取 immutable routing view；提交时 reservePinned() 重新在
admission lock 下确认 capacity。reservation 估算 KV 为请求输入 token 加有效输出 token，并受
endpoint 总 KV 容量限制。maxKvUsagePercent 和可选 maxEngineRequests 是引擎面的 admission
边界；QUEUE 若没有能够替换 Decode victim 的 preemption policy，可将 transient Decode capacity
检查推迟到交付前。

取消或交付结果不确定时，不会乐观释放 Decode KV/并发。RequestRegistry 和 DecodeEndpoint
保留 fence，直到引擎接受、取消或终态证据使该精确 reservation 可结算，避免同一容量被二次分配。

## Preemption 与观测

只有 PRIORITY ordering 可配置 preemption。EvictionManager 和 DecodePreemptionCoordinator 基于
RequestRegistry 的 exact ownership 计划 victim；若允许取消引擎已拥有的 Decode 请求，还必须配置
engine cancellation ACK 与 completion timeout。FIFO 或未配置 preemption 的队列不会把临时
Decode 满载解释为可抢占事件。

HttpLoadBalanceServer 的 GET /rtp_llm/inflight_status 输出 Prefill 活动请求、Decode layered
admission view、KV reservation 和 dispatch permit 的诊断快照。BatchSchedulerReporter 和
RequestSchedulerReporter 上报队列、交付、准入、preemption 与 endpoint 账本指标；观测异常不得
改变资源所有权。
