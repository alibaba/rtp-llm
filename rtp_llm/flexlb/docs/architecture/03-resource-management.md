# Resource Management

容量由 endpoint 的本地预留与引擎观测共同决定。`EndpointRegistry` 管理 worker 代际，
`PrefillState`、`DecodeState`、`EncoderEndpoint` 分别管理各阶段的资源；
`RequestContext` 保存请求生命周期，`AbstractRequestScheduler` 将状态变化投影到资源账本。

## Worker 代际与地址

`EndpointRegistry` 按角色和 `ip:httpPort@engineIndex` 管理逻辑 worker，包括 N=1 的 `@0`。
网络 RPC 使用物理地址与相应的 frontend/status 端口。选点先读取不可变快照，再取得对应代际的
`GenerationPin`；代际变化时重新选点。退休先关闭路由入口，再在锁外等待派发和清理完成。
旧回调只能修改与其 WorkerStatus、请求 ID、reservation token 匹配的记录。

## Prefill 队列与容量

QUEUE 请求进入选定 Prefill 的 `PrefillQueueRunner`，由 `PrefillState` 保存队列、正在准备和已提交的批次。
`GroupingPolicy` 使用同一份约束完成实际凑批与路由时的排队预测：最大请求数、Token/KV 容量、
预测执行时间和收集窗口。引擎没有报告 Token 上限时使用配置的 fallback batch token capacity。
收集窗口可动态更新；已经提交的 DecisionGroup 保留当时的策略与成员。
Prefill 放置队列不以请求数量过滤缓存命中最优 worker，也不通过驱逐排队请求腾出入口。
QUEUE 先比较全体可调度 worker 的 cache 与预计工作量，请求并发上限在交付阶段执行；
DIRECT 仍在选定 worker 后检查其并发容量。

BATCH 在准备和发送时持有批次资源；NON_BATCH 由
`dispatcher.maxInflightPerPrefillWorker` 限制未确认的请求数。引擎有效状态将本地预留转为
已观测任务，避免重复记账。未明确发送的失败可以回滚；发送结果不确定时继续保留资源，
直到独立的完成或清理证据到达。

## Decode 预留与引擎执行

`DecodeResources.AdmissionCapacity` 统一检查最大引擎请求数、硬 KV 与输入加输出上限的 KV。
请求注册时冻结其配置，路由、预留、抢占和派发使用同一约束。
`tryReserveQueuedRequest` 在路由发布前预留并发与 KV；已经进入 Prefill 队列的请求继续占用
placement 容量，但不占引擎 dispatch 并发。真正派发前取得精确 reservation 对应的 dispatch permit。

有效 WorkerStatus 将预留转换为引擎确认记录。完整活动任务表中缺失已确认请求时，释放其容量，
同时保留原请求关联，标记 `waitingForWorkerFinishedReport`，等待迟到的 Finished、请求清理或代际退休。
重复缺失不会再次释放容量；任务重新出现时恢复实际容量。不能把“没有占用容量”当成“已收到完成”。

RPC 抢占先确认所有 victim 的请求与取消地址，再建立精确声明，收到独立的 Decode 释放证据后
提交 incoming 预留。取消 ACK 本身不能证明 Decode 已释放。
RETURN 模式在同一资源事务中声明所有 victim 并预留 incoming，将 victim ID 随路由返回；
incoming 发送前的本地回滚释放该声明，发送或引擎确认后继续等待 victim 的独立完成证据。

## Encoder

Encoder 有独立的阶段队列和 `EncoderEndpoint`，同一请求 ID 可同时存在 Encoder 与 Generation 记录。
首次 WorkerStatus 前按 `seq_len - encoder_cache_hit_len` 记录预计编码工作量；观测到任务后
使用去重的活动任务 `input_length` 替换预测值。全命中任务可能直接出现在 Finished 中，
其原始 prompt 长度不参与在途编码负载。完成、取消或失活会清除本地记录；Encoder 不发送生成阶段的 Cancel RPC。

## 观测

指标使用 `getMetricIpPort()`：单引擎为 `ip:port`，多引擎包含 `@engineIndex`。
`DeliveryMetricsReporter` 上报队列、批次与在途请求，`RequestSchedulerReporter` 上报准入、抢占、取消
和 Decode 分层容量。指标失败不能改变资源记账或业务结果。
