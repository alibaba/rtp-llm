# gRPC 与状态同步观测口径

网络大小按成功 RPC 响应的 protobuf `getSerializedSize()` 字节数计算，不是线上压缩后的网络流量，也不是对象在堆中的占用。失败不填零字节。

## 复用指标

| 指标 | 口径 / 类型 | 维度 | 可定位的问题 |
| --- | --- | --- | --- |
| `app.grpc.call.count` | 每个逻辑调用最终完成一次；重试不另算调用；Counter | engineIp、enginePort、role、service、retry、code | 哪个引擎、方法成功/失败；重试后结果；QPS 用 counter 的 rate |
| `app.grpc.call.duration` | 开始到最终 RPC 完成，包含重试；Timer | 同上但不含 code | RPC 耗时，和本地提交耗时分离 |
| `app.grpc.response.size` | 仅成功响应的序列化字节数；DistributionSummary | engineIp、enginePort、role、service、retry | 响应数、累计字节数、固定分桶分布及滑动窗口最大值 |
| `app.grpc.channel.pool.size` / `app.grpc.connection.duration` | 保留原有语义和维度 | 原有维度 | 连接池规模与建连耗时 |
| `app.engine.balancing.thread.pool.info` | active、queue、pool size；Gauge | threadPool、type | 复用物理池占用和队列深度；健康/缓存分池 |
| `app.engine.balancing.event.loop.group.info` | 保留 pending task 和未关闭 executor 数；2 秒采样 | name、type | EventLoop 队列；active-executor-count 不是忙线程数 |
| `flexlb.grpc.server.executor.*` | 保留服务器原有 Gauge / FunctionCounter | 无 | server 回调池占用、完成、拒绝；caller.runs 历史名称仍表示 AbortPolicy 拒绝 |
| 原有健康/缓存成功 QPS、成功周期、cache 增删 Gauge | 保留 | 原有维度 | 历史面板延续；成功提交周期不是 RPC RT |

`call.duration` 从 Gauge 改为 Timer，`response.size` 从 Gauge 改为 Summary，名称复用但 Prometheus 序列形态发生变化：Timer 导出 seconds 的 count/sum/bucket/max，Summary 导出 bytes 数值的 count/sum/bucket/max。原 Gauge 查询需改为新序列。累计 bytes/count 为平均响应大小，rate(bytes sum) 为成功响应字节率。时间指标的入口单位为 ms，Micrometer Timer 导出的 `_seconds_*` 为秒。

## 新增指标

| 指标 | 口径 / 类型 | 维度 | 可定位的问题 |
| --- | --- | --- | --- |
| `app.sync.convert.ms` / `app.sync.callback.queue.ms` / `app.sync.apply.ms` | 响应到转换完成 / 转换完成到回调开始 / 回调开始到结束；Timer | engineIp、enginePort、role、service=status/cache | 分离转换、队列、锁与提交；apply 含锁等待和 GC，不是纯 CPU |
| `app.sync.phase` / `app.sync.phase.age.ms` | 0 idle、1 RPC、2 conversion、3 pending、4 applying；当前阶段墙钟年龄，Gauge | 同上 | 尚未完成的阻塞不会因没有完成样本而显示正常 |
| `app.sync.response.age.ms` / `app.sync.commit.age.ms` | 最后合法响应的收到时间 / 最后成功提交时间距今；Gauge；未取得有效值和已消失 Worker 为 NaN | 同上 | 引擎已响应但本地迟迟未更新；response 合法性在转换完成后确认，时钟回指响应收到时间 |
| `app.executor.queue.wait.ms` / `app.executor.task.run.ms` | 任务入池到运行 / 运行到结束，含 CallerRuns 的墙钟时间；Timer | threadPool | client、forwarder、server、发现、健康、缓存池的排队和占用时间 |
| `app.executor.oldest.pending.ms` / `app.executor.oldest.running.ms` | 队头任务等待 / 池内仍运行任务最长时间；1 秒 Gauge | threadPool | 线程被卡住时观察未完成任务；CallerRuns 不属于池内运行线程 |
| `app.executor.queue.capacity` | 当前队列大小 + 剩余容量；Gauge | threadPool | 队列容量与瞬时拒绝日志对照 |
| `app.executor.rejected.total` / `app.executor.caller.runs.total` / `app.executor.completed.tasks` | 进程生命周期累计值；Gauge | threadPool | 拒绝与 CallerRuns 分开；服务器原有 FunctionCounter 同时保留 |
| `app.netty.event.loop.lag.ms` | 每组最大探针排队用时；未完成探针报等待年龄；2 秒采样 | name | EventLoop 调度卡顿；每个 loop 至多一个待执行探针 |
| `app.jvm.heap.used.bytes` / `app.jvm.heap.max.bytes` / `app.jvm.direct.used.bytes` | JVM MXBean 及 direct BufferPool；1 秒 Gauge | 无 | 堆 / direct 压力；不是 RSS 或所有 native 内存 |
| `app.jvm.gc.pause.ms` | 每次 GC notification 的持续时间；Timer | collector | GC 次数、持续时间；逐次 GC 的时刻、原因在日志中 |
| `app.sync.object.entries` | cache 快照 key、状态任务、local key、global key/mapping 条目数；1 秒 Gauge | object_kind | 持久状态规模；status 含已提交 running tasks 和在处理响应中的 finished tasks |
| `app.sync.object.entry.budget.bytes` | `cache/local key * 64`，`status task * 256`，`global key * 64 + mapping * 48`；Gauge | object_kind、estimate_method=cardinality_budget_v1 | 仅条目基数预算，排除变长字符串、共享引用、临时 diff 集合和 JVM 布局；不能证明 GC 根因或替代 retained heap |
| `app.sync.pending.response.bytes` | 尚在转换/队列/提交阶段的响应序列化 bytes 之和；1 秒 Gauge | 无 | 尚未处理的响应负担；不是实际堆占用 |
| `app.cache.diff.scanned.keys` / `.added.keys` / `.removed.keys` / `.compute.ms` | 每次已完成 diff 的旧集合+新集合大小、实际增/删量及计算墙钟用时；Counter / Timer | engineIp、role | 真实 diff 工作量；卡住未完成的 diff 由 phase / task age 辅助识别 |
| `app.grpc.metric.binding.overflow` / `app.sync.metric.binding.overflow` | RPC 绑定降级次数 Counter / Worker 代际绑定失败累计 Gauge | 无 | 超过每类 4096 个历史地址的观测覆盖缺口；RPC 降级至 overflow，Worker 不导出伪造单引擎数值 |
| `app.jvm.gc.events.dropped` | 128 条固定 GC 环缓冲写满丢弃累计值；Gauge | 无 | GC 日志缺口，不假装采集完整 |

Timer 使用有限 SLO 桶，不在每个引擎上分配 HDR percentile 直方图。byte Summary 使用 1KiB、16KiB、256KiB、1/4/16/64/256MiB 桶，含 +Inf；max 是 Micrometer 时间窗口最大值，瞬时响应大小在 snapshot 日志中。Worker generation、version、GC id 只进日志，不进持久指标标签。

单引擎定位沿用默认开启的 engine tags；显式配置 `flexlb.monitor.per-engine-tags=false` 会删除 engineIp 和 enginePort，日志仍保留地址，单引擎 Gauge 不再有定位意义，不得当作全舰队最大值。启用 metric-whitelist 时必须加入所需的新指标，否则不会导出。标签预绑定后的白名单按启动配置使用，运行中不动态切换。

## 三类结构化日志

1. `snapshot_changed` / `cache_diff` / `sync_slow`：地址、role、generation、版本、条目数、response bytes、扫描/增删量、分段耗时、状态年龄。snapshot / diff 每地址最多 1 条/秒；slow 每地址每服务最多 1 条/10秒，含 suppressed。
2. `rpc_failed` / `executor_rejected` / `worker_retiring` / `cache_apply_failed`：最终 grpc code、原因分类、失败类别、过期时响应和提交年龄、阶段；拒绝瞬间包含 queue size/capacity、active、最长等待/运行、heap used。RPC/池异常各最多 1 条/10秒；真实生命周期事件保留。已存在的 WorkerStatus 创建日志用于关联恢复。
3. `gc_state`：GC id、collector、cause、起止 epoch、pause、heap before/after、最近条目数/byte 预算、pending response bytes、状态采样时间和距 GC 结束的年龄。GC 线程只拷贝标量至固定环缓冲，独立采样线程写日志；大于一秒的 STW 会让状态样本变旧，时间字段必须保留。

监控不持有 protobuf、集合、任务详情或 key 内容，不主动 GC、不做 heap 遍历、不逐 key 打点、不按请求输出新日志。每个 executor task 添加一个小计时包装；标签与 meter 预绑定复用。每秒采样复杂度是 Worker 和线程数，队列仅 peek，不扫描队列；remove/purge/shutdownNow 仍保持原有 Future / 取消语义。
