# Worker Sync and KV Cache

worker 状态同步与 cache 元数据采用不同的提交边界：WorkerDirectory / EndpointRegistry 负责
可路由的 worker generation；CacheAwareService 编排 cache 查询和元数据更新。两者都使用
logical worker identity ip:httpPort@engineIndex。

## Worker 状态同步

MasterEngineSynchronizer 在进程启动时读取 workerRegistry 配置，并以
statusPollIntervalMs（默认 20ms）启动定时轮询。它为模型拓扑中的每个 required role 提交
EngineSyncRunner；同步和 status-check executor 的线程数来自 internalRuntime（默认各 32），
队列容量为 15,000。status RPC timeout 默认 5,000ms，worker status 超过默认 10,000ms 未更新
会被视为陈旧。

EngineSyncRunner 每轮执行以下工作：

1. 通过 WorkerAddressService 获取 role 的发现结果。一个 frontend 按 endpoint 的
   multi_engine_num 展开为 N 个 logical worker；N>1 使用 worker_status_port + index，
   N=1 未配置该端口时使用发现到的引擎 gRPC port。
2. 对每个逻辑地址在 WorkerDirectory 中取得或创建 WorkerStatus generation。发现本身不创建
   可路由 endpoint。
3. 以 WorkerStatus.PollLease 的 CAS gate 确保同一 worker 同时至多一个 status poll；KVCM
   未启用时，cache poll 另有独立 gate。
4. 对消失的发现条目保留一个与轮询间隔相关的宽限期；宽限期后启动该 generation 的 retirement。

GrpcWorkerStatusRunner 将响应先冻结为 WorkerStatus.StatusObservation。严格更大的
statusVersion 才会生成 PreparedStatus；endpoint reducer 用该不可变 observation 更新自己的
账本，全部 reducer 成功后才通过 publishPreparedStatus() 原子提交新的 EngineObservation 和
cursor。相同版本的 heartbeat 只投影活跃任务事实，不重放版本化的 mutation。

连续三次 status RPC transport failure、空/非法状态或 reducer 失败都会使该 generation 退役。
retirement 先从 EndpointRegistry 移除路由入口，等待已发出的 GenerationPin，再清理目录和本地
cache。状态同步成功后，physical-group health 要求一个共享 frontend 的所有 engine sibling
都已发布且 reportedAlive，才能被路由器使用。

WorkerStatus 公开的并不是可变的 localTaskMap。它分别发布 topology、不可变
CommittedWorkerStatus（引擎容量、KV、运行任务和 cursor）与 PollHealth。调度的本地请求、
Prefill 队列和 Decode fence 由 endpoint / RequestRegistry 持有，详见
[03-resource-management](03-resource-management.md)。

## LOCAL_SYNC cache 元数据

当 cacheMatching.type=LOCAL_SYNC 时，GrpcCacheStatusCheckRunner 按动态间隔获取
PREFILL/PDFUSION 的 cache status。它仍受独立 poll lease 保护，且只将版本变化的 cached keys
交给 CacheAwareService.updateFromWorkerStatus()。

LocalSyncCacheMatchProvider 的实现由以下两级索引组成：

- GlobalCacheIndex：block hash 到 logical worker 集合的倒排索引；
- EngineLocalView：logical worker 到 block hash 集合的正排索引；
- KvCacheManager：对两级索引执行 diff、更新、移除与前缀匹配。

匹配依请求 block 链顺序进行：候选 worker 在第一个 miss 后被淘汰，结果是连续前缀长度。动态
间隔服务根据 cache diff 大小调整下一次轮询，默认目标 diff 为 30，间隔范围 50ms 到 3,000ms。
KVCM 启用时 LOCAL_SYNC 元数据轮询和更新关闭。

## KVCM 与 Local Standby

cacheMatching.type=KVCM 时，KvcmCacheMatchProvider 使用 KvcmGrpcClient 访问当前 KVCM leader，
按 role、group、block size 和 worker kvCacheGroupMode 查询 host 前缀命中。leader 刷新、请求
timeout、重试、心跳和恢复阈值都由 KVCM 配置控制；默认请求 timeout 为 500ms，leader 刷新为
10 秒。

KVCM 模式仍维护 Local Standby，但它不是 worker cache status 的副本：

- LocalStandbyCacheMatchProvider 只从已经成功路由的 PREFILL/PDFUSION 请求异步写入映射；
- 映射有容量上限和 TTL，接近容量时缩短 TTL；可按 Local Standby 的 block size 异步补算 hash；
- LocalStandbyComparisonService 把 standby 预测和之后的 engine feedback 对比，用于在切换前
  评估质量。

CacheMatchQueryOrchestrator 的查询顺序是：

1. LOCAL_SYNC 模式直接查本地全量索引。
2. KVCM 模式且 active source 为 LOCAL_STANDBY 时，直接查 standby。
3. 否则查 KVCM；若查询异常，本次请求同步降级查 standby，但 active source 仍保持 KVCM。
4. CacheMatchFailoverManager 依据 KVCM 健康自动切换，或响应 ACTIVATE_FALLBACK /
   RECOVER_PRIMARY 手动操作。控制入口是 GET /flexlb/cache_match/status 和
   POST /flexlb/cache_match/failover。

CacheMatchResult 携带产生该结果的 block size；路由器用 block size × 连续命中块数换算 token，
并限制在请求 token 数内。KVCM 对 N=1 仅在 logical key 不命中时兼容查询旧 physical key；其余
路径要求 exact logical identity。

## Block hash

RequestBlockHashService 在 schedule 的路由前准备 block_cache_keys。请求已携带 keys 时直接使用；
否则必须提供 input_ids，服务根据当前存活 PREFILL（没有时 PDFUSION）worker 的
BlockHashConfig 计算。没有任一种输入时，调度请求无效。

BlockHashExecutor 在专用线程池执行计算，默认 core=8、max=32、队列=16,384。VLLM 策略使用
链式 sha256_cbor block hash；SGLANG 策略使用 parent digest 与 little-endian token 页哈希，
支持 lookahead 为 0 或 1。三种 cache matcher 都消费有序 hash 链并在首个 miss 截断，不因 hash
算法改变前缀匹配语义。
