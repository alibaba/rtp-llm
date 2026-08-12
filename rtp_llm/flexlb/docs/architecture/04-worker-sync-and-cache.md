# Worker Sync and KV Cache

worker 健康与容量信息由后台线程高频异步同步；KV cache 元数据用于 cache 感知路由。
本分支（feature/flexlb-kvcm）支持三种 cache 匹配源：LOCAL_SYNC（本地全量索引）、
KVCM（外部 KV Cache Manager）、LOCAL_STANDBY（KVCM 的本地兜底）。

主要代码：`flexlb-sync/src/main/java/org/flexlb/sync/`、`service/grpc/`，
`flexlb-cache/src/main/java/org/flexlb/cache/`，`flexlb-grpc/.../KvcmGrpcClient.java`。

## Worker 状态同步

### 调度拓扑

- `MasterEngineSynchronizer`：`ScheduledThreadPoolExecutor(5)` 每 **20ms**
  （`SYNC_STATUS_INTERVAL`）触发一轮，单次 gRPC 超时 200ms。每轮按 模型×角色 提交
  `EngineSyncRunner` 到共享线程池。
- `AbstractEngineStatusSynchronizer`：两个静态线程池（core 500 / max 1000 / 队列 15000 /
  AbortPolicy）——`engine-sync-executor`（状态同步）与 `status-checker-executor`。
- `EngineSyncRunner`：从服务发现拉 worker 列表（陈旧条目要过 `max(3×同步间隔, 1s)` 宽限期
  才移除），然后对每个 worker：
  - 提交 `GrpcWorkerStatusRunner`，用 `statusCheckInProgress` CAS 保证**每 worker 同时至多
    一个在途状态检查**；
  - **仅当 `!kvcmEnabled`** 时提交 `GrpcCacheStatusCheckRunner`（`cacheCheckInProgress` CAS）。

### GrpcWorkerStatusRunner

gRPC `getWorkerStatus`（VIT 走 multimodal 变体）携带 `latest_finished_version` 做增量拉取。
响应字段（`WorkerStatusResponse`）：`alive`、`available_concurrency`、running/waiting/finished
任务表（Map<requestId, TaskInfo>）、`status_version`、`step_latency_ms`、`iterate_count`、
dp/tp size、内嵌 `cache_status`、`block_hash_lookahead_tokens`、`cache_match_rollback_blocks`、
`kv_cache_group_mode` 等。没有显式 TTFT 字段——负载估计由 `stepLatencyMs` 与本地
`runningQueueTime` 组成。

处理逻辑：版本号新才全量更新（并发/任务表/队列时间）；版本号旧也更新 alive、时间戳并做任务
对账；`cache_status` 总量恒更新（used = total − available）。带 `CacheHitFeedback` 的完成
任务会异步送 `CacheAwareService.buildCacheHitComparison`（预测 vs 实际命中对比，出指标 + pv 日志）。

### WorkerStatus 的本地预测与对账

`WorkerStatus`（flexlb-common）的原子性是**字段级**（AtomicLong/AtomicBoolean +
ConcurrentHashMap），不是快照级：

- 路由选中 → `putLocalTask()`：任务记为 IN_TRANSIT，`runningQueueTime` 加上估算 prefill
  时间，`availableKvCacheTokens`/`usedKvCacheTokens` 预扣 `inputLength − prefixLength`；
- 引擎状态到达 → `updateTaskStates()` 状态机对账：IN_TRANSIT→CONFIRMED→RUNNING→FINISHED，
  超时未确认判 LOST；`updateKvCacheTokens()` 在 `getAndSet` 引擎值前**加回在途任务的
  cache-miss 部分**，避免双重计数。
- 状态转变耗时：`updateTaskStates()` 顺带产出 `TaskStateUpdateResult` 里的延迟列表——
  FlexLB 观测值（dispatch→waiting confirm、waiting confirm→running）与引擎侧真实值
  （received→waiting、waiting→running，取自 TaskInfoPB 的 `request_received_time_ms`/
  `waiting_entered_time_ms`/`running_entered_time_ms`，`0` 视为未知跳过），由
  `GrpcWorkerStatusRunner` 分别上报供对账。
- `ExpirationCleaner`（`@Scheduled(fixedRate=3000)`）：移除 `statusLastUpdateTime` 超过
  3s 的 worker；按 `taskConfirmTimeoutMs`（默认 300,000ms）清理确认超时/LOST 任务并出
  pv 日志。

## Cache 状态同步（LOCAL_SYNC 路径，仅 KVCM 关闭时）

`GrpcCacheStatusCheckRunner`：挂在 20ms 同步 tick 上，但 PREFILL/PDFUSION 按
`DynamicCacheIntervalService.getCurrentIntervalMs()` 降频（跳 tick 实现）。请求携带当前
cache 版本做增量；响应恒更新 KV token 总量，版本更新时把 `cached_keys`（block hash 集合）
经 `CacheAwareService.updateFromWorkerStatus()` 喂给本地索引（仅 PREFILL/PDFUSION）。

**动态间隔**：`DefaultDynamicCacheIntervalService` 维护 30 样本滚动平均 diff 大小，目标
`CACHE_STATUS_DIFF_SIZE(30)`；偏差 >10% 时按 ±30% 调整间隔，钳制在
[`CACHE_STATUS_MIN_INTERVAL_MS(50)`, `CACHE_STATUS_MAX_INTERVAL_MS(3000)`]——diff 大则加快
同步，diff 小则放慢。

## flexlb-cache：三种匹配源

### LOCAL_SYNC 两级索引

- **大表** `GlobalCacheIndex`：`ConcurrentHashMap<Long blockHash, Set<String engineIpPort>>`，
  变更加单把 `ReentrantLock`。`batchCalculatePrefixMatchLength`：按序遍历请求 block 链，
  用候选集过滤 + 首个未命中即淘汰该引擎（早停），返回每引擎的前缀匹配块数。
- **小表** `EngineLocalView`：`ConcurrentHashMap<String engineIpPort, Set<Long>>`。
  `calculateDiff` 在专用 ForkJoinPool 上并行算 added/removed，diff 大小回馈动态间隔服务。
- `KvCacheManager`：门面——`findMatchingEngines`（候选来自 `WorkerStatusProvider`）、
  `updateEngineCache`（diff 后双表应用）、`removeStaleEngineCaches`、`clear`。

### KVCM（外部 KV Cache Manager）

- 开关：`MODEL_SERVICE_CONFIG` 里 ServiceRoute 级 `kvcm.enabled`；
  `CacheMatchConfiguration` 推导不变量 **`localSyncEnabled = !kvcmEnabled`、
  `localStandbyEnabled = kvcmEnabled`**。
- `KvcmGrpcClient`（flexlb-grpc）：向 KVCM **leader** 发 `GetHostCacheState`
  （namespace = `deploymentName_blockSize`，QueryType 按 worker `kvCacheGroupMode` 映射
  QT_PREFIX_MATCH / QT_PREFIX_MATCH_WITH_MAMBA），响应 `HostCacheMatch{host_ip_port,
  prefix_match_blocks}`；`p2p_host_count` 默认 5，只对 local 命中最长的前 N 个 host 计算
  P2P，配置为 0 时跳过 P2P；查询失败重试至 `maxQueryRetryCount`。
- 健康管理：daemon 线程每 `leaderRefreshIntervalMs(10s)` 刷 leader（`GetClusterInfo`）与
  worker 元数据；心跳/查询失败计数对 `heartbeatFailureThreshold(3)` /
  `queryFailureThreshold(10)` 判不健康，连续 `recoverySuccessThreshold(3)` 次心跳成功恢复；
  预热期（warmup）失败忽略。健康变化通知监听者。

### LOCAL_STANDBY（兜底索引）

- 近似索引，**只由已路由请求写入**（write-on-route）：PREFILL/PDFUSION 路由成功后
  `HttpLoadBalanceServer` 调 `updateFromRoutedRequest` 异步落库。master/follower 之间不复制。
- `LocalStandbyCacheIndex`：`ConcurrentHashMap<Long blockHash, ConcurrentHashMap<worker,
  lastUpdatedNanos>>`，TTL 过期（用量超 `ttlReductionStartRatio(0.8)` 后 TTL 从
  `ttlMs(300s)` 线性降至 `minimumTtlMs(100s)`），容量上限
  `min(存活 worker HBM 估算块数 × capacityMultiplier(10), maximumEntries(200万))`，
  达到上限拒绝新映射；daemon 清理线程每 10s 增量扫描。
- 匹配时对每个 worker 的命中块数**减去其 `cacheMatchRollbackBlocks`**（下限 0）。
- `LocalStandbyComparisonService`：KVCM 为主时持续影子预测，与引擎实际命中
  （`CacheHitFeedback`）对比出 delta 指标——failover 前即可评估兜底质量。

### 查询编排与 failover

`CacheMatchQueryOrchestrator.findMatchingEngines()`：

1. KVCM 关闭 → LOCAL_SYNC。
2. KVCM 开启：`CacheMatchFailoverManager.activeSource()` 为 LOCAL_STANDBY → 查兜底
   （指标 `standby_fallback{active_source}`）。
3. 否则查 KVCM；成功时同步做一次 standby 影子预测记录；**单次查询抛异常时该请求同步降级
   查 standby**（`standby_fallback{kvcm_query_failure}`）。

`CacheMatchFailoverManager`：监听 KVCM 健康——不健康且 `autoSwitch` 开 → 切 LOCAL_STANDBY；
恢复健康 → 切回 KVCM；手动 `ACTIVATE_FALLBACK` 覆盖一切，`RECOVER_PRIMARY` 要求 KVCM 已健康
（HTTP 入口 `POST /flexlb/cache_match/failover`，非 master 会转发给 master；状态查询
`GET /flexlb/cache_match/status`）。

`CacheMatchResult` 恒携带**应答源自己的 blockSize**（KVCM/standby 的块大小可能与请求主
hash 不同），路由侧统一用 `blockSize × 匹配块数` 折算 token，并以请求 token 数作为上限。

## Block hash 计算

- 策略：`BlockHashStrategy`（flexlb-cache）由 `FlexlbConfig.blockHashStrategy` 选择，默认
  `VLLM`；`FLEXLB_CONFIG` JSON 或 `BLOCK_HASH_STRATEGY` 环境变量可切换为 `SGLANG`。
- `VllmBlockHashStrategy` 委托 `BlockCacheKeyCalculator`（flexlb-common）计算 vLLM 兼容的
  `sha256_cbor` 链式块哈希（`PYTHONHASHSEED=0` 语义）：每满块 CBOR 编码
  `[parentHash, tokens, null]` → SHA-256 → 取低 64 位为 Long key；末尾不满块丢弃。
- `SglangBlockHashStrategy`：每页计算
  `SHA256(parentFullDigest || tokenIdsAsUint32LittleEndian)`，用完整 32-byte digest 串联下一页，
  取 digest 高 64 位作为有符号 Long key；与 SGLang page_size 对齐，末尾不满页不计算。
  `block_hash_lookahead_tokens=0`
  时每个逻辑单元是单 token；值为 1 时匹配 SGLang EAGLE，把 N 个 raw tokens 表示为 N-1 个
  overlapping `(t_i, t_{i+1})`，每个 pair 的两个 token 都写入 hash。其他 lookahead 值请求失败。
  可缓存前缀同样按完整页截取：普通模式按 `floor(N/blockSize)`、EAGLE 按
  `floor((N-1)/blockSize)`。
- 配置解析：`WorkerBlockHashConfigResolver` 每 1 分钟从存活 PREFILL（退化 PDFUSION）worker
  的 `blockSize` + `blockHashLookaheadTokens` 刷新，不可用时保留上次有效值。
- 执行：`BlockHashExecutor` 专用线程池（默认 core 8 / max 32 / 队列 16384，
  `flexlb.block-hash.*` 可调），出队等待/执行耗时指标，完成后 `publishOn(parallel)` 不占
  hash 线程。请求自带 `block_cache_keys` 时直接采用不再计算。
- Local Standby 块大小与主请求不同时，由 `LocalStandbyHashService`（低优先级独立线程池 +
  Caffeine 60s 结果缓存）异步补算，路由只等主 hash；主 hash 与 standby 共用同一个
  `BlockHashStrategy` bean。
- LOCAL_SYNC、Local Standby 与 KVCM 查询都消费有序 block key 链，按请求顺序连续匹配并在
  首个 miss 停止；这与 SGLang HashTree 的前缀匹配语义一致，不按算法拆分 matcher。
