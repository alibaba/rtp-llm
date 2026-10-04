# Worker Sync and KV Cache

worker 健康与容量信息由后台线程高频异步同步；KV cache 元数据用于 cache 感知路由。
本分支（feature/flexlb-kvcm）支持三种 cache 匹配源：LOCAL_SYNC（本地全量索引）、
KVCM（外部 KV Cache Manager）、LOCAL_STANDBY（KVCM 的本地兜底）。

主要代码：`flexlb-sync/src/main/java/org/flexlb/sync/`、`service/grpc/`，
`flexlb-cache/src/main/java/org/flexlb/cache/`，`flexlb-grpc/.../KvcmGrpcClient.java`。

## Worker 状态同步

### 请求 ID 协议

Worker Status 的共享 PB 协议定义为 `string TaskInfoPB.request_id = 1`，
Java、C++ 和 Python 使用同一字段定义生成代码。
`RequestId.parse` 优先保留原始字符串；字符串缺失时，从同字段编号的 unknown fields 中读取旧 int64 编码并转换成字符串；两者都缺失时会报错。
Running 和 Finished 任务表均以解析后的 ID 为键，缓存反馈与调度请求使用同一 ID 关联。
Master 的请求模型、调度、取消、状态查询、资源预留和 tracing 接口均只接收字符串 ID。
客户端的 Schedule、Cancel、GetRequestState gRPC 请求使用相同的边界兼容读取；HTTP JSON 的数字 ID 在反序列化请求模型时转换成字符串。
Engine 的生成、取消、结果拉取和 KV 传输协议仍使用现有 int64 字段；
相关转换仅发生在对接这些 Engine RPC 的边界，不用于 Master 内部请求标识。

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

`WorkerAddressService` 按 Endpoint 保存最近一次成功的非空发现快照。查询异常、超过 500ms、
线程池拒绝任务或返回空列表时保留该快照，已有 worker 继续接受状态探测；首次发现尚无缓存时
返回空列表。发现线程池使用 AbortPolicy，饱和时不在同步线程执行网络查询。超时查询即使稍后
完成，也不会更新快照。成功的非空结果替换快照，允许正常的部分缩容与节点更换。

`EngineAddressResolver` 的定期查询和订阅回调同样忽略空列表，保留该 Endpoint 的地址，
避免关闭现有 channel 或清除已有 worker cache。真实 worker 失效 由健康探测处理。

一个服务发现 frontend 会按 Endpoint `multi_engine_num` 展开为 N 个逻辑 worker，map key
统一为 `ip:httpPort@index`（N=1 也是 `@0`）。frontend HTTP/gRPC 地址保持共享；第 i 个
`GrpcWorkerStatusRunner` 连接显式配置的 `worker_status_port + i`，N=1 时同样接受该覆盖；
未配置时使用发现归一化后的 gRPC port。N>1 必须显式配置 status base，配置加载时同时校验 count、base 和
`base + N - 1 <= 65535`。

`EngineAddressResolver` 向唯一的 pooled `EngineGrpcClient` 发布结构化 `WorkerHost`，保留业务
gRPC 端口和 worker status 端口。客户端按 `ip:实际RPC端口:serviceType` 复用 channel：
`GetWorkerStatus` / `GetCacheStatus` 使用 `worker_status_port + i`，`EnqueueBatch` / `Cancel`
使用共享 frontend gRPC 端口。多引擎的业务 channel 去重，状态 channel 按 engine 端口独立创建。
KVCM 开启时不创建 `GetCacheStatus` channel。该规则对所有 discovery provider 一致；worker IP
完全退出发现集合后关闭其 channel。endpoint 配置不热更新，端口变更依赖进程重启。

worker 地址表示由不可变 `WorkerIdentity` 一次性预计算并保存，调用方不再解析或临时拼接：

| 表示 | 格式 | 用途 |
|---|---|---|
| raw IP | `ip` | 网络连接与服务发现 |
| raw port | `port` | 共享 frontend 端口 |
| raw engine index | `engineIndex` | 逻辑引擎序号 |
| physical IP-port | `ip:port` | 共享 frontend 身份、发现侧 cache 退役匹配 |
| logical IP-port | `ip:port@index` | 路由、rollback、KVCM 与 cache key |
| metric IP-port | N=1 为 `ip:port`；N>1 为 `ip:port@index` | 可从 `WorkerStatus` / `ServerStatus` 归属的 `engineIp` 指标标签 |

`WorkerHost` 在服务发现展开时持有该 identity；N=1 的内部 logical identity 仍保留 `@0`。仅在
metrics 边界通过 `WorkerStatus.getMetricIpPort()`（或 `ServerStatus` 对应方法）选择兼容地址，
不会改变路由、cache 或 rollback 的 identity。

### GrpcWorkerStatusRunner

gRPC `getWorkerStatus`（VIT 走 multimodal 变体）携带 `latest_finished_version` 做增量拉取。
Encoder 使用常规 `GetWorkerStatus`，角色字符串为 `ENCODER`，`RoleTypePB` 枚举值为
`ROLE_TYPE_ENCODER = 5`。`EncoderEndpoint` 将
`running_task_info` 和 `finished_task_list` 投影到 Encoder 阶段的请求记录；运行中的
任务使本地待观察并发与引擎并发对账，完成任务中的错误码决定完成或失败。当前
PAI-vLLM 端的 Encoder 状态上报不属于 FlexLB 实现范围。
`WorkerStatusPB` 由 `EngineStatusConverter.convertToStatusObservation` 转成不可变的
`WorkerStatus.StatusObservation`。其中 `runningTasks` 对应引擎的 `running_task_info`，
保存所有尚未完成的任务，包括 `PENDING`、`RECEIVED`、`KV_ALLOCATED` 和 `RUNNING` 阶段；
`finishedTasks` 单独保存完成上报。任务按字符串 request ID 索引，并保留各自的 `phase`。
纯运行任务需按 `phase == RUNNING` 筛选，不能用 `runningTasks.size()` 代替；
`runningQueryLen` 与 `waitingQueryLen` 则直接保留引擎上报的数量。
其余字段包括 `alive`、`available_concurrency`、`status_version`、`step_latency_ms`、
`iterate_count`、dp/tp size、KV cache 容量、`block_size`、`block_hash_lookahead_tokens`、
`cache_match_rollback_blocks`、`kv_cache_group_mode` 等。没有显式 TTFT 字段——负载估计由 `stepLatencyMs` 与本地
`runningQueueTime` 组成。

处理逻辑：版本号新才全量更新（并发/任务表/队列时间）；版本号旧也更新 alive、时间戳并做任务
对账；`cache_status` 总量恒更新（used = total − available）。带 `CacheHitFeedback` 的完成
任务会异步送 `CacheAwareService.buildCacheHitComparison`（预测 vs 实际命中对比，出指标 + pv 日志）。
连续 3 次状态探测失败（RPC 失败或响应转换失败）会把该逻辑 worker 标为不健康并移除其 endpoint；
同一 frontend 的其他已发布 logical worker 不受影响。任务既没有有效字符串 ID，也没有可兼容读取的
旧 int64 编码时，整份状态观测被拒收，不发布部分任务状态，并按上述连续失败规则处理。
新发现的 worker 在首次接受有效状态前不可路由。空响应标为不健康；未初始化状态
（`status_version=0`）与响应处理异常跳过本轮更新。

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
- `LocalSyncCacheMatchProvider` 仅在 LOCAL_SYNC 模式订阅 `EngineAddressResolver`，按最新
  `WorkerHost` 集合清理已下线 frontend 的本地 cache；KVCM 模式不注册该 listener。

上述 LOCAL_SYNC key、KVCM `host_ip_port`、LOCAL_STANDBY 映射与 cache-hit comparison 均使用
逻辑 `ip:httpPort@index`。KVCM 对 N=1 worker 兼容旧 physical `ip:httpPort` key：logical key
未命中时才回退查询 physical key；N>1 或非 KVCM source 仍要求 exact match，无法匹配时按零命中忽略。

### KVCM（外部 KV Cache Manager）

- 开关：`FLEXLB_CONFIG.cacheMatching.type=KVCM`；`MODEL_SERVICE_CONFIG.kvcm` 只提供
  KVCM address/namespace/port/discovery 定位信息；
  `CacheMatchConfiguration` 推导不变量 **`localSyncEnabled = !kvcmEnabled`、
  `localStandbyEnabled = kvcmEnabled`**。
- `KvcmGrpcClient`（flexlb-grpc）：向 KVCM **leader** 发 `GetHostCacheState`
  （namespace = `deploymentName_blockSize`，QueryType 按 worker `kvCacheGroupMode` 映射
  QT_PREFIX_MATCH / QT_PREFIX_MATCH_WITH_MAMBA），响应 `HostCacheMatch{host_ip_port, local,
  global}`；`global` 是 local、P2P 与远端 pool 来源联合后的前缀命中块数，已包含 `local`。
  请求侧 `medium` 默认空列表（空表示匹配全部介质，取值原样透传给 KVCM）；
  `globalKvsHostCount` 默认 3，映射到 PB 的 `global_kvs_host_count = 7`（int32），按 local
  降序取前 N 个逻辑引擎计算远端命中，0 表示只算本地；`enableP2p` 默认 `false`，
  映射到 PB 的 `enable_p2p = 8`（bool）。PB 与已部署 KVCM 的
  `alibaba/tair-kvcache@f9196aaff4f0dad3520b9523ae55721eb4955b2f` 保持一致。
  PB 不再声明 `p2p_host_count`、`p2p_1_fetch`、`p2p_1_total_match`；返回结果只消费 `local/global`。
  查询失败重试至 `maxQueryRetryCount`。
- 健康管理：daemon 线程每 `leaderRefreshIntervalMs(10s)` 刷 leader（`GetClusterInfo`）与
  worker 元数据；心跳/查询失败计数对 `heartbeatFailureThreshold(3)` /
  `queryFailureThreshold(10)` 判不健康，连续 `recoverySuccessThreshold(3)` 次心跳成功恢复；
  预热期（warmup）失败忽略。健康变化通知监听者。
- 参数热更新：`CacheMatchConfiguration` 注册 `ConfigService` 更新监听器，用 volatile 字段发布
  最新的 `KvcmCacheMatchingConfig`；`KvcmGrpcClient` 与 `KvcmLeaderResolver` 不再缓存构造期快照，
  因此 `requestTimeoutMs`、`maxQueryRetryCount`、健康阈值与三个查询参数在更新后即时生效。
  单次查询开始时读取一次快照，同一查询的所有重试复用该快照。`leaderRefreshIntervalMs`、
  `localStandby.*` 与 `cacheMatching.type` 仍在启动时固定，修改需重启。

### LOCAL_STANDBY（兜底索引）

- 近似索引，**只由已路由请求写入**（write-on-route）：PREFILL/PDFUSION 路由成功后
  `FlexlbServiceImpl` 在本地调度成功后调 `updateFromRoutedRequest`，由独立线程池异步落库，
  不等待索引写入完成；更新失败不影响调度响应。转发成功的 follower 不写入，
  follower 在本地兜底调度成功时写入自己的索引。master/follower 之间不复制。
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
3. 否则查 KVCM；成功时同步做一次 standby 影子预测记录；**内部重试耗尽后查询抛异常时当前请求同步降级
   查 standby，但 active source 保持 KVCM**（`standby_fallback{kvcm_query_failure}`）。KVCM gRPC client 同时报告
   `app.cache.kvcm.query.failure.qps`。
   Local Standby 的结果登记为预测；预测结果与选定 Worker 齐备时由 Local Standby 组件上报该 Worker 的预测命中值。有效的 engine feedback
   可进一步生成 cache-hit comparison 指标和 PV。预测登记异常不触发 KVCM 降级或 Local Standby 路由失败。

`CacheMatchFailoverManager`：监听 KVCM 健康——不健康且 `autoSwitch` 开 → 切 LOCAL_STANDBY；恢复健康 →
切回 KVCM；手动 `ACTIVATE_FALLBACK` 覆盖一切，
`RECOVER_PRIMARY` 要求 KVCM 已健康
（HTTP 入口 `POST /flexlb/cache_match/failover`，非 master 会转发给 master；状态查询
`GET /flexlb/cache_match/status`）。

`CacheMatchResult` 携带**应答源自己的 blockSize**（KVCM/standby 的块大小可能与请求主
hash 不同）和 `querySucceeded`。空匹配是成功结果，查询失败的结果不产生预测指标。
路由侧统一用 `blockSize × 匹配块数` 折算 token，并以请求 token 数作为上限。

## 客户端 block cache keys

- 客户端按目标引擎的哈希算法、block size 和 lookahead 规则计算有序 `block_cache_keys`，
  通过 Schedule 的 `cache_key_block_size` 提供对应块大小。非空 keys 必须携带正数块大小。
- Master 不接收原始 `input_ids`，不计算或重算 block hash，也不从 worker 推导请求块大小。
  未提供 keys 时按零 cache block 路由，支持不足一个完整 block 的请求。
- `BlockHashStrategy` 接口和 `VllmBlockHashStrategy`、`SglangBlockHashStrategy` 实现作为独立
  算法代码保留，包括所需的 `TokenIds`、vLLM CBOR 计算工具和测试；调度链路没有调用这些策略。
- LOCAL_SYNC、Local Standby 与 KVCM 复用同一组客户端 keys 和块大小，按请求顺序连续匹配，
  在首个 miss 停止。Local Standby 索引在路由成功后异步写入这些 keys。
