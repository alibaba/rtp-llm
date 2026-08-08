# Configuration and Observability

## 配置加载机制

`ConfigService`（flexlb-common `config/`，`@Component` 具体类）：

1. 读 env `FLEXLB_CONFIG`（JSON）反序列化为 `FlexlbConfig`，缺省时全部用默认值；
2. **逐字段环境变量覆盖**：字段名的 UPPER_SNAKE_CASE 形式即覆盖变量
   （如 `enableQueueing` → `ENABLE_QUEUEING`），支持基本类型/包装类型/枚举。

### FLEXLB_CONFIG 全量字段（`FlexlbConfig.java`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `loadBalanceStrategy` | `SHORTEST_TTFT` | PDFUSION/PREFILL 策略 |
| `decodeLoadBalanceStrategy` | `WEIGHTED_CACHE` | DECODE 策略 |
| `vitLoadBalanceStrategy` | `RANDOM` | VIT 策略 |
| `weightedCacheDecayFactor` | 0.001 | WeightedCache 指数衰减系数 |
| `enableQueueing` | false | 队列模式开关 |
| `maxQueueSize` | 1000000 | 队列容量 |
| `maxRetryCount` | 0（不限） | 路由重试上限 |
| `routingRetryIntervalMs` | 10 | 重试间隔 |
| `taskConfirmTimeoutMs` | 300000 | IN_TRANSIT 任务确认超时 |
| `prefillQueueSizeThreshold` | 3 | Prefill 停止分流阈值（低于该值可选）及水位 100% 刻度 |
| `prefillCacheHitDiscount` | 0.7 | cache 命中 token 折扣 |
| `p2pHitDiscount` | 0.2 | KVCM 单远端 P2P 拉取后新增命中 block 的路由折扣；本地命中始终按 1.0 计入 |
| `shortestTtftSimilarityThresholdRatio` | 0.2 | ttft 相似阈值比例 |
| `cacheAffinityFirstQueueToleranceFactor` | 2.0 | cache leader 可额外承受的 TTFT：缓存 token 领先量 × cache 折扣 × 该系数 |
| `cacheAffinityFirstAbsoluteToleranceTokens` | 0 | cache leader 可额外承受 TTFT 的绝对 cache token 容忍值；与 factor 计算值取较大者后再应用 cache 折扣 |
| `decodeAvailableMemoryThreshold` | 90 | Decode 可用性滞回阈值（%） |
| `hysteresisBiasPercent` | 15 | Decode 可用性滞回带宽（%） |
| `scheduleWorkerSize` | CPU 核数 | 调度线程数 = 最大许可数 |
| `fixedScheduleWorkerPermits` | false | true 时许可固定为 `scheduleWorkerSize`，不随资源水位动态缩减 |
| `resourceCheckIntervalMs` | 10 | 动态许可重算间隔 |
| `decodeFullSpeedThreshold` / `decodeStopThreshold` | 40 / 80 | Decode 水位线性区间（%） |
| `nettySelectThreadMultiplier` / `nettyWorkerThreadMultiplier` | 1 / 2 | Netty 线程倍数 |

### MODEL_SERVICE_CONFIG

Spring 属性/env，缺省**启动失败**。反序列化为 `ServiceRoute`：

- `service_id`（必填）、`role_endpoints: List<GroupRoleEndPoint>`（必填非空）、`kvcm`。
- `GroupRoleEndPoint`：`group` + `prefill_endpoint` / `decode_endpoint` / `vit_endpoint` /
  `pd_fusion_endpoint`（各为 `Endpoint{address, protocol, path, worker_status_port,
  discovery}`）。
- `DiscoveryConfig`：`type`（`vipserver` / `dashscope` / `static-env`）、`base_url`
  （默认 `http://127.0.0.1:8880`）、connect/read timeout 500ms、poll 1000ms、`hosts`。
- `KvcmConfig`：`enabled`、`address`、`namespace`、`port=6381`、`discovery`、
  `request_timeout_ms=500`、`leader_refresh_interval_ms=10000`、
  `heartbeat_failure_threshold=3`、`query_failure_threshold=10`、`max_query_retry_count=1`、
  `recovery_success_threshold=3`、`local_standby`。
- `LocalStandbyConfig`：`auto_switch=true`、`block_size=0`（0=沿用引擎块大小）、
  `entry_ttl_ms=300000`、`minimum_entry_ttl_ms=100000`、`ttl_reduction_start_ratio=0.8`、
  `maximum_entries=2000000`、`capacity_multiplier=10.0`、`async_queue_capacity=100000`、
  `hash_thread_count=4`、`hash_queue_capacity=100000`。

### FLEXLB_SYNC_CONSISTENCY_CONFIG（可选）

`LBConsistencyConfig{needConsistency=false, masterElectType=ZOOKEEPER,
zookeeperConfig{zkHost, zkTimeoutMs}}`；另需 env `HIPPO_ROLE`。见
[05-lifecycle-and-consistency](05-lifecycle-and-consistency.md)。

## HTTP 端点（主端口 7001，全部 WebFlux RouterFunction）

| 端点 | 方法 | 用途 |
|---|---|---|
| `/rtp_llm/schedule` | POST | 路由主入口；错误映射：非法参数→400 `INVALID_REQUEST`，hash 线程池饱和→503 `QUEUE_FULL`，其他→500 |
| `/rtp_llm/master/info` | POST | `{realMasterHost, podIp, instanceIp, queueLength}` |
| `/rtp_llm/schedule_snapshot` | POST | dump LB 状态（当前 TODO 占位，恒成功） |
| `/rtp_llm/notify_master` | POST | master 变更通知接收端 |
| `/rtp_llm/queue_snapshot` | GET | 队列快照写 `/tmp/flexlb-queue-snapshots/`（保留 10 个） |
| `/flexlb/update_log_level` | POST | 动态调 Spring logger group `flexlb` 级别（trace…error） |
| `/flexlb/cache_match/status` | GET | cache 匹配源状态快照 |
| `/flexlb/cache_match/failover` | POST | 手动 failover（非 master 转发） |
| `/health` | GET | 200 / 404（停机信号或未预热） |
| `/hook/process_ok`、`/hook/after_start`、`/hook/pre_stop` | GET | 生命周期 hook，仅限本机调用 |

## 日志架构（`logback-spring.xml`）

- 路径属性：`LOG_PATH` ← `flexlb.log.path`（默认 `/home/admin/ai-whale/logs`）、
  `APP_LOG_PATH` ← `flexlb.log.app-path`。文件：application.log、pv.log（APP_LOG_PATH）；
  sync.log、sync_consistency.log、flexlb.log（LOG_PATH）。
- 所有文件 appender 为 RollingFile（50MB/文件、maxHistory 5、totalSizeCap 2–4GB）+
  AsyncAppender（`neverBlock=true`，队列 `flexlb.log.async-queue-size` 默认 16384）。
- 命名 logger（additivity=false）：`pvLogger`→pv.log（每请求 JSON 记录）、`syncLogger`→
  sync.log、`syncConsistencyLogger`→sync_consistency.log、`flexlbLogger`→flexlb.log。
- Spring profile 行为：`pre,test` → root 到文件+控制台；生产（`!pre,!test`）→ 只写文件；
  **`dashscope`** → root 与 pvLogger 走 stdout，且 `application-dashscope.yml` 把
  `flexlb.log.app-path` 改为 `${FLEXLB_APP_LOG_PATH:/home/admin/logs}`（application/pv 移到
  /home/admin/logs，sync 系仍在 /home/admin/ai-whale/logs）。profile 块可叠加：dashscope +
  生产时 root 同时写文件和 stdout。
- 运行时调级：logger group `flexlb` = `org.flexlb,flexlbLogger,syncLogger,
  syncConsistencyLogger`（application.yml），经 `/flexlb/update_log_level` 修改。

## 指标体系

- 抽象：`FlexMonitor`（flexlb-common `metric/`）——`register(name, type[, priority])` +
  `report(name[, tags], value)`；类型 GAUGE/COUNTER/QPS，优先级即聚合窗口
  PRECISE(1s)/CRITICAL(5)/MAJOR(10)/NORMAL(20)/TRIVIAL(60)。
- 实现：默认 `NoOpFlexMonitor`（`flexlb.monitor.provider: noop`）；**internal Maven profile**
  （`../../../internal_source` 存在时自动激活）引入 KMonitor/VipServer 依赖；
  `FLEXLB_MONITOR_PROVIDER=kmonitor-prometheus` 时 appctl.sh 注入 kmonitor.properties，
  Prometheus exporter 端口 4142。opensource 构建没有 spring-boot-starter-actuator，
  application.yml 里的 management 配置不生效。
- 指标名集中在 `MetricConstant`（flexlb-common），主要族：
  - `app.engine.health.*` / `app.engine.worker.*`：同步成功周期、worker 数、并发、RT/QPS、
    队列时间、任务表大小；worker status 状态转变耗时——FlexLB 观测值
    （`app.engine.worker.status.observed.decision.to.waiting.ms`、
    `...flexlb.observed.waiting.to.running.ms`）与引擎侧真实时间戳
    （`app.engine.worker.status.engine.received.to.waiting.ms`、
    `...engine.waiting.to.running.ms`，由 TaskInfoPB 的 `request_received_time_ms`/
    `waiting_entered_time_ms`/`running_entered_time_ms` 计算）对账，差值即 FlexLB 观测延迟；
  - `app.routing.*`：队列长度/入队/超时/拒绝/取消 QPS、排队等待、路由执行耗时、
    单次路由尝试耗时（`app.routing.route.attempt.execution.time.ms`，不含 retry sleep）、
    成功/失败（tag `code`）/每次实际 retry 的重试 QPS（`RoutingQueueReporter`，全 PRECISE）；
  - `app.cache.*`：两级索引规模、命中数/率、预测 vs 实际对比（`hit.comparison.*`）、
    Local Standby 默认生效 block size（`app.cache.local.standby.block.size`）、容量/拒绝/映射数、
    `cache.match.active.source`、`standby.fallback.qps`、
    `kvcm.query.retry.qps`、diff 大小、find/update RT；
  - `app.block.hash.*` / `app.local.standby.hash.*`：hash 排队/执行耗时、线程池状态；
  - `app.request.input.ids.count`（实际 `input_ids` 数量）与
    `app.request.body.bytes`（HTTP `Content-Length`，未声明长度的请求不报该值）；
  - `app.worker.permit.capacity`（1s）、`graceful.lifecycle.event`、
    `app.forward.to.master.result`、`app.engine.zk.master.*`。
- 上报器：`EngineHealthReporter`（~40 个指标，2s 周期性 worker 计数/线程池）、
  `RoutingQueueReporter`、`ResourceMonitorReporter`、`GracefulLifecycleReporter`、
  `CacheMetricsReporter`（flexlb-cache）、`GrpcReporter`/`KvcmMetricsReporter`（flexlb-grpc）。
- Tracing：OpenTelemetry 仅配置 W3C TraceContext 传播；OTLP endpoint 经
  `OTEL_EXPORTER_OTLP_ENDPOINT`，skip pattern 经 `OTEL_TRACE_SKIP_PATTERN`。

## 端口与 Spring profile

- `application.yml`：`server.port: 7001`、`management.server.port: 7002`、
  `server.shutdown: graceful`（`${FLEXLB_SHUTDOWN_TIMEOUT:30s}`）。
- Spring profiles：`default`（生产日志）、`pre`/`test`（staging/testing Docker 镜像，
  test 下跳过 GracefulOnlineService）、`dashscope`（日志走 stdout）；生产镜像用
  `-Dspring.profiles.active=${FLEXLB_ACTIVE_PROFILES}` 由环境注入。
- 其他 env：`MAX_IN_MEMORY_SIZE`（codec 默认 10MB）、`FLEXLB_LOG_LEVEL`、`FLEXLB_LOG_PATH`、
  `FLEXLB_APP_LOG_PATH`、`FLEXLB_MONITOR_PROVIDER`、
  `FLEXLB_BLOCK_HASH_{CORE_THREAD_COUNT,MAX_THREAD_COUNT,KEEP_ALIVE_SECONDS,QUEUE_CAPACITY}`。
