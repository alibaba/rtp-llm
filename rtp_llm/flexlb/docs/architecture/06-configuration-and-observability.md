# Configuration and Observability

## 配置加载机制

`ConfigService`（flexlb-common `config/`，`@Component` 具体类）是 FlexLB 运行配置的
统一读取入口。配置来源统一实现 `ConfigSource` 并注册为 Spring Bean：
`EnvironmentConfigSource` 负责环境变量，`NacosConfigSource` 负责 Nacos 初始读取和监听。
各来源 Bean 在自身初始化完成后，根据启用条件主动调用 `ConfigService.register`，注册到静态容器。
`ConfigService` 根据来源自身声明的 priority 从低到高依次加载。
当前环境变量 priority 为 1，Nacos priority 为 2：

Nacos 来源在 Bean 初始化阶段创建 client、注册 Nacos listener 并缓存首次读取结果；
`load()` 只返回已经缓存的配置内容。

1. 读 env `FLEXLB_CONFIG`（JSON）反序列化为 `FlexlbConfig`，缺省时全部用默认值；
2. **逐字段环境变量覆盖**：字段名的 UPPER_SNAKE_CASE 形式即覆盖变量
   （如 `enableQueueing` → `ENABLE_QUEUEING`），支持基本类型/包装类型/枚举；
3. 如果配置了 `FLEXLB_NACOS_SERVER_ADDR`，启动时从 Nacos 获取一个非空的部分
   `FlexlbConfig` JSON，并以 Nacos 中实际存在的字段覆盖环境变量基线；未出现在 Nacos
   中的字段仍使用环境变量或默认值；
4. Nacos listener 收到合法更新后，以当前内存配置为基础覆盖推送中存在的字段，并原子
   替换配置快照。删除或省略 Nacos 字段时保留当前内存值，不再回退环境变量。

最终优先级为：`FlexlbConfig 默认值 < FLEXLB_CONFIG < 逐字段环境变量 < Nacos 字段`。

### Nacos 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FLEXLB_NACOS_SERVER_ADDR` | 无 | Nacos server address；不配置时完全禁用 Nacos |
| `FLEXLB_NACOS_DATA_ID` | `HIPPO_ROLE` | 显式 DataId；为空时使用当前部署的 `HIPPO_ROLE` |
| `FLEXLB_NACOS_GROUP` | `DEFAULT_GROUP` | Nacos group |
| `FLEXLB_NACOS_NAMESPACE` | 空 | Nacos namespace |

配置了 Nacos 地址后，`FLEXLB_NACOS_DATA_ID` 和 `HIPPO_ROLE` 至少一个必须非空。
Nacos DataId 必须存在，内容必须是非空 JSON object。可识别的 `FlexlbConfig` 字段会覆盖
当前配置，未知字段会被忽略。例如：

```json
{
  "enableQueueing": true,
  "cacheAffinityFirstOutstandingUncachedTokensThreshold": 800000
}
```

启动阶段连接、读取、DataId 缺失、空配置或 Jackson 无法反序列化的内容都会阻止应用启动。
运行阶段的非法推送不会修改当前配置，应用保留 last-known-good 快照并记录错误；下一次
启动仍会被该非法配置阻止。配置管理层不标记配置是否支持运行时生效：动态读取的调用方
会看到新快照，构造时复制的配置需要重启后生效。

### FLEXLB_CONFIG 全量字段（`FlexlbConfig.java`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `modelServiceConfig` | 无（缺失则启动失败） | 模型路由、服务发现、KVCM 与 Optimizer 配置；可由 `MODEL_SERVICE_CONFIG` 覆盖，更新后重启生效 |
| `blockHashStrategy` | `VLLM` | cache block hash 策略：`VLLM` / `SGLANG`；可由 `BLOCK_HASH_STRATEGY` 覆盖 |
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
| `p2pHitDiscount` | 0.2 | KVCM 单远端 P2P 拉取后新增命中 block 的路由折扣；本地命中始终按 1.0 计入 |
| `shortestTtftSimilarityThresholdRatio` | 0.2 | ttft 相似阈值比例 |
| `cacheAffinityFirstMaxExtraWorkTokens` | 25000 | cache leader 相对最短 estimated-work worker 最多可增加的 token-equivalent work；不随 cache lead 或 cache 折扣缩放 |
| `decodeAvailableMemoryThreshold` | 90 | Decode 可用性滞回阈值（%） |
| `hysteresisBiasPercent` | 15 | Decode 可用性滞回带宽（%） |
| `scheduleWorkerSize` | CPU 核数 | 调度线程数 = 最大许可数 |
| `fixedScheduleWorkerPermits` | false | true 时许可固定为 `scheduleWorkerSize`，不随资源水位动态缩减 |
| `resourceCheckIntervalMs` | 10 | 动态许可重算间隔 |
| `decodeFullSpeedThreshold` / `decodeStopThreshold` | 40 / 80 | Decode 水位线性区间（%） |
| `nettySelectThreadMultiplier` / `nettyWorkerThreadMultiplier` | 1 / 2 | Netty 线程倍数 |

### MODEL_SERVICE_CONFIG

统一配置字段为 `FlexlbConfig.modelServiceConfig`，环境变量 `MODEL_SERVICE_CONFIG` 仍然兼容，
Nacos 中使用对象字段 `modelServiceConfig`。缺省**启动失败**。配置更新后由 ConfigService 保存
最新快照，模型路由相关组件在启动时初始化，因此需要重启生效。反序列化为 `ServiceRoute`：

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
  `ttl_ms=300000`、`minimum_ttl_ms=100000`、`ttl_reduction_start_ratio=0.8`、
  `maximum_entries=2000000`、`capacity_multiplier=10.0`、`async_queue_capacity=100000`、
  `hash_thread_count=4`、`hash_queue_capacity=100000`。
- `OptimizerConfig`（可选）：`enabled`、`address`、`path=/api/optimizer`、
  `discovery`。启用后仅在成功调度结束时，由专用 `doFinally` 线程池异步发送
  `/traceQuery`；`instance_id` 根据 selected worker 的 role、group 和 block size 解析
  KVCM namespace。请求收尾从 Reactor event-loop 卸载；线程池不设置任务队列，瞬时
  饱和时由提交线程执行，executor shutdown 后允许丢弃晚到任务。

### flexlbSyncConsistencyConfig（可选）

统一配置字段为 `FlexlbConfig.flexlbSyncConsistencyConfig`：
`LBConsistencyConfig{needConsistency=false, masterElectType=ZOOKEEPER,
zookeeperConfig{zkHost, zkTimeoutMs}}`。环境变量
`FLEXLB_SYNC_CONSISTENCY_CONFIG` 仍然兼容；Nacos 中使用对象字段
`flexlbSyncConsistencyConfig`。该配置在启动时初始化，更新后需要重启生效；另需 env
`HIPPO_ROLE`。见
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
- Spring profile 行为：`pre,test` → root 到文件+控制台；生产（`!pre,!test`）→ 只写文件。
  application.log 和 pv.log 的路径可通过 `FLEXLB_APP_LOG_PATH` 独立配置。
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
  - `app.optimizer.trace.query.skipped.qps` / `app.optimizer.trace.query.failed.qps`：
    trace query 的跳过与失败 QPS，tag `reason`；成功发起不单独上报。
- 上报器：`EngineHealthReporter`（~40 个指标，2s 周期性 worker 计数/线程池）、
  `RoutingQueueReporter`、`ResourceMonitorReporter`、`GracefulLifecycleReporter`、
  `CacheMetricsReporter`（flexlb-cache）、`GrpcReporter`/`KvcmMetricsReporter`（flexlb-grpc）、
  `OptimizerClient`。
- Tracing：OpenTelemetry 仅配置 W3C TraceContext 传播；OTLP endpoint 经
  `OTEL_EXPORTER_OTLP_ENDPOINT` 配置（默认 `http://127.0.0.1:4317`），skip pattern 经
  `OTEL_TRACE_SKIP_PATTERN` 配置（默认 `/health|/hook/.*`）。

## 端口与 Spring profile

- `application.yml`：`server.port: 7001`、`management.server.port: 7002`、
  `server.shutdown: graceful`（`${FLEXLB_SHUTDOWN_TIMEOUT:30s}`）。
- Spring profiles：`default`（生产日志）、`pre`/`test`（staging/testing Docker 镜像，
  test 下跳过 GracefulOnlineService）。
- 其他 env：`MAX_IN_MEMORY_SIZE`（codec 默认 10MB）、`FLEXLB_LOG_LEVEL`、`FLEXLB_LOG_PATH`、
  `FLEXLB_APP_LOG_PATH`、`FLEXLB_MONITOR_PROVIDER`、
  `BLOCK_HASH_STRATEGY`（`VLLM` / `SGLANG`）、
  `FLEXLB_BLOCK_HASH_{CORE_THREAD_COUNT,MAX_THREAD_COUNT,KEEP_ALIVE_SECONDS,QUEUE_CAPACITY}`。
