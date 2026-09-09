# Configuration and Observability

## 配置边界

FlexLB 保留两个职责独立的配置文档：

- `FLEXLB_CONFIG`：唯一的 FlexLB 行为配置，包含调度、分发、路由、worker registry、
  observability、服务发现运行参数、cache matching、Optimizer 和一致性/主选举。
- `MODEL_SERVICE_CONFIG`：只保存模型、endpoint、KVCM、Optimizer 的地址和服务发现定位信息。

`MODEL_SERVICE_CONFIG` 不属于动态 `FlexlbConfig` 快照，也不会被 UniConfig 或 Nacos
的动态更新覆盖。UniConfig 开关、Nacos 连接、部署标识、日志路径、Spring/OTEL 等
基础设施环境变量仍各自独立；它们不构成
FlexLB 行为配置的别名。

## FlexlbConfig 加载与动态更新

`ConfigService` 是统一读取入口。`ConfigSourceSelection` 在启动时根据 FlexLB 进程的
环境变量选择唯一的行为配置字符串来源，先判断 UniConfig，再判断 Nacos：

| 条件（按顺序判断） | 字符串来源 | 动态更新方式 |
|---|---|---|
| `FLEXLB_UNICONF_ENABLE=true` | `UniConfigConfigSource` | 轮询本机 Turbo UniConfig HTTP 接口 |
| 否则，`FLEXLB_NACOS_SERVER_ADDR` 非空 | `NacosConfigSource` | Nacos listener |
| 否则 | `EnvironmentConfigSource` | 启动时读取 `FLEXLB_CONFIG` |

`true` 忽略大小写和两端空白。开启 UniConfig 后，即使配置了 Nacos 地址，也不会
创建 Nacos 客户端或 listener。使用外部来源时，环境变量中的 `FLEXLB_CONFIG` 不参与
校验或合并；初始文档中省略的行为字段使用对应格式解析器和 `FlexlbConfig` 的默认值。
`EnvironmentConfigSource` 仍负责读取独立的 `MODEL_SERVICE_CONFIG` 启动拓扑。
来源选择和连接参数在启动时确定，修改它们需要重启进程。

三种来源都返回原始字符串，复用 `ConfigDocumentParserResolver` 和原有 v0/v1 解析器：
优先使用文档内的 `schemaVersion`，否则使用 `FLEXLB_CONFIG_SCHEMA_VERSION`（默认 `0`）。
UniConfig 和 Nacos 的内容直接使用同一种配置 JSON，不增加包装层。

格式归一化后，由现有 `FlexlbConfigMerger` 执行递归部分更新：
对象字段递归覆盖，未出现或从外部配置删除的字段保留当前内存值，数组和标量整体替换；
tagged union 的 `type` 变化会替换整个分支。v1 文档 `{"schemaVersion":1}` 是 no-op；
没有显式版本的 `{}` 则按版本选择规则解析，默认走 v0 兼容转换。
每次合并后都使用与 `FLEXLB_CONFIG` 相同的严格解析和跨字段校验：

- 拒绝重复 key、未知字段、`null`、标量 coercion、数值枚举和尾随 JSON；
- 拒绝 tagged union 非活动分支的字段；
- 拒绝违反 scheduler / dispatcher / router 等组合约束的配置。

选定的外部来源初次读取或校验失败会阻止应用启动，不会自动切换到低优先级来源。
运行时读取失败或非法更新不会替换当前 last-known-good 快照，后续更新恢复正常后
继续应用；合法更新原子替换 `FlexlbConfig`，随后通知监听器。

配置来源层不区分“热生效”与“重启生效”：它只发布最新有效快照。业务组件每次读取快照，
就可以热生效；在 Bean 初始化时缓存的值，则在重启后生效。

旧的字段级行为变量 `BLOCK_HASH_STRATEGY`、`FLEXLB_LOG_LEVEL`、
`ENABLE_STDOUT_LOG`、`ENABLE_FALLBACK` 不再覆盖 JSON。行为配置只认
所选来源的 `FLEXLB_CONFIG` 文档，避免嵌套字段到环境变量名的隐式转换。

### UniConfig 连接

Spectrum 部署的扩展配置中启用 Turbo UniConfig：

```json
{
  "turbo": {
    "env": {
      "UNICONF_ENABLE": "true"
    }
  }
}
```

同时需要在部署的 worker 环境变量中显式配置，让 FlexLB Java 进程读取到：

```text
FLEXLB_UNICONF_ENABLE=true
```

两个开关分别控制不同进程，启用 UniConfig 时需要同时配置：Turbo 的
`turbo.env.UNICONF_ENABLE=true` 开启本机配置服务；Java 的
`FLEXLB_UNICONF_ENABLE=true` 选择 UniConfig 配置来源。Java 只读取带 `FLEXLB_` 前缀的
开关，未设置或不为 `true` 时继续按 Nacos 地址、环境变量的顺序选择来源。
部署标识沿用 `DeploymentIdentity`，要求设置 `SPECTRUM_WORKSPACE_ID`、
`SPECTRUM_APPLICATION_NAME` 和 `SPECTRUM_DEPLOYMENT_NAME`。

`UniConfigConfigSource` 读取以下部署级 key 的 HTTP 响应正文：

```text
GET http://127.0.0.1:18080/v2/configs/modelstudio.spectrum.deployment.<workspace>.<deployment>.runtime.meta
```

正文就是完整的配置 JSON 字符串，与 Nacos 文档格式一致。连接和读取超时各为 3 秒。
启动时立即读取初始配置，后续每次请求完成后等待 30 秒再轮询，仅在正文变化时发布更新。
Turbo 侧通常有约 1 分钟缓存，采用 30 秒轮询可兼顾本机请求频率和变化发现延迟；缓存刷新后，
Java 还可能等待接近 30 秒才发现新内容。因此两段等待叠加时，生效延迟可能接近 90 秒，
另加请求耗时，不能将轮询间隔理解为端到端生效时间保证。
启动阶段遇到本机 agent 尚未监听端口导致的连接拒绝（`ConnectException`）时，每隔 1 秒
重试，最多尝试 30 次；重试耗尽或线程被中断后启动失败。非 HTTP 200（包括 key 不存在时
的 404）和非法配置仍会阻止启动。启动前应先在部署 UniConfig 页面保存合法配置。
运行时 HTTP 异常保留有效快照并继续按 30 秒间隔重试。

### 凑批窗口热更新

`scheduler.decision.maxCollectionWaitMs` 在每次固定窗口决策时从配置服务的当前内存快照读取。
同一次决策的预检查与最终组选择共用一个窗口值。路由预估缓存将窗口值作为有效性条件，
窗口变化时重新生成预估输入。读取不会触发 UniConfig 或 Nacos 网络请求。

配置更新不主动打断已有定时等待；队列事件、状态事件或原定截止时间触发下一次决策时，
使用最新窗口值。因此缩短窗口不会保证已有等待立即结束，后续窗口无需重启即可使用新值。
此行为仅覆盖窗口时长，不承诺调度模式、交付模式或其他启动时配置的热切换。

### Nacos 连接

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FLEXLB_NACOS_SERVER_ADDR` | 无 | 仅在 UniConfig 未开启且地址非空时启用 Nacos |
| `FLEXLB_NACOS_DATA_ID` | 部署标识 | 显式 DataId |
| `FLEXLB_NACOS_GROUP` | `DEFAULT_GROUP` | Nacos group |
| `FLEXLB_NACOS_NAMESPACE` | 空 | Nacos namespace |

未显式配置 DataId 时，部署标识优先使用
`SPECTRUM_WORKSPACE_ID`、`SPECTRUM_APPLICATION_NAME`、
`SPECTRUM_DEPLOYMENT_NAME` 组成
`spectrum:<workspace>:<application>:<deployment>`；旧环境回退 `HIPPO_ROLE`。

UniConfig / Nacos 的 v1 部分更新示例：

```json
{
  "schemaVersion": 1,
  "router": {
    "availabilityHysteresisPercent": 12
  },
  "observability": {
    "logging": {
      "level": "warn",
      "stdoutEnabled": true
    }
  }
}
```

## FLEXLB_CONFIG 结构

公共 schema 当前为 version 1，按责任分区：

- `scheduler`：`DIRECT` / `QUEUE`；QUEUE 拥有 ordering、capacity 和 lifecycle。
- `dispatcher`：`BATCH` / `NON_BATCH`。
- `router`：角色 availability、execution estimator、selector、cache affinity 和
  group selector。
- `workerRegistry`：worker health 与 cache-status 刷新策略。
- `observability.cacheHit`：recent-key window、指标和理论命中日志。
- `observability.logging`：FlexLB logger group 级别与 root/PV stdout 开关。
- `serviceDiscovery`：connect/read timeout、poll interval 与连接池运行参数。
- `cacheMatching`：`LOCAL_SYNC` / `KVCM` tagged union；KVCM 分支拥有查询、健康、P2P
  和 Local Standby 参数。
- `optimizer`：启用开关和服务发现轮询间隔。
- `consistency`：`NONE` / `ZOOKEEPER` tagged union；ZooKeeper 分支拥有连接和 master
  刷新参数。
- `blockHashStrategy`：cache block hash 策略。
- `enableFallback`：默认 `false`；启用时调度入口在转发和路由前返回错误码 `8600`，
  由调用方执行 domain fallback。
- `fallbackBatchTokenCapacity`：默认 `1048576`；Engine 未声明
  `max_batch_tokens_size` 和 `max_seq_len` 时使用的最终 batch token 容量兜底值。
  优先使用 Engine 上报的 `max_batch_tokens_size`，其次使用 `max_seq_len`。
- `internalRuntime`：代码内部设置，不接受公共 JSON 输入。

`DIRECT + BATCH` 非法；可选配置应省略，不能写 `null`。完整示例和 selector
矩阵见根目录 [README](../../README.md)。

## MODEL_SERVICE_CONFIG

独立反序列化为 `ServiceRoute`：

- `service_id` 和 `role_endpoints`；
- endpoint 的 `address`、`protocol`、`path`、`worker_status_port`、`discovery`；
- discovery 定位字段：类型 `static-env`、`vipserver`、`dashscope`，以及可选
  `base_url` / `hosts`；
- 可选 KVCM 定位字段 `address`、`namespace`、`port`、`discovery`；
- 可选 Optimizer 定位字段 `address`、`port`、`path`、`discovery`。

旧的 `load_balance`、KVCM/Optimizer `enabled`、discovery timeout/poll、KVCM 健康与
Local Standby 等行为字段会被拒绝，并提示迁移到 `FLEXLB_CONFIG`。该拓扑在相关 Spring
Bean 创建时使用；动态 `FLEXLB_CONFIG` 更新不会改变模型拓扑。服务发现 provider 仍通过
`DiscoveryConfig` 的运行参数 getter 读取当前 `FlexlbConfig.serviceDiscovery`，因此不需要
把运行参数复制回 topology JSON。

## 日志

`logback-spring.xml` 定义 application、PV、sync、sync-consistency 和 FlexLB 文件输出。
日志路径由 `FLEXLB_LOG_PATH` / `FLEXLB_APP_LOG_PATH` 控制，AsyncAppender queue
由 `FLEXLB_LOG_ASYNC_QUEUE_SIZE` 控制。

logger group `flexlb` 包含 `org.flexlb`、`flexlbLogger`、`syncLogger`、
`syncConsistencyLogger`。`FlexlbLogManager` 监听配置快照：

- `observability.logging.level` 热更新整个 group；
- `observability.logging.stdoutEnabled` 动态挂载或卸载 root/PV 的
  `CONSOLE-async`，文件输出始终保留。

`/flexlb/update_log_level` 继续提供显式 HTTP 调级入口。

### 请求 PV 与回放

gRPC Schedule 在校验请求 ID 前创建基础上下文，记录入口时间和 arrival。
请求 ID 在入口解析一次，业务请求、调度回调和转发补偿复用解析结果。缺少 ID 的请求返回
`INVALID_ARGUMENT`，不获取活跃请求计数 token；拒绝分支记录 completion 和 `ENTRY_ERROR` PV，
即使响应 observer 抛出异常也执行收尾。业务请求初始化失败时，基础上下文仍用于完成统计与 PV。

本地 Schedule 的正常完成、异常、取消和 RPC deadline 到期统一经过处理链完成回调。
`completeOnce` 的完成门闩保证收尾只执行一次，`finally` 负责耗时记录、PV 输出和请求计数释放。
路由 Future 完成回调在 `finally` 中移除其注册的取消监听器。取消监听器只触发取消；处理链完成前不读取 PV。哈希阶段收到取消时，
哈希回调完成后跳过选路；选路阶段的取消由调度器在释放处理权后完成结果 Future。
已取消的 RPC 不发送响应，PV 保留取消或超时结果及收尾前完成的遥测。
PV 中 worker 的预测 TTFT 和缓存匹配从候选快照读取，`server_status` 不输出 `prefill_time` 或 `debug_info`。
PV 顶层记录请求 ID、成功标识、错误码和错误消息；`response` 记录 worker 结果和队列信息。准入拒绝原因仅在
确有准入拒绝时记录，成功、取消、超时和其他无准入拒绝原因的结果省略该字段。

`totalUs` 是入口到记录 PV 前的单调时钟耗时；`arrivalMs` 是服务入口时间减调用方
`requestTimeMs`，受两端时钟偏差影响。`seqLen` 记录请求的输入序列长度。
`cacheMatchCount/cacheMatchUs`累计实际缓存查询尝试，角色的缓存选择和决策记录反映最近一次路由尝试。

哈希和路由遥测由串行处理阶段在请求独立的 `RoutingTelemetryState` 中原地累计。
终态读取依赖处理链的完成发布，不与写入并发；字段记录和角色 Map 操作不使用同步锁。
PV 读取时创建不可变 `RoutingTelemetry` 快照，选路中的次数与原因读取不创建快照。
WorkerBatcher 的 `decisionGroup` 独立发布，在 PV 中包含提交组 ID、committedSize、reason、提交时间及请求等待时长；提交组大小不等于保证交付数量，
Engine 的 `batchId` 标识交付批次。

`routingDecisions` 记录 CostBased 的实际候选值。每个角色最多记录 5 个候选并标注截断，
Prefill 包含选中、最短 TTFT、最高有效缓存命中候选。`projectedTtftMs`、`projectedDrainMs`、
`incomingPrefillMs` 的单位为毫秒；无法建模的估计省略。Prefill 记录候选的预测耗时、缓存命中、pending 和 ownershipVersion；Decode 记录 KV 用量、可用量与采样 logWeight。

缓存反馈以请求 ID、角色和 Worker 实例代次关联路由预测，`worker` 使用完整的
`ip:port@engineIndex` 逻辑身份；`prefill_worker_status` 同时记录 `workerIp` 和 `engineIndex`。
Engine 的 `prefixLengthValid`
表示实际命中值有效；有效的 0 表示零命中，无效值不参与差异计算。每个关联记录最多生成一次
`cache_hit_comparison` 和一次 `prefill_worker_status`。比较事件包含实际命中、路由预测、
KVCM 本地匹配、KVCM 本地加 P2P 总匹配，以及 Local Standby 预测；差值统一为实际值减预测值。
预测关联最多保留 100,000 条，保存期限为一小时。Local Standby 对照异步完成，反馈等待上限
为一秒；不可用时省略 Standby 对照，其余比较正常输出。观测回调在 Worker 状态锁外执行。

`tools/pv_request_replay/build_workbook.py` 支持 `routingDecisions` 和
`shortestTtftDecisions` 两种日志结构，以及包含两种结构的日志窗口。Requests 展示请求与缓存
证据；Routing Decisions 展示 CostBased 候选、拒绝统计和决策组；Decision Snapshot
Top5 展示 `shortestTtftDecisions` 的 token-work 估计。预测耗时与 Engine 实测耗时分别展示。
空值表示未记录，零表示已记录且数值为零。HTML 回放通过工作簿读取这些数据，按请求展示候选
及缓存对照。回归测试验证原始 PV 到工作簿、HTML 的字段传递、单位、角色和实例隔离。

PV 不输出 `inputIdsCount`、`requestMessageBytes`、`hashWaitUs`、`hashUs`、
`realMasterHost` 和 `prefillPolicy`。请求长度由 `seqLen` 表达，策略配置通过配置入口查询。
角色选择原因和缓存选择以 `routingDecisions` 为准；外层 `selectionReasons` 和
`cacheMatchSelections` 仅记录候选快照未覆盖或值不同的信息。回放支持顶层终态字段和嵌套终态字段，
并从选中候选读取缓存匹配与选择原因。Schedule 协议响应保留请求 ID，不包含 Master 地址；`/rtp_llm/master/info` 提供 Master 地址供客户端心跳识别。
缓存对比事件在 `source=KVCM` 时通过 `kvcm.hit/delta` 表达调度采用的预测，省略 `routing`；
其他来源使用 `routing`。实际命中、KVCM 本地/P2P 匹配和 Local Standby 匹配各自保留。

请求收尾上报 `app.request.input.ids.count` 和 `app.request.message.bytes`，后者是 Protobuf 序列化大小，
不含 gRPC framing/compression。HTTP Content-Length 使用 `app.request.body.bytes`；未知值不按零上报。


## 指标

全局调度队列成功接收请求时上报 `app.routing.queue.entry.qps`，容量拒绝和关闭后的提交不计入。
`app.flexlb.scheduler.queue.size` 周期性记录全局队列与 Worker 交付队列的等待请求总数，空队列记录 0。
Prefill 与 PDFUSION 均参与周期性队列和在途观测，角色标签使用 Worker 实际角色。
KV 容量与 Waiting 数来自 WorkerStatus；CacheStatus 查询负责缓存键数量和查询周期。
KVCM 选中节点指标记录本地、P2P 拉取和 P2P 后总匹配 Token 数，使用与路由一致的逻辑 Worker 身份。


`FlexMonitor` 提供 GAUGE、COUNTER、QPS 与优先级窗口抽象。opensource 默认使用
`NoOpFlexMonitor`；internal profile 可启用 KMonitor/Prometheus provider。

指标名集中在 `MetricConstant`，主要覆盖：

- engine health、worker status 与状态转换时延；
- routing、queue、dispatch、forward-to-master；
- cache hit、KVCM retry/failure、Local Standby capacity/fallback/comparison；
- block hash、线程池、graceful lifecycle；
- request payload、optimizer trace 与 PV decision 数据。

上报器分布在 common、grpc、cache、sync 模块。新增指标应复用现有 reporter ownership，
不要恢复已删除的旧监控层。

## HTTP 与端口

- 主服务：7001。
- 管理端口：7002。
- `/health` 与 lifecycle hook。
- `/flexlb/update_log_level`。
- `/flexlb/cache_match/status`、`/flexlb/cache_match/failover`。
- `/rtp_llm/schedule`、master notification 和 queue snapshot 等路由入口。
- gRPC 调度入口及 follower-to-master forwarding。

Spring profile 与日志、监控 provider 的具体默认值见
`flexlb-api/src/main/resources/application.yml`。

## Multi-engine endpoint 与观测

`MODEL_SERVICE_CONFIG` 的 endpoint 支持 `multi_engine_num`（默认 1）。显式
`worker_status_port` 必须处于 `[1, 65535]`；N>1 必须指定该端口，并保证
`worker_status_port + N - 1 <= 65535`。每个 index 使用独立的 status gRPC 端口，
frontend HTTP/gRPC 仍为共享物理地址。protocol 只解释 frontend discovery port。
N=1 沿用 discovery 的 legacy gRPC port，因而兼容未配置 `worker_status_port` 的 RTP-LLM。

逻辑 worker identity 为 `ip:http_port@engineIndex`，包括 N=1 的 `@0`。
经 `WorkerStatus` 或 `ServerStatus` 归属的引擎与 cache 指标的 `engineIp` 在 N=1 使用 physical
`ip:http_port`，在 N>1 使用完整 logical identity `ip:http_port@engineIndex`。网络连接仍使用
物理地址。schedule 在 N=1 时省略 `engine_index`，内部 identity 仍保留 index 0。

### Scheduler step 采样

WorkerStatus 的 `last_step_metrics` 表示最近完成的非空 scheduler step，包含 step ID、完成时间、
总调度 Token 数、Prefill 请求数、Prefill Token 数、Token 预算及预算填充率。
字段缺失表示引擎尚无可用 step 观测；纯 Decode step 的 Prefill 请求数和 Token 数为 0。
FlexLB 按逻辑 Worker 和 step ID 去重后，通过 `app.engine.worker.step.*` 上报五项数值。
`phase=prefill` 表示该 step 含 Prefill，`phase=decode` 表示纯 Decode；模型、角色、组和
`engineIp` 标签沿用 Worker 身份。预算填充率为总调度 Token 数 / Token 预算，包含 Decode Token。

这些指标是 WorkerStatus 轮询采样，轮询之间完成的中间 step 不会全部保留。KMonitor 使用
GAUGE + SUMMARY 聚合实际采样值；Micrometer provider 只暴露最近上报值，不提供逐 step 分布。
指标标签不包含 step ID 或完成时间。凑批比较应筛选 `phase=prefill`，避免纯 Decode step
稀释 Prefill 预算填充率。Turbo 转发后的指标前缀为 `dashscope_turbo_backend_flexlb_app_engine_worker_step_`。
