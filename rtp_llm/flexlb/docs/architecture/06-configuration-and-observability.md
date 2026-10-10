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

Spring 按标准规则读取环境变量，例如 `SERVER_PORT` 对应 `server.port`，
`FLEXLB_MONITOR_PROVIDER` 对应 `flexlb.monitor.provider`。内源部署通过
`FLEXLB_MONITOR_PROVIDER=kmonitor` 启用 KMonitor 指标上报。

容器启动脚本以物理内存和 cgroup 限额的较小值计算 JVM 内存预算；至少 16GiB 的容器
保留 2GiB direct memory 上限，小规格继续按比例分配。默认堆同时受 direct memory、
metaspace、code cache 以及容器内存的 1/8 native 余量约束。显式堆覆盖也校验初始堆不超过最大堆，
且上述预算总和不超过限额；超配时启动脚本报错。该校验约束 JVM 内存池配置，线程栈和
native allocation 的实际占用仍取决于负载。

显式堆大小必须携带 `k/m/g` 单位，例如 `2048m` 或 `2g`；不接受无单位数值，避免将业务希望
配置的 MB 数按 JVM 的字节语义使用。GC 线程数优先采用有效的正整数 CPU 配额；无效或缺失时
读取系统 CPU 数，仍不可用时使用 1，确保 G1 的并行 GC 线程数至少为 1。

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

三种来源都返回原始字符串，复用 `ConfigDocumentParserResolver` 和 v0/v3 解析器：
优先使用文档内的 `schemaVersion`，否则使用 `FLEXLB_CONFIG_SCHEMA_VERSION`（默认 `0`）。
UniConfig 和 Nacos 的内容直接使用同一种配置 JSON，不增加包装层。

格式归一化后，由现有 `FlexlbConfigMerger` 执行递归部分更新：
对象字段递归覆盖，未出现或从外部配置删除的字段保留当前内存值，数组和标量整体替换；
`cacheMatching`、`consistency`、`scheduler.ordering` 和 `scheduler.decision` 的 `type`
变化会替换整个分支，避免保留新模式不支持的抢占或凑批参数。`scheduler`、`dispatcher`
自身的 `type` 字段变化仍按递归覆盖处理，
保留未出现的兄弟字段。v3 文档 `{"schemaVersion":3}` 是 no-op；
没有显式版本的 `{}` 则按版本选择规则解析，默认走 v0 兼容转换。
每次合并后都使用与 `FLEXLB_CONFIG` 相同的严格解析和跨字段校验：

- 拒绝重复 key、未知字段、`null`、标量 coercion、数值枚举和尾随 JSON；
- 拒绝 tagged union 非活动分支的字段；
- 拒绝违反 scheduler / dispatcher / router 等组合约束的配置。

选定的外部来源初次读取或校验失败会阻止应用启动，不会自动切换到低优先级来源。
运行时读取失败或非法更新不会替换当前 last-known-good 快照，后续更新恢复正常后
继续应用；合法更新原子替换 `FlexlbConfig`，随后通知监听器。

`scheduler.type`、`dispatcher.type` 和 `scheduler.decision.type` 在启动时确定，运行时保持
当前模式。部分更新中已知但不同的模式值会被忽略，同一对象中的数值参数仍继续合并；未知模式值
仍会校验失败并保留当前快照。切换模式需要重启 Master，并在启动文档中配置完整的目标分支。
其余业务组件每次读取快照的参数可以热生效；在 Bean 初始化时缓存的值，则在重启后生效。

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
启动阶段读取配置失败时，每隔 1 秒持续重试，没有次数上限；连接失败、非 HTTP 200
（包括 key 不存在时的 404）和非法 JSON 都会触发重试。文档既没有 `consistency`，
也没有 `flexlbSyncConsistencyConfig` 时同样重试。
线程被中断后停止等待并启动失败。文档读取成功后，业务配置校验失败仍会阻止启动。
启动前应先在部署 UniConfig 页面保存合法配置。
运行时 HTTP 异常保留有效快照并继续按 30 秒间隔重试。

### Decode 抢占交付方式

`scheduler.ordering.preemption.allowedVictimStages` 包含 `DECODE_ENGINE_OWNED` 时，
`engineCancellation.mode` 可设为 `RPC` 或 `RETURN`，默认 `RPC`。
`RPC` 使用主动 Cancel 和完成确认；`RETURN` 将待抢占请求的字符串 ID 随目标 Decode
路由返回，由客户端/Decode 消费，要求 `dispatcher.type=NON_BATCH`。
抢占整体超时由 `scheduler.ordering.preemption.timeoutMs` 控制。该配置随请求快照绑定，
外部配置来源与其他调度字段一致。

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
`spectrum:<workspace>:<application>:<deployment>`。Spectrum 三元组不完整时，
要求 `BIZ_NAME`、`DEPLOYMENT_NAME`、`ZONE_NAME` 全部非空，依次以冒号拼接为
`<bizName>:<deploymentName>:<zoneName>`，例如 `dash_pd:ea118_RTX_PRO_5000_72GB:master`。
变量值会去除首尾空白。两组三元组均不完整时启动失败，错误信息列出六个字段及其值；
`null` 表示变量未设置或只有空白。使用第二组三元组时，部署身份不属于 Spectrum，不能用于 UniConfig。
部署标识同时用于 ZooKeeper 选举和主节点变更通知；
显式设置 `FLEXLB_NACOS_DATA_ID` 只覆盖 Nacos DataId。

选中 Nacos 配置来源时，启动阶段必须读取到非空配置正文。配置不存在、正文为空或只有空白、
读取异常均导致启动失败。配置缺失或空白的错误包含 DataId、group 和 namespace；读取异常保留原始原因。
启动过程不回退其他 DataId、环境变量配置或默认配置。

UniConfig / Nacos 的 v3 部分更新示例：

```json
{
  "schemaVersion": 3,
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

公共 schema 当前为 version 3，按责任分区：

- `scheduler`：`DIRECT` / `QUEUE`；QUEUE 拥有 ordering、capacity 和 lifecycle。
- `dispatcher`：`BATCH` / `NON_BATCH`。QUEUE 模式可用
  `maxInflightPerEncoderWorker` 限制每台 Encoder 的在途请求数；未配置时不设额外上限，
  不影响 DIRECT 或未配置 Encoder 的模型。
- `router`：角色 availability、execution estimator、selector、cache affinity 和
  group selector。
- `workerRegistry`：worker health 与 cache-status 刷新策略。
- `observability.cacheHit`：recent-key window、指标和理论命中日志。理论命中查询、历史池更新、
  指标与请求日志由独立单线程按入队顺序处理；请求线程仅提交任务。
  后台任务直接读取路由完成后保持不变的请求字段；日志与指标开关在执行时从配置服务读取。
  等待队列最多容纳 100,000 个任务，满时丢弃统计样本并汇总告警，不阻塞请求线程。
  `recentKeyWindow.maxKeyOccurrences` 默认 `1000000`，限制保留的唯一 key 数量；
  `durationMs` 默认 `1800000`。实际容量取配置上限与 `10 × Prefill 数量 × (单台总 KV Token / blockSize)`
  的较小值。Prefill 的最大 KV Cache 和 blockSize 相同，只读取第一台的容量；该台上报容量后，
  后台线程才分配历史池；容量尚未产生时不记录样本。历史池使用固定数组，不存储 KV 内容；
  默认上限对应约 49 MiB JVM 堆，容量估算较小时实际占用更低。
  配置上限和窗口时长在 Master 初始化时读取，池容量在首次分配时确定；修改 Nacos 配置或
  Prefill 规模后重启 Master 重新计算容量。
  理论命中监控使用 `app.cache.theory.hit.count`、`app.cache.theory.total.count` 与
  `app.cache.theory.hit.ratio`。前两个指标是 Counter，每个请求分别上报理论命中 Tokens 和
  输入 Tokens；命中率是该请求的理论命中 Tokens 与输入 Tokens 的比值。历史记录有效期只影响
  本次请求是否匹配，不影响已经上报的 Counter。
- `observability.logging`：FlexLB logger group 级别与 root/PV stdout 开关。
- `serviceDiscovery`：connect/read timeout、poll interval 与连接池运行参数。
  持续空结果保留已有地址，不按空结果次数或持续时间撤销 worker。
- `cacheMatching`：`LOCAL_SYNC` / `KVCM` tagged union；KVCM 分支拥有查询、健康、远端命中
  （`medium` / `topKHostCount` / `backendTypes`）和 Local Standby 参数。
- `optimizer`：启用开关和服务发现轮询间隔。
- `consistency`：`NONE` / `ZOOKEEPER` tagged union；ZooKeeper 分支拥有连接和 master
  刷新参数。
- `enableFallback`：默认 `false`；启用时调度入口在转发和路由前返回错误码 `8600`，
  由调用方执行 domain fallback。
- `fallbackBatchTokenCapacity`：默认 `1048576`；Engine 未声明
  `max_batch_tokens_size` 和 `max_seq_len` 时使用的最终 batch token 容量兜底值。
  优先使用 Engine 上报的 `max_batch_tokens_size`，其次使用 `max_seq_len`。
- `internalRuntime`：代码内部设置，不接受公共 JSON 输入。

`DIRECT + BATCH` 非法；可选配置应省略，不能写 `null`。完整示例和 selector
矩阵见根目录 [README](../../README.md)。

### KVCM 查询参数热更新

KVCM 模式下，Nacos v3 配置可设置并热更新查询参数：

```json
{
  "schemaVersion": 3,
  "cacheMatching": {
    "type": "KVCM",
    "topKHostCount": 3,
    "backendTypes": []
  }
}
```

`topKHostCount` 为非负整数，默认 3；0 只计算本地命中。
`backendTypes` 默认空列表，只接受 `ST_TAIRMEMPOOL` 和 `ST_EVENT_REPORT_L2`，
例如 `["ST_TAIRMEMPOOL", "ST_EVENT_REPORT_L2"]`。空列表原样发送给 KVCM。
运行时省略字段保留当前值；显式配置 `[]` 清空来源列表。非法值或 JSON `null`
会拒绝整次更新，保留上一份有效配置。旧 `globalKvsHostCount` / `enableP2p` 不再接受。

Nacos 监听更新通过 `ConfigService` 校验并发布到 `CacheMatchConfiguration`；
下一次查询使用新配置，同一次查询的重试继续使用原配置快照和原绝对 Deadline。
查询线程只读取内存快照，不访问 Nacos。`cacheMatching.type` 仍需重启才能切换。

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

gRPC Schedule 在校验请求 ID 前创建基础上下文，记录入口时间、请求大小和 arrival。
请求 ID 在入口解析一次，业务请求、调度回调和转发补偿复用解析结果。缺少 ID 的请求返回
`INVALID_ARGUMENT`，不获取活跃请求计数 token；拒绝分支记录 completion 和 `ENTRY_ERROR` PV，
即使响应 observer 抛出异常也执行收尾。业务请求初始化失败时，基础上下文仍用于完成统计与 PV。
同一业务请求先调度 Encoder、再调度 Generation 时，Schedule PV 的 `phase` 区分两次决策，
`server_status.role` 保留实际选中的角色。Encoder 选点失败和 WorkerStatus 生命周期异常
沿用请求错误码与状态日志。调度摘要日志输出可选的 `encoder_cache_hit_len`，Schedule PV 输出
可选的 `encoderCacheHitLen`；缺失值与显式 0 可区分。

本地 Schedule 的正常完成、异常、取消和 RPC deadline 到期统一经过处理链完成回调。
`completeOnce` 的完成门闩保证收尾只执行一次，`finally` 负责耗时记录、PV 输出、取消监听器
移除和请求计数释放。取消监听器只触发取消；处理链完成前不读取 PV。
选路阶段的取消由调度器在释放处理权后完成结果 Future。
已取消的 RPC 不发送响应，PV 保留取消或超时结果及收尾前完成的遥测。
PV 顶层与嵌套 `response` 的成功标识、错误码和错误消息表达同一终态；准入拒绝原因仅在
确有准入拒绝时记录，成功、取消、超时和其他无准入拒绝原因的结果省略该字段。

`totalUs` 是入口到记录 PV 前的单调时钟耗时；`arrivalMs` 是服务入口时间减调用方
`requestTimeMs`，受两端时钟偏差影响。gRPC 路径记录收到的
`requestMessageBytes`（gRPC 接收的未压缩 protobuf 字节数，不含帧头）。transport tracer
累计已读取/解压的字节，并通过 gRPC Context 传到入口；调度线程不为监控额外遍历 PB。
`app.request.message.bytes` 保留该 Message 口径；绕过 transport 的直接调用不制造 0 样本。
`cacheMatchCount/cacheMatchUs`累计实际缓存查询尝试，角色的缓存选择和决策记录反映最近一次路由尝试。

路由遥测由串行处理阶段在请求独立的 `RoutingTelemetryState` 中原地累计。
终态读取依赖处理链的完成发布，不与写入并发；字段记录和角色 Map 操作不使用同步锁。
PV 读取时创建不可变 `RoutingTelemetry` 快照，选路中的次数与原因读取不创建快照。
WorkerBatcher 的 `decisionGroup` 独立发布，包含提交组 ID、policy、dispatcher、
worker、committedSize、reason、提交时间及请求等待时长；提交组大小不等于保证交付数量，
Engine 的 `batchId` 标识交付批次。

`routingDecisions` 记录 CostBased 的实际候选值。每个角色最多记录 5 个候选并标注截断，
Prefill 包含选中、最短 TTFT、最高有效缓存命中候选。`projectedTtftMs`、`projectedDrainMs`、
`incomingPrefillMs` 的单位为毫秒；无法建模的估计省略。Prefill 记录策略参数、缓存亲和阈值、
候选总体最小/最大命中、pending 和 ownershipVersion；Decode 记录 KV 用量、可用量与采样 logWeight。

缓存反馈以请求 ID、角色和 Worker 实例代次关联路由预测，`worker` 使用完整的
`ip:port@engineIndex` 逻辑身份；`prefill_worker_status` 同时记录 `workerIp` 和 `engineIndex`。
选定 Worker 时上报 `app.cache.kvcm.predicted.tokens/ratio`；Local Standby 预测结果与所选 Worker 齐备时上报
`app.cache.local.standby.predicted.tokens/ratio`。有效的空匹配产生 0，查询失败不产生预测数据点。
Engine 的 `prefixLengthValid`
表示实际命中值有效；有效的 0 表示零命中，无效值不参与差异计算。每个关联记录最多生成一次
`cache_hit_comparison` 和一次 `prefill_worker_status`。实际命中 Counter 及输入 Tokens Counter
仅在收到有效反馈后上报，用于计算实际全局命中率。比较事件包含实际命中、路由预测、
KVCM 本地匹配、KVCM 本地加远端的 global 总匹配，以及 Local Standby 预测；差值统一为实际值减预测值。
预测关联最多保留 100,000 条，保存期限为一小时。Local Standby 对照异步完成，反馈等待上限
为一秒；不可用时省略 Standby 对照，其余比较正常输出。观测回调在 Worker 状态锁外执行。

`tools/pv_request_replay/build_workbook.py` 支持 `routingDecisions` 和
`shortestTtftDecisions` 两种日志结构，以及包含两种结构的日志窗口。Requests 展示请求与缓存
证据；Routing Decisions 展示 CostBased 候选、策略参数、拒绝统计和决策组；Decision Snapshot
Top5 展示 `shortestTtftDecisions` 的 token-work 估计。预测耗时与 Engine 实测耗时分别展示。
空值表示未记录，零表示已记录且数值为零。HTML 回放通过工作簿读取这些数据，按请求展示候选
及缓存对照。回归测试验证原始 PV 到工作簿、HTML 的字段传递、单位、角色和实例隔离。

## 指标

`FlexMonitor` 提供 GAUGE、COUNTER、QPS 与优先级窗口抽象。opensource 默认使用
`NoOpFlexMonitor`；internal profile 可启用 KMonitor/Prometheus provider。

指标名集中在 `MetricConstant`，主要覆盖：

- engine health、worker status 与状态转换时延；
- routing、queue、dispatch、forward-to-master；
- cache hit、KVCM retry/failure、Local Standby capacity/fallback/comparison；
- 线程池、graceful lifecycle；
- request payload、optimizer trace 与 PV decision 数据。

`app.engine.health.check.engine.worker.number.service.discovery.result{model,role}` 统计所有已配置 endpoint
成功返回的原始 Host 数量；空返回上报 0，即使健康检查继续沿用缓存。某 endpoint 查询失败时
不以缓存或 0 伪装完整的发现结果，本轮跳过总数上报，失败由 discovery error/timeout 指标记录。
`app.engine.health.check.engine.{prefill,decode,encoder}.worker.number` 则统计 WorkerDirectory
中已展开的逻辑 Worker 数，使用公共 BIZ_NAME 与 Master 标签定位部署，不含 model 标签。
两种数量含义不同，不能互相替代。

Block Size 保留 `app.cache.block.size{role}`：同一角色的 Worker 共享相同配置，
每 2 秒读取该角色首个大于 0 的 WorkerStatus 值，覆盖 PREFILL、DECODE、PDFUSION、ENCODER。
首次状态到达前 EngineObservation 已初始化，blockSize=0，不上报未就绪值。
该指标不增加 engineIp 标签；大盘按 BIZ_NAME 与 role 展示，删除旧 model 筛选。

既有 WorkerStatus/cache 的轮询成功周期、RPC 耗时、运行队列时间、任务列表大小、cache key 数
和 KV 容量指标保留裸 IP 的 `engineIp` 标签。同 IP 多实例使用相同标签上报，Gauge 是逐样本值，
不表示 IP 合计。缓存预测对照使用实际 WorkerStatus 的实例数选择精确身份：单 Engine 为
`ip:port`，多 Engine 为 `ip:port@engineIndex`；PV 和路由/cache 内部继续使用完整 logical identity。
对照的绝对 Token 差值不依赖分母，输入 Token 数缺失或非正时仍可记录；Counter 和比例仅在
输入 Token 数为正时报告，不能将绝对差值的样本数量当作加权命中率的分母。

`app.engine.balancing.master.select.detail` 保留原有 `role/success/code` 标签；独立的
`app.engine.balancing.master.worker.select.detail` 增加有限枚举 `reason` 和精确 `engineIp`。
`app.engine.worker.info.step.latency.var` 与 `app.engine.worker.info.running.query.len.var`
按 role 上报逻辑 endpoint 方差，不含 model。后者 Prefill 使用 work-ms，Decode 和状态角色使用
活动任务数，方差单位分别为 work-ms² 与 count²，不能跨角色混合比较数值。
Step 延迟方差单位为 ms²。上述指标保留原有名称，通过标签聚合和面板说明表达实际口径。
`app.cache.hit.count` 和 `app.cache.input.tokens` 是按所选 worker 累计的 Token Counter，
全局命中率使用同一窗口内的命中 Token 增量除以输入 Token 增量，不能求单请求命中率的平均值。
`app.engine.zk.master.event` 保留按事件类型上报 `1.0` 的既有 Gauge 口径；
独立的 `app.engine.zk.master.event.time.ms` 用 epoch-ms Gauge 展示最近事件时间。
`app.engine.worker.status.scheduler.to.running.ms` 中的 scheduler 是 Engine 调度器；该值
与 `app.engine.worker.status.engine.waiting.to.running.ms` 相同，均取 Engine 的
`running_entered_time_ms - waiting_entered_time_ms`。前者保留已有展示口径，后者显式标明观测来源；
不能将二者相加或作为 Master 与 Engine 的两端耗时比较。
`remote_kv_wait_ms` 是引擎报告的时长，0ms 可以表示没有远程 KV 等待，仍作为有效样本。
phase 时间戳的 0 表示未知，缺少时间戳时不报告相减得到的耗时。

Engine received→waiting 和 waiting→running 使用 WorkerStatus 中 Engine 自己记录的
阶段时间戳计算。轮询间隔影响 Master 收到样本的时间，不用探测到状态的时间估算阶段耗时；
完成任务的时间戳完整时，无需在轮询中逐个观察到这些阶段。

gRPC 服务端执行器与批次发送执行器初始化后立即通过 `FlexMonitor` 上报一次状态，之后每 2 秒上报忙碌线程数、总线程数和
排队任务数；gRPC 服务端另报最大线程数、累计拒绝任务数。它们与其它线程池使用相同的
provider：KMonitor 部署上报 `whale-lb.grpc.server.executor.*` 和
`whale-lb.dispatch.executor.*`，无需单独接入 Micrometer 采集。
忙碌线程、排队数、总线程数、最大线程数使用 GAUGE，scrape 读取最近一次上报的快照，
最多滞后 2 秒；不能用这些快照捕获短于采样间隔的峰值。瞬时指标保留原名。
拒绝任务数使用 `grpc.server.executor.caller.runs` COUNTER。每次拒绝发生时直接上报 1，
由监控 provider 累加，不维护本地累计值或上次上报位置。初始化及每 2 秒上报 0，
使没有拒绝事件时指标仍可查询；零上报不增加计数。Master 重启后计数重新开始。拒绝数包含队列或线程饱和、
线程池关闭后提交触发的拒绝。保留 `grpc.server.executor.caller.runs` 上报名，
面板显示“任务拒绝数”，使用 `increase(...[1m])` 查看窗口拒绝数。
已初始化的空闲线程池持续上报忙碌线程数和排队任务数为 0，总线程数仍包含空闲线程。

`JvmGcMetricsReporter` 接收 JVM 的 GC 通知，每次记录一次回收和本次暂停毫秒数，
通过 `FlexMonitor` 的 PRECISE COUNTER 上报 `app.jvm.gc.collection.count` 与
`app.jvm.gc.pause.total.ms`。按 `gc`、`collector`、`pid` 区分进程与收集器：
G1 的 `young` 包含 Mixed，`full` 对应 `G1 Old Generation`，`concurrent` 只统计
并发周期中的暂停阶段，不表示整个并发标记周期。其它收集器标为 `other`。
每秒上报零增量，保持未发生 GC 的序列可查询；关闭组件时移除 GC listener。
面板展示统计窗口两端的计数差值，以及暂停毫秒数差值除以次数差值；
例如窗口内两次暂停为 10 ms、30 ms，展示 2 次和平均 20 ms。
没有 GC 的窗口次数为 0，平均耗时为空。底层计数器用于差分，不作为累计趋势展示；
PID 隔离重启前后的计数器，缺少窗口边界样本时保留无数据。

Encoder 的 worker 数由周期指标上报 `app.engine.health.check.engine.encoder.worker.number`；
WorkerStatus 成功轮询上报 `app.flexlb.encoder.pending.request.count` 和
`app.flexlb.encoder.selection.load`、`app.flexlb.encoder.uncached.token.load`。
三项按 `engineIp`、`role=ENCODER` 标记，分别表示尚未在 WorkerStatus
看到的本地选点数、`running + waiting + pending` 并发数和在途编码工作量代理值。
最后一项在首次 WorkerStatus 前采用 Client 的 MM token 预测值，之后采用活动任务的合成输入
`input_length`；两者口径可能略有差异，finished 中的长度不参与该指标。
通用 KV 指标仍上报，但不参与 Encoder 选点。
`app.flexlb.tracked.request.count` 按
`role=PREFILL`（Generation）和 `role=ENCODER`（Encoder）分别上报，
`engineIp=scheduler`；Prefill 序列只统计 Generation。节点侧使用 `app.flexlb.inflight.request.count`，按真实 `engineIp` 和
role 分开展示。NON_BATCH（包括 DIRECT）的 `role=PREFILL/PDFUSION`
表示本地排队、已提交未确认、已确认但仍占用容量的请求，
加上引擎报告的额外活动请求。同一请求按 ID 去重；WorkerStatus 仅提供 running/waiting 数量、
缺少请求明细时，对无法确认重合的请求保守计数。该指标不是引擎 GPU 当前执行请求数。
NON_BATCH（包括 DIRECT）的 Prefill 序列与 `dispatcher.maxInflightPerPrefillWorker` 的请求限流计数一致：
设为 4 时，已确认请求仍占用名额，完成或本地清理后才释放。若引擎有其它来源的请求，或本地超时清理后
引擎仍在处理请求，不能仅凭该配置保证引擎实际并发不超过 4。
BATCH 的上限单位是批次，仍使用 `app.flexlb.inflight.batch.count` 查看已提交、尚未清理完成的批数；
该批数不包含派发前短暂预留的批槽，4 个批次可以包含多于 4 个请求。
BATCH 的 `app.flexlb.inflight.request.count` 保持原有口径：已提交且所在批次尚未被 WorkerStatus
确认的请求数，不含本地排队请求；不新增逐请求确认状态，也不上报新 unconfirmed 指标。

`role=DECODE` 保持原有口径：本地未确认预留数减去 Master 排队数，收到 `KV_ALLOCATED/RUNNING`
后不再计入；仅收到 `RECEIVED` 仍计入。不新增 Decode 指标或改变 Decode 容量规则。
Decode 总负载仍使用 `app.flexlb.decode.total.load`（已确认请求加本地预留，含排队）。
因此不能跨 dispatcher 模式或 Prefill/Decode 聚合 `inflight.request.count` 并解释为同一种请求数。

`app.flexlb.worker.status.unconfirmed.request.count` 仅由 NON_BATCH PREFILL/PDFUSION 上报，
单独统计已提交、尚未在 WorkerStatus 中出现的请求，
沿用 `role/engineIp/scope=worker` 标签，不含本地排队请求。首次有效活动任务上报（包括 RECEIVED）
后请求阶段不再是 COMMITTED，即使之后暂时缺少该请求的明细也不重新计入。
该指标直接复用现有请求阶段，不新增状态字段或改变调度、容量释放规则。
NON_BATCH Prefill 的旧 `inflight.request.count` 未确认口径面板应切换到新指标；总占用面板使用原指标名。
BATCH 和 Decode 面板保持原指标名及原口径。
上述 Gauge 默认每 2 秒采样，无法证明采样间隔内没有短暂超限。大盘按 Master 和逻辑 worker 分线，
不要将不同 Master 的样本直接相加来判断单 worker 的限流效果。
`/rtp_llm/inflight_status` 的 `scheduler_tracked` 返回两个阶段正在跟踪的请求总数，包含排队请求；
`scheduler_inflight` 保留为该值的兼容别名，供现有清账和测试工具使用。它不表示节点侧的未确认在途数。
Decode 预留字段 `inputKvTokens = max(0, seqLen)`，
`inputAndMaxOutputKvTokens = inputKvTokens + max(0, maxNewTokens)`（溢出时饱和）。
两者分别表示输入预留和输入加最大输出预留，不是输出长度预测，后者包含前者。
引擎确认后的容量使用引擎上报的实际 KV 数，不再使用请求上限。
监控分别使用 `app.flexlb.decode.inflight.hard.kv.reserved.tokens` 与
`app.flexlb.decode.inflight.kv.reserved.tokens`，包含排队预留；中文图例使用上述计算口径。
周期统计复用 Prefill 快照，调度器一次遍历计算数量和最大年龄；Decode admission
上报仅收集数值，不构造请求明细。

`auto_tpm.preemption.target_invalid.count` 统计 Decode 抢占目标校验失败的尝试次数，
使用 QPS + NORMAL（20 秒聚合）上报。一次计划即使包含多个目标也只计一次，标签仅有
`mode=return/rpc` 和固定的 `reason`：`victim_state_changed`（预留或派发状态变化）、
`victim_already_claimed`（目标已被其他抢占占用）、`priority_not_preemptible`（目标优先级不允许抢占）、
`cancel_target_unavailable`（RPC 取消地址无法取得）、`request_claim_rejected`（RPC 无法锁定该请求进行抢占）。
这类失败会结束本次高优先级请求的调度；容量不足、取消 RPC 返回 NOT_FOUND、超时不计入此指标。
未发生事件时可能没有时间序列，不能据此断言上报链路正常。
`auto_tpm.request.lifecycle.failures.total{stage}` 是 PRECISE COUNTER，
`stage=registration` 记录注册异常，`stage=terminal_cleanup` 记录资源清理失败或请求最终状态提交异常。
初始化及周期采样对这两个 stage 报告 0，尚未发生异常时也能显示已初始化的计数。
这些异常保留 request_id 日志供定位，指标标签不包含请求 ID 或错误文本。
旧 `auto_tpm.inflight_settle_miss.count{kind}` 随独立 inflight 台账的清理路径一起停用：
当前注册、资源跟踪和结束记录由同一个 RequestSlot 保持，迟到或重复回调无需找回旧台账。
它不映射为抢占目标校验失败。旧竞态告警应退役，另启用新的生命周期异常增量告警。

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
Encoder、step、阶段耗时及缓存预测对照的 `engineIp` 在 N=1 使用 physical
`ip:http_port`，在 N>1 使用完整 logical identity `ip:http_port@engineIndex`；上述兼容指标保留裸 IP。
网络连接仍使用
物理地址。schedule 在 N=1 时省略 `engine_index`，内部 identity 仍保留 index 0。

### Scheduler step 采样

WorkerStatus 的 `last_step_metrics` 表示最近完成的非空 scheduler step，包含 step ID、完成时间、
总调度 Token 数、Prefill 请求数、Prefill Token 数、Token 预算及预算填充率。
字段缺失表示引擎尚无可用 step 观测；纯 Decode step 的 Prefill 请求数和 Token 数为 0。
FlexLB 按逻辑 Worker 和 step ID 去重后，通过 `app.engine.worker.step.*` 上报五项数值。
`phase=prefill` 表示该 step 含 Prefill，`phase=decode` 表示纯 Decode；角色、组和
`engineIp` 标签沿用 Worker 身份。预算填充率为总调度 Token 数 / Token 预算，包含 Decode Token。

这些指标是 WorkerStatus 轮询采样，轮询之间完成的中间 step 不会全部保留。KMonitor 使用
GAUGE + SUMMARY 聚合实际采样值；Micrometer provider 只暴露最近上报值，不提供逐 step 分布。
指标标签不包含 step ID 或完成时间。凑批比较应筛选 `phase=prefill`，避免纯 Decode step
稀释 Prefill 预算填充率。Turbo 转发后的指标前缀为 `dashscope_turbo_backend_flexlb_app_engine_worker_step_`。

### Prefill 非末块 Token 数

完成 Prefill 请求后，`app.engine.worker.status.prefill.nonfinal.chunk.min.tokens` 和
`app.engine.worker.status.prefill.nonfinal.chunk.max.tokens` 分别上报该请求内非末块 Chunk 的
最小、最大 Token 数。末块可能不足一个完整 Chunk，不参与统计；没有非末块样本的请求不向
这两个指标报告 0，仍正常报告 `app.engine.worker.status.prefill.step.count`。
例如分为 8192、8192、616 Tokens 的请求，Step 数为 3，非末块最小和最大值均为 8192。

两个指标保留请求内统计口径，以 GAUGE + TRIVIAL 注册，KMonitor 每 60 秒发送一次聚合结果，
不生成额外的 SUMMARY 分位指标；显式配置的 `FLEXLB_MONITOR_PRIORITY` 可覆盖此周期。
曲线均值分别是有非末块样本请求的最小值均值、最大值均值，不是所有 Chunk 的均值或窗口极值。
业务调用只更新本地统计，不逐请求发送网络数据，也不额外抽样。指标名以单位 `tokens` 结尾，
请求内最小、最大值由 `min` / `max` 中间段区分。

监控迁移配置在 `tools/monitoring/metric_migration.json`；
`python3 tools/monitoring/migrate_metrics.py --input dashboard.json --output dashboard-migrated.json`
读取 Grafana 导出的 JSON，保留现有指标名称与已正确的查询，仅修正失效标签筛选、
累计 Counter 查询与展示口径，并补充缺失面板和独立告警规则；不会写入线上 Grafana。
在对应版本代码发布时使用生成的配置；保留导出原件以便整体回退。
