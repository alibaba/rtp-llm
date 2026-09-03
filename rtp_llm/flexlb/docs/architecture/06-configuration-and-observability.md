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

正文就是完整的配置 JSON 字符串，与 Nacos 文档格式一致。连接和读取超时各为 3 秒，
每 10 秒轮询一次，仅在正文变化时发布更新。Turbo 侧通常有约 1 分钟缓存，因此控制台
提交后会有分钟级传播延迟，不能视为提交后立即生效。
初次读取遇到连接失败、非 HTTP 200（包括 key 不存在时的 404）或非法配置会启动失败；
启动前应先在部署 UniConfig 页面保存合法配置。运行时 HTTP 异常保留有效快照并继续重试。

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

## 指标

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
