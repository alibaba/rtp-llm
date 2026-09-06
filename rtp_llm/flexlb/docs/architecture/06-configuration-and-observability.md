# Configuration and Observability

## 配置边界

FlexLB 保留两个职责独立的环境文档：

- `FLEXLB_CONFIG`：唯一的 FlexLB 行为配置，包含调度、分发、路由、worker registry、
  observability、服务发现运行参数、cache matching、Optimizer 和一致性/主选举。
- `MODEL_SERVICE_CONFIG`：只保存模型、endpoint、KVCM、Optimizer 的地址和服务发现定位信息。

`MODEL_SERVICE_CONFIG` 不属于动态 `FlexlbConfig` 快照，也不会被 Nacos 覆盖。Nacos
连接、部署标识、日志路径、Spring/OTEL 等基础设施环境变量仍各自独立；它们不构成
FlexLB 行为配置的别名。

## FlexlbConfig 加载与动态更新

`ConfigService` 是统一读取入口。Spring 启动时先初始化两个 `ConfigSource`：

1. `EnvironmentConfigSource`（priority 1）读取严格的 `FLEXLB_CONFIG` 基线。
2. 若配置了 Nacos，`NacosConfigSource`（priority 2）读取初始配置并注册 listener。

来源按 priority 从低到高合并。Nacos 内容是 `FlexlbConfig` 的递归部分 JSON：
对象字段递归覆盖，未出现或从 Nacos 删除的字段保留当前内存值，数组和标量整体替换；
tagged union 的 `type` 变化会替换整个分支。空字符串和 `{}` 都是合法 no-op。每次合并后都使用与
`FLEXLB_CONFIG` 相同的严格解析和跨字段校验：

- 拒绝重复 key、未知字段、`null`、标量 coercion、数值枚举和尾随 JSON；
- 拒绝 tagged union 非活动分支的字段；
- 拒绝违反 scheduler / dispatcher / router 等组合约束的配置。

非空初始来源读取或校验失败会阻止应用启动。运行时非法推送不会替换当前
last-known-good 快照；合法推送原子替换 `FlexlbConfig`，随后通知监听器。

Nacos 层不区分“热生效”与“重启生效”：它只发布最新有效快照。业务组件每次读取快照，
就可以热生效；在 Bean 初始化时缓存的值，则在重启后生效。

旧的字段级行为变量 `BLOCK_HASH_STRATEGY`、`FLEXLB_LOG_LEVEL`、
`ENABLE_STDOUT_LOG`、`ENABLE_FALLBACK` 不再覆盖 JSON。行为配置只认
`FLEXLB_CONFIG`，避免嵌套字段到环境变量名的隐式转换。

### Nacos 连接

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FLEXLB_NACOS_SERVER_ADDR` | 无 | 未配置时禁用 Nacos |
| `FLEXLB_NACOS_DATA_ID` | 部署标识 | 显式 DataId |
| `FLEXLB_NACOS_GROUP` | `DEFAULT_GROUP` | Nacos group |
| `FLEXLB_NACOS_NAMESPACE` | 空 | Nacos namespace |

未显式配置 DataId 时，部署标识优先使用
`SPECTRUM_WORKSPACE_ID`、`SPECTRUM_APPLICATION_NAME`、
`SPECTRUM_DEPLOYMENT_NAME` 组成
`spectrum:<workspace>:<application>:<deployment>`；旧环境回退 `HIPPO_ROLE`。

Nacos 部分更新示例：

```json
{
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
