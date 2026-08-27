# Configuration and Observability

## 配置边界

FlexLB 保留三个职责独立的环境文档：

- `FLEXLB_CONFIG`：调度、分发、路由、worker registry 与 observability 行为。
- `MODEL_SERVICE_CONFIG`：模型拓扑、服务发现、KVCM 与 Optimizer。
- `FLEXLB_SYNC_CONSISTENCY_CONFIG`：ZooKeeper 一致性和主选举。

后两个文档不属于动态 `FlexlbConfig` 快照，也不会被 Nacos 覆盖。

## FlexlbConfig 加载与动态更新

`ConfigService` 是统一读取入口。Spring 启动时先初始化两个 `ConfigSource`：

1. `EnvironmentConfigSource`（priority 1）读取严格的 `FLEXLB_CONFIG` 基线。
2. 若配置了 Nacos，`NacosConfigSource`（priority 2）读取初始配置并注册 listener。

来源按 priority 从低到高合并。Nacos 内容是 `FlexlbConfig` 的递归部分 JSON：
对象字段递归覆盖，未出现的字段保留当前值，数组和标量整体替换。每次合并后都使用与
`FLEXLB_CONFIG` 相同的严格解析和跨字段校验：

- 拒绝重复 key、未知字段、`null`、标量 coercion、数值枚举和尾随 JSON；
- 拒绝 tagged union 非活动分支的字段；
- 拒绝违反 scheduler / dispatcher / router 等组合约束的配置。

初始来源读取或校验失败会阻止应用启动。运行时非法推送不会替换当前
last-known-good 快照；合法推送原子替换 `FlexlbConfig`，随后通知监听器。
`updateTrafficPolicy` 同样创建并发布新快照，不原地修改已发布对象。

环境基线额外兼容：

- `BLOCK_HASH_STRATEGY`：`VLLM` / `SGLANG`；
- `FLEXLB_LOG_LEVEL`：`TRACE` / `DEBUG` / `INFO` / `WARN` / `ERROR`；
- `ENABLE_STDOUT_LOG`：严格的 `true` / `false`；
- `ENABLE_FALLBACK`：严格的 `true` / `false`，覆盖 JSON 中的 `enableFallback`。

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
- `blockHashStrategy`：cache block hash 策略。
- `enableFallback`：默认 `false`；启用时调度入口在转发和路由前返回错误码 `8600`，
  由调用方执行 domain fallback。
- `internalRuntime`：代码内部设置，不接受公共 JSON 输入。

`DIRECT + BATCH` 非法；可选配置应省略，不能写 `null`。完整示例和 selector
矩阵见根目录 [README](../../README.md)。

## MODEL_SERVICE_CONFIG

独立反序列化为 `ServiceRoute`：

- `service_id` 和 `role_endpoints`；
- endpoint 的 `address`、`protocol`、`path`、`worker_status_port`、`discovery`；
- discovery 类型 `static-env`、`vipserver`、`dashscope`；
- 可选 `kvcm`、`local_standby` 和 `optimizer`。

该拓扑在相关 Spring Bean 创建时使用；动态 `FLEXLB_CONFIG` 更新不会改变模型拓扑。

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
