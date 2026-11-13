# Lifecycle and Consistency

FlexLB 的上线由同机 HTTP hook 驱动；下线由 HTTP hook 或 Spring 容器关闭事件驱动。
高可用由 ZooKeeper LeaderSelector 主选举与 follower 请求转发实现。

主要代码：`flexlb-api/src/main/java/org/flexlb/api/ApplicationLifecycle.java`、
`flexlb-util/src/main/java/org/flexlb/util/ApplicationWarmupState.java`，`flexlb-api/src/main/java/org/flexlb/api/server/AppStateHookServer.java`、
`HealthCheckServer.java`、`api/server/FlexlbGrpcServer.java`，以及 `flexlb-master/.../master/`。

## 生命周期 Hook

### 预热状态

`flexlb-util` 中的 `ApplicationWarmupState` 是独立 Spring Bean，使用 volatile 字段
共享预热状态。`ApplicationLifecycle` 通过 `setWarmupFinished()` 更新状态，
KVCM 客户端通过 `isWarmupFinished()` 读取状态。该状态类不依赖 gRPC 服务、调度器
或生命周期编排器。

`ApplicationLifecycle` 在上线开始时将状态置为 false，等待初始 Worker 同步后置为 true；
健康检查读取同一状态，并同时检查下线标志。`KvcmGrpcClient` 注入
`ApplicationWarmupState`，在预热完成前不累计心跳和查询失败次数，避免提前触发容灾切换。
生命周期编排器依赖 `FlexlbGrpcServer` 完成请求排空；KVCM 客户端只依赖独立状态 Bean，
不反向依赖生命周期编排器。

### 编排与触发

以下 HTTP 端点由 `AppStateHookServer` 提供，仅允许 loopback 或本机地址访问：

| 端点                    | 行为                                                                          |
| ----------------------- | ----------------------------------------------------------------------------- |
| `GET /hook/process_ok`  | `ApplicationReadyEvent` 后返回 200，否则 503                                  |
| `GET /hook/after_start` | 同步执行 `ApplicationLifecycle.online()`，上报 `online_complete`              |
| `GET /hook/pre_stop`    | 在 boundedElastic 上执行 `ApplicationLifecycle.offline()`，排空完成后返回 200 |

**上线顺序**（`test` profile 下跳过）：

1. 重置下线标志和预热状态，调用 `LBStatusConsistencyService.start()` 启动主选举。
2. 等待 3 秒供初始 Worker 同步，然后标记预热完成；等待被中断时恢复线程中断标志，
   预热完成标志仍在 finally 中置位。

**下线顺序**：

1. 置下线标志，使健康检查返回不可用。
2. 调用 `LBStatusConsistencyService.offline()` 下线并让主。
3. 调用 `FlexlbGrpcServer.drain()`，等待没有新 Schedule 请求的静默期；期间仍处理迟到请求。
   静默期由 `grpcServer.shutdownQuietPeriodMs` 控制，默认 5 秒。
4. 关闭 gRPC 新请求入口，等待已接受的 RPC 完成，再允许 Spring 销毁服务资源。
   强制终止期限由平台管理。

`ApplicationLifecycle` 以最高优先级处理所属 Spring 容器的 `ContextClosedEvent`，
同步执行下线流程；子容器的关闭事件不触发服务下线。成功完成后的重复下线调用直接返回。

`ApplicationLifecycle` 在入口排空后调用 `AbstractRequestScheduler.shutdown(deadlineNanos)`，
等待截止时间由调用方传入，调度器不制定宽限期或计算 deadline。
不持有投递执行器、请求任务池、Endpoint 或调度计时器。`SchedulerConfiguration` 组装
调度器并启动调度；EndpointRegistry 等关闭依赖在构造时传入，不依赖事后注入的 shutdown 配置。
生产与测试共用其停机顺序。

调度器通过 `ExpirationTimer.close()` 停止并等待请求超时回调，不持有底层计时器线程池。
`taskTimer` 和 `deadlineTaskTimer` 由 Spring 创建，在调度停机事件处理完成后的 bean
销毁阶段关闭；手工创建计时器的测试调用方负责相同的资源释放。

调度资源关闭时，先停止接收新请求，等待已接受请求响应至调用方指定的截止时间；随后关闭请求注册，
停止并等待 scheduler 和抢占任务，取消剩余请求。DIRECT 等待已提交的选路任务，QUEUE
等待调度线程、planner 和剩余计划处理完成。接着等待投递及其准备资源释放回调、请求处理
任务结束，再关闭 Endpoint。Endpoint 关闭产生的后续清理任务在关闭请求执行器前再次排空。
这些等待复用各组件的执行状态，调度基类不另维护选路操作计数。

请求任务池在改变接纳状态之前拒绝自身任务调用 close，避免任务等待自己完成。
本地清理和远端 stop 使用不同排空条件；提交者尚未完成交接时也保留工作登记。
执行器关闭期间仅允许正在执行的准确请求续接，防止丢失已接纳工作或重新开放无关请求。
响应失败不重新建立发布登记，也不承担计时器安装恢复；deadline 在响应构造前通过 Timer.update 同步。

停止接收后，尚未注册的新请求返回 `SCHEDULER_UNAVAILABLE`（8518）。Frontend 在响应
不含入队标记、路由结果或生命周期记录时，以原请求重试不同的备用 FlexLB 节点。
已接收请求的取消不使用此错误码；备用节点也不可用时保留拒绝结果，不无限重试。

### 健康检查

`GET /health` 通过 `ApplicationLifecycle.isHealthy()` 判断状态：
预热完成且未收到下线信号时返回 200，否则返回 404。

### 指标

`GracefulLifecycleReporter`：gauge `graceful.lifecycle.event`，tag `type`
（`process_ok`/`zk_node_online`/`warmer_complete`/`online_complete`/`health_check_offline`/
`zk_node_offline`/`shutdown_complete`/`shutdown_timeout`）+ `duration_ms`，值为时间戳。

## 主选举与一致性

### 配置

一致性行为已收拢到 `FLEXLB_CONFIG.consistency` tagged union。默认
`{"type":"NONE"}`，相关 start/offline/destroy 都是 no-op、`isMaster()` 恒 false；启用时使用：

```json
{
  "consistency": {
    "type": "ZOOKEEPER",
    "connectString": "zk-1:2181,zk-2:2181",
    "sessionTimeoutMs": 30000,
    "connectionTimeoutMs": 30000,
    "masterRefreshIntervalMs": 5000
  }
}
```

一致性组件在 Bean 初始化时取得配置，因此 Nacos 可以保存和替换这部分字段，但当前进程
是否启用一致性及 ZooKeeper 客户端参数在重启后生效。选举路径和主节点变更通知使用
`DeploymentIdentity`：Spectrum 三元组齐全时为 `spectrum:<workspace>:<application>:<deployment>`；
Spectrum 三元组不完整时使用完整的 `BIZ_NAME:DEPLOYMENT_NAME:ZONE_NAME` 对应值的冒号拼接。
两组三元组均不完整时启动失败。端口取
Spring `server.port`，再回退 JVM `-Dserver.port` 和默认 7001（假定所有副本同端口）。

### ZookeeperMasterElectService

- Curator recipe：**LeaderSelector**（非 LeaderLatch），namespace `whale-master`，路径
  `/master_lb_leader/<deploymentId>`，`setId(本机IP)`，`autoRequeue()`，重试
  `ExponentialBackoffRetry(1000, 3)`。
- `takeLeadership()`：置 `isMaster=true`，**主动 HTTP 通知所有非 leader 参与者**
  `POST http://<ip>:<port>/rtp_llm/notify_master`（1s 超时）；然后阻塞在 CountDownLatch 上
  保持领导权。
- `stateChanged()`：`SUSPENDED`/`LOST` → 抛 `CancelLeadershipException` 放弃领导权
  （LOST 同时清空 master 缓存）。
- master 缓存：每 5s `updateLatestMaster()` 从 `leaderSelector.getLeader().getId()` 刷新
  `cachedMasterHostIp`；每 1s 上报 master 节点指标。
- **优雅让主**：`offline()` 置 `markOffline`、关闭 autoRejoin；若自己是 master，释放 latch
  后（多节点时）**每 1s 轮询直到 leader 变成别的 IP 才返回**——pre_stop 会等领导权实际转移。

### LBStatusConsistencyService（Spring 门面，实现 MasterElectService）

- `handleMasterChange(req)`：`/rtp_llm/notify_master` 的接收端——校验 `roleId` 匹配后
  `refreshMasterHost(true)` 强刷缓存。
- `getMasterHostIpPort()`：master IP + 本机 serverPort。
- `syncLBStatusFromMaster`（每 500ms 调度）与 `dumpLBStatus()` 目前是 **TODO 空实现**
  （`/rtp_llm/schedule_snapshot` 恒返回成功占位）。

### Master 转发与本地降级

`FlexlbServiceImpl` 在启用一致性且本节点非 Master 时通过 gRPC 转发调度请求。
Master 缺位、明确返回未接管请求的 `NOT_MASTER`，或转发发生网络 I/O 故障时，
本节点使用已有的路由与调度流程执行本地决策，PV 日志记录 `LOCAL_FALLBACK`。

网络故障通过异常因果链识别：gRPC `UNAVAILABLE` 或 `UNKNOWN` 携带
`IOException`（包括连接拒绝、连接重置、连接关闭、DNS 解析失败）或
`UnresolvedAddressException` 时允许本地决策。只有状态码而没有网络异常证据的失败、
业务拒绝、RPC 超时和取消不触发本地降级。已过期或取消的请求不再进入本地调度。

本地降级优先保证可用性，不依赖 ZooKeeper 选主完成。连接断开不能证明原 Master
未接收请求，因此该模式不提供跨 Master 的严格单次调度保证；本地调用仍沿用原请求 ID
和截止时间。`FlexlbControlServer` 的 cache-match failover 操作在非 Master 时转发，
Master 不可用时返回 503。
