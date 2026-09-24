# Lifecycle and Consistency

FlexLB 的上线由同机 HTTP hook 驱动；下线由 HTTP hook 或 Spring 容器关闭事件驱动。
高可用由 ZooKeeper LeaderSelector 主选举与 follower 请求转发实现。

主要代码：`flexlb-api/src/main/java/org/flexlb/service/grace/ApplicationLifecycle.java`、
`flexlb-common/.../listener/ApplicationWarmupState.java`，`flexlb-api/.../AppStateHookServer.java`、
`HealthCheckServer.java`、`FlexlbGrpcServer.java`，以及 `flexlb-sync/.../consistency/`。

## 生命周期 Hook

### 预热状态

`flexlb-common` 中的 `ApplicationWarmupState` 是独立 Spring Bean，使用 volatile 字段
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

| 端点 | 行为 |
|---|---|
| `GET /hook/process_ok` | `ApplicationReadyEvent` 后返回 200，否则 503 |
| `GET /hook/after_start` | 同步执行 `ApplicationLifecycle.online()`，上报 `online_complete` |
| `GET /hook/pre_stop` | 在 boundedElastic 上执行 `ApplicationLifecycle.offline()`，排空完成后返回 200 |

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

### "只有 master 路由"的实际语义

靠 **slave 转发而非拒绝**，只有两处检查 `isMaster()`：

1. `HttpLoadBalanceServer.processScheduledRequest`：启用一致性且非 master →
   `forwardRequestToMaster()` 把原始请求代理到 `http://master:port/rtp_llm/schedule`；
   **master 为空/不可达/超时时降级为本地路由**（`fallbackToLocalRouting`，上报
   `MASTER_NULL`/`TIMEOUT`/`CONNECT_FAILED`）。所有响应都携带 `realMasterHost` 供客户端
   感知真正的 master。
2. `FlexlbControlServer`：cache-match failover 操作非 master 时转发给 master（master 不可用
   返回 503）。

因此该保证是 best-effort：网络分区或 master 缺位时 slave 会自行路由（可用性优先）。
