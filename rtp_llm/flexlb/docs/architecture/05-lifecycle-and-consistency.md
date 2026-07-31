# Lifecycle and Consistency

FlexLB 的优雅上下线不靠 Spring 生命周期或 JVM 信号，而由**同机 sidecar 通过本机 HTTP hook
端点驱动**；高可用由 ZooKeeper LeaderSelector 主选举 + slave 请求转发实现。

主要代码：`flexlb-sync/src/main/java/org/flexlb/service/grace/`、`consistency/`，
`flexlb-api/.../AppStateHookServer.java`、`HealthCheckServer.java`。

## 生命周期 Hook

### 接口（flexlb-common `listener/`）

| 接口 | 方法 | 备注 |
|---|---|---|
| `AppOnlineHooker` | `afterStartUp()` + `priority()` | `priority()` 已声明但**无调用方**——实际执行顺序硬编码在编排服务里 |
| `AppShutDownHooker` | `beforeShutdown()` | 无优先级方法 |
| `ApplicationWarmupState` | `isWarmupFinished()` | 供健康检查与 gRPC 客户端预热判断 |

失败语义：各 Hook 内部自行 catch（`LbConsistencyHooker` 上线 catch Exception、下线 catch
Throwable；`ActiveRequestShutdownHooker` catch InterruptedException），实践中单个 Hook 失败
不会中断链；若真有未捕获异常，`AppStateHookServer` 返回 HTTP 500。

### 编排与触发

**触发点是三个仅限本机调用的 HTTP 端点**（`AppStateHookServer`，非 loopback/本机地址一律 403）：

| 端点 | 行为 |
|---|---|
| `GET /hook/process_ok` | `ApplicationReadyEvent` 后返回 200，否则 503 |
| `GET /hook/after_start` | 同步执行 `GracefulOnlineService.online()`（故意阻塞事件循环），上报 `online_complete` |
| `GET /hook/pre_stop` | 在 boundedElastic 上执行 `GracefulShutdownService.offline()`；活跃请求排干成功返回 200，否则 503 |

**上线顺序**（`GracefulOnlineService.online()`，`test` profile 下整体跳过）：
1. `LbConsistencyHooker.afterStartUp()`——`LBStatusConsistencyService.start()` → ZK
   LeaderSelector 启动（`zk_node_online`）；
2. `QueryWarmerHooker.afterStartUp()`——**固定 sleep 10 秒**等依赖就绪（并有 10s 兜底
   Timer 强制置位），完成后 `warmupFinished=true`（`warmer_complete`）。名字里的
   "warm" 目前不预热任何查询路径。

**下线顺序**（`GracefulShutdownService.offline()`）：
1. `HealthCheckHooker`——置静态 volatile `isShutDownSignalReceived=true`，`/health` 立即
   开始返回 404，摘除流量（`health_check_offline`）；
2. `LbConsistencyHooker`——ZK 下线/让主（`zk_node_offline`）；
3. `ActiveRequestShutdownHooker`——排干在途请求：每 500ms 轮询
   `ActiveRequestCounter.getCount()`，需要**连续 5s 静默**（quietPeriodMs，期间任何活跃请求
   重置窗口）才算成功；硬超时 300s（`shutdown_timeout`）。计数来源：每个
   `/rtp_llm/schedule` 请求通过 `Mono.using(activeRequestCounter::acquire, ...,
   RequestToken::close)` 包裹（token 幂等关闭）。

### 健康检查

`GET /health`（`HealthCheckServer`）：`isShutDownSignalReceived` → 404 "shutdown received"；
warmup 未完成 → 404 "warm not finish"；否则 200 "success"。

### 指标

`GracefulLifecycleReporter`：gauge `graceful.lifecycle.event`，tag `type`
（`process_ok`/`zk_node_online`/`warmer_complete`/`online_complete`/`health_check_offline`/
`zk_node_offline`/`shutdown_complete`/`shutdown_timeout`）+ `duration_ms`，值为时间戳。

## 主选举与一致性

### 配置

env `FLEXLB_SYNC_CONSISTENCY_CONFIG`（JSON → `LBConsistencyConfig`）：`needConsistency`
（默认 false，缺省时一切成为 no-op、`isMaster()` 恒 false）、`masterElectType`（仅
`ZOOKEEPER`）、`zookeeperConfig{zkHost, zkTimeoutMs}`。另需 env `HIPPO_ROLE`（做选举路径），
端口取 `-Dserver.port`（默认 7001，假定所有副本同端口）。

### ZookeeperMasterElectService

- Curator recipe：**LeaderSelector**（非 LeaderLatch），namespace `whale-master`，路径
  `/master_lb_leader/<HIPPO_ROLE>`，`setId(本机IP)`，`autoRequeue()`，重试
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
