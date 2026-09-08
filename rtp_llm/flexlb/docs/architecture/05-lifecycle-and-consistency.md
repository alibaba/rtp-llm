# Lifecycle and Consistency

FlexLB 的上线和下线由本机 sidecar 调用 HTTP hook 驱动；高可用是可选的 ZooKeeper
LeaderSelector 选举，并通过 FlexLB 自己的 gRPC 服务把 follower 请求交给 master。

## 生命周期 hook

AppStateHookServer 只接受 loopback 或与本机地址相同的调用方，其他远端地址返回 403。

| 端点 | 行为 |
|---|---|
| GET /hook/process_ok | ApplicationReadyEvent 后返回 200；此前返回 503 |
| GET /hook/after_start | 同步执行 ApplicationLifecycle.online()，完成后才返回 200 |
| GET /hook/pre_stop | 在 boundedElastic 执行 ApplicationLifecycle.offline()；排干成功返回 200，否则 503 或 500 |

ApplicationLifecycle 是固定编排器，不再遍历 AppOnlineHooker/AppShutDownHooker 列表：

1. online：test profile 整体跳过；其他 profile 启动 LBStatusConsistencyService，随后等待
   固定 3 秒初始 worker 同步并设置 warmUpFinished。
2. health：GET /health 只有 warmUpFinished 且尚未收到 shutdown 时返回 200；其他情况返回 404。
3. offline：先设置 shutdownReceived 并立即让 health 失败，再执行一致性 offline；然后每
   500ms 观察 ActiveRequestCounter，要求连续 5 秒没有活跃请求。硬超时为 300 秒。

ActiveRequestCounter 的 token 在每个 gRPC Schedule 调用完成、取消或出错时幂等关闭。它衡量的是
FlexLB 接收中的 gRPC 请求，不是 endpoint 队列深度或引擎运行请求数。GracefulLifecycleReporter
记录 process_ok、zk_node_online/offline、warmer_complete、online_complete、health_check_offline、
shutdown_complete 和 shutdown_timeout 事件。

## 一致性配置与主选举

FLEXLB_CONFIG.consistency 是 tagged union：

- NONE（默认）：LBStatusConsistencyService 的 start/offline 为 no-op，isMaster() 返回 false。
- ZOOKEEPER：ZookeeperConsistencyConfig 保存连接、session/connection timeout 和
  masterRefreshIntervalMs；该对象在 Bean 创建时读取，切换类型或连接参数需要重启。当前
  ZookeeperMasterElectService 的实际缓存刷新任务固定为 5 秒。

ZookeeperMasterElectService 使用 Curator LeaderSelector，namespace 为 whale-master，路径为
/master_lb_leader/{deploymentId}，selector id 是本机 IP，retry policy 是
ExponentialBackoffRetry(1000, 3)，并启用 autoRequeue。

获得领导权后服务将 isMaster 置为 true，异步 HTTP 通知其他 participant 的
/rtp_llm/notify_master，然后阻塞在 latch 上保持领导权。SUSPENDED 或 LOST 会抛
CancelLeadershipException；LOST 同时清空缓存 master。缓存 leader 每 5 秒刷新，master 节点
指标每 2 秒上报。

offline 会禁止 rejoin 并释放 leader latch。多节点时最多等待约 30 秒确认本机不再是 leader；
单节点或查询 participant 失败时不等待转移。GET /rtp_llm/schedule_snapshot 返回当前的领导权
诊断快照（是否启用、是否 master、本机与缓存 master），不是调度队列的状态复制。

## follower gRPC 转发

Schedule、GetRequestState 和 Cancel 的权威生命周期在 master。启用一致性且本机不是 master 时：

1. FlexlbServiceImpl 通过 FlexlbGrpcForwarder 读取 cached master，并在 gRPC 端口
   server.port + 2 上转发调用。
2. 请求携带 forward_hop，最大只允许一次转发；self-target 或 hop limit 会拒绝再次转发。
3. 没有 master 地址且尚未尝试 RPC 时，Schedule、状态查询和取消可本地处理。
4. 一旦已选择 master 并尝试 Schedule/Cancel RPC，超时或连接失败都是不确定交付：
   不在 follower 本地执行相同 reducer。Schedule 会尽力向 master 发送取消协调，以免 master
   已提交请求后调用方已离开。

因此一致性是“有可达 master 时单一调度所有者”的保证；master 缓存缺失时优先可用性。本地
fallback 只发生在没有选中 master 的早期边界，不能把 RPC 失败解释为安全的本地重试。

Cache match failover 的 HTTP 控制操作也在 follower 转发给 master；没有 master 或转发失败返回
503，而不是在 follower 改变 local active source。
