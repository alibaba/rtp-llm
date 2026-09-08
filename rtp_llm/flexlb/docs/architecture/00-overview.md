# Overview

FlexLB 是 RTP-LLM 模型推理的负载均衡器。它在 Java 21 / Spring Boot 2.7 WebFlux
进程中提供 FlexLB 自己的 gRPC 调度服务、worker 状态同步、cache 感知选址、队列调度与
可选的 ZooKeeper 主选举。

本目录描述当前代码的稳态设计，而不是部署方案或演进计划。代码结构或运行时所有权发生
变化时，应同步更新本目录。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-routing-and-balancing](01-routing-and-balancing.md) | 角色选址、候选策略、DIRECT 和 QUEUE 的提交边界 |
| [02-queue-scheduling](02-queue-scheduling.md) | 全局队列、endpoint batcher、交付与请求生命周期 |
| [03-resource-management](03-resource-management.md) | worker generation、endpoint 账本与容量事件 |
| [04-worker-sync-and-cache](04-worker-sync-and-cache.md) | worker 状态提交、cache matching 与 block hash |
| [05-lifecycle-and-consistency](05-lifecycle-and-consistency.md) | hook 生命周期、主选举和 gRPC follower 转发 |
| [06-configuration-and-observability](06-configuration-and-observability.md) | 配置边界、配置来源、端口与观测入口 |

## 模块划分

| 模块 | 职责 |
|---|---|
| `flexlb-api` | WebFlux 控制与 hook 端点；`FlexlbGrpcServer`、`FlexlbServiceImpl` 和 follower gRPC forwarder |
| `flexlb-common` | 配置契约、请求/worker 数据模型、服务发现拓扑、指标抽象和公共工具 |
| `flexlb-grpc` | 引擎 worker/cache 状态客户端、KVCM 客户端、channel 和 name resolver |
| `flexlb-sync` | `DefaultRouter`、`RequestScheduler`、endpoint 账本、worker 同步、主选举、生命周期与调度观测 |
| `flexlb-cache` | block hash、LOCAL_SYNC 索引、KVCM/Local Standby 匹配与 cache 遥测 |

`flexlb-sync` 的核心包如下：

```
balance/
├── scheduler/    RequestScheduler / GlobalQueueCoordinator / WorkerBatcher /
│                 RequestRegistry / DefaultRouter
├── endpoint/     EndpointRegistry / PrefillEndpoint / DecodeEndpoint /
│                 generation lifecycle
├── delivery/     batch 与 route-decision 的交付边界
├── strategy/     CostBasedPrefillStrategy / CostBasedDecodeStrategy / RandomStrategy
└── eviction/     priority preemption 与引擎取消协调
sync/             MasterEngineSynchronizer / runners / WorkerDirectory
consistency/      LBStatusConsistencyService / ZookeeperMasterElectService
service/          RouteService、grace、monitor、grpc 与 optimizer
```

## 请求主链路

```
gRPC FlexlbService.Schedule（HTTP server.port + 2）
    ↓ 解析请求、计入 ActiveRequestCounter、校验 cache identity
    ↓ 若启用一致性且本机是 follower：单跳 gRPC 转发到 master
    ↓ 本地路由时准备 block cache keys，并由 RouteService 绑定 FlexlbConfig 快照
    ├─ DIRECT: DefaultRouter 选址并在同一提交事务中获取 endpoint 所有权
    └─ QUEUE: RequestScheduler 注册 RequestSlot 后写入 GlobalQueueCoordinator
            ↓ 决策线程从全局有序队列规划，DefaultRouter 为完整角色集生成选址
            ↓ 在精确 Prefill/Decode generation 上提交预留
            ↓ Prefill WorkerBatcher 按 SINGLE 或 FIXED_WINDOW 组织交付
            ↓ DeliveryStrategy: EnqueueBatch，或返回 route decision
    ↓ RequestRegistry 处理 ACK、接受确认、取消、deadline 和终态，完成原始 future
```

`MODEL_SERVICE_CONFIG` 中的拓扑决定必须路由的角色；`ModelMetaConfig` 将已配置角色按
`PDFUSION → DECODE → PREFILL → VIT` 的固定顺序保存。运行时某个 worker map 是否为空不会
改变这份请求角色集。

## 核心不变量

- **逻辑身份贯穿内部状态**：一个物理 frontend 的每个 engine 都是
  `ip:httpPort@engineIndex`；N=1 仍为 `@0`。对外 `ServerStatus` 使用物理地址，只有多
  engine 响应显式携带 `engine_index`。
- **发现、状态与路由 generation 分离**：`WorkerDirectory` 持有已发现的
  `WorkerStatus` generation；`EndpointRegistry` 只公开已提交有效状态的 endpoint generation。
  路由必须取得 `GenerationPin`，同地址替换或退役不能让旧选择获得新资源。
- **QUEUE 只有一个生命周期所有者**：`RequestRegistry` 拥有 request id 去重、全局准入、
  deadline、delivery claim、取消与终态。全局队列负责顺序和选址；每个 `WorkerBatcher`
  只负责已选定 Prefill endpoint 的分组及交付。
- **资源以精确 endpoint 账本提交**：Prefill 的活动请求与 Decode 的 shadow reservation/
  engine fence 都绑定 request generation；失败、取消和退役经相应 reducer 释放，不能由
  选择阶段的临时快照直接释放。
- **物理组健康是逻辑 worker 的共同门槛**：共享 frontend 的所有期望 engine sibling
  均已发布且健康时，该物理组中的逻辑 worker 才可路由。
- **cache 匹配源由配置决定**：`LOCAL_SYNC` 使用 worker cache 状态更新的本地索引；
  `KVCM` 使用外部查询并维护 Local Standby 作为故障转移源。两种主索引不会同时写入。
- **一致性优先单一所有者**：follower 仅在尚未得到 master 地址时可本地路由；已向 master
  发起的调用发生超时或失败时结果属于不确定交付，不再本地重试，必要时发送取消协调。
