# Routing and Balancing

路由与负载均衡是 flexlb-sync 的核心。`Router` 定义路由契约，`LoadBalancer` 定义单一角色内的
worker 选择契约，两者组合完成多角色多阶段路由。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/`、`balance/strategy/`。

## Router / DefaultRouter

当前 `Schedule` 可传 `schedule_roles`（Encoder、Prefill、Decode、PDFusion）只选择指定角色；
未传时选择模型配置的全部角色。显式角色去重后按 `ModelMetaConfig.requiredRoles()` 的顺序决策；
未配置角色以及非法枚举值返回 `INVALID_REQUEST`。现有 VIT 不属于该新增请求枚举，
其选点仍使用 `RandomStrategy`。
EPD 部署当前按客户端分别请求 Encoder 与 Generation 阶段；混合角色请求不在已验证链路内。

Encoder 使用独立的 `EncoderStrategy`：候选必须是已发布且存活的 Encoder endpoint。
请求携带可选 `encoder_cache_hit_len` 时，先比较各节点在途的未命中编码工作量代理值，
并列时比较 `running_query_len + waiting_query_len + 本地待观察请求数`；未携带时仅比较并发数。
Encoder 不按可用 KV cache 筛选或决胜。Encoder 单独请求按 `scheduler.type` 走
DIRECT 或独立的 QUEUE，FlexLB 返回 `ENCODER` 端点，不调用 EnqueueBatch。
QUEUE 还会过滤已达到 `dispatcher.maxInflightPerEncoderWorker` 的节点，并在容量释放后重试；
DIRECT 没有候选时返回 `NO_ENCODER_WORKER` (8408)。
本地待观察请求在 endpoint 的并发 map 中逐请求更新，选点读取允许短暂滞后；并发请求可能依据
同一份负载快照选中同一 worker。Prefill/Decode 与 PDFusion 的既有策略保持原样。

`Router` 接口只有一个方法：`Response route(BalanceContext balanceContext)`。

`DefaultRouter`（`balance/scheduler/DefaultRouter.java`）由 Prefill、Decode、VIT 和
Encoder 选择器组成，按请求有效角色顺序逐个选择，并在失败时关闭已捕获的 endpoint pin。

### route() 流程

1. **校验**：缺少请求或显式角色未配置 → `INVALID_REQUEST`。
2. **角色列表**：`ModelMetaConfig.requiredRoles()` 给出配置顺序；请求带
   `schedule_roles` 时取其交集，未传时使用全部配置角色。
3. **逐角色选择**：Prefill/PDFusion 调用 `CostBasedPrefillStrategy`，Decode 调用
   `DecodeSelector`，VIT 调用 `RandomStrategy`，Encoder 调用 `EncoderStrategy`。
   `group` 初始为 null；每个角色选中后 `group = serverStatus.getGroup()`——**首个成功角色的
   worker group 约束后续所有角色的候选集**（group 亲和链，如 DECODE 选中的 group 决定
   PREFILL 只能在同 group 中选）。任一角色失败立即关闭之前的选择并返回错误。
4. **响应**：全部成功 → success 响应（携带 `List<ServerStatus>`；物理
   `server_ip/http_port/grpc_port` 保持不变，N>1 时每个成功项带逻辑 `engine_index`，
   N=1 时省略该字段）；没有可用角色时按该角色的错误类型返回。

`RoleType.getErrorType()` 映射：PREFILL→`NO_PREFILL_WORKER`(8402)、DECODE→`NO_DECODE_WORKER`
(8403)、PDFUSION→`NO_PDFUSION_WORKER`(8404)、VIT→`NO_VIT_WORKER`(8405)、
ENCODER→`NO_ENCODER_WORKER`(8408)，均可重试。

### 角色与策略 / 资源指标映射

映射不在 `RoleType` 上，而在 `FlexlbConfig`（`flexlb-common/.../config/FlexlbConfig.java`）：

| RoleType           | 策略（可配）                | 默认策略                                   | 资源指标（硬编码）                                           |
| ------------------ | --------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| PDFUSION / PREFILL | `loadBalanceStrategy`       | `SHORTEST_TTFT`                            | `WAIT_TIME`                                                  |
| DECODE             | `decodeLoadBalanceStrategy` | `WEIGHTED_CACHE`                           | `REMAINING_KV_CACHE`                                         |
| VIT                | `vitLoadBalanceStrategy`    | `RANDOM`                                   | `WAIT_TIME`                                                  |
| ENCODER            | 独立 `EncoderStrategy`      | 提供命中长度时最少编码工作量，否则最少并发 | 在途预测／实际输入长度、`running + waiting + 本地待观察请求` |

### gRPC Schedule 响应

`FlexlbService.Schedule` 的 `FlexlbServerStatusPB` 使用 `optional int32 engine_index = 9`。
字段号 5、7 保持 reserved。N=1 不设置字段；N>1 设置所选 index，包含显式 0，调用方应通过
字段 presence 区分这两种情况。`FlexlbServiceImpl` 从内部 `ServerStatus` 转换时保留该语义；
master forwarding 透传 protobuf 响应。物理地址字段与既有 service path 不变，旧客户端可忽略
新增字段。此字段不改变 `EnqueueBatch` 的派发协议。

### Prefill Chunk 装箱

`router.roles.prefill.chunkBatchingEnabled` 默认 `false`。启用后，支持 Chunk 的
Prefill engine 通过 WorkerStatus 上报原始输入 token 额度、请求槽位、分配截止时间，
以及独立的 step token 预算。输入长度小于 step 预算的请求进入 Chunk 选址；等于或
大于预算的请求走 NORMAL。PDFusion、Decode 和 Encoder 不携带 Prefill 预约。

Chunk 的身份由逻辑 engine、P 启动 `engine_epoch` 和 `chunk_id` 共同确定。
Master 在可容纳请求且未到分配截止时间的候选中选择剩余 token 额度最小者，
同等条件优先截止更早者。在选中 worker 的 generation 锁内再次校验并原子扣减
tokens 与一个槽位，防止并发超发和旧端点复用。重复上报同一 Chunk 保留已消费
余额，失效或已清除的预约不能通过重报重新开放。

成功响应在 `FlexlbServerStatusPB.prefill_chunk=11` 返回预约身份和截止时间，
物理地址及 `engine_index` 仍沿用已有路由字段。Frontend 拿到预约后直接向 P
发送请求，不调用 `EnqueueBatch`。无可用 Chunk 使用显式 fallback 状态 8600，
由 SDK 沿用原请求 deadline 走 NORMAL；这不改变其他业务拒绝的终止语义。

Master 的余额不持久化。主备切换后重新接收 P 的上报，可能对同一 Chunk 再次
授予额度；P 必须用有界积压承接，并由 scheduler 的实际 step 硬预算限制计算。
Master 只使用上报的 `dispatch_deadline_ms`，不推算 P 的冻结时间，也不再次
扣除安全余量。取消或发送结果不明时不能直接退回已经发布的预约额度。

请求截止时间必须晚于 Chunk 的分配截止时间才能预约。由于冻结时间不在协议中，
该判断不保证极短 TTFT 请求一定能等到 P 冻结；如果 P 侧原始 deadline 先到，
请求会取消而不会延长预算。上线时需实测等待窗口与业务 TTFT 的配合。

## 回滚机制

`DefaultRouter.rollBackRoutingFailure()`：对 `RoutingResult` 中已成功的每个 `ServerStatus`，
以内部 `serverIp:httpPort@routingEngineIndex` 调用对应策略的 `rollBack(ipPort, requestId)`。
`routingEngineIndex` 不序列化且始终存在，因此 N=1 对外省略 `engine_index` 时仍精确回滚
`@0`；N>1 的 wire `engine_index` 只负责把选择结果传给共享 frontend。

回滚的实体是**选中时的本地记账**——`WorkerStatus.removeLocalTask(requestId)` 逆转
`putLocalTask()`：从 `runningQueueTime` 扣回估算 prefill 时间（下限 0）、把
`inputLength - prefixLength` 的 KV token 从 used 还给 free、从 `localTaskMap` 移除。
`lastSelectedTime` 不回滚。

各策略实现：`RandomStrategy` 为 no-op（它 select 时不记账）；`ShortestTTFTStrategy`（及其
子类 CacheAffinityFirst）回滚时**硬编码查 PREFILL** 的 worker map；`WeightedCacheLoadBalancer`
**硬编码查 DECODE**——回滚正确性耦合于默认的策略↔角色配对，改配对时须注意。

## 四种策略

均实现 `LoadBalancer.select(ctx, roleType, group)`，注册名见 `LoadBalanceStrategyEnum`
（RANDOM / SHORTEST_TTFT / CACHE_AFFINITY_FIRST / WEIGHTED_CACHE）。

所有策略只从 `EndpointRegistry` 已发布的 logical worker 读取候选；单个 worker 的状态失败或
退役只会移除该 `ip:httpPort@index`，不会连带过滤同一 frontend 的 sibling。资源、队列、cache
命中、outstanding tokens 和 score 都读取该 logical worker 自己的状态。快照选中的候选仍须捕获
其 exact generation，避免在选择与提交之间使用已替换或已退役的 endpoint。

### RandomStrategy

公共物理健康门控后随机起点环形扫描，取第一个 `isAlive()` 的逻辑 worker。不看资源水位、不做 cache 匹配、
**不 `putLocalTask()`**（因此 rollBack 为空实现）。

### ShortestTTFTStrategy（PREFILL/PDFUSION 默认）

1. **候选过滤**：`isAlive()` 且 `ResourceMeasure.isResourceAvailable()`（PREFILL 用等待队列
   长度 + 滞回，见 [03-resource-management](03-resource-management.md)）。
2. **cache 匹配**：`CacheAwareService.findMatchingEngines()`，用逻辑
   `ip:httpPort@index` exact-match cache ownership（见
   [04-worker-sync-and-cache](04-worker-sync-and-cache.md)）。
3. **打分**：每个 worker 计算
   `hitCacheTokens = blockSize × 匹配块数`，
   `prefillTime = seqLen − hitCacheTokens`，
   `ttft = prefillTime + runningQueueTime`（本地维护的队列时间估计）。
4. **选择**：
   - 按 ttft 升序排（并列时 `lastSelectedTime` 早者优先）；
   - 仅 PREFILL/PDFUSION 且配置了正的 `outstandingUncachedTokensThreshold` 时，过滤
     `outstandingUncachedTokens + max(0, seqLen − hitCacheTokens)` 超阈值的 worker；若仍有
     eligible worker，后续选择只在其中进行；若全部超阈值，恢复完整候选集并重跑后续
     Top 候选/相似/cache 偏好流程（仍可能选中超阈值的 cache-preferred worker），决策 reason
     记录 `SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK`；
   - 取 Top 候选：≤3 个全取，否则 `max(2, ⌈数量×0.3⌉)`；
   - 相似阈值 `max(minTTFT × shortestTtftSimilarityThresholdRatio(0.2), 标准差 × 0.5)`，
     筛出与最短 ttft 相近的 worker；
   - 相似集合内做 **cache 偏好**：cache 命中高于最短 ttft worker 时改选 cache leader；
   - **CAS 抢占**：按偏好序对 `lastSelectedTime` 做 `compareAndSet(快照值, now)`，第一个
     成功者当选——防止并发调度线程用同一快照选中同一 worker。
5. **提交**：`putLocalTask()` 记账（预扣队列时间与 KV token）、上报 cache 命中指标、
   `ctx.recordCacheMatch()`。

### CacheAffinityFirstStrategy（继承 ShortestTTFT，只重写选择步骤）

- **终态 prefill 工作守卫**：每个 worker 的 `ttft` 已包含当前请求的未命中 Prefill token 和已有队列
  未命中 token。计算 cache leader 相对最短 ttft worker 的额外工作；只有该差异不大于
  `cacheAffinityFirstMaxExtraWorkTokens`，且缓存命中更多时，才选 leader（`CACHE_LEADER`），否则选最短
  ttft（`SHORTEST_TTFT`）。在选 cache leader 前也复用同一 outstanding-threshold guard；低于
  `cacheAffinityFirstMinHitRate` 的 cache leader 不参与 cache affinity
  （`SHORTEST_TTFT_LOW_CACHE_HIT`）；超阈值的 cache leader 直接回退到最短 eligible worker
  （`SHORTEST_TTFT_OUTSTANDING_GUARD`）。与 ShortestTTFT 不同，全部 worker 超阈值时**不做
  cache 取舍，直接按最短 ttft 选择**（同样记录 `SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK`）。
- CAS 抢占失败时按剩余 eligible worker 继续选择；决策 reason 为
  `CACHE_AFFINITY_FALLBACK` 或 `SHORTEST_TTFT_FALLBACK`。决策快照（debug 级）写入
  `BalanceContext.shortestTtftDecisionByRole`。

### CostBasedPrefillStrategy

按候选的资源、排队和预测执行时间做成本筛选与打分。它与 ShortestTTFT 一样通过统一的
`CacheAwareService.findMatchingEngines(CacheMatchQuery)` 取得 cache 匹配，因此 PREFILL 和
PDFUSION 的 cost-based 路径也遵循 [04-worker-sync-and-cache](04-worker-sync-and-cache.md) 的
源选择：`LOCAL_SYNC`、`KVCM` 或 KVCM 不可用时的 `LOCAL_STANDBY`。本地命中按全量折算；KVCM
返回的远端增量命中（`global - local`，负值按 0 处理）按 `cacheAffinity.remoteDiscount`
折算（默认 `0.2`，取值范围 `[0, 1]`），再进入原有的成本估算。
这只改变 cache 命中的输入来源，不改变 cost-based 的候选过滤、成本公式和提交/回滚语义。

### WeightedCacheLoadBalancer（DECODE 默认）

1. 候选过滤：`isAlive()` + `isResourceAvailable()`（DECODE 用 KV 使用率 + 滞回）。
2. **加权随机**（权重基于 KV cache 使用量，与 cache 匹配无关）：
   `weight = exp(−weightedCacheDecayFactor(0.001) × (cacheUsed − 平均值))`——用得越少权重越高；
   总权重非正或全相等时退化为均匀随机；轮盘赌选择。
3. **cache 匹配只做记账**：选中后再查 `findMatchingEngines()`，得到
   `prefixLength = blockSize × 匹配块数`，用于 TaskInfo 的 KV 预扣精度（`inputLength −
prefixLength`），不影响选谁。
4. `putLocalTask()` 记账。

## BalanceContext 路由相关字段

`flexlb-common/.../dao/BalanceContext.java`：输入 `config`/`request`（`seqLen`、`blockSize`、
`blockCacheKeys`、Local Standby keys 等）；输出 `response`；per-role 记录
`cacheMatchSelectionByRole`（角色→选中 ip + 命中 token 数，`recordCacheMatch()` 同时累计
查询耗时/次数与 `cacheMatchSource`）和 `shortestTtftDecisionByRole`（debug 决策快照）。
队列字段见 [02-queue-scheduling](02-queue-scheduling.md)。

## 并发要点

- worker map 是 `ConcurrentHashMap`，`WorkerStatus` 计数全部 Atomic；选择路径不加锁
  （`WorkerStatus.lock` 存在但选择路径不使用）。
- 选择的互斥靠 `lastSelectedTime` CAS；记账/回滚靠 Atomic 加减。
- worker 是否可选由 `PrefillResourceMeasure` / `DecodeResourceMeasure` 基于活性、
  待处理请求、KV 使用率和可选并发上限判定（见
  [03-resource-management](03-resource-management.md)）。
