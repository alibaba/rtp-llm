# Routing and Balancing

路由实现的入口是 DefaultRouter。它不再通过可配置的 Router / LoadBalancer 工厂注册策略；
角色选择器是构造函数注入的三个明确实现：

| 角色 | 选择器 | 说明 |
|---|---|---|
| PREFILL、PDFUSION | CostBasedPrefillStrategy | cache 感知的 TTFT 投影与候选选择 |
| DECODE | CostBasedDecodeStrategy | Decode 容量过滤与 KV/load 加权随机选择 |
| VIT | RandomStrategy | 物理组健康、group 过滤后的随机选择 |

## 角色集与 group

DefaultRouter 在构造时从 ModelMetaConfig.requiredRoles() 取得请求所需角色。该列表来自
MODEL_SERVICE_CONFIG.role_endpoints，按 PDFUSION → DECODE → PREFILL → VIT 排序；不会因
某个角色当前没有 worker 而静默跳过。

路由先尝试 router.groupSelector 为请求解析目标 group。若规则返回 group，所有角色均在该
group 内选址；若没有规则，首个成功角色的 group 约束后续角色。任何角色无可用候选时，
本次选中的 SelectedRole 会关闭 generation pin，且不会留下已提交的 endpoint 账本。

WorkerDirectory 在所有选择器之前执行 physical-group health 门控：共享同一 frontend 的每个
预期 logical engine 必须已经发布为存活 endpoint。随后选择器仍按各逻辑 engine 自己的状态、
cache 与账本投影计算候选。

## Prefill / PDFusion 选择

CostBasedPrefillStrategy 的一次选择按如下步骤进行：

1. 从 WorkerDirectory.prefillRoutingSnapshot() 取得非拥有的候选 generation 快照，并查询
   CacheAwareService.findMatchingEngines()；匹配结果会记录在 BalanceContext。
2. 对每个候选读取 PrefillEndpoint 的 route projection。投影结合 endpoint 已拥有工作、
   请求未命中 token、cache 命中和 PrefillTimePredictor，形成预测 TTFT / drain time。
3. 过滤不适合的候选。候选可因 pending 或 drain outlier、未建模的 engine 工作、以及 cache
   affinity 的 outstanding-uncached-token guard 被排除或降级。
4. 依据 router.roles.prefill.candidateChoice 选择：BEST_ONLY、在相对/最小容忍窗口内
   随机选择，或在最短 TTFT pool 内按 least-recently-used 选择。
5. 若配置 cacheAffinity，只有 cache 命中率达到最小阈值，且相对最短 TTFT 的额外代价不超过
   maxExtraTtftMs 时，才优先 cache leader。
6. 最终按地址重新取得精确 GenerationPin。快照过期、generation 更替或物理组变为不健康时，
   本轮选择失败并交由调用方重新规划。

这条路径将 cache 命中换算为 token；KVCM 的 P2P 命中折扣由
router.roles.prefill.cacheAffinity.p2pHitDiscount 控制。选择器不在此阶段向引擎发送请求。

## Decode 与 VIT 选择

CostBasedDecodeStrategy 从 EndpointRegistry 的 Decode routing view 捕获整批候选，然后：

- 依据 router.roles.decode.availability 的 KV 使用率、可选的 maxEngineRequests 和 endpoint
  账本过滤候选；QUEUE 是否在选址阶段检查 transient Decode 容量取决于
  defersDecodeCapacityUntilDispatch() 与 preemption 配置。
- 排除 engine load 或 KV 已明显偏离群体的 outlier；对其余候选按 decayPerToken 与
  loadDecayPerRequest 形成稳定的加权随机分布。
- 在选中 routing view 后重新取得同一 Decode generation 的 pin。精确 reservation 只在
  DIRECT 提交或 QUEUE admission 提交时创建，选择快照本身不改变容量。

RandomStrategy 仅服务 VIT，随机循环扫描已发布且健康的 endpoint，按 group 过滤后返回一个
带 pin 的无状态 SelectedRole。

## DIRECT 与 QUEUE 的提交边界

RouteService 为每个 BalanceContext 绑定当前 FlexlbConfig 快照，然后按 scheduler.type 调用
不同入口。

### DIRECT

DefaultRouter.routeDirect() 选齐全部所需角色后，在一个本地提交事务中转移它们的 generation pin：

- Prefill/PDFusion 通过 PrefillEndpoint.registerDirectRequest() 取得精确活动请求所有权；
- Decode 通过 DecodeEndpoint.reservePinned() 建立 shadow reservation；
- VIT 只保留 endpoint generation 所有权。

任何提交叶子失败时，已取得的 registration/reservation 按逆序回滚并关闭 pin；成功后构造
Response 并提交所有权。因此 DIRECT 没有 WorkerBatcher 队列，也不支持 BATCH dispatcher。

### QUEUE

DefaultRouter.routeForQueue() 只返回 QueueRouteAdmission：其中包含已 pin 的角色选择、成功响应
和将来提交所需的精确 endpoint 信息。GlobalQueueCoordinator 在其有序决策点将该 admission
提交给 endpoint；若 endpoint 状态或容量已变化，admission 会关闭并重规划。

路由响应中的地址始终是物理 frontend 地址。ServerStatus 内部保留 selected engine index；
多 engine 的 protobuf 响应才设置 engine_index（含显式的 0），单 engine 保持字段未设置。

## 选择与资源的分工

选择器的职责是产生可验证的候选与 generation pin；EndpointRegistry、PrefillEndpoint、
DecodeEndpoint 和 RequestRegistry 才拥有资源、交付和终态。不要在策略中加入临时
WorkerStatus 记账或回滚逻辑：该状态无法覆盖队列、delivery ACK、取消和 generation 退役。
详见 [02-queue-scheduling](02-queue-scheduling.md) 与
[03-resource-management](03-resource-management.md)。
