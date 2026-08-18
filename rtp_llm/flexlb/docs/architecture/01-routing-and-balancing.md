# Routing and Balancing

路由与负载均衡是 flexlb-sync 的核心。`Router` 定义路由契约，`LoadBalancer` 定义单一角色内的
worker 选择契约，两者组合完成多角色多阶段路由。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/`、`balance/strategy/`。

## Router / DefaultRouter

`Router` 接口只有一个方法：`Response route(BalanceContext balanceContext)`。

`DefaultRouter`（`balance/scheduler/DefaultRouter.java`）：

- 类上标注 `@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy",
  "cacheAffinityFirstStrategy"})`——4 个策略 bean 都在**各自构造函数里**调用
  `LoadBalanceStrategyFactory.register()` 自注册，`@DependsOn` 保证注册先于 DefaultRouter 构造。
- 构造时按 `FlexlbConfig.getStrategyForRoleType(roleType)` 为每个 `RoleType` 解析一个
  `LoadBalancer`，存入 `EnumMap<RoleType, LoadBalancer>`；策略未注册则启动即抛异常。

### route() 流程

1. **校验**：`request == null` → `INVALID_REQUEST`；全局 worker 状态未初始化 →
   `NO_AVAILABLE_WORKER`。
2. **角色列表**：读静态单例 `EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS` 的
   `getRoleTypeList()`——按**固定顺序 PDFUSION → DECODE → PREFILL → VIT**，只加入
   worker map 非空的角色。没有请求级角色过滤：所有非空角色都会被路由。
   - 因此 PD 分离部署下实际顺序是 **DECODE 先于 PREFILL**；融合部署只有 PDFUSION；
     有 VIT worker 时 VIT 追加在最后。
3. **`routeByRoleType()`**：逐角色调用 `loadBalancer.select(ctx, roleType, group)`。
   `group` 初始为 null；每个角色选中后 `group = serverStatus.getGroup()`——**首个成功角色的
   worker group 约束后续所有角色的候选集**（group 亲和链，如 DECODE 选中的 group 决定
   PREFILL 只能在同 group 中选）。任一角色失败立即返回
   `RoutingResult.failure(已成功列表, 失败角色, 错误信息)`。
4. **响应**：全部成功 → success 响应（携带 `List<ServerStatus>`）；失败 → 先
   `rollBackRoutingFailure()` 再构造错误响应，错误码取
   `failedRoleType.getErrorType().getErrorCode()`。

`RoleType.getErrorType()` 映射：PREFILL→`NO_PREFILL_WORKER`(8402)、DECODE→`NO_DECODE_WORKER`
(8403)、PDFUSION→`NO_PDFUSION_WORKER`(8404)、VIT→`NO_VIT_WORKER`(8405)，全部 `canRetry=true`。

### 角色与策略 / 资源指标映射

映射不在 `RoleType` 上，而在 `FlexlbConfig`（`flexlb-common/.../config/FlexlbConfig.java`）：

| RoleType | 策略（可配） | 默认策略 | 资源指标（硬编码） |
|---|---|---|---|
| PDFUSION / PREFILL | `loadBalanceStrategy` | `SHORTEST_TTFT` | `WAIT_TIME` |
| DECODE | `decodeLoadBalanceStrategy` | `WEIGHTED_CACHE` | `REMAINING_KV_CACHE` |
| VIT | `vitLoadBalanceStrategy` | `RANDOM` | `WAIT_TIME` |

## 回滚机制

`DefaultRouter.rollBackRoutingFailure()`：对 `RoutingResult` 中已成功的每个 `ServerStatus`，
以 `serverIp:httpPort` 调用对应策略的 `rollBack(ipPort, requestId)`。

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

### RandomStrategy

随机起点环形扫描，取第一个 `isAlive()` 的 worker。不看资源水位、不做 cache 匹配、
**不 `putLocalTask()`**（因此 rollBack 为空实现）。

### ShortestTTFTStrategy（PREFILL/PDFUSION 默认）

1. **候选过滤**：`isAlive()` 且 `ResourceMeasure.isResourceAvailable()`（PREFILL 用等待队列
   长度 + 滞回，见 [03-resource-management](03-resource-management.md)）。
2. **cache 匹配**：`CacheAwareService.findMatchingEngines()`（见
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
- 路由并发度由 `DynamicWorkerManager` 的许可信号量在调度层门控（见
  [03-resource-management](03-resource-management.md)）。
