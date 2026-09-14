# Spec: GlobalQueue 联合 Prefill 决策

## 1. Goal / Restated Understanding

为 FlexLB 增加可配置的全局 FIXED_WINDOW：在选择 worker 之前收集一批请求，综合该批请求在各 worker 上的缓存收益、预测 TTFT 和资源竞争，统一产生落点。保留当前 shortest TTFT / cache affinity 配置和 AutoTPM 优先级语义。

典型问题：256k–1M 长请求 A、B 先后到达，A 低缓存命中、B 高缓存命中，两者独立选择同一 worker。A 产生大量 KV 写入，可能淘汰 B 依赖的缓存。只预测新增计算负载不足以表达该干扰；仅先发送 B 也不保证引擎先执行 B。

本轮先形成可审阅、可实施、可恢复的详细 Feature Spec；用户确认后在同步路由接口上以 TDD 落地基础 BEST_ONLY，不提交或推送。按 SDD-RIPER-ONE Light 的 checkpoint、验证和回写方式维护。

## 2. Done Contract

- 本轮完成：已确认约束、当前源码事实、配置/API 设计、算法、AutoTPM 不变量、失败语义和验收矩阵落盘；源码链接与示例经过静态检查。
- 功能完成：后续实现通过兼容性、优先级、容量/生命周期、联合分配测试以及全仓质量检查；运行时性能收益需独立实测。
- 仅有算法示例或 mock 成功不代表实际 TTFT/缓存收益得到证明；实现授权与详细参数批准仍待后续确认。

## 3. 决策状态与范围

### 3.1 用户已确认的硬约束

| ID | 约束 |
|---|---|
| C01 | 联合规划允许为整批收益改变某条请求原本最优的 worker。 |
| C02 | shortest TTFT 与 cache affinity 是可选偏好，复用现有 prefill config。 |
| C03 | 全局 FIXED_WINDOW 收集发生在 worker 选择之前。 |
| C04 | 开启全局窗口时文档要求 `scheduler.decision.type = SINGLE`；不增加代码 assert 或交叉配置强制拒绝，不自动改配置。 |
| C05 | 不新增引擎协议；FlexLB 规划、提交、发送顺序均不能被解释为引擎执行顺序。 |
| C06 | 抽取 `PrefillStrategy` 公共基类，保留独立的 `CostBasedPrefillStrategy`，新增独立的 `CostBasedBatchedPrefillStrategy`。 |
| C07 | 保持现有 AutoTPM 优先级逻辑，禁止用低优先级收益换取高优先级损失。 |
| C08 | 实现已获后续确认；保持当前 dirty/untracked 并行修改，不修改其他项目。 |

### 3.2 方案提议（不是已经实现的事实）

- 使用机会损失优先的贪心分配，加有界单请求迁移/双请求交换。
- 复用真实生命周期提交，规划过程只改变虚拟状态，不做全批原子事务。
- 缓存干扰作为有证据边界的启发式；不虚构精确淘汰概率、命中率或引擎顺序。
- 新增配置字段、接口草图、预算常量和策略聚合语义需在实现前完成详细评审。

### 3.3 Out of scope

引擎侧 pin/cache lease、引擎调度顺序协议、跨模型全局优化、新 AutoTPM 排序/抢占策略、修改 KVCM 协议、改变 DIRECT 语义、整批分布式事务、历史分支整体迁移、生产部署。

## 4. 当前事实与代码索引

初次核对时间：2026-09-11；实现基线于 2026-09-13 重新确认为同步接口 HEAD `6b93c7bf7e15c2597ea91566443a2f5d6707bea9`。事实来自当前工作区（包含未提交修改），不是仅来自 HEAD。

以下链接相对于本 spec；它们指向现有代码，不代表新增设计已落地。

| 入口 | 已核对事实 |
|---|---|
| [SchedulerConfig](../../flexlb-common/src/main/java/org/flexlb/config/SchedulerConfig.java) / [DecisionPolicyConfig](../../flexlb-common/src/main/java/org/flexlb/config/DecisionPolicyConfig.java) | 现有 decision 支持 SINGLE/FIXED_WINDOW，含 maxRequests、maxCollectionWaitMs、maxPredictedExecutionMs；默认 FIXED_WINDOW。 |
| [GlobalQueueCoordinator](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/GlobalQueueCoordinator.java) | nextPlanningFrontier 取有界 eligible 前缀；plan(entry) 逐条 routeForQueue；每次 commit 前检查更高优先级；精确阻塞、REPLAN、priority rescue 已存在。 |
| [OrderedRequestQueue](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/OrderedRequestQueue.java) | FIFO 按入队顺序；PRIORITY 按优先级降序、同级按入队顺序。 |
| [PriorityNormalizer](../../flexlb-common/src/main/java/org/flexlb/util/PriorityNormalizer.java) | AutoTPM 优先级 1–100，数值越大越高；保留现有归一化和默认值，不重新解释原始 header/proto。 |
| [DefaultRouter](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DefaultRouter.java) | 验证、路由组和多角色选择后构造 QueueRouteAdmission。 |
| [QueueRouteAdmission](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/QueueRouteAdmission.java) | 拥有选定 endpoint generation pins；精确容量由发布事务获取；close 释放未转移所有权。 |
| [CostBasedPrefillStrategy](../../flexlb-sync/src/main/java/org/flexlb/balance/strategy/CostBasedPrefillStrategy.java) / [RoutingConfig](../../flexlb-common/src/main/java/org/flexlb/config/RoutingConfig.java) | cacheAffinity 非空启用缓存偏好；maxExtraTtftMs 默认 0；候选默认 RANDOM_WITHIN_TOLERANCE；BEST_ONLY 为严格最短预测 TTFT。 |
| [RouteProjection](../../flexlb-sync/src/main/java/org/flexlb/balance/projection/RouteProjection.java) | 已有冻结预测器和 delivery-aware 投影契约，应扩展复用而非复制计算公式。 |
| [WorkerBatcher](../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler/WorkerBatcher.java) | worker 内决策组及交付所有权继续保留。 |
| [调度详细文档](../priority-scheduler-delivery-modes.md) | 描述高优先级提交前重检、精确 endpoint 抢占、全局 outstanding 满时准入替换与异步顺序边界。 |

`../architecture/02-queue-scheduling.md` 仍含旧类名/旧配置表述，不以其覆盖当前源码。历史分支的 cacheAffinityFirstMaxExtraWorkTokens 与当前 maxExtraTtftMs 不可数值直译。本 spec 不要求恢复旧配置。

## 5. 配置设计

新增 `SchedulerConfig.globalDecision: GlobalDecisionConfig`，默认 `type=SINGLE`，保持现有逐请求路径。

```json
{
  "scheduler": {
    "type": "QUEUE",
    "globalDecision": {
      "type": "FIXED_WINDOW",
      "maxRequests": 8,
      "maxCollectionWaitMs": 5,
      "maxPlanEvaluations": 4096
    },
    "decision": { "type": "SINGLE" }
  },
  "router": {
    "roles": {
      "prefill": {
        "candidateChoice": { "type": "BEST_ONLY" },
        "cacheAffinity": {
          "maxExtraTtftMs": 50,
          "minPrefixHitPercent": 5
        }
      }
    }
  }
}
```

示例是局部配置片段；8、5 ms、4096、50 ms 为提议值，不是生产调优结论。建议新增字段默认
maxRequests=8、maxCollectionWaitMs=5、maxPlanEvaluations=4096；全局功能默认关闭。

| 字段 | 语义 |
|---|---|
| globalDecision.type | SINGLE 走现有路径；FIXED_WINDOW 使用联合策略；只用于 QUEUE。 |
| maxRequests | 每次捕获的最大请求数，满额提前关闭窗口。 |
| maxCollectionWaitMs | 收集等待上限；0 表示不主动等待，仍可一起规划当前已在队列中的请求。 |
| maxPlanEvaluations | 单个 global batch 的 completion search 与每个局部优化阶段的最大评估次数；默认 4096。 |
| scheduler.decision | 保持原 worker 内决策语义；全局 FIXED_WINDOW 配合 SINGLE 仅为文档契约。 |
| prefill.* | 原 estimator、candidateChoice、cacheAffinity 配置为唯一偏好来源；不加 batch 专属 affinity 开关/阈值。 |

新配置执行常规类型/范围校验（maxRequests >= 1，等待 >= 0，maxPlanEvaluations >= 1）；不增加 C04
禁止的交叉 assert。使用独立配置类型，不把 worker 的 maxPredictedExecutionMs 误用成全局规划时间预算。
未配置新字段时所有旧配置解析行为应保留；不升级 wire schema。

配置捕获沿用现有运行时约定：不可把不同配置语义的请求当成同一个评分域；具体热更新生效边界要与现有 scheduler runtime 核对。不得为了此功能顺带重构配置热更新。

## 6. 窗口与队列语义

1. 由一个模型的 GlobalQueueCoordinator 维护收集期限，不跨模型、路由隔离域混用候选。
2. 以最早 eligible 请求首次进入收集状态的时刻建立绝对截止时间；新请求不延长窗口。取消最老请求不能不断刷新其他成员的期限。
3. 达到 maxRequests、收集到期或请求绝对过期处理所需的唤醒时刻，触发处理。请求过期仍走原生命周期取消，不靠“抢在超时前规划”绕过过期门。
4. 窗口关闭时重新从 OrderedRequestQueue 按原顺序捕获 eligible 前缀。低优先级不能先填满不可替换的固定槽位，把后来高优先级排除在外。
5. PRIORITY 模式若更高优先级在收集期间到达，建议立即关闭当前收集等待并重新捕获，避免它等待低优先级的剩余窗口。已开始计算时依靠提交前重检作废未提交低优先级部分。
6. 纯收集等待不占每条请求的 AdmissionMutation，以免长时间阻止原 outstanding-priority admission 转移许可；规划/提交期的 mutation 保持原所有权规则并及时释放。
7. 容量不足请求仍停放在原 blocker 域，由相关事件唤醒；已付过收集等待的重规划不重复等待窗口。阻塞请求不计入可凑批数，不定时忙轮询。
8. 每个模型只提交一份当前有效联合计划；候选 IO 可并发，不能让多个联合规划器独立消费同一份虚拟容量。

## 7. 架构与接口草图

```text
GlobalQueueCoordinator: 收集 / eligible / deadline / 优先级 / commit
    → DefaultRouter: 校验、角色约束，并通过统一 PrefillStrategy 契约选择单请求或批量入口
    → PrefillStrategy: 共用候选发现、硬过滤、cache/TTFT 评估、结果 materialize
        CostBasedPrefillStrategy: 单请求选择
        CostBasedBatchedPrefillStrategy: 继承单请求选择并提供全局联合规划
          → 不可变 PlanningRequest + 批内虚拟状态
          → 机会损失贪心 + 有界迁移/交换
    → DefaultRouter: 指定 Prefill 落点的其他角色完成与 generation pin
    → QueueRouteAdmission: 每条精确发布事务
    → WorkerBatcher SINGLE: 现有 dispatcher 交付
```

接口为设计草图，最终类型名可以贴合现有代码，但所有权不可变化：

```java
abstract class PrefillStrategy {
    public final PlacementResult select(BalanceContext context, RoleType role, String group);
    public List<PlacementResult> selectBatch(List<BatchRequest> requests);
    // package-private shared candidate discovery, filtering, cache/TTFT evaluation and materialization
}

final class CostBasedPrefillStrategy extends PrefillStrategy {
}

final class CostBasedBatchedPrefillStrategy extends PrefillStrategy {
    // The only Spring PrefillStrategy bean, used for both call shapes
    @Override
    List<PlacementResult> selectBatch(List<BatchRequest> requests);
    List<Integer> plan(List<PlanningRequest> requests); // internal joint BEST_ONLY plan
}
```

- 基类持有同步候选查询所需依赖和公平游标，统一候选发现、RouteProjection、原单请求策略、cacheAffinity 门控、决策遥测、generation pin 与 materialize。单请求 concrete strategy 只暴露稳定的 `select()` API；批量 concrete strategy 复用 package-private 候选快照/materialize 能力，不把 `PrefillCandidateSet` 暴露到包外。两个策略共享精确平局公平游标，使 B=1 在模式切换前后延续单请求语义。
- `PlanningRequest` 是批量算法使用的不可变候选/预测数据。批量 concrete strategy 一次消费整个列表，不循环调用单请求 `select()` 伪装联合算法。
- `DefaultRouter` 只注入一个 `PrefillStrategy`，不感知 concrete strategy，也不判断 batch/non-batch 配置。`GlobalQueueCoordinator` 根据已经形成的 planning frontier 调用 `routeForQueue()` 或 `routeBatchForQueue()`，Router 再分别调用同一策略的 `select()` 或 `selectBatch()`。
- 这里选择单个 `abstract class PrefillStrategy`，不再叠加 interface 与 `AbstractCostBasedPrefillStrategy`。共享实现依赖 `WorkerDirectory`、`CacheAwareService`、`EngineHealthReporter` 和实例公平游标，用 interface default/private method 会迫使实现类暴露依赖 getter 或引入额外 context 参数，不能减少耦合。
- Spring 只创建 `CostBasedBatchedPrefillStrategy` 这一份生产 Bean；它直接持有公共依赖和公平游标，同时支持单请求与联合规划。`CostBasedPrefillStrategy` 保留为独立对比实现，但不自动注册。`DefaultRouter` 的构造器只出现 `PrefillStrategy`。`globalDecision` 来自每次请求绑定的运行期配置快照，因此不能用 Spring 启动期 `@ConditionalOnProperty` 按当时配置只实例化一个行为受限的 Bean，否则动态配置切换后 Bean 能力与请求配置不一致。
- 规划输出只包含候选索引；真实容量仍由后续逐请求发布事务获取。生成的 `SelectedRole` 只持有 generation pin，不提前预留 delivery capacity。
- 批量策略的 `BatchCandidates` 为每条请求保存请求身份、endpoint discovery、候选投影、缓存匹配、拒绝原因和 blocker；`PlanningRequest` 保存联合算法需要的不可变 worker 成本与策略参数。
- `List<Integer>` 规划结果与原始队列顺序一一对应；内部机会损失顺序只作为计算过程，materialize 和提交仍按原队列顺序执行。
- 虚拟 worker 状态为一次调用私有；不修改真实 ServerStatus、endpoint 计数或共享缓存索引。不复用 ThreadLocal cursor 到异步线程或跨请求共享。
- 保持 PREFILL/PDFUSION 与 Decode 完整路由合法性。Decode 等剩余角色无法完成时反馈可解释的 blocked/replan 结果，不静默重新选择 Prefill。
- 原有未知执行时长降级（UNMODELED_PENDING_LRU）必须保留等价路径，不把未知值当 0 或捏造 TTFT。该路径不能承诺 affinity 的毫秒上限。

## 8. 联合规划算法

### 8.1 输入和目标

设 R 为窗口请求，W 为各请求合法 worker 集合，x 为完整落点方案。T_i(x) 为基于现有队列、虚拟新增工作、真实 SINGLE 交付语义预测的 TTFT；H_iw 为观测到的可复用 token。所有成本评估以原提交顺序投影，不能把内部贪心顺序当引擎先后顺序。

同一优先级层内 shortest TTFT 主目标为最小化 sum(T_i)。PRIORITY 从高到低冻结层级结果；低层不得修改高层落点、容量预算，也不能通过其新增工作使高层建模结果变差后仍声称无影响。若模型不能区分低层干扰，应采取保守方案并标为风险，不保证引擎层严格隔离。

### 8.2 基线、机会损失和搜索

1. 复用当前策略生成可行基线，作为退化比较和规划预算耗尽时的保留方案。
2. 建立 R×W 候选矩阵；硬过滤沿用现有规则。单一可选 worker 的请求优先；没有合法落点的请求保留明确 blocker，不虚构分配。
3. 对当前优先级层，计算最佳与次佳合法候选的成本差（机会损失）。差越大越先在虚拟计划内分配；平局使用原队列序号保证可复现。
4. 每次分配后更新虚拟负载、WorkerStatus 可用 KV 与交付 request 容量，重算受影响候选。不同请求只能读取各自真实 cache-match，不把前一个请求将要产生的缓存当作已存在；KV 缺失按现有 WorkerBatcher 的 unlimited 兼容语义处理。
5. 完成初始方案后尝试单请求迁移/双请求交换；按完整受影响 worker 的投影评估，不能只算被移动请求自己。只接受硬约束满足且当前目标改善的方案。
6. 贪心结果不完整时，按候选数优先执行有界完成搜索，目标先最大化可落点请求数，再最小化总 TTFT；随后才做局部改进。工作量/轮数有界；到界返回已找到的最佳容量安全方案，不提交搜索中的半成品状态。

建议首版内部上限为两轮局部改进并限制评估次数；具体常量通过 B=1/8/32、多 worker 的规划耗时测试确定，不新增公共调参面。需要避免 B×W 全量 IO/内存失控；基线候选范围要保留缓存优胜 worker，不能只取 TTFT top-K 而提前丢掉 B 唯一高命中落点。

### 8.3 cache affinity

- 对同一快照、同一优先级层，先得到 shortest-TTFT 批量基线 T_i(base)。
- 启用 affinity 后，在每个受影响请求满足 T_i(x) <= T_i(base)+maxExtraTtftMs 的范围内，优先提高可复用 token 总量；继续遵守 minPrefixHitPercent、P2P 折扣和 outstanding guard 的现有含义。
- 上限相对固定基线，不在每次迁移后重置；后续层不能利用低优先级收益放宽高优先级约束。
- RANDOM_WITHIN_TOLERANCE / LRU 用于符合当前策略约束的近似等价方案选择；精确平局、随机候选集合和 LRU 副作用必须有兼容测试。搜索过程不能更新真实 lastSelectedTime；只对最终选择执行现有时机的记账。
- 此为新批量路径的聚合语义，不要求单条旧路径改变。B=1 的选择、原因和副作用应与原路径等价。

### 8.4 缓存干扰

A、B 同机时 A 可能破坏 B 已观测缓存，属于风险而非确定淘汰。首版不能把 A 的未命中 token 直接从 B 命中量扣除，也不能假定同 batch 新缓存共享、pin 或引擎排序。

建议启发式：在满足 TTFT/affinity/优先级约束的近似等价方案间，优先减少“高命中依赖 + 大量低命中新增 KV”集中于同 worker 的组合。信号仅使用已确认可获得的新增 KV 压力与缓存依赖量；公式和归一化待真实字段核对后固定，不能引入无单位混加的评分。

无可信空闲 KV 或淘汰信号时，记录 co-location 风险，采用保守 tie-break，不宣称缓存保护。无替代 worker 时允许可行的同机分配；不靠延迟/饿死 A 人为制造漂亮命中率。

## 9. AutoTPM 不变量

| ID | 必须保留 |
|---|---|
| P01 | 复用 normalized priority，不改变 1–100、默认值和上游来源。 |
| P02 | PRIORITY 降序、同级 sequence；FIFO 不因存在 priority 字段自动变成 PRIORITY。 |
| P03 | 每条提交前 hasHigherPriorityEntry 重检；高优先级到达后废弃未提交的低级后缀，释放 pin/mutation，再捕获。 |
| P04 | 优化算法的访问顺序与提交顺序分离；不能为了 cache hit 跨级提交。 |
| P05 | 精确 endpoint 容量 miss 继续 park；只有版本/相关事件证明可变时重规划，避免失败后不断重选 worker 绕过原规则。 |
| P06 | tryPriorityRescue 复用 EvictionManager 和精确 Prefill/Decode 落点，不新增受害者策略，不二次调用 selector。 |
| P07 | 全局 outstanding 满时原有低优先级 queued permit 转移仍工作；收集窗口不提前占住不可转移 mutation。 |
| P08 | 继续保留同 endpoint blocker 顺序与独立 endpoint 可前进的规则；不要把高优先级阻塞误改成全模型停止。 |

先前访谈允许过同级尽力重排，但后续“保持现有 AutoTPM 逻辑”是收敛边界：首版只改变联合落点和内部搜索顺序，不新增对外提交顺序规则。同机缓存保护不能依赖重排获得保证。

## 10. 提交、并发与失败

- 一份联合计划逐条转为 QueueRouteAdmission，复用 request generation、取消、精确容量预留和发布事务。
- 任何一条取消/过期都不影响已完成请求；取消释放虚拟假设后，对依赖该假设的未提交部分重新评估。
- generation/placement stale：关闭未转移 pin/mutation，保留已提交结果，对剩余重新捕获状态；不得重复发布。
- unchanged capacity miss：走现有 blocker/priority rescue；不能把所有 miss 都当 snapshot stale。后续依赖该分配的虚拟状态应重建，不继续使用错误容量。
- 部分成功非全批回滚；已发布副作用不可撤销。最终计划仅是建议，提交检查仍是容量真相。
- 相关事件在 snapshot 与 park 之间发生的竞态继续由 availability sequence/原锁边界防丢唤醒。
- 队列锁内只管理索引和状态，不查询缓存、预测、等待 future 或执行 RPC。
- 无预测/查询失败沿用现有明确降级；编程异常不伪装成容量不足或无限重试。
- 关闭全局功能恢复原路径；不取消已经发布的请求，运行中的旧配置计划遵守现有配置快照/运行时生命周期边界。

## 11. 文件改动计划（后续实现）

| 模块 | 拟改动 |
|---|---|
| flexlb-common config/parser/validator | GlobalDecisionConfig 与兼容解析、基础范围验证。 |
| flexlb-sync scheduler | 全局收集状态、联合规划入口、保留逐条提交与 AutoTPM 重检。 |
| flexlb-sync strategy | PrefillStrategy、公用 cost/evaluation、CostBasedBatchedPrefillStrategy。 |
| flexlb-sync projection | 多虚拟请求投影，按 SINGLE 和现有顺序建模。 |
| flexlb-sync/api tests | 策略、全局队列、生命周期、AutoTPM 集成回归。 |
| docs | 实现后更新架构/调度配置文档；本轮不把未来设计写成稳态事实。 |

不要求创建所有拟名文件：输入/输出小 record 可内聚在规划器附近。公共逻辑抽取须以实际复用为依据，避免一次性框架。

## 12. 验收矩阵

| ID | 场景 | 预期证据 |
|---|---|---|
| T01 | 无 globalDecision / SINGLE | 现有配置与单条行为不变。 |
| T02 | 满批/到期/等待0/孤立一条 | 一次全局收集；0等待可规划多条；计时不被新请求刷新。 |
| T03 | A两机相近、B只在W1高命中 | 在容量允许时 A→W2、B→W1；反转同级输入顺序仍能找到优良分配。 |
| T04 | 多请求争同一容量 | 虚拟预算无重复消费，精确提交不超限。 |
| T05 | affinity 多次迁移 | 每条相对固定基线遵守 maxExtraTtftMs；低命中门控、P2P、guard 保留。 |
| T06 | B=1、未知预测、LRU/随机 | 旧选择语义与副作用兼容；未知不当0。 |
| T07 | 高优先级收集/计算/部分提交时到达 | 未提交低级不越过；已提交不重复撤销/发布。 |
| T08 | outstanding 满、准入替换、抢占 | 复用原 victim/permit/精确落点语义，无 pin/mutation 泄漏。 |
| T09 | 高优先级阻塞不同/相同 endpoint | 保留原 blocker 冲突规则，不扩大为全局 HOL。 |
| T10 | cancel/timeout/generation失效/park竞态 | 单次终态、单次资源释放，无丢唤醒。 |
| T11 | PD/PDFUSION、多 routingGroup/配置 | 合法域隔离、Decode失败不暗中改 Prefill。 |
| T12 | 长低命中A与高命中B | 验证启发式选择；引擎重排模拟不能被误报为保证缓存命中。 |
| T13 | 搜索预算耗尽 | 返回预算内放置数最多、TTFT 最低的方案；无法形成完整方案时显式保留 blocked 结果，规划开销有界。 |

重点复用已有 OutstandingPriorityAdmissionTest、EvictionManagerTryAdmitTest、EvictionPlannerPrefillContractTest、EvictionPlannerDecodeContractTest、TransientCapacityQueueContractTest、CostBasedPrefillSelectionMetricTest 等用例并增加批量覆盖。

质量验证：实施时从 FlexLB 根运行针对性测试，随后 Maven 全仓测试与 `./mvnw spotless:check -Pspotless-check`；记录退出码、Surefire 和 Reactor Summary。并行工作导致的基线失败单独归因，不覆盖他人改动。

性能验证对比原单条、全局FIXED_WINDOW（相同路由偏好/流量/worker/缓存初态），分别测试 shortest TTFT 与 affinity、低负载/高负载、256k–1M 混合缓存、优先级混合。记录端到端 TTFT p50/p95/p99（按优先级）、实际命中 token、吞吐、超时/拒绝、全局收集等待、规划耗时、重规划次数。现阶段不捏造提升阈值；量化性能门槛需根据基线确认。

## 13. 风险与未决细节

1. 已批准并已进入实现：总体架构、公共策略基类、配置复用、AutoTPM不变和基础 BEST_ONLY。
2. 待核对：实际缓存容量/压力字段足以支持哪种干扰 tie-break；无字段时明确降级，不能承诺解决全部淘汰场景。
3. 后续范围：批量 RANDOM/LRU 精确语义和性能验收阈值；首版 BEST_ONLY 搜索预算固定为 4096 次方案评估。
4. 配置热更新以请求绑定的不可变快照为评分域；不同快照形成边界并先关闭旧窗口。非 Prefill 角色及未知预测继续沿用既有路径。
5. AutoTPM“无低级牺牲高级”是规划和准入规则，不保证引擎执行隔离；window 本身增加的等待需按优先级分别测量。
6. 当前实现基于同步 `routeForQueue` 接口；保留工作区已有未跟踪文档和并行修改，不从陈旧 HEAD 覆盖当前工作。

## 14. Checkpoint / Approval / Validation

- 当前理解：详细 spec 已确认，基础 BEST_ONLY 已在同步接口基线上实现。
- 核心目标：将已确认联合规划与 AutoTPM 边界固化为可测试契约。
- 当前进度：单层 `PrefillStrategy` 抽象基类、Router 单一策略依赖、全局窗口、配置、review 修复、测试和架构文档 reverse sync 均已完成。
- 下一步：用同一流量分别测量“单请求决策 + worker FIXED_WINDOW 批量提交”和“全局 FIXED_WINDOW + worker SINGLE”。
- Execution Approval：文档与基础 BEST_ONLY 业务实现 Approved。
- Goal Alignment：实现保持 BEST_ONLY 首版范围、原 AutoTPM 优先级和逐请求精确提交语义；无目标偏离。
- Validation：review 后的聚焦配置/策略/GlobalQueue 测试与全仓 Spotless 已通过。Maven 全仓回归中 common/grpc/cache/sync/api 全部通过，mock-engine 仅 `JavaLoadClientResultCollectionTest.slowHeadFutureDoesNotBlockCompletedTailFutures` 因主机时间跳变触发一次 10 秒超时；该测试随后单独重跑通过。真实引擎性能收益仍需按对比方案实测。
- Human confirmation：C01–C08、BEST_ONLY 首版范围和测试边界均来自本会话明确决策。

## 15. Change Log / Reverse Sync

- 2026-09-11：首次落盘。根据源码补充提交前高优先级重检、精确阻塞/抢占、收集期不持有 mutation 的约束；明确规划顺序不改变提交顺序。
- 2026-09-13：实现基线切换到同步接口 HEAD。批量入口落在独立 `CostBasedBatchedPrefillStrategy`；`PrefillStrategy` 承载共用候选评估、materialize 和统一的单/批调用契约；`DefaultRouter` 仅注入 Spring 主 `PrefillStrategy`，由 coordinator 决定调用形态。
- 2026-09-13：code review 后收窄内部候选类型可见性；补入生产 WorkerStatus KV/request 容量；B=1 复用单请求流程和公平游标；PRIORITY 按层提交后再规划下一层；窗口携带不可变配置快照并在配置边界关闭；补充贪心不完整时的有界完成搜索与失败路径 pin 释放。
- 2026-09-14：`DefaultRouter` 收敛为单一 `PrefillStrategy` 依赖；Spring 仅注册同时支持 `select()` 与 `selectBatch()` 的批量实现，移除 Router 对具体类型和 Bean 名称的感知。
- 2026-09-14：启动期按 `globalDecision.type` 绑定 single 或 batched concrete strategy；两个 decision type
  在运行期更新中冻结并记录 warning，fixed-window 的 B/等待继续热更新。全局 fixed-window 与 worker
  fixed-window 改为互斥配置，移除 batched strategy 对 single strategy 的复制构造器。
- 2026-09-14：completion search 提取为独立 `BatchCompletionSearch`；`globalDecision.maxPlanEvaluations`
  默认 4096，并作为 completion search 与每个局部优化阶段的可热更新评估上限。

## 16. Resume / Handoff

当前状态：基础 BEST_ONLY 与 code review 修复已实现；review 后聚焦测试、全仓 Spotless 和一次完整
Maven 回归已执行，唯一时间敏感失败已单独重跑通过。恢复时先读第3、7、9、13、15节并核对当前 diff。

下一步是用相同流量分别测量单请求决策 + worker FIXED_WINDOW 和全局 FIXED_WINDOW + worker SINGLE。

## 17. Project Sync Candidates

候选：全局收集/worker决策/引擎执行三者边界、联合规划下 AutoTPM 提交约束。

同步状态：已同步到 docs/architecture/02-queue-scheduling.md、01-routing-and-balancing.md、06-configuration-and-observability.md 及 docs/priority-scheduler-delivery-modes.md。不修改 AGENTS 或个人 memory。

## 18. 实现授权与首版范围收敛

用户已授权基于 TDD 实现基础 BEST_ONLY，用于对比“全局批量决策 + worker SINGLE”与“原单请求决策 + worker FIXED_WINDOW 批量提交”。本节覆盖第13/14/16节旧的实现 Pending 状态。

- Execution Approval：Approved；配置、策略、RequestScheduler 与 HTTP/mock 边界已确认。
- 首版批量路径仅考虑 BEST_ONLY；RANDOM_WITHIN_TOLERANCE / LRU 暂不实现，原单条路径仍保持其原有能力。
- shortest TTFT 与可选 cacheAffinity 继续复用原配置；AutoTPM 不变量不变。
- 对比必须记录 globalDecision、decision、dispatcher 三个维度；两组分别控制全局/worker窗口，不能把交付粒度差异归因为算法收益。
- 已确认 seams：配置公开解析入口（新字段、兼容）；PrefillStrategy 包内规划契约（落点和策略约束）；RequestScheduler.submit/cancelRequest/getRequestState（窗口、优先级、取消与容量收敛）；现有 HTTP/mock-engine 性能入口（两组运行配置和结果输出）。
- 只在外部缓存/worker/时钟边界使用 fake，不 mock 新规划器或内部评估器；逐行为 red → green，不一次写完全部测试。
- 当前 TDD 覆盖：配置解析、策略算法、真实 KV 容量、B=1、公平游标、配置边界、全局窗口/优先级/取消、完整路由和 materialize 失败清理均有自动化用例。真实引擎性能、搜索预算在大候选集上的耗时和缓存淘汰效果留给对比压测验证；实现授权无需重复索取。
