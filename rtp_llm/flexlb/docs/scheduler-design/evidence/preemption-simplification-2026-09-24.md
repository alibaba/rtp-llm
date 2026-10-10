# 抢占状态与候选规划简化（2026-09-24）

## 改动与职责

- `DecodePreemptionCoordinator` 删除 `cancelStarted`、`committed`、`acceptedAcknowledgement` 及未使用的 RequestScheduler 引用。回调安装前的线性流程已保证 Cancel 协议准备完成；`closed` 唯一记录协调器清理责任是否结束。
- victim 继续区分 RELEASABLE、OUTBOUND、TRANSFERRED、TERMINAL：发送边界未知不能本地回滚，迟到终态仍由精确 claim 接收。
- ACK 全部结束后只创建一个有超时的终态聚合等待，替代每 victim 单独建立相同期限的定时器与汇总循环。超时只完成聚合 Future，绝不取消独立终态观察。超时完成放在 delayedExecutor 执行，避免把端点结算放在 CompletableFuture 的全局定时线程。
- `DecodeEndpointSnapshot` 用一个不可变 candidates 列表代替按阶段拆分的 reserved/accepted/running 三列表。阶段唯一来源是 DecodeRequestView.phase；已被其他尝试 claim 的请求仍在捕获时排除。
- `EvictionPlanner` 统一严格低优先级、已知优先级、排除 ID 与 KV 条件的过滤；MASTER_LOCAL 与 ENGINE_CANCEL 仍分别规划、比较成本，不混合提交。显式排序保证不依赖捕获时的列表顺序。
- 删除历史多端点 planDecode 包装。生产唯一调用原本只传 List.of(selected)，现在直接接收已经选中的端点快照；不改变路由或候选成本计算。
- `EvictionManager` 复用已有提交遥测方法，删除重复异常隔离包装和不可达的 planner 改换端点检查。

## 代码量

| 生产文件 | 修改前 | 修改后 | 减少 |
| --- | ---: | ---: | ---: |
| DecodePreemptionCoordinator | 657 | 576 | 81 |
| DecodeEndpointSnapshot | 54 | 31 | 23 |
| EvictionPlanner | 355 | 313 | 42 |
| EvictionManager | 568 | 548 | 20 |

sync 生产 Java：**28,759 → 28,593，净减 166 行**。从初始 30,842 行累计减少 2,249 行，距 25,000 行还差 3,593 行。未移动生产代码到其他模块。

## 验证

- 基线归档：本地 `/tmp/flexlb-before-preemption-simplify.tar.gz`；本轮独立差异 `/tmp/flexlb-preemption-simplify.diff`。
- 首轮全量回归出现一个新增断言错误：fixture.reserve 使用 reserveUnqueued，状态应为 ENGINE_MAY_HAVE_SEEN，不能把“reserved”列表等同本地 QUEUED。确认原 fixture 语义后修正测试预期。
- `/tmp/flexlb-preemption-reactor-final.log`：完整 API reactor BUILD SUCCESS，common 225、cache 37、grpc 15、sync 1098；API 175 项中 1 项跳过，其余通过。
- 最后的单端点接口收敛及双 victim 测试两个变体：`/tmp/flexlb-preemption-final-focused.log`，相关 sync 53 项通过，BUILD SUCCESS。
- 双 victim 测试覆盖：两者均发送后一个先终态、另一个未知并超时；发送前已经终态而跳过其 Cancel；超时不释放远端可能持有的 claim，迟到终态不重新提交 incoming。
- 三个独立 subagent 从并发与所有权、设计与优先级规则、测试与行为覆盖审查，无遗留问题。git diff --check 通过。
- 按用户最新要求，本轮局部简化只运行本地测试，未重跑远端性能。此前突发 P99 未达标的结论仍有效，不声称性能验收完成。

## 后续：Decode 阶段成为唯一状态来源

- 删除 DecodeRequestState.queued 和 DecodeRequestView.queued 两份排队布尔字段；排队、发送待确认、Engine 确认直接由 phase 表达。snapshot 只构造一次请求视图，queued() 由不可变 phase 推导。锁外读取先捕获一次 volatile phase，避免并发清理导致两次读取不一致。
- 删除 PreemptionClaim.owner 及 ClaimOwner 枚举。claim 的请求是否确认由所在 DecodeRequestState.phase 唯一决定；WorkerStatus 缺席或回退到 RECEIVED 仍保留原确认阶段，并继续保留合成 slot/KV。engineLifecycleOwned 和 KV hold 代表独立事实，继续保留。
- 校准中的合成占用统计和普通缺席项清理合成一次遍历。终结时 confirmed 与 shadow 使用互斥分支结算，消除清空 phase 后再次进入 shadow 分支的无效尝试。
- RequestScheduler 将 fenced Cancel 与 worker priority-canceled 的共同收尾合并；保留后者同时结清 Prefill 的语义区别。删除重复 invariant 检查、未使用返回值及由 terminal source 枚举推导的布尔字段。

### 代码量与证据

生产 Java 从 **28,593 → 28,538，净减 55 行**；累计从 30,842 行减少 **2,304 行（7.47%）**，离 25,000 行还差 **3,538 行**。仍是 95 个生产 Java 文件，没有跨模块搬移。

- 基线：`/tmp/flexlb-before-decode-phase.tar.gz`；最终独立 diff：`/tmp/flexlb-decode-phase-final.diff`。后半段 ClaimOwner 独立基线：`/tmp/flexlb-before-claim-owner.tar.gz`。
- 阶段合并定向测试：`/tmp/flexlb-decode-phase-final-focused.log`，193 项通过；删除 ClaimOwner 的首轮定向测试：`/tmp/flexlb-claim-owner-focused.log`，BUILD SUCCESS。
- 最终完整 API reactor：`/tmp/flexlb-claim-owner-reactor.log`，BUILD SUCCESS；common 225、cache 37、grpc 15、sync 1101 项全部通过，API 175 项中 1 项跳过，其余通过。git diff --check 通过。
- 扩展原入队/派发/确认测试，检查视图状态及捕获后不变性。新增两个参数化变体覆盖 shadow claim 后确认、缺席或 RECEIVED 回退、重复报告只持有一次 KV、NOT_SENT 不释放已确认 victim、终态精确结算及迟到状态不复活。
- 三个 subagent 已分别只读 review 两轮生产变更，覆盖并发/身份与账本、设计与优先级语义、测试覆盖；均无阻断发现。
- 按用户要求，本轮只跑本地测试，未运行远端性能。既有突发 P99 未达标结论未改变。

## 后续：合并 victim 选择，删除计划包装

- EvictionPlanner 的槽位与 KV 选择共用排序前缀遍历及 cost/harm/release 累计。槽位按所需数量停止，KV 按 hard 和 expected 两个缺口停止；候选严格低优先级、排除已选 ID、所有权隔离的规则保留。
- 删除 PlanCost 类。精确优先级 harm 和最小 requestId 直接属于不可变 DecodeEvictionProposal；victim 数从不可变列表大小读取。比较顺序仍是 harm、人数、最小 ID、endpointId；标量饱和 cost 只作诊断。
- 删除 EvictionManager.PlannedDecodeEviction。提交直接接收此前选中的 endpoint 与 proposal，需求从原 RouteAdmission 获取；没有再次选路，提交仍重验精确身份和实时容量。
- 生产 Java **28,538 → 28,488，净减 50 行**，文件 **95 → 94**；累计减少 **2,354 行**，距离 25,000 行目标还差 **3,488 行**。

### 验证

- 独立基线 `/tmp/flexlb-before-victim-selection.tar.gz`，最终差异 `/tmp/flexlb-victim-plan.diff`。
- 新旧规划器差分工具仅放在 `/tmp/flexlb-victim-diff`，未加入仓库。固定 seed=20260924，30,000 组输入比较 victim 顺序、case、hard KV、标量成本、精确 harm、tie break 和失败原因；结果全部一致。覆盖零值、Long.MAX_VALUE 附近容量、阶段/优先级、功能开关和 Cancel 支持差异。结果为 slot 322、KV 754、不可行 28,924，**没有随机命中成功的 combined 计划**。日志 `/tmp/flexlb-victim-plan-differential.log`。
- 因上述覆盖限制，新增两个固定 combined 缺口用例，分别走 slot→KV 和 KV→slot，断言选取顺序、不重复、成本/harm、释放量以及最终容量满足。`/tmp/flexlb-victim-plan-final-focused.log`：sync 24 项通过。
- 三个独立 subagent 审查未发现阻断问题；测试 reviewer 提出的 combined 覆盖缺口已补齐。
- `/tmp/flexlb-victim-plan-clean-reactor.log`：完整 API reactor 执行 clean test 后 BUILD SUCCESS，common 225、cache 37、grpc 15、sync 1103 全部通过，API 175 项中 1 项跳过；删除 PlanCost 不依赖残留编译产物。
- reviewer 指出最初 KV→slot 用例即使漏排除已选 ID 也不重复；第一次强化数据反而使 slot-only 已满足两项容量，定向测试失败（`/tmp/flexlb-victim-plan-exclusion-test.log`）。修正为需 3 slot、4000 KV，id1/id5 各 2000 KV，其余三个各 100 KV；两种单侧计划均不足，最终必须选择 [1,5,2]。最后两个参数化用例通过：`/tmp/flexlb-victim-plan-exclusion-final.log`，BUILD SUCCESS。生产逻辑未因测试失败修改。
- 本轮未执行远端性能；既有性能验收缺口保持记录。

## 后续：Prefill 只归并本次发生终结的 batch

- 全量 WorkerStatus 先解析精确 terminal，再为受影响 BatchWork 收集全部存活成员；没有 terminal 的 batch 不创建临时 BatchReduction。原实现对未受影响 batch 的 projectedCompletion 必然返回 null，且不重新预测它们，因此这部分工作可删除。孤儿清理仍遍历所有 batch。
- ACTIVE 的归并保持独立：无 terminal 的 batch 仍更新 Engine phase、剩余时间和 unknown 计数。
- 删除 reconcileEngineStatus 私有转发层及 canonicalMutationStarted 重复标记。StatusReconciliation 在首个账本修改前一次构造，非 null 即表示进入该边界；归并和发布失败共用出口，保留冻结事实并触发 failedReduction。边界前失败仍抛出原 RuntimeException/Error，边界后失败保留在结果中。
- 生产 Java **28,488 → 28,450，净减 38 行**；累计减少 **2,392 行**，仍有 **3,450 行**待收敛到目标。
- 基线 `/tmp/flexlb-before-prefill-reduction.tar.gz`，独立差异 `/tmp/flexlb-prefill-reduction.diff`。
- `/tmp/flexlb-prefill-reduction-final-focused.log`：sync 148 项通过。新增两变体验证完成 batch 与无 terminal 的 RUNNING batch 同轮处理，以及 publication/retirement 回调同时失败时仍保留首因、scheduler facts 和 BatchCompletion。已有部分终结重新预测及预测失败不修改账本的测试继续通过。
- 三个 subagent 从并发与精确身份、设计与阶段语义、测试覆盖审查，无阻断发现。git diff --check 通过。本轮只运行本地测试，未声称远端性能已验收。
- 最终 `/tmp/flexlb-prefill-reduction-reactor.log`：完整 API reactor BUILD SUCCESS；common 225、cache 37、grpc 15、sync 1105 全部通过，API 175 项中 1 项跳过，其余通过。

## 后续：资源统计读取及指标观察接口收拢

- PrefillEndpoint 的三个计数 getter 合为 ownershipStats()，读取现有不可变 Stats；指标和 HTTP 各读取一次。HTTP JSON 键、total/individual/batch 的含义不变。
- BatchSchedulerReporter 删除逐项转发接口，批派发直接报告，完成事件在 reporter 内逐项隔离 RuntimeException。保留 dispatch reason 覆写入口、现有 metric/tag/value 和 endpoint 的后置观察异常隔离。
- sync 生产 Java **28,450 → 28,347，净减 103 行**，仍 94 文件；累计减少 2,495 行，距离 25,000 行还差 3,347 行。API 生产增加 1 行（缓存同一快照），没有搬移实现。
- 基线 `/tmp/flexlb-before-ownership-metrics.tar.gz`；独立差异 `/tmp/flexlb-ownership-metrics.diff`。
- 初次定向编译发现测试缺少 eq 导入，修复后 `/tmp/flexlb-ownership-metrics-focused-final.log` 88 项通过。
- `/tmp/flexlb-ownership-metrics-reactor.log` 完整 API reactor：common 225、cache 37、grpc 15、sync 1109 全部通过；API 175 项中 1 项跳过、1 项失败。失败来自 HTTP mock 未 stub 新 Stats 接口；已补齐快照，并增加三个 JSON 字段值与单次读取断言。
- `/tmp/flexlb-ownership-metrics-final.log`：mock-engine reactor BUILD SUCCESS，所有模块及 mock-engine 71 个测试源文件编译通过；定向 sync 89 项通过。该 reactor 不包含 API；HTTP 修复另行验证。包括 predicted/actual/gap 各项上报故障及正常完成变体。未执行 mock-engine 全量测试。
- 三个 subagent 已完成并发、设计和测试只读 review，无阻断问题；测试 reviewer 建议的 gap 故障变体已补齐。本轮未跑远端性能，既有 P99 验收缺口保持。
- HTTP 最终定向验证 `/tmp/flexlb-ownership-http-final.log`：3 项全部通过，BUILD SUCCESS。

## 后续：Decode 所有权结算使用同一出口

- 合并 removeShadowExactLocked/removeConfirmedExactLocked 为 removeRequestOwnershipLocked，先校验当前对象身份及是否仍有所有权，再按 phase 释放 shadow 的 reserved/queued/dispatch 或 confirmed slot；claim 与终态历史保留独立生命周期。
- 普通 terminal、priority claim terminal、EXPIRED 删除重复 phase 分支及重复发送许可清理；所有原 shadow 入口继续由前置校验限定释放范围。
- TTL 保留判定是外部 LongPredicate，可能重入并确认请求；在判定返回后保留 !request.confirmed() 的复验，避免通用结算误删除刚确认的 owner。新增用例验证这一边界。
- sync 生产 Java **28,347 → 28,298，净减 49 行**；累计从 30,842 减少 **2,544 行**，距 25,000 仍差 **3,298 行**，目标未完成。
- 基线 `/tmp/flexlb-before-decode-settlement.tar.gz`，最终差异 `/tmp/flexlb-decode-settlement.diff`。
- `/tmp/flexlb-decode-settlement-focused.log` 226 项通过；加入重入保护与用例后，`/tmp/flexlb-decode-settlement-final.log` **227 项通过**。覆盖 Decode*、Eviction*、QueuedDecodeWithdrawal、PreemptionRegistration、RequestResourceAccounting。四阶段×terminal/expiry 用例检查旧 token、邻居资源、重复终结、许可失效与各资源计数；现有测试覆盖 claim 转确认、合成 KV、迟到终态及过期。
- 三个 subagent 完成只读 review，无阻断问题；并发 reviewer 已复核最终 TTL phase 复验。git diff --check 通过。按用户要求本轮局部结算简化只运行本地测试，没有重跑远端性能或扩大既有验收结论。

## 后续：定时器注册失败统一回收，删除过时返回对象

- ExpirationTimer.register 以 installed 为唯一安装事实，finally 统一取消未安装的任务；删除 schedule 私有转发层及重复拒绝/安装失败取消分支。成功安装后的 publishAfterInstall、提前触发五状态协议、注册计数与 close 等待顺序保持不变。
- 三种 deadline 的 cancel 重载收敛为一个 DeadlineRegistration 入口，保留 exact.owner 身份验证。DeadlineRegistration 改为包内可见以匹配包内取消接口；具体语义句柄类型仍分别保留。
- 删除只被测试读取、生产回调直接丢弃的 DecisionExpiry。onDecisionVisibilityDeadline 改为 void；测试直接检查 SUSPECTED_LOST、活动请求身份、期限状态及旧句柄调用无副作用。
- 新增早于安装触发的 install/reject/throw 三变体：单线程 ScheduledThreadPoolExecutor barrier 确认任务先运行，再验证成功安装只回调一次，拒绝和异常取消后不可再次消费或发布。
- sync 生产 Java **28,298 → 28,263，净减 35 行**；累计减少 **2,579 行**，距 25,000 行还差 **3,263 行**。没有移动实现或压缩格式。
- 基线 `/tmp/flexlb-before-deadline-cleanup.tar.gz`，最终差异 `/tmp/flexlb-deadline-cleanup.diff`。
- `/tmp/flexlb-deadline-cleanup-final-focused.log`：RequestLifetimeTest 30、EndpointCleanupDeadlockTest 6，共 36 项通过。三个 subagent 从并发、设计、测试进行只读 review，无生产阻断问题；测试 reviewer 建议的取消后即时无需确认断言已恢复。
- 最终 `/tmp/flexlb-deadline-cleanup-reactor.log`：完整 API reactor BUILD SUCCESS，common 225、cache 37、grpc 15、sync 1122 全部通过，API 175 项中 1 项跳过，其余通过；包含最后补回的即时断言。git diff --check 通过。
- 本轮只跑本地回归。既有远端突发 P99 未达标的结论不变；目标保持未完成。

## 后续：准入成员只记录尚未移交的所有权

- 删除 MemberOwnership 三态枚举。所有旧读者只区分 ADMISSION_OWNED 与其他状态，现用 admissionOwned 表示准入过程是否仍负责释放；正常 dispatch 的三种结果都交出该责任，dispatch 抛异常仍保留以便 close 回收。
- 删除 CommittedAdmissionOwner.bound，以非空 handoff 表示完成绑定。Batch 和 Route 的真实提交路径都返回具体 CommittedHandoff，绑定仍只赋引用，不在提交后分配对象或新增可失败验证。
- Member 与 accepted 包装尚未逃逸时的捕获失败直接回滚精确 permit，无需构造第二条 Member 状态清理路径。
- 删除 createCommittedOwner 工厂及两策略 prepare 私有转发，Route 的单调用 predict 包装直接使用既有预测边界。CommittedAdmissionOwner 仍在账本 commit 前完成分配；Route 部分 reservation 转移前缀、Batch 发送阶段保持独立。
- sync 生产 Java **28,263 → 28,217，净减 46 行**；累计减少 **2,625 行**，距 25,000 仍差 **3,217 行**。基线 `/tmp/flexlb-before-admission-owner.tar.gz`，差异 `/tmp/flexlb-admission-owner.diff`。
- `/tmp/flexlb-admission-owner-final-focused.log`：7 类准入/交付/资源结算测试 **115 项通过**，BUILD SUCCESS。新增五变体验证 TRANSFERRED、OWNERSHIP_LOST、ENDPOINT_RETIRED、dispatch 抛错和无 Decode；未绑定与外来身份拒绝、重复移交、剩余成员回收及 handoff 只关闭一次均通过。
- 三个 subagent 完成并发、设计和测试 review，无阻断发现。git diff --check 通过。按用户要求仅跑本地测试；该小改动没有重跑远端性能，既有性能验收缺口未关闭。

## 后续：WorkerBatcher 的停止清理与提交回收归属

- stopAndDrain 删除 queueLocked 标记，获取锁成功后由 finally 解锁；监听器注销失败仍继续唤醒，解锁、线程中断和逐项 reservation 释放继续独立汇总错误，不影响后续队列项回收。停止的主执行者、共享结果及精确失败项保留机制保持不变。
- commitPreparedSelection 的 committed/handedOff 收敛为 ownsCommitted。提交返回后由当前方法负责 abort；交给 handoff 前完成参数检查并转交该责任，handoff 的 finally 负责处理未移交资源，避免诊断异常引起外层重复 abort。
- sync 生产 Java **28,217 → 28,202，净减 15 行**；累计从 30,842 减少 **2,640 行**，距 25,000 仍差 **3,202 行**。基线 `/tmp/flexlb-before-batcher-cleanup.tar.gz`，独立差异 `/tmp/flexlb-batcher-cleanup.diff`。
- `/tmp/flexlb-batcher-cleanup-final-focused.log`：WorkerBatcher*Test、QueuedBatchDeliveryTest、BatchDeliveryStrategyTest、RouteDeliveryStrategyTest 共 **89 项通过**，BUILD SUCCESS。新增注销失败的停止用例检查队列排空、跨线程读取不被遗留锁阻塞、退休无残留身份以及重复停止返回同一错误。另新增 WorkerBatcher 实际事务边界的移交前/移交时异常两变体，abort 同时抛错，验证 commit/handoff/abort/close 次数、原始错误及 suppressed。事务是 mock，真实 Batch/Route 资源路径由同轮已有测试覆盖。
- 三个 subagent 完成并发、设计、测试 review，无阻断发现；测试 reviewer 提出的移交边界覆盖缺口已补齐并复核。git diff --check 通过。
- 本轮仅运行本地测试。25,000 行目标与远端突发 P99 性能验收尚未完成。

## 后续：发布器关闭结果与响应结果去重

- RequestCompletionPublisher 删除 PublisherPhase、closeFailure 和跟随关闭者的手写 wait 循环。closeCompletion 为空表示开放，非空表示关闭已由唯一执行者认领，完成后承载同一清理结果；准入与认领继续在 lifecycleMonitor 下互斥，跟随者 join 不丢失中断标记。
- 在途 publication 计数和 ThreadLocal publicationDepth 仍分别承担排空与回调重入保护。回调内关闭继续交给独立 closer，不能等待自身；启动 closer 失败仍按旧逻辑关闭 executor 并共享失败。
- 删除 abortClaimedPublication、ownedBy 纯转发及重复的锁外断言，复用已有 closePublication 与 RequestScheduler.requireOutsideSlotLock。
- SelectedPublication 不再展开复制 selected/completion/response/failure/interrupt，而直接引用 Context 已选的不可变 ResponseResult；null 表示本次没有取得响应，仍先执行 finishRequest。所有响应类型、精确对象比较及原始 Future 身份保持不变，没有新建第二个 ResponseResult。
- sync 生产 Java **28,202 → 28,163，净减 39 行**（包含 Java import 分组规范整理）；累计减少 **2,679 行**，距 25,000 行还差 **3,163 行**。基线 `/tmp/flexlb-before-publisher-close.tar.gz`，独立差异 `/tmp/flexlb-publisher-close.diff`。
- `/tmp/flexlb-publisher-result-final-focused.log`：8 类请求发布、终态结算、关闭编排与生命周期测试 **163 项通过**，BUILD SUCCESS。新增正常/失败两变体用真实 permit 阻塞关闭，检查停止新准入、并发关闭共享首因、中断保留和第三次关闭后仍仅 shutdown 一次；失败变体只将 executor.shutdown 替换为抛错 mock，正常变体使用真实线程池。既有 256 个回调及回调内关闭、三类响应、ACK 与终态竞争继续通过。
- 三个 subagent 完成关闭协议与响应结果去重的并发、设计、测试 review，无阻断发现；测试建议的唯一 shutdown 断言已补齐。最终测试后仅整理 imports/Javadoc 和 switch 空格，无逻辑修改；git diff --check 通过。
- 该局部状态简化只运行本地测试。25,000 行目标及既有远端突发 P99 验收均未完成。

## 后续：继续执行器以唯一任务队列判断排空

- RequestContinuationExecutor 删除 pending 镜像计数。每个精确 Context 的队列 entry 在 fact 运行期间保持存在（deque 可以暂时为空），仅唯一 drain 在确认无下一任务时移除；queues.isEmpty 直接证明无运行中和排队中的任务。
- 最后一个 entry 移除时 notifyAll，删除每条 fact 完成后为更新 pending 单独获取 monitor 的路径。相同 Context 的重入事实仍交给同一个 drain，其他 Context 继续并发执行，fact/日志失败隔离保持。
- close 在 lifecycle monitor 内复用 awaitIdle，返回后同锁停止接受任务，避免排空判定与准入关闭之间出现竞态。closing/accepting/closed 三个既有事实各有独立职责，本轮保留。
- sync 生产 Java **28,163 → 28,153，净减 10 行**；累计减少 **2,689 行**，距 25,000 行还差 **3,153 行**。基线 `/tmp/flexlb-before-continuation-drain.tar.gz`，独立差异 `/tmp/flexlb-continuation-drain.diff`。
- `/tmp/flexlb-continuation-drain-focused.log`：6 类执行器、终态/投递结算、Inactivity、Registry、清理死锁测试 **107 项通过**，BUILD SUCCESS。新增 awaitIdle/close × 正常池/recovery 四变体，验证 deque 为空但任务仍运行时不能提前排空、任务内重入提交仍被等待、中断保留、空闲后同 Context 新执行者及关闭后拒绝提交。原多 Context 串行/并行测试保留。
- 三个 subagent 完成并发、设计和测试 review，无阻断发现。git diff --check 通过。本轮局部状态简化只运行本地测试，未宣称热路径锁操作减少等于远端性能已达标；目标及 P99 缺口仍未完成。

## 后续：Prefill 提交校验与 OPEN 回滚收拢

- 删除单调用 commitRoutesUnderLock 层，将 route 提交保留在 commitRouteGroup 的同一锁内。锁前仍冻结 reservation 列表并验证 ledger 身份，锁内先校验所有权与构造 handoff，再移除队列索引及提交成员；不改变资源生效边界。
- validateActiveGroup 与 validateRouteGroup 合为 validateGroup。queuedOnly 明确区分 batch 必须拥有活动队列索引与 route 可包含 DIRECT 未入队请求；route 原有 queue membership 一致性检查保留，不统一两类资源账本。
- releaseOpenLease 删除 capacityReleased 标记：非 OPEN 在锁内早退并由 finally 解锁；OPEN 认领回滚后锁外关闭可选 handoff，finally 通知容量。保留非空判断，异常继续由 Failures 保留原 Throwable。
- 首次 `/tmp/flexlb-prefill-commit-cleanup-focused.log` 出现 30 个错误：误将可空的 route generationHandoff 传给不支持 null 的 Failures.close。三个 reviewer 与本地测试发现同一问题，已恢复非空判断。没有修改通用 Failures 语义或放宽测试。
- 修复后 `/tmp/flexlb-prefill-commit-cleanup-final-focused.log` 182 项通过；补两变体后 `/tmp/flexlb-prefill-commit-cleanup-verified.log` **184 项通过**。新测试覆盖 route 无 handoff、batch handoff 关闭抛错、容量通知自身抛错、锁外通知恰好一次、重复 close、批次容量归零及保留 queued owner。
- sync 生产 Java **28,153 → 28,129，净减 24 行**；累计减少 **2,713 行**，距 25,000 仍差 **3,129 行**。保留原长字符串换行格式，没有压缩格式计入削减。基线 `/tmp/flexlb-before-prefill-commit-cleanup.tar.gz`，差异 `/tmp/flexlb-prefill-commit-cleanup.diff`。
- 三个 subagent 均复核修复，无剩余本轮阻断问题；git diff --check 通过。只运行本地验证，既有远端 P99 验收缺口保持。
- 完整 `/tmp/flexlb-prefill-commit-cleanup-reactor.log`：API reactor BUILD SUCCESS；common 225、cache 37、grpc 15、sync **1138** 全部通过，API 175 项中 1 项跳过、其余通过。覆盖最近发布器、继续执行器和 Prefill 资源清理改动组合；该结果不代表远端性能验收。

## 后续：批次地址构造去重与发送转发层移除

- DefaultBatchDispatcher 的 Prefill/Decode protobuf 地址缓存共用 addRoleAddr；两份缓存仍按实际角色保留，缺失 Decode 只省略当前请求字段，不清空可复用对象。sameRoleAddr 的角色/IP/HTTP/gRPC 匹配规则及 DP rank 排序不变。
- 内联 BatchTransaction.send 的单一调用，保持 SUBMITTED 校验、队列深度校验、发送与 transportAccepted 的原顺序。两个 transport 状态转换方法仍保护发送前后边界。
- sync 生产 Java **28,129 → 28,094，净减 35 行**；累计减少 **2,748 行**，距 25,000 仍差 **3,094 行**。基线 `/tmp/flexlb-before-batch-address.tar.gz`，独立差异 `/tmp/flexlb-batch-address.diff`。
- `/tmp/flexlb-batch-address-final.log`：DefaultBatchDispatcherTest、BatchDeliveryStrategyTest、QueuedBatchDeliveryTest **73 项通过**。新增单 rank/多 rank 两变体覆盖 Decode 缺失后对象复用、Decode 各地址变化，以及 reviewer 建议补齐的 Prefill IP/HTTP/gRPC 改变。真实构造 protobuf、模拟 gRPC ACK，不涉及远端吞吐。
- 三个 subagent 已复核最终增量，无阻断发现。本轮局部变更只跑本地测试；25,000 行目标和既有远端突发 P99 验收缺口未完成。

## 后续：Decode 发送许可直接记录消费结果

- EngineDispatchPermit 删除专用五态 Resolution 枚举。dispatchResult 为空表示尚未消费；非空即后续 dispatch 应返回的固定结果。首次归还成功仍向本次 release 返回成功，但缓存 OWNERSHIP_LOST，后续调用不能重新取得发送权；已发送、已退休、身份丢失的结果仍分别保留。
- resolve 的 synchronized、applyDispatch 的资源及通知边界不变；调用抛错时不缓存结果，与旧 ACQUIRED 行为相同。没有合并 endpoint 的实际资源账本与许可消费结果。
- RequestScheduler.applyPrefillActivityLocked 改为 void：该函数只在锁内归并 NOT_FOUND 后的 Worker 活动，原来始终返回 null。两处调用仍在原位置，其他真正返回发布/清理操作的函数保留。
- sync 生产 Java **28,094 → 28,078，净减 16 行**，仍 94 文件；累计从 30,842 减少 **2,764 行**，距 25,000 仍差 **3,078 行**。基线 `/tmp/flexlb-before-preemption-install.tar.gz`，最终差异 `/tmp/flexlb-dispatch-resolution.diff`。
- `/tmp/flexlb-dispatch-resolution-focused.log`：Decode*、Eviction*、PreemptionRegistration、RequestResourceAccounting、RequestLifetime、QueuedDecodeWithdrawal 共 **257 项通过**。补强退休后 dispatch→release→dispatch 与 release→dispatch 的两个现有用例后，`/tmp/flexlb-dispatch-resolution-final.log` **44 项通过**；首次归还后 dispatch 必须 OWNERSHIP_LOST，首次 dispatch 已取得 ENDPOINT_RETIRED 后失败的 release 不改写该结果。
- 三个 subagent 独立 review 了状态映射、锁/异常边界、既有测试覆盖，无阻断发现；两处过时注释已同步。git diff --check 通过。本轮只运行本地测试，未重复远端性能；生产行数和突发 P99 目标仍未完成。

## 后续：类图检查与统一路由提交

- 类图暴露的主要结构问题：PrefillEndpoint 从 WorkerBatcher 反取 PrefillState；DIRECT 也依赖 batcher 的版本化路由投影缓存。后续迁移资源账本必须同时处理缓存归属、共同锁与版本失效，不能简单删除 DIRECT batcher。
- 本轮先统一独立的重复提交路径：EvictionManager 取得精确 Decode reservation 后调用 RouteAdmission.tryEnqueue，复用普通请求的创建、提交时间、Prefill offer 和提交标记。删除 commitQueuedRequest/commitQueued 两个入口，createScheduledRequest 收窄为包内可见；抢占与普通路径仍分别 adopt/reserve，不重复预留资源。
- sync 生产 Java **28,078 → 28,068，净减 10 行**，累计从 30,842 减少 **2,774 行**，距 25,000 仍差 **3,068 行**。基线 `/tmp/flexlb-before-shared-route-commit.tar.gz`，差异 `/tmp/flexlb-shared-route-commit.diff`。
- 三个 subagent review 未发现生产语义回归。测试 review 指出的成功路径缺口已补：QueuedDecodeWithdrawalTest 使用真实 EvictionManager/RouteAdmission/DecodeEndpoint，覆盖入队成功和失败两种情况，检查精确 reservation、原 future、单次 offer、victim requeue，以及失败 finally 释放。新 fixture 的 requestId/ipPort 缺失已修正。
- `/tmp/flexlb-shared-route-final.log` **69 项通过，BUILD SUCCESS**。本轮局部变更按约定仅运行本地测试；远端性能门槛仍未验收通过。

## 后续：队列版本由成员索引统一维护

- 删除 WorkerBatcher 独立 AtomicLong queueVersion 和 publishActiveIndexUnderLock、detachNextStopTerminalUnderLock、removeUnderLock 三个计数包装函数。PrefillActiveIndex.Ordered 在成功 add/remove、非空 clear 时发布 volatile revision；Disabled 始终为 0。投影缓存、等待条件和诊断读取同一索引版本。
- enqueue、抢占置换、回滚、组批提交、终态移除、stop detach、retirement clear 均通过同一成员索引修改。版本用于相等判断与失效，不承诺每事务加一；组提交多个成员以及失败事务中成功 add/remove 都可以使版本前进多次。PrefillState.mutationVersion 保留，因为 DIRECT/UNINDEXED 的已提交工作不会改变队列成员。
- sync 生产 Java **28,068 → 28,044，净减 24 行**；累计减少 **2,798 行**，距 25,000 仍差 **3,044 行**。基线 `/tmp/flexlb-before-index-version.tar.gz`，增量 `/tmp/flexlb-index-version.diff`。
- 三个 subagent 从设计、并发、测试方向 review，无阻断发现。PrefillActiveIndexTest 的 400 步随机序列补强成功变更版本前进、重复失败操作及空 clear 版本不变；PrefillStateSnapshotTest 补强真实双成员 batch commit 版本前进及只剩未提交成员，OPEN route/batch lease rollback 不改变队列成员版本。
- `/tmp/flexlb-index-version-focused.log` **180 项通过**；`/tmp/flexlb-index-version-reactor.log` **BUILD SUCCESS**：common 225、cache 37、grpc 15、sync 1,142、API 175（1 skip），合计 1,594 项记录。本轮局部变更仅本地测试，既有远端 P99 缺口尚未完成。
- 下一步移除 DIRECT WorkerBatcher 必须一并完成：Endpoint 持有同一 ledger/lock/index；唯一 ProjectionSource 和捕获缓存移到 Endpoint，QUEUE/DIRECT 共用；DIRECT 不构造 Batcher；status/learning/routeReady 的失效在 Endpoint 汇合，QUEUE 额外唤醒线程。仅迁移 ledger 构造而保留 DIRECT Batcher 会增加装配和转发，因此不作为独立完成项。
- 最终补强断言重新执行：`/tmp/flexlb-index-version-final.log` **27 项通过，BUILD SUCCESS**；git diff --check 通过。

## 后续：Endpoint 持有唯一账本与投影缓存，DIRECT 不再创建 Batcher

- PrefillEndpoint 创建 PrefillState，WorkerBatcher 显式借用该账本的同一锁、索引和容量通知回调。删除 ownedState 反向取账本接口；Batcher 构造只接受 QUEUE，删除 DIRECT 空线程的启动/投递/关闭分支。
- 原 ProjectionSource 整体归属 Endpoint，QUEUE/DIRECT 共用三元版本失效、锁内捕获、锁外延迟物化及旧捕获不能覆盖新版本的机制。WorkerBatcher 只提供 QUEUE 特有约束和阻塞头；DIRECT 没有 worker/runtime，沿用自己的已提交工作投影与 generation retirement。移动代码不计作删减，实际净减来自旧装配/空运行时/转发流程删除。
- schedulingInputVersion 由共享 PrefillState 维护，QUEUE 等待和 Endpoint 投影读取同一版本。容量 listener 注册/移除与 ledger 的永久回调使用同一 Runnable 身份，避免重复通知。review 找到 initializeFromPreparedStatus 遗留 runtime 方法引用导致 DIRECT NPE，已改成 Endpoint 自身失效入口并回归通过。
- 队列测试显式创建共享 ledger；投影并发与捕获性能测试改走真实 Endpoint 入口，保留共享物化、锁外构建、旧 capture 不覆盖新 block、work 对象复用断言。新增 DIRECT 无 Batcher、缓存复用/失效、资源回滚用例。SnapshotBench 改用两版本都支持的真实 Endpoint 构造和公开投影入口，已编译验证。
- sync 生产 Java **28,044 → 28,019，净减 25 行**，累计减少 **2,823 行**，距 25,000 仍差 **3,019 行**。基线 `/tmp/flexlb-before-direct-batcher-removal.tar.gz`，本轮差异 `/tmp/flexlb-direct-batcher-removal.diff`。
- 三个 subagent review 完成，已修问题无遗留阻断。`/tmp/flexlb-direct-removal-reactor.log` **BUILD SUCCESS**：common 225、cache 37、grpc 15、sync 1,143、API 175（1 skip）。`/tmp/flexlb-direct-removal-before-tool-check.log` 与 `/tmp/flexlb-direct-removal-tool-check.log` 的 2,000 selections/2,000 plans/4,000 candidates 签名一致：`f8ee1a60677270f92efb9d51785512d54eec8ecf3a0be1dec889c62cce2d1062`。该工具验证纯投影结果，不代替并发回归和远端端到端性能。
- 已同步指定远端目录，容器 luoli_gpu，原源码备份 `/tmp/flexlb-before-direct-removal-20260925.tar.gz`（host）。工具同步遇到容器生成的 __pycache__ 权限失败，排除缓存后重试；未把环境错误当作测试结果。源码清单 `/tmp/flexlb-direct-removal-sources.sha256` 共 600 文件，摘要 `3260515b21ee4a817a7956776c9ed8ddc7f32ebd9b095f353dd20c773044970c`。远端按原门槛执行 Decode 锚点、sync 性能、完整 API 性能和 750P/750D 四场景；日志容器 `/tmp/flexlb-direct-removal-perf-20260925/`，结果待收集。
- 远端结果已完成：Decode **8/8**、sync 性能 **4/4**、750P/750D **4/4** 通过；完整 API 性能 **14/16** 通过。剩余突发 Master P99 **783 ms > 250 ms**，以及 1P/2D 10,000 QPS 投递等待 P99 **62 ms > 50 ms**。单规划线程对照提高吞吐并降低 batch wait，但整体 P99 **713 ms** 仍失败，因此不改生产默认线程数。详见 `remote-performance-2026-09-24.md` 第四快照；远端结果不能称为达标。


## 2026-09-25：清理操作归回唯一请求执行者

- 对照当前类关系，删除 `RequestTerminalCleanup` 独立执行层：它持有清理操作表，却反向调用 Scheduler 查询同一进度、routing、preemption 并提交终结。
- `RequestScheduler` 直接持有 cleanupOperations，负责清理调度、精确进度仲裁和终结；端点仍执行资源释放。删除 start/forget/progress 转调、构造注入和 scheduler 反向引用。进度类以及仅供旧清理类使用的终结方法收窄为 private。
- 保持 RUN_AGAIN 防丢唤醒、同一 progress 身份核对、锁外结算、Prefill/Decode 独立失败聚合和 UNKNOWN 资源约束。未把执行状态放回 BalanceContext。
- 测试初始化不再注入 mock 清理器，走真实清理流程；原断言和并发屏障保留。定向 96/96 通过，日志 `/tmp/flexlb-cleanup-owner-focused.log`。三位 reviewer 完成并发、设计、测试审查；发现的初次代码插入位置错误已修复。
- 本轮生产物理行数 28,010 → 27,968，净减 42；距 25,000 仍有 2,968。此前 Capture 列表优化的 9 行变化在本轮基线中，性能取舍仍待远端 A/B，不能归入本轮收益。
- 本轮基线 `/tmp/flexlb-before-cleanup-owner.tar.gz`，独立差异 `/tmp/flexlb-cleanup-owner.diff`。全量 reactor 日志 `/tmp/flexlb-cleanup-owner-reactor.log`，结果完成后补记。本轮不改变调度算法或线程，先执行本地回归；此前远端性能失败尚未解决。
- 全量本地 reactor 最终 BUILD SUCCESS：common 225、cache 37、grpc 15、sync 1,143、API 175（跳过 1）；共 1,595 条测试记录、无失败/错误。当前总目标仍未完成。

## 2026-09-25：完成列表构建的远端验证并撤回

- 远端 A/B 否定保留 `Stream.toList` 替换的收益：深队列省约 22%–23% 分配，但耗时增加约 9%–10%；浅队列分配增加。完整方法和数据见 remote-performance-2026-09-24.md 新节。
- 仅恢复 PrefillActiveIndex.java，当前文件与 `/tmp/flexlb-before-capture-list.tar.gz` 内基线逐字节一致；远端指定目录/容器同文件已同步并核对 SHA256。清理层合并保留。
- 恢复后本地定向 134/134 通过，日志 `/tmp/flexlb-capture-revert-focused.log`；三位 reviewer 复核撤回范围、测试及性能结论，无阻断问题。
- 当前生产物理行数 27,977；相对 30,842 累计减少 2,865，距离 25,000 尚有 2,977。端到端性能仍未验收，不宣称目标完成。
- 本轮检查 ScheduledRequest 字段发现：Decode 输入、当次路由和期限有冻结语义，不能未经证明改为读取可变 Context；后续应先规范请求级顺序初始化和精确投递凭据，再删除副本。

## 2026-09-25：抢占事务统一使用精确 reservation 身份

- DecodeState.EndpointPreemptionAttempt 直接持有 incoming ReservationHandle，删除拆分 requestId/token 表示和退休时重新构造；过期、COMMIT 与 incoming 匹配复用精确句柄。
- remainingVictims 从 Map<requestId, ReservationHandle> 改为 Set<ReservationHandle>，去掉重复键；接管 begin 内独占创建的集合，不再复制。所有读写仍受 admissionLock 保护。同一句柄重复输入在集合检查拒绝，同 ID 不同 token 由 isExactReservation 拒绝；退休最后仍排序。
- 采纳设计 review，操作 owner 保留普通 final class，避免可变集合参与 record 的结构 equality。保留请求协议与资源账本各自的阶段，未将 UNKNOWN 视为释放证据。
- 新增重复 exact/stale token 两个参数场景及活跃抢占退休测试：拒绝时账本版本/claim/incoming 不变；close 两次只回报一次精确 owners，清空账本及事务。
- 本地相关抢占、Decode、撤回测试 214/214 通过，日志 `/tmp/flexlb-preemption-identity-focused.log`。三个 reviewer 已审查生产变更；测试 reviewer 指出的覆盖缺口已补齐并复核。此轮局部账本表示变化，仅本地回归，未重跑远端 perf；此前端到端性能失败仍待解决。
- 生产行数 27,977 → 27,962（净减 15），距目标 2,962。基线 `/tmp/flexlb-before-preemption-identity.tar.gz`；生产独立差异 `/tmp/flexlb-preemption-identity.diff`。

## 2026-09-25：投递成员使用标准资源关闭协议

- PrefillAdmissionResources.Member 实现 AutoCloseable。未交接成员的 Decode permit 释放与本地 admissionOwned 结束集中到 close；finally 保证释放抛错后也结束本地清理责任，已交接成员 close 不再释放。
- 删除 rollbackMember/rollbackReservation/rollbackPermit 三个专用函数，Member 与 Prefill Reservation 共享 rollback(AutoCloseable, priorFailure)，permit 捕获失败复用现有 Failures.run。BATCH/NON_BATCH 的顺序、部分成功前缀和逐资源异常聚合保持。
- DIRECT 使用 try(member)，删除专用 finally。明确行为改进：提交和关闭同时失败时保留提交首因、清理错误 suppressed，旧 finally 会覆盖提交错误。
- 新增 DirectAdmissionContractTest.directCommitKeepsPrimaryFailureWhenPermitCleanupAlsoFails：真实 endpoint ledger，Prefill 提交失败，Decode 先真实释放 permit 再抛错误，断言首因同一对象、唯一 suppressed、一次 release、permit 和临时 reservation 清空。
- 本地准入/投递/批次/资源守恒/撤回定向 323/323 通过，日志 `/tmp/flexlb-member-close-focused.log`。并发及设计 review 无阻断，测试 review 提出的 DIRECT 双异常覆盖已补齐。此次局部资源关闭重构未跑远端性能；端到端性能遗留失败仍未解决。
- 本轮生产行数 27,962 → 27,939，净减 23，距目标 2,939。基线 `/tmp/flexlb-before-member-close.tar.gz`，差异 `/tmp/flexlb-member-close.diff`。

## 2026-09-25：全局队列边界审查与容量版本归一化

- 审查未支持合并 OrderedRequestQueue / PlacementWaitQueue / GlobalQueueCoordinator：O(1) 精确撤队、有限扫描 cursor、ready retry、域的一次重试许可、在途计划与控制票据各有独立语义。已有 cleanupWakeDuringCapturePreservesPriorityWithinTheFrontier 测试证明取消期间的同步唤醒会改变候选次序，不能直接删除结果排序/去重。
- 落实可证明的重复状态删除：PlacementAvailability 不再同时保存带 group 的 exact endpoint 版本和 role/address 规范版本。PlacementKey.capacityDomain 统一读、写、park、release 的规则；组/角色聚合版本与原 key 通知保持。
- 新增同地址换组测试：两个 exact 查询共享最新版本，两个 group 仍各自推进，其他地址不受影响。乱序发布 max 保护测试保留。
- 本地定向 BUILD SUCCESS：sync 44、cache 12，均无失败/跳过，日志 `/tmp/flexlb-capacity-domain-focused.log`。并发、设计 review 无阻断；测试 review 独立复核。本轮仅局部版本表示改变，未追加远端性能测试。
- 生产行数 27,939 → 27,937，净减 2；这是重复版本存储的消除，非大幅行数进展。基线 `/tmp/flexlb-before-capacity-domain.tar.gz`，差异 `/tmp/flexlb-capacity-domain.diff`。目标和此前端到端性能缺口仍未完成。

## 2026-09-25：删除无调用者的旧超时事件

- 全仓调用检查确认 DeferredTerminal.timeout/Kind.TIMEOUT 没有生产、测试或反射消费者，删除事件、工厂方法及 decideRequestEndLocked 中对应旧分支。
- 真正的调度期限仍经 onSchedulingDeadline 记录首次 DEADLINE_EXCEEDED；inactivity 保留独立事实，TerminalOutcome.timeout 与对外 TIMED_OUT/BATCH_SLO_EXPIRED 映射未删除。待定原始事件与已选择 TerminalOutcome 的生命周期不同，未强行合并。
- 本地定向 BUILD SUCCESS：sync 261、common 2，无失败/跳过。日志 `/tmp/flexlb-obsolete-timeout-focused.log`。并发、设计 review 确认无遗留调用或兼容依赖；测试 review 核对真实超时路径。
- 生产行数 27,937 → 27,927，净减 10，距目标 2,927。基线 `/tmp/flexlb-before-obsolete-timeout.tar.gz`，独立差异 `/tmp/flexlb-obsolete-timeout.diff`。删除不可达分支未追加远端 perf；原性能缺口仍待解决。


## 2026-09-25：抢占登记去除执行入口

- `PreemptionRegistration` 删除 Scheduler 引用与 applyPhase/release/completePreemption 三个转发方法；Coordinator 直接调用 Scheduler。
- Scheduler 接口仅接收精确 claim，从其不可变 owner 获取原 Context，再检查自身 map 中的 claim 实例身份。保留原锁与状态迁移逻辑。
- `AttemptCapability` 成为 Coordinator 的内部实例类；异步执行期间持有服务对象，持久登记对象不再反向引用 Scheduler。
- 生产 Java 物理行数 27,927 → 27,922，净减 5；距 25,000 尚差 2,922。不将此次职责收敛声称为大幅减量。
- 前一轮未完成的 ExpirationTimer cleanup helper 实验已从独立基线恢复，未混入此轮。
- 本地 focused 105/105 成功，无失败/跳过：`*Preemption*Test,*Eviction*Test,DeliverySettlementTest,RequestRegistryTest,RequestSlotTerminalSettlementTest`。新增两个 Scheduler 同请求 ID 时拒绝异主 claim 的测试；保留同 ID 重注册后的迟到事件覆盖。
- 日志 `/tmp/flexlb-passive-preemption-tests.log`；基线 `/tmp/flexlb-before-passive-preemption.tar.gz`；diff `/tmp/flexlb-passive-preemption.diff`。
- 三个 reviewer 分别审查并发身份、职责结构、测试迁移，均无阻断发现。此次小范围调用归属调整未跑远端性能；既有性能失败仍未解决，总目标保持未完成。
- 按测试 reviewer 建议加强异主调用后的 phase 断言：原 owner 必须仍能首次进入 CANCEL_IN_FLIGHT，防止“更新了状态却返回 false”漏检。增强后 RequestSlotTerminalSettlementTest 14/14 通过，日志 `/tmp/flexlb-passive-identity-test.log`。


## 2026-09-25：Engine 抢占直接交接精确 reservation

- DecodeState 的锁内收尾返回 attempt.incoming；COMMIT 保留精确 token 与 victims 清空校验。对外分为 `commitPreemption` 返回 ReservationHandle、`abortPreemption` 返回 boolean，删除 PreemptionDecision；内部共用收尾逻辑。
- PreemptionResult 保存 reservation，committed 从非 null 推导。EvictionManager 删除成功后按 requestId 重新查找 reservation 与相应失败分支，直接采用精确结果。
- RouteAdmission 仍验证 generation pin 与 markQueued 的精确 token，采用失败按原能力释放；不存在从失败 COMMIT 或 ABORT 结果取得可交接能力的路径。
- 生产 Java 27,922 → 27,916，净减 6。距离 25,000 仍差 2,916；本轮解决身份交接边界，尚未完成大幅代码收敛。
- 本地 `*Preemption*Test,*Eviction*Test,*Decode*Test,QueuedDecodeWithdrawalTest` 共 214/214 通过，无失败/跳过。覆盖真实 endpoint 精确句柄、重复提交失败、coordinator 结果传播，以及 manager 采用失败时 RESOURCE_EXHAUSTED、不入队、关闭 admission。
- 日志 `/tmp/flexlb-preemption-handoff-tests.log`；生产与原调用测试基线 `/tmp/flexlb-before-preemption-handoff.tar.gz`，diff `/tmp/flexlb-preemption-handoff.diff`。随后补强的 EvictionManagerTryAdmitTest 不在这份局部基线中，已在最终 214 项测试中执行。
- 三个 reviewer 审查并发、结构、测试；采纳公开 COMMIT/ABORT API 分拆及采用失败响应断言建议。此轮不涉及调度热路径算法或线程配置，按小改动运行本地测试，未宣称远端性能已达标。


## 2026-09-25：期限清理复用已有异常汇总

- ExpirationTimer.DetachedDeadlines.release 的三段取消及 maintain 的两段维护改用现有 Failures.run，保留取消顺序、released 一次性保护、所有独立操作均尝试、首异常与 suppressed 顺序。
- 删除该类重复的私有 rethrow，统一 Failures.rethrow。review 发现原私有实现只抛 RuntimeException/Error，与捕获 Throwable 的共同工具组合会静默丢弃 checked 回调异常，现已修复并新增 cause 身份回归。
- 新增 ExpirationTimerTest 三项：独立取消全部失败仍按序各调用一次；terminal records 清理失败后仍 sweep；意外 checked 回调异常仍报告。
- focused `ExpirationTimerTest,Request*Test,DeliverySettlementTest` 253/253 通过，无失败/跳过；日志 `/tmp/flexlb-timer-cleanup-tests.log`，基线 `/tmp/flexlb-before-timer-cleanup.tar.gz`，生产 diff `/tmp/flexlb-timer-cleanup.diff`。
- 三个 reviewer 完成；并发 reviewer 的异常传播问题已修复并复核。此轮小改动只跑本地回归，未跑远端性能。
- 生产 Java 27,916 → 27,886，净减 30，距目标尚差 2,886。整体功能/性能验收未完成。
- 结构审查没有找到可证实净删 50 行以上的 RequestScheduler 终态仲裁重复；Prefill/Decode 退休和普通 Worker 终态的 preemption/cleanup 条件确有区别，未强行合并。


## 2026-09-25：Decode 心跳确认态计数归并

- doCalibrate 删除 actualConfirmed/retainedConfirmed/syntheticallyHeldSlots 三个分散计数器，在已有的 confirmed reconciliation 遍历中直接统计最终保留的 confirmed owner。普通缺席请求删除后 continue，claim 缺席仍保留并维持原 KV hold。
- 同一 admissionLock 内、相同 terminal 结算次序，无新遍历或分配。正常唯一 requestId 输入等价；重复 requestId 观测不会再把一个账本 owner 重复计数。
- 新增混合心跳回归，同轮包含缺席 claim、RECEIVED 回退、RUNNING 和普通缺席请求，连续两轮断言 confirmed=3、含 incoming shadow 的 totalLoad=4，普通缺席已移除。
- 原 scoped `*Decode*Test,*Preemption*Test,*Eviction*Test,QueuedDecodeWithdrawalTest` 214/214 成功；新增后 DecodeEndpointLayeredViewTest 29/29 成功。日志 `/tmp/flexlb-confirmed-count-tests.log` 与 `/tmp/flexlb-confirmed-count-mixed-test.log`。
- 三个 reviewer 均无阻断。基线 `/tmp/flexlb-before-confirmed-count.tar.gz`、生产 diff `/tmp/flexlb-confirmed-count.diff`。小范围状态计数调整仅本地测试，不声称性能验收完成。
- 生产 Java 27,886 → 27,883，净减3行；核心收益是一个资源事实只在最终保留点计数。距25000尚差2883行，整体目标未完成。


## 2026-09-25：抢占观测与失败响应去重

- EvictionManager 七处相同的 RuntimeException 指标隔离边界共用私有 report(requestId, operation, metrics)，块内指标顺序/循环范围保持原样。
- 六处 admissionError 原本均传 RESOURCE_EXHAUSTED/RESOURCE_EXHAUSTED，工厂现只接收 message，错误响应字段不变；删除 placement 的恒定 kind 参数。
- 不改变抢占协议、准入、资源交接或清理流程。指标失败 warning 统一为 operation + incoming requestId + throwable，较旧日志少 endpoint/victim/case 明细，这是诊断文本取舍。
- 原两个 completion timeout 用例扩为四个含指标故障组合；模拟 reportEvictionPlan/reportCancelRequest/reportVictim 抛异常，仍验证 preempt 参数、精确 adoption、RESOURCE_EXHAUSTED、never enqueue/lookup 和 admission close。
- scoped `*Eviction*Test,*Preemption*Test,QueuedDecodeWithdrawalTest` 39/39 通过，无失败/跳过。日志 `/tmp/flexlb-eviction-observers-tests.log`；基线 `/tmp/flexlb-before-eviction-observers.tar.gz`；diff `/tmp/flexlb-eviction-observers.diff`；目标文件 diff check 通过。
- 生产 Java 27,883 → 27,847，净减36，距25000尚差2847；小改动仅本地测试，远端性能既有问题仍未解决。
- 三个 reviewer 完成，无业务阻断；采纳测试 reviewer 建议，显式 verify 三个故障指标均确实执行，增强后 EvictionManagerTryAdmitTest 13/13 通过，日志 `/tmp/flexlb-eviction-observers-fault-tests.log`。


## 2026-09-25：拒绝 Entry 提前投影构造实验

- 尝试删除 PrefillActiveIndex.Entry 的 volatile/synchronized 懒构造，在 add 时根据 ScheduledRequest 冻结字段创建 GroupPlanner.Item。
- 本地 `PrefillActiveIndexTest,*Projection*Test,WorkerBatcher*Test` 79 项中 2 failures + 1 error：failedMaterializationCanBeRetriedWithoutPublishingPartialResult 的异常从 projectedItems 前移到 add；concurrentReadersShareOneVersionWithoutHoldingQueueLock、capacityBlockInvalidatesAnUnfinishedProjectionCapture 的锁外物化屏障不再触发。
- 值冻结不等于物化时机可变。GroupPlanner.Item 还校验 seqLen/hitCache，提前构造改变失败重试边界，故直接撤回，未修改或放宽测试。
- 恢复后相同 79/79 通过。实验日志 `/tmp/flexlb-eager-entry-tests.log`，恢复日志 `/tmp/flexlb-eager-entry-revert-tests.log`，实验代码 `/tmp/PrefillActiveIndex-eager-rejected.java`，基线 `/tmp/flexlb-before-eager-entry.tar.gz`。
- 已同步至用户指定远端目录后发现上述失败；随即恢复远端同一文件。当前本地与 luoli_gpu 内 SHA-256 均为 403e147a5bc6b6ddf2bc36ed46352d8c9f6208852c87f434bd6fe5a6027209b6。远端 A/B 脚本仅已准备/上传，未启动，不存在性能通过结论或运行中性能任务。
- 三个 reviewer 审查实验与恢复；并发审查最初仅核对字段冻结而漏掉物化时机契约，测试审查指出具体反例后撤回。最终生产代码无本轮增量，仍 27,847 行；后续不能重复采用提前构造或通过修改测试绕过该边界。


## 2026-09-25：缓存指标去除转发层

- CostBasedPrefillStrategy 直接注入已有 CacheMetricsReporter，删除 EngineHealthReporter 的四个纯转发方法及依赖，并删除 strategy 内单行 affinity 转发；调用参数、顺序与异常边界保持。
- 所有 sync/API/mock-engine 显式构造调用已迁移，无兼容构造器；性能夹具 reporter 保留 stubOnly。生产 Java 27,847 → 27,817，净减 30。
- 定向 reactor 测试 common 3、cache 7、sync 80，共 90 项通过；API 源码与测试编译通过，本次未执行 API 测试。日志 /tmp/flexlb-cache-metric-delegates-tests.log。
- mock-engine test-compile 成功，71 个测试源编译；未执行测试。日志 /tmp/flexlb-cache-metric-delegates-mock-compile.log。
- 并发、设计、测试三个 reviewer 均通过，确认异常时 generation pin 清理与指标断言未弱化。小改动未跑远端性能；既有性能未达标问题仍未解决。


## 2026-09-25: Exact local preemption handoff

- DecodeState returns the exact ReservationHandle from reserveLocked through DecodeEndpoint and RequestScheduler to EvictionManager. Null means no replacement. Both adoption and failure cleanup use this handle; requestId lookups are removed.
- Production Java: 27,817 to 27,806 (-11); 2,806 remain above target. Baseline: /tmp/flexlb-before-local-handoff.tar.gz; diff: /tmp/flexlb-local-handoff.diff.
- Scoped tests: 217 passed (/tmp/flexlb-local-handoff-tests.log). Strengthened identity assertions: 56 passed (/tmp/flexlb-local-handoff-identity-tests.log). Final withdrawal class: 13 passed (/tmp/flexlb-local-handoff-reuse-tests.log).
- New deterministic test replaces the incoming reservation with a new token for the same requestId during requeue, then throws. Cleanup preserves the new reservation and original exception, and terminates the victim.
- Three reviewers approved, including the final token-reuse regression. No new blocker; pre-existing post-commit exception windows are unchanged. No remote performance run for this small change; full target and previous performance failures remain unresolved.


## 2026-09-25: Remove concrete delivery forwarding interfaces

- Removed five RequestScheduler wrappers: prepareBatchDelivery, prepareBatchMember, prepareRouteMember, claimBatchDelivery, claimRouteDelivery. Strategies and DIRECT admission invoke existing prepareDispatch/claimDelivery with the same lazy operations, kind and correlation id; preparation and handoff remain under the same Context monitor. prepareDispatch is package-private, not public. No new state or abstraction.
- Production Java: 27,806 to 27,786 (-20). Baseline /tmp/flexlb-before-delivery-boundary.tar.gz; diff /tmp/flexlb-delivery-boundary.diff.
- Removed duplicate test fixture stubs while retaining preparation loss, partial prefix, handoff failure, primary/suppressed exceptions and exact identity assertions. Initial 193 tests passed; final expanded scope including DIRECT and router: 229 passed, /tmp/flexlb-delivery-boundary-final-tests.log.
- All three reviewers approved the final five-wrapper scope, including test migration and unchanged failure assertions. Local verification only for this call-through simplification; no performance claim. EvictionManager asynchronous admission ownership and placement consolidation remain unfinished.


## 2026-09-25: Current full API reactor verification

- Ran mvnw -B -f rtp_llm/flexlb/pom.xml -P opensource,!internal -pl flexlb-api -am test on current production code (27,786 sync Java lines). BUILD SUCCESS, 1m53s. No production edits during this verification.
- common 225, cache 37, grpc 15, sync 1156, API 175 (1 skipped): 1,608 test records, 1,607 passed, 1 skipped, zero failures/errors. Log: /tmp/flexlb-state-consolidation-full-reactor.log. This excludes dedicated performance profiles and mock-engine module.
- Post-run Java source manifest: /tmp/flexlb-state-consolidation-full-sources.sha256 (472 files); SHA256 f7c3ba00dd4213014fc15c172d7c478ca430a46834fcf095c9baca860c062ac5.
- Three focused read-only reviews examined request snapshots and continuation closure. ScheduledRequest.future is redundant for canonical registered production requests, but current commit checks reject mismatched supplied futures; removal must move or redesign that validation and migrate standalone fixtures. Worker FIFO sequence/time cannot become mutable Context reads.
- Rejected a closure-field consolidation via CompletableFuture: fewer booleans alone does not simplify the protocol. Accepting during drain remains required for subsequent events generated by running facts.
- Target <=25,000 and remote performance constraints remain incomplete. This turn completes the full functional gate for accumulated local changes, not the overall goal.


## 2026-09-25: One canonical response future

- Removed ScheduledRequest.future storage and both constructor future parameters. future() delegates to its exact Context, whose registered future cannot be replaced. RouteAdmission no longer passes a future through tryEnqueue/createScheduledRequest; EvictionManager drops two now-unused private future parameters.
- Removed the two comparisons of an item future to the same Context future. Current Context identity, stage, admission and exact routing identity checks remain. No future lookup by requestId was introduced. Standalone fixtures explicitly initialize Context futures; null-only endpoint fixtures retain their previous behavior.
- Added assertions to the existing requestId-reuse regression: old route still references its original future, differs from the replacement request future, and replacing the registered old Context future throws without changing its reference.
- Full API reactor passed: common 225, cache 37, grpc 15, sync 1156, API 175 (1 skipped): 1607 passed, 1 skipped. Log /tmp/flexlb-canonical-future-tests.log. After private-parameter cleanup and added assertions, focused RequestRegistry/EvictionManager/QueuedDecodeWithdrawal 58/58 passed: /tmp/flexlb-canonical-future-identity-tests.log. SnapshotBench compiled against current test classpath with JDK21; not a performance run.
- All three reviewers approved identity, visibility and test migration. Removed the duplicate fixture setter identified by review. Baseline /tmp/flexlb-before-canonical-future.tar.gz; final diff /tmp/flexlb-canonical-future.diff.
- Production Java 27,786 to 27,779 (-7), target still 2779 lines away. No remote performance claim.


## 2026-09-25: Single-pass Prefill batch outcomes

- Replaced prepareBatchPredictionsUnderLock plus BatchReduction.projectedCompletion with prepareBatchOutcomesUnderLock. One affected-member scan computes repacking eligibility, survivors and completion facts without adding persistent state or duplicating batch membership.
- Preserve executionStarted/current RUNNING/terminal execution evidence as reasons not to repredict. Emit completion only when every remaining member has a terminal observation. Prediction and immutable result construction still precede StatusReconciliation publication and all canonical mutations.
- Short-circuit once both allTerminal and repack are false, preserving early exit for running batches with survivors. Canonical terminal settlement remains a separate complete traversal. Negative execution times cannot increase the accumulated max (initially zero), so removed the redundant non-negative branch.
- Added partialSuccessThenFailureCompletesBatchOnceWithoutLearningOrRepacking: first success 300ms plus another RUNNING leaves batch open; later error 200ms produces exactly one success=true/learning=false completion with actualWork=300ms; repeated terminal produces none.
- Final scoped Prefill/projection/delivery/WorkerBatcher suite: 289/289 passed, /tmp/flexlb-batch-outcomes-final-tests.log. All three reviewers approved including final short-circuit. Baseline /tmp/flexlb-before-batch-outcomes.tar.gz; final diff /tmp/flexlb-batch-outcomes.diff.
- Production Java 27,779 to 27,762 (-17). No remote performance rerun for this small consolidation; the latest remote source remains 27,779 with two API latency failures. Line and performance goals remain unmet.
- Follow-up candidate from state review: DecodeState settledAtMs appears replaceable with lastSeenAtMs for history-only entries, subject to verifying all rememberSettledLocked/prune paths and delayed heartbeat fencing. Not yet implemented.


## 2026-09-25：统一普通与抢占后的路由提交

- sync 生产 Java：27,762 → 27,589，净减 173 行；距 25,000 目标仍有 2,589 行。
- EvictionManager 删除 RouteAdmission、AdmissionHandle、BatchSchedulerReporter 依赖，
  删除新请求安装、失败响应和路由释放的重复编排。tryReserve 只返回精确抢占结果。
- GlobalQueueCoordinator 持有原选路调度权，普通/抢占成功共用 submitRoute。
  异步 callback 挂接前从 Plan 转移 route 与 handle，callback 用 TWR 结算。
- 本地冲突返回 null 并继续等待；已启动 Engine 抢占的失败是结果，不能当作可重试的未接管。
- review 发现并修复：ROUTING 期间已接受取消但 future 未完成，必须在选出 proposal 后
  检查 isAdmissionOpen，才能开始撤 victim 或 Engine Cancel。该检查同时覆盖 Scheduler 关停。
- 预期失败直接生成拒绝信息，不用抛异常表达分支；异常日志保留 cause。
- 完整 API reactor 回归：common 225、cache 37、grpc 15、sync 1163、API 175（1 skip），
  共 1614 passed / 1 skipped / 0 failures / 0 errors。
  日志 `/tmp/flexlb-unified-placement-final-tests.log`。
- mock-engine 测试源码编译成功（未执行该模块测试），日志
  `/tmp/flexlb-unified-placement-mock-compile.log`。
- 回归包含同步/异步成功、typed 失败、exceptional completion、采用资源失败、Prefill 阻塞、
  取消后禁止抢占、本地 victim 重排及精确 reservation 释放。
- 三个独立 agent review 完成，无剩余阻断发现。测试迁移时修正了两类断言问题：
  reservation record 比较值而非对象引用；TWR 关闭前后的资源归属分别验证。
- closePlacement 与 closeAdmission 分开期间的异步回调窗口在基线已存在，本轮不宣称修复。
- 基线 `/tmp/flexlb-before-unified-placement.tar.gz`；本轮核心 diff
  `/tmp/flexlb-unified-placement.diff`。类图已更新为当前实现。
- 远端 API performance gate 15/16 通过，burst P99 705ms > 250ms，仍未完成性能目标。


## 2026-09-25：Decode 历史记录与队列状态收敛

- sync 生产 Java：27,589 → 27,558，净减 31 行，距目标仍差 2,558 行。
- DecodeRequestState 删除 createdAtMs、lastSeenAtMs、settledAtMs，统一为 observedAtMs：
  shadow 使用创建时间，confirmed 使用最后 Engine 观测时间，history 使用终结时间。
  confirmed 不会转换回 shadow；所有状态/TTL写入仍由 admissionLock 保护。
- settled() 由无请求归属且无协议 owner 推导，不再把时间是否非零当作独立终态标记。
  rememberSettledLocked 的所有调用都在旧 owner 被删除后创建 token=0 的新历史条目，
  删除旧的复用状态和返回值分支。
- 四个入/出队状态转发方法合为 setQueuedLocked，阶段与 queuedUsage 在同一位置更新。
  删除 clearShadowAccounting 的无用 requestId 和 removePreemptionClaim 的无人使用返回值。
- 锁外 stats 先读取观测时间、再检查 volatile 阶段，防止确认过程中误将新观测时间当创建时间。
  TTL 保留 retention callback 后的 !confirmed 复查，因为该回调可重入确认请求。
- 新参数化用例覆盖过期、counterpart结束、FULL缺席三条历史创建路径；重复终态与迟到active
  在TTL内不复活请求，TTL后可观测untracked请求但旧token仍失效。
- 最终定向回归 222/222 passed，0 failure/error/skip。
  `/tmp/flexlb-decode-history-final-tests.log`；首轮同范围也通过。
- 基线 `/tmp/flexlb-before-decode-history.tar.gz`，diff `/tmp/flexlb-decode-history.diff`。
- 本轮为局部账本简化，仅跑本地定向回归。远端仍是27,589行版本，性能结论不冒充本轮结果；
  上轮burst P99 705ms仍未达门槛。
- 三个 subagent 已复核最终增量（状态归属、并发、测试覆盖），未发现剩余阻断问题。


## 2026-09-25：Prefill 预留身份与批次唯一性边界

- sync 生产 Java：27,558 → 27,537，净减 21 行，距目标差 2,537 行。
- RouteReservation.requestId 与 BatchReservation.headRequestId 均为原始 owner ID 的副本，
  删除字段和构造参数，直接引用唯一 originalOwner。route commit 改为比较精确 owner 对象。
- 删除 commitBatchUnderLock 的 findBatchWorkUnderLock 全表扫描及专属方法：
  reserveBatch 已在同一锁下由 findBatchReservationUnderLock 检查 OPEN/已提交两类批次。
  唯一 OPEN lease 在 commit 中转 OWNED，任何存活成员均保留 batch ID；最后成员结束才能重用。
  保留预留端唯一性检查和 route group 的重复成员防护，不添加批次索引。
- 新测试覆盖 OPEN 重复 ID、首成员终结后重复 ID、最后成员终结后同 ID 重用及释放。
- 定向 Prefill/Projection/Delivery/WorkerBatcher 回归 290/290 passed，0 failure/error/skip：
  `/tmp/flexlb-prefill-lease-identity-tests.log`。
- 基线 `/tmp/flexlb-before-prefill-lease-identity.tar.gz`；diff
  `/tmp/flexlb-prefill-lease-identity.diff`。
- 本轮为局部删除冗余扫描和身份字段，仅本地回归；没有声明吞吐或延迟收益。
  远端性能仍以27,589行版本的705ms burst P99失败为最新实测。
- 三个 subagent 完成当前实现与回归覆盖 review，无剩余阻断发现。


## 2026-09-25：投递成员索引去重

- CommittedAdmissionOwner 删除 membersByIdentity，保留唯一冻结 Member[]。
  transferToEndpoint 使用原成员下标与 ScheduledRequest 引用相等校验，保持 O(1) 查询。
- Batch 的 items 投影与 members 同序；Route 的 prepared 与其 members 投影同序；
  DIRECT 只有下标0。取消跳过只影响后续 submitted 列表，不压缩交接使用的原下标。
- closed、handoff绑定、admissionOwned 检查及按原准备顺序清理均保留。
  新断言验证错下标不会交接同组另一请求的 Decode permit。
- 本轮主要收益是移除重复索引与构造开销；不把分配减少宣称为实测性能收益。
  sync 生产 Java 27,537 → 27,536，净减1行，距25,000仍有2,536行。
- 定向 Delivery/Admission/RequestRegistry/RequestResourceAccounting/WorkerBatcher
  279/279 passed、0 failure/error/skip，日志 `/tmp/flexlb-delivery-member-index-tests.log`。
- 基线 `/tmp/flexlb-before-delivery-member-index.tar.gz`；diff
  `/tmp/flexlb-delivery-member-index.diff`。跨调用点测试fixture已迁移，无兼容重载残留。
- 局部索引删除仅跑本地回归；远端仍为27,589行版本，burst P99 705ms失败尚未解决。
- 三个 subagent 完成只读 review，无实现阻断发现。按测试review建议把Batch取消用例扩为
  三成员、取消首项或中间项，逐一验证survivor dispatch与lost release，补充后该类18/18通过：
  `/tmp/flexlb-delivery-member-index-skip-tests.log`。生产代码在279项回归后未再改动。


## 2026-09-25：请求终态的取消优先级收敛

- RequestScheduler.decideRequestEndLocked 将三个事件分支内重复的取消结果选择合为一次
  cancellationWins 判断与一次 cancellation outcome/error 构造；事件差别只决定详细文本。
- 保留 FAILURE 在 deliveryClaimKind.NONE 前后不同的 outcome：已认领投递后仍可 FAILED，
  即使客户端 Response 沿用先前 cancellationResponse。Outcome 与 Response 继续分别选择。
- INACTIVITY_EXPIRED 仍要求先有取消首因；Worker成功、Worker错误、抢占、Decode代际退出
  无取消时的结果不变。退休默认文案、Worker来源文案及FAILURE响应备用文本逐分支对照。
- 无新增类型、字段、锁或异步边界。sync生产Java 27,536 → 27,528，净减8行。
- Request/Preemption/Deadline/Expiration 定向247/247通过：
  `/tmp/flexlb-terminal-decision-tests.log`；`git diff --check`通过。
- 基线 `/tmp/flexlb-before-terminal-decision.tar.gz`；diff `/tmp/flexlb-terminal-decision.diff`。
- 本轮只跑本地回归。远端仍为27,589行版本，burst P99 705ms失败尚未解决。

## 2026-09-25：投递事务直接持有资源，删除中间所有者

- sync 生产 Java：27,528 → 27,463（净减 65 行；距 25,000 尚差 2,463）。
- 删除 `PrefillAdmissionResources.CommittedAdmissionOwner`，包括重复冻结的 Member[]、
  closed、handoff 绑定步骤和中间转交查找。Batch/Route 事务保留原有成员序列，直接持有
  `PrefillState.CommittedHandoff`；DIRECT 在调用栈内持有。
- `Member.transferToEndpoint/close` 同一 monitor 保护一次 Decode permit 转交或释放；
  TRANSFERRED/OWNERSHIP_LOST/ENDPOINT_RETIRED 消耗本地责任，dispatch 抛异常时保留给 close。
- `closeCommitted` 无独立状态，索引遍历成员并隔离释放失败，finally 关闭 generation handoff。
  保留先成员、后 generation 的关闭顺序；每条路径已有事务阶段确定唯一执行者。
- Batch 保留原下标与 exact item 校验。Route 使用同一 PreparedRoute 绑定的 item/member，
  DIRECT 使用当前 item 创建的 member，删除跨对象下标身份映射。
- 新测试：首成员 release 抛错仍释放 sibling，再关闭 handoff；重复 member.close 不重复释放。
  原五种 dispatch 结果、取消成员跳过与部分成员发送契约保留。
- 定向 280/280：`/tmp/flexlb-transaction-ownership-tests.log`。
- API 全量 reactor：1,621 条记录，1,620 通过、1 跳过，0 失败/错误：
  `/tmp/flexlb-transaction-ownership-full-tests.log`。
- cleanup 改为索引循环后的最终定向 46/46：`/tmp/flexlb-transaction-ownership-final-tests.log`。
- 三个 subagent 分别审查请求并发、设计/资源、测试；均无阻断。并发依据：Route 的
  takeCommitted 在同步区唯一消费 handoff；Batch 在 COMMITTED 同步仲裁 abort/handoff，
  SUBMITTED 后唯一 delivery 执行者推进；DIRECT 同调用栈 finally。未新增反向锁链。
- 基线 `/tmp/flexlb-before-transaction-ownership.tar.gz`；增量 diff
  `/tmp/flexlb-transaction-ownership.diff`。没有重置已有工作区修改。
- mock-engine test-compile 通过（未执行其测试）：`/tmp/flexlb-transaction-ownership-mock-compile.log`。
- 远端 API 性能 16 项中 15 通过，唯一失败 burst Master P99 735 ms > 250 ms；吞吐达标，
  环境无阻碍。详见 remote-performance-2026-09-24.md 本轮记录；本目标仍未完成。

## 2026-09-25：删除测试专用队列快照，合并 ACTIVE 终结

- sync 生产 Java：27,463 → 27,401，净减 62 行；目标仍差 2,401 行。
- 删除 WorkerBatcher.QueueSnapshot、WorkerBatcher.captureQueueSnapshot、PrefillEndpoint
  的对应转发方法。仅测试消费，生产选路/投影仍使用原有 PrefillActiveIndex.Capture。
  测试直接在 ownershipLock 下捕获既有 Capture，不新增生产兼容层。
- PrefillState 合并 ACTIVE 与 ACTIVE Route 终结入口，共用精确 entry 查找、索引撤回、
  request 删除、mutation 更新；nonnull Route lease 先校验 owner/identity/OPEN，再关闭。
  null 分支保留原 OPEN Batch lease；rollbackFreshActiveUnderLock 删除重复选择分支。
- 所有生产、测试与 SnapshotBench 调用迁移；旧入口无 Java 引用残留。
- 定向最终 362/362 通过：`/tmp/flexlb-queue-boundary-final-tests.log`，覆盖 WorkerBatcher、
  Prefill、Delivery、Admission、EndpointCleanup。新增 wrongRouteLeaseCannotDetachEitherQueuedOwner
  验证错 lease 不撤任一 owner；原快照测试增加空/入队前捕获的不变性断言。
- 三个 subagent 分别审查状态/设计、并发/ABA、测试，未发现阻断；测试审查建议已补
  ACTIVE 撤回后 OPEN Batch lease 继续由准备事务持有的直接用例。
- 原停止 detach→callback→ack 协议保留；其保留回调失败后的 canonical entry 供退休重放。
  runtimeState/stopped 含义不等价，未合并。入队 reservation 失败仍只回滚新 ACTIVE 项。
- 基线 `/tmp/flexlb-before-queue-snapshot-removal.tar.gz` 与
  `/tmp/flexlb-before-active-terminal-merge.tar.gz`；diff `/tmp/flexlb-queue-boundary.diff`。
- 本轮属局部删除/分支合并，仅本地回归，未同步远端。最新远端仍是 27,463 行版，
  burst Master P99 735 ms > 250 ms；不得将本轮本地测试称为性能验收。
- 补测后的 PrefillStateSnapshotTest 27/27 通过：`/tmp/flexlb-active-terminal-lease-tests.log`。
  ACTIVE 移除后队列/请求数归零，但 batch lease 计数保持 1、retirement drain 未触发；
  lease.close 两次后计数归零、容量通知与 generation drain 都仅发生一次。

## 2026-09-25：DefaultRouter 删除失效配置回退与清理包装

- sync 生产 Java：27,401 → 27,362，净减 39 行；目标仍差 2,362 行。
- BalanceContext.config 已是 final 且构造器 requireNonNull，无 setConfig。删除 Router 的
  ConfigService 字段、构造注入和 null-config 重新加载分支，选路只消费请求配置。
- 删除单用途 validateRequest 包装，入口仍对 null Context/Request 返回 INVALID_REQUEST。
  statuses 收集与成功响应构造合一，成员顺序与 pin 移交时点不变。
- 删除 closeSelection / closeSelections 包装，复用既有 Failures.close/append/rethrow。
  PinnedRouting 仍 TWR 逆序逐项关闭；append 失败的未入列表 selection 单独关闭并保留首因。
- sync/API/mock-engine 的 router 构造和测试子类全部迁移，无兼容构造器残留。
- 定向 sync 126 + API 21 项通过：`/tmp/flexlb-router-cleanup-tests.log`。
  mock-engine test-compile 通过：`/tmp/flexlb-router-cleanup-mock-compile.log`（未跑该模块测试）。
- 三个 subagent review 无阻断；按测试审查建议补多个 pin 关闭均失败时的逆序与 nested
  suppressed 首因测试。未修改任何错误码、资源交接或性能阈值。
- 本轮检查了抢占 target 预扫/claim 两遍读取，但未合并：目标缺失在全部认领前返回，
  claim 冲突则撤销已经取得的能力，合并需要改变 preflight 副作用与失败分类，尚无净删证据。
- 基线 `/tmp/flexlb-before-router-cleanup.tar.gz`，diff `/tmp/flexlb-router-cleanup.diff`。
  本轮局部改动仅本地验证；远端仍是 27,463 行版本，性能目标未通过。
- 最终 DefaultRouterTest 29/29 通过：`/tmp/flexlb-router-cleanup-final-tests.log`，新增用例
  证明 selector 主异常不变、两个 selection 逆序关闭、两个清理错误按原嵌套 suppressed 保留。

## 2026-09-25：全局队列 Future 单一来源与发送优先级冻结

- sync 生产 Java：27,362 → 27,350，净减 12 行；目标仍差 2,350 行。
- GlobalQueueEntry 删除重复 future 字段/构造参数；GQC.offer 不再接收独立 future，
  从已注册 Context 读取，registered 索引、控制回调、撤回与异步 claim 仍使用同一引用。
  BalanceContext 注册后拒绝替换 Future，生产 register 后才入队，因此身份与可见性边界不变。
- DefaultBatchDispatcher 删除可变 Request 优先级读取及无效 null 分支，使用 item.priority()
  的冻结 DecodeBinding 值。测试冻结60后修改原 Request 为7，真实发送 PB 仍为60；无优先级
  的0 sentinel保持。正常 API 入口优先级已归一化；手工0值的全局队列默认50仍属既有行为。
- 初轮328项中4失败来自 RequestSchedulerTest.Fixture 的 mock Context 没有 getFuture stub，
  与真实 register 会保存 Future 的契约不一致。补充同一 Future 返回，未增加生产兼容分支。
- 三个 subagent review 均无阻断：核对旧 requestId/Context/Future 隔离、撤回/关闭、
  全局队列发布 happens-before 与优先级冻结测试。
- 基线 `/tmp/flexlb-before-request-facts.tar.gz`；diff `/tmp/flexlb-request-facts.diff`。
  本轮局部改动只本地回归，远端仍为27,463行版，性能未重新验收。
- 修正fixture后 API全量reactor：1,624条记录，1,623通过、1跳过、0失败/错误；
  `/tmp/flexlb-request-facts-full-tests.log`。common225/cache37/grpc15/sync1172/API175。

## 2026-09-25：SchedulerRuntime 删除测试专用入口与重复遍历

- sync 生产 Java：27,350 → 27,301，净减49行，目标仍差2,301行。
- 删除四参数 no-op dispatcher 排空构造、私有 Runnable 适配构造；唯一 @Autowired
  构造显式持有 DefaultBatchDispatcher。测试以 mock dispatcher 表明边界，不在生产提供绕过入口。
- Prefill/Decode 两份重复快照、循环、叶子错误隔离逻辑合为 reportEndpoints；两角色报告次序、
  RuntimeException捕获范围和日志Throwable隔离保持。无新状态、DTO或中间服务。
- 关停顺序完全保留：beginShutdown、placement、admission、endpoint、dispatcher、expiration、
  continuations、outstanding、finally publisher；不改变beginShutdown唯一认领。
- 定向 sync47/API24 全通过：`/tmp/flexlb-runtime-cleanup-tests.log`。
- 新增测试 Prefill snapshot 抛错，Decode batch metric 再抛错，Decode admission metric 仍执行；
  RequestOrchestratorsTest 最终10/10：`/tmp/flexlb-runtime-cleanup-final-tests.log`。
- 三个subagent review无阻断；确认泛型方法只抽取相同控制流，未扩大异常隔离范围。
- 基线 `/tmp/flexlb-before-runtime-cleanup.tar.gz`，diff `/tmp/flexlb-runtime-cleanup.diff`。
- 本轮只本地回归，没有同步远端；最新远端性能仍为27,463行版本、burst P99 735ms>250ms。
- mock-engine test-compile通过（未运行该模块测试）：`/tmp/flexlb-runtime-cleanup-mock-compile.log`。


## 2026-09-25：原子入队收拢到 PrefillState

- 生产 Java 27,301 → 27,258，净减43行，距目标2,258行；没有新增状态字段或类。
- PrefillState.enqueueForDeliveryUnderLock 完成 ACTIVE 发布、退休检查、NON_BATCH lease
  申请与绑定，失败 finally 使用 exact lease 撤销新 entry。WorkerBatcher 只检查止收和成功唤醒。
- 删除 WorkerBatcher.rollbackFreshActiveUnderLock、PrefillEndpoint.reserveRouteOwnership；
  reserveRoute 改锁内精确操作，删除重入锁、结果分配及当前唯一生产调用不可达的拒绝分支。
- bindPublishedRouteReservation 改 public 供跨包账本使用；其他精确资源交接不变。
- 定向回归最终243/243通过，0失败/错误/跳过：`/tmp/flexlb-atomic-enqueue-final-tests.log`。
  覆盖 Worker/Prefill、队列投递、Batch/Route策略、Endpoint清理和Direct准入。
- 新增退休交错：在 ACTIVE 已发布时关闭退休门槛，offer 拒绝且账本/索引归零、无 item lease；
  原故障用例改在绑定时抛错，验证只撤销新请求、旧请求不受影响、容量可再次使用。
- 中途两次 testCompile 失败分别为测试 helper 缺引用、跨包访问非公开方法，已修正后重跑。
  审查期间把退休检查位置保持在 ACTIVE 后；最终243项执行的是修正后的源码。
- 三个 subagent 对最终增量复核无阻断：并发/所有权、结构、测试边界。
- 基线 `/tmp/flexlb-before-atomic-enqueue.tar.gz`，diff `/tmp/flexlb-atomic-enqueue.diff`。
- 本轮小改动只做本地回归；未同步远端。最新远端仍27,463行版本，burst P99 735ms>250ms，
  总体性能和25,000行目标均未验收。


## 2026-09-25：排队抢占复用原子入队和终结

- 生产 Java 27,258 → 27,222，净减36行；距离25,000行仍差2,222。
- PrefillState.replaceQueuedRoutesUnderLock 独占选择、取句柄、入队、终结与失败恢复。
  WorkerBatcher 删除 replaceQueuedRequestsUnderLock，仅保留唤醒和锁外事件通知。
- 删除原抢占路径单独的 RequestEntry/RouteReservation 构造、索引/目录写入回滚和删除流程；
  分别复用 enqueueForDeliveryUnderLock、terminalizeActiveUnderLock。选择器改 private，
  删除不再存在外部输入时的名单重复/长度检查，保留实际句柄身份校验。
- 新请求绑定返回 false 或抛异常均通过共享入队 finally 回滚，再由抢占 finally 恢复 victims。
  两个参数回归验证原队列/计数不变、无抢占事件、重试成功及迟到 victim 清理隔离。
- 245项定向回归全部通过：`/tmp/flexlb-queued-replacement-final-tests.log`。
- 三个subagent未发现生产阻断；测试review提出多victim取句柄前缀恢复缺口，已补测试。
- 基线 `/tmp/flexlb-before-queued-replacement.tar.gz`，diff `/tmp/flexlb-queued-replacement.diff`。
- 本轮局部重构只本地验证，远端未更新；此前burst P99 735ms>250ms的未达标结论仍有效。
- 多victim前缀恢复用例已通过：最终 WorkerBatcherRequestCapacityTest 16/16，
  `/tmp/flexlb-queued-replacement-prefix-tests.log`。失败后同一incoming可重试替换两victim。
  初版在解锁后用 endpoint.removeQueued 清理新请求，与后台worker处理竞争而断言失败；
  本用例直接调用账本替换，已将结尾精确终结断言保持在同一锁内，避免混入运行层竞态。
  原offer级用例仍独立覆盖运行层通知与迟到清理。


## 2026-09-25：Decode 退休提交收拢

- 生产 Java 27,222 → 27,198，净减24行；距25,000行目标仍差2,198行。
- 合并仅单调用的 retireGenerationOwnershipLocked 到锁内 retire，删除 addRetiredOwner
  转发方法。删除快照前的 permit 标记遍历，在已有 permit 清理遍历中完成退休标记。
- 先冻结精确 owner 去重/排序结果，再改变 permit 与账本；迟到 dispatch/release 使用的
  retiredByEndpoint 保留，所有观察与修改仍受同一 admissionLock 保护。
- 230项 Decode/Endpoint清理/资源计数定向回归全部通过：
  `/tmp/flexlb-decode-retirement-tests.log`。
- 新增 failedRetirementSnapshotDoesNotInvalidateTheLiveDispatchPermit：在读取 generation
  构造快照处抛错，验证版本不变、permit 仍实际释放容量；随后重新获得 permit 并成功退休，
  精确 owner 返回、容量归零、迟到释放与重复退休正常。最终测试日志：
  `/tmp/flexlb-decode-retirement-final-tests.log`。
- 三个subagent review无阻断。未删除必要的资源claim/终态分类，没有新增状态或包装层。
- 基线 `/tmp/flexlb-before-decode-retirement.tar.gz`，diff `/tmp/flexlb-decode-retirement.diff`。
- 本轮小改动只本地回归，远端未更新；整体行数与性能目标均未达成。


## 2026-09-25：删除选路判空触发的完整请求列表物化

- PrefillEndpoint 在 ownershipLock 内直接读取 activeIndex.isEmpty；此前为判空调用
  Snapshot.activeItems 会分配完整请求列表。冻结Capture与锁外预测物化保持不变。
- 删除 Snapshot.activeItems getter，测试直接使用 active().items()。总行数27,198→27,194。
- 219项快照/投影/Worker/选路回归通过：`/tmp/flexlb-projection-empty-tests.log`。
- 当前累计变更API全量reactor：1630项记录、1629通过、1跳过、0失败/错误；
  common225/cache37/grpc15/sync1178/API175，日志 `/tmp/flexlb-current-ownership-full-tests.log`。
- 三个subagent review无阻断。曾建议删除projectedItems缓存，经复核List.copyOf可复用现有
  JDK不可变列表且Capture跨ownership版本复用，已撤销该建议，未进行此修改。
- 基线 `/tmp/flexlb-before-projection-empty-check.tar.gz`；diff `/tmp/flexlb-projection-empty.diff`。
- 当前27194行版已经同步指定目录并在luoli_gpu跑API性能。首次改后burst P99 862ms超标，
  1P/2D 10k吞吐8476.1低于8500，性能未通过；完整前后及复跑证据见remote-performance文档。
- 同源码性能复跑16项15通过1失败，矩阵恢复通过，burst P99仍771ms>250ms；
  归档 `/tmp/flexlb-projection-empty-repeat-results.tar.gz`。保留两次结果，不归因单次差异。

## 2026-09-25：Batch 首成员准备收拢与类结构复核

- BatchDeliveryStrategy 删除 prepareAdmission、blocked 工厂和事务 prefill 字段；首成员资源准备与 append 使用同一次 prepareDispatch 请求锁，batchId 在初始化时一次赋值。
- 初始化失败仍保持主异常与 submission.close 的 suppressed 异常；finally 只回收事务仍拥有的资源。Route 的已转移前缀补偿保留。
- 生产 Java 27,194 → 27,153（净减 41）。基线 `/tmp/flexlb-before-batch-initialization.tar.gz`。
- 初次专项 151 个测试通过，日志 `/tmp/flexlb-batch-initialization-tests.log`。
- 新增真实 Scheduler 的准备/取消交错测试。首次编译使用不存在的取消枚举，已修正为 CLIENT_CANCELLED；随后断言错误地把接受取消视为完成取消，按 cancelRequest 的实际契约修正为 CANCEL_REQUESTED。最终两个测试类 36/36 通过，日志 `/tmp/flexlb-batch-initialization-final-tests.log`；仍断言取消后不能 claimDelivery，submission/reservation 各关闭一次且不发送。
- 三个专项 agent 完成生产代码并发、设计、测试复核，无阻断发现。测试 reviewer 指出 latch 只能证明取消线程已开始，锁内 holdsLock 与完成顺序共同验证边界。
- 类图审查新增当前职责、规模与改进顺序，见 `class-structure-review.md` 开头。当前改动未重跑远端性能；最近远端 27,194 行版本 burst Master P99 771 ms，仍未达到 250 ms。

## 2026-09-25：排队撤回归入 PrefillState

- WorkerBatcher 三处调用统一进入 PrefillState.removeQueuedUnderLock，删除 removeTerminalActiveUnderLock 与 restoreRouteReservation。取精确句柄、终结及失败 finally 恢复顺序不变；Batch OPEN lease 仍由事务关闭。
- 生产 Java 27,153 → 27,140，净减13行。此次主要收拢职责和删除恢复辅助函数，未删除新的状态机。
- 基线 `/tmp/flexlb-before-active-removal.tar.gz`；最终diff `/tmp/flexlb-active-removal.diff`。
- 本地专项134/134通过，日志 `/tmp/flexlb-active-removal-tests.log`。三个 subagent 分别复核并发/资源归属、设计/停止协议、测试覆盖，均无阻断；现有精确身份、错 lease、Batch OPEN lease 与迟到 victim 测试足够支持本次等价改动，无新增镜像测试。
- 停止 detach→callback→ack 不能并入普通撤回：callback 失败需要保留 canonical owner 供退休重放。锁外 Route lease.close 也不能并入持锁 detach。
- 静态唯一方法名扫描所发现六项均为配置Bean或生命周期/定时入口，未仅因无普通Java调用就删除。
- 本轮小改动仅本地回归，未同步远端；25,000行及性能门槛仍未达成。

## 2026-09-25：Decode 退休复用终态 reducer

- RequestScheduler.applyDecodeRetirementLocked 保留精确 endpoint/reservation 校验，调用 processRequestEndLocked；删除重复的 cleanup 结算、routing 暂存与普通终结流程。
- 共同入口保留退休专用抢占分支，放在 isFinished 检查前：记录事件、结束 claim、detach、选择原退休事件的结果并锁外通知。退休属于 authoritativeWorker，FINALIZING 的 cleanup 仍可被结算。
- 净减16行，生产27,140 → 27,124。基线 `/tmp/flexlb-before-retirement-reducer.tar.gz`；最终diff `/tmp/flexlb-retirement-reducer.diff`。
- 初始专项263/263通过，日志 `/tmp/flexlb-retirement-reducer-tests.log`。
- 扩展 DeliverySettlementTest 为 preempting×retirement 四变体：Prefill cleanup 失败阻塞时，错误 Decode token 不能结束 claim，精确退休不能跳过尚未完成的 Prefill cleanup，后续重试完成且保留原响应。最终该类36/36通过，日志 `/tmp/flexlb-retirement-cleanup-tests.log`。
- 扩展 RequestSlotTerminalSettlementTest 的 admission 持有场景为 Worker terminal/退休两变体，验证关闭 admission 后重放仍保留首个取消原因。该类15/15通过，日志 `/tmp/flexlb-retirement-admission-tests.log`。
- 三个 subagent 审查生产并发、设计/重放、测试覆盖，无未解决阻断。测试审查起初误读 authoritativeWorker 的退休分支，复核源码后撤回误报；抢占信号审查的疑似 ACK 漏通知未证明生产可达，未据此改代码。
- finalizationEffects 的额外 signal 不删除：它通知已经 detach 的 claim，TerminalAction 不再持有该 claim。
- 本轮小改动仅本地验证，无远端环境阻碍；未重跑性能。总行数和性能目标尚未完成。

## 2026-09-25：退休暂存事实与派生访问器清理

- PendingPrefillRetirement 的 outcome/response 替换为冻结 detail，字段净减一个；采用退休终态时才构造结果，取消优先和精确路由身份不变。此部分行数不变。
- 删除无生产调用的 Candidate.requiredProjectedTtftMs 与 DecodeRoutingView.engineFacingKvAvailable；同步迁移测试及 PlanningBench/SnapshotBench。未知 TTFT 的工具异常类型改为 OptionalLong.orElseThrow 的 NoSuchElementException；正常输入与生产协议不变。
- 生产27,124 → 27,115，净减9行。基线 `/tmp/flexlb-before-retirement-facts.tar.gz`（src目录）；最终diff `/tmp/flexlb-retirement-facts.diff` 含工具的精确单行替换。
- 退休专项184/184通过 `/tmp/flexlb-retirement-facts-tests.log`；派生访问器专项77/77通过 `/tmp/flexlb-derived-accessors-tests.log`。
- 两个基准工具通过 javac -proc:none 编译，类路径来自当前 DeliverySettlementTest surefire java.class.path。未运行微基准，无新的性能结论。
- 三个 subagent 完成并发、设计、测试review，无阻断；设计review明确指出工具未知值异常类型变化。
- 静态扫描未直接删除 DecodeEndpoint.reserveUnqueued：测试构造的 engine-facing shadow 与真实 dispatch 后 engineLifecycleOwned 有差异，迁移必须先保证测试仍表达原资源归属。expireInactiveRequest 的测试同步行为也不同于异步 deadline 入口，未盲删。
- 当前距25,000行还差2,115；最近远端性能结论仍未达标，本轮无环境阻碍且未重跑远端。

## 2026-09-25：当前版本整体回归和 burst 定位

生产保持27,115行，本轮未修改默认配置或生产代码。本地完整API依赖回归1634项记录、1633通过、1跳过，日志 `/tmp/flexlb-27115-full-tests.log`。远端同步486个文件校验通过后，同源依次运行默认256/32/8 planner的burst，Master P99分别889/723/643ms，全部未达到250ms；详见 remote-performance-2026-09-24.md 最新小节。只有定位实验，不是完整性能矩阵验收。无环境阻碍。

## 2026-09-25：Worker 有界前缀快照

- snapshotActiveQueue(maxRequests)在同一ownershipLock中只复制排序后的最多maxRequests项，保留版本与List.copyOf不可变性。原GroupPlanner、过期扫描及capacityfeasibleprefix均不访问更后的成员；captureHead的小队列最老年龄扫描不变。
- Capture.items失去生产热路径调用后，删除volatile items缓存与同步懒构造，观察接口从不可变entries直接生成不可变请求列表。projectedItems缓存未改。
- 净减7行：27,115 → 27,108。基线 `/tmp/flexlb-before-worker-prefix.tar.gz`；最终diff `/tmp/flexlb-worker-prefix.diff`。
- 本地120/120专项通过 `/tmp/flexlb-worker-prefix-tests.log`。三个subagent分别验证并发快照、真实规划边界及测试覆盖，无阻断，无新增镜像测试。
- 默认256 planner、2GiB堆远端burst：Master QPS5412.8、P99 836ms，失败于250ms门槛。与前一次默认889ms均为单样本，不能声称稳定提速。原Capture可复用稳定队列快照，新逻辑每次复制一个批次，有取舍；确定改善的是复制范围和缓存状态数量。
- 当前距25,000行还差2,108；性能目标未完成。最近完整本地API回归仍是27,115行版本的1633通过/1跳过，本轮为上述专项。


## 2026-09-25：预测边界统一与类结构复核

RouteTimelineProjector.PredictionBoundary 的 batchDurationMs 复用已有
batchPlanningDurationMs 与 committedGroupDurationMs；singleton 预测与整数转换合为一个入口，
删除仅剩一次调用的特征 helper 和空构造器。缓存身份、成功写入时点、异常分类、ceil 与
Long 饱和转换不变。生产代码27,108 → 27,083，净减25行。

原专项128/128通过（`/tmp/flexlb-prediction-flow-tests.log`）；删除空构造器后的最终源码
再跑 RouteProjectionTest,*Prediction*Test，40/40通过
（`/tmp/flexlb-class-review-final-tests.log`）。三个subagent均未发现此小改的阻断。
本轮未重新跑远端性能；最新远端27,108版本burst P99 836ms仍未达标。

随后按用户要求重新梳理类图，结论写入class-structure-review.md开头：路由提交编排、
请求输入冻结、Worker账本封装和抢占双侧状态。文档中的目标接口尚未实施，
不计为减行或性能收益。当前距25,000行仍差2,083行。


## 2026-09-25：victim 拥有自己的 Cancel ACK

DecodePreemptionCoordinator 将每个 ACK future 收进现有 ClaimedVictim，删除独立
acknowledgements 列表与 handleAcknowledgements 参数、command.victims/claims/ACK
三者按下标读取的逻辑。全部 terminal observer 仍在第一个 Cancel 前安装，
allOf 在所有 ACK 字段赋值后建立。请求侧取消协议与 endpoint 资源账本不变。

生产27,083 → 27,073，净减10行；删除一份平行关联容器，增加已有 victim 上的
一个 ACK 引用，不新增类。基线 `/tmp/flexlb-before-victim-ack.tar.gz`，
精确生产差异 `/tmp/flexlb-victim-ack.diff`。

- 专项79/79通过：`/tmp/flexlb-victim-ack-tests.log`。
- 新增双victim乱序ACK测试3项，分别验证FAILED/NOT_FOUND/REQUEST_FENCED；
  ACK完成顺序不能改变作用对象，全ACK屏障和第一victim终态证明仍须成立。
  CoordinatorTest共8/8通过：`/tmp/flexlb-victim-ack-order-tests.log`。
- 并发与结构review均无阻断；测试review指出的乱序与不同ACK组合已补上。
- 小范围对象聚合调整仅跑本地，未重新跑远端性能，既有836ms失败结果仍有效。

路由提交消环尚未实施：只搬方法仍会留下 DIRECT 资源取得输出和清理状态，
必须结合具体所有权交接收拢；不能把搬代码计作向25,000行目标的减行收益。
当前目标还差2,073行，功能全量与性能最终验收仍待完成。


## 2026-09-25: Delivery prediction interface convergence

Removed DeliveryStrategy.projectGroupDurationMs and the unused Batch implementation.
WorkerBatcher uses newGroupPredictor; Route now returns the same per-prefix summation
inside its callback. Evaluation order, floating-point summation and validation remain
unchanged. Test implementations and mocks now implement/delegate the production entry.
Production lines: 27,073 -> 27,052 (-21), remaining 2,052 to the target.
Baseline: /tmp/flexlb-before-prediction-interface.tar.gz.
Diff: /tmp/flexlb-prediction-interface.diff.
Full API reactor regression: 1,637 records, 1,636 passed, 1 skipped, no failures/errors.
Log: /tmp/flexlb-prediction-interface-full-tests.log.
Final import cleanup verified by scoped regression:
/tmp/flexlb-prediction-interface-final-tests.log. Three reviews found no blockers.
No remote performance rerun for this equivalent interface change; previous 836ms
burst P99 remains above the 250ms gate. External custom DeliveryStrategy implementations,
if any, must implement newGroupPredictor; all repository implementations were checked.

Broader review did not establish a safe large deletion: deferred terminal and Prefill
retirement facts can coexist; endpoint attempt/request indexes protect different
identities; DispatchGate protects synchronous callbacks before transport acceptance.
The proposed deletion of shutdownAndAwait was retracted after finding the production
method reference SchedulerRuntime dispatcher::shutdownAndAwait. Request input freezing
remains unimplemented pending explicit retry/configuration semantics; live worker
capacity and cache observations must not become cached request facts.


## 2026-09-25：路由提交编排归入 RequestScheduler

QUEUE 调用 Scheduler.enqueueRoute，DIRECT 调用 Scheduler.commitDirectRoute；
RouteAdmission 不再接收/调用 Scheduler，资源操作 offerQueued、reserveDecode、
blockOnPrefill 仍封装 pin/reservation/阻塞版本事实。DIRECT 的单调用
commitImmediateRoute 合入主流程；member、handoff、routeCommit、reservation 的
关闭顺序不变。共享 commitRoute(BooleanSupplier) 仍保护请求绑定与锁外发布，
PrefillReservationAttempt 仍保存 DIRECT 准入结果供提交与失败回收。

生产27,052 → 27,039，净减13行；没有新增状态或类。大部分属于编排职责迁移，
不计作大规模删除，也不声称消除了所有 publication 间接调用。
基线 /tmp/flexlb-before-route-orchestration.tar.gz；最终差异
/tmp/flexlb-route-orchestration.diff。三个review均未发现本轮新增阻断。

验证记录：
- 首次定向失败4项：测试schedulerMock默认跳过新移入的commitDirectRoute，
  已改为调用真实实现，资源取得和交接仍由测试覆盖。
- 定向123/123通过：/tmp/flexlb-route-orchestration-tests.log。
- 首次全量GlobalQueueProgressTest一项latch超时：迁移后多个规划线程动态
  stub同一个Scheduler mock。改为fixture启动前安装固定Answer；每个route仍
  保存独立返回item，未放宽latch和积压/槽位补入断言。测试review确认等价。
  失败日志保留于 /tmp/flexlb-route-orchestration-full-tests.log。
- 最终全量：1637项记录，1636通过、1跳过，0失败/错误；
  /tmp/flexlb-route-orchestration-final-full-tests.log。

本轮没有改变规划算法、线程模型或容量协议，使用本地全量验证，未重跑远端性能。
最新远端burst P99 836ms仍未达250ms门槛，25,000行目标还差2,039行。
并发review还提醒原publication回调内markCommitted关闭pin的异常边界：若队列已
发布而pin关闭随后抛错，旧commitRoute的统一catch可能回滚请求绑定。该疑点本轮
没有复现实验，未新增也未解决，需后续结合真实generation continuation确认。


## 2026-09-25：删除生产不可达的第二套 Batch 规划器

重新追踪实际入口，修正此前“非增量模型需要BatchPlanning”的判断：
Evaluator.newBatchPrediction 默认返回全批重算适配器；FormulaPredictor 在公式无法
增量聚合时也使用该默认实现。唯一生产 Predictions 实现 PredictionBoundary 始终
返回追加会话，因此 BatchProjection 的 null 分支仅由旧测试替身触发。

删除 BatchPlanning 类、PLANNING ThreadLocal 和 null 分支；Predictions 只要求
newBatchPrediction，不再暴露 batchPlanningDurationMs。后者失去共享调用后，
Boundary.batchDurationMs 直接使用原 predictCommittedBatchMs；异常分类及数值
校验/取整相同。AppendPlanning 与 BatchService 的前缀缓存、模型隔离保持原路径。

生产27,039 → 26,997，净减42行。基线
/tmp/flexlb-before-projection-fallback.tar.gz，差异
/tmp/flexlb-projection-fallback.diff。
首次与最终定向测试均132/132通过：
/tmp/flexlb-projection-fallback-tests.log、
/tmp/flexlb-projection-fallback-final-tests.log。
IncrementalPredictionTest 的 full evaluator 只实现 estimate/predictBatchMs，实际使用
默认适配器；与公式模型进行150组过期/优先级/probe位置投影差分，继续覆盖全批模型。
测试还覆盖前缀位精度、越界预算的前一个前缀、重排、模型替换和并发会话。
公开 Predictions 接口收窄：仓外直接实现者须提供追加会话；本仓实现已全部检查。
小范围删除生产死分支仅跑本地，未重跑远端性能；最新836ms仍未达标。
当前25,000行目标还差1,997行。

pin关闭疑点的补充调查：GenerationPin.close 经 HandoffPermit.close 释放计数；
最后一个permit只把退休任务提交给共享executor，closeEndpoint抛出的异常在任务内
捕获并记录，不回传到发布线程。正常路径没有找到之前假设的退休清理异常回传。
未为未证明可达的异常增加新的发布标志或补偿状态。

最终三个subagent review均无阻断；并发review已包含batchDurationMs单调用包装合并。


## 2026-09-25：删除一次性 service 查询的持久缓存层

生产唯一调用点 RouteTimelineProjector 每个 ready group 创建service后只查询一次：
probe在组内取该成员，否则取末成员。删除GroupService、BatchService、两份SERVICE
ThreadLocal和重复service入口；新completionOffsetMs直接接受items/index/predictions/planning。
Batch保持exact planning前缀命中与全批重算顺序，Route保持逐项saturatedAdd。
原服务缓存的重复查询能力无生产消费者，规划缓存仍保留。

生产26,997 → 26,914，净减83行，无新增类或状态。
基线 /tmp/flexlb-before-service-cursor.tar.gz；差异 /tmp/flexlb-service-cursor.diff。
专项139/139：/tmp/flexlb-service-cursor-tests.log。
含API全量1635项记录、1634通过、1跳过：/tmp/flexlb-service-cursor-full-tests.log。
服务游标memo专用断言已删除，改为每次精确前缀预测/不读suffix/bounds无模型调用的断言；
150组增量与全批投影差分、规划前缀复用与模型替换仍执行。三个review均无阻断。
公开DeliveryProjection接口和GroupService变更会影响仓外自定义实现；本仓无遗漏引用。
远端完整profile另记 remote-performance-2026-09-24.md。


### 选择提交：以事务阶段替代 Worker 重复标记

生产 26,914 → 26,899，净减15行；距离25,000还差1,899行。State只接收精确成员
校验和边界移除职责，Worker保留同锁提交编排。删除ownsCommitted与边界异常暂存。
Route.abort仅处理COMMITTED；未提交/部分转移由外层TWR close回滚。
实验中的State→Transaction调用及SelectionCommit结果对象经结构review后已删除。

三个review完成，无阻断。RouteDeliveryStrategyTest增强两种reservation失配下
commit失败→abort不提前释放→close清理的验证。最终专项255/255通过，日志
`/tmp/flexlb-selection-commit-final-tests.log`；过程中全量1634通过、1跳过，
日志`/tmp/flexlb-selection-commit-full-tests.log`（全量编译的是中间版本，
最终结构修正由上述专项覆盖）。末次仅恢复多行Java格式，git diff --check通过。
基线`/tmp/flexlb-before-selection-commit.tar.gz`。本轮未重新运行远端性能；
此前远端已恢复service-cursor版本且486文件SHA校验通过，不含本轮提交改动。


### Route规划游标只保存累计值

生产26,899→26,891，净减8行。GroupPlanning约定requiredThroughIndex不下降；
RouteTimelineProjector按min(probePosition,prefix.size-1)调用，GroupPlanner只追加，
超预算撤回末项后立即结束，不再请求较小索引。RouteCursor因此删去offsets数组、
ensureCapacity和through转发，保留累计durationMs、computedThrough和预测器。
新游标reset清零；预测抛错时累计值和已完成索引均不提前修改。
降序调用现显式拒绝，属于对已有SPI前置约束的检查，仓内无合法调用受影响。

专项75/75通过：投影、增量预测、组规划与Route投递；新增失败重试和饱和测试，
保留增长、仅计算probe前缀及下个队列重置测试。日志
`/tmp/flexlb-planning-scalar-tests.log`；生产基线`/tmp/flexlb-before-planning-scalar.java`。
本轮仅本地验证，未宣称性能提升或达标。

三个subagent分别复核调用契约、失败/并发状态、测试覆盖，均无阻断。


### 单请求Prefill账本消除重复时钟

RequestEntry.lastObservedAtMs与phaseBaseMs的全部写入均相同，删除前者，TTL与stats
读取后者；observeIndividualPhase复用已有individualRemaining。BatchWork的时钟
存在独立touch/re-predict/terminal用途，未合并。生产26,891→26,885，净减6行。
专项187/187通过，日志`/tmp/flexlb-individual-clock-tests.log`；
基线`/tmp/flexlb-before-individual-clock.java`。未运行远端性能，整体门槛仍未通过。

三个review无阻断；逐写入点证明单请求两个时钟恒等，未增加只镜像实现的测试。


### 投递结果直接返回后续动作

RequestScheduler.acceptDeliveryResult删除跨锁failed重复分支和临时结果转存；
失败直接生成后续动作，成功/不确定/失效claim各自按原条件返回。
投递失败与准备失败共用publishFailureAndCleanUp，继续隔离响应发布异常后清理。
生产26,885→26,879，净减6行，没有新增状态。专项217/217通过，
日志`/tmp/flexlb-delivery-effects-tests.log`，基线`/tmp/flexlb-before-delivery-effects.java`。
全局Plan的提交与异常关闭边界本轮未改，避免改变AdmissionHandle暂存事件重放时机。

三个review完成，无阻断；迟到拒绝、request ID重用与UNKNOWN确认路径保持原语义。


### 快照捕获时刻复用统一剩余工作量计算

WorkSnapshot两个无时间参数查询改为调用对应At(capturedAtMs)，删重复求和与
unknown分支；缓存、饱和规则和公开接口保留。相等时间先得到elapsed=0，
包括Long极值，不发生时间差溢出。生产26,879→26,873，净减6行。
专项119/119通过，三个review无阻断。日志`/tmp/flexlb-work-time-tests.log`，
基线`/tmp/flexlb-before-work-time.java`。未跑远端性能，未宣称达标。
