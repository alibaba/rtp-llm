# Decode 队头有限绕行

只在 PRIORITY 队列的队头因 Decode 容量不足而阻塞时尝试绕行。沿用原有选择、资源准备、提交和发布流程。

## WorkerBatcher 与 NON_BATCH

WorkerBatcher 是每个 Prefill 的本地队列执行器。三个配置轴独立：调度方式决定是否排队，决策方式决定一次选择几个请求，投递方式决定如何发布。

| 配置               | WorkerBatcher 的职责                               | 提交与发布                                                                     |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| DIRECT + NON_BATCH | 不参与                                             | Scheduler 直接提交并发布单个路由                                               |
| QUEUE + NON_BATCH  | 排序、SINGLE/FIXED_WINDOW 决策、容量等待、有限绕行 | 按请求占用 Prefill 额度，逐个发布路由；无 batchId、批次许可或 EnqueueBatch RPC |
| QUEUE + BATCH      | 同一队列决策流程                                   | 占用批次额度和发送许可，交给 Dispatcher 发送 EnqueueBatch、处理 ACK            |

FIXED_WINDOW + NON_BATCH 仍会等待决策窗口；预算按各请求预测之和计算，提交时保留每项预测。先 claim 全组，再逐个发布；失去 claim 的请求不计入后续累计预测。SINGLE + BATCH 仍发送一个成员的真实批次。

普通选择和绕行共用 `decideGroup → processSelection`。`bypassCandidates` 只确定允许绕过队头的候选前缀；提交成功才消耗额度。完整 BATCH 组复用分组预测，实际提交前缀缩短时重算。随后按投递方式进入 `publishRoutes` 或 `Dispatcher.deliverBatch`。

## 接口与状态

- `WorkerBatcher.canBypass(head, candidate)`：检查同优先级、不同 Decode 账本、请求仍在队列且未过期、队头额度未用。此接口只判断资格，不预留容量。
- `PrefillState.RequestEntry.bypassUsed`：唯一新增的持久状态。成功提交绕行决策组时，在原有队列锁内重新校验并置位；失败不消耗额度。唤醒、重试、新请求到达和临时队头变化均不重置它，队列项移除时一起释放。
- `CommitPlan.bypassedHead`：可选的原队头，用于提交时复验并消费额度。普通提交保持为空。

## 执行顺序

1. 先按原顺序尝试队头。只有它因 Decode 容量不足而失败，才检查后续候选。
2. 只跳过这一个队头，收集紧邻其后的合格前缀；遇到第二个不合格或阻塞请求就停止，不继续向后找。
3. 候选使用原有分组规则，检查 Prefill KV/计算容量，按投递方式申请 Decode 与 Prefill 容量；仅 BATCH 申请发送许可。机会性绕行不额外等待一次收集窗口。
4. 提交前检查队头是否已可重试；已恢复则放弃本次绕行。队列锁内再次检查资格，成功提交后消费一次额度。
5. 复用原有发布、失败回滚和容量等待。额度按一次成功提交的决策组消耗；SINGLE 是一个请求，NON_BATCH 的决策组不是远端批次。

## 行为边界

同一个持续排队的队头最多被一个成功提交的决策组绕过；提交后的路由发布失败或批次 RPC 失败也不返还额度。候选不使用队头所等待的 Decode 账本，但仍共享 Prefill，因此可能使原队头多等这一组请求。

队头恢复检查是无锁容量快照，与 Decode 恢复并非跨账本原子操作；并发恢复时，已经进入提交的绕行批次仍可能完成。新请求到达可唤醒尚有额度的 Decode 等待；额度用完后不会因继续到达而反复重试。

这是机会性的绕行：只订阅原队头的 Decode 容量事件。候选所在 Decode 单独恢复不会额外唤醒此队列，须等待原有事件或新请求到达；没有新增候选监听器、线程或后台任务。路由预测保留原有保守等待边界。

NON_BATCH 单项预测失败按成员归因：预算分组阶段记录失败成员的索引，只移除该请求，正常前缀下轮继续；准备阶段将失败转成该成员的 CapacityBoundary，正常前缀可一起提交。异常不持有请求或输入对象，不增加正常路径分配。BATCH 的整体预测失败仍按原有组首处理。

## 队列流程收敛验证

基于 `d661bb360e` 的隔离副本，在 `luoli_gpu`、JDK 21 中验证；未混入其他工作区改动。

- 完整 UT：3305 项，0 失败、0 错误、1 跳过；Mock Engine 400 项全部执行。
- 新增覆盖：NON_BATCH 的窗口与逐项预算、BATCH + SINGLE 的真实批次、完整/截短绕行预测、失效 QUEUED 提交，以及单项预测失败的准确归因。NON_BATCH 绕行测试从初始化就使用对应运行时配置。
- 两个 agent 复核了模式边界、热点、许可回收、读写与锁顺序。发现的单项预测归属问题已修复。全量 UT 另发现校验移动改变 handoff 与异常契约，已修正：QUEUED 保留 Endpoint 提前拒绝，BATCH 只在 State 校验；修正后重跑全量 UT 通过。
- 无新增生产类。WorkerBatcher 仍为 28 个显式方法，本轮改动的生产方法均不超过 100 行。未修改的 `stopAndDrain` 104 行、`PrefillEndpoint.reduceStatus` 115 行仍是既有超限项。
- 性能 21/25 通过、4 项失败，尚未验收。API 14/16、Sync 5/5；750P/750D 的 BATCH 0/2、NON_BATCH 2/2。使用 64g 堆、ExitOnOutOfMemoryError、Netty advanced；矩阵预热和测量各 10 秒，保留 98% 吞吐和 50ms P99 门槛。

| 750P/750D 模式       | 目标 QPS | 客户端 QPS | 客户端 P99 | 服务端 P99 | 结论            |
| -------------------- | -------: | ---------: | ---------: | ---------: | --------------- |
| BATCH + FIXED_WINDOW |     3000 |     2996.8 |   59.332ms |       13ms | 客户端 P99 超限 |
| BATCH + FIXED_WINDOW |    10000 |     9989.2 |  179.507ms |       28ms | 客户端 P99 超限 |
| NON_BATCH + SINGLE   |     3000 |     2999.4 |    3.144ms |        1ms | 通过            |
| NON_BATCH + SINGLE   |    10000 |     9994.8 |   18.018ms |        3ms | 通过            |

另外两项失败为 API 突发测试（服务端 P99 608ms）与 1P/2D、10000 QPS（服务端 P99 109ms）。本轮未重新压测基线，不据此宣称性能提升或没有回退。

## 初版绕行验证（历史结果）

基线为 `e2e129d4db3d2d1fd22a9509f721427511ca1983`。在只包含本轮 5 个 Java 文件改动的隔离副本中，进入 `luoli_gpu` 使用 JDK 21 验证；共享 proto 和 telemetry 测试数据来自同一基线。

- 完整 UT：3296 项，0 失败、0 错误、1 跳过，其中 Mock Engine 400 项全部执行。
- 回归覆盖：同优先级/不同 Decode、FIFO 禁止绕行、额度不可重置、SINGLE/NON_BATCH、Prefill KV 上限、预测失败归因、真实线程到达唤醒、队头恢复/变更，以及提交前取消队头或候选后的许可归还。
- 两个 agent 独立复核了许可回收、额度、等待唤醒和锁顺序。发现的预测异常归因和 Prefill 容量复验问题已修复并增加回归；复核未发现剩余确定问题，上述行为限制仍然保留。
- 外层显式方法（包含构造器）：WorkerBatcher 28 个，PrefillState 30 个；本轮修改的生产方法均不超过 100 行。未修改的 `WorkerBatcher.stopAndDrain` 为 104 行，是原有超限项。
- 完整性能门槛：20/25 通过、5 项失败，性能尚未验收。API 13/16，Sync 5/5；750P/750D、3000/10000 QPS 的 BATCH 为 0/2，NON_BATCH 为 2/2。使用 64g 堆、ExitOnOutOfMemoryError 和 Netty advanced 泄漏检测，未调整原有门槛。
