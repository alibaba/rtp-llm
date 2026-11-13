# QueuedRequestScheduler 最终交接与实施记录

## 2026-11-13：从决策输入重新核对调度状态

本节基于目录迁移后的当前实现：`balance/schedule/QueuedRequestScheduler`、
`SchedulingRequests`、`AbstractRequestScheduler`、`DefaultDecodePreemption`，以及
`balance/queue` 与 `worker/endpoint/CapacityNotifier`。下文区分源码事实、已有行为测试
和待验证的精简建议；后面的旧记录不能作为当前状态清单。
本轮修改目录开发规范并纠正 Scheduler 类说明，没有实施下列生命周期或索引重构。

### 决策最少需要知道什么

| 要做的决定                 | 最少输入                                                                                | 当前提供者                                                          |
| -------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 是否接收并保留一次排队登记 | Scheduler 是否停止；请求是否仍 OPEN、未取消、Future 未完成；本次登记是否仍有效          | Scheduler、RequestContext、registered                               |
| 下一次先选谁               | 优先级、原始 FIFO 序号、是否有未扫描/已唤醒条目                                         | SchedulingQueue                                                     |
| 能否开始新的选址任务       | 已提交但未收尾的选址任务数、并发上限；本条目是否已有任务或仍在等待                      | SchedulingRequests、CapacityWaitQueue                               |
| 能否选择并提交 Worker      | 请求输入、group、截止时间、当前选址编号、Endpoint 当前资源和代际                        | Context、RequestRouteSelector、Endpoint；资源账本不复制到 Scheduler |
| 失败后重试还是继续等       | RETRY/BLOCKED/结束结果；等待哪个 role/group；选择期间是否已有容量变化；是否等待抢占名额 | 选址结果、CapacityNotifier、两类等待索引                            |
| 能否结束或关闭             | 旧选址是否已返回并完成回滚；抢占结果是否已采用或释放；执行线程是否已退出                | 未结束任务记录、planner executor、抢占结果处理完成通知              |

因此，`SchedulingQueue + registered` 足以描述排队集合及其查找关系，但不能推出
取消后仍未返回的异步任务。一次请求的排队、选址尝试、远端资源和本地任务退出也不是同一个状态。

### 按实际流程看状态交接

1. `enqueue` 在 Scheduler 锁内先登记精确 entry，再入队；恢复请求保留原始序号。
   `registered` 与队列正常情况下包含相同 entry，登记先于入队也使入队异常后的请求仍可关闭。
2. driver 按可用名额扫描队列；在投递 executor **之前**记录选址任务，避免任务已被接纳但尚未运行时重复投递。
   扫描不把条目移出队列，选址中的条目仍是排序成员。
3. planner 通过 Context 开始精确尝试，Selector 取得资源，基类负责提交或回滚。
   成功/拒绝/关闭移除登记；BLOCKED 登记容量等待；RETRY 发布重试资格。
4. 抢占可在选址任务退出前启动，因而同一 entry 的 deciding 与 preemption 可同时存在。
   原选址任务先收尾，再订阅结果；结果处理后才释放抢占名额及关闭等待通知。
5. 取消移除登记和等待索引，但不提前返还旧任务名额。旧 entry 的回调不能删除同 Context 的替代登记。
6. 关闭先停止接纳并排空登记，等待 planner 退出，再等待抢占**结果处理**完成，最后关闭抢占组件。

### 请求登记与未结束任务：每份状态的原因

| 状态                          | 设置和清除                                                            | 根本原因、可否推导及判断                                                                                                                                                                                        |
| ----------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `registered: Context → entry` | enqueue 登记；remove 精确删除；关闭清空                               | 按 Context 找当前排队 entry，取消时无需扫描 N 个队列成员；精确 value 防止旧回调删新登记。它主要是同一排队集合的 O(1) 查找索引，不是另一套请求生命周期。可让队列承担查找与原子登记后删除外部索引，不能直接拿掉。 |
| `work: entry → Work`          | 投递选址或启动抢占前建立；两种工作都收尾后删除                        | 保留登记移除后仍可能交回资源的任务。按 entry 而非 Context 区分旧任务和替代登记。必要的是未结束工作事实，Map 和 Work 对象只是当前表示方式。                                                                      |
| `Work.deciding`               | startDecisions 设置；finishDecision 清除                              | 表示已占用选址名额且结果尚未处理完，包含 executor 排队、选址、响应或回滚；不等于 Context.selectionId 非零，也不等于 executor 正在用 CPU。                                                                       |
| `Work.preemption`             | tryStartPreemption 创建；finishPreemption 取出并清除；锁外完成 Future | 非空表示尚未处理完的抢占结果；Future 同时供 close 等待。上游返回的 Future 已完成时，晚到资源可能仍在回滚，不能用上游 Future 代替它。Future 对象可被总任务计数与条件等待替换，但不能删除等待清理结束的保证。     |
| `decisions`                   | startDecisions 加；finishDecision 减                                  | 由 work 中 deciding 条目数派生的 O(1) 计数。不是新业务事实；可改为扫描 work。设并发上限 P，work 中最多约 2P 个 entry，扫描成本是 O(P)，不是 O(N)。未测量前不能声称缓存计数一定更快。                            |
| `preemptions`                 | tryStartPreemption 加；finishPreemption 减                            | 同样是 work 中 preemption 非空数量的缓存；用途为抢占并发限制。不能用 work.size，因为一个 entry 可同时占两种名额。                                                                                               |
| `waitingRequestCount`         | 按需计算，不单独存储                                                  | registered.size 减去仍登记且有 work 的条目数；重叠任务只扣一次，取消的旧任务不扣替代登记。此处已避免再维护一个 waiting 计数。                                                                                   |
| `preemptionQuotaWaiters`      | 名额不足时加入；取得名额、移除请求或名额释放唤醒时删除                | 等待本地抢占名额的第二种唤醒条件。Worker 容量不变时也必须重试，因此不能只等 CapacityNotifier。它可与容量等待同时存在，表达 OR；可把事实放 entry 字段，但无集合索引时名额释放须扫描 N 个登记。                   |

Work 的空间随未结束工作数变化，当前 Work 只含一个布尔值和一个完成通知，
没有必要再为普通选址结果增加持久 DTO。把这两个字段放进 GlobalQueueEntry 可以少一次
Work 分配，却会让所有排队 entry 都携带执行状态，且关闭时仍需要查找取消后的任务；
只有比较实际排队规模、P、分配和清理路径后才能选择，搬字段本身没有减少独立事实。

### 排序与容量等待：哪些是性能索引，哪些保存独立事件

| 状态                                                     | 根本原因与维护位置                                                                                                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entry.context` / `routingGroup`                         | Context 引用提供精确身份；group 是 enqueue 时按请求输入和配置求出的路由约束，供重复选址与抢占恢复沿用。group 可重算，但删除缓存还需确认重试期间是否允许重新解释配置。           |
| `entry.sequence` / Context.placementSequence             | 队列排序编号与移除路由后恢复位置的编号。entry 会被替换，Context 保留编号跨越这次替换；不是可以无条件合并的同生命周期副本。                                                      |
| `entry.removed`、`previous/next`                         | 队列维护物理成员关系和双向链表，支持取消 O(1) 摘除、忽略旧 entry；不表示请求已结束。若改用 Context 查找索引推导 removed，队列会反向依赖 Scheduler，不能仅为了少一个字段这样做。 |
| `buckets.head/tail`、`order`                             | FIFO/优先级排序本体；常规加入和删除不扫描全队列。恢复原序号可能扫描一个 bucket。                                                                                                |
| `buckets.nextToScan`、`pendingPriorities`                | 保存有界向前扫描位置及哪些优先级还有未扫描条目。否则每次空闲名额出现都从高优先级队头重扫已阻塞请求。                                                                            |
| `readyRetries`                                           | 容量或抢占名额唤醒的请求可能位于扫描游标之后，需单独纳入有界扫描并与新请求统一排序；不是第二套成员登记。                                                                        |
| `size`、`priorityCounts`、`nextSequence`                 | 前两项是队列成员的计数缓存，用于 O(1) 长度与诊断；nextSequence 产生新的 FIFO 身份。计数可重算，序号产生器不能由当前最大值代替，否则队列空后可能重复使用旧编号。                 |
| `CapacityWaitQueue.domains`                              | role/group 到等待组的查找索引；物理通知带 endpoint，等待时归并为 role/group，下一次仍重新选址，不把唤醒源当成资源预留。                                                         |
| `Domain.entries`                                         | 同一容量条件下的等待成员，按全局 queue.order 排序。避免一次通知唤醒全部积压请求。                                                                                               |
| `membership`                                             | entry 到等待组的反向索引；精确取消和重新等待时 O(1) 定位，不必扫描所有组。                                                                                                      |
| `ready`                                                  | 已获得检查机会且没有在途试探的等待组，按组内首个 entry 排序；是可运行条件的索引，不重复承担请求登记。                                                                           |
| `Domain.retry`                                           | 本组已放行、尚未确认结果的一个试探请求；在排到 planner 之前就需占住机会，因此不能从 work.deciding 推出。成功或取消允许下一个尝试，仍阻塞则停止这一轮。                          |
| `Domain.available`                                       | 有无尚未消费的容量变化机会；试探中又到达通知时必须记住，即使该组暂时不能加入 ready。它不是 ready.contains(domain) 的同义布尔值。                                                |
| `CapacityNotifier.sequence/lastChanged/forgottenThrough` | 分别保存全局通知版本、role/group 最近版本、清理旧历史后的版本下界；park 比较选址前的版本，防止“通知先到、等待后登记”丢唤醒。下界防止历史删除后被误认作没有变化。                |

这些索引可以改成每次扫描 registered 并计算资格，但代价不只是 Map 数量变化：
会把通知 O(1) 定位和有界扫描变为 O(N) 查找，且仍需保留扫描轮次、一个组的试探请求、
试探期间新到事件等事实。本轮没有测量吞吐或分配，不把现有索引收益写成性能实测结论。

### Scheduler 自身控制与临时结果

| 状态                                                                  | 根本原因与判断                                                                                                                                                                                                       |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `closed`                                                              | 停止接纳和启动工作的跨线程事实；队列为空不能推出关闭。代码只 get/set，没有 CAS，AtomicBoolean 包装可简化为 volatile boolean，关闭事实仍须保留。                                                                      |
| `started`                                                             | 持锁保证 driver 只启动一次；与 closed 不同，允许尚未 start 就 stop。也可从 Thread 状态判断，但不比一个明确布尔值更直观。                                                                                             |
| `notifications`                                                       | 通知版本。driver 在扫描前后比较，防止扫描期间的阻塞消耗 unpark permit 后睡过头；与 Worker 容量版本不同，入队、完成和停止也会改变它。                                                                                 |
| `planningThread`                                                      | 标记当前线程属于本 Scheduler 的 planner，阻止关闭时等待自己。不是请求状态；不能根据线程名称判断归属。                                                                                                                |
| `schedulerThread` / `planners`                                        | 分别执行全局有界扫描和可能阻塞的选址，保留 join/awaitTermination 的真实退出条件；不能用 registered 为空替代。                                                                                                        |
| `capacityListener`                                                    | 向全局容量通知器登记并在 stop 时解除的精确监听实例；是需释放的外部关联，不是请求状态。                                                                                                                               |
| `priorityOrdering` / `plannerCount` / `scanBudgetMultiplier` / `lock` | 启动时固定的排序、并发、扫描预算和同步约束；依赖引用不作为另一套动态请求状态计算。                                                                                                                                   |
| `observedCapacity` / `plan` / `retry` / `operation`                   | 一次调用的采样版本、实际选择结果、收尾决定和 Context 尝试身份；有真实跨步骤用途，但不应复制到长期请求账本。                                                                                                          |
| `preemptionContinuation`                                              | 推迟抢占结果订阅，直到原选址任务归还名额；否则立即完成的 Future 可能先发布重试，driver 在旧任务仍 busy 时消费掉重试资格。延迟交接有原因，但把 Runnable 放进 RequestResult 混合了数据与调度行为，是后续协议收拢候选。 |
| `queueSchedulingCheck` / `scheduleAt` 时间返回值                      | queueSchedulingCheck 永远是 MAX；scheduleAt 只返回 now 或 MAX，当前没有真正的“未来检查时刻”。这是可删的表示层复杂度，不是必要调度事实。                                                                              |
| `closed && !hasWork` 分支                                             | 此分支返回 MAX；closed 为真时下方 ready 也必为 false，同样返回 MAX。分支冗余，不能反推关闭时无需等待未结束任务。hasWork 若无其他生产调用，应随分支一起清理而不是只为 UT 留接口。                                     |
| `drainOnClose` 的两个集合并集                                         | 正常时 queue 与 registered 等价；自定义队列入队/移除部分失败后可能不一致，关闭仍需收齐条目。只有先定义并验证队列原子失败契约，才能去掉兜底并集。                                                                     |

### 具体反例与已有测试证据

- A 正在选址，用户取消 A 并移除登记，B 到来：A 未返回前不能释放选址名额，
  否则持续取消会积累超过 P 个未清理任务。
  `GlobalQueueProgressTest.cancelledPlannerKeepsItsSlotUntilItsResourcesAreClosed` 覆盖该行为。
- A 的原登记已移除，同 Context 建立新 entry；旧 A 的回调到达：只能结束旧 Work，
  不能删除或扣减新登记。`SchedulingRequestsTest.oldCallbacksCannotRemoveOrDiscountAReplacementRegistration` 覆盖身份边界。
- 抢占名额先释放，B 随后才登记容量等待：需要保留 B 的名额唤醒订阅直到原选址结果处理完，
  再移除容量等待并发布重试；物理容量版本未必变化。
  `GlobalQueueProgressTest.preemptionSlotReleaseCannotBeLostBeforeCapacityParking` 覆盖两种时序。
- 上游抢占 Future 已完成，晚到预留正在回滚：close 不能提前返回。
  `GlobalQueueProgressTest.closeWaitsForLatePreemptionReservationReleaseAfterItsFutureCompletes` 覆盖该边界。
- 容量事件在选址与 park 之间到达、或者在本组 retry 中到达，分别由
  `CapacityWaitQueueTest.edgeDuringPlanningRequiresRetryInsteadOfSleeping` 和
  `aNewEdgeDuringAnActiveRetryIsNotLost` 覆盖。
- 这些是读过源码且纳入本轮全量运行的现有测试；没有新增竞态复现或性能基准。

### 精简取舍与实施顺序

1. 可先删无效的未来检查时间表达及冗余关闭分支；它们不承载新事实。
2. 保留“当前登记”和“未结束工作”两个生命周期集合。不要先把 Work 改成一个互斥枚举，
   决策与抢占存在短暂重叠。先比较 O(P) 推导两个计数与 O(1) 缓存的成本，再决定计数是否保留。
3. 默认 DefaultDecodePreemption 只发取消并返回已完成的 BLOCKED/CLOSED Future，
   没有直接移交预留；但 DecodePreemption 是可替换接口，现有异步/成功移交测试表达扩展契约。
   可以研究已完成结果的同步处理路径来减少默认路径的闭包、Future 和再次投递，
   同时保留异步结果晚到释放与关闭等待；不能因默认实现同步就删除整个异步协议。
4. 然后评估将条目查找与队列成员修改收进同一个队列入口，减少 registered 的双处维护。
   必须同时迁移自定义 SchedulingQueue，先固定入队和移除失败后的精确成员契约。
5. 容量与抢占名额等待均保留事件事实，若统一表示则在一个持锁入口同时登记 OR 唤醒条件，
   替换并删除现有两处登记/清理协议；不再叠加第三套 wait 状态。

根本判断：需要保留的不是所有现有 Map 和中间类，而是精确身份、未结束副作用、
排序资格和未消费事件。先减少独立事实来源与交接，再用规模和测量决定索引，
不能以“测试通过”证明所有现有存储都必要，也不能以“只剩两个集合”证明实现更简单。

### 本轮全量验证（顶层 Maven 目录调整前）

在独立副本执行 `./mvnw clean test -fn`，完成全部模块及当前默认绑定的 route-performance
门禁；共享 telemetry fixture 随副本提供，未修改测试阈值或通过条件。
共执行 2543 项：6 项 assertion failure、2 项 error、3 项跳过。
普通 UT 的 2 项断言失败与 2 项 Mockito UnnecessaryStubbing 错误与迁移前记录一致；
API 普通 UT 224 项无失败，mock-engine 307 项无失败。

额外 4 项路由性能用例全部因 macOS 不存在 `/proc/self/status` 而无法取得进程资源样本；
其中 10k QPS 两项同时未通过 client p99 门槛，QUEUE 为 5.156083 ms，DIRECT 为 7.006625 ms。
四组请求均无 RPC 错误且完成吞吐接近目标，但这不等于性能门禁通过，也不能据此断言延迟失败
来自环境或目录迁移。完整日志位于 `/tmp/flexlb-scheduler-audit-full-ut.log`。
全 reactor Spotless、修改类成员顺序和两个开发 skill 的结构验证均通过。
顶层目录后续调整须重新编译和全量运行，不能沿用本段结果作为新布局的验证。

### 七个 Maven 模块和最新目录的验证

最新布局为 api、config、balance、worker、cache、master、util 七个独立业务模块；
Mock Engine 保留为 reactor 中的测试工具。Worker 按 endpoint/rpc/status/cache 组织，
配置按业务用途和 loading 流程组织，公共通信组件归 util.network。
源码根包与模块一一对应，源码引用与 Maven 依赖均未发现跨模块环。

完整普通 UT 在隔离副本执行，共 2539 项，2 项断言失败、2 项 Mockito 错误、3 项跳过；
失败用例与迁移前一致，Mock Engine 的 307 项测试通过。随后删除指标测试中未被调用的
`observation.role()` stub，两项参数化测试单独复验均通过，原有指标断言不变。
剩余普通 UT 断言失败为 DecodeCapacityPreemptionTest 的容量竞争用例和
EncoderStrategyTest 的分组变更用例。

路由性能矩阵 12 项全部执行，11 项 assertion failure、1 项端口占用 error；其中包括
macOS 缺少 `/proc/self/status`、延迟门槛及容量压力错误分类问题。选路性能 2 项均未通过
分配量门槛：750 个 Worker 的稳态分配约 51043 bytes/op，高于单 Worker 的约 7816 bytes/op。
不能据此宣称性能通过，也未修改任何阈值。

默认 `./mvnw test` 已取消对 performance-regression 标签的排除，选路性能与原有路由矩阵
均纳入默认验证。最终全 reactor Spotless 和成员排序通过，修改后的开发 skill 结构验证通过。
完整 UT、选路性能和 Mockito 补验日志分别为 `/tmp/flexlb-final-layout-ut.log`、
`/tmp/flexlb-final-layout-selection-perf.log`、`/tmp/flexlb-mockito-fixed.log`；路由矩阵日志为
`/tmp/flexlb-modules-full-ut-first.log`。默认性能标签变更后尚未重新跑整套矩阵，不能将上述分次
结果写成一轮全绿的默认 Maven 构建。

## 2026-11-10 全目录重复校验核对

范围为整个 FlexLB 目录，包括各模块生产代码、测试、夹具、工具和示例；排除生成目录、构建产物和缓存。
首次检索 1068 个文件，随后取消 Git 忽略规则补扫至 1898 个文件，没有新增含校验调用的文件。
原始匹配：requireNonNull 253 行 / 73 个文件，checkState 131 处 / 34 个文件，checkArgument 158 处 / 50 个文件。
逐文件查看匹配位置及所属入口，再核对直接调用、构造关系和是否跨越锁或异步交接。

本轮校验改动涉及 18 个生产文件，删除 22 处空值校验、2 处直接状态校验和 3 处参数校验调用。
此外收拢重复的锁检查调用、条件表达式和只为已验证参数服务的转换，不把这些都混算为直接 checkState 删除数。

| 位置                                   | 收拢方式与保留的保证                                                                                                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RoutingResult、CapacityBoundary 的工厂 | payload 合法性统一由 record 构造器检查；工厂的空 payload 现在统一抛 IllegalArgumentException，原先工厂抛 NullPointerException。新增合同测试覆盖工厂和直接构造。                          |
| RequestContext → RequestData           | 配置在 RequestData 构造器校验一次；绑定参数及未绑定状态由 Repository 在相同 Data 锁内校验，内部 register 直接绑定。                                                                      |
| DeliveryDispatcher → Delivery.Member   | 入口核对发送模式和 batchId，initialize 保留“不能重复初始化”的状态校验，删除相同参数校验。                                                                                                |
| WorkerStatus                           | prepareStatus、recordSuccessfulPoll、updateDiscoveryLabels 通过 requireActiveGeneration 一次检查锁和活跃状态；publishPreparedStatus 自己检查锁，由 publishReducedStatus 检查未退休状态。 |
| EndpointRegistry                       | 映射发布保留 current == null，一并拒绝已有的相同或不同代际；删除被该条件完全包含的判断。关闭前后、跨锁边界的复查保留。                                                                   |
| RoleSelection                          | 角色类型由各个 sealed 子类精确检查一次；父类仍检查成功元数据，Prefill 子类同时拒绝空 role。                                                                                              |
| DecodeEndpoint                         | 新候选初始化的 observation 身份统一由 calibrateLocked 检查；涉及 beginRetirement 副作用之前的身份校验继续保留。                                                                          |
| ArithmeticFormula → Parser             | 外层 copyOf 完成空值校验与不可变封装，包内 parser 使用已经验证的绑定，不重复 copyOf/requireNonNull。                                                                                     |
| CapacityRelease.plus                   | 两个不可变 CapacityRelease 已在构造时检查非负容量，plus 直接饱和相加；原有饱和加法校验 helper 仍供未验证的计数输入使用。                                                                 |
| 定时任务、退休回调                     | ScheduledThreadPoolExecutor.schedule 和 CompletableFuture.thenRun 已拒绝空任务，删除转交前相同检查。                                                                                     |
| 排队恢复与选路                         | 恢复节点的 removed 条件在插入时检查；RoutingContext 构造器保证 context 非空，选路仍检查其是否为当前准确上下文。                                                                          |

保留项包括独立对象的构造约束、外部 RPC/策略返回值校验、CAS 成功与否、容量转移结果、
下溢和溢出检查，以及状态可能在锁或线程边界发生变化后的复查。未进行吞吐或延迟测量，不以删除检查数量声称性能收益。

验证：空值清理专项用例通过；状态/参数校验专项 148 个用例通过。
全量回归执行 common 461、grpc 274（原有跳过 1）、cache 201、sync 2697 个用例。
sync 唯一失败用例用 Mockito 伪造了构造器不允许的 RoutingContext(null, ...)，
已改为合法但不是当前 admission 的真实 RoutingContext，保留原 IllegalStateException、状态不变和资源未准备的断言。
修正夹具后，相关同步用例 40 个及 common 46 个重跑通过。
收尾复验 SchedulerArchitectureTest 与 DirectAdmissionContractTest 共 64 个用例通过；全 reactor 测试编译、Spotless 与成员排序通过。

## 2026-11-10 交付链路 UT 入口与重复状态清理

本轮以交付边界改造完成后的工作区为基线，核对 Delivery、DeliveryDispatcher、DeliveryExecution、
PrefillQueueScheduler、PrefillDeliveryScheduler、PrefillEndpoint、DecodeEndpoint、RequestContext、RequestRepository。
先区分实际生产调用、方法引用、Spring 入口和扩展契约，再清理 UT 专用入口，随后复核状态写入与清理。

删除的生产入口：

- `PrefillEndpoint.isQueued`：只有 UT 使用，断言改用现有队列快照核对精确 route 身份。
- `PrefillEndpoint.committedSnapshot`：只有 UT 使用，完整诊断快照获取移到 EndpointTestSupport；
  保留原来的锁内采样、锁外计算、可控时钟和工作量断言，不增加生产测试钩子。
- `PrefillEndpoint.queueHeadLocked`：没有生产读取方，只出现在锁约束 UT 与 mock 名单中，删除整个入口和对应名单项。
- `PrefillEndpoint.RequestRecord.belongsTo`：没有生产调用，只剩 mock 与绑定断言；删除 stubbing，真实绑定继续断言精确 route 身份。
- `DecodeEndpoint.placementVersion`：只供 UT 读取 admissionVersion，改为生产路由快照已有的 admissionVersion 字段。

重复状态处理：

- `PrefillEndpoint.rollback` 删除记录后再次清空 committedWorkSnapshot，而 removeEntry 已根据记录的阶段处理缓存失效。
  排队选址登记不会改变已提交工作量，外层重复清空会造成无意义重建。
  `rollingBackQueuedSelectionKeepsUnchangedCommittedWorkSnapshot` 先在旧实现失败，修正后验证复用工作快照且容量归零。
- `PrefillDeliveryScheduler` 删除调用 waitCurrent 前将“没有 request 的 decision”再转换为 null 的重复分支；
  waitCurrent 已统一处理空 decision 和空 request，保留原有后续调度条件。

核实后保留的状态：

- DeliveryExecution 的发送许可与 pendingDeliveryResponses：许可先归还，响应处理可能仍在运行，关闭需等待两者。
- PrefillDeliveryScheduler 的 ready/running 与并发任务数：运行中可以收到后续通知；并发计数是避免每次扫描全部队列的缓存。
- RequestRepository 的 stopping/closed：停止接收与关闭登记、清理阶段不同；retainsIdentity 有真实生产方法引用。
- PrefillEndpoint 的排队、预留、工作登记状态和工作量缓存：影响容量、预测与清理的事实不同，保留精确阶段与身份。

本轮生产代码净减少 39 行（PrefillEndpoint -34、DecodeEndpoint -4、PrefillDeliveryScheduler -1），
相对交付边界改造最初基线累计净减少 83 行。测试夹具与断言同步迁移；372 个针对性用例通过。
全量回归执行 sync 2664 个用例；其中 15 个测试类因测试夹具仍反射写入旧 activeIndex 字段而失败。
将夹具对齐当前源码的 queuedRequests 字段后，重跑这 15 类共 360 个用例全部通过。
该字段改名已存在于当前工作区，保留其命名，只迁移失效的测试反射引用。
整个 Maven reactor 的 Spotless 和成员排序检查通过。

## 2026-11-10 交付出队边界收缩

本轮修改 `PrefillEndpoint`、`PrefillQueueScheduler`、`DeliveryDispatcher`、`Delivery` 的生产实现及相关测试。
先增加队头和批次容量边界 UT，在旧实现上通过后删除反向回写接口，再迁移调用方。

- 删除 `PrefillEndpoint.commitDelivery(Delivery)` 和对外可访问的 `commitLocked`。
  新入口 `tryStartQueuedDelivery(DeliverySelection)` 在一个方法内加锁、校验和修改账本，删除单次转调的私有包装。
- 删除 `Delivery.committed(...)`、`committed`、`removedFailedMember`、`selectionTimeMs`、
  `maxInflightBatches` 以及类级 Lombok Getter、`items()`、`failedMember()`。
  临时许可和成员集合只在 scheduler 包内供 Dispatcher 使用，不再向 Endpoint 暴露。
- Dispatcher 的旧包级 `commit` 改为私有 `dequeuePrepared`，接收 Endpoint 返回值。
  出队结果只在本次 submit 内判断；异步发送需要的工作快照和队列深度由 Dispatcher 保存。
- 保留自定义调度器实际使用的 public `submit` 和 `Delivery` 构造器；
  保留 RequestContext 的发送原子校验、Decode 许可转移及发送结果不确定时的清理协议。
- 测试夹具直接构造出队输入、读取出队结果，不再创建空 Delivery 或调用无资源可释放的 close。
- 出队只处理选中的有效前缀：删除 DeliverySelection.failedMember 和 DequeueResult.removedFailedMember，
  删除 Endpoint 对未选中失败成员的校验与移除、Dispatcher 中对应的专用通知分支。
  PrefillQueueScheduler 在有效前缀交付提交后复用 commitBoundary 处理失败成员，统一移除、通知和异常记录。
  未开始交付时仍保留原有边界返回和容量等待路径。

关键流程：

```java
// PrefillQueueScheduler：只决定连续队头是否到了交付条件。
boundary = dispatcher.submit(delivery, candidates, prediction, prefixPolicy, selectionStillAllowed);
if (boundary == null && preparationHasFailedMember) {
    commitBoundary(delivery.blocked, delivery.boundary, 0L);
}
return boundary;

// DeliveryDispatcher.submit：持有临时资源，协调交接与失败。
prepare(delivery); // 准备预测、执行许可与 Decode 临时许可
if (prefixNotAllowed) return boundary;
dequeued = dequeuePrepared(delivery); // 计算批次预测，调用 Endpoint
if (dequeued == null) return boundary; // finally 归还临时资源
dispatchCommitted(delivery); // 提交任务或发布路由
// 若出队后提交任务失败，上报每个成员的准备失败；Scheduler 清理请求登记。

// PrefillEndpoint.tryStartQueuedDelivery：所有队列和 Prefill 账本变化在同一把锁内。
lock();
now = clock.getAsLong();
checkExactWaitingMembersAndPrefix(selection, now);
checkCapacityAndEndpointAvailability();
capturePrecedingWork();
removeSelectedPrefixAndRegisterWork();
return new DequeueResult(work, queueDepth);

// 发送任务：构造 payload 后再次在请求锁内决定该请求能否真正开始发送。
ctx.validateDelivery(exactRoute, now);
decodePermit.dispatch();
ctx.recordDeliveryStarted(member, now);
// 锁外调用 gRPC；结果未知不能当成未发送重试。
```

额外发现并修复：旧实现使用 Dispatcher 在等待 Endpoint 锁之前采样的时间检查截止期限。
`deadlineIsCheckedWhenTheEndpointDequeuesRatherThanWhenSelectionWasPrepared` 先证明旧实现错误出队，
然后改为 Endpoint 持锁后取时间，删除跨组件时间参数。
新增/调整 UT 还覆盖输入集合与预测数组不能被调用方改写、同一队头并发出队只有一次成功、
容量不足不能移除失败边界成员，保留取消、执行器拒绝、锁边界和资源释放断言。

代码量相对本轮开始的工作区：生产代码净减少 44 行（Endpoint +2、PrefillQueueScheduler +7、Dispatcher -23、Delivery -30）。
输入/结果 record 保留明确的数据边界；继续删除单次转调、只在计算中使用的持久 features 字段及重复 evaluator 参数。
Endpoint 用一个有序 RequestRecord 集合完成校验、去重及工作登记，删除保存相同记录的 list 与 set。
RequestRecord 是没有 equals/hashCode 覆写的 final 类，集合仍按记录身份去重，遍历顺序与请求预测数组一一对应。
没有新增生产顶层类。
`successfulPrefixIsPublishedBeforeHandlingAnUnrelatedPreparationFailure` 先在旧实现失败，
再验证正常前缀先发布，失败成员的通知异常仍记录到 shutdown failure，且不影响正常请求的临时许可释放。
Endpoint 的原失败成员原子提交测试改为验证未选中成员既不被校验也不被出队；请求失败处理仍由队列层完成。

验证：先在旧流程上运行新增 UT，确认正常前缀发布前已进入失败成员通知；改造后相关 195 个专项用例通过。
最终 `./mvnw -pl flexlb-balance -am test` 完整通过，sync 2663 个用例无失败、无错误、无跳过；
依赖模块也通过（grpc 有 1 个原有跳过用例）。成员排序和整个 Maven reactor 的 Spotless 检查通过。
共用测试夹具中已失效的 telemetry 字段引用已迁移到实际 reporter 字段。

## 2026-11-10 调度包逐类逐函数复核

范围是 `org.flexlb.balance.scheduler` 全部 36 个顶层类及其嵌套类型，不把此结果声称为整个 FlexLB 的全仓审核。
逐函数阅读实现，并核对直接调用、方法引用、Spring 生命周期、测试夹具与扩展契约；下表包含本轮未修改类的保留理由。
函数清单按当前源码抽取，含手写构造器与重载参数个数，不含 Lombok 自动生成方法；清单用于防止遗漏，不能代替行为验证。

### 本轮实际删改与证据

- `RequestContext`/`ContextResult`：`RouteState` 只有 `preparedDelivery` 一个生产判断点，其他条件已经检查 route、pendingEnd、outcome、取消与抢占。
  `retainRouteForCleanup` 始终保留结束原因，随后 `beginClose` 保留 outcome，因此移除重复字段、枚举与多处写入。
  `RoutingContextTest.unpublishedCleanupRouteCannotBecomeDispatchable` 增加路由完成后和结束开始后的禁止投递断言，先在旧实现通过。
- `recordAbandonment` 和 Abandonment 事件没有生产入口；测试改为真实 `requestCancel`，保留第一次原因、不把路由关闭伪装成引擎 NOT_SENT 的断言。
  清理了两个测试兜底路径向不支持该事件的 onDeliveryResult 传入 Abandonment 的旧代码。
- `DeliveryDispatcher`：timer 字段只在构造器赋值，删除字段、构造参数和夹具注入；移除 publishRoute 纯转调。
  新屏障 UT 同时暂停两个 putIfAbsent，旧实现 activeStops 留下两个对象，预期只有一个；提交拒绝 UT 也复现已关闭任务仍留在 activeStops。
  修复失败登记对象及被拒绝任务的清理，继续等待真正已接受的工作与回调，不用清空整个索引掩盖残留。
- `RequestRepository.pendingDeliveryRequestCount` 只有测试调用，移入已有共享 testFixtures；不在生产目录留下用于测试等待的统计遍历。
- `RequestRouteSelector` 与 `RouteResult` 共用完全相同的 endpoint 失效检查；`RequestCloseQueue` 合并转调和无人读取的返回值。
  `QueuedRequestScheduler` 不再在锁外重复展开 RequestContext 的撤回条件；保留 routeFailure 快速判断避免普通事件无谓进入撤回检查。
- `AbstractRequestScheduler` 删除同一快照上重复的 notificationsDone 判断；`ExpirationTimer` 修正“没有请求 map、负责归档维护”的过时说明。

### 本轮代码量与验证

以本轮开始保存的全部 Java 源码快照为基线，生产代码净减少 **92 行**（含空行、注释），没有新增生产类或生产文件迁移。
其中 RequestContext −25、ContextResult −18、RequestRouteSelector −18、RequestRepository −14、
DeliveryDispatcher −6、QueuedRequestScheduler −6、ExpirationTimer −6、RequestCloseQueue −4、RouteResult +5；
AbstractRequestScheduler 修改一处重复判断，物理行数不变。测试净增 61 行，其中普通测试 +55、共享 testFixtures +6。

`./mvnw -pl flexlb-balance -am test`：common 453、cache 201、sync 2658，共 **3312** 个用例，零失败、零错误、零跳过。
日志 `/tmp/scheduler-deep-full.log`。旧实现的两个停止任务残留失败见 `/tmp/scheduler-deep-stop-before.log`，
改后定向测试见 `/tmp/scheduler-deep-focused.log`。完整 reactor Spotless、所有本轮修改 Java 文件的成员顺序检查通过。
`./mvnw test -DskipTests` 完整 reactor 编译通过，包含 API 与 mock-engine 的测试源码（`/tmp/scheduler-deep-reactor.log`）；不将编译计作这些模块的行为测试。
未执行端到端性能基准，未提交或推送。

### 逐类结论与函数清单

**AbstractRequestScheduler**（62 个手写函数/构造器）：删除完成通知入口中对同一不可变快照的重复判断；保留响应、资源清理和关闭顺序。

函数：`AbstractRequestScheduler/8`、`submit/1`、`submit/2`、`cancel/3`、`cancel/4`、`cancel/3`、`recordRequestFailure/1`、`publish/3`、`isAdmissionOpen/1`、`currentRoutingContext/1`、`finishRouting/1`、`completeRouting/2`、`onDeliveryResult/1`、`isCurrentRequest/1`、`onRouteFailure/3`、`recordRemoteCleanup/1`、`failRouting/2`、`onRequestStateChanged/1`、`onWorkerRequestEvent/2`、`onEndpointRetired/2`、`rejectRoute/2`、`onDeadline/2`、`shutdown/1`、`checkShutdownThread/0`、`tryBeginRouteSelection/1`、`schedulingTimeoutError/0`、`describeWait/3`、`completeWaitDiagnostics/2`、`describeWaitAtStage/3`、`checkRequestMode/2`、`removeRequest/1`、`onRouteFailure/2`、`processPendingRequestActions/1`、`commitEncoderRoute/1`、`encoderRequestLimit/0`、`recordDeliveryPrediction/1`、`recordPreparationFailure/1`、`throwIfRequestFailed/0`、`finishUnpublishedRoute/2`、`releaseRouteResources/2`、`preparationFailed/2`、`completeRouting/3`、`finishRoutingRequest/5`、`closeRequest/2`、`scheduleClose/1`、`tryCompleteFromExternal/5`、`closeOutstandingAndTerminalize/0`、`closeExecutors/0`、`onDeliveryNotStarted/1`、`recordDeadline/4`、`recordPredictedDelivery/3`、`recordRequestChange/2`、`confirmDelivery/1`、`reserveResponse/2`、`needsFollowUpActions/2`、`submitClose/2`、`releaseBeforeResponse/1`、`runClose/2`、`cleanupRequest/1`、`requestDeliveryStop/1`、`completeRequestEndNotifications/1`、`response/3`。

**BatchEnqueueHelper**（13 个手写函数/构造器）：保留 token 原始字节、元数据和地址缓存；这些是局部解析/分配优化，不是请求生命周期副本。

函数：`parseInput/1`、`payloadSizeUpperBound/1`、`prepare/2`、`retain/2`、`validate/4`、`mergeMetadata/4`、`addressSizeUpperBound/1`、`acknowledge/5`、`input/2`、`role/2`、`readField/3`、`writer/0`、`finish/0`。

**CancelReason**（3 个手写函数/构造器）：保留取消原因到引擎协议的映射。

函数：`CancelReason/1`、`getMessage/0`、`toRequest/1`。

**CapacityKey**（4 个手写函数/构造器）：保留角色/组等待域与具体 endpoint 的区分；后者供诊断及抢占使用。

函数：`CapacityKey/0`、`anyGroup/1`、`exact/3`、`waitingDomain/0`。

**CapacityNotifier**（11 个手写函数/构造器）：保留变化序号、分组历史及历史删除下限，防止登记期间或退休期间丢失变化。

函数：`addListener/1`、`removeListener/1`、`notifyChanged/2`、`notifyChanged/1`、`clearRetired/1`、`sequence/0`、`lastChangedSequence/1`、`notifyChanged/3`、`discardHistory/2`、`onEndpointChanged/1`、`onCapacityChanged/1`。

**CapacityWaitQueue**（15 个手写函数/构造器）：保留 membership、每域排序及 retry 身份；isWaiting 与 contains 对进行中的重试含义不同。

函数：`CapacityWaitQueue/2`、`isWaiting/1`、`contains/1`、`park/3`、`remove/1`、`capacityChanged/1`、`hasReadyRequests/0`、`size/0`、`resumeReady/2`、`clear/0`、`remove/2`、`unindex/1`、`index/1`、`release/1`、`Domain/1`。

**ContextResult**（34 个手写函数/构造器）：删除 RouteState、测试专用 Abandonment 事件及仅测试读取的 failedRole 派生方法；失败 endpoint 仍供扩展策略读取。

函数：`ContextResult/0`、`RouteFailure/3`、`RouteFailure/0`、`End/0`、`publicationFailure/2`、`failure/4`、`withdrawnFailure/1`、`externalFailure/3`、`deliveryFailure/2`、`terminal/0`、`end/0`、`end/0`、`delivery/0`、`nowMs/0`、`predict/2`、`observe/2`、`expireVisibility/0`、`visibility/5`、`owner/0`、`recordTransport/2`、`recordResult/4`、`abandon/2`、`releasable/1`、`mergeReleaseProof/3`、`recordEnd/2`、`recordReport/3`、`mergeSettlement/2`、`withSettlement/2`、`beginEnding/2`、`retainEnd/1`、`deliveryReleasable/0`、`canConfirmDelivery/0`、`phase/0`、`preparedDelivery/0`。

**Delivery**（21 个手写函数/构造器）：保留临时 permit、队列提交事实与请求完成信号；close 返回临时许可，complete 表示请求资源已清理，不能合并。

函数：`Delivery/5`、`committed/3`、`items/0`、`failedMember/0`、`close/0`、`releaseMembers/0`、`Member/2`、`owner/0`、`close/0`、`completion/0`、`initialize/2`、`recordFailure/1`、`prepare/2`、`complete/0`、`completionFuture/0`、`Result/0`、`delivered/0`、`notSent/1`、`prefillRejected/1`、`uncertain/1`、`failed/0`。

**DeliveryDispatcher**（48 个手写函数/构造器）：删除未使用 timer 依赖和 publishRoute 转调；修复竞争登记失败方及提交拒绝后 activeStops 残留。

函数：`DeliveryDispatcher/6`、`mode/0`、`cancel/6`、`isStoppingOnCurrentThread/0`、`awaitStops/0`、`submit/5`、`routeDelivered/3`、`validateRequestInput/1`、`releaseUnusedInput/1`、`capacityAvailable/2`、`watchCapacity/3`、`deliverBatch/1`、`stop/1`、`shutdown/1`、`close/1`、`commit/5`、`commitSelectedDelivery/2`、`commitQueuedDelivery/3`、`commitDelivery/4`、`tryCommitDelivery/4`、`report/1`、`report/3`、`logFailure/3`、`cancelEvidence/1`、`deliverCommitted/1`、`failBeforeSend/3`、`dispatchBatch/4`、`logDispatch/5`、`reportDelivery/3`、`record/3`、`awaitDeliveryCleanup/2`、`submitted/1`、`dispatchCommitted/1`、`publishRoutes/1`、`reportRoutes/3`、`prepare/3`、`prepareMember/3`、`StopTask/3`、`run/0`、`start/0`、`signal/0`、`close/1`、`finish/1`、`stopRemotely/0`、`result/0`、`await/2`、`requestProof/1`、`applyReply/2`。

**DeliveryExecution**（23 个手写函数/构造器）：保留 permit 数与响应回调数；先返还发送容量后，回调仍可能运行，关闭必须等待两者。

函数：`DeliveryExecution/2`、`activeThreads/0`、`queuedTasks/0`、`poolSize/0`、`completedTasks/0`、`awaitDeliveries/0`、`nextBatchId/0`、`tryReserveDelivery/0`、`capacityAvailable/2`、`deliveryCapacityAvailable/0`、`watchDeliveryCapacity/1`、`stopDeliveryAdmission/0`、`observeDeliveryResponse/2`、`finishDeliveryResponse/0`、`finishDeliveryDrain/0`、`releaseDeliveryPermit/0`、`close/0`、`tryChargeBytes/1`、`submit/2`、`releaseIfNoResponse/0`、`observeResponse/2`、`hasTaskOwner/0`、`release/0`。

**DirectRequestScheduler**（16 个手写函数/构造器）：保留启动前 pending、提交中的 scheduling 及 acceptedWork；分别覆盖启动缓存、提交交错和已接受异步任务。

函数：`DirectRequestScheduler/9`、`start/0`、`onEndpointChanged/1`、`close/0`、`stop/0`、`awaitStopped/1`、`scheduleRequest/1`、`selectAndPublish/2`、`schedulingTimeoutError/0`、`removeRequest/1`、`selectAndPublish/2`、`commitDirectRoute/1`、`drainPendingRequests/0`、`submitSelection/1`、`dispatchRegistered/1`、`selectAndCommit/1`。

**ExpirationTimer**（23 个手写函数/构造器）：修正与实现不符的类说明；保留修订号、安装失败、先触发后安装及取消协调，TaskTimer 明确允许返回前执行回调。

函数：`ExpirationTimer/2`、`ExpirationTimer/3`、`close/0`、`update/2`、`needsUpdate/2`、`remove/1`、`replaceHandles/5`、`install/1`、`delayUntil/1`、`beginRegistration/0`、`endRegistration/0`、`awaitRegistrations/0`、`Token/0`、`schedule/4`、`token/1`、`without/1`、`token/5`、`hasFailedInstallation/0`、`Handle/1`、`install/1`、`deliver/0`、`fire/0`、`cancel/0`。

**GlobalQueueEntry**（7 个手写函数/构造器）：markQueued/markRemoved 为外部 SchedulingQueue 扩展维护成员身份，不能只按内置调用数删除。

函数：`GlobalQueueEntry/2`、`context/0`、`routingGroup/0`、`sequence/0`、`markQueued/1`、`markRemoved/0`、`priority/0`。

**OrderedRequestQueue**（15 个手写函数/构造器）：保留桶链表、扫描游标、readyRetries 和优先级统计；分别承担删除、有限扫描、精准重试与统计。

函数：`OrderedRequestQueue/1`、`add/1`、`priorityCounts/0`、`size/0`、`restore/1`、`scanForPlanningCandidates/3`、`hasUnscannedRequests/0`、`markRequestReadyForRetry/1`、`drain/0`、`remove/1`、`order/0`、`insert/2`、`bucketIndex/1`、`insert/2`、`remove/1`。

**PreemptionRegistration**（5 个手写函数/构造器）：保留准确抢占身份及惰性 Future，完成结果与资源预留是不同事实；未用新增公共抽象合并 Future 代码。

函数：`PreemptionRegistration/4`、`completion/0`、`result/0`、`complete/1`、`completionFuture/0`。

**PrefillDeliveryScheduler**（8 个手写函数/构造器）：复核前轮结论；ready 可与 running 同时成立，timed 为截止时间索引，共享 planner 不等于共享名额语义。

函数：`PrefillDeliveryScheduler/6`、`endpointChanged/1`、`stop/0`、`clearAfterTasksStopped/0`、`scheduleAt/1`、`runQueueScheduling/1`、`finishQueueScheduling/3`、`rejectedDuringStop/1`。

**PrefillQueueScheduler**（23 个手写函数/构造器）：复核前轮删除停止协议后的单任务边界；保留容量订阅后的复查和队首连续前缀校验。

函数：`PrefillQueueScheduler/4`、`prefillEndpoint/0`、`captureQueue/1`、`settings/0`、`getLatestQueueWaitSnapshot/0`、`recordQueueWait/2`、`schedulingConstraints/0`、`processSelection/4`、`stopping/0`、`commitBoundary/3`、`observeWait/1`、`waitCurrent/2`、`scheduleOnce/0`、`removeExpiredRequest/1`、`waiting/0`、`failQueuedRequests/1`、`capacityAvailable/2`、`watchCapacity/3`、`scheduleGroup/3`、`notifyRemovedRequest/2`、`updateCapacityListener/1`、`capacityBlocked/3`、`awaitingSchedulingChange/5`。

**QueueExecutionSettings**（6 个手写函数/构造器）：保留不可变配置与容量归一化；固定比较器仍用于预测新请求按尾部追加。

函数：`queueOrder/0`、`capture/2`、`constraints/1`、`capacityChanged/2`、`tokenCapacity/1`、`kvCapacity/1`。

**QueuedRequestScheduler**（51 个手写函数/构造器）：删除重复展开的撤回准入条件，RequestContext 在锁内统一判定；保留 selectAndPublish 的自定义调度扩展用途。

函数：`onCapacityChanged/1`、`onEndpointChanged/1`、`QueuedRequestScheduler/14`、`queuedRequestCount/0`、`start/0`、`shouldRetryRouteFailure/1`、`scheduleRequest/1`、`onEndpointChanged/1`、`blockedRequestCount/0`、`close/0`、`stop/0`、`awaitStopped/1`、`allowsRoutePreemption/0`、`selectAndPublish/2`、`schedulingTimeoutError/0`、`encoderRequestLimit/0`、`removeRequest/1`、`onRouteFailure/2`、`processPendingRequestActions/1`、`describeWait/3`、`enqueue/1`、`selectAndPublish/2`、`enqueueRoute/1`、`notifyQueuedRoute/1`、`completeRouting/3`、`run/0`、`scheduleAt/1`、`withdrawUnavailableRoutes/0`、`planningCandidates/1`、`removeRequest/1`、`requestRun/0`、`awaitPlanners/1`、`remainingNanos/1`、`submitPlan/1`、`place/1`、`planRoute/3`、`executeQueuedRouting/3`、`startOrWaitForDecodePreemption/2`、`resumeAfterPreemption/4`、`resumeRoute/5`、`finishDecision/3`、`parkOrRetry/3`、`completeDecisionResponse/2`、`placementFailed/2`、`finishSchedulingTask/3`、`retryPreemptionWaiters/0`、`rejectedDuringStop/1`、`drainOnClose/0`、`shutdownRequest/1`、`withdrawFailedRoute/1`、`RoutingPlan/0`。

**RequestCloseQueue**（7 个手写函数/构造器）：合并 submit/execute，删除无人读取的 boolean 返回及多余包装；保留每请求串行、清理合并、响应不可合并与拒绝恢复。

函数：`awaitIdle/0`、`isExecutingOnCurrentThread/0`、`submit/5`、`Lane/1`、`run/0`、`reject/1`、`run/0`。

**RequestContext**（51 个手写函数/构造器）：删除 routeState 可变字段及所有写入，交付准入直接使用 route/pendingEnd/outcome 等事实；删除测试专用 recordAbandonment，测试改走 requestCancel。

函数：`RequestContext/1`、`data/0`、`register/5`、`snapshot/0`、`beginRouteSelection/0`、`beginRoutePreemption/1`、`publishRoute/2`、`completeRouting/3`、`recordDeliveryResult/1`、`recordDeliveryPrediction/1`、`recordRemoteCleanup/1`、`recordPreparationFailure/1`、`recordWorkerRequestEvent/1`、`recordEndpointRetirement/1`、`requestCancel/1`、`recordDeadline/3`、`beginPreemption/3`、`canReconcilePreemption/1`、`completePreemption/3`、`selectResponse/1`、`beginClose/2`、`releaseReason/0`、`validateDelivery/2`、`validateDelivery/3`、`canPublishRoute/1`、`recordResourcesReleased/4`、`recordCloseNotifications/3`、`failUnsentRoute/2`、`beginFailedRouteWithdrawal/0`、`unsentDeliveryFailure/1`、`retainRouteForCleanup/2`、`beginRoutingCompletion/1`、`recordRoutingDiagnostics/2`、`recordDirectDeliveryStarted/3`、`recordDeliveryStarted/2`、`isCurrentEndpointRetirement/1`、`confirmDelivery/1`、`readyToClose/1`、`matchesDelivery/1`、`acceptRoute/1`、`completeClose/1`、`commitSnapshot/1`、`deriveDeadlines/1`、`canBeginRouting/0`、`canWithdrawRoute/0`、`recordWorkerResourceRelease/1`、`waitsForPlacement/1`、`retainEnd/1`、`waitsForPreemption/1`、`classify/1`、`matches/3`。

**RequestData**（31 个手写函数/构造器）：保留输入解析 FutureTask、原请求身份、排队元数据与遥测；解析值/异常复用和输入释放有实际调用。

函数：`RequestData/1`、`setOriginalNumericRequestId/1`、`setPhase/1`、`setRequestedRoles/1`、`setRequest/1`、`setSchedulingMetadata/1`、`setFuture/1`、`getRequestId/0`、`getPriority/0`、`getRequestExpiresAtMs/0`、`getResponse/0`、`setGenerateInputPb/1`、`initializeExecutionOrder/1`、`recordRequestTiming/2`、`finishRequestTiming/0`、`recordCacheQuery/2`、`recordCacheSelection/3`、`beginRoutingAttempt/1`、`recordRoutingDecision/1`、`routingAttempt/1`、`selectionReason/1`、`getRoutingTelemetry/0`、`register/2`、`hasGenerateInput/0`、`prepareGenerateInput/0`、`getGenerateInput/0`、`discardGenerateInput/0`、`releaseSchedulingInput/0`、`initializeExecutionOrder/2`、`acknowledge/2`、`RoutingTelemetry/0`。

**RequestRepository**（25 个手写函数/构造器）：删除仅测试使用的 pendingDeliveryRequestCount，计数移到共享测试夹具；保留 active/归档目录、停止接收与关闭登记的区分。

函数：`RequestRepository/0`、`maintainExpiration/0`、`ownerOf/1`、`ownerOf/2`、`findActive/1`、`findActive/2`、`findTerminal/1`、`findTerminal/2`、`retainsIdentity/1`、`closeRegistration/0`、`stopAccepting/0`、`awaitResponses/1`、`snapshotActive/0`、`isStopping/0`、`isClosed/0`、`oldestLiveRequestAgeMs/0`、`snapshotActiveRequests/0`、`liveRequestCount/0`、`isCurrent/1`、`getRequestState/2`、`getRequestState/3`、`register/5`、`archive/2`、`expireTerminalRecords/1`、`removeExactTerminal/2`。

**RequestRouteSelector**（20 个手写函数/构造器）：部分选址与完整路由的 endpoint 失效检查复用 RouteResult 的实现；保留部分注册失败回滚和 Decode 线程缓存。

函数：`RequestRouteSelector/11`、`select/2`、`routeVit/1`、`selectedVitIsValid/1`、`blockedEndpoint/1`、`select/4`、`resolvePolicyGroup/1`、`selectEncoder/3`、`selectVit/3`、`captureVit/4`、`selectPrefill/4`、`registerPrefill/4`、`reportCacheAffinity/3`、`reportSelection/4`、`selectDecode/3`、`registerDecode/4`、`addDecode/4`、`refreshDecode/1`、`readDecodeCapacity/1`、`rollback/1`。

**RequestScheduler**（10 个手写函数/构造器）：保留自定义调度器和 Spring 启动/关闭契约；start 经 Bean initMethod 调用，不是 UT 专用。

函数：`start/0`、`scheduleRequest/1`、`onEndpointChanged/1`、`allowsRoutePreemption/0`、`shouldRetryRouteFailure/1`、`queuedRequestCount/0`、`blockedRequestCount/0`、`stop/0`、`awaitStopped/1`、`close/0`。

**RequestState**（4 个手写函数/构造器）：保留只读对外状态投影；不是另一套可写生命周期。

函数：`RequestState/0`、`from/1`、`matchesBatch/1`、`isTerminal/0`。

**RequestTaskExecutor**（1 个手写函数/构造器）：保留请求任务执行契约及 request 参数，扩展实现可以按请求分配工作。

函数：`execute/2`。

**RequestTaskPool**（8 个手写函数/构造器）：awaitIdle 在生产 shutdown 中以方法引用调用；提交计数和单次释放保护必须保留。

函数：`execute/2`、`awaitIdle/0`、`isExecutingOnCurrentThread/0`、`close/0`、`completed/0`、`SubmittedTask/1`、`run/0`、`rejectBeforeStart/0`。

**ResponseCompletionExecutor**（27 个手写函数/构造器）：保留注册、提交、执行、完成及拒绝阶段；同步外部 Future 与异步响应回调具有不同调用语义。

函数：`ResponseCompletionExecutor/1`、`close/0`、`isCompletingOnCurrentThread/0`、`register/0`、`release/0`、`invoke/1`、`submit/2`、`finishClose/0`、`Registration/4`、`close/0`、`reject/1`、`prepare/3`、`isUnsubmitted/0`、`complete/3`、`publish/1`、`completeNow/1`、`submit/1`、`failIfUnsubmitted/1`、`run/2`、`RequestFuture/1`、`complete/1`、`completeExceptionally/1`、`cancel/1`、`completeOwned/1`、`completeExceptionallyOwned/1`、`cancelOwned/0`、`complete/3`。

**RoleSelection**（17 个手写函数/构造器）：保留各角色固定元数据与精确注册；资源释放仍由各自 Endpoint 执行。

函数：`RoleSelection/2`、`endpoint/0`、`dpRank/0`、`hitCacheTokens/0`、`capacityKey/0`、`role/0`、`group/0`、`toServerStatus/1`、`ip/0`、`httpPort/0`、`grpcPort/0`、`requestId/0`、`Prefill/4`、`workMs/0`、`Decode/3`、`release/3`、`Auxiliary/2`。

**RouteResult**（28 个手写函数/构造器）：汇总完整路由及部分选择的 endpoint 失效检查，删除一份重复实现；保持原错误信息和重试分类。

函数：`RouteResult/2`、`requestId/0`、`roles/0`、`successResponse/1`、`planningInput/0`、`prefillReservation/0`、`role/1`、`encoderEndpoint/0`、`prefillWorkMs/0`、`ctx/0`、`prefillDpRank/0`、`prefillEndpoint/0`、`decodeEndpoint/0`、`decodeReservation/0`、`enqueuedAtMs/0`、`requestExpired/1`、`expiresAtMs/0`、`priority/0`、`enqueueSeq/0`、`seqLen/0`、`hitCache/0`、`validateEndpoints/1`、`validateEndpoints/2`、`selectedEndpointsAvailable/0`、`firstUnavailableEndpoint/0`、`endpoint/1`、`usesAnyEndpoint/1`、`capacityKey/1`。

**RoutingContext**（1 个手写函数/构造器）：保留对象身份；即使 record 值相等，也不能由旧调用结束新的选路。

函数：`RoutingContext/0`。

**ScheduledTaskTimer**（4 个手写函数/构造器）：两个构造器均有生产用途，清理与请求截止时间使用不同线程配置。

函数：`ScheduledTaskTimer/0`、`ScheduledTaskTimer/2`、`schedule/2`、`close/0`。

**SchedulerConfiguration**（17 个手写函数/构造器）：逐个核对 @Bean、initMethod、destroyMethod 与 ConditionalOnMissingBean；无普通 Java 调用不能作为删除依据。

函数：`taskScheduler/0`、`requestTasks/0`、`taskTimer/0`、`deadlineTaskTimer/0`、`expirationTimer/2`、`responseCompletions/1`、`deliveryExecution/2`、`prefillSelectionStrategy/2`、`decodeSelectionStrategy/2`、`vitSelectionStrategy/0`、`groupPlanner/1`、`workerQueues/0`、`schedulingQueue/1`、`decodePreemptionPolicy/0`、`decodePreemption/8`、`decodeAdmissionCapacity/1`、`requestScheduler/14`。

**SchedulingQueue**（10 个手写函数/构造器）：保留自定义排序、恢复和有限扫描契约；不能为精简内置实现删除扩展能力。

函数：`add/1`、`restore/1`、`remove/1`、`scanForPlanningCandidates/3`、`markRequestReadyForRetry/1`、`hasUnscannedRequests/0`、`size/0`、`priorityCounts/0`、`drain/0`、`order/0`。

**TaskTimer**（1 个手写函数/构造器）：保留一次任务与取消句柄契约；允许回调早于 schedule 返回，不能删掉使用方的安装交错处理。

函数：`schedule/2`。

本清单合计 650 个手写函数/构造器；只读辅助函数不机械增加 UT，状态和异步变更使用行为测试验证。

## 2026-11-10 测试专用入口与重复状态清理

本轮基于上一轮固定队序的完成状态，核对 Prefill 调度、失败清理及其调用方：

- `PrefillQueueScheduler.stopAndAwait()` 只有 UT 调用。测试线程的停止和等待归入
  `RequestExecutionTestSupport`，生产类删除该入口，不再保存 `stopOwner` 和 `stopped` Future。
- 生产中只有当前队列调度任务调用失败清理，正常关闭则先等待所有任务结束，再清除订阅。
  因此删除 `waitLock` 和重复停止协调；队列关闭事实统一读取 `PrefillEndpoint.queueStopped()`。
  `failQueuedRequests(cause)` 先关闭队列，再取消订阅，最后在 Endpoint 锁外逐个通知请求。
  订阅移除或某个请求回调失败不能跳过后续通知；未完成释放的资源仍由 Endpoint 保留。
- `processSelection` 删除生产调用恒定的 `prefixPolicy`、`selectionStillAllowed` 参数。
  内置队列固定连续前缀并在提交前检查关闭状态；`DeliveryDispatcher` 的自定义整体提交策略仍保留，
  对应 UT 直接调用 dispatcher。删除 `PrefillDeliveryScheduler.requestRun` 单纯转调层。
- 保留 ready/running：运行中的任务可以同时收到下一次调度事件。保留 timed 索引、全局容量等待
  和抢占任务收尾，它们表达不同事实，不能因字段多而合并。

先运行原失败清理测试，再用 `failedQueueRejectsEveryRequestOutsideEndpointLock` 固定锁外回调、
后续请求继续清理、关闭后拒绝入队及失败资源保留的契约，确认旧实现通过后修改生产代码。
移除只验证多线程调用测试专用停止入口、重复读取同一个清理错误的断言。

本轮生产 Java 物理行数（含注释、空行；不含 UT 和文档）：`PrefillQueueScheduler` 476 → 433，
`PrefillDeliveryScheduler` 221 → 217，合计 **净减少 47 行**，没有新增或迁移的生产文件。
统计基线为本轮修改前保存的源码快照，未使用包含历史改动的 HEAD 差异。

验证：`./mvnw -pl flexlb-balance -am test` 共 3311 个用例通过，零失败、零错误、零跳过
（`/tmp/prefill-cleanup-full.log`）。最后通过完整 reactor 编译及 RequestExecutionQueueTest 的
19 个定向用例（`/tmp/prefill-cleanup-reactor.log`）；完整 reactor Spotless、修改文件的成员顺序
及 diff 空白检查通过。未提交或推送。

## 2026-11-10 固定队序实施记录

本节记录下方固定队序计划的实际落点；历史旁路策略不再作为当前实现要求。

- 先增加 UT 并复现旧实现的三种行为：优先级/原序号改变 Prefill 接收顺序、阻塞队首被旁路、
  路由发布暂停期间队列快照已能看见它。之后再修改生产代码。
- `OrderedPrefillRequestQueue.fifo` 用插入有序集合及准确身份索引保存成员，追加和删除平均为 O(1)。
  删除排序比较器、队内序号和 bypassedHead。`PrefillRequestQueue.Factory.create()` 不再接收排序参数，
  Snapshot 只保存已有顺序；预测中的新到请求按固定尾插规则处理，不再让队列和预测各维护一套排序。
- `PrefillQueueScheduler` 从全局调度类移为独立类。`scheduleOnce/scheduleGroup` 只处理连续前缀，
  删除替代选择、旁路参数和接口。`PrefillEndpoint.ownsSelectionLocked/commitDelivery` 核对提交时的真实前缀。
  部分准备失败、取消、过期和发送结果未知的处理仍保留；重复 batch 成员仍在队列变化前拒绝。
- `PrefillDeliveryScheduler` 接收 Prefill 事件，独立维护 ready/timed 索引、运行任务名额及关闭后的订阅清理。
  全局调度类只调用 endpointChanged、scheduleAt、stop 和清理入口，不直接修改这些状态。
  共用原执行器与驱动线程，没有新增线程池或第二份待投递请求集合；迁移本身不计为减少系统状态。
- `PrefillEndpoint.acceptQueuedRoute` 持 RequestContext → Endpoint 锁完成本地入队和发布；
  发布拒绝时移除准确临时成员，未交出资源仍由现有选路清理负责。通知保留在成功交接之后的锁外。
  删除消费者的 `Route commit in progress` 等待。审核的内置交付、上报、关闭和抢占路径未发现反向嵌套；
  自定义队列契约明确禁止在 Endpoint 锁内调用请求生命周期。
- 普通优先级选路/抢占配置仍由全局调度器保存，从 QueueExecutionSettings 移除不再用于 Prefill 投递的优先级字段。
  waitingRequests、preemptionQuotaWaiters 和 pendingPreemptions 保留原有等待与收尾保证，未宣称本轮已统一 OR 等待。

新增测试入口：

| 测试类                          | 新增方法                                                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| OrderedPrefillRequestQueueTest  | acceptanceOrderIgnoresPriorityAndOriginalEnqueueSequence；removingMiddleItemPreservesOrderAndReceivingItAgainAppendsIt |
| PrefillRequestQueueContractTest | selectionMustBeAContinuousPrefixOfTheAcceptedRequests                                                                  |
| QueuedBatchDeliveryTest         | blockedHeadPreventsSendingALaterRequestOnAnAvailableDecode；selectingALaterRequestCannotSkipTheAvailableHead           |
| PrefillEndpointTest             | queueCaptureCannotObserveARouteBeforeItsRequestAcceptsIt                                                               |

旧插队/旁路断言迁移为固定队序；取消与锁外清理、单次释放、准确资源预留、容量上限和关闭等待断言保留。
并发消费者在提交旧快照失败后重新读取队首，不能继续提交自己此前任选的成员。
验证结果以本节后续记录为准；下方历史批次数字不能用于本次验收。

本次最终验证：

- `./mvnw -pl flexlb-balance -am test`：common 453、cache 201、sync 2657，共 **3311** 个用例，
  零失败、零错误、零跳过。日志 `/tmp/prefill-sync-verified.log`。
- `./mvnw test -DskipTests`：完整 Maven reactor 的生产代码和测试代码编译通过，包含 API 和 mock-engine；
  不将这条命令算作这两个模块的行为测试。日志 `/tmp/prefill-reactor-compile.log`。
- 完整 reactor `./mvnw spotless:check -Pspotless-check` 通过；相关生产类与主要新 UT 的成员顺序检查通过，
  定向 `git diff --check` 通过。日志 `/tmp/prefill-format-verified.log`。
- 迭代过程中迁移了旧排序断言、旧发布入口 mock、搬迁类的反射夹具，以及故障注入时的优先级配置。
  首次全量还出现过一次未修改的 `DecodePreemptionProtocolTest` 监听器测试 NPE，后续定向和最终全量均通过；
  未将其记为本次修复，也未削弱监听器清理断言。
- 未执行端到端性能基准，固定队首顺序的吞吐及尾延迟取舍仍需实际负载测量。
  不报告跨批次净减行：当前工作区已有大量其他改动，不能把相对 HEAD 的差异归到本次。
- 未提交或推送。本节覆盖实际实施与验证，下面保留的伪代码只说明原设计意图。

## 2026-11-10 新计划：固定 Prefill 队序与阶段交接

**状态：本轮已按计划实施，实际函数归属见上方实施记录；以下保留设计伪代码，不可直接粘贴为 Java。**
本节落实本轮用户提出的方向：Prefill 入队后顺序固定，允许取消、超时和抢占删除，
减少重复登记及跨阶段状态操作。它取代下文历史方案中“保留 Prefill 优先级插队和旁路额度”的要求；
历史代码量、测试结果不证明本计划已实现。全局调度优先级、请求原始序号和绝对期限继续保留。
本计划最初只作为文档评审；用户随后授权实施并要求 UT 先行，实施结果单独记录在顶部。

### 1. 行为范围与明确取舍

- QUEUE 的 Prefill 队列保留，用于等待 batch 聚合和发送容量；DIRECT 不增加队列，Encoder 保留原交付路径。
- 新路由追加到尾部；这里的顺序是 **Prefill 接收顺序**，不是请求最初进入全局队列的序号。
  并行选路可能后来的请求先完成，不能按原序号将它插到已经接收的路由前面。
- 发送只选择当前队首的连续前缀。遇到不能发送的成员时停止扩展，不能跳过它继续取后面的成员。
- 取消、超时、抢占可以移除准确的路由，剩余成员相对顺序不变；不能靠自动移除再入队恢复旁路行为。
- 抢占成功移除旧路由后，仍有效的请求通过普通调度入口重排，保留全局原序号和截止时间；
  新路由再次到达 Prefill 时按新的接收顺序追加，不恢复旧 Prefill 位置。
- 进入发送准备/提交的路由能否移除，以现有交付状态和准确资源身份为准；不能当普通排队项直接删除。
- 优先级在全局调度与抢占选择中仍有效；后到的高优先级请求不直接插入 Prefill 队首。
  此规则会改变现有 Prefill 优先级排序及有限旁路策略，应迁移对应配置说明和扩展契约，不能称为纯等价重构。
- 接受队首阻塞：A 等 D1 时，即使 B 对应 D2 空闲，B 也等待。需测量吞吐、队列等待及尾延迟，
  不承诺性能无回退；不得用未说明的抢占或轮询恢复旧行为。

### 2. 阶段最小契约和事实归属

| 阶段             | 输入                            | 完成与交接                                                         | 失败责任                                                   |
| ---------------- | ------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- |
| 全局等待         | 有效请求、优先级、原序号、期限  | 启动唯一一次选路；队列不能再次消费它                               | 调度器决定等待、重试或结束                                 |
| 选路及资源登记   | 当前 RoutingContext、策略       | 完整 RouteResult 交给 Prefill 接收，或原有 DIRECT/Encoder 交付路径 | 选路流程清理未交出的部分资源；清理失败不能伪装成可立即重试 |
| Decode 抢占分支  | 本次请求、目标 Decode、资源需求 | 返回准确预留；选路接收或释放后，该段工作才全部结束                 | 抢占处理自身副作用；晚到预留由结果接收方释放               |
| Prefill 排队投递 | 已完成发布的路由                | 按连续前缀移交发送组件                                             | 投递处理局部失败并向请求生命周期报告，不自行发起全局重试   |
| 发送及执行反馈   | 准确交付和资源关联              | 按协议反馈更新执行、释放事实                                       | 未知发送结果保留跟踪；不能直接当作未发送重排               |

| 事实                                    | 唯一修改者                         | 其他类的限制                                                 |
| --------------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| 请求取消、当前路由及本次 RoutingContext | RequestContext                     | 不再复制一套请求有效/已完成布尔状态                          |
| 是否可参与全局选路                      | 全局调度队列及其调度入口           | 容量等待索引只能使它重新可调度，不能直接启动选路             |
| 是否属于 Prefill 待投递队列             | PrefillEndpoint 锁内的队列成员记录 | 投递器只能通过 Endpoint 操作成员，不能另建一份待发送请求集合 |
| 资源是否仍登记                          | 对应 Endpoint 的准确资源记录       | 持句柄调用释放，不用响应 Future 推断已释放                   |
| 交付是否仍需跟踪                        | DeliveryDispatcher 及现有交付记录  | 全局调度器不能消费发送失败并自行重发                         |

索引、统计和快照可以保留，但不能成为另一套可独立推进请求的登记。每个索引必须能从事实记录
说明更新与删除时点。交接中允许短暂存在两份引用，不允许两个消费者同时获得推进同一条路由的权力。
请求响应完成与资源清理完成是不同事实，不应为减少状态而合并。

### 3. 函数落点：固定顺序和连续前缀

以下路径均相对 `flexlb-balance/src/main/java/org/flexlb/balance/`。

#### 3.1 `endpoint/OrderedPrefillRequestQueue.add/remove`

当前按 comparator 排序，且保存 bypassedHead。改为按实际 add 成功顺序追加，删除旁路字段及其更新。
可采用带准确身份索引的双向链表，使追加、按身份删除为 O(1)；不强制新增 Node 类，优先复用现有 Entry。
身份索引只指向链表成员，不另存请求生命周期。优先级计数如仍被容量统计使用则保留。

```java
// 均由所属 PrefillEndpoint 的锁保护。
add(route):
    validateAllInputsBeforeMutation(route)
    if identityIndex.contains(route): return false
    entry = entryFor(route)
    appendToTail(entry)
    identityIndex.put(route, entry)
    updateCountsAndInvalidateSnapshot()
    return true

remove(exactRoute):
    entry = identityIndex.get(exactRoute)
    if entry == null: return false
    unlink(entry)                    // 其他成员不移动
    identityIndex.remove(exactRoute)
    updateCountsAndInvalidateSnapshot()
    return true
```

联动修改 `PrefillRequestQueue`、`QueueExecutionSettings.queueOrder` 和
`PrefillEndpoint.enableQueue`：Prefill 不再复用全局 priorityOrdering 决定插队。
删除 `candidatesAfterBlockedHead`、`recordDispatch` 及调用方；迁移 Factory 和 Snapshot.order 的实际调用方，
不能给预测模块提供与实际队列顺序不同的 comparator。若预测确需顺序键，只用本次 Prefill 接收顺序。
自定义队列仍可替换存储，但必须通过追加、精确移除、连续前缀契约测试；旧的重排序实现需显式迁移。

#### 3.2 `QueuedRequestScheduler.scheduleQueueOnce/scheduleGroup`

删除 Decode 容量阻塞后再次调用 scheduleGroup 的整个替代选择分支。
`scheduleGroup` 删除 bypassedHead 和仅为旁路使用的 selectionStillAllowed 参数；
保留取消、准确成员、发送容量和 Endpoint 代际校验，不能把这些校验当作旁路逻辑一起删掉。

```java
scheduleQueueOnce(queue):
    snapshot = endpoint.captureQueue(maxRequests) // 已发布的前缀
    if snapshot.empty(): return NO_WAIT
    if snapshot.head.expired():
        removeExpiredRequest(snapshot.head)
        return NO_WAIT
    // 不再读取 routingContext 来等待 "Route commit in progress"
    // 前提是 4.1 的发布与可消费性保证已实现并通过测试。
    return scheduleGroup(queue, snapshot, evaluator, constraints)

scheduleGroup(queue, snapshot, evaluator, constraints):
    selected = grouping.select(snapshot.items, constraints, predictor)
    require selected.items is a contiguous prefix of snapshot.items
    if collectionWindowNotReached(selected):
        return waitUntil(min(collectionDeadline, requestDeadline))
    result = processSelection(selected, ORDERED_PREFIX)
    if result.capacityBlocked(): return waitFor(result.capacity, deadline)
    if result.failed(): return handlePreparationFailure(result)
    return NO_WAIT
```

这段伪代码省略现有容量复查、预测异常处理及通知异常收尾，实施时必须保留其行为。
`planner/BuiltinGroupPlanner.select`、自定义 `GroupPlanner.select` 必须只输出连续前缀。
空选择不能导致无期限空转；保持现有无法规划的失败/等待语义。某个成员过期或准备失败可以精确移除，
但需要重新检查当前前缀，不能把旧候选列表的剩余项直接作为新的 batch 提交。

#### 3.3 `PrefillEndpoint.ownsSelectionLocked/commitDelivery`

当前 ownsSelectionLocked 主要检查成员仍在队列且未过期。固定队序后还必须核对前缀，防止旧快照越过队首。

```java
ownsSelectionLocked(items, now):
    require endpointLockHeld()
    if items.empty(): return false
    cursor = activeIndex.iterator()
    for route in items:
        if !cursor.hasNext() || cursor.next() != route: return false
        if route.expired(now): return false
    return true

commitDelivery(delivery):
    with endpointLock: // 所有返回和异常均释放锁
        if !ownsSelectionLocked(delivery.items, now): return notCommitted
        validateExistingDeliveryAndCapacityRules()
        // 同一锁内：移出待投递队列，写入现有交付记录。
        for member in delivery.items:
            activeIndex.remove(member.route)
            member.commit(...)
        recordMutation()
        return committedDelivery
```

联动 `scheduler/DeliveryDispatcher` 和 `scheduler/Delivery`：删除 bypassedHead 透传和旁路专用复查；
保留 ORDERED_PREFIX 的失败边界、已取得资源回滚和结果未知跟踪。
其他路径仍使用的 ALL_SELECTED 不因本计划而全局删除。

### 4. 函数落点：发布完成后才能消费

#### 4.1 `QueuedRequestScheduler.enqueueRoute` 与 `RequestContext.publishRoute`

当前是 Endpoint 插入成员后再 publishRoute，消费者因此可能看到未发布路由。
目标是将两步封装为一次本地接收操作：对队列消费者而言，两步一起可见；取消不能在中间修改请求。
推荐候选锁方案为 **RequestContext → PrefillEndpoint**，复用现有请求锁与 Endpoint 锁，不加第三个全局锁。
下列 `acceptQueuedRoute` 是拟新增的协调入口名，不是当前已存在 API。

```java
enqueueRoute(route):
    accepted = acceptQueuedRoute(exactRoutingContext, route)
    if !accepted: return existingRejectionOrClosedResult()
    // 调用方记录 routeAccepted 后，在锁外执行 notifyQueuedRoute。
    return success(route)

acceptQueuedRoute(exact, route):
    synchronized route.ctx:
        with route.prefillEndpoint.lock: // 所有返回和异常均释放锁
            validateCurrentRoutingAndEndpoint(exact, route)
            if invalid: return false
            // add 异常必须保持成员未加入，沿用队列接口现有契约。
            if !addQueuedLocked(route): return false
            try:
                if !ctx.publishRoute(exact, route):
                    removeExactUnpublishedMemberLocked(route)
                    return false
            catch failure:
                removeExactUnpublishedMemberLocked(route)
                throw failure
    return true
```

这里的 remove 只撤销本次未发布的本地入队，不替代路由全部资源的回滚。失败返回后仍由
`executeQueuedRouting` / `finishUnpublishedRoute` 负责准确的未交出资源；接收成功后禁止该路径再释放。
`routeAccepted` 必须在任何通知或响应回调之前记录成功，不能因通知抛异常触发未发布路由清理。
实现让 accept 返回成功、由 executeQueuedRouting 记录 routeAccepted 后再通知，
通知异常沿用已接收路由的故障处理，不得向外伪装成接收失败。

**锁方案实施前置检查：** 核对所有 Endpoint 持锁调用 RequestContext 的路径，尤其交付提交、
抢占、关闭和 worker 上报；若存在 Endpoint → RequestContext，必须先消除反向嵌套再采用此方案。
不能仅凭上述伪代码宣称无死锁。publishRoute 必须只修改本地请求事实，不执行回调、通知、RPC 或资源释放；
还要验证其异常不会留下“已发布但返回失败”的部分结果。若无法满足，须修订交接实现，
不能先删除消费者的发布中检查。本节给出具体候选方案与验收门槛，不声称锁审计已完成。

#### 4.2 `QueuedRequestScheduler.executeQueuedRouting/finishDecision`

```java
executeQueuedRouting(exact, selection):
    acquired = null
    routeAccepted = false
    try:
        acquired = selection.get()
        if !acquired.success(): return acquired
        outcome = acceptQueuedRoute(exact, acquired.route)
        routeAccepted = outcome.accepted
        if routeAccepted: notifyQueuedRoute(acquired.route)
        return outcome
    finally:
        if !routeAccepted: finishUnpublishedRoute(exact, acquired)
        completeThisRoutingWorkUsingExistingLifecycle()

finishDecision(entry, outcome):
    if outcome.accepted:
        removeExactGlobalEntry(entry)
        // 此时即使清理登记稍晚，也不得再次消费 entry。
    else:
        applyExistingWaitRetryOrFailurePolicy(entry, outcome)
```

以上保留当前 Encoder 等分支；不把通用方法强行改成只有 Prefill。
全局队列引用在任务完成时才删除并不等于双重消费：从启动选路到收尾必须一直不可再次取出。
因此不能先删 inFlight 再希望通过 Prefill 队列判断“是否还在选路”。

### 5. 函数落点：抢占、取消和旧路由重排

`DefaultDecodePreemption.startPreemption` 保持当前两个分支：引擎取消与本地排队预留替换，
不沿用历史 RPC/RETURN 的描述。抢占选择可以按优先级选目标，但移除目标不能改变剩余 Prefill 队序。

```java
// 对应 DefaultDecodePreemption 的本地路由抢占与 Endpoint.removeQueuedLocked。
removeQueuedRouteForPreemption(exactRoute):
    useExistingRequestAndReservationIdentityChecks()
    underEndpointLock:
        if exactRoute has entered delivery: return cannotRemoveAsQueued
        removeExactQueuedMember(exactRoute)
    completeExistingRouteResourceCleanup()
    // 清理成功且请求有效，才通过 completeRouting/finishRoutingRequest 重排。

// 对应 QueuedRequestScheduler.resumeRoute。
resumeRoute(exact, result):
    if result.failed: applyExistingPreemptionFailurePolicy()
    else if requestCancelledOrExactRoutingObsolete(exact):
        release(result.exactReservation)
    else:
        selectRemainingRolesAndHandOff(exact, result.exactReservation)
    // finishSchedulingTask(PREEMPTION) 只能在上述处理结束后执行。
```

`AbstractRequestScheduler.cancel` 仍只推进请求取消及现有关闭工作，不恢复 reconcile 调用。
`QueuedRequestScheduler.completeRouting` / `AbstractRequestScheduler.finishRoutingRequest` 是旧路由清理后
进入 `scheduleRequest` 的协调位置；释放失败不能重排，取消不能重排，旧回调不能修改新 RoutingContext。
不新增 Prefill 自己的“失败后重新选路”入口。

### 6. 函数落点：事件、等待与关闭各守范围

`QueuedRequestScheduler.onEndpointChanged` 当前同时操作全局容量等待和 Prefill 执行索引。
迁移目标是分别通知两个职责，而不从外层设置 ready/running/nextCheck。
下列 `prefillDelivery` 表示拟独立的投递职责，最终类名在实现时确定；不为此新建第二份请求队列。

```java
onEndpointChanged(endpoint):
    globalScheduling.capacityChanged(endpoint)
    prefillDelivery.endpointChanged(endpoint)

prefillDelivery.endpointChanged(endpoint):
    if endpoint is Prefill: signalItsQueue(endpoint)
    // Decode 容量由当前队首已有的准确资源订阅触发，不扫描所有 Prefill 队列。

runQueueScheduling(queue):
    decision = scheduleQueueOnce(queue)
    installOrReplaceLocalWait(decision)
    recheckAfterSubscription() // 不能漏掉登记期间已恢复的容量
    arrangeNextRunOrDeadline()
```

迁移现有 `schedulePrefillQueues/runQueueScheduling/finishQueueScheduling` 和
`PrefillQueueScheduler.observeWait/waitCurrent/stopAndDrain` 时，ready/timed 索引、任务额度和订阅清理
一起留在投递职责内部；外层不读取内部状态决定是否运行。ready/running 的具体压缩另行证明，
本计划不预设把两个可同时为真的标记合成枚举，也不以固定线程数替代提交额度。

`PrefillEndpoint.isQueueWaitCurrent` 删除“队首阻塞时扫描后续替代项”的分支。
容量等待主要核对准确队首是否还在、是否过期以及所等资源；batch 聚合等待仍需关注尾部新到达、
队列/输入变化和窗口到期。不能机械删除所有版本：它们也保护预测与准确选择，不只服务旁路。

`waitingRequests` 与 `preemptionQuotaWaiters` 的 OR 等待合并不纳入本次必做项。
当前一个请求对应一个容量范围，直接塞入名额 key 不成立。先保留现有去重、通知和优先级保证；
只有能减少登记/删除协议且通过名额释放竞态 UT 的替代方案才迁移。

关闭继续分两步：停止各阶段接收新工作；等待已接收工作处理完交接、资源回滚和晚到结果，再关闭其执行设施。
`pendingPreemptions` 不能仅用抢占服务 Future 替代。相关新增 UT：
`GlobalQueueProgressTest.closeWaitsForLatePreemptionReservationReleaseAfterItsFutureCompletes`，
验证资源释放阻塞时 close 不提前结束；该测试已经通过，不代表本节其他改造已有测试覆盖。

### 7. 删除清单、实施顺序和验证

| 顺序 | 改动与真实删除目标                                                                    | 验证位置及关键断言                                                                                                                               |
| ---- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | 固定追加顺序，迁移 comparator/预测顺序、自定义队列契约                                | OrderedPrefillRequestQueueTest、PrefillRequestQueueContractTest：高优先级后到仍在尾部；删除中间成员不重排；相同字段不同身份不混淆                |
| 2    | 删除替代选择、bypassedHead、candidatesAfterBlockedHead、recordDispatch 及专用回调参数 | RequestExecutionSchedulingTest、GroupPlannerTest、DeliveryDispatcherTest：队首阻塞不发后项；连续前缀；旧快照不能越过当前队首；准备失败不重复发送 |
| 3    | 发布与可消费性原子交接，通过后删除 Route commit in progress 消费者等待                | RoutingContextTest、RequestExecutionSchedulingTest：插入/发布/取消屏障交错；发布拒绝无残留；通知异常不撤销已接收资源；锁顺序无反向嵌套           |
| 4    | 投递事件与执行索引归投递职责，外层删除对其 ready/running/定时字段的直接操作           | SchedulerExecutionTest、QueuedSchedulerStopRaceTest：运行期间信号不丢、batch 无事件仍到期、拒绝归还额度、关闭解除订阅                            |
| 5    | 核对抢占移除、重排及晚到结果，删除任何新增的阶段外重试入口                            | GlobalQueueProgressTest、抢占与撤路测试：剩余队序不变；清理后重排；取消不重排；释放失败不重排；close 等待晚到资源处理                            |

每一步先补能区分新旧行为的 UT，再改实现、删旧分支。优先级插队/旁路旧测试按明确的新策略迁移，
不能仅删除失败断言；自定义 GroupPlanner/队列违反前缀契约时应清楚拒绝，不能悄悄改写结果。
不要求一次性重写整个调度器；固定顺序与旁路清理可先落地，发布交接的锁审计未通过前保留旧安全检查。

实现完成还需通过相关模块测试、全 reactor Spotless 和成员排序，并更新稳态架构文档。
性能对比至少包含：单 D 热点、多 D 中一个被阻塞、长短请求混合、高优先级持续到达、频繁抢占及 batch 低流量窗口。
记录吞吐、P95/P99 队列等待、预测/候选扫描次数与 CPU；没有测量前只报告删除了哪些协议，不能报告性能改善。
代码量统计必须包含迁移出的类及接口，单纯搬移不计为净减少。

## 2026-11-10 后续收尾：固定回调与拒绝清理

本批基线：`/var/folders/hp/trr4w7bx56bctwg22ymmsph00000gn/T/queued-cleanup-_3tfy2k0/baseline`。
本批只改 QueuedRequestScheduler 的生产实现，1840 → 1836 行，净减 4 行；新增 UT 33 行。
累计本系列自身生产改动净减 31 行，其他工作区变更不计入。行数收益小，不将本批称为整体架构治理完成。

- 删除仅绑定同一个内部实现的 tryPreemption 字段及两层 recover/BiFunction 传递。
  内部 allowDecodePreemption 明确哪些入口允许开始抢占；自定义选址 Supplier、Router、
  DecodePreemption 及队列接口仍保留。抢占续接仍在原任务释放名额后注册。
- 普通规划和 Worker 队列提交共用 rejectedDuringStop，只忽略“调度器已停、执行器已关闭”的
  RejectedExecutionException。其他错误继续上报，不改变抢占结果续接在停服后的清理路径。
- 新 UT rejectedPlannerReturnsItsSlotEvenWhenQueueRemovalFails 使用拒绝执行器与首次 remove
  抛错的自定义队列。旧实现上真实 schedule/关闭路径结束后，inFlight 仍为 1；新增断言失败。
  修复将未被执行器接纳的规划名额归还放在 finally 中，即使请求失败清理又抛错也会归还。
  请求清理错误仍传播给调度驱动，不通过吞掉异常通过测试。
- 本批保留 scheduleRequest → enqueue 入口。尝试合并时发现撤路、取消与重排测试在 enqueue 边界
  注入成功、拒绝和抛错；直接改为拦截 scheduleRequest 会绕过原有前置检查，并弱化这些测试。
  因此撤回本批的入口合并，不为三行收益批量替换成更弱的测试断言。此项需迁移到真实
  SchedulingQueue 注入后另行实施，不能将其列为本批删除成果。
- 容量挂起入口改为 parkOrRetry，返回值只回答“是否需要立即重试”；调用处不再取反，也不再
  同时传入 result 和可以从 result 得到的 blocker。成功/拒绝/关闭共用移除队列成员的步骤。

验证：先在未改生产代码的副本运行 46 个用例，只有新增名额断言失败（expected 0, actual 1）。
日志 `/tmp/queued-cleanup-baseline.log`。第一次修改后 313 用例中新增 UT 通过，但既有
schedulerInvalidationOnlyRemovesTheExactGlobalQueueEntry 出现一次 Mockito UnfinishedStubbingException；
日志 `/tmp/queued-cleanup-tests.log`。未修改该测试断言，不将此次执行记为全绿。
入口合并尝试因旧 enqueue 测试编译失败而撤回，对应 `/tmp/queued-cleanup-final.log`，不是完成版本。
最终版本执行相同 13 类回归：313 用例，零失败、零错误、零跳过，退出码 0；日志
`/tmp/queued-cleanup-verified.log`。上述 Mockito 错误未在此次复跑出现，原因尚未确认，未宣称修复了它。
生产文件和新增 UT 与被测副本逐字节一致。主工作区全 reactor Spotless、成员顺序与定向
diff 空白检查均通过，格式检查日志 `/tmp/queued-cleanup-spotless.log`。本批未做性能基准。

## 2026-11-10 再次收敛：纠正“修复完成但整体增行”

前一批净增 7 行只证明修复与局部整理完成，没有达到整体精简目标。本批以独立工作区基线
`/var/folders/hp/trr4w7bx56bctwg22ymmsph00000gn/T/queued-simplify-w6ij8mho/baseline`
继续实施；不覆盖期间其他 agent 对交付上报边界的修改。下文前一批的行数与测试记录保留为历史。

实际删除的协议：

- `finishPlanning` 与 `finishPreemption` 合并成 `finishSchedulingTask`：归还对应任务名额后，
  重试条件、quota waiter 唤醒、驱动通知只执行一套代码。抢占完成 Future 仍在锁外完成。
  内部 SchedulingTask 枚举只区分本次完成哪种任务，不是请求的新状态，也没有新增任务登记。
- `selectAndPublish` 复用已有 planRoute 的开始选路检查，删除一份重复准入分支。
- removeRequest 直接使用 ReentrantLock 的重入语义，删除手工判断本线程是否已持锁及对应分支。
- 抢占恢复直接使用既有 `router.select(request, group, endpoint, reservation)`。
  删除私有方法间的 Function 参数传递，并删除 RequestRouteSelector.selectionAfterPreemption
  这个只生成闭包、没有独立行为的工厂。自定义 Router/策略仍由原 select 入口调用，
  准确预留释放、取消校验和策略内资源登记没有迁移或删除。

本批不是将等待登记和任务退出完全合成一个事务：仍保留已由 UT 证明必要的
retryPreemptionWaiters 对运行中任务的保护。此次消掉的是两种任务各自维护公共收尾的重复实现，
不宣称所有等待索引或全部异步协调都已经消除。

| 本批生产文件           | 修改前 | 修改后 |  净变化 |
| ---------------------- | -----: | -----: | ------: |
| QueuedRequestScheduler |   1861 |   1840 |     -21 |
| RequestRouteSelector   |    654 |    641 |     -13 |
| 合计                   |   2515 |   2481 | **-34** |

计入前批自身改动的 +7，两批自身生产改动累计 **-27 行**；不把其他并行修改计入收益。
没有新增生产文件，本批没有修改测试断言，也没有压缩格式或注释获得减量。
测试沿用先前已复现竞态的 UT，先在本批基线上运行四类规划、取消、停服、抢占测试通过；
修改后再运行相关回归。最终结果在本节补齐，不复用前批验证数字。

本批最终验证：13 个测试类、312 个用例，零失败、零错误、零跳过；日志
`/tmp/queued-simplify-final-tests.log`。两份生产源码与隔离验证副本逐字节一致。
主工作区全 reactor Spotless 通过（`/tmp/queued-simplify-spotless.log`），成员顺序、定向
diff 空白检查及技能校验通过。本批没有重新运行全部模块行为测试或性能基准。

## 2026-11-10 合并后的执行依据

本节合并本文原审核与 lifecycle-simplification-development-manual.md 顶部的 Q1–Q5。
下面保留的 2026-11-09 分析是历史证据；方案冲突时以本节及本轮实测结果为准。
使用合并后的 [flow-complexity-review](../../skills/flow-complexity-review/SKILL.md)，
同时遵守仓库开发、设计及实现技能。工作区基线保存在
`/var/folders/hp/trr4w7bx56bctwg22ymmsph00000gn/T/queued-final-zdquyjp_/baseline`。
开始时 QueuedRequestScheduler 为 1884 行；所有净变化须相对此基线计算，不使用混有其他修改的 HEAD diff。

### 保留的契约

- 自定义 scheduler、选址策略、SchedulingQueue、PrefillRequestQueue、GroupPlanner 与 PrefixPolicy。
- 有限扫描、候选缓存和局部重算；原顺序、绝对期限及旁路额度。
- 同一 RoutingContext 跨越抢占等待与恢复；先释放规划名额，再注册已启动抢占的续接。
- 未成功抢占当前仍结束请求，不在本次重构中改成无限重试。
- 取消不等于任务退出；inFlight 和 pendingPreemptions 必须保留到各自收尾。
- 物理入队及路由接收后，通知失败不得撤销已交出的资源；发送开始后不得按未发送撤路。
- ready/running 可以同时为真，表示执行期间收到新通知；不能合成互斥状态丢事件。

### 本轮实施范围和验收顺序

| 项目            | 合并后的具体决定                                                                                                                                                | UT 与删除目标                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| A：抢占等待     | 保留容量变化或本地名额释放均可唤醒的语义。quota waiters 是第二种唤醒条件的索引，不是另一份请求计数。名额释放不能清掉仍在规划的 waiter；其规划结束后再发布重试。 | 固定名额释放发生于容量 park 前的屏障交错；一个请求 blocked count 为 1。删除通知被过早消费的交接。                          |
| B：旁路         | head 和阻塞原因作为局部动态条件传递，提交前仍检查原 head 是否恢复。                                                                                             | 保留队头恢复、换头、失败不消耗旁路额度及 ALL_SELECTED UT；删除 bypass/blockedHead、BooleanSupplier 身份和 finally 清字段。 |
| C：一轮队列执行 | 名称明确表示执行调度及返回等待信息；过期移除与准备失败使用不同入口。                                                                                            | 通知失败仍完成取消/失败回调；删除 null+alreadyRemoved 控制业务类型的协议，不新建计划执行框架。                             |
| D：路由完成     | 撤路特有步骤留在 Queue；公共的结束原因优先级及后续动作协调只保留一份。                                                                                          | 释放失败不得重排、保留首个取消原因、旧 delivery 不影响新路由；删除复制的公共完成分支，不搬迁整个撤路方法。                 |
| E：预算/扫描    | 核实真实成本、能力和语义，本轮不改变线程池公平策略或建立反向索引账本。能保持语义的局部算法改善可实施。                                                          | 不把两个提交额度误称为线程超限，不把 queuedRequestCount 替换成全局队列长度；记录确定的约束与未测量项。                     |

### 对上一份方案的修正

旧 Q1 建议将容量等待与名额等待改成二选一，会丢掉当前“容量恢复也能推进名额等待者”的行为。
本轮不执行该语义变更，也不为解决局部交错新增 RoutingPlan 的第三种业务状态。
解决点应是：唤醒索引和任务退出之间的交接，以及统计时去重。
startOrWaitForDecodePreemption 的命名必须明确有启动或登记等待的效果；返回 Runnable
仅用于在原规划任务退出后订阅已启动操作，不宣称它是尚未启动的抢占计划。

旧 Q4 的统一 CPU 额度只是候选设计，尚无公平性和时延证据，不作为本轮强制改造。
旧 Q5 的失效节点扫描可先将每个 route 遍历失效集合改成固定角色身份查询；不为此增加新的
endpoint→request 索引。若无法保持精确 endpoint 身份就保留现状并说明依据。

### 已落地的代码与证据

- A：新增参数化 UT `GlobalQueueProgressTest.preemptionSlotReleaseCannotBeLostBeforeCapacityParking`。
  `false` 分支等待实际挂起后检查唯一请求计数；`true` 分支阻塞在路由操作收尾处，使抢占名额
  先释放、随后才进入容量 park。旧实现两个用例均失败：计数为 2，以及未再次准入。
  修复后两者通过。名额等待者仍在 inFlight 时保留登记；finishPlanning 移除 inFlight 后，在
  有空闲抢占名额时统一唤醒。重新开始抢占前清除旧唤醒登记，避免它污染下一次尝试。
  没有新增请求状态、版本或任务集合。blockedRequestCount 对两种唤醒索引的交集去重，
  代价是统计时遍历 quota waiters；没有为指标再维护一个同步计数。
- B：删除 bypass、blockedHead、PrefillQueueScheduler 的 BooleanSupplier 身份及清空字段的
  finally。局部回调仍由 Dispatcher 在准备和提交阶段动态检查，保留队头恢复时中止旁路的行为。
- C：`scheduleQueueOnce` / `scheduleGroup` 明确执行效果，`NO_WAIT` 只表示没有下一项等待条件。
  `removeExpiredRequest` 与 `commitBoundary` 分别表达超时和准备失败，通过 notifyRemovedRequest
  共享通知错误处理；删除 failure=null / alreadyRemoved 的混合参数协议。相关旧 UT 只迁移入口，
  不改异常和通知断言。
- D：基类和 Queue 共用 `finishRoutingRequest` 的结束优先级、取消/重排及异常收尾。
  Queue.completeRouting 保留撤路特有资源步骤；删除其重复结束处理和对 Context 同步方法的外层
  重复锁。没有宣称整个 completeRouting 已合并，也没有新增清理结果状态或生命周期接口。
- E：失效 endpoint 使用身份集合，RouteResult.usesAnyEndpoint 遍历自身固定数量的角色，
  将 O(活动请求数 × 失效 endpoint 数 × 角色数) 改为 O(活动请求数 × 角色数 + 失效 endpoint 数)。
  仍然遍历活动请求，没有新建反向索引。只验证算法及行为，未运行端到端性能基准。
  线程池仍限制实际并行数；全局规划、Worker 队列、未完成抢占分别有不同含义的额度。
  本轮不把这些事实合成一个计数，不改变公平策略，不声称解决了全部调度预算问题。

### 验证记录与版本范围

1. 修正测试夹具中的 Mockito 嵌套配置后，旧实现上运行两个新用例：2 failures、0 errors。
   权威复现日志：`/tmp/queued-final-red-corrected.log`。第一版夹具失败不作为竞态证据。
2. 修改后第一批 7 个测试类共 204 用例通过，包含新用例，以及真实旁路、撤路、通知失败与资源
   身份测试；日志 `/tmp/queued-final-green.log`。
3. 隔离工作区执行 `./mvnw -pl flexlb-balance -am test -q`，退出 0：common 453、cache 201、sync 2693，
   合计 3347 用例，失败、错误、跳过均为 0。日志 `/tmp/queued-final-regression.log`。
   隔离目录为基线同级的 work，避免其他并行构建干扰。该次不是 API/mock-engine 的全量行为测试。
4. 完整回归后还简化了等价的结束原因条件，并移除撤路前置条件已保证的空值判断与重复锁；
   最后补跑相关路径和自定义 scheduler 的 11 个测试类、309 个用例，退出 0，失败、错误、
   跳过均为 0；日志 `/tmp/queued-final-last-regression.log`。这 309 个用例与第 3 项有重叠，
   不能相加声称独立用例数量。四个修改后的生产文件均已逐字节核对与被测源码一致。
5. 主工作区全 reactor `./mvnw spotless:check -Pspotless-check` 通过，日志
   `/tmp/queued-final-spotless.log`。合并后的技能及旧技能兼容入口均通过 quick_validate。
   修改文件的成员顺序检查和定向 `git diff --check` 均通过。

### 代码量：本轮没有达成总体减行

| 生产文件                 | 本轮前 | 本轮后 | 净变化 |
| ------------------------ | -----: | -----: | -----: |
| QueuedRequestScheduler   |   1884 |   1861 |    -23 |
| AbstractRequestScheduler |   1341 |   1367 |    +26 |
| CapacityWaitQueue        |    150 |    154 |     +4 |
| RouteResult              |    218 |    218 |      0 |
| 合计                     |   3593 |   3600 | **+7** |

新增行为 UT 为 112 行，其余三个测试文件仅迁移调用入口，行数不变。文档与技能不计入生产代码。
Queue 中部分后续逻辑移至基类，已计入上表，不能把 -23 当作净收益。实际收益是删除两个临时
状态字段、一组设置/清空协议、一套重复结束裁决及 null/boolean 业务分支，并修复已复现竞态。
名额释放与规划退出的协调增加了代码，抵消部分删除；这不是“整体复杂度已彻底解决”的证据。

剩余结构工作包括规划/队列执行预算的明确契约，以及是否值得消除活动请求全表扫描；需有
公平性、资源索引覆盖和测量依据后再做。不要靠删必要清理、缩短格式或隐藏新增文件补足减行目标。
本轮未提交、未推送。

---

## 2026-11-09 原始审查证据

日期：2026-11-09。范围：当前工作区的 `QueuedRequestScheduler` 及其直接调用链。
本轮仅分析和编写交接材料，没有实施下述重构，也没有为本轮审查执行行为测试。
工作区存在其他并行修改；接手时以方法名重新定位，不把本文代码摘录当作可直接覆盖的版本。

## 目标和判断标准

沿用此前对 AbstractRequestScheduler 的要求：第一次读代码，能看懂方法做什么、返回值代表什么、
状态保存什么，以及资源和异步任务由谁继续处理。不要只换一批抽象名称，也不要把大类拆成多个文件后宣称精简。
应删除隐式传参、重复判断和重复协调；必要的资源身份检查、取消处理和并发交接必须保留。

主要源码：[QueuedRequestScheduler.java](../../flexlb-balance/src/main/java/org/flexlb/balance/schedule/QueuedRequestScheduler.java)。
以下问题区分“已确认的实现/可读性问题”和“需要验证的行为选择”；没有把可读性问题直接认定为生产故障。

## 实际流程

1. `scheduleRequest → enqueue`：进入全局排序队列；恢复请求复用原序号。
2. `scheduleAt → planningCandidates → submitPlan → place`：领取选路名额，执行选路。
3. `planRoute → executeQueuedRouting`：选择结果成功后进入 Prefill 队列，或提交 Encoder 路由；阻塞时可能发起 Decode 抢占。
4. `finishDecision`：决定移除全局队列成员、立即重试或登记容量等待。
5. `schedulePrefillQueues → runQueueScheduling → decideQueueScheduling → decideGroup`：选择 Prefill 队列前缀并提交投递；阻塞时订阅容量或定时再检查。
6. `resumeAfterPreemption → resumeRoute`：抢占结果到达后续选，成功发布或清理晚到资源，最后释放抢占名额。
7. Endpoint 不可用时撤回未发送路由；取消/关闭处理队列成员、已接纳任务和异步结果的清理。

全局选路和 Prefill 投递是两个不同阶段，不能合成一个模糊的“调度中”。

## 1. 决策方法实际在执行投递、删除和取消

定位：`decideQueueScheduling`、`decideGroup`、`PrefillQueueScheduler.processSelection`、`commitBoundary`。

代码摘录：

```java
if (head.requestExpired(nowMs)) {
    queueScheduler.removeAndNotify(head, null, false);
    return PrefillQueueScheduler.Decision.NO_ACTION;
}
```

```java
boundary = queueScheduler.processSelection(
        selection.items(), reason, evaluator, prediction,
        bypassedHead, Delivery.PrefixPolicy.ORDERED_PREFIX, queueScheduler);
return queueScheduler.commitBoundary(first, boundary, retryVersion);
```

沿调用方确认：`processSelection` 构造 Delivery 并调用 `dispatcher.submit`；
`commitBoundary` 可能报告路由失败、删除队列成员并通知请求失败。
因此返回 `NO_ACTION` 不表示什么都没做，也可能表示刚删除一个过期请求，或者已完成一次处理。

**判断：确认的语义问题。** 方法名像纯决策，返回值却只是“执行完这一轮后是否等待”的信息。
接手时优先把入口表达为“执行一轮 Prefill 调度”，把等待结果表达清楚；已有 GroupPlanner 可以继续承担候选选择。
不要为了命名一致再造一套完整的计划对象和执行框架。

## 2. 队首绕行条件通过临时成员字段隐式传递

定位：`decideQueueScheduling`、`PrefillQueueScheduler.getAsBoolean`。

```java
queueScheduler.bypass = head;
queueScheduler.blockedHead = decision;
try {
    // 调用 decideGroup，再把 queueScheduler 当 BooleanSupplier 传到投递路径。
} finally {
    queueScheduler.bypass = null;
    queueScheduler.blockedHead = null;
}
```

```java
public boolean getAsBoolean() {
    return bypass == null || !capacityAvailable(bypass, blockedHead.unavailable());
}
```

**判断：确认的隐式传参问题，优先处理。** 读者必须追踪先写字段、传整个对象、再由回调读字段的过程，
才能知道这是“只有原队首仍被 Decode 容量阻塞时，才允许绕行”。`bypass` 和 `blockedHead` 还必须成对设置和清空。

建议用本轮局部变量捕获原队首和阻塞原因，显式传入用途清晰的检查函数，移除这两个临时字段和整个类的 BooleanSupplier 身份。
先核对 Dispatcher 的检查时点，确保仍在真正提交前重查容量；不能把动态检查提前计算成一个 boolean。

## 3. null 和 boolean 混合表达不同业务结果

定位：`enqueue`、`park`、`finishDecision`、`PrefillQueueScheduler.removeAndNotify`。

```java
if (context.data().getFuture().isDone()) return true; // enqueue：实际没有入队
if (closed.get()) return false;
```

```java
case BLOCKED -> !park(entry, result.blocker(), observedCapacity, result);
```

`park` 在请求已移除或 Future 完成时也返回 true；所以 true 同时表示“已登记等待”和“不需要重试”。
false 则可能表示选路期间容量已经变化，应立即重试。方法名没有表达完整含义。

```java
void removeAndNotify(RouteResult item, CapacityBoundary failure, boolean alreadyRemoved) {
    if (!alreadyRemoved && !prefillEndpoint.removeQueuedMember(item)) return;
    // failure == null 表示按 DEADLINE_EXCEEDED 取消；非 null 表示准备失败。
}
```

**判断：确认的接口语义问题。** 尤其 `removeAndNotify(head, null, false)` 无法从调用处读出“删除过期请求并报告超时”。
建议分清“过期移除”和“报告已移除请求的准备失败”；公共的通知异常处理可以复用。
对 enqueue/park，先根据调用方决定使用明确命名还是明确结果类型，不强制每个 boolean 都换成新枚举。

## 4. 抢占入口返回 Runnable，但抢占已经启动；null 也可能有副作用

定位：`tryPriorityRescue`、`place`、`resumeAfterPreemption`。

```java
if (pendingPreemptions.size() >= plannerCount) {
    preemptionQuotaWaiters.add(entry);
    return null;
}
pendingPreemptions.put(entry, new CompletableFuture<>());
```

```java
var result = decodePreemption.tryPreempt(entry.context, endpoint);
if (result == null) return null;
return () -> resumeAfterPreemption(entry, operation, endpoint, result);
```

**判断：确认的命名和交接可读性问题。** `Rescue` 没说清是 Decode 抢占；返回的 Runnable 不是“执行抢占”，
而是给已经启动的异步抢占注册后续处理。null 既可能表示不适用，也可能表示已经登记等待抢占名额。

建议明确区分：是否适用、领取/等待名额、发起抢占、注册结果续接。名字应表达实际动作，例如开始 Decode 抢占，
不能只把 Rescue 换成另一个抽象词。
保留 `place` 先执行 `finishPlanning`、再注册续接的顺序；已完成 Future 的回调不能重入尚未退出的规划任务。
也不要简单把协议的所有细节移回 Scheduler；远端取消与资源释放仍由 DecodePreemption 负责。

## 5. completeRouting 一次承担多种不同后果

定位：覆盖的 `completeRouting(RoutingContext, RoutingCompletion, Response)`。

处理链包括：

```text
领取本次收尾权
→ 从 Prefill 队列移除旧路由
→ 关闭投递许可、释放路由资源
→ 提交 RequestContext 状态
→ 完成旧投递通知或重新唤醒 Prefill
→ 结束请求 / 超时取消 / 重新入队 / 执行其他后续动作
```

**判断：确认的流程可读性问题，不是这些动作都不必要。** 同一个入口同时处理正常选路完成、旧路由撤回、
恢复和失败，内部又修改 result/failure 并积累 problem，读者难以看清每种结果的后果。

先列清楚“旧路由释放成功/失败、请求已取消/过期、路由是否仍保留”的转换与动作，再让相邻代码按这一顺序执行。
优先让路由撤回处理形成完整、明确的操作；不要只把几段 try/catch 移进新类。
必须保留：释放失败后的可重试清理、首个结束原因、已发布结果不因通知失败而回滚。

## 6. 多个集合不是同一份状态，但其关系缺少集中表达

定位：字段、`planningCandidates`、`removeRequest`、`finishPlanning`、`finishPreemption`。

| 结构                         | 实际保存什么                     | 是否可直接删除                               |
| ---------------------------- | -------------------------------- | -------------------------------------------- |
| queuedEntries / orderedQueue | 请求到队列条目的索引 / 排序成员  | 不可仅因重复引用同一条目就删除               |
| inFlight                     | 已领取且结果尚未处理完的选路名额 | 请求取消后也要等任务真正交回名额             |
| waitingRequests              | 等待哪个资源域、何时允许重试     | 与任务执行状态不同                           |
| pendingPreemptions           | 等待抢占续接收尾的条目及完成信号 | close 会等待该信号，不只是资源 Future 的副本 |
| preemptionQuotaWaiters       | 等待本地抢占并发名额             | Worker 容量变化不能替代名额释放通知          |

**判断：关联管理分散是问题；不能据此把所有集合合并成一个状态枚举。**
取消时 `removeRequest` 不立即移除 inFlight/pendingPreemptions，有实际任务生命周期原因。
改动前应明确每个结构的登记、清除和关闭等待点，再删除确实冗余的记录。

同样，PrefillQueueScheduler 的 ready/running 不是必然互斥：运行期间收到新通知可以同时为 true。
如果改状态表达，必须保留“本轮执行中又有新事件”的事实。

## 7. 两类调度任务共用驱动，边界需要明确，不能靠拆文件解决

`scheduleAt` 同时驱动全局选路和 Prefill 队列投递；两者使用同一个 planners，但分别使用 inFlight 与 queueSchedulingInFlight 限流。
这是两个提交额度，并不意味着线程池会超过 plannerCount 并发运行；也不能未经验证合并成一个计数。

建议保留上层对执行预算、事件和停服的统筹，把单个 Prefill 队列的一轮处理保持在明确的职责边界内。
当前外部静态方法直接操作内部对象的 waiting、ready、running、bypass 等字段，值得收紧，但纯粹搬到另一个文件不会降低复杂度。

低优先级命名问题：`place`、`placementFailed`、`tryPriorityRescue` 和 Placement 错误文案仍未统一；
`queuedRequestCount` 实际统计活跃且尚无 delivery 的请求，包含规划/等待期间的请求，不能仅按名称理解成 orderedQueue.size()。

## 需要单独确认的行为选择

`resumeRoute` 在 result 未 committed 时调用 `failRouting` 并返回 CLOSED，而不是自动重新排队。
这是当前策略，不在本次可读性审查中判定为 bug。若要把“抢占未成功”改成“继续等待”，应先区分普通资源竞争失败、
控制失败、请求已失效，并补 UT；不能借改名或整理控制流悄悄改变。

## 推荐落地顺序和验证

1. 先清除绕行临时字段，显式传递动态条件；补队首资源恢复、队首更换、投递前重查的 UT。
2. 澄清 Prefill 一轮处理及过期/失败入口的语义；验证删除失败、通知抛错时仍能完成必要清理。
3. 澄清抢占开始与结果订阅的交接；验证同步完成、失败、名额耗尽、取消、关闭后晚到结果。
4. 再整理撤回路由的完整流程和队列状态关系；避免一次重写所有调度状态。

可复用的现有测试入口（不是已通过的本轮结果）：

- `RequestExecutionSchedulingTest`、`RequestExecutionQueueTest`、`QueuedBatchDeliveryTest`：队列投递、容量通知、绕行。
- `GlobalQueueProgressTest`、`CapacityWaitQueueTest`：排序与阻塞唤醒。
- `QueuedDecodeWithdrawalTest`、`QueuedDeliveryWithdrawalRaceTest`：旧路由撤回和竞争。
- `QueuedPublicationNotificationFailureTest`、`RequestPublicationFailureOwnershipTest`：发布后通知失败不能回收已交出的资源。
- `QueuedSchedulerStopRaceTest`、`QueueRegistrationCancellationTest`、`RequestPreemptionRegressionTest`：关闭、取消与抢占。

先读这些测试确认场景，再决定补充哪些行为测试；用屏障/锁存器构造关键交错。
每批修改后执行适用 UT、成员排序检查和全 reactor Spotless。当前仅完成阅读分析，不声称上述改动已经验证。

## 行数与交付

后续修改开始前保存受影响生产文件的基线；本工作区混有其他改动，不能拿全工作区 Git diff 冒充本轮净收益。
对迁出/迁入代码配对统计，排除 target 中的格式化副本和测试。区分搬迁、行为新增、重复逻辑、注释和换行。
交付要列出具体删掉的间接层或重复状态及验证结果，不承诺尚未测量的减行数。
