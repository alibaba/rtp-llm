# Scheduling and Request Lifecycle

FlexLB 逐条选择机器并返回路由，由调用方向引擎发送推理请求。
没有组批决策、收集窗口、EnqueueBatch 发送或 worker 内的第二层派发队列。

## 调度入口

同一个业务 request_id 按 RequestPhase 分为 ENCODER 与 GENERATION 两条生命周期。
AbstractRequestScheduler 负责注册、期限、取消、响应及资源清理，并统一执行选址、路由提交和失败回滚；
具体调度器负责启动选址以及容量不足时的拒绝、等待或抢占策略。

- DIRECT 立即进行一次选址，成功后返回路由；无法接纳时返回对应拒绝。
- QUEUE 按 FIFO 或 PRIORITY 选择一个请求，成功后立即返回路由。
  没有可用机器或容量不足时保留全局队列成员，等待节点或容量变化再尝试。
  请求到期、取消或停止接纳后结束等待。

SINGLE 是固定行为，无须配置决策策略或交付模式。
自定义调度器、排序与选址策略仍通过各自入口扩展。
SchedulingRequests 统一管理排队登记与未结束任务，同一请求的决策和抢占结果共用一份执行记录。
等待数从排队登记中排除仍有任务的请求；决策和抢占重叠时只扣一次，已取消任务不扣减新登记。
取消或恢复排队只改变登记，旧任务的名额和关闭等待仍保留到回调返回。
CapacityWaitQueue 与抢占名额等待集合只负责唤醒，不再用于拼接等待请求统计。

## 选址与路由响应

RequestRouteSelector 选择所需角色并登记资源，部分角色失败时撤销已取得的登记。
完整路由构造后直接交给 Scheduler；executeRouteSelection 统一将资源关联到 Context，
再校验端点并提交或回滚。Selector 和路由发布入口不重复关联 Context。
PrefillEndpoint 在登记时检查当前代际、资源版本和 compute-token 容量；
DecodeEndpoint 独立检查 KV 和请求容量。

选址成功后，Scheduler 在请求锁内再次核对期限、取消和精确选址身份，
核对 Decode 预留仍存在并记录路由发布。
提交成功后，同一次选址收尾直接准备路由响应；没有独立发送阶段、发送状态枚举或二次发布回报。
选址尚未收尾时，取消与 Worker 反馈先记录事实，资源回滚完成后再执行结束动作。
网络取消、容量通知和用户响应回调在请求锁外执行。
返回路由不表示引擎已经完成请求，也不立即删除资源记录。

路由失效或提交失败时，先完成本次尝试的资源回滚。
QUEUE 依据失败原因等待或重新选址，旧尝试不能修改新请求的记录。

## Prefill 工作量

PrefillTimePredictor 只计算 computeTokens：

    computeTokens = max(0, inputTokens - effectiveCacheHitTokens)

Endpoint 按请求保存已登记的 compute tokens。
选址按节点已有工作量加当前请求的 compute tokens 评分；
每台机器的硬容量使用 router.roles.prefill.maxComputeTokens。
启用 cache first 时，cacheAffinity.maxComputeTokens 是另一条绝对上限；
两条上限都检查加入当前请求后的总工作量，超限的机器直接跳过，全部不满足则等待。
缓存命中更高也不能越过上限。
没有时间公式、在线学习、批量预测、虚拟插队或未来组批模拟。
经过的墙上时间不扣减 token 工作量。

## 请求状态与结束

RequestRepository 保存活动请求及最近结束的 RequestResult；归档结果直接携带所属调度器，不再包装 TerminalRecord。
RequestContext 统一持有输入、配置、优先级、Future、遥测和生命周期事实，
并串行记录本次选址、路由响应、取消、Worker 上报和清理进展。
只读 Snapshot 在请求锁内按需捕获，只提供关闭、响应及诊断所需的信息。
Worker 观察细节、取消错误码、过期标记和最后上报时间留在 Context 内部；routingPhase 从选址编号推导。
pendingEndEvent 在 Context 内部保存待处理的结束事件，不通过 Snapshot 暴露；
requestEndResult 保存已确定的结束结果，responseForFuture 保存选定的响应。
结束结果确定后立即清除 pendingEndEvent；结束事件只携带原因，资源释放进展统一保存在 Context。
调度截止时间直接读取不可替换的 SchedulingMetadata，不再复制保存。
WorkerReleaseConfirmed 表示远端释放已确认，CleanupCompleted 表示本地清理已完成；两者不互相替代。
每次状态更新结束时统一检查清理、通知和响应条件，满足后生成 requestRecordForArchive，由 Repository 归档。
清理先完成而响应尚未选定时继续保留请求；后来补选响应的同一次状态更新即可完成归档判断。
定时器依据提交后的归档状态计算，不重复维护归档条件。
refreshState 统一完成归档检查、定时器和最后变更时间的更新。
选址取得的路由保存在 RequestContext.route 中，提交和失败清理使用同一份记录。
提交时在请求锁内检查选址身份、取消、期限和资源有效性，同时结束选址并确定成功响应；
没有独立交付确认、响应就绪标记或选址完成认领阶段。响应执行器只在锁外执行已确定的回调。
提交前取消阻止成功响应；提交后的取消处理引擎执行，不再改写首次响应。
取消、Worker 结束和超时在没有进行中选址时直接确定结束结果；Scheduler 预留响应执行资源后，
由 Context 在同一次调用中选定结束响应。预留失败保留结束事实和待响应请求，不能提前归档。
成功提交路由同时结束选址，不再重复执行选址收尾；未提交的选址在回滚后调用
finishSelectionWithoutCommit，落实等待中的取消或失败，再由调度策略决定等待、重试或清理。
选址退出直接调用所属 Scheduler 的实例入口，不通过静态方法再次分派。
回滚成功清除路由，回滚失败保留原记录并进入请求清理，不另设失败后补登记流程。
选址尚未返回时只记录取消或端点下线事实，不能提前移除正在选址的队列项；选址结束后统一决定重试或清理。
提交时间和 P/D 远端释放证据直接记录在请求上。结束结果归档不携带路由引用。
每轮清理在请求锁外提交取消和释放资源，再一次记录成功释放的各侧资源及通知进展并检查归档；
部分失败保留已成功的进度。归档还要求响应已选定或 Future 已完成。
Timing 统一保存观测时间：AtMs 是 Unix 毫秒时间点，Nanos 是单调时钟时间点，ElapsedUs/DelayMs 是耗时。
登记时间同时用于队列观测和生命周期查询；没有第二份 enqueueTime。已废弃的 ACK 阶段及其延迟统计不再保留。
PreemptionStatistics 一次保存已提交抢占的诊断成本和请求数量，成本不解释为毫秒。
请求与选址尝试通过请求对象和选址编号匹配，Endpoint 仍按精确资源记录和代际校验；不额外维护请求 incarnation 计数。

WorkerStatus 的活跃、完成和失败上报推进按请求的资源账本。
Worker 上报、Endpoint 关闭与取消 RPC 的远端清理确认统一为 RequestEvent，经 Scheduler.onRequestEvent 进入 Context.recordEvent。
REMOTE_CLEANED 确认 P/D 远端资源已释放；REMOTE_FENCED 只确认 Prefill，不确认 Decode。取消被接受、请求未找到或 RPC 失败不代表资源释放。
Context 在请求锁内校验并记录事件，直接返回是否需要更新定时器或推进清理；Scheduler 在锁外执行后续动作，不比较前后快照。
Prefill 完成与整个 P/D 请求完成分开处理。
Endpoint 退役、失活清理和重复反馈都针对精确请求记录，不能释放新尝试的资源。

响应完成、引擎执行完成与本地资源清理完成是不同事实。
RequestCloseQueue 串行处理同一请求的通知、释放和归档；
scheduleRequestCleanup 统一安排执行顺序：Endpoint 关闭先发布响应，普通结束先清理再响应，
无活动超时将队列移除延后到收尾任务执行；任务失败不阻断其余独立清理或响应动作。
资源释放失败保留尚未完成的清理责任。

## Decode 抢占与取消

Decode 抢占由 policy 选择请求，执行普通请求取消并等待 Endpoint 的容量变化。
RequestContext 不保留独立抢占事务、取消阶段或恢复交付协议。
RPC 模式由 EngineCancellation 向原引擎发送单请求 Cancel，并处理有界重试和晚到回复；
RETURN 模式将取消指令交给调用方透传。
Scheduler.cancel 记录本地取消并提交清理后返回，不等待引擎回复，也不表示容量已经释放。
EngineCancellation 通过 Scheduler.onEngineCancelReply 回报结果，由 Scheduler 校验原请求身份、
转换清理事件并推进资源释放；Context 校验具体路由。已归档请求的晚到回复被忽略，
取消任务持有的 Context 引用不依赖请求仍在活跃索引中。
取消确认和资源释放使用精确路由、预留及请求身份。
请求级 RequestResult 同时携带抢占完成状态或释放规划线程后订阅抢占结果的后续动作，不再使用 RoutingPlan。

## 关闭

停止接纳后，停止调度来源并推进已有请求结束。
等待已接纳的选址、取消、清理和响应工作，再关闭相关执行器。
请求工作通过标准 Executor 接口提交；RequestTaskPool 只管理已接纳任务的计数、排空和关闭。
计时线程由 JDK ScheduledThreadPoolExecutor 管理，TaskTimer 只保留可替换的单次定时入口。
没有批次发送执行器、在途 payload 字节预算或批次回调排空流程。
