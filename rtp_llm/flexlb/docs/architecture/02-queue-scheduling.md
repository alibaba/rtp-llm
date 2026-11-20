# Scheduling and Request Lifecycle

FlexLB 逐条选择机器并返回路由，由调用方向引擎发送推理请求。
没有组批决策、收集窗口、EnqueueBatch 发送或 worker 内的第二层派发队列。

## 调度入口

同一个业务 request_id 按 RequestPhase 分为 ENCODER 与 GENERATION 两条生命周期。
AbstractRequestScheduler 负责注册、期限、取消、响应及资源清理，并统一执行选址、路由提交和失败回滚；
具体调度器负责启动选址以及容量不足时的拒绝、等待或抢占策略。

- DIRECT 立即进行一次选址，成功后返回路由；无法接纳时返回对应拒绝。
- QUEUE 独立组合 priority.enabled 与 policy.type（FIFO / PREFILL_COST），成功后立即返回路由。
  没有可用机器或容量不足时保留全局队列成员，等待节点或容量变化再尝试。
  请求到期、取消或停止接纳后结束等待。

SINGLE 是固定行为，无须配置决策策略或交付模式。
自定义调度器、排序与选址策略仍通过各自入口扩展。
SchedulingQueue 直接保存 RequestContext；OrderedRequestQueue 的节点、成员索引与扫描游标不对外暴露。
Scheduler 用 executing 记录尚未结束的任务，队列成员和任务各有一个负责位置。
取消移除队列成员，但执行记录保留到选址或抢占结果清理结束，期间不会重复选路。
容量不足的请求由 CapacityWaitQueue 按角色、分组暂停，仍留在排序队列。容量变化只触发重新扫描；
Scheduler 按现有排序选出请求，在登记选路任务的同一把锁下扣重试额度。
CapacityKey 只标识角色、分组和可选端点，不包含候选集合。SelectionResult 携带本次选路的候选代际提示，
CapacityWaitQueue 的 Waiter 按请求保存；重新挂起时更新，取消或离开等待时清除。
返回 RETRY 时结清本次额度，再等待重新派发。整组满载的周期探测额度归该等待分组，和 Worker 额度分别消费。
容量通知按现有队列顺序触发重试检查。候选提示只决定先尝试哪份预算；实际选路仍重新查询 cache。
调度器收到 Endpoint 的容量通知后，读取同一资源版本下的剩余请求名额；共享同一 Worker 的等待域共同消耗这份重试预算，
重复或晚到的旧版本通知不会增加名额。Prefill 预算同时受请求数与 Engine 队列水位限制；Decode 的请求数预算
不能替代按请求进行的 KV 准入检查。实际容量仍由 Endpoint 在登记时核对，不把重试预算当作资源预留。
一次重试在分配选路线程时消耗预算，任务结束不补发当前版本的名额。容量刷新会计入仍在选路的请求，
旧版本的尝试结束时按结果结清；本次失败重新挂起时会处理期间到达的容量变化。
每隔 capacityCheckIntervalMs（默认 100ms）在调度锁外重新读取各 Worker 的本地容量快照，
补偿漏通知。周期快照与通知用 Worker 资源版本合并到同一份预算。采样期间如果收到了新 Worker 的容量通知，
调度保留快照里缺席的预算并继续派发，下一次完整采样再清理缺席的 Worker。
排序队列按当前策略检查可派发请求。有旧候选提示时，优先用匹配候选的预算；
没有匹配的预算时可用同一等待范围内其他 Worker 的额度。容量通知与周期检查使用同一规则。
每个组按请求顺序处理一轮固定的等待集合；失败请求和新进入等待的请求留到下一轮，
不遮住本轮尚未尝试的请求，也不让持续到来的请求无限延长本轮。容量通知和周期检查
继续未完成的一轮，本轮结束后按优先级和先后顺序开始下一轮。没有请求空位时，每个 role/group 每轮只共享一次控制检查，用于恢复抢占或尚未观察到的容量变化；
这不是 Engine 接纳名额。任务完成不会补发容量重试额度。
plannerCount 限制并行调度任务，executing 统计已安排但未结束的选路、抢占恢复和估计刷新任务。
容量重试从剩余线程名额内取出。FIFO（含优先级排序）按 `ceil(remaining × scanBudgetMultiplier)`
检查队列条目，默认倍率为 2。PREFILL_COST 检查整条队列的可派发请求，用堆保留前 `remaining` 个候选。
首次选路负责识别候选及启动抢占，已在容量等待中的请求实际派发前再扣重试额度。
planner 处理完选路结果后释放执行名额并通知调度线程。调度线程按当时可用的名额补派任务。
抢占结果回调先使结果可读，再通知调度线程。调度线程在调度锁内检查是否有空闲名额和就绪动作，
有空闲才读取抢占就绪列表，准备派发普通请求时才读取抢占等待集合。恢复选路与普通选路共用 plannerCount 个执行名额。
关闭先停止接纳并结束排队请求，保留规划线程处理晚到结果，待任务清理完成后关闭线程池。

## 选址与路由响应

RequestRouteSelector 选择所需角色并登记资源，部分角色失败时撤销已取得的登记。
完整路由构造后直接交给 Scheduler；executeRouteSelection 统一将资源关联到 Context，
再校验端点并提交或回滚。Selector 和路由发布入口不重复关联 Context。
PrefillEndpoint 在登记时检查当前代际、工作量、请求数上限及适用的 Engine 队列水位；
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
选址的评分与准入分别处理：普通 COST_BASED 保留当前 cache affinity；TOKEN_BALANCE 每次选路按当前 cache 查询结果确定热缓存候选组。
最大命中量与候选代际一次计算形成 CacheCandidates，节点选择与阻塞结果共用这份事实；没有独立的重试候选策略。
统一请求数上限默认 128，Engine 已有但本地未知的请求也计入；没有计算量硬上限。
Engine 队列准入属于 TOKEN_BALANCE，默认 waitingThreshold=10。
waiting > threshold - unobserved 时立即阻止新增；没有单独的持续拥塞计时配置。

OrderedRequestQueue 保存唯一成员索引并直接提供 FIFO 顺序；启用成本排序时组合 PrefillCostOrder。
PREFILL_COST 按 compute/input 的精确比例、compute、input、入队顺序排序；优先级启用时先比较 priority。
可派候选的估计有未知时按优先级和入队顺序派发。不同成本或优先级的候选可以并行选路，排序决定派发顺序。
估计刷新复用有限 planner 名额，缓存查询在调度锁外执行，旧刷新票据不能回写恢复后的新队列成员。
热缓存候选代际只作为请求级唤醒提示，不参与容量等待域的身份。

requestLifecycle 位于顶层：dispatchTimeoutMs 默认 60000，DIRECT/QUEUE 都从入口计到路由提交；
inactivityTimeoutMs 默认 60000，从提交开始，仅匹配该路由的 Engine 请求反馈续期。
engineVisibilityMultiplier 默认 2.0。可见性使用单请求估计毫秒数
computeTokens + 0.02 × reusedTokens。PrefillEndpoint 估计此前工作与当前请求的剩余时长，
AbstractRequestScheduler 乘系数、加 10000ms 观察余量并生成绝对期限。
缺少可靠工作量/耗时则不预测；排序分数与时间估计不混用，墙上时间不扣减 token 工作量。
ExpirationTimer 继续管理 REQUEST、INACTIVITY、VISIBILITY；VISIBILITY 仅诊断，不结束或释放请求。

## 请求状态与结束

RequestRepository 保存活动请求及最近结束的 RequestState。归档存储在 Repository 内将摘要与所属调度器关联，查询只返回纯数据摘要；摘要不持有路由或调度器引用。
RequestContext 统一持有输入、配置、优先级、Future、遥测和生命周期事实，
并串行记录本次选址、路由响应、取消、Worker 上报和清理进展。
只读 Snapshot 在请求锁内按需捕获，只提供关闭、响应及诊断所需的信息。
Worker 观察细节、取消错误码、过期标记和最后上报时间留在 Context 内部；routingPhase 从选址编号推导。
pendingEndEvent 只保存一份请求级待结束原因，不保存原始 WorkerRequestEvent。
endRequest 是请求结果的唯一提交入口，统一处理取消、超时、本地失败及 Scheduler 判断成立的 Worker 请求结束。
endState、endCause、endError、endDetail 保存结果；EndCause 同时表达本地原因和 Worker 导致的请求完成、失败或抢占取消。
选址期间保留优先级最高的待结束原因，同级保留先到者：完成或执行失败高于抢占取消，抢占取消高于普通本地失败，端点关闭最低。
已提交响应的发布失败保留自身结果；其他结束仍结合已记录的取消原因决定取消或超时结果。
提交前且没有 Decode 执行证据的 Endpoint 关闭只使本次选址失效，selectionEndpointCloseDetail 保留关闭说明并阻止提交。
选址回滚结束时重新检查执行观察和清理结果；未提交、没有 Decode 执行证据且本地资源全部回滚才允许调度器重试。
若期间收到 Decode 活动上报，即使资源随后回滚也不能重试；没有更强的请求结束原因时，以 Endpoint 关闭结束请求。
选址退出清除关闭说明，结果确定清除待结束事件；本地失败不会因关闭事实被撤销而丢失。
最终 endState 仍结合此前取消原因确定，结束原因不直接代表资源已经释放。
结束结果确定后立即清除待结束事件；资源释放进展统一保存在 Context，responseForFuture 引用选中的响应执行登记。
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
选址回滚由 rollbackSelectedRouteResources 撤销本次取得的登记，不隐含结束选址；
退出选址时由 finishUnpublishedRoute 先回滚，再完成选址收尾。
cleanupRequest 依据提交情况和 P/D 远端释放确认推进清理。未提交的资源安全回滚；
已提交但没有完成、清理、隔离或代际关闭证据时保留登记与容量。无反馈超时进入统一结束和取消流程，
不直接删除在途资源。正常响应已确定但另一角色迟迟未释放时，也由无反馈超时启动取消，保留原响应结果。
Timing 统一保存观测时间：AtMs 是 Unix 毫秒时间点，Nanos 是单调时钟时间点，ElapsedUs/DelayMs 是耗时。
登记时间同时用于队列观测和生命周期查询；没有第二份 enqueueTime。已废弃的 ACK 阶段及其延迟统计不再保留。
PreemptionStatistics 一次保存已提交抢占的诊断成本和请求数量，成本不解释为毫秒。
请求与选址尝试通过请求对象和选址编号匹配，Endpoint 仍按精确资源记录和代际校验；不额外维护请求 incarnation 计数。

WorkerStatus 的活跃、完成和失败上报推进按请求的资源账本。
Worker 上报与 Endpoint 关闭通过 WorkerRequestEvent 传递精确登记、角色、Endpoint 和事件事实。
Scheduler 创建的接收入口捕获本次 RequestRoute；handleWorkerEvent 在请求锁内调用 Context.updateWorkerStatus，核对路由、角色登记和 Endpoint 身份并更新执行与资源事实。
Scheduler 根据上报类型、角色和提交状态决定继续等待、使选址失效或生成 RequestEndEvent；Context 不决定角色完成是否意味着整个请求结束。
Decode 不再通过 Repository 查当前路由；P/Encoder 不保存完整路由。
取消 RPC 已持有原路由，直接调用 Context.recordRemoteCleanup，不再重复构造 Worker 事件。
REMOTE_CLEANED 确认 P/D 远端资源已释放；REMOTE_FENCED 只确认 Prefill，不确认 Decode。取消被接受、请求未找到或 RPC 失败不代表资源释放。
上报解释与 Context 接受结果在同一次请求持锁期间完成，防止取消或路由变化插入；Scheduler 根据前后状态及结束动作，在锁外更新定时器或推进清理。
Prefill 完成与整个 P/D 请求完成分开处理。
Endpoint 退役、失活清理和重复反馈都针对精确请求记录，不能释放新尝试的资源。

响应完成、引擎执行完成与本地资源清理完成是不同事实。
RequestContext 记录清理执行状态，执行期间的新通知合并为再检查一次；Scheduler 在请求锁外执行释放和归档，并跟踪执行中的清理供停机等待；
scheduleRequestCleanup 统一安排执行顺序：Endpoint 关闭先发布响应，普通结束先清理再响应，
无活动超时将队列移除延后到收尾任务执行；任务失败不阻断其余独立清理或响应动作。
资源释放失败保留尚未完成的清理责任。

## Decode 抢占与取消

Decode 抢占由具体 DecodePreemptionPlanner 选择请求，DefaultRequestPreemption 执行普通取消并等待 Endpoint 的容量变化。
DefaultRequestPreemption 报告等待结果是否就绪，并在登记新的抢占等待或收到相关 Endpoint 的容量通知时递增版本。
QueuedRequestScheduler 在版本变化或定时检查到期时重新检查当前等待，并在有空闲规划线程时派发本轮剩余的检查。
RequestContext 不保留独立抢占事务、取消阶段或恢复交付协议。
EngineRequestCanceller 向原引擎发送单请求 Cancel，处理有界重试和晚到回复；一轮耗尽后仍保留精确路由，
由定时恢复入口继续下一轮。关闭时停止恢复并等待已提交回调。
Scheduler.cancel 记录本地取消并提交清理后返回，不等待引擎回复，也不表示容量已经释放。
EngineRequestCanceller 通过 Scheduler.onEngineCancelReply 回报结果，由 Scheduler 校验原请求身份、
转换清理事件并推进资源释放；Context 校验具体路由。已归档请求的晚到回复被忽略，
取消任务持有的 Context 引用不依赖请求仍在活跃索引中。
取消确认和资源释放使用精确路由、预留及请求身份。
RequestRoute 不携带抢占资源或 continuation。RequestContext.currentResourceWait 指向当前精确等待，
DecodeEndpoint 保存同一对象；取消、结束或选路退出清除 Context 引用后，残留索引立即失去优先级约束作用。
多个等待者取有效等待中的最高优先级；更高优先级请求仍可接纳。
等待的 Future 只在 Endpoint 锁外完成，晚到登记由结果接收方精确回滚；等待本身不占规划线程。

## 关闭

停止接纳后，停止调度来源并推进已有请求结束。
等待已接纳的选址、取消、清理和响应工作，再关闭相关执行器。
DIRECT 通过标准执行器为每次选路启动虚拟线程。并发上限取 JVM 可见 CPU 数，最少 2、最多 8。
等待名额的请求留在 DIRECT 调度器。超时后续和异步清理共用独立的虚拟线程执行器。
调度器跟踪已接纳的超时和清理工作供停机分阶段等待，由应用最终关闭该执行器。
计时线程由 JDK ScheduledThreadPoolExecutor 管理，TaskTimer 只保留可替换的单次定时入口。
没有批次发送执行器、在途 payload 字节预算或批次回调排空流程。
