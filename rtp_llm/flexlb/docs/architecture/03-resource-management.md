# Resource Management

## 术语与记录

Worker 是远端推理实例，WorkerStatus 保存它的身份、代际与引擎状态。
Endpoint 是 Master 内对应某一代 Worker 的本地对象，维护请求记录、队列和容量。
RoleSelection 保存单个角色的选择事实和精确登记；RouteResult 聚合这些角色并保留同一次尝试的精确身份。Endpoint 引用本身不阻止下线。

EndpointRegistry 按角色和逻辑地址维护 Endpoint。选路快照提供候选；
selectEndpoint 检查可用性；tryRegisterSelection 在每个 Endpoint 的状态锁内验证评分版本并登记影响。
入队和发送再次检查可用性，独立的发送容量通过派发许可检查。
旧 Endpoint 引用不能修改同地址新代际的请求。

请求上报与 Endpoint 下线使用独立入口。WorkerRequestEventType 只描述请求的活跃、
资源分配确认、完成、失败和抢占取消，由 onWorkerRequestEvent 更新请求执行事实。
Endpoint 下线通过 onEndpointRetired 传入具体实例、路由和 Decode 预留，核对当前关联后
处理资源释放及请求去向，不更新请求活跃时间，也不表示 Worker 已接受或完成请求。
准入期间的结束原因保存在同一份 pendingEnd；未交付路由完整回滚后可撤销该退役原因并重选，
取消、超时等其它已接受原因继续保留。

## Endpoint 状态互斥

每个 Endpoint 使用一把 ReentrantLock 保护本地请求、批次、容量和退休状态。
准入检查和记录修改在同一临界区内完成；不同 Endpoint 可以并行操作。
所有新增交接统一使用 `WorkerEndpoint.acceptsNewRequests()`：WorkerStatus 可调度且 Endpoint
仍接受请求才允许入队、预留容量或提交派发。任一条件关闭都拒绝新增交接，已有请求的状态反馈、
回滚和资源释放仍可执行。
未交付路由失效时，Scheduler 先通过 shouldRetryRouteFailure 选择重试策略。
需要重试的失败统一记录在 RequestContext 的 routeFailure 中；失败角色不决定清理范围。
Context 使用原有 placement 凭据取得整条路由的撤回权，与最终交付提交互斥；
BATCH 的投递登记与开始发送在同一个临界区完成。
基类直接协调释放 Prefill/PDFUSION、Decode、Encoder 实际取得的全部资源；VIT 只清除选择引用。
删除精确 Decode 登记时一并清除其临时派发额度，旧许可和旧提交随后只能匹配旧记录，
不再等待批次准备完成通知，也不会释放同 ID 新登记的额度。
准入回滚、路由撤回和请求终止共用基类的 `releaseRouteResources`，Dispatcher 不再实现各 role 的组合清理。
清理成功后，未过期请求通过 `onRequest` 恢复调度，不再进行第二次重试决策：Endpoint 不可调度的排队请求保留原序号、优先级和截止时间重新入队。
准备错误保留原来的失败发布与异步清理；清理失败禁止重试。正常的派发容量等待不撤销路由。
正在抢占/取消协议中的请求沿用原有协议收尾，不能跳过远端执行状态确认直接重试。
已交付、正在发送或发送结果不确定的请求不能直接撤回重发；只有精确匹配的 NOT_SENT 结果可以证明未发送。
失效 Endpoint 通知由排队调度器合并处理，在调度锁外查找受影响路由并发起撤回。
同步操作由调用线程加锁执行，不创建操作队列、Future 或切换执行线程。
业务方法直接使用 lock/try/finally/unlock 展示临界区，不通过回调包装加锁。
状态锁及要求调用者持锁的内部方法仅在 Endpoint 包内可见。
Scheduler、Dispatcher 通过 Endpoint 的队列版本检查、条件删除和派发提交方法完成原子操作。
Scheduler 自己的等待决策、容量订阅和停止归属由独立的等待锁保护；
监听注册与移除不持有 Endpoint 锁，停止完成后的等待决策不能重新挂载监听。
监听增删只更新本地监听集合，与等待、停止状态一起在等待锁内完成；容量探测和 Endpoint
版本核对在等待锁外执行。
先订阅容量变化，再检查队列和资源事实，避免在订阅过程中丢失容量唤醒。
调度唤醒等异步工作继续使用现有执行器和 CAS 标志，避免重复提交。
现有请求生命周期和 Worker 状态同步事务仍有各自的并发保护。
RPC、耗时预测和请求完成通知在锁外执行。
Prefill 的阻塞原因探测在锁外执行，捕获队列前核对版本，版本变化则重新探测。
Encoder 直接提交不持有 Context 监视器等待 Endpoint；提交后仍由请求状态校验拒绝取消的路由，
并回滚晚到的资源记录。
Encoder 在锁内捕获引擎任务快照和本地未确认请求的工作量，在锁外遍历不可变引擎任务；
退休时在锁内移交请求表，在锁外生成通知。
Encoder 状态对账只遍历本地跟踪请求，在锁内更新记录并发布对应状态 CAS，锁外复制通知列表。
Prefill 和 Decode 退休在锁内捕获清理所需记录并清空记账，排序、不可变复制和日志在锁外执行。
Decode 在锁内捕获同一版本的优先级计数，在锁外构造准入摘要，缓存发布前重新核对版本。
心跳反馈列表的不可变复制也在锁外完成。
状态对账需要重新预测批次时，先捕获成员，预测完成后再核对成员和下线状态，
防止把已删除请求写回。

下线在同一把状态锁保护下停止接收请求、清空记录，然后在锁外通知请求生命周期。
较早完成的准入由下线统一清理，较晚的准入直接拒绝。
不再需要 GenerationPin 或选路引用计数。

Selector 在私有局部作用域中收集已登记的 RoleSelection，并协调组合失败回滚；没有跨类共享的可变 Builder。
选点完成、入队和直接提交边界统一通过 validateEndpoints 检查全部角色，
包括没有容量预留的 VIT。选点中的版本/代际变化由 Strategy 局部处理；
已选组合在发布前失效统一返回 RETRY，释放未交付登记后重选；结果只保留兼容响应所需的失败原因，不再嵌套旧选择结果。
未交付的预留先完整回滚，DIRECT 在选点和提交之间共用一次重新决策额度，
QUEUE 则重新规划，不把下线当作容量不足等待。普通容量不足继续按容量策略处理，
参数错误、取消、超时和已交付请求仍按原有生命周期处理。
尚未交付且已完整回滚的准入，不因该次 Endpoint 下线通知而终止请求。
客户端携带 selectedVit 时仍须匹配绑定的代际；过期绑定返回 VIT_ROUTE_STALE，
不能静默换到没有该次 VIT 结果的节点。

## Prefill

PrefillEndpoint.RequestRecord 保存不可变排队输入、SELECTED/WAITING/RESERVED/COMMITTED 阶段、预计剩余耗时、
阶段计时起点和所属批次。每个请求一条记录，等待队列是这些请求的有序索引。
SELECTED 已计入投影但尚不可派发；发布时同一记录进入 WAITING。Direct 在选择时登记为 RESERVED。
组合完成时由 RouteResult 构造器绑定已有记录，禁止将同一记录绑定到第二个执行身份；未绑定记录下线时直接清除，由选择尝试检查失败并撤销其他角色。
入队和 DIRECT 发布只接受选择时的精确记录，不创建缺失记录。

PrefillEndpoint.BatchRecord 是 Endpoint 内部的已提交批次记录，保存批次 ID、成员、原始预测输入、
剩余耗时与执行结果。组批候选不创建账本记录，也不占批次槽位。
提交时在同一 Endpoint 锁内检查最新批次上限、成员身份和队列版本，然后创建完整批次；
竞争失败的候选释放临时派发凭据，成员仍留在等待队列。最后一个成员结束后释放批次名额。
不存在空批次预留、未提交批次回滚或 committed 双态。

直接调度使用请求记录，批次调度使用批次记录，不再继承只保存 owner 的 Reservation 基类。
回滚和释放核对具体记录或请求身份，旧回调不能删除同 ID 的新请求。

## Decode

DecodeEndpoint 维护每个请求的本地预留、引擎确认、释放和抢占状态。
DecodeResources.RequestReservation 是不可变的预留标识：
Endpoint 代际 ID、请求 ID、预留编号。容量和任务阶段仍保存在 Endpoint 中。

tryRegisterSelection 检查评分版本、容量并登记；登记即参与后续 P/D 选择。
提交阶段不再首次预留 Decode。排队请求替换和引擎抢占完成时，Endpoint 直接将新登记置为可排队状态，
在锁外通知容量变化；Scheduler 只检查精确登记仍可用并接收，不再执行 markQueued 转移。
排队预留保护 KV 容量；发送前通过 EngineDispatchPermit 检查引擎并发和 KV 限制。
引擎状态到达后对账；发布结果不确定时保留相关容量，直到取得完成或取消证据。

## Encoder 与监控

EncoderEndpoint 保存尚未完成的请求、是否已被引擎观察到及预计未命中编码长度。
提交和下线使用同一把状态锁，完成、取消或下线后清除对应记录。

调度读取不可变快照。监控只在 Endpoint 空闲时捕获一致快照，忙时跳过采样，
通过 tryLock 避免等待。预测、指标格式化和通知在请求记录变更之外执行。
