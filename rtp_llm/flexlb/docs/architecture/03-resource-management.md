# Resource Management

## 请求记录和容量

WorkerStatus 保存远端实例的身份、代际和状态；Endpoint 保存这一代实例的本地请求记录和容量。
EndpointRegistry 按角色和地址管理实例。RoleSelection 保存单个角色的选择和精确登记，
RequestResult 组合本次选择的全部角色。旧代际回调不能修改同地址新实例的请求。

Registry 按角色缓存不可变、按地址排序的 Endpoint 列表。节点发布和撤出统一更新对应角色的列表，
选址读取现成列表，不在每个请求上复制或排序。Prefill 策略直接使用 PrefillEndpoint，
不再包装额外的路由目录项；旧列表只用于候选遍历，登记时仍校验当前节点和代际。
Endpoint 查询统一为 `endpoints(role, group)` 和 `getEndpoint(role, address)`：前者按角色返回类型化列表，
group 为 null 时包含全部分组；后者只查当前地址映射。角色策略负责读取容量快照，并在登记前检查接纳状态及代际。
监控直接遍历同一列表，不再构造统计专用的节点 Map。

PrefillEndpoint.SelectionSnapshot 只包含 generationId、resourceVersion、computeTokens 和 maxComputeTokens。
策略读取快照比较工作量，tryRegisterSelection 在 Endpoint 状态锁内检查代际、版本和剩余容量，
登记后立即计入工作量。冲突时重新选择，容量不足时尝试其他候选；没有候选则等待容量变化。

选择过程中取得的各角色资源由同一次尝试负责。完整路由提交前仍需检查取消和节点可用性；
失败时回滚全部已取得资源。完整路由与返回状态一起提交，不保留等待派发的路由。
回滚失败的记录留给异步清理，避免资源失去清理入口。

P/D/E 共用 WorkerEndpoint 的 RegistrationStatus、Registration<T> 和 RequestRegistration 契约，
各角色保留自己的具体登记与 SelectionSnapshot。P/E 共用本地 REGISTERED/COMMITTED 阶段；
Decode 的 UNCONFIRMED/CONFIRMED 表示 Worker 是否已确认执行，与本地提交阶段不同。
RoleSelection 通过 SelectionRollback 调用统一的 rollbackSelection，各 Endpoint 自行核对身份和阶段，
旧登记或重复回滚不能影响同 ID 的新记录。正常完成或远端清理后的释放使用 releaseRequest。

## 状态更新和关闭

每个 Endpoint 使用一把 ReentrantLock 保护本地请求和容量。
WorkerEndpoint 统一检查代际、加锁和发布状态；角色实现 updateStatusLocked。
请求事件、容量通知和网络操作在锁外执行。

Registry 对所有已登记 Endpoint 调用统一的 `evictExpiredRequests`，不按角色分派或转换类型。
Prefill、Decode 和 Encoder 各自负责其本地记录的 TTL 与精确身份复核；没有请求登记的 VIT 默认不做清理。
EncoderStrategy 直接读取 router.roles.encoder.maxRequests，筛选候选和 Endpoint 登记使用同一上限。
DIRECT 和 QUEUE 都遵守此上限；Scheduler 只根据容量不足结果决定拒绝或等待，不持有 Encoder 容量参数。
只有调度器已不再保留的记录才可清理：先在锁内捕获候选，锁外查询保留状态，再持锁复核后移除。
Encoder 在接纳和每次 ACTIVE 上报时更新记录时间；清理查询期间发生替换或新上报时，旧候选不能删除新记录。

WorkerEndpoint.acceptsNewRequests() 同时检查节点健康和本地接纳状态。
stopAcceptingRequests() 停止新增登记；closeAsync() 清理这一代记录并返回共享的完成结果。
Registry 等待精确代际关闭后清理注册与缓存。已有请求的状态反馈、回滚和释放仍可执行。
单节点下线统一调用 Registry.closeWorker，Registry 自动启动清理并返回只读完成结果，
调用方不再携带清理对象或另行启动后续操作。WorkerStatus 锁内停止接纳并撤出路由，
异步任务等待原状态事务解锁后，依次等待 Endpoint 关闭、清理缓存并删除精确状态。
等待节点接纳锁和执行清理时不持有 Registry 全局锁，其他节点仍可发现、发布和接纳请求。
定期扫描先关闭不可用节点的接纳，再启动清理；状态锁或接纳锁忙碌的节点留待后续扫描，
不阻塞同一轮对其他节点的处理。
整体关闭仍分别等待正在发布的节点和已登记的下线任务；清理失败通过完成结果及整体关闭报告。
Decode 关闭先清除本地请求记录，再逐个通知请求和容量变化。普通通知异常记录日志并继续，
不阻止 Registry 收尾；本地资源清理失败仍通过关闭 Future 报告，Registry 不将其视为清理完成。

## Prefill

PrefillEndpoint.RequestRegistration 保存 requestId、computeTokens、绑定的完整路由和 REGISTERED/COMMITTED 状态。
登记时预留 compute tokens，完整路由绑定后提交；完成、回滚或清理时移除精确记录并扣减工作量。
本地请求按登记值计数，不根据经过时间衰减，也不学习执行耗时。

上报的未知运行请求按 inputLength 减有效 prefixLength 统计 compute tokens。
缺少输入长度或任务明细时保守占满配置容量。已知本地请求不会再次计入未知工作量。
可用容量增加后通知全局等待队列；Endpoint 内不保存等待派发队列或批次记录。

## Decode

DecodeEndpoint 保存本地请求记录，并按完整 worker 上报统计引擎占用。
tryRegisterSelection 在同一次持锁期间核对选址快照、请求名额和 KV 容量，成功即预留；
后续路由提交不再领取临时派发许可。调用方持有原始 DecodeEndpoint.RequestRegistration 对象，
释放时比较对象身份，旧回调不能释放同 ID 的新记录，不再维护预留编号。

本地请求只维护一个 confirmed 标志。未确认时，同时预留请求名额和 KV；
首次出现在 worker 活跃请求上报中即确认分配，两项本地预留同时移除，由 worker 容量上报负责。
运行阶段直接取自 worker，不在本地重复维护。未知引擎请求计入容量，不创建本地记录。
选址与接纳共用一份 CapacityUsage；监控直接统计计数，不复制请求表。
只有抢占和容量不足归因读取请求明细，优先级归因直接遍历明细，不另建优先级数组快照。
确认未发送或远端已清理后，调用统一的 releaseRequest；仅发出取消不能释放。
未确认请求释放时扣除本地请求名额及 KV 预留，并在锁外通知调度。
已确认请求清理时只删除本地关联，请求数和 KV 容量均等待 worker 上报更新，不另存已释放 ID 或提前修正上报。
引擎实际可用 KV 以 worker 上报为准，不把 inputLength 当作取消释放的物理 KV。

Endpoint 不保存 A 与等待请求 B 的抢占关系，也不替 B 保留 A 释放的容量。
B 超时不影响 A 的普通取消和清理；A 释放容量后，所有等待请求按正常调度规则竞争。

请求 Context 中的结清标志保存准确 route 的清理观察，供结束和归档使用；它不参与 Endpoint 容量记账。
远端允许释放的证据与本地记录已经移除是不同事实。Endpoint 中查不到 requestId，不能单独证明某次旧尝试已经结清。
结束流程先处理停止、deadline 和已就绪的完成通知，再释放资源。
成功路径不再回到清理循环顶部推进通知，也不推迟已经具备条件的本地完成。
保留最多三次实际失败尝试，等待远端证据时退出，由后续事件推进。准确资源身份保证重复释放不重复改变容量或发布通知。

## Encoder 与监控

EncoderEndpoint 在选址时通过 tryRegisterSelection 记录精确 RequestRegistration，立即计入并发和未命中工作量。
commitSelection 将完整路由关联到同一登记，不再次扣减容量；是否已关联路由决定 REGISTERED/COMMITTED 阶段。
提交和下线使用同一把状态锁，完成、取消或下线后清除对应记录。
登记接收请求侧计算好的未命中 token 数，不从请求 Context 读取输入长度与缓存命中参数。

调度读取不可变快照。监控只在 Endpoint 空闲时捕获一致快照，忙时跳过采样，
通过 tryLock 避免等待。预测、指标格式化和通知在请求记录变更之外执行。
