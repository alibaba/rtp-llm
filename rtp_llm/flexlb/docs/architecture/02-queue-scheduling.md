# Scheduling and Request Lifecycle

## 调度所有者

`RequestScheduler` 暴露 submit 和按阶段 cancel。启动时根据配置创建
`DirectRequestScheduler` 或 `QueuedRequestScheduler`，共同使用
`AbstractRequestScheduler` 的生命周期执行逻辑。`SchedulerRuntime` 管理执行器、
计时器和关闭顺序；`RequestRepository` 维护索引，`RequestContext` 决定状态转换。

每个请求使用不可变 Request、当前 FlexlbConfig 快照和 SchedulingMetadata。
`(request_id, RequestPhase)` 是登记身份；活跃和保留终态中的重复身份都会被拒绝。
Encoder 与 Generation 可以共用字符串业务 ID，且独立取消、过期和查询。未传 phase 时
定位 Generation。原始数值 ID 独立保存，不由字符串规范化或猜测生成。

## 提交、排队和派发

DIRECT 在调用链内选点、提交资源并发布路由。QUEUE 使用有序全局队列和按 blocker 组织的
容量等待队列，FIFO / PRIORITY 决定顺序。全局等待请求不再受额外数量准入门限制，
工作量受期限、各 worker 的容量和有界派发执行器约束。

QUEUE 提交先登记 future 和调用方取消回调，再把工作交给调度线程。入口只验证协议结构
和规范 int64 身份；准备任务通过 `GenerateInputPayload` 解析元数据并验证 token 编码，
保留编码后的 token 字节供 Engine 解码，避免在 Master 分配 token 数组。派发时只更新
路由、优先级等元数据。NON_BATCH 不保留生成输入；归档时清除已缓存的输入，已取得输入
的派发任务仍可使用自己的不可变引用。准备期间取消或过期不会进入后续选点。

选点捕获 endpoint generation pin。调度器在任何对外发布前提交预留、RequestRoute 和
对应的 DeliveryClaim；旧 claim、旧 batch 或旧 endpoint 的回调不能变更新代际。

`PrefillQueueRunner` 根据独立的 GroupingPolicy 组织决策组：SINGLE、FIXED_WINDOW 或
执行预算分组。`DeliveryDispatcher` 负责 BATCH EnqueueBatch；NON_BATCH 将路由交给
调用方。两者共用容量和结果协议。分组策略与 dispatcher 类型是独立配置。
每个已提交组在交付回调可见前记录 group ID、冻结策略、成员数、原因和等待时间。
`maxCollectionWaitMs` 的热更新影响后续收集及预测，不改变已经提交的组。
选组和 Decode 选择复用线程内的临时列表，返回前清除引用；递归调用使用独立存储，
已发布的决策组始终是不可变副本。

独立 ViT 选路查询（`vit_route_only=true`）不创建 Generation 生命周期，也不要求
`generate_input`；这一规则同样适用于 BATCH 派发配置。

## Encoder

模型包含 Encoder 且启用 QUEUE 时创建独立 `EncoderQueueCoordinator`。
队列遵循 FIFO / PRIORITY，用 `maxInflightPerEncoderWorker` 约束
`running + waiting + 本地待观察请求`。状态更新、结束和退役唤醒等待者；发布等待前
重查容量版本，防止漏掉释放事件。

Encoder 选点进入 DISPATCHING，成功路由提交后为 ACKNOWLEDGED。Frontend 发送实际
编码工作；FlexLB 不做 Encoder EnqueueBatch 或 Encoder Cancel RPC。WorkerStatus、
退役和失活超时只结束 Encoder 阶段，Generation 记录不受影响。

## 取消与清理

`cancel(requestId, expectedBatchId, reason, phase)` 校验精确身份和 batch。
客户端完成响应与引擎资源释放分别记录。尚未发出的预留可以本地回滚；可能已到达引擎
的请求必须保留关联和容量义务，直到 Finished、有效 fence、退役或有界清理证据到达。
取消统一通过 `EngineGrpcClient.cancelAsync` 发起：普通取消先检查 Engine 清理能力，
优先级抢占沿用原 Cancel 协议；RPC 使用独立于调用方取消的上下文，且不做传输层重放。
每个精确 delivery 由 Dispatcher 维护一份停止记录，普通取消、抢占和迟到 ACK 共同推进该记录；
重试预算与在途 RPC 也由该记录保存，不重新启动独立清理任务。停止事件使用独立执行器，
仍计入请求任务的登记和停服排空；网络等待不占住执行器线程。资源释放继续由调度器结合
Worker 反馈和明确清理证据判断。
正在等待抢占反馈的请求同样可以完成客户端取消响应并启动派发清理，不以取消响应代替
远端资源释放证据。

FULL WorkerStatus 中缺失已确认 Decode 任务时，只清除一次容量统计，保留原请求与
reservation token 等待晚到 Finished。重复缺失不会重复释放；重新出现的任务恢复统计。
终态快照不保留整个 RequestContext，按配置保留期清除，之后才允许复用身份。

## Decode 抢占

`DecodeAdmission` 规划精确 victim 集合。RPC 模式先确认所有 victim 的生命周期
所有者都存在，再原子占用并发/KV 释放计划，向原 worker 发 Cancel 并等待权威反馈。
中途 claim 失败会撤回已经取得的 claim。ACK 阶段结束时登记等待完成的截止时间，
固定本次 incoming 能否使用释放容量的结果；后续协调排队不延长预算。等待结束只移除
本次观察者，晚到的 Worker 事实仍可收敛 victim 的资源记录。
`NOT_FOUND` 不把抢占前的历史分配视为恢复证据；只有本次抢占登记后观察到的 ACTIVE
或 Decode KV_ALLOCATED，才用于确认任务已恢复并收敛对应 claim。

RETURN 要求 NON_BATCH：原子登记 victim 与 incoming reservation，将字符串 victim ID
随 Decode 路由返回，由客户端透传执行。incoming 获得专用容量许可，不能让旁路请求
借用已承诺释放的容量。交付前 LOCAL_ROLLBACK 可以撤回计划；交付后或 incoming 已结束
不能把仍在执行的 victim 误当作空闲。晚到 Finished 继续按精确身份收敛。

`DecodeAdmission` 将 incoming reservation 和需要返回的 victim ID 列表一起交给
`QueuedRequestScheduler.Plan`；只有当前请求仍可准入且预留成功转为排队状态时才接受。
被拒绝的交接由 `DecodeAdmission` 回滚预留，已接受的交接由 Plan 在路由发布前负责回滚。
容量限制使用请求建立时保存的配置，后续配置更新不改变该请求已经取得的容量约束。

## 停服

停止登记后依次关闭 Generation 和 Encoder 放置队列，等待准入，取消活跃请求，排空
派发与清理，再关闭 endpoint、计时器和请求执行器。清理失败继续处理其他资源，并保留
第一原因及后续异常供上层报告。状态锁、代际锁内不运行用户 future 回调或远程 RPC。
