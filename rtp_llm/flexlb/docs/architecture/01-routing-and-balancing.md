# Routing and Balancing

核心入口是 `RequestRouteSelector`，协调 Prefill/PDFusion、Decode、VIT、Encoder 的角色策略。
Scheduler 决定何时调度请求、失败后等待还是重试；Strategy 决定当前候选中选谁，以及版本冲突后如何重算。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/`、`balance/strategy/`、`balance/endpoint/`。

## 选择与登记

`select(context, group)` 返回 `RoutingResult<RouteResult, CapacityKey>`。
一次成功直接返回绑定请求与已登记角色的 `RouteResult`；队列和回调沿用该结果的精确身份。VIT 仅选择，Encoder 沿用独立提交边界。

1. 校验请求、SchedulingMetadata 和请求指定的角色。`schedule_roles` 必须属于模型配置；
   按 `ModelMetaConfig.requiredRoles()` 顺序处理，未指定时处理全部配置角色。
2. 捕获角色候选并在锁外评分。Prefill 的输入包含引擎工作、实际等待队列、在途工作，以及已选中尚未发布的请求；
   Decode 的输入包含引擎观测和本地未确认登记，包括尚未发送的并发、hard KV 和 expected KV。
3. Strategy 选出最优候选，通过回调调用该 Endpoint 的 `tryRegisterSelection`。
   Endpoint 在自己的状态锁内检查可用性、评分版本、容量，并登记本次影响。
   相同快照的并发登记只有一个能成功；之后的请求能看到已登记影响。
4. 版本冲突时，Strategy 只刷新和重算冲突的候选，再与其余缓存分数比较。
   Prefill 复用本轮 cache 查询；Decode 只重算该节点的成本。每次角色选择最多尝试 8 次，耗尽后返回 `RETRY`。
   节点不可用时从本轮候选移除；代际信息留在 Strategy/Endpoint 边界内。
5. 某角色失败或请求已取消时，Selector 在返回前释放本轮所有已登记资源。
   普通等待队列不保留部分 P/D 登记。清理失败抛出异常，不能作为可重试失败交回队列。
6. 全部角色成功后一次构造 `RouteResult`，交给 Scheduler 发布；没有结果包装、补登记或资源二次转交。
   指定的策略 group 始终优先；未指定时，首个成功角色的 group 约束后续角色。

角色之间不同时持有 Endpoint 锁，不存在先占 P 锁再等 D 锁的过程。
预测、cache 查询和资源通知均在 Endpoint 状态锁外；没有全局选点串行队列。

选点元数据由 RoleSelection 冻结。组批构造 RPC 地址、抢占构造取消目标时直接读取冻结的地址字段，
内部直接读取冻结的角色字段；对外响应才由 RouteResult 生成独立协议对象。
这些字段不从 Endpoint 的后续动态观测重新读取，保持已选路由的地址一致性。

## 数据归属

| 粒度           | 类型                        | 保存的信息                                                                                     |
| -------------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| 单角色事实     | `RoleSelection`             | Endpoint、冻结拓扑、DP rank、对外协议需要的观测                                                |
| Prefill 事实   | `RoleSelection.Prefill`     | 本请求预计工作量、精确 Prefill 登记                                                            |
| Decode 事实    | `RoleSelection.Decode`      | 精确 Decode 登记                                                                               |
| 单次多角色结果 | `RouteResult`               | Context、冻结 requestId、有序 RoleSelection 引用；同时作为队列和回调的精确身份，不复制角色字段 |
| 生命周期       | `ContextResult.Snapshot`    | 引用 RouteResult，记录发布、交付、取消和清理事实                                               |
| 路由上下文     | `RoutingContext`            | 保存请求上下文和待撤回路由；以对象身份区分当前路由过程，不复制资源或分数                       |
| 诊断           | `RoutingDecision`           | 选择原因与候选估计；不能代替资源账本                                                           |
| 外部响应       | `Response` / `ServerStatus` | 按需从 RouteResult 生成独立协议对象                                                            |

Selector 用方法内的角色列表收集已登记结果，失败时撤销其中的登记；列表不进入等待队列，也不构造 SelectionAttempt 中间对象。
不存在对外暴露的 RouteResult.Builder；ViT 服务只返回自己的 RoleSelection，不能修改整个组合。
角色循环统一检查成功选择恰好新增一个对应角色；P/D 和 Encoder 复用策略返回的结果，
不再把成功结果二次包装成另一份单角色 RoutingResult。角色事实只从已登记列表读取。
评分版本只在 Strategy/Endpoint 登记时校验，不保存在已完成的角色结果中。
Prefill 登记保存不可变的排队输入和本次工作量，可以立即参与其他请求的投影，无需读取半成品路由。
`RouteResult` 构造时角色组合完整，并绑定已取得的 Prefill 登记；没有 attachReservation 入口，也不再创建另一个执行包装对象。
规划直接使用 Prefill 登记中的不可变输入，不再另建规划缓存、加锁初始化或重复构造输入。
`RoutingContext` 保存 `requestContext` 和可为空的 `routeToWithdraw`，不实现 AutoCloseable；Scheduler 通过对象身份核对当前路由过程，并负责完成及清理。
同 ID 的重选使用新的执行身份，旧回调只能释放旧记录。

## Prefill / PDFusion 策略

`RequestTimeEstimation` 仅根据 Endpoint 冻结的队列、工作量和模型计算预测成本与可建模性；
`CostBasedPrefillStrategy` 比较预测与缓存亲和，并登记选中 Prefill 上本请求的工作量。
DIRECT 模式同时检查该 Prefill 的请求容量。DecodeSelector 只检查本请求允许的 Decode 候选。
全局发送许可、队首请求的 Decode 派发容量由派发阶段检查，不影响另一个请求的 Prefill 选点或登记。
预测不读取 Scheduler 的上次 `Decision`；等待决策和 Decode / 执行器容量变化不使预测缓存失效。
队列和工作量的真实变化仍由 Endpoint 更新版本；登记时校验真实版本，冲突时仅刷新相应候选。
缓存匹配来自统一的 `CacheAwareService`：本地命中按全量折算，远端增量按
`cacheAffinity.remoteDiscount` 折算；详见 [缓存同步](04-worker-sync-and-cache.md)。

排序保留配置的 TTFT/cache affinity 规则。最优候选容量不足时返回该节点的 `BLOCKED`，
不因为登记失败直接退到次优节点。下次因资源变化调度时重新比较允许候选；不永久绑定上次节点。
若版本冲突后重算表明另一个候选更好，可以改选另一个候选；这与容量不足时绕过原策略不同。

## Decode 策略

`DecodeSelector` 先区分 READY、BUSY、IMPOSSIBLE，再用配置的成本公式比较同组候选，平分时轮转。
DIRECT 和 QUEUE 的选点准入均计入本地尚未发送的登记，避免同时向低负载节点集中承诺容量。
BUSY 节点仍可作为阻塞原因或抢占候选，但登记不成功就不会返回成功路由。
静态容量不足返回终态拒绝；临时容量不足由 Scheduler 决定等待或抢占。
发送许可使用独立的实际派发容量检查，复用同一请求登记，不再次创建请求资源记录。

## VIT 与 Encoder

VIT 使用 `VitWorkerSelector`；已给出的 VIT cache 选择通过 `VitCacheSelector` 校验。
VIT 不登记 Prefill/Decode 容量；cache 结果失效返回 `VIT_ROUTE_STALE`，不能偷偷换节点。

Encoder 使用 `EncoderSelectionStrategy`：有 `encoder_cache_hit_len` 时比较未命中工作量，
并列时比较并发；未携带时比较并发，不按 KV cache 筛选。
Encoder 单角色按 DIRECT 或独立 QUEUE 提交；QUEUE 受 `maxInflightPerEncoderWorker` 限制。
EPD 的 Encoder 与 Generation 分阶段请求；本次选点登记变更不扩大混合阶段请求范围。

### gRPC Schedule 响应

`FlexlbService.Schedule` 的 `FlexlbServerStatusPB` 使用 `optional int32 engine_index = 9`。
字段号 5、7 保持 reserved。N=1 不设置字段；N>1 设置所选 index，包含显式 0，调用方应通过
字段 presence 区分这两种情况。`FlexlbServiceImpl` 从内部 `ServerStatus` 转换时保留该语义；
master forwarding 透传 protobuf 响应。物理地址字段与既有 service path 不变，旧客户端可忽略
新增字段。此字段不改变 `EnqueueBatch` 的派发协议。

## 发布、失败和清理

Direct 和 Queued 各自实现 `selectAndPublish(request, selection)`，在同一调用中开始选路、执行策略、
校验并按本调度模式发布路由，返回或抛异常前结束本轮操作。基类提供共同的状态校验和资源清理，
不根据可空队列参数选择流程。调用方不持有待发布预留，也不配对清理。
Prefill 的 SELECTED 记录进入实际队列后成为 WAITING；Direct 复用已登记的 RESERVED 记录。
Decode 复用选择时的登记；发送阶段只取得独立的 dispatch permit。
Direct/Encoder 的路由与投递在同一个 Context 快照中接收；投递前的预测或代际校验失败只回滚本轮资源，
不再提前发布 Context.route 后再通过 discardRoute 撤销。

登记存在不代表已经发布。Queue 在物理入队和请求接收成功后返回，操作完成后再通知消费者；
Direct 在提交前完成工作量和可见期限计算，接收成功后返回 RouteDelivered。交接前失败由本轮清理，
交接后由请求生命周期清理，不通过 Context 当前是否还保留 route 反推资源归属。
P/D 均只有选择登记入口；提交缺少精确登记时拒绝发布，不补建记录。
策略只有 `selectAndRegister` 合约，成功意味着登记已完成；没有纯选择与登记入口之间的兼容适配。

QUEUE 容量失败在角色/group 域等待，资源变化触发重选，不只订阅上次失败节点；等待登记使用 availability sequence 防止丢失唤醒。
DIRECT 对版本冲突或未交付代际失效使用有限重试，不进入长期等待队列。
抢占由 QueuedRequestScheduler 发起，策略和执行器统一位于 `balance.preemption`：
`DecodePreemptionPlan` 保存 `requestsToPreempt`，执行器通过 `tryPreempt` 发起抢占，
其中 `DefaultDecodePreemption.beginRoutePreemption` 校验排队路由的优先级、资源身份和可抢占状态，
再由 `RequestContext` 在请求锁内登记状态；scheduler 只通过 `allowsRoutePreemption` 声明是否允许抢占。
需要取消引擎请求时，`DefaultDecodePreemption.tryRegisterPreemption` 查找请求及取消目标，
由 `RequestContext` 登记抢占编号和精确资源预留，防止同一请求被并发重复抢占。
按请求所处阶段撤回本地路由或取消引擎请求，再处理资源释放和请求重新排队；
成功取得的 incoming Decode 登记交给 Selector 继续完成其他角色，不重复登记。
取消、超时及发送结果未知的资源规则见 [队列调度](02-queue-scheduling.md) 和 [资源管理](03-resource-management.md)。
