# Routing and Balancing

核心入口是 `RequestRouteSelector`，协调 Prefill/PDFusion、Decode、VIT、Encoder 的角色策略。
Scheduler 决定何时调度请求、失败后等待还是重试；Strategy 决定当前候选中选谁，以及版本冲突后如何重算。

主要代码：`flexlb-sync/src/main/java/org/flexlb/balance/scheduler/`、`balance/strategy/`、`balance/endpoint/`。

## 选择与登记

`select(context, group)` 返回 `PlacementResult<RouteResult, PlacementKey>`。
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
| 放置操作       | `PlacementOperation`        | 精确 Context/撤回身份，不复制资源或分数                                                        |
| 诊断           | `RoutingDecision`           | 选择原因与候选估计；不能代替资源账本                                                           |
| 外部响应       | `Response` / `ServerStatus` | 按需从 RouteResult 生成独立协议对象                                                            |

Selector 用方法内的角色列表收集已登记结果，失败时撤销其中的登记；列表不进入等待队列，也不构造 SelectionAttempt 中间对象。
不存在对外暴露的 RouteResult.Builder；ViT 服务只返回自己的 RoleSelection，不能修改整个组合。
角色循环统一检查成功选择恰好新增一个对应角色；P/D 和 Encoder 复用策略返回的结果，
不再把成功结果二次包装成另一份单角色 PlacementResult。角色事实只从已登记列表读取。
评分版本只在 Strategy/Endpoint 登记时校验，不保存在已完成的角色结果中。
Prefill 登记保存不可变的排队输入和本次工作量，可以立即参与其他请求的投影，无需读取半成品路由。
`RouteResult` 构造时角色组合完整，并绑定已取得的 Prefill 登记；没有 attachReservation 入口，也不再创建另一个执行包装对象。
规划直接使用 Prefill 登记中的不可变输入，不再另建规划缓存、加锁初始化或重复构造输入。
`PlacementOperation` 是操作身份，不实现 AutoCloseable；实际完成及清理由 Scheduler 的完成路径负责。
同 ID 的重选使用新的执行身份，旧回调只能释放旧记录。

## Prefill / PDFusion 策略

`CostBasedPrefillStrategy` 用 `RequestTimeEstimation` 同时计算排序成本和准入状态。
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

Scheduler 调用 `publishPlacement`，在同一尝试身份上校验并发布队列或直接路由。
Prefill 的 SELECTED 记录进入实际队列后成为 WAITING；Direct 复用已登记的 RESERVED 记录。
Decode 复用选择时的登记；发送阶段只取得独立的 dispatch permit。

登记存在不代表已经发布。发布责任以 Context 的精确 route/placement 事实交接；未发布失败由本轮清理，
已发布请求由生命周期清理。P/D 均只有选择登记入口；提交缺少精确登记时拒绝发布，不补建记录。
策略只有 `selectAndRegister` 合约，成功意味着登记已完成；没有纯选择与登记入口之间的兼容适配。

QUEUE 容量失败在角色/group 域等待，资源变化触发重选，不只订阅上次失败节点；等待登记使用 availability sequence 防止丢失唤醒。
DIRECT 对版本冲突或未交付代际失效使用有限重试，不进入长期等待队列。
抢占由 QueuedRequestScheduler 发起：原有 eviction 逻辑取消 victim、释放资源并恢复受害请求队列；
成功取得的 incoming Decode 登记交给 Selector 继续完成其他角色，不重复登记。
取消、超时及发送结果未知的资源规则见 [队列调度](02-queue-scheduling.md) 和 [资源管理](03-resource-management.md)。
