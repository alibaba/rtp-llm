# Routing and Balancing

核心入口是 `RequestRouteSelector`，协调 Prefill/PDFusion、Decode、VIT、Encoder 的角色策略。
Scheduler 决定何时调度请求、失败后等待还是重试；各角色 Strategy 负责捕获候选、评分和登记。
P/D/V/E 策略继承 `AbstractWorkerSelectionStrategy`，共用有界登记重试循环；`RequestRouteSelector` 只组合角色结果并负责跨角色回滚。

主要代码：`flexlb-balance/src/main/java/org/flexlb/balance/strategy/`、`balance/schedule/`，
以及 `flexlb-worker/src/main/java/org/flexlb/worker/endpoint/`。

## 选择与登记

`select(context, group)` 返回 `RequestResult`。
结果直接保存成功选择的角色、失败原因或容量等待条件；成功结果绑定请求与已登记角色。队列和回调沿用该结果的精确身份。VIT 仅选择；Encoder 在选址时登记容量，提交完整路由时关联同一份登记。

1. 校验请求、SchedulingMetadata 和请求指定的角色。`schedule_roles` 必须属于模型配置；
   按 `ModelMetaConfig.requiredRoles()` 顺序处理，未指定时处理全部配置角色。
2. 捕获角色候选，在 Endpoint 锁外评分。Prefill 读取计算 token 快照并查询一次 cache；
   Decode 读取引擎观测与本地登记，Encoder 读取并发和未命中工作量。
   每个角色策略提供完整选址入口，评分方法返回候选；登记及元数据构造失败时由该策略回滚本次登记。
3. `AbstractWorkerSelectionStrategy` 调用评分方法，再调用子类的登记和刷新方法访问 Endpoint。
   P/D/E 的 `tryRegisterSelection` 在状态锁内检查接纳条件、记录身份和容量；P/D 还核对快照版本。
   成功登记返回实际的 `RoleSelection`，公共循环只在登记成功后返回。
4. 共同循环只刷新版本冲突的候选，最多处理 8 次冲突；容量不足不消耗冲突次数。
   Prefill 和普通 VIT 可尝试其他候选；Decode 和 Encoder 的容量不足返回当前阻塞条件。
   不可用候选被排除，本轮候选全部失效时返回重试，由下一轮重新捕获目录。
   Prefill 刷新时复用本轮 cache 查询；Decode 的评分缓冲区在单次策略调用的 finally 中释放。
5. 某角色失败或请求已取消时，`RequestRouteSelector` 在返回前通过 `SelectionRollback.rollbackSelection`
   回滚本轮已登记资源，P/D/E 各自核对精确登记与允许回滚的阶段。
   普通等待队列不保留部分 P/D 登记。清理失败抛出异常，不能作为可重试失败交回队列。
6. 全部角色成功后一次构造 `RequestResult`，交给 Scheduler 发布；没有结果包装、补登记或资源二次转交。
   指定的策略 group 始终优先；未指定时，首个成功角色的 group 约束后续角色。

角色之间不同时持有 Endpoint 锁，不存在先占 P 锁再等 D 锁的过程。
预测、cache 查询和资源通知均在 Endpoint 状态锁外；没有全局选点串行队列。

选点元数据由 RoleSelection 冻结。返回路由、抢占构造取消目标时直接读取冻结的地址字段，
内部直接读取冻结的角色字段；对外响应才由 RequestResult 生成独立协议对象。
这些字段不从 Endpoint 的后续动态观测重新读取，保持已选路由的地址一致性。

## 数据归属

| 粒度           | 类型                              | 保存的信息                                                                                     |
| -------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- |
| 单角色事实     | `RoleSelection`                   | Endpoint、冻结拓扑、DP rank、对外协议需要的观测                                                |
| Prefill 事实   | `RoleSelection`                   | 本请求预计工作量、精确 Prefill 登记                                                            |
| Decode 事实    | `RoleSelection`                   | 精确 Decode 登记                                                                               |
| 单次多角色结果 | `RequestResult`                   | Context、冻结 requestId、有序 RoleSelection 引用；同时作为队列和回调的精确身份，不复制角色字段 |
| 生命周期       | `RequestContext`                  | 引用 RequestResult，记录发布、交付、取消和清理事实                                             |
| 选址凭据       | `RequestContext.SelectionAttempt` | 只读的请求引用和选址编号；状态只由 RequestContext 持有                                         |
| 诊断           | `RoutingDecision`                 | 选择原因与候选估计；不能代替资源账本                                                           |
| 外部响应       | `Response` / `ServerStatus`       | 按需从 RequestResult 生成独立协议对象                                                          |

Selector 用方法内的角色列表收集已登记结果，失败时撤销其中的登记；列表不进入等待队列，也不构造 SelectionAttempt 中间对象。
不存在对外暴露的 RequestResult.Builder；ViT 服务只返回自己的 RoleSelection，不能修改整个组合。
角色循环统一检查成功选择恰好新增一个对应角色；每个角色的成功结果直接加入列表，角色事实只从已登记列表读取。
评分版本只在 Strategy/Endpoint 登记时校验，不保存在已完成的角色结果中。
Prefill 登记保存不可变的排队输入和本次工作量，可以立即参与其他请求的投影，无需读取半成品路由。
`RequestResult` 构造时角色组合完整，并绑定已取得的 Prefill 登记；没有 attachReservation 入口，也不再创建另一个执行包装对象。
规划直接使用 Prefill 登记中的不可变输入，不再另建规划缓存、加锁初始化或重复构造输入。
RequestContext 同时持有请求输入、遥测和生命周期事实，不再有 RequestData 或 ContextResult 容器。
选址开始时分配递增编号，回调同时核对请求对象和编号；完成后清除活动编号，旧回调不能完成下一次选址。
Snapshot 在读取时生成，仅是不可变视图；请求不维护另一份完整状态副本。
同 ID 的重选使用新的执行身份，旧回调只能释放旧记录。

## Prefill / PDFusion 策略

PrefillTimePredictor.computeTokens 统计输入 tokens 减去可复用缓存 tokens。
CostBasedPrefillStrategy 按节点当前工作量加本请求 compute tokens 选择节点，
并跳过超过 router.roles.prefill.maxComputeTokens 的候选。
cacheAffinity 启用时优先选择命中更多且工作量不超过 cacheAffinity.maxComputeTokens 的节点；
该上限省略时使用普通 Prefill 上限。无合适缓存候选时仍按工作量选择满足上限的节点。

缓存匹配来自 CacheAwareService；本地命中按全量折算，远端增量按 remoteDiscount 折算。
登记失败会尝试其他候选，版本冲突会重新读取快照。全部候选容量不足时由全局队列等待唤醒。

## Decode 策略

`RequestRouteSelector` 根据 Endpoint 容量判断 READY、BUSY、IMPOSSIBLE；
`DecodeStrategy` 只用配置的成本公式比较同组候选，平分时轮转。
DIRECT 和 QUEUE 的选点准入均计入本地尚未发送的登记，避免同时向低负载节点集中承诺容量。
BUSY 节点仍可作为阻塞原因或抢占候选，但登记不成功就不会返回成功路由。
静态容量不足返回终态拒绝；临时容量不足由 Scheduler 决定等待或抢占。
`tryRegisterSelection` 在同一次持锁中核对容量并登记，后续提交复用同一份登记。

## VIT 与 Encoder

VIT 使用 `VitStrategy`；普通选择与缓存亲和选择均经过基类的登记重试循环。
`VitStrategy` 从 `VitCacheDirectory` 查询缓存信息、比较缓存层级命中，复查选中节点后通知目录记录临时 key 去向；该记录不是请求容量预留。
已给出的 VIT cache 选择通过 `VitStrategy` 校验节点身份、代际和 group。
VIT 不登记 Prefill/Decode 容量；cache 结果失效返回 `VIT_ROUTE_STALE`，不能偷偷换节点。

Encoder 使用 `EncoderStrategy`：有 `encoder_cache_hit_len` 时比较未命中工作量，
并列时比较并发；未携带时比较并发，不按 KV cache 筛选。
Encoder 单角色按 DIRECT 或独立 QUEUE 提交；QUEUE 受 `maxInflightPerEncoderWorker` 限制。
EPD 的 Encoder 与 Generation 分阶段请求；本次选点登记变更不扩大混合阶段请求范围。

### gRPC Schedule 响应

`FlexlbService.Schedule` 的 `FlexlbServerStatusPB` 使用 `optional int32 engine_index = 9`。
字段号 5、7 保持 reserved。N=1 不设置字段；N>1 设置所选 index，包含显式 0，调用方应通过
字段 presence 区分这两种情况。`FlexlbServiceImpl` 从内部 `ServerStatus` 转换时保留该语义；
master forwarding 透传 protobuf 响应。物理地址字段与既有 service path 不变，旧客户端可忽略
新增字段。

## 发布、失败和清理

Direct 和 Queued 各自实现 `selectAndPublish(request, selection)`，在同一调用中开始选路、执行策略、
校验并按本调度模式发布路由，返回或抛异常前结束本轮操作。基类提供共同的状态校验和资源清理，
不根据可空队列参数选择流程。调用方不持有待发布预留，也不配对清理。
Prefill 提交选择时取得的请求登记，不进入第二层请求队列。
Decode 复用选择时的登记，不再登记第二份容量。
Direct/Encoder 的路由与投递在同一个 Context 快照中接收；投递前的预测或代际校验失败只回滚本轮资源，
不再提前发布 Context.route 后再通过 discardRoute 撤销。

登记存在不代表已经提交。Direct 和 Queue 均在本次选址中完成资源提交，随后直接准备路由响应，
不等待 RouteDelivered 或独立发送器回报。提交前失败由本轮回滚，提交后由请求生命周期清理；
选址期间到达的取消和完成反馈在选址收尾后处理。
P/D 均只有选择登记入口；提交缺少精确登记时拒绝发布，不补建记录。
策略只返回候选；共同 Selector 确认登记成功后才返回 P/D 选择结果，没有策略调用登记器的反向回调。

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

请求状态查询、结束原因、响应选择与抢占完成使用 RequestResult 的对应字段，不再建立单独状态或结果包装。
RoleSelection 用角色字段区分 Prefill、Decode 和辅助角色，资源登记与 compute tokens 直接保存在同一个对象中。
