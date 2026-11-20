# Routing and Balancing

调度输入 `Request` 和调度响应 `Response` 位于 `balance.request`，由 Balance 与 API 使用。
Endpoint 不接收完整调度请求，也不构造调度响应：Decode 容量评估只接收硬 KV、预估 KV、
优先级和容量限制，返回同一次持锁检查的快照、缺口及容量原因；DecodeStrategy 将原因映射为
调度错误。Cache 的选路后更新只接收请求 ID、block size、缓存键和选中的 Worker。

`RequestRoutingService` 接收本地请求，负责 VIT 单独选址、已有 VIT 选择复核和成功后的缓存反馈；
普通请求交给 Scheduler。API 负责协议转换和响应交付。

`RequestRouteSelector` 按角色调用各策略的 `selectAndRegister(context, group)`，入口由 `AbstractWorkerSelectionStrategy` 声明。
PDFUSION 走 Prefill 策略的融合入口，复用同一套 Prefill 排名与登记逻辑。
Scheduler 决定何时调度请求、失败后等待还是重试；各角色 Strategy 负责捕获候选、评分和登记。
P/D/V/E 策略继承 `AbstractWorkerSelectionStrategy`，共用有界登记重试循环；`RequestRouteSelector` 只组合角色结果并负责跨角色回滚。

主要代码：`flexlb-balance/src/main/java/org/flexlb/balance/strategy/`、`balance/schedule/`，
以及 `flexlb-worker/src/main/java/org/flexlb/worker/endpoint/`。

## 选择与登记

`RequestRouteSelector.selectAndRegister(context, group)` 返回 `SelectionResult<RequestRoute, CapacityKey>`；成功值为完整路由，未成功时返回等待、重试、拒绝或关闭原因。
结果直接保存成功选择的角色、失败原因或容量等待条件；成功结果绑定请求与已登记角色。队列和回调沿用该结果的精确身份。VIT 仅选择；Encoder 在选址时登记容量，提交完整路由时关联同一份登记。

1. 校验请求、SchedulingMetadata 和请求指定的角色。`schedule_roles` 必须属于模型配置；
   按 `ModelMetaConfig.requiredRoles()` 顺序处理，未指定时处理全部配置角色。
2. 捕获角色候选，在 Endpoint 锁外评分。Prefill 读取计算 token 快照并查询一次 cache；
   Decode 读取引擎观测与本地登记，Encoder 读取并发和未命中工作量。
   每个角色策略提供完整选址入口，评分方法返回候选；登记及元数据构造失败时由该策略回滚本次登记。
3. `AbstractWorkerSelectionStrategy` 调用评分方法，再调用子类的登记和刷新方法访问 Endpoint。
   P/D/E 的 `tryRegisterSelection` 在状态锁内检查接纳条件、记录身份和容量；Encoder 同时复核请求 group。
   Prefill 核对排序使用的工作量，Decode 核对排序快照的版本；Encoder 核对请求数，以及本次评分需要的 token 工作量。
   成功登记返回实际的 `RoleSelection`，公共循环只在登记成功后返回。
4. 共同循环最多处理 8 次排序冲突；容量不足不消耗冲突次数。
   Prefill、Decode、Encoder 和普通 VIT 排除本轮容量不足的候选并比较其他机器。
   P/D/E 登记先检查当前容量；仍有容量但排序依据变化时，重新读取候选并比较。
   P/D/E 每轮评分读取当前 Endpoint 快照，在同一次遍历中过滤 group，不先复制分组列表。
   Prefill 登记后又回滚、工作量恢复原值时，不因记账版本变化而要求重排。
   Prefill 在默认公式且整次查询无缓存命中时，找到空闲候选即可结束比较；自定义公式及有缓存匹配时仍完整评分。
   Decode 全部候选已满时不调用登记；QUEUE 返回带抢占目标的阻塞结果，DIRECT 返回容量不足。
   不可用候选被排除，本轮候选全部失效时返回重试，由下一轮重新捕获目录。
   Prefill 刷新时复用本轮 cache 查询；Decode 的评分缓冲区在单次策略调用的 finally 中释放。
5. 某角色失败或请求已取消时，`RequestRouteSelector` 在返回前通过 `SelectionRollback.rollbackSelection`
   回滚本轮已登记资源，P/D/E 各自核对精确登记与允许回滚的阶段。
   普通等待队列不保留部分 P/D 登记。清理失败抛出异常，不能作为可重试失败交回队列。
6. 全部角色成功后一次构造 `RequestRoute`，通过 SelectionResult.success 交给 Scheduler 发布，不重新登记资源。
   指定的策略 group 始终优先；未指定时，首个成功角色的 group 约束后续角色。

角色之间不同时持有 Endpoint 锁，不存在先占 P 锁再等 D 锁的过程。
预测、cache 查询和资源通知均在 Endpoint 状态锁外；没有全局选点串行队列。

RoleSelection 复用选中时的 TopologySnapshot，保存本次命中量、预计工作量及精确登记。
地址、分组和 Engine 索引读取保存的拓扑；角色与 DP rank 从 Endpoint 获取，不重复保存。
对外响应由 RequestRoute 提供 requestId，通过 RoleSelection.toServerStatus 生成独立协议对象；
ViT 代际读取 Endpoint 绑定的 WorkerStatus，内部选址不再构造成功协议对象。

## 数据归属

| 粒度           | 类型                              | 保存的信息                                                                                     |
| -------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- |
| 单角色事实     | `RoleSelection`                   | Endpoint、选中时的拓扑、本请求命中量                                                           |
| Prefill 事实   | `RoleSelection`                   | 本请求预计工作量、精确 Prefill 登记                                                            |
| Decode 事实    | `RoleSelection`                   | 精确 Decode 登记                                                                               |
| 单次多角色结果 | `RequestRoute`                    | Context、冻结 requestId、有序 RoleSelection 引用；同时作为队列和回调的精确身份，不复制角色字段 |
| 生命周期       | `RequestContext`                  | 引用 RequestRoute，记录发布、交付、取消和清理事实                                              |
| 选址凭据       | `RequestContext.SelectionAttempt` | 只读的请求引用和选址编号；状态只由 RequestContext 持有                                         |
| 外部响应       | `Response` / `ServerStatus`       | 按需从 RequestRoute 生成独立协议对象                                                           |

Selector 用方法内的角色列表收集已登记结果，失败时撤销其中的登记；列表不进入等待队列，也不构造 SelectionAttempt 中间对象。
不存在对外暴露的 RequestRoute.Builder；ViT 服务只返回自己的 RoleSelection，不能修改整个组合。
角色循环统一检查成功选择恰好新增一个对应角色；每个角色的成功结果直接加入列表，角色事实只从已登记列表读取。
评分版本只在 Strategy/Endpoint 登记时校验，不保存在已完成的角色结果中。
Prefill 登记保存不可变的排队输入和本次工作量，可以立即参与其他请求的投影，无需读取半成品路由。
`RequestRoute` 构造时角色组合完整，不修改 Endpoint 登记。Selector 成功返回后，Scheduler 才调用各角色的 `confirmSelection`，传入该角色的精确登记和事件接收入口。
P/Encoder 在同一次持锁操作中确认登记并关联反馈；Decode 只关联反馈，不改变 Worker 是否已确认执行，也不转移 KV 占用。
Endpoint 不持有完整路由；事件接收入口由 Scheduler 创建并捕获本次 RequestRoute。

`firstUnavailableEndpoint(selections)` 只返回不可用的已选 Endpoint，不改写选择结果；调用方在角色选择失败和最终发布前使用。
各角色的 `confirmSelection` 在锁内返回 `COMMITTED`、`UNAVAILABLE` 或 `INVALID_REGISTRATION`，由 Scheduler 决定后续动作；发布失败后不再扫描 Endpoint 推测原因。
完整路由交接前由 Selector 回滚已取得的登记，交接后由 Scheduler 的统一失败清理释放；局部 Endpoint 提交成功不代表路由发布成功。取消和发布竞争时继续按精确登记清理，不释放其他重试的登记。

规划直接使用 Prefill 登记中的不可变输入，不再另建规划缓存、加锁初始化或重复构造输入。
RequestContext 同时持有请求输入、遥测和生命周期事实，不再有 RequestData 或 ContextResult 容器。
选址开始时分配递增编号，回调同时核对请求对象和编号；完成后清除活动编号，旧回调不能完成下一次选址。
Snapshot 在读取时生成，仅是不可变视图；请求不维护另一份完整状态副本。
同 ID 的重选使用新的执行身份，旧回调只能释放旧记录。

## Prefill / PDFusion 策略

PrefillTimePredictor.computeTokens 统计输入 tokens 减去可复用缓存 tokens。
PrefillSelectionStrategy 抽象基类负责缓存估计、原子登记和重试，具体选点只有两个实现：
CostBasedPrefillStrategy 按当前工作量加本请求 compute tokens 评分，普通 cacheAffinity 优先缓存命中，
满载时允许退到其他可接纳节点。TokenBalancedPrefillStrategy 的热请求（有效命中比例严格大于阈值）
先找最高命中节点。登记返回 `CAPACITY_FULL` 时保留该节点的等待资格，继续尝试并列节点；
登记返回 `UNAVAILABLE` 时排除该节点，按剩余节点重算最高命中。最高命中节点全部满载时等待。
冷请求比较已有工作量加完整输入长度，不利用低于阈值的缓存亲和。
两者都检查 admission.maxInflightRequests；只有 TOKEN_BALANCE 启用 Engine 等待水位准入。
compute tokens 是排序依据，不再是可配置容量上限。

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

Follower 转发 `Schedule`、`Cancel` 和 `GetRequestState` 时携带原请求的 gRPC metadata，
包括自定义键、重复值和二进制值。转发 span 有有效的 W3C trace context 时，由该 span 注入下一跳的追踪字段。

## 发布、失败和清理

DIRECT 的 `selectAndPublish` 在一次调用内开始选路、执行策略并结束本轮选路。
QUEUE 自行开始选路，等待抢占结果时可以保留本轮选路，稍后继续执行。
两者复用基类的选路执行、路由提交和资源回滚逻辑。
Prefill 提交选择时取得的请求登记，不进入第二层请求队列。
Decode 复用选择时的登记，不再登记第二份容量。
Direct/Encoder 的路由与投递在同一个 Context 快照中接收；投递前的预测或代际校验失败只回滚本轮资源，
不再提前发布 Context.route 后再通过 discardRoute 撤销。

登记存在不代表已经提交。Direct 和 Queue 均在本次选址中完成资源提交，随后直接准备路由响应，
不等待 RouteDelivered 或独立发送器回报。提交前失败由本轮回滚，提交后由请求生命周期清理；
选址期间到达的取消和完成反馈在选址收尾后处理。
P/D 均只有选择登记入口；提交缺少精确登记时拒绝发布，不补建记录。
策略只返回候选；共同 Selector 确认登记成功后才返回 P/D 选择结果，没有策略调用登记器的反向回调。

QUEUE 容量不足时保留普通队列成员，Endpoint 通知或固定周期触发下一轮检查；
没有 role/group 等待域、逐请求重试索引或通知版本号。
DIRECT 对版本冲突或未交付代际失效使用有限重试，不进入长期等待队列。
QueuedRequestScheduler 对可抢占的 Decode 容量阻塞调用 RequestPreemption。
DefaultRequestPreemption 使用具体 DecodePreemptionPlanner 计算低优先级请求组合，先登记资源等待，再执行普通取消。
DecodeEndpoint.ResourceWait 的 Future 完成后，结果处理与普通选路共用有界规划线程池。
RequestRouteSelector 接收已取得的 RoleSelection，继续选择其余角色，失败时回滚全部登记。
未提交 P/D 选择的抢占及角色策略生成计划仍是实施方案中的待完成项，当前默认执行器只处理已交付 Decode。
取消、超时及发送结果未知的资源规则见 [队列调度](02-queue-scheduling.md) 和 [资源管理](03-resource-management.md)。

RequestRoute 只保存已选角色和精确登记。RequestContext 直接保存结束状态、原因和说明；响应选择引用 ResponseCompletionExecutor.Registration，执行登记同时作为精确响应身份。查询按 Context 快照生成 RequestState，清理完成后保留同一种摘要；摘要不提供回到完整路由的导航接口。
RoleSelection 从 Endpoint 获取角色，资源登记与 compute tokens 直接保存在同一个对象中。
