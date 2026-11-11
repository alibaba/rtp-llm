# FlexLB 开发术语与命名约定

用于功能开发、重构、审核、注释和设计讨论。目标是让读者从名称看懂对象保存什么、
方法改变什么、状态正在等待什么。依据当前实现及已确认的命名约定维护，不把历史类名
或尚未落地的方案当作现行架构。

## 使用规则

- 先说明业务含义，再选择下表中的词；不要先找一个英文词，再给现有实现套解释。
- 同义概念统一命名；存在真实语义差异时保留区别。`Preemption`、`Cancel`、`Release`
  是不同动作，不能为了统一名字而合并。
- 新命名使用明确的业务对象和动作。改名时检查调用方、测试、反射字符串、Spring 装配
  和现有架构说明；不因术语清理改变重试、超时、状态转换或资源释放语义。
- 下表的代码名称是定位示例。以实际字段、调用方和测试确认行为，不要求为每个词新建类。
- 内部旧名应在授权范围内迁移；协议字段、配置键、日志检索字段和指标名需要兼容评估。
  例如 Java 中可用 `allowedPreemptionStages`，现有配置键 `allowedVictimStages` 仍需保持兼容。

## 请求、角色与身份

| 中文                | 统一英文 / 命名示例                  | 在 FlexLB 中的含义与边界                                                        |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| 请求                | `Request`、`requestId`               | 一次业务请求。仅有 request ID 不足以识别所有阶段、重试及资源记录。              |
| 请求上下文          | `RequestContext`、`requestContext`   | 保存请求生命周期中的事实，并在请求锁内校验和修改状态；不是选址结果。            |
| 请求阶段            | `RequestPhase`、`requestPhase`       | 同一业务请求的处理阶段；当前为 `ENCODER`、`GENERATION`，不是运行/完成状态。     |
| 请求状态            | `RequestState`                       | 请求的可查询状态；不要把 worker 状态或某个 RPC 结果当成整个请求状态。           |
| Worker              | `Worker`、`worker`                   | 承担模型服务的工作节点或实例。不要与 master 的调度线程混用。                    |
| 引擎                | `Engine`、`engine`                   | Worker 侧执行推理、上报状态并接受请求的执行系统；本地预留不代表引擎已经执行。   |
| Worker 端点         | `WorkerEndpoint`、`endpoint`         | master 侧管理某个 worker 的对象，含角色相关的请求和资源操作；不只是 IP 字符串。 |
| 角色                | `RoleType`、`role`                   | Worker 承担的职责，如 Prefill、Decode、Encoder；不是生命周期状态。              |
| Prefill             | `PREFILL`、`prefill`                 | 处理输入上下文、建立后续生成需要的 KV；Prefill 完成不一定代表整个请求完成。     |
| Decode              | `DECODE`、`decode`                   | 执行后续 token 生成；与 Prefill 的容量、完成反馈分别处理。                      |
| PD 合设             | `PDFUSION`                           | 同一角色承担 Prefill 与 Decode；不要默认每条路由都有独立的 P、D 端点。          |
| Encoder / ViT       | `ENCODER`、`VIT`                     | 多模态编码相关角色；按当前角色配置和实现区分，不当作完全相同的能力。            |
| Worker 分组         | `group`、`workerGroup`               | 候选 worker 的分组边界；与一次调度选中的请求组不同。                            |
| 决策请求组          | `DecisionGroup`、`decisionGroup`     | 一起参与一次调度决策的请求集合；不必等于一次实际发送的 batch。                  |
| Worker 代际         | `endpointGenerationId`               | 区分旧 worker 实例及替换后的新实例；相同地址不代表相同代际。                    |
| 预留编号            | `reservationToken`                   | 区分某个请求的不同资源预留记录；释放、抢占及回调必须核对相应记录。              |
| 批次编号            | `batchId`                            | 标识协议中的交付批次；不是 request ID、抢占编号或资源预留编号。                 |
| 逻辑地址 / 物理地址 | `logicalAddress` / `physicalAddress` | 内部逻辑 worker 身份与实际连接地址分开；内部索引标记不能直接当作网络拨号地址。  |

身份字段必须写明粒度和比较方式。当前 `RoutingContext` 通过对象身份区分路由过程，
不能用相同字段值或相同 request ID 替代；资源预留则需要核对 worker 代际、请求和预留编号。

## 调度、选址与路由

| 中文               | 统一英文 / 命名示例                | 在 FlexLB 中的含义与边界                                                                |
| ------------------ | ---------------------------------- | --------------------------------------------------------------------------------------- |
| 调度               | `Scheduling`、`Scheduler`          | 决定请求何时处理、按什么顺序处理，以及何时等待或重试。                                  |
| 选址 / 选择 Worker | `Selection`、`select`              | 从候选 worker 中选择目标。若同时登记资源，应使用 `selectAndRegister` 等名称明确副作用。 |
| 路由               | `Routing`、`Route`                 | 为请求确定角色与 worker 的组合，并处理相关路由流程；不直接等于发送成功。                |
| 单角色选择结果     | `RoleSelection`                    | 一个角色选中的 endpoint、固定元数据及对应资源登记。                                     |
| 完整路由结果       | `RouteResult`                      | 一个请求的角色选择组合；与一个 role 的结果分开。                                        |
| 路由处理结果       | `RoutingResult`                    | 表达某一步成功、阻塞、需要重试、拒绝或关闭；成功值可能是单角色结果或完整路由。          |
| 路由上下文         | `RoutingContext`、`routingContext` | 识别请求当前的一次路由处理，并关联需要移除的旧路由；不是 `RouteResult`。                |
| 路由阶段           | `RoutingPhase`、`routingPhase`     | 当前正在选址、结束选址、移除旧路由或完成移除；不是引擎执行阶段。                        |
| 路由发布           | `publishRoute`                     | 将选好的路由提交到后续队列或处理流程；仅选中 worker 不代表发布成功。                    |
| 调度队列           | `SchedulingQueue`                  | 保存待调度请求及其顺序。                                                                |
| 容量等待队列       | `CapacityWaitQueue`                | 保存因容量条件不满足而等待的请求，并根据容量变化安排重新检查。                          |
| Worker 请求队列    | `PrefillRequestQueue` 等           | 保存已经关联到具体 worker、等待后续处理的请求；与全局调度队列分开。                     |
| 重试               | `retry`                            | 重新执行明确的一段流程；名称或邻近代码说明是重新选址、重新发布还是其他步骤。            |
| 重新入队           | `requeue` / `onRequest`            | 将仍有效的请求交回调度；不等同于立即选址或重新发送。                                    |
| 回滚               | `rollback`                         | 撤销未成功完成的本地登记或准备工作；发送结果未知时不能当作从未发送而回滚。              |

`DIRECT` / `QUEUE` 是调度模式；`NON_BATCH` / `BATCH` 是交付模式。
FIFO / PRIORITY 表达排队顺序。这些概念不能互换，也不能仅从是否有 batch 推断调度策略。

## 容量、资源与通知

| 中文           | 统一英文 / 命名示例                 | 在 FlexLB 中的含义与边界                                                            |
| -------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| 容量           | `Capacity`                          | 某类资源可接纳请求的约束，明确是请求数、KV token 还是其他单位。                     |
| 准入           | `Admission`、`AdmissionCapacity`    | 判断能否接纳请求的资源条件；不表示引擎已经收到或执行请求。                          |
| 容量使用量     | `CapacityUsage`                     | 当前容量计算的输入。实际可用 KV、本地硬预留和预估使用量必须区分。                   |
| 资源预留       | `Reservation`、`RequestReservation` | 本地为请求登记的容量关联；不代表引擎物理资源已经分配。                              |
| 执行许可       | `Permit`、`EngineDispatchPermit`    | 允许跨过某个交付边界的具体许可；说明取得、转交和释放时机，不笼统称为所有权。        |
| 资源登记       | `Registration`、`register`          | 将请求及其资源影响登记到明确对象；说明登记什么、重复登记如何处理。                  |
| 资源释放       | `Release`、`releaseResources`       | 更新或释放指定资源，名称需说明范围；不暗示删除整个请求或完成全部清理。              |
| 容量标识       | `CapacityKey`                       | 标识 role、group 及可选 endpoint，用于等待、通知和诊断；不是资源预留。              |
| 容量变化通知器 | `CapacityNotifier`                  | 记录容量/可用性变化版本，并通知监听者重新检查；不拥有等待请求，不保证存在空闲容量。 |
| 容量变化通知   | `notifyChanged(endpoint)`           | 发布指定 worker 的变化。通知可以导致路由失效检查或等待请求重选，不是容量授予。      |
| 容量变化版本   | `sequence` / `capacityVersion`      | 判断等待登记前后是否发生变化，避免漏掉唤醒；不是请求优先级或 worker 代际。          |
| KV 缓存        | `KV cache`                          | 模型推理的 key/value 缓存资源；按实际计量区分 token 数、块数和字节数。              |
| 缓存命中       | `cacheHit`、`hitCacheTokens`        | 缓存匹配结果。历史或预测命中不自动构成资源预留，也不保证发送时缓存仍存在。          |

命名中明确单位：`timeoutMs`、`deadlineNanos`、`kvTokens`、`requestCount`。
相同数值类型不代表可以相加；真实容量、预估工作量和本地资源预留不得重复计数。

## 交付、抢占、取消与结束

| 中文                | 统一英文 / 命名示例                               | 在 FlexLB 中的含义与边界                                                           |
| ------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 交付                | `Delivery`、`DeliveryDispatcher`                  | 将路由决策或请求输入交给后续处理方；需区分返回路由决策与发送 BATCH 输入。          |
| 发送                | `send`、`SendState`                               | 某次网络/交付操作的发送事实；`UNKNOWN` 不能解释为 `NOT_SENT`。                     |
| 确认回复            | `ACK`、`acknowledged`                             | 对某个操作的接收确认，必须说明确认的是交付还是取消；不等于推理完成。               |
| Worker 请求事件     | `WorkerRequestEvent`、`workerEvent`               | 引擎上报的活动、分配、完成或失败等事实；先核对路由、角色和资源身份，再更新请求。   |
| 抢占                | `Preemption`                                      | 为高优先级请求取得容量，处理选中的低优先级请求；本地路由抢占和引擎取消是不同路径。 |
| 待抢占请求          | `requestToPreempt` / `requestsToPreempt`          | 被选中处理的请求或请求视图；不是已经抢占成功的请求。                               |
| 待抢占资源预留      | `reservationToPreempt` / `reservationsToPreempt`  | 参数实际是资源预留时使用此名，不称为 request。                                     |
| 抢占计划            | `DecodePreemptionPlan`                            | 保存计划处理的请求和容量目标；计划存在不代表抢占已登记或资源已释放。               |
| 抢占登记            | `tryRegisterPreemption`、`PreemptionRegistration` | 绑定当前抢占编号、请求资源及取消目标，防止并发重复抢占；不发送取消、不释放容量。   |
| 开始抢占路由        | `beginRoutePreemption(routeToPreempt)`            | 在允许时登记抢占当前选好但尚未交付的路由；实际移除和资源处理由后续流程执行。       |
| 取消                | `Cancel`、`cancelRequest`                         | 提出或执行停止请求的操作；区分客户端取消、超时取消和抢占取消的原因。               |
| 移除路由            | `removeRoute`、`REMOVING_ROUTE`                   | 移除已选路由及相应队列关联的阶段。可能来自抢占或路由失败，不统一叫抢占。           |
| 缓存淘汰            | `Eviction`、`evictCache`                          | 淘汰缓存条目或缓存块；与请求抢占不同，不机械替换为 `Preemption`。                  |
| 清理                | `Cleanup`、`cleanupRequest`                       | 完成必要通知、资源释放和关联移除；可以在响应完成后继续。                           |
| 归档                | `Archive`、`archive`                              | 将满足结束条件的请求保存为可查询的结束记录；不是引擎上报完成的同义词。             |
| 截止时间 / 超时长度 | `deadline` / `timeout`                            | 绝对时间点与持续时间分开；要注明时钟来源和单位。                                   |
| 上报不活跃超时      | `inactivityTimeout`                               | 在规定时间内没有匹配的引擎请求反馈；不能称为“确认引擎已经停止”。                   |
| 停止接收 / 关闭     | `stopAccepting` / `shutdown`                      | 停止接纳新请求与等待已有工作、释放组件资源分开，名称应对应实际范围。               |

抢占的策划、登记入口和执行编排属于抢占组件；请求状态的原子校验属于 `RequestContext`。
不要因为调度器首先发起调用，就把所有抢占方法放进 scheduler。

## 状态命名的边界

当前代码中的以下状态不是同一套枚举：

- `RequestPhase.ENCODER / GENERATION`：业务阶段。
- `ContextResult.Phase.OPEN / ENDING / CLOSED`：请求生命周期。
- `RoutingPhase`：选址和移除旧路由的处理阶段。
- `RouteState`：没有路由、可执行路由、仅为清理保留的路由。
- `SendState`：交付发送情况。
- `PreemptionCancelPhase`：抢占取消协议的进度。

`OPEN` 不表示 worker 正在运行；`ENDING` 不表示资源已经全部释放；Future 已完成也不表示
请求可以归档。P/D 分离时，Prefill 完成不能直接解释为整个请求完成。

状态名应能读成一个判断，例如 `waitingForWorkerFinishedReport`，并说明何时设置、
何时清除以及对调度和容量的影响。不要仅靠 `active`、`done`、`terminal` 表示多个不同事实。
不要为了好看再维护一份与既有事实重复的状态。

设计方案中的状态名和接口不能直接当作已实现契约；采用前需检查当前代码并落实迁移。

## 方法与变量如何写清楚

| 形式                        | 应表达的实际效果                         | 约束或示例                                                                                |
| --------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| `tryRegister...`            | 尝试登记某个具体对象，可能因正常竞争失败 | 写清失败返回值；不能把所有异常吞成失败。                                                  |
| `begin...`                  | 检查条件并开始一个明确阶段               | `beginRouteSelection()` 与 `beginRoutePreemption(route)` 分开，不用 `null` 切换业务动作。 |
| `record...`                 | 在本地记录一个具体事实或反馈             | 不隐含网络发送、发布用户响应或释放整条请求链路的资源。                                    |
| `selectAndRegister...`      | 选目标并登记其资源影响                   | 名字体现两个动作；成功边界与纯选择不同。                                                  |
| `publish...`                | 把具体结果交给队列、交付方或响应执行方   | 必须写对象，如 `publishRoute`、`publishResponse`，避免裸 `publish` 成为多种语义入口。     |
| `notify...`                 | 通知别人某事实发生变化                   | `notifyChanged(endpoint)` 不承诺通知接收方一定调度成功。                                  |
| `release...` / `remove...`  | 释放资源 / 移除某个记录或队列项          | 两者不是同义词，不暗示超出方法实际执行的清理。                                            |
| `finish...` / `complete...` | 完成名称中指定的对象或阶段               | 说明完成的是选址、交付、响应还是清理，不能统一当作请求完成。                              |

变量优先说明内容：`workerEvent`、`requestContext`、`routingContext`、`routeToPreempt`、
`previousState`、`updatedState`、`eventProcessing`。`ctx` 可用于上下文唯一且很短的局部代码；
同时存在多种 context 时使用完整名称。

`Context`、`Result`、`Snapshot` 不是任意添加的后缀：Context 保存处理过程所需关联，Result
表达某次处理的输出，Snapshot 表达某个时点的视图。不能只改后缀而保留含混或重复职责。

## 旧词的处理方式

| 旧词或宽泛名称                       | 新增代码的选择原则                                                                                                                                                                            |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Placement*`                         | 按职责使用 `RoutingPhase`、`RoutingContext`、`RoutingResult`、`CapacityNotifier`、`CapacityKey`、`CapacityWaitQueue`、`SchedulingQueue`、`SchedulerConfiguration`；不是统一替换成同一个前缀。 |
| 抢占中的 `Victim*` / `victim`        | 使用 `Preemption*`、待抢占请求或待抢占资源预留的具体名称。外部字段先处理兼容。                                                                                                                |
| 抢占中的 `Eviction` / `tryReclaim`   | 抢占流程使用 `Preemption` / `tryPreempt`；真实的缓存淘汰和资源回收保留其具体含义。                                                                                                            |
| `tryClaim`                           | 说明登记什么；抢占登记使用 `tryRegisterPreemption`。不把请求状态互斥误称为已经释放资源。                                                                                                      |
| `withdrawal`                         | 新业务接口优先使用“移除旧路由”或“抢占路由”的具体动作；旧内部协议枚举需按语义迁移，不自动全局替换。                                                                                            |
| `Operation` / `Attempt`              | 不作为不知道对象内容时的默认名称；先判断保存的是上下文、结果、登记还是具体编号。实际表达尝试次数时仍可用 `attemptCount`。                                                                     |
| `ownership` / `settlement` / `proof` | 必须具体说明哪个请求、哪份预留、哪个释放确认；不能用这些词同时概括执行状态、容量占用和请求关联。                                                                                              |
| `advanceRequest`                     | 说明触发事件和实际动作；若方法混合多套状态决策，先梳理职责，不以更长的泛化名字掩盖复杂度。                                                                                                    |

## 审核时的五个问题

1. 不看实现，能否知道这个变量装的是请求、路由、资源预留还是通知？
2. 方法名是否准确说明动作及完成边界？有没有通过空值或布尔参数暗中切换业务流程？
3. 是否混淆“已登记、已发布、已发送、已确认、已完成、已释放、已归档”？
4. 同义概念是否出现多套命名，或同一个词是否被用于不同事实？
5. 旧名清理是否覆盖当前改动的调用方和测试，并保留必要的外部兼容？

实现入口可从 [调度目录](../../../flexlb-sync/src/main/java/org/flexlb/balance/scheduler)、
[抢占目录](../../../flexlb-sync/src/main/java/org/flexlb/balance/preemption)、
[资源目录](../../../flexlb-sync/src/main/java/org/flexlb/balance/endpoint) 查找；
模块划分参见 [FlexLB 架构概览](../../../README.md#architecture)，与代码不一致时核对实现及测试，
不要沿用过期类图或术语。
