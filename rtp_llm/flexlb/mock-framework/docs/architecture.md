# 架构与设计边界

[返回使用说明](../README.md)

Python/Java 模块、主要类与调用关系的逐项说明见[钉钉架构文档](https://alidocs.dingtalk.com/i/nodes/6LeBq413JA9BOdm2iqd79zQ4JDOnGvpb)。Client 请求查询位于 `interfaces/client/`，运行管理位于 `execution/client_group.py`；输入生成代码只供测试使用，不提供独立请求查询 CLI。

本文说明组件职责、调度与协议边界，以及接口扩展方式。运行步骤见 README，Case 与启动配方用法见 [Case 指南](case-guide.md)。设计目标中尚未实现的能力仍按原文标明。文中路径和命令均以框架目录为基准。

## Scheduler 输入、执行完成、传输完成与状态查询

Scheduler 是每个后端的请求状态与调度策略所有者，不只是选择 batch 的函数。公共接入代码解析协议、检查格式并发送回执；`input` 决定初始状态、队列与资源准入，成功登记不等于资源已就绪。计算结果、传输结果和协议许可分别进入明确入口，外部组件不直接修改请求阶段或队列。

- `input(request, listener)`：登记请求并执行后端准入策略。
- `schedule()`：返回本次 Forward 的执行计划，不暴露或取走内部 running 容器。
- `computeCompleted(work, result)`：提交本次计算进度，决定继续计算、等待交接或终结。Compute 不修改后端状态。
- `transferCompleted(request, result)`：接收异步 KV 传输结果，由后端决定下一步；传输结束不自动等于整个请求结束。
- `control(request, event)`：处理后端协议许可等少量控制输入；GENERATE、LOAD ACK 等类型属于原生接入与 RTP 后端的约定，不是共享状态机。
- `tasks()`：按需返回当前执行任务的不可变观测快照，不推进状态机；内部状态机和队列仍由后端自行定义。

PD 的初始通知/初始化和 KV 传输分开。当前接入协议共用，KV 传输使用公共异步组件，第一版只建模零耗时完成；保留完成、失败、取消和资源使用权接口，不加入带宽、分块或争用模型。P 计算完成后可以退出计算集合，但仍持有 KV；本地传输结束和 D 的 LOAD ACK 是不同事实，确认前不能释放源 KV。网络协议仍由 RTP 适配器执行，状态决策及转移归 Scheduler。

RtpEngineLoop 负责唤醒、执行和结果交接，不判断后端队列和协议阶段。新输入、Compute/Transfer 完成、输出恢复可写、取消、资源释放和关闭触发唤醒；同时按最近请求 deadline 定时唤醒。无事件、无期限时阻塞等待，不固定周期轮询。Forward 在途时仍处理状态维护和 I/O 事件，但不提交第二个 Forward。资源可用变化必须有明确通知，不能在移除轮询后遗漏容量唤醒。

状态转换与任务快照复制使用同一调度锁。查询可读到转换前或转换后的已提交状态，不能读到队列移动一半的状态；不等完整调度轮、Forward 或 PD 完成，不维护影子状态。活动记录移除与完成增量发布在同次状态提交完成，网络通知在锁外执行。Cache 单独采样一次，再由既有 GetWorkerStatus 编码组装；不承诺任务与 Cache 来自同一全局时刻。这与 real RTP 分开读取运行记录、Stream 阶段和 Cache 的边界一致，不复制其类之间的分工。

## 设计边界

本节记录已确定的职责边界，包含尚待实现的部分。当前已迁移 Python Case 接口、Step/ref 和 Prometheus 查询，并拆出协议无关的 RtpEngineLoop、请求进度/终态和 RTP 调度域；本批接通 Engine 宿主、实例扩缩容和发现更新；Client 已接通标准压缩输入播放与 client0 协调，其他后端仍待实现。其后的“当前代码职责”及运行说明描述可执行范围。

### Python Case 操作与 Java 实例能力

Python 提供面向 Case 的业务接口，Java Engine 提供实例原子能力。Client 的多 JVM 发压由 Java client0 协调，Python 转交活动级命令，不逐个调度请求。

| 层次 | 职责 |
| --- | --- |
| Python Runner | 顺序执行 step、解析引用、固定时间范围、预算和失败处理、记录结果。 |
| Python Case 接口实现 | 选择 Engine、扩缩容、向 client0 提交流量任务、跨实例查询与聚合；使用进程管理和发现文件能力。 |
| Java 实例协议 | 停止实例、查询自身状态、接收已分配的流量计划等。Engine 宿主另负责创建和关闭本 JVM 内的 Engine 对象。 |
| Java client0 协调者 | 固定参与者、分配总流量与输入、协调准备/启动/停止、汇总活动状态；与本地请求执行分开。 |
| Java 公共 Mock Engine | 请求接入、调度循环、资源机制、Forward 执行、输出和指标；组合 RTP 或 vLLM Scheduler。 |

缩容一个 Decode 时，Python 从全局候选集合中选择并固定一个地址，再通知其宿主关闭该实例，更新发现文件并返回实际结果。Java 不接收全局 `count: 1`；不能对每个 JVM 广播一次缩容，否则会缩掉多个实例。Client 总 QPS、请求数和并发预算由 client0 只分配一次，各 Client 收到自己的份额，不能广播完整总量。

Case 操作放在独立的 Python 实现中，不在 Runner 主循环按接口名堆分支。简单实例调用复用公共传输；Engine 扩缩容、client0 流量协调、Python 查询聚合保留各自的业务逻辑，不用一个通用广播器代替。

参数引用只表达取值：前置 step 的返回结果，或已定义的全局配置/发现信息。不增加专用的 leader/group/instance 选择器语言；查询主 Master 后直接引用查询结果。目标省略时默认对应操作域的全部实例，显式空集合或失效地址报错，不能退回全部，也不能自动缩成当前可达集合。需要唯一目标的操作校验数量。引用统一写成 `{ref: steps.<id>}`、`{ref: steps.<id>.<field>}` 或 `{ref: globals.<field>}`。当前全局字段为 masters/prefill/decode/vit 的服务地址列表，目标参数统一为 `targets`。不新增 YAML 流程宏。

### Python 接口定义和 Step 模型

Case 接口由 Python 函数实现，装饰器登记稳定接口名、读写属性和必要单位，参数与返回类型来自函数签名。参数范围和跨字段校验放在参数类型中，不在装饰器另抄一份字段表。Registry 只负责登记、查找和定义检查；不要求每个函数继承一个接口基类，也不做动态插件发现。运行状态由本次 Case 的上下文持有，不能存入全局注册函数。

配置编译按 Case → Phase → Step 推进，环境编译分别处理部署参数、组件配置文件和 Master 原生 JSON。普通值与 ref 共用 validation 的 schema 遍历；编译器只为 ref 叶子提供类型检查和转换，不再复制对象/列表/union 的校验规则。值解析、schema 校验和类型兼容检查统一在 config/validation.py；不再保留 values.py。编译器通过 Interface.arguments 检查无引用参数，dataclass 构造与结果转换作为 registry 内部实现，不直接从编译器调用。Runner 在 ref 展开后只校验并构造一次参数，再调用 Interface.call；Interface.invoke 供独立调用时从原始参数完成同样的准备。参数失败发生在调用前，结果校验失败发生在调用后，保留各自的错误阶段和写操作 outcome_unknown 语义。

Step 统一为一次接口调用，包含 id、describe、interface、args、within 和可选 expect。删除 YAML 的 call/check 外壳及两套重复 Step 类：无 expect 时执行并保存结果，有 expect 时在同一次返回后判断。读取也可以只取值供后续引用；写操作的回执可以断言，但不因断言失败重放。接口读写属性与是否断言相互独立，不把普通检查隐式变成轮询。

执行链统一为解析引用、校验实参、固定查询时刻和预算、调用 Python 函数、校验返回值、保存结果、执行可选断言。编译阶段检查引用必须来自前置 step 或允许的全局信息，检查字段路径与类型；运行时解析后再次校验。Python 类型注解本身不等于运行时校验。

普通数值查询直接返回数值，单位声明在接口上；地址集合返回列表，动作返回明确回执，逐 Master 状态保留映射。断言按结构和运算符判断，不按 MetricValue、PerMasterEngines 等业务类型名称偷偷聚合。接口文档和清单可以由 Python 定义导出，用于保存实验现场，不再依赖 Java 构建才能编译 Case。

Python async 只表达非阻塞等待，不代表任意 step 可以脱离 Runner 后台执行。client/start 完成准备和开始命令后返回，Java 自行播放；client/stop 停止并收尾。Client 不引入 task ID、任务 ref 或通用后台任务框架。

本轮重构按职责变化整体重写 Registry、Step 模型、编译和执行主链，联动 CLI、启动配方、夹具和构建入口；复用严格 YAML 解析、进程回收、采集与记录等仍成立的机制。Java 内部端点保留明确的请求/响应协议，但不再声明整套 Case API；配置类型留在所属组件，不能随旧接口目录误删。保留引用重校验、固定时间窗口、错误结果和取消清理的验收，不保留另一套旧 Case 执行入口作为兼容分支。

### KV 容量的明确简化边界

每个 Engine 使用一套统一的逻辑 KV block 容量计量。不同模型网络层的 KV 成本合并在这个预算中；不实现按模型层分池、Full/SWA/Linear 多资源组、各组独立配额及它们之间的联合分配。这是有意的 Mock 模型边界，不是后续必须补齐的 real 能力。Device/Host 仅表示已有的缓存存放介质，不扩展成每层或每种注意力各自的资源池。涉及真实混合池瓶颈和逐层容量的实验不由此模型给出结论。

Scheduler 决定申请、增长和提交时机；KvCache 管匹配、分配、索引及引用。正常结束且 Compute/PD 使用结束后，P 提交已计算 prompt；D 提交 prompt 加实际已计算的输出前缀，排除最后一个仅采样、尚未计算 KV 的 token。共享 CacheSequence 保存语义输入和 cache-only 多模态区间，并统一生成完整 block key；生成 token 值由接入层从原生 GENERATE 传入，不能猜测为 prompt 最后一个 token。失败或取消不发布新缓存。

当前 Host↔Device 保持瞬时逻辑迁移。对齐的 feature 源码中，Host→Device 加载使用异步任务和完成上下文，Device→Host 转存也由后台任务执行；调用方在源/目标资源尚未就绪时保留引用。因而瞬时模型不覆盖搬运延迟、传输重叠和在途额外占用，不能表述成 real 使用同步瞬时迁移。

### 公共 Engine 与独立调度域

RTP 和 vLLM 的设计目标是复用请求事实、资源机制、RtpEngineLoop、Forward、输出和生命周期工具；各自的 Scheduler 拥有队列模型、准入、组批、抢占恢复和结果后的状态转换。替换调度策略不应修改 Compute 或 RtpEngineLoop。原生协议和计算模型参数仍在各自接入/配置边界适配，不宣称两个真实后端只有调度差异。

当前已打通 Java 装配入口 `RtpNode.start(config, Scheduler.Factory)`，指定监听端口时使用 `start(config, rpcPort, controlPort, factory)`。工厂的 `create(lock, cache, finished)` 为每个实例创建一个新 Scheduler，收到协议接入层与 RtpEngineLoop 共用的调度锁，以及本实例独占的 RTP Full-KV 管理器 KvCache 和统一的 finished 增量记录。工厂不关闭缓存；节点在关闭或调度器创建失败时回收它。默认入口只在 RtpNode 选择 RTP 工厂，PrefillRpcServer/DecodeRpcServer 不再创建或引用 RtpScheduler，公共执行层不识别 RTP 配置。

P 接入层保留原生报文、模型上下文及生成参数校验，单批 token 预算等调度策略由 Scheduler 判定。RTP 默认策略仍拒绝超出单批预算的 Prefill，并返回原有 INVALID_ARGUMENT/605；替换 Scheduler 不会被接入层的重复策略校验拦截。请求数量保护上限统一来自 RtpConfig，不再由接入层读取具体调度器的常量。

本批仅完成装配与公共组件边界；vLLM Scheduler 尚未实现，Python 后端选择和对应 Case 接线留到后续。没有自动插件扫描、空后端占位或静默回退到 RTP。

```text
网络/控制线程 → input / control / transferCompleted / finish
                           ↓
RtpEngineLoop：维护到期与依赖 → Scheduler.schedule → Compute.forward
                           ↑                         ↓
                      下一轮调度 ← Scheduler.computeCompleted
                                      ↓
                         锁外发送输出/就绪/终态通知
```

每个 Engine 同时至多一个 Forward。初次和空闲后先调度；每轮计算结束后，结果交回 Scheduler 提交进度、检查有效性并决定继续或结束。公共 RequestState 保证终态与资源清理只发生一次，但不定义后端阶段或替调度器决定队列迁移。Scheduler 不持有执行器、计时线程或 RtpEngineLoop，也不发送网络响应。

| 边界 | 职责 |
| --- | --- |
| `input(Input, Listener)` | 校验身份、接纳并登记请求，返回只读请求句柄。外部不能取得可变队列或修改请求进度。 |
| `schedule()` | 检查预算与资源，调整内部队列，返回本轮 Work/ForwardBatch；没有可执行工作时返回空。 |
| `computeCompleted(work, outcome)` / `failed(work, error)` | 提交本轮有效结果，决定后续队列和阶段完成，清理本轮资源引用。 |
| `transferCompleted(request, result)` / `control(request, event)` | 分别接收传输完成结果和后端协议许可，不把传输成功当作自动获准生成。 |
| `finish` / `beginTransfer` | 终止请求，或为已经就绪的数据取得短期读取引用；读取结束不等于收到远端传输确认。 |
| `tasks()` / `queues()` | 同一调度锁内读取已提交任务的观测事实及真实队列计数；不推进状态，不返回可变请求。 |

ForwardBatch 是本轮执行输入，不是 running 容器本身。它固定请求成员、本轮 token 数和执行身份；请求的持续状态由 Scheduler 保留。P 通常只执行一轮，D 在 running 中跨轮保留，不每轮回到 waiting。外部线程可在 Forward 期间经公共入口提交请求或依赖事实，但不能修改已经提交的 batch。

Scheduler 调用在同一短锁下串行化；锁不覆盖计算等待或网络发送。公共 RtpEngineLoop 使用一个虚拟线程等待事件和最近 deadline，入队、控制/执行完成、容量释放和关闭都会唤醒；没有固定周期轮询，容量版本未变化时不反复尝试初始分配。没有可执行 batch 也必须处理取消、期限和传输结果，不能依赖下一次 Forward 才收尾。

通知按请求顺序、锁外执行。每个请求最多一个尚未处理完的调度通知；RTP 输出通知只交给协议容器，不等待 Fetch 或网络可写，不暂停后续计算。计算接纳上限覆盖等待调度通知完成的请求；协议结果按独立保留上限清理。通知失败进入该请求的失败清理；终态通知等基础设施异常使 Engine 的后续状态读取明确报错。

批次取消按成员处理：取消成员不再提交结果，但资源引用保留到该 Forward 结束。只有全部成员终止或 Engine 关闭时才取消整个模拟执行。旧回调按执行对象和请求句柄核验，不能用复用的请求 ID 找到新请求继续推进。一次完成的 Forward 只记录一次耗时，计算量包含这轮实际执行的全部成员，输出量只计有效提交。

### 队列模型、观察与注入

队列模型属于 Scheduler，不另建强制共享的 QueueModel 层。一个实现可以有独立 loading 队列，另一个可以把 loading 作为 waiting 成员的状态；抢占也可能回到 waiting 重新计算，而不是进入 swap。外部只通过操作和快照契约交互。公共请求事实需要逐步覆盖输入、缓存命中、已完成 Prefill、生成进度和本轮工作量；尚未实现分块 Prefill，不能把重复执行当作分块支持。

当前 RTP 保留普通 FIFO waiting、显式 batch 队列、持续 running 和请求上的容量/输出/传输就绪状态。`concurrency_limit` 控制 P/D 每批请求数，默认 32。P 按完整输入长度和 `最长序列长度×序列数` 检查 `max_batch_tokens_size`，生成模式额外保留一个 token；合法 singleton 可超过该 batch token 预算。预算为 0 时 P 仍可执行 singleton，不表示无限组 batch。D 按请求数上限接纳，不复用 P 的 token/矩形限制。Compute 使用扣除命中后的实际计算长度。P/D 初始分配遇到临时容量不足等待容量变化；已进入生成的 D 增长 KV 失败则终止，对齐当前 real RTP，不实现 vLLM 抢占/重算策略。

`max_inited_kv_cache_streams` 为正时限制 P 已取得 KV 的请求数，包含仍持有数据等待 PD 的来源；0 不限制。普通 P 准入达到 `max_batch_tokens_without_cache` 后停止继续接纳，本轮最后一个成员可跨过阈值，不把它实现成严格总和上限。该配额不额外套到 real 的显式组路径。CP padding 和多序列宽度仍不展开模拟。

按请求计数，不能直接使用 group/batch 容器的 size：

- `waiting`：等待计算准入或初始 KV 容量的请求。
- `running`：已进入持续计算集合的请求，包含轮次之间等待输出交给协议容器的 D；不等于此刻 Forward 的成员数。
- `loading`：资源已预留，等待 KV 加载或生成许可的请求。
- `ready`：P 计算阶段已完成，仍持有传输数据的请求。

上述四类互斥，由 Scheduler 统计；终止但仍被执行引用的成员不计入四类，KV 占用继续保留。公共 QueueSnapshot 不再根据“不是 FORWARD 就是 waiting”推导。Mock 指标保留四类；原生状态接口只有 waiting/running，因此将 waiting、loading、ready 合并为原生 waiting，与任务列表的 is_waiting 一致，避免 Master 漏算已预留资源的请求。

Scheduler 的 `input(Input, Listener)` 提交普通 single；`input(List<InputItem>)` 一次提交一个完整显式 batch，每项返回独立的 Request 或失败，登记在同一 RtpEngineLoop 锁内完成后才唤醒调度。组来自调用边界，不按相同 groupId 拼接，也不等待缺失成员。每个成员仍各自管理 Listener、deadline、取消和 KV 租约。

P 默认保留 Master batch 边界。正常组只从本组选择实际 Forward 成员，不与其他组或普通 single 混合；组内按输入顺序扫描，暂时放不下的成员保留，后续较小成员仍可准入，剩余成员留在原组。成员失败、超时或取消不默认结束整组。有效组成员数超过 concurrency_limit，或超过已开启的 max_inited_kv_cache_streams，按 real 退回普通 single。普通工作与组在空 Forward 边界切换；组完全无法推进时允许普通请求推进，不因组屏障造成容量依赖死锁。无需等待一组在 D 全部生成完成才开始下一组 P Forward。

EnqueueBatch/EnqueueGroup 先并行准备各成员 D 资源，收齐准备结果后，将成功且未取消的成员一次提交到 P 的组入口；准备失败逐项返回错误，不回滚其他成功成员。ACK 表示正式入队，不保证已经完成 KV 准入或 Forward，后续执行失败通过 Fetch/输出返回。D ALLOCATE 清除 P 的组字段，继续按独立 Decode 请求调度。注入只能走正式提交与依赖入口，不能跳过资源和传输前置条件。本地搬运同步完成，不新增异步组装队列；暂停、换出和重算仍不实现。

### 请求观测与接入交接

对外只有一次 GetWorkerStatus。PrefillRpcServer 在同一执行锁内合并接入阶段与执行阶段：准备中的 Session 以 PENDING 上报；提交成功后持有 Scheduler.Request，由 Scheduler.tasks() 按需返回执行事实。Loop 不持有接入登记表，不过滤 Batch 成员，也不映射逐项结果。状态查询不驱动执行，Scheduler 不向 RtpEngineLoop 逐次回报阶段变化，也不需要使用共享状态机。RTP 将自己的内部状态映射为 RECEIVED/KV_ALLOCATED/RUNNING；其他后端负责自己的映射。

接入层在接受有效请求时固定 started 和绝对 deadline，单请求、多模态准备和显式组提交沿用同一预算。取消与提交使用同一执行锁，旧句柄不能删除或结束同身份的新请求。显式组中已取消的成员不再提交，其他合法成员继续准入。

TaskHistory 位于Engine 模块，每个 Engine 只有一份 finished 版本序列，由接入层创建，经 RtpEngineLoop 传给 Scheduler.Factory。提交前失败由接入 Session 发布，提交后由 Scheduler 在本地终态和资源清理条件满足时发布；不另外维护一份活动执行状态。错误保留 RequestFailure，到 RpcTasks 编码时才转换原生协议错误。准备中的任务不计入 Scheduler waiting/running，不占 KV；协议输出容器和 Fetch 保留结果同样不计入执行负载。P 交接完成后的本地 finished 与其继续代理 D 输出的会话结束是两个边界，后续网络结束不重复发布 P 的本地终态。

### PD 数据就绪与资源生命周期

PD 通过“数据就绪”和“传输完成”衔接。Scheduler 判断阶段是否完成、发布就绪通知并管理依赖；协议/传输组件负责 ALLOCATE、LOAD、GENERATE、网络读写和 ACK；公共引用机制保证资源不会提前释放。Compute 不在每轮完成后自动传输，分块 Prefill、加载命中和提前发起 LOAD 都不能依赖这一假设。

P 完成 Forward 后退出计算集合，保留 KV 和请求期限。交接期间额外持有资源引用；P 收到 D 的 LOAD ACK 后才提交传输完成并释放阶段资源。没有确认、传输失败或超时仍走失败清理，不发布成功缓存。释放 P KV 后，端到端 P RPC 仍可继续转发 D 输出。

D 只有 KV 加载成功和生成许可均满足后才进入 Decode；Prefill-only 完成走独立通知，不伪造 D Forward。加载或输出失败通过同一请求句柄返回调度域。所有通知在没有计算工作的情况下也能推进；过期请求的晚到通知不能复活请求。

不设公共 finished 调度队列。需要通知执行端清理状态的后端可维护完成 ID 集合；这不等于所有完成请求都进入一个额外流水线。

对齐参考：本仓 `NormalEngine::step`、`FIFOScheduler`、`GenerateStateMachine`、P/D RPC；vLLM 设计参考固定的 [v0.10.1 V1 SchedulerInterface](https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/v1/core/sched/interface.py) 和 [KVConnector](https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)。vLLM 实际对齐版本、KV connector 和能力范围尚未选定，不能把接口边界验证当作 vLLM 行为验收。首版不模拟多批异步流水或逐层 KV 传输。

### JVM 放置、配置和生命周期

业务操作只感知实例及其服务地址，不感知 JVM。物理放置先只支持两种 Engine 形态：

| 对象 | Engine 共享模式 | Engine 独立模式 |
| --- | --- | --- |
| Master | 每个 Master 独立 JVM | 每个 Master 独立 JVM |
| P/D/V Mock Engine | 全部放在同一个 Engine JVM | 每个 Engine 一个 JVM |
| Client | 每个 Client 独立 JVM | 每个 Client 独立 JVM |

本地 P/D/V 各实例直接启动一个 JVM。Case 通过服务地址引用实例。

实例使用服务地址区分，同 IP 下用端口区分。控制、业务 RPC 和 metrics 地址之间的对应关系留在发现信息中，不向 Case 暴露额外的 JVM 身份。实例之间统一通过 TCP 通信，共 JVM 不另走 in-process 传输。Python 内部仍须保存进程与实例归属，以处理宿主退出和清理。

进程 env、heap 和资源限制属于 JVM；每个 Engine 的业务参数来自自己的配置对象。共 JVM 时不能用修改进程 env 或全局 System properties 的方式切换实例配置。原生 env 风格字段可以保留其名字和值，由实例加载器从显式配置读取。

停止 Engine 会终止所属 JVM，释放实例监听器和缓存。其他实例继续运行。

Runner 登记预期停止，确认进程退出后撤销端点。意外退出使实验失败并回收其余进程。

### 发现文件和产物

实例在自己的目录写配置、端点和日志。ready 与 endpoints 原子发布。Python 独占拓扑与服务发现的写入。

Python 维护一份权威的当前发现信息，增加、移除和重启换地址均更新该信息。Master 等消费者能否动态加载发现变更由其自身实现负责，Case 可检查传播结果，框架不替消费者伪造路由更新。显式引用旧地址不能偷偷改指向新实例；一次运行内，Engine 的业务 RPC、控制/metrics 和宿主控制端口统一不复用，跨用途也不能复用，例如不能把旧 metrics 端口分给新 RPC。该约束只覆盖本次 Runner 管理的 Engine 和宿主，不是操作系统级端口预留。

### Client：播放器控速，Client 发送与管理生命周期

所有 Client 同时只有一份相同的 start 计划。Python 只操作 client0，Java 负责拆分计划和自行发压；Python 不轮询播放进度、不实时检查 QPS。先调用 client/stop 收集上一轮结果，再启动下一轮。

| 组件 | 职责 |
| --- | --- |
| Python | 启动和监视各 JVM，调用 start/stop，最后通过已有指标查询路径执行 QPS 等 Case 检查。 |
| client0 协调者 | 固定全部参与者，分配总请求数、QPS 和并发额度，协调准备/开始/停止，汇总执行异常与最终结果。 |
| 播放器 Player | 读取并生成请求，控制产出速度。只有播放器做速率控制。 |
| Client（LoadClient + NativeClient） | 管理小队列和在途并发，发送请求，消费完整响应，处理超时、取消和终态，产生实际调用指标。 |

client0 同时参与本地发压，不增加协调进程或选主。单 Client 与多 Client 使用同一流程，client0 到本地和其他 Client 的控制都走 TCP。准备完成后，各 JVM 独立产出和发送，请求与响应不经 client0 中转。client0 意外退出使整体失败，不做接管。

#### 产出、队列与并发

```text
播放器按速率产出 → 非阻塞 offer → 有界小队列 → Client 有并发容量时取出 → 原生调用
```

Client 取快了就在队列为空时等待；取慢了允许小队列暂时积压。队列满时报告执行异常并停止整组，不让播放器阻塞降速，不丢弃后继续，也不自动扩成无限队列。Client 不再把取出的请求移到第二条待执行队列，并发槽位覆盖调用至终态。

请求不携带 planned_at，不增加逐请求发送偏差验收。播放器内部使用单调时钟控制节奏；Client 只从实际调用开始计 TTFT 和请求耗时。开始调用不等于 Master 已接收，客户端终态也不证明远端 KV 已释放。

队列满说明当前配置无法继续执行计划，不能单凭它归责服务端或发压端。播放器慢但队列未满时，留给最终 QPS check 判断。队列会改变短时间发送节奏，本版不承诺严格复现 trace 到达时间。

总流量只分配一次。均匀播放的全局序号 k 对应相对时间 k/Q，再按 k % Client 数分片，各 Client 不从同一个相位重复完整速率。总并发均分，余数按固定参与者顺序分配；不运行时借用配额或重分片。首版每个 Client 至少分到一个请求和一个并发槽位，否则拒绝启动。

#### 只暴露 start/stop

- `client/start`：接收全局计划，所有 Client 准备后下发同一个未来开始时间，成功返回 true。返回只代表开始命令已被接收，不代表播放成功。启动中或尚未收集上一轮结果时再次 start 报错。
- `client/stop`：参数为空，停止全部播放器及新增调用，等待有界收尾，返回按 Client 服务地址保存的结果。已经自然结束则直接收集原结果；重复 stop 不重放或重新开始。

没有独立的 status/wait/drain Case 接口，没有任务选择参数。Java 内部保留轮次序号以拒绝旧控制命令，Case 不需要知道它。当前 stop 固定整组；部分 Client 停止尚未接通，不能暗中把指定目标扩成全部。

自然结束时，Client 发送完队列剩余项并等待在途请求；主动或异常停止时，未发出的请求不再发送，计入 unsent。单请求业务错误留给 Case 指标门禁，不默认中断整组。队列满、准备失败、参与者失联、收尾超时和产物写失败属于执行异常，client0 停止其余参与者并保留首个原因；故障传播并非多个 JVM 同时发生。

Python 在 stop 或最终清理时收集异常，不增加实时发压监控。开始响应丢失也必须尝试最终 stop，不能据此重发 start。Runner 原有进程退出监视保留。收尾先停 Client、获取结果，再做 Prometheus 最后采样和进程清理；不能遗漏 stop 后仍给出虚假的成功。收尾超时取消剩余调用，记录执行错误，不补造远端释放事实。

#### 标准压缩输入与分片

`client/start` 只接受 `requests_file`，路径指向标准 Protobuf＋Zstd 文件；不接受 inline requests、JSONL、xz 或格式兼容开关。相对路径基于 Case YAML 所在目录，也接受绝对路径。当前所有 Client 位于同一机器并共享文件系统，不支持 URL 或跨机器分发。

压缩回放定义为 `java/client/src/main/proto/client_input.proto`，Java 由 Maven 生成，Python 测试辅助目录 tests/helpers 提交同一 protoc 3.25.1 生成的 client_input_pb2.py；输入读写器 input.py 供框架测试与镜像终验生成、检查压缩数据，不发布到 mock_framework 运行包。Request 保存未展开的回放输入：非空、非负 int32 token_ids，正 int32 max_new_tokens（本格式中的固定目标输出长度，LoadClient 发送 ignore_eos=true），可选非负 int32 priority（缺省为 50，显式 0 保留），可选非负 int64 arrival_offset_ms（相对于源起点的毫秒偏移，尚未除以倍速）。可选 images 保存 url、width、height、offset；每张图只占 token_ids 中的一个位置，offset 按升序排列且不得重复。LoadClient 按 Client 显式配置的 image_token_id 将这些位置写为图片标记；Mock V 默认异步下载真实图片，使用 Java 读取尺寸并按文件内容摘要生成模拟特征身份；本地 mock.vit.skip.download=true 时使用回放宽高，其他准入、排队和计算路径相同。Client 查询 V 元数据后计算路由长度和缓存 key，发送给后端的仍是原始媒体输入。P 在进入 Scheduler 前展开图片位置，保留语义图片标记，将逐行特征 hash 单独保存在缓存区间中；hash 不替换语义 token。完整生成参数及真实观测仍不是此播放格式的内容。未知字段、非法请求及不支持的版本均拒绝。

文件以 `MCKINP01` 标识输入格式 v1，之后每批为 4 字节大端压缩长度加一个 Zstd 压缩的 Protobuf InputChunk。帧必须为 single-segment、声明解压长度并带校验和，压缩及解压单批上限均为 4 MiB；每请求 Protobuf 上限 1 MiB，每批至多 1024 个请求。写入器按约 1 MiB 或 1024 请求封批，不累积整个文件。InputChunk.sequence 从 1 连续递增，末批只含 total_requests；缺末批、计数不符、截断或尾随数据均报错。InputChunk 仅用于文件分块，不决定请求发送或 Engine 计算的 batch。输入和分片使用同一份格式与读写器，输入消息与执行证据消息分开定义。

client0 在 start 的步骤预算内流式校验整个输入，按全局序号 `k % Client 数` 写本轮压缩分片；校验、封口、关闭和发布全部完成后才 prepare/begin。分片位于 `components/client-0/coordinator/<轮次>/shards/`，直接写压缩 `.part`，成功后原子发布为 `.binpb.zst`。input.json 只记录条数和摘要等元数据，不保存可读请求副本。准备失败不启动播放；源文件在准备期间发生可检测的大小、修改时间或文件身份变化会报错。

每份 worker 计划只包含分片路径、条数、全局序号起点/步长、请求 ID 起点、SHA-256 和发流参数。Player 使用同一 InputFile.Reader 按批次解压，交给原有小队列；不把完整输入放进堆，也不把请求/响应经 client0 转发。请求 ID 在多轮 start 之间单调分配，主动停止留下的未发请求不会重新分配给后续轮次。播放读失败、条数或完整读取后的 SHA-256 不符使整组失败；主动 stop 不扫描未消费的后缀，剩余请求计入 unsent。

masters 可省略，Client 默认持续发现本实验的全部 Master；显式填写时固定为该 step 的地址集合，播放期间只在其中选主备。qps 和 replay_speed 必须且只能提供一个正有限值。qps 模式按全局请求序号控制均匀发送；replay_speed 模式要求每条请求有 arrival_offset_ms，准备时检查时间非递减，Player 按 offset/replay_speed 发送。所有分片保留同一个源起点，不按各自首条请求重新归零；相同时间允许突发，不再叠加 QPS 限速。concurrency 是全局预算。queue_capacity 默认每 Client 16，允许 1–100000；request_timeout 和 drain_timeout 默认 30s，允许 1ms–1h 的整毫秒值。计划时长最多 24h；每个 Client 至少分到一个请求和并发槽。准备时间计入 start 的 within/phase timeout，大文件需明确足够的预算。

参与发压的 Client 在本轮固定；单请求 timeout 从 NativeClient 调用开始，不包含小队列等待。只支持本机多 JVM，未承诺跨机器时钟同步。准备后 30s 未收到 begin 的 Client 报错退出本轮。请求循环、多轮对话、ramp、动态参与者和运行时旧格式转换不在当前范围。

```yaml
- id: load
  describe: 全体 Client 合计 24 QPS
  interface: client/start
  within: 60s
  args:
    requests_file: traffic.binpb.zst
    qps: 24
    concurrency: 16
  wait_after: 10s
- id: stop_load
  describe: 收集全部 Client 的执行结果
  interface: client/stop
```

测试代码通过 tests/helpers/input.py 构造 Protobuf Request 并写压缩文件（下面的导入用于 pytest 或已将 tests 加入 sys.path 的镜像终验脚本）：

```python
from helpers.input import pb, write_inputs
write_inputs("traffic.binpb.zst", (
    pb.Request(token_ids=[1, 2, i + 3], max_new_tokens=3)
    for i in range(20)
))
```

测试写入器拒绝覆盖已有文件，生成失败删除本次不完整文件。正常运行 Case 只传入标准二进制文件路径，由 Java InputFile/RequestFiles 校验、读取和分片，不导入 Python 输入辅助代码。基本 Case 直接携带标准二进制输入。控制接口仍传小型 HTTP 元数据，不在 JSON 控制报文中传请求数据。

`RequestFilesTest` 在独立 48 MiB 堆的 JVM 中分片并完整播放解压后超过 64 MiB 的输入，覆盖非法输入、截断、校验和、内容变化和准备取消。跨语言测试验证 Python 写→Java 读、Java 写→Python 读；真实 Master 接线测试在 1/2 个 Client JVM 中播放并校验分片总量、全局请求 ID 和发压窗口内的合计 QPS。这些是有界读取和功能接线验证，不测量最大可达 QPS。

`client/qps` 复用公共 Python 查询，统计 mock_client_requests_issued_total，是发出 QPS；最终检查选择实际发压窗口，不能用含空闲和收尾的实验全时长稀释目标。latest 在自然结束后可能为零，all 尚不支持；缺实例或采样不足按公共查询规则报错。各 Client 仅上报自己的指标，client0 不重复上报全局汇总，不再增加一套发压有效性门禁。

每个 Client 在自己的 playback/<轮次>/ 保存计划与结果，client0 另在 coordinator/<轮次>/ 保存总计划、参与者和汇总。文件单写并原子替换，多 JVM 和后续轮次不会覆盖旧结果。每轮输入分片与逐请求执行证据分别保存，输入不作为已发送/已完成的证明；磁盘须容纳输入和各轮分片。Client 结果包含 produced/issued/finished/unsent、在途/队列余量、终态计数和异常，业务错误计数不等于执行异常。

启动配方中的 Client 使用 java -jar <framework>/java/client/target/mock-client-1.0-SNAPSHOT-all.jar，由 Runner 注入配置、拓扑和独立输出目录。部分停止与队列/并发默认值的压力校准仍待讨论。

### Client 逐请求记录与查询

`NativeClient` 在实际执行边界记录 ISSUED、SCHEDULE_START、SCHEDULE_RESULT、FIRST_TOKEN 和 TERMINAL。同一个逻辑请求的主备尝试共用 request_id，issued 只计一次。记录包含墙钟毫秒、单调耗时、目标 Master、尝试序号、gRPC 状态与业务错误码、首 token 耗时、实际输出 token 数及终态。不记录输入 token 正文或完整响应，错误说明最多 4096 字符，截断会标记。Schedule 响应分别保留传输状态、原始业务 code（成功通常为 200）和可空 response_success；业务拒绝不会伪装成 gRPC 传输失败。Schedule 超时/取消的结果只表示 Client 观察，不证明远端未接收；Cancel 仍是尽力而为。

每个 Client 每轮独占 `components/<client>/playback/<轮次>/requests.binpb.zst`，目录与记录文件拒绝覆盖。公共字段定义是 `java/client/src/main/proto/client_evidence.proto`，Java 在 Maven 构建时生成，Python 在 `python/mock_framework/interfaces/client/` 提交同一 protoc 3.25.1 生成的 `client_evidence_pb2.py`；修改 proto 时必须重新生成两侧，不能分别手写协议。

每个文件由连续帧组成：4 字节大端压缩长度，再接一个带校验和的独立 Zstd frame；解压后为 varint 长度和一个 Protobuf EventBatch。每帧携带版本、Client、playback、帧序号与分片身份，每条事件有连续序号。最后一帧携带 Seal 的 issued/finished 计数。没有额外 JSONL 执行记录或永久逐请求索引，`client/requests` 只在显式查询时解码。

网络回调只向有字节上限的队列追加事件，单独写线程按约 1 MiB 或 1 秒成帧并 force 到磁盘。当前队列上限 8 MiB、Zstd level 3，是初始参数，不是最大吞吐承诺。入队、写入、持久化不是同一个状态；Client 只有等待最终 Seal 写入并 force 成功才完成。队列满、超大事件、写入失败或收尾超时使 worker 报错，client0 沿用现有监督机制停止整轮。进程意外退出可能留下未完成请求；不能补造终态，也不能把无 Seal 的文件判完整。

`client/stop` 及 Runner 收尾使用 Java 返回的最终汇总，检查参与 Client、完成状态、队列、inflight、issued/finished、outcomes 和 errors，不再重复扫描请求文件。Java 在最终 Seal 写入后才发布完成状态，写盘/封口错误通过 errors 返回。只有显式调用 `client/requests` 时才流式读取文件并检查完整性；过滤和返回条数限制不会跳过其余记录的验证。事后文件缺失或损坏由该查询报错，不追溯改写已保存的 stop 结果。

`client/requests` 是 Python Case 读取接口，不经过 Java 服务。默认查最新一轮、全部 Client；`targets` 使用现有 Client RPC 地址，`playback` 可选择历史轮次。`request_ids`、`outcome`、`since_ms`、`until_ms` 可筛选，其中时间按请求发送墙钟的 `[since_ms, until_ms)` 计算。返回 matched、筛选后 outcomes、records、truncated；limit 默认 20、范围 0–1000，只限制明细条数。明细按 Client 选择顺序和各自完成顺序返回，不承诺全局时间排序。运行中尚未封口的轮次报错；先 stop/收集再查询。

```yaml
- id: collected
  describe: 停止并核对本轮全部请求
  interface: client/stop
- id: failed_requests
  describe: 查看本轮失败请求
  interface: client/requests
  args: {outcome: error, limit: 20}
```

实现集中在 `python/mock_framework/interfaces/client/`：`__init__.py` 定义现有 start/stop/requests 接口、参数和播放目录选择；`_records.py` 只包含返回值结构、流式解码和完整性校验；`client_evidence_pb2.py` 为生成消息。镜像终验直接复用读取函数。不提供独立请求查询 CLI，也不增加 Java 查询服务、parser 类或存储抽象。

### Python 指标查询与公共指标

Java 产生指标并暴露 `/metrics`；Python 直接调用 Prometheus HTTP API，不经过 Engine 查询代理，也不新增 Java 查询服务。原始 PromQL 可在 Prometheus 存活期间查询其已采集的数据，框架不维护实例历史。

公共 Engine 使用同一组指标含义和采集事件，Python 默认只实现一套公共语义查询。公共客户端处理 HTTP、超时和响应解析，指标函数处理 PromQL、单位、目标范围和结果形态。确有后端原生指标差异时，映射和公式放在 RTP/vLLM 对应 Python 模块中，Runner 直接调用；不为相同指标复制两份实现。原始 `prometheus/query` 走公共查询客户端，不进行后端改写或自动附加过滤。

指标按实例上报、按查询语义聚合：总 QPS 求和；成功率和命中率分别汇总分子/分母后求比率；平均值按样本数加权；p99 不能平均各实例 p99，须合并兼容直方图或使用完整请求记录。时间窗口、目标覆盖和采样完整性要校验，缺数据不补零。语义查询只使用当前活动实例；显式 targets 也必须仍然活动，不根据时间窗重建过去的成员集合。需要查询已下线地址时，直接使用原始 PromQL。不同 Master 的状态保留逐实例结果，不取并集掩盖差异。

去掉业务指标上的 RTP/vLLM `engine`、逻辑 `node`、JVM 和主动添加的实验 ID 标签，保留必要的角色与实例地址区分。独立采集端点可使用 Prometheus `instance`；共享 `/metrics` 时必须在样本上区分实例服务地址，不能让同名同标签序列碰撞或重复采集。`host_ip`/`container_ip` 保持物理含义，共 JVM 时不能仅靠相同 IP 区分 Engine。实验默认独占 Prometheus，跨实验合并分析时再补来源标签，不能在合并后无法区分数据来源。

一个真实完成的 Forward batch 记录一次耗时，不能按 batch 中请求数重复记录。请求终态按请求计数，token 按本轮实际计算和提交分别统计。新增两项 Forward 指标，不增加对应服务接口：

- `rtp_llm_model_forward_us`：Prefill 完成 Forward 的窗口均值，单位微秒。目标生产 MTP Decode 版本缺少该时序。P/D 的最近一轮模拟实际耗时使用 `mock_model_forward_actual_us`。
- `mock_model_forward_planned_us`：公式和可选噪声确定的本轮计划耗时，单位微秒。

实际计时从提交本轮计算到公共执行层收到完成通知，包含计时器延迟和回调调度等待，在处理结果及提交 token/KV 前结束。计划和实际对同一完成 batch 各记一次，取消批次不补完成样本。差值可以揭示宿主调度/GC 等干扰，不能单独归因为 CPU 不足；Client TTFT 仍取真实接收时间。本仓 real `NormalExecutor.cc` 围绕 model_->forward 计时，`RtpLLMMetrics.cc` 将该字段注册并报告为 Gauge；本轮因此采用同名微秒 Gauge，计划耗时也用 Gauge。Gauge 保留最近一批，不能拿采样点平均值冒充逐批平均；吞吐仍使用累计 token 与耗时 Histogram 的 sum/count。KMonitor 与 Prometheus 的采样、聚合管线不据此宣称完全相同。

### 待定边界与验收

强制 group、分块 Prefill、抢占/重算、vLLM 对齐版本和 connector 范围仍待确定。当前 RTP 的有界合批只覆盖现有调度契约。

验收覆盖缩容和扩容后的真实 TCP 请求、取消后的 KV 释放、Forward 指标及异常退出回收。

## 当前代码职责

Python 接口与 Java 内部协议已分开；Engine 已接通共享/独立宿主和实例扩缩容；Master 与 Client 保持逐实例进程。

| 目录 | 负责内容 |
| --- | --- |
| `python/mock_framework/config/` | YAML 解析、检查、点分配置展开、引用和可执行步骤编译。 |
| `python/mock_framework/execution/` | 运行环境装配、统一 Step 执行、角色生命周期、拓扑、采集、HTTP 传输、断言与记录；各模块边界见下表。 |
| `python/mock_framework/interfaces/` | 已接通的 Case 函数：逐 Master 状态、Client start/stop、公共指标与原始 PromQL。 |
| `java/protocol/` | 控制 HTTP、媒体 RPC 和跨组件 Cache Key 编码。 |
| `java/engine/` | P/D/V 单实例入口、RTP 接入、调度、Forward、KV 和 ViT。 |
| `java/metrics/` | 指标标签、窗口统计和 Prometheus / KMonitor 导出。 |
| `java/client/` | 分片、回放、原生请求和逐请求证据。 |
| `java/master/` | 本地真实 Master 启动和状态查询。 |
| `java/zookeeper/` | 独立 ZooKeeper JVM。 |

Python 运行时按以下边界组织。`RunEnvironment` 装配实例并安排启动和收尾顺序，不分发 Case 业务操作；各服务显式接收所需依赖，不接收整个运行环境再修改它的内部状态。

| 模块 | 唯一负责的状态与行为 |
| --- | --- |
| `execution/runner.py` | Step 的引用、参数、预算、调用、返回校验、断言、等待与结果；不判断具体接口名。 |
| `execution/environment.py` | `RunEnvironment` 的静态装配、启动与收尾顺序。 |
| `execution/processes.py` | `ProcessSupervisor` 的子进程所有权、退出监视、操作中断和回收；不识别 Master、Engine 或 Client 业务。 |
| `execution/topology.py` | `Topology` 的当前实例端点及文件发布；读者拿到独立快照。 |
| `execution/components.py` | Master/Client 通用的配置文件、进程启动、ready 检查与端点登记；原生配置与端点额外校验由调用方提供。 |
| `execution/master.py` | `MasterGroup` 的 ZK 选举环境、独立 JVM 增减、原生配置和 Master discovery 发布。 |
| `execution/engines.py` | `EngineProcesses` 的实例分配、进程启停、端点回读和扩缩容。 |
| `execution/client_group.py` | `ClientGroup` 的 Client JVM、当前播放待收尾状态、client0 控制与最终汇总校验；流量分片和控速仍由 Java 负责。 |
| `execution/monitoring.py` | `Prometheus` 的采集器、目标、基线、末次采样和采集完整性记录；不执行 Case 语义查询。 |
| `execution/launch.py` | 启动配方的结构校验与规范化，不持有运行状态。 |
| `interfaces/` | Case 参数/返回契约及操作组合；调用对应服务或传输/查询函数。运行时服务不反向导入这一层。 |

拓扑是 Python 根据已校验的实例端点生产的运行记录，不代表生产 Master 的成员状态或主备判断。`Topology` 统一登记与退役，发布当前 topology.json，并把快照交给 `MasterGroup` 更新活动 Master 的 discovery。Master、Client 等消费者自行加载配置；Python 不替它们推导主节点。发布失败使实验报错，保留已登记状态与进程所有权，随后清理；不宣称多个文件的发布是一个事务。

启动顺序固定为 Prometheus、ZK、Engine、Master、Client、采集基线。收尾先停止 Engine 健康检查，再收集 Client、完成最后采样，依次清理 Client、Master、Engine、ZK、Prometheus，最后由进程监督器回收剩余进程。某一步失败仍继续后续清理，错误进入最终结果；进程未回收时保留其所有权，不能仅删除内部记录。最终没有存活的受管进程后清空活动拓扑。

新增 Case 操作在 `interfaces/` 用现有装饰器注册；新增角色生命周期行为进入对应服务；只有新的跨服务启动/收尾顺序才修改 `RunEnvironment`。不为接口名、角色名或 Case 名向 Runner 主循环增加分支，也不引入通用插件框架或事件总线。配置入口和业务 Case 的下一批接线以这些边界为基础，本批保持现有 YAML 参数、返回值与产物格式。

这里的 RtpEngineLoop 是 P/D 文本生成执行内核，不是 JVM 宿主循环，也不是 ViT 调度循环。runtime 仅提供访问和实例管理设施；RtpEngineLoop 驱动 Scheduler 选批及 Forward 执行，Scheduler 提交生成进度、队列和资源变化。ViT 保留自己的图片合批循环。RequestState 负责进度、终态及资源使用计数；QueueSnapshot 只是队列计数值，不统一各调度器的队列容器。

multimodal 供 Client、P 接入层和 ViT 共用媒体标识、输入转换及 TCP 调用；图片几何、模拟输出 hash 生成和缓存大小计算归 ViT 的 VitGeometry。它们不反向依赖 ViT 服务实现。shared/metrics 只保留 MetricTags 和 WindowMetrics；RTP 指标名称、单位及 TPS/容量统计放在后端 RtpMetrics，ViT 仍使用 VitMetrics。指标记录不参与模拟计算；窗口统计在 scrape 时推进，因此同一目标应由一个采集者抓取，额外 scrape 会改变窗口划分。

ViT 的终态分三步：VitScheduler 只完成已准备图片的计算阶段 Future，不依赖 VitCache；VitPreparation.Task 等待阶段结果，统一调用 VitCache.publish 发布成功/失败，随后移除任务、释放准入额度。VitCache 保留同 key 的共享结果条目、等待者和驻留淘汰，不再从 pending 数量推导执行额度。mm_inflight_inputs 来源于 Preparation 的在途任务数，包含发布尚未返回的任务。排队取消通过阶段 Future 返回给任务所有者；已经选入 Forward 的工作保留到实际完成。关闭同样由 Preparation 收口任务终态，Cache.close 仅关闭缓存驻留。


KvTransfer 是接入层使用的零耗时异步 PD 传输执行器，放在 admission 包内，不作为公共调度接口。传输开始与完成仍通过 RtpEngineLoop 交给 Scheduler 处理；移动执行器不改变资源使用权和 LOAD ACK 顺序。KvCache/BlockTree 目前固定为 RTP Full-KV，保留已有模块，不声明仅替换 Scheduler 就能兼容任意后端，也不增加缓存抽象或新模块。

共享指代码复用，不是不同实例共享资源。RtpRpcService 只路由协议入口和编码查询结果；PrefillRpcServer 统一管理 P 的本地单 token、P/D 和显式组接入，DecodeRpcServer 管理 D 的协议会话。两者保留原生校验、输入准备、协议顺序和响应投递，通过 Scheduler.Request 句柄与 RtpEngineLoop 交互。RpcTasks 转换完整输入和观测快照，RpcErrors 转换原生错误码；KV、队列、计算和本地完成条件由后端 Scheduler 管理。Control/TransferResult 是现有原生接入与 RTP 后端的协议约定；共享 Scheduler 只提供扩展点，不定义所有后端必须经历的状态推进序列。RtpScheduler 负责普通 FIFO 准入，RtpGroupScheduler 单独负责显式组调度；两者共用基类，无 gRPC/protobuf/执行器依赖。

RequestState 是共享的请求进度、终态和资源清理机制。调度器决定何时申请和成功发布缓存；公共机制确保终态只通知一次、发布失败也尝试释放、取消请求不能再提交进度。Forward 持有资源使用权，终止请求不会提前释放仍被使用的块；RtpEngineLoop 对执行对象做身份校验，晚到回调不能影响下一次执行。当前模拟 Forward 使用可取消计时；单成员取消保留整批执行，全部成员终止或关闭 Engine 时才取消计时并释放使用权。

P 的 retained 请求计算完成后变成 READY，释放 Forward 槽位但保留 KV，直到 LOAD ACK、取消或期限到达；读取期间另持资源引用。D 的接入/传输与计算进度分开保存，完成后的 token/迭代数只由调度域推进。请求终态先清理资源；Scheduler 在同次状态提交中记录指标、发布原生完成增量并移除活动请求。随后由协议适配器异步完成 RPC 响应，不影响状态查询的可见性。

当前生产路径支持多个独立请求合批，P/D 使用相同调度循环。测试覆盖持续 Decode、批内独立取消、临时容量等待、输出阻塞和 PD 确认边界；完整 group、分块 Prefill 和 vLLM 未实现。

## 当前接口扩展方式

在 Python 所属接口模块定义 async 函数，参数使用 dataclass、返回值使用类型注解，添加 `@case_interface`，再加入 `interfaces.FUNCTIONS`。例如：

```python
@dataclass(frozen=True)
class QueueArgs:
    targets: list[str] | None = None

@case_interface("engine/queue/size", effect="read", unit="requests")
async def queue_size(ctx, args: QueueArgs) -> int:
    # 实现负责目标选择、读取和缺失目标校验；此处只是扩展示意。
    ...
```

装饰器生成参数/返回结构，不要求接口基类，也不允许只登记名称而没有实现。边界使用 Annotated/Bounds、duration/range 标记，跨字段规则写在参数 dataclass 的 __post_init__。新增接口不改编译器或 Runner 分发逻辑；非平凡行为必须有测试。运行产物 interfaces.json 保存本次 Python 注册定义，构建不再产生 Java Case catalog。

Java 内部端点仍可使用 ApiServer 和 Endpoint 的序列化/校验机制；它只约束实例请求与响应，不承担 Case 目标选择、扩缩容计数、指标聚合或 Python ref 语法。当前 MasterStateApi 使用这条内部协议；未接通的 Java Case 声明及 PrometheusQueries 已删除。

### Fetch 与计算解耦

RTP 的 Enqueue ACK 后持续推进 LOAD、Prefill 和 Decode，不等待 Fetch。Scheduler 仍统一管理
RECEIVED、KV_ALLOCATED、RUNNING、KV 资源和按版本查询的 finished 记录；P 在 LOAD ACK 后结束
本地阶段，D 在最终输出交给协议容器后结束计算阶段。Fetch 接入、网络可写和输出消费不参与这些状态转换。

协议层缓冲输出并异步发送，Fetch 断开只丢弃该连接的待发输出，不取消计算；显式 Cancel 和请求 deadline
仍终止计算。输出注册表不叠加固定请求数上限，未接入结果按 fetch_attach_timeout_ms（默认十分钟）
过期并回收已完成项，不另设会拒绝活跃请求的容量上限。正常性能 Case 应在 Prefill 完成前接入 Fetch；晚接入可读取
保留输出，但补发间隔不代表真实 TPOT。不模拟输出消费者背压。

公共 RtpEngineLoop 使用 Condition 等待事件，避免 Java 21 Object.wait 占用共享虚拟线程承载线程。

Master HA Case 在 environment 中声明两个独立 Master JVM，启动后通过 `master/get` 加 `is_leader: true` 记录唯一主节点 A，再以 `is_leader: false` 记录备用节点 B；后续按保存的地址执行故障与恢复，不依赖启动编号，也不通过动态新增节点获得备用身份。

原生 Master 的正常停止和最终清理使用每进程 60 秒退出预算，覆盖选举交接、gRPC 静默期和资源回收。最终清理按启动尝试的逆序逐个等待退出，期间保持其他 Master、Engine 和 ZooKeeper 存活；超时仍强杀并记录清理错误，不把退出失败当作成功。

接入层只处理原生 RPC 的解析、响应关联和异步输出；Scheduler 是队列、计算进度、容量和终态的唯一来源。D 的网络不可写只阻塞输出缓冲发送，不再阻塞 Decode Forward 或 KV 释放；缓冲仍受请求生成长度和整体输出保留策略约束。PD 交接只保留 ALLOCATE→Prefill 完成→LOAD/ACK→GENERATE 或 PREFILL_COMPLETE，不实现额外来源注册表、Reader 所有权及自定义 proto。Wire 阶段仅用于拒绝乱序帧，不代替 Scheduler 的调度状态。

接入实现位于 `java/engine` 中的 `service` 包。对外入口为 `RtpRpcService`，接收 `RtpServiceConfig`、实例自己的 `KvCache` 和现有 `Scheduler.Factory`；RtpNode 额外传入启动时确定的显式组接口开关，关闭时 EnqueueBatch/EnqueueGroup 返回 UNIMPLEMENTED。`RtpServiceConfig` 只含身份、角色、块大小、TP、序列/批 Token 上限和 ComputeModel，不包含排队、抢占或组批策略。RTP 后端通过 `RtpConfig.endpoint()` 提供这些值，由 `RtpNode` 装配 `RtpScheduler.factory(config)`：P 的 `mock.scheduler.mode=fifo`（默认）选择 RtpScheduler，`group` 显式选择 RtpGroupScheduler；D 始终使用普通调度。换 Scheduler 不需要复制接入代码。模块只有当前这一套 RPC，不增加协议选择、接入基类或插件注册机制。

接入层的独立测试只依赖共享组件，用独立测试 Scheduler 验证 P/D 入队、错误返回和锁归属；真实 RTP 策略的 RPC/PD 集成测试仍在后端模块中运行。模块依赖方向为后端→接入层→共享运行组件，不存在接入层→后端依赖。

## 构建与检查

在框架目录使用 Java 21、仓库 Maven Wrapper 和 uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/build.sh
PROMETHEUS_BIN=/path/to/prometheus ./scripts/check.sh
```

`build.sh` 编译 Java 模块并运行测试；`check.sh` 继续运行 Python 测试。缺少 `PROMETHEUS_BIN` 时跳过真实采集集成测试。多 Master 测试需要操作系统支持对应的 127/8 地址；macOS 未配置回环别名时跳过双 Master 测试，不自动改系统网络。生成目录不入库，Python 依赖由 `uv.lock` 锁定。

测试按能力归属：Java 验证组件行为和配置边界，Python 验证编译、协调、指标与跨语言接线。非法参数的组合在负责校验的组件内覆盖，不为每组参数启动 JVM；已有接线测试能覆盖的行为直接补进原测试。集成测试使用最小输入，不依赖正式 Case 的部署规模、倍率或等待时间。所有随仓 Case 统一检查可编译及输入文件存在；完整场景通过 `mock-case run` 执行，判定条件由 YAML 管理，不另写 Python 配置快照或逐个 Case 的 pytest 包装。进程回收、部分失败、取消、数据完整性和协议兼容的行为覆盖保留。

构建复用生产 `flexlb-common`、`flexlb-grpc` 的公式与协议。Master 还使用生产 cache/sync/api，Spring 只进入 Master 运行时。框架构建跳过 Boot 重打包和 Whale 分发归档，不改变生产模块独立构建的默认行为。

### 独立 Engine 服务与本地实验的边界

EngineMain 本地启动时原子发布 effective-config、endpoints 和 ready。Whale 启动时在日志记录有效配置和端点，平台管理发现和重拉。

Engine 的 HTTP 与原生健康检查读取现有 Loop/Scheduler 失败状态，不采集指标。独立进程每秒调用同一检查，不可恢复失败退出。不开新的健康管理模块，不把普通请求失败当作 Engine 故障。

本分支使用 DSV4.1 的原始多模态 RPC 与内部 PD cache spans；Prepared 输入准备仍在 Scheduler 之前完成，Scheduler 消费长度和只读缓存身份，不处理媒体 RPC。缓存元信息归 shared/cache，Master RPC 与共享 KvcmReporter 消费同一份有效块状态。RTP 实例仅装配 Reporter 的配置、实例身份和实际缓存能力；缓存锁只投递变更，网络线程负责 KVCM 注册、事件与恢复，远端读写另由 shared/cache 的 CacheLoad 与 remote/RemoteCache 管理，当前只实现非 group 的加载流程，未接真实存储。具体范围见组件参考中的 DSV4.1 Engine / KVCM 契约。

### 非 group 的异步缓存准备

KvCache 统一拥有本地分配、有效前缀提交和异步读写引用。CacheLoad 代表一次准备操作，立即就绪和远端等待使用同一入口；RemoteCache 只负责外部协议，不操作调度队列。普通 RtpScheduler 独立推进 loading，请求准备就绪后重新参与 FIFO 组批。RtpEngineLoop 通过 KvCache.onChange 接收容量/准备完成信号，网络调用不持有调度或缓存锁。RtpGroupScheduler 不支持远端适配器，保持已有本地组边界。后台写持有缓存引用，不使用 RequestState.pin 延迟请求终态。真实远端存储、payload 和 KVCM/KVS 适配尚未实现。

### 状态与观测边界

Scheduler 的 allocator loading 与 D 的 PD 协议等待分开统计。WorkerStatus 按 real 输出 phase/is_waiting，不再填写 real 未实现的 waiting_query_len/running_query_len；活动 execution_time_ms=-1，终态总耗时减最后一次离开 WAITING 时冻结的 wait 得到 execution_time_ms。正常加载后重新调度会更新 wait，加载中取消则保留之前的 wait；不能把该字段当作纯 Forward 时长。微秒阶段指标在终态上报，具体名称、零值和统计窗口见 component-reference。

KvCache 投递类型化的 match/read/write 完成事件，由 RTP 指标层映射 real 名称与单位，不把协议细节放入 Scheduler。读取未成功不增加远端复用，写入确认数由适配器返回；独立拉取 LOAD/STORE 引用与逻辑在途状态，后台写失败不依赖请求是否还存活。未模拟的 SDK/payload 子阶段没有指标样本。
