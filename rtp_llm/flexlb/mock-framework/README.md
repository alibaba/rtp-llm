# Mock 框架

这是独立重写的实验框架。当前版本由 Python 编译 Case、编排进程、执行步骤和门禁，Python 函数定义 Case 接口，Java 实现组件和实例协议。RTP Mock 使用生产协议，真实 Master 使用生产调度与 ZK 选举；共享组件不依赖旧 Mock 或 online_eval。

当前已接通本机真实 Master、新 RTP P/D、通用 V、Client 原生收发核心、标准压缩输入播放、client0 多 JVM 协调和 Prometheus。Engine 默认共享一个宿主 JVM，也支持每实例独立 JVM，实例间统一使用网络 gRPC。Master 停启、替换及 Client 主备发现已接通；旧采集格式仅支持离线转换，运行时只读新格式；ViT 图像元数据链路已接通，vLLM 尚未接通；未实现的能力不登记为可调用 Case 接口。已有验证是小流量功能接线，不能替代开发机/CI 的规模和压力验收。

## Scheduler 输入、执行完成、传输完成与状态查询

Scheduler 是每个后端的请求状态与调度策略所有者，不只是选择 batch 的函数。公共接入代码解析协议、检查格式并发送回执；`input` 决定初始状态、队列与资源准入，成功登记不等于资源已就绪。计算结果、传输结果和协议许可分别进入明确入口，外部组件不直接修改请求阶段或队列。

- `input(request, listener)`：登记请求并执行后端准入策略。
- `schedule()`：返回本次 Forward 的执行计划，不暴露或取走内部 running 容器。
- `computeCompleted(work, result)`：提交本次计算进度，决定继续计算、等待交接或终结。Compute 不修改后端状态。
- `transferCompleted(request, result)`：接收异步 KV 传输结果，由后端决定下一步；传输结束不自动等于整个请求结束。
- `control(request, event)`：处理后端协议许可等少量控制输入；RTP 的 GENERATE、LOAD ACK 等类型留在 RTP 模块。
- `status(latestFinishedVersion)`：按最终上报语义返回不可变的任务快照及原生协议要求的完成增量，不推进状态机。

PD 的初始通知/初始化和 KV 传输分开。当前接入协议共用，KV 传输使用公共异步组件，第一版只建模零耗时完成；保留完成、失败、取消和资源使用权接口，不加入带宽、分块或争用模型。P 计算完成后可以退出计算集合，但仍持有 KV；本地传输结束和 D 的 LOAD ACK 是不同事实，确认前不能释放源 KV。网络协议仍由 RTP 适配器执行，状态决策及转移归 Scheduler。

EngineLoop 负责唤醒、执行和结果交接，不判断后端队列和协议阶段。新输入、Compute/Transfer 完成、输出恢复可写、取消、资源释放和关闭触发唤醒；同时按最近请求 deadline 定时唤醒。无事件、无期限时阻塞等待，不固定周期轮询。Forward 在途时仍处理状态维护和 I/O 事件，但不提交第二个 Forward。资源可用变化必须有明确通知，不能在移除轮询后遗漏容量唤醒。

状态转换与任务快照复制使用同一调度锁。查询可读到转换前或转换后的已提交状态，不能读到队列移动一半的状态；不等完整调度轮、Forward 或 PD 完成，不维护影子状态。活动记录移除与完成增量发布在同次状态提交完成，网络通知在锁外执行。Cache 单独采样一次，再由既有 GetWorkerStatus 编码组装；不承诺任务与 Cache 来自同一全局时刻。这与 real RTP 分开读取运行记录、Stream 阶段和 Cache 的边界一致，不复制其类之间的分工。

## 设计边界

本节记录已确定的职责边界，包含尚待实现的部分。当前已迁移 Python Case 接口、Step/ref 和 Prometheus 查询，并拆出协议无关的 EngineLoop、请求进度/终态和 RTP 调度域；本批接通 Engine 宿主、实例扩缩容和发现更新；Client 已接通标准压缩输入播放与 client0 协调，其他后端仍待实现。其后的“当前代码职责”及运行说明描述可执行范围。

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

Step 统一为一次接口调用，包含 id、describe、interface、args、within 和可选 expect。删除 YAML 的 call/check 外壳及两套重复 Step 类：无 expect 时执行并保存结果，有 expect 时在同一次返回后判断。读取也可以只取值供后续引用；写操作的回执可以断言，但不因断言失败重放。接口读写属性与是否断言相互独立，不把普通检查隐式变成轮询。

执行链统一为解析引用、校验实参、固定查询时刻和预算、调用 Python 函数、校验返回值、保存结果、执行可选断言。编译阶段检查引用必须来自前置 step 或允许的全局信息，检查字段路径与类型；运行时解析后再次校验。Python 类型注解本身不等于运行时校验。

普通数值查询直接返回数值，单位声明在接口上；地址集合返回列表，动作返回明确回执，逐 Master 状态保留映射。断言按结构和运算符判断，不按 MetricValue、PerMasterEngines 等业务类型名称偷偷聚合。接口文档和清单可以由 Python 定义导出，用于保存实验现场，不再依赖 Java 构建才能编译 Case。

Python async 只表达非阻塞等待，不代表任意 step 可以脱离 Runner 后台执行。client/start 完成准备和开始命令后返回，Java 自行播放；client/stop 停止并收尾。Client 不引入 task ID、任务 ref 或通用后台任务框架。

本轮重构按职责变化整体重写 Registry、Step 模型、编译和执行主链，联动 CLI、启动配方、夹具和构建入口；复用严格 YAML 解析、进程回收、采集与记录等仍成立的机制。Java 内部端点保留明确的请求/响应协议，但不再声明整套 Case API；配置类型留在所属组件，不能随旧接口目录误删。保留引用重校验、固定时间窗口、错误结果和取消清理的验收，不保留另一套旧 Case 执行入口作为兼容分支。

### 公共 Engine 与独立调度域

RTP 和 vLLM 的设计目标是复用请求事实、资源机制、EngineLoop、Forward、输出和生命周期工具；各自的 Scheduler 拥有队列模型、准入、组批、抢占恢复和结果后的状态转换。替换调度策略不应修改 Compute 或 EngineLoop。原生协议和计算模型参数仍在各自接入/配置边界适配，不宣称两个真实后端只有调度差异。

当前已打通 Java 装配入口 `RtpNode.start(config, Scheduler.Factory)`，指定监听端口时使用 `start(config, rpcPort, controlPort, factory)`。工厂的 `create(lock, cache)` 为每个实例创建一个新 Scheduler，收到协议接入层与 EngineLoop 共用的调度锁，以及本实例独占的 RTP Full-KV 管理器 KvCache。工厂不关闭缓存；节点在关闭或调度器创建失败时回收它。默认入口只在 RtpNode 选择 RTP 工厂，LocalPrefill/DecodeAdmission 不再创建或引用 RtpScheduler，公共执行层不识别 RTP 配置。

P 接入层保留原生报文、模型上下文及生成参数校验，单批 token 预算等调度策略由 Scheduler 判定。RTP 默认策略仍拒绝超出单批预算的 Prefill，并返回原有 INVALID_ARGUMENT/605；替换 Scheduler 不会被接入层的重复策略校验拦截。请求数量保护上限统一来自 RtpConfig，不再由接入层读取具体调度器的常量。

本批仅完成装配与公共组件边界；vLLM Scheduler 尚未实现，Python 后端选择和对应 Case 接线留到后续。没有自动插件扫描、空后端占位或静默回退到 RTP。

```text
网络/控制线程 → input / control / transferCompleted / finish
                           ↓
EngineLoop：维护到期与依赖 → Scheduler.schedule → Compute.forward
                           ↑                         ↓
                      下一轮调度 ← Scheduler.computeCompleted
                                      ↓
                         锁外发送输出/就绪/终态通知
```

每个 Engine 同时至多一个 Forward。初次和空闲后先调度；每轮计算结束后，结果交回 Scheduler 提交进度、检查有效性并决定继续或结束。公共 RequestState 保证终态与资源清理只发生一次，但不定义后端阶段或替调度器决定队列迁移。Scheduler 不持有执行器、计时线程或 EngineLoop，也不发送网络响应。

| 边界 | 职责 |
| --- | --- |
| `input(Input, Listener)` | 校验身份、接纳并登记请求，返回只读请求句柄。外部不能取得可变队列或修改请求进度。 |
| `schedule()` | 检查预算与资源，调整内部队列，返回本轮 Work/ForwardBatch；没有可执行工作时返回空。 |
| `computeCompleted(work, outcome)` / `failed(work, error)` | 提交本轮有效结果，决定后续队列和阶段完成，清理本轮资源引用。 |
| `transferCompleted(request, result)` / `control(request, event)` | 分别接收传输完成结果和后端协议许可，不把传输成功当作自动获准生成。 |
| `finish` / `beginTransfer` | 终止请求，或为已经就绪的数据取得短期读取引用；读取结束不等于收到远端传输确认。 |
| `status(latestFinishedVersion)` | 同一调度锁内复制已提交任务、计数和原生完成增量；不推进状态，不返回可变请求。 |

ForwardBatch 是本轮执行输入，不是 running 容器本身。它固定请求成员、本轮 token 数和执行身份；请求的持续状态由 Scheduler 保留。P 通常只执行一轮，D 在 running 中跨轮保留，不每轮回到 waiting。外部线程可在 Forward 期间经公共入口提交请求或依赖事实，但不能修改已经提交的 batch。

Scheduler 调用在同一短锁下串行化；锁不覆盖计算等待或网络发送。公共 EngineLoop 使用一个虚拟线程等待事件和最近 deadline，入队、控制/执行完成、容量释放和关闭都会唤醒；没有固定周期轮询，容量版本未变化时不反复尝试初始分配。没有可执行 batch 也必须处理取消、期限和传输结果，不能依赖下一次 Forward 才收尾。

通知按请求顺序、锁外执行。每个请求最多一个尚未处理完的调度通知；RTP 输出通知只交给协议容器，不等待 Fetch 或网络可写，不暂停后续计算。计算接纳上限覆盖等待调度通知完成的请求；协议结果按独立保留上限清理。通知失败进入该请求的失败清理；终态通知等基础设施异常使 Engine 的后续状态读取明确报错。

批次取消按成员处理：取消成员不再提交结果，但资源引用保留到该 Forward 结束。只有全部成员终止或 Engine 关闭时才取消整个模拟执行。旧回调按执行对象和请求句柄核验，不能用复用的请求 ID 找到新请求继续推进。一次完成的 Forward 只记录一次耗时，计算量包含这轮实际执行的全部成员，输出量只计有效提交。

### 队列模型、观察与注入

队列模型属于 Scheduler，不另建强制共享的 QueueModel 层。一个实现可以有独立 loading 队列，另一个可以把 loading 作为 waiting 成员的状态；抢占也可能回到 waiting 重新计算，而不是进入 swap。外部只通过操作和快照契约交互。公共请求事实需要逐步覆盖输入、缓存命中、已完成 Prefill、生成进度和本轮工作量；尚未实现分块 Prefill，不能把重复执行当作分块支持。

当前 RTP 保留普通 FIFO waiting、显式 batch 队列、持续 running 和请求上的容量/输出/传输就绪状态。`max_generate_batch_size` 控制 P/D 每批请求数，默认 1；删除重复的 `max_context_batch_size`。P 按完整输入长度和 `最长序列长度×序列数` 检查 `max_batch_tokens_size`，生成模式额外保留一个 token；合法 singleton 可超过该 batch token 预算。预算为 0 时 P 仍可执行 singleton，不表示无限组 batch。D 按请求数上限接纳，不复用 P 的 token/矩形限制。Compute 使用扣除命中后的实际计算长度。P/D 初始分配遇到临时容量不足等待容量变化；已进入生成的 D 增长 KV 失败则终止，对齐当前 real RTP，不实现 vLLM 抢占/重算策略。

`max_inited_kv_cache_streams` 为正时限制 P 已取得 KV 的请求数，包含仍持有数据等待 PD 的来源；0 不限制。普通 P 准入达到 `max_batch_tokens_without_cache` 后停止继续接纳，本轮最后一个成员可跨过阈值，不把它实现成严格总和上限。该配额不额外套到 real 的显式组路径。CP padding 和多序列宽度仍不展开模拟。

按请求计数，不能直接使用 group/batch 容器的 size：

- `waiting`：等待计算准入或初始 KV 容量的请求。
- `running`：已进入持续计算集合的请求，包含轮次之间等待输出交给协议容器的 D；不等于此刻 Forward 的成员数。
- `loading`：资源已预留，等待 KV 加载或生成许可的请求。
- `ready`：P 计算阶段已完成，仍持有传输数据的请求。

上述四类互斥，由 Scheduler 统计；终止但仍被执行引用的成员不计入四类，KV 占用继续保留。公共 QueueSnapshot 不再根据“不是 FORWARD 就是 waiting”推导。Mock 指标保留四类；原生状态接口只有 waiting/running，因此将 waiting、loading、ready 合并为原生 waiting，与任务列表的 is_waiting 一致，避免 Master 漏算已预留资源的请求。

Scheduler 的 `input(Input, Listener)` 提交普通 single；`input(List<InputItem>)` 一次提交一个完整显式 batch，每项返回独立的 Request 或失败，登记在同一 EngineLoop 锁内完成后才唤醒调度。组来自调用边界，不按相同 groupId 拼接，也不等待缺失成员。每个成员仍各自管理 Listener、deadline、取消和 KV 租约。

P 默认保留 Master batch 边界。正常组只从本组选择实际 Forward 成员，不与其他组或普通 single 混合；组内按输入顺序扫描，暂时放不下的成员保留，后续较小成员仍可准入，剩余成员留在原组。成员失败、超时或取消不默认结束整组。有效组成员数超过 max_generate_batch_size，或超过已开启的 max_inited_kv_cache_streams，按 real 退回普通 single。普通工作与组在空 Forward 边界切换；组完全无法推进时允许普通请求推进，不因组屏障造成容量依赖死锁。无需等待一组在 D 全部生成完成才开始下一组 P Forward。

EnqueueBatch/EnqueueGroup 先并行准备各成员 D 资源，收齐准备结果后，将成功且未取消的成员一次提交到 P 的组入口；准备失败逐项返回错误，不回滚其他成功成员。ACK 表示正式入队，不保证已经完成 KV 准入或 Forward，后续执行失败通过 Fetch/输出返回。D ALLOCATE 清除 P 的组字段，继续按独立 Decode 请求调度。注入只能走正式提交与依赖入口，不能跳过资源和传输前置条件。本地搬运同步完成，不新增异步组装队列；暂停、换出和重算仍不实现。

### PD 数据就绪与资源生命周期

PD 通过“数据就绪”和“传输完成”衔接。Scheduler 判断阶段是否完成、发布就绪通知并管理依赖；协议/传输组件负责 ALLOCATE、LOAD、GENERATE、网络读写和 ACK；公共引用机制保证资源不会提前释放。Compute 不在每轮完成后自动传输，分块 Prefill、加载命中和提前发起 LOAD 都不能依赖这一假设。

P 完成 Forward 后退出计算集合，保留 KV 和请求期限。Read 期间额外取得读取引用，Read 返回仅表示本地交付结束；P 收到 D 的 LOAD ACK 后才提交传输完成并释放阶段资源。没有确认、传输失败或超时仍走失败清理，不发布成功缓存。释放 P KV 后，端到端 P RPC 仍可继续转发 D 输出。

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

本地运行默认采用共享模式，P/D/V 实例放在一个 JVM；独立模式用于需要进程隔离的实验。模式在运行开始前确定。共享模式新增 Engine 就在已有宿主创建对象；独立模式新增 Engine 就启动专属 JVM。不实现任意多 JVM/多实例装箱、动态迁移或共享宿主不足时自动另开 JVM。共享指代码复用，实例仍独立拥有缓存容量、请求状态和 Scheduler。

实例使用服务地址区分，同 IP 下用端口区分。控制、业务 RPC 和 metrics 地址之间的对应关系留在发现信息中，不向 Case 暴露额外的 JVM 身份。实例之间统一通过 TCP 通信，共 JVM 不另走 in-process 传输。Python 内部仍须保存进程与实例归属，以处理宿主退出和清理。

进程 env、heap 和资源限制属于 JVM；每个 Engine 的业务参数来自自己的配置对象。共 JVM 时不能用修改进程 env 或全局 System properties 的方式切换实例配置。原生 env 风格字段可以保留其名字和值，由实例加载器从显式配置读取。

停止 Engine 是关闭对象：拒绝新请求、终止活动、关闭自己拥有的 RPC/连接、释放资源并移除发现地址。共享宿主中的其他实例继续运行；最后一个 Engine 停止后宿主可保留至实验收尾，供后续扩容。独立模式完成停止后回收专属 JVM。对象清理不依赖 System.gc，也不承诺能强杀某个 Java 线程；关闭超时使实验失败并进入宿主清理。

Runner 在动作前登记预期停止，结合子进程退出和有界健康检查判断结果；预期停止仍需确认完成，不能吞掉所有异常。停止单个 Engine 不允许对应共享 JVM 意外退出。宿主意外退出映射到其全部实例，记录首个根因，停止剩余发压并清理实验；不自动重启或重放未知结果的写操作。业务请求错误按 Case 门禁判断，不与基础设施退出混为一谈。

### 发现文件和产物

进程启动材料和实例产物分目录保存。每个输出文件只有一个写入者，不让多个 Engine 共享同一个输出目录并各自覆盖 `endpoints.json`；宿主若汇总多个实例端点，由宿主统一写。ready/endpoints 采用临时文件加原子替换发布，部分启动失败要回收已经启动的对象。沿用 `components/<角色>-<序号>/` 保存每个实例的 config、effective-config 和 endpoints；共享宿主自己的启动材料、stdout/stderr 单独保存，不能把宿主日志冒充某个 Engine 的专属日志。聚合 topology/discovery 由 Python 单写，实例不得各自覆盖这份文件。

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

唯一输入定义为 `java/client/src/main/proto/client_input.proto`，Java 由 Maven 生成，Python 提交同一 protoc 3.25.1 生成的 client_input_pb2.py。Request 保存已展开的生成输入：非空、非负 int32 token_ids，正 int32 max_new_tokens（本格式中的固定目标输出长度，LoadClient 发送 ignore_eos=true），可选非负 int32 priority（缺省为 50，显式 0 保留），可选非负 int64 arrival_offset_ms（相对于源起点的毫秒偏移，尚未除以倍速）。可选 images 保存 url、width、height、offset、length；token_ids 中提前为每张图展开 length 个占位 token，按 offset 排序且不得重叠，实际长度必须与 V 输出一致。Client 和 P 用 V 返回的逐行特征 hash 替换这些占位值，普通 token 保持不变。完整生成参数及真实观测仍不是此播放格式的内容。未知字段、非法请求及不支持的版本均拒绝。

文件以 `MCKINP01` 标识输入格式 v1，之后每批为 4 字节大端压缩长度加一个 Zstd 压缩的 Protobuf Batch。帧必须为 single-segment、声明解压长度并带校验和，压缩及解压单批上限均为 4 MiB；每请求 Protobuf 上限 1 MiB，每批至多 1024 个请求。写入器按约 1 MiB 或 1024 请求封批，不累积整个文件。Batch.sequence 从 1 连续递增，末批只含 total_requests；缺末批、计数不符、截断或尾随数据均报错。输入和分片使用同一份格式与读写器，输入消息与执行证据消息分开定义。

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

离线准备时直接构造共享 Protobuf Request 并写压缩文件：

```python
from mock_framework.input import pb, write_inputs
write_inputs("traffic.binpb.zst", (
    pb.Request(token_ids=[1, 2, i + 3], max_new_tokens=3)
    for i in range(20)
))
```

写入器拒绝覆盖已有文件，生成失败删除本次不完整文件。基本 Case 直接携带标准二进制输入。离线转换示例：`python scripts/convert_lineage.py <source.xz> <output.binpb.zst> --output-cap <tokens> --output-mean <tokens> --seed <seed>`；输出文件和 manifest 均拒绝覆盖。控制接口仍传小型 HTTP 元数据，不在 JSON 控制报文中传请求数据。

`RequestFilesTest` 在独立 48 MiB 堆的 JVM 中分片并完整播放解压后超过 64 MiB 的输入，覆盖非法输入、截断、校验和、内容变化和准备取消。跨语言测试验证 Python 写→Java 读、Java 写→Python 读；真实 Master 接线测试在 1/2 个 Client JVM 中播放并校验分片总量、全局请求 ID 和发压窗口内的合计 QPS。这些是有界读取和功能接线验证，不测量最大可达 QPS。

`client/qps` 复用公共 Python 查询，统计 mock_client_requests_issued_total，是发出 QPS；最终检查选择实际发压窗口，不能用含空闲和收尾的实验全时长稀释目标。latest 在自然结束后可能为零，all 尚不支持；缺实例或采样不足按公共查询规则报错。各 Client 仅上报自己的指标，client0 不重复上报全局汇总，不再增加一套发压有效性门禁。

每个 Client 在自己的 playback/<轮次>/ 保存计划与结果，client0 另在 coordinator/<轮次>/ 保存总计划、参与者和汇总。文件单写并原子替换，多 JVM 和后续轮次不会覆盖旧结果。每轮输入分片与逐请求执行证据分别保存，输入不作为已发送/已完成的证明；磁盘须容纳输入和各轮分片。Client 结果包含 produced/issued/finished/unsent、在途/队列余量、终态计数和异常，业务错误计数不等于执行异常。

启动配方中的 Client 使用 java -jar <framework>/java/client/target/mock-client-1.0-SNAPSHOT-all.jar，由 Runner 注入配置、拓扑和独立输出目录。部分停止与队列/并发默认值的压力校准仍待讨论。

### Client 逐请求记录与查询

`NativeClient` 在实际执行边界记录 ISSUED、SCHEDULE_START、SCHEDULE_RESULT、FIRST_TOKEN 和 TERMINAL。同一个逻辑请求的主备尝试共用 request_id，issued 只计一次。记录包含墙钟毫秒、单调耗时、目标 Master、尝试序号、gRPC 状态与业务错误码、首 token 耗时、实际输出 token 数及终态。不记录输入 token 正文或完整响应，错误说明最多 4096 字符，截断会标记。Schedule 响应分别保留传输状态、原始业务 code（成功通常为 200）和可空 response_success；业务拒绝不会伪装成 gRPC 传输失败。Schedule 超时/取消的结果只表示 Client 观察，不证明远端未接收；Cancel 仍是尽力而为。

每个 Client 每轮独占 `components/<client>/playback/<轮次>/requests.binpb.zst`，目录与记录文件拒绝覆盖。公共字段定义是 `java/client/src/main/proto/client_evidence.proto`，Java 在 Maven 构建时生成，Python 提交同一 protoc 3.25.1 生成的 `client_evidence_pb2.py`；修改 proto 时必须重新生成两侧，不能分别手写协议。

每个文件由连续帧组成：4 字节大端压缩长度，再接一个带校验和的独立 Zstd frame；解压后为 varint 长度和一个 Protobuf EventBatch。每帧携带版本、Client、playback、帧序号与分片身份，每条事件有连续序号。最后一帧携带 Seal 的 issued/finished 计数。没有额外 JSONL 执行记录或永久逐请求索引，`mock-requests` 只在查看时解码输出。

网络回调只向有字节上限的队列追加事件，单独写线程按约 1 MiB 或 1 秒成帧并 force 到磁盘。当前队列上限 8 MiB、Zstd level 3，是初始参数，不是最大吞吐承诺。入队、写入、持久化不是同一个状态；Client 只有等待最终 Seal 写入并 force 成功才完成。队列满、超大事件、写入失败或收尾超时使 worker 报错，client0 沿用现有监督机制停止整轮。进程意外退出可能留下未完成请求；不能补造终态，也不能把无 Seal 的文件判完整。

`client/stop` 及 Runner 收尾会流式读取各 Client 的本轮记录，与控制回执及 result.json 对账。缺文件、损坏、事件缺序、身份不符、未完成请求、缺 Seal 或计数不符都使实验 ERROR。Python 同时只保留在途请求状态，不把所有请求载入内存。查询过滤和返回条数限制不会跳过其余记录的完整性验证。

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

离线工具直接读取各 Client 目录中的 endpoints.json，不依赖拓扑历史文件。离线查看：`uv run mock-requests <Case输出目录> --playback 1 --outcome timeout`；可重复 `--target <Client地址>` 或 `--request-id <ID>`。命令流式输出 JSON 到终端，需要时可重定向导出；遇到坏尾部会非零退出，已输出部分仅作为恢复前缀，不能当作完整结果。

### Python 指标查询与公共指标

Java 产生指标并暴露 `/metrics`；Python 直接调用 Prometheus HTTP API，不经过 Engine 查询代理，也不新增 Java 查询服务。原始 PromQL 可在 Prometheus 存活期间查询其已采集的数据，框架不维护实例历史。

公共 Engine 使用同一组指标含义和采集事件，Python 默认只实现一套公共语义查询。公共客户端处理 HTTP、超时和响应解析，指标函数处理 PromQL、单位、目标范围和结果形态。确有后端原生指标差异时，映射和公式放在 RTP/vLLM 对应 Python 模块中，Runner 直接调用；不为相同指标复制两份实现。原始 `prometheus/query` 走公共查询客户端，不进行后端改写或自动附加过滤。

指标按实例上报、按查询语义聚合：总 QPS 求和；成功率和命中率分别汇总分子/分母后求比率；平均值按样本数加权；p99 不能平均各实例 p99，须合并兼容直方图或使用完整请求记录。时间窗口、目标覆盖和采样完整性要校验，缺数据不补零。语义查询只使用当前活动实例；显式 targets 也必须仍然活动，不根据时间窗重建过去的成员集合。需要查询已下线地址时，直接使用原始 PromQL。不同 Master 的状态保留逐实例结果，不取并集掩盖差异。

去掉业务指标上的 RTP/vLLM `engine`、逻辑 `node`、JVM 和主动添加的实验 ID 标签，保留必要的角色与实例地址区分。独立采集端点可使用 Prometheus `instance`；共享 `/metrics` 时必须在样本上区分实例服务地址，不能让同名同标签序列碰撞或重复采集。`host_ip`/`container_ip` 保持物理含义，共 JVM 时不能仅靠相同 IP 区分 Engine。实验默认独占 Prometheus，跨实验合并分析时再补来源标签，不能在合并后无法区分数据来源。

一个真实完成的 Forward batch 记录一次耗时，不能按 batch 中请求数重复记录。请求终态按请求计数，token 按本轮实际计算和提交分别统计。新增两项 Forward 指标，不增加对应服务接口：

- `rtp_llm_model_forward_us`：模拟 Forward 的实际耗时，单位微秒；对齐 real 模型 Forward 调用区间的含义，不包含请求排队或网络 TTFT。
- `mock_model_forward_planned_us`：公式和可选噪声确定的本轮计划耗时，单位微秒。

实际计时从提交本轮计算到公共执行层收到完成通知，包含计时器延迟和回调调度等待，在处理结果及提交 token/KV 前结束。计划和实际对同一完成 batch 各记一次，取消批次不补完成样本。差值可以揭示宿主调度/GC 等干扰，不能单独归因为 CPU 不足；Client TTFT 仍取真实接收时间。本仓 real `NormalExecutor.cc` 围绕 model_->forward 计时，`RtpLLMMetrics.cc` 将该字段注册并报告为 Gauge；本轮因此采用同名微秒 Gauge，计划耗时也用 Gauge。Gauge 保留最近一批，不能拿采样点平均值冒充逐批平均；吞吐仍使用累计 token 与耗时 Histogram 的 sum/count。KMonitor 与 Prometheus 的采样、聚合管线不据此宣称完全相同。

### 落地顺序与待定边界

1. 从 LocalPrefill/DecodeAdmission 分离协议、调度域和公共执行组件，保留现有超时、取消、P/D 交接及清理行为；先让 RTP 使用公共 Engine。
2. 把队列、容量等待、选批和完成后的重新入队一起移入 Scheduler，不能只抽 `advance()`。用第二种明确的调度行为验证边界，再按选定版本实现 vLLM 策略；当前 RTP 有界合批不等于完整 RTP/vLLM 调度。
3. 将 Case 级接口与 Prometheus 查询迁移到 Python，保留实例协议、Java 指标产生及既有校验。迁移完成后删除旧查询服务和重复定义，不长期维护两条调用链。
4. Engine 两种放置方式、动态上下线和发现更新已接通；Client 已接通上述第一批能力，生成输入文件、图片元数据和有界分片已接通，后续按剩余边界推进采集格式转换和请求执行记录。每批改动同时补齐对应调用方、配置、测试和文档，避免旧入口继续承载另一套语义。

本批落实公共 Scheduler 操作、私有队列、持续 running、有界多请求 Forward、部分批次取消和 PD 确认后释放。强制 group、分块 Prefill、抢占/重算、vLLM 对齐版本及 connector 范围仍待确定。

验收关注可观察行为：两种 JVM 放置下缩容数量和流量总量一致；共 JVM 停单实例不影响其他实例；不同队列容器报告相同请求数；batch 内取消不损伤其他请求；取消/抢占无 KV 泄漏或提前复用；每批 Forward 只记一次；任一基础设施失败不产生虚假的整体 PASS。共享宿主的实际延迟可以不同，不承诺两种放置方式性能相同。

## 当前代码职责

从本节起记录现有实现和命令。Python 接口与 Java 内部协议已分开；Engine 已接通共享/独立宿主和实例扩缩容；Master 与 Client 保持逐实例进程。

| 目录 | 负责内容 |
| --- | --- |
| `python/mock_framework/config/` | YAML 解析、检查、点分配置展开、引用和可执行步骤编译。 |
| `python/mock_framework/execution/` | 运行环境装配、统一 Step 执行、角色生命周期、拓扑、采集、HTTP 传输、断言与记录；各模块边界见下表。 |
| `python/mock_framework/interfaces/` | 已接通的 Case 函数：逐 Master 状态、Client start/stop、公共指标与原始 PromQL。 |
| `java/interfaces/` | Java 内部端点的类型/校验及共享模型定义；不再定义或导出 Case API。 |
| `java/runtime/` | HTTP 接口绑定、参数和回执校验、调用上下文、实例生命周期句柄和进程监听地址。 |
| `java/host/` | P/D/V 统一实例装配和可执行 JAR；单实例入口 EngineMain、本地多实例宿主 EngineHost。宿主不包含调度或集群策略。 |
| `java/master/` | 真实 Master 的启动入口、生产状态到公共读取接口的适配、独立 ZK 启动入口。 |
| `java/client/` | client0 协调、Player 控速、LoadClient 队列/并发/生命周期、NativeClient 原生收发与实际计时。 |
| `java/shared/vit/` | 跨后端共用的 V Engine：图像编码、组批、embedding/hash 字节缓存和节点装配。 |
| `java/engine/rtp/` | RTP 协议适配、P/D 调度策略、资源申请时机、P/D 交接与节点装配。 |
| `java/shared/multimodal/` | 原生媒体输入、Mock 元数据校验与 TCP 客户端，无 V 服务、计算或缓存依赖。 |
| `java/shared/engine/` | EngineLoop/Scheduler 执行边界及工厂契约、RequestState 进度与终态/资源使用权、QueueSnapshot 请求计数。 |
| `java/shared/compute/` | Forward batch、公式、噪声、Decode 推进、停止模型和可取消计时。 |
| `java/shared/cache/` | BlockPool 物理容量与引用、KvCache 块租约及前缀发布、BlockTree 叶子候选与驱逐。 |
| `java/shared/metrics/` | Engine 指标、注册表和 Prometheus 导出；不提供查询代理。 |
| `tests/`、各模块 `src/test/` | Python/跨语言集成与 Java 测试；探针不作为正式业务入口。 |
| `scripts/` | 构建和检查入口。 |

Python 运行时按以下边界组织。`RunEnvironment` 装配实例并安排启动和收尾顺序，不分发 Case 业务操作；各服务显式接收所需依赖，不接收整个运行环境再修改它的内部状态。

| 模块 | 唯一负责的状态与行为 |
| --- | --- |
| `execution/runner.py` | Step 的引用、参数、预算、调用、返回校验、断言、等待与结果；不判断具体接口名。 |
| `execution/environment.py` | `RunEnvironment` 的静态装配、启动与收尾顺序。 |
| `execution/processes.py` | `ProcessSupervisor` 的子进程所有权、退出监视、操作中断和回收；不识别 Master、Engine 或 Client 业务。 |
| `execution/topology.py` | `Topology` 的当前实例端点及文件发布；读者拿到独立快照。 |
| `execution/components.py` | Master/Client 通用的配置文件、进程启动、ready 检查与端点登记；原生配置与端点额外校验由调用方提供。 |
| `execution/master.py` | `MasterGroup` 的 ZK 选举环境、独立 JVM 增减、原生配置和 Master discovery 发布。 |
| `execution/engines.py` | `EngineHosts` 的实例分配、JVM 放置、扩缩容、成员检查和宿主退出影响范围。 |
| `execution/clients.py` | `ClientGroup` 的 Client JVM、当前播放待收尾状态、client0 控制与结果/证据完整性检查；流量分片和控速仍由 Java 负责。 |
| `execution/monitoring.py` | `Prometheus` 的采集器、目标、基线、末次采样和采集完整性记录；不执行 Case 语义查询。 |
| `execution/launch.py` | 启动配方的结构校验与规范化，不持有运行状态。 |
| `interfaces/` | Case 参数/返回契约及操作组合；调用对应服务或传输/查询函数。运行时服务不反向导入这一层。 |

拓扑是 Python 根据已校验的实例端点生产的运行记录，不代表生产 Master 的成员状态或主备判断。`Topology` 统一登记与退役，发布当前 topology.json，并把快照交给 `MasterGroup` 更新活动 Master 的 discovery。Master、Client 等消费者自行加载配置；Python 不替它们推导主节点。发布失败使实验报错，保留已登记状态与进程所有权，随后清理；不宣称多个文件的发布是一个事务。

启动顺序固定为 Prometheus、ZK、Engine、Master、Client、采集基线。收尾先停止 Engine 健康检查，再收集 Client、完成最后采样，依次清理 Client、Master、Engine、ZK、Prometheus，最后由进程监督器回收剩余进程。某一步失败仍继续后续清理，错误进入最终结果；进程未回收时保留其所有权，不能仅删除内部记录。最终没有存活的受管进程后清空活动拓扑。

新增 Case 操作在 `interfaces/` 用现有装饰器注册；新增角色生命周期行为进入对应服务；只有新的跨服务启动/收尾顺序才修改 `RunEnvironment`。不为接口名、角色名或 Case 名向 Runner 主循环增加分支，也不引入通用插件框架或事件总线。配置入口和业务 Case 的下一批接线以这些边界为基础，本批保持现有 YAML 参数、返回值与产物格式。

共享指代码复用，不是不同实例共享资源。本批按职责重写 LocalPrefill/DecodeAdmission：它们保留原生校验、ALLOCATE/LOAD/GENERATE 协议状态和响应编码，通过公共 Scheduler.Request 句柄与 EngineLoop 交互；RtpTasks 只转换输入与状态编码，RtpErrors 负责原生错误码；上报阶段、准入等待时间和终态完成记录由 RtpScheduler 提交。RtpScheduler 无 gRPC/protobuf/执行器依赖，拥有 FIFO、持续 running、资源准入和执行结果提交。

RequestState 是共享的请求进度、终态和资源清理机制。调度器决定何时申请和成功发布缓存；公共机制确保终态只通知一次、发布失败也尝试释放、取消请求不能再提交进度。Forward 持有资源使用权，终止请求不会提前释放仍被使用的块；EngineLoop 对执行对象做身份校验，晚到回调不能影响下一次执行。当前模拟 Forward 使用可取消计时；单成员取消保留整批执行，全部成员终止或关闭 Engine 时才取消计时并释放使用权。

P 的 retained 请求计算完成后变成 READY，释放 Forward 槽位但保留 KV，直到 LOAD ACK、取消或期限到达；读取期间另持资源引用。D 的接入/传输与计算进度分开保存，完成后的 token/迭代数只由调度域推进。请求终态先清理资源；Scheduler 在同次状态提交中记录指标、发布原生完成增量并移除活动请求。随后由协议适配器异步完成 RPC 响应，不影响状态查询的可见性。

当前生产路径支持多个独立请求合批，P/D 使用相同调度循环。测试覆盖持续 Decode、批内独立取消、临时容量等待、输出阻塞和 PD 确认边界；完整 group、分块 Prefill 和 vLLM 未实现。

## 构建与检查

在本目录使用 Java 21、仓库 Maven Wrapper 和 uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/build.sh
PROMETHEUS_BIN=/path/to/prometheus ./scripts/check.sh
```

`build.sh` 编译 Java 模块并运行测试；`check.sh` 继续运行 Python 测试。缺少 `PROMETHEUS_BIN` 时跳过真实采集集成测试。多 Master 测试需要操作系统支持对应的 127/8 地址；macOS 未配置回环别名时跳过双 Master 测试，不自动改系统网络。生成目录不入库，Python 依赖由 `uv.lock` 锁定。

测试按能力归属：Java 验证组件行为和配置边界，Python 验证编译、协调、指标与跨语言接线。非法参数的组合在负责校验的组件内覆盖，不为每组参数启动 JVM；已有接线测试能覆盖的行为直接补进原测试。集成测试使用最小输入，不依赖正式 Case 的部署规模、倍率或等待时间。所有随仓 Case 统一检查可编译及输入文件存在；完整场景通过 `mock-case run` 执行，判定条件由 YAML 管理，不另写 Python 配置快照或逐个 Case 的 pytest 包装。进程回收、部分失败、取消、数据完整性和协议兼容的行为覆盖保留。

构建复用生产 `flexlb-common`、`flexlb-grpc` 的公式与协议。Master 还使用生产 cache/sync/api，Spring 只进入 Master 运行时。框架构建跳过 Boot 重打包和 Whale 分发归档，不改变生产模块独立构建的默认行为。

## 第一个可运行 Case

`cases/basic/case.yaml` 使用真实 Master、RTP Prefill/Decode 和一个 Client。20 条请求保存在同目录的 requests.binpb.zst，总 QPS 为 2、总并发为 2。Case 检查唯一主 Master，在播放中段的 4 秒窗口检查 QPS，然后收尾并检查 20 条请求全部成功。QPS 门禁范围为 1.5–2.5，用于小流量接线检查，不是压力或容量验收。

在本目录完成 Java 构建后，指定 Java 21 和 Prometheus 的可执行文件，生成本机启动配方，再运行同一份 Case：

```sh
uv run --locked mock-case launch cases/basic/case.yaml \
  --java "$JAVA_HOME/bin/java" \
  --prometheus "$PROMETHEUS_BIN" \
  --engine-mode shared \
  --output /tmp/basic-launch.yaml

uv run --locked mock-case run cases/basic/case.yaml \
  --launch /tmp/basic-launch.yaml \
  --output /tmp/basic-result
```

配方文件和结果目录都要使用新路径。`launch` 不启动组件、不下载工具；它检查本地构建产物及 Master classpath，并复用 `prepare_launch` 的配置校验，输出当前 `run --launch` 接受的 YAML。配方中的路径为本机绝对路径，不提交到仓库；需要调整 JVM 参数时编辑生成的配方。默认设置 `-XX:ActiveProcessorCount=2`，只限制本地 JVM 看到的处理器数，不代表 CPU 配额保证。

本机 Mock 配方临时为 Master 添加 `-Dflexlb.monitor.metric-whitelist=flexlb_`，导出全部 FlexLB 业务指标；JVM 等非 FlexLB 指标沿用现有导出。该设置不修改生产 Master 的默认值。全量采集在规模压测下的序列数、采样耗时及资源开销尚需验证，暂不据此取消生产限制。

`shared` 把 P/D 放进一个 Engine JVM；改为 `--engine-mode isolated` 可生成每实例独立 JVM 的配方，Case 不变。Master 与 ZK 始终独立 JVM。Client 按当前拓扑自行发现主 Master。输出包括 result.json、gates.jsonl、组件日志、Client 请求证据及 Prometheus 数据；异常会中断并执行统一清理。

## 实验结束后查看基础 view

默认大盘的分区、聚合、地址筛选、标签来源和未实现观测见 [views/README.md](views/README.md)。

基础 Case 在末尾用 `view: ../../views/default.json` 引用通用的原生 Grafana 大盘 JSON，独立于 environment。执行时把这份 view 保存为结果目录下的 view.json，避免以后修改源文件改变已完成实验的展示。配置仍由配置加载器负责，view 不参与发压、调度或门禁判定。

```sh
uv run --locked mock-view /tmp/basic-result \
  --output /tmp/basic-view \
  --prometheus "$PROMETHEUS_BIN" \
  --grafana-home "$GRAFANA_HOME"
```

`GRAFANA_HOME` 是包含 bin/grafana、conf 和 public 的 Grafana 安装目录；输出目录必须是新的、位于实验目录之外的目录。命令先在回环地址提供回看页面并打印 URL，再启动独立的 Prometheus 与 Grafana。页面显示指标服务的启动状态，启动失败仍可查看配置和门禁。Ctrl+C 或 SIGTERM 统一回收页面服务和所拥有的子进程；页面关闭不等于停止服务。端口冲突直接报错，不连接其他查看会话或已有服务。

查看服务复制已完成实验的 TSDB，原始结果与数据不被重新打开写入；它不启动任何 Engine/Master/Client、不重新执行检查。多个查看会话使用不同输出目录和端口，可以独立查看同一份实验。

一个页面展示原始 Case、完整配置/启动配方、组件生效配置、代码版本/工作区差异、Phase/Step 时间线、保存的门禁，以及 Grafana 在线曲线。默认大盘所有 row 折叠，Prefill/Decode 分区并显式过滤各自 role；唯一线上参考是 RTP-LLM 大盘（包括其中的 FlexLB 分区）。当前已上报的 Mock 观测与 real 待埋点面板分开说明，ViT 和共享 JVM 的宿主资源缺口不伪造数据。Client 的 TTFT/请求耗时不能冒充 Engine 延迟。完整来源与覆盖见 views/coverage.md。每实例曲线保留 instance/role 等原有标签，支持 Grafana 自己的缩放与曲线选择。

点击门禁卡片定位其保存的指标窗口；执行侧 metric_window 事件记录真正使用的起止时间，展示层不根据当前实现重新计算。原始 PromQL 与非指标检查只标记查询点，图表显示附近上下文，不解析表达式推导窗口。每个 Step 只导入一个图表标注：普通操作标在开始时刻；有门禁时使用查询点或已记录的查询窗口，并在注释中展示接口、判定、预期条件、实际值（含单位）和错误。Phase 和 Step 的完整开始/结束记录仍保留在执行时间线。

事件线默认全部显示。在 Case 引用的 View JSON 顶层可加 `"annotation_steps": ["traffic", "kill_a", "kill_b"]` 指定默认选中的 Step，空数组表示默认全部隐藏；省略表示全选。它只属于展示配置，不扩展 Step，也不改变门禁卡片或执行记录。查看时校验重复和未知 ID；错误只影响指标展示，不改变实验结果。页面“事件线”可以逐项勾选、全部显示或全部隐藏，独立 Grafana 页面也有对应的 Step 多选变量。筛选只改变查询标签，所有标注仍然保留，不重启指标进程、不重跑实验。曲线在线查询保留的 TSDB，不固化进 HTML。

实验判定、产物完整性和查看服务状态分别展示：

- result.json 是核心记录，必须含有效的 status、带时区的 started_at/finished_at；缺失或损坏时命令报告文件和字段，拒绝猜测实验结果。只查看已完成实验。
- 配置、代码版本、接口元数据、采集记录和事件/门禁日志属于附加记录。缺失或损坏时显示具体文件；JSONL 中损坏的行标出行号，保留其他有效行。页面说明记录不完整，不把缺失门禁当作通过。没有配置也能按保存的 step 名展示门禁。
- TSDB、view.json 或依赖程序不可用，或指标进程启动失败/运行中退出时，指标区域说明原因，配置、门禁和时间线仍可读。已启动的指标进程统一回收，不自动重试。完整异常和进程 stdout/stderr 留在查看目录；修复后重新运行查看命令，使用新输出目录。
- ERROR/FAIL/CANCELLED、清理错误、采集不完整以及缺失指标均如实显示，缺失数据不补零。即使查看服务故障，也不改写实验的 PASS/FAIL/ERROR/CANCELLED。

实现分为 CLI（view/__main__.py）、服务生命周期与固定只读 HTTP 路由（view/service.py）、产物读取（view/records.py）、Grafana 配置与事件导入（view/grafana.py）、页面（view/page.html）。进程生命周期直接复用现有 ProcessSupervisor，不复制进程管理实现；Runner 只保存实验事实和 view 快照，不依赖展示层。

查看只读取指定的一次实验，不增加实验索引、实例历史台账或运行中控制。页面只轮询查看服务自身的指标服务状态，不轮询 Case 接口或重新判定门禁。自定义 view 继续使用原生 Grafana JSON；复用线上大盘时，需要把查询和标签改为本地实际存在的指标，不自动兼容线上 VIP 或 KMonitor。

## Case 配置和编译

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

每个 Case 独立目录组织自己的配置，数据集可共享。私有相对路径基于 Case 目录；显式提供共享根时，`data/`、`datasets/`、`views/` 基于共享根。编译不启动进程、不查询指标，也不检查数据内容。Client 只接受 start 参数 requests_file 指向的标准 Protobuf＋Zstd 文件，不提供运行时旧格式兼容。`scripts/convert_lineage.py` 仅用于离线转换，复用仓库的 lineage 解码和输出采样规则，将完整 token 序列、精确长度、前缀关系和原始相对到达时间写成当前 Protobuf＋Zstd 格式，并生成来源、摘要和转换参数 manifest；运行时不加载旧 codec。

顶层为 `id`、可选 `name`、`describe`、`environment`、`phases` 和可选 `view`。phase 和 step 完全顺序执行；每个 step 直接写 `interface` 与 `args`，可选 `expect`、`within` 和 `wait_after`；不再使用 call/check 外壳或 run 字段。phase 必须有正的 `timeout`。ID 全局唯一，引用必须来自前置步骤或允许的全局发现字段。

固定等待写成 step 的 `wait_after: 5s`，默认 `0s`，必须是非负的带单位时长。执行顺序为“调用接口 → 检查 expect → wait_after → 下一步”；最后一个 step 也执行等待。等待计入该 step 的 within 和 phase 的 timeout，期间继续监视组件退出并响应取消；调用或断言失败不执行等待。返回值和 ref 仍是原调用结果，等待不会刷新结果或查询时间；等待后的检查需要下一次接口调用。

不再登记 `runtime/wait`，旧 Case 的等待步骤应迁到前一个业务 step 的 `wait_after`，不保留兼容入口或只等待的空 step。接口成功后等待超时/中断，记录保留其 result 和 `error_stage: wait_after`，不能把已知成功的写操作标成 outcome_unknown，也不会重试动作。

组件配置分为 `deployment` 和 `config`。原生业务配置原样保留；Mock 点分键在本层展开成对象，不展开原生 JSON 文本。重复键、YAML merge、循环引用、点分路径冲突均报错。`FLEXLB_CONFIG` 保留 JSON 字符串，编译器只检查它是对象且无重复键，业务字段由 Master 加载器检查。

Client 的 `deployment.replicas` 表示 JVM 数量，不能把总 QPS 乘以副本数。当前启动器不落实 CPU、内存或 heap 限制，配置这些限制会拒绝运行。配置编译通过不代表资源限制或其他未接通能力已执行。

下面的读取保存 Leader 地址列表，后续步骤直接引用整个返回值，不另造句柄或 selector：

```yaml
phases:
  - id: observe
    describe: 检查发现信息
    timeout: 10s
    steps:
      - id: leader
        describe: 检查唯一主 Master
        interface: master/leadership/get
        expect: {count: 1}
        wait_after: 1s
      - id: leader_engines
        describe: 记录主 Master 的 Prefill 发现结果
        interface: master/engine/get
        args:
          targets: {ref: steps.leader}
          role: prefill
      - id: tps
        describe: 检查所有 Prefill 的计算吞吐
        interface: engine/prefill/tps
        args:
          targets: {ref: globals.prefill}
          range: [-60s, 0s]
        expect: {gte: 1}
```

省略 targets 等价于该操作域全部实例；显式空列表、重复地址或未知地址报错。全局发现信息在 step 开始时取值；已保存的 step 返回值保持原样，不跟着发现变化自动改写。Client 控制接口单独约定：start 作用于全部已配置 Client，stop 无参数；指标接口仍可指定 targets。

### 查询和断言

普通指标优先使用语义接口，YAML 不写 PromQL。当前支持 `latest` 和相对该 step 固定查询时刻的区间 `[-60s, 0s]`；`client/success_rate` 必须显式提供区间，不接受 latest 或 all，也不隐式切换为整轮累计成功率。其他指标的时间语义保持各自定义；`all` 的跨轮次查询窗口尚未实现，目前拒绝。

`latest` 使用前 60 秒内最近两次样本的 irate；区间使用 increase。执行器固定 `query_time`，指标实现记录实际查询点、表达式和窗口，不重新取墙钟。`expect` 支持 eq、列表 count 或数值 gte/lte。eq 是结构相等，列表顺序参与比较；实例查询返回排序后的地址列表。普通数值直接返回 float，不包装为 `{value, unit}`；单位在接口定义里。逐 Master 状态返回 `{master服务地址: [engine服务地址, ...]}`。

原始 `prometheus/query` 只接受 promql，在固定 query_time 执行一次 instant query；历史窗口写进表达式。返回 `{samples: [{labels, value, sample_time}], query_time}`，不自动把它压成数值或对全部序列应用同一阈值。空 vector 保留为空；业务数值门禁使用语义接口，缺数据报错而不是补零。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。Master/Client 按 replicas 启动独立进程；Engine 的 replicas 表示逻辑实例数，编号 `<组件>-<序号>`，由 engine_mode 决定放置；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。配方不再包含 bindings；目标选择和查询聚合由 Python 接口函数完成。

真实 Master 配方需要顶层 `zookeeper`；P/D/V 使用 `java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
engine_mode: shared  # shared（默认）：一个 Engine JVM；isolated：每实例一个 JVM
zookeeper:
  command: [<Java21>, -cp, <Master的classpath>, org.flexlb.mock.master.ZookeeperNode]
components:
  master:
    command: [<Java21>, -cp, <Master的classpath>, org.flexlb.MasterNode]
    startup_timeout: 60s
  prefill:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
  decode:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
  vit:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
```

配方必须覆盖 Case 实际声明的组件。组件通过 `MOCK_CONFIG_FILE` 读取本实例配置和完整 environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR` 提供身份和目录。初始化并绑定端口后原子发布 `MOCK_READY_FILE`，启动超时或提前退出都失败并回收已有进程。

统一 Engine JAR 的默认入口为 `EngineMain`，只启动配置指定的一个 P/D/V Engine，不启动管理接口。Whale 每个 Pod 用 `java -jar <Engine JAR>` 运行一个实例，生命周期由 Whale 管理。

本地 Runner 显式用 `java -cp <Engine JAR> org.flexlb.mock.host.EngineHost` 启动宿主；同一个宿主管理 P/D/V，不为 V 单设管理者。共享模式下 P/D/V 的 command、cwd、env 和 startup_timeout 必须一致；角色差异写入各自 config，不能用进程 env 区分同 JVM 的实例。宿主读取 `MOCK_COMPONENTS_DIR`，按实例目录读取配置并创建对象。Runner 的 isolated 模式仍使用每 JVM 一个对象的本地宿主，便于验证统一扩缩容协议；这不是 Whale 的启动方式。两种入口共用实例装配，RtpNode/VitNode 不再各自提供进程入口或可执行 JAR。

宿主的 `host/create`、`host/stop`、`host/status` 是内部原子接口，不接收集群副本数。`create` 接收 Python 分配的实例 id、rpcPort 和 controlPort；metrics 与该实例的 control 共用 HTTP 监听端口。宿主控制端口通过 `MOCK_HOST_PORT` 指定。Python 在启动监听前选择本次实验未使用的端口，Java 绑定指定端口，Runner 再核对返回地址与分配一致。端口探测不构成预留，若被其他进程抢占，启动失败并结束实验，不偷偷换端口；不能先启动服务再检查是否复用了旧地址。Python `execution/engines.py` 管理放置和实例归属，`interfaces/engines.py` 提供两项 Case 操作：

| 接口 | 参数和回执 |
| --- | --- |
| `engine/scale_out` | role 默认 prefill；count 默认 1、范围 1–1024。复用该角色的 Case config，返回新服务地址列表。角色必须已配置。 |
| `engine/scale_in` | role 默认 prefill；targets 省略时选择该角色全部活动实例，count 省略时关闭整个候选集。count 提供时按候选顺序只选一次；显式 targets 保持输入顺序，否则按创建顺序。返回实际关闭的服务地址列表。 |

空集合、重复/失效地址和超过候选数量的 count 在任何停止前报错。操作成功表示实例与发现文件更新完成，不表示 Master 已传播更新；在后续 step 读取各 Master 的实际观察。部分执行后失败会保留已知事件并结束实验，不重试或伪造整体回滚。

```yaml
- id: added
  describe: 增加一个Decode
  interface: engine/scale_out
  args: {role: decode, count: 1}
- id: removed
  describe: 关闭刚增加的实例
  interface: engine/scale_in
  args:
    role: decode
    targets: {ref: steps.added}
```

主动下线会终止该实例的在途请求；生产 Master 淘汰旧地址前仍可能有路由失败，不能将当前操作称为无损缩容。Case 应按场景判断过渡期错误和收敛后的成功率。生命周期验收持续记录请求结果，验证两种 JVM 放置下剩余实例继续服务、真实 Master 最终移除旧地址、新实例能接请求，以及历史指标仍可查询。另用故意保留旧 Prometheus target 的场景检查采集不会串到新实例，并注入第二个创建/停止失败，检查已完成操作不重放、整次实验报错并清理。

共享宿主关闭最后一个 Engine 后仍保留，允许再次扩容；独立宿主在关闭实例后回收进程。每次新增使用新编号、目录和未用过的监听地址。编号表示一次创建，例如删除 prefill-0 后再次扩容会创建 prefill-2，不填补旧编号；创建失败也不回收编号。它只用于内部对象映射和产物路径，Case 仍使用服务地址，不增加 node/generation 标签或选择器。实验目录和实例目录拒绝覆盖；Java 以 CREATE_NEW 写入 effective-config.json，另一宿主也不能重新使用旧实例目录。重新创建必须使用新编号与新目录，Runner 断点恢复、Engine 同身份原地重启不在当前能力范围。`hosts/engine-shared/` 或 `hosts/engine-<实例>/` 保存 JVM 日志，`components/<实例>/` 保存 config、effective-config、endpoints 和 ready，不复制宿主日志到各实例。下线删除 ready；endpoints、日志和配置作为该组件的原始产物保留，不另行维护实例历史。

`topology.json` 只保存当前活动实例，不生成 topology-history.json，不保存实例起止时间或退役台账。活动 Master 的 discovery.json 随当前 Engine 更新，角色缩到零时写空地址列表。Master 查询直接返回生产 WorkerDirectory 当前观察到的地址，不经实例 ID 或历史地址表转换；因此生产发现尚未淘汰的旧地址仍会如实返回。消费者的刷新和淘汰由自身实现负责，Case 检查其收敛结果。最终清理清空当前拓扑。

Prometheus 对每个实例独立抓取，以 RPC 地址作为 instance；宿主不重复输出业务指标。新增实例先完成采集基线，下线前完成末次抓取再移出目标；不保留退役目标、实例生命期或退役实例的末次采样台账。语义查询按当前活动实例选择范围并检查采样覆盖；时间窗只限定指标数据，不改变成员选择。原始 PromQL 直接查询 Prometheus 已采集的数据。

Runner 每秒开始一轮宿主健康检查，单次 HTTP 调用限时 10 秒，检查实际成员与 Engine 状态；进程退出另行实时监视。操作期间用同一管理锁隔离成员变更与健康快照。共享宿主意外退出记录受影响实例并使整个实验 ERROR；预期停止仅登记将回收的独立宿主，不屏蔽共享 JVM 意外退出。Master 始终单独 spawn，不进入 EngineHost。

Python 运行环境持续存活；`ProcessSupervisor` 持有进程并监视退出，各角色服务负责自己的生命周期，`RunEnvironment` 统一安排最终清理。进程监督器不识别 Master 业务接口，也不合并返回值。真实 Master 的状态查询由 `interfaces/master.py` 按 targets 逐实例调用，并保留每个 Master 的独立结果。

`master/stop(targets)` 由 Python 停止选定的独立 Master JVM，省略 targets 时停止全部活动 Master；空集合、重复或失效地址在任何停止前报错。停止前完成末次采样、登记预期退出，确认退出后移出活动发现与 Prometheus 目标，删除 ready，保留原 endpoints、日志和配置。默认 `mode: terminate` 使用正常停止；若需要 SIGKILL 兜底仍报错。显式 `mode: kill` 直接杀掉选定进程组并等待回收，只有这次退出视为预期故障注入；其他实例退出、操作前已退出及部分停止失败仍报错，不重放已完成动作。

`master/start(count=1)` 的 count 范围为 1–1024，从 environment.master 的配置创建新 JVM，复用本实验的 ZK、HIPPO_ROLE 和公共服务端口，使用递增编号、新回环 IP、新产物目录。控制 HTTP 也绑定新 IP，防止旧控制引用命中新进程。初始启动和动态启动共用同一路径；补回时允许某个 Engine 角色已缩到零，按当前空发现启动。返回新 RPC 地址列表，选举和采集基线完成才返回。这里的 start 是新增/补回，不恢复旧身份。旧 step 引用始终指向旧地址，不能自动替换成新地址。Java 不提供 Master 集群生命周期服务。

`master/restart(targets)` 恢复本次实验中已停止的 Master，省略 targets 时恢复全部已停止 Master；活动、未知、重复或显式空目标拒绝执行。恢复保留原服务地址、逻辑身份和选举组，启动新进程，当前拓扑与采集重新注册。每次启动使用独立进程监督身份和 `components/<实例>/restart-<序号>/` 目录；原配置、endpoints 和日志不覆盖。管理器只在内存保留可恢复实例的配置与当前启动信息，不增加实例历史文件。`master/start` 仍表示新建实例，旧地址引用不会自动指向它。

滚动替换直接组合查询、stop、start。下面使用固定观察等待，实际 Case 应按需要配置等待时间及门禁；新主产生不保证在途请求无损：

```yaml
- id: old_leader
  describe: 查询当前主节点
  interface: master/leadership/get
  expect: {count: 1}
- id: stopped
  describe: 停止原主 JVM
  interface: master/stop
  args: {targets: {ref: steps.old_leader}}
  wait_after: 10s
- id: elected
  describe: 检查剩余实例产生唯一新主
  interface: master/leadership/get
  expect: {count: 1}
- id: added
  describe: 补回一个独立 Master JVM
  interface: master/start
  args: {count: 1}
  within: 60s
```

Engine 独立宿主在对象关闭后登记预期退出并回收；其他运行中退出仍是异常。Master 的 `/hook/pre_stop` 仅是生产生命周期钩子，不等于 JVM 已停止。

所有步骤共享所在 phase 的单调时钟预算，step 的 within 只能缩短它。Python 在引用解析后校验参数，返回后校验结果；Java 内部 HTTP 协议另有独立校验与剩余预算。写操作不自动重试，调用开始后的异常或超时记录 outcome_unknown；回执已返回而 expect 不满足记 FAIL，不记结果未知。

进程退出、调用异常、空指标结果或门禁失败都会中止后续步骤并收尾。收尾对本次拥有的进程组先 TERM 后 KILL；主动取消也清理。结果为 PASS/FAIL/ERROR/CANCELLED，清理或采集异常不能得到 PASS。

运行目录不覆盖已有目录，保存原始 Case、编译配置、启动配方、接口清单、代码版本和工作区差异、组件配置和日志、phase/step 时间线、门禁查询时刻/真实值/阈值/结果及实际 PromQL。Client 逐请求记录和离线读取已接入，Engine 全链路记录尚未实现。测试 TransportProbe 只验证进程与传输，不替代业务实现。

## 真实 Master 适配

`org.flexlb.MasterNode` 与生产 Application、同步器和调度器在同一 JVM，不另起 Master 子进程，也不实现 Mock 调度器。Case 的 `environment.master.config` 使用原生环境变量字符串：`FLEXLB_CONFIG` 是业务 JSON，`MODEL_SERVICE_CONFIG` 是路由 JSON。

Runner 先启动独占 ZK，再启动 Engine、Master 和 Client，与 YAML 书写顺序无关。已启动 Engine 的 `endpoints.json` 形成 topology/discovery；Runner 只替换 `MODEL_SERVICE_CONFIG` 的 hosts/discovery_file，保留 service_id、组和角色域名，不读取线上 VIP，不改写 `FLEXLB_CONFIG`。当前只支持一个 endpoint 组、分离角色和 gRPC；ViT 使用相同的发现和实例生命周期。

发现使用实际 Engine gRPC 端口，生产 WorkerAddressService 再转换内部 HTTP 身份，不假定 Mock 控制端口相邻。原始组件配置、实际环境、effective-config 和 discovery 分开保存；Master 暴露生产 `/prometheus`。

只绑定 `MasterStateApi` 的两个读取接口：

| 接口 | 行为 |
| --- | --- |
| `master/leadership/get` | 读取各 Master 的生产 isMaster 声明，返回声明自己为 Leader 的 Master 服务地址列表。数量由 expect 判断，不伪造唯一 Leader。 |
| `master/engine/get` | 读取各 Master 的 WorkerDirectory，按角色映射 Engine 服务地址，逐 Master 返回。未知地址报错，不能按部署数量补齐或将各节点取并集。 |

单/多 Master 全部使用生产 ZK 选举，不提供 standalone 分支。独立 ZookeeperNode 使用已有依赖、独立 JVM/数据目录；Runner 分配地址和本次 HIPPO_ROLE，运行时同步配置为 needConsistency=true、会话超时 6s，并保存实际值。ready 必须等到真实选举产生 Leader。

Master 使用 127.0.0.1、127.0.0.2 等地址，相同 HTTP 端口，gRPC=HTTP+2，适配生产以 IP 作为身份和共用转发端口的约束。端口启动时被抢占则失败，不静默换协议。Linux 默认支持 127/8；macOS 缺少别名时明确报错。实验之间不共享 ZK 或选举路径。清理先停止组件，再停止 ZK，最后停止 Prometheus，保留日志和数据。

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。`tests/test_master_lifecycle.py` 在两种模式下持续运行同一 Client 计划，停止主节点、补回新 IP 的 Master、再停止旧节点；在旧请求 deadline 之后检查新主持续成功接流，并检查独立进程退出、各组件独立产物和当前采集目标。切换错误如实保留，不要求整个过渡期零错误。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

## RTP 节点与公共校验

节点仅监听本机回环地址；每个实例独立绑定原生 gRPC、控制 HTTP 和指标端点，共 JVM 不合并这些端点。`endpoints.json` 写实际地址、支持的 RPC 和阶段范围；`/health` 只列已绑定的公共接口，当前 Engine 没有绑定 Case 查询接口，Python 查询直接访问 Prometheus。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      SEQ_SIZE_PER_BLOCK: "64"
      MAX_SEQ_LEN: "32768"
      MAX_BATCH_TOKENS_SIZE: "8192"
      REUSE_CACHE: "true"
      TEST_BLOCK_NUM: "1025"
      RESERVE_BLOCK_RATIO: "5"
      compute.forward.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持 server_args 小写或 Whale 大写 env 名，同一字段同时提供两种写法报错；整数/布尔接受对应原生文本，其余原生内容保留，但不能据此宣称功能已模拟。当前要求显式 block size、max sequence length 和 test_block_num，只支持 `dp_size=1`、`dp_rank=0`；P/D 批量上限取 1–1024，默认 1。Mock 一个 Engine 表示完整计算体，不拆 CP/TP/DP rank 或模拟 rank 间通信；内部并行的影响由整 Engine 的性能公式、容量和调度配置表达。block size 是本 Engine 的逻辑分配粒度，PD 不根据 CP 数乘除。配置导入时应先统一容量口径。不加载模型层/head 或 GPU 布局，不从部署内存推导 KV 块数。

RTP 请求公共支持范围集中在 `RtpRequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：多模态、adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 处理原生协议，RtpScheduler 管理资源和队列，只有 GENERATE 后进入计算队列；每个 Engine 一轮一个 Forward，按配置合并多个请求。初始 KV 分配遇临时不足等待容量变化；生成期间每轮先增长 KV，失败则终止，成功后才执行与提交输出。

P 的内部 `PrefillTransfer.start()` 注册活跃来源并复用 LocalPrefill 计算。Forward 完成后 KV 仍由请求持有；网络 `Read` 交付结束后仍保留来源，P 收到 D 的 LOAD ACK 后才完成计算阶段并释放。正式服务**只保留 Read RPC，没有外部 Prepare RPC**；测试从包内 start 建立来源或调用正式生成链路，不为探针注册额外协议。

D 的 LOAD 从 peer_addrs 对应 P 发起真实网络 Read，校验 client/request 身份、完整 prompt 和单 Reader 所有权；不能用缓存中存在的 key 冒充活跃来源。PD 按完整输入的 token 覆盖范围交接，返回元数据只有 input_tokens 和 prefill_finished（首 token 是否已经使请求结束），不传 block size、块编号或 cache keys。P/D 各自按本地 block size 分配和生成前缀 keys，两端块大小不要求相同或成倍数；最后不足一块的 token 也纳入交接。D 验证 token 数、min/max/ignore_eos 一致性、首 token 结束事实及预算后才能确认 LOAD，传入 KV 不算 D 历史命中，也不提前提交为缓存。不复制 KV 张量，不模拟 RDMA/TCP cache-store 或带宽。

Read 等待计算就绪；D 取消/超时会取消 P 来源，P 取消/关闭也使 LOAD 失败。`load_cache_timeout_ms` 默认 5s，非正值按原生回退 5s，同时受总 deadline 限制。没有 Reader 或 Read 入口校验失败时，来源仍由正式 P 请求所有者管理，受请求取消、timeout/deadline 和节点关闭约束。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0、文本 P/D 和 max_new_tokens>1。入队 ACK 前必须取得 D 分配确认；P 可在 Fetch 前计算并持有 KV，但不能先发 LOAD 或执行 D。Fetch 单次接入后使用相同输出链路，不缓存整段结果。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。incremental 返回新 token，否则返回累计输出；aux.output_len 始终累计。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。Mock 保护上限是 1024 活动请求，TaskHistory 保留 1024 项。旧游标越过历史窗口时，按 real 的恢复方式继续返回完整当前状态、仍保留的完成记录和推进后的游标，不以 OUT_OF_RANGE 阻断健康同步；超前游标不倒退。每次缺口读取在组件日志中记录 `finished_history_gap`，包含 requested_version、dropped_through、latest_version、retained，不能把这段完成记录当成完整请求历史。生产 proto 没有历史完整性字段，缺口证据保存在实验组件日志；需要完整逐请求证据时不能依赖 WorkerStatus 的有限窗口。real 当前为 1000 条/5s 保留，Mock 的 1024 条是明确的模拟边界，不宣称保留策略完全相同。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。D 当前要求 input+max_new_tokens<max_seq_len，未实现完整真实截断。显式投机按 RTP reserveStep 预留，成功且 reuse 与缓存层开启时写回完整 prompt 块；失败不提交，生成内容不额外建索引。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

共享 `BlockPool` 只管理物理容量和引用。每个实例/存储层各自建池，每个所有者持有独立 Reference，close 幂等，不能释放其他所有者份额或混用池。REQUEST/CACHE/LOAD/STORE/EVICTION 分别计数；只有 CACHE 引用的块算可回收容量，free 和 available 分开，分配失败不部分扣账，释放唤醒等待者。

`KvCache` 和 `BlockTree` 目前固定实现 RTP Full-KV 语义，文件保留在 `java/shared/cache`，不为未知的 vLLM 行为增加策略层或适配接口。每个 Engine 各自拥有 Device/Host 池、前缀索引和请求租约，同 JVM 不共享缓存内容。树内 `lru/lfu/fifo` 只改变合法候选的排序，Device 和 Host 可以分别配置。一次 match 用同一访问序号刷新整条命中路径，release/重叠 publish 不刷新；每次成功进入新存储层都会刷新该层的 FIFO 入层顺序。

正式 `acquire` 在同一锁内完成前缀匹配、取得使用引用、申请缺少的 Device 容量和本地读取；没有先查询再占用的 Scheduler 接口。状态/前缀快照只读，不保证后续请求仍然命中。本地 Host/Device 读取和转存瞬间完成，不建立本地 loading 队列、异步任务或带宽模型；PD 网络交接仍按原有生命周期执行。

对齐当前 real BlockTree：开启 Device cache 时，Host 读取成功后把已有有效前缀注册到 Device，再移除 Host 副本；关闭 Device cache 时，Host 源保留，Device 工作副本不注册。申请失败不能消费 Host。驱逐候选按所在层的直接子节点判断叶子，不要求整棵树的全局叶子；Full-KV 前缀被删除时，所有失效后缀跨层清理。Device 水位触发时向 Host 转存；Host 分配失败时保留 Device 源，不为了转存强制删除 Host 缓存，也不退化成丢弃 Device。未配置 Host 时水位回收可以直接删除 Device 索引。紧急 Device 分配允许直接回收 Device 缓存及失效后缀，不额外删除无关 Host 根。层间移动保持逻辑 key 和索引版本，真正插入/删除才改变 key 版本。Host 不计入 WorkerStatus/CacheStatus 的 Device 执行容量。

容量使用原生 `test_block_num` 和 `reserve_block_ratio`。Device 可用块数=test_block_num−1；Host 使用 `memory_cache_blocks` 个块。reserve 默认 5%，约束首次分配，运行中增长可以使用预留。需求超过单实例上限为永久不足，暂时被其他请求占用为可重试；增长一次补齐差额，失败不留下半次新增分配。容量不足后的等待/结束决定仍在 Scheduler，运行中的 RTP Decode 增长失败会结束该请求，不实现主动换出。显式投机的首次 D 分配也包含 reserveStep，在 PD 交接前准备目标容量。

容量失败处理沿用上述边界：首次 `acquire` 返回 RETRYABLE 时，P 留在 waiting、D 留在容量等待阶段；同一容量版本不重复申请，资源释放后唤醒重试，并保留请求原 deadline。PERMANENT 立即结束单请求，返回容量错误 602；等待中超时返回 603，取消和超时后不再分配。运行中 Decode 的 `ensureCapacity` 无论暂时还是永久不足，都结束该请求，等待其 Compute/PD 使用归零后释放引用，其他请求继续；释放的容量可供等待请求重新准入。不增加抢占、请求换出或重新排队重算接口。

Host 转存目标不足由 KV 内部保留源，不直接上升为请求失败。成功终态的 `publish` 抛异常时，RequestState 将请求改为内部错误，并在 finally 中关闭租约；失败/取消不发布。`acquire/ensureCapacity` 的预期容量失败必须以 Failure 返回，未预期的内部异常按 Engine 故障停止处理，不能伪装成容量不足继续运行。

Scheduler 与 KV 管理器按 Token 交互：`acquire(requiredTokens, keys)` 首次取得资源，`Lease.ensureCapacity(requiredTokens)` 保证本轮容量，`reusedTokens()` 返回复用 Token 数。容量的 Block 换算和向上取整封装在 KV 管理器中；已有容量足够或需求变小时不缩容、不重复申请，超过容量上限返回永久不足。容量配置、物理池和状态快照仍使用原生 Block 口径。

不再每轮调用 markValid，也不在 Lease 中重复保存计算进度。Scheduler 在成功终态且所有 Compute/PD 使用结束后调用 `publish(keys, validTokens)`，一次声明有效范围并注册完整 prompt 块；KV 管理器校验有效范围未超过已分配容量、key 数未超过完整有效块数。`publish` 不关闭租约，`close` 只释放请求引用。投机拒绝区域、预留空间、失败/取消请求不发布，生成内容暂不额外建立前缀索引。多个请求可共享已注册的不可变 Device 块。缓存索引引用与请求物理引用独立：索引可以在请求运行中注销或迁往 Host，请求仍保留原 Device 块，最后一个引用释放才归还物理容量。acquire/publish 仅在同步事务内临时保护完整前缀。状态中的 referenced 只统计仍同时被索引和请求引用的 Device 物理块；注销或转存后仍被请求持有的块计入 held。Host-only 的私有 Device 副本与 Host 记作两层真实占用。

Device/Host 水位按物理占用比例工作：达到 high 后，尽量回收到 low，未达到 low 时保留触发状态。加载、发布、增长和请求释放完成后同步检查，顺序为 Host、Device，不建后台扫描线程；运行中请求的引用可能使物理占用暂时无法下降。`enable_memory_cache` 要求显式正数 `memory_cache_blocks`。Disk/Remote 仍拒绝开启。Host 传输超时保留在 pending_config，本地瞬间完成不使用该超时。Host-only 写回暂保留可容纳的部分前缀，尚未对齐 real 的整批目标预分配与整批失败回滚；Device+Host 主路径不依赖此简化。异步传输队列、Disk/Remote 和 Hybrid/SWA/Linear 布局仍不实现。

## 通用 V 与输入准备

Engine 是运行实例的概念，prefill/decode/vit 是 Engine 的角色；RTP/vLLM 是 Engine 后端，不能与 P/D/V 混为一层。当前配置中的 `component.engine` 字段选择 P/D 后端；V 统一实现于 `java/shared/vit`，不需要选择后端，也不与 `java/engine/rtp` 并列为后端。以后 P/D 使用哪个后端都复用这一份 V。V 的媒体协议与 TCP 客户端仍在 `java/shared/multimodal`，Client/P 只依赖该协议模块。

P/D/V 都实现 `EngineInstance` 生命周期句柄，各自拥有端口、队列、缓存和产物目录；V 拥有自己的调度器。本地同一个 `EngineHost` 平等管理三种 Engine；Whale 每 Pod 直接启动一个 Engine，无需 EngineHost。统一产物是 `java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`，同时包含通用单实例入口和本地宿主入口，不为 V 单独打包。Master 始终独立 JVM。

原生 `RemoteMultimodalEmbedding` 的一个请求可含多张图，并发请求由 V 调度器合成 batch，无额外 Batch RPC。`gpu_max_batch_size=1` 表示单请求执行并忽略收集窗口；否则按请求数、图数和收集窗口选 FIFO batch。队列和 RPC 等待者均有上限，超出返回 RESOURCE_EXHAUSTED。Client/P 只依赖 shared/multimodal，不依赖 V 服务模块。首版不拆分一个请求的图片组，超过单批图数预算明确拒绝。取消/超时仅释放当前等待者；同 key 的计算可被多个请求共享，无消费者且尚未执行的工作不再 forward，已开始的 batch 可以完成并登记缓存。

`environment.vit` 不指定 engine 身份，配置示例：

```yaml
vit:
  deployment:
    replicas: 1
  config:
    patch_size: 14
    spatial_merge_size: 2
    hidden_size: 4096
    gpu_max_batch_size: 8
    gpu_max_batch_images: 200
    gpu_batch_wait_ms: 10
    mm_max_queue_size: 1024
    mm_cache_gpu_max_bytes: 67108864
    mm_cache_cpu_max_bytes: 268435456
    mm_hash_key_cache_max_bytes: 16777216
    compute.forward.expression: "2 + 0.00001 * totalImagePixels"
```

图像输入携带明确的 width/height（原生 mm_preprocess_config），不下载/解码实际图片。输出行数为 ceil(width/(patch_size*spatial_merge_size))*ceil(height/(patch_size*spatial_merge_size))。这是可配置的 Mock 几何模型，不宣称复现某个真实视觉模型的 resize/crop。首版只接受 IMAGE URL 身份和显式尺寸；视频、音频、tensor 输入、V4.1 prepared patches 在 V 模块边界明确拒绝，不由 Client 或 P 的核心校验器按多模态类别拒绝。数据文件的 Image.length 要事先按模型几何生成，旧数据仅离线转换，不新增运行时兼容格式。

V 以 miss 的图片组成 ForwardBatch，复用公式和 SimulatedCompute。计算完成后登记缓存、通知等待者；Compute 不直接改变缓存。命中和在途合并不重复计入 forward。提供 real 名称的 py_rtp_vit_embedding_forward_rt、batch_size、cache 指标及 mock_vit_theoretical_forward_rt；元数据模拟不能代替真实模型计时校准。

Embedding Device/Host 按逻辑 rows*hidden_size*BF16(2 bytes) 计费，各自 LRU；Device 淘汰转存 Host，Host 淘汰移除 embedding。Hash 独立按行数和 key 字节计费，因此 hash_hit 与 embedding_hit 不等价。在途计算使用独立 claim，失败移除，旧 claim 的迟到完成不能覆盖新 claim。消费者持有不可变结果，不依赖缓存继续驻留。缓存 key 对齐 Linux real MultimodalInput 的 URL/type/preprocess 身份，hash 内容是稳定模拟值，不宣称与实际 tensor 的 hash 相等。

Client 按 Master 的 V-only route 选 V，读取元数据、准备特征 token，再带 selected_vit 请求完整 P/D 路由；Master 切换使 pin 失效时重新取一次 V 路由。P 在输入准备完成后才把规范化 token 序列交给 KV/Scheduler，并用同一输入准备 D ALLOCATE 和 PD 交接。P 的输入准备发生在调度锁外，受原请求 deadline/取消约束；这段准备暂不计入 P Scheduler 的队列快照，但计入 Client 端到端耗时。D 接收已准备输入，不重复编码。

本地单 token 的输入准备由 RtpRpcService 负责，P/D 单请求和 Master batch 由 PrefillGeneration 负责；LocalPrefill 的单个与批量提交只接受已准备输入，拒绝尚带媒体内容的请求，不调用 V。本地入口保留准备前的请求开始时间；各路径沿用原请求 deadline，不在准备完成后重置超时预算。

VP 同 JVM 和跨进程都走 TCP gRPC，只传 shape/dtype、split_size 和逐行 INT32 hash，不生成 embedding payload，也不伪造 RDMA lease。Mock P 校验形状、长度和展开 span 后使用；此响应不能交给真实 tensor 解码器，VP 网络带宽和 GPU 内存复制不在模拟范围内。

Whale 的 P/D/V 统一使用 `java -jar java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`，无需 Python Runner、`MOCK_HOST_PORT`、`MOCK_COMPONENTS_DIR` 或宿主管理服务。入口 `EngineMain` 从 `MOCK_CONFIG_FILE` 中的 component_role 决定启动哪个 Engine；设置 MOCK_OUTPUT_DIR、MOCK_READY_FILE，以及固定 MOCK_RPC_PORT、MOCK_CONTROL_PORT。实例配置采用与 Runner 相同的 id/component_role/component.config 结构，P/D 另外提供 component.engine。通过 MOCK_BIND_HOST=0.0.0.0 和 MOCK_ADVERTISE_HOST=PodIP 区分监听和发现地址。每个实例独立写入 effective-config、endpoints 和 ready；关闭删除 ready，旧产物目录拒绝覆盖。共享 JVM 的实例沿用同一进程地址，各自端口不同。无论是否共 JVM，VP/PD 都走 TCP、只传元数据。

## Compute、Forward、Decode 和 EOS

每个组件只配置一份 Compute；Forward、Decode 进度和 EOS 都归 Compute，不再配置组级 decode/generation，也不保留旧 compute.expression 入口。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      compute:
        forward:
          expression: "1.23 * (1 + 0.01*totalComputeTokens)"
        eos:
          probability: 0.0025
          seed: 20260922
  decode:
    engine: rtp
    config:
      compute:
        forward:
          expression: "19.5 + 0.175*batchSize"
        decode:
          tokens_per_step: 2.6
        eos:
          probability: 0.0025
          seed: 20260922
```

RtpConfig 只加载 ComputeSpec；ComputeModel 组合 ForwardModel、可选噪声、DecodeStepModel 和 StoppingModel。非默认 Decode 进度参数只允许配置在 D，P/D 都可以配置 EOS。旧字段、未知字段、null、错误类型或缺少必要 seed 明确拒绝，没有别名和兼容分支。模型校验来自 Java，Python 只编译组件配置结构。effective-config 顶层保存生效值，requested_config 保存请求配置，pending_config 单列仍未实现的原生参数。

调度前 Compute.prepare 固定请求快照和本轮验证量/KV 需求，不抽投机接受量。Forward 完成后 Compute.completed 依次计算接受后的候选产出、EOS 和长度截断，返回 committedTokens/finished。Scheduler 只校验请求仍有效并提交结果、更新队列、处理资源和 PD；不判断正常结束是 EOS 还是达到上限。取消、超时、关闭和计算失败仍走独立失败路径，不能成为正常 finished，也不能因旧回调再提交 token。

P 每轮只产首 token，finished 表示整个请求是否结束，不表示 Prefill 阶段是否完成。P 阶段完成后退出计算集合，保留传输需要的 KV；即使首 token EOS，也仍完成 LOAD 确认，再通过 PREFILL_COMPLETE 让 D 结束，不启动 D Forward。PD 不再传预测的 stop_after，不由 P 决定 D 的未来停止长度。

ForwardBatch 保留请求长度、实际缓存命中、已生成长度、stepTokens、优先级和图片 shape。ForwardModel 只计算 Duration，SimulatedCompute 只执行可取消等待；公共 Compute 的产出计算不修改请求或队列。一个 Engine 同时至多一个 Forward，完成结果只提交一次。

公式复用生产 ArithmeticFormula 的 param(name, value)、sum 等函数，单位 ms，保留到 ns；非法变量、负值、非有限值和溢出拒绝。逐请求变量只在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。RTP 额外绑定 cpSize、cpEnabled（0/1）、tpSize、dpSize，只有显式引用才参与计算，不自动缩放已有拟合公式。默认 FormulaForward 不猜测多个 DP 的组合方式。

倍率直接乘入 forward.expression，不提供独立 scale/noise.scale。噪声位于 forward.noise，默认 stddev_ms/stddev_ratio=0；启用时要求 seed 和正 max_deviation_ms，扰动有界、最终耗时不小于 0。seed、Engine 身份、角色和 batch 身份固定一次耗时样本，取消批次不补完成指标。当前 DeepSeek Case 不启用噪声。

Decode 有两种互斥模型。默认普通模式每轮一个 token；平均模式 tokens_per_step>=1 按 ceil((n+1)*v)−ceil(n*v) 产生候选数量，每轮每请求计一个 target 输入，不虚构 draft/accept 统计。显式投机模式配置 draft_tokens、acceptance_rate、seed：抽取接受前缀 A，候选产出为 A+1；固定独立接受率 α、草稿长 k 时 E[C]=1+α+…+α^k。这是可复现的统计近似，不模拟真实 MTP 的 logits、draft 张量和独立 draft KV；P 的初始化成本包含在 P 整体公式中。

长度边界在接受结果后统一处理。显式投机模式按固定 k 个草稿、k+1 个验证位置及 seq_len+k+1 的 KV 预留执行（普通 Decode reserveStep=0），对齐当前 RTP 的固定验证窗口；不能因为只剩 R 个输出位置就提前缩小验证量。候选产出 5 个而只剩 4 个时，只提交 4 个并返回 finished=true，验证成本仍按完整计划累计。平均进度模型每轮仍计一个 target 输入，KV 仅覆盖最终可能提交的输出前缀。EOS 只检查接受后的候选 token，按绝对输出位置生成固定概率事件；相同 seed、请求身份和概率下，改变 batch、轮宽、接受模型或调用顺序不会重抽已经走过的位置。拒绝的草稿不参与结束判断。

StoppingModel 遵守 min_new_tokens，ignore_eos 只屏蔽 EOS，不屏蔽长度上限。完成判定的上限同时受 max_new_tokens 与 maxSequence−inputTokens 约束；RTP 接入仍保留前述较严格的序列准入规则；显式投机在首次分配和后续增长中另外预留 reserveStep。EOS 位于本轮中间时截断实际产出，但本轮已执行的验证成本和在途资源引用不撤销。固定概率只是 Mock EOS hazard，不宣称真实输出分布。标准回放发送 ignore_eos=true，所以 Case 可保留 P/D 的 EOS 参数而不再次缩短数据中的目标长度。

real 依据：MtpExecutor::decodeStep 先 target verify Forward，再执行 speculative_sampler；MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo 提取接受后的 token，GenerateStream::specUpdate 经 NormalGenerateStream::updateOutput 检查 EOS/max；CompleteTokenIds::matchEosToken 截断到第一个 EOS。PrefillRpcServer::remoteLoadCacheEnd 保留 LOAD ACK、释放源 KV 和 PREFILL_COMPLETE 顺序。当前 Mock 不支持 stop words/beam search 等未实现条件，显式请求会拒绝。

## P/D 配置治理

组件只接受已登记、角色适用的原生参数；支持小写字段或对应大写环境变量名，两者同时出现、未知参数、已删除参数和错放角色的参数报错。Python 不复制这份字段表，RTP 的字段归属与待实现参数校验位于 RtpSettings，实际值解析位于 RtpConfig。没有静默透传一批未消费配置的入口。

CP/TP/DP 保留在执行配置及 effective-config 中，不转成来源说明。一个 Mock Engine 仍表示整体计算单元；不创建内部 rank，不模拟并行通信。当前执行域仍只支持 dp_size=1、dp_rank=0，其他值明确拒绝；保留 dp_size 并不表示已经支持内部多 DP。并行度作为公式输入，已校准的公式、容量和 block size 不再隐式换算。

下表是当前 DeepSeek HA 配置的处理结果。待实现参数保留实际数值、校验类型与范围，但只进入 pending_config，不表示行为已经对齐。Host 容量、策略和水位已经接通；开启尚不存在的 disk/remote cache 仍拒绝启动；其他待实现项允许保留并警告，相关调度、压力和超时结论不得作为已对齐结果。

| 参数 | P | D | 当前处理 |
| --- | --- | --- | --- |
| cp_size / cp_enabled | 4 / true | 1 / false | 保留，公式可读取；不模拟内部 CP。 |
| tp_size / dp_size | 4 / 1 | 1 / 1 | 保留；tp_size 还用于原生状态上报。 |
| max_seq_len | 1048576 | 1048576 | 生效。 |
| seq_size_per_block / test_block_num | 512 / 21554 | 128 / 221485 | 本地容量生效；PD 按 token 交接。 |
| reserve_block_ratio | 1 | 8 | 生效，百分比。 |
| max_generate_batch_size | 64 | 128 | P/D 统一入口，生效；删除 max_context_batch_size。 |
| max_batch_tokens_size | 512000 | 1048576 | P token/矩形准入生效；D 不作为每轮 verify token 上限。 |
| max_batch_tokens_without_cache | 50000 | 不配置 | P 普通队列按未缓存 token 配额准入；D 不配置。 |
| max_inited_kv_cache_streams | 128 | 不配置 | P 初始化 KV 数量限制和超大组退回条件均生效；D 不配置。 |
| reuse_cache / enable_device_cache / device_eviction_policy | false / true / lru | false / true / lru | 保留现有语义，不为接 Host 而打开 reuse。 |
| enable_memory_cache / memory_cache_blocks | true / 52295 | false / 不配置 | P Host 容量生效，本地瞬间周转。 |
| memory_eviction_policy | lru | 不配置 | P Host 策略/水位生效；D 不配置。 |
| block_tree_device_evict_low/high_watermark_ratio | 0.82 / 0.9 | 0.82 / 0.9 | Device 水位生效；释放资源时同步回收到低水位。 |
| block_tree_memory_evict_low/high_watermark_ratio | 0.9 / 0.94 | 不配置 | P Host 策略/水位生效；D 不配置。 |
| block_tree_business_queue_max_size / block_tree_transfer_queue_max_size | 10000 / 10000 | 10000 / 10000 | 待实现队列背压，不以无界队列冒充等价实现。 |
| memory_cache_sync_timeout_ms | 10000 | 不配置 | P 本地瞬间传输不使用，保留 pending_config。 |
| enable_disk_cache / enable_remote_cache | false / false | false / false | 明确关闭；开启拒绝。 |
| prefill_retry_times | 0 | 不配置 | 当前不重试；非零值列为待实现。 |
| prefill_max_wait_timeout_ms | 600000 | 不配置 | P 待实现独立等待期限。 |
| max_rpc_timeout_ms | 7200000 | 7200000 | 待实现独立 RPC 上限；已有请求总期限不因此失效。 |
| load_cache_timeout_ms | 不配置 | 5000 | D 生效；删除 P 无效项。 |
| compute | 1.23 * (原公式)，不配噪声 | 19.5+0.175*batchSize | 保留，不按 CP/TP 二次缩放。 |
| compute.eos | probability=0.0025，seed=20260922 | 同 P | 参数保留；固定长度回放通过 ignore_eos 屏蔽 EOS。 |
| compute.decode.tokens_per_step | 不配置 | 2.6 | D 平均进度模型，不等于真实投机参数。 |

以下参数从当前 Case 删除，不复刻 native 线程池、轮询和描述符分批：block_tree_transfer_worker_count、memory_cache_max_descriptors_per_transfer_batch、prefill_prepare_resource_pool_size、prefill_stop_stream_wait_timeout_ms、decode_retry_interval_ms。取消、资源释放和有界请求生命周期仍必须实现，不能随这些参数一起删除。

decode_retry_times、decode_retry_timeout_ms 在当前 real C++ 执行代码中没有读取处，不迁入 Mock。当前 P 的 cp_force_single_prefill=false、D 不执行对应 Prefill 准入，均省略；未来 P 启用该限制时只需增加逻辑调度约束。prefill_retry_timeout_ms 当前因重试次数为 0 省略，非零重试的次数与预算一起待实现。

## Client 原生收发核心

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。每次请求固定一份主备快照，先向主节点 Schedule，传输失败时最多向备节点尝试一次；收到业务响应、deadline 超时、主动取消或进入 Engine 取流后都不重发。要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D/V 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每个 Client JVM 独立运行 MasterDiscovery：每秒刷新 topology.json 并 POST 真实 `/rtp_llm/master/info`，单次心跳 500ms、最多 4 个并发探测；连续两次失败判不健康。对齐 real Frontend `server/host_service.py` 的选择规则：优先健康且声明自己为主的节点，否则保留仍健康的原目标，再按 IP 选择健康候选；备用节点从未声明为主的其他健康节点中选择。发现暂时漏掉的健康节点仍继续探测，失败后停止保留，不再发现且已不健康的记录在最后探测 30s 后清理。文件损坏或非法端点属于执行错误，会终止整个 Client 播放；没有健康 Master 是请求失败，不直连 Engine，也不按列表轮流广播。

发送边界对齐 `server/master_client.py`：只有 Schedule 的传输失败尝试一次备节点；业务拒绝、deadline 和取流失败不尝试其他 Master。整个请求仍受同一个总 deadline 约束，重试消耗剩余预算。这个策略不承诺 exactly-once：响应丢失后原主可能已经接收请求，保留与 real Frontend 相同的传输重试风险。Client QPS 仍按逻辑请求 issued 计一次，备节点尝试不会再发放流量预算。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；最终长度必须达到 min_new_tokens，ignore_eos 时必须等于 max_new_tokens，提前 finished 不能冒充固定长度回放成功；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向该请求最后尝试的 Master 尽力发一次有界 Cancel，不随新的选举结果改变取消目标；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。标准 Protobuf 输入的 Player、LoadClient、client0 协调和 start/stop 已接通，主备发送只改变 Schedule 目标选择与传输失败重试边界。Client 逐请求执行记录已接通；旧采集格式已有离线转换；Engine 与 Master 的完整服务端证据仍未接通。

## 指标、采集和查询

EngineMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。业务标签只保留 role；Prometheus 为每个采集目标设置 instance=实际 RPC 服务地址，KV 分池使用 pool_name=device/host、tier=DEVICE/HOST，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| rtp_llm_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| rtp_llm_framework_requests_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| rtp_llm_host_kv_tokens | Gauge | 本地 Host total/free token 容量，与 Device 分开。 |
| rtp_llm_cache_lookup_tokens_total / rtp_llm_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| rtp_llm_forward_tokens_total | Counter，kind=input/compute/output | 完成 Forward 的 token；input/compute 汇总本批全部执行成员，output 只计有效提交。全部取消的 Forward 不补完成样本。 |
| rtp_llm_model_forward_us | Gauge | 最近一个完成 Forward 的实际耗时，微秒；计时到执行完成通知，不包含等待调度域锁和结果提交。 |
| mock_model_forward_planned_us | Gauge | 同一个完成 Forward 的计划耗时，微秒，包含公式和可选噪声。 |
| rtp_llm_model_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| rtp_llm_scheduler_requests | Gauge，state=waiting/running/loading/ready | 调度器报告的四类互斥请求数。 |
| rtp_llm_kv_tokens | Gauge，state=total/available/free/held | token 容量，available 含可驱逐缓存，不等于 free。 |

RTP 原生观测直接导出：`rtp_llm_wait_stream_size`、`rtp_llm_running_stream_size`、`rtp_llm_kv_cache_left_seq`（Tokens）、`rtp_llm_kv_cache_{total,free,available,used}_blocks`（Blocks）和 `rtp_llm_kv_cache_used_ratio`（0–100，不可用比例）。Host 暂只上报 total/free/used Blocks，不用 free 冒充 available。loading/ready 仍是调度器阶段辅助观测，不冒充 real 的加载缓存队列。

Prefill 的 context TPS 用计算 Tokens／执行时间，含缓存 TPS 用输入 Tokens／执行时间；墙钟 TPS 用对应 Tokens／实际窗口。Decode generate TPS 统计提交的输出 Tokens，total TPS 统计模型计算 Tokens，二者在投机采样时不同。`rtp_llm_framework_qps` 统计终态，不是接纳；error 包含 timeout，cancel 单列。上述速率 Gauge 的窗口是两次采集间的单调时钟间隔，无后台采样线程；空闲窗口报告零。real 默认按后台周期报告，窗口边界可能不同，不能要求逐点相等。

没有 real 同名指标的真实观测，以 RTP Prometheus 辅助 Counter/Histogram 保留供 Python 门禁和统计使用，不拿它们的累计值直接冒充线上 QPS/TPS Gauge。理论 Forward、ViT 运行图片数和 hash 缓存预算保留 `mock_` 前缀。此次更名不保留旧 `mock_engine_*` 别名；旧实验保持原始 view.json 和 TSDB，不承诺新查询可以直接读取旧指标名。

P/D 已补 `rtp_llm_{context,generate}_batch_size`、执行 Token 数、最长序列、逐请求等待/首 Token/总延迟、输入/输出长度与迭代数；Decode 传输等待接入 `rtp_llm_stream_loading_cache_latency_us`。Query batch 的 native 语义是 beam 数，不能用调度批大小代替。

ViT 使用相同的物理元数据与 Prometheus instance 发现；QPS、Embedding 队列/等待/Forward/合批、请求图片数/输出长度，以及缓存命中/未命中/在途合并/淘汰和逻辑驻留量均按 real 名称导出。毫秒指标不转换成微秒名，未实现的下载、预处理子进程和 CUDA Graph 不补零。详见 [指标覆盖与边界](views/README.md)。

新增事件 Gauge 为采集窗口均值，无事件的窗口不发耗时样本；其 `_sum` / `_count` 辅助累计量不清零。QPS 使用真实窗口秒数换算，空闲归零；窗口与 KMonitor 固定周期不保证逐点相同。原有 Forward 最近值与 Histogram 口径保持不变。

累计量预建 0，抓取不清零；分母为 0 不补比率。以下查询在 Python `interfaces/metrics.py` 实现，直接调用本次托管 Prometheus：

| 接口 | 口径 |
| --- | --- |
| engine/cache/hit_rate | P 历史命中 token 增量／输入 token 增量，token 加权；可选 targets。 |
| engine/prefill/tps | P 实际计算 token 增量／含实际计算的 Forward 耗时增量。 |
| engine/prefill/tps_with_cache | P 输入 token 增量／有输入的 Forward 耗时增量，输入含命中。 |
| engine/decode/tps | D 完成 Forward 输出 token 增量／墙钟时间。 |
| client/success_rate | 必填 range 区间内成功终态计数增量／全部终态计数增量，先对各 Client 算增量再聚合；可选 targets。 |
| prometheus/query | 固定 query_time 的一次 scalar 或 float instant vector，保留标签、数值和 sample_time。 |

普通接口在本次独占 Prometheus 内按 role 和 instance 地址过滤，先逐条 Counter 算增量再聚合以处理 reset；hit_rate 不能平均节点比率。显式空 targets 报错，选择器转义；每个选中实例的分子/分母至少需要两次采样，重复或缺失序列报错。range 在固定 end 执行单次 instant query，窗口=end−start，精度 ms，低于 1ms 拒绝；latest 用检查前 60s 内最近两次采样的 irate，不平均整分钟。缺少采样或没有可计算 Forward 时报错。

`client/success_rate` 的 range 是必填的 `[start, end]`（start < end <= 0），按固定查询时刻定位窗口，使用 Counter 的 increase，按完成时间统计，分母包含 success/error/timeout/cancelled。它没有最小请求数门槛；窗口无终态时分母为零，或任一 Client 的终态序列采集不完整时均报错。至少两次抓取是计数器增量的计算条件，不是请求数要求。近期完成请求可能在更早时发出；此口径不表示同一发送阶段的成功率。

`cases/master_ha_failover/case.yaml` 使用已转换的 DeepSeek trace，按 YAML 中的 `replay_speed` 回放贯穿杀 A、恢复 A、杀 B、恢复 B，检查结束后提前停止。操作通过已保存的地址列表引用，恢复前不杀另一侧；恢复后检查原主不被抢占。Case 在保守等待后直接检查最近完成请求的成功率，最后收集 Client 终态，不使用进度接口、最少样本数、精确事件窗口或 HA 专用执行器。启动配方仍由 `mock-case launch <case>/case.yaml` 生成；双 Master 要求本机可绑定两个回环 IP。该 Case 保留原 DeepSeek preset 的 Host 开关、容量与水位，不为演示命中而打开原配置关闭的 reuse_cache。参数来自 legacy_unverified preset，配置和数据迁移不代表端到端验收通过。

原始 PromQL 不自动加过滤。matrix/string/native histogram、非有限值、查询 warning 或后端失败都不作为有效门禁；普通指标恰好一个有限值。原始查询返回结构化结果，不隐式聚合或对各样本套用数值门禁。查询不重试，HTTP 和求值均受步骤剩余预算限制。

### Runner 托管 Prometheus

启动配方可加入：

```yaml
prometheus:
  binary: /path/to/prometheus
  scrape_interval: 1s
  startup_timeout: 30s
```

Runner 不下载 binary；默认 scrape 1s、startup 30s，目前验证 Prometheus 3.5.0。它在独立运行目录启动采集器，从结构化日志读取系统分配端口，URL 仅保存在 Python 运行上下文与采集产物中，不向 Engine 配置注入查询地址。

组件 ready 后必须提供合法 metrics URL。Runner 原子更新 file discovery，初始节点全部成功抓取两次才执行 phase；新增 Engine 完成两次基线采样后扩容操作才返回，下线前完成最后采样，再从活动 targets 移除，不另建采集历史台账。每次独立 TSDB、job=mock，不要求用户定义实验 ID。当前 Python 查询只支持 Runner 托管 Prometheus，不接受组件内 prometheus.url/job 作为备用入口。

收尾等待每个目标在收尾开始后成功采样，再停止组件、ZK，最后 TERM Prometheus，超时才 KILL。最后采样失败、采集器异常退出会标记数据不完整并令实验 ERROR。启动失败/取消仍清理并保留现场。

prometheus/ 保存配置、targets、版本、日志、data/TSDB/WAL 和 collection.json；complete 仅说明采集边界及关闭成功，不代替业务指标或中途 up 判断。门禁 FAIL/主动取消也可以有完整采集。曲线不导出 JSON，数据不随收尾删除，后续回放服务在线查询；目前仍用默认保留策略，没有长期归档或正式报告服务。Grafana 大盘映射尚未实现；Client 已提供 issued/finished Counter 及 TTFT/请求耗时 Histogram。

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
仍终止计算。输出注册表最多保留 65536 个请求，未接入结果按 fetch_attach_timeout_ms（默认十分钟）
过期，满时先回收已完成项，不覆盖活跃项。正常性能 Case 应在 Prefill 完成前接入 Fetch；晚接入可读取
保留输出，但补发间隔不代表真实 TPOT。不模拟输出消费者背压。

公共 EngineLoop 使用 Condition 等待事件，避免 Java 21 Object.wait 占用共享虚拟线程承载线程。
