# Mock 框架

这是独立重写的实验框架。当前版本由 Python 编译 Case、编排进程、执行步骤和门禁，Python 函数定义 Case 接口，Java 实现组件和实例协议。RTP Mock 使用生产协议，真实 Master 使用生产调度与 ZK 选举；共享组件不依赖旧 Mock 或 online_eval。

当前已接通本机真实 Master、新 RTP P/D、Client 原生收发核心、显式文本请求列表播放、client0 多 JVM 协调和 Prometheus。Engine 默认共享一个宿主 JVM，也支持每实例独立 JVM，实例间统一使用网络 gRPC。Master 停启、替换及 Client 主备发现已接通；数据集读取、ViT/vLLM 及历史查看服务尚未接通；未实现的能力不登记为可调用 Case 接口。已有验证是小流量功能接线，不能替代开发机/CI 的规模和压力验收。

## 设计边界

本节记录已确定的职责边界，包含尚待实现的部分。当前已迁移 Python Case 接口、Step/ref 和 Prometheus 查询，并拆出协议无关的 EngineLoop、请求进度/终态和 RTP 调度域；本批接通 Engine 宿主、实例扩缩容和发现更新；Client 已接通显式文本请求列表播放与 client0 协调，其他后端仍待实现。其后的“当前代码职责”及运行说明描述可执行范围。

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

RTP 和 vLLM 复用请求事实、资源机制、EngineLoop、Forward、输出和生命周期工具；各自的 Scheduler 拥有队列模型、准入、组批、抢占恢复和结果后的状态转换。替换调度策略不应修改 Compute 或 EngineLoop。原生协议和计算模型参数仍在各自接入/配置边界适配，不宣称两个真实后端只有调度差异。

```text
网络/控制线程 → submit / dependencyCompleted / finish
                           ↓
EngineLoop：维护到期与依赖 → Scheduler.schedule → Compute.forward
                           ↑                         ↓
                      下一轮调度 ← Scheduler.completed
                                      ↓
                         锁外发送输出/就绪/终态通知
```

每个 Engine 同时至多一个 Forward。初次和空闲后先调度；每轮计算结束后，结果交回 Scheduler 提交进度、检查有效性并决定继续或结束。公共 RequestState 保证终态与资源清理只发生一次，但不替调度器决定队列迁移。Scheduler 不持有执行器、计时线程或 EngineLoop，也不发送网络响应。

| 边界 | 职责 |
| --- | --- |
| `submit(Input, Listener)` | 校验身份、接纳并登记请求，返回只读请求句柄。外部不能取得可变队列或修改请求进度。 |
| `schedule()` | 检查预算与资源，调整内部队列，返回本轮 Work/ForwardBatch；没有可执行工作时返回空。 |
| `completed(work, outcome)` / `failed(work, error)` | 提交本轮有效结果，决定后续队列和阶段完成，清理本轮资源引用。 |
| `dependencyCompleted(request, result)` | 接收 KV 加载、生成许可、输出就绪、传输确认等明确事实，不把传输成功当作自动获准生成。 |
| `finish` / `readReady` | 终止请求，或为已经就绪的数据取得短期读取引用；读取结束不等于收到远端传输确认。 |
| `requests` / `queues` | 返回请求快照及调度器定义的统计，不暴露容器。 |

ForwardBatch 是本轮执行输入，不是 running 容器本身。它固定请求成员、本轮 token 数和执行身份；请求的持续状态由 Scheduler 保留。P 通常只执行一轮，D 在 running 中跨轮保留，不每轮回到 waiting。外部线程可在 Forward 期间经公共入口提交请求或依赖事实，但不能修改已经提交的 batch。

Scheduler 调用在同一短锁下串行化；锁不覆盖计算等待或网络发送。公共 EngineLoop 每 50ms 维护一次期限与外部容量变化，入队、依赖完成和执行完成也会立即唤醒；容量版本未变化时不反复尝试初始分配。没有可执行 batch 也必须处理取消、期限和传输结果，不能依赖下一次 Forward 才收尾。

通知按请求顺序、锁外执行。每个请求最多一个尚未处理完的输出；慢消费者只暂停自己的下一轮，不堵住其他请求的计算。接纳上限同时覆盖等待通知完成的请求，不能把它们搬到另一条无界队列。通知失败进入该请求的失败清理；终态通知等基础设施异常使 Engine 的后续状态读取明确报错。

批次取消按成员处理：取消成员不再提交结果，但资源引用保留到该 Forward 结束。只有全部成员终止或 Engine 关闭时才取消整个模拟执行。旧回调按执行对象和请求句柄核验，不能用复用的请求 ID 找到新请求继续推进。一次完成的 Forward 只记录一次耗时，计算量包含这轮实际执行的全部成员，输出量只计有效提交。

### 队列模型、观察与注入

队列模型属于 Scheduler，不另建强制共享的 QueueModel 层。一个实现可以有独立 loading 队列，另一个可以把 loading 作为 waiting 成员的状态；抢占也可能回到 waiting 重新计算，而不是进入 swap。外部只通过操作和快照契约交互。公共请求事实需要逐步覆盖输入、缓存命中、已完成 Prefill、生成进度和本轮工作量；尚未实现分块 Prefill，不能把重复执行当作分块支持。

当前 RTP 使用 FIFO waiting、持续 running 和请求上的容量/加载/输出/传输就绪状态。`max_context_batch_size`、`max_generate_batch_size` 控制 P/D 每批请求数，默认均为 1；`max_batch_tokens_size` 限制本轮 token 预算，0 表示当前 Mock 不额外限制；P 按完整输入长度准入，Compute 使用扣除命中后的长度，D 按本轮 verify token 数准入。P 初始分配遇到临时容量不足等待变化；D 初始分配也等待容量，已进入生成的 D 增长 KV 失败则终止，对齐当前 real RTP `GenerateStateMachine::handleRunning` 的行为。不是 vLLM 抢占/重算策略。

按请求计数，不能直接使用 group/batch 容器的 size：

- `waiting`：等待计算准入或初始 KV 容量的请求。
- `running`：已进入持续计算集合的请求，包含轮次之间等待本请求输出完成的 D；不等于此刻 Forward 的成员数。
- `loading`：资源已预留，等待 KV 加载或生成许可的请求。
- `ready`：P 计算阶段已完成，仍持有传输数据的请求。

上述四类互斥，由 Scheduler 统计；终止但仍被执行引用的成员不计入四类，KV 占用继续保留。公共 QueueSnapshot 不再根据“不是 FORWARD 就是 waiting”推导。Mock 指标保留四类；原生状态接口只有 waiting/running，因此将 waiting、loading、ready 合并为原生 waiting，与任务列表的 is_waiting 一致，避免 Master 漏算已预留资源的请求。

普通批量提交和不可拆分的 group 是不同契约。本批实现多个独立请求合并 Forward。现有原生 EnqueueBatch/EnqueueGroup 接口仍按成员独立准入并保留 group 元数据，不保证组内同轮计算；不可拆分 group 的调度语义尚未实现，不能据接口名宣称已经支持。注入只能走正式提交与依赖入口，不能跳过资源和传输前置条件。暂停、换出、释放 KV 后重算也不能混为一个 swap 队列；本批不实现这些尚未对齐的策略。

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

本地运行默认采用共享模式，已实现的 P/D Engine 放在一个 JVM，ViT 接入后沿用同一放置规则；独立模式用于需要进程隔离的实验。模式在运行开始前确定。共享模式新增 Engine 就在已有宿主创建对象；独立模式新增 Engine 就启动专属 JVM。不实现任意多 JVM/多实例装箱、动态迁移或共享宿主不足时自动另开 JVM。共享指代码复用，实例仍独立拥有缓存容量、请求状态和 Scheduler。

实例使用服务地址区分，同 IP 下用端口区分。控制、业务 RPC 和 metrics 地址之间的对应关系留在发现信息中，不向 Case 暴露额外的 JVM 身份。实例之间统一通过 TCP 通信，共 JVM 不另走 in-process 传输。Python 内部仍须保存进程与实例归属，以处理宿主退出和清理。

进程 env、heap 和资源限制属于 JVM；每个 Engine 的业务参数来自自己的配置对象。共 JVM 时不能用修改进程 env 或全局 System properties 的方式切换实例配置。原生 env 风格字段可以保留其名字和值，由实例加载器从显式配置读取。

停止 Engine 是关闭对象：拒绝新请求、终止活动、关闭自己拥有的 RPC/连接、释放资源并移除发现地址。共享宿主中的其他实例继续运行；最后一个 Engine 停止后宿主可保留至实验收尾，供后续扩容。独立模式完成停止后回收专属 JVM。对象清理不依赖 System.gc，也不承诺能强杀某个 Java 线程；关闭超时使实验失败并进入宿主清理。

Runner 在动作前登记预期停止，结合子进程退出和有界健康检查判断结果；预期停止仍需确认完成，不能吞掉所有异常。停止单个 Engine 不允许对应共享 JVM 意外退出。宿主意外退出映射到其全部实例，记录首个根因，停止剩余发压并清理实验；不自动重启或重放未知结果的写操作。业务请求错误按 Case 门禁判断，不与基础设施退出混为一谈。

### 发现文件和产物

进程启动材料和实例产物分目录保存。每个输出文件只有一个写入者，不让多个 Engine 共享同一个输出目录并各自覆盖 `endpoints.json`；宿主若汇总多个实例端点，由宿主统一写。ready/endpoints 采用临时文件加原子替换发布，部分启动失败要回收已经启动的对象。沿用 `components/<角色>-<序号>/` 保存每个实例的 config、effective-config 和 endpoints；共享宿主自己的启动材料、stdout/stderr 单独保存，不能把宿主日志冒充某个 Engine 的专属日志。聚合 topology/discovery 由 Python 单写，实例不得各自覆盖这份文件。

Python 维护一份权威的当前发现信息，增加、移除和重启换地址均更新该信息；历史端点保留在实验记录中。Master 等消费者能否动态加载发现变更由其自身实现负责，Case 可检查传播结果，框架不替消费者伪造路由更新。显式引用旧地址不能偷偷改指向新实例；一次运行内，Engine 的业务 RPC、控制/metrics 和宿主控制端口统一不复用，跨用途也不能复用，例如不能把旧 metrics 端口分给新 RPC。该约束只覆盖本次 Runner 管理的 Engine 和宿主，不是操作系统级端口预留。

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

#### 第一批可运行范围

当前输入是显式文本请求列表：每项包含 token_ids、max_new_tokens 和可选 priority（默认 50）。不复制旧数据解码器，不做数据生成、数据集读取、循环、多轮、trace、ramp 或动态参与者。每次控制请求受现有 1 MiB 限制，这个入口用于功能接线，不能冒充大规模流式数据播放。后续数据读取接在 Player 的内容来源处，Client 的发送/并发/生命周期边界不变。

启动参数：masters 可省略，默认持续发现本实验的全部 Master；显式填写时是非空、无重复的已知地址列表，播放期间只在这个固定范围内选主备。`masters: {ref: globals.masters}` 也会固定为该 step 执行时的列表，后续补回实例不会自动加入该列表；需要跟随扩容/替换时省略 masters。requests 是非空请求列表；qps 和 concurrency 是全局预算。queue_capacity 默认每 Client 16 条，允许 1–100000；request_timeout 和 drain_timeout 默认 30s，允许 1ms–1h 的整毫秒值。计划时长最多 24h。容量默认值仅用于接线，尚未经过压力校准。

Master 在本轮固定，不动态跟随主节点。单请求 timeout 从 NativeClient 调用开始，不包含小队列等待。当前只支持本机多 JVM；开始时间经本机墙钟转换为各 JVM 的单调时钟，未承诺跨机器时钟同步。准备后 30s 未收到开始命令的 Client 自行报错退出本轮。

```yaml
# environment 增加 client: {deployment: {replicas: 2}, config: {}}
# 以下 steps 放在已有 phase 内；省略 masters，持续跟随本实验的 Master 发现。
- id: load
  describe: 两个 Client 分担同一份流量
  interface: client/start
  args:
    qps: 2
    concurrency: 2
    requests:
      - {token_ids: [1, 2, 3], max_new_tokens: 3}
      - {token_ids: [4, 5, 6], max_new_tokens: 3}
  wait_after: 3s
- id: stop_load
  describe: 收集全部 Client 的执行结果
  interface: client/stop
```

启动配方中的 client 使用 `java -jar <framework>/java/client/target/mock-client-1.0-SNAPSHOT-all.jar`，仍由 Runner 注入配置、拓扑和独立输出目录。

`client/qps` 复用公共 Python 指标查询，统计 `mock_client_requests_issued_total`，不是完成 QPS。沿用 targets 和 range 参数；最终检查必须选择实际发压窗口，不能用包含空闲和收尾的整个实验时长稀释目标。latest 在自然结束后可能已经为零，all 尚不支持。没有第二套“发压有效性”自动门禁。

各 Client 只上报自己的 issued/finished、TTFT 和请求耗时；client0 不把全局汇总重复上报。缺实例或采样不足按公共查询规则报错。Client 执行结果包含 produced/issued/finished/unsent、在途/队列余量、各终态计数和异常，不把业务错误计数当成执行异常。

每个 Client 在自己的 `playback/<轮次>/` 保存计划与结果；client0 另在 `coordinator/<轮次>/` 保存总计划、参与者和汇总。文件单写并原子替换，多 JVM 和下一轮不会覆盖旧结果。当前仅落盘计划和汇总，逐请求完整记录及查询仍待实现。

剩余待讨论项收敛为：实际数据集如何读取/分片，是否需要部分停止，逐请求记录格式，以及适合真实负载的队列/并发默认值。QPS 的容忍度和窗口由 Case 检查指定，不在 Client 再定义一份。

### Python 指标查询与公共指标

Java 产生指标并暴露 `/metrics`；Python 直接调用 Prometheus HTTP API，不经过 Engine 查询代理，也不新增 Java 查询服务。Engine 全部停止后，仍能在 Prometheus 存活期间查询历史窗口。

公共 Engine 使用同一组指标含义和采集事件，Python 默认只实现一套公共语义查询。公共客户端处理 HTTP、超时和响应解析，指标函数处理 PromQL、单位、目标范围和结果形态。确有后端原生指标差异时，映射和公式放在 RTP/vLLM 对应 Python 模块中，Runner 直接调用；不为相同指标复制两份实现。原始 `prometheus/query` 走公共查询客户端，不进行后端改写或自动附加过滤。

指标按实例上报、按查询语义聚合：总 QPS 求和；成功率和命中率分别汇总分子/分母后求比率；平均值按样本数加权；p99 不能平均各实例 p99，须合并兼容直方图或使用完整请求记录。时间窗口、目标覆盖和采样完整性要校验，缺数据不补零。窗口查询包含范围内已退役实例的历史，不能只看当前存活发现集合。不同 Master 的状态保留逐实例结果，不取并集掩盖差异。

去掉业务指标上的 RTP/vLLM `engine`、逻辑 `node`、JVM 和主动添加的实验 ID 标签，保留必要的角色与实例地址区分。独立采集端点可使用 Prometheus `instance`；共享 `/metrics` 时必须在样本上区分实例服务地址，不能让同名同标签序列碰撞或重复采集。`host_ip`/`container_ip` 保持物理含义，共 JVM 时不能仅靠相同 IP 区分 Engine。实验默认独占 Prometheus，跨实验合并分析时再补来源标签，不能在合并后无法区分数据来源。

一个真实完成的 Forward batch 记录一次耗时，不能按 batch 中请求数重复记录。请求终态按请求计数，token 按本轮实际计算和提交分别统计。新增两项 Forward 指标，不增加对应服务接口：

- `rtp_llm_model_forward_us`：模拟 Forward 的实际耗时，单位微秒；对齐 real 模型 Forward 调用区间的含义，不包含请求排队或网络 TTFT。
- `mock_model_forward_planned_us`：公式、scale 和噪声确定的本轮计划耗时，单位微秒。

实际计时从提交本轮计算到公共执行层收到完成通知，包含计时器延迟和回调调度等待，在处理结果及提交 token/KV 前结束。计划和实际对同一完成 batch 各记一次，取消批次不补完成样本。差值可以揭示宿主调度/GC 等干扰，不能单独归因为 CPU 不足；Client TTFT 仍取真实接收时间。本仓 real `NormalExecutor.cc` 围绕 model_->forward 计时，`RtpLLMMetrics.cc` 将该字段注册并报告为 Gauge；本轮因此采用同名微秒 Gauge，计划耗时也用 Gauge。Gauge 保留最近一批，不能拿采样点平均值冒充逐批平均；吞吐仍使用累计 token 与耗时 Histogram 的 sum/count。KMonitor 与 Prometheus 的采样、聚合管线不据此宣称完全相同。

### 落地顺序与待定边界

1. 从 LocalPrefill/DecodeAdmission 分离协议、调度域和公共执行组件，保留现有超时、取消、P/D 交接及清理行为；先让 RTP 使用公共 Engine。
2. 把队列、容量等待、选批和完成后的重新入队一起移入 Scheduler，不能只抽 `advance()`。用第二种明确的调度行为验证边界，再按选定版本实现 vLLM 策略；当前 RTP 有界合批不等于完整 RTP/vLLM 调度。
3. 将 Case 级接口与 Prometheus 查询迁移到 Python，保留实例协议、Java 指标产生及既有校验。迁移完成后删除旧查询服务和重复定义，不长期维护两条调用链。
4. Engine 两种放置方式、动态上下线和发现更新已接通；Client 已接通上述第一批能力，后续按剩余边界推进数据输入和记录。每批改动同时补齐对应调用方、配置、测试和文档，避免旧入口继续承载另一套语义。

本批落实公共 Scheduler 操作、私有队列、持续 running、有界多请求 Forward、部分批次取消和 PD 确认后释放。强制 group、分块 Prefill、抢占/重算、vLLM 对齐版本及 connector 范围仍待确定。

验收关注可观察行为：两种 JVM 放置下缩容数量和流量总量一致；共 JVM 停单实例不影响其他实例；不同队列容器报告相同请求数；batch 内取消不损伤其他请求；取消/抢占无 KV 泄漏或提前复用；每批 Forward 只记一次；退役实例的历史仍可查询；任一基础设施失败不产生虚假的整体 PASS。共享宿主的实际延迟可以不同，不承诺两种放置方式性能相同。

## 当前代码职责

从本节起记录现有实现和命令。Python 接口与 Java 内部协议已分开；Engine 已接通共享/独立宿主和实例扩缩容；Master 与 Client 保持逐实例进程。

| 目录 | 负责内容 |
| --- | --- |
| `python/mock_framework/config/` | YAML 解析、检查、点分配置展开、引用和可执行步骤编译。 |
| `python/mock_framework/execution/` | 初始化、统一 Step 执行、HTTP 传输、断言、记录和清理；`master.py` 负责 Master 原生配置接线，`processes.py` 管理进程和端点。 |
| `python/mock_framework/interfaces/` | 已接通的 Case 函数：逐 Master 状态、Client start/stop、公共指标与原始 PromQL。 |
| `java/interfaces/` | Java 内部端点的类型/校验及共享模型定义；不再定义或导出 Case API。 |
| `java/runtime/` | HTTP 接口绑定、参数和回执校验、调用上下文。 |
| `java/master/` | 真实 Master 的启动入口、生产状态到公共读取接口的适配、独立 ZK 启动入口。 |
| `java/client/` | client0 协调、Player 控速、LoadClient 队列/并发/生命周期、NativeClient 原生收发与实际计时。 |
| `java/engine/rtp/` | RTP 协议适配、P/D 调度策略、请求资源申请、P/D 交接与前缀树缓存。 |
| `java/shared/engine/` | EngineLoop/Scheduler 执行边界、RequestState 进度与终态/资源使用权、QueueSnapshot 请求计数。 |
| `java/shared/compute/` | Forward batch、公式、噪声、Decode 推进、停止模型和可取消计时。 |
| `java/shared/cache/` | 每个池独立的物理 block 容量、所有者引用与释放。 |
| `java/shared/metrics/` | Engine 指标、注册表和 Prometheus 导出；不提供查询代理。 |
| `tests/`、各模块 `src/test/` | Python/跨语言集成与 Java 测试；探针不作为正式业务入口。 |
| `scripts/` | 构建和检查入口。 |

共享指代码复用，不是不同实例共享资源。本批按职责重写 LocalPrefill/DecodeAdmission：它们保留原生校验、ALLOCATE/LOAD/GENERATE 协议状态和响应编码，通过公共 Scheduler.Request 句柄与 EngineLoop 交互；RtpTasks/RtpErrors 在边界转换输入、状态和错误码。RtpScheduler 无 gRPC/protobuf/执行器依赖，拥有 FIFO、持续 running、资源准入和执行结果提交。

RequestState 是共享的请求进度、终态和资源清理机制。调度器决定何时申请和成功发布缓存；公共机制确保终态只通知一次、发布失败也尝试释放、取消请求不能再提交进度。Forward 持有资源使用权，终止请求不会提前释放仍被使用的块；EngineLoop 对执行对象做身份校验，晚到回调不能影响下一次执行。当前模拟 Forward 使用可取消计时；单成员取消保留整批执行，全部成员终止或关闭 Engine 时才取消计时并释放使用权。

P 的 retained 请求计算完成后变成 READY，释放 Forward 槽位但保留 KV，直到 LOAD ACK、取消或期限到达；读取期间另持资源引用。D 的接入/传输与计算进度分开保存，完成后的 token/迭代数只由调度域推进。请求终态先清理资源、记录指标，再通知协议适配器记录原生历史和完成响应。历史通知异步，资源释放与历史可见之间可能有短暂间隔。

当前生产路径支持多个独立请求合批，P/D 使用相同调度循环。测试覆盖持续 Decode、批内独立取消、临时容量等待、输出阻塞和 PD 确认边界；完整 group、分块 Prefill 和 vLLM 未实现。

## 构建与检查

在本目录使用 Java 21、仓库 Maven Wrapper 和 uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/build.sh
PROMETHEUS_BIN=/path/to/prometheus ./scripts/check.sh
```

`build.sh` 编译 Java 模块并运行测试；`check.sh` 继续运行 Python 测试。缺少 `PROMETHEUS_BIN` 时跳过真实采集集成测试。多 Master 测试需要操作系统支持对应的 127/8 地址；macOS 未配置回环别名时跳过双 Master 测试，不自动改系统网络。生成目录不入库，Python 依赖由 `uv.lock` 锁定。

构建复用生产 `flexlb-common`、`flexlb-grpc` 的公式与协议。Master 还使用生产 cache/sync/api，Spring 只进入 Master 运行时。框架构建跳过 Boot 重打包和 Whale 分发归档，不改变生产模块独立构建的默认行为。

## Case 配置和编译

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

每个 Case 独立目录组织自己的配置，数据集可共享。私有相对路径基于 Case 目录；显式提供共享根时，`data/`、`datasets/`、`views/` 基于共享根。编译不启动进程、不查询指标，也不检查数据内容。数据集格式读取尚待实现；目前 Client 只接受 start 参数中的显式文本请求列表，不提供旧格式兼容、转换器或生成器。

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

普通指标优先使用语义接口，YAML 不写 PromQL。当前支持 `latest` 和相对该 step 固定查询时刻的区间 `[-60s, 0s]`；`all` 的跨轮次查询窗口尚未实现，目前拒绝。

`latest` 使用前 60 秒内最近两次样本的 irate；区间使用 increase。执行器固定 `query_time`，指标实现记录实际查询点、表达式和窗口，不重新取墙钟。`expect` 支持 eq、列表 count 或数值 gte/lte。eq 是结构相等，列表顺序参与比较；实例查询返回排序后的地址列表。普通数值直接返回 float，不包装为 `{value, unit}`；单位在接口定义里。逐 Master 状态返回 `{master服务地址: [engine服务地址, ...]}`。

原始 `prometheus/query` 只接受 promql，在固定 query_time 执行一次 instant query；历史窗口写进表达式。返回 `{samples: [{labels, value, sample_time}], query_time}`，不自动把它压成数值或对全部序列应用同一阈值。空 vector 保留为空；业务数值门禁使用语义接口，缺数据报错而不是补零。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。Master/Client 按 replicas 启动独立进程；Engine 的 replicas 表示逻辑实例数，编号 `<组件>-<序号>`，由 engine_mode 决定放置；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。配方不再包含 bindings；目标选择和查询聚合由 Python 接口函数完成。

真实 Master 配方需要顶层 `zookeeper`；P/D 使用 `java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
engine_mode: shared  # shared（默认）：一个 Engine JVM；isolated：每实例一个 JVM
zookeeper:
  command: [<Java21>, -cp, <Master的classpath>, org.flexlb.mock.master.ZookeeperNode]
components:
  master:
    command: [<Java21>, -cp, <Master的classpath>, org.flexlb.MasterNode]
    startup_timeout: 60s
  prefill:
    command: [<Java21>, -jar, <RTP的JAR绝对路径>]
  decode:
    command: [<Java21>, -jar, <RTP的JAR绝对路径>]
```

配方必须覆盖 Case 实际声明的组件。组件通过 `MOCK_CONFIG_FILE` 读取本实例配置和完整 environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR` 提供身份和目录。初始化并绑定端口后原子发布 `MOCK_READY_FILE`，启动超时或提前退出都失败并回收已有进程。

Engine JAR 的默认入口为 `EngineHost`。共享模式下 P/D 的 command、cwd、env 和 startup_timeout 必须一致；角色差异写入各自 config，不能用进程 env 区分同 JVM 的实例。宿主读取 `MOCK_COMPONENTS_DIR`，按实例目录读取配置并创建对象。底层 `RtpNode` 入口仍可用 classpath 直接运行，供配置校验与单节点探针使用。

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

共享宿主关闭最后一个 Engine 后仍保留，允许再次扩容；独立宿主在关闭实例后回收进程。每次新增使用新编号、目录和未用过的监听地址。编号表示一次创建，例如删除 prefill-0 后再次扩容会创建 prefill-2，不填补旧编号；创建失败也不回收编号。它只用于内部对象映射和产物路径，Case 仍使用服务地址，不增加 node/generation 标签或选择器。实验目录和实例目录拒绝覆盖；Java 以 CREATE_NEW 写入 effective-config.json，另一宿主也不能重新使用旧实例目录。重新创建必须使用新编号与新目录，Runner 断点恢复、同身份原地重启不在当前能力范围。`hosts/engine-shared/` 或 `hosts/engine-<实例>/` 保存 JVM 日志，`components/<实例>/` 保存 config、effective-config、endpoints 和 ready，不复制宿主日志到各实例。下线删除 ready，保留 endpoints 作为历史。

`topology.json` 保存当前实例，`topology-history.json` 保存历史端点和起止时间；两者由 Python 原子发布。活动 Master 的 discovery.json 随活动 Engine 更新，停止后保留最后一份发现快照，角色缩到零时写空地址列表。Master 读取接口动态加载历史地址映射，所以仍能如实报告尚未清除的旧地址；是否完成刷新由生产发现实现决定。本仓 Master 的 workerRegistry.health.statusStaleAfterMs 默认 10000，下线后的短期旧地址仍如实返回，验收需覆盖淘汰窗口。最终清理后清空当前发现，历史记录保留。

Prometheus 对每个实例独立抓取，以 RPC 地址作为 instance；宿主不重复输出业务指标。新增实例先完成采集基线，下线前完成末次抓取再移出目标，保留退役目标与历史数据。语义查询按固定时间窗选择生命期有交集的实例，显式 targets 不能引用该窗口内从未存在的实例，覆盖不足仍报错。

Runner 每秒开始一轮宿主健康检查，单次 HTTP 调用限时 10 秒，检查实际成员与 Engine 状态；进程退出另行实时监视。操作期间用同一管理锁隔离成员变更与健康快照。共享宿主意外退出记录受影响实例并使整个实验 ERROR；预期停止仅登记将回收的独立宿主，不屏蔽共享 JVM 意外退出。Master 始终单独 spawn，不进入 EngineHost。

Runner 是持续存活的进程所有者，负责启动、退出监视和最终清理。`Processes` 不识别 Master 业务接口，也不合并返回值。真实 Master 的状态查询由 `interfaces/master.py` 按 targets 逐实例调用，并保留每个 Master 的独立结果。

`master/stop(targets)` 由 Python 停止选定的独立 Master JVM，省略 targets 时停止全部活动 Master；空集合、重复或失效地址在任何停止前报错。停止前完成末次采样、登记预期退出，确认退出后移出活动发现与 Prometheus 目标，删除 ready，保留原 endpoints、日志、配置和历史生命期。意外退出仍使整个实验 ERROR；强制 SIGKILL、部分停止失败同样报错，不重放已完成动作。

`master/start(count=1)` 的 count 范围为 1–1024，从 environment.master 的配置创建新 JVM，复用本实验的 ZK、HIPPO_ROLE 和公共服务端口，使用递增编号、新回环 IP、新产物目录。控制 HTTP 也绑定新 IP，防止旧控制引用命中新进程。初始启动和动态启动共用同一路径；补回时允许某个 Engine 角色已缩到零，按当前空发现启动。返回新 RPC 地址列表，选举和采集基线完成才返回。这里的 start 是新增/补回，不恢复旧身份。旧 step 引用始终指向旧地址，不能自动替换成新地址。Java 不提供 Master 集群生命周期服务。

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

运行目录不覆盖已有目录，保存原始 Case、编译配置、启动配方、接口清单、代码版本和工作区差异、组件配置和日志、phase/step 时间线、门禁查询时刻/真实值/阈值/结果及实际 PromQL。逐请求持久化及正式历史查看服务尚未接入。测试 TransportProbe 只验证进程与传输，不替代业务实现。

## 真实 Master 适配

`org.flexlb.MasterNode` 与生产 Application、同步器和调度器在同一 JVM，不另起 Master 子进程，也不实现 Mock 调度器。Case 的 `environment.master.config` 使用原生环境变量字符串：`FLEXLB_CONFIG` 是业务 JSON，`MODEL_SERVICE_CONFIG` 是路由 JSON。

Runner 先启动独占 ZK，再启动 Engine、Master 和 Client，与 YAML 书写顺序无关。已启动 Engine 的 `endpoints.json` 形成 topology/discovery；Runner 只替换 `MODEL_SERVICE_CONFIG` 的 hosts/discovery_file，保留 service_id、组和角色域名，不读取线上 VIP，不改写 `FLEXLB_CONFIG`。当前只支持一个 endpoint 组、分离角色和 gRPC；ViT 发现映射可表达，但 ViT Engine 尚未实现。

发现使用实际 Engine gRPC 端口，生产 WorkerAddressService 再转换内部 HTTP 身份，不假定 Mock 控制端口相邻。原始组件配置、实际环境、effective-config 和 discovery 分开保存；Master 暴露生产 `/prometheus`。

只绑定 `MasterStateApi` 的两个读取接口：

| 接口 | 行为 |
| --- | --- |
| `master/leadership/get` | 读取各 Master 的生产 isMaster 声明，返回声明自己为 Leader 的 Master 服务地址列表。数量由 expect 判断，不伪造唯一 Leader。 |
| `master/engine/get` | 读取各 Master 的 WorkerDirectory，按角色映射 Engine 服务地址，逐 Master 返回。未知地址报错，不能按部署数量补齐或将各节点取并集。 |

单/多 Master 全部使用生产 ZK 选举，不提供 standalone 分支。独立 ZookeeperNode 使用已有依赖、独立 JVM/数据目录；Runner 分配地址和本次 HIPPO_ROLE，运行时同步配置为 needConsistency=true、会话超时 6s，并保存实际值。ready 必须等到真实选举产生 Leader。

Master 使用 127.0.0.1、127.0.0.2 等地址，相同 HTTP 端口，gRPC=HTTP+2，适配生产以 IP 作为身份和共用转发端口的约束。端口启动时被抢占则失败，不静默换协议。Linux 默认支持 127/8；macOS 缺少别名时明确报错。实验之间不共享 ZK 或选举路径。清理先停止组件，再停止 ZK，最后停止 Prometheus，保留日志和数据。

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。`tests/test_master_lifecycle.py` 在两种模式下持续运行同一 Client 计划，停止主节点、补回新 IP 的 Master、再停止旧节点；在旧请求 deadline 之后检查新主持续成功接流，并检查独立进程退出、历史产物和采集目标退役。切换错误如实保留，不要求整个过渡期零错误。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

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
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持 server_args 小写或 Whale 大写 env 名，同一字段同时提供两种写法报错；整数/布尔接受对应原生文本，其余原生内容保留，但不能据此宣称功能已模拟。当前要求显式 block size、max sequence length 和 test_block_num，只支持 `dp_size=1`、`dp_rank=0`；P/D 批量上限取 1–1024，默认 1。不加载模型层/head 或 GPU 布局，不从部署内存推导 KV 块数。

RTP 请求公共支持范围集中在 `RtpRequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：多模态、adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 处理原生协议，RtpScheduler 管理资源和队列，只有 GENERATE 后进入计算队列；每个 Engine 一轮一个 Forward，按配置合并多个请求。初始 KV 分配遇临时不足等待容量变化；生成期间每轮先增长 KV，失败则终止，成功后才执行与提交输出。

P 的内部 `PrefillTransfer.start()` 注册活跃来源并复用 LocalPrefill 计算。Forward 完成后 KV 仍由请求持有；网络 `Read` 交付结束后仍保留来源，P 收到 D 的 LOAD ACK 后才完成计算阶段并释放。正式服务**只保留 Read RPC，没有外部 Prepare RPC**；测试从包内 start 建立来源或调用正式生成链路，不为探针注册额外协议。

D 的 LOAD 从 peer_addrs 对应 P 发起真实网络 Read，校验 client/request 身份、完整 prompt、block size 和单 Reader 所有权；不能用缓存中存在的 key 冒充活跃来源。返回元数据为 block size、输入长度、完整块 cache keys 和 stop_after；不复制 KV 张量，不模拟 RDMA/TCP cache-store 或带宽。D 验证成功且预算有效后才能确认 LOAD，传入 KV 不算 D 历史命中，也不提前提交为缓存。

Read 等待计算就绪；D 取消/超时会取消 P 来源，P 取消/关闭也使 LOAD 失败。`load_cache_timeout_ms` 默认 5s，非正值按原生回退 5s，同时受总 deadline 限制。没有 Reader 或 Read 入口校验失败时，来源仍由正式 P 请求所有者管理，受请求取消、timeout/deadline 和节点关闭约束。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0、文本 P/D 和 max_new_tokens>1。入队 ACK 前必须取得 D 分配确认；P 可在 Fetch 前计算并持有 KV，但不能先发 LOAD 或执行 D。Fetch 单次接入后使用相同输出链路，不缓存整段结果。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。incremental 返回新 token，否则返回累计输出；aux.output_len 始终累计。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。Mock 保护上限是 1024 活动请求，TaskHistory 保留 1024 项。旧游标越过历史窗口时，按 real 的恢复方式继续返回完整当前状态、仍保留的完成记录和推进后的游标，不以 OUT_OF_RANGE 阻断健康同步；超前游标不倒退。每次缺口读取在组件日志中记录 `finished_history_gap`，包含 requested_version、dropped_through、latest_version、retained，不能把这段完成记录当成完整请求历史。生产 proto 没有历史完整性字段，缺口证据保存在实验组件日志；需要完整逐请求证据时不能依赖 WorkerStatus 的有限窗口。real 当前为 1000 条/5s 保留，Mock 的 1024 条是明确的模拟边界，不宣称保留策略完全相同。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。D 当前要求 input+max_new_tokens<max_seq_len，未实现真实截断/reserveStep。成功且 reuse/device 均开启时写回完整 prompt 块；失败不提交，生成内容不额外建索引。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

共享 `BlockPool` 只管理物理容量和引用。每个实例/存储层各自建池，每个所有者持有独立 Reference，close 幂等，不能释放其他所有者份额或混用池。REQUEST/CACHE/LOAD/STORE/EVICTION 分别计数；只有 CACHE 引用的块算可回收容量，free 和 available 分开，分配失败不部分扣账，释放唤醒等待者。

RTP 的 `BlockTree` 管前缀与叶子候选，`DeviceCache` 管准入、复用、增长、完成和取消。按 2026-10-28 fetch 的 main `7cdc40370e5129c5f3d70a40d7cdc54633a7955`，real 统一使用 BlockTreeCache；`lru/lfu/fifo` 是树内候选排序，不是 tree 之外的模式。只驱逐未引用叶子，按需逐个回收，保留可匹配父节点；match 刷新热度，release/重叠 insert 不刷新。

容量使用原生 `test_block_num` 和 `reserve_block_ratio`，不支持旧 `capacity.device_blocks/reserve_blocks`。block 0 不可分配，可用块数=test_block_num−1；token 容量再乘 block size。reserve 是原生百分比向下取整，默认 5%，约束首次分配，允许运行中增长使用预留。先引用命中再校验净新增需求，失败回滚；增长失败保留已取得部分，由请求最终释放。

仅缓存完整块，命中排除 prompt 最后 token；遵守全局/请求 reuse 和 device 开关，D 还遵守 disable_decode_first_malloc_device_reuse。成功完成才提交新 key，取消不提交。当前只有 Full-attention device 池；host/disk/remote 尚未接通，显式开启对应 enable_*_cache 会拒绝启动。共享容量类可供各层使用，迁移与驱逐策略仍由 Engine 实现。

## Compute、Decode 和停止模型

每个组只在自己的 config 中放一份模型，没有 `environment.compute` 或覆盖优先级：

```yaml
environment:
  prefill:
    engine: rtp
    config:
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
      compute.noise.scale: 1
      compute.noise.stddev_ratio: 0.05
      compute.noise.max_deviation_ms: 20
      compute.noise.seed: 42
      generation.eos_probability: 0.01
      generation.seed: 42
  decode:
    engine: rtp
    config:
      compute.expression: "1 + 0.1*totalStepTokens + 0.001*totalContextTokens"
      decode.draft_tokens: 3
      decode.acceptance_rate: 0.8
      decode.seed: 42
```

组件加载器负责读取自己的文件外壳。RTP 用 `RtpConfig.load()` 提取本组 compute，再调用 `ComputeModel.fromJson()`；共享 Compute 不认识 Runner 的 component/environment 格式，不再提供 fromComponentFile。测试探针自行提取配置。effective-config 保存补齐默认值后的结果。模型类型及语义校验来自 Java ComputeSpec/DecodeSpec/StoppingSpec，Python 不复制字段表；编译只处理组件配置结构，模型错误由 Java 启动加载器拒绝，编译通过不代表模型已验证。

`ForwardBatch` 保留 DP、请求输入长度/cache 匹配/目标输出/已生成/stepTokens/优先级/图片原始 shape。Engine 构造 batch；ForwardModel 只返回 Duration。默认 FormulaForward 只接受一个非空 DP，不暗自合并多个 DP。

公式复用生产 ArithmeticFormula 的函数、`param(name, value)`、`sum(...)`。param 初始值按解析时求值，只用常量；不新增参数映射。结果单位 ms，可保留到 ns，不设隐式 1ms 下限；未知变量、负数、非有限值和溢出报错。逐请求变量只能在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。`totalComputeTokens^2` 和 `sum(computeTokens^2)` 不等价。computeTokens=input−hit，contextTokens=input+generated，stepTokens 由 Engine 提供，图片不计算特征 token。

噪声先乘 scale，再加高斯；默认 scale=1，stddev_ms/stddev_ratio=0，二者按平方和开根合并。启用随机噪声须有 seed 与正的 max_deviation_ms，裁剪扰动后耗时下限为 0。按 seed、Engine 身份、角色和 batch 身份确定一次 Sample，不随调用顺序改变；不同真实 batch 使用不同身份。

`plan()` 只算耗时，`SimulatedCompute.submit()` 才等待；Execution 可取消，唯一终态为 COMPLETED/CANCELLED。关闭取消未完成项并拒绝新任务，回调在独立虚拟线程执行。共享 Compute 不提交缓存、不释放 Engine 资源，也不把计时完成当请求成功。

DecodeStepModel 默认每轮 1 token。推测配置用 draft_tokens、acceptance_rate、seed：按首次拒绝得到接受前缀 A，再提交纠正/奖励 token，即 C=A+1；固定独立接受率 α、草稿长 k 时是封顶几何模型，E[C]=1+α+…+α^k，见[经典推测解码论文](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)。这不宣称复现所有 MTP/EAGLE。计算按验证数收费，指标按实际提交数累计，KV 覆盖验证上下文，接近输出上限缩短草稿预算。

StoppingModel 只配在 P 的 generation，默认 eos_probability=0，随机概率要求 seed。固定 EOS hazard 是 Mock 的封顶几何近似，不是线上长度规律；遵守 min/max_new_tokens，ignore_eos 强制上限。P 按稳定请求身份只抽样一次，经 LOAD 传 stop_after，D 不重复抽样；跨过停止位置只提交允许部分，仍计验证成本。首 token 即结束时使用 PREFILL_COMPLETE 释放 D，不再执行 Decode。真实 observed 长度或输出覆盖由 Client 指定目标并 ignore_eos。

## Client 原生收发核心

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。每次请求固定一份主备快照，先向主节点 Schedule，传输失败时最多向备节点尝试一次；收到业务响应、deadline 超时、主动取消或进入 Engine 取流后都不重发。要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每个 Client JVM 独立运行 MasterDiscovery：每秒刷新 topology.json 并 POST 真实 `/rtp_llm/master/info`，单次心跳 500ms、最多 4 个并发探测；连续两次失败判不健康。对齐 real Frontend `server/host_service.py` 的选择规则：优先健康且声明自己为主的节点，否则保留仍健康的原目标，再按 IP 选择健康候选；备用节点从未声明为主的其他健康节点中选择。发现暂时漏掉的健康节点仍继续探测，失败后停止保留，不再发现且已不健康的记录在最后探测 30s 后清理。文件损坏或非法端点属于执行错误，会终止整个 Client 播放；没有健康 Master 是请求失败，不直连 Engine，也不按列表轮流广播。

发送边界对齐 `server/master_client.py`：只有 Schedule 的传输失败尝试一次备节点；业务拒绝、deadline 和取流失败不尝试其他 Master。整个请求仍受同一个总 deadline 约束，重试消耗剩余预算。这个策略不承诺 exactly-once：响应丢失后原主可能已经接收请求，保留与 real Frontend 相同的传输重试风险。Client QPS 仍按逻辑请求 issued 计一次，备节点尝试不会再发放流量预算。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向该请求最后尝试的 Master 尽力发一次有界 Cancel，不随新的选举结果改变取消目标；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。显式请求列表的 Player、LoadClient、client0 协调和 start/stop 已接通，主备发送只改变 Schedule 目标选择与传输失败重试边界。数据集读取与完整逐请求持久化尚未接通。

## 指标、采集和查询

EngineMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。业务标签只保留 role；Prometheus 为每个采集目标设置 instance=实际 RPC 服务地址，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| mock_engine_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| mock_engine_requests_finished_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| mock_engine_cache_lookup_tokens_total / mock_engine_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| mock_engine_forward_tokens_total | Counter，kind=input/compute/output | 完成 Forward 的 token；input/compute 汇总本批全部执行成员，output 只计有效提交。全部取消的 Forward 不补完成样本。 |
| rtp_llm_model_forward_us | Gauge | 最近一个完成 Forward 的实际耗时，微秒；计时到执行完成通知，不包含等待调度域锁和结果提交。 |
| mock_model_forward_planned_us | Gauge | 同一个完成 Forward 的计划耗时，微秒，包含公式、scale 和噪声。 |
| mock_engine_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| mock_engine_requests | Gauge，state=waiting/running/loading/ready | 调度器报告的四类互斥请求数。 |
| mock_engine_kv_tokens | Gauge，state=total/available/free/held | token 容量，available 含可驱逐缓存，不等于 free。 |

累计量预建 0，抓取不清零；分母为 0 不补比率。以下查询在 Python `interfaces/metrics.py` 实现，直接调用本次托管 Prometheus：

| 接口 | 口径 |
| --- | --- |
| engine/cache/hit_rate | P 历史命中 token 增量／输入 token 增量，token 加权；可选 targets。 |
| engine/prefill/tps | P 实际计算 token 增量／完成 Forward 耗时增量。 |
| engine/prefill/tps_with_cache | P 输入 token 增量／完成 Forward 耗时增量，输入含命中。 |
| engine/decode/tps | D 完成 Forward 输出 token 增量／墙钟时间。 |
| prometheus/query | 固定 query_time 的一次 scalar 或 float instant vector，保留标签、数值和 sample_time。 |

普通接口在本次独占 Prometheus 内按 role 和 instance 地址过滤，先逐条 Counter 算增量再聚合以处理 reset；hit_rate 不能平均节点比率。显式空 targets 报错，选择器转义；每个选中实例的分子/分母至少需要两次采样，重复或缺失序列报错。range 在固定 end 执行单次 instant query，窗口=end−start，精度 ms，低于 1ms 拒绝；latest 用检查前 60s 内最近两次采样的 irate，不平均整分钟。缺少采样或没有可计算 Forward 时报错。

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

组件 ready 后必须提供合法 metrics URL。Runner 原子更新 file discovery，初始节点全部成功抓取两次才执行 phase；新增 Engine 完成两次基线采样后扩容操作才返回，下线前完成最后采样，再从活动 targets 移除，历史采集记录保留。每次独立 TSDB、job=mock，不要求用户定义实验 ID。当前 Python 查询只支持 Runner 托管 Prometheus，不接受组件内 prometheus.url/job 作为备用入口。

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
