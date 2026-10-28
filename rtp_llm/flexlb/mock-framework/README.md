# Mock 框架

这是独立重写的实验框架。当前版本由 Python 编译 Case、编排进程、执行步骤和门禁，Python 函数定义 Case 接口，Java 实现组件和实例协议。RTP Mock 使用生产协议，真实 Master 使用生产调度与 ZK 选举；共享组件不依赖旧 Mock 或 online_eval。

当前已接通本机真实 Master、新 RTP P/D、Client 原生收发核心和 Prometheus。每个 Engine 独立 JVM，实例间使用网络 gRPC。Client 播放器、多 JVM 发压协调、Master 停启和滚动重启、动态缩容、ViT/vLLM 及历史查看服务尚未接通；未实现的能力不登记为可调用 Case 接口。已有验证是小流量功能接线，不能替代开发机/CI 的规模和压力验收。

## 设计边界

本节记录已确定的职责边界，包含尚待实现的部分。当前已迁移 Python Case 接口、Step/ref 和 Prometheus 查询，并拆出协议无关的 EngineLoop、请求进度/终态和 RTP 调度域；共享 Engine JVM、动态扩缩容和发现更新仍待接通。其后的“当前代码职责”及运行说明描述可执行范围。

### Python Case 操作与 Java 实例能力

Python 提供面向 Case 的业务接口，Java 提供实例原子能力。

| 层次 | 职责 |
| --- | --- |
| Python Runner | 顺序执行 step、解析引用、固定时间范围、预算和失败处理、记录结果。 |
| Python Case 接口实现 | 选择实例、扩缩容、分配总流量、跨实例查询与聚合；使用进程管理和发现文件能力。 |
| Java 实例协议 | 停止实例、查询自身状态、接收已分配的流量计划等。Engine 宿主另负责创建和关闭本 JVM 内的 Engine 对象。 |
| Java 公共 Mock Engine | 请求接入、调度循环、资源机制、Forward 执行、输出和指标；组合 RTP 或 vLLM Scheduler。 |

缩容一个 Decode 时，Python 从全局候选集合中选择并固定一个地址，再通知其宿主关闭该实例，更新发现文件并返回实际结果。Java 不接收全局 `count: 1`；不能对每个 JVM 广播一次缩容，否则会缩掉多个实例。Client 总 QPS、请求数和并发预算也只分配一次，实例收到的是自己的份额。

Case 操作放在独立的 Python 实现中，不在 Runner 主循环按接口名堆分支。简单实例调用复用公共传输；扩缩容、流量分配、查询聚合保留各自的业务逻辑，不用一个通用广播器代替。

参数引用只表达取值：前置 step 的返回结果，或已定义的全局配置/发现信息。不增加专用的 leader/group/instance 选择器语言；查询主 Master 后直接引用查询结果。目标省略时默认对应操作域的全部实例，显式空集合或失效地址报错，不能退回全部，也不能自动缩成当前可达集合。需要唯一目标的操作校验数量。引用统一写成 `{ref: steps.<id>}`、`{ref: steps.<id>.<field>}` 或 `{ref: globals.<field>}`。当前全局字段为 masters/prefill/decode/vit 的服务地址列表，目标参数统一为 `targets`。不新增 YAML 流程宏。

### Python 接口定义和 Step 模型

Case 接口由 Python 函数实现，装饰器登记稳定接口名、读写属性和必要单位，参数与返回类型来自函数签名。参数范围和跨字段校验放在参数类型中，不在装饰器另抄一份字段表。Registry 只负责登记、查找和定义检查；不要求每个函数继承一个接口基类，也不做动态插件发现。运行状态由本次 Case 的上下文持有，不能存入全局注册函数。

Step 统一为一次接口调用，包含 id、describe、interface、args、within 和可选 expect。删除 YAML 的 call/check 外壳及两套重复 Step 类：无 expect 时执行并保存结果，有 expect 时在同一次返回后判断。读取也可以只取值供后续引用；写操作的回执可以断言，但不因断言失败重放。接口读写属性与是否断言相互独立，不把普通检查隐式变成轮询。

执行链统一为解析引用、校验实参、固定查询时刻和预算、调用 Python 函数、校验返回值、保存结果、执行可选断言。编译阶段检查引用必须来自前置 step 或允许的全局信息，检查字段路径与类型；运行时解析后再次校验。Python 类型注解本身不等于运行时校验。

普通数值查询直接返回数值，单位声明在接口上；地址集合返回列表，动作返回明确回执，逐 Master 状态保留映射。断言按结构和运算符判断，不按 MetricValue、PerMasterEngines 等业务类型名称偷偷聚合。接口文档和清单可以由 Python 定义导出，用于保存实验现场，不再依赖 Java 构建才能编译 Case。

Python async 只表达非阻塞等待，不代表任意 step 可以脱离 Runner 后台执行。Client 活动的启动回执、跟踪、stop/drain 以及原 run: background 的替代方式留待 Client 设计确定；本轮不引入通用后台任务框架。

本轮重构按职责变化整体重写 Registry、Step 模型、编译和执行主链，联动 CLI、启动配方、夹具和构建入口；复用严格 YAML 解析、进程回收、采集与记录等仍成立的机制。Java 内部端点保留明确的请求/响应协议，但不再声明整套 Case API；配置类型留在所属组件，不能随旧接口目录误删。保留引用重校验、固定时间窗口、错误结果和取消清理的验收，不保留另一套旧 Case 执行入口作为兼容分支。

### 公共 Engine 与独立调度域

RTP 和 vLLM 共用一个 Mock Engine 执行框架。复用请求数据、缓存机制、Forward 模型、输出、生命周期工具和指标；各自实现队列模型和调度策略。原生协议、配置及错误码确有差异时，在接入边界适配，不塞进 Scheduler。这个划分是 Mock 的实现目标，不宣称两个真实后端只有调度差异。

```text
协议接入、校验、公共 Request
            ↓
RTP Scheduler / vLLM Scheduler
队列、接纳、资源申请时机、抢占/恢复、组批
            ↓
公共 Forward：计算模型 → 模拟执行 → 本轮结果
            ↓
Scheduler 更新进度，决定下一轮或结束
            ↓
公共输出、终态记录和资源释放
```

Forward 完成后回到 Scheduler。后续可能继续 Decode、执行下一块 Prefill、等待资源或结束请求，不能做成只向前走一次的流水线。

| 公共机制 | Scheduler 决定的内容 |
| --- | --- |
| Request 输入、进度和资源记录 | 内部队列种类、成员、顺序及状态转换 |
| BlockPool、缓存匹配、租约、增长和释放 | 分配时机、容量不足处理、缓存发布及保留策略 |
| ForwardBatch、计算公式、噪声和执行器 | 本轮请求、阶段、token 数和 batch 预算 |
| 输出、停止模型、终态与清理工具 | 何时提交进度、重新入队、抢占或结束 |
| 指标记录与快照结构 | 内部调度状态如何映射到公共观察语义 |

调度域负责提交、取消、生成下一轮 batch、处理本轮完成、读取快照和关闭。公共 EngineLoop 只依赖 `schedule/completed/failed`，串行推进本轮执行；协议适配器不操纵内部队列。当前 RtpScheduler 集中持有请求、期限和容量等待任务，内部 P/D 策略只决定排队、资源申请时机与重新入队，不各建 RPC 服务或执行器。D 只在容量版本变化后重新申请；定时唤醒用于检查中断/期限。输出背压通过就绪回调唤醒。

公共 Request 保存一份事实状态，调度期间由 Scheduler 修改；公共执行器只接收本轮不可变描述并返回结果，不维护另一套队列状态。请求要区分输入总量、历史缓存命中、已完成 Prefill、已生成输出和本轮工作量。分块 Prefill 不能每轮都用 `inputTokens - hitCacheTokens` 作为本轮计算量，也不能把之前已算过的分块算成历史缓存命中。

调度与本轮资源预留在同一串行边界完成。容量不足由策略决定等待、跳过、拒绝或抢占；没有入选的尝试不得泄漏新增预留。意外异常释放本轮新增资源，不能误释放请求原来持有的资源。执行中的 batch 拥有相应资源使用权，取消或抢占不能提前释放仍被该 batch 使用的块。

一个 batch 包含多个请求时，取消其中一个不应直接取消其他请求的 Forward。完成回调检查具体请求及本轮执行是否仍有效，取消请求不再提交输出；旧回调不能推进后来复用身份的请求。请求终态和资源释放使用公共机制保证只执行一次。

### 队列观察与注入

共享 Scheduler 的操作和观察契约，不向外暴露可变队列。一个实现可保存 batch/group，另一个可展开请求；不能用容器 `size()` 直接报告请求数。例如两个 batch 含 10 和 5 个请求，应报告 15 个等待请求。

公共入口表达提交请求、提交一组请求、取消请求和读取状态。外部不能直接对 waiting/running 容器执行 add/remove，也不能绕过 KV、传输和进度前置条件把请求塞进 running。注入通过正式提交入口完成校验、去重、登记和调度唤醒。容器可以在内部复用，但不强制两个策略采用同一容器或元素类型。

统一按请求计数：`running` 是正在执行当前 Forward 的请求；`waiting` 是其余已接纳、尚未终止的请求，包括等待 KV、传输、下一轮、输出就绪，以及 P 计算完成后等待消费的请求。已请求终止、仍等待 Forward 释放资源的成员不计入这两个数，资源占用仍如实保留。公共 QueueSnapshot 从不可变请求快照统计，原生状态和指标使用同一口径，不按队列容器数量计数。

批量提交与必须一起调度的 group 也要区分。前者可以逐项接纳，后者不能任意拆散；支持范围和部分失败规则在协议中明确。不先引入描述任意队列或组策略的配置语言。

### JVM 放置、配置和生命周期

业务操作只感知实例及其服务地址，不感知 JVM。物理放置先只支持两种 Engine 形态：

| 对象 | Engine 共享模式 | Engine 独立模式 |
| --- | --- | --- |
| Master | 每个 Master 独立 JVM | 每个 Master 独立 JVM |
| P/D/V Mock Engine | 全部放在同一个 Engine JVM | 每个 Engine 一个 JVM |
| Client | 每个 Client 独立 JVM | 每个 Client 独立 JVM |

目标是本地运行默认采用共享模式，所有 P/D/V Engine 放在一个 JVM；独立模式用于需要进程隔离的实验。模式在运行开始前确定。共享模式新增 Engine 就在已有宿主创建对象；独立模式新增 Engine 就启动专属 JVM。不实现任意多 JVM/多实例装箱、动态迁移或共享宿主不足时自动另开 JVM。共享指代码复用，实例仍独立拥有缓存容量、请求状态和 Scheduler。

实例使用服务地址区分，同 IP 下用端口区分。控制、业务 RPC 和 metrics 地址之间的对应关系留在发现信息中，不向 Case 暴露额外的 JVM 身份。实例之间统一通过 TCP 通信，共 JVM 不另走 in-process 传输。Python 内部仍须保存进程与实例归属，以处理宿主退出和清理。

进程 env、heap 和资源限制属于 JVM；每个 Engine 的业务参数来自自己的配置对象。共 JVM 时不能用修改进程 env 或全局 System properties 的方式切换实例配置。原生 env 风格字段可以保留其名字和值，由实例加载器从显式配置读取。

停止 Engine 是关闭对象：拒绝新请求、终止活动、关闭自己拥有的 RPC/连接、释放资源并移除发现地址。共享宿主中的其他实例继续运行；最后一个 Engine 停止后宿主可保留至实验收尾，供后续扩容。独立模式完成停止后回收专属 JVM。对象清理不依赖 System.gc，也不承诺能强杀某个 Java 线程；关闭超时使实验失败并进入宿主清理。

Runner 在动作前登记预期停止，结合子进程退出和有界健康检查判断结果；预期停止仍需确认完成，不能吞掉所有异常。停止单个 Engine 不允许对应共享 JVM 意外退出。宿主意外退出映射到其全部实例，记录首个根因，停止剩余发压并清理实验；不自动重启或重放未知结果的写操作。业务请求错误按 Case 门禁判断，不与基础设施退出混为一谈。

### 发现文件和产物

进程启动材料和实例产物分目录保存。每个输出文件只有一个写入者，不让多个 Engine 共享同一个输出目录并各自覆盖 `endpoints.json`；宿主若汇总多个实例端点，由宿主统一写。ready/endpoints 采用临时文件加原子替换发布，部分启动失败要回收已经启动的对象。沿用 `components/<角色>-<序号>/` 保存每个实例的 config、effective-config 和 endpoints；共享宿主自己的启动材料、stdout/stderr 单独保存，不能把宿主日志冒充某个 Engine 的专属日志。聚合 topology/discovery 由 Python 单写，实例不得各自覆盖这份文件。

Python 维护一份权威的当前发现信息，增加、移除和重启换地址均更新该信息；历史端点保留在实验记录中。Master 等消费者能否动态加载发现变更由其自身实现负责，Case 可检查传播结果，框架不替消费者伪造路由更新。显式引用旧地址不能偷偷改指向新实例；一次运行内不复用已经退役的实例服务端口。

### 多 Client 的流量和结果（待讨论，本轮不实施）

本节保留讨论中的候选方案，尚未解决的 Client 生命周期、输入分片、并发和总 QPS 协调不能视为已定契约。本轮保持 NativeClient 原生收发核心，不实现播放器或据此固化 Case 接口。

一次流量任务开始前固定 Client 集合，Python 统一规划总 QPS、请求数、并发预算和数据分片，Java Client 读取并执行自己的计划。发请求、消费响应和逐请求计时留在 Java，Python 不进入逐请求发压路径。全局请求序号、输入顺序和随机采样不随 JVM 数变化；trace 时间和 warmup/ramp 不能在各分片重新从头计算。拆分输入避免每个 Client 重复全量解压和播放。

首版多 Client 不支持多轮会话，不在运行中增加参与者或自动重分片。停止默认作用于本次任务的全部 Client，也可指定部分；部分停止后其余 Client 保持原分配，不自动补齐被停止部分的 QPS。stop 停止新增发送，drain 等待已有请求结束；最终收尾覆盖原任务全部参与者，已停止实例的历史仍属于任务结果。

请求记录按 Client 分文件保存，保留全局请求身份、实际发送/首 token/终态事实和完成边界；跨实例查询按目标集合合并，分页不能冒充完整集合。任一 Client 基础设施失败使任务整体失败并停止其他参与者，不通过忽略缺失实例得到成功结果。保存已确认和结果未知的请求，不重放未知请求，也不补造完成记录。

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
2. 把队列、容量等待、选批和完成后的重新入队一起移入 Scheduler，不能只抽 `advance()`。用第二种明确的调度行为验证边界，再按选定版本实现 vLLM 策略；当前单请求 FIFO 不等于完整 RTP/vLLM 调度。
3. 将 Case 级接口与 Prometheus 查询迁移到 Python，保留实例协议、Java 指标产生及既有校验。迁移完成后删除旧查询服务和重复定义，不长期维护两条调用链。
4. 接通 Engine 两种放置方式和发现更新；Client 等设计确定后单独实施。每批改动同时补齐对应调用方、配置、测试和文档，避免旧入口继续承载另一套语义。

waiting/running 的统一含义已落实。批量提交/group 约束、vLLM 对齐版本和能力范围，以及完整批处理策略仍需确定；本节不把示意方法固化为完整接口。

验收关注可观察行为：两种 JVM 放置下缩容数量和流量总量一致；共 JVM 停单实例不影响其他实例；不同队列容器报告相同请求数；batch 内取消不损伤其他请求；取消/抢占无 KV 泄漏或提前复用；每批 Forward 只记一次；退役实例的历史仍可查询；任一基础设施失败不产生虚假的整体 PASS。共享宿主的实际延迟可以不同，不承诺两种放置方式性能相同。

## 当前代码职责

从本节起记录现有实现和命令。Python 接口与 Java 内部协议已分开；Engine 当前仍逐实例 JVM。共享宿主和动态生命周期属于上面的设计，不是本轮已经实现的运行能力。

| 目录 | 负责内容 |
| --- | --- |
| `python/mock_framework/config/` | YAML 解析、检查、点分配置展开、引用和可执行步骤编译。 |
| `python/mock_framework/execution/` | 初始化、统一 Step 执行、HTTP 传输、断言、记录和清理；`master.py` 负责 Master 原生配置接线，`processes.py` 管理进程和端点。 |
| `python/mock_framework/interfaces/` | 已接通的 Case 函数：等待、逐 Master 状态、公共指标与原始 PromQL。 |
| `java/interfaces/` | Java 内部端点的类型/校验及共享模型定义；不再定义或导出 Case API。 |
| `java/runtime/` | HTTP 接口绑定、参数和回执校验、调用上下文。 |
| `java/master/` | 真实 Master 的启动入口、生产状态到公共读取接口的适配、独立 ZK 启动入口。 |
| `java/client/` | 生产协议的调度、Fetch/Generate、真实客户端计时与取消。 |
| `java/engine/rtp/` | RTP 协议适配、P/D 调度策略、请求资源申请、P/D 交接与前缀树缓存。 |
| `java/shared/engine/` | EngineLoop/Scheduler 执行边界、RequestState 进度与终态/资源使用权、QueueSnapshot 请求计数。 |
| `java/shared/compute/` | Forward batch、公式、噪声、Decode 推进、停止模型和可取消计时。 |
| `java/shared/cache/` | 每个池独立的物理 block 容量、所有者引用与释放。 |
| `java/shared/metrics/` | Engine 指标、注册表和 Prometheus 导出；不提供查询代理。 |
| `tests/`、各模块 `src/test/` | Python/跨语言集成与 Java 测试；探针不作为正式业务入口。 |
| `scripts/` | 构建和检查入口。 |

共享指代码复用，不是不同实例共享资源。本批按职责重写 LocalPrefill/DecodeAdmission：它们保留原生校验、ALLOCATE/LOAD/GENERATE 协议状态和响应编码；RtpTasks/RtpErrors 在边界转换输入、状态和错误码。RtpScheduler 不依赖 gRPC、protobuf 或 StreamObserver，内部 Prefill 策略在 FIFO 入选时分配 KV，Decode 策略在 ALLOCATE 时申请 KV、等待传输、逐轮轮转并按需增长。容量/输出等待都不占用 Forward。

RequestState 是共享的请求进度、终态和资源清理机制。调度器决定何时申请和成功发布缓存；公共机制确保终态只通知一次、发布失败也尝试释放、取消请求不能再提交进度。Forward 持有资源使用权，终止请求不会提前释放仍被使用的块；EngineLoop 对执行对象做身份校验，晚到回调不能影响下一次执行。当前模拟 Forward 只有可取消计时，取消确认后同步回收使用权，不等待异步通知线程再回调。

P 的 retained 请求计算完成后变成 READY，释放 Forward 槽位但保留 KV，直到传输消费、取消或期限到达。D 的接入/传输与计算进度分开保存，完成后的 token/迭代数只由调度域推进。所有请求终态先清理资源、记录历史和指标，再通知协议适配器。

当前仍是每轮一个请求，P FIFO 与 D 轮转均有不经过 RPC 的调度测试；原生 P/D 测试继续验证端到端行为。共享资源使用权测试覆盖同批成员的独立终止，但生产路径尚无多请求 Forward batch，不能据此宣称已经实现完整组批、部分批次取消、分块 Prefill 或 vLLM。公共状态目前只落实现有 P/D 所需的生成进度；这些能力接入时再增加实际需要的状态。

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

每个 Case 独立目录组织自己的配置，数据集可共享。私有相对路径基于 Case 目录；显式提供共享根时，`data/`、`datasets/`、`views/` 基于共享根。编译不启动进程、不查询指标，也不检查数据内容。最新数据格式由 Java Client 读取；播放器和数据读取尚待实现，不提供旧格式兼容、转换器或生成器。

顶层为 `id`、可选 `name`、`describe`、`environment`、`phases` 和可选 `view`。phase 和 step 完全顺序执行；每个 step 直接写 `interface` 与 `args`，可选 `expect` 和 `within`；不再使用 call/check 外壳或 run 字段。phase 必须有正的 `timeout`。ID 全局唯一，引用必须来自前置步骤或允许的全局发现字段。

组件配置分为 `deployment` 和 `config`。原生业务配置原样保留；Mock 点分键在本层展开成对象，不展开原生 JSON 文本。重复键、YAML merge、循环引用、点分路径冲突均报错。`FLEXLB_CONFIG` 保留 JSON 字符串，编译器只检查它是对象且无重复键，业务字段由 Master 加载器检查。

Client 的 `deployment.replicas` 表示 JVM 数量，不能把总 QPS 乘以副本数。当前启动器不落实 CPU、内存或 heap 限制，配置这些限制会拒绝运行。配置编译通过不代表播放器、资源限制或其他未接通能力已执行。

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

省略 targets 等价于该操作域全部实例；显式空列表、重复地址或未知地址报错。全局发现信息在 step 开始时取值；已保存的 step 返回值保持原样，不跟着发现变化自动改写。Client 的播放、分布输入、多 JVM 计划及 run/stop/drain Case 接口暂不登记。

### 查询和断言

普通指标优先使用语义接口，YAML 不写 PromQL。当前支持 `latest` 和相对该 step 固定查询时刻的区间 `[-60s, 0s]`；`all` 依赖尚未确定的 Client 活动边界，目前拒绝。

`latest` 使用前 60 秒内最近两次样本的 irate；区间使用 increase。执行器固定 `query_time`，指标实现记录实际查询点、表达式和窗口，不重新取墙钟。`expect` 支持 eq、列表 count 或数值 gte/lte。eq 是结构相等，列表顺序参与比较；实例查询返回排序后的地址列表。普通数值直接返回 float，不包装为 `{value, unit}`；单位在接口定义里。逐 Master 状态返回 `{master服务地址: [engine服务地址, ...]}`。

原始 `prometheus/query` 只接受 promql，在固定 query_time 执行一次 instant query；历史窗口写进表达式。返回 `{samples: [{labels, value, sample_time}], query_time}`，不自动把它压成数值或对全部序列应用同一阈值。空 vector 保留为空；业务数值门禁使用语义接口，缺数据报错而不是补零。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。每类组件按 replicas 启动独立进程，编号 `<组件>-<序号>`；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。配方不再包含 bindings；目标选择和查询聚合由 Python 接口函数完成。

真实 Master 配方需要顶层 `zookeeper`；P/D 使用 `java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
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

Runner 是持续存活的进程所有者，负责启动、退出监视和最终清理。`Processes` 不识别 Master 业务接口，也不合并返回值。真实 Master 的状态查询由 `interfaces/master.py` 按 targets 逐实例调用，并保留每个 Master 的独立结果。

**Master 的 stop/start 尚未接通。** 后续必须由 Runner 侧调用适配器使用进程管理能力实现，不能依赖被停止 Master 自身的控制端口。届时要在发出停止前登记有意退出，启动时恢复该节点端点和监视；意外退出仍须失败。当前任何组件运行中退出仍是异常，不把 `/hook/pre_stop` 当成已经实现的进程停启或滚动重启。

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

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

## RTP 节点与公共校验

节点仅监听本机回环地址，独立 JVM 提供原生 gRPC、控制 HTTP 和指标端点。`endpoints.json` 写实际地址、支持的 RPC 和阶段范围；`/health` 只列已绑定的公共接口，当前 Engine 没有绑定 Case 查询接口，Python 查询直接访问 Prometheus。

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

原生字段支持 server_args 小写或 Whale 大写 env 名，同一字段同时提供两种写法报错；整数/布尔接受对应原生文本，其余原生内容保留，但不能据此宣称功能已模拟。当前要求显式 block size、max sequence length 和 test_block_num，只支持 `dp_size=1`、`dp_rank=0`、`max_context_batch_size=1`。不加载模型层/head 或 GPU 布局，不从部署内存推导 KV 块数。

RTP 请求公共支持范围集中在 `RtpRequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：多模态、adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 管理资源和状态，只有 GENERATE 后进入计算队列；一轮一个执行槽、轮转请求，尚不合并动态 batch。每轮增长 KV、执行公式等待后才提交输出；临时容量不足等待 allocation generation 变化，永久不足返回 602。

P 的内部 `PrefillTransfer.start()` 注册活跃来源并复用 LocalPrefill 计算。Forward 完成后 KV 仍由请求持有；只有网络 `Read` 成功交付后才完成 P 并释放。正式服务**只保留 Read RPC，没有外部 Prepare RPC**；测试从包内 start 建立来源或调用正式生成链路，不为探针注册额外协议。

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

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。请求只调度一次；要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向原 Master 尽力发一次有界 Cancel，不重试结果未知的请求；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。播放器、数据集读取、多 JVM 分片/总 QPS 协调、Client 指标与请求记录、traffic run/stop/drain 的业务实现仍待接通。

## 指标、采集和查询

EngineMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。业务标签只保留 role；Prometheus 为每个采集目标设置 instance=实际 RPC 服务地址，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| mock_engine_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| mock_engine_requests_finished_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| mock_engine_cache_lookup_tokens_total / mock_engine_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| mock_engine_forward_tokens_total | Counter，kind=input/compute/output | 成功完成 Forward 的 token；P 为输入、输入减命中、1；D 的 input/compute 为验证数，output 为实际提交数。取消的 Forward 不补成功计算。 |
| rtp_llm_model_forward_us | Gauge | 最近一个完成 Forward 的实际耗时，微秒；公共执行层收到通知并取得调度域锁后停止计时。 |
| mock_model_forward_planned_us | Gauge | 同一个完成 Forward 的计划耗时，微秒，包含公式、scale 和噪声。 |
| mock_engine_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| mock_engine_requests | Gauge，state=waiting/running | 当前等待/执行数。 |
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

组件 ready 后必须提供合法 metrics URL。Runner 原子更新 file discovery，固定节点全部成功抓取两次才执行 phase；动态目标增删尚未接通。每次独立 TSDB、job=mock，不要求用户定义实验 ID。当前 Python 查询只支持 Runner 托管 Prometheus，不接受组件内 prometheus.url/job 作为备用入口。

收尾等待每个目标在收尾开始后成功采样，再停止组件、ZK，最后 TERM Prometheus，超时才 KILL。最后采样失败、采集器异常退出会标记数据不完整并令实验 ERROR。启动失败/取消仍清理并保留现场。

prometheus/ 保存配置、targets、版本、日志、data/TSDB/WAL 和 collection.json；complete 仅说明采集边界及关闭成功，不代替业务指标或中途 up 判断。门禁 FAIL/主动取消也可以有完整采集。曲线不导出 JSON，数据不随收尾删除，后续回放服务在线查询；目前仍用默认保留策略，没有长期归档或正式报告服务。Grafana 大盘映射和 Client 指标尚未实现。

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
