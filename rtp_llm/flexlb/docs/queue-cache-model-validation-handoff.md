# FlexLB 排队与缓存复用模型验证交接

记录时间：2026 年 10 月 9 日，Asia/Shanghai。仓库：`/Users/hena/Documents/project/RTP-LLM/github-opensource`。记录时 HEAD：`7180432a690daef1a28a79a8024890dac0d73a60`。执行前重新核对代码，行号和实现可能变化。

当前任务是先验证模型与真实系统的差异。用户希望启动真实 Engine，让 FlexLB 自动积累数据，先验证排队预测，再验证缓存复用预测；之后才考虑利用模型改变路由、组批或资源配置。

目前只完成了论文讨论、代码检查和本地样例数据审计。尚未启动真实 Engine、采集实验数据、实现预测评估组件或验证预测准确率。本文之外没有为此修改产品代码。此前对话中的收益数字均为教学算例，不能作为验证结果。

## 给接手 agent 的任务

先阅读本文及适用的 AGENTS.md、FlexLB CLAUDE.md。检查真实 Engine 的可用环境、连接方式、请求入口和流量来源。围绕现有 FlexLB 与真实 Engine，完成自动采集、旁路排队预测、预测持久化和事后评分。第一阶段保持原调度行为，以真实等待请求数为验证目标，报告误差及适用范围。排队验证得到可接受证据后，再进行独立的缓存复用预测验证。

不要把 Mock Engine 的队列当成真实 Engine 的队列，也不要把论文假设、代码中已有字段或本次设计建议当成已经验证的生产事实。运行条件缺失时，先完成不依赖这些条件的准备，再向用户索取具体缺失信息。

## 用户已经明确的意图

- 用户希望理解理论对 FlexLB 的实际作用，要求能够量化，而非只讨论抽象概率或罗列落地场景。
- 主要排队指标是排队等待的请求数分布，不是所有未完成请求数。
- 用户认为流量、请求结构、实例数量都会变化，模型应依据近期真实数据更新。不能把理论状态永远递推成线上真实状态。
- 用户明确要求先比较排队建模值与实际值，再比较缓存请求模型的建模值与实际值。
- 用户不希望必须先手工整理历史日志；接受运行真实 Engine，由 FlexLB 自动统计数据的路线。
- 最终可能评估组批、路由、缓存热点分流、容量、拒绝和扩缩容，但这些不是当前阶段的交付目标。
- 用户有基础计算机知识，排队理论基础较少。沟通应短，每次说明一个问题，用具体数字和小表格。详细技术材料放在交接或实验报告中。

用户最后提出的是记录背景、交给另一个 agent 执行。本轮未创建另一个用户任务、未向其他任务发送消息，也未启动服务。接手后应在目标任务中继续执行。

早期用户提供过一张随时间明显变化的 QPS 图，用来质疑固定参数和单一长期分布是否适合线上系统。尚未取得该图对应的原始时序，也未确认它表示到达还是完成 QPS。用户曾表示可以提供更多数据，随后选择优先采用真实 Engine 运行中自动采集，不要把手工导出历史日志设为唯一前置条件。

## 两篇论文的作用与边界

### 论文一

[Effect of Parameters on Geoa/Geob/1 Queues: Theoretical Analysis and Simulation Results](https://www.scirp.org/journal/paperinformation?paperid=82709)，Lorente 与 Sánchez，2018。

原模型是单服务台、离散时间、固定到达批量 a、固定服务批量 b。每个节拍以概率 α 到来 a 个请求；忙碌服务以概率 β 完成一批。未满 b 个就不开始下一批。到达和服务采用文中的独立性假设，稳定条件为 aα < bβ。原文的状态 n 是系统内请求总数，包含正在处理的请求；等待请求数需要另外转换。

贡献是从状态变化规则计算队列分布。a、α 等是给定参数，论文不提供从线上数据自动学习它们的方法，也不预测未来请求的长度、前缀或缓存复用。不能直接据此证明某次 A/B 路由更好。

在 FlexLB 中可借鉴其状态演化方法，但真实系统的超时发批、多实例、不同请求工作量、缓存和 P/D 阶段需要显式扩展。原模型的几何服务时间、严格满批规则不应强行套用。

### 论文二

[Queueing Analysis of GPU-Based Inference Servers with Dynamic Batching: A Closed-Form Characterization](https://arxiv.org/html/1912.06322v3)，Yoshiaki Inoue。

模型考虑批大小相关的服务时间。理论中的简化策略是服务器空闲且有请求时立即处理全部等待请求；使用泊松到达等假设，在线性批耗时条件下推导平均延迟上界。它帮助描述组批效率与延迟的关系。

论文中的 GPU 推理实验不能直接代表 LLM continuous batching、P/D 集群或 KV 缓存路由；平均延迟上界也不是 P99 TTFT 保证。原始公式的参数符号与论文一含义可能不同，应避免混用。

### 已纠正的过度推断

- 两篇论文的公式不能直接拼接成 FlexLB 预测器。
- 论文一可以在给定到达规律后推算到达量和队列，不知道后面具体来哪些请求。
- 此前“发给 A 总共节省若干毫秒”的数字仅说明计算口径，没有线上数据支持。
- 缓存保留价值不能简单计算为未来访问次数乘以未命中代价：被淘汰后可能在首次访问时重建，后续不再损失。
- 计算耗时是 GPU 或 Engine 工作时间，不能直接与多个请求的等待时间总和相减。需要传播到请求时间线上再比较。

## 当前执行范围和分阶段顺序

| 阶段 | 工作 | 完成证据 |
| --- | --- | --- |
| 准备 | 明确队列口径、真实 Engine 环境、流量、采集来源及数据质量 | 可复现启动信息和真实观测样本 |
| 第一阶段 | 真实系统自动采集与排队模型验证 | 冻结的预测、随后实际值、误差及简单基线比较 |
| 第二阶段 | 缓存前缀复用预测验证 | 未见未来时生成的预测与实际前缀访问结果 |
| 后续阶段 | A/B 路由收益或组批策略评估 | 在可信模型和独立实验上的服务质量改善证据 |

当前优先第一阶段。可以提前记录第二阶段需要的前缀标识，以免重复采集，但不要把缓存预测器接进第一阶段并同时调整两个模型。

## 等待人数的定义

至少分别记录以下范围，最终报告不能只写含糊的 queue length：

1. Master 全局排队与等待派发的请求。
2. Endpoint WorkerBatcher 中尚未派发的请求。
3. Engine 已接收但尚未开始执行的本地请求。
4. 正在执行的请求，单独记录，不能混入等待人数。

这些集合可能存在所有权或同步时间重叠。全链路人数需要按请求身份、尝试次数、worker generation 和阶段去重；不能简单把所有 gauge 相加。P 与 D 分别建模，不能把同一个请求在多个阶段的记录误当多个业务请求。第一版若仅覆盖 Prefill 或 Master，标题与结论必须明确该范围。

对指定等待集合，守恒关系为：

```text
下一时刻等待人数
= 当前等待人数
+ 进入该等待集合的请求
- 离开该等待集合的请求
```

具体减项取决于集合边界：Master 队列在派发转出时减去请求，Engine 等待队列在开始执行时减去请求，全链路等待集合中的内部转移不改变总数。取消或失败导致退出集合时也要减去。ACK 不等于开始执行；重试和抢占后重新排队属于再次进入。取消不等于成功完成。

### 采样真值与事件真值

已有同步接口可以支持采样时刻的等待人数观测，但默认轮询可能漏掉两个采样点之间完整发生的短等待及峰值。

- 第一版可以验证相同观测时刻、相同采样分辨率下的预测人数及采样分布。
- 不能据此声称还原了精确的连续时间队列占用分布。
- 精确等待时长和连续队列曲线需要可靠的入队、开始执行、取消等事件时间戳。被抢占后多次等待也需要记录。
- 状态陈旧、采集失败、断连、重启、终态缺失必须显式标记，不能填零。
- 不同机器时钟偏差、RPC 传输延迟和采样完成时间要纳入观测误差。区分本地单调时钟耗时与跨机器墙钟时间。

## 已核查的代码能力与限制

以下为检查时的实现。执行 agent 应复核相关路径，并遵循已有所有权、版本和 generation 机制。

| 入口 | 已确认事实 | 对验证工作的意义 |
| --- | --- | --- |
| [GroupPlanner.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/planner/GroupPlanner.java:195) | 按请求上限、收集窗口和预测执行预算决定发批 | 模拟应复用实际策略语义；不是论文一的严格满批规则 |
| [DecisionPolicyConfig.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/config/DecisionPolicyConfig.java:10) | 支持 SINGLE、FIXED_WINDOW；代码默认请求上限 8、窗口 300 ms | 默认值不是线上实际配置，必须保存启动时生效配置 |
| [RouteTimelineProjector.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/projection/RouteTimelineProjector.java:65) | 基于冻结队列和已提交工作做串行时间投影 | 已有工作预测基础，不是持续引入未来到达的完整排队分布预测器 |
| [RoutingConfig.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/config/RoutingConfig.java:35) | 耗时估计支持 FORMULA、LEARNING，默认 FORMULA | 不能假设所有运行都启用了在线学习；需记录实际模式和参数 |
| [PrefillEndpoint.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/endpoint/PrefillEndpoint.java:668) | 批完成回调比较 predicted 与 actual；符合条件的样本进入 learn | 可复用耗时学习入口；需验证各派发模式是否都有有效样本 |
| [LearningPredictor.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/prediction/LearningPredictor.java) | 利用批大小、计算 token、复用 token 等特征学习耗时 | 尚不能等同于服务耗时分布；预测残差和不确定性需要另外评估 |
| [RequestRepository.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestRepository.java:85) | getQueuedRequestCount 按 owner 和阶段过滤；liveRequestCount 是所有活跃请求 | 两者不能混用；核对阶段范围和是否覆盖 WorkerBatcher |
| [HttpLoadBalanceServer.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-api/src/main/java/org/flexlb/httpserver/HttpLoadBalanceServer.java:203) | queue_snapshot 写出 snapshotActiveRequests，并返回文件路径与总数 | 总数不是纯等待人数；不宜靠频繁调用落盘调试接口充当高频采集 |
| [BatchSchedulerReporter.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/service/monitor/BatchSchedulerReporter.java:179) | 有 batcher 队列、派发等待和耗时等指标；部分 priority gauge 不补零 | 历史残留 gauge 不能直接求和；Master 派发等待不含 Engine 内部等待 |
| [ServerScheduleLatencyRecorder.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-api/src/main/java/org/flexlb/httpserver/ServerScheduleLatencyRecorder.java:145) | QPS 由记录窗口的首末时间与样本数计算；周期打印不等于固定窗口重置 | 应自行保留到达时间桶，不能把现成值误当最近一秒到达量 |
| [FlexlbServiceImpl.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-api/src/main/java/org/flexlb/httpserver/FlexlbServiceImpl.java:105) | arrival 记录位于可能的 master 转发之前 | 多 Master 汇总可能重复计数；schedule 完成也不是推理完成 |
| [WorkerRegistryConfig.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/config/WorkerRegistryConfig.java:15) | statusPollIntervalMs 代码默认 20 ms；缓存刷新单独控制 | 记录实际频率、延迟和丢样；不要仅根据默认值宣称采样精度 |
| [GrpcWorkerStatusRunner.java](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/sync/runner/GrpcWorkerStatusRunner.java) | 已有带游标与同步所有权的状态拉取链路 | 优先在已有观测后异步采集，避免另造干扰状态提交的同步路径 |

### 真实 Engine 的数据陷阱

[LocalRpcServer.cc GetWorkerStatus](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/cpp/model_rpc/LocalRpcServer.cc:493) 实际填充任务 request_id、phase、is_waiting、batch_id、prefix_length、input_length、waiting_time_ms、execution_time_ms、终态等字段，并提供实例状态及 KV 容量。

[model_rpc_service.proto](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto:564) 虽声明顶层 waiting_query_len、running_query_len、step_latency_ms 等字段，当前检查到的生产 GetWorkerStatus 路径没有填充这些汇总字段。其默认零不是实际无排队。available_concurrency 也不能假设已有真实填充值。

[RpcServerRuntimeMeta.h](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h:41) 有以下限制：

- 活跃任务在轮询时刷新 phase，但 waiting_time_ms 等初始数据不等于持续更新的当前等待耗时。结束时才通过运行快照更新等待、执行等耗时。
- running_task_info 列表并不表示全部已执行任务；需看 phase。同时可能包含无本地 stream 的取消控制 overlay，不能把它当真实等待工作。
- 完成数据按 latest_finished_version 增量返回。完成缓存容量定义为 1000 条，代码也定义了 5000 ms 清理超时；应核查清理调用与实际保留行为。该接口不是可靠持久化事件日志，断连或高负载可能导致缺口。
- Engine 的 execution_time_ms 是其计时口径下扣除等待后的 wall time，不能自动视为 GPU kernel 时间。核查 PD 阶段和 begin time 重置语义。
- Master 逻辑批次与真实 Engine 执行批次未必一一对应。不能把一个逻辑批次各成员的耗时简单相加当成实际批耗时。

因此，采集层要维护数据完整性、阶段解释和观测时间，而不是只把 protobuf 序列化后假设所有字段都可靠。

## 已有样例数据审计

文件：[trace_30min.jsonl](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/tools/online_eval/data/online_logs/trace_30min.jsonl)。

- 2,378,410 字节，8,332 条，均可解析。
- ts 是相对毫秒，范围 0 至 796225，覆盖 13 分 16.225 秒；文件名的 30min 不是实际跨度。
- 有 ts、il、ol、ttfb、total、pep、dep、bh、cached、priority 等字段。
- bh 在 6,486 条记录中非空，1,846 条为空；cached 全部为 0，不可当成经过验证的实际命中结果。
- 缺少真实队列时序、开始执行时间、完整实例状态和阶段事件，不能构造本次目标的线上等待人数真值。
- ttfb 包含排队、执行及其他链路时间，不能代替等待时长；尚未出首 token 的人数也不能当作等待人数。
- ol 为 0 的记录有 2,290 条。[trace_loader.py](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/tools/online_eval/online_eval/trace_loader.py:99) 的默认 skip 策略会去掉它们。不能无依据把这些请求当失败或直接删除，否则到达流量已改变。

该文件被 [online_eval README](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/tools/online_eval/README.md:237) 定义为脱敏回放形状。可考虑作为流量来源，但需要核对请求构造和复用关系是否保持。它不能提供真实排队的验证答案。

已有 online_eval 默认评估 FlexLB 与 Mock Engine。可复用其采集和报告思路，但运行该工具不等于已经完成真实 Engine 验证。某些性能模式会缩放 mock 耗时，也不能用于宣称真实容量。若用模拟器验证实现，结果必须清楚标记为模拟实验。

## 第一阶段实现建议

### 先确定运行条件

尚未提供真实 Engine 地址、认证方式、目标机器或 GPU、模型及 P/D 拓扑、启动配置、请求入口和流量方案。不要假设当前 macOS 工作目录能运行目标 GPU 模型。

先检查已有启动脚本、已连接环境和服务，复用用户现有部署方式。需要的核心信息是可用 Engine 与请求入口，而不是让用户再次手工整理一份历史数据。日志保留和实验参数应由工具自动记录。

流量需要覆盖低负载、接近饱和及突发恢复等情况。仅在长期空队列下测试会使“预测为零”获得虚假的好成绩。开始时保留原有策略，实验范围和负载按可用环境确定。

### 采集与预测评分分离

```text
真实请求进入 FlexLB 与真实 Engine
                    |
       原始请求事件和引擎观测自动留存
                    |
        只使用当时可见的数据生成预测
                    |
             预测结果立即持久化
                    |
        未来实际结果到达后独立评分
                    |
      后续轮次使用已经完成的历史更新模型
```

优先在已有生命周期和状态同步观测点异步采集。保持有界开销并报告丢记录情况；不能为了采集阻塞调度线程。若已有字段不够准确，补事件记录的必要性应明确到某项验证目标。

至少留存以下数据。具体文件格式可按现有工具组织，不必照本文名称创建多套框架。

| 数据组 | 必需内容 |
| --- | --- |
| 运行元数据 | run_id、代码版本、生效配置、模型、拓扑及变化、硬件类别、采样周期、时间基准、输入流量来源 |
| 请求 | 唯一请求与尝试身份、到达、阶段、派发、取消及失败原因、长度和优先级 |
| 工作状态 | worker generation、角色与 DP rank、任务与批次身份、等待或执行阶段、观测时间、状态新鲜度 |
| 执行观测 | 可获取的真实等待与执行耗时、终态、样本有效性；批次与请求不能混淆 |
| 预测 | 生成时间、输入截止时间、起始状态、预测时域、模型版本、预测分布与摘要 |
| 评分 | 对齐的实际观测、有效时间覆盖、缺失原因、各误差指标及对应基线 |

原始 prompt 文本通常不是必要数据。缓存阶段需要稳定的前缀 key 及命名空间，而不是把敏感请求内容写入日志。未命中请求也必须保留访问标识。

### 排队模型怎么更新

运行规则来自实际调度实现；从历史数据更新的是到达过程、请求工作量、执行耗时及其波动。不能只把 QPS 拟合成一个固定 α 后永久使用。

可先比较简单的近期窗口与加权历史方案。窗口长度 W、预测时域 H、更新步长和采样间隔分别记录；对话中的 10 秒、10 分钟仅为例子，没有形成固定需求。依据历史滚动验证选择参数，并保留最终未参与选择的验证时段。

每次从最新真实状态开始，保留已有积压和已提交工作。未来到达与执行不确定性通过多个场景形成预测分布。不稳定或持续增长的负载下应报告积压增长，不能强行求一个稳态分布。

若要复现论文一原模型，可另做其假设下的概率推进和随机模拟一致性检查。这只验证数学或实现，不代表原模型对 FlexLB 有效。论文表格或前文演示数字不能未经复算直接作为正确答案。

### 将两类验证分开报告

1. **已知到达流量的回放验证**：使用事后已知的实际到达序列，检查处理规则、服务模型能否解释排队。它回答系统演化模型是否可靠，不检验到达预测。
2. **真正的在线预测验证**：在预测时点只使用过去和当前信息，预测后续到达并推进队列。它包含流量预测误差，才对应用户要求的提前预测能力。

若输入了验证段的真实执行耗时或缓存命中，这属于额外的条件诊断，应单独标记，不能混进在线预测成绩。第一阶段没有缓存预测器，并不意味着可以偷偷使用未来真实缓存结果。

## 排队验证指标与交付

使用时间顺序滚动验证，不能随机打散时间记录。预测生成后冻结；验证段用于评分，评分后才允许用于下一轮训练。只有完整结束的历史窗口可以提供训练标签。

至少报告：

| 指标 | 具体口径 |
| --- | --- |
| 等待人数平均绝对误差 | 相同有效观测时刻上，预测人数与实际人数之差的绝对值再取平均，单位为请求数 |
| 系统性偏差 | 平均预测人数减平均实际人数，指出持续高估或低估 |
| 高峰低估 | 在预先定义的高负载时段或实际高队列样本上，漏估多少请求、峰值时间差多少 |
| 分布差异 | 相同窗口和采样口径下的等待人数直方图与累计分布；可用离散 Wasserstein 距离表示相差的人数尺度 |
| 区间覆盖率 | 若预测提供 90% 等区间，统计实际落入比例与区间宽度，避免无限放宽区间 |
| 数据质量 | 有效样本比例、丢记录、终态缺失、状态陈旧、时间对齐误差 |

至少比较“未来人数保持为当前值”的简单基线；按需要增加近期均值等基线。必须覆盖有积压和突发的时段，分角色、节点或负载水平报告，避免总体平均掩盖热点误差。

动态流量下称为滚动预测或窗口内占用分布。只有对近似稳定、满足稳定条件的时段，才讨论长期稳态分布。把不同时段混成一张直方图，不能证明动态预测准确。

尚未约定业务可接受的误差阈值。应预先提出可解释的阈值与理由，或先完整报告结果供用户判断。不能看到结果后调阈值，也不能仅因优于基线就宣称足以控制路由。

第一阶段应交付可复现的启动与采集命令、模型配置、原始记录、冻结预测、评分结果、预测和实际队列曲线、分布图、误差摘要及失败时段说明。数据不足或真实 Engine 不可用时，明确停在哪一步，不用伪造真值补齐结果。

## 第二阶段缓存复用模型

排队验证之后再推进本阶段。首先预测全局请求是否会再次使用某前缀，再结合路由和节点状态计算命中。全局复用与节点命中是两个问题，不能直接将历史某节点命中率学习成缓存固有价值。

最低输入包括所有请求的到达时间、模型或版本等缓存命名空间、可复用前缀或块链、长度；后续命中与淘汰成本评估还需要各节点缓存成员、占用、引用状态、淘汰和重建信息。共享前缀和父子块关系要按真实缓存语义处理，不能重复计算同一段收益。

第一版可估计“未来 H 秒内是否再次访问”的概率，使用近期访问次数、距上次访问时间、请求类别等特征，以相似历史情形的真实复用比例作为简单基线。冷启动前缀和样本少的类别需保留不确定性。

该概率不足以模拟所有排队影响。多次淘汰和重建还需要访问次数与时间，第一版可从相似的历史请求片段采样完整场景，保留突发和前缀访问相关性，而不是对所有请求独立随机生成。

滚动验证时，未来 H 秒的复用标签必须等 H 秒实际结束后才能进入训练。检验内容为：

- 复用概率校准，例如预测约 70% 的样本实际是否接近 70%。
- 概率误差，可用 Brier score，即预测概率与 0/1 实际结果的平方差均值；与近期频率等简单基线比较。
- 热门与冷门区分能力，不能只用总体 70% 覆盖率宣称准确。
- 访问次数、时间和复用 token 的预测误差，以及热门前缀迁移时的表现。
- 复用预测通过后，再独立检查真实节点命中、淘汰重建和额外计算预测。

预测复用正确并不证明路由收益正确。离线历史只发生过一种路由；改变路由后缓存与排队都会变化，最终仍需可信回放或受控对照实验确认服务质量改善。

## 后续路由价值问题的上下文

用户提出的典型场景是：B 是缓存命中高但压力大的热点机器；把请求送 A 可以缓解 B，却可能挤占 A 的缓存，影响 A 后续请求。

可用的统一比较口径是，在相同初始状态、同样外部请求场景和相同评价范围下，比较 A/B 两种选择导致的请求延迟和超时：

```text
选择 A 相对 B 的延迟收益
= 当前请求提前的时间
+ B 上其他请求提前的时间
- A 上其他请求推迟的时间
```

最后一项包含新增工作、缓存淘汰、额外重算及其排队传播。实际应逐请求比较两条完整时间线，避免将重复的影响多次扣减。先比较按业务优先级约定的超时，再看总体延迟与资源成本；拒绝和系统导致的取消不能被简单排除以美化结果。评价窗口末尾留下的未完成工作和缓存变化也不能忽略，否则策略可能只是把代价推到窗口之外。

这是未来扩展的评价方法，不是论文一已经给出的路由算法。当前还没有足够数据计算其线上真实收益。

## 接手时的检查顺序

1. 复核代码版本、适用说明和现有启动工具，确认真实 Engine 运行位置及连接方式。
2. 明确第一阶段等待队列覆盖范围，检查身份去重、终态完整性和时钟口径。
3. 接上最小自动采集链路，先用真实小流量验证记录是否解释得通。
4. 保存原调度配置，建立简单基线与旁路预测；先冻结预测再评分。
5. 采集覆盖不同负载的实验，分别报告条件回放和真正在线预测误差。
6. 交付排队模型的实测结论，再决定是否推进缓存复用模型。

执行过程中，给用户短更新：当前验证到了哪里、发现什么、下一步需要什么。不要继续用大量假设算例代替真实验证结果。

## 参考入口

- [FlexLB 仓库说明](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/CLAUDE.md)
- [FlexLB Sync 说明](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/CLAUDE.md)
- [online_eval 工具说明](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/tools/online_eval/README.md)
- [Mock Engine 说明](/Users/hena/Documents/project/RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-mock-engine/README.md)
- [时间序列滚动验证方法](https://otexts.com/fpp3/tscv.html)
- [缓存复用与淘汰建模研究](https://www.usenix.org/conference/atc16/technical-sessions/presentation/hu)：可参考其问题划分，不能直接当作 LLM KV 缓存验证证据。
