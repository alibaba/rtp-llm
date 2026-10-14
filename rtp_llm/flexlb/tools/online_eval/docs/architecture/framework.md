# 框架结构

```text
config/scenarios/*.yaml   身份、环境、输入、阈值与视图选择
        ↓
cases/<case>/program.py   步骤、分支、检查与能力声明
        ↓
scenario compiler        校验并生成有类型的执行计划
        ↓
parallel runner          冻结候选执行计划，固定 lane 租约，动态领取实例
        ↓
stage executor           执行固定步骤并清理 Master / Mock
        ↓
frozen evidence/metrics  原始证据、指标定义与完整序列
        ↓
case analysis/report     明确判定后装配 HTML bundle
```

## 格式版本

版本字段采用 `<format>_schema_version`，字段名标明格式，整数标明该格式的版本。各格式独立演进，数字不表示项目版本或代码是否过时。

| 格式 | 版本字段 | 当前值 |
|---|---|---|
| case YAML | `case_schema_version` | 2 |
| 指标集合 | `metric_plan_schema_version` | 5 |
| 报告视图 YAML | `report_view_schema_version` | 1 |
| Suite 与 CI 策略 | `suite_schema_version` | 3 |
| mode profile | `mode_profiles_schema_version` | 1 |
| Python 内部 program document | `program_schema_version` | 1 |
| 冻结执行计划 | `execution_plan_schema_version` | 1 |
| 编译 / 列举的实例清单 | `instance_catalog_schema_version` | 1 |
| 子执行器结果 / 父 runner 汇总 | `scenario_results_schema_version` / `run_summary_schema_version` | 1 |
| runner 计划 / 端口租约 / 耗时缓存 | `runner_plan_schema_version` / `lease_schema_version` / `timings_schema_version` | 1 |
| 冻结指标归档 | `metrics_schema_version` | 1 |
| 报告图表 / bundle manifest / 分析封装 | `report_spec_schema_version` / `report_manifest_schema_version` / `report_analysis_schema_version` | 1 |
| 报告运行信息 | `run_meta_schema_version` | 1 |
| 门禁冻结 manifest | `gate_result_schema_version` | 1 |
| workload 证据 / 分析 | `workload_evidence_schema_version` / `workload_analysis_schema_version` | 1 |
| 请求与引擎关联证据 / 客户端请求记录 | `request_engine_evidence_schema_version` / `client_record_schema_version` | 1 |
| 观测快照 / 冻结观测窗口 | `observation_snapshot_schema_version` / `observation_window_schema_version` | 1 |
| 性能证据 | `performance_evidence_schema_version` | 2 |
| 性能分析 / cache 分析 | `performance_analysis_schema_version` / `cache_scale_in_analysis_schema_version` | 2 |
| 实验归档 manifest / 请求计划 manifest | `archive_manifest_schema_version` / `request_plan_manifest_schema_version` | 1 |
| Master 模板 / 流量保真度分析 | `master_template_schema_version` / `traffic_fidelity_schema_version` | 1 |

case YAML 通过 `program: default` 生成内部 program document；后者包含 stages 和 typed reference，不是旧用户 YAML 的兼容入口。读取方校验本格式的版本字段及整数类型，拒绝其他格式或旧通用版本字段，不能根据文件名或版本数字猜测格式。报告生产器组装的内存图表可以暂不带版本字段，由 bundle 写入器明确标记；落盘文件和离线读取必须带有相应字段。

真实采集输入、固定合成流量 profile 及 Java 共享性能 / 流控协议保留它们原有的 `schema_version`，由相应协议读取器校验，不进入 case、执行计划、指标和报告的读取边界。它们的语义与准入见[数据目录](../../data/README.md)。调整这类字段需要同步固定输入及协议双方。

版本字段的名称属于格式身份；字段或语义发生不兼容变化时升级该格式，并同步其读写双方，不为数字一致整体替换。配置分层见[新增 case](../development/adding-cases.md)。

## 编译与执行边界

Master discovery pass 对缺失引擎的退役锁只做非阻塞尝试；锁忙时由下一轮重试相同 generation。同一 Master 只有一轮 discovery 拥有缺失引擎的退役批次，其他轮次跳过这批清理，继续提交存活引擎的状态轮询，防止批量索引清理占满同步执行池。批次结束后释放所有权，过期判断、generation 校验和正式退役仍在持锁后执行。

编译按 document、variant、profile 三层展开，读取声明、preset 和实现源码，但不启动服务。环境声明与标量合同由 `environment_declaration` 维护，Master patch 由 `master_patch` 校验，preset、profile 渲染、偏离检查和快照冻结由 `environment_resolver` 负责。

内部 program document 使用显式 `implementation` 字段携带实现与配置来源；普通 mapping 复制和 JSON 序列化必须保留该字段。编译后的 `metadata.kind` 与 `execution` 是执行策略的来源，`test_kind`、`collection_profile`、`workload_runtime` 是 plan 生成并复核的一致性投影。suite 只选择实例，不补齐执行策略。

端口窗口准入、探测与锁由 `adapters.port_windows` 提供。`admit_port_window()` 返回包含选定端口、来源、逐 lane 状态、诊断及锁所有权的 `PortAdmission`，不改写参数或进程环境，不打印或退出。父编排器在上下文作用域内持有窗口锁与输出锁，规划失败、执行失败和正常返回都释放所有权；子进程端口环境从选定租约生成。命令入口负责准入错误的退出码和诊断。子结果的身份和嵌套证据由 `scenario.result_validation` 复核，耗时缓存由 `scenario.timing_store` 维护。编译、冻结计划读取、lease 校验与父进程结果接收各自保留边界校验，共享字段和参数规则。

后台采样由 `runtime.sample_worker` 管理预算、取消和线程退出；服务协议与 epoch 数据源由 `adapters.observation_sources` 提供，`adapters.observation_window` 在线程 join 后冻结 cohort。action 负责参数、资源归属、检查和产物发布。

## 代码归属

| 目录 | 内容 |
|---|---|
| `cases/<case>/` | 该 case 的 program、action、输入合同、纯分析、指标生产、报告与离线重判 |
| `case_api/` | 作者使用的构建、注册、输入与取证扩展合同、门禁插件 |
| `scenario/` | YAML 声明与编译、静态校验、实例调度、基础 action、资源句柄与取证作用域、阶段执行、清理及最终状态裁决 |
| `runtime/` | deadline、进程、网络、通用端口区间检查、有界采样与时钟计算 |
| `adapters/` | 服务与客户端协议、请求账本、Fetch 流消费、环境生命周期、Java mock 资源布局及具体取证适配 |
| `workload/` | 持续测量策略、请求与 engine 证据关联、采集有效性 |
| `analysis/` | 可复用统计、标量裁决、冻结指标检查桥接及输入保真度计算，不定义某个 case 的门槛 |
| `diagnostics/` | 离线诊断的文件发现、输入审计、拟合与生产生成器编排 |
| `monitoring/`、`reporting/` | 指标查询/归档与图表/spec/bundle 的公共能力 |
| `traffic/`、`artifacts/` | 流量构造与播放；制品管理、启动记录及冻结证据格式读写 |

归属由语义决定，不由文件名或当前调用数决定：包含特定阶段、门槛、指标输出或报告身份的代码属于 case；同一合同能被不同 case 使用的能力才放在公共组件。注册表可以指向专属实现，公共执行器不按 case 名分支。

文件摘要统一由 `file_digest.sha256_file` 计算；字段与格式头校验分别由 `input_contract`、`schema_contract` 提供，执行实现身份由 `implementation_identity` 描述。这些基础件不依赖业务包。`analysis.checks` 提供通用检查和裁决，`analysis.metric_checks` 负责冻结 MetricStore 的检查桥接。

运行底座只依赖标准库和包内机制。`runtime.network` 提供通用 HTTP 传输和端口探测，Master 的 info、inflight 控制路由由 `adapters.master_control.master_url` 统一定义。具体客户端等待调用 `runtime.startup.wait_ready`，协议状态与进程退出码由 `adapters.flow_control.wait_flow_started` 解释；`artifacts.startup.startup_phase` 使用同一 deadline 保存启动耗时。`runtime.resource_plan` 检查通用端口区间，`adapters.java_mock_resources` 定义 Java mock 的保留端口、累计扩容容量、子进程环境及租约投影。

集中归属不合并职责。`analysis.py` 根据显式输入计算，不访问网络、进程或报告；`actions.py` 操作现场并取证；`metrics.py` 发布声明的指标；`report.py` 展示既定结论，不能隐式重判。门禁动作先冻结独立 result 和校验 manifest，再返回共用的 `CheckResult`。数值投影在最终遥测导出后执行，报告在统一的 renderer 阶段生成一次，不重新裁决或改写冻结判定。

## Mock RPC 与缓存同步

Mock cluster 的 Netty 线程只处理传输，RPC 方法由同一 cluster 所有的有界执行器运行。缓存 RPC 与普通、状态 RPC 使用独立队列；队列饱和时拒绝执行，不让调用方或 Netty 线程代跑。动态引擎复用所属 cluster 的执行器，单引擎移除不关闭共享线程池；传输线程组结束后回收执行器。

Device 与 Memory 缓存维护不可变的 keys 成员快照，仅成员增删和清空使其失效，引用计数和 LRU 访问不重建集合。Memory 快照只用于遍历发布，以不可变列表保存唯一成员，避免在缓存锁内再构建 HashSet。缓存 RPC 仍返回完整 keys，包括请求方已经持有同一版本的情况；响应字段及 keys 快照不变时复用 protobuf 和编码字节，每次发送使用独立读取游标，保留 gRPC 的 KnownLength / Drainable 快速 framing 接口。容量或 keys 变化使响应缓存失效，不能以响应复用为由隐藏容量变化。

Master 在所属 worker 代次与索引版本检查后，才在回调执行器中复制 keys 并更新索引。成功取得同版本、已经索引的 Prefill 响应时，向动态缓存刷新间隔反馈零差异；旧版本、失效代次、尚未成功索引的响应和 RPC 失败不产生零差异反馈。刷新策略沿用配置的上下限，不能只以出现变化的样本估计刷新频率。

## 执行与资源契约

每个 case 有 `default` program，变体只追加测试点。YAML 保存数据，不能写 action、通用 `$ref` 或任意表达式；具名观测窗口只能选择 program 声明的时间输出，不能编排步骤。配置层次和 action 边界见[新增 case](../development/adding-cases.md)。

资源句柄绑定环境代次。重建环境前清理旧消费者及进程，旧句柄只能显式作为历史证据读取。动态 worker 添加保留尝试预算；移除不返还容量预算。端口租约由父 runner 拥有，子执行器启动前核对范围与最大拓扑。父 runner 将完整 compiled instances 写入 `execution-plan.json`，记录规范化内容摘要与代码、运行配置的文件摘要。子执行器使用父进程提供的摘要验证计划，再验证当前依赖与租约，直接执行冻结步骤，不重新编译场景 YAML。规划后改变场景文件不改变已经冻结的参数；代码或运行配置变化使启动失败，必须重新规划。

`ResourceScope` 持有句柄、环境代次、取证适配器和清理栈；`RuntimeContext` 绑定输出引用及实际事件；`StageExecutor` 负责阶段分发与输出检查；`RunFinalizer` 负责证据和报告预算、检查点与结果提交。取证 exporter 返回 `case_api.extension_contracts.ResourceEvidence`，客户端的具体转换位于 `adapters.resource_evidence`，公共资源作用域只校验合同。deadline 位于运行底座。gRPC 客户端持有 channel，stream 持有消费线程，Mock 控制能力只使用所属环境的 HTTP 地址，不另建资源所有者。

阶段使用剩余 deadline，清理使用独立预算并逆序执行。同一资源回调内的独立清理操作使用 `cleanup_all`，逐项尝试并汇总错误，诊断采样器失败不得阻断进程回收。启动部分失败也要回收已登记资源；未退出线程、未回收进程或清理异常不能成为 PASS。普通失败阻断依赖步骤，独立观察应放在最终判定之前。finding 只能声明具体检查，不豁免证据、超时和清理。

观察预算覆盖预热、流程等待和取证窗口；排空预算覆盖停止发送后等待请求终态；分析预算覆盖门禁所需的取证和判定；报告发布使用独立的 `execution.report_timeout_s`。需要单独分析阶段的负载在 `procedure.analysis_timeout_s` 声明预算，不能由客户端请求超时推导。

请求发出、Schedule ACK、Fetch、业务 FINISHED、取消和资源释放分别取证，不互相推断。服务启动及诊断 API 成功应答不代替业务完成。具体协议见[请求生命周期](request-lifecycle.md)。

指标和报告使用冻结定义与序列，不通过 debug API、日志或文件自动兜底；缺失来源显式报错。采集、门禁与交付规则分别见[指标契约](metrics.md)、[结果与指标](../development/results.md)和[报告契约](reporting.md)。

## 现场观测与冻结证据

持续时间和轮询由阶段的单调时钟控制；epoch 秒用于关联客户端流水和 Prometheus。`RuntimeContext.start_observation` 记录观测开始事件，并把两种时钟的锚点交给 `runtime.observation.ObservationClock`；样本传入当前单调时间，用同一锚点加时间差生成 `epoch_s`、`monotonic_s`、`elapsed_s`。已有证据中的毫秒字段与相对 `t` 是格式投影，不形成第二个时钟来源。

`RuntimeContext.record_event` 是事件时间的权威记录。证据内事件保存同一记录及相对时间投影，供独立离线重判；stage 的起止记录描述执行生命周期，不能代替采样点或业务事件。具名窗口声明负责边界，测量实现负责 cohort 与统计口径。

采样复用 `poll_samples` 和 `SampleBudget`。`observation.capture` 显式声明样本数与字节上限；超限保留已有现场和错误，证据不完整时不得判 PASS。先保留已获得的失效现场再检查流量状态，允许诊断停止原因；错误样本不能被当作有效门禁证据。HTTP 请求复用 `runtime.network`，timeout 受当前 deadline 或可停止的短请求预算约束。

门禁的获取、终态补全和发布均通过 `artifacts.json_io.write_json` 原子更新同一证据文件，崩溃时保留最近的完整版本。公共获取信封包含格式版本、clock、criteria、samples、errors、provenance；业务 payload 保持所属格式的字段与语义。格式版本描述结构，`measurement_policy` 描述统计口径，两者独立。历史时钟字段只由 `artifacts.gate_inputs.evidence_origin` 转换；缺少锚点且没有样本时失败，不补零。

`run_provenance.collect` 收集实际环境代次的启动输入；门禁复用其严格获取路径并冻结业务源码摘要。`adapters.java_flow.evidence_environment` 定义可比较客户端环境中的 run-local 字段排除规则。源文件位置变化只改变摘要，不改变历史证据的重判口径。

停止发流和排空由 program 显式编排；门禁仅读取所声明的测量证据，全量 Prometheus 查询归档由采集器清理统一完成，公共 finalize 物化指标一次；门禁 handler 保存判定，最终归档后投影指标，再由注册 renderer 生成报告。采集或归档失败保留为 errors，产生 INVALID/ERROR；分析或判定保存异常成为 stage ERROR，报告交付错误单独记录。门禁证据句柄使用 `gate_evidence`，与基础快照分开；historical 表示可以显式读取旧环境证据，默认读取仍拒绝环境换代。功能程序没有持续采样需求时不创建空观测组件。

跨阶段后台工作由 context 登记的资源对象持有，action 只负责参数校验、登记、调用与输出。客户端启动、finish、证据快照与 cleanup 使用同一对象，不另外包装第二套生命周期。Java 发流控制复用 `adapters.flow_control` 的原子命令、身份校验和排空计数；发现方式及业务证据校验留在所属能力。独立资源的清理复用 `cleanup_all`，启动失败与清理失败同时存在时保留启动错误及其原因链。

Java 流量控制使用 `FlowState`、`FlowStatus`、`StopCommand` 和 `DrainResult` 表达状态、命令及最终结果，JSON 只用于文件协议和证据输出。Java 按 `STARTING → SENDING → DRAINING → DRAINED / INCOMPLETE` 转换，在进入排空时冻结 `sending_end_reason`：`stop_command` 表示接受控制命令，`natural` 表示到达自身时长或输入边界；排空期间的迟到命令不能改写结束原因。

控制观察先查询进程退出码，再读取状态文件，确保已退出进程对应最终状态。生命周期等待只读轻量控制状态；明确需要请求计数的 checkpoint 才读日志。drain 确认进程退出后读取稳定日志，校验提交、启动、终态计数和命令确认。成功结果不可变，重复 drain 或 cleanup 直接返回，失败不缓存并允许继续收尾。stop 与自然结束并发时，显式自然结束可以替代命令确认；未知结束原因、错误确认、非正常退出或不完整请求均失败。

## 收尾与持久化

`execution.timeout_s` 约束阶段执行，`cleanup_timeout_s` 单独保留资源清理预算。workload 清理后，证据导出与分析受 `finalize_timeout_s` 约束，报告生成受 `report_timeout_s` 约束；所有实例的压缩工件交付受 `archive_timeout_s` 约束，父 runner 的进程预算包含这些收尾时间。执行、取证或报告失败后仍尝试归档已有证据；归档失败只影响交付状态，不修改门禁与运行有效性。同步工作受主线程中断式期限保护，资源导出同时检查合作式 deadline。 Prometheus 原始查询归档使用 teardown/cleanup 的剩余预算，并预留 `collector_shutdown_s` 给独立的进程关闭期限；归档失败仍尝试关闭、升级终止和回收进程，保留查询错误及原有运行有效性判定。

清理结束后立即原子保存 `result.json`，每段收尾前后保存检查点。未完成的检查点使用 `status: FINALIZING`，不能作为终态 PASS；`execution_status` 保留阶段与清理的结果，`finalization` 保存各段预算、状态、耗时和错误。最终 `duration_ms` 包含执行、清理、证据和报告。证据阶段失败使 runtime validity 为 INVALID，并阻止报告阶段；报告失败记录 report status，保留已有 runtime validity 和独立门禁结果。整体执行器结果仍报告交付失败，调用方不能把未完成的交付当成成功。

细项检查使用 `CheckResult` 的 id、status、actual、expected、detail、evidence；`gate_checks` 汇总有效阈值检查与完整性错误。ERROR 对应 INVALID，FAIL 对应有效观测越过门槛，SKIP 表示不适用，WARNING 表示 advisory。持续异常、窗口归属和请求 cohort 的计算属于各 case，公共组件只统一结果协议与聚合。

## 结果状态

`result.json.outcome` 分别保存 `execution`、`gate`、`validity` 和 `delivery`。流程异常、业务未达标、证据不足与交付失败分别记录，不用一种错误覆盖其他事实。`scenario.outcome.RunOutcome` 是最终状态的唯一计算规则，父 runner 使用同一规则验证子结果一致性。交付失败仍使任务失败，但不能改变已冻结的门禁 verdict；finding 只匹配明确失败的检查。

没有执行有效检查，或只有 `SKIP`，不构成 PASS。`WARNING` 是实际完成的建议性检查；证据不足始终为 ERROR，不受建议性策略豁免。可缺失的测量输出必须显式声明 `nullable_number`，只有伴随证据 ERROR 时才能输出 null；成功结果仍要求实际有限数值。

## 声明与解析结果

数据型 case YAML 在入口解码为 `CaseDeclaration`、`VariantDeclaration`、`ExecutionPolicy` 和 `MonitoringPolicy`。case 的业务参数由所属程序校验，公共构建层不通过任意模块属性寻找入口。`CaseDefinition` 声明全部能力，注册器按目录发现并校验冲突。

环境覆盖通过统一规则合并：配置字段整体替换，profile 按名称及字段继承，列表与完整模型整体替换。编译阶段将最终 Master JSON、preset 性能数据及运行选项冻结为带 SHA 的 `EnvironmentSnapshot`。运行时使用冻结数据投影进程参数，不重新读取 preset 或应用配置默认值。

## 执行领域对象与序列化边界

`ScenarioCompiler.compile()` 返回不可变的 `CompiledInstance`，其中阶段为 `StagePlan`，公共请求、句柄引用和比较参数分别使用具体参数类型。`ExecutionPlan` 组合实例与依赖摘要，`ExecutionPlanCodec` 负责落盘、完整性验证和领域解码。摘要一致之外，加载时还检查身份、预算、资源范围和环境快照的一致性。`compile_scenarios()`、`freeze_plan()` 和 `load_plan()` 是清单及文件格式的投影入口，不作为运行时状态容器。

`ScenarioRunner.run()` 接收编译实例并返回 `RunResult`。`StageExecutor` 返回 `StageExecutionSummary`，阶段、清理和交付状态具有独立的结果类型。`RunOutcome.evaluate()` 使用 `OutcomeEvidence` 统一判定执行、gate、有效性和交付；报告失败不重写阶段检查。`RunFinalizer` 在证据和报告各自的 deadline 内调用适配器，并在阶段开始、完成或失败时保存检查点。

环境声明、Master patch 和解析结果分别由 `EnvironmentDeclaration`、`MasterPatch` 与 `ResolvedEnvironment` 表达。`EnvironmentResolver` 读取一次 preset，校验所有具名 profile patch，再为选定 profile 渲染。性能与缓存 patch 在编译时合入最终快照；backend 仅绑定租约和进程启动参数，不重新读取 preset 或解释性能 patch。简单字段的类型、默认值和范围声明在领域字段 metadata，环境词汇从字段定义生成。

case action 保留自身参数和输出的所有权。公共 core action 使用具体参数对象；扩展 action 的冻结传输由 `AdapterParameters` 承载，在进入注册适配器时转换为该适配器的参数格式。报告视图、来源信息和扩展证据作为序列化边界数据保存，不通过公共层识别某个 case 的字段名。

回放配置由 `PlaybackConfiguration` 表达，`PlaybackConfigurationCodec` 校验 uniform/true-ts、圈数和请求身份策略。`normalize()` 生成 Java 环境变量；Java 客户端控制发送与循环，发送端的 `PLAYBACK_V1` 快照用于离线重建和一致性验证。

lineage codec 分别解码 v2 的块对齐格式与 v3 的精确长度格式，返回 `LineageModel`。`LineageEvent` 明确区分 token 长度、共享完整块和额外标签预留；`LineageExpander` 管理父路径及标签预算，`LineageProjector` 校验固定输入并生成请求文件。过滤作用于展开后的事件，保留原始索引和父路径，不改变请求身份或随机采样索引。压缩格式中的版本号描述数据语义，不表示 Python 类的代次。

启动等待共用 stage 的 `Deadline`，不会在端口、服务、拓扑和首采样之间重置预算。单/双 Master 共用进程存活、HTTP、同步与连续拓扑稳定检查；客户端创建进程后必须收到同一 run/group/phase 的发送状态确认，才能进入后续观测。`startup.json` 记录环境、流量输入、监控及发送确认的分项成本；各 case 的固定预热、条件预热和故障稳定期仍由其 YAML 与 program 决定。

严格文件读取由 `document_io` 提供，共享异常定义在 `document_io`。该模块不依赖 program 装载或编译总装；`scenario.loader` 只负责读取 case 数据、应用实验参数和调用注册程序。`scenario.references` 统一 `$ref` 语法及先前 stage 的输出类型解析，契约和冻结参数读取直接复用该规则，不回调 stage 编译实现。

`scenario.backend_spec` 将冻结环境计划转换为受租约约束的启动规格。`EnvManager` 通过显式的环境创建与进程变化回调交付所有权；外部有界清理负责回收时，必须同时提供两个回调。启动和重启失败也记录已产生的进程，保留部分环境供独立清理预算回收。启动负例使用单独的 mock 规格，避免无效 Master 配置阻止 mock 启动或临时修改 live environment 的规格。Master 拓扑和 owner ledger 的证据解码由 `adapters.master_evidence` 提供，readiness、残留上界及收敛判定仍由 action 决定。


## 异常收尾与资源监管

`ResourceScope` 为清理项保留 ACTIVE、STOPPING、CLOSE_FAILED、CLOSED 状态。清理先向所有声明了 `request_stop` 的资源发出有界停止请求，再按 LIFO 等待关闭；失败项保留到后续独立预算重试。预算耗尽后不执行任意用户回调。每次失败记录保留，后续回收成功不抹掉原始执行或清理错误。

实例父进程通过独立 session 的管道监管进程启动 runner。正常 TERM/INT 传递给 workload 执行协作清理；父进程死亡导致管道 EOF，监管进程强制结束本次 session。父编排器的异常、TERM 和 KeyboardInterrupt 都先停止调度并回收子进程，再等待线程池退出。残留进程、端口占用、超时或无法验证释放时，lane 不复用。

session 监管只覆盖留在该进程组的后代；主动另建 session 的后代、内核不可中断状态及监管进程自身被同时杀死，需要 CI executor 的任务容器或 cgroup 生命周期兜底。禁止将 Python finally、JVM shutdown hook 或进程名扫描当作覆盖这些故障的保证。

监控关闭失败后继续保留 server/线程所有权，只有验证退出后才设置 closed。查询归档失败不能跳过进程回收。结果判定、清理状态与 evidence/report/archive 交付状态独立保留。
