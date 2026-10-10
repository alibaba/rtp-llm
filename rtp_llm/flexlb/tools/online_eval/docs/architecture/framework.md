# 框架结构

```text
config/scenarios/*.yaml   身份、环境、输入、阈值与视图选择
        ↓
cases/<case>/program.py   步骤、分支、检查与能力声明
        ↓
scenario compiler        校验并生成有类型的执行计划
        ↓
parallel runner          冻结执行计划，分配 lane 与租约
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
| CI suite | `suite_schema_version` | 2 |
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

## 代码归属

| 目录 | 内容 |
|---|---|
| `cases/<case>/` | 该 case 的 program、action、输入合同、纯分析、指标生产、报告与离线重判 |
| `cases/config.py`、`cases/registry.py` | 公共 case 构建接口与显式注册 |
| `scenario/` | 配置编译、基础 action、资源句柄、阶段执行与清理 |
| `runtime/` | 端口/进程/Java/HTTP/gRPC 等协议与资源底座 |
| `workload/` | 连续负载的通用采集生命周期、证据核对、来源与全量视图 |
| `analysis/` | 可复用统计、只读检查及输入保真度分析，不定义某个 case 的门槛 |
| `monitoring/`、`reporting/` | 指标查询/归档与图表/spec/bundle 的公共能力 |
| `traffic/`、`artifacts/` | 流量构造、播放与制品管理 |

归属由语义决定，不由文件名或当前调用数决定：包含特定阶段、门槛、指标输出或报告身份的代码属于 case；同一合同能被不同 case 使用的能力才放在公共组件。注册表可以指向专属实现，公共执行器不按 case 名分支。

集中归属不合并职责。`analysis.py` 根据显式输入计算，不访问网络、进程或报告；`actions.py` 操作现场并取证；`metrics.py` 发布声明的指标；`report.py` 展示既定结论，不能隐式重判。门禁动作先冻结独立 result 和校验 manifest，再返回共用的 `CheckResult`。数值投影在最终遥测导出后执行，报告在统一的 renderer 阶段生成一次，不重新裁决或改写冻结判定。

## 执行与资源契约

每个 case 有 `default` program，变体只追加测试点。YAML 保存数据，不能写 action、通用 `$ref` 或任意表达式；具名观测窗口只能选择 program 声明的时间输出，不能编排步骤。配置层次和 action 边界见[新增 case](../development/adding-cases.md)。

资源句柄绑定环境代次。重建环境前清理旧消费者及进程，旧句柄只能显式作为历史证据读取。动态 worker 添加保留尝试预算；移除不返还容量预算。端口租约由父 runner 拥有，子执行器启动前核对范围与最大拓扑。父 runner 将完整 compiled instances 写入 `execution-plan.json`，记录规范化内容摘要与代码、运行配置的文件摘要。子执行器使用父进程提供的摘要验证计划，再验证当前依赖与租约，直接执行冻结步骤，不重新编译场景 YAML。规划后改变场景文件不改变已经冻结的参数；代码或运行配置变化使启动失败，必须重新规划。

`ResourceScope` 持有句柄、环境代次、取证适配器和清理栈；`RuntimeContext` 绑定输出引用及实际事件；`StageExecutor` 负责阶段分发与输出检查；`RunFinalizer` 负责证据和报告预算、检查点与结果提交。deadline 位于运行底座。gRPC 客户端持有 channel，stream 持有消费线程，Mock 控制能力只使用所属环境的 HTTP 地址，不另建资源所有者。

阶段使用剩余 deadline，清理使用独立预算并逆序执行。同一资源回调内的独立清理操作使用 `cleanup_all`，逐项尝试并汇总错误，诊断采样器失败不得阻断进程回收。启动部分失败也要回收已登记资源；未退出线程、未回收进程或清理异常不能成为 PASS。普通失败阻断依赖步骤，独立观察应放在最终判定之前。finding 只能声明具体检查，不豁免证据、超时和清理。

观察预算覆盖预热、流程等待和取证窗口；排空预算覆盖停止发送后等待请求终态；分析预算覆盖门禁所需的取证和判定；报告发布使用独立的 `execution.report_timeout_s`。需要单独分析阶段的负载在 `procedure.analysis_timeout_s` 声明预算，不能由客户端请求超时推导。

请求发出、Schedule ACK、Fetch、业务 FINISHED、取消和资源释放分别取证，不互相推断。服务启动及诊断 API 成功应答不代替业务完成。具体协议见[请求生命周期](request-lifecycle.md)。

指标和报告使用冻结定义与序列，不通过 debug API、日志或文件自动兜底；缺失来源显式报错。采集、门禁与交付规则分别见[指标契约](metrics.md)、[结果与指标](../development/results.md)和[报告契约](reporting.md)。

## 现场观测与冻结证据

持续时间和轮询由阶段的单调时钟控制；epoch 秒用于关联客户端流水和 Prometheus。`ObservationClock` 冻结观测起点的两种时钟，样本用同一锚点加单调时间差生成 `epoch_s`、`monotonic_s`、`elapsed_s`。已有证据中的毫秒字段与相对 `t` 是格式投影，不形成第二个时钟来源。

`RuntimeContext.record_event` 是事件时间的权威记录。证据内事件保存同一记录及相对时间投影，供独立离线重判；stage 的起止记录描述执行生命周期，不能代替采样点或业务事件。具名窗口声明负责边界，测量实现负责 cohort 与统计口径。

采样复用 `poll_samples` 和 `SampleBudget`。`observation.capture` 显式声明样本数与字节上限；超限保留已有现场和错误，证据不完整时不得判 PASS。先保留已获得的失效现场再检查流量状态，允许诊断停止原因；错误样本不能被当作有效门禁证据。HTTP 请求复用 `runtime.network`，timeout 受当前 deadline 或可停止的短请求预算约束。

门禁的获取、终态补全和发布均通过 `artifacts.json_io.write_json` 原子更新同一证据文件，崩溃时保留最近的完整版本。公共获取信封包含格式版本、clock、criteria、samples、errors、provenance；业务 payload 保持所属格式的字段与语义。格式版本描述结构，`measurement_policy` 描述统计口径，两者独立。历史时钟字段只由 `evidence_origin` 转换；缺少锚点且没有样本时失败，不补零。

`run_provenance.collect` 收集实际环境代次的启动输入；门禁复用其严格获取路径并冻结业务源码摘要。`runtime.java_flow.evidence_environment` 定义可比较客户端环境中的 run-local 字段排除规则。源文件位置变化只改变摘要，不改变历史证据的重判口径。

停止发流和排空由 program 显式编排；门禁 handler 保存判定，最终归档后投影指标，再由注册 renderer 生成报告。采集或归档失败保留为 errors，产生 INVALID/ERROR；分析或判定保存异常成为 stage ERROR，报告交付错误单独记录。门禁证据句柄使用 `gate_evidence`，与基础快照分开；historical 表示可以显式读取旧环境证据，默认读取仍拒绝环境换代。功能程序没有持续采样需求时不创建空观测组件。

跨阶段后台工作由 context 登记的资源对象持有，action 只负责参数校验、登记、调用与输出。客户端启动、finish、证据快照与 cleanup 使用同一对象，不另外包装第二套生命周期。Java 发流控制复用 `runtime.flow_control` 的原子命令、身份校验和排空计数；发现方式及业务证据校验留在所属能力。独立资源的清理复用 `cleanup_all`，启动失败与清理失败同时存在时保留启动错误及其原因链。

## 收尾与持久化

`execution.timeout_s` 约束阶段执行，`cleanup_timeout_s` 单独保留资源清理预算。workload 清理后，证据导出与分析受 `finalize_timeout_s` 约束，报告生成受 `report_timeout_s` 约束；父 runner 的进程预算包含这两段时间。同步工作受主线程中断式期限保护，资源导出同时检查合作式 deadline。

清理结束后立即原子保存 `result.json`，每段收尾前后保存检查点。未完成的检查点使用 `status: FINALIZING`，不能作为终态 PASS；`execution_status` 保留阶段与清理的结果，`finalization` 保存各段预算、状态、耗时和错误。最终 `duration_ms` 包含执行、清理、证据和报告。证据阶段失败使 runtime validity 为 INVALID，并阻止报告阶段；报告失败记录 report status，保留已有 runtime validity 和独立门禁结果。整体执行器结果仍报告交付失败，调用方不能把未完成的交付当成成功。

细项检查使用 `CheckResult` 的 id、status、actual、expected、detail、evidence；`gate_checks` 汇总有效阈值检查与完整性错误。ERROR 对应 INVALID，FAIL 对应有效观测越过门槛，SKIP 表示不适用，WARNING 表示 advisory。持续异常、窗口归属和请求 cohort 的计算属于各 case，公共组件只统一结果协议与聚合。

## 结果状态

`result.json.outcome` 分别保存 `execution`、`gate`、`validity` 和 `delivery`。流程异常、业务未达标、证据不足与交付失败分别记录，不用一种错误覆盖其他事实。`runtime.outcome.RunOutcome` 是最终状态的唯一计算规则，父 runner 使用同一规则验证子结果一致性。交付失败仍使任务失败，但不能改变已冻结的门禁 verdict；finding 只匹配明确失败的检查。

没有执行有效检查，或只有 `SKIP`，不构成 PASS。`WARNING` 是实际完成的建议性检查；证据不足始终为 ERROR，不受建议性策略豁免。可缺失的测量输出必须显式声明 `nullable_number`，只有伴随证据 ERROR 时才能输出 null；成功结果仍要求实际有限数值。

## 声明与解析结果

数据型 case YAML 在入口解码为 `CaseDeclaration`、`VariantDeclaration`、`ExecutionPolicy` 和 `MonitoringPolicy`。case 的业务参数由所属程序校验，公共构建层不通过任意模块属性寻找入口。`CaseDefinition` 声明全部能力，注册器按目录发现并校验冲突。

环境覆盖通过统一规则合并：配置字段整体替换，profile 按名称及字段继承，列表与完整模型整体替换。编译阶段将最终 Master JSON、preset 性能数据及运行选项冻结为带 SHA 的 `EnvironmentSnapshot`。运行时使用冻结数据投影进程参数，不重新读取 preset 或应用配置默认值。
