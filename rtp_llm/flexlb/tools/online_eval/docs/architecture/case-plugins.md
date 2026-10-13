# Case 插件结构

新增 case 是组合已有能力并补充业务差异，不是重新实现采集、归档、判定发布和 HTML 系统。YAML 表达运行输入、指标与窗口选择、检查门槛和视图；Python 声明流程和业务算法。唯一必需的代码入口是 `program.py` 导出的 `CASE: CaseDefinition`。

## 两种执行模型

单窗口或整窗测量在观测和请求收尾后收集完整证据，再计算测量值与检查。分阶段检查在每个所需阶段取窗口证据并判定，随后由执行器汇总。两者均使用 `CheckResult`，观测不足返回 ERROR/INVALID，阈值越界返回 FAIL。不能为了统一目录强迫分阶段检查等待整个 run 结束，也不能将整体测量的窗口、cohort 或归约拆成不等价的局部判定。

编译期构建固定计划；现场 handler 操作服务并输出证据；测量函数读取显式证据；发布和报告消费已冻结判定。测量不足、操作失败与报告交付失败是不同结果，不互相覆盖。

## 文件职责与依赖

| 文件 | 职责 | 何时需要 |
|---|---|---|
| `program.py` | 读取经校验输入、组合阶段、注册 `CASE` | 必需 |
| `inputs.py` | 输入字段、数值规则、窗口与指标绑定约束；不访问现场 | 有业务输入对象时 |
| `actions.py` | 专属操作、等待、取证和资源登记 | 基础 action 不足时 |
| `analysis.py` | 纯业务测量与检查；接收显式证据，不读目录、不启动采集 | 通用比较或测量不足时 |
| `observation.py`、`client.py` | 专属现场协议的解释与客户端会话 | 特殊数据源或协议需要时 |
| `metrics.py` | 业务证据到指标的投影与口径声明 | PromQL 或已有计算能力不足时 |
| `report.py`、`panels.py` | 业务面板、门槛线或诊断附录；复用公共曲线与发布 | view YAML 和公共报告不足时 |
| `replay.py` | 特殊证据补充和显式离线重判的命令入口 | 确有重判需求时 |

不创建空的层次。小型 case 可以只写 program；大型协议实现可拆文件，但必须归属一个明确能力。公共能力只有跨 case 的合同相同才抽取，不能仅因函数名字相同就合并算法。

依赖方向是输入与纯测量 → 现场能力 → program 的能力组合。门禁通过 `CASE.gates` 注册，现场能力通过 `ctx.services.gate(name)` 获取；动作不反向导入 program。输入模块不得混入监控查询；分析器不得调用报告；报告不得触发重判。所有模块导入无网络、进程或运行产物副作用。

## 整窗门禁能力

`case_api.gate_plugin.GatePlugin` 组合三种业务函数：

```python
GATE = GatePlugin(
    name="gate",
    analyze=analyze,
    project_metrics=project_metrics,
    write_report=write_report,
)
```

| 业务函数 | 输入 | 输出与边界 |
|---|---|---|
| `analyze(evidence)` | 已收齐的显式证据 | 结果 dict，含 `verdict` 和 `checks`；业务解释字段可以附加，判定必须与 `CheckResult` 汇总一致 |
| `project_metrics(directory, evidence, result)` | 已校验的冻结证据与判定 | 发布所声明指标；不决定 PASS/FAIL、不改写门禁文件 |
| `write_report(directory, evidence, result, *, run=None, presentation=None)` | 冻结判定、可选 run 与已选择的 view | 返回 bundle 目录；只负责展示，不分析或发布指标 |

插件名决定既有门禁证据、结果和 manifest 的文件身份。公共 `freeze_gate/load_gate` 负责原子 JSON 写入与校验；业务内容保持所属测量的证据格式，不伪装成通用指标。回调签名在构造时检查，返回内容、覆盖与清理仍需测试。

### 生命周期

1. 现场 handler 收齐原始证据，并在可能失败的分析前保存已获得证据。
2. `GATE.commit(directory, evidence)` 调用分析一次，校验检查与判定的一致性，冻结门禁文件，返回 `GateCommit(result, path)`。它不发布指标、不生成报告。
3. `GATE.project(directory)` 读取并校验已提交的门禁，再调用指标投影。该方法可作为 `MetricProducer` 的 GATE 阶段执行函数。中断的 run 可能尚未提交门禁，此时不伪造指标；已存在 manifest 的损坏必须报错。
4. `GATE.render(directory, run, presentation)` 读取并校验冻结判定，传递已选择的 view，返回 HTML 路径。该方法可作为 `ReportView` 的 renderer。缺少门禁不能生成一份假定通过的报告。

普通现场门禁不调用 `publish()`。它用于已经明确得到判定的离线发布：冻结 → 指标投影 → 校验冻结文件 → 展示。该方法从不隐式调用分析。显式离线重判先经 `load_reinterpretation` 检查源证据与新输出目录，再调用 `evaluate()`，随后发布。报告错误或指标发布错误不会将已冻结 PASS 改写为 FAIL，也不能把已冻结 FAIL 改成 PASS。

报告回调收到已校验的 `ViewDefinition`，通过属性访问配置；声明校验器也收到同一对象，区块列数通过 `validate_section_contract` 验证。报告的共用时间轴、运行信息补充和 bundle 写入由 `reporting.gate_report` 承担。case 提供业务 spec 与已有来源信息，明确选择在线 `write_gate_report(..., run=...)` 或离线 `write_archived_gate_report`；装配逻辑保留在公共发布函数中。报告时间轴与门禁窗口的职责见[报告契约](reporting.md)。

分阶段检查直接使用 `StageHandler` 和 `analysis.checks` 的测量/比较接口，不必额外构造一个整体 `GatePlugin`。数值采样来源通过 `CASE.sources` 注册，结构化快照通过 `CASE.snapshot_sources` 注册；数据来源与指标合同见[指标契约](metrics.md)。

## 新作者的接入路径

先确认是否能复用已有 program、action 和测量，再创建 YAML。需要新流程时新增 program；需要新业务算法时新增纯测量；需要特殊协议时新增 adapter。只有实际出现特殊展示或重判需求才增加对应模块。

输入和证据合同必须说明字段、单位、身份标签、时间基准、窗口与缺失行为。测量须说明统计对象及归约，不能靠指标名推断算法。协议输出使用阶段引用，不要求转换为指标。请求 cohort、survivor 筛选等算法保留 Python，公共层只统一可验证的调用边界。

可执行的最小例子、注册参数与测试矩阵见[新增 case](../development/adding-cases.md)。静态校验使用 `validate_case.py`；真实取证、缺采、超时和清理需额外验证。`tests/test_gate_plugin.py` 验证公共生命周期：判定和交付分离、损坏证据拒绝、已选择视图传递，以及交付失败保留原判定。

## 公共源码的归属

公共目录按能力职责划分；同一能力的协议实现与阶段适配可以分层，但不复制算法。

| 目录 | 负责 | 不负责 |
|---|---|---|
| `runtime` | deadline、进程、网络、通用请求记录、观察预算与资源合同 | 实例调度、服务协议、case 步骤、HTML 发布 |
| `adapters` | mock/master HTTP 与 gRPC、Java 客户端会话、环境与服务生命周期、启动配置 | 选择 case、声明阶段、业务门禁与报告 |
| `scenario` | 配置编译、实例选择与调度、阶段执行、资源登记、收尾 | 实现 RPC 协议、业务测量与 HTML 渲染 |
| `scenario/actions` | 将现场能力适配为校验参数、有预算、有输出和资源归属的步骤 | 复制 adapter 的协议和进程实现 |
| `traffic` | 流量采集与拟合、数据格式、长度与 prefix 语义、输入生成及 pacing 合同 | 启停客户端进程、现场缩容与门禁 |
| `workload` | 持续测量的执行策略、请求与 engine 证据关联、整轮采集有效性 | 公共门禁存储、报告发布、实例并行调度 |
| `analysis` | 显式输入上的统计与检查 | 启动服务、采集、发布报告 |
| `monitoring` | 采集计划、来源、Prometheus 生命周期与冻结指标 | case 流程及业务门槛 |
| `artifacts` | 证据/判定冻结、归档、来源校验、离线读取与安全保留 | 重新决定业务判定、执行 case 步骤 |
| `reporting` | 冻结结果到视图、公共图表、HTML 与报告 bundle | 现场操作、重判与隐藏缺采 |
| `case_api` | 作者的构建、注册、输入约束、门禁扩展合同 | 具体实验参数、流程或业务算法 |
| `cases/<case>` | 该业务的输入、步骤组合、测量和专属展示 | 复制公共执行、采集和报告基础设施 |

`scenario/actions/java_flow.py` 定义阶段合同，`adapters/java_flow.py` 管理客户端会话；`engine_control` action 依赖 `adapters/mock_control.py` 的协议访问。名字相近不表示冗余。新步骤先组合 adapter；只有出现新的现场协议才扩展 adapter。`runtime` 和 `adapters` 不依赖 `scenario`、`cases`、`workload` 或 `reporting`，避免基础能力反向调用业务和调度层。

新增参数组合复用现有 program，仅新增 YAML；新增流程从 `program.py` 开始，按需注册 action；新增测量才增加 analysis 或 producer。无需了解实例调度、归档或 HTML 内部实现。删除能力前检查注册发现、命令入口、配置引用、延迟导入与测试；缺少普通 import 不能单独证明无用。

## 扩展调用边界

`case_api.extension_contracts` 定义作者侧回调 Protocol。builder 接收 `CaseBuilder`；action 的参数校验接收 `PlanContext`，执行接收 `RuntimeContext` 与 `Deadline`，返回 `StageOutput`。领域参数和证据结构由所属 case 校验，Protocol 不代替运行时数据校验。`EvidenceSource` 当前是 Prometheus 证据适配配置工厂，不是任意文件或协议的通用采集插件。

专属视图、检查依赖的 producer 和证据来源必须由消费 case 的 `CASE` 显式声明。共享能力由各使用方声明同一实现；全局发现不是跨 case 的调用授权。默认报告和公共 action 由框架提供。

FINALIZE producer 接收 `FinalizationContext(directory, run)`，而非可写的主分析结果。`run` 顶层只读，嵌套数据是该次调用独立的深拷贝；修改嵌套数据不会影响主结果。producer 返回可 JSON 序列化且不含非有限数值的 metadata mapping，公共层将其写入 `producer_results`。指标发布仍走 `MetricStore`，不能通过返回 metadata 覆盖检查或判定。

编译结果的 `implementation.case_sources` 记录 case 目录全部 Python 文件的相对路径与 SHA256，`case_sources_sha256` 绑定这份确定性清单。它涵盖业务分析、操作及子模块；公共依赖的复现仍需整个仓库代码版本，不能把 case 指纹当作全部代码身份。

## 作者输入与运行时扩展合同

`InputContract` 由 case 持有。`InputContract.from_fields()` 从字段声明派生准入字段与数值规则；嵌套共享规则可使用 `numeric_fields()`。`CASE.input_contract` 将规则提供给编译器，program 通过 `case.bind(contract)` 读取。组名由 case 声明；五个常用组是组织约定。`decode` 可将校验后的分组转换为 case 自己的 frozen dataclass。预校验数值不消费参数；未知字段和未读取叶子仍被拒绝。

可选数值用 `InputField(..., required=False)` 或 `from_fields(..., optional=...)` 声明。缺失不会补默认值；存在时必须满足类型与范围，启用检查时仍须显式提供阈值。`CheckSet` 的每个 `CheckSpec` 同时声明分析字段、指标、比较、单位、窗口和数值规则，派生 numeric_parameters、分析字段集合和指标依赖。业务算法、统计对象和 YAML 的具体阈值仍由 case 持有。

`case.env` 与 `plan.env` 提供严格字段访问；未知属性立即报错。`resolved_config(profile)` 只接受本次选中的 profile，渲染结果在视图内缓存，返回副本。case 不导入 compiler。不同 profile 的值不一致时，读取必须明确指定 profile。

| 扩展 | 注册 | 调用与资源边界 |
|---|---|---|
| 自定义客户端 | `CASE.clients[name] = ClientFactory(factory, evidence=...)` | factory 接收 `(services, params, deadline)`；返回实现 `cleanup(deadline)` 的客户端。`ctx.services.client()` 登记 epoch 句柄与清理；证据 exporter 接收 `(client, profile, deadline)` |
| 结构化采样 | `CASE.snapshot_sources[name] = SnapshotSource(factory)` | factory 接收 `(services, params)`，只构造 reader；`read(deadline)` 执行 I/O，可选 `close(deadline)` 自动登记清理。snapshot/observe 负责 epoch、样本和字节预算及错误记录 |
| 门禁 | `CASE.gates[name] = GatePlugin(...)` | `ctx.services.gate(name)` 只解析当前 case 声明的能力 |

这些名称在 case 内解析，同名扩展可以属于不同 case。foundation action 的名字以及数值 producer/source/view 的显式共享身份继续执行既有冲突检查。factory 在构造失败时负责回收未返回的资源；返回后的资源由 scope 管理。扩展的回调签名在注册时校验，返回协议在使用时校验。

`ctx.services.environment` 提供当前环境、Master 地址与客户端进程操作。`ClientRecordAdapter(read, decode)` 将特殊记录明确转换为 client-record v1；公共读取器检查 schema 与 epoch，不替原始 Java 记录伪造身份。`FinalizationContext.deadline` 将收尾预算传给 producer。`ReportView.deadline_renderer` 可接收第四个 deadline 参数；整窗插件提供 `render_with_deadline`，离线 renderer 仍可独立调用。

分阶段检查的发布使用 `StagedGateProjection(action, project)`：公共层只读取 result.json、筛选已完成检查和保存指标，case 回调负责标签、时间与投影口径。它不重新执行判定。`ProducerDescriptions` 提供 `(producer, metric)` 表驱动口径，复杂动态口径仍可提供 describe 回调。

`InputContract.from_fields()` 接受嵌套字段声明，同时派生顶层字段准入与数值叶子规则；`None` 表示没有数值约束。共享数值路径通过 `section_fields()` 组合，检查阈值通过 `CheckSet.input_fields` 组合。嵌套声明不替代数据所有者的形状、单位和跨字段校验。新增数值输入只需要一处契约声明、业务消费及 YAML 值，不再维护独立的数值路径表或字段集。`optional` 可以指定字段或数值叶子路径，缺省不会隐式消费参数。

case 可以在契约的 `projection` 注册 `(value, context)` 校验函数，使用 `admit_projection()` 准入投影后的阶段参数或离线证据。编译阶段与冻结计划读入复用这一权威函数；进程内转换消费已经准入的数据。离线证据仍是外部输入，必须重新准入，不能用“已校验”的布尔开关绕过检查。

`monitoring.producers.produce(..., phase=ProducerPhase.GATE/FINALIZE)` 两个阶段都从冻结指标定义选择 producer，同名 producer 在单次调度中只执行一次。GATE 在整轮分析前发布已提交判定，FINALIZE 在分析后消费隔离快照并返回 metadata；两者都受收尾 deadline 约束。新增执行策略须复用该调度器，不另外从 program 查找或遍历 producer。

任意指标命名空间使用 `bind_metrics()`，显式提供 labels、mode 与可选 namespace；`bind_mock_engine_metrics()` 明确限定 mock scrape 的引擎身份维度。profile 阈值从选中的 `case.profiles` 校验，其他维度须显式传给 `metric_criterion(..., dimensions=...)`。`TimestampOutput` 集中声明时间输出字段和 epoch 秒域，窗口不依赖字段必须命名为 epoch_s。
