# 用例与公共能力

YAML 声明环境、输入、阈值和视图；Python 定义步骤与业务算法。每个 case 的入口是 `program.py` 导出的 `CASE: CaseDefinition`。仅改变参数时复用现有程序；基础 action 足够时无需专属采集器、指标或报告文件。

## 职责与依赖

| 内容 | 归属与职责 |
|---|---|
| `program.py` | 读取 `case.config`，组合类型化步骤，声明 `CASE` |
| `inputs.py` | 输入字段、数值规则、窗口及指标约束，不访问现场 |
| `actions.py` | 操作、等待、取证及资源登记；不导入 program |
| `analysis.py` | 从显式证据计算业务检查，不采集、不生成报告 |
| `metrics.py` | 指标口径和冻结证据的数值投影 |
| `report.py` | 展示冻结判定，复用公共视图和 HTML 发布 |
| `replay.py` | 显式离线重判；报告展示不隐式重判 |

公共编译、执行和资源管理位于 `scenario`；服务协议位于 `adapters`；进程、时钟及有界采样机制位于 `runtime`。输入模块不依赖编译器，分析器不依赖报告，adapter 不依赖场景执行。模块导入不启动网络、进程或采集。

## 输入与动作

`InputSchema` 使用一份嵌套字段声明派生必需字段、可选路径与数值规则。`None` 表示由领域拥有者校验的非数值对象。编译器先合并基础参数和变体，应用 `parameter_schema` 的收紧约束，再解析独立的 `CaseConfig`；program 通过 `case.config` 读取。`CaseConfig` 是类型标注过的普通字典，不维护分组包装、绑定对象或投影回调。

字段集合和数值规则只声明一次。嵌套协议对象使用其领域解析器，跨字段约束由 case 普通校验函数维护。编译入口和冻结计划读取面对不同输入边界，均须检查合法性；共用同一规则，不维护平行规则表。缺值、未知字段、非法类型、收紧约束冲突及未消费参数均失败。

`CaseBuilder.step()` 和 `observe()` 生成 `StageDefinition`。action 名称是冻结计划中的稳定身份；执行器通过所属 case 的 `StageHandler` 找到普通执行函数。handler 保留输出类型、检查 ID、资源预算、所需能力和执行前提。只做字段与资源引用检查的参数校验复用 `scenario.parameters.reference_parameters`；有业务含义的验证保持在 case 内。

外层阶段用于独立超时、前置条件、故障操作和资源生命周期。普通请求批次可以在一个动作内完成提交、等待和多项断言；持续负载保留启动与停发排空的边界。观察时段内部记录起止事件，窗口切片与独立断言集中分析，不为每个中间列表登记资源。合并动作使用子 deadline 保留各操作预算，并受所属阶段和整轮超时约束。

批量动作可以通过 `StageHandler.checks(params)` 从已校验参数派生检查 ID；编译器将其冻结到计划，执行器仍核对返回的完整 ID 集合。每个业务检查保留独立结果、阈值、样本有效性和证据；报告中的完整身份为所属阶段 ID 与检查 ID 的组合。ERROR/INVALID 不得折算成业务通过。

阶段执行消费原生参数对象，`AdapterParameters` 仅持有独立的字典快照；JSON 用于文件边界及可序列化性检查，不用作内部参数状态。输出句柄绑定环境代次，部分启动失败也登记并回收资源。

`CaseDefinition.execution_requirements` 与 `StageHandler.execution_requirements` 声明逐请求证据和监控依赖。编译与冻结计划加载拒绝冲突的 `collection: aggregate`、`capture_metrics: false`，不隐式调整门槛或窗口。

## 判定与冻结

整窗测量在收齐证据后调用 `commit_gate(directory, name, evidence, analyze)`，由纯 `analyze(evidence)` 产生结果，`freeze_gate` 核对 `verdict` 与检查一致后原子发布证据、结果和 manifest，返回 `(result, path)`。业务测量失败、证据无效和交付失败分别记录。

`gate_result(checks, errors, requirements=...)` 统一派生 `verdict`、`validity` 和检查明细。实验前提未满足时，将对应失败检查记为 ERROR；业务阈值未满足保持 FAIL。case 保存一份 `checks`，展示时按检查身份分组，不重复保存功能与有效性检查列表。

分阶段检查可在所需阶段完成并冻结 `CheckResult`，不必等整轮结束。两种执行方式不能改变原来的 cohort、窗口和归约语义。ERROR/INVALID 表示证据不足，FAIL 表示业务阈值未达标。

`project_gate(directory, name, project)` 校验冻结文件后调用数值投影。未提交的门禁不伪造指标；已存在但损坏的 manifest 必须报错。`render_gate` 只读取冻结判定并调用报告函数。`publish_gate` 供离线发布已计算的结果使用，指标投影后重新读取冻结判定，避免投影中的可变对象污染展示。

门禁名称保留其文件身份，不注册为运行时插件，不通过能力查询间接调用分析器。离线重判必须显式要求，写入新目录，并保留证据来源与模拟边界。

## 指标与报告

`CASE.metrics` 指向所属 case 的指标模块。模块的 `CONTRACTS` 是来源 ID 到口径函数的声明索引，签名为 `describe(source_id, metric_id, calculation)`；它用于校验来源、单位、计算方法和实现出处，不分派执行函数。指标文件的 `producer` 字段保留为已有来源标识。

最终遥测导出后，执行器直接调用 `project(directory, selected, deadline)`，将冻结门禁或阶段检查转换为已声明指标。需要整轮分析的模块可提供 `finalize(directory, analysis, selected, deadline)`；其输入是顶层只读、嵌套独立复制的快照，返回以已选择来源 ID 为键的 metadata，保留到兼容的 `producer_results` 字段。回调不得扩大本轮选中的指标集合，也不能覆盖执行检查或判定。

统一的 `monitoring.publication.publish` 校验定义、序列身份、来源和实现口径，再更新 `MetricStore`。缺失来源不回退到其他文件或 debug API。指标和图表需求的选择发生在编译期，运行投影不能扩大采集计划。

视图由 `CASE.report_views` 声明 `ReportView`。`scenario.registry.view/write_views` 在组合入口解析视图与指标计划，把已选择的描述符交给通用读取和发布函数。`reporting` 不导入执行注册表，也不查找 case；离线报告按冻结定义分类指标，不要求安装当时的注册快照。renderer 接收目录、运行分析、视图定义和可选 deadline，返回 HTML 路径；只展示既定判定，不分析或重新发布指标。离线 renderer 校验归档格式与摘要。

## 来源与资源边界

少量结构化来源通过 `CASE.sources` 显式声明。公共 snapshot 源提供引擎、请求和客户端记录；Master 观测遵循[公共诊断契约](master-diagnostics.md)，不假设任意 Master 工件提供某个调试接口。资源 exporter 使用 `adapters.resource_evidence.ResourceEvidence`，数据合同和现场投影不反向依赖场景执行。

自定义动作、视图、指标模块和来源必须由消费 case 显式声明；共享实现也由各使用方声明，自动发现不等于跨 case 授权。名称冲突、回调签名和资源预算在启动前检查。case 源码指纹覆盖其目录全部 Python 文件，公共依赖另由冻结执行计划校验；case 指纹不等于完整代码身份。
