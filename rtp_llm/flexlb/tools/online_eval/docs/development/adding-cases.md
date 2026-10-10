# 新增 case

YAML 定义输入，注册的 Python program 定义流程。只改变拓扑、阈值或流量时修改现有 YAML；改变业务步骤时修改所属 case 的 Python。所有 YAML 平铺在 `config/scenarios/`，文件位置不决定测试分类或 CI 选例。

## YAML 的阅读顺序

场景字段按下表排列。顺序是编辑规范，不改变合并优先级或执行顺序。

| 顺序 | 字段 | 阅读目的 |
|---|---|---|
| 1 | `case_schema_version`、`case`、`program` | 格式、身份与默认入口 |
| 2 | `metadata`、`test`、`profiles` | 用途、测试性质、采集与运行形态 |
| 3 | `environment`、`execution` | 模型/拓扑、配置与时间预算 |
| 4 | `parameters`、`parameter_schema` | 流量输入、指标绑定、门槛及约束 |
| 5 | `variant_axis`、`variants`、`profile_overrides` | 默认程序之外的测试点和覆盖 |
| 6 | `analysis`、`reports` | 分析策略及交付视图 |

`parameters` 按 `traffic` → `procedure` → `observation` → `checks` 排列，只声明 program 使用的组。流量先写来源和播放方式，再写并发、预算与客户端资源。指标集合按身份/继承 → 来源查询 → Python 输出排列；视图按身份 → 查询依赖/事件 → 曲线绑定 → 面板排列，曲线先写 `metric_id`/`labels` 再写展示属性。命名集合及列表顺序保留，不能按字母重排阶段、曲线或面板。

从 `rtp_llm/flexlb` 检查或整理配置：

```bash
python3 tools/online_eval/scripts/commands/format_configs.py --check
python3 tools/online_eval/scripts/commands/format_configs.py
```

唯一的字段排序表在 `scripts/pipeline/config_order.py`。命令只移动已有字段块，不补默认值、改标量或重排列表，并在写入前检查解析值不变。新业务参数按其输入含义组织，不为了格式工具另建 YAML schema。

每个 case 顶层必须声明 `program: default`，Python 必须提供 `default(case)`。`variants` 只追加额外测试点，不能占用 `id: default`。`test.kind` 为 functional 或 workload，`collection` 为 aggregate、request 或 diagnostic；性质、说明和采集档位必须完整。CI 必跑项显式登记到 `config/suites.yaml`，不从 category 或目录位置推断。

## 数据、运行策略与参数约束

| 段 | 负责什么 | 主要消费者 |
|---|---|---|
| `metadata` | 测试目标、分类与标签，供检索和结果展示 | 实例清单、报告 |
| `test` | functional/workload、采集档位及监控设置，决定执行和取证策略 | suite 选择、执行器、客户端与采集器 |
| `environment` | Master / Mock 的配置、模型和拓扑，如 worker 数量、性能档案、缓存容量 | 环境渲染与启动、端口和资源预算 |
| `execution` | 实例、阶段和清理的时间预算 | runner、阶段执行器 |
| `parameters` | 按流量、流程、观测与检查分组的 program 输入 | `CaseBuilder.inputs/number`、业务输入校验 |
| `parameter_schema` | 数值参数的整数类型、最小值和最大值约束 | program 构建前统一校验；`CaseBuilder.number(path)` 可显式读取 |

`environment.n_prefill` 是启动多少个 Prefill worker；`parameters.traffic.count` 是 program 发出多少个请求；`parameter_schema["traffic.count"].maximum` 是该数量允许的上界。实际取值与允许范围分别维护，调整约束不自动改变请求数量。

`traffic.kind` 显式选择 `request_batch`、`java_flow` 或 `ha_replay`，各 program 只接受自己的字段合同。`traffic` 保存请求、来源与发流条件；`procedure` 保存操作及流程预算；`observation` 保存观测时窗、指标输入绑定与采样要求；`checks` 保存检查条件和门槛。发流 QPS 只在 `traffic` 定义，program 将同一个值冻结到门禁证据，不维护另一份目标 QPS。环境启动配置和实例执行预算仍分别属于 `environment`、`execution`。

program 用 `case.inputs(...)` 声明各组允许和必需的字段，得到 `ProgramInputs`。基础参数和 variant 合并后都执行严格校验；未知字段、未读取参数和缺失必需值报错。嵌套业务对象复用其拥有者的输入校验，不能用“已读取父级 dict”代替子字段校验。复杂流程仍在 Python，不为每个 YAML 字段创建类或表达式语言。

`observation.windows` 是观测边界的权威声明。跨阶段窗口使用 `stage`、`field: epoch_s` 和可选 `offset_s`，经 `ObservationWindow` 编译成有类型的输出引用；单次观测内的窗口使用 `event` 和显式 `offset_s`，测量能力验证允许的锚点及边界方向。时长从边界计算，不再另写同义的时长参数。program 选择当前流程可用的窗口。`checks.<id>.window` 指向声明窗口；不存在或当前流程不可用的窗口报错。YAML 不能借此定义步骤顺序或分支。

`parameter_schema` 的每条 dotted path 在 program 构建前统一校验，variant 合并后的值也受约束。未声明字段不会推断边界；缺字段、错误类型、非有限或越界值失败。校验本身不把字段标为“业务已使用”，未被 program 读取的输入仍会被拒绝。复杂对象和跨字段关系由 program 或 action 合同校验；schema 不提供缺省值，也不承担环境配置校验。

观测输入统一使用 `source: metric_store` 和 `fields: {本地字段: {metric: namespace/name, labels: {...}}}`。绑定方向不因 case 改变；同一物理指标的不同标签投影可以共存，同一 ID 和标签选择重复绑定时报错。`metric_store` 指定读取后端，运行时来源实例另由采集目标决定。query plan 是实际单位、标签与采样模式的权威定义；Python 消费合同声明所需维度，绑定时比对，防止定义被误标。身份标签按必需子集校验，允许 exporter 附加标签；测量能力另行校验角色、基数、允许字段和必需字段。

字段集合校验统一由 `input_contract.mapping_fields` 实现；case 与 action 提供字段集和定位路径。配置入口将输入错误包装为带来源路径的 `ScenarioError`，运行时协议或执行错误保持原始异常类型。四组是公开输入语义，内部分析器可以接收经编译的数值投影；投影不得提供缺省值，也不得成为第二份配置来源。历史证据的数值 criteria 字段保持其重判含义。

数值检查写明 `metric`、`unit`、`window`、`op`、`expected`；程序校验其与已注册测量规则一致，并将门槛冻结到运行证据。布尔或协议输出检查用 `output` 明确引用已声明阶段输出，不伪造数值指标。SLO 的逐请求定义、持续异常的阈值构造等是测量参数，放在 `observation`；复杂归因和持续性计算仍由 Python 负责。

QPS 只取 `traffic.client.playback.qps`，编译时核对请求数量和 goodput 下界不超过允许的 offered-load 范围。修改负载不自动缩放绝对阈值，需要同时核对时窗和门槛。priority 由 `traffic.source.parameters.priority` 定义，客户端环境从该值生成；显式客户端 PRIORITY 与来源冲突时报错。

`test` 保持独立：将执行与采集策略藏入 `metadata` 会使说明字段承担控制作用。`metadata.description` 描述业务目标，`test.description` 描述测试或取证策略；两者应避免重复。指标绑定和名称的规则见[指标契约](../architecture/metrics.md)。

## Python 的归属与注册

`src/cases/config.py` 和 `registry.py` 是公共构建、能力注册接口；业务代码集中在 `src/cases/<case>/`：

| 文件 | 职责 |
|---|---|
| `program.py` | `default`、额外变体、步骤顺序、输出引用与门禁依赖 |
| `actions.py` | 专属现场操作、取证、deadline 与清理 |
| `inputs.py`、`analysis.py` | 业务指标绑定、输入合同与根据显式证据计算的门禁 |
| `metrics.py` | 将业务证据投影成声明的指标；采集、归档与通用统计仍复用公共组件 |
| `report.py`、`panels.py`、`view.py` | 业务结果展示、面板装配与专属视图校验；HTML 交互和 bundle 协议复用 `reporting/` |
| `runtime.py` | 业务所需的有界现场采样或客户端会话，不复制公共进程、HTTP 或 gRPC 底座 |
| `comparison.py` | 业务声明的对齐策略校验，不重算跨 run 门禁 |
| `publication.py`、`replay.py` | 产物发布与显式离线重判 |

按实际需要建文件，小型 case 只需 `program.py`，不创建空的层次。这里的归属表示业务合同由谁维护，不表示代码永远不可复用：同类 program 可显式注册同一实现；跨业务一致的统计、协议、采集或渲染功能才提到公共组件。不能因文件名同为 `analysis.py` 或 `metrics.py` 就把不同业务口径合并。

`cases.registry.PROGRAMS` 将稳定 case 名映射到 program 模块；YAML 只能选择注册入口，不能导入代码。通用构建接口是 `cases.config.CaseBuilder`；`output(stage, name)` 声明有类型的前序输出引用。

program 的 `ACTION_HANDLERS` 声明所属能力；分析策略用 `ANALYSIS_POLICY_VALIDATOR` 校验，未声明则拒绝顶层 `analysis`。指标用 `case.metric(id, ...)` 声明依赖，编译检查 ID、单位与身份标签。复杂指标的 producer 注册到 `monitoring/producers.py`，定义数据放在 `config/monitoring/`，规则见[指标配置](../../config/monitoring/README.md)。

若需最终归档后刷新报告，program 声明可调用的 `REPORT_FINALIZER(directory)`；只读取已发布的结果与指标，不重判或重复生产指标。门禁 bundle 声明 `role="gate"`，发现与汇总通过 manifest 校验，不根据目录名猜测。报告协议见[报告契约](../architecture/reporting.md)。

## Action 的边界

公共 action 位于 `src/scenario/actions/`，每个 action 承担单一目的、预算、输出与清理责任；可包含轮询和多次 RPC，不以代码行数划分原子大小。

- 相同合同可被不同 case 直接使用，才进入公共 action；不含实验阶段组合或业务门槛。
- 专属能力留在该 case 的 `actions.py`。同类 program 可显式声明同一组实现，重复或冲突声明报错；编译器检查所有权，不根据 YAML 的展示 ID 授权。
- 协议访问复用 `runtime/`，参数校验复用 `scenario.parameters.validate_fields`；不导入其他 action 的私有函数共享工具。
- 多操作的流程与编译时分支放在 program；单操作的现场判断与有界重试放在 handler。跨阶段动态跳转没有现成契约，不能通过 YAML 表达式或隐式跳步实现。

handler 用 `StageHandler` 声明参数、输出、能力与检查 ID。未知字段、类型错误和非法引用在启动前拒绝；所有等待使用剩余 deadline，后台资源立即登记清理，异常保留已获得证据。每个场景至少声明一个检查。执行器的预算与资源规则见[框架结构](../architecture/framework.md)。

基础 `check` 与业务分析复用 `analysis.checks` 的标量比较。`check_metric` 只读取冻结的 `MetricStore`，显式指定指标、标签、时窗、归约和覆盖要求，不发起采集或填补缺失值。归约必须选中一条 series；跨 worker 聚合由 PromQL 或明确的 Python 测量计算负责。检查结果保留定义、来源、实际窗口和样本信息；有效数据越过门槛为 FAIL，缺少有效观测为 ERROR/INVALID，契约错误直接报错，advisory 只改变普通阈值失败。

请求集合采用哪个时间字段、如何归属节点、如何判定终态，以及持续异常等业务计算留在 case。请求集合的半开时窗与指标采样点的闭区间选择分别声明，不互相推断；报告展示既定结果，不重新计算门禁。

## 配置覆盖与变体

配置按全局默认 → profile 模板 → case 覆盖渲染。profile 身份由 decision × dispatcher 决定，`PROFILE_SPECS` 是唯一来源；case 可重述相同身份值，不能改变它。`ordering`、优先级、抢占和非身份预算是可配置策略；有效能力由最终配置计算。

`FUNCTIONAL_DEFAULTS` 提供模式无关基线，`FUNCTIONAL_PROFILE_KWARGS` 提供模式相关模板。窗口字段只在 FIXED_WINDOW 输出，SINGLE 不携带；`GENERATOR_DEFAULTS` 只用于低层 schema 构造。字段允许范围以配置 schema 和校验器为准，文档不维护第二份字段清单。`queue_timeout_ms: {omit: true}` 表示使用 Java 默认期限，不表示无限等待。

`perf_preset` 引用固定采集档案。模式相关 Master 配对参数由 profile 解释；case 的非身份覆盖优先。测试派生的模型、容量或拓扑偏离用 `environment.model_override` 声明基线与原因，不能改写采集档案冒充现场观测；规则见[数据目录](../../data/README.md)。

`profiles` 选择运行形态；`environment.profile_overrides` 覆盖非身份环境字段；检查的 `expected_by_profile` 声明形态相关阈值；`warning_profiles` 只将指定形态下的普通阈值失败标为警告，不能豁免缺失数据。所有 profile 名均验证，包括未被选中的规则。

变体选择已注册的 Python program 或经 `parameter_schema` 校验的数据覆盖。维度、字段与身份必须一致，不把多项无关偏离包装成同一个参数变体；字段合并以加载器契约为准。未知或隐藏兜底值必须失败，不能靠 program 补齐未声明的输入。

## 验证

先用 `list_cases.py` 与 `run_cases.py --dry-run` 检查编译实例、预算和视图，再执行必要的真实 case。运行入口见[运行测试](running.md)。本地回归从 `tools/online_eval` 执行：

```bash
PYTHONPATH=src:. python3 -m pytest -q tests
```

新增判定须保存实际值、期望值和来源；调整规模或负载后重新验证门槛。删除能力时同时检查 registry、program、CLI、文档和测试，保留公共工具及现役失败边界。编译、假传输和单测不代替真实服务运行验收。
