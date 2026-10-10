# 新增 case

YAML 定义输入，注册的 Python program 定义流程。只改变拓扑、阈值或流量时修改现有 YAML；改变业务步骤时修改所属 case 的 Python。所有 YAML 平铺在 `config/scenarios/`，文件位置不决定测试分类或 CI 选例。

## YAML 的阅读顺序

场景字段按下表排列。顺序是编辑规范，不改变合并优先级或执行顺序。

| 顺序 | 字段 | 阅读目的 |
|---|---|---|
| 1 | `case_schema_version`、`case`、`program` | 格式、身份与默认入口 |
| 2 | `metadata`、`profiles` | 用途、测试性质与运行形态 |
| 3 | `environment`、`execution` | 模型/拓扑、采集策略与时间预算 |
| 4 | `parameters`、`parameter_schema` | 流量输入、指标绑定、门槛及约束 |
| 5 | `variant_axis`、`variants`、`profile_overrides` | 默认程序之外的测试点和覆盖 |
| 6 | `reports` | 交付视图 |

`parameters` 按 `traffic` → `procedure` → `observation` → `analysis` → `checks` 排列，只声明 program 使用的组。`observation` 内按采样与覆盖设置 → `inputs` → `capture` → `windows` 排列；未使用的字段不补齐。根参数及变体覆盖都由配置排序命令检查，指标绑定集合、窗口名和检查项的内部顺序保持原样。流量先写来源和播放方式，再写并发、预算与客户端资源。指标集合按身份/继承 → 来源查询 → Python 输出排列；视图按身份 → 查询依赖/事件 → 曲线绑定 → 面板排列，曲线先写 `metric_id`/`labels` 再写展示属性。命名集合及列表顺序保留，不能按字母重排阶段、曲线或面板。

从 `rtp_llm/flexlb` 检查或整理配置：

```bash
python3 tools/online_eval/scripts/commands/format_configs.py --check
python3 tools/online_eval/scripts/commands/format_configs.py
```

唯一的字段排序表在 `scripts/pipeline/config_order.py`。命令只移动已有字段块，不补默认值、改标量或重排列表，并在写入前检查解析值不变。新业务参数按其输入含义组织，不为了格式工具另建 YAML schema。

每个 case 顶层必须声明 `program: default`，Python 必须提供 `default(case)`。`variants` 只追加额外测试点，不能占用 `id: default`。`metadata.kind` 为 functional 或 workload，`execution.collection` 为 aggregate、request 或 diagnostic；性质、说明和采集档位必须完整。CI 必跑项显式登记到 `config/suites.yaml`，不从 category 或目录位置推断。

## 数据、运行策略与参数约束

| 段 | 负责什么 | 主要消费者 |
|---|---|---|
| `metadata` | 测试性质、目标、分类与标签；`kind` 参与 CI 筛选和执行器选择 | suite 选择、实例清单、执行器与报告 |
| `environment` | Master / Mock 的配置、模型和拓扑，如 worker 数量、性能档案、缓存容量 | 环境渲染与启动、端口和资源预算 |
| `execution` | 实例、阶段和清理预算，以及 `collection` 和 `monitoring` 取证策略 | runner、阶段执行器、客户端与采集器 |
| `parameters` | 按流量、流程、观测、分析与检查分组的 program 输入 | `CaseBuilder.inputs/number`、业务输入校验 |
| `parameter_schema` | 场景对 program 数值契约的范围收紧 | program 构建前统一校验；`CaseBuilder.number(path)` 可显式读取 |

`environment.n_prefill` 是启动多少个 Prefill worker；`parameters.traffic.count` 是 program 发出多少个请求。整数类型、正负和协议范围由 Python 数值契约定义；`parameter_schema["traffic.count"].maximum` 可以进一步限制该场景的请求预算。实际取值与允许范围分别维护，调整约束不自动改变请求数量。

`traffic.kind` 显式选择 `request_batch`、`java_flow` 或 `ha_replay`，各 program 只接受自己的字段合同。`traffic` 保存请求、来源与发流条件；`procedure` 保存操作、流程等待和超时上限；`observation` 保存观测时窗、指标输入绑定与采样要求；`parameters.analysis` 保存 SLO、持续异常等证据解释规则；`checks` 保存检查条件和门槛。发流 QPS 只在 `traffic` 定义，program 将同一个值冻结到门禁证据，不维护另一份目标 QPS。环境启动配置和实例执行预算仍分别属于 `environment`、`execution`。

program 用 `case.inputs(...)` 声明各组允许和必需的字段，得到 `ProgramInputs`。基础参数和 variant 合并后都执行严格校验；未知字段、未读取参数和缺失必需值报错。嵌套业务对象复用其拥有者的输入校验，不能用“已读取父级 dict”代替子字段校验。复杂流程仍在 Python，不为每个 YAML 字段创建类或表达式语言。

`observation.windows` 是取证边界的权威声明，支持两种锚点：

- 跨阶段窗口使用 `stage`、`field: epoch_s` 和可选 `offset_s`，经 `ObservationWindow` 编译成有类型的输出引用。`procedure` 中的 `wait_s` 控制现场流程实际等待；窗口引用完成后记录的时间戳，有效时长由实际边界之差计算，不能把操作的 `timeout_s` 当成窗口长度，也不再另写同义的窗口时长。
- 单次观测内的窗口使用 `event` 和显式 `offset_s`，测量能力验证允许的锚点及边界方向，并从边界推导预热、测量时长和采集截止时间。偏移量定义取证区间；改变边界也可能改变观测执行时长，并非只改变报告显示。

两种形式遵守同一规则：流程等待、操作超时与取证区间是不同输入，不相互冒充，也不重复声明同一个窗口长度。无条件等待属于 `procedure`；观测能力为等待指标就绪或覆盖达标设置的有界预算属于 `observation`。program 选择当前流程可用的窗口。`checks.<id>.windows` 使用非空、无重复的窗口名列表；不存在或当前流程不可用的窗口报错。单窗口检查也使用列表；仅支持单窗口的测量能力拒绝多窗口，不能隐式拼接数据。多个窗口的切分和归约由 Python 测量契约决定，列表不定义算法。`full_run` 等整体范围及阶段输出窗口由 program 显式声明，不用组合字符串代替窗口引用。YAML 不能借此定义步骤顺序或分支。

运行身份使用 `case::variant::profile`，配置、制品和实际流量用冻结的 SHA 与运行配置追溯。观测参数不另声明手工实验身份；性能证据必须保留实例身份、配置 SHA、制品和流量 SHA，缺失或损坏的证据不能通过门禁。

program 通过 `NUMERIC_PARAMETERS` 把必需的 dotted path 绑定到 `cases.numeric_parameters` 中的语义类型：计数、正整数、非负实数、`FRACTION`、有符号偏移、priority 和 Java 请求长度。类型和协议范围只在公共类型中定义，公共字段组直接复用；program 可以进一步收紧运行预算，但不能改变公共字段类型或放宽通用范围。`unit: ratio` 不是类型约束，倾斜比等比值可以大于 1；只有明确绑定 `FRACTION` 的比例限制在 `[0,1]`。

`parameter_schema` 可省略，只声明场景自己的 `minimum`、`maximum`，不能改变整数类型或放宽 program 契约。variant 的范围只能在场景范围上继续收紧。所有字段在 program 构建前校验，包括 variant 合并后的值；未绑定字段、缺字段、错误类型、非有限或越界值失败。展开后的每个 variant 数值契约写入 `implementation.numeric_parameters`，供编译结果与运行证据核对。

数值校验不把字段标为“业务已使用”，未被 program 读取的输入仍会被拒绝。复杂对象和跨字段关系由 program 或 action 合同校验；数值契约不提供参数缺省值，也不承担环境配置校验。

观测输入统一使用 `fields: {本地字段: {metric: namespace/name, labels: {...}}}`；输入组只绑定指标，不声明固定的存储后端。绑定方向不因 case 改变；同一物理指标的不同标签投影可以共存，同一 ID 和标签选择重复绑定时报错。读取能力由 Python program 提供，实时观测可通过 monitoring session 读取 Prometheus，离线分析消费冻结证据；运行时来源实例由采集目标决定。query plan 是实际单位、标签与采样模式的权威定义；Python 消费合同声明所需维度，绑定时比对，防止定义被误标。身份标签按必需子集校验，允许 exporter 附加标签；测量能力另行校验角色、基数、允许字段和必需字段。

字段集合校验统一由 `input_contract.mapping_fields` 实现；case 与 action 提供字段集和定位路径。配置入口将输入错误包装为带来源路径的 `ScenarioError`，运行时协议或执行错误保持原始异常类型。五组是公开输入语义，内部分析器可以接收经编译的数值投影；投影不得提供缺省值，也不得成为第二份配置来源。历史证据的数值 criteria 字段保持其重判含义。

固定测量能力在 Python 声明指标、单位、窗口和方向，YAML 提供阈值；通用指标检查可由 YAML 选择指标与窗口，但仍验证定义与允许选择。前者保障持续性、cohort 等算法的前提，后者用于无专属测量流程的标量比较。两者复用比较与依赖绑定，不把测量算法复制成 YAML 规则。消费方单位是算法要求，query plan 单位是生产声明；必须相等，不能直接抄生产单位代替消费断言。

数值检查写明 `metric`、`unit`、`windows`、`op`、`expected`；程序校验其与已注册测量规则一致，并将门槛冻结到运行证据。布尔或协议输出检查用 `output` 明确引用已声明阶段输出，不伪造数值指标。SLO 的逐请求定义、持续异常的阈值构造等是测量参数，放在 `parameters.analysis`；复杂归因和持续性计算仍由 Python 负责。场景不声明根级 `analysis`；多运行对比通过公共入口读取冻结报告，事件对齐由对比命令显式指定。

QPS 只取 `traffic.client.playback.qps`，编译时核对请求数量和 goodput 下界不超过允许的 offered-load 范围。修改负载不自动缩放绝对阈值，需要同时核对时窗和门槛。priority 由 `traffic.source.parameters.priority` 定义，客户端环境从该值生成；显式客户端 PRIORITY 与来源冲突时报错。

`metadata.description` 是唯一的业务目标说明；`metadata.kind` 必须显式声明为 functional 或 workload，不从目录、标签或参数推断。`execution.collection` 为 aggregate 时保留汇总计数，为 request 时保留逐请求证据，为 diagnostic 时额外启用诊断。`execution.monitoring` 管指标集合、采样周期、缺采预算与采集器收尾预算；业务窗口和输入绑定仍属于 `parameters.observation`。指标绑定和名称的规则见[指标契约](../architecture/metrics.md)。

case 主报告文件使用 `<case>.yaml`，公共视图使用 `default.yaml`，额外视角使用 `<case>_<视角>.yaml`。目录区分场景、指标集合和报告职责，同一 case 的配置可使用相同文件名。报告生产能力与 bundle 身份独立声明，不由文件名推断。

## 新增指标前的来源选择

新增 case 时先列出每个检查和图表需要测量的事实，再选择数据来源。来源与计算方式不同：读取 Prometheus 样本后做 survivor 筛选或整窗归约，仍是 `source_type: prometheus`，必要时通过 `produced` 发布结果；直接可表达的查询放在 `sources`。具体口径边界只在[指标契约的选择统计口径](../architecture/metrics.md#选择统计口径)维护。

Prometheus 优先适用于监控指标；请求是否完成、RPC 是否符合协议等功能断言使用 action 的协议输出，不要求先转换为指标。

按以下顺序设计：

1. 普通吞吐、队列、资源和状态趋势优先查已有 exporter，使用 PromQL。先确认物理指标实际存在、单位、身份标签和采样间隔；存在相似名称不代表口径相同。图表说明 rate 窗口、直方图估计和缺采行为。
2. 检查指定发送 cohort 的最终结果、请求路由或多个 SLO 的联合满足情况，以及精确逐请求分位数、TPOT 时，保留请求流水，声明 `client_journal`。明确窗口采用发送还是完成时间，不能用全局 counter 或 histogram 替换请求配对。
3. 必须读取 exporter 未提供的状态账本或接口可回读性时，才使用 `debug_api`。通过注册数据源接入公共生命周期，适配器只做一次有界读取，不在 case 中自行启动轮询线程。逐字段声明指标，规定身份、采样和字段校验；接口可回读与 Prometheus `up` 分别测量不同事实。日志或文件仅作为必要的取证输入，不能成为缺失指标的回退来源。
4. 只有被检查、曲线或明确诊断消费的测量才注册为指标。算法分子、分母和边界值可保留在冻结分析证据中，不为每个中间变量新增 `metric_id`。通用 calculator 按需声明，专属 producer 只发布其已声明输出，缺失必需结果报错。
5. 需要保留同一物理量的两种口径时，明确各自用途、窗口和权威消费者；只需一种口径时，从实际 plan 中移除另一条定义。不能同时继承两套指标，靠报告隐藏其中一套。

提交前验证 query plan 展开、检查绑定和报告分类：每个指标能说明由谁消费；非 Prometheus 输入有具体的必要性；故意缺失或损坏数据时门禁保持 ERROR/INVALID，图表断线，不填零、不切换来源。切换展示口径须验证报告标识、单位和来源；切换判定口径须另行证明窗口、统计对象、覆盖和精度满足门禁合同。编译与单测只验证配置及逻辑，新增采集口还需真实运行验证 exporter 和数据覆盖。

## Python 的归属与注册

`src/cases/config.py` 和 `registry.py` 是公共构建、能力注册接口；业务代码集中在 `src/cases/<case>/`：

| 文件 | 职责 |
|---|---|
| `program.py` | `default`、额外变体、步骤顺序、输出引用与门禁依赖 |
| `actions.py` | 专属现场操作、取证、deadline 与清理 |
| `inputs.py`、`analysis.py` | 业务指标绑定、输入合同与根据显式证据计算的门禁 |
| `metrics.py` | 将业务证据投影成声明的指标；采集、归档与通用统计仍复用公共组件 |
| `report.py`、`panels.py`、`view.py` | 业务结果展示、面板装配与专属视图校验；HTML 交互和 bundle 协议复用 `reporting/` |
| `client.py`、`observation.py` 等现场模块 | 专属客户端会话或数据源协议解释；HTTP 观测注册适配器，由公共标准 exporter 与 Prometheus 管理调度、预算和收尾 |
| `comparison.py` | 业务声明的对齐策略校验，不重算跨 run 门禁 |
| `publication.py`、`replay.py` | 产物发布与显式离线重判 |

按实际需要建文件，小型 case 只需 `program.py`，不创建空的层次。这里的归属表示业务合同由谁维护，不表示代码永远不可复用：同类 program 可显式注册同一实现；跨业务一致的统计、协议、采集或渲染功能才提到公共组件。不能因文件名同为 `analysis.py` 或 `metrics.py` 就把不同业务口径合并。

`cases.registry.PROGRAMS` 将稳定 case 名映射到 program 模块；YAML 只能选择注册入口，不能导入代码。通用构建接口是 `cases.config.CaseBuilder`；`output(stage, name)` 声明有类型的前序输出引用。

program 的 `ACTION_HANDLERS` 声明所属能力；单次运行的分析参数由所属 program 校验。指标用 `case.metric(id, ...)` 声明所有门禁及 producer 的采样输入依赖，编译检查 ID、单位与身份标签。采集清单来自这些依赖、报告曲线与显式诊断项的并集；目录中无人消费的能力不采集，不能靠 producer 运行时加载整份 plan 扩大清单。复杂指标的 producer 注册到 `monitoring/producers.py`，定义数据放在 `config/monitoring/`，规则见[指标配置](../../config/monitoring/README.md)。

program 可声明 `produce_gate_metrics(directory)`，在最终遥测导出后投影已冻结的门禁数值。报告通过 `REPORT_VIEWS` 注册 `ReportView(validator, renderer)`；renderer 接收目录、运行分析和视图定义，读取校验后的冻结判定，在统一报告阶段生成 bundle，返回 HTML 路径，不重判或重新发布指标。门禁 bundle 声明 `role="gate"`，发现与汇总通过 manifest 校验，不根据目录名猜测。报告协议见[报告契约](../architecture/reporting.md)。

## Action 的边界

公共 action 位于 `src/scenario/actions/`，每个 action 承担单一目的、预算、输出与清理责任；可包含轮询和多次 RPC，不以代码行数划分原子大小。

- 相同合同可被不同 case 直接使用，才进入公共 action；不含实验阶段组合或业务门槛。
- 专属能力留在该 case 的 `actions.py`。同类 program 可显式声明同一组实现，重复或冲突声明报错；编译器检查所有权，不根据 YAML 的展示 ID 授权。
- 协议访问复用 `runtime/`，参数校验复用 `scenario.parameters.validate_fields`；不导入其他 action 的私有函数共享工具。
- 多操作的流程与编译时分支放在 program；单操作的现场判断与有界重试放在 handler。跨阶段动态跳转没有现成契约，不能通过 YAML 表达式或隐式跳步实现。

handler 用 `StageHandler` 声明参数、输出、能力与检查 ID。未知字段、类型错误和非法引用在启动前拒绝；所有等待使用剩余 deadline，后台资源立即登记清理，异常保留已获得证据。需要导出证据的资源在 `register_resource(..., evidence=exporter)` 显式登记适配器；`exporter(value, collection_profile)` 返回 `ResourceEvidence`，声明请求记录、完整性错误、生产者身份和预期中断。公共策略通过 `export_evidence` 枚举声明，不猜测资源方法或 case 名。每个场景至少声明一个检查。执行器的预算与资源规则见[框架结构](../architecture/framework.md)。

基础 `check` 与业务分析复用 `analysis.checks` 的标量比较。`check_metric` 只读取冻结的 `MetricStore`，显式指定指标、标签、时窗、归约和覆盖要求，不发起采集或填补缺失值。归约必须选中一条 series；跨 worker 聚合由 PromQL 或明确的 Python 测量计算负责。检查结果保留定义、来源、实际窗口和样本信息；观测不足统一通过 `invalid_check` 返回 ERROR，并将 `validity: INVALID` 放在 `evidence`，不当作实际测量值；有效数据越过门槛为 FAIL，缺少有效观测为 ERROR/INVALID，契约错误直接报错，advisory 只改变普通阈值失败。

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

## 检查与输出

检查返回共用 `CheckResult`，实际测量与比较在 case 的纯分析能力中完成。发布指标、生成 HTML 属于收尾阶段，不能成为业务检查的前置条件。证据不足返回 `invalid_check` 并保留原因；不要以零代替缺失，也不要用 `SKIP` 宣称通过。

handler 的错误路径同样遵守输出合同。数值在证据不足时可以为空的，声明 `nullable_number`，输出键仍必须存在，并伴随 ERROR 检查；依赖有效数值的普通输出继续使用 `number`。报告附录通过 `view_details`、`view_table` 获得稳定 `case.<id>`，公共检查与有效性区由报告骨架统一添加，不依赖中文标题识别结构。
