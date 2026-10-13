# 新增 case

YAML 定义输入，注册的 Python program 定义流程。只改变拓扑、阈值或流量时修改现有 YAML；改变业务步骤时修改所属 case 的 Python。所有 YAML 平铺在 `config/scenarios/`，文件位置不决定测试分类或 CI 选例。

## 从最小 case 开始

先判断需要哪种扩展，避免照抄大型 workload 的全部文件：

| 需求 | 必需内容 | 可复用能力 |
|---|---|---|
| 请求、操作与协议断言 | 场景 YAML、`program.py` | 基础 action 与默认报告 |
| 标准指标时窗比较 | 增加指标集合、窗口与输入绑定 | 公共采集、`analysis.metric_checks.check_metric` 与通用图表 |
| 特殊业务测量或数据源 | 按需增加专属 handler、producer 或 adapter | 注册、deadline、证据冻结和报告骨架 |

下面的最小例子验证请求终态。创建 `src/cases/protocol_terminal/`，加入空的 `__init__.py` 和下列 `program.py`，再创建 `config/scenarios/protocol_terminal.yaml`。名称仅用于演示新增目录的接入方式；所有取值由示例 YAML 明确给出，预算和阈值不是框架默认值。小型 case 不需要专属采集器、producer、报告或重判入口。

```python
from case_api.builder import output
from case_api.inputs import fields
from case_api.numeric_parameters import JAVA_LENGTH, POSITIVE_COUNT, NONNEGATIVE, COUNT
from case_api.registry import CaseDefinition
from traffic.contracts import driver


def default(case):
    data = case.inputs(
        traffic={"kind", "input_len", "output_len", "count"},
        procedure={"setup_timeout_s"}, checks={"no_errors"},
    )
    criterion = data.checks["no_errors"]
    fields(criterion, {"output", "unit", "windows", "op", "expected"},
           "parameters.checks.no_errors")
    if (criterion["output"] != "error_count" or criterion["unit"] != "errors"
            or criterion["windows"] != ["terminal"] or criterion["op"] != "eq"):
        raise ValueError("no_errors requires the terminal error_count equality check")
    request = {key: case.number("traffic." + key) for key in driver(data.traffic, "request_batch")}
    case.step("setup", "setup", timeout_s=data.procedure["setup_timeout_s"])
    case.step("submit", "request", params=request)
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    case.step("no_errors", "check", params={
        "actual": output("terminal", criterion["output"]),
        "op": criterion["op"], "expected": criterion["expected"],
    })
    case.step("cleanup", "teardown")


CASE = CaseDefinition(
    builders={"default": default},
    numeric_parameters={
        "traffic.input_len": JAVA_LENGTH, "traffic.output_len": JAVA_LENGTH,
        "traffic.count": POSITIVE_COUNT, "procedure.setup_timeout_s": NONNEGATIVE,
        "checks.no_errors.expected": COUNT,
    },
)
```

```yaml
case_schema_version: 2
case: protocol_terminal
program: default
metadata:
  kind: functional
  description: 验证提交的请求终态没有流错误
  category: status
  tags: []
profiles: [single-nonbatch]
environment:
  backend: java_mock
  n_prefill: 1
  n_decode: 1
execution:
  timeout_s: 300
  stage_timeout_s: 60
  cleanup_timeout_s: 120
  collection: diagnostic
parameters:
  traffic:
    kind: request_batch
    input_len: 16
    output_len: 2
    count: 1
  procedure:
    setup_timeout_s: 180
  checks:
    no_errors:
      output: error_count
      unit: errors
      windows: [terminal]
      op: eq
      expected: 0
```

从 `rtp_llm/flexlb` 执行：

```bash
python3 tools/online_eval/scripts/commands/format_configs.py
python3 tools/online_eval/scripts/commands/validate_case.py \
  --source tools/online_eval/config/scenarios/protocol_terminal.yaml --json
```

然后添加输入错误、证据不足和清理失败测试，按[运行测试](running.md)完成现场验证。新增目录会被自动发现；是否纳入 CI 必跑集合由 `config/suites.yaml` 显式决定。标准指标检查读取冻结的 `MetricStore`，需要明确传入标签、时窗、归约、最少样本和缺采预算；尚未冻结的数据不能通过这个分析函数冒充现场观测。

## YAML 的阅读顺序

场景字段按下表排列。顺序是编辑规范，不改变合并优先级或执行顺序。

| 顺序 | 字段 | 阅读目的 |
|---|---|---|
| 1 | `case_schema_version`、`case`、`program` | 格式、身份与默认入口 |
| 2 | `metadata`、`profiles` | 用途、测试性质与运行形态 |
| 3 | `environment`、`execution` | 模型/拓扑、采集策略与时间预算 |
| 4 | `parameters`、`parameter_schema` | 流量输入、指标绑定、门槛及约束 |
| 5 | `variants`、`profile_overrides` | 默认程序之外的测试点和覆盖 |
| 6 | `reporting`、`reports` | 时间轴与交付视图 |

`parameters` 按 `traffic` → `procedure` → `observation` → `analysis` → `checks` 排列，只声明 program 使用的组。`observation` 内按采样与覆盖设置 → `inputs` → `capture` → `windows` 排列；未使用的字段不补齐。根参数及变体覆盖都由配置排序命令检查，指标绑定集合、窗口名和检查项的内部顺序保持原样。流量先写来源和播放方式，再写并发、预算与客户端资源。指标集合按身份/继承 → 来源查询 → Python 输出排列；视图按身份 → 查询依赖/事件 → 曲线绑定 → 面板排列，曲线先写 `metric_id`/`labels` 再写展示属性。命名集合及列表顺序保留，不能按字母重排阶段、曲线或面板。

从 `rtp_llm/flexlb` 检查或整理配置：

```bash
python3 tools/online_eval/scripts/commands/format_configs.py --check
python3 tools/online_eval/scripts/commands/format_configs.py
```

唯一的字段排序表在 `scripts/pipeline/config_order.py`。命令只移动已有字段块，不补默认值、改标量或重排列表，并在写入前检查解析值不变。新业务参数按其输入含义组织，不为了格式工具另建 YAML schema。

每个 case 顶层必须声明 `program: default`，Python 必须注册 `default` builder。`variants` 只追加额外测试点，不能占用 `id: default`。`metadata.kind` 为 functional 或 workload，`execution.collection` 为 aggregate、request 或 diagnostic；性质、说明和采集档位必须完整。CI 必跑项显式登记到 `config/suites.yaml`，不从 category 或目录位置推断。

## 数据、运行策略与参数约束

| 段 | 负责什么 | 主要消费者 |
|---|---|---|
| `metadata` | 测试性质、目标、分类与标签；`kind` 参与性质筛选和执行器选择 | suite 选择、实例清单、执行器与报告 |
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

program 通过 `NUMERIC_PARAMETERS` 把必需的 dotted path 绑定到 `case_api.numeric_parameters` 中的语义类型：计数、正整数、非负实数、`FRACTION`、有符号偏移、priority 和 Java 请求长度。类型和协议范围只在公共类型中定义，公共字段组直接复用；program 可以进一步收紧运行预算，但不能改变公共字段类型或放宽通用范围。`unit: ratio` 不是类型约束，倾斜比等比值可以大于 1；只有明确绑定 `FRACTION` 的比例限制在 `[0,1]`。

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

先阅读[case 插件结构](../architecture/case-plugins.md)，理解编译、取证、测量、判定和交付的生命周期。目录按职责拆分，不要求每个 case 搭齐所有模块。

`src/case_api/builder.py` 和 `src/case_api/registry.py` 是公共构建、能力注册接口；业务代码集中在 `src/cases/<case>/`：

| 文件 | 职责 |
|---|---|
| `program.py` | `default`、额外变体、步骤顺序、输出引用与门禁依赖 |
| `actions.py` | 专属现场操作、取证、deadline 与清理 |
| `inputs.py`、`analysis.py` | 业务指标绑定、输入合同与根据显式证据计算的门禁 |
| `metrics.py` | 将业务证据投影成声明的指标；采集、归档与通用统计仍复用公共组件 |
| `report.py`、`panels.py`、`view.py` | 业务结果展示、面板装配与专属视图校验；HTML 交互和 bundle 协议复用 `reporting/` |
| `client.py`、`observation.py` 等现场模块 | 专属客户端会话或数据源协议解释；HTTP 观测注册适配器，由公共标准 exporter 与 Prometheus 管理调度、预算和收尾 |
| `comparison.py` | 业务声明的对齐策略校验，不重算跨 run 门禁 |
| `replay.py` | 专属证据补充与显式离线重判；整窗门禁复用 `GatePlugin` 发布 |

按实际需要建文件，小型 case 只需 `program.py`，不创建空的层次。这里的归属表示业务合同由谁维护，不表示代码永远不可复用：同类 program 可显式注册同一实现；跨业务一致的统计、协议、采集或渲染功能才提到公共组件。不能因文件名同为 `analysis.py` 或 `metrics.py` 就把不同业务口径合并。

`case_api.registry` 自动发现 `cases` 下包含 `program.py` 的直接子目录。program 导出 `CASE: CaseDefinition`，明确声明构建器、参数约束、动作、报告、指标生产者和采集源。目录名就是稳定 case 名；新增 case 不修改公共名单。注册快照校验能力冲突并保持不可变，YAML 只能选择已发现入口，不能导入代码。通用构建接口是 `case_api.builder.CaseBuilder`；`output(stage, name)` 声明有类型的前序输出引用。

`CASE.actions` 声明所属能力；单次运行的分析参数由所属 program 校验。指标用 `case.metric(id, ...)` 声明所有门禁及 producer 的采样输入依赖，编译检查 ID、单位与身份标签。采集清单来自这些依赖、报告曲线与显式诊断项的并集；目录中无人消费的能力不采集，不能靠 producer 运行时加载整份 plan 扩大清单。复杂指标的 producer 注册到 `CASE.producers`，`monitoring/producers.py` 负责调用这些能力；定义数据放在 `config/monitoring/`，规则见[指标配置](../../config/monitoring/README.md)。

`CASE.producers` 注册 `MetricProducer(describe, execute, phase)`。GATE 的 `execute(directory)` 在最终遥测导出后投影已冻结的门禁数值，FINALIZE 的 `execute(directory, context)` 发布收尾证据；函数名没有自动注册含义。报告通过 `CASE.report_views` 注册 `ReportView(validator, renderer)`；renderer 接收目录、运行分析和视图定义，读取校验后的冻结判定，在统一报告阶段生成 bundle，返回 HTML 路径，不重判或重新发布指标。门禁 bundle 声明 `role="gate"`，发现与汇总通过 manifest 校验，不根据目录名猜测。报告协议见[报告契约](../architecture/reporting.md)。


注册接口的参数契约定义在 `case_api.extension_contracts`，发现阶段检查可调用参数，不执行能力函数：

| 能力 | 调用约定 |
|---|---|
| builder | `(case)` |
| action | `validate(params, plan)`、`execute(ctx, params, deadline)` |
| report | `validator(path, data, fail)`、`renderer(directory, analysis, presentation)` |
| producer | `describe(producer, metric_id, calculation)`；GATE `execute(directory)`，FINALIZE `execute(directory, context)` |
| source | `factory(environment, fields)` |

所有模块导入只定义能力，不能启动进程、访问网络或创建运行产物。导入失败会定位到所属 case 并阻止执行，不忽略坏 case。签名检查不能证明返回结构正确或 deadline 得到遵守；新增能力须测试合法结果、错误结果和资源收尾。

## Action 的边界

公共 action 位于 `src/scenario/actions/`，每个 action 承担单一目的、预算、输出与清理责任；可包含轮询和多次 RPC，不以代码行数划分原子大小。

- 相同合同可被不同 case 直接使用，才进入公共 action；不含实验阶段组合或业务门槛。
- 专属能力留在该 case 的 `actions.py`。同类 program 可显式声明同一组实现，重复或冲突声明报错；编译器检查所有权，不根据 YAML 的展示 ID 授权。
- 协议访问复用 `runtime/`，参数校验复用 `scenario.parameters.validate_fields`；不导入其他 action 的私有函数共享工具。
- 多操作的流程与编译时分支放在 program；单操作的现场判断与有界重试放在 handler。跨阶段动态跳转没有现成契约，不能通过 YAML 表达式或隐式跳步实现。

handler 用 `StageHandler` 声明参数、输出、能力与检查 ID。未知字段、类型错误和非法引用在启动前拒绝；所有等待使用剩余 deadline，后台资源立即登记清理，异常保留已获得证据。需要导出证据的资源在 `register_resource(..., evidence=exporter)` 显式登记适配器；`exporter(value, collection_profile)` 返回 `ResourceEvidence`，声明请求记录、完整性错误、生产者身份和预期中断。公共策略通过 `export_evidence` 枚举声明，不猜测资源方法或 case 名。每个场景至少声明一个检查。执行器的预算与资源规则见[框架结构](../architecture/framework.md)。

基础 `check` 与业务分析复用 `analysis.checks` 的标量比较。`analysis.metric_checks.check_metric` 只读取冻结的 `MetricStore`，显式指定指标、标签、时窗、归约和覆盖要求，不发起采集或填补缺失值。归约必须选中一条 series；跨 worker 聚合由 PromQL 或明确的 Python 测量计算负责。检查桥接通过 `MetricStore.reduce_with_evidence` 一次选择并归约，取得独立的定义与观察证据，不直接访问 store 的文档结构。检查结果保留定义、来源、实际窗口和样本信息；观测不足统一通过 `invalid_check` 返回 ERROR，并将 `validity: INVALID` 放在 `evidence`，不当作实际测量值；有效数据越过门槛为 FAIL，缺少有效观测为 ERROR/INVALID，契约错误直接报错，advisory 只改变普通阈值失败。

请求集合采用哪个时间字段、如何归属节点、如何判定终态，以及持续异常等业务计算留在 case。请求集合的半开时窗与指标采样点的闭区间选择分别声明，不互相推断；报告展示既定结果，不重新计算门禁。

## 配置覆盖与变体

配置按全局默认 → profile 模板 → case 覆盖渲染。profile 身份由 decision × dispatcher 决定，`PROFILE_SPECS` 是唯一来源；case 可重述相同身份值，不能改变它。`ordering`、优先级、抢占和非身份预算是可配置策略；有效能力由最终配置计算。

`FUNCTIONAL_DEFAULTS` 提供模式无关基线，`FUNCTIONAL_PROFILE_KWARGS` 提供模式相关模板。窗口字段只在 FIXED_WINDOW 输出，SINGLE 不携带；`GENERATOR_DEFAULTS` 只用于低层 schema 构造。字段允许范围以配置 schema 和校验器为准，文档不维护第二份字段清单。`queue_timeout_ms: {omit: true}` 表示使用 Java 默认期限，不表示无限等待。

`perf_preset` 引用固定采集档案。模式相关 Master 配对参数由 profile 解释；case 的非身份覆盖优先。测试派生的模型、容量或拓扑偏离用 `environment.model_override` 声明基线与原因，不能改写采集档案冒充现场观测；规则见[数据目录](../../data/README.md)。

`profiles` 选择运行形态；`environment.profile_overrides` 覆盖非身份环境字段；检查的 `expected_by_profile` 声明形态相关阈值；`warning_profiles` 只将指定形态下的普通阈值失败标为警告，不能豁免缺失数据。所有 profile 名均验证，包括未被选中的规则。

变体用于表达实验目的、实验对象或实现流程的变化，例如正反对照、故障类型或替代实现。设计时不宜为每个 QPS、时长或容量值分别新增变体；参数扫描使用独立的实验输入覆盖，保持同一实验身份。这是 case 设计规范，不由字段差异自动推断或强制判定。

变体用唯一 `id` 标识，可组合 `program`、`environment`、`parameters` 和 `parameter_schema`，不需要声明变化轴，名字也不必等于 program 名。`program` 必须已注册；覆盖数据仍按字段、类型、数值范围、未消费参数和 profile 身份合同校验。`parameter_schema` 只能收窄 program 声明的数值范围。根配置的 `profiles`、执行预算和报告声明适用于所有变体；变体不能写入这些未支持的覆盖字段，避免配置被接收后静默忽略。YAML 仍只表达数据，步骤与分支由 Python 实现。

## 验证

先用 `validate_case.py --source <场景路径>` 校验所有变体和 profile；`--json` 输出合并后的参数、数值约束、阶段检查、窗口、指标消费者、报告与资源预算；`collector_health` 标明采集器用于完整性判断的 `up` 指标，不把它们当作业务检查或图表。命令复用正式编译器，不启动服务、不发请求、不创建产物；退出码 0 表示静态通过，2 表示配置或能力错误。静态通过不代表 exporter、采样覆盖、门禁实际值或清理已经验证。再用 `run_cases.py --dry-run` 检查执行计划并执行必要的真实 case。运行入口见[运行测试](running.md)。本地回归从 `tools/online_eval` 执行：

```bash
PYTHONPATH=src:. python3 -m pytest -q tests
```

新增 case 的最小测试矩阵如下；公共注册、输入错误和文档示例的可执行测试见 `tests/test_case_authoring.py`，业务测试按自己的合同构造证据。

| 边界 | 应验证的结果 |
|---|---|
| 合法输入与变体 | 全部声明的 variant/profile 可编译，检查、窗口和指标依赖完整 |
| 未知字段、越界与非法引用 | 启动前拒绝，错误定位到输入路径或能力名称 |
| 缺指标、采样不足或损坏证据 | ERROR/INVALID，不补零、不换数据来源、不降成通过 |
| 阈值越界 | 有效证据产生 FAIL，实际值和期望值可追溯 |
| 超时、取消与操作异常 | 释放已登记资源，清理失败明确留在结果中 |
| 专属报告 | 消费冻结判定，缺曲线按公共规则展示，不能重新计算门禁 |

新增判定须保存实际值、期望值和来源；调整规模或负载后重新验证门槛。删除能力时同时检查 registry、program、CLI、文档和测试，保留公共工具及现役失败边界。编译、假传输和单测不代替真实服务运行验收。

## 检查与输出

检查返回共用 `CheckResult`，实际测量与比较在 case 的纯分析能力中完成。发布指标、生成 HTML 属于收尾阶段，不能成为业务检查的前置条件。证据不足返回 `invalid_check` 并保留原因；不要以零代替缺失，也不要用 `SKIP` 宣称通过。

handler 的错误路径同样遵守输出合同。数值在证据不足时可以为空的，声明 `nullable_number`，输出键仍必须存在，并伴随 ERROR 检查；依赖有效数值的普通输出继续使用 `number`。报告附录通过 `view_details`、`view_table` 获得稳定 `case.<id>`，公共检查与有效性区由报告骨架统一添加，不依赖中文标题识别结构。

## 收尾指标扩展

注册为 `ProducerPhase.FINALIZE` 的函数签名是 `execute(directory, context: FinalizationContext) -> dict`。从 `context.run` 读取调用快照，通过返回 metadata 提交自己的诊断信息，不修改主分析结果。自定义视图、producer 和来源必须在所属 `CASE` 声明；复用另一 case 的实现也须显式声明。调用边界和源码指纹见 [Case 插件结构](../architecture/case-plugins.md)。

## 扩展功能时的修改边界

先区分新增配置、业务算法与新协议，不把这些独立事实合并成一段隐式推导。新增数值字段优先用 `InputContract.from_fields` 单点声明类型、必填性与准入字段；新增指标检查用 `CheckSpec` 单点声明绑定和数值规则。YAML 负责具体取值，指标定义负责来源口径，纯分析负责业务算法。它们是必要的修改点；不要再手工重复维护 RULES、数字路径及分析字段集合。

```python
from case_api.input_contract import InputContract
from case_api.numeric_parameters import NONNEGATIVE

INPUTS = InputContract.from_fields(
    {"custom": {"name": None, "threshold": NONNEGATIVE}},
    optional={"custom.threshold"},
)
# CASE = CaseDefinition(builders={"default": default}, input_contract=INPUTS)
# data = case.bind(INPUTS)
```

自定义客户端或采样器在 case 的实现文件编写协议，并在 program 的 `CASE` 注册；不修改公共 source 白名单或执行器分支。注册与实现可以同处一个小型 program，也可以拆成两个文件。调用客户端必须保存返回的真实资源句柄，采样器必须使用传入 deadline 限制 I/O；不要自建脱离 scope 的后台线程。接口职责与签名只在[case 插件结构](../architecture/case-plugins.md)维护。

扩展验证应记录实际 diff 涉及的作者文件、重复合同声明与公共层修改数，测试文件单列。至少验证正向执行、错签名、未声明能力、缺失与非法输入、过期 epoch、超时与清理，以及冻结检查的投影不重新判定。新增能力不能改变原 case 的窗口、阈值、cohort、缺采处理或运行判定。
