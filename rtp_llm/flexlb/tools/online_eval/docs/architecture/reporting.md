# 报告契约

报告展示已确定的结果，不负责取证或决定门槛。case 专属报告位于 `cases/<case>/report.py`，公共 `reporting/` 提供曲线绑定、物化与面板投影、配对、spec 校验、renderer 与 bundle 校验。曲线统一处理单位缩放、时间原点和来源；选中面板保留曲线的 `hidden`，不再用生产者另算全局可见性。面板只有有限实测值才算有效，零值有效，null/NaN 不补零，缺测使用 `empty_caption`，部分缺测列出缺少的曲线。配置门禁线由 case 在投影后添加，不算实测数据。case 程序在 `REPORT_VIEWS` 中按文件名声明 `ReportView` 校验与渲染能力；`case_api.registry` 自动汇集已注册程序的声明，同一视图可以复用相同能力；能力冲突或声明非法直接报错；YAML 只能引用视图文件，不能指定 Python 模块。

## 输入与装配

视图 YAML 在 `config/report_views/` 使用 `report_view_schema_version: 1`。顶层顺序固定为版本、`kind`、`report`、`metrics`、`charts`、`sections`，只出现当前视图需要的块；不允许把文案、曲线策略、表头或 case 私有字段追加到顶层。

| 块 | 职责 |
|---|---|
| `kind` | `default` 展示全部已归档指标，`selected` 展示 YAML 选择的指标；不能在 YAML 指定 Python 模块 |
| `report` | `subtitle`；已生产报告另声明 `id`、`producer`，声明最终 bundle 身份和生产者 |
| `metrics` | `query_plan` 选择指标集合；`diagnostic_only` 明确哪些已采指标不绘图 |
| `charts` | `curves`、`panels`、事件显示名和时间轴文案；全量视图在这里声明分组、采样和曲线可见性 |
| `sections` | 按稳定 section ID 声明附录标题、表头和默认开合状态 |

`sections.<id>` 只包含 `title`、`opened` 和表格的 `columns`。`checks` 表示专属门禁说明，`monitoring`、`validity`、`metrics`、`sources` 表示相应诊断；case 专属附录由注册的 Python 校验器声明。每个 case 校验附录 ID 集合及表格列数，不能依赖生产时的 KeyError。表格和详情使用相同开合组件，Python 决定内容和顺序，YAML 只控制展示。公共骨架使用稳定的 `run.checks`、`run.validity` 区块 ID 和 `run.execution`、`run.validity` KPI ID；视图附录使用 `case.<id>`。结构去重只比较 ID，不比较标题或展示文字。公共组件生成标准 KPI 与检查表，case 提供专属面板和附录；运行报告不重复展示 case 的另一份检查表。

曲线通过 `charts.curves.<curve_id>.metric_id` 与 `labels` 选择冻结指标，面板通过 `charts.panels[].curve_ids` 选择曲线；展示名不参与判定或身份匹配。`reporting/catalog.py` 只提供通用调色板和 UI 主题，不按指标名称推断单位、分组或业务含义。全量诊断视图按冻结指标的 `unit` 标注坐标轴，保留原始指标身份。

视图不声明运行元信息，不用固定字符串宣称采样或来源；运行信息由归档的 `run_meta` 提供，曲线来源由冻结的 provenance 提供。新增展示项放入相应块并同时补充 Python 字段合同，不保留旧扁平字段或转换适配器。

报告读取 `metrics.json` 中的冻结定义与序列，不重新查询服务、解析日志或生产数值指标。复杂门禁的报告必须接收明确结果，不能在 result 缺省时隐式重判。查询与 producer 规则见[指标契约](metrics.md)，主报告、默认视图和产物清单见[结果与指标](../development/results.md#收取产物)。

所有生产器直接输出同一 panel 契约：声明 `axes`，每条 series 携带 `points: [{x, y}]`；`timeX: true` 表示时间坐标，缺采以 `y: null` 保留。类目图的 `x` 是字符串，数值图的 `x` 是数值。`spec.py` 严格校验输入，不接受旧的 `x`/`xNums` + `series.data` 或显示模式标志，也不执行格式转换。`pairing.py` 可按归档事件或相对秒平移序列，差值只在同一时刻两侧都有值时产生，不补缺采。

## 事件

事件标记使用稳定 ID，显示文案不参与匹配或对齐。`charts.events.<id>` 声明 `label` 和 `source`：阶段边界使用 `source: stage`、`stage`、`boundary: start | end`；case 内事件使用 `source: case`、`event`。编译时校验阶段引用，阶段边界由运行框架自动记录；复杂程序在真实事件发生处调用 `ctx.record_event(id)`。归档保存事件 ID、epoch 秒与 monotonic 秒；阶段结束记录还保留状态，失败标记附带状态文案，阶段结束不代表操作成功。

`charts.event_ids` 显式声明所有面板的事件选择；`charts.panels[].event_ids` 可以覆盖它，空列表表示该面板不显示事件。未选中事件仍保留在运行证据中，不在图上绘制；未发生的事件不补造。默认视图的指标面板共享它声明的事件选择。

公共投影按真实 `epoch_s` 和图表的 `timeOriginEpochS` 计算相对秒，冻结到 spec 的 `events` 与各 panel 的 `events`。专属图表可使用自己的观测或测量原点，重新装配不得改成 workload 原点。事件源缺少有效时间直接失败，不读取日志、文件或其他来源兜底。对照报告按事件 ID 对齐；同 ID 多次发生时不能推断唯一对齐点。

## Bundle 与发现

`write_bundle` 写出自包含 HTML、`analysis.json`、`report-spec.json` 和最后写入的 `manifest.json`。`read_bundle` 校验 identity、manifest 与 SHA，`load_analysis` 接受带 `report_analysis_schema_version` 的分析封装或 bundle；原始 case 分析不是报告封装。三类落盘 JSON 分别校验自己的版本字段，不按相同的版本数字推断格式，也不改写归档。

`kind` 为 run 或 comparison，表示单 run 或运行对照；`role` 由生产者明确声明。`discover_reports(root, kind=..., role=...)` 按这两个字段发现并校验报告，不从目录名猜测。门禁使用 `role="gate"`。

生产阶段未完成的失败运行生成默认报告并标明未生成的视图；已经存在但损坏的 bundle 必须报错。报告重新装配只补充归档运行信息与曲线，原始 verdict 保持不变。

通用 bundle 外的输入保真度诊断由 `reporting/fidelity.py` 生成 `fidelity.html`，使用公共渲染器展示冻结阈值、ECDF 与联合密度；它不参与运行门禁或 bundle 发现。

## 展示与交互

专属视图的 `charts.curves` 每条声明都必须被面板的 `curve_ids` 引用，未接入面板的样式报错。所有选中视图都显式声明 `metrics.diagnostic_only`（可以为空）：每个查询与 Python 输出序列必须被实际面板使用或列为诊断；已注册 gate producer 的标量单列为 `GATE_EVIDENCE`。编译检查当前查询集合，报告装配再次检查冻结指标，并在诊断区记录 `metric_classification`；自动 `up` 查询用于采集完整性，Python 产出的门禁标量保留既定判定与有效性证据，不强制画成曲线。

面板可选的 `presets` 使用 `visible`、`names`、`contains` 或 `groups` 选择当前面板的曲线，展开结果进入 spec 和 HTML 的切换按钮。未声明时不生成额外按钮；曲线和面板的默认可见性由视图明确声明，缺采仍按实测值处理。全量默认视图的概要/明细 presets 负责不同的投影，不与专属视图的曲线选择器混用。

单 run 标题统一由运行身份生成 `case : variant : profile`，视图 YAML 不声明或覆盖标题。副标题由视图的 `report.subtitle` 提供；离线重生成使用证据中冻结的运行身份。`run_meta` 只展示本次已归档的制品、配置、模型、拓扑、输入和播放参数，不拼接其他运行。公共组件按字段分组，长配置可展开。

门禁检查使用 `CheckNode` 树展示冻结结果中的父检查与具体子检查；空子列表不隐藏父结果。外层 action 的 `actual` 只记录 verdict，完整结果用校验后的工件路径引用，不把整份结果及窗口序列塞入表格单元格。门禁检查默认展开；有效性、诊断与附件使用同一折叠组件。较大的实际值展示摘要，完整值保留供展开，原始证据仍留在运行目录。

场景 YAML 的 `reporting.time_axis` 决定整个 run 的报告零点与默认展示范围，所有视图和时间面板共享，不由 view YAML 分别定义。`origin` 选择 `{event: <id>}` 或 `{stage: <id>, boundary: start|end}`；`range.from` 使用 `origin` 或同型选择器，`range.until` 使用同型选择器。阶段名在编译期验证；事件 ID 引用执行层的实际记录，运行后必须唯一发生。选择自定义 case 事件时由 Python 调用 `ctx.record_event()`，不得把启动请求的时刻冒充操作实际生效时间。

`traffic_started` 是公共层保留的事件：从客户端已发出请求的流水取最早 `send_start_epoch_ms`，包含失败与尚未完成的请求；多发流器取全 run 最早发送时间，并冻结来源资源和请求身份。场景也可选择故障或其他阶段作为零点；零点与展示起点独立，故障前数据可以显示为负秒。例：

```yaml
reporting:
  time_axis:
    origin:
      event: traffic_started
    range:
      from: origin
      until:
        stage: drain
        boundary: end
```

公共层将声明、解析后的绝对时间与来源记录冻结到 `workload-evidence.json`、`analysis.json` 和 `metrics.json` 的运行上下文。曲线、事件和配置参考线统一平移；原始指标仍保存绝对时间。默认范围控制视窗，不删除归档点；没有数据时不补零，不增加采集或来源回退。成功运行缺少边界事件、重复匹配或范围逆序必须报错；失败运行的事件未发生时保留门禁和错误信息，明确标示时间轴未建立，不选择替代零点。

门禁窗口与展示时间轴相互独立。性能展示序列使用完整业务流量周期，覆盖 warmup 和请求排空；门禁仍使用冻结的 measurement 窗口与请求集合。离线发布从指标归档读取冻结的报告时间轴，不按当前场景 YAML 重新选择零点。

时间曲线支持拖拽、输入秒数和还原区间，各时间面板同步。读数显示当前值与选区有效样本的等权 avg，并显示有效/总点数；缺采不补零。avg 用于阅读，不是时间加权均值，也不重判。全量 HTML 可以降采样，但完整序列与统计留在指标归档。

## 冻结对照

`scripts/commands/compare_runs.py` 接收至少两个冻结的 run bundle 目录或 `experiment.eval.zip`，允许混合输入。bundle 通过公共读取器校验清单与全部文件摘要；实验容器通过 `ExperimentReader` 校验所消费的执行记录、冻结布局和数据集。原始附件、普通 HTML、零散 JSON 与 comparison bundle 不能作为单 run 输入。容器缺少主分析时仅保留冻结执行记录，不补造业务结论。

`--label` 按输入顺序重复传入唯一名称；`--baseline` 选择其中一个名称作为条件比较基准，默认第一项。`--view` 按输入顺序指定冻结 ZIP 视图名，`auto` 使用自动选择；bundle 对应项只能为 `auto`。同一冻结输入不能重复当作多个实验。

```bash
python3 scripts/commands/compare_runs.py <run-bundle> <experiment.eval.zip> \
  --label reference --label candidate --baseline reference \
  --view auto --view <frozen-view.yaml> \
  --alignment-event <event-id> --output <comparison-output>
```

对照只读取冻结结果，不调用领域分析器、不重跑门禁。各 run 的 verdict、validity、执行与清理记录保持独立，comparison 不产生总体 PASS/FAIL。运行清单和差异矩阵展示实验身份、代码、配置、流量、环境、时间及判定条件；未知字段不能视为相同。存在多个差异时不能将曲线变化自动解释为某一个参数的因果效果。

默认按基准报告的 panel ID 组织多指标图，保留指标组合、轴、颜色、隐藏状态和原预设。同 ID 图内再按指标 ID、标签、单位、测量与计算口径、来源、缩放、轴定义和时间原点配对；不要求整张图的曲线集合相同。至少一个指标兼容时合图，未匹配的基准指标保留，其他 run 的未匹配指标单独展示。同指标出现在多个不同 ID 图中，不据此自动映射图表。图例支持按 run 和指标选择，run 由线型区分，颜色保持指标含义。身份未知、口径不兼容、非时序曲线单独展示。同一 run 同一指标身份对应不同数据时不推断是哪条，保留原曲线。缺采不补零，不插值，不自动合并统计窗口。

`--alignment-event` 使用唯一且有效的事件 ID，将时间曲线和交互统计样本同时平移；任一 run 的事件缺失或重复时，所有 run 保留原时间轴。默认保留原零点，合图采用基准报告声明的显示范围，不用范围外采样点扩张横轴；未匹配图保留所属 run 的范围。未请求对齐时，原点语义由冻结的报告时间轴 origin selector 或 clock.origin_kind 明确提供，仅文字标签相同不构成兼容。不同或未知原点的图不共享选区，各 run 内的图保持同步。显式事件对齐后显示范围同步平移。采样点、窗口长度和各 run 的原判定不因对齐变化。

默认仅展示合图和未配对曲线；`--include-original-panels` 同时保留全部原图。输出包含自包含 `report.html`、冻结比较分析、spec、manifest 和按输入编号保存的单 run HTML 附件；附件同样进入摘要校验。ZIP 原图只从冻结布局重绘，不执行容器中的程序，也不引用原机器。完整比较 bundle 可独立搬迁；单独 HTML 中的图表与表格可离线打开，原图链接需携带附件。

## 门禁与发布

在线门禁独立保存 `<gate>-gate-evidence.json`、`<gate>-gate-result.json` 和 `<gate>-gate-manifest.json`。manifest 最后原子提交，记录输入和结果的摘要；读取时校验固定文件名和摘要。缺少提交的 manifest 表示尚未产生门禁，已提交工件缺失或校验失败属于损坏，不能回退成普通缺失视图。

每个专属视图由 program 的 `ReportView` 显式注册 renderer。在线执行在资源清理、证据分析和数值投影完成后调用 renderer，直接写出包含运行上下文的最终 bundle。公共报告层不读取旧 bundle 补写运行上下文；重绘只消费校验后的冻结结果和归档指标，不重判。缺少未提交门禁且运行已失败时，可生成默认监控视图，并明确显示未生成的视角。

HA 的逐阶段检查直接根据已选请求证据产生 `CheckResult`，其输入选择、样本数、窗口边界与实测值进入阶段结果。收尾阶段从已落盘的阶段检查发布门禁标量，不重新测量或比较；门禁判定不依赖指标写入或 HTML 成功。公共报告元信息只接受明确的 `run_meta`，未知信息保留为空，不转换旧 `meta` 字段补造配置。

## 数据边界

每个实例统一交付 `experiment.eval.zip` 主工件与 `experiment-<sha256>.raw.zip` 原始附件，由 `result.json.artifacts` 定位。主工件冻结编译计划、代码与配置依赖摘要、视图声明、执行与清理记录、请求流水、门禁输入及结果、指标定义和全量序列。原始附件保存日志及其他非结构化文件；主工件记录附件整体摘要与文件清单。工作目录保留用于即时诊断，ZIP 是可独立搬迁的交付格式，HTML 是可再生的展示产物。

容器索引使用 `ArtifactEntry`、`ArtifactRole`、`MetricDescriptor` 和 `ViewDescriptor`。文件按内容 SHA 存储，相同内容的多个逻辑路径引用同一个 blob；物理来源路径只作为 provenance，`ExperimentReader.resolve_source` 在包内解析，不访问原机器。视图布局和点数组分开保存，多个视图重复选择的数据按内容复用。主工件不重复收录工作目录中的整套报告 bundle 和 HTML。已有报告仍保留其独立的目录协议。

指标目录区分 `raw_observation`、`query_result`、`derived`：原始采样保留实际时间、值、身份和缺采；查询结果保留 PromQL 与查询定义；Python 加工结果保留 producer、计算声明及逐序列来源。门禁输入独立冻结，不用最终指标序列替代较早窗口的输入快照。主分析结果引用唯一的 `metrics.json`，读取时按冻结零点还原展示序列；时间变换不改变原始采样。未知 case 数据作为有角色的扩展文件保留，不按文件后缀推断业务指标含义。

ZIP 使用 Deflate level 3，已压缩文件直接存储；请求流水、日志及压缩证据均完整保留，不按大小裁剪头尾。文件通过流式快照计算摘要与压缩，manifest 最后提交，容器原子替换。读取器按需校验所消费的 blob；完整校验同时检查冻结门禁 manifest 的输入摘要，原始附件可单独校验。符号链接和秘密名称文件的遗漏原因进入完整性记录；缺少证据或不可用的报告视图明确记录，不包装成完整实验。

`ExperimentReader` 从冻结布局重绘已有专属视图；主分析与默认视图声明齐全时，也可生成未在线发布的默认全指标视图，不读取当前 YAML、不重新判定门禁。其他新视图仍需明确的数据需求与兼容的 renderer；容器不执行内嵌 Python。主工件记录实验执行与报告交付结果，归档交付本身由外部实例检查点记录，避免将归档阶段当成实验操作。

报告发布将分析结果中的 `series`、`statistic_sources` 无损存为 gzip 证据工件，`analysis.json` 通过 `series_evidence` 引用工件摘要与字段清单。证据按序列编码，不额外构建完整序列 JSON 字符串；压缩不删除采样点，也不改变缺采、时间戳、单位或来源。发布阶段复用相同序列对象的压缩结果，复用范围只覆盖一次视图发布，不跨运行缓存可变分析结果。

压缩证据在 `reports/evidence/` 按内容摘要存储，并硬链接到各 bundle；不支持硬链接的文件系统使用复制。每份 bundle 的 manifest 校验其本地证据文件，单独复制 bundle 后仍可离线读取，不依赖原目录的共享池。`load_analysis` 校验引用并还原完整分析结构；直接读取 `analysis.json` 得到的是归档结构。HTML 和报告 spec 各自保留交互统计所需的数据，可独立打开和重绘。冻结对照通过公共读取接口取得完整结果，不把压缩方式解释为测量口径差异。

一次多视图发布通过 `frozen_metric_reads` 复用已解析的 `metrics.json`。缓存使用绝对路径与文件设备、inode、大小和修改时间识别工件，原子替换后重新读取；发布结束或异常退出时释放缓存。该范围从报告生成开始，不覆盖采集、指标生产或门禁阶段。视图只读冻结指标，不修改共享的 `MetricStore`；原始归档与门禁判定保持独立。

`RunPresentation` 在运行分析进入展示层时解析 identity、validity、检查树和 provenance；内部通过属性访问。`FrozenReport` 与 `ChartPanel` 在归档进入对照时解析报告契约，原始 case analysis 只作为附件展示。外部 YAML/JSON 在入口严格校验，序列化保持 JSON 工件格式。

视图映射在 `view_schema.py` 的加载边界校验。`view_model.py` 定义曲线、面板、坐标轴、预设选择、章节和字幕模板对象，渲染组件通过这些对象访问展示字段。坐标轴上下界必须有限且 `min < max`，一个面板不能重复选择同一曲线；字幕只允许 `{verdict}`、`{monitoring_status}` 普通占位符，不接受格式规格或转换。视图文件和报告产物仍采用数据格式，测量值与门禁结论不由展示模型改变。
