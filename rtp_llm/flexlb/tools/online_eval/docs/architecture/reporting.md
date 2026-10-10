# 报告契约

报告展示已确定的结果，不负责取证或决定门槛。case 专属报告位于 `cases/<case>/report.py`，公共 `reporting/` 提供曲线绑定、物化与面板投影、配对、spec 校验、renderer 与 bundle 校验。曲线统一处理单位缩放、时间原点和来源；选中面板保留曲线的 `hidden`，不再用生产者另算全局可见性。面板只有有限实测值才算有效，零值有效，null/NaN 不补零，缺测使用 `empty_caption`，部分缺测列出缺少的曲线。配置门禁线由 case 在投影后添加，不算实测数据。case 程序在 `REPORT_VIEWS` 中按文件名声明 `ReportView` 校验与可选渲染能力；`cases.registry` 自动汇集已注册程序的声明，同一视图可以复用相同能力；能力冲突或声明非法直接报错；YAML 只能引用视图文件，不能指定 Python 模块。

## 输入与装配

视图 YAML 在 `config/report_views/` 使用 `report_view_schema_version: 1`。顶层顺序固定为版本、`kind`、`report`、`metrics`、`charts`、`sections`，只出现当前视图需要的块；不允许把文案、曲线策略、表头或 case 私有字段追加到顶层。

| 块 | 职责 |
|---|---|
| `kind` | `default` 展示全部已归档指标，`selected` 展示 YAML 选择的指标；不能在 YAML 指定 Python 模块 |
| `report` | `subtitle`；已生产报告另声明 `id`、`producer`，用于定位和校验 bundle |
| `metrics` | `query_plan` 选择指标集合；`diagnostic_only` 明确哪些已采指标不绘图 |
| `charts` | `curves`、`panels`、事件显示名和时间轴文案；全量视图在这里声明分组、采样和曲线可见性 |
| `sections` | 按稳定 section ID 声明附录标题、表头和默认开合状态 |

`sections.<id>` 只包含 `title`、`opened` 和表格的 `columns`。`checks` 表示专属门禁说明，`monitoring`、`validity`、`metrics`、`sources` 表示相应诊断；case 专属附录由注册的 Python 校验器声明。每个 case 校验附录 ID 集合及表格列数，不能依赖生产时的 KeyError。表格和详情使用相同开合组件，Python 决定内容和顺序，YAML 只控制展示。标准 KPI 的标签由公共组件统一提供，视图不另造一套 KPI 身份。

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

通用 bundle 外的输入保真度诊断由 `reporting/traffic_fidelity.py` 生成 `fidelity.html`，提供阈值、ECDF 与联合密度交互；它不参与运行门禁或 bundle 发现。

## 展示与交互

专属视图的 `charts.curves` 每条声明都必须被面板的 `curve_ids` 引用，未接入面板的样式报错。所有选中视图都显式声明 `metrics.diagnostic_only`（可以为空）：每个查询与 Python 输出序列必须被实际面板使用或列为诊断；已注册 gate producer 的标量单列为 `GATE_EVIDENCE`。编译检查当前查询集合，报告装配再次检查冻结指标，并在诊断区记录 `metric_classification`；自动 `up` 查询用于采集完整性，Python 产出的门禁标量保留既定判定与有效性证据，不强制画成曲线。

面板可选的 `presets` 使用 `visible`、`names`、`contains` 或 `groups` 选择当前面板的曲线，展开结果进入 spec 和 HTML 的切换按钮。未声明时不生成额外按钮；曲线和面板的默认可见性由视图明确声明，缺采仍按实测值处理。全量默认视图的概要/明细 presets 负责不同的投影，不与专属视图的曲线选择器混用。

单 run 标题统一由运行身份生成 `case : variant : profile`，视图 YAML 不声明或覆盖标题。副标题由视图的 `report.subtitle` 提供；离线重生成使用证据中冻结的运行身份。`run_meta` 只展示本次已归档的制品、配置、模型、拓扑、输入和播放参数，不拼接其他运行。公共组件按字段分组，长配置可展开。

门禁检查默认展开；有效性、诊断与附件使用同一折叠组件。较大的实际值展示摘要，完整值保留供展开，原始证据仍留在运行目录。

时间曲线支持拖拽、输入秒数和还原区间，各时间面板同步。读数显示当前值与选区有效样本的等权 avg，并显示有效/总点数；缺采不补零。avg 用于阅读，不是时间加权均值，也不重判。全量 HTML 可以降采样，但完整序列与统计留在指标归档。

## 冻结对照

`comparison.py` 接收多个 run bundle，先校验再读取结果与 spec，不导入 case 分析器、不读取原始 evidence、不重建单 run 报告。每侧 verdict、KPI、缺采、说明与原报告链接保留；输入归档应留在原位置。

合图只配对 ID、指标集合、单位及坐标轴一致的时间面板。时间原点不一致时分别展示；指定事件缺失、重复或无效时全部保留原坐标，不推导统计窗口、排名或原因。

控制变量来自冻结的 configuration、workload、environment 与 criteria。字段缺失标 UNKNOWN，不从两边同时缺失推断一致。对比不产生顶层 verdict，命令退出码只描述读取、校验与写入是否成功；入口见[命令导航](../development/entrypoints.md)。

离线重判必须显式指定 `--reinterpret`，输出目录位于原证据归档之外且为空。原归档只读；来源证据 SHA 与当前分析器 SHA 写入重判证据，指标物化和报告发布只写新目录。`--json-only` 同样遵守目录保护和溯源规则。
