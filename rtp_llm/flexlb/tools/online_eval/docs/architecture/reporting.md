# 报告契约

报告展示已确定的结果，不负责取证或决定门槛。case 专属报告位于 `cases/<case>/report.py`，公共 `reporting/` 提供曲线绑定、配对、spec 校验、renderer 与 bundle 校验。case 视图种类通过 `cases.registry.VIEW_KINDS` 注册校验和渲染实现；YAML 只能引用视图文件，不能指定 Python 模块。

## 输入与装配

视图 YAML 在 `config/report_views/` 声明曲线、名称、分组、单位、轴、颜色和面板。曲线用 `metric_id` 与 `labels` 选择冻结指标，面板用稳定的 `curve_ids` 选择曲线；展示名不参与判定或身份匹配。`reporting/catalog.py` 只提供通用调色板和 UI 主题，不按指标名称推断单位、分组或业务含义。全量诊断视图按冻结指标的 `unit` 标注坐标轴，保留原始指标身份。

报告读取 `metrics.json` 中的冻结定义与序列，不重新查询服务、解析日志或生产数值指标。复杂门禁的报告必须接收明确结果，不能在 result 缺省时隐式重判。查询与 producer 规则见[指标契约](metrics.md)，主报告、全量 opt-in 和产物清单见[结果与指标](../development/results.md#收取产物)。

所有生产器直接输出同一 panel 契约：声明 `axes`，每条 series 携带 `points: [{x, y}]`；`timeX: true` 表示时间坐标，缺采以 `y: null` 保留。类目图的 `x` 是字符串，数值图的 `x` 是数值。`spec.py` 严格校验输入，不接受旧的 `x`/`xNums` + `series.data` 或显示模式标志，也不执行格式转换。`pairing.py` 可按归档事件或相对秒平移序列，差值只在同一时刻两侧都有值时产生，不补缺采。

## Bundle 与发现

`write_bundle` 写出自包含 HTML、`analysis.json`、`report-spec.json` 和最后写入的 `manifest.json`。`read_bundle` 校验 identity、manifest 与 SHA，`load_analysis` 接受已有分析 JSON 或 bundle，不改写旧归档。

`kind` 为 run 或 comparison，表示单 run 或运行对照；`role` 由生产者明确声明。`discover_reports(root, kind=..., role=...)` 按这两个字段发现并校验报告，不从目录名猜测。门禁使用 `role="gate"`。

生产阶段未完成的失败运行生成精简报告并标明未生成的视图；已经存在但损坏的 bundle 必须报错。报告重新装配只补充归档运行信息与曲线，原始 verdict 保持不变。

通用 bundle 外的输入保真度诊断由 `reporting/traffic_fidelity.py` 生成 `fidelity.html`，提供阈值、ECDF 与联合密度交互；它不参与运行门禁或 bundle 发现。

## 展示与交互

单 run 标题为 case、variant、profile，副标题由视图提供。`run_meta` 只展示本次已归档的制品、配置、模型、拓扑、输入和播放参数，不拼接其他运行。公共组件按字段分组，长配置可展开。

门禁检查默认展开；有效性、诊断与附件使用同一折叠组件。较大的实际值展示摘要，完整值保留供展开，原始证据仍留在运行目录。

时间曲线支持拖拽、输入秒数和还原区间，各时间面板同步。读数显示当前值与选区有效样本的等权 avg，并显示有效/总点数；缺采不补零。avg 用于阅读，不是时间加权均值，也不重判。全量 HTML 可以降采样，但完整序列与统计留在指标归档。

## 冻结对照

`comparison.py` 接收多个 run bundle，先校验再读取结果与 spec，不导入 case 分析器、不读取原始 evidence、不重建单 run 报告。每侧 verdict、KPI、缺采、说明与原报告链接保留；输入归档应留在原位置。

合图只配对 ID、指标集合、单位及坐标轴一致的时间面板。时间原点不一致时分别展示；指定事件缺失、重复或无效时全部保留原坐标，不推导统计窗口、排名或原因。

控制变量来自冻结的 configuration、workload、environment 与 criteria。字段缺失标 UNKNOWN，不从两边同时缺失推断一致。对比不产生顶层 verdict，命令退出码只描述读取、校验与写入是否成功；入口见[命令导航](../development/entrypoints.md)。
