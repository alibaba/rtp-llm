# 报告装配契约

报告装配用于把分析结果组织成可离线阅读的图表和审计信息。门禁、阈值、统计值由分析层决定；装配层只决定名称、分组、轴、颜色、显示顺序和说明文字。新增报告时先复用 `reporting/` 的词汇表、配对函数和 bundle 写入入口，不从 HTML 或已写出的报告反向提取数据。

流程为「读取并校验既有分析产物 → 分析层给出结果与有效性 → 装配 spec → `write_bundle`」。`reporting/catalog.py` 管理曲线展示名称、分组、轴、颜色和默认可见性。`reporting/pairing.py` 接收已分析的序列，按阶段边界、事件时间或相对秒平移，配对时保留缺采样的 `null`；差值仅在同一时刻两侧都有数值时产生。累计计数的差分在 `reporting/statistics.py` 定义。新增曲线先补 catalog，再让报告装配器按数据身份取定义；不得以展示名作为统计或门禁依据。

`reporting/assembly.py` 是 panel 适配入口。历史的 `x`/`xNums` + `series.data` 和 `series.points` 均可读取；时间曲线在写入时派生 `points`，保留原字段和空值。overlay 与普通图表可混排，`representation` 指定交互组件；bar/scatter 仍保留其类目或二维坐标。旧归档不改写，`read_bundle` 继续校验原 manifest 和 SHA，`load_analysis` 读取旧 `analysis.json`。新 bundle 仍写自包含 HTML、`analysis.json`、`report-spec.json`、`manifest.json`，由 `write_bundle` 最后写 manifest。

`kind` 表示制品关系：`run` 是单次运行，`comparison` 是两个或多个运行的对照（含时间线），`sweep` 是参数扫描。`discover_reports(root, kind=..., role=...)` 在所选 kind 下读取并校验 bundle；`role` 可选，由生产者声明，不从目录名猜测。默认 kind 为 `run`，兼容现有门禁发现调用。

workload case 可在 YAML 的 `reports` 段引用 `config/report_views.yaml` 中的公共模板，并声明附加的 `gate` 视角；未声明时仍发布默认 run 视角。模板按 epoch、监控源和指标合图，均值与最大值仅作展示投影，逐引擎原线可在指标选择器中查找。HTML 为控制体积保留分桶极值和缺采点；`analysis.json` 保留原始序列全集，spec 中每条线标明来源、计算方式与采样方式。run 与 gate 是独立 bundle，各自保留时间原点，run 仅以相对路径链接 gate。YAML 只选视角和可见性，不定义计算式或门禁判定。

`cache_scale_in` 与 `master_performance` 声明默认 run 和各自分析器产出的 gate；`trace_scale_out` 声明默认 run。`config/report_views/cache_scale_in.yaml` 对应 `cache_scale_in`，`config/report_views/scale_out.yaml` 对应 `trace_scale_out`，但这两份是 `workload.views` 的跨 run 时间线规格，须通过 `--view` 与已有 run 归档显式生成 comparison，不能作为单次运行的 gate 视角。当前模板中的 `gate/*` 和 `statistics/1/per_second/*` 必须先在输入分析产物中确认存在；缺失时 `workload.views` 会绘出空曲线，不能把它当作有效的专属报告。

输入归档保持各自的历史格式。`reporting.core.load_analysis` 接受旧分析 JSON 与带 manifest 的新 bundle；cache 和性能证据仍由对应分析器按原格式读取。不得因为呈现格式统一，就把不同证据类型混作同一分析输入。

独立的 `traffic_fidelity_report.py` 保留自包含 `fidelity.html` 作为显式例外：它提供可调阈值、ECDF 和三幅联合密度热图，通用 renderer 尚无等价交互组件。该例外只影响合成输入保真度诊断，不进入运行门禁或 `discover_reports`。若迁入 bundle，先为这些交互补齐通用组件并逐项核对信息完整性。压测的 `aggregate.py` 子进程链路属于证据生成，不改变装配契约；它继续经 `write_bundle` 发布报告。

`stress_report.py`、`compare_twin.py`、`monitoring/session.py` 的专属装配保留：其图表或监控视图重复度低，整体迁移成本高，均经 `write_bundle` 和 panel 适配入口发布。豁免只覆盖各自的专属布局，不允许新增独立配色、配对或落盘规则；需要复用的呈现决策仍应进入 `reporting/`。
