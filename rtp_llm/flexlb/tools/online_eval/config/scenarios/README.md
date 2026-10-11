# 场景配置

本目录按 case 名平铺 schema v2 YAML。每个 case 声明 `program: default`；`variants` 只增加额外测试点。运行清单由 `profiles`、变体和 `config/suites.yaml` 决定，以 `list_cases.py` 输出为准。

字段顺序、配置覆盖和新增流程见[新增 case](../../docs/development/adding-cases.md)。执行方式见[运行测试](../../docs/development/running.md)。

`execution.monitoring.query_plan` 选择[指标集合](../monitoring/README.md)，公共集合为 `config/monitoring/default.yaml`。

`reports` 选择报告视图。未声明时使用 `default.yaml`，按归档数据决定是否显示指标；声明专属视图时只生成所选报告。运行后遍历 `result.json → workload.reports` 取齐 HTML，见[结果与指标](../../docs/development/results.md#收取产物)。
