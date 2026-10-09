# 场景配置

本目录按 case 名平铺 schema v2 YAML。每个 case 声明 `program: default`；`variants` 只增加额外测试点。运行清单由 `profiles`、变体和 `config/suites.yaml` 决定，以 `list_cases.py` 输出为准。

字段顺序、配置覆盖和新增流程见[新增 case](../../docs/development/adding-cases.md)。执行方式见[运行测试](../../docs/development/running.md)。

`test.monitoring.query_plan` 选择[指标集合](../monitoring/README.md)，`reports` 选择报告视图。未声明 workload 视图时生成精简门禁报告；全量视图 `workload.yaml` 需显式选择。运行后遍历 `result.json → workload.reports` 取齐 HTML，见[结果与指标](../../docs/development/results.md#收取产物)。
