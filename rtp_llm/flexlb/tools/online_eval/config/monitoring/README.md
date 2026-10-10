# 指标配置

`default.yaml` 提供公共默认指标集合；case YAML 通过 `test.monitoring.query_plan` 选择本目录中的集合，case 集合可引用默认集合并显式增删指标。

物理名称、稳定 `metric_id`、曲线 ID、PromQL、继承与缺失规则统一见[指标身份与采集契约](../../docs/architecture/metrics.md)。采集定义和序列冻结到运行目录的 `metrics.json`，Python 门禁和 view 使用同一个指标 ID；报告装配见[报告契约](../../docs/architecture/reporting.md)。
