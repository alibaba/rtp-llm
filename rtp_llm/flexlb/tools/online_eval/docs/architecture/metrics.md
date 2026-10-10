# 指标身份与采集契约

case 和报告共用 `metrics.json` 中的稳定 `metric_id`。exporter 的物理指标名、查询 ID、曲线 ID 和展示名承担不同职责；只有指标定义决定取数口径，不能通过展示名反推查询。

## 名称与身份

| 名称 | 所属位置 | 用途 |
|---|---|---|
| 物理指标名 | Java exporter 与 PromQL | 描述服务实际上报的 counter、gauge 或 histogram；保留生产对齐的名称与单位 |
| `metric_id` | 指标集合、Python program、`metrics.json` | 门禁和报告共用的逻辑身份，格式为 `<namespace>/<name>` |
| 标签、来源实例、环境代次 | 指标定义和序列 | 区分角色、引擎、Master、采样窗口等具体序列 |
| 本地 `curve_id` | view YAML 的 `curves` 键 | 选择一个指标的标签投影和展示方式，由面板的 `curve_ids` 引用 |
| `name` | view YAML | 面向读者的图例，可使用中文或变量模板，不参与取数和门禁 |

Prometheus 查询的完整 ID 由 `sources` 的来源种类和查询键组成；例如 `sources.mock.running_avg` 对应 `mock/running_avg`。这里的 `mock` 是采集来源种类，物理名称仍可以是 `rtp_llm_running_stream_size`。Python producer 在 `produced` 中直接声明完整 ID。

命名遵循以下规则：

- 指标 namespace 和名称使用小写 `snake_case`，同一个 ID 在不同集合中保持相同的取数定义、单位、值类型与身份标签。`required` 是集合内的采集要求，可以按用途设置。
- 原始抓取可以使用 exporter 名，或明确的业务简称；映射只在指标 YAML 的 `promql` 中定义。Python 门禁和 view 均引用 `metric_id`，不再另建一套 exporter 到门禁名称的转换表。
- 聚合、分位数和统计口径体现在名称中，例如 `_sum`、`_mean`、`_max`、`_p99`、`_per_engine`；仅有相同来源或单位不能合并 ID。对 priority 样本直接取平均，与先逐引擎求和再取平均，结果可以不同。
- 同一指标的不同角色或实例通过标签、`source`、`epoch` 选择；已有的多角色查询不为每个角色再定义相同 PromQL。需要多个图例时定义本地曲线，均绑定同一个 `metric_id`。
- 改展示名、颜色或面板不改变指标 ID。改变业务含义、归约口径或单位时使用明确的新 ID；所有实际定义及 SHA 随运行冻结，报告读取冻结产物。

例如，一条 `running_avg` 查询可在 view 中分别绑定 `role: PREFILL` 和 `role: DECODE`，使用不同曲线 ID 和中文图例。两条曲线都来自 `mock/running_avg`；门禁也通过这个 ID 和明确的标签、时窗读取数据。

## 指标集合与查询

场景通过 `test.monitoring.query_plan` 选择 `config/monitoring/` 中的集合。`default.yaml` 是公共默认集合，case 集合通过 `include` 引用；它与 `config/report_views/default.yaml` 全量诊断视图是独立配置。默认集合的文件名由 `monitoring.query_plan.DEFAULT_PLAN` 声明。

```yaml
metric_plan_schema_version: 2
include: [default.yaml]
sources:
  mock:
    queue_depth:
      promql: sum(rtp_llm_wait_stream_size${selector})
      unit: requests
      value_kind: gauge
      labels: []
```

上例的指标 ID 是 `mock/queue_depth`。`sources` 支持 `mock`、`client`、`master`；采集器仅注入 `${selector}` 和 `${window_ms}`，运算使用 PromQL 原文。默认 `mode: evaluated`；需要原始抓取时间的门禁声明 `mode: scrape`，此时 PromQL 只接受原始指标名加 `${selector}`。`labels` 声明每条结果必须携带的身份标签，`required: true` 表示缺失该查询会使监控归档失败。

`include` 合并集合，`exclude` 按完整指标 ID 显式删除。重复定义、循环引用和删除不存在的指标均失败；本契约不支持同 ID 覆盖，要调整口径时修改所属集合，或新增不同 ID 并排除旧指标。展开后的定义及 SHA 写入编译产物与运行归档。

难以用 PromQL 表达的请求归因、阶段统计由注册 Python producer 实现，在 `produced` 声明输出 ID、`producer`、`source_type`、`unit`、`value_kind`、`labels`。来源类型为 `derived`、`client_journal` 或 `debug_api`；产物记录 producer 模块及源码 SHA。原始证据保留供审计，不能作为缺失 Prometheus 指标的自动替代。

视图 YAML 的 `curves` 用本地曲线 ID 声明 `metric_id` 和 `labels` 选择，并设置名称、颜色、轴和换算；面板用 `curve_ids` 选曲线。Python program 用 `case.metric(id)` 声明依赖，编译时拒绝未定义 ID。运行时 `MetricStore.select` 显式选择标签与时间窗、检查样本数及最大间隔，`reduce` 只对单条已选序列归约；缺失数据抛出 `MetricUnavailable`，定义冲突抛出 `MetricContractError`，均不能补零。
