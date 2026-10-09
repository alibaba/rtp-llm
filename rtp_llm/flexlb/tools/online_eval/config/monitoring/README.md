# 指标配置

场景通过 `test.monitoring.query_plan` 选择本目录的 YAML。指标定义与实际数据一起冻结在运行目录的 `metrics.json` 中；报告与门禁使用稳定指标 ID，不按展示名或 exporter 名猜测。运行与报告契约见[报告装配](../../docs/architecture/reporting.md)。

```yaml
schema_version: 2
include: [workload.yaml]
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
