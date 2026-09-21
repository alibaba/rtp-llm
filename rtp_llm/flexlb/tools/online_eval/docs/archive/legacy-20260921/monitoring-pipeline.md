> [!WARNING]
> 历史快照：此文件不是当前操作依据。当前入口为 `tools/online_eval/README.md`。

# 实验监控与专用测试证据

曲线的数据源只能是监控系统。线上使用 KMonitor；线下使用 Prometheus 采集同一套生产指标定义。HTTP `/metrics` 是监控 exporter，不是从业务 API 推测指标的适配器。

`/snapshot`、控制 API、日志、请求 JSONL 只用于专用测试的断言、请求关联和故障取证，不能补齐缺失的监控曲线。例如 snapshot 的 `running` 是未完成任务集合，包含等待请求，不能当作 `rtp_llm_running_stream_size`。缩容门禁已经改用标准执行/等待指标。

## 运行方式

执行机器需要 Prometheus。设置 `PROMETHEUS_BIN=/absolute/path/to/prometheus`，或把二进制加入 PATH。已验证版本为 3.12.0。缺少二进制或目标首次采样失败时，实验报错，不降级到 Python poller。

框架 workload 和 `stress/run_online_eval.sh` 共用 `online_eval.monitoring.PrometheusSession`：

1. 每个环境启动独立的 Prometheus，抓取 mock `/metrics?per_engine=true`、各 Master `/prometheus`。
2. 客户端用 Micrometer Counter/Timer histogram 暴露 `/metrics`。发现文件只登记端点；首次采样成功后，框架写入 ready 标记，客户端才开始发流量。
3. 专用测试读取 TSDB 中的实际 scrape 样本。多个消费者不会重复访问会推进 TPS 窗口的 mock exporter。
4. 报告通过 PromQL 得到聚合序列。窗口速率采用先对各 counter 做 `rate` 再聚合；客户端延迟用 `histogram_quantile`，不平均各客户端 p99。
5. 结束监控区间后导出查询结果并停止 Prometheus。原始指标只存 TSDB，不再逐秒复制成 `.prom`、JSONL 和展开的每引擎报告。

每个环境的产物：

```
telemetry/<epoch>/
  prometheus.json   # 实际 scrape 配置
  session.json      # 端点、采样间隔、启动时间
  queries.json      # PromQL、起止时间、step、结果、缺失查询与错误
  data/            # Prometheus TSDB，可重新挂载查询
  prometheus.log
reports/run/<id>/   # 统一报告 bundle
```

默认采样为 1 秒，速率窗口为 `max(4 × scrape_interval, 10 秒)`。执行 TPS 与 wall TPS 分开展示，模拟 forward 耗时明确标为 simulated。waiting/running 同时提供按上报引擎的平均、总量和最大值，缩容图默认展示平均值。其分母是该次 scrape 中存在的引擎序列，不能误称为 Master 在册引擎数。

## 按测试需要保留证据

`suites.yaml` 的用例 `collection` 指定专用证据需求；原始 stress 入口使用 `COLLECTION_PROFILE`，默认 `aggregate`。

| 档位 | 监控曲线 | 客户端请求证据 | engine events / 详细诊断 |
|---|---|---|---|
| aggregate | Prometheus | 不写逐请求文件，不累积全部完成请求 | 关闭 |
| request | Prometheus | 保留 | 关闭 |
| diagnostic | Prometheus | 保留 | 保留，用于专用断言/关联 |

已有工作负载用例包含请求完成性、发送时刻或放置验证，因此显式声明 `request`。有逐请求契约的用例不能为了减量改成 aggregate。功能测试保留 diagnostic 的原有证据能力。缓存缩容门禁保留逐请求发送完整性校验；其判定窗口是专用测试证据，报告曲线不会使用这些窗口或 JSONL 重新计算。

旧 `eval_collectors.py` 和 `SharedMetricSource` 已移除。工作负载不再自动执行旧压测聚合或复制所有日志。旧档案解析器与显式请求诊断工具保留用于历史证据复核；新运行不通过它们生成监控曲线。

## 缺失与归档边界

- `up=0` 对应曲线缺口，不填零、不插值、不从日志补值。
- mock running/waiting 和监控健康序列缺失属于采集错误；其他未上报指标记录在 `missing_queries`，不会生成假数据。
- 引擎序列携带 incarnation，避免同名重启拼接。专用缓存门禁仍拒绝重启/重置破坏的判定窗口。
- 客户端开始/结束文件是生命周期边界，不包含曲线数值。突然退出而没有结束边界不能视作正常采集结束。
- TSDB 默认配置 24 小时、1 GB retention。它是 Prometheus 的保留策略，不是包含 WAL/head 的严格磁盘限额。超出保留范围的实验不能声明完整覆盖。
- 正常退出保留 TSDB；重新分析时可对这个目录启动 Prometheus，然后重放归档中的查询。不要同时启动两个进程写同一 TSDB。
- `test_valid` 表示采集证据有效性，`performance_verdict=NOT_EVALUATED`；没有性能门限判定时不会把采集成功包装成性能门禁通过。

Prometheus 的 [HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/) 与 [TSDB 存储及保留策略](https://prometheus.io/docs/prometheus/latest/storage/) 为时间序列查询和存储依据。Python 只做生命周期编排、结果归档与呈现。

旧 `stress/compare_ab.py` 的门限建立在历史 aggregate schema 上，只用于旧档案。它明确拒绝新 Prometheus aggregate，避免空字段产生假通过；新实验使用 workload 或 cache-gate 的 A/B 对照。
