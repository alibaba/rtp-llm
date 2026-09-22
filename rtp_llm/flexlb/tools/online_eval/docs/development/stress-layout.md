# 压测代码边界

- [`scripts/commands/run_stress.py`](../../scripts/commands/run_stress.py)：日常压测入口；使用 `src/runtime/harness.py` 的 `EnvManager`、`ClientOps`、`ProcessOps` 启动与清理 Java 进程。
- `src/runtime/stress.py`：压测独有的流量、Prometheus、JFR、多 shard 和归档编排；不另起一套 Java 启动器。
- [`scripts/commands/`](../../scripts/commands/)：日常命令；[`scripts/pipeline/`](../../scripts/pipeline/)：内部执行和证据加工步；[`scripts/probes/`](../../scripts/probes/)：低频 mock/real 逼真度探针。
- `src/analysis/`：证据归位、聚合和比较；`src/reporting/`：报告 spec 与 HTML；`src/monitoring/`、`src/traffic/`、`src/artifacts/`：监控、流量和归档组件。

入口完整映射见[入口清单](../reference/entrypoints.md)，运行参数见[参数参考](../reference/parameters.md)。运行数据写到 `run/` 或指定输出目录，不提交进代码。
