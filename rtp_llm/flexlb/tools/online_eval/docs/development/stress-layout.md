# 压测代码边界

- [`scripts/stress/run_online_eval.sh`](../../scripts/stress/run_online_eval.sh)：压测编排入口；`scripts/stress/lib/` 是 shell 辅助函数。
- `scripts/{aggregate_run,consolidate_run,compare_ab,compare_twin,render_report}.py`：日常数据处理和报告命令。
- `src/analysis/`：压测证据合并、聚合与 real/mock 对照。
- `src/reporting/`：报告 spec、HTML 渲染与内嵌资产。
- `src/monitoring/`、`traffic/`、`artifacts/`：各类测试共用的监控、流量和归档组件。

流程见[压测 runbook](stress.md)，参数见[参数参考](../reference/parameters.md)。运行数据写到 `run/` 或指定输出目录，不提交进代码。
