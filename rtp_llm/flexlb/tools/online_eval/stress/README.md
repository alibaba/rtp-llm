# 压测实现目录

压测代码按职责分层：

- `run_online_eval.sh`：唯一压测入口，负责进程编排和采集生命周期。
- `lib/`：入口脚本使用的 shell 公共函数。
- `analysis/`：原始采集产物的合并、聚合和 real/mock 对照。
- `reporting/`：报告 spec 与 HTML 输出；`assets/` 保存随报告内嵌的前端资源。

通用监控、流量和报告协议放在相邻的 `online_eval/` Python 包中；这里不放场景执行器、实验结果或历史报告。

操作流程见[压测 runbook](../docs/development/stress.md)，参数见[参数参考](../docs/reference/parameters.md)，结果口径见[结果与指标](../docs/reference/results.md)。历史说明通过 Git 追溯。
