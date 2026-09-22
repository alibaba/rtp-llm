# 命令入口与迁移

从 `rtp_llm/flexlb` 运行时，以下路径加 `tools/online_eval/` 前缀。`commands/` 是 runbook 直接使用的五类命令；`pipeline/` 只由入口或维护流程调用；`probes/` 用于独立验收。`materialize_traffic.py` 与 `derive_master_templates.py` 是流量模型内部维护工具，位于 `pipeline/`。

| 旧入口 | 新入口 | 主要同步点 |
|---|---|---|
| `scripts/stress/run_online_eval.sh` | `scripts/commands/run_stress.py` | 压测 runbook、监控/流量/配置测试；Whale bundle 无引用，shell 及 `lib/load_client.sh` 一并退役 |
| `scripts/test_runner.py`、根层 `test_runner.py` | `scripts/commands/run_cases.py` | 功能/场景 runbook、suite 说明 |
| `scripts/scenario_runner.py`、根层 `scenario_runner.py` | `scripts/commands/list_cases.py` | case 列表命令、`src/runtime/instance_runner.py`、子进程协议测试 |
| `scripts/parallel_runner.py`、根层 `parallel_runner.py` | `scripts/pipeline/execute_cases.py` | `run_cases.py` import、端口锁和并行执行测试 |
| `scripts/aggregate_run.py` | `scripts/pipeline/calculate_metrics.py` | 内部聚合入口；指标由 `src/analysis/aggregate.py` 唯一计算 |
| `scripts/consolidate_run.py` | `scripts/pipeline/organize_evidence.py` | 内部文件归位入口；不负责统计计算 |
| `scripts/compare_ab.py` | `scripts/commands/compare_runs.py` | 压测 runbook、CLI 可达性测试；比较两次运行及配置 |
| `scripts/compare_twin.py` | `scripts/probes/check_mock_fidelity.py` | CLI 可达性测试；比较 mock/real 分布，不用于迁移前后等价判定 |
| `scripts/render_report.py` | `scripts/commands/render_stress_report.py` | 压测 runbook、CLI 可达性测试；产出 HTML |
| `scripts/materialize_traffic.py` | `scripts/pipeline/materialize_traffic.py` | 压测 Python 编排及交通模型测试；模型维护工具 |
| `scripts/derive_master_templates.py` | `scripts/pipeline/derive_master_templates.py` | `data/README.md`、派生模板测试；模型维护工具 |

`run_cases.py` 调用 `execute_cases.py`，后者按 lane 启动 `list_cases.py` 的执行协议。单独列清单只运行 `list_cases.py --list-json`。压测运行 `run_stress.py`，以共享 Python runtime 启动 JVM，并调用专业 Prometheus 会话生成曲线。

旧 shell 的 `START_MOCK=0`、`START_FLEXLB=0`、`FLEXLB_START_CMD`、`FLEXLB_NETWORK_ISOLATED` 和并发检查关闭开关在仓库里没有调用方；新入口只编排本次拥有的完整 Java 环境。需要接外部进程时应新增明确的 Python 输入与进程所有权契约，不能复用这些隐式环境变量。
