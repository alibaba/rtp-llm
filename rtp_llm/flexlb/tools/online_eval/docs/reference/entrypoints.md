# 命令入口

从 `rtp_llm/flexlb` 运行时，路径加 `tools/online_eval/` 前缀。具体参数以各命令 `--help` 和[参数参考](parameters.md)为准。

| 层级 | 命令 | 用途 |
|---|---|---|
| 日常 | `scripts/commands/run_stress.py` | 启动压测并采集 Prometheus 证据 |
| 日常 | `scripts/commands/run_cases.py` | 运行功能与场景 case |
| 日常 | `scripts/commands/list_cases.py` | 列出可运行 case |
| 日常 | `scripts/commands/render_stress_report.py` | 从压测证据渲染 HTML |
| 日常 | `scripts/commands/compare_runs.py` | 对比两次运行；Prometheus 归档仅做描述性比较 |
| 管线 | `scripts/pipeline/execute_cases.py` | 按 lane 并行执行 case |
| 管线 | `scripts/pipeline/calculate_metrics.py` | 计算派生指标 |
| 管线 | `scripts/pipeline/organize_evidence.py` | 整理证据文件 |
| 管线 | `scripts/pipeline/materialize_traffic.py` | 将流量源物化为请求计划 |
| 管线 | `scripts/pipeline/derive_master_templates.py` | 从固定模型生成 Java 回归夹具 |
| 探针 | `scripts/probes/check_mock_fidelity.py` | 独立检查 mock 与 real 的分布逼真度 |

压测与 case 共用 `src/runtime/` 的 Java 进程启动和清理组件。压测独有的 Prometheus、JFR、分片负载和归档编排在 `src/runtime/stress.py`。数据输入见[数据目录](../../data/README.md)，手写配置见[配置目录](../../config/README.md)。
