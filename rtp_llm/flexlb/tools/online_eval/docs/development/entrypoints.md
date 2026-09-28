# 命令入口

从 `rtp_llm/flexlb` 运行时，路径加 `tools/online_eval/` 前缀。具体参数以各命令 `--help` 和[参数参考](parameters.md)为准。

| 层级 | 命令 | 用途 |
|---|---|---|
| 日常 | `scripts/commands/run_stress.py` | 启动压测并采集 Prometheus 证据 |
| 日常 | `scripts/commands/run_cases.py` | 运行功能与场景 case |
| 日常 | `scripts/commands/list_cases.py` | 列出可运行 case |
| 日常 | `scripts/commands/render_stress_report.py` | 从压测证据渲染 HTML |
| 日常 | `scripts/commands/compare_runs.py` | 并排展示两份或多份冻结 run bundle |
| 管线 | `scripts/pipeline/execute_cases.py` | 按 lane 并行执行 case |
| 管线 | `scripts/pipeline/calculate_metrics.py` | 计算派生指标 |
| 管线 | `scripts/pipeline/organize_evidence.py` | 整理证据文件 |
| 管线 | `scripts/pipeline/materialize_traffic.py` | 将流量源物化为请求计划 |
| 管线 | `scripts/pipeline/derive_master_templates.py` | 从固定模型生成 Java 回归夹具 |
| 分析 | `scripts/commands/compare_traffic.py` | 对比合成流量与真实捕获的统计结构 |
| 离线 | `python3 -m workload.performance_gate` | 从逐请求证据重算单 run 性能结论 |

压测与 case 共用 `src/runtime/` 的 Java 进程启动和清理组件。压测独有的 Prometheus、JFR、分片负载和归档编排在 `src/runtime/stress.py`。数据输入见[数据目录](../../data/README.md)，手写配置见[配置目录](../../config/README.md)。

对比统一使用以下入口，输入直接指向含 `manifest.json` 的单 run 报告目录：

```bash
python3 tools/online_eval/scripts/commands/compare_runs.py \
  /path/to/A/reports/run/<report> /path/to/B/reports/run/<report> \
  --output /path/to/comparison
```

可继续追加其他 run bundle。输出位于 `reports/comparison/runs/`，含 HTML、冻结结果和 manifest。
输入按顺序标为 A、B、C；报告展示各 run 独立结论、控制变量差异、缺采和无法配对的面板。
退出码 0 表示报告生成成功；2 表示参数、输入校验或输出写入失败。控制差异和单 run 的
FAIL/INVALID 不影响对比命令成功生成报告。

`--alignment-event <name>` 指定归档事件为共同零点。任一 run 的事件缺失、重复或时间无效时，
全部保留原时间轴并提示原因。未指定时使用归档时间坐标，不推导稳态窗或重新计算统计值。
图表、预设和分析结果来自已校验的 bundle，对比时不读取原始请求、日志或监控采样。
