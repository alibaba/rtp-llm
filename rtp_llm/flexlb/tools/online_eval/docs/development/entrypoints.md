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
| 分析 | `scripts/commands/compare_traffic.py` | 对比合成流量与真实捕获的统计结构 |
| 离线 | `python3 -m workload.performance_gate` | 从逐请求证据重算单 run 性能结论 |
| 离线 | `python3 -m workload.performance_compare` | 比较两份性能证据和声明的配置差异 |
| 离线 | `python3 -m workload.cache_gate_ab` | 比较缓存实验的两份运行归档，不裁决 A/B 成败 |
| 探针 | `scripts/probes/check_mock_fidelity.py` | 独立检查 mock 与 real 的分布逼真度 |

压测与 case 共用 `src/runtime/` 的 Java 进程启动和清理组件。压测独有的 Prometheus、JFR、分片负载和归档编排在 `src/runtime/stress.py`。数据输入见[数据目录](../../data/README.md)，手写配置见[配置目录](../../config/README.md)。

模块入口需设置 `PYTHONPATH=tools/online_eval/src:tools/online_eval`。缓存实验 A/B 的通用形式：

```bash
PYTHONPATH=tools/online_eval/src:tools/online_eval python3 -m workload.cache_gate_ab \
  /path/to/baseline-run /path/to/candidate-run \
  --config /path/to/scenario.yaml --output /path/to/comparison
```

`analysis` 只声明 `comparison: cache_scale_in` 与可选的 `alignment_event`，不接受
`mode` 或 `expected_verdicts`。A/B 仅保留各 run 独立 verdict；控制变量缺失和差异
逐项展示，不推导整场 PASS/FAIL。退出码 0 仅表示报告生成成功，输入错误或写入失败仍报错。

`alignment_event` 指定双方曲线的共同零点；未声明时保留各 run 相对时间。
任一侧事件缺失、重复或时间无效时，报告提示并保留双方原时间轴，不改变单 run 判定。
目录、图例与结构化字段统一使用 A/B；实际 JAR 与配置身份分别展示相同、不同或未知。
