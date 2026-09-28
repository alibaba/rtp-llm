# 压测

压测用于吞吐、延迟、容量曲线和同配置 A/B。它运行真实 Java Master、Java Mock Engine 与 Java load client，不使用 GPU。

先完成[编译与运行底座](build-and-runtime.md)，再从 `rtp_llm/flexlb` 执行。

## 标准运行

```bash
python3 tools/online_eval/scripts/commands/run_stress.py \
  --run-root /path/to/new-output-root \
  --run-id stress-$(date +%Y%m%d-%H%M%S) \
  --n-prefill 12 --n-decode 40 --master-mode wb \
  --send-mode replay --loop --duration-s 120 --replay-speed 4 \
  --warmup-s 10 --fetch-output-stream 1
```

真实流量从 `data/traffic_trace/` 按文件名选择：`--traffic-model <model-name>`。
默认选择以 `--help` 为准；模型是匿名 prefix DAG，不能传入原始访问日志。
脚本验证模型 SHA，在运行目录生成 `traffic-plan.jsonl` 及 manifest，Java 读取物化后的计划。
事件数与采集跨度从模型 manifest 查询。全量单轮的名义平均速率可按下式估算：

```text
speed = target_qps × event_span_seconds / valid_request_count
```

`--traffic-model` 与 `--traffic-source-spec` 互斥。换源后记录模型 SHA 和发送节奏，
重新确认采集口径和实验条件，不直接沿用其他来源的结论。

`valid_request_count` 是模型事件数；实际 `--limit`、时长和发送拥塞会改变实发
QPS，报告中的实发 QPS 才是结果口径。要使用参数化合成源，设置
`--traffic-source-spec /path/to/source.json`，文件遵守
`synthetic/realistic/1` 的 `kind/model/version/parameters` 格式；到达节奏仍由
`--send-mode`、`--send-mode-qps` 等客户端参数控制。合成源的时间戳只是序号，
因此默认使用 uniform 与 `--send-mode-qps 650`，显式 replay 会报错。脚本拒绝外部 `TRACE_FILE` 环境变量，
因此不会绕过两种源的校验。实验间须固定模型 SHA、参数、节奏及输出长度。

`--fetch-output-stream 1` 用于正式端到端 A/B；设为 `0` 时仅验证调度和引擎执行，客户端不会形成完整输出链路。

## 参数选择

- replay 保留 prefix DAG 模型的到达间隔与 burst；uniform 用于显式容量扫描。两者结果不能直接混作同一基线。
- `--n-prefill/--n-decode` 决定拓扑；改变拓扑就是改变实验条件。
- `--master-mode` 决定调度和投递形态。
- `--collection-profile aggregate|request|diagnostic` 决定证据量；默认 `aggregate`。
- `--config-override` 只覆盖 Master 配置字段；最终配置保存在运行目录。

完整含义见[参数参考](parameters.md)。

## 收结果

运行目录是 `--run-root/--run-id` 指定的路径，也可用 `--run-dir` 直接指定。默认单档结果就在运行目录：

```bash
find /path/to/new-output-root/stress-YYYYMMDD-HHMMSS -name aggregate.json -print
```

每个有效 case 至少应有 `aggregate.json`、`run_meta.json`、组件 JSON/日志和逐请求明细。先检查 `aggregate.json` 的 `summary.test_valid` 与 `validity_checks`，再解释性能。报告生成成功或进程退出码为零都不能替代有效性检查。

重新生成 HTML：

```bash
python3 tools/online_eval/scripts/commands/render_stress_report.py \
  --aggregate /path/to/aggregate.json \
  --out /path/to/report.html
```

## 运行对照

先为每次运行生成报告 bundle，再使用统一入口：

```bash
python3 tools/online_eval/scripts/commands/compare_runs.py \
  /path/to/A/reports/run/<report> /path/to/B/reports/run/<report> \
  --output /path/to/comparison
```

对比读取冻结的分析结果与曲线，展示控制变量差异、采集错误和缺口；这些标注不阻止报告生成。
对比入口不读取 `aggregate.json` 或重新选择统计窗。执行状态与各 run 独立判定分开，
退出码 0 只表示报告已生成。指标单位和解释边界见[结果与指标](results.md)。
