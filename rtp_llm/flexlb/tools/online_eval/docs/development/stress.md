# 压测

压测用于吞吐、延迟、容量曲线和同配置 A/B。它运行真实 Java Master、Java Mock Engine 与 Java load client，不使用 GPU。

先完成[编译与运行底座](build-and-runtime.md)，再从 `rtp_llm/flexlb` 执行。

## 标准运行

```bash
export PROMETHEUS_BIN=/path/to/prometheus
export RUN_ROOT=/path/to/new-output-root
export RUN_ID=stress-$(date +%Y%m%d-%H%M%S)
export N_PREFILL=12 N_DECODE=40
export FLEXLB_MASTER_MODE=wb
export SEND_MODE=replay LOOP=1 DURATION_S=120
export REPLAY_SPEED=82
export FLEXLB_WARMUP_SECONDS=10
export FETCH_OUTPUT_STREAM=1

bash tools/online_eval/stress/run_online_eval.sh
```

`REPLAY_SPEED=82` 只对应仓库当前 `trace_30min.jsonl` 的约 650 名义 QPS 标定。替换 trace 或目标 QPS 时按下式重算：

```text
speed = round(target_qps × (max(valid_ts)-min(valid_ts)) / valid_request_count)
```

`valid_request_count` 只统计 `output_len > 0` 的请求。报告中的实发 QPS 才是结果口径，不能用名义 QPS 替代。

`FETCH_OUTPUT_STREAM=1` 用于正式端到端 A/B；设为 `0` 时仅验证调度和引擎执行，客户端不会形成完整输出链路。

## 参数选择

- replay 保留 trace 的到达间隔与 burst；uniform 用于显式容量扫描。两者结果不能直接混作同一基线。
- `N_PREFILL/N_DECODE` 决定拓扑；改变拓扑就是改变实验条件。
- `FLEXLB_MASTER_MODE` 决定调度和投递形态。
- `COLLECTION_PROFILE=aggregate|request|diagnostic` 决定证据量；默认 `aggregate`。
- `FLEXLB_CONFIG_OVERRIDE` 只覆盖 Master 配置字段；最终配置保存在运行目录。

完整含义见[参数参考](../reference/parameters.md)。

## 收结果

运行根目录是 `$RUN_ROOT/$RUN_ID`。单档运行的实际 case 目录位于其下一层。用文件名定位，不要假定目录名：

```bash
find "$RUN_ROOT/$RUN_ID" -name aggregate.json -print
```

每个有效 case 至少应有 `aggregate.json`、`run_meta.json`、组件 JSON/日志和逐请求明细。先检查 `aggregate.json` 的 `summary.test_valid` 与 `validity_checks`，再解释性能。报告生成成功或进程退出码为零都不能替代有效性检查。

重新生成 HTML：

```bash
python3 tools/online_eval/stress/reporting/report.py \
  --aggregate /path/to/aggregate.json \
  --out /path/to/report.html
```

## A/B

基线与候选必须使用同一 trace、拓扑、Master 配置、时长、Fetch 模式和稳态窗口，且两边都有效：

```bash
python3 tools/online_eval/stress/reporting/compare_ab.py \
  --run-a /path/to/A/aggregate.json \
  --run-b /path/to/B/aggregate.json \
  --out /path/to/comparison.json --html
```

以 `compare_ab.py --help` 为当前参数契约。比较结果用于指出变化；如果输入 provenance 不一致，停止比较并重新跑。指标单位、聚合和误判边界见[结果与指标](../reference/results.md)。
