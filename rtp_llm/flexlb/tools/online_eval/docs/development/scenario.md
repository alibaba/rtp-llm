# 场景测试

场景测试对应 runner 的 `workload` suite，用于持续负载下的故障、扩缩容、流量迁移、缓存压力和恢复。它与功能测试共用执行底座，但额外保存阶段时间线和连续监控。

先完成[编译与运行底座](build-and-runtime.md)，并设置 `PROMETHEUS_BIN`。

## 选择与预览

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios \
  --suite workload --list-json > /tmp/flexlb-workloads.json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --suite workload \
  --instances 'balance_distribution::sustained_mix::batch-window' \
  --parallel 1 --dry-run
```

场景测试默认单 lane。延迟、恢复时间和阶段曲线会受到并行邻居影响；需要可比较结果时保持 `--parallel 1`。

## 执行

```bash
export PROMETHEUS_BIN=/path/to/prometheus
OUT=/path/to/new-output/scenario

python3 tools/online_eval/scripts/commands/run_cases.py \
  --suite workload \
  --profile batch-window \
  --parallel 1 \
  --out-dir "$OUT" \
  --json "$OUT/aggregate.json" \
  --archive "$OUT.exp.zip"
```

场景 YAML 定义 P/D 规模、请求节奏、干预时点、观察窗口和阈值；Python case 定义步骤、分支和断言。不要从命令行按比例扩大拓扑后继续沿用原 band，除非已经重新校准。

## 收结果

每个 workload 实例应包含：

- `result.json`：步骤、检查和原始状态。
- `workload-evidence.json`：阶段、进程代次、资源身份和请求证据。
- `telemetry/`：Prometheus 原始采样。
- `reports/run/<instance>/analysis.json`：可复算分析。
- `reports/run/<instance>/report.html`：阶段与曲线报告。

判定顺序是：证据完整性 → 合同检查 → 性能观察。`runtime_validity=INVALID` 时，即使某个业务检查显示 PASS，也不能作为成功结论。故障注入可能让压测聚合器的 `test_valid=false`，该字段需要结合场景预期解释，不能强改为 true。

## 比较两次场景

```bash
PYTHONPATH=tools/online_eval/src:tools/online_eval python3 -m workload.compare \
  --baseline /path/to/A/analysis.json \
  --candidate /path/to/B/analysis.json \
  --out /path/to/comparison
```

只能比较相同实例和声明配置。阶段按同名步骤对齐；缺采样显示为 `MISSING_DATA`，不能当作零。变化排序是调查入口，不自动等于产品回归。

## Cache 缩容的单 run 与 A/B

`config/scenarios/cache_scale_in.yaml` 是真实前端前缀谱系流量的单 run 缩容门禁。运行大型 125P/536D 拓扑前须设置 `FLEXLB_FT_WORKER_PORT_CAPACITY=700`。分别使用普通启动路径运行旧、新 Master；可用 `FLEXLB_FT_MASTER_JAR` 指定 JAR，`FLEXLB_FT_MASTER_CONFIG_FILE` 指定实际配置。`FLEXLB_FT_MASTER_SOURCE_COMMIT` 只声明源码来源；每轮证据独立记录实际 JAR 哈希与生效配置，不要求事前 manifest。

两轮完成后，再读取同一场景 YAML 中的 `analysis` 策略生成 A/B 报告：

```bash
PYTHONPATH=tools/online_eval/src:tools/online_eval python3 -m workload.cache_gate_ab \
  OLD_RUN_DIR NEW_RUN_DIR \
  --config tools/online_eval/config/scenarios/cache_scale_in.yaml \
  --output AB_DIR
```

报告核对流量、拓扑、容量、性能和 Master 配置，缺字段会显示 UNKNOWN；曲线按缩容事件对齐。默认强判定观察 old FAIL / new PASS；`--mode weak` 只核对控制变量，`--mode none` 只出报告。A/B 不修改单 run 的 PASS / FAIL / INVALID 结论。

## Master 性能绝对门禁

见 [性能门禁](performance-gate.md)。单 run 以 TPS、延迟和 100% 成功率判定；版本或 batch/non-batch A/B 仅辅助观察。
