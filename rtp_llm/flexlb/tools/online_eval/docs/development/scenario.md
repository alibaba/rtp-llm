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
  --instances '<exact-instance-id>' \
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

## 干预与恢复取证

故障、扩缩容和主备切换分别保留基线、过渡期、恢复期。基线样本应非空且成功；
过渡期逐笔记录成功、拒绝、传输错误和 deadline，所有已发请求必须有界结束；
确认成员和路由收敛后，另发独立的恢复请求，不能用故障前晚完成请求代替恢复流量。
采集器取消、丢记录、线程不退出或资源账目不收敛属于真实失败。

缓存命中取请求准入时的观测，晚完成请求仍归其发送窗口。容量、驱逐和副本数量保留
连续采样；终态恢复不能掩盖中途越界，采样峰值不能证明采样间隙不存在更高峰值。
校准 band 仅使用独立有效的基线样本，保存参数、窗口、样本路径和 SHA；
改变负载、容量、拓扑或运行形态后重新校准，不用干预结果调整健康门槛。

专用离线分析器从实例产物读取证据与 YAML 中的分析策略，命令参数以其 `--help` 为准。
各实例的阈值、拓扑和流程留在配置与程序中，不在文档重复维护。

## Master 性能绝对门禁

见 [性能门禁](performance-gate.md)。单 run 以 TPS、延迟和 100% 成功率判定；版本或 batch/non-batch A/B 仅辅助观察。
