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

workload 实例可生成多份报告 HTML。运行结束后，遍历该实例 `result.json` 中的 `workload.reports`（视图文件名 → HTML 绝对路径）取齐全部报告；`workload.yaml` 是默认视图，还可能有 YAML 声明的专属视图。`run_cases.py` 完成实例时也会在 stdout 枚举这份清单。运行前查看场景 YAML 的 [`reports:` 列表](../../config/scenarios/README.md)，或使用 `--dry-run` 查看本次选中实例的视图名。

每个 workload 实例的主要产物包括：

- `result.json`：步骤、检查和原始状态。
- `workload-evidence.json`：阶段、进程代次、资源身份和请求证据。
- `telemetry/`：Prometheus 原始采样。
- `reports/run/<report-identity-slug>/analysis.json`：对应视图的可复算分析。
- `reports/run/<report-identity-slug>/report.html`：对应视图的阶段与曲线报告。每份视图各有一个目录，专属视图的报告 identity 带视图后缀；不能只按文件名 `report.html` 去重。

判定顺序是：证据完整性 → 合同检查 → 性能观察。`runtime_validity=INVALID` 时，即使某个业务检查显示 PASS，也不能作为成功结论。故障注入可能让压测聚合器的 `test_valid=false`，该字段需要结合场景预期解释，不能强改为 true。

## 受控策略对照

场景可在健康基线成立后干预缓存亲和等待预算。环境显式设置 `master_test_control: true` 才会为隔离 Master 生成独立令牌并启用测试控制接口；普通服务不注册该接口。干预使用带预期旧值的更新，随后独立读回，将三次响应保存在门禁证据中。

正反向对照共用输入、预热策略、拓扑、发流速率和观测窗口，只改变干预参数及预期行为。`expected_behavior: stable` 要求无持续下降；`collapse` 要求出现持续下降。报告同时展示预期行为与实际行为；采集不完整、基线不健康或干预未经验证始终判为 INVALID。

大流量源可通过 flow 的 `max_requests` 显式投影到前若干条请求；投影不改变原始顺序或前缀关系，来源 SHA、投影数量与展开结果 SHA 随产物保存。请求预算必须覆盖最长预热、拓扑收敛和完整观测。输入准备的启动预算与预热预算分别设置。

通用监控的短缺口预算与缓存门禁的证据预算分别声明。缓存缩容场景允许前后均有成功采集、间距不超过 10 秒的通用监控缺口，并保留报警；缓存基线和缩容观测仍独立执行 `gate.max_gap_s: 5`。长期断采、未被成功采集包围的缺口和不完整的缓存门禁证据不能据此变为有效。

## 对照场景报告

```bash
python3 tools/online_eval/scripts/commands/compare_runs.py \
  /path/to/A/reports/run/<instance> /path/to/B/reports/run/<instance> \
  --output /path/to/comparison
```

对比保留冻结曲线和每个 run 的有效性；缺采点不补零。配置差异逐项展示，不阻止报告生成。
默认保留归档的时间坐标；需要共同零点时，用 `--alignment-event` 显式指定归档事件。
不按阶段重新计算均值或生成变化排名。参数与产物见[命令入口](entrypoints.md)。

## 干预与恢复取证

故障、扩缩容和主备切换分别保留基线、过渡期、恢复期。基线样本应非空且成功；
过渡期逐笔记录成功、拒绝、传输错误和 deadline，所有已发请求必须有界结束；
确认成员和路由收敛后，另发独立的恢复请求，不能用故障前晚完成请求代替恢复流量。
观察窗口结束后可通过客户端 stop-sending 控制停止发新请求，再等待在途请求自然完成；必须核对停止命令身份、已提交数和终态记录数，不用杀进程代替排空。固定发流时长作为上限，不应迫使已完成恢复验证的场景继续发流。
采集器取消、丢记录、线程不退出或资源账目不收敛属于真实失败。

缓存命中取请求准入时的观测，晚完成请求仍归其发送窗口。容量、驱逐和副本数量保留
连续采样；终态恢复不能掩盖中途越界，采样峰值不能证明采样间隙不存在更高峰值。
校准 band 仅使用独立有效的基线样本，保存参数、窗口、样本路径和 SHA；
改变负载、容量、拓扑或运行形态后重新校准，不用干预结果调整健康门槛。

专用离线分析器从实例产物读取证据与 YAML 中的分析策略，命令参数以其 `--help` 为准。
各实例的阈值、拓扑和流程留在配置与程序中，不在文档重复维护。

## Master 性能绝对门禁

见 [性能门禁](performance-gate.md)。单 run 以 TPS、延迟和 100% 成功率判定；版本或 batch/non-batch A/B 仅辅助观察。
