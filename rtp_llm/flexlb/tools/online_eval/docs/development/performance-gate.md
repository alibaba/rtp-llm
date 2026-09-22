# Master 性能绝对门禁

每个 run 独立按固定标准判定：完成 input/output TPS 达标、延迟不超过上限、积压增长受限，且所有已发送请求成功率为 100%。A/B 是可选观察，不要求存在劣化版本，也不会修改单 run 结论。

当前 `master_performance::synthetic_steady` 是流程验收画像：2P/4D、P 固定 100ms、10 QPS、输入 1024、输出目标 8。它不是 V4 Flash/Pro 的生产 SLO。生产画像与 TPS/延迟边界需要校准后单独版本化。

这个低负载用例验证判定和证据链，不用于证明 batch 的容量优势。达到输入流量上限时，两种配置 TPS 相同是正常结果；后续容量验证需要固定更高负载或扫描负载。

## 运行

按 [编译与运行底座](build-and-runtime.md)构建 Master、mock 并设置 Prometheus。下面命令从 `rtp_llm/flexlb` 执行，各自使用新的输出目录。

```bash
python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir tools/online_eval/config/scenarios/master_performance.yaml \
  --suite workload --profile single-nonbatch --parallel 1 \
  --instances 'master_performance::synthetic_steady::single-nonbatch' \
  --out-dir /path/to/nonbatch

python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir tools/online_eval/config/scenarios/master_performance.yaml \
  --suite workload --profile batch-window --parallel 1 \
  --instances 'master_performance::synthetic_steady::batch-window' \
  --out-dir /path/to/batch
```

所有标准在 `config/scenarios/master_performance.yaml`，必填，不从一次测量反推。`max_error_rate` 必须为 0。`test.collection: request` 保证逐请求证据。场景作为 workload 可显式选取；尚未加入默认 core CI 清单。

场景显式开启所需 Master 指标白名单。公共监控使用 `auto_tpm.schedule.latency_ms` 的 timer count 统计调度响应 QPS；它不代表推理完成。推理完成率与 TPS 从完整 Fetch 的请求终态计算。

## 口径与有效性

- 完成 TPS：固定测量墙钟窗口内成功完成的请求，其输入 token、实际输出 token 分别除以窗口时长。
- goodput：在窗口内到达、最终成功且满足 TTFT/E2E/TPOT 标准的请求数除以时长。窗口末尾的慢请求仍属于 cohort，不因晚完成而被删除。
- 输出使用终止响应 AuxInfo 的累计 `observed_output_tokens`；请求目标 `output_len` 不作为实际输出替代。只有单输出响应受支持；缺少观测字段为 INVALID。
- TTFT/E2E 的 p99 来自成功 cohort；错误率独立覆盖整个 flow（含预热、排空），任一失败即 FAIL。全部失败也为 FAIL，不把空延迟分位数当采集缺失。
- TPOT = (E2E − TTFT)/(实际输出 token − 1)，是每请求平均生成间隔，不是 inter-token tail；单 token 请求不适用。
- inflight 由 issued 与终态时间重建，涵盖 Master 等待及引擎执行。增长为窗口末与初之差/时长，属于有限窗口合同，不证明无限期稳定。
- 缺少 JAR/配置/轨迹证据、请求未闭合、观测缺口、发压偏差或 pacing 超限均为 INVALID。它不能通过顶层 runner。

单 run 产物包括 `performance-gate-evidence.json`、`reports/run/master-performance/analysis.json` 与 HTML。离线复算：

```bash
PYTHONPATH=tools/online_eval/src:tools/online_eval python3 -m workload.performance_gate \
  /path/to/performance-gate-evidence.json --output /path/to/reanalysis
```

退出码：PASS=0、FAIL=1、INVALID=2。场景框架中 INVALID 映射为 ERROR；顶层 workload 的采集、清理检查也必须通过，不把专用 PASS 当整轮通过。

## 配置或版本 A/B

```bash
PYTHONPATH=tools/online_eval/src:tools/online_eval python3 -m workload.performance_compare \
  /path/to/A/performance-gate-evidence.json /path/to/B/performance-gate-evidence.json \
  --output /path/to/ab \
  --allow-master-change /actual_master_config/scheduler/decision \
  --allow-master-change /actual_master_config/dispatcher/type
```

允许差异必须是实际归档配置中的明确 JSON Pointer 路径，可以声明配置字段或子树；报告保留该路径两侧完整值。这里声明 decision 子树，是因为 FIXED_WINDOW 相比 SINGLE 还增加窗口参数。应先读归档配置，再声明，不按模式名猜路径。不允许豁免性能模型、流量或门禁阈值。

跨 profile 的 trace request ID 会带不同运行命名空间；另存 workload SHA，仅排除 `rid`，其余 token、顺序、长度、priority 全部参与校验，原始 SHA 仍保留。比较结果含各自 verdict 和控制变量状态。比较命令退出 0 只表示输入有效且差异已声明，**不是候选门禁通过**。

## 边界

当前实现使用开发机 Java flow。Whale 不复用开发机 runner；后续适配需提供同样的逐请求证据、实际制品和配置身份，才能调用同一离线分析。只有聚合 TPS 或 schedule ACK 时不能给出本门禁的 PASS。暂不自动发布、采集生产请求或修改生产配置。
