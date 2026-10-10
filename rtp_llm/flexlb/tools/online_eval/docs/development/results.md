# 结果与指标

## 判定结果

先核对实际代码与制品、配置、拓扑、流量 SHA、Fetch 模式和窗口，再检查启动、采样、请求终态与清理。最后解释业务检查及性能数据。退出码为零、HTML 可打开或 Schedule 成功都不能单独证明运行有效。

| 状态 | 含义 |
|---|---|
| `PASS` | 所选合同成立；不证明未测试的合同或生产容量 |
| `FAIL` | 有效观察得到反例或超出预声明门槛 |
| `ERROR` / `TIMEOUT` | 运行、证据或预算不足以完成判断 |
| `BLOCKED` | 前置阶段失败，依赖步骤未执行 |
| `runtime_validity=INVALID` | 证据不足以发布成功结论，即使局部检查为 PASS |

性能门禁的 `INVALID` 映射为执行器 `ERROR`。专属门禁 PASS 不能代替整轮采集与清理通过。故障窗口的预期业务错误由阶段合同解释，采集缺失仍是证据错误。

## 收取产物

workload 默认交付一份主报告。专属视图包含门禁、有效性、运行信息与所选曲线；未声明视图时生成 `default.yaml`，没有归档序列时不生成指标面板，有序列时展示全部已归档指标。专属视图不会附带额外默认报告；需要全指标诊断时显式追加 `default.yaml`。运行前用 `--dry-run` 或 YAML 的 `reports` 查看视图，运行后以 `result.json → workload.report` 定位主报告，**遍历 `workload.reports`（视图文件名 → HTML 绝对路径）取齐所有报告**。stdout 也枚举该清单。功能实例读取执行结果报告。

| 产物 | 用途 |
|---|---|
| `aggregate.json`、实例 `result.json` | 汇总、阶段与检查状态 |
| `workload-evidence.json`、原始 journal | 请求、资源身份、进程代次与阶段证据 |
| `telemetry/` | 查询、原始采样和采集错误 |
| `metrics.json` | 冻结的指标定义、来源、标签和完整时间序列 |
| `reports/run/<report-identity-slug>/` | 每个视图独立的 HTML、分析、spec 和 manifest |

每个报告目录内都叫 `report.html`，专属 identity 带视图后缀；不能按文件名去重。省略全量 HTML 不减少原始指标和证据。已存在但校验失败的 bundle 必须报错，不能被精简报告掩盖。

## 指标与门禁口径

指标 ID 与来源、单位、标签、窗口一起解释。定义见[指标配置](../../config/monitoring/README.md)，请求完成含义见[生命周期](../architecture/request-lifecycle.md)。

- 发送 QPS、成功完成 QPS、名义速率是不同量，不能互换。
- counter 先处理差分与 reset，再除以实际间隔；gauge 按采样值解释。
- 引擎执行 TPS 保留 role、engine、priority 和 incarnation；同一引擎同一采样点归并 priority 后取引擎均值，不当作集群墙钟吞吐。
- 客户端完成 TPS 取固定墙钟窗内成功完成的实际 token 数；目标 `output_len` 不替代终态 `observed_output_tokens`。
- goodput 取窗口内到达、最终成功且满足 SLO 的请求数除以窗口时长。晚完成请求仍属于到达 cohort。
- TTFT/E2E 的 p99 来自成功 cohort；逐秒 p99 或其平均值不等于整窗 p99。TPOT 是每请求 `(E2E − TTFT)/(实际输出 token − 1)`，单 token 不适用。
- 请求闭合、发压偏差、pacing、采样间隔、引擎身份与制品来源先决定有效性。合法的零 TPS 参与阈值判定，缺指标不能补零。
- 错误率合同覆盖所声明的流量范围；性能合同要求完整 Fetch 与全部请求成功。只有 schedule ACK 或聚合吞吐不能证明端到端 PASS。

容量干预中，可用成员由新请求准入与 Master 拓扑共同确认；排空进程不等于可用容量。全局 offered load 只说明发流节奏，幸存者、被摘机与未知归属须分别记账。完成 counter 使用完成采样窗，客户端归因使用发送与终态时间，不能互换。峰值采样不能证明采样间隙不存在更高峰值。

## 对照与离线重判

`compare_runs.py` 校验并展示冻结 run bundle，不重新分析原始证据，不产生整体 verdict。对比退出码只说明报告是否生成成功。配置、负载、拓扑、窗口及模型差异逐项展示；来源缺失标 UNKNOWN，不据此认定一致。曲线单位或口径不一致时独立展示。

`analyze_performance.py` 从完整冻结请求证据显式复算，PASS/FAIL/INVALID 分别返回 0/1/2。两个入口都遵守[报告契约](../architecture/reporting.md)中的显式重判、输出目录保护与溯源规则。`reinterpret_cache.py` 可用 `--client-snapshot` 补充完整历史请求归属；历史能力限制必须保留，不能凭现有配置补造旧观察。

对比和离线命令见[命令入口](entrypoints.md)；报告写入及交互契约见[报告装配](../architecture/reporting.md)。
