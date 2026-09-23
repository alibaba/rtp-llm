# Master 性能门禁

每个 run 按预先声明的吞吐、延迟、积压和成功率标准独立判定。A/B 用于比较已声明的
配置或版本差异，不修改单 run 结论，也不根据测量结果回写门槛。
Mock 回归标准不等于生产容量或 SLO 认证；配置一致、流量可比和证据完整是解释结果的前提。

## 配置与执行

场景 YAML 声明拓扑、容量、性能模型、流量 SHA、输出模型、预热/测量窗口及 `criteria`。
规模和阈值从所选配置读取，不能直接沿用另一场景的参数。`max_error_rate` 必须为 0；
`test.collection: request` 保证逐请求证据。显式实验通过 `--case-dir` 选择，
加入实验目录不代表加入默认 suite 或 CI。

先按 [编译与运行底座](build-and-runtime.md)构建制品并配置 Prometheus。从 `rtp_llm/flexlb` 执行：

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source /path/to/scenario.yaml --suite workload --list-json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir /path/to/scenario.yaml --suite workload \
  --instances '<exact-instance-id>' --parallel 1 --dry-run

python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir /path/to/scenario.yaml --suite workload \
  --instances '<exact-instance-id>' --parallel 1 --out-dir /path/to/new-run
```

按拓扑配置 `FLEXLB_FT_WORKER_PORT_CAPACITY`、`FLEXLB_FT_MOCK_HEAP`，将
`FLEXLB_FT_PARALLEL_MASTER_BASE` 和 `FLEXLB_FT_PARALLEL_MOCK_BASE` 绑定到独占租约区间。
Master 与 Mock 端口不重叠，每轮使用新的输出目录。缓存预热完成须由占用、复用和驱逐
证据确认，固定预热时长不能保证满缓存。

## 流量与 TPS 口径

采集按 `request_enter_ts_epoch_ms` 筛选左闭右开到达窗 `[start, end)`，缺到达时间戳
显式计数，不以完成时间替代。capture→fit 契约、版本及归档规则见
[数据规则](../../data/README.md)和 [capture_contract.py](../../src/traffic/capture_contract.py)。

v3 保存精确总长；`max_input_tokens` 是包含上界的播放过滤，不截断请求或修改快照。
展开时保留被过滤父请求的前缀身份。v2 长度有块对齐损耗，不适合精确边界过滤。
相同长度范围不保证用户群体、前缀分布或流量来源可比。

回放显式 `ol` 会绕过引擎 EOS 采样。独立输出模型必须记录分布、seed 和 cap，
不能把错误截断的输出拟合为正常分布。事件序号驱动的采样在过滤和截取后仍保持身份；
具体控制见 [播放调节与复现](playback-controls.md)。

`rtp_llm_context_tps`、`rtp_llm_context_tps_with_cache` 和 `rtp_llm_generate_tps`
是引擎执行 TPS。先按同一引擎、同一采样点归并 priority，再计算引擎算术均值；
保留引擎身份、采样间隔和覆盖情况，不相加称作集群墙钟吞吐。
客户端完成 token / 墙钟秒与引擎执行 TPS 分开报告。

`criteria.engine_tps` 声明完整下界时，finish 阶段保存三项原始 range-vector 抓取。
缺引擎、缺 priority 点、超过 `max_gap_s` 的断档或 incarnation 改变判 INVALID；
零 TPS 保留并参与 FAIL 判定。未声明该字段的场景只有客户端合同，不能宣称引擎 TPS 达标。
Master `auto_tpm.schedule.latency_ms` 的 timer count 只表示调度响应 QPS。

终态证据在分析前原子落盘，保留所有请求的判定字段、原始 journal 路径与 SHA256。
完整实验生成单 run HTML；A/B 保留合图、A 图、B 图和各自报告。FAIL/INVALID 同样保留诊断证据，
`--json-only` 仅用于显式离线诊断。

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

允许差异必须是实际归档配置中的明确 JSON Pointer 路径，可以声明配置字段或子树；报告保留该路径两侧完整值。decision 子树包含调度窗口参数。应先读归档配置，再声明，不按模式名猜路径。不允许豁免性能模型、流量或门禁阈值。

跨 profile 的 trace request ID 会带不同运行命名空间；另存 workload SHA，仅排除 `rid`，其余 token、顺序、长度、priority 全部参与校验，原始 SHA 仍保留。比较结果含各自 verdict 和控制变量状态。比较命令退出 0 只表示输入有效且差异已声明，**不是候选门禁通过**。

## 边界

当前实现使用开发机 Java flow。Whale 不复用开发机 runner；其他运行环境需提供同样的逐请求证据、实际制品和配置身份，才能调用同一离线分析。只有聚合 TPS 或 schedule ACK 时不能给出本门禁的 PASS。暂不自动发布、采集生产请求或修改生产配置。

## 报告曲线

报告沿用公共 multi_curve 组件：A/B 合图、A 图、B 图，分别支持核心、TPS、延迟、流量、队列、规模、KV、模拟执行视角及指标搜索。
A 为第一个输入，B 为第二个输入；A 虚线，B 实线。同指标同色。单 run 各指标仍独立使用绝对标准。
请求曲线按 1 秒分桶：TPS 按成功完成时刻，延迟/成功率按到达 cohort（包含窗口后终态），不能把桶 p99 当作整个测量窗口 p99。
监控曲线从 evidence 同级 `telemetry/*/queries.json` 读取，保留 P、D 两种角色和缺采。
离线比较会复制这些查询归档到 A/B 子目录；报告不会从日志伪造缺失的监控曲线。
