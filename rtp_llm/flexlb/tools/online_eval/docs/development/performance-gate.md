# Master 性能绝对门禁

## 冻结 Whale Mock 后做回归

2026-09-22 暂停继续追齐真实集群 TPS，保留当前已知差异，不调整门槛来制造通过。
`config/flash_whale_reference_20260922.json` 固定观察来源、48P/192D、Device/Memory 容量、
P 公式及 1.23 scale、D step 模型、EOS400 和差异范围。
`config/scenarios/master_performance_frozen.yaml` 用此画像做单请求 / batch 配置 A/B；
两个 profile 的独立绝对结论不受比较结果影响。

该场景使用较早采集的 Flash 前缀轨迹，在播放时过滤 0–32k；不能说它来自用户最后一次复制源调整。
数据不提交仓库；运行前须提供该 YAML 指定 SHA256 的 `data/traffic_models/local_flash_20260922_v3.xz`。
输出上限为 8192，正常终止由固定 seed 的 EOS400 模型控制，不能再用旧350上限截短长度分布。
预热300秒、测量180秒；是否进入满缓存阶段须另外看驱逐证据，固定预热时长不保证满缓存。

首次合同预先声明：P context >=50000、with-cache >=100000、D generate >=2500 tok/s
（逐引擎算术均值），以及 YAML 中客户端吞吐、积压、TTFT p99<=2s、E2E p99<=30s、
请求平均 TPOT p99<=50ms 和全流程100%成功。这些是暂定的 **Mock 回归标准**，
不是生产容量/SLO认证；本次A/B结果不得回写这些阈值。

`criteria.engine_tps` 声明三项完整下界后，finish阶段直接归档 Prometheus 原始抓取时间点。
先按同一引擎、同一时间点求 priority 之和，再计算引擎均值；不相加成集群 TPS。
缺引擎、缺 priority 点、超过 max_gap_s 的断档、引擎 incarnation 改变都判 INVALID；零TPS保留并判 FAIL。
旧场景未声明此字段时保留客户端合同，不能称为引擎 TPS 门禁。

暂不生成 HTML 时，离线门禁及 A/B 命令均支持 `--json-only`，保存 evidence、analysis.json 和指标差值。
运行流程仍复用下文 runner，将 case-dir 改为 `master_performance_frozen.yaml`，
实例改为 `master_performance::flash_frozen_mock::<profile>`。

## 真实复制流量与线上 TPS 口径

`capture_frontend_prefix.py --format xz` 保存全长度前缀摘要，按
`request_enter_ts_epoch_ms` 严格筛选 `[start,end)`；缺少到达时间的记录计数后跳过，
不使用完成时间替代，避免混入切流前请求。原始 token 不落入采集文件。

拟合时选择 `fit_frontend_prefix.py --model-version 3`，保留精确输入长度。
回放 source 使用 `kind: trace / model: prefix_lineage / version: '3'`，
参数仍需固定 `path, sha256, count, output_tokens, priority`，可额外指定
`max_input_tokens: 32768`。该上界是包含边界的输入长度过滤，不截断请求，
不修改完整模型；被过滤父请求的前缀标签仍参与展开，后续请求共享关系不变。
v2 保持兼容，但其长度经过块对齐，不能用于精确 32k 边界过滤。

新报告以 `rtp_llm_context_tps`、`rtp_llm_context_tps_with_cache` 为主要
Prefill TPS 视角：先按引擎/DP 汇总 priority，再提供逐引擎曲线和引擎算术均值。
这两个 gauge 的分母是相应 batch 执行时间；不能与客户端完成 token/墙钟秒混用，
也不能将各引擎执行 TPS 相加称为集群吞吐。均值仅用于概览，线上比较须保留
引擎身份、采样间隔和缺失情况。客户端吞吐保留在独立视角中。

`flash_capture_diagnostic` 仅声明 P=512 / D=64 的块配置，沿用默认耗时模型；
它不是 Flash 性能校准。生产 TPS 阈值应在同口径校准后另行固定。
现有完成 TPS/SLO 判定仍是独立诊断合同，不能据此宣告线上容量达标。
长度过滤不能消除复制流量用户群体的差异；失败的测试前端完成时延和输出长度
不能用来拟合真实引擎。采集文件、缺失 Pod、时间窗口、SHA 和独立指定的输出行为
必须随实验保存。

每个 run 独立按固定标准判定：完成 input/output TPS 达标、延迟不超过上限、积压增长受限，且所有已发送请求成功率为 100%。A/B 是可选观察，不要求存在劣化版本，也不会修改单 run 结论。

当前唯一执行画像 `master_performance::flash_online_scale` 使用 48P/192D、1600 QPS、固定输入 3270 / 输出目标 350。已经撤销低规模用例，不能用低规模 PASS 推断线上表现。

它只对齐所选线上部署的 P/D 和负载量级；执行耗时、KV 容量、缓存复用、请求长度分布仍为 synthetic，单 Master 承接全量流量，也不同于线上双 Master 拓扑。生产容量和正式生产门禁阈值仍需校准。

## 运行

按 [编译与运行底座](build-and-runtime.md)构建 Master、mock 并设置 Prometheus。下面命令从 `rtp_llm/flexlb` 执行，各自使用新的输出目录。

```bash
python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir tools/online_eval/config/scenarios/master_performance.yaml \
  --suite workload --profile single-nonbatch --parallel 1 \
  --instances 'master_performance::flash_online_scale::single-nonbatch' \
  --out-dir /path/to/nonbatch

python3 tools/online_eval/scripts/commands/run_cases.py \
  --case-dir tools/online_eval/config/scenarios/master_performance.yaml \
  --suite workload --profile batch-window --parallel 1 \
  --instances 'master_performance::flash_online_scale::batch-window' \
  --out-dir /path/to/batch
```

运行前设置 `FLEXLB_FT_WORKER_PORT_CAPACITY=300`、`FLEXLB_FT_MOCK_HEAP=32g`，并显式配置自己的租约端口基址（见后文）。

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

## 线上量级探针与图表

`config/scenarios/master_performance.yaml` 是显式选择的 workload 规模探针，不加入默认 core 套件：
48P/192D、1600 QPS、固定输入 3270 / 输出目标 350，预热 30 秒、测量 60 秒。
这些拓扑和负载量级参考 2026-09-22 的线上只读观测；固定长度、全冷输入、100ms synthetic prefill 和默认 decode 模型尚未完成 V4 校准。
它验证 Master 在这一合成规模下的行为，不能代表生产容量或生产门禁 SLO。
其中 90% TPS/goodput 下界、TTFT 2s / E2E 15s / TPOT 50ms 和 100% 成功均是实验前声明的 synthetic 合同，不根据结果调低。

沿用上面的 runner，将 `--case-dir` 改为此配置，实例改为
`master_performance::flash_online_scale::<profile>`。
运行前设置 `FLEXLB_FT_WORKER_PORT_CAPACITY=300`、`FLEXLB_FT_MOCK_HEAP=32g`，
并将 `FLEXLB_FT_PARALLEL_MASTER_BASE` 和 `FLEXLB_FT_PARALLEL_MOCK_BASE` 显式绑定到自己的租约区间
（本例至少预留 320 个端口，Master 与 mock 区间不得重叠）。
每组使用独立输出目录，顺序运行，不与其他租约共用端口。

报告沿用公共 multi_curve 组件：A/B 合图、A 图、B 图，分别支持核心、TPS、延迟、流量、队列、规模、KV、模拟执行视角及指标搜索。
A 为第一个输入，B 为第二个输入；A 虚线，B 实线。同指标同色。单 run 各指标仍独立使用绝对标准。
请求曲线按 1 秒分桶：TPS 按成功完成时刻，延迟/成功率按到达 cohort（包含窗口后终态），不能把桶 p99 当作整个测量窗口 p99。
监控曲线从 evidence 同级 `telemetry/*/queries.json` 读取，保留 P、D 两种角色和缺采。
离线比较会复制这些查询归档到 A/B 子目录；报告不会从日志伪造缺失的监控曲线。
