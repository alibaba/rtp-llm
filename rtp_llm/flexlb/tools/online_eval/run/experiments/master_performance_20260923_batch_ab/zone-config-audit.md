# Master frozen 门禁：Whale zone 配置核对与 A/B

检查对象：Whale 测试部署 `DeepSeekV4-Flash-Batch_l20c_test / na130_L20C_4TP`（部署 ID `6aa3b6979ffe080d001c3dbb`）、真实 Flash `0_32k_v8`（只读，部署 ID `6a9a8fc5c0237e4ddf24d973`），以及本地 frozen B 场景（提交 `280c22f9b8`）。Whale 使用生效部署的 zone 和 final Carbon target；测试 Master Pod 的环境变量另做读回。报告不记录 OSS 凭据。

## 配置对照

| 关键项 | Whale 测试部署运行态 | 本地 frozen B | 真实 Flash 0–32k |
|---|---|---|---|
| 模型、frontend prefix | `deepseek_v4`，frontend block 512，CP sharded 1、CP size 4；模型 revision 与生产相同，OSS 位置不同 | 同一 Flash 流量模型；trace 仅来自 16/20 个 frontend Pod | `deepseek_v4`，frontend block 512，CP sharded 1、CP size 4 |
| Master 版本和组批 | **legacy schema-v1 BATCH**；64 请求、350 ms 收集、350 ms early dispatch、每 P 2 个 inflight batch | **schema-v3 FIXED_WINDOW/BATCH**；64 请求、350 ms 收集、350 ms predicted budget、每 P 2 个 inflight batch | legacy schema-v1 BATCH；64/350/350、每 P 2 个 inflight batch |
| 路由与限制 | 亲和额外 TTFT 100000000 ms、最小 prefix 命中 5%；D 请求上限 128、KV 使用上限 98% | 亲和 20 ms/20%；D 请求上限 132、KV 使用上限 90% | 与 Whale 测试 Master 相同的 100000000/5 和 128/98 |
| 规模 | Mock bundle 逻辑 **48P/192D**，Master 1 Pod，frontend 20 Pod；独立 P/D zone 为 0 | **48P/192D**，单 Java client 直发 Master | zone 配置 P 64 行，D 10 行、4 partition、DP16，约 160 个逻辑 D；Master 2、frontend 50 |
| P/D block、pool | P 512、D 128；P pool 21553 blocks、D pool 221484 blocks；D max concurrency 128 | 同值 | P 物理 block 128/CP4，frontend 有效粒度 512；D block 128。真实设备和 Memory 容量不能直接由 mock block 数等同 |
| 性能与 Memory | `MOCK_PERFORMANCE_CONFIG_JSON` 与本地 preset JSON 逐字段相同；P scale 1.23、FIFO max requests 64、Memory 52295 blocks、D 2.6 tokens/step | 同一 preset | 真 GPU L20A；P Memory zone 为 100000 MB、GPU prefix tree 开启，D device cache 关闭。模拟公式需以监控验证 |
| 输出长度 | 运行态 EOS geometric mean 400；同日监控实际均值约 389 | 回放显式 geometric mean 294，绕过引擎 EOS | 同窗实际均值约 339 |
| 请求路径 | frontend `RTP_LLM_MOCK_NON_BATCH=1`，`FLEXLB_EXPECT_FETCH_RESPONSE=1`；Mock Master `FETCH_OUTPUT_STREAM=1` | Java client 直接向 Master 发请求，`FETCH_OUTPUT_STREAM=true`；无 Whale frontend | 真实 frontend / Master / GPU P-D 路径 |
| 制品与资源 | final Carbon Master 是 mock-bundle 镜像 tag 含 `30b0b8a38`；Master/Mock heap 各 32g，Master CPU 6400 millicores；模板 NV 镜像 tag `973c8fdcb` **不是**当前 Master final image | 源码 `280c22f9b8` 本地编译，Mock heap 32g；真实 Whale 镜像身份未与本地 JAR 证明一致 | 真实 P/D GPU 资源，Master CPU 6400 millicores |

**结论：** frozen B 与 Whale 测试部署的模拟引擎规模、block/pool、性能 JSON 和组批数值已对齐；Master schema/路由/Decode 限制、frontend 路径与输出长度仍不一致。Whale 测试部署也未按当前真实 Flash 0–32k 的 P/D 规模运行，因此它是有用的虚拟回归环境，但不能标为“生产配置完全等价”。

## 运行健康与监控

查询时 Whale 测试 Master 1/1 ready，frontend **0/20 ready**；Carbon frontend 进程显示 `HT_ALIVE / WT_READY`，服务状态为 `SVT_UNAVAILABLE`，部署 summary 为 `PUBLISHING`。Carbon 健康检查设为 `/frontend_health`，但该测试 frontend VIPServer 域的 HTTP checker 仍设 `/health`，25/25 IP invalid；在一个 frontend Pod 实测 `/health=503`、`/frontend_health=200`。这是独立于 TPS 的发布/服务发现问题，不能用 Master 指标正常掩盖。此处只读检查，未修改域名或生产部署。

同日 14:04–14:09（Asia/Shanghai），KMonitor 按精确 `hippo_app`、P/D `hippo_role`、`avg` + `1m-avg`（6 点）读取：测试 Mock P context 51.05k、with-cache 107.25k token/s/engine，context batch 14.56、forward 370 ms；真实 Flash P context 43.64k、with-cache 85.99k，batch 18.31、forward 458 ms。测试 Mock 输出均值 388.6，真实 339.1。两边负载、模型硬件和角色规模不同，这些数字用于定位配置差异，不作为同负载性能 A/B。

## 本地 A/B 定义

A：原组批 `32 / 10 / 550`；B：已提交的 `64 / 350 / 350`。两边来自同一提交 `280c22f9b8` 和同一次 Maven 构建，同一模型 trace SHA `29d0fbee…e9e911ca`、生成 JSONL SHA `12dd2a0a…4d9bce9`、Master JAR SHA `adc4b0d8…f74f21`、Mock JAR SHA `b4e27358…f17784`，同一输出模型、48P/192D、1536 QPS、300 秒预热、180 秒测量、全 Fetch 和绝对门禁。正式比对器给出 `controls=ALIGNED`，唯一声明变化为 `/actual_master_config/scheduler/decision`，A/B 用于观察，绝对门禁分别判定。

| 测量窗指标 | A | B | B 相对 A |
|---|---:|---:|---:|
| 发送 QPS | 1535.99 | 1536.01 | 同目标 1536 |
| 完成成功数 / 失败数 | 276479 / 0 | 276481 / 0 | 均 100% 成功 |
| Prefill context TPS / P | 53117 | 57108 | +7.51% |
| Prefill with-cache TPS / P | **94234** | **106354** | **+12.86%** |
| Decode generate TPS / D | 2354.55 | 2353.40 | −0.05% |
| TTFT p99 | 971.96 ms | 911.59 ms | −6.21% |
| TPOT p99 | 18.65 ms | 18.69 ms | +0.23% |
| E2E p99 | 13030.03 ms | 12775.71 ms | −1.95% |
| Inflight 首/尾 | 5300 → 5233 | 4936 → 4905 | 均未持续增长 |

A 的 12 项绝对检查中仅 `with-cache TPS/P ≥100000` 未通过，因此内层门禁 **FAIL**；B 的 12 项内层检查均 **PASS**。变化主要在 P 侧，与 D generate TPS 几乎不变一致。这是同 trace、同构建下的组批相关性，不代表 Whale 测试部署或真实生产配置已经完全等价。

**证据完整性限制：** 场景运行器额外检查整个生命周期的 Prometheus 抓取。A 的 Master collection 有 4 个空点（相对时刻 248、249、592、593 秒），B 有 1 个（207 秒）；它们不在上述 180 秒正式测量窗的 TPS/延迟取数段，且两边请求 journal 完整、端到端成功率 100%。但外层 runtime validity 均为 `INVALID`，A 外层结果 `FAIL`、B 外层结果 `ERROR`。因此本报告可用于定位组批差异，**不能宣称整轮场景或生产门禁全通过**；修复采集短缺口后需要复跑一次作为正式验收。

远端原始证据：host 111，租约 run `master-batch-ab-20260923` 的 `A/`、`B/` 和 `ab/`。本地 [A/B 合图](ab/report.html)、[A 单图](a/report.html)、[B 单图](b/report.html) 均为自包含 HTML；[归档说明](README.md) 记录文件哈希和证据限制。
