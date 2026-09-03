# flexlb_ft — FlexLB mock engine case test 套件（场景测试）

FlexLB mock engine **case test**（场景测试）框架：每个 case 由 `EnvManager` 启动一个小型
mock 引擎集群 + FlexLB master，对一个具体场景钉住**一条调度器行为契约**。入口 runner 为
上层目录的 `flexlb_functional_tests.py`，场景定义在本目录 `cases/` 的九个分类模块中，
框架设施为 `harness.py` / `context.py` / `engine_ops.py` / `grade.py`。

## 术语（2026-09 task #85 统一）

| 术语 | 含义 |
| --- | --- |
| **case test（场景测试）** | 本套件：`flexlb_functional_tests.py` + `flexlb_ft/cases/` 九分类。逐场景验证调度器行为契约 |
| **stress test（压测）** | online_eval 负载管线（`run_online_eval.sh` + eval/分析脚本）：QPS / ramp / duration 负载形态与时序分析，独立血统，勿混称 |

旧 "e2e test" / "chaos test" 措辞已废弃：**故障注入是 case 内的机制**（`engine_fault` /
`status` / `direct` 等分类的构造手段），不是套件名。

## 快速上手

前置：先构建两个 jar（缺失时 harness 会直接报错指路）——
`flexlb-mock-engine` 的 all-in-one jar 与 `flexlb-api` jar（maven，见 `harness.MOCK_JAR` /
`harness.API_JAR`）。

```bash
cd rtp_llm/flexlb/tools/online_eval

python3 flexlb_functional_tests.py --list                        # 列出用例
python3 flexlb_functional_tests.py --category all --profile batch-window   # 全量（默认）
python3 flexlb_functional_tests.py --category kv --json results.json       # 单分类 + JSON 结果
python3 flexlb_functional_tests.py --filter cancel_t1 --profile single-nonbatch   # 子串过滤
python3 flexlb_functional_tests.py --category cancel --grade strict --keep # 严档断言，跑完保留环境
```

CLI 一览：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--category` | `all` / `cancel` / `status` / `kv` / `balance` / `elastic` / `engine-fault` / `master` / `admission` / `direct` | 九大场景分类（默认 `all`） |
| `--profile` | `batch-window` / `single-nonbatch` / `single-batch` / `window-nonbatch` | 调度 profile（默认 `batch-window`） |
| `--grade` | `strict` / `normal` / `loose` | 断言档位（默认 `normal`） |
| `--filter` | 子串 | 按用例名子串过滤 |
| `--list` | — | 列出当前过滤条件下的用例并退出 |
| `--json` | 路径 | 把逐 case 结果写成 JSON |
| `--keep` | — | 跑完保留环境不 teardown |

### 四 profile 矩阵（schema-v2 FLEXLB_CONFIG 四轴）

profile 是 scheduler.ordering.decision.dispatcher 四个行为轴的命名组合（见
`harness.PROFILES` / `PROFILE_SPECS`）。Phase-1 档位集**scheduler 恒为 QUEUE、ordering 恒为
FIFO**，变量只有 decision × dispatcher 两轴（PRIORITY ordering / DIRECT / 抢占变体留给后续
阶段）：

| profile | decision | dispatcher | `--list` 计数 |
| --- | --- | --- | --- |
| `batch-window` | fixed_window | batch | 80 |
| `single-nonbatch` | single | non_batch | 36 |
| `single-batch` | single | batch | 39 |
| `window-nonbatch` | fixed_window | non_batch | 36 |

全集 81 例；各 profile 计数不同是因为 case 可声明 `profiles`（显式限定）或 `requires`
（语义能力需求，如 `enqueue_batch` 只在 BATCH 投递 profile 下满足，`generate_stream` 只在
NON_BATCH 下满足——见 `harness.PROFILE_CAPS`）。

### 分级断言（GRADE_BANDS）与运行档位

balance / kv 一带的结果属性用例（P 系列 / M 系列）不是布尔断言：属性值对着中央 band 表
（`grade.GRADE_BANDS`，strict/normal/loose 三档）评估**实际达成档**（achieved），case 取
各属性最差档。`--grade` 定运行档：**超过运行档界值即 FAIL**，achieved 仍如实记录。
硬不变量（P2 无饥饿、P6 完备性）无 band——任何违反直接不可用。运行级 verdict 四级：
**优异**（全 strict）/ **良好**（全 ≥normal）/ **边缘**（有 loose、无 fail）/ **不可用**
（超 loose 或破不变量）。band 校准记录与 false-fail 推导见 `grade.py` docstring。

## 框架要点速览

细节以代码 docstring 为准，这里只列索引：

- **EnvManager + EnvSpec**（`harness.py`）：环境声明式描述（P/D 数量、perf、master
  profile / 覆盖 env、discovery 模式、cache blocks、稳定窗等）。`ensure(spec)` 按
  **fingerprint** 复用：指纹相同直接复用现有环境，不同才 teardown 重建；构建中途失败
  也会停掉已启动的部分 JVM 防泄漏。
- **master 三段式就绪**（`harness.EnvManager.start_master`）：① HTTP 端口 up 且确认是
  自己的 JVM（防外来 master 占端口误判）；② `POST /rtp_llm/master/info` 的 `ready=true`
  （引擎同步完成）；③ 稳定窗——`alive == discovered == spec 拓扑` 持续
  `master_stable_window_s`（默认 3s）才返回，跳过冷启动首连风暴（期间健康引擎可能被
  3-strike 误标死；`master_coldstart_burst` 特意关掉此窗）。
- **95s TTL 排空约定**（`harness.TTL_DRAIN_TIMEOUT_S = 95.0`）：泄漏的 inflight 槽位最坏
  要 `staleInflightTimeoutMs(30s) + ExpirationTimer 60s 周期 sweep + 5s 余量 ≈ 90s` 才被
  扫走；排空断言一律用 95s 窗。等待更长是"等 settle"而非放松断言——目标仍是全零，真
  泄漏照样超时挂掉（更短的窗曾让残渣流进共享环境的下一个 case，造成 16 个假 FAIL）。
- **正确契约断言原则**：断言写**正确预期行为**而非当前行为——跑挂即 finding。预期挂的
  case 在 docstring 的 Prediction 里声明（如 `status_zombie_fake_running` 是声明式
  P2 probe；`kv_storm_hot_churn` 是 FINDING case）。
- **RID family**（`context.RID_BASES` / `rid_base`）：请求 id 按 category × profile 分配
  百万段基址，再叠加 `case_seq × 1M` 与 pid 偏移——复用 master 的 dedup 表跨 profile、
  跨重跑、跨并行进程永不冲突。

## 九分类逐类说明

当前全集 **81 例**（以 `--list` 实测为准）。每类一节：主题一段话、全量用例清单表、
分类色注（profile 限制与 EnvSpec 特点）。表内"期望"是一句话契约摘要，**详细断言以各
case 函数 docstring 为准**。

### cancel（13 例）— 请求取消契约

客户端 Cancel 必须终止流、释放引擎侧请求状态、（master 投递模式下）排空 master inflight
账目——幂等、与兄弟请求隔离、覆盖生命周期每个阶段、乃至覆盖 master 从未见过的请求。
`deliveryClaimKind` 边界是唯一不可回退点：NONE = master 还持有，BATCH_ENQUEUE /
ROUTE_DECISION = 已投递引擎；每个新 case 都在探测边界某侧的取消语义。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `cancel_t1` | 请求流式输出中客户端发显式 Cancel RPC（仍在运行） | 流终止；引擎侧取消 BATCH 契约保证可见（NON_BATCH 观测）；master inflight 排空；后续请求正常 |
| `cancel_t2` | Cancel 一次后立刻再 Cancel 一次 | 第二次 Cancel 幂等不报错；流终止；恢复正常 |
| `cancel_t3` | 3 并发请求，B 长 decode（output_len=500）中被取消，A/C 为短请求 | B 流终止且未完成；A/C 照常完成；引擎记录取消；恢复正常 |
| `cancel_t4` | 请求（output_len=1）完成后才 Cancel | 取消幂等成功、无二次结算；恢复正常 |
| `cancel_t5` | 对从未存在的 rid 发 Cancel | 调用不抛异常（幂等处理） |
| `cancel_t6` | A 在 prefill 相位、B 在 decode 相位分别取消 | 两相位取消都终止流；恢复正常 |
| `cancel_anomaly_path` | 流式输出中 Cancel——失败请求客户端侧的取消路径（E1 移入） | 流终止、取消延迟记录；（BATCH）inflight 清零；恢复正常 |
| `cancel_deadline_exempt_inflight` | queueTimeout 到期落在 claim 之后（M2 豁免，enqueue_delay 拖住 ACK） | 请求**不被**取消：完整输出正常完成；无引擎侧取消记录；账目走普通完成路径 |
| `cancel_schedule_drop_delivered` | 批已 claim 后客户端取消 Schedule RPC 本身（BATCH only） | master 仍向原 prefill 发**真引擎 Cancel**；引擎记录取消；账目经 CANCELLED reconcile 结算 |
| `cancel_engine_notfound_settle` | 请求跑完后 Cancel 迟到地分别打到 master 与原 prefill 引擎 | master Cancel 幂等成功；引擎答 NOT_FOUND；终态保持完成不被复写；账目不重开 |
| `cancel_preemption_victim` | 专用 1P+1D PRIORITY 环境：P30 长 decode 受害者 RUNNING 后 P70 到达（M3） | 受害者流以 8429（PRIORITY_PREEMPTED）非完成终态终止；引擎记录取消；弱 Cancel 真实发出；P70 正常完成；无泄漏 |
| `cancel_stream_break_prefill_autonomous` | 长请求首输出后客户端自行断开 FetchResponse 流、不发 Cancel（C1，BATCH only） | 引擎感知断流自主清理（记录取消、inflight 归零）；master 经 CANCELLED reconcile 结算（预期 FINDING，依赖 mock 断流感知能力落地） |
| `cancel_stream_break_decode_autonomous` | GenerateStreamCall 直投递 decode 中客户端断流（C2，NON_BATCH only） | decode 感知断流**提前**上报终态而非等 stale TTL；引擎与 master 无残留（预期 FINDING） |

色注：无显式 profile 限制（两个断流用例按 `requires` 分别限 BATCH / NON_BATCH 投递）。
默认共享 `smoke_spec`（2P+4D）；`cancel_preemption_victim` 专用 1P+1D PRIORITY 环境
（master_fixed_window.json 生产抢占块 + decode maxEngineRequests=1）。

### status（19 例）— engine→master 状态上报契约

WorkerStatus 事实（ACTIVE / TERMINAL）既 settle 请求槽又刷新 stale-inflight 活动时钟；
本分类往这条通道注入故障（mock `/inject` 的 `status_*` / `enqueue_ack_*` 家族），钉住
**正确的 master 契约**。分级：P0 = 发布阻塞契约，P1 = 健壮性，P2 = 声明式 finding probe。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `status_inflight_ttl_cleanup` | 双 prefill 慢 10s + 6 个 fire-and-forget 请求 + `/stop_engine` 卡死 inflight（S1 移植） | 卡住的 scheduler inflight 在 TTL+余量内清零；幸存 prefill 继续服务 |
| `status_ack_partial_fail` | 4 请求批的 ack 中 k=1 个成员带失败 | k 个成员收带注入码的 TERMINAL、其余成员**仍成功**（部分失败不毒化整批）；账目排空；新批恢复 ≥95% |
| `status_ack_multi_error` | 两个批分别注入不同错误码 8431 / 8510 | 错误码**按请求逐个透传**（A 批全 8431、B 批全 8510）；master 不崩；失败批不复活 |
| `status_ack_empty_no_crash` | prefill 丢弃整个 enqueue ack（空 ack = 投递不确定） | master 装不确定 fence：残渣**有界不增长**且最终可清空（排空断言为声明式契约候选 finding）；master 保持 200 |
| `status_prefill_suppress_all` | 所有 prefill 压制 running+finished（状态通道全静默） | master 保持 200；请求以合法终态结束（ok 或超时类）；TTL 兜底清账；清除注入后恢复 |
| `status_prefill_suppress_finished` | 所有 prefill 只压制 finished（持续显示 RUNNING，TTL 被活动时钟解除武装） | 请求在 queueTimeout(10s)+余量内以合法终态退出；清除注入后整个账目排空 |
| `status_status_no_respond` | prefill 完全停止应答 WorkerStatus 轮询 | alive 数在 3-strike 窗口内下降（代际退役）；inflight TTL 内排零；清除后拓扑完全恢复 |
| `status_unknown_rid_finished` | 引擎对 master 未见过的 rid 上报终态（一次性） | master 忽略未知 rid 终态：inflight 指纹前后**逐位一致**；master 保持 200 |
| `status_version_regress` | prefill 持续以回退版本应答状态轮询 | 无效版本累计进 3-strike：alive 下降（代际退役）；inflight 排零；master 200 |
| `status_decode_suppress_finished` | 所有 decode 引擎压制 finished（prefill 正常） | 请求在 deadline/TTL 内合法终态；prefill 批正常排空；清除注入后 decode inflight 在放宽 TTL 窗内排零 |
| `status_decode_before_prefill` | prefill 压制整批全部事实、decode 正常上报 | P/D 分离下 decode 侧 finished 完成结算：4 请求全部成功终态；prefill 批 TTL 内排零；master 200 |
| `status_unknown_rid_running` | 引擎对未知 rid 上报一次性 RUNNING 幽灵事实 | 注入停止后 scheduler inflight 在 TTL+余量内归零（禁止永久幽灵常驻） |
| `status_unknown_batchid` | 真实 rid + 伪造 batchId 的终态与真实流量并发 | 真实请求结算不受影响：目标与对照请求都成功完成（只按 rid 结算的实现会挂——即 finding） |
| `status_duplicate_finished` | 引擎重放 finished 上报 | 重放幂等：无双重结算、无账目复活；inflight 指纹在重放窗内逐位不变 |
| `status_cursor_regress` | 3 请求完成后完成游标回退 3（旧终态重投递） | 旧终态重投递幂等：无重复终态、无复活；指纹逐位不变；新请求仍成功 |
| `status_finished_then_running` | 请求完成后先重放 finished、再持续上报 RUNNING | 终态不可复活 / 不可回滚：账目跨两段注入保持干净；master 200 |
| `status_zombie_completed_running` | decode 引擎持续把已完成任务重报 RUNNING（tombstone 路径） | master 200；decode 侧无新确认条目泄漏（inflight 保持排空）；全账目干净 |
| `status_zombie_fake_running` | 引擎持续 ≥2×TTL 为 N 个幽灵 rid 上报 RUNNING（**声明式 P2 probe**） | 注入停止后 inflight 必须可清零——活动时钟刷新使 TTL 永不触发的常驻 inflight 即 finding（预期当前实现挂） |
| `status_fetch_error` | 批模式 FetchResponse 流发一条未完成输出后失败 | 客户端观察到错误；引擎侧 inflight 立即排空；master 账目由 30s stale TTL 清理；清除后新请求成功 |

色注：全部声明 `batch-window`——共享 `_status_spec`（2P+2D）用 FLEXLB_CONFIG 钉死旧
故障轴（PRIORITY + FIXED_WINDOW + BATCH）+ `staleInflightTimeoutMs=30s` +
`queueTimeoutMs=10s`（zombie keep-alive 场景的兜底底线）。换 `--profile` 重跑执行的
是同一份配置（label honesty + 回归效率）；NON_BATCH 变体是后续阶段素材。

### kv（15 例）— 前缀缓存生命周期契约

master 视角的 KV 前缀缓存生命周期：per-engine 账目隔离、驱逐事件同步、共享（全局索引）
块、同步收敛、引擎下线清理、热前缀 churn 风暴、亲和 vs 容量冲突、decode 侧 KV 容量
停车。分组：per-engine 3 例 / global 5 例 / storm+capacity 3 例 / 亲和路由 3 例（P9/M2/M3
分级属性）/ LRU 容量 1 例。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `kv_pe_admit_isolation` | ledger 分离播种：family-0 钉 A、family-5 钉 B，B 慢 5s，其余 family 只可能在 A admit | 只有 A 的 `cache_key_set` 增长（B 保持只有自家 seed）；后续同前缀续接粘 A（P9）；全局广播会把命中摊平 |
| `kv_pe_evict_zero_match` | family-0 在 X 上 prime 后 `/cache_evict` 全量逐出并等同步收敛 | 逐出事件同步过 master 索引：后续同前缀批不再粘 X（P1 零命中摊开）；陈旧索引会把 max-share 钉死 1.0 |
| `kv_pe_prefix_continuity` | family-0 共享于 e1/e2；e1 抠掉中段块留缺口、e2 留前 8 块连续 | 连续前缀匹配按 **token** 计：请求只匹配 e2 的 8 块（hitTokens 8192），批集中落 e2（M3）；按块数或后缀口径都会路由错 |
| `kv_g_shared_block_both_match` | 共享块双持有（全局索引映射到持有者**集合**） | maxHit==minHit → NO_CACHE_LEAD：同前缀请求在持有者间摊开（P1/P2），持有者并集份额 100% |
| `kv_g_partial_release_redirect` | 共享块被 e1 单侧 `/cache_evict` 释放 | 收敛后 e2 成唯一持有者：后续同前缀请求全部重定向 e2（P9） |
| `kv_g_full_release_no_ghost` | 共享块被两侧全部逐出并静默收敛 | 索引无残影：同前缀请求零命中摊开（P1）；幽灵条目会把 family 卡死在单引擎 |
| `kv_g_sync_convergence` | 交错 admit/evict 事件流后静默 ≥3.5s | 收敛后路由与引擎快照一致：无持有者的 family 摊开、唯一持有者的 family 粘住（P9）；乱序 / 丢更新会落错桶 |
| `kv_g_engine_down_cleanup` | 3P 环境 family 双持有于 h1/h2，`remove_engine` 永久下线 h1 | 只清 h1 的持有条目：family 仍归 h2（P9 全落 h2）；过度清理会把 family 打散到全部幸存引擎 |
| `kv_storm_hot_churn` | 4 个热 family 轮转 vs 每引擎 24 块小 LRU（**FINDING case**） | 复制因子有界、holder 翻转有界、M3 命中浓度 ≥loose；无复制抑制时自逐出导致命中率塌方——塌方即 finding |
| `kv_capacity_conflict_overflow` | family-0 满命中钉在持有者 e1，但 e1 ledger 热约 2s（e2 已恢复） | 亲和**让位**容量：OVER_CAP 溢出——波次泼到不匹配引擎（P5）、短请求受保护（P7）、无队列超时停车（P6） |
| `kv_prefix_stickiness` | 双 family 分离播种 + 30 串行请求（60% family 续接 + 40% 唯一键自由流） | P9 亲和保真 + P2 自由流多引擎摊开 + P6 完备（分级属性） |
| `kv_hot_prefix_tension` | 70% 流量的热 family（16 块长前缀）+ 7:3 交错 40 串行请求 | P9 family 粘持有者 **且** M2 持有者总份额有上限（首次校准）+ P2 + P6 |
| `kv_match_mixed` | 全命中 / 半命中 / 零命中三层对比（各 10 串行） | M3 全命中与半命中层集中在各自持有者（半命中 50% 仍过亲和门）、零命中层摊开（P2）+ P6 |
| `kv_lru_eviction_affinity` | 容量 4 的 LRU：prime 2 键 → 同键复放 → 同前缀+3 新键（G10 端到端） | 快照证明 cache_keys 封顶 4、evictions ≥1（最老块被逐出）；前缀命中保持复放请求亲和落同引擎 |
| `kv_decode_capacity_park` | 所有 decode 引擎 KV 打满 → 探针 Schedule（短客户端 deadline）→ master Cancel → 解压（G1） | KV 耗尽是 **WAIT 不是拒绝**：Schedule 挂起（客户端 DEADLINE_EXCEEDED、无拒绝响应、parked rid 未投递任何引擎）；master Cancel 释放无残留；解压后新请求完成 |

色注：无 profile 限制（P/D 两投递模式共享同一 delivery-capacity 准入）；大多共享默认
`smoke_spec`，个别自带专用 spec（如 3P 动态发现）。断言策略即"正确契约"；依赖 mock 的
`cache_key_set` / `/cache_evict` 能力（task #84 对齐契约）。

### balance（6 例）— 调度结果属性

**结果属性**用例（P 系列）：断言可观测的结果属性而非机制叙事；每个属性对着中央 band 表
分级（见上文"分级断言"）。全部 profile 无限制（共享默认 smoke 环境）。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `balance_uniform_serial` | 20 串行同构请求 × 两变体（plain / 单引擎慢 200ms） | P1 请求均匀性 max-share + P2 无饥饿（客户端落点计数；引擎速度不进评分，倾斜=缺陷） |
| `balance_concurrent_mix` | 20 请求 20 路并发突发 | P1（放宽带继承）+ P2 + P6：不塌缩单引擎；批准入背压（≥8 保留底线）之外的失败=缺陷 |
| `balance_overload_avoid_prefill` | 双 prefill 慢 5s + 147k token 种子制造单引擎热，恢复冷引擎后打基线 + 5 请求波 | P5 过载规避（热引擎份额≈0）+ P6 + P7 短请求保护（TTFT / 完成时长按投递模式双口径） |
| `balance_overload_avoid_decode` | 一个 decode 引擎 KV 打满（S11 强化） | P5 delta 口径（受压引擎新增完成 ≤1）+ P6 + 接管断言：其余 decode 实际吸收分流、至少两个引擎接活 |
| `balance_decode_spread` | decode 流量 n=10 / n=50 两档样本 | P2（最少引擎使用数）+ P1（4 引擎 case 校准带——KV 加权随机非均匀抽签）+ P6 |
| `balance_len_mixed` | 5 波双峰长度混合（每波 2 长 @131k–147k + 6 短 @512） | P3 token 加权 max-share ≈0.5（首次校准）+ P2 短请求两引擎都有份 + P6；请求数均匀性与 token 均衡在此场景真实冲突，故不断言 |

### elastic（8 例）— 动态扩缩容

引擎经 mock 控制面（`/add_engine` + `/remove_engine`）与文件动态发现链
（mock `--discovery-file` → DiscoveryFileStore 原子重写 → master
`FLEXLB_DISCOVERY_FILE` → FileServiceDiscovery → EngineSyncRunner → 路由）加入 / 离开
集群：master 必须收敛新拓扑、背景流量跨过渡存活、健康窗口内驱逐被移除引擎、扛住并发
增删风暴。扩缩容是**正常功能需求**（用户 2026-08 拍板），不是故障场景；收敛上限取
秒级（add/remove 各 10s、驱逐 30s）容慢 CI。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `elastic_add_flow` | 背景流量运行中 `/add_engine` 加 1 个 prefill | 新引擎在收敛上限内接到流量；背景流量成功率 ≥90% |
| `elastic_remove_flow` | 新引擎接入流量后被 `remove_engine` | 流量成功率 ≥90%；引擎从 mock services 与 discovery 文件**双双**消失；master inflight 95s 排空窗内清零 |
| `elastic_add_remove_cycle` | 3 轮 add→验证→remove→验证 | 每轮 discovery 文件同步、alive 收敛、新引擎接流量、移除后文件可解析；最终拓扑恢复初值、恢复正常 |
| `elastic_rebalance` | 基线 50 请求后扩容 3P，再发 50 请求 | 新引擎份额 >0 且 <60%（成本感知再平衡、非独占）；第二阶段零错误 |
| `elastic_stop_after_add` | 新引擎接流量后 `/stop_engine`（3-fail 逐出）→ 停机期请求 → `/start_engine` | alive 3→2 逐出；停机期请求在幸存 prefill 上成功；恢复后 alive 回 3 且新引擎重新接流量 |
| `elastic_concurrent_ops` | 10s 双加双删风暴 + 每秒健康探测与 master 探活 | master 全程 HTTP 200；discovery 文件可解析且条目数与 mock snapshot 一致；健康探测成功率 ≥50%（风暴合法压低可用性，全黑仍挂） |
| `elastic_remove_pending_drain` | 受害 prefill 双 inflight 批租约占满时 `remove_engine`（有请求已 QUEUED 未投递——用户识别缺口） | 每个滞留请求在 stale-TTL+余量（40s）内到达**可见终态**（P6）；master 账目 50s 内回基线；拓扑收敛 victim 消失 |
| `elastic_add_preference` | 稳定唯一键背景流 + 扩容第 3 个 prefill，45s 量测窗（10s 瞬态 + 35s 稳态）——用户点名缺口（"流量会不会偏好没排队的新引擎"） | 新引擎接到流量；瞬态高份额允许（观测校准带）；稳态份额 <60%（P1 覆写带）、老引擎各 ≥10%（P2）；全程 200、背景流成功率 ≥90% |

色注：全部 `batch-window`（`elastic_spec` 用 FLEXLB_CONFIG 钉旧故障轴）。共享
`elastic_spec`：2P+4D、`discovery="discovery_file"` 动态发现、TTL=30s；`remove_pending_drain`
用专用 2P+2D 私有环境。

### engine_fault（13 例）— 引擎作为故障受害者

prefill / decode 引擎死亡、摇摆、ack 中途 crash、enqueue 报错、停摆（no_respond / 延迟
enqueue / 执行膨胀）：master 必须观测损失、靠幸存者继续服务、账目有界、引擎回来后完全
重收敛。含**恢复家族 E1–E6**（期望行为断言：恢复代际 bump、KV 全量快照重同步、不复活、
短 gap 不退役、长 gap 退役并栅栏、KV 用量归零）。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `engine_fault_down_phases` | 五相位引擎下线断言集（S2/S4 合并：下线 / 恢复 / 流量 / TTFT 回归门） | master 存活 + 幸存引擎接管 + 恢复率达标 |
| `engine_fault_flap` | 单 prefill ≥5 次快速 stop/start 摇摆（3-strike 逐出 vs 重发现竞速），背景流量常开 | 全程 master HTTP 200；摇摆停止后引擎被重新发现（拓扑收敛）、路由与请求恢复 ≥95%、95s 排空窗内 inflight 无泄漏 |
| `engine_fault_crash_after` | `/inject crash_after`：首次 enqueue 即真 crash（内存全清 + 端口死）+ 空回执 | 恰一引擎 crash、master 观测损失（alive 降）；幸存引擎服务 ≥60% 突发；`/start_engine` 后拓扑完全恢复；空批 master 账目残渣**有界不增长**（uncertain fence 隔离区），引擎侧完全干净 |
| `engine_fault_no_respond` | 所有 prefill 注入 no_respond（E2：请求停摆） | 错误表面化到客户端（schedule 错误或流超时 / 错误）；清除注入后恢复正常 |
| `engine_fault_enqueue_error` | 所有 prefill 注入 enqueue_error（E3） | 错误表面化（流不完成）；清除后恢复 |
| `engine_fault_enqueue_delay` | `enqueue_delay=1500ms` 推迟整个 enqueue runnable（BATCH only） | 端到端延迟增量 ≥1.2s 且请求**仍成功**（延迟非失败）；清除后延迟恢复 |
| `engine_fault_generate_delay` | `generate_delay=1500ms` 抬高 prefill 执行估计（全 profile） | TTFT 增量 ≥1.2s 且请求仍成功；清除后 TTFT 恢复 |
| `engine_fault_recovery_generation_bump` | 引擎 gRPC 拒绝一段有界窗口后恢复（E1） | 恢复必须发布新 WorkerStatus 代际：retire 落地、created-count 严格增长、新代际账目从零起步、新请求完成——旧代际账目不得泄漏 |
| `engine_fault_recovery_kv_resync` | 恢复时缓存视图必须全量快照重建（E2，内存保留 / 丢失双 regime） | regime A：LRU 存活 → 持有关系保持（同前缀 ≥4/5 粘回）；regime B：键集被清 → 重建视图反映空集（≤3/5 摊开）；增量基线跨代际 resync 即 finding |
| `engine_fault_recovery_no_resurrect` | 崩溃前在途请求不得在恢复后复活（E3，真 crash 语义） | 恢复引擎账目零 inflight、全局 inflight TTL 内排零；引擎**空内存**回来（无运行任务 / 无持有块 / 空 KV）；旧 rid 永不完成；新流量正常调度 |
| `engine_fault_status_gap_no_bump` | 2 个轮询 tick（~40ms）的短状态空窗（E4） | 短空窗=网络抖动：**不**退役代际（created-count 不变）、拓扑保持 discovered==alive、空窗期流量照常成功 |
| `engine_fault_status_gap_long_retire` | 5s 长状态空窗（≈4+ 次超时轮询 > 3-strike）（E5） | 长空窗=崩溃：代际退役落地并**栅栏**账目 / inflight（TTL 内结算零——永不清算即 F7 缺口）；上报恢复后新代际服务新流量 |
| `engine_fault_recovery_kv_usage_reset` | 引擎重启且 KV 内存丢失（E6） | KV 用量自报**归零**（空 LRU；续旧读数=重启保真缺陷）；恢复引擎持续接流量（不被旧占用率拉黑）；代际确实翻新 |

色注：`down_phases` / `flap` / `crash_after` / 恢复家族 6 例声明 `batch-window`（fault
spec 钉旧故障轴）；`no_respond` / `enqueue_error` / `generate_delay` 无 profile 限制
（共享默认 smoke 环境跑真实 per-profile 配置）；`enqueue_delay` 按 `requires` 限 BATCH
投递 profile。

### master（3 例）— master 进程作为故障受害者

FlexLB master 自身下线（kill -9 + 重启）、按配额阻塞准入、冷启动首连风暴下被压：工作
流量必须收敛到健康拓扑，在途状态必须结算（TTL 或显式清理），重启后的 master 必须带
干净状态回来。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `master_kill` | 基线请求成功后 kill -9 master → 重启（HA） | 拓扑重新收敛（P≥2 / D≥4 alive）；新 master inflight 干净（干净状态）；恢复请求成功 |
| `master_quota_block` | 1P+1D、maxInflightBatches=1：填满配额 → 停唯一 prefill → 新请求 → TTL 清理 → 重启引擎（S3 移植） | 配额阻塞期新请求失败 ≥50%；TTL 清理后重启引擎恢复 ≥90% |
| `master_coldstart_burst` | master 刚 ready 瞬间打 20 请求（intake 缺陷回归探针：CONNECT_TIMEOUT 20ms / 首连 3-strike 误标 / 非原子 getOrCreateWorkerStatus） | 记录失败率与误标死样本数作 intake 修复基线（预期今日 FAIL 或边缘过） |

色注：全部 `batch-window`；`master_kill` / `master_quota_block` 用共享 elastic 环境
（quota 为专用 1P+1D spec），`master_coldstart_burst` 用专用 coldstart spec（稳定窗置 0，
流量直击首连风暴窗口）。

### admission（3 例）— 准入门契约

被门拒绝的请求必须**快速、响亮、带类型地**失败——不挂起、不消失、不泄漏 inflight
状态；压力解除后系统必须恢复。一例一条拒绝路径。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `admission_queue_depth_reject` | 每个 prefill 持 ≥1 慢 pending 请求占满 queue_depth 门（G8） | 下一个 enqueue 被**快速拒绝**（"queue depth limit exceeded" → BATCH_DISPATCH_FAILED）而非无界堆叠；门解除后占用者完成、新请求成功、无泄漏 |
| `admission_slo_queue_deadline` | kv_pressure 把每个 prefill 可用 KV 挤到 0 + queueTimeoutMs=1500（G11） | KV 门是 WAIT 条件：请求挂起至 deadline 约 1.5s 以带类型的 deadline 错误快速终态失败；解除压力后新请求成功 |
| `admission_master_capacity_reject` | `maxOutstandingRequestsGlobal=2` + PRIORITY（G11） | 超出全局预算的请求在 submit 路径**同步快拒** RESOURCE_EXHAUSTED（不排队不泄漏）；在途请求终止后顺序请求再次成功 |

色注：`queue_depth_reject` 按 `requires` 限 BATCH 投递 profile（门只在 EnqueueBatch 入口
检查，共享默认环境跑真实 per-profile 配置）；另两例 `batch-window`（专用 spec 钉旧故障轴）。

### direct（1 例）— client-direct gRPC 契约

完全绕过 master 的直连请求（load-client direct 部署形态）：引擎侧注入必须表面化在
GenerateStreamCall 入口，故障清除后恢复，且引擎侧无 inflight 残留。

| 用例 | 场景 | 期望（契约） |
| --- | --- | --- |
| `direct_generate_error` | 直连（不经 master）在 GenerateStreamCall 入口注入 generate_error（G6/G7） | 直连流立即以注入错误失败且引擎侧无 inflight 注册；清除后新直连请求正常完成 |

色注：无 profile 限制——direct stub 序列不经过 master 的 dispatcher，任何 profile 下
执行的都是同一条路径。

## 扩展指南：加一个新 case

最短路径（以 `cases/cancel.py` 为模板）：

1. **注册**：在对应分类模块用本模块的 `@case` 装饰器注册（进该模块的
   `*_CASES` 列表；新分类则建新模块 + 在 `cases/__init__.py` 导出，runner 的
   `ALL_CASES` 与 `--category` choices 需同步）。
2. **docstring 四要素**：`Scenario`（场景构造）→ `Behaviour`（被测机制）→
   `Expected (contract)`（**正确契约**，非当前行为）→ `Prediction`（预判
   passes / UNCERTAIN / FINDING；预期挂的必须声明 finding 及依据）。
3. **RID**：请求 id 一律 `ops.next_request_id(rid_base(ctx, "<category>"))`——
   段基址按 category×profile 预分，复用 master 的 dedup 表不冲突；绝不手写裸数字
   （`cancel_t5` 的 99999 是刻意的幽灵 rid）。
4. **finally 卫生**：fire-and-forget 请求必须消费到终态（cancel 兜底）；注入要清除；
   动态引擎要移除；perf 要复位——共享环境上的残留会毒化后续 case。
5. **排空断言**：依赖 stale-inflight TTL 清账的断言用 `TTL_DRAIN_TIMEOUT_S`（95s），
   不要自造更短的窗。
6. **同步计数**：runner 模块 docstring 的分类计数与本 README 清单须随 `--list`
   实测同步更新。

## 维护注记

- 用例数与清单必须与 `python3 flexlb_functional_tests.py --list` 实测一致（当前全集
  81：cancel 13 / status 19 / kv 15 / balance 6 / elastic 8 / engine_fault 13 /
  master 3 / admission 3 / direct 1；注意 `--list` 受 `--profile` 过滤，全集计数以
  `ALL_CASES` 为准）。
- 本 README 只是**索引**：每个用例的详细断言、构造手法、band 校准与 finding 预判以
  `cases/<category>.py` 内各 case 函数 docstring 及模块 docstring 为准；分级语义细节见
  `grade.py`，环境 / 就绪 / 排空细节见 `harness.py`。
