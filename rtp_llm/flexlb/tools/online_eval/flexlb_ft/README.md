# FlexLB Case Tests

FlexLB 调度器的场景测试套件：每个用例启动一小片 mock 引擎集群 + FlexLB master，对一个具体场景钉住**一条调度器行为契约**。与 `online_eval` 压测管线（`run_online_eval.sh`，QPS / 时长负载形态与时序分析）互补——压测看性能，本套件看行为正确性。

## 在 v1 老栈上运行的预期结果

本套件契约按 v2 master 行为编写；跑在 v1 老栈（本分支的 flexlb-api / flexlb-sync）上，94 例实测分布为 **57 PASS / 32 FAIL / 3 finding-confirmed / 1 finding-resolved / 1 SKIP**（admission / status 为本分支断言口径适配后的重跑结果，其余分类沿用同一代码基线的既有运行）。

- **SKIP（1）**：`kv_decode_capacity_park` —— park 是 v2-only 机制，v1 无对应路径，显式跳过。
- **finding-confirmed（3）/ finding-resolved（1）**：声明式预期失败探针（expected_fail），确认即套件的审计产出，不计入 FAIL。
- **FAIL（32）**：全部为 v1 真实行为缺口，分四族——
  1. **park 机制缺失**：v2 的容量 park（等待而非拒绝）在 v1 无对应路径，master 侧 park 不可观察、受压请求无终态（`admission_batcher_queue_capacity_park`、`admission_engine_decode_hard_gate_unbounded_park`、`admission_placement_pool_wait`）；
  2. **generation / retire 语义**：状态版本回退不触发 3-strike 退役；停止应答场景的退役耗时超出观察窗（`status_version_regress`、`status_status_no_respond`）；
  3. **inflight 账目泄漏**：特定故障注入后账目残留——RECEIVED 相位条目释放后不清、僵尸 RUNNING 驻留、伪造 batchId / 哨兵 id 注入产生登记、KV 内存压力 probe 挂起（`status_decode_waiting_before_prefill`、`status_zombie_completed_running`、`status_unknown_batchid`、`status_special_ids`、`admission_engine_kv_lack_mem_fast_reject`）；
  4. **错误码不透传 / 终态兜底**：ack 错误码不逐请求透传、deadline 拒绝无 typed 载体、decode 抑制后 prefill 账目无兜底结算、终态重放非幂等（`status_ack_multi_error`、`admission_batcher_queue_deadline`、`status_decode_suppress_finished`、`status_duplicate_finished`）。

这些 FAIL 是套件对老栈的审计产出（finding），不是套件或环境问题。泄漏类 FAIL 由产生泄漏的用例自身承担；账目基线类用例（指纹逐位对比、干净 inflight 基线）通过 isolated 环境与其它用例隔离，前序用例的泄漏残留不会污染它们的断言。

**断言口径的 v1/v2 双标适配**（栈间行为差异，非缺口）：准入拒绝错误码 v2 细分 typed 码（8502 QUEUE_FULL / 8510 / 8511），v1 统一走 8431 RESOURCE_EXHAUSTED 族（如 "master outstanding capacity exhausted"、"post-success backpressure"）——相关 admission 用例验证拒绝语义（同步、typed、可恢复），双口径接受两套体系；入队拒绝可经响应流终态（v2）或 Schedule RPC 同步失败（v1）两种 surface 传递，用例均接受；TTL 清账断言在 master 未实现 TTL-eviction prometheus counter 时降级为行为断言（账目在 TTL 窗口内归零 + 时间窗），counter 存在的栈上断言保持硬性。

## 快速开始

前置：先用 maven 构建两个 jar（缺失时启动会直接报错指路）——`flexlb-mock-engine` 的 all-in-one jar 与 `flexlb-api` jar（路径见 `harness.MOCK_JAR` / `harness.API_JAR`）。

```bash
cd rtp_llm/flexlb/tools/online_eval

python3 parallel_runner.py                    # 全量：默认 4 路 case 级分片（profile=batch-window grade=normal）
python3 parallel_runner.py --dry-run          # 只打印分片矩阵与端口矩阵，不执行
python3 parallel_runner.py --parallel 1       # 串行等价（单进程全量）
python3 parallel_runner.py --categories kv --parallel 2   # 只跑指定分类
```

首轮没有逐例耗时基线时 case 均匀切分；跑完自动落盘耗时基线（见「并行编排」节），第二次起自动按实测耗时做 LPT 均衡。

## CLI（parallel_runner.py）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--parallel` | `1..N` | lane 数（默认 4；1 = 单进程串行等价；上限由 mock stride 推导：默认 2000 → 6 路，`--mock-stride 500` → 21 路）|
| `--shard` | `case` / `category` | 分片粒度（默认 `case`：逐例 LPT 摊平，同家族 case 可散到不同路；`category`：按家族装箱，最重家族决定 wall 下限）|
| `--categories` | 逗号分隔 | 只跑指定分类（连字符/下划线均可）|
| `--timing-json` | 路径 | 逐例耗时基线（默认自动读 `/tmp/flexlb_ft_timing_baseline.json`，可用 env `FLEXLB_FT_TIMING_BASELINE` 改址）|
| `--mock-stride` | `≥153` | lane 间 mock 端口步长（默认 2000；实测每路 mock 窗口恒 153 口，500 有 ~3x 余量）|
| `--profile` / `--grade` | — | 透传给 runner（默认 `batch-window` / `normal`）|
| `--json` / `--out-dir` | 路径 | 聚合 JSON 与 lane 产物目录（默认 `/tmp/flexlb_ft_parallel_<ts>/aggregate.json`）|
| `--keep` / `--dry-run` | — | 透传给 runner / 只打印计划不执行 |

### 定向复跑（flexlb_functional_tests.py）

底层 runner 保留独立 CLI，主要供编排器内部 spawn（每路一次 `--cases <逗号列表>` 调用）与单例定向复跑：

```bash
python3 flexlb_functional_tests.py --list                              # 列出用例
python3 flexlb_functional_tests.py --cases a,b,c                       # 精确 case 名列表（优先于 --category/--filter）
python3 flexlb_functional_tests.py --category kv --json results.json   # 单分类 + JSON 结果
python3 flexlb_functional_tests.py --filter cancel_basic --profile single-nonbatch   # 子串过滤
```

`--cases` 为精确 case 名逗号列表，仍受 `--profile` 过滤，未知名报错退出（rc=2）。其余 runner 参数（`--run-root` 等）见其 `--help`；全集 **118 例**（10 分类），`--list` 按当前 profile 过滤，默认 batch-window 下 103 例（priority 14 例仅 single-nonbatch、1 例仅 NON_BATCH 投递形态适用），用例间环境按需复用 / 重建。

`--profile` / `--grade` 的取值语义（两个入口通用）：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--profile` | `batch-window` / `single-nonbatch` / `single-batch` / `window-nonbatch` | 调度形态（默认 `batch-window`）：decision（fixed_window / single）× dispatcher（batch / non_batch）两轴组合 |
| `--grade` | `strict` / `normal` / `loose` | 断言档位（默认 `normal`）：数值断言按档位边界评估，超出运行档界值即 FAIL；逐例记录实际达到档并汇总为运行判定（优异 / 良好 / 边缘 / 不可用） |

## 并行编排与耗时基线

串行全量 103 例约 35–55 分钟，瓶颈是各 case 的等待窗口（batch drain / TTL / 收敛）而非 CPU。默认的 `--shard case` 把每个 case 按实测耗时逐例 LPT 摊到 N 路（status 24 例不再独占一路，wall 跟随均衡总和而非最重家族）；每条 lane 是一个独立 runner 子进程树，端口与 run 目录显式分段，互不相碰。

**耗时基线自维护**：每轮跑完（任何分片模式、任何子集），编排器把逐例 `duration_ms` **合并**写入共享基线 `/tmp/flexlb_ft_timing_baseline.json`（原子写；合并语义——定向子集 run 只刷新它跑过的 case，不破坏全量基线；env `FLEXLB_FT_TIMING_BASELINE` 改址，多操作员共享一台机时可各用各的）。case 分片未显式传 `--timing-json` 时自动读该文件：首轮不存在则均匀切分（正常态，无告警）；个别 case 无记录退化为该家族单例权重（stderr 警告）。显式 `--timing-json` 仍可覆盖（缺失/不可读 → 均匀切分 + 警告）。

端口分段（lane i，0 起）：

| 段 | 区间 | 说明 |
| --- | --- | --- |
| master 组 | `18080+10i .. 18080+10i+5` | http/mgmt/grpc = +0/+1/+2（单 master、HA Tier-1 A、Tier-3 三选一路径共用组首）；HA Tier-1 B = +3..+5 |
| mock 窗口 | `55151+S*i .. +151` | 显式 `FLEXLB_FT_MOCK_BASE_GRPC_PORT`，消除并发自动扫描的 bind TOCTOU；实测占用宽度恒为 153 口（http=base-1、engines=base..base+n-1、victim zone=base+149..151） |

- `--parallel` 上限由 mock stride 推导：`base+stride*(N-1)+151 ≤ 65535`（默认 stride 2000 → 6 路；`--mock-stride 500` → 21 路上限，容器实测承载 4–8；stride 下限 153 = 窗口宽度，代码内校验）。
- 同机与他人共用且对方占用默认段时，用 `FLEXLB_FT_PARALLEL_MASTER_BASE` / `FLEXLB_FT_PARALLEL_MOCK_BASE` 整体平移（stride 不变，lane 间仍互斥）。
- 其余 env（如 `FLEXLB_FT_HA_DUAL_MASTER=1`）原样透传给每条 lane；HA 分组与 mock 段已按 lane 同步分段，无需手工干预。

聚合 `--json` 保持单 runner schema（summary + cases[]），另加：`cases[].lane`、`lanes[]`（各路 category 集合 / exit_codes / wall_s）、`summary.parallel / wall_time_s / serial_case_time_s`（最后一项为逐例耗时之和，是串行 wall 的下界，报告加速比时对标实测串行 35–55 分钟而非它）；case 级分片另记 `summary.shard`（category|case）与 `lanes[].case_names`（各路精确 case 名单，分片矩阵是 run 记录的一部分）。退出码 = 任一 lane runner 非零或存在 FAIL。`--parallel 1` 单 lane 跑全量（case 模式为一次 `--cases` 全列表调用，category 模式走 `--category all` 单进程路径），与直接串行等价，可作编排无回归的冒烟基线。

实测参考（110 开发机容器，batch-window profile，共享负载）：串行单进程 wall 4918s（98 例快照）；category 级 4 路 wall 2444s（2.01x）——wall 被最重家族钳制（status 24 例实测 2104s，占串行 43%）；case 级 6 路 wall 941s（5.22x，15.7 分钟，最重路 16 例），8 路（`--mock-stride 500`，mock base 平移避开他人占用段）wall 703s（6.99x，11.7 分钟）——逐例摊平后钳制消除。等价性口径：并行 run 对串行基线逐例对照 + FINDING 集一致；实测 6 路 89/98 一致、9 例翻转全部单向好转（对翻转例同 jar 同 env 定向复跑两轮结果稳定，属快照漂移而非编排回归）；8 路对 6 路 FINDING 集完全相等。

## 测试分类（118 例）

断言一律写**正确契约**而非当前实现——跑挂即 finding。表内「期望」为一句话摘要，完整断言与构造细节以各用例 docstring 为准。

### cancel（13 例 · 全 profile）

客户端取消的全生命周期契约：流终止、引擎侧释放、master 账目排空——幂等、跨投递边界（master 持有 vs 已投引擎）、覆盖 prefill / decode 各相位。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `cancel_basic` | 请求流式输出中客户端发显式 Cancel | 流终止；引擎侧取消可见；master inflight 排空；后续请求正常 |
| `cancel_idempotent` | 同一请求连续 Cancel 两次 | 第二次幂等不报错；流终止；恢复 |
| `cancel_sibling_isolation` | 3 并发请求，长 decode 的 B 中途被取消，A/C 为短请求 | B 终止且未完成；A/C 照常完成；引擎记录取消；恢复 |
| `cancel_after_terminal` | 请求已完成后再发 Cancel | 幂等成功、无二次结算；恢复 |
| `cancel_unknown_rid` | 对从未存在的 rid 发 Cancel | 调用不抛异常（幂等处理） |
| `cancel_phase_timing` | A 处于 prefill 相位、B 处于 decode 相位时分别取消 | 两相位取消都终止流；恢复 |
| `cancel_anomaly_path` | 注入失败后的请求走客户端侧取消路径 | 流终止、取消记录可见；（BATCH 投递）inflight 清零；恢复 |
| `cancel_deadline_exempt_inflight` | queueTimeout 到期落在请求已被引擎认领之后 | 不取消：完整输出、无引擎侧取消记录、走普通完成路径 |
| `cancel_schedule_drop_delivered` | 批已认领后客户端取消 Schedule RPC 本身（仅 BATCH 投递） | master 仍向原 prefill 发真引擎 Cancel；账目经 CANCELLED reconcile 结算 |
| `cancel_engine_notfound_settle` | 迟到的 Cancel 到达时引擎已无该请求 | master 幂等处理；引擎 NOT_FOUND 不报错；已有终态不被复写 |
| `cancel_preemption_victim` | PRIORITY 排序下高优先级（priority=70）请求驱逐正在运行的低优先级（priority=30）请求 | 受害者以抢占错误码 8429 终态结算；系统继续运行 |
| `cancel_stream_break_prefill_autonomous` | 客户端直接断流（不发 Cancel），prefill 引擎侧自主清理（仅 BATCH 投递） | 引擎感知断流并清理自身状态；账目无残渣 |
| `cancel_stream_break_decode_autonomous` | 客户端直接断流，decode 侧自主提前终态（仅 NON_BATCH 投递） | decode 不等 stale-inflight TTL、主动上报终态；账目无残渣 |

### status（24 例 · 固定 batch-window）

engine→master 状态上报通道的故障注入：ack 丢失 / 部分失败 / 错误码、终态抑制、伪造任务、重放、版本 / 游标回退、僵尸 RUNNING——master 账目在每种畸变下都必须收敛到正确终态。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `status_inflight_ttl_cleanup` | 引擎侧请求卡死、终态永不到达 | stale-inflight TTL 到期清理账目；后续请求不受污染 |
| `status_ack_partial_fail` | 一批 4 个请求的 enqueue ack 中 1 个失败（瞬时码 13 / 永久码 8431 两档矩阵） | 失败请求表面化、其余照常完成；失败成员及时清出账目；瞬时码应重试至成功或 SLO 终态，永久码快速终态不重试 |
| `status_ack_multi_error` | ack 携带两种不同错误码（8431 / 8510） | 错误码按请求逐个透传，各自正确 |
| `status_ack_empty_no_crash` | ack 整体丢失（空 ack，投递结果不确定） | master 不崩溃；不确定栅栏有界且最终可清空 |
| `status_prefill_suppress_all` | 所有 prefill 状态消息全静默 | TTL 兜底清账；master 存活；恢复 |
| `status_prefill_suppress_finished` | 请求终态被抑制但持续上报 RUNNING | queueTimeout 成为唯一出口并按时触发 |
| `status_status_no_respond` | 引擎停止应答状态轮询 RPC | 陈旧代际退役；其余引擎继续服务 |
| `status_unknown_rid_finished` | 上报 master 未见过的 rid 的终态 | 被忽略；账目指纹逐位不变 |
| `status_version_regress` | 状态消息版本号回退（陈旧代） | 3-strike 机制退役陈旧代际 |
| `status_decode_suppress_finished` | decode 侧终态被抑制 | 请求仍经兜底路径拿到终态 |
| `status_decode_before_prefill` | decode 侧先于 prefill 上报完成（prefill 事实全程被抑制） | D 终态足以结算请求，并事件驱动（≤10s）释放 prefill 账目——不死等 TTL |
| `status_decode_running_before_prefill` | decode 只报 RUNNING 不报终态，prefill 事实被抑制 | D 中间态零驱动：prefill 账目不被提前清理、请求不被结算；释放后 TTL 收敛 |
| `status_decode_waiting_before_prefill` | decode 只报早期 RECEIVED 相位（合成事实），prefill 事实被抑制 | D 中间态零驱动：RECEIVED 不结算请求、不提前清 prefill 账目；释放后 TTL 收敛 |
| `status_unknown_rid_running` | 一次性幽灵 RUNNING 条目（未知 rid） | 不驻留账目：TTL 内归零 |
| `status_unknown_batchid` | 伪造 batchId 搭配真实 rid 上报终态 | 错配不得结算真实请求 |
| `status_special_ids` | 边界 id：负值 rid 幽灵、batch_id=0 哨兵 / batch_id=-1 搭配真实请求 | 负 rid 幽灵终态被忽略（账目指纹不变）；哨兵 / 负 batchId 不得结算真实请求（batch_id=0 语义歧义单独标注） |
| `status_unbatched_single_request` | 无批上下文的孤立状态上报（batch_id 缺省 / 0 × RUNNING / finished 四组合） | 每种组合对账目零推进：不登记幽灵、不产生幻影结算 |
| `status_foreign_batchid` | 超范围 batchId（模拟另一 master 派发空间）+ 并发真实流量 | 外来事实被整体忽略；真实请求不被跨空间别名误结算 |
| `status_duplicate_finished` | 同一终态上报两次 | 重放幂等；无二次结算 |
| `status_cursor_regress` | 完成游标回退 3 步 | 幂等；已结算的不回退 |
| `status_finished_then_running` | 已终态请求随后又上报 RUNNING | 终态不可复活 |
| `status_zombie_completed_running` | 已完成任务的僵尸 RUNNING 持续上报 | tombstone 吸收；账目不回退 |
| `status_zombie_fake_running` | 永久驻留的假 RUNNING 探针 | 假 inflight 最终清零（不得永久驻留） |
| `status_fetch_error` | 批量 FetchResponse 流中途故障 | 故障表面化到客户端；账目收敛；恢复 |

### kv（15 例 · 全 profile）

KV 前缀缓存生命周期契约：per-engine 账本隔离、全局共享块的多持有者语义、增量同步收敛、热前缀风暴、容量冲突与亲和路由。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `kv_pe_admit_isolation` | 前缀族分别钉在引擎A、引擎B；零命中的族因引擎B减速全部落到引擎A | 只有引擎A的缓存 key 集增长（引擎B保持原样）；后续同前缀请求粘引擎A |
| `kv_pe_evict_zero_match` | 前缀族在引擎A上 prime 后被 /cache_evict 整族强制逐出 | 逐出同步进 master 索引：同前缀批次不再粘引擎A，按零命中平摊 |
| `kv_pe_prefix_continuity` | 共享族在引擎A挖出缺口、引擎B保留前 8 块连续 | 命中按连续前缀计算（缺口截断）：批次落在引擎B而非块数更多的引擎A |
| `kv_g_shared_block_both_match` | 同一前缀族被双投共享给两个引擎（全局索引 key→持有者集合） | 双持有者等命中 → 平局平摊：不钉单引擎，两引擎都接流量 |
| `kv_g_partial_release_redirect` | 共享族只从引擎A逐出 | 收敛后引擎B成为唯一持有者；同前缀请求全部重定向到引擎B |
| `kv_g_full_release_no_ghost` | 共享族从两个持有者全部逐出 | 索引无残渣：同前缀请求按零命中平摊，无引擎被钉死 |
| `kv_g_sync_convergence` | 交错 admit/evict 事件流后静默 ≥3.5s | 静默后路由与引擎快照一致：无持有者族平摊、唯一持有者族粘住 |
| `kv_g_engine_down_cleanup` | 共享族双投两个引擎后永久下线其一 | 只清下线引擎的条目：幸存引擎保留该族并继续接同前缀流量 |
| `kv_storm_hot_churn` | 4 个热前缀族轮换（50 请求）vs 每引擎 24 块的小 LRU | 复制因子与持有者翻转有界、命中率不崩塌 |
| `kv_capacity_conflict_overflow` | 持有者引擎账本已满（全命中但预测 TTFT 高）时同前缀波次到达 | 亲和让位：波次溢出到非匹配引擎；短请求不受拖累；无排队超时 |
| `kv_prefix_stickiness` | 多前缀族复用流量 + 自由流量混合 | family 续连粘住持有引擎；自由流量多引擎散布；全部完成 |
| `kv_hot_prefix_tension` | 70% 流量集中于单一热前缀族 | 粘性保持且持有者总份额有上限；另一引擎仍接自由流量 |
| `kv_match_mixed` | 全命中 / 半命中 / 零命中三档流量 | 全命中与半命中集中在持有者；零命中平摊多引擎 |
| `kv_lru_eviction_affinity` | 容量 4 的 LRU：同键回放 + 前缀扩展新键 | 回放粘原引擎；扩展触发 LRU 逐出最老块且 key 数封顶；前缀命中保亲和 |
| `kv_decode_capacity_park` | 所有 decode 引擎 KV 耗尽时新请求到达 | 请求被 park（等待而非拒绝）；Cancel 释放无残渣；清压后恢复路由 |

### balance（6 例 · 全 profile）

多引擎负载分布契约：均匀性、并发突发不塌缩、过载规避（prefill / decode 双侧）、token 加权均衡。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `balance_uniform_serial` | 同质串行流量 × 两变体（无注入 / 单引擎减速） | 请求在等价引擎间均匀分布；无饥饿（引擎速度不进入路由评分） |
| `balance_concurrent_mix` | 20 请求 20 路并发混合突发 | 不塌缩到单引擎（放宽均匀带）；无饥饿；全部完成 |
| `balance_overload_avoid_prefill` | 长请求种子使单一 prefill 账本过载 | 热引擎份额受抑、流量分流；短请求延迟不受拖累 |
| `balance_overload_avoid_decode` | 一个 decode 引擎 KV 耗尽 | 受压引擎停接新活（增量有界）；健康引擎实际接管流量 |
| `balance_decode_spread` | decode 流量 n=10 / n=50 两档采样 | decode 舰队无饥饿；份额分布在校准带内（KV 加权随机） |
| `balance_len_mixed` | 双峰长度混合（每波 2 长 + 6 短） | 按 token 足迹而非请求数均衡；短请求两引擎都接；全部完成 |

### elastic（8 例 · 固定 batch-window）

文件发现链路（discovery file → master 同步 → 路由）的动态扩缩容契约：拓扑收敛、切换期流量存活、计划内缩容零失败（优雅摘除：先摘路由、等在途排空再下线）。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `elastic_add_flow` | 负载流运行中新增一个 prefill 引擎 | 收敛窗口内 master 收编新引擎并开始接流；背景流成功率 ≥90% |
| `elastic_remove_flow` | 负载流运行中移除一个引擎 | 计划内缩容零失败：背景流无一失败、无一悬挂（每请求到达终态）；其余引擎持续服务；被删引擎从快照与发现文件消失；inflight 排空 |
| `elastic_add_remove_cycle` | 3 轮 新增→验证→移除→验证 循环 | 每轮发现文件 / 拓扑 / 流量全过且文件始终可解析；每轮移除时背景流零失败；拓扑还原 |
| `elastic_rebalance` | 扩容后持续投放流量 | 新引擎分到份额（>0）且不超过 60%（成本再均衡） |
| `elastic_stop_after_add` | 新增引擎→接流→/stop_engine→/start_engine | 3-strike 健康逐出下线；重启后重新被发现并恢复服务 |
| `elastic_concurrent_ops` | 10 秒双线程并发加 / 删风暴 | master 全程健康；操作计数一致；无拓扑残渣 |
| `elastic_remove_pending_drain` | 缩容时请求已在 master 队列排队、尚未投递到被删引擎 | 不搁浅：请求在窗口内拿到可见终态；无饥饿 |
| `elastic_add_preference` | 扩容后 45s 量测窗（前 10s 瞬态 + 35s 稳态） | 瞬态偏爱新引擎允许；稳态新引擎份额 <60%、老引擎 ≥10% |

### engine_fault（13 例 · 9 例固定 batch-window、4 例全 profile）

引擎进程级故障（下线 / 摇摆 / 真崩溃 / 停应答 / 入队错误 / 延迟）与恢复契约（代际、缓存重建、不复活、状态空窗、KV 归零）。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `engine_fault_down_phases` | 五相位引擎下线：基线→下线→接管→重启→恢复 | master 存活；幸存引擎接管；恢复率与 TTFT 回归门达标 |
| `engine_fault_flap` | ≥5 次快速 stop/start 摇摆 | 3-strike 逐出与重发现竞态下 master 不错杀、不残留 |
| `engine_fault_crash_after` | enqueue 计数触发真 crash（内存清零 + 端口击杀）+ 空回执 | 失败表面化；不确定栅栏有界；残渣不外溢；恢复 |
| `engine_fault_no_respond` | 所有 prefill 停止应答 | 失败请求表面化；账目收敛；恢复 |
| `engine_fault_enqueue_error` | 所有 prefill 每次入队都报错 | 失败请求表面化；账目收敛；恢复 |
| `engine_fault_enqueue_delay` | enqueue ack 延迟 ≥1.2s（仅 BATCH 投递） | 体现为调度延迟而非失败；请求仍成功 |
| `engine_fault_generate_delay` | prefill 执行时间膨胀 ≥1.2s | TTFT 增量如实反映；请求成功（全 profile 生效） |
| `engine_fault_recovery_generation_bump` | 引擎崩溃后恢复 | 发布新端点代际；旧账目不泄漏进新代 |
| `engine_fault_recovery_kv_resync` | 崩溃引擎带 / 不带 KV 内存恢复 | 缓存视图从全量快照重建：存活族粘回原引擎、空集摊开 |
| `engine_fault_recovery_no_resurrect` | 崩溃时有 inflight 请求 | 恢复引擎内存为空；崩溃前请求不复活、不假完成 |
| `engine_fault_status_gap_no_bump` | 2 个 tick 的短状态空窗 | 不误退役代际（抖动容忍） |
| `engine_fault_status_gap_long_retire` | 长状态空窗 | 代际退役；其账目 / inflight 被栅栏 |
| `engine_fault_recovery_kv_usage_reset` | 引擎全量重启后 | KV 用量从零起步（不沿用旧读数）；不被误拉黑 |

### master（8 例 · 固定 batch-window）

master 自身进程级故障与冷启动行为，以及双实例 HA 链路（冻结判活 / kill -9 failover / 全下线直连兑底 / 回切 wrap-around）。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `master_kill` | kill -9 杀死 master 后重启 | 拓扑重新收敛；inflight 干净；恢复率达标 |
| `master_quota_block` | 1P+1D 小集群配额阻塞 | 阻塞经 TTL 清理后恢复 ≥90% |
| `master_coldstart_burst` | 冷启动 ready 瞬间 20 请求风暴 | 首连风暴下不误杀健康引擎、请求被服务（回归探针） |
| `master_freeze` | SIGSTOP/SIGCONT 冻结 sticky master，短挂（6s）与长挂（46s）两档 | 短挂解冻后行不蒸发不切换；长挂判死后同请求重试切走，解冻后同进程账目连续、无错误风暴 |
| `master_ha_failover` | 双 master、sticky A，kill -9 A | 同请求重试兑底切 B（切换窗错误≈0）；切换后 B 100% 服务；无重复 rid |
| `fallback_direct` | ENABLE_FALLBACK 下杀掉全部 master | in-flight 双连接失败触发直连引擎兑底：fallback 路成功率健康、master 路为 0、无重复 rid |
| `fallback_negative_errorcode` | 业务错误码（8431）与短 deadline（800ms 冻结）下 fallback 已武装 | 均不触发兑底：业务错误经 master 应答带码、deadline 失败不重试不切换；清除后账目收敛 |
| `failback_wraparound` | 重建 sticky B 端态后重启 A、再杀 B | A 60s 内重新收敛并接恢复流量；对称切换绕回 A；inflight 干净、无 8511 风暴 |

### admission（15 例 · 14 例固定 batch-window、1 例仅 BATCH 投递）

准入门全谱——可等待 park 与快速拒两种语义：引擎侧（prefill 并发、decode 硬门、等待批上限、KV 块池、队列深度）与 master 侧（batcher 队列容量、placement 池、准入许可、全局 outstanding），外加 SLO 排队超时终态。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `admission_queue_depth_reject` | 引擎队列深度超限（仅 BATCH 投递） | 快速拒绝（BATCH_DISPATCH_FAILED）；注入清除后恢复 |
| `admission_slo_queue_deadline` | KV 压力门触发 WAIT 后排队超时 | ~1.5s 内带类型的 deadline 错误；恢复 |
| `admission_master_capacity_reject` | 全局 outstanding 上限（=2）打满 | 同步快速拒绝（8502 QUEUE_FULL / TooManyRequests）；排空后恢复 |
| `engine_prefill_concurrency_gate_park` | 单 prefill 并发门=1 下连发 4 请求，后续批整批驻留引擎等待队列 | 门为 WAIT 语义：零拒绝、等待可见；FIFO 全部完成；排空后账目干净并恢复 |
| `engine_decode_hard_gate_unbounded_park` | decode 硬并发门=128、master 路由上限放开，150 请求溢出 | 无界 park 不拒绝：Schedule 全部成功、等待可见；≥95% 完成后排空、账目干净并恢复 |
| `admission_priority_incomer_reject` | 唯一准入许可被低优先级请求占用，高优先级新来者到达且无抢占块 | 快速带类型 8431 拒绝、不悬挂；受害者不被抢占正常完成；许可释放后恢复 |
| `admission_batcher_queue_capacity_park` | batcher 等待队列容量收紧为 2，7 请求逐发溢出 | master 侧 park：零快速拒、parked≥1；FIFO 串行完成（批间隔≥1.2s）；排空后账目干净并恢复 |
| `admission_batcher_queue_deadline` | batcher 队列容量门下 queueTimeout=1.5s，溢出波 park 后到期 | 排队者 1-5s 内带类型 deadline 终态（8511，与 KV 门同码分类统一）；已投递者不受扰；账目干净并恢复 |
| `admission_placement_pool_wait` | prefill placement 池仅 1 席，A 运行中 B 到达被拒入池 | 池满为 WAIT：B 驻留 master 侧，池释放后被唤醒重试并晚于 A≥1s 完成；账目干净并恢复 |
| `admission_engine_waiting_batch_cap_reject` | 引擎等待批上限=1（运行时注入）打满后探测批到达 | 非等待门：快速整批 backpressure 拒绝；占用者不受扰；同压力下放开 cap 可 park；账目干净并恢复 |
| `admission_engine_kv_lack_mem_fast_reject` | 17 块引擎 KV 池被两个 8 块租 约占满后第 3 个 8 块请求入队 | 非等待门：快速 602 LACK_MEM 拒绝（引擎侧码非 8431）；租约完成后归还；恢复后新请求成功、账目干净 |
| `engine_prefill_token_budget_split` | 引擎内双预算重组（#8）：token 预算 1024，4×512 请求合一 master 批（Σ2048 超预算 2x） | 拆散前缀+尾段 park：全部完成；执行批计数 2 批/4 请求/最大 2；成员 batch_id 归属同一 master 批；账目干净并恢复 |
| `engine_prefill_token_budget_split_fifo` | 同拆散场景钉执行序（lifecycle end_ms） | 尾段成员完成严格晚于前缀 >1s（到达序贯穿重组）；执行批计数增量 2 批/4 请求/最大 2；账目干净并恢复 |
| `engine_prefill_token_budget_boundary` | 预算 2048 == 4×512（恰好等于） | 不拆：1 批/4 请求/最大 4 verbatim；执行窗口内无 park；账目干净并恢复 |
| `engine_prefill_regroup_disabled_verbatim` | 双 0 关闭重组（复现旧行为） | master 批原样执行：1 批/4 请求/最大 4；无 park；成员同批归属；账目干净并恢复 |

### priority（15 例 · 14 例固定 single-nonbatch + PRIORITY 轴 case 层注入、1 例全 profile）

优先级排序 + auto-TPM 抢占/降级契约（2026-09 自源线 `flexlb-priority-auto-tpm-ft` 迁移）。PRIORITY ordering 轴经 case 层 JSON 注入（`_prio_config` 走 `build_flexlb_config(ordering="priority", decision="single", dispatcher="non_batch")`，不扩 profile 表）；优先级双通道——proto field 14 与 `x-dashscope-inner-qos-level` header，由 PriorityNormalizer 归一（proto > header > defaultPriority）。基线标注含义：`[EV-1-FIXED]` 校准于 intake3 PendingPlacementCoordinator 线（拉式 park，6ad0315f10），`[EV-2]` 为 decode 驱逐不可达基线——两者在 intake3-rebuild Java 上的成立性待远端探针核对，断言口径迁移自源线不改。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `prio_order_basic` | 混优先级波（70/50/30…）在 inflight=1 的 prefill 串行窗口排队 | dispatch 严格 priority 降序、inversion_ratio=0（PR1；[EV-1-FIXED] 首个 parker 豁免） |
| `prio_same_level_fifo` | 7 个同 priority=50 逐发（rid 升序，首个兼占位） | dispatch 序=提交序（PR2 同级 FIFO 不受 priority 干扰） |
| `prio_normalize` | 三通道归一：proto field 14 显式 70、QoS header 显式 80、双缺省走 defaultPriority=30 | metric 面 `auto_tpm.request.count` 分桶 70/80/30 各就位（PR3），行为面缺省按 30 排在显式 70 后 |
| `prio_low_no_starvation` | 非饱和负载：30x4 先发、70x4 后到（共享 env、无 inflight cap） | 30 完成率 1.0——非挂起即唯一机械防饿保护（PR8 calibre） |
| `prio_queue_timeout_terminal` | queueTimeout=8s、高优先级持续占位、3 个 30 排队 | 到期即 8511 终态、绝不悬挂过 deadline（PR8 band） |
| `atpm_preempt_prefill_queued` | PREFILL_QUEUED 抢占开启，波内混优先级排队 | [EV-1-FIXED] 拉式 park 设计终形（PR10/PR5/PR6/PR4）；抢占计数与客户面终端对齐 |
| `atpm_preempt_decode_engine_owned` | DECODE_RESERVED/DECODE_ENGINE_OWNED 抢占（engineCancellation 必配）、decode 满载波 | [EV-2] decode 驱逐不可达：零 8429、占用者完成、incoming 走 EV2 拒绝族；AT5 闭环预算 band |
| `atpm_same_priority_zero_eviction` | 8 个显式 priority=50 + 第 9 个 50 全部 park | 零抢占（8400/8429 均不出现）；design-final 完成形状（PR4/AT3） |
| `atpm_preemption_disabled_zero_eviction` | PRIORITY ordering 但无 preemption 块（disabled），两轮饱和 | 零 8400/8429/8430（EvictionManager 前置拒绝）；[EV-1-FIXED] 饱和后 park 而非拒（AT2） |
| `atpm_timeout_attribution` | 短 queueTimeout 下排队到期归因 | 到期统一 8511 + reason=UNSPECIFIED（PR7；归因分类器 intake3 零调用点——已申报观察缺口） |
| `atpm_comparator_frozen_weak` | 同一负载形态分别跑 PRIORITY env 与 FIFO env（F1 控制） | priority 半场 dispatch 序显著异于 FIFO 半场（构造期序型决定行为，PR9 弱断言） |
| `atpm_error_code_family` | 三段独立构造 8400/8402/8403/8429/8511 触发条件 | 各码只在自身触发下出现、互不串扰（AT4 带类型终端可观测） |
| `atpm_config_strict_reject` | 3 个非法 FLEXLB_CONFIG 原始 JSON 变体（removed 字段 / FIFO+defaultPriority / owned 无 engineCancellation） | master 启动失败 + 严格解析器报文族命中；rejected 后 current=None（AT1） |
| `atpm_decode_reservation_priority` | decode 面三波：30<70 / 50==50 / kvBucket 偏好（D1 共享 env，kv_pressure 注入时序纪律） | [EV-2] 三波零驱逐、victim metric delta=0；[EV-1-FIXED] incoming 8511 park 终态（AT7 跨阶段一致性） |
| `atpm_observability_integrity` | ENV-O1 复合编排（debug 日志 + FLEXLB_MONITOR_MODE=all） | 客户面形状 + `auto_tpm.request.count` 分桶 4/3/2/1 + latency success 桶 + `[priority-scheduler]` 日志 + pv.log admissionRejectReason 全在场（AT8/AT6） |

### direct（1 例 · 全 profile）

直连引擎入口（不经 master 调度决策）的故障注入。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `direct_generate_error` | 直连引擎入口 GenerateStreamCall 注入 generate_error | 立即失败；不注册 inflight；注入清除后恢复 |

## 新增用例

1. 在 `cases/<分类>.py` 用 `@case("name")` 注册函数；docstring 写明场景与期望契约（断言写正确行为而非当前实现，预期挂掉的用例在 docstring 声明）。
2. 需要时声明 `profiles`（限定调度形态）或 `requires`（能力需求，如 `enqueue_batch`）。
3. 函数返回 `(passed, detail)`；带数值断言时用 `GradeReport` 返回三元组。
4. `python3 flexlb_functional_tests.py --filter <name>` 单例验证。
5. 更新本 README 对应分类表，保持与 `--list` 同步。

## 更多细节

完整断言与构造细节见各 case 函数 docstring 与 `harness.py`。
