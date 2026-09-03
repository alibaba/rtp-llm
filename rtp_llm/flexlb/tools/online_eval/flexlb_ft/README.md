# FlexLB Case Tests

FlexLB 调度器的场景测试套件：每个用例启动一小片 mock 引擎集群 + FlexLB master，对一个具体场景钉住**一条调度器行为契约**。与 `online_eval` 压测管线（`run_online_eval.sh`，QPS / 时长负载形态与时序分析）互补——压测看性能，本套件看行为正确性。

## 快速开始

前置：先用 maven 构建两个 jar（缺失时启动会直接报错指路）——`flexlb-mock-engine` 的 all-in-one jar 与 `flexlb-api` jar（路径见 `harness.MOCK_JAR` / `harness.API_JAR`）。

```bash
cd rtp_llm/flexlb/tools/online_eval

python3 flexlb_functional_tests.py                 # 全量（默认 category=all profile=batch-window grade=normal）
python3 flexlb_functional_tests.py --list          # 列出用例
python3 flexlb_functional_tests.py --category kv --json results.json                  # 单分类 + JSON 结果
python3 flexlb_functional_tests.py --filter cancel_basic --profile single-nonbatch      # 子串过滤
```

## CLI

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--category` | `all` / `cancel` / `status` / `kv` / `balance` / `elastic` / `engine-fault` / `master` / `admission` / `direct` | 场景分类（默认 `all`） |
| `--profile` | `batch-window` / `single-nonbatch` / `single-batch` / `window-nonbatch` | 调度形态（默认 `batch-window`）：decision（fixed_window / single）× dispatcher（batch / non_batch）两轴组合 |
| `--grade` | `strict` / `normal` / `loose` | 断言档位（默认 `normal`）：数值断言按档位边界评估，超出运行档界值即 FAIL；逐例记录实际达到档并汇总为运行判定（优异 / 良好 / 边缘 / 不可用） |
| `--filter` | 子串 | 按用例名子串过滤 |
| `--json` | 路径 | 逐用例结果写成 JSON |
| `--list` | — | 列出当前过滤条件下的用例并退出 |
| `--keep` | — | 跑完保留环境不 teardown |

全集 **86 例**；`--list` 按当前 profile 过滤，默认 profile 下显示 85 例（1 例仅 NON_BATCH 投递形态适用）。用例间环境按需复用 / 重建。

## 测试分类（86 例）

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

### master（3 例 · 固定 batch-window）

master 自身进程级故障与冷启动行为。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `master_kill` | kill -9 杀死 master 后重启 | 拓扑重新收敛；inflight 干净；恢复率达标 |
| `master_quota_block` | 1P+1D 小集群配额阻塞 | 阻塞经 TTL 清理后恢复 ≥90% |
| `master_coldstart_burst` | 冷启动 ready 瞬间 20 请求风暴 | 首连风暴下不误杀健康引擎、请求被服务（回归探针） |

### admission（3 例 · 2 例固定 batch-window）

三级准入拒绝：引擎队列深度快速拒、KV 压力 SLO 等待-超时、全局 outstanding 上限。

| 用例 | 场景 | 期望 |
| --- | --- | --- |
| `admission_queue_depth_reject` | 引擎队列深度超限（仅 BATCH 投递） | 快速拒绝（BATCH_DISPATCH_FAILED）；注入清除后恢复 |
| `admission_slo_queue_deadline` | KV 压力门触发 WAIT 后排队超时 | ~1.5s 内带类型的 deadline 错误；恢复 |
| `admission_master_capacity_reject` | 全局 outstanding 上限（=2）打满 | 同步快速拒绝（RESOURCE_EXHAUSTED）；排空后恢复 |

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
