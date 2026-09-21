> [!WARNING]
> 历史快照：此文件不是当前操作依据。当前入口为 `tools/online_eval/README.md`。

# FlexLB 功能与场景测试逐项审查

> 基线：`codex/ft-case-framework` 提交 `e580ca2fb25f65cc2cecb90668ba51de5e54a85f`，2026-09-18。清单来自 `scenario_runner.py --source scenarios --list-json --suite all`，并核对 YAML 参数和对应 Python 程序。`elastic/lifecycle.yaml`、`case_programs/elastic_lifecycle.py`、`scenario/actions/elastic_lifecycle.py` 在审查时另有未提交改动；本报告描述当时工作区可编译出的实例，相关判断须在这些改动定稿后复核。本文是**设计审查**，不是通过率报告；没有把一次 PASS/FAIL 当成测试价值的依据。

源文件入口：[场景 YAML](../scenarios/) · [Python 用例程序](../flexlb_test_framework/case_programs/) · [编译/执行入口](../scenario_runner.py)。复核命令：`python3 scenario_runner.py --source scenarios --list-json --suite all`（在 `tools/online_eval` 目录运行）。

## 口径与总量

“测试”按 `case::variant` 计，一种变体可在 sb/sn/wb/wn 等 master 模式下展开为多个执行实例。当前有 **33 族、190 变体、388 实例**：功能 149 变体/320 实例，持续负载或复杂场景 41 变体/68 实例。功能测试应证明一个相对稳定的外部契约，短流量、单一触发、明确终态；场景测试应有多阶段流量、容量变化/故障、窗口观测和恢复结论。下文按分类、每类内由简单到复杂排列。

建议标签含义：**保留**＝可作为独立契约；**删除独立例**＝删除此独立 E2E 行，若其断言仍有价值则并入同族主例或更低层单测；**转场景**＝把此行为并入可扩展的多阶段场景；**重设计**＝目标有价值，但现有小规模、时序、阈值或观测不足，不能直接作为稳定信号。建议不是立即删除代码的清单；迁移后需核对覆盖台账，再删除旧入口。每一行的“怎么测/期望”描述现有测试意图，必要时在“建议”指出它不应继续按当前口径验收。

## 1. 请求闭环与状态协议：先证明能完成，再看异常状态

这一类要验证请求从接收、P/D 执行到客户端终态，以及引擎状态上报乱序、遗漏、重复时 master 的资源账不泄漏。稳定的客户端终态适合功能测试；跨 TTL、重启和持续状态缺口的恢复过程应转场景。状态协议大量“一个字段一个例”的变体应合并，否则太依赖当前内部报文。

| 测试（当前类型） | 目的（大白话） | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `request_completion::immediate`（F） | 普通请求能走完。 | 发短请求并正常收取输出。 | 成功终态、资源清零。 | 保留，核心烟测。 |
| `request_completion::deferred_fetch`（F） | 晚些取输出也不会丢请求。 | 先提交，延后 Fetch。 | 仍能拿到正确终态且不泄漏。 | 保留；明确 Fetch 开关口径。 |
| `request_completion::client_no_fetch`（F） | 压测不 Fetch 模式仍能完成 P→D。 | 只提交、不取流，读引擎与 master 状态。 | 引擎完成、无 Fetch RPC、资源归零。 | 保留，独立于客户端成功率。 |
| `observed_terminal_cohort::deferred`（F） | 延迟收割时观察证据不能凭空消失。 | 先提交一批，冻结窗口快照，再收割。 | 客户端终态与引擎快照可对应。 | 删除独立例；并入 `deferred_fetch` 的观测校验。 |
| `status_protocol::unknown_rid_finished`（F） | 陌生请求的完成消息不能改账。 | 注入未知 RID 的 finished 状态。 | 真请求和负载不受影响。 | 删除独立例；并入“非法状态报文”表驱动契约。 |
| `status_protocol::unknown_rid_running`（F） | 陌生 running 消息不能造在飞请求。 | 注入未知 RID 的 running 状态。 | 不增长资源账，清理后为空。 | 删除独立例；与上一项合并。 |
| `status_protocol::unknown_batchid`（F） | 陌生批次号不影响真批次。 | 真请求执行时上报未知 batch ID。 | 真请求成功、批次账不混。 | 删除独立例；放入状态协议单测/表驱动例。 |
| `status_protocol::foreign_batchid`（F） | 别人的批次号不能串账。 | 将请求状态挂到外来 batch ID。 | 原批次仍完成，外来状态被忽略。 | 删除独立例；与未知 batch 合并。 |
| `status_protocol::special_ids`（F） | 特殊 ID 不能污染请求表。 | 注入负值/特殊请求标识。 | 不出现幽灵请求，正常请求可完成。 | 删除独立例；移到协议边界单测。 |
| `status_protocol::unbatched_single_request`（F） | 单请求的缺省批次信息可处理。 | 上报缺省或单请求状态。 | 不误扣/漏记，终态正确。 | 保留一个跨模式契约，去掉报文字段细节断言。 |
| `status_protocol::duplicate_finished`（F） | 完成消息重复也只能结算一次。 | 正常完成后重放 finished。 | 不重复释放、不生成负负载。 | 保留，幂等核心契约。 |
| `status_protocol::cursor_regress`（F） | 旧游标重放不能让状态倒退。 | 完成后回退状态游标重发。 | 已完成仍完成，账不复活。 | 保留；与版本回退合并为一个异常状态矩阵。 |
| `status_protocol::finished_then_running`（F） | 完成后又报 running 不应复活。 | 先完成，再注入旧 running。 | 终态单调、无幽灵在飞。 | 删除独立例；并入游标/版本回退矩阵。 |
| `status_protocol::version_regress`（F） | 旧版本状态不能覆盖新状态。 | 活跃请求期间注入版本倒退。 | 所有权最终收敛，不回滚账。 | 保留；做异常状态矩阵的主例。 |
| `status_protocol::zombie_completed_running`（F） | 已完成请求的残留 running 不能拖住系统。 | 完成后让 D 继续报 running。 | 请求已终结、资源最终清空。 | 转场景；与长期状态缺口合并。 |
| `status_protocol::zombie_fake_running`（F） | 虚假 running 不能无限占容量。 | 空闲时持续报不存在的 running。 | 占用有界、健康请求可恢复。 | 转场景；观察窗口与容量恢复。 |
| `status_protocol::decode_before_prefill`（F） | D 先完成时也不能卡住 P 的结算。 | 屏蔽 P 终态，让 D 先上报完成。 | 请求终结，P 所有权及时释放。 | 保留，跨 P/D 乱序契约。 |
| `status_protocol::decode_running_before_prefill`（F） | D 只报 running 不等于请求完成。 | 压住 P 终态，先送 D running。 | 中间态不提前结算，最终能收敛。 | 删除独立例；并入上一项的状态序列。 |
| `status_protocol::decode_waiting_before_prefill`（F） | D waiting 也不能误当完成。 | P/D 终态暂缓，仅送 D waiting。 | 请求保持在飞，补齐终态后清理。 | 删除独立例；与 running 顺序例合并。 |
| `status_protocol::prefill_suppress_finished`（F） | P 完成消息暂时丢了仍应可收敛。 | 跑请求，屏蔽 P finished 后恢复。 | 客户端合法终态，P 批次账最终清空。 | 转场景；与 `decode_suppress_finished` 构成双侧缺口。 |
| `status_protocol::decode_suppress_finished`（F） | D 完成消息暂时丢了仍应可收敛。 | 跑请求，屏蔽 D finished 后恢复。 | P 可独立释放，D 恢复后清空。 | 转场景；与上一项合并。 |
| `status_protocol::prefill_suppress_all`（F） | P 所有状态都断档时不要无限挂账。 | 批量流量中屏蔽 P 状态。 | 请求有合法终态，master 最终退账。 | 转场景；加持续流量与恢复窗。 |
| `status_protocol::no_respond`（F） | 引擎长期不回应时最终要退出调度。 | 在飞请求后让状态/RPC 不响应。 | 有界超时与退场、恢复后可服务。 | 转场景；将真实 10 秒保留期计入预期。 |
| `status_protocol::inflight_ttl_cleanup`（F） | 在飞请求超 TTL 后账不能泄漏。 | 造慢请求并屏蔽 P/D 上报，等待 TTL。 | 旧所有权释放，新请求恢复。 | 转场景；时间窗太长，不适合快功能回归。 |
| `status_protocol::fetch_error`（F） | Fetch 失败能透传并恢复。 | 注入 Fetch 故障，再发健康请求。 | 错误可见，新请求成功，资源清零。 | 保留；只在真实 Fetch 模式执行。 |
| `status_protocol::debug_snapshot`（F） | 调试快照能看到已终结请求的记录。 | 完成请求后查询 debug/tombstone。 | 可查询且不占资源。 | 删除独立例；属于调试接口自测，不是调度契约。 |
| `status_protocol::normal_no_fetch`（F） | 不 Fetch 时引擎自主接续并结算。 | schedule-only 发请求并看 P/D 状态。 | 完成、无 Fetch、无残账。 | 删除独立例；与 `client_no_fetch` 重复，保留更端到端者。 |
| `late_completion::after_missing`（F） | 漏一次状态后晚到终态仍有效。 | 故意制造缺报，再补终态。 | 旧账退休，不影响后续请求。 | 删除独立例；并入 P/D 状态缺口场景。 |
| `batch_ack_and_execution::ack_partial`（F） | 批次 ACK 部分失败时只退失败成员。 | 一批请求注入部分 ACK 失败。 | 成功成员继续，失败成员及时释放。 | 保留，批处理核心边界。 |
| `batch_ack_and_execution::execution_partial`（F） | ACK 成功后执行中部分失败不能整批误杀。 | 批次执行中让个别成员失败。 | 各成员独立终态、账最终清零。 | 保留；与 ACK 部分失败并列。 |
| `batch_ack_and_execution::ack_multi_error`（F） | 多种引擎拒绝码要按原样归类。 | 分别注入 8431/8510 等拒绝。 | 错误码透传且健康请求仍能成功。 | 删除独立例；收进 ACK 错误表驱动校验。 |
| `batch_ack_and_execution::ack_drop`（F） | ACK 丢失时不应重复派发或永久占位。 | 丢 ACK，观察 fence/TTL，随后恢复。 | 未知结果有界收敛、无重复执行。 | 转场景；含时间窗和恢复阶段。 |

## 2. 准入与排队：容量真的满时，拒绝、等待和恢复各有边界

这类要区分 master 排队、P/D 并发、KV 水位与 batch token 上限，不能把任意一个“满”都当成同一种错误。单个硬门槛可保留功能契约；持续背压、恢复与多门槛相互作用应扩展成场景。当前若只用 1P/1D 或几块缓存触发，应重新验算硬容量、水位及请求需求。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `priority_admission::permit_released_without_preemption`（F） | D 已运行的旧请求不再占 P 准入许可。 | 让低优先级请求进入 D，再送高优先级请求。 | 两者可并行完成，旧请求不被误抢占。 | 保留，明确许可释放时点。 |
| `engine_admission_gate::prefill_concurrency`（F） | P 并发上限生效且可恢复。 | 放慢 P、同时发超过上限的请求。 | 执行数有界，排队者最终完成。 | 保留；去掉脆弱的瞬时 park 断言。 |
| `engine_admission_gate::decode_hard_gate`（F） | D 的硬并发门槛不能被绕过。 | 放慢 D 并批量发请求。 | 准入受限，释放后恢复。 | 重设计；与 D KV 容量分开，规模/压力要能稳定触门槛。 |
| `engine_admission_gate::prefill_waiting_cap`（F） | P waiting 上限满后应明确背压。 | 占满 waiting，再送探针和恢复请求。 | 探针快速且正确地拒绝，放位后成功。 | 重设计；避免只匹配易变错误文本，验证状态与错误族。 |
| `engine_admission_gate::kv_pool_capacity`（F） | P KV 池满时不能超分配。 | 小池承载占用者，再送溢出请求。 | 容量拒绝或等待符合合同，释放后准入。 | 重设计；17 块小池需按真实水位与块大小复算。 |
| `admission_queue::queue_depth`（F） | 排队深度满时应快速拒绝并恢复。 | 慢 P 占位，填满队列，再送探针。 | 明确背压、占位者完成、后续成功。 | 保留；错误语义用类型/码，不锁文本。 |
| `admission_queue::slo_deadline`（F） | 等待过期限应结束，不能无限挂起。 | KV 压力下给请求短期限，随后解除压力。 | 期限内有界失败，恢复请求成功。 | 保留；阈值用宽容区间。 |
| `batcher_placement_admission::batcher_queue_capacity_park`（F） | 批组装等待时容量限制不能丢请求。 | 慢 P 后发一波，观察 admission 与结束。 | 全部有终态并恢复。 | 删除独立例；与 P 并发/队列满测试重合，策略性 park 不应硬断言。 |
| `batcher_placement_admission::batcher_queue_deadline`（F） | 组批等待中到期应按期限退出。 | 慢 P、短 queue timeout、一波请求。 | 到期者正确错误族，幸存者完成。 | 删除独立例；并入 `slo_deadline` 的组批分支。 |
| `batcher_placement_admission::placement_pool_wait`（F） | 已分配位置的请求可等资源而不串账。 | 限 P 在飞数，先占位再送后一请求。 | 后者等前者释放后完成，账归零。 | 保留；以终态/资源账验收，不测内部 park 名称。 |
| `prefill_batch_token_budget::split`（F） | 超过 P batch token 预算会拆批。 | 设置 1024 token 上限并同时送请求。 | 各成员完成，实际批次不超预算。 | 保留，组批容量核心。 |
| `prefill_batch_token_budget::split_fifo`（F） | 拆批不破坏同级先后顺序。 | 拆批条件下按顺序发多请求。 | 客户端顺序和批次结算一致。 | 保留；可与 `split` 合为一个参数化例。 |
| `prefill_batch_token_budget::boundary`（F） | 正好在上限附近不能错拆或超装。 | 设置 2048 token 边界，对照基线再发波次。 | 批次形状正确，完成和 TTFT 不异常。 | 保留一个边界参数，别单独维护整套编舞。 |
| `prefill_batch_token_budget::regroup_disabled`（F） | 关闭 token 组批限制不应仍按旧值拆。 | 将上限设 0，再发同样请求。 | 批次配置生效，成员全完成。 | 删除独立例；并入 token 预算的配置轴。 |

## 3. 取消与 fence：一次取消必须落在正确请求、正确引擎和正确终态

这一类的共同思想是取消只影响目标请求；客户端、master、P/D 和 fence 对同一终态达成一致，重复/晚到消息不造成第二次副作用。基础幂等是功能；跨重启、断流、故障注入和等待窗口是场景。`batch`/`nonbatch` 配对只是执行轴，不应复制两套断言实现。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `cancel_lifecycle::basic_batch`（F） | 批派发后取消正在输出的请求。 | 等首输出，再从 master 取消。 | 目标终止、占用释放。 | 保留，基础取消合同。 |
| `cancel_lifecycle::basic_nonbatch`（F） | 非批派发也能取消。 | 同样等首输出，经 worker 取消路径。 | 终止和资源释放与批模式一致。 | 保留；作为同一例的模式轴。 |
| `cancel_lifecycle::idempotent_batch`（F） | 取消两次不能多做一次。 | 同一 RID 连续取消，读计数和终态。 | 引擎取消至多一次，终态不反复。 | 保留；合入基础取消的第二阶段。 |
| `cancel_lifecycle::idempotent_nonbatch`（F） | 非批模式重复取消也幂等。 | 同一 RID 经 worker 路径取消两次。 | 无重复副作用。 | 删除独立例；由统一幂等程序覆盖模式轴。 |
| `cancel_lifecycle::after_terminal_batch`（F） | 完成后才取消不能重开请求。 | 等成功终态后发送取消。 | 保持成功、无新占用。 | 删除独立例；并入基础取消的终态后步骤。 |
| `cancel_lifecycle::after_terminal_nonbatch`（F） | 非批完成后迟到取消也无害。 | 等终态后经 worker 路径取消。 | 终态不变、无漏账。 | 删除独立例；统一 late-cancel 参数。 |
| `cancel_lifecycle::unknown_rid`（F） | 取消不存在的请求要有明确答复。 | 对陌生 RID 发取消。 | typed not-found，系统保持健康。 | 删除独立例；放到取消 API 单测。 |
| `cancel_lifecycle::anomaly_path_batch`（F） | 异常取消路径不能把请求挂住。 | 首输出后触发非正常取消分支。 | 目标最终结束，后续请求成功。 | 重设计；先明确具体故障类型，避免“异常路径”宽泛口径。 |
| `cancel_lifecycle::anomaly_path_nonbatch`（F） | 非批异常路径同样收敛。 | 在 worker 取消链路触发异常。 | 有界终态、资源清零。 | 删除独立例；并入清晰定义后的异常场景模式轴。 |
| `cancel_lifecycle::sibling_isolation_batch`（F） | 取消一条不能伤及同批兄弟。 | A/B/C 同批并发，只取消 B。 | A/C 正常完成，B 终止。 | 保留，局部性核心合同。 |
| `cancel_lifecycle::sibling_isolation_nonbatch`（F） | 非批并发取消也要隔离兄弟。 | A/B/C 同时运行，只取消 B。 | A/C 不受影响。 | 删除独立例；作为同一局部性例的模式轴。 |
| `cancel_lifecycle::phase_timing_batch`（F） | P 阶段取消和 D 阶段取消不能混账。 | 在不同生命周期阶段分别取消。 | 各阶段都正确终结、释放对应资源。 | 保留；改为阶段表驱动。 |
| `cancel_lifecycle::phase_timing_nonbatch`（F） | 非批的 P/D 分阶段取消正确。 | 同样在 P/D 阶段取消 worker 请求。 | 终态与资源账正确。 | 删除独立例；与阶段表驱动合并。 |
| `cancel_lifecycle::deadline_exempt_inflight_batch`（F） | 已进引擎的请求不能被排队期限误杀。 | 短 queue timeout、慢 enqueue，等待完成。 | 在飞请求完成，不收到错误取消。 | 保留，期限和在飞边界。 |
| `cancel_lifecycle::deadline_exempt_inflight_nonbatch`（F） | 非批在飞同样不受排队 TTL 误伤。 | 相同条件走非批派发。 | 不误取消。 | 删除独立例；保留模式轴。 |
| `cancel_lifecycle::schedule_drop_delivered`（F） | Schedule 调用丢连接时不应丢失已投递请求。 | 请求已送到引擎时断开客户端/调度连接。 | 所有权有界结算、无重复投递。 | 转场景；需要请求级链路证据和重试窗口。 |
| `cancel_lifecycle::stream_break_prefill_autonomous`（F） | 客户端流断后 P 自主取消。 | 首输出后断流，查 P 收据和排空。 | P 停止、D 不遗留。 | 转场景；与 D 断流组成 P/D 故障矩阵。 |
| `cancel_lifecycle::stream_break_decode_autonomous`（F） | 客户端流断后 D 自主取消。 | D 输出期间断流，查 D 收据和排空。 | D 停止、请求终结。 | 转场景；与上一项合并。 |
| `cancel_lifecycle::preemption_victim_batch`（F） | 被抢占者走取消链路而非静默消失。 | 低优先级 D 运行后送高优先级请求。 | victim 有明确终态，幸存者完成。 | 转场景；归入抢占剧本，避免两族重复编舞。 |
| `cancel_lifecycle::preemption_victim_nonbatch`（F） | 非批抢占者取消同样可见。 | 同上，经非批取消路径。 | 终态/所有权一致。 | 删除独立例；作为抢占场景的模式轴。 |
| `cancel_fence_settlement::engine_notfound_settle_batch`（F） | 引擎已完成后取消返回 not-found 也要结算。 | 完成后对原 P 发取消。 | master 不因 not-found 留 fence。 | 保留；与普通 after-terminal 例合并断言。 |
| `cancel_fence_settlement::engine_notfound_settle_nonbatch`（F） | 非批 not-found 也能结算。 | 完成后经 worker 和原 P 取消。 | 不挂 fence。 | 删除独立例；保留模式轴。 |
| `cancel_fence_settlement::prefill_dead_await_terminal`（F） | P 死亡时取消不能永远等它答复。 | 首输出后停 P，再取消。 | 客户端有终态，fence 最终关闭。 | 转场景；纳入摘机/重启窗口。 |
| `cancel_fence_settlement::decode_retire_closes_fence`（F） | P/D 相继退场时 fence 能关闭。 | 取消后依次停 P、D。 | 终态唯一、fence 不泄漏。 | 转场景；合入故障恢复场景。 |
| `cancel_fence_settlement::transport_failure_one_shot`（F） | 取消 RPC 传输失败不能无限重试。 | 注入一次 cancel 传输故障。 | 有界终态、恢复请求成功。 | 保留，独立传输错误合同。 |
| `cancel_fence_settlement::unexpected_status_await_terminal`（F） | 意外取消状态仍须等待明确终态。 | 注入异常 status，观察终态窗口。 | 不提前假成功，也不永久占位。 | 重设计；先定义允许的真实状态族。 |
| `cancel_fence_settlement::engine_restarted_tombstoned_settle`（F） | 引擎重启后旧请求墓碑可结算。 | 在飞请求触发 P 崩溃、重启后取消。 | 旧 RID 不复活，fence 关闭。 | 转场景；与重启/代际测试合并。 |
| `cancel_fence_settlement::fencing_lost_on_engine_restart`（F） | 重启使旧 fence 丢失也不能重复执行。 | 故意丢 fence 后重启并重试旧请求。 | 旧所有权最终结算且不双执行。 | 转场景；需要跨代际请求证据。 |

## 4. 优先级与抢占：顺序应正确，抢占必须确实可行

这一类要把“按优先级排队”和“为高优先级驱逐低优先级”分开。前者可用小拓扑证明稳定排序；后者需要可解的容量算术、被驱逐者真实处于允许阶段、8400 和受益者成功同时出现。先验“应该抢占”却没制造唯一可行解的例子要重设计。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `priority_queue::same_level_fifo`（F） | 同优先级按到达顺序处理。 | 慢 P 占位后同级并发入队。 | FIFO 顺序和最终终态正确。 | 保留。 |
| `priority_queue::order_basic`（F） | 高优先级先于低优先级出队。 | 先造占位，再混合优先级发波次。 | 优先级顺序成立、无丢失。 | 保留，排序主例。 |
| `priority_queue::queue_timeout_terminal`（F） | 排队超时有明确终态。 | 占位后让队列请求过期。 | 超时错误可见、占用清零。 | 删除独立例；并入 admission 的 `slo_deadline`。 |
| `priority_queue::normalize_default50`（F） | 缺省优先级按配置默认值处理。 | 发未指定优先级请求。 | 排序采用默认值。 | 删除独立例；配置解析单测足够。 |
| `priority_queue::normalize_default30`（F） | 改默认值后归一化仍生效。 | 设默认 30，再送有/无优先级请求。 | 相对顺序正确。 | 删除独立例；与默认值参数化单测合并。 |
| `priority_queue::normalize_channels`（F） | 多输入通道的优先级口径一致。 | 混合明确/缺省优先级渠道。 | 归一化后排序一致。 | 保留一个外部渠道契约。 |
| `priority_queue::normalize_metrics`（F） | 归一化指标与请求一致。 | 送不同优先级并读计数。 | 指标分桶正确。 | 删除独立例；移到指标映射单测。 |
| `priority_queue::low_no_starvation`（F） | 高优先级流量不能一直饿死低优先级。 | 两轮高低优先级波次。 | 低优先级最终完成。 | 转场景；需持续到达与等待分布，不宜两轮定性。 |
| `priority_preemption::same_priority_zero_eviction`（F） | 同级请求不能互相抢占。 | 塞满位置后送同级请求。 | 零驱逐、两者按容量完成。 | 保留，抢占负例。 |
| `priority_preemption::config_strict_reject`（F） | 禁用/不适用的抢占配置不应悄悄生效。 | 用严格配置启动并检查行为。 | 不发生非法驱逐。 | 删除独立例；配置校验下沉到配置生成单测。 |
| `priority_preemption::prefill_queued`（F） | 高优先级到达时可踢 P 队列里的低优先级者。 | 慢 P、占满等待位，确认 victim queued 后送高优先级。 | victim 8400，高优先级成功。 | 重设计；必须证明到达瞬间只有驱逐能腾位。 |
| `priority_preemption::timeout_attribution`（F） | 抢占与自然超时要区分。 | 排队占位、期限与高优先级同时触发。 | 终态原因和计数归因正确。 | 转场景；需多轮边界时间窗，而非固定毫秒编舞。 |
| `priority_preemption::comparator_frozen_weak`（F） | 冻结/弱比较器不应乱踢对象。 | 固定候选队列及比较条件。 | 选对 victim 或拒绝抢占。 | 重设计；过度绑定内部 comparator，改外部受益/损失合同。 |
| `priority_preemption::decode_engine_owned`（F） | D 已持有请求时抢占仍按允许阶段执行。 | 让 D 占容量，再送高优先级。 | 被允许 victim 有 8400，incoming 成功。 | 重设计；4D 等布局先证明驱逐可解、再看行为。 |
| `priority_preemption::decode_reservation_priority`（F） | D 预留阶段优先级抢占正确。 | 占用 D 预留位后送高优先级。 | 预留 victim 取消、incoming 准入。 | 重设计；与 live reserved 合并，避免重复小拓扑。 |
| `priority_preemption::decode_reserved_live_window`（F） | 活的 D 预留位可被更高优先级接替。 | 先排空 placeholder，保留 victim 1 块，再送需 4 块的 incoming。 | victim 8400、incoming 200、decode_reserved 指标一致。 | 保留为算术锚点；5 块池和 1 块水位需锁定并验证时序。 |
| `priority_preemption::observability_integrity`（F） | 驱逐指标应与真正的 victim 对上。 | 完成一轮抢占并查事件、计数、日志。 | 同一 victim 的终态和指标一致。 | 删除独立例；作为抢占主例的证据校验。 |
| `priority_preemption::cancel_not_found`（F） | 抢占时目标已结束，not-found 不应卡住新请求。 | 冻结状态、让 victim 先结束，再送 incoming。 | 旧请求不重复取消，新请求最终结束。 | 转场景；属于晚到取消竞态。 |
| `priority_preemption::cancel_absent_fence`（F） | fence 不存在时取消也能结算。 | 先完成/清场，再触发取消。 | 无残余所有权。 | 删除独立例；并入取消 fence 矩阵。 |
| `priority_preemption::cancel_tombstoned`（F） | 重启后 tombstone 不能让抢占卡住。 | victim 执行时 P 崩溃重启，再来高优先级。 | victim 终态明确、incoming 可恢复。 | 转场景；与引擎重启取消合并。 |

## 5. 引擎故障与恢复：身份、租约和请求不能互相冒名

核心是区分短暂状态缺口、真正崩溃、重启后的新 generation 与旧请求；故障期间允许既定的有界失败，但不能泄漏所有权或把旧缓存当成新引擎缓存。一次 RPC 错误可做功能；多个阶段、代际、恢复窗口属于场景。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `engine_rpc_fault::enqueue_error`（F） | P 接收请求报错可被清楚传给客户端。 | 注入 Enqueue 错误，再发健康请求。 | 错误归类正确、恢复请求成功。 | 保留，RPC 基本合同。 |
| `engine_rpc_fault::enqueue_delay_batch`（F） | Enqueue 慢不会被错认成功。 | 先测基线，再注入 Enqueue 延迟。 | 延迟/超时有界、后续恢复。 | 删除独立例；与其他 RPC 延迟合并成参数矩阵。 |
| `engine_rpc_fault::generate_delay_batch`（F） | 批模式 Generate 慢能正常处理。 | 注入 Generate 延迟并做基线对照。 | 请求结果与延时边界一致。 | 删除独立例；并入 RPC 延迟矩阵。 |
| `engine_rpc_fault::generate_delay_nonbatch`（F） | 非批 Generate 慢也按同样口径处理。 | 非批路径注入相同延迟。 | 终态正确，不漏账。 | 删除独立例；以模式轴覆盖。 |
| `engine_fault_recovery::generation_bump`（F） | 真重启后引擎代际必须变化。 | 完成基线，重启目标引擎，再发请求。 | 新 generation 可发现，旧资源账清零。 | 保留，代际主合同。 |
| `engine_fault_recovery::status_gap_no_bump`（F） | 暂停上报不等于重启。 | 短暂制造状态缺口但不重启。 | generation 不变，恢复流量正常。 | 保留，和上一项成正反对照。 |
| `engine_fault_recovery::kv_usage_reset`（F） | 引擎 KV 使用量清零后 master 不应保留旧占用。 | 清缓存、同步状态，再发流量。 | 可用量刷新，目标继续接请求。 | 保留；以真实容量单位和快照验证。 |
| `engine_fault_recovery::kv_resync`（F） | 重启后旧缓存地址簿不能当成仍命中。 | 先种缓存、重启或清空，再请求同前缀。 | 旧 holder 失效，新命中由实际缓存决定。 | 转场景；加前后多个窗口和不同前缀。 |
| `engine_fault_recovery::down_phases`（F） | 引擎下线各阶段的路由行为可解释。 | 基线、停机、保留期、摘除、恢复逐段采样。 | 保留期失败按真实策略计，摘除后不再派死机。 | 转场景；不能硬断言瞬时摘除零失败。 |
| `engine_fault_recovery::crash_after`（F） | 接受请求后崩溃不会留下永生租约。 | 指定 Enqueue 后崩溃，观察残留与重启。 | 请求有终态、旧引擎租约最终清空。 | 转场景；要求请求级落点和代际证据。 |
| `engine_fault_recovery::no_resurrect`（F） | 崩溃前请求不能在重启后复活。 | 慢请求在飞时崩溃并恢复引擎。 | 旧请求不再执行，新请求成功。 | 转场景；先证明崩溃注入真的触发。 |
| `engine_fault_recovery::status_gap_long_retire`（F） | 长状态缺口最终应退出路由。 | 在飞流量中长时间屏蔽上报。 | 经保留期后退休、恢复请求成功。 | 转场景；使用真实 10 秒 grace 的时间轴。 |
| `engine_fault_recovery::flap`（S） | 短时间反复上下线仍能收敛。 | 持续流量下多次停/启同一引擎。 | 各窗错误有界，最终发现/资源账收敛。 | 保留场景；扩大 P/D 和波次做稳定性校准。 |

## 6. KV 缓存与亲和：先验清“存在哪”，再谈 master 选得对不对

这类要分开三件事：引擎侧块是否保存/驱逐、状态上报形成的 holder 集是否正确、master 在候选容量和前缀命中之间如何选择。单个 LRU/容量边界可做功能，但热点、跨机复制、全局风暴、恢复速度必须是窗口化场景。尤其不能用“路由理论命中率”代替落点引擎的实际 device/memory reuse。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `cache_churn::lru_affinity`（F） | 热块被重用，冷块先淘汰。 | 4 块小池种数据、重放、加压力。 | 热块保留、冷块驱逐，路由可重试。 | 重设计；先用引擎缓存单测定 LRU，再以真实量级做 E2E。 |
| `cache_churn::referenced_occupancy`（F） | 正在使用的块不能被淘汰。 | 4 块 P 池种数据并 pin，在压力下取快照。 | 引用期间不驱逐，释放后可淘汰。 | 重设计；device/memory pin 与水位须与真实引擎一致。 |
| `cache_local_index::evict_batch`（F） | 本机块被驱逐后命中索引要删。 | 种前缀、逐出目标块，再发同前缀。 | 不再宣称旧命中，路由重新分散。 | 保留；16 块仅作协议边界，不当线上容量模型。 |
| `cache_local_index::evict_nonbatch`（F） | 非批派发下同样去掉旧索引。 | 同样驱逐并走非批请求。 | holder 状态一致。 | 删除独立例；以模式轴运行同一程序。 |
| `cache_capacity_recovery::decode_pool_exhaustion_terminal`（F） | D KV 池完全放不下要终结而非死等。 | 1D、3 块池先占位再发大请求。 | 明确容量错误、释放后恢复。 | 重设计；小池水位取整会改变准入算术。 |
| `cache_capacity_recovery::decode_capacity_park`（F） | D 暂时没空间时可以等资源。 | 多 D 制造压力，再释放容量。 | 请求不被误判永久失败，最终有终态。 | 重设计；不把内部 park 计数当功能断言。 |
| `cache_local_index::continuity_batch`（F） | 同一前缀跨请求延续时仍选对 holder。 | 两机种共享前缀，再连发续写。 | 连续前缀命中和路由一致。 | 转场景；连续多波请求有明显负载特征。 |
| `cache_local_index::continuity_nonbatch`（F） | 非批延续路由也正确。 | 同样种子和续写，改派发模式。 | holder 连续性成立。 | 删除独立例；作为场景模式轴。 |
| `cache_local_index::isolation_batch`（F） | 不相关家族不能混入同一缓存索引。 | 种 A/B 两家族与 filler，交错续写。 | A/B 命中隔离，B 不误路由。 | 转场景；用多窗口背景流量。 |
| `cache_local_index::isolation_nonbatch`（F） | 非批下 A/B 索引仍隔离。 | 同样交错流量走非批派发。 | 各家族命中独立。 | 删除独立例；以场景模式轴覆盖。 |
| `cache_global_holders::shared_batch`（F） | 多机共享同一前缀时 holder 集应含所有持有者。 | 在不同 P 种同一前缀，继续发请求。 | holder 并集完整、落点合理。 | 转场景；共享缓存要有多轮演化。 |
| `cache_global_holders::shared_nonbatch`（F） | 非批共享前缀 holder 集也完整。 | 同样种缓存，改非批。 | 不漏 holder。 | 删除独立例；并入共享 holder 场景轴。 |
| `cache_global_holders::release_batch`（F） | 缓存释放后要从全局 holder 集移除。 | 种共享前缀、释放一台缓存、继续发。 | 无幽灵 holder，幸存者可承接。 | 转场景；加入驱逐和再填充窗口。 |
| `cache_global_holders::release_nonbatch`（F） | 非批缓存释放也不能留幽灵 holder。 | 同样释放/续发。 | holder 集及时更新。 | 删除独立例；并入 release 场景轴。 |
| `cache_global_holders::redirect_batch`（F） | 旧 holder 不可用时流量应转到新 holder。 | 种缓存、改变可用性、连发续写。 | 请求转移且最终命中重建。 | 转场景；观测转移速度而非一个落点。 |
| `cache_global_holders::redirect_nonbatch`（F） | 非批也能重定向缓存流量。 | 同样变更 holder 并续写。 | 重新收敛。 | 删除独立例；作为场景模式轴。 |
| `cache_global_holders::down_batch`（F） | holder 下线后幸存缓存仍可用。 | 3P 中一台下线，继续同前缀流量。 | grace 后不再选死机，幸存者命中。 | 转场景；把摘机保留期与失败窗说清。 |
| `cache_global_holders::down_nonbatch`（F） | 非批 holder 下线同样收敛。 | 相同下线过程、非批流量。 | 幸存者恢复。 | 删除独立例；并入 down 场景模式轴。 |
| `cache_global_holders::mixed_batch`（F） | 不同前缀家族各找各的 holder。 | 先分别种缓存，交错续写。 | 两族命中不串、路由可解释。 | 转场景；可与 local isolation 聚合。 |
| `cache_global_holders::mixed_nonbatch`（F） | 非批混合家族也不串缓存。 | 相同混合流量改非批派发。 | 家族隔离。 | 删除独立例；作为聚合场景模式轴。 |
| `cache_capacity_recovery::pool_saturation_evict_reject_recover`（F） | P 池饱和后驱逐、拒绝、恢复链条要闭环。 | 连续种缓存、加压、送溢出请求、再释放。 | 驱逐与拒绝有据，恢复请求成功。 | 转场景；27 块小池的数值需重新标定。 |
| `cache_capacity_recovery::capacity_conflict_overflow`（F） | 缓存亲和不能压过真实剩余容量。 | 种热点 holder、降低其容量、分波送请求。 | 无不可能的准入，其他 P 分担。 | 转场景；增加 P 数和窗口级落点/成功率。 |
| `cache_affinity::prefix_batch`（F） | 命中前缀时优先走缓存 holder。 | 两 P 种前缀，混入无关请求后重复发送。 | 前缀亲和与空闲分流兼顾。 | 转场景；30 次续发已是负载过程。 |
| `cache_affinity::prefix_nonbatch`（F） | 非批也应按前缀命中选路。 | 同样前缀/背景流量走非批。 | 亲和与分流一致。 | 删除独立例；作为同一场景模式轴。 |
| `cache_affinity::mixed_tiers`（S） | 全命中、半命中、零命中要有可解释的分层路由。 | 分别种三档前缀并持续发。 | 全命中最集中，零命中可扩散。 | 保留场景；补落点实际 reuse 分层。 |
| `cache_affinity::hot_tension`（S） | 热点亲和与全局公平之间有边界。 | 种热点、混合热点/普通请求持续发。 | 热点尽量命中，冷请求不过度饥饿。 | 保留场景；阈值按流量和容量校准。 |
| `cache_churn::hot_churn`（S） | 多家族轮换时命中和 holder 不应无界抖动。 | 四家族跨十个窗口轮流访问。 | 驱逐/复制/命中处于预定 band。 | 保留场景；记录独立校准样本。 |
| `cache_affinity::leader_spill_batch`（S） | 2P 热点 P 饱和时是否触发缓存踩踏。 | 小池中制造热点、filler、慢 P、多阶段观察。 | 基线命中、风暴/恢复与复制份数可判读。 | 重设计；12 块旧构造和指标阈值与新版 P 轴重复且易失真。 |
| `cache_affinity::leader_spill_nonbatch`（S） | 非批 2P 热点溢出形态。 | 同样热点/慢 P 改非批派发。 | 路由、命中、恢复可解释。 | 重设计；移作 P 轴的派发模式参数。 |
| `cache_affinity::leader_spill_p2`（S） | 2P 风暴复现锚点。 | 1 热 P + 1 接收 P、快节奏请求、解除慢速。 | 零失败，基线/崩塌/恢复都有窗口证据。 | 保留场景；按拓扑独立校准 band。 |
| `cache_affinity::leader_spill_p3`（S） | 多一个接收 P 后风暴是否更深。 | 同一剧本改为 3P，逐窗比落点和复制。 | 解释扩散机器数与恢复窗数。 | 保留场景；与 P2 同源配置，不能沿用阈值。 |
| `cache_affinity::leader_spill_p4`（S） | 4P 扇出是否进一步放大风暴。 | 4P 重复同样阶段并独立记录。 | 可量化最低命中率/扩散/恢复。 | 保留场景；P 系列结论才是交付物。 |

## 7. 负载均衡：单点正确性与分布曲线分开

这一类不是要求每台机器每时刻都均分，而是用落点、请求长度、D 压力与延迟证明策略在已声明条件下合理。一次路由边界是功能；持续混合流量和阶段性偏斜是场景。过小 P/D 或几笔请求得出的“均衡率”不具有可推广性。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `balance_distribution::uniform_serial`（F） | 空闲机器之间不应永久只用一台。 | 逐笔串行发相似请求，记录 P/D 落点。 | 多个健康候选都能被使用。 | 删除独立例；太简单，合入分布场景的基线窗。 |
| `balance_distribution::concurrent_mix`（F） | 并发混合请求能分散到 P/D。 | 短波次并发、按落点统计。 | 无单点异常聚集、全部有终态。 | 转场景；单波样本少，补稳定窗口。 |
| `balance_distribution::decode_spread`（F） | D 从少到多时分布会随容量改变。 | 对比 10/50 请求的 D 落点。 | 健康 D 参与，分布差异可解释。 | 重设计；固定 10/50 笔与真实规模不相称。 |
| `balance_distribution::length_mixed`（F） | 长短请求混合不应只看请求数。 | 多轮长/短请求与 P/D 落点、token 份额对照。 | token 负担和延迟有界，短请求不饿死。 | 转场景；本来就是五轮负载。 |
| `balance_distribution::sustained_mix`（S） | 持续混合负载下整体均衡与恢复。 | 4P/8D、多窗口发流量、停止后排空。 | 成功、延迟、P/D 份额处于经校准 band。 | 重设计；拓扑和流量先与目标部署规模/单位对齐。 |
| `balance_overload_transfer::decode_pressure`（F） | D 过载时流量能转到其他 D。 | 注入 D 压力，前后比较请求落点。 | 压力机份额下降、其他 D 接住、终态正确。 | 转场景；按压力/恢复窗口看转移曲线。 |
| `balance_overload_transfer::prefill_pressure`（F） | P 变慢时新请求能重新分布。 | 先测基线、放慢一个 P、发波次。 | 高负载 P 少接，恢复后重新参与。 | 转场景；真实 P 时延模型和缓存代价一起观察。 |

## 8. Master 生命周期与客户端故障转移：端点变化不能制造重复请求

此类要区别单 master 的冷启动/重启、双 master 的请求级 failover，以及客户端对业务错误不应误重试。单次配置或短 burst 不足以代表高可用；必须有持续负载、请求唯一性、故障前后窗口和两台 master 的落点证据。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `master_coldstart::burst_twenty`（F） | master 刚 ready 就能接第一波请求。 | 10 并发发 20 请求并记拓扑/落点。 | 请求最终完成，初始路由可解释。 | 重设计；20 笔不足以测冷启动承载，保留少量 smoke 并扩为流量爬坡场景。 |
| `master_dispatch_quota::single_prefill_ttl`（F） | 单 P 配额占满后 TTL 不留永久残账。 | 填满 quota、停 P、送阻塞请求、重启恢复。 | 旧账有界清理，新请求成功。 | 转场景；单 P 是边界但不代表弹性系统。 |
| `master_lifecycle::kill_single`（F） | 单 master 重启后能重新发现引擎。 | 先发基线，kill/restart master，再发请求。 | 拓扑恢复，旧在飞账清零，新请求成功。 | 保留，单 master 恢复合同。 |
| `client_fallback_failback::direct_generate_error`（F） | 直接 Generate RPC 错误可见且下一请求可恢复。 | 单 master 基线后注入 Generate 错误。 | 当前请求正确失败，不污染后续请求。 | 保留；这是错误透传，不是 HA。 |
| `client_fallback_failback::negative_errorcode`（F） | 业务/期限错误不能触发换 master 重试。 | 制造业务错误、期限错误并观察路由。 | 一次请求只处理一次，原错误保留。 | 保留，客户端重试边界。 |
| `client_fallback_failback::all_masters_down`（S） | 双 master 都不可用时客户端应有界失败。 | 持续流量中依次停 A/B，随后恢复。 | 故障窗错误可解释，无重复请求，恢复后成功。 | 保留场景；明确无法服务窗不要求零错误。 |
| `client_fallback_failback::wraparound`（S） | A/B 轮换重启后客户端能回到健康节点。 | kill A、重启 A、kill B，记录各阶段落点。 | 每阶段唯一请求，恢复分布稳定。 | 重设计；0.75/0.85 等瞬时份额阈值需按稳态窗重校准。 |
| `master_ha_failover::standalone_a_to_b`（S） | A 故障时同一请求能切到 B。 | 双独立 master 持续流量中 kill A。 | B 接管，重试不双执行，故障窗可解释。 | 保留场景；稳态窗允许采样波动。 |
| `master_lifecycle::kill_dual_b_to_a`（S） | B 故障后 A 能接管并最终恢复 B。 | 双 master 流量中 kill B、观察 A、重启 B。 | 唯一请求、可见终态、恢复稳定。 | 保留场景；与 A→B 组成双向轴，不复制编排。 |
| `master_lifecycle::freeze_short_long`（S） | 短暂停顿与长停顿应产生不同故障转移。 | 双 master 上短/长冻结，跨期限观察。 | 短暂停顿可恢复，长停顿有界切换，无双执行。 | 重设计；长短窗口、超时与客户端重试时钟需统一。 |

## 9. 弹性与持续流量：看变化过程和收敛，不看瞬时零失败

这一类本质上都是场景：新引擎加入、旧引擎摘除、P/D 容量与缓存重新分布需要基线→变化→过渡→新稳态→恢复的窗口。真实设计允许短时“疑似 dead”保留约 10 秒，故在故障窗硬断言零失败不符合实际；应分别断言发现/摘除时间、请求终态、受影响范围与稳态恢复。当前多项变体仅靠 normal/strict 或四 master 模式复制，建议改成少量场景 + 声明式轴。

| 测试（当前类型） | 目的 | 怎么测 | 期望 | 建议 |
| --- | --- | --- | --- | --- |
| `elastic_added_worker_fault::default`（F） | 新加 D 真能接流量，坏了还能恢复。 | 加 D、送流量、停机、重启、再送。 | 存活者继续服务，新 D 重新可用。 | 转场景；已有四阶段生命周期。 |
| `elastic_added_worker_fault::single_batch`（F） | SINGLE/BATCH 下新 D 同样可恢复。 | 同一加入/故障/重启流程改 master 模式。 | 终态和恢复一致。 | 删除独立例；作为上一场景的 master 轴。 |
| `elastic_added_worker_fault::single_nonbatch`（F） | SINGLE/NON_BATCH 下新 D 可恢复。 | 同剧本改非批。 | 不误杀新增 D。 | 删除独立例；保留 master 轴。 |
| `elastic_added_worker_fault::window_nonbatch`（F） | FIXED_WINDOW/NON_BATCH 下新 D 可恢复。 | 同剧本改窗口非批。 | 发现与恢复稳定。 | 删除独立例；保留 master 轴。 |
| `elastic_pending_drain::legacy_terminal`（S） | 带 pending 负载缩 P 时有可见终态。 | 慢两台 P、发波次、remove 一台再排空。 | 旧请求结算，幸存者恢复。 | 重设计；若 remove 前负载已完成，测试无效。 |
| `elastic_pending_drain::zero_errors`（S） | 收敛后新请求应零错误。 | 同样缩 P 后专测恢复窗口。 | 收敛后零错误，故障窗不强求。 | 保留场景；与 terminal 变体合并为阶段断言。 |
| `elastic_pending_drain::single_batch_terminal`（S） | SINGLE/BATCH 下 pending 缩容终态可见。 | 同样负载/缩 P 改模式。 | 终态有界、资源清理。 | 删除独立例；作为合并场景的模式轴。 |
| `trace_scale_out::prepared_background`（S） | 后台持续负载不会因 Python 阶段结束而停。 | 先启动 Java 背景流，再扩 D、观察阶段。 | 负载贯穿扩容，新增 D 承接且可排空。 | 保留场景；独立于短波次 elastic 例。 |
| `elastic_concurrent_mutation::default`（S） | 多引擎并发增删后全局状态能稳定。 | 流量中并发增删四 worker，随后静置。 | 拓扑和请求账有界收敛。 | 保留场景主例；用最终窗口而非瞬时零误差。 |
| `elastic_concurrent_mutation::single_batch`（S） | SINGLE/BATCH 下并发增删收敛。 | 同剧本切 master 模式。 | 拓扑、请求终态最终一致。 | 删除独立例；作为模式轴。 |
| `elastic_concurrent_mutation::single_nonbatch`（S） | SINGLE/NON_BATCH 下并发增删收敛。 | 同剧本改非批。 | 最终一致。 | 删除独立例；作为模式轴。 |
| `elastic_concurrent_mutation::window_nonbatch`（S） | FIXED_WINDOW/NON_BATCH 下并发增删收敛。 | 同剧本改窗口非批。 | 最终一致。 | 删除独立例；作为模式轴。 |
| `elastic_concurrent_mutation::single_nonbatch_convergence`（S） | 非批增删后的稳态单独达标。 | 同剧本多观察静置收敛窗。 | 最终拓扑/成功率达标。 | 删除独立例；并入主例恢复阶段。 |
| `elastic_concurrent_mutation::window_nonbatch_convergence`（S） | 窗口非批增删后的稳态达标。 | 同剧本观察收敛窗。 | 最终稳定。 | 删除独立例；并入主例模式轴。 |
| `elastic_concurrent_mutation::batch_convergence`（S） | 批模式增删后稳态达标。 | 同剧本观察收敛窗。 | 最终稳定。 | 删除独立例；并入主例恢复阶段。 |
| `elastic_concurrent_mutation::single_batch_convergence`（S） | SINGLE/BATCH 增删后稳态达标。 | 同剧本观察收敛窗。 | 最终稳定。 | 删除独立例；并入主例模式轴。 |
| `elastic_lifecycle::decode_scale_out_protection`（S） | 新 D 加入不能压死旧 D 或被直接打死。 | 旧 D 带载时加 D，观察准入/完成。 | 新旧 D 有界分担，新增 D 不异常死亡。 | 重设计；2P/2D、64 块池不足以代表目标规模。 |
| `elastic_lifecycle::rebalance`（F） | 新 P/D 加入后落点开始重新分配。 | 基线后加 worker、比较前后份额。 | 新 worker 获得合理流量。 | 转场景；重平衡需要持续窗口。 |
| `elastic_lifecycle::rebalance_single_batch`（F） | SINGLE/BATCH 重平衡。 | 同一加入/份额比较改 master 模式。 | 新 worker 参与。 | 删除独立例；作为重平衡场景模式轴。 |
| `elastic_lifecycle::rebalance_single_nonbatch`（F） | SINGLE/NON_BATCH 重平衡。 | 同剧本改非批。 | 新 worker 参与。 | 删除独立例；保留模式轴。 |
| `elastic_lifecycle::rebalance_window_nonbatch`（F） | FIXED_WINDOW/NON_BATCH 重平衡。 | 同剧本改窗口非批。 | 新 worker 参与。 | 删除独立例；保留模式轴。 |
| `elastic_lifecycle::normal`（S） | 常规加、删和三轮弹性循环能恢复。 | ramp→加 worker→三次缩扩→恢复。 | 每轮终态可见，最终拓扑与成功率达标。 | 保留场景主例；各轮独立归档窗口。 |
| `elastic_lifecycle::strict`（S） | 严格口径检验相同生命周期。 | 同一三轮过程用更严阈值。 | 稳态满足经校准的较严 band。 | 删除独立例；严格阈值不应复制一份剧本。 |
| `elastic_lifecycle::normal_single_batch`（S） | SINGLE/BATCH 跑常规弹性循环。 | 相同生命周期换模式。 | 最终稳态恢复。 | 删除独立例；模式轴。 |
| `elastic_lifecycle::strict_single_batch`（S） | SINGLE/BATCH 严格阈值。 | 同一生命周期和更严断言。 | 稳态在 band 内。 | 删除独立例；模式轴 + 断言等级。 |
| `elastic_lifecycle::normal_single_nonbatch`（S） | SINGLE/NON_BATCH 常规弹性循环。 | 相同生命周期换模式。 | 最终恢复。 | 删除独立例；模式轴。 |
| `elastic_lifecycle::strict_single_nonbatch`（S） | SINGLE/NON_BATCH 严格阈值。 | 相同生命周期换模式。 | 稳态在 band 内。 | 删除独立例；模式轴 + 断言等级。 |
| `elastic_lifecycle::normal_window_nonbatch`（S） | FIXED_WINDOW/NON_BATCH 常规弹性。 | 相同生命周期换模式。 | 最终恢复。 | 删除独立例；模式轴。 |
| `elastic_lifecycle::strict_window_nonbatch`（S） | FIXED_WINDOW/NON_BATCH 严格弹性。 | 相同生命周期换模式。 | 稳态在 band 内。 | 删除独立例；模式轴 + 断言等级。 |
| `elastic_lifecycle::steady_recovery`（S） | 摘机后新稳态能恢复。 | 持续流量、移除 worker、观察过渡与稳态。 | grace 后摘除生效，稳定期成功率恢复。 | 保留场景；不强求摘机瞬时零失败。 |
| `elastic_lifecycle::transient_imbalance`（S） | 摘机瞬间的不均衡会自行收敛。 | 3P/2D 下先打流量、摘 P、看恢复。 | 短时偏斜有界，幸存者稳定承载。 | 重设计；小 P 拓扑和波次不足以推断规模效应。 |
| `elastic_lifecycle::kv_full_shrink`（S） | D KV 满载时缩容请求不会无限积压。 | 2P/2D 小池加压、移除一台后恢复。 | 过渡期有界终态，新稳态能完成。 | 重设计；24 块、固定慢速和错误窗要按真实水位校准。 |
| `elastic_lifecycle::kv_skew_hot`（S） | 扩缩容期间热点缓存偏斜可恢复。 | 种热点，弹性变化后多窗口持续请求。 | 热点落点/命中率收敛。 | 保留场景；实际 device/memory reuse 分开报告。 |
| `elastic_lifecycle::kv_skew_cold`（S） | 无热点时弹性分布也稳定。 | 冷前缀流量重复相同弹性阶段。 | 不把冷流量强制粘在一台。 | 删除独立例；作为 hot/skew 场景冷流量对照轴。 |

## 汇总：删什么、留什么、移什么、重做什么

| 分类 | 保留独立例 | 删除独立例 | 功能转场景 | 重设计 | 合计变体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 请求闭环/状态 | 11 | 13 | 8 | 0 | 32 |
| 准入/排队 | 8 | 3 | 0 | 3 | 14 |
| 取消/fence | 8 | 10 | 8 | 2 | 28 |
| 优先级/抢占 | 5 | 7 | 4 | 4 | 20 |
| 引擎故障 | 5 | 3 | 5 | 0 | 13 |
| KV/亲和 | 7 | 9 | 10 | 6 | 32 |
| 均衡 | 0 | 1 | 4 | 2 | 7 |
| master/客户端 | 6 | 0 | 1 | 3 | 10 |
| 弹性 | 6 | 22 | 2 | 4 | 34 |
| **合计** | **56** | **68** | **42** | **24** | **190** |

按现有类型交叉看：149 个功能变体中，40 个保留、52 个删除独立行、42 个转场景、15 个重设计；41 个现有场景中，16 个保留、16 个合并掉独立行、9 个重设计。仅执行“删除独立行”这一步，维护单元可从 190 降到至多 122；这**不是**承诺保留 122 份不同的 Python 代码，也不是说 68 个行为可以无覆盖删除。四 master 模式和 P 数等应作为 YAML 轴，不用复制程序；实例数量要在变体合并后重新编译计算。

目标类型也要明确：所有“转场景”进入场景层；所有现有场景的“保留/重设计”仍留在场景层；15 个现有功能的“重设计”中，`balance_distribution::decode_spread` 和 `master_coldstart::burst_twenty` 应变成场景（冷启动另保留极短 smoke），其余 13 个仍是功能边界，但要先修构造。由此在删除重复独立行之后，暂估 **53 个功能变体、69 个场景变体**；这只是设计审查的上限，继续聚合时还能减少。这里的“删除独立例”是维护单元的归并或下沉，不等同于直接删掉独特行为。

### 删除判据如何落地

1. **过时、易变化**：`debug_snapshot`、`normalize_metrics`、`config_strict_reject` 等内部接口/指标格式测试从 E2E 套件移出；需要的话在接口或配置单测验证。`comparator_frozen_weak` 不能按旧内部比较器继续验收，应重设计外部行为。
2. **太简单、只测一个小点**：未知 RID/batch ID、特殊 ID、默认优先级 30/50、单个 RPC 延迟等独立行合并成表驱动功能契约；它们仍可作为输入样本，而非一个输入一套引擎启动。
3. **不通用、只测局部过程**：策略性 park、瞬时某一步的计数/份额和一次调试快照不应单独决定 E2E 结果。保留对客户端终态、容量与最终所有权的断言。
4. **没价值或重复**：`normal_no_fetch` 与 `client_no_fetch`，`queue_timeout_terminal` 与 `slo_deadline`，成对的 batch/nonbatch 编舞，elastic 的 normal/strict 与四模式复制，删除重复**入口**。先把唯一断言并入主例，再删旧变体。

### 转场景的聚合方案

- **状态缺口与恢复**：`prefill/decode_suppress_finished`、`no_respond`、`inflight_ttl_cleanup`、late completion 和 zombie 状态组成“基线→缺口→持续请求→恢复→排空”场景，分别报告 P/D 终态、错误窗和资源账。
- **取消与重启**：断流、schedule 投递后断链、P/D 退场、tombstone/fence 丢失及 preemption victim 组成一个请求级事件时间线；基础取消和幂等仍做功能测试。
- **缓存亲和**：local index、global holders、prefix、churn、capacity conflict 共用“种缓存→背景流量→驱逐/摘机→续写→收敛”模板；`leader_spill_p2/p3/p4` 保持独立拓扑报告行和独立 band。
- **均衡/弹性**：持续混合流量、P/D 压力转移、加删 worker 与新稳态使用同一窗口采集器。去掉仅靠 10/50 笔请求推导分布的断言，摘机故障窗遵守真实 grace，稳态单独验收。
- **master HA**：单 master 重启功能例保留；双 master A↔B、短/长冻结和双失联用统一客户端请求 ID/路由/重试记录比较，不把一次份额波动当故障。

### 重设计的先决条件

24 个“重设计”变体不能通过简单调松阈值变绿。优先做四件事：①把每个小池的 `ceil(请求 token/块大小)+reserve` 和并发占用列成可达到的块数账，特别是 3/4/5/17/24 块池；②对分布、弹性、缓存风暴使用足够 P/D、足够窗口和可声明的流量节奏，不从 1P/1D、2P/2D 或十几笔请求外推线上结论；③把“真实慢速”“真实 cache reuse”“client Fetch 与否”“10 秒摘机 grace”写入环境与验收口径；④每个 band 按拓扑和流量独立多次校准，保存样本与原始分子/分母。`leader_spill_batch/nonbatch` 只有在新版 P 轴完成对照后才可移除，避免丢掉已知 2P 锚点。

### 建议的实施顺序与验收

先把“删除独立例”中重复的模式、normal/strict、单字段边界收成 YAML 轴/步骤，编译前后核对每种独特断言是否仍存在；再把 42 个功能变体迁进持续流量模板；最后修 24 个不可直接信任的构造。建议最终按九组 Python 程序维护：请求/状态、准入/组批、取消/fence、优先级、故障恢复、KV/亲和、均衡、master HA、弹性；YAML 负责 topology、模式、负载、节奏和 band。九组是**组织目标**，不是在未审查依赖前强行合并现有文件。

迁移验收不以“新套件绿了”代替覆盖：编译清单要列出旧 `case::variant` 到新场景/功能步骤的映射，列出保留下来的客户端终态、引擎状态、master 所有权和指标断言；对被删除的独立行说明是在新位置覆盖、下沉单测，还是确认无价值后彻底删除。报告仅依据源码和编译清单，未运行这 388 个实例，尤其不代表 24 个待重设计例当前可靠。
