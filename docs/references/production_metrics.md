# P→D 与 Writeback 可观测性

本文说明 C++ P2P 路径的指标分层、采样边界，以及 Decode→Prefill writeback 当前已经具备和仍然缺少的观测能力。
指标通过 KMonitor 上报；以下名称都是代码中的注册名，不包含部署侧前缀或导出器转换。
“已实现”指当前分支有相应埋点，不等于已经验证线上监控面板。

## 1. 怎么分层

**整体结果、scheduler、worker 应当分开；TCP/RDMA 再作为底层传输层单独看。**
这里的“整体”还需要区分业务请求和一次缓存传输，不能用同一个 QPS 或耗时代表两者。

| 视角 | 回答的问题 | P→D 正向已有能力 | Writeback 当前状态 |
| --- | --- | --- | --- |
| 业务请求整体 | 请求是否变快？本轮实际复用了多少 KV？按现有计费口径复用多少？ | 引擎 stream 耗时、首 token 耗时、缓存复用统计 | 沿用请求指标，新增 `actual_reuse_len` / `reuse_len` 区分物理复用与计费口径 |
| 一次传输整体 | 发起了多少次？成功、失败、跳过各多少？何时真正结束？ | Decode scheduler 的结果和总耗时 | 已有两端 scheduler 尝试的结果、总耗时、成功传输量、发布量；不是独立的业务请求统计 |
| Scheduler 调度层 | 慢在排队、规划、远端协调、资源准入还是缓存发布？是否仍占着资源？ | 两端调度阶段、inflight、资源保留与租约耗时 | 已有两端阶段、inflight、清理与 KV 保留指标 |
| Worker 执行层 | 哪个发送/接收步骤慢？是 worker 排队、提交还是异步完成慢？ | Prefill 发送 worker、Decode 接收 worker 的独立埋点 | **尚无 writeback 专属 worker 分阶段埋点，不能认为已经与正向对齐** |
| TCP/RDMA 传输层 | 单个底层传输任务是否出错？等待和传输执行耗时是多少？ | 沿用已有传输指标及 reporter 覆盖范围 | 不新增方向/协议标签，不改内源 RDMA；writeback 失败看上层指标 |

“一次传输整体”目前直接使用发起端 scheduler 的汇总，不再额外上报一套相同的总量。
Writeback 总体优先看 `side=decode`，Prefill 端用来解释接收、发布和清理；两端不能相加当成两次 writeback。

### 两个方向的角色

```text
P→D 正向：
  D scheduler 发起读取 → P scheduler 规划、协调
  P worker 等待 Prefill 的 KV 就绪并发送 → D worker 接收
  D scheduler 确认完成，并处理资源/租约清理

D→P writeback：
  D 请求结束 → 入口校验 → D scheduler 保留源 KV、发起写回
  P scheduler 规划、准入目的缓存 → P worker 注册接收
  P scheduler 通知 D worker 发送 → 等待本地加载及参与传输的 worker 完成
  P scheduler 发布缓存、清理目的资源 → D scheduler 确认停止、释放源 KV
```

正向传输关系到当前请求取得 KV、开始 Decode；writeback 在 Decode 结束后触发，主要服务后续请求。
因此，“写回成功”和“下一轮真的命中”是两个不同结论，必须分别观察。

Decode 本地缓存复用与 writeback 是独立开关：`reuse_cache=0` 或关闭本地 DEVICE 缓存入库，
不应阻止启用 `p2p_writeback_enable` 的 Decode 写回。写回仍要求请求成功结束、非假请求、非 beam search，
并通过路由、完整 KV 块和资源预留等校验；异步发送完成前单独保留源 KV 引用，不需要先将其放入本地复用缓存。
Prefill 端能否在后续请求中命中，仍取决于 Prefill 自己的缓存配置和请求前缀。

## 2. P→D 正向是怎么做的

正向不是只有一条“传输耗时”，而是使用 **scheduler 汇总 + 调度阶段 + worker 执行阶段 + 底层传输**。
本节正向 P2P 指标统一省略前缀 `rtp_llm_p2p_connector_`，请求级指标除外。

### 2.1 业务请求与传输整体

- 请求级沿用 `rtp_llm_latency_us`、`rtp_llm_first_token_latency_us` 等 stream 指标。它们不是专门的 KV 传输耗时，也不等同于客户端 HTTP 全链路耗时。
- 正向发起端使用 `decode_schedule_qps`、`decode_schedule_failed_qps`、`decode_schedule_cost_time_us` 看调度结果和总耗时。
- Prefill 对端使用 `prefill_scheduler_qps`、`prefill_scheduler_failed_qps`、`prefill_scheduler_total_cost_time_us` 看对端处理情况，不能再与 Decode 端相加统计业务请求。

### 2.2 Scheduler：定位控制和协调耗时

| 位置 | 代表性指标后缀 | 观察内容 |
| --- | --- | --- |
| Decode 发起排队、规划 | `decode_schedule_kickoff_queue_time_us`、`decode_schedule_plan_cost_time_us` | 调度入口排队、传输计划生成 |
| Decode 远端调用、广播 | `decode_schedule_server_call_cost_time_us`、`decode_schedule_server_submit_time_us`、`decode_schedule_broadcast_submit_time_us`、`decode_schedule_tp_sync_cost_time_us` | RPC、广播提交和 TP 协调，不是纯网络数据拷贝 |
| Decode 租约与资源 | `decode_schedule_lease_query_time_us`、`decode_schedule_lease_hold_time_us` | 租约查询和资源持有 |
| Prefill 规划、广播 | `prefill_scheduler_plan_cost_time_us`、`prefill_scheduler_broadcast_submit_time_us`、`prefill_scheduler_broadcast_wait_time_us` | 接收读请求后的计划和 worker 协调 |
| Prefill 资源与旁路信息 | `prefill_scheduler_resource_wait_time_us`、`prefill_scheduler_resource_register_time_us`、`prefill_scheduler_side_channel_wait_time_us`、`prefill_scheduler_side_channel_fill_time_us` | 资源就绪、注册和附加信息等待 |
| Decode 存量状态 | `decode_scheduler_inflight_context_count`、`decode_scheduler_retained_kv_bytes`、`decode_scheduler_retained_kv_percent`、`decode_scheduler_retained_context_count`、`decode_scheduler_oldest_kv_hold_ms` | 未释放上下文、清理期间保留的 KV 和最长持有时间 |

### 2.3 Worker：定位实际发送、接收过程

| 位置 | 代表性指标后缀 | 采样含义 |
| --- | --- | --- |
| Prefill 发送 worker 汇总 | `prefill_worker_write_qps`、`prefill_worker_write_failed_qps`、`prefill_worker_write_total_cost_time_us` | 一次 worker 处理的结果和总耗时 |
| Prefill 发送准备 | `prefill_worker_write_add_buffer_time_us`、`prefill_worker_write_dispatch_time_us` | 数据缓冲准备和任务分发 |
| Prefill 发送线程池排队 | `prefill_worker_write_sender_queue_time_us` | 发送任务入池到真正开始执行 |
| Prefill 发送提交 | `prefill_worker_write_send_submit_time_us` | 任务开始执行到 `sender->send()` 返回，包含同步准备部分 |
| Prefill 发送完成 | `prefill_worker_write_send_complete_time_us` | 任务开始执行到发送完成回调，**不是** `send()` 返回就算完成 |
| Prefill 分层就绪与回调等待 | `prefill_worker_write_first_layer_wait_time_us`、`prefill_worker_write_last_layer_wait_time_us`、`prefill_worker_write_callback_wait_time_us` | KV 分层就绪和异步发送完成等待 |
| Decode 接收 worker 汇总 | `decode_worker_qps`、`decode_worker_failed_qps`、`decode_worker_total_cost_time_us`、`decode_worker_total_block_count` | worker 接收处理的结果、总耗时和块数 |
| Decode 接收准备与等待 | `decode_worker_prepare_time_us`、`decode_worker_recv_wait_time_us`、`decode_worker_first_layer_wait_time_us` | 接收准备、等待和首层就绪 |
| Decode 单个接收任务 | `decode_worker_recv_task_time_us` | 单个接收任务发起到完成回调；采样数量不等于 worker 请求数 |

还有 `stream_store_*`、`prefill_worker_store_*` 等指标用于 KV 就绪与存储等待。
上表中的 `prefill_worker_write_*` 属于 **P→D 正向发送**，实际埋点位于 `P2PWorkerPrefillRead.cc`，不能因为名字含 `write` 就把它当成 writeback 指标。

对 writeback 应借鉴的是这套分层以及“提交不等于异步完成”的采样边界，而不是机械复制所有首层、末层指标。
Writeback 当前是完整缓存块范围的写回，不具备与正向逐层流水完全相同的等待语义。

## 3. Writeback 整体结果：是否写回、是否发布、是否复用

### 3.1 后续请求的复用收益

这两个指标是业务请求的缓存复用样本，**不使用 writeback 前缀**，也不按一次后台写回任务计数。

| 完整指标名 | 类型 / 单位 | 含义 |
| --- | --- | --- |
| `rtp_llm_actual_reuse_len` | GAUGE / token | Prefill 实际复用的前缀 KV 长度，包含 writeback 缓存首次命中的部分 |
| `rtp_llm_reuse_len` | GAUGE / token | 按现有 prompt-cache 计费规则，可计入缓存折扣的复用长度 |

- 标签为 `side=prefill|decode|normal`；有 reporter 时，零命中也上报，不能只用命中请求计算平均值。
- Decode 端取的是对端 Prefill 的复用结果，不把 Decode 本地缓存命中冒充 Prefill 复用。PD 场景选定一侧统计，不重复计算两端。
- 当前实现满足 `0 <= reuse_len <= actual_reuse_len`。两者之差表示当前请求中实际复用、但不进入既有折扣口径的部分，不是写回缓存的全生命周期收益。
- 不新增 `writeback_first_use_len` 等独立复用指标；内部协议字段是否存在是另一件事，本次埋点不改变计费规则。
- 既有 `rtp_llm_reuse_length`、各缓存层级的复用指标保留原有含义，不直接与这两个新指标混用。

### 3.2 写回尝试的结果

以下及第 4 节指标统一省略前缀 `rtp_llm_p2p_writeback_`。
它们虽然放在“整体结果”视图中，**实际是在 scheduler 层上报，计数单位是单侧写回处理尝试，不是用户请求**。

| 指标后缀 | 类型 | 上报时机和含义 |
| --- | --- | --- |
| `qps` | QPS | 单侧处理最终结束；Decode 在源资源实际释放后上报，Prefill 正常接纳的请求在目的资源清理后上报；参数等前置拒绝直接上报 |
| `failed_qps` | QPS | 已结束处理中的错误子集；成功结束量可由同侧 `qps - failed_qps` 得到 |
| `skipped_qps` | QPS | 写回入口校验判定不执行，例如不满足启用、路由或完整块范围等条件；不是所有业务请求的跳过统计 |
| `no_transfer_qps` | QPS | 成功结束、但不需要跨端传输的子集，例如目的端已经缓存；不代表一定没有本地缓存加载 |

这组指标携带 `side=prefill|decode`、`submitted=true|false`、`reason`：

- Decode 的 `submitted` 表示已进入对端调用阶段；Prefill 的 `submitted` 表示接收 worker 广播已成功提交到控制队列，不表示所有 worker 已完成注册。
- `reason` 是跳过原因、错误码，或特定的 `prefix_evicted_or_demoted` 原因；请求 ID 和错误详情放日志，不放标签。
- 统计完成失败率时，固定同一 `side`，对 `reason`、`submitted` 聚合后计算 `failed_qps / qps`。跳过不放进完成失败率分母。
- 入口异常还有第 4 节的 `stage=submission` 样本；阶段只采样耗时和结果标签，不单独统计阶段 QPS。不要把入口跳过、两端完成数强行拼成严格一一对应的业务请求漏斗。

### 3.3 传输量、发布量与命中量分开看

| 指标后缀 | 类型 / 单位 | 含义及边界 |
| --- | --- | --- |
| `planned_bytes` | GAUGE / byte | Decode 是候选范围的计划数据量；Prefill 是准入后实际接纳路由的计划数据量。两端可能不同，不是已完成传输量 |
| `transferred_bytes` | GAUGE / byte | 仅 Prefill 在本地加载及参与传输的 worker 均成功完成后上报；取接纳路由的 payload 总和，覆盖参与路由的各 rank。是成功确认后的逻辑数据量，不是网卡实测流量 |
| `committed_token_count` | GAUGE / token | Prefill 本次回写成功提交的后缀 token 数，按准入后后缀块数乘规范缓存键块长计算；无需回传后缀时为零。跨请求不去重，不代表缓存净增量；不按 TP rank 或缓存组重复累加，也不等于本次新生成 token 数 |

`transferred_bytes` 不统计失败、部分完成任务的流量，也不包含报文头和重传开销。
传输成功后，缓存发布仍可能失败，此时可以有成功传输量而没有成功发布量。
**发布成功也不能证明下一轮命中，必须再看后续请求的 `actual_reuse_len` 和 `reuse_len`。**
这两个复用指标不单独标记缓存来源，因此只看它们也不能把所有命中都归因于 writeback；需要结合测试上下文或请求日志确认。

## 4. Writeback Scheduler：慢在哪里、资源何时释放

### 4.1 单侧总耗时和清理耗时

这些指标与第 3.2 节的最终结果一起上报，沿用 `side`、`submitted`、`reason` 标签。
统一前缀仍为 `rtp_llm_p2p_writeback_`。

| 指标后缀 | 类型 / 单位 | Decode 端边界 | Prefill 端边界 |
| --- | --- | --- | --- |
| `cost_time_us` | GAUGE / 微秒 | 写回上下文创建到源资源实际释放并上报 | `StartWriteBack` 处理开始到最终上报，包含清理；前置拒绝为开始到拒绝上报 |
| `hold_time_us` | GAUGE / 微秒 | 不上报，与 `cost_time_us` 重复 | 目的缓存准入返回资源后，到目的资源实际释放；未取得资源时为零 |
| `drain_time_us` | GAUGE / 微秒 | 首次失败触发清理，或正常完成进入最终释放，到源资源释放 | 主处理逻辑返回成功/失败、进入清理，到目的资源释放；前置拒绝不产生该样本 |

逻辑上的取消/失败可以先返回，但资源未实际释放时仍有在途状态；不能仅凭业务回调认为清理完成。
上述耗时不是用户请求延迟，正常写回也不能把 `hold_time_us` 再加到 `cost_time_us` 上。

### 4.2 调度阶段

统一使用以下耗时指标，通过标签区分阶段，不为各阶段额外注册 QPS：

| 指标后缀 | 类型 / 单位 | 含义 |
| --- | --- | --- |
| `stage_cost_time_us` | GAUGE / 微秒 | 该次阶段的耗时 |

标签为 `side`、`stage`、`success=true|false`，没有结果组的 `submitted`、`reason`。
`transferred_bytes` 随成功的 Prefill `transfer` 阶段上报，也使用这组标签。

| `side` | `stage` | 当前实际采样边界 | 不能理解成什么 |
| --- | --- | --- | --- |
| `decode` | `kickoff_queue` | 上下文创建到 kickoff 开始；只有真正开始的任务有该样本 | 不是发送 worker 线程池排队 |
| `decode` | `plan` | `planFor()` 调用前后 | 不是所有路由构造、校验和准备的总和 |
| `decode` | `peer_call` | 发起对端调用到观察到 `StartWriteBack` 结果 | 不只是 RPC 握手；可能包含 Prefill 的准入、传输、发布和清理 |
| `decode` | `submission` | 入口捕获到异常时记录的失败阶段 | 不是每次正常写回都会产生的提交耗时 |
| `prefill` | `plan` | `planFor()` 调用前后 | 不是整个接收准备阶段 |
| `prefill` | `admission` | `admitWriteBackDecodeCache()` 调用前后 | 不是后续异步本地加载全部完成的耗时 |
| `prefill` | `transfer` | 本地加载提交、接收 worker 提交与注册、发送通知、等待加载及发送/接收完成的协调区间 | **不是纯网络耗时，也不是已补齐的 worker 发送/接收耗时** |
| `prefill` | `commit` | `commitWriteBackDecodeCache()` 调用前后 | 不是后续请求真正命中缓存的耗时 |

阶段失败后，后续阶段可能根本不执行；没有某个阶段样本，不代表该阶段耗时为零。
`peer_call` 等区间会包含对端阶段，阶段之间也可能重叠，不能把所有阶段的均值或分位数相加得到端到端耗时。

### 4.3 在途、清理和 KV 保留状态

以下都是 GAUGE，只增加 `side` 标签，不携带某个请求的 `reason`、`submitted`。

| 指标后缀 | 单位 | 含义 |
| --- | --- | --- |
| `inflight_context_count` | 上下文数 | 已登记、尚未物理释放的在途上下文；包含正常执行和清理中的任务，不只统计异常 |
| `oldest_inflight_ms` | 毫秒 | 最老在途上下文的年龄；两端各自从本地上下文建立时计时 |
| `retained_context_count` | 上下文数 | 清理期间仍保留 KV 的上下文数 |
| `retained_kv_bytes` | byte | 清理保留的 DEVICE KV，按缓存组/物理块去重，沿用资源泄漏保护的核算口径 |
| `retained_kv_percent` | 百分比，0–100 | 保留 DEVICE KV 占已配置设备缓存容量的比例 |
| `oldest_kv_hold_ms` | 毫秒 | 清理保留资源中最长的持有时间 |

状态会周期刷新，空闲或等待卡住任务时也会刷新，清理结束后回到零。
`retained_*` 统计的是进入清理保护路径后仍不能释放的资源，**不是所有正常传输持有的 KV，更不是进程总显存**。
因此，在途数非零而 `retained_kv_bytes=0` 可以是正常情况。

## 5. Writeback Worker：当前缺口与补齐方向

**当前没有 writeback 专属 worker 阶段指标。**
`P2PWorkerDecodeWrite.cc` 实际负责发送，`P2PWorkerPrefillWrite.cc` 实际负责接收；目前已有的是 scheduler 汇总，底层指标沿用现有 reporter 覆盖范围，见第 6 节。
不能用 `stage=transfer` 代替下表中的测量，也不能把正向的 `prefill_worker_write_*` 当作反向覆盖。

以下均为**待补设计，不是已实现指标，也不是已经注册的新指标名**：

| 位置 | 应补观测项 | 建议采样边界 | 参考正向 |
| --- | --- | --- | --- |
| Decode 发送 worker | worker 处理次数、失败次数、总耗时、块数 | worker 请求进入到本次任务组真正完成；每个 worker 请求一次，不按每个分片回调重复计数 | Prefill worker 汇总 |
| Decode 发送线程池 | 排队耗时 | 任务入池到开始执行；入池失败单独记录错误，不伪造完成样本 | `prefill_worker_write_sender_queue_time_us` |
| Decode 单个发送任务 | 提交耗时 | 执行开始到 `sender->send()` 返回，区分同步准备开销 | `prefill_worker_write_send_submit_time_us` |
| Decode 单个发送任务 | 完成耗时和结果 | 执行开始到发送完成回调，包括同步回调、异常、取消路径，确保只上报一次 | `prefill_worker_write_send_complete_time_us` |
| Prefill 接收 worker | worker 处理次数、失败次数、总耗时、块数 | worker 请求进入到任务组接收完成/失败 | Decode worker 汇总 |
| Prefill 接收准备 | 准备与注册耗时 | 接收请求解析、缓冲和接收任务建立，到报告注册完成；异常路径单独记失败 | `decode_worker_prepare_time_us` |
| Prefill 单个接收任务 | 接收完成耗时和结果 | `receiver->recv()` 发起到该任务真正完成，不能只测函数返回 | `decode_worker_recv_task_time_us` |

补齐时应把 worker 请求汇总和单个传输任务样本分开，并复核现有 reporter 能覆盖哪些 rank。
是否增加 rank 标签需要先确定采集覆盖和基数预算，不能仅增加标签就声称已覆盖所有 worker。
这些是下一步 runtime 埋点工作，本次文档分层本身不会使它们自动具备。

## 6. TCP/RDMA：保持现有底层实现

底层直接复用原有 reporter，不新增 `direction` / `transport` 标签，也不修改内源 RDMA 的 reporter 透传。
Writeback 失败仍通过传输任务结果传递到上层，由 `rtp_llm_p2p_writeback_failed_qps` 等指标观测，不依赖底层指标是否上报。
下表列出已有指标定义，每行使用自己的完整前缀；定义存在不代表当前 P2P 路径都会上报。

| 完整前缀 | 指标后缀 | 观察内容 |
| --- | --- | --- |
| `rtp_llm.transfer.client.` | `qps`、`error_qps`、`block_count`、`total_block_size`、`latency_us` | TCP 发送侧任务结果、数据规模、耗时 |
| `rtp_llm.transfer.server.` | `qps`、`error_qps`、`block_count`、`block_size`、`wait_task_latency_us`、`latency_us` | TCP 接收侧任务结果、规模、任务等待和总耗时；注册名确实是 `block_size` |
| `rtp_llm.transfer.rdma.` | `qps`、`error_qps`、`block_count`、`total_block_size`、`pending_latency_us`、`write_latency_us` | 内源已有定义；当前 P2P receiver 的工厂调用未透传 reporter，不能依赖此路径上报 |

注意以下边界：

- RDMA 的 `write_latency_us` 是既有命名，当前 `RdmaConnection::read()` 路径在 `ReadBatch` 提交和完成回调附近计时，不能根据名称推断使用的是 RDMA Write。
- 这里按底层传输任务采样，不是用户请求数；字节样本也不天然等于成功传输的有效业务数据量。失败任务可能仍带请求字节数。
- reporter 是否存在决定哪些 worker 真正上报。默认引擎只在 TP rank 0 创建 reporter，不能把 rank 0 样本简单乘 TP 数当成完整统计。
- Writeback 的 `transferred_bytes` 来自 Prefill scheduler 对接纳路由的汇总，覆盖该计划参与的 rank；不能与单个 worker 的底层样本相加。
- TCP client/server 也不能相加当作两份独立业务流量。连接失败等控制面 RPC 错误，应与数据面传输错误分开排查。

## 7. 看板和排障顺序

建议按下面四组展示，避免把不同层的时间、次数和数据量混在一个面板里：

1. **整体结果**：业务请求耗时、`actual_reuse_len` / `reuse_len`；Decode 侧 writeback 完成、失败、跳过；Prefill 侧成功传输量和缓存提交量。
2. **Scheduler**：按 `side`、`stage`、`success` 看各结果下的阶段耗时；同时看总耗时、资源持有、清理耗时、在途年龄和保留 KV。
3. **Worker**：正向看发送排队、提交/完成、接收准备/任务完成；writeback 对应面板标为待补，不拿 scheduler 的 `transfer` 顶替。
4. **TCP/RDMA**：结合部署协议、已有底层样本和日志排查数据面问题；没有方向标签，不据此区分正向/回写。当前 P2P 路径不依赖 `rtp_llm.transfer.rdma.*` 上报来判断 writeback 失败。

典型排查例子：

- 写回成功，但下一轮 `actual_reuse_len=0`：不能凭成功计数认定复用生效；继续查请求前缀、完整块范围、缓存发布与淘汰。
- `actual_reuse_len > reuse_len`：区分物理复用和既有折扣口径，不直接判为复用失效或计费错误。
- Prefill `transfer` 慢、底层传输不慢：继续查本地加载、注册与协调；当前缺 worker 阶段，不能直接归因网络。
- `failed_qps` 暂时没增长，但在途年龄、保留 KV 持续增长：可能尚未物理收尾，最终结果指标还没有上报，应优先检查清理和停止确认。

### 聚合注意事项

- QPS 是事件发生率，失败率使用同一侧、同一层、同一统计窗口的分母。
- 耗时、字节数、块数和 token 数都是 GAUGE；其中结果/阶段指标是逐事件样本，`inflight`、`retained` 等是周期状态值，两者不能按同一种方式累加。
- 如需数据量吞吐，必须确认监控系统保留了逐事件样本和窗口 sum/count 口径；不能直接对一个“最后值/平均值”GAUGE 调用计数器的 `rate()`。
- 阶段分位数不相加，P/D 两端不重复累加，底层多任务统计不当成业务请求统计。
- 请求 ID、peer 地址、unique key 放日志，用来关联具体请求，不做高基数指标标签。

## 8. 实现位置索引

路径相对于仓库根目录；下表用于从指标回查代码，而不是表示这些位置都已经补齐了 writeback 埋点。

| 层次 | 主要文件 |
| --- | --- |
| 指标注册、标签与上报条件 | `rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h`、`rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.cc` |
| 业务请求耗时与复用长度 | `rtp_llm/cpp/engine_base/stream/GenerateStream.cc`、`rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc`、`rtp_llm/cpp/metrics/RtpLLMMetrics.cc` |
| 正向 scheduler | `rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillRead.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc` |
| 正向 worker | `rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillRead.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeRead.cc` |
| Writeback 入口与 scheduler | `rtp_llm/cpp/cache/KVCacheManager.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc` |
| Writeback 清理资源统计 | `rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.cc` |
| Writeback worker，专属埋点待补 | `rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.cc`、`rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.cc` |
| 底层 backend 创建与原有 reporter 传递 | `rtp_llm/cpp/cache/connector/p2p/P2PConnectorBackend.cc` |
| TCP 指标 | `rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.cc` |
| RDMA 指标及采样边界，位于内源 | `internal_source/rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMetric.cc`、`internal_source/rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.cc` |
