# Decode KV Cache 回传 Review：功能、资源与可观测性

- 整理日期：2026-10-21
- 审查主机：110（`b300-host-0`）
- 仓库：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource`
- 审查版本：`bed7c2d8934a9562149e0d64a2a28660730b3a6b`，对应此前分析的 `rym/feat/decode-writeback-v1.3`。
- 范围：截至该提交的完整 Decode writeback 链路，不限于该提交新增的 diff。
- 方法：静态代码审查与已有测试内容检查；本次未运行测试，也未确认线上服务运行此版本。
- 版本提醒：审查期间远端 checkout 曾切换到 `develop/dsv41_ced_release`；文档入库前已确认返回 `rym/feat/decode-writeback-v1.3`，HEAD 为上述审查提交。本文结论仍以固定提交为准。

本文共 10 条意见，区分已由代码确认的行为问题与设计改进建议，不将全部意见视为已复现的 bug。P1 表示建议优先处理的高影响问题或设计风险，P2 表示常规修复和改进。

## 1. [P2，已确认] 按 Prefill 实际接收范围裁剪 Decode 的发送 routes

Prefill 算出 `start_block` 后，只裁剪了接收 routes；下发 `WRITE` 时直接复用了 `request.decode_workers()` 中的原始发送 routes。

例如 Prefill 已有 B4、B5，只缺 B6、B7，Decode 仍会通过 TCP 发送 B4～B7。接收端按 key 忽略多余数据，但 D2H、序列化和网络开销已经发生。本地命中能减少对应的 H2D，当前不能减少这些 blocks 的 TCP 发送开销。

建议下发前按接收 key 集合裁剪发送 routes，并验证实际发送字节数。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:549`

补充证据：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTaskContext.cc:86`，接收端以注册的 expected blocks 为准进行拷贝。

## 2. [P2，已确认] commit 返回成功，需要反映实际发布结果

`insertExternalBlocks()` 调用普通 Tree store 后直接返回 OK，没有检查回传 suffix 是否真正被接纳。

已有单测明确覆盖：分配后并发出现同 key 的 HOST 节点，commit 返回 OK，但后续 suffix 没有成功插入。这样 RPC 成功和缓存收益会脱节。

建议返回实际发布数量及有效复用长度，至少区分“已发布”“并发重复”“因层级冲突未发布”。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:223`

测试证据：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/test/SingleTypeKVCacheAllocatorTest.cc:452`，`ExternalInsertDoesNotOverwriteHostSuffixCreatedAfterAllocation`。

## 3. [P2，已确认] SWA active tail 不足以覆盖复用窗口时，应提前识别无收益写回

当前允许按 `active_tail_blocks` 回传，但下一次匹配按 sliding window 判断。已有测试中，commit 成功后 `matched_device_blocks` 仍等于原来的 `start`，没有增长。

建议 admission 时检查最终 endpoint 是否能同时满足 FULL/SWA/LINEAR 的复用条件；无法增加复用长度时，明确跳过或记录部分收益，不要仅凭传输成功判断效果。

这是已确认的收益不足场景，不等同于已确认的数据正确性错误。

测试证据：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/test/HybridTypeKVCacheAllocatorTest.cc:2309`，`SwaWindowDoesNotExpandAllocatedActiveTail`。

## 4. [P2，已确认] 对端 client 缓存需要容量或空闲回收机制

Prefill 的 `source_clients_` 按 Decode 地址列表缓存，Decode 的 `servers_` 按 Prefill 地址缓存，当前只新增、不淘汰。每个 `BroadcastManager` 还会创建独立 completion thread。

实例滚动发布、扩缩容且地址不断变化时，会持续积累旧 client、线程和连接资源。建议增加 idle TTL/LRU，确保有在途 RPC 的 client 不被提前销毁。

代码位置：

- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:98`
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.cc:41`
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/model_rpc/BroadcastManager.h:265`

## 5. [P1，容量设计建议] 增加按 KV 字节数计算的 writeback admission 限额

当前会先持有 Decode 源 blocks，并在 Prefill 分配目标 blocks；线程池队列容量限制不了已经在传输中的总显存占用。

大量长请求同时完成、对端变慢时，后台写回可能长期占住两端显存，影响前台推理。建议两端都增加 `max_inflight_bytes`、请求数和显存水位限制，在获取资源前做 admission。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:147`

## 6. [P1，故障隔离建议] 明确单次 writeback 触发进程退出的上线策略

当前到达 `deadline + lease_query_timeout` 仍无法确认传输停止，就执行 `abort()`。这是为了避免释放仍在使用的显存，但控制面故障也可能把缓存优化失败扩大成整个实例退出。

建议补充故障注入验证，并明确是否采用“隔离未确认的 blocks、停止接收新 writeback、超过隔离预算后退出”的策略。不能简单超时释放 blocks。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc:945`

## 7. [P2，资源优化建议] 缩小 Decode 源资源的引用范围

`asyncWriteBack()` 当前对所有 group 的 `0～ready_blocks` 增加引用，包括无需发送的完整 prompt prefix，以及未开启 prefix reuse 的 group。

长 prompt、短 output 时，可能为了回传很少的 suffix，额外持有很大的 prefix。建议按实际发送 plan 的 group/key 范围保留物理 blocks，保持逻辑位置不变，但减少无关引用。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/KVCacheManager.cc:1402`

## 8. [P2，策略与设计建议] 按 group 类型放松 required prefix 的 DEVICE 约束

建议将 writeback 的 prefix admission 与推理 prefix matching 分离：FULL 保留连续、稳定的 DEVICE prefix 要求；LINEAR/SWA 不应仅因旧 prefix 状态位于 HOST/DISK 而拒绝回传，也不应为接收新 suffix 强制加载已不再需要的历史状态。

### 当前限制与放松边界

当前 `validateDevicePrefixLocked()` 按各 group 的 `computeReuseBlockCount(required_prefix_blocks)` 校验 required prefix：FULL 检查整个 prefix，LINEAR 检查最后一个 block，SWA 检查 prefix 末尾的窗口。这些被检查的资源均须处于有效稳态、未 detached，并拥有完整 DEVICE blocks；并不是所有 group 都检查整段历史。

建议分别处理：

| Group | 旧 required prefix 的建议要求 | 新回传结果的有效性要求 |
| --- | --- | --- |
| FULL | 保留 key 路径及连续、稳定、完整的 DEVICE prefix，避免低层节点夹在 DEVICE 路径中 | 新增 FULL blocks 完整，保持 DEVICE 连续性 |
| LINEAR | 不要求旧 endpoint 状态位于 DEVICE，可保留在 HOST/DISK | 按新的 endpoint 判断状态是否完整、有效 |
| SWA | 不要求旧 prefix 窗口全部位于 DEVICE，可保留在 HOST/DISK | 按最终 endpoint 的复用窗口确定需要的数据，而非按旧 prefix 窗口要求加载 |

这里放松的是不同节点之间的驻留层级约束，不是取消资源完整性、引用生命周期或并发迁移保护，也不是允许同一 group 的同一节点同时保有两个活动 serving tier。

### 示例：旧窗口已经不再需要

假设 required prefix 为 B0～B3（R=4），回传后完整 block 范围为 B0～B7（N=8），SWA 窗口为 2 个 blocks。D 表示 DEVICE，H 表示 HOST，`·` 表示该示例不要求保有的状态：

```text
             required prefix | 新 suffix
             B0 B1 B2 B3     | B4 B5 B6 B7
FULL          D  D  D  D     | D  D  D  D
SWA           ·  ·  H  H     | ·  ·  D  D
LINEAR        ·  ·  ·  H     | ·  ·  ·  D
```

上述结果应允许接纳。最终 SWA 需要 B6、B7，LINEAR 需要 B7，没有必要先把旧 SWA 的 B2、B3 或旧 LINEAR 的 B3 加载到 DEVICE。旧状态在 DISK 时同理。

边界情况：如果只新增 B4（N=5），SWA 最终窗口是 B3、B4。B3 在 HOST 不应直接导致 admission 拒绝，但如果希望新 endpoint 立即形成完整 DEVICE 命中，则仍需将 B3 加载到 DEVICE。允许发布缓存，不等于已经增加 DEVICE 可复用长度；此处需要与第 3 条的收益判断配合。

### 实现建议

1. **拆分 admission 校验。** 将统一的 DEVICE prefix 校验拆成 FULL 连续性校验和非 FULL 的资源安全检查。保留 key 路径与 FULL 约束，对实际访问、加载或修改的 LINEAR/SWA 资源检查状态与所有权；不要只因旧状态在 HOST/DISK 拒绝。
2. **拆分写回规划与推理匹配。** 当前普通 loader 面向所有 group 的推理复用，allocator 又用统一匹配长度决定 `start_block`。仅跳过非 FULL 的 DEVICE 校验，仍可能触发旧状态加载或缩短统一匹配长度。建议明确 FULL 连续锚点、每个 group 的接收 keys，以及最终窗口确实需要的本地加载 keys，避免为了统一 start 恢复无关历史状态。尤其不要让 start 退到 Decode 未提供源 routes 的 required prefix 以内。
3. **同步调整 commit 校验。** `insertExternalBlocks()` 在提交时还会调用同一 DEVICE prefix 校验，且参数为 `start_block`；不能只修改 admission。提交时应重新确认 FULL 连续性、新资源完整性及并发状态，并按新的 LINEAR endpoint/SWA 窗口报告实际复用收益。
4. **明确同 key 的低层冲突处理。** Tree 插入逻辑对非 FULL 不施加 FULL 的跨节点连续性限制，但遇到同 key 已存在 HOST/DISK 资源时，不会自动用新 DEVICE 资源覆盖。应明确复用并加载现有资源、受控 promotion 或保留原资源的策略，并返回真实接纳结果，不能把“不怕夹心”理解为可以直接覆盖低层节点。
5. **补充测试与指标。** 覆盖旧 LINEAR endpoint/旧 SWA 窗口在 HOST 和 DISK、SWA 最终窗口跨 prompt 边界、同 key 低层冲突以及 admission 后并发迁移。验证不加载已过期窗口、最终有效复用长度正确，并确认 FULL 的 HOST/DISK prefix 仍按原约束拒绝。拒绝指标区分 group 类型、低层可恢复、真正缺失及并发状态冲突。

FULL prefix 的有预算本地恢复可以作为独立后续优化评估，不应与本次 LINEAR/SWA 放松混在一起；恢复后仍需满足 FULL 的 DEVICE 连续性要求。

代码位置（均对应本文固定审查提交）：

- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:158`：writeback 匹配入口及 prefix 校验。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:218`：commit 再次校验 prefix。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTree.cc:196`：FULL 路径连续性限制；后续资源接纳逻辑不覆盖已有非空资源。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTreeLoader.cc:142`：面向各 group 推理复用的匹配结果及加载规划。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/KVCacheAllocator.cc:227`：由匹配结果决定写回 start 及本地加载。

本条是设计修改建议，尚未实施或运行测试验证。

## 9. [P2，超时策略建议] 明确后台 writeback 是否应继续受原生成 deadline 限制

writeback 在生成完成后发起，却使用 `min(now + writeback_timeout, 原请求 deadline)`。

例如生成完成时距离原 deadline 只剩 100ms，即使配置 writeback timeout 为 5 秒，写回仍只有约 100ms。建议将后台写回预算与生成预算的关系明确配置；至少增加“发起时剩余预算”指标，避免把这种失败误判为网络慢。

代码位置：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:155`

## 10. [P2，可观测性建议] 将 writeback 拒绝从通用失败中独立提取，补充阶段、错误分类与日志

### 现状：有通用指标和错误码，缺少独立拒绝分类

当前除显式 skip 分支外，只要 `error.hasError()`，就会计入 `rtp_llm_p2p_writeback_failed_qps`，admission 拒绝和执行失败混在一起。已有标签为 `side=prefill/decode`、`submitted=true/false` 和 `reason`。reason 通常取错误码名称，只有错误文本恰好为 `prefix_evicted_or_demoted` 时做字符串特判。

当前 Decode 的 `submitted=true` 表示开始调用 Prefill，Prefill 的同名标签表示已提交 worker 接收任务，语义不同。`prefix_evicted_or_demoted` 也同时覆盖 admission 拒绝和 commit 前重新校验失败。

现有指标可以通过以下标签组合筛选 Prefill 在提交接收任务前发生的 prefix 拒绝：

```text
rtp_llm_p2p_writeback_failed_qps
  side=prefill
  submitted=false
  reason=prefix_evicted_or_demoted
```

这只是现有实现下的特定场景筛选，不能作为通用拒绝分类。不能把全部 `submitted=false` 当成拒绝，因为其中可能包含规划、分配、加载提交等错误；也不能仅过滤 reason，否则会混入 commit 失败。

当前主要错误码如下，数字为内部 `ErrorCode`，不是未经转换的 RPC 枚举值：

| 场景 | 当前内部错误码 | 分类不足 |
| --- | --- | --- |
| Prefix 缺失、降级或资源状态不满足 | `CACHE_STORE_STORE_FAILED`（8309） | 多种原因合并为 `prefix_evicted_or_demoted`，且 admission/commit 共用 |
| 参数不合法、receiver 不可用、配置不支持 | `INVALID_PARAMS`（605） | 请求错误与服务状态混在一起 |
| 重复请求、取消或停止 | `CANCELLED`（8100） | 无法直接区分重复与取消 |
| 队列不可用 | `OUTPUT_QUEUE_FULL`（8102） | 接收前拒绝和执行中失败可能共用 |
| KV 分配失败 | `MALLOC_FAILED`（602） | 缺少 writeback admission 语义 |
| Admission 过期、传输超时、发布超时 | `GENERATE_TIMEOUT`（603） | 缺少发生阶段 |

已检查的 Prefill `reject()` 和 allocator admission 失败路径会返回 `error_code/error_message` 并上报指标，但没有统一的逐请求拒绝日志。已有提交异常、RPC 失败、worker 执行失败、清理无法确认停止等日志，不能据此完整统计 admission 拒绝；RPC 返回错误文本不等于该文本已经落日志。

### 建议一：新增独立拒绝指标，明确接纳边界

新增 `rtp_llm_p2p_writeback_rejected_qps`，建议按以下语义区分结果：

| 结果 | 定义 | 指标 |
| --- | --- | --- |
| skipped | Decode 本地未发起，例如无完整 suffix | 原有 `rtp_llm_p2p_writeback_skipped_qps` |
| rejected | Prefill admission 未通过，未进入执行阶段 | 新增 `rtp_llm_p2p_writeback_rejected_qps` |
| failed | 已接纳，后续加载、传输、commit 等失败 | `rtp_llm_p2p_writeback_failed_qps` 的目标口径 |
| no_transfer | 成功，但不需要远端传输 | 原有 `rtp_llm_p2p_writeback_no_transfer_qps` |

不要使用 `submitted` 代替“是否通过 admission”。应显式记录 admission 结果，明确在执行本地 load 或提交接收任务之前完成接纳状态转换。本地 load 已启动但接收任务尚未提交的失败，不能再归类为 admission 拒绝。

Decode 本地排队失败应单独标记 `stage=kickoff`，不混入 Prefill 拒绝率；请求取消也应有独立 outcome，避免混同缓存策略拒绝。

### 建议二：使用结构化分类，不解析错误字符串

在内部结果及 RPC 响应中增加明确的 outcome、stage、reason，保留原始错误码用于诊断。建议阶段包含 `kickoff/admission/load/transfer/commit/cleanup`，拒绝 reason 使用有限枚举，例如：

```text
prefix_missing
prefix_not_device
prefix_state_conflict
insufficient_kv_capacity
receiver_unavailable
unsupported_topology
invalid_request
duplicate_request
expired_before_admission
```

- `prefix_not_device` 可增加低基数标签 `group_type=full/linear/swa`、`tier=host/disk`。按第 8 条放松约束后，旧 LINEAR/SWA 状态在 HOST/DISK 不应再自动触发拒绝。
- commit 时发现 prefix 不满足，应为 `outcome=failed, stage=commit`，不能计入 admission 拒绝。
- 队列满按发生阶段分类，不能仅凭 `OUTPUT_QUEUE_FULL` 判定拒绝。
- 保留首个根因及其阶段，清理阶段的次生错误单独记录，避免覆盖原始拒绝或失败原因。
- 补充实际发送 bytes、实际发布 blocks、有效复用增长量和在途持有 bytes，区分传输成功与真实缓存收益。

### 建议三：增加统一拒绝日志

在 Prefill 作出 admission 拒绝决定时统一记录，例如以下为建议格式，并非现有日志：

```text
writeback_rejected request_id=... unique_key=...
stage=admission reason=prefix_not_device
error_code=8309 group_type=full tier=host
required_prefix_blocks=4 failed_block_index=2
remaining_budget_ms=...
```

预期的缓存 miss、低层驻留拒绝采用采样 INFO；容量问题采用限频 WARNING；协议或内部不变量错误单独告警。指标全量、日志采样；request_id、unique_key、完整错误文本和 block key 不放入指标标签，避免高基数。

### 建议四：明确单侧统计和指标迁移口径

1. 先新增 `rejected_qps` 和结构化分类，保留旧 `failed_qps` 口径，避免已有告警突然下降。过渡期明确 rejected 是旧 failed 的子集，不能将二者相加。
2. 看板、告警迁移完成后，再让 failed 只表示执行失败，或改用带 outcome 标签的统一结果指标；明确切换时间和历史数据不可直接比较的边界。
3. Prefill 拒绝在 Prefill rank0 每次 admission 决策上报一次。Decode 可记录对端结果，但拒绝率只用 Prefill 侧统计，不能将 P、D 两侧相加。
4. 拒绝率分母使用同口径的 admission 决策次数（接纳加拒绝），不要直接使用两侧 writeback QPS 之和；若需要按逻辑请求去重，应另外定义重试、重复请求的计数语义。
5. 补充测试：prefix admission 拒绝与 commit 失败分别计数、P/D 不重复汇总、本地 load 已启动后失败不算拒绝、重复请求与取消分类、reason 枚举及旧指标兼容性。

代码位置（均对应本文固定审查提交）：

- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.cc:135`：现有指标标签和 failed 上报逻辑。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:307`：StartWrite 校验、reject 响应及指标上报。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:482`：allocator admission；后续本地 load、接收提交和 commit。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:174`：prefix 错误分类；commit 复用此校验。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc:1145`：Decode 侧结束上报。
- `/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/utils/ErrorCode.h:15`：内部错误码定义。

本条为可观测性设计建议，尚未实施或运行测试验证。

## 审查结论校正

此前讨论中的两点需要按本次代码核对结果理解：

1. “Prefill 本地命中较长 suffix 后，Decode 只传剩余部分”不符合当前 TCP 实现。当前发送端仍提供整个候选 suffix，接收端仅拷贝缺失部分，见第 1 条。
2. “非零 TP rank 没有独立 pin 可能导致提前复用”不列为正式缺陷。正常推理调度由 rank0 统一负责，不能仅凭非零 rank 没有本地引用就判断存在生命周期 bug。建议通过真实多进程 TP 并发淘汰测试验证。

相关代码：`/home/xinfei.sxf/work/remote_ft/rtp-llm/github-opensource/rtp_llm/cpp/normal_engine/NormalEngine.cc:784`
