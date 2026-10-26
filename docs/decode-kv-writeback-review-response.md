# Decode KV Cache 回写 Review 意见处理对照

- 对照对象：`docs/decode-kv-writeback-review-functional.md`（下称《功能》）和 `docs/decode-kv-writeback-review-code-design.md`（下称《代码设计》）。两份文档的审查版本是 `bed7c2d893`。
- 历史修改前基线：分支 `rym/feat/decode-writeback-v1.3`，commit `77c8ada15e`。
- 最新更新：2026-10-26，分支 `p2p-decode-writeback`，基于 `f7cc9bf86c` 的工作区修改。此次补充功能第 2、6 条的修复及时间预算调整，并同步此前已完成的功能第 8 条；本轮代码和文档尚未提交。
- 代码位置均相对于仓库根目录。2026-10-26 新增条目的行号以当前工作区为准；历史条目的行号可能因 rebase 和后续修改偏移，以函数名定位为准。
- 约束：尽量不改上游（`origin/feat/p2p-block-tree-cache`）原有的函数和逻辑。上游文件里本分支自己加的回写函数可以改，新增配置字段时紧挨本分支已有的回写字段。确需改动上游文件的地方，在第五节逐一说明。

## 一、总览

| 文档 | 编号 | 意见 | 处理结果 |
|---|---|---|---|
| 《功能》 | 1 | 按 Prefill 实际接收范围裁剪 Decode 的发送 routes | **已修改** |
| 《功能》 | 2 | commit 返回成功需要反映实际发布结果 | **已修改**（校验可复用端点，详细结果分类暂缓） |
| 《功能》 | 3 | SWA active tail 不足以覆盖窗口时识别无收益写回 | **删除人为构造的测试**，不改业务代码 |
| 《功能》 | 4 | 对端 client 缓存需要容量或空闲回收 | **已修改** |
| 《功能》 | 5 | 按 KV 占用给 writeback admission 设上限 | **已修改**（配置项 + 两端逐 pool 保留线） |
| 《功能》 | 6 | 明确单次 writeback 触发进程退出的上线策略 | **已修改**（超时后保活，超保留容量阈值退出） |
| 《功能》 | 7 | 缩小 Decode 源资源的引用范围 | **已修改** |
| 《功能》 | 8 | 按 group 类型放松 required prefix 的 DEVICE 约束 | **已修改**（此前已完成，本次同步状态） |
| 《功能》 | 9 | 后台 writeback 是否继续受原生成 deadline 限制 | 本轮不改，保留现有策略 |
| 《功能》 | 10 | 拒绝分类、阶段和日志 | 部分改善，剩余暂缓 |
| 《代码设计》 | 1 | 源资源注册后缺少异常回滚 | **已修改** |
| 《代码设计》 | 2 | 析构函数中无保护执行回调 | 暂缓（待加固） |
| 《代码设计》 | 3 | 持有 context 锁时调用外部回调 | 暂缓 |
| 《代码设计》 | 4 | `handleWrite()` 职责过多 | 本轮不改（大重构） |
| 《代码设计》 | 5 | 多个布尔状态组合 | 本轮不改（大重构） |
| 《代码设计》 | 6 | admission 多个输出参数封装 | 本轮不改 |
| 《代码设计》 | 7 | Prefill/Decode route 解析重复 | 本轮不改（涉及上游代码） |
| 《代码设计》 | 8 | bool 返回值与 response 错误码含义不同 | **已修改** |
| 《代码设计》 | 9 | 完成通知与轮询混用 | 本轮不改 |
| 《代码设计》 | 10 | 空 `catch (...) {}` 丢失原始失败原因 | **已修改** |

## 二、已修改的意见

### 《功能》第 1 条：按 Prefill 实际接收范围裁剪 Decode 的发送 routes

- **问题**：P 算出缺失范围后只裁剪了自己的接收 routes，下发 `WRITE` 时原样转发 D 提供的全部候选 routes。TCP 下，P 已命中的块仍会被 D 拷出、序列化并发送。
- **改法**：P 下发 `WRITE` 前，用接收 routes 的 `(route_id, layer_id, cache_key)` 集合筛选 D 的发送 routes，只保留 P 实际要接收的 key 和对应 block。两端的 `route_id` 来自同一份 plan，plan digest 已经校验一致。如果有接收 key 没被 D 的 routes 覆盖，直接拒绝本次回写，不会注册接收后空等。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:308`（`selectSendRoutes`），调用点 `:559`。
- **效果**：TCP 不再发送已命中的块。RDMA 本来只读取有接收目标的块，这里主要减少了控制面的描述信息和 MR 查询。D worker 收到空的 route 列表时直接返回成功，裁剪不会让某个 rank 报错。
- **测试**：`P2PConnectorWriteSchedulerTest.SenderRouteSelectionRejectsIncompleteKeyBlockPairs`、`ReceiverSelectsSuffixWithoutReturningRangeToSource`、`LocalLoadRunsAlongsideReceiveAndPublicationWaitsForBoth`；真实 TCP 集成测试 `DecodeWritebackIntegrationTest.ReceiverPullsOnlyMissingSuffixFromCandidateSourceBlocks`，实际发送的块数从 4 降为 2。

### 《功能》第 2 条：commit 返回成功需要反映实际发布结果

- **问题**：`insertExternalBlocks()` 调用普通 Tree store 后直接返回 OK。分配后如果并发出现同 key 的 HOST 节点或 busy 节点，storer 可能没有接纳完整 suffix，但 RPC 仍报告成功。
- **改法**：在同一把 BlockTree 锁内，`storeLocked()` 返回后调用 `validateDevicePrefixLocked(cache_keys, cache_keys.size())`，确认最终端点按各 group 的正常规则确实可以在 DEVICE 上复用。未满足时返回 `CACHE_STORE_STORE_FAILED`，错误信息为 `writeback publication incomplete: final reuse endpoint unavailable`；P 的失败响应将 `accepted_block_count` 置为 0，D 收到失败后仍按物理停止确认规则释放源资源。
- **并发语义**：不覆盖已有 HOST/busy 节点；并发重复插入但最终端点已可复用时仍算成功。失败前已安全发布的部分前缀不回滚，不能把失败响应中的 0 理解成树里完全没有新增块。
- **与原建议的差别**：本次修复成功/失败语义，没有修改上游 `storeLocked()` 的接口，也没有统计实际新增块数或新增“并发重复”等结构化结果分类；这些仍属于《功能》第 10 条的后续工作。
- **代码**：`rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:203`（`insertExternalBlocks`，发布后检查在 `:225`）；`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:739`（失败响应的接纳块数）。
- **测试**：`SingleTypeCoordinatorCacheManagerTest.ExternalInsertDoesNotOverwriteHostSuffixCreatedAfterAllocation`、`ExternalInsertDoesNotOverwriteBusySuffix` 改为断言发布失败；`P2PConnectorWriteSchedulerTest.ConcurrentBusySuffixFailsPublicationAndReleasesSource` 验证错误能传回 D、源资源最终释放。

### 《功能》第 3 条：SWA active tail 不足以覆盖窗口

> **原意见**：当前允许按 `active_tail_blocks` 回传，但下一次匹配按 sliding window 判断。已有测试中，commit 成功后 `matched_device_blocks` 仍等于原来的 `start`，没有增长。
>
> 建议 admission 时检查最终 endpoint 是否能同时满足 FULL/SWA/LINEAR 的复用条件；无法增加复用长度时，明确跳过或记录部分收益，不要仅凭传输成功判断效果。
>
> 这是已确认的收益不足场景，不等同于已确认的数据正确性错误。

- **处理**：删除测试 `SwaWindowDoesNotExpandAllocatedActiveTail`，不改业务代码。
- **原因**：这条意见唯一的证据就是这个测试。它人为把 `sliding_window_size` 从 2 个 block 改成 4 个，却没有同步调整 `active_tail_blocks`（仍是 2）。正常配置下两者是匹配的。测试本身证明的是另一件事：窗口不完整时匹配逻辑会拒绝复用，不会误命中，正确性没有问题。没有证据表明真实模型配置会进入这个场景，所以不在 admission 里增加专门判断。
- **影响**：删除后，"窗口大于 active tail"这种不匹配配置下的分配行为不再有测试覆盖。正常配置下 FULL/SWA/LINEAR 的回写分配和复用，仍由 `WritebackMultiGroupAllocatorTest` 的其余用例覆盖。

### 《功能》第 4 条：对端 client 缓存需要回收

- **问题**：P 按 D 地址列表缓存 client，D 按 P 地址缓存 `BroadcastManager`，两边都只增不删。每个缓存项都带一个完成线程和一组 gRPC 连接。滚动发布或扩缩容时地址不断变化，旧连接会一直累积。上游正向链路没有这种缓存，这是回写引入的问题。
- **改法**：新增缓存模板 `PeerClientCache`，P、D 两侧共用。每次查询时顺带回收同时满足两个条件的 client：除缓存外没人持有，且闲置超过 10 分钟。
- **安全性**：
  - 被回收的 client 不会有在途 RPC：10 分钟远长于默认的回写 RPC 超时，结果对象不持有 manager，manager 析构时会等完成队列排空。
  - 回收的 client 在释放锁之后才析构，因为析构要 join 完成线程，不能卡住其他查询。
  - 创建失败不会被缓存。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h:37`（`PeerClientCache`，TTL 常量在 `:16`）；使用处 `rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.cc:47`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:102`。
- **说明**：
  - 没有做容量上限（LRU）。地址更替后旧地址不再被使用，自然会被闲置回收；硬性上限反而可能淘汰仍在用的连接。
  - 10 分钟 TTL 是固定常量，不是由代码保证大于 RPC 超时，前提是回写相关超时远小于 10 分钟（默认 5 秒）。即使有人把超时配到 10 分钟以上，后果也只是回收时等在途 RPC 结束，不会访问已释放的对象。`get()` 只在 control 线程池和 kickoff 线程上调用，不会在 client 自己的完成线程上调用，所以不存在线程 join 自己的问题。
- **测试**：`P2PWriteRpcUtilTest.PeerClientCacheEvictsOnlyIdleUnheldClients`。

### 《功能》第 5 条：在途回写的 KV 上限

- **问题**：两端原本没有按 KV 占用限制在途回写。P 变慢或失联时，D 必须继续持有尚未确认传输停止的源块，不能仅凭业务超时释放；P 侧的回写分配也会和前台 prefill 抢块。
- **改法**：新增配置 `p2p_writeback_kv_reserve_ratio`，默认 0.1，合法范围 [0, 1]，0 表示不限制。两端统一一条规则：**每个物理 block pool 的可用块，都不低于「比例 × 该 pool 总块数」时才回写**。可用块 = 空闲块 + 只被缓存引用、可以淘汰的块。
  - **按物理 pool 分别检查**：不同 pool 之间不能互借块。如果把各 pool 的余量相加，FULL pool 已经耗尽时，SWA pool 的余量也会让检查通过，而前台请求实际上拿不到 FULL 块。多个 group 共用同一个 pool 时只检查一次。
  - **D 侧**：`KVCacheManager::asyncWriteBack` 选完源块后检查，用的是"本请求释放之后"每个 pool 的可用块。由 `DeviceBlockPool::availableBlocksAfterRelease` 在 pool 锁内按引用计数计算：只有本请求持有该块的全部请求引用、而且没有 LOAD/STORE/EVICTION 引用的块，释放后才计为可用；仍被其他请求共享的 prompt 块、回写要保留的块都不计入。低于保留线就跳过本次回写（skip 原因 `kv_reserve`），本地释放照常进行。这样长 prompt 请求不会因为 prompt 块还没释放就被误判跳过，也不会因为共享 prompt 被误算成可释放而放行。持有中的源块始终不算可用，持有越积越多时可用块自然下降，降到保留线以下就不再接新的回写，不需要额外计数。
  - **P 侧**：`handleWrite` 在 `admitWriteBackDecodeCache` 分配完成后检查每个 pool 的实际剩余，低于保留线就以 `MALLOC_FAILED` 拒绝，并走已有的失败清理路径释放刚分配的块。分配后再检查，pool 里已经反映了这次回写在各个 group、各个 CP rank 上的真实占用，也能看到并发准入的其他回写；分配前估算做不到这些。P 已经拥有全部块、无需接收也无需加载时直接成功返回，不做这项检查，不会把一次空操作报成失败。
  - 比例越界或为 NaN 时，两端 scheduler 的 `init()` 失败，和非法超时配置的处理方式一致。不做这个校验的话，负数转成 `size_t` 是未定义行为。
- **与原建议的差别**：原文建议按字节数、请求数和显存水位三个维度限制。这里只用一个"可用块数"的保留线：同一 pool 内块数和 KV 字节数成正比，而"不能挤占前台"正好可以表达为"始终保留一部分可用块"。没有单独限制请求数，因为在途请求数已经受线程池队列和保留线间接约束。
- **为什么不是"D 限持有总量、P 限剩余量"**：这是最初讨论的方案。但同一个比例调大时，D 会变松而 P 会变严，方向相反，所以两端统一成保留线语义。
- **软水位**：检查只代表检查那一刻的状态。检查通过之后，其他请求仍可能继续申请块，所以保留线不是硬配额，不能保证任何时刻都留有固定比例。
- **代码**：
  - 配置：`rtp_llm/cpp/config/ConfigModules.h:404`、`rtp_llm/server/server_args/cache_store_group_args.py:26`、`rtp_llm/ops/libth_transformer_config.pyi:201`、pybind 与 pickle `rtp_llm/cpp/pybind/ConfigInit.cc`（字段数 28→29）。
  - 判断函数：`rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.cc:31`（检查当前可用）、`:35`（检查释放后可用），两者共用 `:19` 的逐 pool 比较；释放后可用块的计算在 `rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.cc:116`。
  - 调用处：D 侧 `rtp_llm/cpp/cache/KVCacheManager.cc:1408`；P 侧 `rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:550`。
  - 启动校验：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h:18`，在 `P2PSchedulerPrefillWrite.cc:85`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:41` 使用。
- **测试**：
  - 比例校验：`P2PWritebackKvReserveTest.RatioMustBeWithinZeroAndOne`（含 NaN）、`P2PConnectorWriteSchedulerTest.OutOfRangeKvReserveRatioFailsInit`。
  - P 侧：`P2PConnectorWriteSchedulerTest.ReceiverRejectsWritebackThatWouldEatKvReserve`、`ReceiverChecksEachIndependentPoolAndReleasesRejectedTargets`、`FullyCachedHandshakeNeedsNoKvReserve`。
  - D 侧（真实 allocator 集成测试）：`DecodeWritebackIntegrationTest.KvReserveSkipsWritebackAndPreservesLocalRelease`、`KvReserveDoesNotCreditPromptHeldByAnotherRequest`、`KvReserveCreditsReleasedLongPromptBeforeWriteback`、`DecodeWritebackIndependentPoolsIntegrationTest.KvReserveChecksEachPoolBeforeHoldingSuffix`。
  - Pool：`DeviceBlockPoolTest.AvailableAfterReleaseExcludesSharedAndActiveTreeReferences`、`AvailableAfterReleaseCountsRepeatedReferencesOnce`。
  - 配置：`rtp_llm/test/engine_config_test.py` 的 pickle 往返。
- **已知后续**：P 因保留线拒绝时，D 会按普通失败打一条 `writeback failed` WARNING。P 持续紧张时日志会偏多，这属于《功能》第 10 条的分类问题，留到那条一起处理。

### 《功能》第 6 条：单次超时不再直接触发进程退出

- **问题**：原来到了 `deadline + p2p_lease_query_timeout_ms` 仍未确认停止，就直接 `abort()`。一次控制 RPC 丢失也可能让整个实例退出。
- **改法**：对齐 rebase 后上游正向链路的保活策略。业务超时或失败后继续取消、查询停止状态，不再以固定时间作为退出条件；每次控制 RPC 使用独立的有界预算，失败后退避重试。`p2p_lease_query_timeout_ms` 不再表示资源保活的最终期限。
- **释放条件不放松**：D 必须确认所有 sender 已停止且 P 已停止接收，P 必须确认 receiver 和本地 load 都停止。没有确认就保留 KV 引用，不能超时强制释放。
- **容量保护**：两侧共用 `WritebackRetainedKv`，复用已有配置 `p2p_kv_leak_exit_threshold_percent`（默认 30，合法范围 1～100）。按 `(cache_tag, block_id)` 去重、按各 group 的块字节数累计失败后保留的 DEVICE KV，严格超过本侧拓扑 DEVICE KV 总容量的配置百分比时才触发容量保护退出；同时记录保留量和恢复日志。
- **与准入保留线的区别**：《功能》第 5 条的 `p2p_writeback_kv_reserve_ratio` 仍控制是否接纳新回写；本条处理失败后已经不能安全释放的资源。没有新增“任何一次失败后全局暂停回写”的开关。
- **生命周期**：完成态 D context 从 checker 移除，晚到或重复请求仍由 worker 的 tombstone 拦截；P 只单独扫描仍在保活的资源，避免扫描全部已完成请求。P 请求和 worker 的去重记录按 `p2p_cancelled_keys_ttl_ms` 保留，不使用单次查询超时充当 TTL。
- **保留的 fail-stop**：本次移除的是“到时间就退出”。P 的物理停止清理路径若出现无法处理的异常，原有 `catch (...)` 中的 `abort()` 仍保留，不能在无法保证安全时继续释放资源。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.cc:65`（阈值初始化）、`:86`（去重计量及容量保护）；`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc:1347`（D checker）；`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:233`（P 保留量检查）、`:658`（物理停止清理）。
- **测试**：原来要求定时 fail-stop 的两项用例替换为 `P2PConnectorWriteSchedulerTest.MissingWorkerLeaseStatusRetainsTargetsUntilRecovery`、`MissingSenderAcknowledgmentsRetainSourceUntilRecovery`；`P2PWritebackRetainedKvTest.ThresholdDeduplicatesSharedBlocksAndUsesStrictGreaterThan` 覆盖去重、阈值边界及超限退出；`MirroredRoutesAndAllRanksRequiredBeforePublicationAndSourceRelease` 同时验证完成态移除后重复请求仍被拦截。

### 《功能》第 7 条：缩小 Decode 源资源的引用范围

- **问题**：原来对所有 group 的 `0～ready_blocks` 都加引用，包括不需要发送的 prompt 前缀和未开启 prefix reuse 的 group。
- **改法**：`KVCacheManager::asyncWriteBack` 先构造一份未加引用的快照，交给 `P2PSchedulerDecodeWrite::selectSourceBlocks`。选块只计算要发送哪些块：用发起回写时同一份 plan 和同一个范围，按 tag 汇总所有 route 会发送的逻辑位置；同一 group 的所有层共用一张块表，所以每个 group 只需对一层做一次位置到物理块的转换，转换函数和生成 routes 时用的是同一个（`LayerCacheBufferUtil::convertLayerTagForRoute`）。只保留这些块，其余位置填空洞以保留原逻辑位置，末尾的空洞截掉。随后只对保留下来的块加引用。
- **为什么不生成 routes**：选块在 decode 主循环释放请求时同步执行，如果在这里生成完整的 routes，开销是 route 数 × 层数 × 块数，而且发起回写时还要再生成一遍。现在只需 route 数 × 块数的位置汇总，加上每个 group 一次转换；完整的 routes 只在后台 kickoff 线程上生成一次。
- **为什么不会和真正发送的 routes 对不上**：选块和发起回写共用 `writeRange()` 计算按 P 的 CP 单元对齐的范围，plan 也是同一份缓存，位置汇总规则与生成 routes 时一致。即使以后两者出现偏差，生成 routes 时遇到缺失或空的块会直接报错退出，不会发送已释放的块。
- **保留的边界**：CP 对齐后跨 prompt/response 边界的整个 CP 单元仍会被引用。未开启 prefix reuse 的 group 不再持有任何块。
- **未做的部分**：没有在 P 返回命中范围后进一步提前释放引用。P 的命中范围要等 StartWrite 之后才知道，而现在持有的已经只有候选 suffix。
- **代码**：`rtp_llm/cpp/cache/KVCacheManager.cc:1403`；`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:144`（`writeRange`）、`:159`（`selectSourceBlocks`）。
- **测试**：`P2PConnectorWriteSchedulerTest.SourceSelectionKeepsOnlyCandidateBlocksAtOriginalPositions`、`SourceSelectionUsesGroupTailPoliciesAndSkipsNonReusableGroups`、`CpWritebackIncludesPromptBoundaryAndPublishesCanonicalKeys`；集成测试 `DecodeWritebackIntegrationTest.FinishedStreamReleasesBeforeTcpCopyAndNextRequestReusesSuffix`（prompt 块不再被回写引用，位置保留为空洞）。

### 《功能》第 8 条：按 group 类型检查 required prefix

- **状态**：此前已修复，2026-10-26 同步文档，不是本轮新增的策略改动。
- **改法**：`validateDevicePrefixLocked()` 通过各 group 的 `computeReuseBlockCount(required_prefix_blocks)` 决定检查范围：FULL 检查完整 prefix，SWA 检查所需窗口，LINEAR 检查末端状态。只要求这些实际需要复用的块完整、稳定且位于 DEVICE，不再要求 SWA/LINEAR 的所有历史块都在 GPU。
- **边界**：真正需要的窗口或末端状态如果缺失、位于低层或正处于迁移中，仍然拒绝。admission 与 commit 前的 prefix 校验共用该函数；本轮《功能》第 2 条新增的 commit 后校验也使用同样的 group 规则，没有重新引入全链 DEVICE 限制。普通 Tree matching/store 逻辑不改。
- **代码**：`rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc:173`（`validateDevicePrefixLocked`）。
- **测试**：`WritebackMultiGroupAllocatorTest.SparsePrefixUsesGroupReuseRangesAndPublishesOnlyFilledSuffixBlocks`、`BusySwaWindowRejectsAdmissionWithoutTakingReferences`、`WritebackMissingGroupAllocatorTest.RejectsAdmissionWithoutTakingReferences`。

### 《代码设计》第 1 条：源资源注册后缺少异常回滚

- **问题**：源资源交给 worker 之后才创建 context。如果中途抛异常，没有任何对象负责释放源资源，worker 的清理逻辑也只回收已释放源资源的条目，这批 KV 会被永久占住。
- **改法**：`P2PSchedulerDecodeWrite::asyncWrite` 中，从创建 context 到回写任务排上队之间如果抛异常：context 已创建就走 `finishWithoutTransfer` 释放，否则直接释放源资源，然后把原异常继续抛给上层。上层 `KVCacheManager::asyncWriteBack` 会记录日志并返回空。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:234`（释放逻辑），`:270`（catch）。
- **测试**：未加。要触发这个路径得人为制造内存分配失败，需要引入仅供测试的注入代码，没有这样做。

### 《代码设计》第 8 条：bool 返回值与 response 错误码含义不同

- **问题**：D 的 `writePerRank()` 写入失败时仍返回 `true`，参数非法的分支又返回"取消是否成功"，和 P 侧以及正向链路"返回值表示业务是否成功"的约定不一致。
- **改法**：成功返回 `true`，失败一律返回 `false`。原有的取消并等待传输停止的逻辑不变；传输是否停止仍由 lease 查询确认，不依赖这个返回值。
- **错误码不会丢**：返回 `false` 时 gRPC 状态非 OK，但 `LocalRpcServer` 会把 response 里的业务错误码放进 `ErrorDetailsPB`（`rtp_llm/cpp/model_rpc/LocalRpcServer.cc:1161`），P 侧 `BroadcastManager` 再解析回来（`rtp_llm/cpp/model_rpc/RpcErrorCode.h:248`），P 仍能拿到 D 侧的具体失败原因。
- **与原建议的差别**：原文建议内部改为结构化结果类型。这里选择和正向链路保持一致，没有新增类型。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.cc:320`、`:375`。
- **测试**：`P2PConnectorWriteWorkerTest` 中相关断言由 `EXPECT_TRUE` 改为 `EXPECT_FALSE`；新增 `InvalidSenderPeerIsRejectedBeforeSending`、`RejectedWriteFencesLateValidRetry`；`P2PConnectorWriteRpcTest.SenderWriteAndCancelResponsesWaitForEveryPhysicalSend` 改为校验 gRPC 状态里带有错误详情。

### 《代码设计》第 10 条：空 `catch (...) {}` 丢失原始失败原因

- **问题**：5 处 `catch (...) {}`（P 侧 3 处，D 侧取消流程 2 处）吞掉异常，最后只剩一句笼统的 "broadcast failed"。
- **改法**：新增 `callForResult()`：执行调用，出异常或返回空时返回空，并把原因（`what()`、"unknown exception" 或 "no result"）写回调用方。5 处统一使用它。
  - P 侧接收、发送两个阶段的失败信息带上阶段名和原因，随 StartWrite 响应返回 D，D 记录时带 `unique_key`。
  - P 侧停止确认的 control 失败、D 侧取消失败会重试，每次失败打一条带 `unique_key` 的 WARNING。
  - 第一条根因仍由 `fail()` 保留，不会被后续清理错误覆盖。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h:20`；`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:486`、`:662`；`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc:1037`。
- **测试**：`P2PWriteRpcUtilTest.CallForResultKeepsFailureCause`。

## 三、未修改的意见及原因

每条先引用原意见（较长的条目为节选，完整内容见原文对应章节），再说明不改的原因。

### 暂缓，后续再处理

#### 《功能》第 10 条：将 writeback 拒绝从通用失败中独立提取，补充阶段、错误分类与日志

> **原意见（节选，完整内容见原文第 10 节）**：当前除显式 skip 分支外，只要 `error.hasError()`，就会计入 `rtp_llm_p2p_writeback_failed_qps`，admission 拒绝和执行失败混在一起。……
>
> - 建议一：新增 `rtp_llm_p2p_writeback_rejected_qps`，按 skipped / rejected / failed / no_transfer 区分结果，明确接纳边界。
> - 建议二：在内部结果及 RPC 响应中增加明确的 outcome、stage、reason，使用结构化分类，不解析错误字符串。
> - 建议三：在 Prefill 作出 admission 拒绝决定时统一记录拒绝日志。
> - 建议四：先新增 rejected 指标并保留旧 failed 口径，明确单侧统计和指标迁移口径。

**原因**：已部分改善。`d1f462f3dd` 增加了 D 侧 `writeback failed` 日志（含 `unique_key` 和错误）；本轮让失败信息带上阶段和原因（《代码设计》第 10 条），并给保留线拒绝分配了独立的 skip 原因 `kv_reserve` 和错误码 `MALLOC_FAILED`。剩下的结构化拒绝分类、新指标和口径迁移涉及指标定义和看板告警，范围较大，后续单独处理。

#### 《代码设计》第 2 条：不应在析构函数中无保护地执行用户回调

> **原意见**：`DoneNotifier::~DoneNotifier()` 调用 `dispatchDoneCallbacks()`，后者直接执行注册的 callback。
>
> 析构函数默认是 `noexcept`，任意回调抛异常都会触发 `std::terminate()`，后续回调也不会执行。建议逐个隔离回调异常，或者把回调分发移出析构函数，并明确回调的异常契约。

**原因**：风险成立，回调接口本身没有"不抛异常"的约定，属于待加固项。暂缓的原因是：目前生产调用链上注册的回调都是内部函数，没有找到会抛异常的实际触发者；而同一个 `DoneNotifier` 也用于上游的正向 read context（`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h:159`），只改回写一侧会和正向不一致，两边一起改又要动上游代码。建议和上游合并后统一加固。

#### 《代码设计》第 3 条：持有 context 锁时调用外部回调，扩大锁依赖

> **原意见**：`finishLocked()` 在持有 `mutex_` 时执行 `on_released_()`，该回调会进一步取消 group、释放 allocator 引用；同一临界区还会上报指标。
>
> 现在没有确认死锁，但任何后续回调重入 context 都可能出问题。建议锁内完成状态转换并取出回调和待释放对象，锁外执行释放、通知和指标上报。清理未完成时，不要提前对外宣称资源已释放。

**原因**：已核对锁顺序：context 锁 → task group 锁 → allocator 锁，没有找到反向加锁的路径，不是已确认的死锁。要改对需要很小心：锁内只做状态转换，锁外执行释放，同时不能在清理完成前对外标记"已释放"。建议出现实际重入需求时，和《代码设计》第 5 条的状态整理一起做。

### 本轮不改

#### 《功能》第 9 条：明确后台 writeback 是否应继续受原生成 deadline 限制

> **原意见**：writeback 在生成完成后发起，却使用 `min(now + writeback_timeout, 原请求 deadline)`。
>
> 例如生成完成时距离原 deadline 只剩 100ms，即使配置 writeback timeout 为 5 秒，写回仍只有约 100ms。建议将后台写回预算与生成预算的关系明确配置；至少增加"发起时剩余预算"指标，避免把这种失败误判为网络慢。

**原因**：当前仍有意让回写的业务预算受原生成 deadline 限制，本轮没有解耦这两种预算。时间预算跨机器传递和本地计时方式已经修复，见第四节；这不等于给回写重新分配一段不受原请求限制的预算。另外，业务预算耗尽不等于资源可以释放，物理停止未确认时仍按《功能》第 6 条继续保活。独立预算及“发起时剩余预算”指标留待后续评估。

#### 《代码设计》第 4 条：handleWrite() 承担过多职责，异常路径难以审查

> **原意见**：这个函数约 360 行，同时处理请求校验、注册、资源分配、本地 load、receiver barrier、发送、commit、取消和最终释放，还通过多个 `[&]` lambda 共享局部变量。
>
> 修改一个变量的生命周期，需要理解整个函数。建议提取请求上下文对象，按 `admit → prepare → transfer → commit → drain` 拆分阶段，把资源和清理责任归到对象上。

**原因**：可读性意见合理，但这是重写整个请求生命周期，容易引入清理顺序的回归。功能稳定、和上游合并之后再按阶段逐步拆分更稳妥。

#### 《代码设计》第 5 条：状态由多个布尔量组合，合法状态和转换约束不明确

> **原意见**：Context 同时维护 `kickoff_started_`、`cancelled_`、`done_`、`released_`、`transfer_stopped_`、`peer_pending_` 等状态。
>
> 尤其 `done()` 可以已经为真，但物理传输和资源释放尚未结束，调用方容易误用。建议分别建模"业务结果"和"传输清理状态"，集中实现状态转换，并为 `released ⇒ 双端已停止` 等不变量加断言。
>
> 不建议简单把所有维度压成一个巨大的 enum；业务终态和资源清理状态需要保持明确区分。

**原因**：业务终态和物理传输是否停止本来就是两个维度，"done 为真但资源仍需保活"是必要语义，原意见也同意不压成一个 enum。按原意见重新建模状态机风险大，收益主要在可读性；可以后续先补不变量断言。

#### 《代码设计》第 6 条：admission 的多个输出参数应封装成一个有所有权的结果

> **原意见**：`admitWriteBackDecodeCache()` 同时修改 `resource`、`start_block`、`load_context`，调用方还必须知道哪些对象需要 commit、abort，以及应该按什么顺序释放。
>
> 建议返回 `StatusOr<WritebackAdmission>`，将资源、范围和 load context 封装在一起，减少"结果只初始化了一半"或清理遗漏的可能。析构只能安全回滚未提交部分，已开始传输的资源仍需走 drain。

**原因**：封装成返回对象会更清晰，但要调整所有权接口和所有调用方，而且目前没有由此导致的功能错误，收益低于风险。适合和《代码设计》第 4 条的拆分一起做。

#### 《代码设计》第 7 条：Prefill/Decode 的 route 解析和校验存在重复实现

> **原意见**：两端分别实现 partition/slice 检查、layer 范围校验、key/block 数量校验、重复 key 检查和 `LayerCacheBuffer` 构造。
>
> 后续增加一种 route 格式时，很容易只改了一侧。建议抽取公共的"解析并验证 worker route"函数，返回经过验证的内部类型，仅保留发送侧 endpoint 解析等方向差异。

**原因**：重复的一方是上游正向链路的 `P2PConnectorPrefill`，合并必须改上游代码，按约定暂不处理。双方已共用 `RouteCodec`、`buildWriteUnits` 等公共组件。

#### 《代码设计》第 9 条：完成通知与轮询混用，存在可避免的唤醒延迟

> **原意见**：Prefill scheduler 先检查条件，再无 predicate 地 `wait_for(10ms)`；通知如果落在检查与等待之间，会等到下一次超时才继续。worker 注册通知又用 `sleep_for(1ms)` 轮询 RPC 结果。
>
> 建议统一使用支持 deadline 的事件等待，采用同锁 predicate 或通知序号避免丢失唤醒，同时保留取消和超时唤醒。

**原因**：最坏情况是 scheduler 多等一个 10ms 的轮询周期、worker 多等 1ms，不会永久挂起。回写在后台运行，这点延迟没有实际影响，改成事件等待则要同时改动取消和超时的唤醒路径，收益低于风险。两处都是本分支代码（`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:465`、`rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.cc:579`），后续整理《代码设计》第 4 条时可以一起改。

## 四、review 意见之外的改动

以下改动不直接对应某条意见，是复核代码时为整洁或正确性做的调整：

- **合并重复计算**：D 侧"按 P 的 CP 单元对齐的回写范围"原本在选块和发起回写两处各算一遍，现在共用 `writeRange()`（`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:144`）。
- **改名**：P 侧筛选函数原名 `buildDecodeRankRoutes`，和另外两个类里"根据 plan 生成 routes"的同名函数含义不同，改为 `selectSendRoutes`。
- **统一参数**：`KVCacheManager::asyncWriteBack` 传给选块和发起回写的就绪长度统一成同一个变量 `ready_tokens`。
- **删除死代码**：去掉 `startAsyncWriteCalls` 里一处执行不到的 `blocks == 0` 分支。
- **合并保留线判断**：两个同名的 `writebackKeepsKvReserve`（按数字比较的版本和按 pool 的版本）原本分处两个头文件。按数字比较的版本只是按 pool 版本的内部步骤，现在并入 `P2PWritebackResourceUtil.cc` 的内部函数 `poolsKeepReserve`；"检查当前可用"和"检查释放后可用"两个函数共用这一个逐 pool 比较（`P2PWritebackResourceUtil.cc:19`）。`P2PConnectorConfig.h` 只保留比例校验。
- **格式**：用仓库的 `.clang-format` 只格式化改动行。

### 2026-10-26：对齐上游时间预算与本地计时

- **问题**：部分回写路径仍把远端的绝对 `deadline_ms` 与本机时钟比较；本地控制等待和清理也依赖 wall clock，机器时钟偏差或本地调时可能让请求提前失败或延后超时。
- **改法**：`StartWriteBack` 接收端按收到的 `timeout_ms` 和本地上限生成本地 deadline，不再取远端绝对时间的最小值。`P2PBroadcastClient` 为 `WRITE`、`HANDLE_WRITE`、`HANDLE_WRITE_READY` 填充已有的 `transfer_timeout_ms`，worker 据此建立本地预算；D 发送队列使用扣除排队时间后的剩余预算。复用已有协议字段，没有增加 protobuf 字段。
- **本地计时**：回写 context、P 请求状态和 worker task group 在进入本机时锚定 steady deadline，控制等待、重试、过期判断和清理使用 steady clock；提交 RPC 时再由剩余预算生成本机 wire deadline。取消先于开始到达时，使用本地配置建立去重记录的保留时间，不按远端时间戳回收。
- **范围**：本轮调整的是回写控制与 worker group 的预算管理，没有重写底层传输 backend 的全部计时逻辑；《功能》第 9 条的原请求预算限制不变，《代码设计》第 9 条的事件等待重构也未实施。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h:18`（steady deadline）、`:24`（worker 本地 deadline）；`rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.cc:88`（扩展预算字段填充）；`P2PConnectorDecode.cc`、`P2PConnectorPrefill.cc` 的回写 worker 入口，以及 `P2PWriteWorkerUtil.cc`、`P2PWorkerDecodeWrite.cc`、`P2PWorkerPrefillWrite.cc`。
- **测试**：`ReceiverUsesRemainingBudgetDespiteEarlierRemoteClock`、`TransferBudgetDoesNotFollowWireClockChanges`、`RemainingBudgetIgnoresRemoteClockOnBothWorkers`、`QueuedSendUsesRemainingTransferBudget`、`QueuedReceiveControlUsesIndependentBoundedQueryBudget`。

### 历史自查与复核（截至 2026-10-23）

这一轮先后经过三方复核：我自己的复核、一个没参与修改的 agent 对整份 diff 的独立审查，以及你安排的 codex review。三方都没有发现 use-after-free、死锁、两端 route/key/CP 对齐不一致这类正确性问题。发现的风险和处理如下：

| 发现 | 来源 | 处理 |
|---|---|---|
| 保留比例为负数或 NaN 时，转成 `size_t` 是未定义行为 | 自查 | 已修：两端 `init()` 校验 [0, 1] |
| 选源块在 decode 主循环上生成完整 routes，开销大且与 kickoff 重复 | 独立审查 | 已修：只汇总位置，不生成 routes，见《功能》第 7 条 |
| D 侧保留线检查发生在请求释放之前，长 prompt 请求会被误判跳过 | 独立审查 | 已修：改为检查"本请求释放之后"的可用块，见《功能》第 5 条 |
| P 侧按 suffix 长度估算新分配块数：CP 分片下多算、多 group 下少算、并发时有竞争 | 独立审查 | 已修：改为分配后按实际剩余检查 |
| P 侧检查排在"无需接收"的早返回之前，P 已拥有全部块时空操作也会被拒 | 自查 | 已修：检查移到早返回之后，新增测试 `FullyCachedHandshakeNeedsNoKvReserve` |
| 不同 pool 的余量直接相加，会掩盖某个 pool 已经耗尽 | codex | 已修（codex）：逐物理 pool 检查，共用 pool 去重 |
| D 侧把仍被其他请求共享的 prompt 块算作即将释放，会高估可用容量 | codex | 已修（codex）：`DeviceBlockPool::availableBlocksAfterRelease` 在 pool 锁内按引用计数计算 |
| 两个同名保留线函数分处两个头文件，按 pool 的两个函数重复同一套逻辑；`KVCacheManager.cc` 留有不再使用的 include | 自查 | 已整理，见上方"合并保留线判断" |
| Python 类型桩 `libth_transformer_config.pyi` 缺少新字段 | 独立审查 | 已修 |
| client 回收 TTL 是固定常量，没有和超时配置关联 | 独立审查、codex | 未改：见《功能》第 4 条的说明 |
| 删除 SWA 测试后少了一块覆盖 | 独立审查 | 未改：按决定删除，影响见《功能》第 3 条 |

### 已知的附带影响

- D 侧上报的 `planned_bytes` 指标按候选 routes 计算，P 裁剪后实际发送量会更小，这个指标现在偏大。
- `writePerRank` 失败时返回非 OK 的 gRPC 状态，每个失败的 rank 会多出两条 WARNING：D 侧 `LocalRpcServer` 的 "P2P ExecuteFunction rejected"，P 侧 `BroadcastManager` 的 "broadcast rpc failed"。这与正向链路的行为一致。

## 五、涉及的上游文件

下表只列上游已有的文件；`P2PSchedulerDecodeWrite`、`P2PSchedulerPrefillWrite`、`DecodeWriteHelper`、`P2PWriteRpcUtil.h`、`P2PWritebackResourceUtil` 以及回写测试文件都是本分支自己的文件。

| 文件 | 改动 |
|---|---|
| `rtp_llm/cpp/cache/KVCacheManager.cc` | 只改本分支新增的 `asyncWriteBack`；新增一行 include |
| `rtp_llm/cpp/cache/connector/p2p/P2PConnector.{h,cc}`、`P2PConnectorDecode.{h,cc}` | 只改本分支新增的回写接口：新增 `selectWriteSource` 转发函数，修改 `writePerRank` |
| `rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.{h,cc}` | 只改本分支新增的 write context/checker：取消保活、steady deadline、保留容量检查及完成态移除；正向 read context 不改 |
| `rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.cc` | 只改本分支新增的回写 worker 入口：剩余预算转本地 deadline、取消去重时间和 ready 通知等待 |
| `rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.cc` | 在已有预算字段填充条件中增加 `WRITE`、`HANDLE_WRITE`、`HANDLE_WRITE_READY`；不改变原有 READ 分支行为 |
| `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc` | 只改本分支新增的回写校验/插入函数：按 group 的复用范围检查 DEVICE，提交后确认最终可复用端点；不改普通 Tree store/match |
| `rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h` | 在本分支已有的回写常量和字段旁，新增比例校验函数和配置字段 |
| `rtp_llm/cpp/cache/connector/p2p/BUILD` | 紧挨本分支已有条目：`components` 新增 `P2PWriteRpcUtil.h`，`connector` 新增 `P2PWritebackResourceUtil.{h,cc}` |
| `rtp_llm/cpp/config/ConfigModules.{h,cc}`、`rtp_llm/cpp/pybind/ConfigInit.cc`、`rtp_llm/server/server_args/cache_store_group_args.py`、`rtp_llm/test/engine_config_test.py`、`rtp_llm/ops/libth_transformer_config.pyi` | 紧挨本分支已有的 `p2p_writeback_*` 字段，新增 `p2p_writeback_kv_reserve_ratio` |
| `rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.{h,cc}`、`IBlockPool.h`、`test/DeviceBlockPoolTest.cc` | **上游 block pool 代码，只新增、不修改已有函数**：新增只读查询 `availableBlocksAfterRelease` 和 protected 访问器 `availableBlocksNumNoLock`，以及两个测试。放在 pool 里的原因：判断一个块释放后是否可用，依赖 pool 内部的引用计数规则（树引用在 `refcounts_` 里占 1、LOAD/STORE/EVICTION 引用不可用等），并且要在同一把锁下读取一致的计数。在 pool 外实现，就得复制这些内部规则、分多次加锁读取。 |
| `rtp_llm/cpp/cache/test/HybridTypeKVCacheAllocatorTest.cc` | 删除本分支自己添加的 SWA 测试 |

## 六、验证

### 2026-10-26：本轮验证

- 编译通过，包含生产库 `//:th_transformer` 和以下测试目标；`git diff --check` 通过。
- 10 个 C++ 测试程序共 **787 项通过**，其中含 CP 分片、FULL/SWA/LINEAR 混合 group、真实 TCP 传输及失败清理覆盖。

| 测试程序 | 结果 |
|---|---|
| `p2p_connector_scheduler_test` | 140/140 |
| `p2p_connector_worker_test` | 114/114 |
| `decode_writeback_integration_test` | 30/30 |
| `p2p_connector_test` | 45/45 |
| `single_type_kv_cache_allocator_test` | 77/77 |
| `hybrid_kv_cache_allocator_test` | 72/72 |
| `hybrid_kv_cache_allocator_cp_shard_test` | 10/10 |
| `p2p_model_rpc_test` | 108/108 |
| `transfer_plan_test` | 48/48 |
| `components_test` | 143/143 |

- **重复验证**：7 个关键 scheduler 用例重复 20 轮通过，覆盖丢失停止确认后的保活与恢复、并发 busy 发布失败、控制请求排队、时钟偏差、完成态移除及重复请求拦截；2 个 worker 预算用例重复 50 轮通过。
- **模型 smoke**：Qwen2.5-0.5B，BF16/TCP、DEVICE-only、CP1，`P_TP1/D_TP2` 和 `P_TP2/D_TP1` 两组均通过，每组 3 个检查全部成功，包含生成后回写及后续命中。服务已退出，0～3 卡恢复测试前的空闲基线。
- **验证边界**：本轮未重跑 DeepSeek-V4 整模 CP smoke，也未运行 RDMA 整模 smoke；下方 DeepSeek-V4 记录属于历史版本，不作为本轮改动的整模验收。
- **已知测试抖动**：上游 `P2PConnectorSchedulerTest.AsyncRead_ReturnFalse_BroadcastFailed` 曾在 `StartLoad` 到达前断言调用次数为 1 而失败；单独重跑整个 scheduler 测试程序后 140 项通过。没有修改该正向用例或相关正向代码。
- **证据**：编译日志 `.cache/writeback-upstream-fixes-build-20261026.log`；C++ 结果在 `.cache/writeback-accounting-20261024/` 的 `*.upstream-fixes-20261026.log`、`*.upstream-fixes-final-20261026.log` 与 `*.upstream-fixes-repeat-final-20261026.log`；最终模型 smoke 日志为 `.cache/writeback-upstream-fixes-smoke-final-20261026.log`、`.cache/writeback-upstream-fixes-smoke-tp2-tp1-final-20261026.log`。

### 2026-10-23：历史单测与集成测试

当时的代码编译通过（包括生产库 `//:th_transformer`、`//:th_transformer_config` 和以下测试目标），5 个测试程序共 **160 项全部通过**：

| 测试程序 | 范围 | 结果 |
|---|---|---|
| `p2p_connector_scheduler_test` | `P2PConnectorWriteSchedulerTest.*`、`P2PConnectorAsyncWriteContextTest.*`、`P2PWriteRpcUtilTest.*`、`P2PWritebackKvReserveTest.*` | 68/68 |
| `p2p_connector_worker_test` | `P2PConnectorWriteWorkerTest.*`、`P2PConnectorWriteRpcTest.*` | 41/41 |
| `device_block_pool_test` | 全部 | 19/19 |
| `hybrid_kv_cache_allocator_test` | `*Writeback*` | 6/6 |
| `decode_writeback_integration_test` | 全部（真实 TCP 传输，含 CP2→CP1 和独立 pool） | 26/26 |

另外用重新编译的 `libth_transformer_config.so` 验证了新配置字段的 pickle 往返（29 个字段）和 `to_string` 输出。

### 2026-10-22～23：历史真实模型 smoke

- **配置**：正式验收配置。DeepSeek-V4-Flash-MTP，P TP2/CP2（EP2）、D TP1（DP2/EP2），TCP、DEVICE-only、MTP；prompt 2016 tokens、生成 33 tokens；保留比例为默认值 0.1。
- **验收项**：P 发布新 cache key；P-only 和 PD 两种路径的热请求都复用 2048 tokens（冷请求为 0），且热、冷输出一致；生成部分复用 32 tokens；D 无终态错误；没有出现保留线拒绝或跳过；本轮新增的告警日志都没有出现。
- **库校验**：每轮都用当时 `bazel-bin` 的库做固定副本，并核对每个服务进程实际加载的是这些副本。

| 轮次 | 库（`libth_transformer.so`） | 覆盖的改动 | 结果 | 证据目录（`.cache/claude-smoke-20261022/attempts/`） |
|---|---|---|---|---|
| 1 | 10-22 19:39，`44d06564…` | 前四轮修复，不含独立审查后的调整 | 通过 | `writeback-20261022-202416-1316943` |
| 2 | 10-22 20:46，`1e129c87…` | 加上独立审查后的 3 处调整和 P 侧检查顺序修正 | 通过 | `writeback-20261022-205112-1366795` |
| 3 | 10-23 最终，`3cd49099…` | 加上 codex 的逐 pool / 精确释放修复和当轮整理，不含 10-26 修复 | 通过 | `writeback-20261023-112504-1477358` |

已知的既有现象（与 10-21 的通过记录一致，与本轮改动无关）：D 启动时有 832 条 `CacheTopology layer=0 does not own tag=hca_kv` 断言日志；服务退出时 P 的一个 rank 在 kmonitor 指标上报中 SIGABRT。

## 附：《代码复用》文档

`docs/decode-kv-writeback-review-code-reuse.md` 讨论的是回写与原有正向链路之间的重复实现，大部分合并要改上游代码。按约定本轮不处理。只在本分支文件内部的重复（D/P 两侧对称的 `planFor`，以及 `buildDecodeRankRoutes` 与 `buildPrefillRankRoutes`）可以在不碰上游的前提下合并，待确认后再做。
