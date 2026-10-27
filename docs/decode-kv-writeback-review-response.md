# Decode KV Cache 回写 Review 意见处理对照

- 对照对象：`docs/decode-kv-writeback-review-functional.md`（下称《功能》）和 `docs/decode-kv-writeback-review-code-design.md`（下称《代码设计》）。两份文档的审查版本是 `bed7c2d893`。
- 历史修改前基线：分支 `rym/feat/decode-writeback-v1.3`，commit `77c8ada15e`。
- 最新更新：2026-10-27，分支 `p2p-decode-writeback`，本次提交的父提交为 `5482bddc36b89e0a392fc62ae7aa2add2ba91a63`。本次包含回调隔离、非阻塞取消、统一 lease 查询、TTL 恢复、混版本停止证明、CP 元数据保活及功能第 9 条的独立回写预算修复；功能第 2、6 条及跨机器时间预算修复已在父提交中。按本轮 review 决定撤回结构化结果协议、分类指标、scheduler util 抽取、worker 增量清理/墓碑索引以及与回写无直接关系的 grammar 停服修复，并精简重复测试。第 9 条完成并交付 review 后，按最新要求提交并推送当前分支；文档仅包含本 response，不包含其他本地设计文档。
- 代码位置均相对于仓库根目录，历史行号可能因 rebase 和后续修改偏移，以函数名定位为准。第六节明确区分当前工作区验证和历史记录。
- 约束：尽量不改上游（`origin/feat/p2p-block-tree-cache`）原有的函数和逻辑。上游文件里本分支自己加的回写函数可以改，新增配置字段时紧挨本分支已有的回写字段。确需改动上游文件的地方，在第五节逐一说明。

## 一、总览

| 文档 | 编号 | 意见 | 处理结果 |
|---|---|---|---|
| 《功能》 | 1 | 按 Prefill 实际接收范围裁剪 Decode 的发送 routes | **已修改** |
| 《功能》 | 2 | commit 返回成功需要反映实际发布结果 | **已修改**（校验可复用端点，未统计树内实际新增块数） |
| 《功能》 | 3 | SWA active tail 不足以覆盖窗口时识别无收益写回 | **删除人为构造的测试**，不改业务代码 |
| 《功能》 | 4 | 对端 client 缓存需要容量或空闲回收 | **已修改** |
| 《功能》 | 5 | 按 KV 占用给 writeback admission 设上限 | **已修改**（配置项 + 两端逐 pool 保留线） |
| 《功能》 | 6 | 明确单次 writeback 触发进程退出的上线策略 | **已修改**（超时后保活，超保留容量阈值退出） |
| 《功能》 | 7 | 缩小 Decode 源资源的引用范围 | **已修改** |
| 《功能》 | 8 | 按 group 类型放松 required prefix 的 DEVICE 约束 | **已修改**（此前已完成，本次同步状态） |
| 《功能》 | 9 | 后台 writeback 是否继续受原生成 deadline 限制 | **已修改**（提交回写时开始独立预算，不再受生成 deadline 截断） |
| 《功能》 | 10 | 拒绝分类、阶段和日志 | **精简**（保留简单日志和原有指标，撤回结构化分类及协议扩展） |
| 《代码设计》 | 1 | 源资源注册后缺少异常回滚 | **已修改** |
| 《代码设计》 | 2 | 析构函数中无保护执行回调 | **已修改**（read/write 完成回调逐个隔离异常） |
| 《代码设计》 | 3 | 持有 context 锁时调用外部回调 | **已修改**（锁外释放，清理结束后才标记 released） |
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
- **与原建议的差别**：本次修复成功/失败语义，没有修改上游 `storeLocked()` 的接口，也没有统计实际新增块数或细分“并发重复”等树插入结果。请求级结构化分类已按《功能》第 10 条撤回；树内新增块数和重复插入计数同样不在本轮范围内。
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
  - 配置：`rtp_llm/cpp/config/ConfigModules.h`、`rtp_llm/server/server_args/cache_store_group_args.py`、`rtp_llm/ops/libth_transformer_config.pyi`；当前 pybind 的 `CacheStoreConfig` pickle 是 28 个字段，回写开关、超时、保留比例位于索引 25～27，读取严格要求 tuple 长度为 28（`rtp_llm/cpp/pybind/ConfigInit.cc:1180`）。
  - 判断函数：`rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.cc:31`（检查当前可用）、`:35`（检查释放后可用），两者共用 `:19` 的逐 pool 比较；释放后可用块的计算在 `rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.cc:116`。
  - 调用处：D 侧 `rtp_llm/cpp/cache/KVCacheManager.cc:1408`；P 侧 `rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:550`。
  - 启动校验：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h:18`，在 `P2PSchedulerPrefillWrite.cc:85`、`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:41` 使用。
- **测试**：
  - 比例校验：`P2PWritebackKvReserveTest.RatioMustBeWithinZeroAndOne`（含 NaN）、`P2PConnectorWriteSchedulerTest.OutOfRangeKvReserveRatioFailsInit`。
  - P 侧：`P2PConnectorWriteSchedulerTest.ReceiverRejectsWritebackThatWouldEatKvReserve`、`ReceiverChecksEachIndependentPoolAndReleasesRejectedTargets`、`FullyCachedHandshakeNeedsNoKvReserve`。
  - D 侧（真实 allocator 集成测试）：`DecodeWritebackIntegrationTest.KvReserveSkipsWritebackAndPreservesLocalRelease`、`KvReserveDoesNotCreditPromptHeldByAnotherRequest`、`KvReserveCreditsReleasedLongPromptBeforeWriteback`、`DecodeWritebackIndependentPoolsIntegrationTest.KvReserveChecksEachPoolBeforeHoldingSuffix`。
  - Pool：`DeviceBlockPoolTest.AvailableAfterReleaseExcludesSharedAndActiveTreeReferences`、`AvailableAfterReleaseCountsRepeatedReferencesOnce`。
  - 配置：`rtp_llm/test/engine_config_test.py` 的 pickle 往返。
- **拒绝观测**：保留原有错误码、错误文本及指标，不再增加结构化分类和拒绝率指标。保留线拒绝仍通过 `MALLOC_FAILED` 及明确的错误文本诊断；本轮未增加日志采样，持续拒绝时日志量仍可能偏多。

### 《功能》第 6 条：单次超时不再直接触发进程退出

- **问题**：原来到了 `deadline + p2p_lease_query_timeout_ms` 仍未确认停止，就直接 `abort()`。一次控制 RPC 丢失也可能让整个实例退出。
- **改法**：对齐 rebase 后上游正向链路的保活策略。业务超时或失败后继续取消、查询停止状态，不再以固定时间作为退出条件；每次控制 RPC 使用独立的有界预算，失败后退避重试。`p2p_lease_query_timeout_ms` 不再表示资源保活的最终期限。
- **释放条件不放松**：D 必须确认所有 sender 已停止且 P 已停止接收，P 必须确认 receiver 和本地 load 都停止。没有确认就保留 KV 引用，不能超时强制释放。
- **非阻塞取消（Claude 发起修复）**：原 D 侧 `CANCEL_WRITE` 会进入无超时的 `waitStopped()`，客户端 RPC 超时重试时，服务端 handler 仍在等待，可能持续积累。现在取消只设置状态、封闭任务组并返回，不等待物理发送结束；复用 `QUERY_LEASE_STATUS`，新增 `lease_operation=WRITE` 到发送 worker 的路由。Decode 侧统一使用 `queryLeaseStatusPerRank(unique_key, operation, response)`，按 READ/WRITE 选择 worker，共用响应填充，不再保留单独的 write 查询函数；缺失读任务仍按原逻辑报告 stopped，未知回写任务返回错误，不能作为停止证明。取消应答只表示取消已处理，D checker 必须另行取得所有发送 rank 的有效 stopped lease，且 P 也已停止，才释放源资源。正常成功响应仍沿用原有的成功完成证明。
- **停止记录过期恢复（后续自查修复）**：取消已确认后，D 原来只查询 lease；若查询持续失败跨过 TTL，非 rank 0 的已停止 worker 记录可能被清理，恢复后会一直返回 `unknown write task`。现在 lease 查询失败或返回无效证明时，清除 `worker_cancelled_`，按原有退避节奏重新发送非阻塞 CANCEL，再查询 lease；有效但仍在传输的 lease 不触发重新取消。重新 CANCEL 可以为已删除的 key 建立封闭记录，但该应答本身仍不能证明 stopped。没有缩短或取消 TTL，也不把 unknown、超时或记录缺失当成停止证明。
- **P 侧同类恢复补修**：接收侧清理也有相同遗漏：`cancel_confirmed` 一旦为 true，后续查询无效不会清除，停止记录过期后会一直查询 unknown。现在每次控制结果完成后用 `status.success` 更新该状态，无效时按原有退避重新发送 `CANCEL_HANDLE_WRITE`；该 RPC 本来就返回接收 lease，仍须所有 rank 的证明有效且 stopped、本地 load 也已停止才能释放目标 KV。有效但尚未停止的 lease 继续查询，不改变 TTL 或停止判定。
- **容量保护**：两侧共用 `WritebackRetainedKv`，复用已有配置 `p2p_kv_leak_exit_threshold_percent`（默认 30，合法范围 1～100）。按 `(cache_tag, block_id)` 去重、按各 group 的块字节数累计失败后保留的 DEVICE KV，严格超过本侧拓扑 DEVICE KV 总容量的配置百分比时才触发容量保护退出；同时记录保留量和恢复日志。
- **与准入保留线的区别**：《功能》第 5 条的 `p2p_writeback_kv_reserve_ratio` 仍控制是否接纳新回写；本条处理失败后已经不能安全释放的资源。没有新增“任何一次失败后全局暂停回写”的开关。
- **生命周期**：完成态 D context 从 checker 移除，晚到或重复请求仍由 worker 保留的停止 group 拦截；P scheduler 只单独扫描仍在保活的资源，避免扫描全部已完成请求。worker 已恢复原有定期遍历 `groups_` 的清理方式，不再压缩成独立墓碑。P 请求和 worker 的去重记录按 `p2p_cancelled_keys_ttl_ms` 保留，不使用单次查询超时充当 TTL。
- **保留的 fail-stop**：本次移除的是“到时间就退出”。P 的物理停止清理路径若出现无法处理的异常，原有 `catch (...)` 中的 `abort()` 仍保留，不能在无法保证安全时继续释放资源。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.cc:65`（阈值初始化）、`:86`（去重计量及容量保护）；`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc:1347`（D checker）；`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.cc:233`（P 保留量检查）、`:658`（物理停止清理）。
- **取消与恢复代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.cc:161`（非阻塞取消）、`:181`（发送 lease 查询）；`P2PConnectorDecode.cc:399`（worker RPC 入口）、`P2PBroadcastClient.cc:308`（扩展 WRITE 查询）；`P2PConnectorAsyncContext.cc:1023`（分离取消确认与停止证明）、`:1045`（D 查询失败后允许重新取消）；`P2PSchedulerPrefillWrite.cc:709`（P 查询失败后允许重新取消）。
- **测试**：原来要求定时 fail-stop 的两项用例替换为 `P2PConnectorWriteSchedulerTest.MissingWorkerLeaseStatusRetainsTargetsUntilRecovery`、`MissingSenderAcknowledgmentsRetainSourceUntilRecovery`；`P2PWritebackRetainedKvTest.ThresholdDeduplicatesSharedBlocksAndUsesStrictGreaterThan` 覆盖去重、阈值边界及超限退出；`MirroredRoutesAndAllRanksRequiredBeforePublicationAndSourceRelease` 同时验证完成态移除后重复请求仍被拦截。
- **新增回归**：`SenderCancelReturnsBeforeEveryPhysicalSendAndLeaseConfirmsStop`、`RepeatedCancelTimeoutsDoNotBlockHandlersOrReleaseSourceBeforeLeaseProof` 验证非阻塞取消、连续超时不积累 handler 和一次性释放；`CancelAcknowledgmentNeedsValidStoppedLeasesFromEverySender` 验证各 rank 的证明。`ExpiredSenderLeaseIsRecreatedBeforeSourceRelease` 使用真实 worker cleanup 和 RPC，推进停止记录到过期时间后验证重新取消及恢复释放；修复前失败、修复后通过，没有实际等待默认的一小时 TTL。
- **P 侧新增回归**：`RealReceiversRecreateCleanedLeaseAndWaitForEveryRankBeforeOneRelease` 通过真实 connector、worker 和 lease 清理路径让一个 rank 的停止记录过期，验证从 unknown 恢复为重新取消；另一 rank 仍未停止时继续保留目标 KV，全部停止后只释放一次且不发布失败回写的 suffix。已删除重复的纯 mock 过期用例及其辅助状态；不新增测试文件，不修改生产环境的 TTL 配置。

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

### 《功能》第 9 条：后台回写使用独立预算

- **问题**：回写在正常生成完成后提交，但 D 侧仍使用 `min(当前时间 + 回写预算, 原生成 deadline)`。生成即将到期时会挤掉后台回写时间；正常完成后若因调度延迟跨过原 deadline，回写甚至无法开始。
- **改法**：`P2PSchedulerDecodeWrite::asyncWrite()` 从提交时重新计算 `deadline = now + min(p2p_writeback_timeout_ms, load_cache_timeout_ms, INT_MAX)`，不再读取 `routing.deadline_ms`。复用已有配置，不新增开关、协议字段或指标。`load_cache_timeout_ms` 仍是传输上限；P 侧继续按收到的剩余预算与本地两个配置上限取最小值。
- **预算边界**：独立于生成请求，不等于无限制。D 的排队、plan/route 构造及握手均消耗同一份回写预算；RPC 只传剩余时间，原有本地 steady clock 和超时取消不变。超时后仍须确认两侧物理传输停止才释放 KV，不给每次重试重新分配业务预算。
- **取消语义**：stream 仍仅在无错误且状态为 FINISHED 时触发回写，已取消、失败或生成超时的 stream 不会因此补做回写。后台任务显式取消及停服 drain 也不改变。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:227`；传递剩余预算沿用 `DecodeWriteHelper::write()`，P 侧沿用 `P2PSchedulerPrefillWrite::handleWrite()` 的预算检查。
- **测试**：现有 scheduler 文件新增一项 `WritebackBudgetIsIndependentOfGenerationDeadline`，覆盖原 deadline 已过期、仅余 100ms、较远和未设置四种情况，并分别检查 writeback/load 两个配置成为上限时的行为。旧实现已复现前两种情况失败。既有排队过期、取消后保活、原请求取消不触发回写和 TCP 集成测试继续覆盖安全边界。

### 《功能》第 10 条：拒绝分类、阶段和日志

- **本轮决定**：删除未提交的 `P2PWritebackResultPB`、响应 `result` 字段、P/D 分类传播和 `classifyResult()`；不以新的内部枚举或结果类型替代。它们只用于观测，不是路由、传输、发布或安全释放的必要协议。
- **保留范围**：沿用 `error_code/error_message/stopped`、原有 `WriteSchedulerMetricsCollector` 和指标口径。P 的早期拒绝日志只记录 request_id、unique_key、原始错误；D 保留失败日志和首因，不再同步 P 的阶段及接纳分类。`model_rpc_service.proto`、`P2PConnectorMetrics.{h,cc}` 相对 HEAD 已无修改。
- **未做范围**：不新增 rejected/admission QPS、阶段标签或拒绝率统计，不改变原有 `reason` 标签。无法仅凭 D 指标准确区分准入拒绝与接纳后失败，这是本轮主动收缩的观测能力，不影响错误返回和资源保活。
- **测试**：删除纯分类测试及分类字段断言；保留参数/reserve 拒绝、本地 load 失败、接收提交失败、发布失败、首因保留以及未确认 stopped 时不得释放资源的行为断言。

### 《代码设计》第 1 条：源资源注册后缺少异常回滚

- **问题**：源资源交给 worker 之后才创建 context。如果中途抛异常，没有任何对象负责释放源资源，worker 的清理逻辑也只回收已释放源资源的条目，这批 KV 会被永久占住。
- **改法**：`P2PSchedulerDecodeWrite::asyncWrite` 中，从创建 context 到回写任务排上队之间如果抛异常：context 已创建就走 `finishWithoutTransfer` 释放，否则直接释放源资源，然后把原异常继续抛给上层。上层 `KVCacheManager::asyncWriteBack` 会记录日志并返回空。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.cc:234`（释放逻辑），`:270`（catch）。
- **测试**：未加。要触发这个路径得人为制造内存分配失败，需要引入仅供测试的注入代码，没有这样做。

### 《代码设计》第 2 条：完成回调异常隔离

- **改法**：read/write context 的完成回调统一经过 `invokeDoneCallback()`，逐个捕获 `std::exception` 和未知异常并记录日志；单个回调失败不影响后续回调，不因 `DoneNotifier` 析构而终止进程。已完成后再注册的回调也用同样规则。
- **范围**：正向 read context 只改两处回调调用点，不改传输、取消或释放协议；没有调整全局 `AsyncContext` 接口或其他实现。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc`（`invokeDoneCallback`、两类 context 的 `onDone/dispatchDoneCallbacks`）。
- **测试**：`ThrowingCompletionCallbacksDoNotStopRemainingCallbacks` 同时覆盖 read/write、标准/非标准异常，以及终态后注册回调。

### 《代码设计》第 3 条：锁外执行释放和指标上报

- **改法**：`tryFinishLocked()` 只确认双端停止并设置 `release_ready_`；退出状态锁后，`finishRelease()` 用 `release_in_progress_` 保证只有一个清理者，在锁外调用 `on_released_`、销毁所持资源及 RPC 引用。清理成功后才设置 `released_`，通知完成并在锁外上报指标。
- **重入与并发**：释放回调可以读取 context 状态、注册完成回调或重入检查/取消；不会重复释放，也不会把已确认的结果改成取消。清理进行中 `resourceHoldPending()` 仍为 true。失败请求仍允许先返回业务终态，不改变“业务完成不等于资源已释放”的原协议。
- **异常边界**：释放回调异常时记录错误，保留回调及所有权，后续 checker 重试，不谎报 released；当前内部释放操作允许重复 cancel/release。重试按 100ms 唤醒，保留资源参与容量阈值检查。指标异常单独记录，不阻断完成回调。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.{h,cc}`（`DoneNotifier`、`tryFinishLocked`、`finishRelease`、`nextWakeupMs`、checker）。
- **测试**：`ReleaseCallbackMayReenterAndKeepsHoldUntilCleanupCompletes`、`FailedReleaseRetainsOwnershipAndRetriesBeforeReportingReleased`；原 deadline 和阻塞 kickoff 用例改为等待 checker 收尾或确认 `resourceHoldPending()==false`，避免将“进入释放回调”当成“已释放完成”。

### 《代码设计》第 8 条：bool 返回值与 response 错误码含义不同

- **问题**：D 的 `writePerRank()` 写入失败时仍返回 `true`，参数非法的分支又返回"取消是否成功"，和 P 侧以及正向链路"返回值表示业务是否成功"的约定不一致。
- **改法**：成功返回 `true`，失败一律返回 `false`。当前版本在失败时发起非阻塞取消并返回业务错误，不在 worker RPC 内等待物理停止；D scheduler 独立查询 lease 并保留源 KV，详见《功能》第 6 条。不能用业务错误返回或取消应答代替停止证明。
- **错误码不会丢**：返回 `false` 时 gRPC 状态非 OK，但 `LocalRpcServer` 会把 response 里的业务错误码放进 `ErrorDetailsPB`（`rtp_llm/cpp/model_rpc/LocalRpcServer.cc:1161`），P 侧 `BroadcastManager` 再解析回来（`rtp_llm/cpp/model_rpc/RpcErrorCode.h:248`），P 仍能拿到 D 侧的具体失败原因。
- **与原建议的差别**：原文建议内部改为结构化结果类型。这里选择和正向链路保持一致，没有新增类型。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.cc:320`、`:375`。
- **测试**：`P2PConnectorWriteWorkerTest` 中相关断言由 `EXPECT_TRUE` 改为 `EXPECT_FALSE`；新增 `InvalidSenderPeerIsRejectedBeforeSending`、`RejectedWriteFencesLateValidRetry`；RPC 用例现为 `SenderCancelReturnsBeforeEveryPhysicalSendAndLeaseConfirmsStop`，同时校验 gRPC 错误详情和独立 lease 停止证明。

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

### 本轮不改

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
- **范围**：此阶段调整的是回写控制与 worker group 的预算管理，没有重写底层传输 backend 的全部计时逻辑。原生成预算限制已在 2026-10-27 按《功能》第 9 条另行解除；《代码设计》第 9 条的事件等待重构仍未实施。
- **代码**：`rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h:18`（steady deadline）、`:24`（worker 本地 deadline）；`rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.cc:88`（扩展预算字段填充）；`P2PConnectorDecode.cc`、`P2PConnectorPrefill.cc` 的回写 worker 入口，以及 `P2PWriteWorkerUtil.cc`、`P2PWorkerDecodeWrite.cc`、`P2PWorkerPrefillWrite.cc`。
- **测试**：`ReceiverUsesRemainingBudgetDespiteEarlierRemoteClock`、`TransferBudgetDoesNotFollowWireClockChanges`、`RemainingBudgetIgnoresRemoteClockOnBothWorkers`、`QueuedSendUsesRemainingTransferBudget`、`QueuedReceiveControlUsesIndependentBoundedQueryBudget`。

### 2026-10-27：混版本停止证明与 worker 清理

- 仅 StartWriteBack 的 gRPC UNIMPLEMENTED 可视为 P 从未开始接收；D 只设置远端 stopped，仍须按原条件确认本地所有发送 lease 停止。CancelWriteBack 的 UNIMPLEMENTED 可能只是取消接口或路由不兼容，不能证明已受理或在途的 StartWriteBack 停止，必须继续保留源 KV。其他 RPC 错误同样不按错误字符串猜测状态。`BroadcastResult` 只新增完成后读取原始 gRPC 状态码的只读接口，复用原广播和完成机制，不改原有正向行为。
- D/P 两侧补充“取消已确认→查询提交被控制队列拒绝→恢复”的故障注入，确认重发 CANCEL、所有 rank 停止前持续保活及单次释放。此前按变化通知驱动 worker 清理的优化已撤回，不改变非阻塞取消及停止证明要求。
- `StreamCacheResource` 在构造回写 Meta 前检查 `writebackEnabled()`；关闭或非 DECODE 角色不再进入 `asyncWriteBack`，也不再从此入口上报 disabled skip。manager 内部保留防御检查。删除无调用方的 `GenerateStream::beginTimeUs()`、`checkTimeout()`。已撤回 scheduler util 抽取，plan 缓存与 route 构造保留在 D/P 两侧原函数中，保留已有校验，不改 read 路径。
- **已撤回优化**：删除 `WriteTaskGroupCleanup`、状态变化通知及 deadline/墓碑索引，D/P worker 恢复各自的简单 `cleanup()`。定期扫描 `groups_`，超时取消，物理停止后释放 tasks；仅在本地 deadline + TTL 到期后删除 group，D 侧还要求源 KV 引用已释放。TTL 内保留 group 用于拦截重复 WRITE、源资源注册和回答 lease 查询，保留真实计数；记录过期后仍返回 unknown，由已有重发 CANCEL 流程恢复。
- **与正向的对照**：P→D 的 D 接收端通过 `P2PWorkerDecodeRead::onRecvTaskDone` 和 `queryLeaseStatus` 清理已停止 lease，提前取消记录在 `pending_cancel_keys_` 中按过期时间扫描。P 发送端通过 `ComputedLayerCacheBufferStore::removeBuffer/checkTimeout` 删除 buffer，并保留一小时的删除记录及过期队列。正向也有轻量删除记录，但没有本轮撤回的 `WriteTaskGroupCleanup`。回写对齐的是物理停止前不释放、取消封口和晚到请求拦截的安全原则；保留对象和回收触发方式并不完全一致，本轮不为统一实现引入新抽象或修改正向代码。
- 三份说明已同步接收集合裁剪、超时保活/容量阈值、角色相关 TP/CP、`CoordinatorCacheManager` 接口与验证版本；本轮测试和边界见第六节。

### 2026-10-27：真实模型停服时的两处生命周期问题

- **Decode SIGSEGV，已知问题，本分支不处理**：`LogitsProcessorFactory` 的进程级 grammar backend 持有 `MetricsReporter`，晚于 `KMonitorFactory::Shutdown()` 才析构。gdb 捕获到 reporter 的 deleter 调用已释放对象的 `monitor->Name()`。此前通过提前释放 backend 验证了修复，但该通用停服问题不是回写功能的依赖；按 review 决定撤回 `LogitsProcessorFactory::shutdown()`、`RtpLLMOp::stop()` 中的调用及对应测试，四个文件恢复到 HEAD。本分支不继续处理该问题，完整模型退出时仍可能出现该崩溃。
- **Prefill 指标线程 SIGABRT，已修复**：V4/V4.1 的 `V4Transformer.set_cp_info()` 保存了 pybind 返回的 CP 元数据引用。该引用的 `reference_internal` 保活链为 CP 元数据 → `PyAttentionInputs` → `PyModelInputs` → cache writer / P2P 回调 → `KVCacheManager`；模型退出后这些引用仍在，传输指标线程因此晚于 KMonitor 停止。原生引用计数诊断确认 P 的 writer 未析构；仅增加 GC 不能解决。为 `PyContextParallelParams` 增加 `__copy__`，两个 Transformer 保存浅拷贝快照，tensor 存储仍共享，不再保活整个请求。
- **协议保持不变**：没有把 `context_parallel_info` getter 改成值返回，原有 `inputs.context_parallel_info.xxx = ...` 写法仍然有效；仅长期保存元数据时创建快照。没有新增 connector stop 框架，也没有改正向/反向传输、RDMA 接口。临时诊断日志已全部移除。
- **当前保留的代码与反例**：`models_py/bindings/OpDefs.cc`、`models_py/modules/{dsv4,dsv41}/transformer.py`。测试复用 `dsv41/test/test_native_prefill_inputs.py`；父请求 weakref 反例在修复前对两个模型均失败，修复后均通过，同时检查 tensor 不复制及原 getter 的原地修改协议。grammar 工厂与测试改动已撤回。

### 2026-10-27：测试范围精简

- **删除已撤回重构的附带测试**：`WritebackPlanCacheKeepsPeerEntriesAndDoesNotCacheInvalidTpSizes` 原用于 scheduler util 抽取，主要断言缓存指针身份。抽取已撤回，原有 plan/route 与非对称传输测试仍在，不保留这项实现细节测试。
- **删除重复模拟**：`ReceiverRecreatesExpiredLeaseBeforeReleasingTargets` 及 mock 的 `expired` 状态、`expireStoppedLease()` 已删除，保留前述真实 worker 的过期恢复用例，避免用两套实现重复验证同一协议。撤掉 `SenderExceptionCancelsRemainingUnits` 中不再需要的停清理线程操作，恢复正常后台清理环境。
- **保留必要覆盖**：READ/WRITE 统一查询、非阻塞取消、所有 rank 的物理停止证明、TTL/查询提交失败恢复、回调异常与重入、混版本 RPC、关闭/非 Decode 角色及 CP 元数据生命周期。测试分别覆盖协议入口、worker、scheduler 和真实 TCP 集成，不按文件整份撤回。
- **范围**：仅修改七个已有测试文件，没有新增测试文件。结构化分类、增量清理索引以及无关 grammar 停服测试已随对应实现撤回。

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

上游基线为 `origin/feat/p2p-block-tree-cache` 的 `d35ddab31826ec04fb13f8fc7561bc386c6fb5f3`。下表列本 review 系列涉及的上游已有文件及当前接口，不是整个分支的所有配套差异；完整分支清单可用 `git diff --name-status d35ddab318 -- rtp_llm` 获取。相对于父提交，本轮主要修改已有回写实现/测试和本 response；公共 `BroadcastManager.h` 增加只读状态访问，read context 仅隔离完成回调异常，另包含下表列出的 CP 元数据生命周期修复及 stream 回写入口精简。

| 上游已有文件（相对 `rtp_llm/`） | 当前回写相关改动 |
|---|---|
| `cpp/cache/KVCacheManager.{h,cc}`、`cpp/cache/CoordinatorCacheManager.{h,cc}` | 回写快照、资源保留、准入和提交；提交后校验可复用端点。Decode 本地不使用 Prefill 的 CP slot mapper。旧 `KVCacheAllocator` 已由上游管理器接口替代，不能再按旧文件名审查 |
| `cpp/cache/CacheConfig.{h,cc}`、`cpp/cache/MHAKVCacheSpec.h` | 实际传输 topology、基础 key 粒度和全局 KV head 元数据；这些属于此前提交的 TP/CP 适配，本轮未改 |
| `cpp/cache/connector/p2p/P2PConnector.{h,cc}`、`P2PConnectorDecode.{h,cc}`、`P2PConnectorPrefill.{h,cc}` | 回写接口/worker 分支，发送端 lease 查询、取消封口、剩余预算和注册通知 |
| `cpp/cache/connector/p2p/P2PConnectorAsyncContext.{h,cc}` | write context/checker 的保活、容量核算、TTL 恢复、锁外释放、首因保留与混版本停止证明；read 部分仅隔离完成回调异常，本轮不改正向状态机 |
| `cpp/engine_base/stream/StreamCacheResource.cc`、`GenerateStream.h` | 回写关闭或非 DECODE 角色时在构造 Meta 前返回正常释放路径；删除此前新增但无调用方的两个 stream 方法 |
| `cpp/cache/connector/p2p/P2PBroadcastClient.{h,cc}` | 回写的异步 WRITE/HANDLE_WRITE lease 查询、预算填充与首因汇总；不改变 READ 分支行为 |
| `cpp/model_rpc/BroadcastManager.h`、`cpp/model_rpc/test/BroadcastManagerTest.cc` | 复用已有广播、首因与完成通知；本轮只新增 `rpcStatusCode()` 及完成前/越界返回 UNKNOWN 的测试，避免解析错误文本或另写 RPC 客户端 |
| `cpp/cache/connector/p2p/P2PConnectorConfig.h`、`P2PConnectorMetrics.{h,cc}`、`cpp/model_rpc/proto/model_rpc_service.proto` | 此前已提交的回写配置、RPC 与指标；本轮结构化结果及新增分类指标已撤回，proto 和 metrics 文件相对 HEAD 无修改 |
| `cpp/cache/connector/p2p/P2PKeyUtil.h`、`P2PWorkerRoute.h`、`plan/{ShardLayout.h,ShardLayoutFactory.h,KVCacheTransferPlanner.cc}` | 回写 key/route、反向 TP/CP plan 与按角色解释的布局；此前已提交，本轮未改 |
| `cpp/cache/block_tree_cache/BlockTreeCache.{h,cc}` | 回写前缀准入、DEVICE 复用范围检查、插入后端点确认；不是改普通 store/match 的成功语义 |
| `cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.{h,cc}`、`IBlockPool.h` 及 `test/DeviceBlockPoolTest.cc` | 新增同锁读取的 `availableBlocksAfterRelease`/`availableBlocksNumNoLock` 和测试，避免在回写代码中复制 pool 引用计数规则 |
| `cpp/config/ConfigModules.{h,cc}`、`cpp/pybind/ConfigInit.cc`、`ops/libth_transformer_config.pyi`、`server/server_args/cache_store_group_args.py`、`test/engine_config_test.py` | 回写开关、超时、保留比例及当前 28 字段 pickle；本轮未改配置 ABI |
| `cpp/cache/connector/p2p/BUILD`、`cpp/cache/connector/p2p/test/{BUILD,P2PBroadcastClientTest.cc}`、`cpp/cache/test/HybridTypeCoordinatorCacheManagerTest.cc` | 回写源文件/依赖、WRITE/HANDLE_WRITE 查询断言；测试引用统一使用当前 Coordinator 管理器文件，旧 allocator 测试路径不再存在 |
| `cpp/models/logits_processor/LogitsProcessorFactory.{h,cc}`、`cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc`、`cpp/models/logits_processor/test/GrammarLogitsProcessorTest.cc` | 通用 grammar 停服修复及测试已撤回；四个文件相对 HEAD 无修改，不纳入回写功能范围 |
| `models_py/bindings/OpDefs.cc`、`models_py/modules/{dsv4,dsv41}/transformer.py`、`models_py/modules/dsv41/test/test_native_prefill_inputs.py` | CP 元数据浅拷贝快照，避免长期保活整份请求；保持原 getter 的引用语义和 tensor 存储共享 |

`P2PScheduler{Decode,Prefill}Write`、`P2PWorker{Decode,Prefill}Write`、`P2PWriteWorkerUtil`、`DecodeWriteHelper`、`P2PWriteRpcUtil`、`P2PWritebackResourceUtil` 和回写专用测试均为本分支新增，不是上游原有读链路。worker 清理保留两侧原有直接实现；scheduler 的 plan 缓存和 route 构造也保留原有直接实现，没有新增清理类或 util 文件。

## 六、验证

### 2026-10-27：功能第 9 条独立回写预算验证

- **范围**：第 9 条的生产代码只修改 `P2PSchedulerDecodeWrite::asyncWrite()` 的 deadline 计算；在已有 scheduler 测试文件中补一项回归，同步本 response。没有新增配置、协议、源文件或测试文件；验证后先停下交付 review，现按要求随本轮修复一并提交。
- **反例**：先保持旧实现运行 `WritebackBudgetIsIndependentOfGenerationDeadline`，原生成 deadline 已过期和仅余 100ms 两种情况均失败，确认预算确实被截断；日志 `.cache/writeback-accounting-20261024/p2p_connector_scheduler_test.independent-budget-red-20261027.log`。
- **构建**：修改后主库及七个现有 C++ 测试目标编译通过，日志 `.cache/codex-smoke-20261027/independent-budget-build.log`。
- **回归**：scheduler 全量 155 项、TCP integration 全量 31 项，共 186 项通过，包含新增预算用例、回写超时后源 KV 保活、取消请求不触发回写、TP/CP 非对称和混合 cache group。日志在 `.cache/writeback-accounting-20261024/`，后缀 `.independent-budget-20261027.log`。`git diff --check` 通过。
- **边界**：本轮未重跑整模 DSv4、真实 RDMA 或下方全部 415 项 C++/3 项 Python 测试；下方记录属于独立预算修复前的工作区，不混记为本次验证。

### 2026-10-27：测试精简后的验证（独立预算修复前）

- **版本范围**：以 `5482bddc36` 为父提交的工作区，尚未包含后续功能第 9 条的独立预算修改。保留取消、lease、回调及 CP 元数据生命周期修复，不包含已撤回的协议/指标扩展、scheduler util、增量清理索引或 grammar 停服修复。七个已有测试文件有改动，无新增测试文件。
- **构建**：主库 `th_transformer`、connector、worker、scheduler、正向 Decode lease、integration、BroadcastManager 构建通过；另补编译 components 目标。日志为 `.cache/codex-smoke-20261027/final-review-build.log`、`final-review-components-build.log`。
- **C++ 回归**：connector 45 项、worker 120 项、scheduler 154 项、正向 Decode lease 23 项、integration 31 项，共 373 项全量通过；另外 P2P 广播客户端/取消生命周期筛选用例 18 项、BroadcastManager 全量 24 项通过，总计 415 项。相较上一轮核心回归少两项，原因是删除冗余用例，不是跳过或失败。日志位于 `.cache/writeback-accounting-20261024/`，后缀 `.final-review-20261027.log`。
- **Python 回归**：`test_native_prefill_inputs.py` 三项全部通过，包括同时检查 V4/V4.1 的 CP 元数据快照不保活父请求、tensor 存储共享及原 getter 原地修改协议；日志 `.cache/codex-smoke-20261027/native-inputs-final-review.log`。
- **范围与边界**：TCP 集成覆盖 TP/CP 非对称、混合 cache group、发布和资源释放；本次未重跑整模 DSv4 或真实 RDMA smoke，历史整模成功记录不能替代本次验证。通用 grammar 停服崩溃仍按约定不处理。`git diff --check` 通过。

### 2026-10-27：撤回 worker 增量清理后的验证

- **范围**：删除 `WriteTaskGroupCleanup`、变化通知与额外索引，恢复 D/P worker 原有定期扫描清理；保留非阻塞取消、统一 lease 查询及 TTL 过期后的取消恢复。删除两项仅针对增量索引的测试，调整既有用例验证真实 lease 计数、TTL 内去重、到期删除、在途传输和源 KV 保活，不新增测试文件。
- **构建**：主库及 connector、worker、scheduler、正向 Decode lease、integration 六个目标通过，日志 `.cache/codex-smoke-20261027/simple-cleanup-build.log`。此前撤回 grammar 修复后的当前代码也包含在本次构建内。
- **回归**：connector 45 项、worker 120 项、scheduler 156 项、正向 Decode lease 23 项、integration 31 项，共 375 项全量通过。日志在 `.cache/writeback-accounting-20261024/`，后缀 `.simple-cleanup-20261027.log`；integration 包含 TCP TP/CP 非对称及混合 cache group。
- **边界**：未重新运行整模 DSv4 或真实 RDMA smoke；grammar 通用停服问题仍按前述约定不处理。正向传输状态机未修改，共享 read context 仅增加完成回调异常隔离，安全原则与回收实现的对照见第四节。

### 2026-10-27：撤回无关 grammar 停服修复

- 下方 377 项回归及完整模型 smoke 均为撤回 grammar 修复前的工作区记录，不代表当前代码已解决该停服问题。本次仅撤回四个文件中的该项改动，不修改回写查询、传输、取消或资源保活逻辑；不再为该独立问题运行整模 smoke 或扩展修复。

### 2026-10-27：统一 Decode lease 查询入口后的验证

- **改动范围**：原 `queryLeaseStatusPerRank` 增加 operation 参数，READ/WRITE 分别查询对应 worker，共用响应填充；删除单独的 write 查询函数。不新增 RPC、协议字段或测试文件，扩展现有 connector 测试覆盖读任务缺失、回写关闭、回写未知、取消后 stopped 和非法 operation。
- **编译与回归**：主库及五个现有测试目标构建通过，日志 `.cache/codex-smoke-20261027/unified-lease-query-build-final.log`。connector 45 项、worker 122 项、正向 Decode lease 23 项、scheduler 156 项、integration 31 项，共 377 项全量通过。测试日志在 `.cache/writeback-accounting-20261024/`，后缀 `.unified-lease-query-20261027.log`。
- **边界**：本次包含真实 RPC 取消/查询及 TCP TP/CP 回写集成测试，未再次运行整模 DSv4 或真实 RDMA smoke；下方完整模型记录属于统一查询入口之前的版本。

### 2026-10-27：撤回 scheduler util 抽取后的验证

- **改动范围**：删除 `P2PWriteSchedulerUtil.{h,cc}` 及构建项，D/P 两侧恢复原有 `planFor` 和 route 构造；不新增测试文件，不撤回取消、资源保活等功能修复。
- **编译与回归**：主库、scheduler、worker、integration 四个目标构建通过，日志 `.cache/codex-smoke-20261027/trim-scheduler-util-build.log`。scheduler/context 89 项、worker/RPC 50 项、integration 31 项，共 170 项通过；测试日志在 `.cache/writeback-accounting-20261024/`，后缀 `.trim-scheduler-util-20261027.log`。
- **完整 DSv4 smoke，通过**：43 个主模型层、1 个 MTP 层，P TP2/CP2/EP2，D TP1/CP1/DP2/EP2，TCP，D 开启 CUDA graph，34 个 cache blocks。2016 输入 + 33 生成后复用 2048 token，其中 32 个来自生成；发布、P-only/PD 复用、首次计费及 warm/cold 输出一致性断言通过。证据 `.cache/codex-smoke-20261027/attempts/writeback-20261027-143046-2903969/acceptance/result.json`，`passed: true`。构建库与受验库 SHA256 一致：`b6ff311cc52fdf0216c917b6286d224e176e8d6df04c3819f2cf57adcb0b8870`。
- **退出**：客户端、服务和监控退出码均为 0，`trim-scheduler-util-full-v4/exit.json` 无遗留测试进程；`stack_trace.log` 为 0 字节，0–3 卡显存已回到基线。未验证真实 RDMA。

### 2026-10-27：撤回结构化结果后的验证

- **范围与构建**：删除 `P2PWritebackResultPB`、分类传播和新增分类指标，保留原错误码、停止证明及资源安全修复。proto 和 metrics 源文件恢复到 HEAD；`th_transformer`、scheduler、worker、integration 四个目标构建通过，日志 `.cache/codex-smoke-20261027/trim-result-build.log`。随后清理 metrics 遗留的 proto 构建依赖，四个目标再次构建通过，日志 `trim-result-build-final.log`。
- **单元回归**：scheduler/context 89 项、worker/RPC 50 项、integration 31 项，共 170 项通过。日志在 `.cache/writeback-accounting-20261024/`，后缀为 `.trim-result-20261027.log`。
- **完整 DSv4 smoke（14:15–14:17），通过**：43 个主模型层、1 个 MTP 层；P TP2/CP2/EP2，D TP1/CP1/DP2/EP2，TCP，D 开启 CUDA graph，34 个 cache blocks。2016 输入 + 33 生成后发布并复用 2048 token，其中 32 个来自生成；P-only、PD 的复用、首次计费及 warm/cold 输出一致性验收通过。证据 `.cache/codex-smoke-20261027/attempts/writeback-20261027-141504-2877131/acceptance/result.json`，`passed: true`。受验库 SHA256 为 `6b31898f58140a73320e4097498ceb49d664620563114b37365bf8cd81c9b5b0`。
- **退出与边界**：客户端、服务、监控退出码均为 0；`trim-result-full-v4/exit.json` 中 `remaining_before_cleanup` 为空，`stack_trace.log` 为 0 字节，0–3 卡恢复基线。真实 RDMA 未验证；此前记录的 Python resource tracker 清理问题不属于本轮协议精简范围。

### 2026-10-27：停服修复后的验证（协议精简前）

- **版本与构建**：仍为 `5482bddc36` 之后的未提交工作区，包含前述六项补修及停服生命周期修复。`th_transformer`、grammar 测试目标构建通过；证据根目录 `.cache/codex-smoke-20261027/`，构建日志 `shutdown-final-build.log`。
- **单元回归**：grammar 28/28、native Prefill inputs 3/3（其中生命周期用例覆盖 V4 与 V4.1）、DSv4 Prefill fast path 40/40，共 71 项通过。日志分别为 `grammar-final.log`、`native-inputs-final.log`、`prefill-fast-path-final.log`。
- **Qwen smoke**：最终库上的 P TP1 → D TP2、P TP2 → D TP1 回写 smoke 均通过，日志分别为 `shutdown-qwen-tp1-tp2-final/launch.log`、`shutdown-qwen-tp2-tp1-final/launch.log`；两次监控退出码均为 0，没有遗留测试进程。
- **DSv4 有界 CP 停服回归**：使用真实本地权重，但仅加载 4 个主模型层和 1 个 MTP 层；P TP2/CP2/EP2，D TP1/CP1/DP2/EP2，TCP，D 开启 CUDA graph。2016 输入 + 33 生成后，P 发布新增的 2048-token canonical block；随后 PD 请求、P-only 两次热请求及冷请求，共 6 次均完成，P-only 实际复用分别为 2048/2048/0。诊断时临时将 `hack_layer_num` 限制到各模型真实层数，避免把单层 MTP 扩成 4 层，该配置改动已还原。此次不是完整模型输出一致性验收，也没有跳过失败断言后宣称完整验收通过。
- **停服证据**：`attempts/writeback-20261027-113211-2808744/outputs/logs/engine.log` 显示 P/D 的 block tree 与 grammar backend 均在 KMonitor 关闭前析构；`stack_trace.log` 为 0 字节，未出现 SIGSEGV/SIGABRT。`shutdown-cp-final/exit.json` 为 exitcode 0、`remaining_before_cleanup: []`。诊断请求与明确的测试范围保存在该 attempt 的 `acceptance/result.json`。
- **完整模型基线**：停服修复前已完成完整 DSv4 Flash CP2→CP1 的发布、首次回写复用计费、再次命中及 warm/cold token/text 一致性验收，见 `attempts/writeback-20261027-103707-2709518/acceptance/result.json`；同轮退出时捕获了本次修复的 Decode 崩溃。修复后最初因可用显存不足，仅运行了上述有界 CP 回归和 Qwen smoke。
- **完整模型最终复验（13:29–13:32），通过**：0–3 卡空闲后，使用最终编译库及本地完整权重（43 个主模型层、1 个 MTP 层），没有 `hack_layer_num` 裁层。P TP2/CP2/EP2，D TP1/CP1/DP2/EP2，TCP，D 开启 CUDA graph，两侧均为 34 个 cache blocks。输入 2016 token、生成 33 token 后，P 发布 2048-token CP 单元；后续 P-only 和 PD 请求均命中，包含 32 个生成 token，首次回写复用计费及两条路径的 warm/cold token ID、文本一致性断言通过。证据：`attempts/writeback-20261027-132953-2847793/acceptance/result.json`，`passed: true`。所加载 `libth_transformer.so` 与最终构建产物 SHA256 相同，为 `39aed02e5ec2b49f122f7b374c86ed5b721d491ccc33923336aa714560a0f7e3`。
- **完整模型停服及剩余边界**：同次复验的 P/D 四个 rank 均在 KMonitor 停止前销毁 block tree、释放 grammar backend；`stack_trace.log` 为 0 字节，没有 SIGSEGV/SIGABRT。客户端及服务退出码均为 0，`shutdown-full-v4-final/exit.json` 的 `remaining_before_cleanup` 为空，GPU 显存回到测试前基线。Python `multiprocessing.resource_tracker` 仍有退出时重启及 semaphore `KeyError` 日志，此前两轮也存在，本轮未处理该独立清理问题；不能将结果表述为日志完全无异常。完整模型 TCP 验收缺口已补齐，真实 RDMA 仍未验证；未终止其他作业。

### 2026-10-27：六项补修完成时的记录（停服修复前）

- **编译**：指定的 `th_transformer`、worker、scheduler、integration，加 `stream_cache_resource_test`、`generate_stream_test`，六个目标全部通过。复用已有的 `rules_python` 兼容缓存覆盖，并将 torch 覆盖至已有 `f9431cf77fd55b828c9d3fd0b2215b26/external/pip_gpu_cuda13_torch_torch`，绕过依赖重取超时；没有修改仓库依赖或构建脚本。日志：`/tmp/writeback-six-review-build-20261027.log`。
- **测试**：WriteWorker/WriteRpc 50/50、WriteScheduler/AsyncWriteContext 85/85、完整 integration 31/31、StreamCacheResource 63/63、GenerateStream 48/48，共 277 项通过。新增取消接口不兼容时的源 KV 保活反例、首次取消后等待物理完成通知、D/P 查询队列拒绝恢复、plan 缓存及非 DECODE 入口测试；更新旧 P RPC 和关闭开关断言。
- **重复与环境**：D/P 队列拒绝及 Cancel UNIMPLEMENTED 反例共三个用例各重复 20 轮，60 次全部通过。GPU 全部占用时先运行无 GPU 回归；检测到空闲后先用 6 卡、再用 2 卡执行 GPU 测试，未操作其他作业。集成测试在独立临时目录按要求设置三个 `TEST_*` 环境变量。日志：`/tmp/writeback-six-review-{workers,schedulers-gpu,integration,repeated-faults,stream_cache_resource_test,generate_stream_test}-20261027.log`；集成目录记录：`/tmp/writeback-six-review-integration-directory`。未运行真实模型 smoke，未提交或推送。

### 2026-10-27：六项补修前的验证记录（不替代当前验收）

- **版本**：父提交为 `5482bddc36b89e0a392fc62ae7aa2add2ba91a63`，包含之后已有的未提交改动及本轮修复；不是一个新提交。受验文件 SHA256 在 `.cache/writeback-review-fixes-20261027/validated-files.sha256`，不能用父提交单独代表这次验证。
- **编译**：按指定 `build-current.py` 构建 `//:th_transformer`、worker、scheduler、integration，并增加 `//rtp_llm/cpp/model_rpc/test:broadcast_manager_test`，五个目标全部通过。原命令先因缓存中 `rules_python` 不识别 `package_annotation(whl_patches=...)` 失败；追加已有的 `--override_repository=rules_python=$PWD/.cache/writeback-review-20261027/rules_python` 后通过，没有修改仓库依赖定义。日志为 `build.log`、`build-final.log`。

| 测试程序 | 最终全量结果 | 日志（相对 `.cache/writeback-review-fixes-20261027/`） |
|---|---|---|
| `p2p_connector_worker_test` | 121/121 | `worker-full.log` |
| `p2p_connector_scheduler_test` | 153/153 | `scheduler-full-final.log` |
| `decode_writeback_integration_test` | 30/30 | `integration.log` |
| `broadcast_manager_test_bin` | 24/24 | `broadcast.log` |

- **总计**：328 项全量通过。要求的 WriteWorker/WriteRpc 共 49 项包含在 worker 全量内；WriteScheduler/AsyncWriteContext 共 81 项还单独通过（`scheduler-focused-final.log`）。公共 broadcast 目标是 wrapper，实际测试二进制为其旁边的 `_bin`。
- **新增与更新**：真实旧 P 未实现 StartWriteBack/CancelWriteBack 的两种 RPC；仅 Start 的 UNIMPLEMENTED 证明远端未开始，Cancel 的 UNIMPLEMENTED 与其他传输错误均不能证明停止；确认双端停止才释放且只释放一次。双端完成 group 回收、墓碑拦截/过期/CANCEL 幂等、有序清理、不访问未变化 group、未物理停止/源引用未释放时不压缩。更新完成态 lease 断言和旧 TTL 恢复测试，公共状态码读取覆盖未完成和越界情况。
- **重复验证**：7 个 worker/RPC 关键用例各重复 50 轮（350 次），混版本 context 用例重复 50 轮（每轮覆盖两种 RPC × 8 种状态），均通过。日志为 `worker-repeat.log`、`context-repeat.log`。
- **配置核对**：当前 `ConfigInit.cc` 明确使用 28 字段；用现有 `bazel-bin/libth_transformer_config.so` 实测 `__getstate__()` 长度、pickle 往返及 `to_string()` 全通过（`config-pickle.log`）。本轮没有改该库或配置 ABI。
- **运行方式与边界**：使用指定 `run-v4-cpp-test.py`；集成测试在独立临时目录设置 `TEST_WORKSPACE`、`TEST_BINARY`、`TEST_SRCDIR` 后运行，目录记录于 `integration-workdir.txt`。GPU 占用变化后改用 4 卡，未操作其他进程。本轮不运行真实模型、整模 CP 或 RDMA smoke。
- **已观察到的时序抖动**：首次 worker 的 `SenderExceptionCancelsRemainingUnits` 在清理线程先压缩后仍断言历史计数 2；已停住该用例的后台清理以验证活跃计数，墓碑零计数由独立用例验证。旧 `FullyCachedHandshakeRejectsDeadlineReachedDuringAdmission` 首次全量出现过 1 次边界失败，单项重复 10 轮通过；第二次全量的上游 `AsyncRead_StartLoadTimeout_HoldsTargetUntilAllRanksStop` 等待查询超时。最终完整重跑 153/153，未修改这两个无关测试或正向逻辑；失败日志保存在 `scheduler-full.log`、`scheduler-full-retry.log`。

### 历史记录：不作为当前修复验收

| 记录时间与代码状态 | 原有验证及证据 |
|---|---|
| 2026-10-26，`5482bddc36` 提交前的受验代码 | 原记录 10 个 C++ 程序共 787 项、CP/混合 group/TCP，以及两组 Qwen2.5-0.5B 非对称 TP smoke；`.cache/writeback-upstream-fixes-build-20261026.log`、`.cache/writeback-accounting-20261024/*.upstream-fixes*.log` |
| 2026-10-26，父提交 `5482bddc36` 之后的未提交阶段 | 暂缓项修复记录 587 项；D 非阻塞取消/TTL 阶段 294 项；P TTL 补修阶段 591 项。三者是不同工作区快照，不是同一提交，均不能代替本轮验证；证据分别为 `.cache/writeback-deferred-build-accepted-20261026.log`、`.cache/writeback-accounting-20261024/*.ttl-fixed-20261026.log`、`*.prefill-ttl-final-20261026.log` |
| 2026-10-22～23，DeepSeek-V4 历史工作区 | P TP2/CP2、D TP1/CP1、TCP/DEVICE-only/MTP smoke 的最终库哈希前缀 `3cd49099…`；日志 `.cache/claude-smoke-20261022/attempts/writeback-20261023-112504-1477358/`。旧记录没有可唯一复现的提交号，不将其标为当前代码或当前 CP/RDMA 验收 |

## 附：《代码复用》文档

`docs/decode-kv-writeback-review-code-reuse.md` 讨论的是回写与原有正向链路之间的重复实现，大部分合并要改上游代码。与正向链路之间的复用按约定暂缓。D/P 两侧 `planFor` 和 route 构造的 util 抽取也已撤回，删除新增文件和构建项；本轮只保留功能修复，代码重构后续单独处理。
