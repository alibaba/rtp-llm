# Decode KV Cache 回传 Prefill 设计

> 状态：设计稿（v1，待 review）
> 范围：`rtp_llm/cpp/cache/connector/p2p/`（decode_entrance 链路的反向传输）
> 阶段：Phase 1，对称配置（两侧 TP 相等、prefill 不开 CP）；Phase 2（非对称）见 §8
> 原则：对现有 P2P 链路做一次方向反转，改动尽可能小，不引入不必要的新模块。写回是 best-effort 旁路，任何失败只放弃写回，绝不影响请求主链路。

---

## 0. 一页总览

PD 分离下，decode 生成 token 的 KV cache 只存在于 decode 侧。多轮对话的下一轮 prompt 是上轮 prompt 加上轮 response，prefill 只能命中 prompt 部分的前缀，response 部分要重算。

这份设计让 decode 在请求正常结束、本地插树的同一时刻，把 prefill 没有的那段增量 KV 块反向 push 给服务过该请求的 prefill 实例；prefill 分配空闲块接收，收齐后原子地插进自己的 block prefix tree。

Phase 1 只做一件事：在**两侧配置对称**（TP 相等、prefill 不开 CP）的前提下把这条链路打通。不支持非对称 TP/CP/DP：入口断言两侧 layout 对称（digest 相等），不对称直接拒绝，放开是 Phase 2 的主题（§8）。

```
decode stream FINISHED（tryReleaseKVBlock）
   │
   ├─ (现状) 本地 insertIntoCache → decode 自己的树
   │
   └─ (新增) asyncWrite ──────────────► prefill rank0: StartWrite
        hold blocks                      ├─ probe：prompt 全在 GPU，后缀无低层/忙碌节点冲突 → [p, p+k)
        直到传输结束/超时                 ├─ mallocForExternalInsert：RAII 接管新块的分配引用
                                         ├─ 广播 recv routes 给 prefill workers
        ◄─ response {accepted: p, k, 接收端点} ─┘
   ├─ 广播 send routes 给 decode workers
   ├─ decode workers 按 route 发送（数据早已算完，一次全量发）
   └─ prefill workers 收齐 → settle → BlockTreeCache::insert → 树上可见
```

---

## 1. 传输内容：增量如何界定

### 1.1 传输内容与准入控制：key 由 decode 算，增量与准入由 prefill 判定

**key 由 decode 计算**：key 是 token 序列的链式 hash，生成的 token 只有 decode 知道。decode 用 `completeTokenIds` 派生全序列 cache_keys，与它本地插树是同一来源，天然一致。握手传的只是元数据（keys 每块 8 字节，1000 块也就 8KB，一次 RPC），KV 数据从头到尾只有被接受的增量上过网。

握手同时带上**完整 token_ids 和 input_length**（4K token 约 16KB，仍是元数据量级）：cache_keys 是当前匹配的快路径；token_ids 供 prefill 用本地 `initCacheKeys` 复算 keys 与 cache_keys 互校——防滚动升级时两侧 hash 算法/版本漂移导致 key 永不匹配，复算不一致直接拒绝本次写回。注意**复算与比较只做到完整块边界**：`initCacheKeys` 会给尾部半块也算 key（§1.2），比较前须按同一丢弃行为截断，否则互校恒失败。token 序列也是未来扩展（按 token 索引的远端缓存、审计校验）的现成原料。

**准入与增量由 prefill 判定**，规则三条：

1. **先判断是否写回？prompt 前缀必须整段驻留 GPU，否则本次不写回**。StartWrite 请求带 `input_length`，prefill 由它推出 prompt 完整块数 prompt_blocks（不单独设字段，避免与 input_length 同义冗余）；prefill match 得到 **device 层连续命中前缀** p_dev（`BlockTreeMatchResult.matched_device_blocks`，零新机制），若 `p_dev < prompt_blocks`——prompt 有块被降级到 CPU/盘或已丢弃——直接拒绝，decode 释放 hold。理由：prompt 被挤出 GPU 说明该会话已冷或 prefill 显存高压，投机写回价值低；拒绝也避免为写回触发 load-back 或混合层挂载。
2. **写回内容 = 两侧持有的差，服从规则 1，并检查已有后缀是否可插入**。通过价值门后，以 **device 层连续命中前缀 p 为界（p ≡ p_dev，与规则 1 同一个量）**，不取跨层命中。必然 p ≥ prompt_blocks；p 可能更大，因为相同前缀的并发写回已补过一段。`[p, 完整块数)` 是候选接受区间，但 p_dev 之后可能存在 HOST/DISK 节点，不能等同于树上不存在：现有 `insertNode` 不会把已有 HOST FULL 节点更新为 DEVICE，还会停止后续插入。**Phase 1 对候选路径上的低层 FULL 节点、忙碌或不完整且不可接纳的节点直接拒绝本次写回**，不触发 load-back、不覆盖节点；检查须为不持块、不登记加载的探测（接口要求见 R3）。只有该检查通过才接受 k 块，k=0 仍走快路径。具体冲突判据及 settle 时的重验见 §3.1。
3. **通过准入的写回是一等公民：GPU 满照常驱逐腾位**。malloc k 块走与普通请求相同的分配路径（`FullKVCacheGroup::malloc → ensureFreeBlocks`，现成代码），空间不足时照常触发 LRU 驱逐更冷的缓存；驱逐后仍不足（极端）才拒绝。三条规则自洽：规则 1 已筛掉低价值写回，剩下的对应活跃会话，其数据配得上挤掉冷缓存的待遇。

为什么增量判定必须在 prefill 而不是 decode 记账：① prefill 的树可能已淘汰部分块，"当初传过的"≠"现在还有的"；② 相同前缀的并发请求可能已写回一段，缺口比 decode 以为的晚；③ 已有 DEVICE 结果可复用，而已有低层或忙碌节点必须按冲突规则处理，不能无条件跳过同 key。价值门和后缀冲突检查需要 R3 中提出的纯探测接口；不能直接丢弃现有 `BlockTreeCache::match()` 的结果，因为它包含引用和加载状态。是否刷新 LRU 由该探测接口明确，不将现有 match 的行为直接当作无副作用的实现。

### 1.2 块边界：只传完整块，跨界块自动划入增量

尾部未写满的块不传。这不是工程妥协，而是按块 hash 的前缀缓存体系下的必然边界：

- **半块的 key 会被显式丢弃，进不了树**。`initCacheKeys` 会给尾部半块也算 key（KVCacheHashUtil.cc:15，注释原文 "including the final partial block"），但半块每多填一个 token hash 就变，所以代码用两道丢弃挡它进树：decode 每步 `updateCacheKeys` 按 `lastBlockAligned` 弹掉旧半块 key（.cc:47-49），`insertIntoCache` 入口再兜一道 `dropLastPartialBlock`（KVCacheManager.cc:355）——被丢弃的 key 永远不会被 match 命中。**实现提示**：写回打包 keys 时必须对齐同一行为（调 `dropLastPartialBlock` 或按 `lastBlockAligned()` 截断），否则存在把半块 key 报给 prefill 的时序窗口。
- **传了也匹配不上**。下一轮 prompt = 上轮 prompt + response + 新的用户消息，跨在 response 尾部和新消息开头的那个块，token 段和 decode 手里的半块不同、hash 不同，即使发明半块传输机制也对不上。
- **损失有硬上界**：下一轮 prefill 最多多算 `seq_size_per_block - 1` 个 token（配置里 8～64）。想吃掉这点收益，正确的杠杆是调小 `seq_size_per_block`，而不是给传输和树加子块粒度的 key 与部分匹配语义。

`accepted_block_count` 以 decode 侧完整块数与 prefill 侧 match 结果共同截断。

而 **prompt 尾块与输出拼成的"跨界块"不需要任何特殊处理**——key 的生成不区分 prompt/输出边界。块大小 8、prompt 13 个 token、生成 10 个时：

```
token:   [p0..p7] [p8..p12, g0..g2] [g3..g9]     (p=prompt, g=generated)
block:      B0          B1             B2(半块)
key:       k0 ✓        k1 ✓           key 被丢弃
```

B1 就是跨界块：前 5 行 KV 由 prefill 算好传来，后 3 行由 decode 解码时写入同一个物理块，写满后在完整序列的 hash 链上拥有定型的 k1。而 **prefill 侧的 B1 是半块，它的 key 被 `dropLastPartialBlock` 丢弃、不进树**（只有 5 个 token，传输后块即释放），所以写回握手时 match 恰好停在 k0，accepted 区间从 B1 开始——跨界块被自然划进增量。这也再次印证了 §1.1 的选择：若严格按"生成 token 所在的块"界定增量，B1 会被误判为"prompt 的块"而漏传，但 prefill 树里恰恰没有它。

**适用前提**：链式 hash 意味着 k1 依赖从 p0 开始的整个前缀。下一轮请求要命中写回的块，prompt 必须与"上轮 prompt + 上轮 response 开头"逐 token 一致——原样拼接历史的多轮对话天然满足；中间件若对历史做了改写（截断、摘要、重排 system prompt），链从改动点断掉，之后的写回块全部失效。写回的收益评估应以"下一轮 prompt 严格追加"为前提。

### 1.3 对称假设下的键集与 route

Phase 1 断言两侧 `ShardLayout` 对称后，plan 退化为每个 rank 一条对角线 route，`KeyShardSpec{modulus=1, residue=0}`：rank i 把接受区间内所有块原样发给 rank i。仍然走 planner 而不是硬编码，因为执行路径就和 Phase 2 完全一致了，Phase 2 只需删掉对称断言、扩 planner 白名单，不用重写执行层；plan 的逐 route 字节校验（`src_bytes == dst_bytes`）也是免费拿到的，配置漂移在编排期报错，而不是拷错数据。

---

## 2. 回传时机

### 2.1 触发点：在请求收尾处异步发起传输

decode 侧请求的收尾逻辑在 `StreamCacheResource::tryReleaseKVBlock`（StreamCacheResource.cc:297），我们在本地插树之后、释放引用之前，异步发起一个写回请求。注意生产链路里**没有 coordinator**（`KVCacheManager` 直接持有 `p2p_connector_`，初始化日志明示 "initialized without coordinator"，KVCacheManager.cc:800；正向的 asyncRead 也是 KVCacheManager 直调（本分支已无 asyncMatch，prefill 登记走 registerResource），.cc:684/702），所以入口走 cache_manager：

```cpp
// StreamCacheResource::tryReleaseKVBlock
  if (reuseCache() && !stream_->hasError() && stream_->getStatus() == StreamState::FINISHED) {
      const Tier target_tier = storeTarget();               // 现状：本地插树还有一道 tier 门
      if (target_tier != Tier::NONE) {
          cache_manager->insertIntoCache(insert_info);      // 现状：本地插树
+         // 新增：异步发起写回（KVCacheManager 新增薄方法，内部转发 p2p_connector_->asyncWrite）。
+         // hold 住资源后立即返回，不阻塞收尾路径；开关未开 / 超并发 / 无 routing context 时直接跳过。
+         // Phase 1 限定 target_tier == DEVICE：发送路径按 device block id 寻址（见下），
+         // 本地直落 HOST 时"insertIntoCache 不拷贝不释放"的前提是否仍成立待确认，先不写回。
+         if (target_tier == Tier::DEVICE) {
+             cache_manager->asyncWriteBack(batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), meta);
+         }
      }
  }
  cache_manager->free(free_info);                           // 照常释放 stream 引用
```

选这个位置的理由：stream 状态明确（FINISHED、无 error、开了 reuse、storeTarget 非 NONE——与本地插树共用同一组条件），token 序列完整（`completeTokenIdsPtr`），块还活着；写回只在其上收紧一档（仅 DEVICE 目标）。cancel、error、超时的请求自然走不进这个分支，不回传。

`asyncWriteBack` 在这里只做几行判断加任务提交就返回，算 keys、RPC、下发 route、等回调全在 scheduler 线程上跑——与正向 `asyncRead` 同一个模式，主进程零阻塞。

显存代价：DEVICE 目标下 `insertIntoCache` **既不拷贝也不释放**，只是把块的所有权移交给 block tree（`free()` 仅减 stream 引用，物理块原地变缓存内容、由水位淘汰回收）。块本来就会留在 decode 树上，hold 不产生额外显存占用。

数据驻留由引用计数保证：`DeviceBlockPool` 的块**只在 refcount 归零时回空闲池**（DeviceBlockPool.cc:115-133）；hold 走 `KVCacheManager::incrKVCacheRef`（.cc:578，shared_ptr 析构自动 -1；`is_connector=true` 的同款调用见 StreamCacheResource.cc:759）：

```
运行期:                     refcount = 1  (stream)
insertIntoCache:            refcount = 2  (stream + tree)
asyncWriteBack 内 hold:     refcount = 3  (stream + tree + connector)   ← 同步完成，必须先于 free
free() 照常执行:            refcount = 2  (tree + connector)
── 传输窗口 ──
  最坏情况：水位淘汰驱逐了树节点（节点是元数据，驱逐只减树的那一票）
                            refcount = 1  (connector)  ← 块不回空闲池，数据原地驻留 GPU
传输完成/超时，hold 释放:    树还在 → 继续当缓存；树已驱逐 → 此刻才真正回收
```

发送路径也不依赖树：decode worker 按 hold 住的 `KVCacheResource` 里的 block id 直接寻址显存，树节点是否被驱逐不影响进行中的传输。因此顺序约束只有一条：hold 必须在 `asyncWriteBack` 内同步完成，之后 `free()` 照常执行。

注意 **hold 的是全序列的块，不是增量**：增量起点 p 要到握手后才知道，而 hold 必须先于 `free()`，所以只能按全序列 keys 全量增引用。这没有额外代价（块本就全部留在树上）；握手后提前释放 `[0, p)` 的引用是可做但不值得的优化。

### 2.2 可选的防御性策略

一般情况下写回窗口很短（一次握手 RTT + k 块传输），且如上所述没有额外显存占用，不需要任何干预。以下策略针对异常和长尾场景，均为可配置项：

**兜底两件套**（Phase 1 实现）：

- 硬 deadline：`p2p_writeback_timeout_ms`（建议默认 5s 量级），超时立即放行释放，传输任务 abort——保证淘汰延迟有上界；
- 并发上限：`p2p_writeback_max_inflight`，进行中的写回超限直接跳过本次。由于没有额外显存占用，它防的只是异常场景下 pin 住过多可淘汰块，可以设得宽松。

**提前写回**（可选扩展，默认关闭）：输出累计满 N 个完整块（`p2p_writeback_early_threshold_blocks`）就分段异步写回，而不是等请求结束一次性传。适合"超长输出 + 短轮间隔"的场景：请求结束时大部分块已在 prefill 侧，收尾只剩 final flush。

它的可行性建立在一个关键性质上：**增量的任意连续前缀都是合法的缓存内容**。提前传的都是前缀链上定型的完整块，请求中途 cancel 也无需撤销——prefill 树上已插入的段仍是有效数据，代价只是可能留下永不复用的块（显存，非正确性）。而且正在 decode 的 stream 本就持有这些块，连防淘汰 pin 都不需要额外处理。

真正的成本是状态维护，需要引入：

- per-stream 写回游标（已写回到第几块）；
- in-flight 批次跟踪（上一段没回来之前是否发下一段、失败后是否停止后续分段）;
- 请求结束时的 final flush 与游标收敛；
- 多次握手/插树带来的 RPC 与树操作放大（每段一次 StartWrite + 一次 insert）。

Phase 1 只实现结束后一次性回传，但入口按"游标从 0 直接推到末尾的特例"来组织代码，给提前写回留好接缝。

---

## 3. Prefill 侧：接收、保留、插树

### 3.1 处理顺序（原子性是底线）

```
纯探测准入及后缀冲突 → RAII 分配空闲块 → 注册 recv task → 收齐全部 (layer, tag, block)
  → settle：同锁完整预检前缀和后缀，再插树 → 可见
```

insert 必须是全部数据到齐后的原子动作：任何一层、一块失败，就放弃插树，确认物理传输停止后释放全部已分配块。半截数据进树会污染前缀匹配——树上可见就意味着可被 match、可被后续请求复用。物理停止的判定和后台持有者仍需按 R1/R4 落实，不能将业务超时或 erase entry 当成物理传输停止。

malloc k 块是**一等公民分配**（§1.1 规则 3）：走与普通请求相同的路径（`FullKVCacheGroup::malloc → ensureFreeBlocks`），GPU 满照常触发 LRU 驱逐腾位，驱逐后仍不足（极端）才在 StartWrite response 里拒绝。多个 DP 副本并发写回的聚合压力由 decode 侧 `p2p_writeback_max_inflight` 与正常驱逐体系共同承接。

**接收侧分配引用：由新分配接口直接返回 owning RAII 句柄**。`FullKVCacheGroup::malloc()` 在分配成功后已调用一次 `incRef`（FullKVCacheGroup.cc:81–85）；`incrKVCacheRef()` 会再加一次引用（SingleTypeKVCacheAllocator.cc:454），其 deleter 只减自己增加的那一次。正向路径由 stream 的 `free()` 释放原始分配引用，写回接收侧没有这个 stream，不能照抄 `malloc → incrKVCacheRef → entry` 后只释放 guard。

因此 `mallocForExternalInsert(...)` 的返回契约明确为：**RAII 句柄直接接管 malloc 产生的原始引用，不再额外调用 `incrKVCacheRef`**。allocator 在分配过程中负责回滚已经成功的部分；成功返回后，原始分配引用的唯一释放责任交给句柄的 deleter。deleter 保持 allocator/pool 存活，并对该句柄实际分配的所有块各减一次引用。entry 和接收任务共享同一个句柄的生命周期，不另造一份分配引用；单纯创建普通 `shared_ptr<KVCacheResource>` 不满足此契约。该接口属于 Slice 2，当前尚未实现。

引用账本（对每个新分配块，物理传输已停止时才允许最后一个接收句柄销毁）：

| 时刻/结果 | 新块引用数与释放责任 |
| --- | --- |
| malloc 成功并返回句柄 | 1，原始分配引用由 RAII 句柄接管 |
| entry/接收任务持有句柄 | 仍为 1；复制句柄不增加 block refcount |
| 插树采纳该块 | 2，分配句柄 1 + 树的 CACHE 引用 1；`BlockTree::publishTier` 增加树引用（BlockTree.cc:173–175） |
| settle 成功后销毁接收句柄 | 采纳的新块从 2 到 1，归树持有；因已有 DEVICE 副本而未采纳的新块从 1 到 0，归还空闲池 |
| 分配/注册失败、接收失败、放弃插树 | 未发布的新块从 1 到 0；释放一次，不依赖 stream 的 `free()` |

**entry 的创建与数据入树**。StartWrite 到达后，handleWrite 现场造 entry：元数据（unique_key/deadline/cache_keys/token_ids/input_length）抄自 RPC；纯探测过价值门及后缀冲突检查后得出 p，`mallocForExternalInsert` 返回 k 个落点块的 owning 句柄，将其交给 entry（现有 `addResource` 依赖 Meta，需增加直接传参重载）。注册失败由局部句柄回收；开始接收后，句柄必须覆盖全部在途任务的生命周期。成功或放弃后，所有新分配块的分配引用都要归还，不能只释放未被树采纳的块。

**已有 HOST 后缀的策略：Phase 1 明确放弃，不做节点升级**。例如 `{prompt_0, response_0}` 已在树中，prompt_0 在 DEVICE，response_0 只有 HOST，则 p_dev=1。即便 prompt 完整命中，普通 `insertNode({prompt_0, response_0, response_1}, ...)` 也会停在 response_0，response_1 不会被插入。`BlockTreeTest.InsertHardStopsAtExistingHostFullNode`（BlockTreeTest.cc:455）明确覆盖此行为，不能依靠“重复 key 自动跳过”补回 GPU。

对 `[p, p+k)` 候选路径中的既有 FULL group，准入及 settle 都按同一规则检查：

- 完整、稳定的 DEVICE 值可以复用，重复写回的新块由接收句柄归还。
- 既有空节点只有在可安全接纳新值时才允许使用（当前对应 `is_removable()`）；不存在的节点可新建。
- HOST/DISK-only、LOADING/DEMOTING 等忙碌状态，以及不完整且不可安全接纳的状态，都拒绝本次写回，记录 `existing_suffix_conflict`。本阶段不删除旧节点、不替换已有缓存、不触发 load-back。

数据收齐后，新增 `insertExternalBlocks` 必须在 `BlockTreeCache::mutex_` 内先做**完整预检**：`[0, p)` 仍整段驻留 GPU，并且 `[p, p+k)` 的全部既有节点都符合上述规则，之后才挂载新块。准入后的 malloc 驱逐和传输期间的并发变化都可能让检查结果失效，因此 settle 必须重验。不能先调用普通 insert，再根据是否插完决定失败，因为前面的空节点可能已被采纳，形成部分发布。该接口须返回整体结果及实际采纳范围；它是 R3 要求的新接口，尚未实现。group 支持范围及非 FULL group 的完整性约束仍由 R6 明确。

**异常控制：传输窗口内前缀被驱逐 → 优雅退出**。准入时前缀全在 GPU，但飞行的几百毫秒内淘汰器仍可能动它——我们不 pin，引用也 pin 不住节点（`isEvictable` 不看 request ref，BlockTreeEvictor.cc:73-78）；在途接收的 k 个新块由 owning 句柄保活。settle 在树锁内完成重验并决定不发布；终态登记及资源回收在退出树锁后执行，不持有树锁等待物理传输停止：

```
① 树锁内不调 insertNode（防止给缺失 key 造出空壳节点，BlockTree.cc:248）
② 退出树锁，标记写回终态；接收上下文继续保活任何仍在途的任务和落点块
③ 确认无在途访问后，移除 entry 并释放接收上下文的 owning 句柄，新块的分配引用归零
④ metric 记 writeback_abandoned（原因：prefix_evicted / prefix_demoted）
decode 侧由传输回调/deadline 结束业务等待，物理保活约束见 R1
```

不变量不变：prefill 要么整段发布、要么本次写回零发布且最终无资源残留；已有缓存不受影响。后缀冲突的放弃也走同一回收流程，原因记为 `existing_suffix_conflict`。

**为什么 prefill 侧不 maintain stream**，两条理由：① **用不上**——写回要的持块、超时释放、key 索引、汇合，decode-entrance 正向链路已在 main 上从 stream 解耦进 ResourceStore entry（addResource/waitAndStealResource），写回同款复用；拿块也不依赖 stream：`initKVBlock` 只是组装 MallocInfo 转调 `cache_manager->malloc`（StreamCacheResource.cc:370 起），真正的分配与驱逐在 group 层，直调即可，无 stream 造 resource 有生产先例（KVCacheManager.cc:643）。② **养不起**——GenerateStream 是引擎状态机的参与者，为写回伪造一个意味着假输入、假状态推进，且它的收尾是"无条件插树"，与写回"收齐才插、失败零残留"相反；假 stream 有过事故（docs/references/deepseek/dsv4_mtp_fake_decode_host_memory_leak.md）。所以 prefill 对写回零长期状态，一致性全靠握手时 match + settle 时重验。

### 3.2 保留、淘汰与热度：全部委托给 BlockTree，零新设计

写回块插树后就是一个普通缓存节点，和本地请求留下的块没有任何区别，BlockTree 的既有机制自动生效：

- **占用与回收**：受 watermark 淘汰管理、可降级 host/disk，"显存浪费"的上界由既有淘汰体系兜住；
- **热度**：insert 时获得最新 `last_access_seq`，之后正常请求的 match 命中自动刷新；StartWrite 纯探测是否刷新热度由 R3 的接口契约明确；
- **可选退路**：`insert` 的 `target_tier` 参数支持直落 HOST（settle 后 D2H、device 块即时归还，命中时走现有 load-back），留作配置项。

因此写回对缓存管理的唯一控制点就是 §3.1 的准入；块进树之后的一切不需要任何新代码。

### 3.3 幂等与并发

- 相同前缀并发写回：通过 §3.1 新增的同锁预检与插入接口串行提交。完整稳定的 DEVICE 重复结果可复用，接收句柄释放本次所有分配引用；低层或忙碌节点触发整次放弃。不能把现有普通 insert 的局部 adopt-or-skip 当成整个写回的原子性或幂等保证。
- decode 中途消失（进程退出）：prefill 在 transfer deadline 后结束业务等待、禁止插树；接收上下文保留 owning 句柄，直到物理传输停止后释放。ResourceStore 的 TTL 负责业务终态，不能单独决定仍在途块的归还时刻（R1/R4）。

---

## 4. 配套重构：side channel 从 ResourceStore 解耦

### 4.1 问题：通用租约容器长着正向链路的器官

side channel 是正向 StartLoad 的推理结果回传：prefill 算出的 first token / reuse len / MTP 张量要交还 decode，而这些数据在 RPC 到达时还不存在，于是进程内造了一套"引擎线程生产、RPC handler 等待"的汇合机制。问题是这套机制长在通用租约容器上：

- entry 内嵌 `SideChannelData` + ready 标志 + 每 entry 一把 mutex/cv（P2PConnectorResourceStore.h:31-47）；
- store 级又有一份独立 map `side_channel_data_map_` + `active_side_channel_deadlines_` + 专属 cv（.h:127-135）——为了处理"entry 已被 steal、notify 才到"的竞态；
- `checkTimeout` 要清两个 map（.cc:344-366），`markTerminal` 的一半语义是封死迟到通知。

写回复用 ResourceStore 做收侧租约（§3.1）之后，它从"正向 load 的私有实现"变成两条链路的公共底座：每个写回 entry 都背着一坨永远为空的 side-channel 死状态，读代码的人要反复辨认"这部分对写回适用吗"。本版一并改掉。

### 4.2 方案：ResourceStore 退回纯租约，side channel 独立成 store

- **`P2PConnectorResourceStore` 只留租约**：unique_key、request_id、kv resource ref、双 deadline、cancelled/terminal 终态。entry 内嵌的 SideChannelData/ready/mutex/cv 全部删除；
- **新建 `PrefillResultStore`**（仅 prefill、仅正向 load 使用）：单一 `map<unique_key, {data, deadline}>` + 一个 cv，API 四个——`notify` / `waitAndFill` / `seal`（承接 markTerminal 里"封死迟到通知"的那一半语义）/ 定期清扫；
- **双存储合并为单 map**：现状 entry 内嵌那份并不承担唯一真相——等待侧本来就在 10ms 轮询独立 map 兜竞态（P2PConnectorPrefill::waitAndFillResponse，P2PConnector.cc:504 起）。合并后 notify 无条件写 map、唤醒 cv，wait 只查 map，数据投递与 entry 生命周期彻底无关，竞态面从"两条时间线 × 两份存储"降为"× 一份"；
- **调用方改动**：NormalGenerateStream 的 notifySideChannelReady 经 KVCacheManager 转发到新 store（签名不变）；handleRead 的 `waitAndFillResponse` 改调 `prefill_result_store_->waitAndFill(...)`，散布全函数的十来处 `clearSideChannelData` 收敛为终态路径上一次 `seal`；**handleWrite 与新 store 零接触**。

协议零改动：response 的 `SideChannelPayloadPB` 原样不动，这是纯进程内重构。

### 4.3 改动位置、验证与落地顺序

| 位置 | 改动 |
|---|---|
| `p2p/P2PConnectorResourceStore` | 删 entry 内嵌 side-channel 字段与全部 side-channel API；markTerminal 保留终态语义、去掉 side-channel 部分 |
| `p2p/PrefillResultStore.{h,cc}` | 新容器：单 map + cv + seal + TTL 清扫；注释注明对应 StartLoad response 的 `SideChannelPayloadPB`（协议字段不改名）（新增） |
| `p2p/P2PConnector` | `waitAndFillResponse` 改调新 store；`clearSideChannelData` 调用点收敛为 seal |
| `NormalGenerateStream` / `KVCacheManager` / Coordinator | notify 转发目标换成新 store，签名不变 |
| 测试 | ResourceStoreTest 的 side-channel 用例迁到新 store 单测；DecodeLoadHelperTest / P2PConnectorTest / 正向 smoke 回归 |

风险集中在 handleRead 的等待/清理时序（正向热路径），靠上表回归兜底。落地顺序上作为 **Slice 0**，先于写回各切片（§7.8）：趁 ResourceStore 还只有一个用户时拆，改动面最小，Slice 2 的 handleWrite 也能在干净地基上写。

---

## 5. 配套重构：在角色纵切内按流拆分 scheduler / worker

> 本章基于当前分支结构（commit "refactor: split P2P connector by role"）。

### 5.1 现状：connector 已按角色纵切，横向门面已删除

`P2PConnector` 现在是薄门面，按 role_type **只构造本角色的纵切**（P2PConnector.cc:103-122）；原 Scheduler/Worker 两个横向门面已随该重构删除，角色子对象由纵切直接持有：

```
P2PConnector（薄门面，按 role_type 二选一）
 ├─ P2PConnectorPrefill：BroadcastClient + SchedulerPrefill + WorkerPrefill(持 sender) + ResourceStore
 └─ P2PConnectorDecode： BroadcastClient + SchedulerDecode  + WorkerDecode(持 receiver)
```

每个纵切里的 scheduler/worker 天然单流——只服务正向 load。写回等于给每个纵切加上第二条流。

### 5.2 拆法：纵切内按流成对，scheduler 与 worker 同构

```
P2PConnectorDecode（decode 进程）
 ├─ P2PSchedulerDecodeRead    现 SchedulerDecode 纯改名
 ├─ P2PSchedulerDecodeWrite   新增：写回发起编排
 ├─ P2PWorkerDecodeRead       现 WorkerDecode 纯改名（持 receiver，正向收）
 └─ P2PWorkerDecodeWrite      新增：持 sender，写回发（接住 init 中被丢弃的半个 backend，§5.4）

P2PConnectorPrefill（prefill 进程）
 ├─ P2PSchedulerPrefillRead   现 SchedulerPrefill 纯改名
 ├─ P2PSchedulerPrefillWrite  新增：handleWrite 编排
 ├─ P2PWorkerPrefillRead      现 WorkerPrefill 纯改名（持 sender，正向逐层发）
 └─ P2PWorkerPrefillWrite     新增：持 receiver，写回收（同上）
```

Write 侧是**轻量新类，不是复制 Read 侧**：方向通用的收发机制都在 `transfer/*`（SendRequest、TransferTaskStore、rendezvous），Write worker 直接用这些原语——`WorkerDecodeWrite` 按 routes 构造一次性 SendRequest 推送；`WorkerPrefillWrite` 把落点块按 key 注册进 TaskStore、收齐后通知 settle。正向独有的逐层流水（ComputedLayerCacheBuffer、StoreWaitContext）全部留在 `WorkerPrefillRead`，不复制。

### 5.3 命名：角色 × 流唯一确定方向，方向不进类名

类名按流命名 Read/Write（decode 视角），锚点是接口层现有的 `asyncRead`/`handleRead`。方向由角色×流推导：DecodeRead=收、DecodeWrite=发、PrefillRead=发、PrefillWrite=收——早先方案里 WorkerSender/WorkerReceiver 的改名**作废**："WorkerDecode 在 prefill 进程当接收器"的绕口令在纵切结构下不存在，写回的接收器就叫 `WorkerPrefillWrite`，长在 prefill 纵切里、按 prefill 命名。

| 改名项 | 说明 |
|---|---|
| SchedulerDecode / SchedulerPrefill → `P2PSchedulerDecodeRead` / `P2PSchedulerPrefillRead` | 纯改名 |
| WorkerDecode / WorkerPrefill → `P2PWorkerDecodeRead` / `P2PWorkerPrefillRead` | 纯改名 |
| 新增四个 Write 类 | 见 §5.2 |
| `RouteCodec::encodeForPrefill/encodeForDecode` → `encodeForSender/encodeForReceiver` | 保留此项——codec 编码的本来就是 route 发送端/接收端字段，方向是它的本质维度 |
| 协议词汇 | 不变：`StartWrite`/`handleWrite`、`HANDLE_WRITE`/`WRITE`、`_wb_` 前缀、`p2p_writeback_*` 配置、`PrefillResultStore`（§4） |

### 5.4 基建校正：接住被丢弃的半个 backend

`createAndRegisterTransferBackend` 在每个进程仍然把 sender+receiver 都建出来并**双向 regMem**（P2PConnector.cc:34-95），但当前各纵切只把本角色用的那半交给 worker，**另一半 init 返回即析构**——prefill 进程没有存活的 receiver（无监听端口），decode 进程没有 sender。写回把这一半交给对应的 `*Write` worker 即可，一行接线。注意：老结构下"收发子对象已构造、端口已监听、零新基建"的说法在当前分支**不再成立**，基建成本修正为"留住 backend 另一半 + 两个轻量 Write worker"。

### 5.5 落地

Read 侧改名是纯重构，与 §4 同属 Slice 0 批次（§7.8）；保留 backend 另一半、增加两个轻量 Write worker 并完成接线归入 Slice 1，两个 Write scheduler 的编排逻辑分别在 Slice 2/3 独立落地和验证，生产请求收尾触发及真实两端接通单独归入 Slice 4。回归：现有 P2PConnectorTest 全量 + 正向 smoke。

---

## 6. 端到端视图：架构映射、数据流与时序

### 6.1 架构映射：同一副骨架，角色换边

P2P connector 在每个进程里是角色纵切下的五层结构（§5.1）：

```
P2PConnector（薄门面，按 role_type 构造本角色纵切）
 └─ P2PConnector{Prefill|Decode}（角色纵切）
     ├─ Scheduler*（仅 rank0）   控制面大脑：算 plan、投影 route、驱动 RPC
     ├─ Worker*（每个 rank）     数据面执行器：发数据 / 注册接收
     ├─ ResourceStore（prefill） 请求级资源租约（持块 refcount + TTL）
     ├─ BroadcastClient（rank0） 集群内控制通道：rank0 → 本侧全部 worker
     └─ transfer/*               纯数据面：sender / receiver / TransferTaskStore
```

写回不新建任何一层，只把每层里"发起方/响应方、发送方/接收方"的角色换边：

| 架构位置 | 正向 load | 反向 write-back | 变了什么 |
|---|---|---|---|
| 发起方（控制面 client） | decode rank0：`asyncRead` | decode rank0：`asyncWriteBack` | **没变**——两个方向都是 decode 发起 |
| 响应方（控制面 server） | prefill rank0：`handleRead` | prefill rank0：`handleWrite` | **没变** |
| 数据发送方 | prefill 纵切的 `WorkerPrefillRead`（持 sender） | decode 纵切新增的 `WorkerDecodeWrite`（持 sender，§5.2） | 反转（轻量新执行器 + backend 接线，§5.4） |
| 数据接收方 | decode 纵切的 `WorkerDecodeRead`（持 receiver + TaskStore） | prefill 纵切新增的 `WorkerPrefillWrite`（持 receiver，§5.2） | 反转（同上） |
| 源侧持块 | prefill ResourceStore entry（请求进入时 registerResource 登记，P2PConnector.cc:244） | decode 的 connector hold | 机制同（refcount），载体不同 |
| 收侧持块 | decode stream 自己的分配 | prefill ResourceStore entry（主动 malloc） | entry 的第二种构造方式 |

浓缩成一句核心不变式：

> **控制面方向不变（decode 永远是 client、prefill 永远是 server），只有数据面方向反转。**

这是整个复用故事的支点：控制面的全部基础设施（RPC caller 模式、广播通道、plan 计算与缓存、对称性校验）因方向不变而原样照搬；数据面的底子也在——`createAndRegisterTransferBackend` 在每个进程都建出 sender+receiver 并对 KV buffer 双向 regMem（P2PConnector.cc:34-95；TCP 模式下 receiver regMem 失败仅告警非致命，RDMA 模式才强制双向成功）。当前未用的那一半在 init 返回时被丢弃（§5.4），所以写回的传输基建成本 = 留住这一半 + 两个轻量 `*Write` worker，没有任何新机制。

### 6.2 数据流

对称 TP=N 时是 N 条并行的 rank-to-rank 管道，每条搬运本 rank 分片的增量：

```
decode rank i 的 device blocks（增量 k 块 × 全部 layer×tag）
  → staged gather D2H（复用 TcpKVCacheSender 现成路径）
  → TCP → prefill rank i 的 transfer server
  → rendezvous → executeCopy → prefill 预分配的 device blocks
  → 全部到齐 settle → BlockTreeCache::insert → 树上可见
```

与正向的一个显著差异：正向 prefill 侧要逐层流水（数据跟着计算逐层就绪，`asyncWriteByLayer` + ComputedLayerCacheBuffer），写回时数据早已全部算完，一次性全量发——反向的数据面反而比正向简单。

### 6.3 时序

```
decode                                     prefill
──────                                     ───────
请求结束：本地插树 + hold 全部块（此刻还不知道 p）
   │
   ├─── StartWrite(keys + token_ids + prompt 边界) ─►  复算互校 → match：价值门 p_dev ≥ prompt_blocks，不满足即拒
   │                                        malloc k 块（不足照常驱逐腾位），workers 注册落点
   ◄────── 应答 {p, k, 接收端点} ───────────┤  （k=0：到此结束；驱逐后仍不足：拒绝）
   │
   ├═══ workers 推送 k 块 × 全部层 ═══════►  workers 按 key 接收进预分配块
   │                                        收齐 → 同锁重验前缀仍全在 GPU + 原子插树
   └─ 完成或超时 → 释放 hold                  （重验/插入失败 → 优雅退出，全部释放）
```

控制面（─►）一来一回各一次；数据面（═►）一次全量推送。任何一步失败都只是放弃写回，两侧零残留。

---

## 7. 代码设计：协议、调用链与改动清单

### 7.1 PD 之间的协议：三层约定

写回的协议不是一个东西，是三层，每层的契约不同。

**第 1 层：跨集群控制面（decode rank0 → prefill rank0）**。`RpcService` 上一个新的 unary RPC，与正向的 `StartLoad` 并排。这一层约定 unique_key 命名空间（`_wb_` 前缀与正向隔离）、全序列 cache_keys 与 token_ids（元数据非数据，双传互校，见 §1.1）、deadline、对称性校验字段，以及唯一带协商语义的东西——accepted range：正向 StartLoad 是"照单执行"，StartWrite 是"报价—回价"（decode 报全量 key，prefill 按 match 结果回 `{p, k}`）。

```protobuf
// rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto
rpc StartWrite(P2PConnectorStartWriteRequestPB) returns (P2PConnectorStartWriteResponsePB);

message P2PConnectorStartWriteRequestPB {
    int64  request_id   = 1;
    string unique_key   = 2;          // 命名空间与正向隔离（"_wb_" 前缀）
    int64  deadline_ms  = 3;
    repeated int64 cache_keys = 4;    // 全序列（只到完整块边界，须对齐 dropLastPartialBlock，见 §1.2）
    // 校验与扩展（§1.1）：prefill 用本地 initCacheKeys 从 token_ids 复算 keys 与 cache_keys 互校，
    // 不一致即拒绝（防两侧 hash 版本漂移）；token 序列也是未来按 token 索引扩展的原料
    repeated int32 token_ids  = 9;    // 完整 token 序列（prompt + response）
    // 价值门（§1.1 规则 1）：prompt 长度（token 数）。prompt 完整块数由它推出
    //（input_length / seq_size_per_block 向下取整），不单独设 prompt_blocks 字段，避免同义冗余
    int32  input_length       = 10;
    // 镜像一致性校验：decode 算的写回方向 plan 的 digest，语义见下文
    int32  decode_tp_size = 5;
    uint64 layout_digest  = 6;
}

message P2PConnectorStartWriteResponsePB {
    int32 error_code      = 1;
    string error_message  = 2;
    int32 accepted_start_block = 3;  // = p（device 层连续命中前缀 p_dev，§1.1 规则 2）
    int32 accepted_block_count = 4;  // = k；0 表示无需传输
    // 数据面端点。写回方向 prefill 是接收方，发送方（decode）必须拿到接收方 workers 的
    // transfer server 端点才能 push——与正向同构：正向把接收方（decode）端点放在 StartLoad
    // 请求里带给发送方（proto TPWorkerInfoPB），反向自然放在应答里带回。decode 侧的
    // P2PRoutingContext 只有 prefill rank0 的 RPC 地址（Meta.h:32-39），推不出 worker 传输端点
    repeated string prefill_worker_transfer_addrs = 5;
}
```

`layout_digest` 的精确语义（易错，写死在这里）：

- **比较对象是"写回方向的 plan"的 digest**：decode 算 `plan(decode_layout, prefill_layout推导)` 随请求发出，prefill 用真实 layout 算同方向 plan 后比对。`digest()` 包含 src/dst_rank 等 route 字段（KVCacheTransferPlanner.cc:568），但对称配置下正反向的对角线计划可能相同，digest 也可能相同。双方必须基于同一个写回方向的计划计算和比较，不能直接复用正向 load plan 的 digest，也不以正反向 digest 是否相等判断方向；
- 它验证**镜像一致性**（两侧从各自配置推出同一份计划，防漂移）；Phase 1 的"对称"是另一个更强的**本地断言**：decode 发 RPC 前检查 plan 已退化为对角线（每条 route `src_rank == dst_rank`、partition {1,0}、无 slice），不满足直接本地放弃；
- TransferPlan.h 注明 digest 原本"不上协议"，写回是首个上协议用途，属有意扩展。可选加固（Phase 1 不强制）：把 digest 短哈希追加进 `_wb_` 传输 key，让残余不一致退化为超时而非拷错数据。

**第 2 层：集群内控制面（rank0 → 本侧 workers）**。复用 `ExecuteFunction` 广播通道，`FunctionRequestPB` 新增两个 function type。这一层约定的内容是 route：每个 worker 收到"和对面哪个 rank、传哪个 tag、块内哪段字节"（对称版 route 平凡：对角线 + 整块）。

- `HANDLE_WRITE`：P0 → Pw，携带 recv routes + 分配好的 block_ids（落点）——接收方只登记落点，不需要知道对端地址；
- `WRITE`：D0 → Dw，携带 send routes + 对端（prefill workers）传输端点 + 本地 block_ids。对端端点来自 StartWrite 应答的 `prefill_worker_transfer_addrs`（见第 1 层）——decode 本地的 routing context 里没有它。

**第 3 层：数据面（worker ↔ worker）——零新增**。完全沿用 TransferTask 的 rendezvous 契约，对方向无感，写回只是把"谁注册、谁 push"换了边：

```
接收方先行：按 key 注册 (block_infos, deadline) 到 TransferTaskStore
发送方后到：push (key, data)
汇合条件：  key 逐字节相同（<unique_key>_wb_<layer>_<tag>_r<route_id>）
拷贝条件：  第 i 个子块长度逐位相等
```

### 7.2 调用链：与正向逐层对照

左列是已存在的正向 load，右列是写回，缩进层级一一对应——这就是"镜像胶水"的直观含义：

```
decode 侧（发起）
  正向 load                                   反向 write-back
  ────────────────────────────               ────────────────────────────
  引擎调度请求                                 StreamCacheResource::tryReleaseKVBlock
  → KVCacheManager::asyncLoadCache            → KVCacheManager::asyncWriteBack           ★新增薄转发
    （asyncRead 直调，.cc:684/702）              （同步 hold：incrKVCacheRef，必须先于 free，§2.1）
  → P2PConnector::asyncRead                   → P2PConnector::asyncWrite                 ★新增门面方法
  → P2PConnectorDecode::read                  → P2PConnectorDecode::write                ★新增
  → SchedulerDecodeRead::asyncRead            → SchedulerDecodeWrite                     ★新增文件
      ├ planFor()（plan 缓存）                    ├ 派生 cache_keys + planFor()
      ├ DecodeLoadHelper → StartLoad ─RPC─►       ├ DecodeWriteCaller → StartWrite ─RPC─►  ★新增
      └ broadcastPerRank(recv routes)             └ broadcastPerRank(send routes)
        → WorkerDecodeRead 注册 recv task           → WorkerDecodeWrite 一次性发送          ★新增文件

prefill 侧（响应）
  RemoteRpcServiceImpl::StartLoad             RemoteRpcServiceImpl::StartWrite           ★新增入口
  → P2PConnector::handleRead                  → P2PConnector::handleWrite                ★新增门面方法
  → P2PConnectorPrefill::processRead          → P2PConnectorPrefill::processWrite        ★新增
    （waitForResourceEntry 取登记的块）           （match + malloc + 建 ResourceStore entry）★真新逻辑
  → SchedulerPrefillRead::sendKVCache         → SchedulerPrefillWrite                    ★新增文件
      ├ planFor()（同一 plan 函数）               ├ planFor()（同一 plan 函数）
      └ broadcastPerRank(send routes)             ├ broadcastPerRank(recv routes + block ids)
        → WorkerPrefillRead 逐层发送                │  → WorkerPrefillWrite 注册 recv       ★新增文件
                                                  ├ 回复 {p, k, 接收端点}
                                                  └ settle → insertExternalBlocks        ★真新逻辑
```

这个对照直接读出两件事：每个 ★新增 函数在左列都有形状相同的对照物，写起来是翻译不是设计；只有标 ★真新逻辑 的两处（match+malloc 建 entry、settle 后外部插树）没有对照物，是实现风险的集中点（§7.4）。另外正向 prefill 侧的逐层流水在右列消失了——数据全好了，一次发。

### 7.3 可直接复用的逻辑：逐项论证

对 §6.1 的角色映射逐项展开，说明每个机制为什么能复用、到 Phase 2 还成不成立：

| 机制 | 正向（load） | 反向（write-back） | Phase 1 为什么可以复用 | Phase 2（非对称）能否复用 |
|---|---|---|---|---|
| 传输计划 | `KVCacheTransferPlanner::plan(prefill, decode)` | `plan(decode, prefill)`，同一个纯函数，参数互换 | 函数体不引用 RoleType，src/dst 的语义是"发送方/接收方"而非"prefill/decode"：dst 侧人人要收（各自独立显存）、src 侧副本类选举，这对语义写回方向同样成立。角色相关知识（pre_sliced 等）封装在 ShardLayout 构造阶段，不在 plan 内部 | 部分。Step 4a（head 配对）、Step 5（rank 展开+选举）、resolveKeys 方向无关，直接用；Step 1 的 CP 白名单（.cc:183，`dst.cp ∈ {1, src.cp}`）和 Step 3 的 `modulus = src.cpSize()`（.cc:320）按"prefill 分片、decode 不分片"设计，写回反转后变成 src 不分片→dst 分片，需要放开白名单并把 modulus 改为 `lcm(src, dst)` |
| 布局推导 | `ShardLayoutFactory::fromTopology` + `peerOf` | 同左，角色互换 | 布局是各侧真实 RoleType 的纯函数，与传输方向无关；`peerOf` 已做角色相关的 CP method 归一化（D4b 测试保证镜像一致） | 可复用，无需改动。写回时仍按真实角色构造：decode 用 `RoleType::DECODE` 建自己、`peerOf(..., PREFILL)` 推对端 |
| route 下发 | `RouteCodec` + `P2PBroadcastClient::broadcastPerRank` | 同左，收发两侧的编码方向互换 | 编码内容本质是"发送方字段"（route_id、对端端点、src partition/slice）和"接收方字段"（dst partition/slice + 解析好的键），与角色无关；广播通道是通用的 rank0→worker 管道 | 大部分可复用。两点适配：encodeForPrefill/encodeForDecode 按角色命名，§5 统一改名为 encodeForSender/Receiver；正向发送方不下发键规则是靠"本地投影=route 键集"的白名单性质，非对称写回后 decode worker 的本地全量≠route 键集，KeyShardSpec 要编进 TransferRoutePB 下发 |
| 数据面 rendezvous | decode 预注册 recv task，prefill push，`TransferTaskStore` 按 unique_key 汇合 | prefill 预注册，decode push | 汇合只认 unique_key 字符串和 recv task 的 block_infos，对谁是 prefill/decode 完全无感 | 原样复用（Phase 1 加的 `_wb_` 命名空间继续用） |
| 传输后端 | `TransferBackendFactory` 同时产出 sender + receiver | 同左 | backend 工厂每进程建出收发对并双向 regMem（P2PConnector.cc:34-95）；当前未用半边在 init 时被丢弃，写回接给 `*Write` worker（§5.4），零新机制、少量接线 | 原样复用 |
| 资源生命周期 | prefill 侧 `P2PConnectorResourceStore` + hold_ms + lease，保证传输期间块不被复用 | decode 侧需要同样的 hold | hold/TTL/超时清理是通用的资源租约机制，与方向和并行度都无关 | 原样复用 |
| 原子落账 | decode asyncRead 的 settle（全部到齐才生效，失败全释放） | prefill 侧 settle 后才 insert | "全部传输单元到齐才生效"的判定只依赖 route×layer 的完成计数，与方向无关 | 原样复用（非对称时每个 rank 的完成条件仍是"覆盖它的所有 route×layer 到齐"，机制不变） |
| 控制面 RPC | `DecodeLoadHelper`（原 PrefillLoadCaller，decode rank0 发起 StartLoad） | `DecodeWriteCaller`（decode rank0 → prefill rank0 发起 StartWrite） | 镜像新增，结构照抄；控制面只携带 cache_keys 与校验字段，不含并行度相关逻辑 | 可复用，仅对称性校验从"digest 相等"改为走 planner 白名单校验 |

### 7.4 需要新增什么

上表的机制层零新造，新增代码分两档。

**镜像胶水**——新写，但每一个都有正向对照物，结构照抄（即 §7.2 调用链里的 ★新增）：

- `DecodeWriteCaller`：照抄 `DecodeLoadHelper`（正向 StartLoad caller）的异步 RPC 骨架；
- decode 侧 `asyncWriteBack()`：照抄 `asyncRead` 的编排模式（hold → plan → RPC → 广播 → 回调收尾）；
- prefill 侧 `handleWrite()`：照抄 decode 收侧的 settle 模式；
- 两侧 worker 的 `write` / `handleWrite`：复用对侧 worker 的收发基础；connector 解析协议并传入 `P2PWorkerRoutePlan`，worker 启动入口返回 `ErrorInfo`，取消和状态查询使用独立接口。`handleWrite` 登记接收任务即返回，不等待发送或接收完成。

任务状态沿用 group + lease 的组织方式：Decode Read 与两侧 Write 共用 `P2PTransferLease`（原 `DecodeTargetWriteLease`），统一任务计数与停止判定；Write group 另行保存业务结果及终态。lease 不提供数据拷贝回滚或插树原子性，完整发布和资源归还仍由 scheduler/settle 负责。

一块"勿抄"牌：正向 `handleRead` 的收尾有一半是 side channel（first token / reuse len / MTP）的等待与清理记账——`waitAndFillResponse` 的阻塞等待（P2PConnector.cc:504 起）加散布全函数的十来处 `clearSideChannelData`。写回没有 side channel：`handleWrite` 填完 `{p, k, 接收端点}` 即返回，数据面成败由 settle 与 deadline 收尾。§4 的解耦（Slice 0）落地后这套记账整体移入 `PrefillResultStore`，handleWrite 天然碰不到。

**真正的新逻辑**——正向链路里没有对应物，是实现风险的集中点（即 §7.2 里的 ★真新逻辑，只有两处）：

1. **prefill 侧"无 stream 的块生命周期管理 + 外部插树"**（`mallocForExternalInsert` / `insertExternalBlocks`）。k 个新块由分配接口返回的 owning RAII 句柄接管原始引用，再由 ResourceStore entry 和接收上下文持有（§3.1），接收端不额外 `incrKVCacheRef`。settle 需要新增同锁接口：完整预检 GPU 前缀和整个后缀的节点冲突，通过后发布，再归还所有新块的分配引用；失败时保证零发布，并在物理传输停止后归还句柄；
2. **握手协商语义**（match → accepted range）。正向是"decode 声明要什么、prefill 照给"；写回是"decode 报全量 key、prefill 算增量再回价"，两阶段协商是新协议逻辑。

这两处与"是否保留 stream"无关：settle 的"收齐才插、失败零残留"和握手协商在任何方案下都要新写。接收端需要的是纯探测、owning 分配和同锁插入接口；StartWrite 返回后的任务持有者与完成汇总仍按 R4 单独明确，不能假定整条链都在握手 RPC 线程内完成。

传输基建没有新机制：双向 regMem 每进程已就绪（§6.1），写回只需留住 init 中被丢弃的半个 backend 并交给 `*Write` worker（§5.4）。第 1 处也是 §7.8 把"prefill 收侧"单独作为 Slice 2、用测试 RPC 先行验证的原因。

### 7.5 改动位置

表中文件/类名均为 §5 重构后的新名。

| 位置 | 改动 | 性质 |
|---|---|---|
| `p2p/DecodeWriteCaller.{h,cc}` | decode rank0 发起 StartWrite 的异步 RPC（结构镜像 `DecodeLoadHelper`） | 新增 |
| `p2p/P2PConnectorResourceStore` | `addResource` 加一个不依赖 Meta 的重载（现有签名依赖 Meta，入口 `registerResource`，P2PConnector.cc:244；写回直接传 unique_key/request_id/deadline） | 扩展 |
| `p2p/P2PSchedulerDecodeWrite.{h,cc}` | 写回发起编排：算 keys/plan、调 caller、下发 send routes、超时管理（hold 在 KVCacheManager 薄转发内同步完成，§2.1）；现 SchedulerDecode 纯改名为 `P2PSchedulerDecodeRead`（§5） | 新增文件 |
| `p2p/P2PSchedulerPrefillWrite.{h,cc}` | `handleWrite()`：校验、match、malloc、下发 recv routes、settle、insert；现 SchedulerPrefill 纯改名为 `P2PSchedulerPrefillRead`（§5） | 新增文件 |
| `p2p/P2PWorkerPrefillWrite.{h,cc}` | 新增：持 receiver（接住 init 中被丢弃的半个 backend，§5.4），按 key 把落点块注册进 TaskStore、收齐通知 settle | 新增文件 |
| `p2p/P2PWorkerDecodeWrite.{h,cc}` | 新增：持 sender（同上），按 routes 构造一次性 SendRequest 推送 | 新增文件 |
| `P2PConnectorDecode` / `P2PConnectorPrefill`（纵切） | init 接线：把 backend pair 中被丢弃的另一半交给本纵切的 `*Write` worker；新增 `write` / `processWrite` 转发入口 | 扩展 |
| `p2p/P2PConnector` | 新增 `asyncWrite` / `handleWrite` 门面方法并按纵切分发（本分支接口简化后已无 asyncWrite stub） | 扩展 |
| `p2p/P2PKeyUtil.h` | `makeWriteBackRouteLayerKey()`（`_wb_` 命名空间） | 新增 |
| `RemoteRpcServiceImpl`（model_rpc） | StartWrite 的 service 端接入，转发给 prefill 侧 P2PConnector | 扩展 |
| `StreamCacheResource::tryReleaseKVBlock`（.cc:297） | FINISHED 分支内、free 之前调用 `cache_manager->asyncWriteBack`（生产链路无 coordinator，KVCacheManager.cc:800） | 修改（约 5 行） |
| `KVCacheManager` / allocator | ① `asyncWriteBack()` 发送侧薄转发仍用 `incrKVCacheRef` hold；② 接收侧 `mallocForExternalInsert(...)` 返回接管原始分配引用的 owning RAII 句柄，含部分分配回滚；③ `insertExternalBlocks(...)` 接入新增的同锁完整预检与插入接口，显式返回整体结果和采纳范围 | 新增方法 |
| `BlockTreeCache` | 纯探测后缀冲突；在一次锁持有期间完成 GPU 前缀、全部既有后缀节点状态检查及发布。Phase 1 对低层或忙碌节点冲突放弃，不新增节点升级能力 | 新增接口（Slice 2，见 R3、R8） |
| decode 侧 `Meta::P2PRoutingContext` | prefill rank0 地址/`prefill_tp_size` 需存活到 stream 结束（现为 load 期使用，实现时确认生命周期）。注意它只有 rank0 的 RPC 地址（Meta.h:32-39），够发 StartWrite 控制面；数据面的 prefill worker 传输端点由 StartWrite 应答带回（§7.1） | 确认/微调 |

### 7.6 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `p2p_writeback_enable` | false | 总开关（两侧都要开：decode 决定发不发，prefill 决定收不收） |
| `p2p_writeback_timeout_ms` | 5000 | decode 侧 hold 的硬 deadline |
| `p2p_writeback_max_inflight` | 4 | decode 侧并发写回上限，超限跳过 |
| `p2p_writeback_early_threshold_blocks` | 0（关闭） | 输出累计满 N 个完整块即分段提前写回（§2.2 可选扩展，Phase 1 仅预留） |
| `p2p_writeback_target_tier` | DEVICE | prefill 落账 tier（DEVICE / HOST） |

### 7.7 失败模式与兜底

| 失败 | 行为 | 影响面 |
|---|---|---|
| decode 侧开关关 / 超并发 / routing context 缺失 | 不发起，直接走原释放路径 | 无 |
| StartWrite RPC 失败 / 超时 | decode 释放 hold，放弃 | 无 |
| decode 侧 planFor()/对称断言本地失败（validateTag 的全部检查在本地跑，此时已 hold） | 不发 RPC，立即释放 hold | 无 |
| decode 本地校验通过，但 prefill 侧 digest 比对不一致 | prefill 拒绝，decode 释放 hold | 无 |
| token_ids 复算 keys 与 cache_keys 不一致（两侧 hash 版本漂移，§1.1） | prefill 拒绝，decode 释放 hold | 无 |
| prefill match 全命中（k=0） | 握手即结束 | 无（快路径） |
| 价值门拒绝：prompt 前缀有块不在 GPU（p_dev < prompt_blocks，§1.1 规则 1） | prefill 拒绝，decode 释放 hold | 无 |
| 准入发现候选后缀已有 HOST/DISK 或忙碌 FULL 节点 | 分配前拒绝，记录 existing_suffix_conflict；不覆盖或升级旧节点 | 已有缓存不变 |
| prefill malloc 不足 | 照常驱逐腾位（规则 3）；驱逐后仍不足（极端）→ 拒绝，decode 释放 | 无 |
| 接收资源登记失败或部分分配失败 | owning 句柄/分配接口回滚原始分配引用；已启动接收时先等待物理停止 | 无分配引用残留 |
| 数据面部分传输失败 / 超时 | prefill settle 失败：释放全部 k 块、不插树；decode 到 deadline 释放 hold | 无残留 |
| 传输窗口内前缀被驱逐/降级出 GPU（严格档重验失败） | 优雅退出：不插树、释放 k 块、erase entry、记 writeback_abandoned | 无残留（白传一趟） |
| settle 发现既有后缀已降到低层或进入忙碌状态 | 完整预检失败，零发布；记 existing_suffix_conflict，确认物理停止后归还分配句柄 | 已有缓存不变，新块无残留 |
| decode 传输中途进程退出 | prefill recv task 超时清理，块释放 | 无残留 |
| prefill settle 成功但 ack 丢失 | decode 到 deadline 释放 hold；prefill 树上已可见（结果正确，仅 metric 记为超时） | 可接受 |

所有路径共同的不变量：decode 侧的 hold 一定在 deadline 内解除；prefill 侧要么整段插树、要么零残留。

顺带说明 ack 的取舍：prefill settle 后不回显式 ack 也能工作（decode 靠传输回调 + deadline 收尾），ack 的唯一价值是 metric 精确性，Phase 1 不加。

### 7.8 可观测性与验证

Metrics（挂 `P2PConnectorMetrics`）：写回发起/跳过（分原因）/成功/失败次数、握手快路径（k=0）占比、传输字节与耗时、prefill 拒绝原因分布、decode hold 时长分布。

测试：

1. planner 单测：src/dst 互换的对称退化断言（`plan/test/`）；
2. `P2PConnectorTest` 集成：成功回传 / k=0 快路径 / malloc 拒绝 / 传输超时释放 / decode 中途退出，五条路径；
3. smoke（挂 `suites_h20_oss.bzl` decode_entrance 组）：两轮请求，第二轮 prompt = 第一轮 prompt + response，断言 aux_info 的 `prefill_local_reuse_len` 覆盖到 response 部分（该字段现成，见 `q_r_dp_sep_p2p_reuse.json` 的断言方式）。
4. 接收引用账本：分配后仅一份引用；部分分配/登记失败回到原始空闲块数；成功插树后仅保留树引用；并发 DEVICE 重复结果对应的新块归零；超时但仍在途时不得提前释放。
5. 后缀冲突：prompt 在 GPU、response 后缀在 HOST/DISK 时准入拒绝；准入后后缀降级时 settle 零发布；较早的空节点后接 HOST/忙碌节点时，前面的空节点也不得被部分填充；完整 DEVICE 重复前缀可安全复用并继续插入新后缀。

实现切片共五片（Slice 0～4，每片独立可编译、可测）：

| 切片 | 实现范围 | 验证边界 |
| --- | --- | --- |
| Slice 0：Read 重构 | §4 side channel 拆分及 §5 现有 scheduler/worker 改为 Read 命名 | 现有正向路径回归 |
| Slice 1：写回基础设施 | proto、两个轻量 Write worker、backend 另一半持有、角色分发和广播控制接口；Decode 使用 `writePerRank → P2PWorkerDecodeWrite::write`，Prefill 使用 `processWritePerRank → P2PWorkerPrefillWrite::handleWrite`；connector 负责协议与操作分发，worker 接收内部 route plan | 显式下发任务验证注册、发送、查询和取消；真实 RPC/TCP 验证，无生产请求触发 |
| Slice 2：Prefill 接收与插树 | `StartWrite` 服务处理、`handleWrite` / `processWrite`、`P2PSchedulerPrefillWrite`；准入与范围协商、owning 分配、接收汇总、settle/insert、失败释放；包括原始分配引用交接、HOST/DISK 后缀冲突拒绝和同锁完整预检与发布 | 测试发起方驱动完整接收流程，验证引用账本和失败收尾 |
| Slice 3：Decode 发起与收尾 | `P2PConnector::asyncWrite` / `P2PConnectorDecode::write`、Write caller 和 `P2PSchedulerDecodeWrite`；源块保活上下文、握手、routes 发送、并发与超时管理、CANCEL/QUERY 和释放 | 模拟 Prefill 服务驱动完整发起流程；使用测试提供的源块 hold 验证生命周期，不接入请求收尾触发 |
| Slice 4：生产链路接通 | `KVCacheManager::asyncWriteBack`、`tryReleaseKVBlock` 触发接线，在 free 前同步建立源块 hold 并交给 Slice 3 上下文；确认 KV 就绪及 routing context 生命周期，连接真实两端 | 端到端集成、资源交接和异常路径测试、正向回归及写回 smoke |

Slice 2/3 开工前先明确共同契约：StartWrite 的接受范围、拒绝/`k=0` 与接收端点语义，任务标识、keys/digest/routes 对齐，worker START 失败后的 CANCEL/QUERY，以及源块和接收块的所有权交接与物理停止条件。两个 scheduler 围绕相同契约分别实现和验证；Slice 4 验证生产触发与真实两端协作。涉及尚待讨论的接口时，先确认对应条目。实现进度、当前命名与验证记录见[实现文档 §1](decode_kvcache_writeback_implementation.md#1-当前范围与实现计划)。

---

## 8. 下一步计划（Phase 2：放开非对称）

Phase 1 走 planner、route 驱动、双端镜像，所以 Phase 2 基本收敛为 planner 的能力扩展：

1. 白名单新形态「src 不分片 → dst CP 分片」：写回方向 decode 恒 cp=1（decode 计算不做 CP，PD 分离下 decode 持整段），prefill 可能 cp=N。数学上简单，每个 dst cp_rank 只收自己剩余类的键，不需要 lcm/CRT；
2. cp_slice 反向：「src 持整块、dst 预切片」的字节切分方向 `assignSlices` 已覆盖（KVCacheTransferPlanner.cc:439），但该分支目前在 planner 用例中**零测试覆盖**（唯一测 cp_slice 的 A9 只测正向），需先补 planner 单测再做执行层验证；
3. TP 不对称：ND1P/NP1D 的 head 配对逻辑方向无关，删除对称断言后直接可用，需补测试；
4. 键集规则需要真正下发到 worker：对称版规则平凡，rank0 可以代解析；非对称后 decode worker 侧的发送键集不再等于本地全量。

---

## 9. Review comments（待讨论）

> R1–R6 保留原有待讨论意见，相关未决接口和行为继续按各条说明确认。R7/R8 是本轮补充的接收引用与后缀冲突问题，正文已明确处理策略，归属尚未实现的 Slice 2；不代表当前 Slice 0 存在接收写回泄漏或已实现节点升级。

### R1 [P1] 超时不能直接释放传输中的块

- **对应章节**：§2.2 的硬 deadline、§7.7 的超时释放与 hold 不变量。
- **问题与依据**：`transfer/TransferTask.cc` 中的 `TransferTask::cancel()` 对正在传输的任务只记录取消意图，等待 `notifyDone()` 才真正结束。业务超时后立即归还源块或目标块，后续传输可能访问已被其他请求复用的显存。现有 `P2PConnectorWorkerDecode` 在取消后也保留 lease，以追踪物理传输是否停止。
- **待讨论建议**：区分业务超时和物理传输停止。超时停止业务等待、禁止插树；确认传输停止后再释放资源。相应调整“hold 必须在 deadline 内解除”的不变量，明确异常传输期间资源由谁继续保活。
- **验证要点**：传输进入 TRANSFERRING 后超时，确认块不会提前归还；迟到完成后只清理一次，并且不能再插树。

### R2 [P1] 完整 token 序列不等于完整 KV 序列

- **对应章节**：§1.1 的 cache_keys 派生与互校、§1.2 的块边界、§2.1 的收尾快照。
- **问题与依据**：最后采样出的 token 通常还没有经过下一次 forward。块大小为 8、结束时共有 16 个 token，但 KV 只计算到前 15 个时，只能回传一个完整块。直接从最终 `completeTokenIds` 重新生成两个完整块的 keys，不能保证与本地插树一致。现有 `KVCacheManager::malloc()` 在计算前维护 keys，`insertIntoCache()` 使用已有 keys 并丢弃半块；`GenerateStream` 的 token 更新发生在采样结果返回后。
- **待讨论建议**：收尾时同步快照本地插树使用的有效 keys、对应资源和 KV 就绪边界，发送范围受该边界约束；接收端复算也只比较已声明有效的范围，不能仅按最终 token 数向下取整。
- **验证要点**：结束长度恰好落在块边界、边界前后各一个 token，以及 MTP 接受多个 token 后结束。

### R3 [P1] 现有 match 不是纯探测，缓存接口需要补充

- **对应章节**：§1.1 的准入探测、§3.1 的同锁重验与插树、§7.5 的 allocator 接口。
- **问题与依据**：`block_tree_cache/load/BlockTreeLoader.cc` 的 `createMatchResult()` 会增加 device 块引用，并为低层命中登记 `LOAD_PENDING` 等加载状态。只读取 `matched_device_blocks` 后丢弃结果，会遗漏引用释放。另一个独立约束是：现有公开 `BlockTreeCache::match()` 和 `insert()` 分别加锁，外层顺序调用无法保证重验与插树原子执行。
- **待讨论建议**：明确新增不持块、不登记加载的探测接口；同时在 BlockTreeCache 内提供“同锁重验前缀并插入”的接口，将 §3.1 的原子性要求落实到 API，明确准入拒绝与重复插入时的引用归还责任。
- **验证要点**：准入拒绝后 refcount 不变、不留下加载状态；传输期间前缀被驱逐或降级时放弃插树；相同前缀并发写回时，未采纳的块正确释放。

### R4 [P2] StartWrite 返回后的完成状态汇总与所有权尚未定义

- **对应章节**：§7.2 的握手和 settle 调用链、§7.4 的 RPC 返回时机与线程模型。
- **问题与依据**：§7.4 一处要求 `handleWrite` 填完握手响应即返回，另一处又称包括 insert 的整条链都在 RPC 线程内完成。现有接收调用包含“注册后等待传输结束”；直接照搬会使握手等待数据，而 decode 又在等待握手后才发送。
- **待讨论建议**：明确各 rank 注册接收任务后何时回复，哪个后台上下文在 StartWrite 返回后继续持有 entry，以及如何汇总所有 rank 的完成结果并触发 settle。区分“接收任务注册成功”和“全部数据传输成功”两个事件。
- **验证要点**：握手能在尚未发送数据时返回；任一 rank 注册失败、传输失败或迟到完成时，整体只进入一次终态且不会提前插树。

### R5 [P2] PrefillResultStore 的无条件 notify 与 seal 冲突

- **对应章节**：§4.2 的单 map、notify、seal 与 deadline 语义。
- **问题与依据**：仅有 `{data, deadline}` 且 notify 无条件写入时，如果 seal 删除记录，迟到 notify 会重新创建已结束请求的数据。当前 `P2PConnectorResourceStore::notifySideChannelReady()` 会检查终态，并约束数据的保留期限；拆分时需要保留这些行为。
- **待讨论建议**：单 map 可以保留，但应明确等待、就绪、终态的状态转换，以及终态记录的保留期限；明确资源租约提前失效如何结束结果等待，并拒绝迟到通知。
- **验证要点**：notify 先到、wait 先到、seal 后迟到 notify、租约提前过期，以及重复终态处理。

### R6 [P2] 对称 TP/CP 不足以保证所有 group 都按 k 个块处理

- **对应章节**：§1.3 的完整接受区间、§3.1 的分配与收齐条件、§7.5 的外部插树接口。
- **问题与依据**：现有 planner 的 `resolveKeys()` 会按 `active_tail_blocks` 截取键集，Hybrid allocator 也允许部分 key 缺少非 FULL group 的资源。仅限制 TP/CP 对称，不能推出每个 group 都为 `[p, p+k)` 分配并发送 k 个物理块；按此假设实现可能等待不会发送的块，或发布没有填充的数据。
- **待讨论建议**：明确 Phase 1 的 group 支持范围。若覆盖 DSV4 的混合 group，需要定义各 group 的实际分配范围、键集、依赖关系和完成条件；若暂时只支持 FULL，则在准入条件中显式限制。
- **验证要点**：混合 group 的尾部键集、缺省资源与插树映射一致；未支持的 group 配置在数据传输前被拒绝。

### R7 [P2] 接收侧原始分配引用缺少释放责任（本轮 comment 4）

- **依据**：`FullKVCacheGroup::malloc` 已增加一次引用，`SingleTypeKVCacheAllocator::incrKVCacheRef` 再增加一次；后者的 guard 只减少一次，写回接收侧没有 stream 的 `free()` 释放原始引用。
- **正文处理**：§3.1 选择由 `mallocForExternalInsert` 返回直接接管原始分配引用的 owning RAII 句柄；接收侧不再额外增引用。成功插树后也释放分配引用，树自行持有 CACHE 引用；未采纳的新块和失败分配由同一所有权规则回收。
- **状态与验证**：Slice 2 待实现，引用账本和失败回滚测试见 §7.8。当前 Slice 0 没有写回接收分配路径，不将其归类为现有泄漏。

### R8 [P2] HOST FULL 后缀会阻断普通插入（本轮 comment 5）

- **依据**：`BlockTreeTest.InsertHardStopsAtExistingHostFullNode` 明确断言已有 HOST FULL 节点不被覆盖，后续新节点也不插入。p=p_dev 只能给出 GPU 连续前缀长度，不能证明后缀不存在。
- **正文处理**：§1.1/§3.1 明确 Phase 1 对低层或忙碌 FULL 节点冲突放弃；准入及 settle 均检查，settle 在同一把树锁内完整预检后再发布。完整稳定的 DEVICE 重复结果可复用，未采纳的新块归还分配引用。
- **状态与验证**：Slice 2 待实现；覆盖准入时已有 HOST 后缀、传输中降级、忙碌节点和部分发布防护。安全的 HOST→DEVICE 节点更新接口不在本阶段范围。
