# BlockTreeCache 验收场景

> 本文定义 `BlockTreeCache`（block cache tree）重构的验收范围与场景清单。  
> 配套设计：[设计文档（主）](rtp_llm_block_tree_cache_design.md) · [淘汰候选](rtp_llm_block_tree_cache_eviction_candidate_design.md) · [引用计数生命周期](rtp_llm_block_tree_cache_refcount_lifecycle_design.md) · [代码参考](rtp_llm_block_tree_cache_code_reference.md) · [示例](rtp_llm_block_tree_cache_examples.md)

---

## 1. 重构范围摘要

### 1.1 替换关系

| 旧组件 | 新组件 / 去向 |
|--------|----------------|
| `SharedBlockCache` + Device 前缀树 + `leaf_lru_` | `BlockTreeCache` + `BlockTree` + `EvictionHeap`（经 `BlockTreeEvictor`） |
| `BlockCache`（扁平 LRU） | 删除；由统一树 + heap 承担 |
| `PrefixTreeMemoryBlockCache` + Host 独立树 | 并入同一棵树的 Host tier |
| `KVCacheMemoryConnector` 中缓存管理 / Disk 协调 | 移入 `block_tree_cache/`（CopyEngine、Demotion、LoadBack、Broadcast） |
| Namespace（`kDefault` / `kGpuLogical` / `kGpuCpCanonical`） | **删除**；CP 由树外 `CPSlotMapper` + canonical keys 处理 |
| `setStateBlockIndependentEviction` | `CacheEvictPolicy::INDEPENDENT` / `enable_independent_group_eviction` |

### 1.2 核心 API 映射（验收时按新名）

| 旧 API | 新 API |
|--------|--------|
| `put` | `BlockTreeCache::insert` |
| `match` / `matchGroup` | `BlockTreeCache::match` → `BlockTreeMatchResult` |
| `selectAndEvict` / `evictAndFree` | `reclaimBlocks`（+ watermark demotion） |
| `selectAndEvictForGroup` / `evictAndFreeForGroup` | `evictForGroup`（+ `DeviceKVCacheGroup::ensureFreeBlocks`） |
| leaf LRU 链淘汰 | FULL：仅叶子入 `EvictionHeap`；SWA/LINEAR：ready 节点可入堆 |
| resident keys | 字段 `InsertInfo.is_resident` 仍在；**allocator 当前未消费**（见 N3） |

### 1.3 主要代码与测试入口

| 类型 | 路径 |
|------|------|
| 实现 | `rtp_llm/cpp/cache/block_tree_cache/` |
| 单测 | `block_tree_cache/test/BlockTreeCacheTest.cc` 等 |
| 淘汰 | `block_tree_cache/test/block_tree_cache_eviction_test/`、`BlockTreeEvictorTest.cc` |
| 集成 | `BlockTreeCacheIntegrationTest.cc`、`BlockTreeCacheBroadcastTest.cc` |
| Allocator/Manager | `rtp_llm/cpp/cache/test/`（含 DSV4 / Hybrid / CP） |

---

## 2. 验收节奏（优先级）

| 阶段 | 覆盖域 | 合入要求 |
|------|--------|----------|
| **P0** 门禁 | A、B、C、D1–D4、E1/E4、H、J1–J4、I1–I2 | 主干合入必过 |
| **P1** L2/L3 | F、G、K、M2/M3 | 开启 Host/Disk 配置前必过 |
| **P2** 策略/压测 | C11、L4、E5–E7、J6、M 全矩阵 | 发布 / 大流量前必过 |
| **签字项** | N1–N4 | 范围裁剪或补齐实现后签字 |

每条场景建议记录：**前置配置 → 操作步骤 → 可观测断言 → 通过/失败**。

---

## 3. 验收场景清单

### A. 拓扑与 Match / Insert

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| A1 | 空树 match | 返回空前缀、无 load_back ticket | P0 |
| A2 | 单链 insert → 全路径 match | 路径长度、pool/block 顺序正确 | P0 |
| A3 | 部分前缀 hit | 只返回共同前缀；可在 hit 节点下继续 insert | P0 |
| A4 | 分叉树（同父多子） | 两叉互不影响；各自可独立 match | P0 |
| A5 | 重复 insert 同 key | 不新建节点；已有 slot 保留；调用方持有「败者」block 所有权 | P0 |
| A6 | 空 keys no-op | match / insert / reclaim 安全返回 | P0 |
| A7 | 多 ComponentGroup 同树 | FULL+SWA(+LINEAR) 共路径；match 按 group policy 收集 blocks | P0 |
| A8 | `insert(parent=matched_node, …)` | 仅挂在指定父下，不误挂 root | P0 |
| A9 | key snapshot / version | 树变更后 version 递增；snapshot 与当前一致（含 limit） | P1 |

### B. 引用计数与生命周期

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| B1 | insert → match → release → reclaim | ref 归零后才可淘汰；持有期间 victim 被 skip | P0 |
| B2 | 双并发 match 同叶 | 首次 release 不入堆；末次 release 后 `refreshCandidate` 恢复 | P0 |
| B3 | request pin / cache hold | 请求占用块不进 demotion / reclaim victim | P0 |
| B4 | shutdown 排空 | 各 tier 树 hold 释放；外部 co-holder 存活时只放树 hold | P1 |
| B5 | ticket 长于 Host/Disk shutdown | ticket abort/commit 语义明确，无 UAF | P1 |
| B6 | `onBlocksReleased` 后 cache-only 块 | 成为 heap 候选并可被 `ensureFreeBlocks` 回收 | P0 |

### C. EvictionHeap / 候选资格

> 对齐 [淘汰候选设计 §12](rtp_llm_block_tree_cache_eviction_candidate_design.md#12-测试与验收)。

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| C1 | 热节点连续大量 upsert | 物理条目恒为 1 | P0 |
| C2 | LRU 只跟真实访问 | refcount 抖动 / 回滚 / 父提升 **不**改相对序 | P0 |
| C3 | FIFO 只跟 admission | match / 回滚不改 `admission_seq` 序 | P1 |
| C4 | LFU 策略 | 访问频次决定序；与 LRU 对照可区分 | P1 |
| C5 | erase 同步 | 有序索引 + node 索引无残留 | P0 |
| C6 | FULL 仅叶子入 device heap | 中间节点只刷访问历史，不入堆 | P0 |
| C7 | SWA/LINEAR ready 节点均可入堆 | 非仅叶子 | P0 |
| C8 | 扩展叶子后父刷新 | 只重新判定直接父；不扫更高祖先 | P0 |
| C9 | insert overlap | 不刷新访问历史 / 不改 LRU | P0 |
| C10 | takeBest 遇 refcount>1 | skip；不把热节点当 victim | P0 |
| C11 | 冷候选 + 热 match 压测 | heap size = ready 候选数；无失效扫描膨胀 | P2 |

### D. 同步 Reclaim / 按 Group 淘汰

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| D1 | `reclaimBlocks(n, DEVICE)` | 释放 ≥n device blocks；**不**分配 host | P0 |
| D2 | 单链顺序 drain | 子→父逐级删；父变叶入堆 | P0 |
| D3 | `evictForGroup(gid, n)` | 只压该 group 的 device；返回数正确 | P0 |
| D4 | `ensureFreeBlocks` 压力路径 | pool 不足时循环 `evictForGroup` 直到 free≥need 或明确失败 | P0 |
| D5 | tier 关闭时 reclaim | 关闭 tier 返回 0；host 关则直接 release | P1 |
| D6 | 节点所有 group 空 | 节点从树删除 | P0 |
| D7 | `CacheEvictPolicy::NONE` | 该 group 永不因 cache 淘汰释放 | P1 |
| D8 | NON_REUSABLE group | **不入树**；allocator 直管；match 不可见 | P0 |

### E. 级联与独立淘汰

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| E1 | FULL device reclaim → 级联 SWA | 同路径 SWA slots 一并处理 | P0 |
| E2 | FULL → LINEAR 级联 | 同上 | P0 |
| E3 | FULL+SWA+LINEAR 三联级联 | 顺序与堆组成符合设计 | P1 |
| E4 | SWA `INDEPENDENT` 淘汰 | **不**拖垮 FULL；FULL 前缀可继续 match | P0 |
| E5 | `enable_reverse_eviction=true` | 任意 leaf group 淘汰反向级联兄弟 group | P2 |
| E6 | reverse 关 | 仅按 priority / cascade 规则，不反向 | P1 |
| E7 | 级联遇 pinned / loading / demoting | skip 该 sibling；成功条保留；失败可重试 | P2 |
| E8 | 分叉：两叶均可淘汰 | 先淘汰一叶，另一叶与共同祖先状态正确 | P0 |

### F. Tier Demotion（水位线 / Device→Host→Disk）

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| F1 | Device 超 watermark | 异步 demotion：source 出堆、`DEMOTING`、成功后 GPU 释放、Host 入堆 | P1 |
| F2 | Host 超 watermark → Disk | 同理；Disk 默认 FIFO | P1 |
| F3 | demotion 失败 | source 按原序恢复；target 释放；无脏 slot | P1 |
| F4 | `device_min_free_blocks` | 请求 release 后强制腾出绝对 headroom | P1 |
| F5 | 仅 Device（Host/Disk off） | demotion 不发生；走 direct reclaim | P0 |
| F6 | 夹心饼干：子全删 → 父变 DeviceLeaf | 自动入堆，无需显式 cascade 检查 | P0 |
| F7 | TP broadcast demotion | 多 rank 编码 / 提交 / 回滚一致 | P1 |

### G. Load Back（Host/Disk → Device）

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| G1 | match 命中 Host → ticket | commit 触发 copy；abort 不 copy | P1 |
| G2 | Disk→GPU 经 Host DMA | **不**在 Host 建缓存条目（仅中转缓冲） | P1 |
| G3 | SWA load_back 仅窗口 | 窗口外不回灌 | P1 |
| G4 | `LOADING_BACK` 第二请求 | 复用同一 AsyncContext；不重复迁移 | P1 |
| G5 | load_back 成功后入堆时机 | copy 完成 + slot 稳 + 外部 ref 释放后才进 device heap | P1 |
| G6 | 队列拒绝 / 目标校验失败 | 回滚 tree holders；请求目标语义正确 | P1 |
| G7 | 映射 metadata 非法 | init/preflight 拒绝（越界/重复/空洞）；原子失败 | P1 |
| G8 | `enable_load_back=false` | 不发 ticket；仅 device 前缀可用 | P1 |
| G9 | shutdown 等待 claimed commit | registry 等待；close 只 abort 一次 | P1 |

### H. Component 语义（FULL / SWA / LINEAR）

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| H1 | FULL validator | 仅完整可用 device slot 可匹配 | P0 |
| H2 | SWA window after gap | gap 后必须满足窗口才 reusable | P0 |
| H3 | SWA 稀疏 / 断连窗口 | match / load_back 边界正确 | P1 |
| H4 | LINEAR chain dependency | resource 侧 linear deps 与树 key 序一致（树边不靠 `BlockDependency`） | P0 |
| H5 | 混合模型前缀复用（如 DSV4） | insert 同前缀 → reuse 命中；SWA 保尾 block | P0 |
| H6 | state / NON_REUSABLE（如 HCA state） | 不进 tree；推理路径仍正确分配 | P0 |

### I. CP / 无 Namespace 回归

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| I1 | CP>1 page-RR insert | 树内仅 canonical keys；无 NamespacedKey 双父 | P0 |
| I2 | CP match 用 last-rank keys | `localCacheKeys` / `cpEffectiveCacheKeys` 与 allocator 一致 | P0 |
| I3 | CP 下 `ensureFreeBlocks` | 只释放本 rank 物理块；无串释放 | P1 |
| I4 | CP 短请求（不足 cp_size 逻辑块） | 空 canonical 可跳过；行为与设计一致 | P1 |

### J. Allocator / Manager 集成

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| J1 | `KVCacheManager` 启动创建 BlockTreeCache | 旧 SharedBlockCache 路径不存在 | P0 |
| J2 | SingleType：malloc → insert → reuse → free → reclaim | 端到端无泄漏 | P0 |
| J3 | Hybrid：多 group reuse + insert | 与 device_group 对齐 | P0 |
| J4 | pool 耗尽触发 eviction | 推理续写不 OOM，或失败可预期 | P0 |
| J5 | factory：budget / pin / disk mount | 配置错误失败可预期；正确配置可建池 | P1 |
| J6 | `enable_independent_group_eviction` vs `evict_policy=INDEPENDENT` | **明确以谁为准**并端到端验证 | P2 |

### K. CopyEngine / Pool / IO

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| K1 | D2H / H2D / H2Disk / Disk2H | 转换成功；非法 layout 拒绝 | P1 |
| K2 | Device / Host / Disk pool malloc/free | 对齐、stride、容量正确 | P1 |
| K3 | Disk mount guard | 锁目录、stale 清理、直接 IO 对齐 | P1 |
| K4 | copy 失败回滚 | Evictor staged-hold 回滚完整 | P1 |

### L. 并发与稳定性

| ID | 场景 | 通过标准 | 优先级 |
|----|------|----------|--------|
| L1 | 多线程 match / insert | 无死锁、无数据竞争 | P0 |
| L2 | match 与 demotion / load_back 竞态 | prepareMove pin 规则；无双释放 | P1 |
| L3 | ticket registry 并发 shutdown | 共享 detached abort 完成 | P1 |
| L4 | 长稳：大量分叉前缀 + 水位淘汰 | heap 有界、节点数可解释、无 FD/内存泄漏 | P2 |

### M. 配置矩阵

| ID | Device | Host | Disk | LoadBack | Reverse | 期望 | 优先级 |
|----|--------|------|------|----------|---------|------|--------|
| M1 | ✓ | ✗ | ✗ | ✗ | ✗ | 纯 GPU 树 + direct reclaim | P0 |
| M2 | ✓ | ✓ | ✗ | ✓ | ✗ | Device↔Host demotion / load_back | P1 |
| M3 | ✓ | ✓ | ✓ | ✓ | ✗ | 三级 demotion + Disk load_back | P1 |
| M4 | ✗ | ✓ | ✓ | — | — | 纯元数据 / 存储节点语义 | P2 |
| M5 | ✓ | ✓ | ✓ | ✓ | ✓ | 反向级联开启 | P2 |
| M6 | ✓ | ✓ | ✓ | ✗ | ✗ | 有 L2/L3 但 match 不回灌 | P1 |

### N. 非目标 / 暂缓（需签字）

| ID | 项 | 说明 | 决策 |
|----|-----|------|------|
| N1 | P2PConnector | 设计写明暂不做；与 block tree 解耦验收 | [ ] 确认排除 |
| N2 | Remote `StorageBackend`（Phase 5） | 未合入则标 Phase 外 | [ ] 确认范围 |
| N3 | `InsertInfo.is_resident` | 字段仍在但 allocator **未消费** | [ ] 删除字段 / [ ] 实现永不淘汰 |
| N4 | 旧 MemoryConnector API | 应已删除 | [ ] 确认无代码路径 / 无回归依赖 |

---

## 4. 与现有 UT 映射（抽样）

| 验收域 | 主要测试 |
|--------|----------|
| A / B / G | `BlockTreeCacheTest.*` |
| C / E / F | `BlockTreeEvictorTest`、`FullEvictionTest`、`FullSWA*`、`FullLinear*`、`BlockTreeCacheIntegrationTest` |
| H / J | `*ComponentGroupTest`、`KVCacheManagerTest`（DSV4 系列）、`Hybrid*Test`、`SingleType*` |
| I | `CPSlotMapperTest`、`HybridKVCacheAllocatorCPShardTest` |
| K | `BlockTreeTransferConverterTest`、`CopyEngine*`、`*BlockPoolTest`、`DiskMountGuardTest` |
| F7 / Broadcast | `BlockTreeCacheBroadcastTest` |

### 4.1 已知缺口（建议优先补测）

1. **D4** `ensureFreeBlocks` ↔ `evictForGroup` 专项（目前多为间接覆盖）
2. **J6** 双独立淘汰开关优先级端到端
3. **N3** resident 行为：实现或删除字段
4. **D7** `CacheEvictPolicy::NONE` 专项
5. DeviceSWA 相关测试中若仍有 `SharedBlockCache` TODO，完成迁移清理

---

## 5. 勾选记录（可选）

| 阶段 | Owner | 日期 | 结果 | 备注 |
|------|-------|------|------|------|
| P0 | | | ☐ Pass / ☐ Fail | |
| P1 | | | ☐ Pass / ☐ Fail | |
| P2 | | | ☐ Pass / ☐ Fail | |
| N 签字 | | | ☐ Done | |

---

## 6. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-07-21 | 初版：基于 BlockTreeCache 设计文档与现有 UT 梳理全量验收场景 |
