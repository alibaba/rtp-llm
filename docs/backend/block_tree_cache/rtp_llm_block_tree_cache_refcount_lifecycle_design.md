# BlockTreeCache Block 引用计数与生命周期重构设计

> 本文定义 `BlockTreeCache` / `BlockTreeEvictor` / `ComponentGroup` 中 block 分配、释放、引用计数、slot 状态更新的最终职责划分和接口设计。
> 目标是后续开发时将所有 block pool 生命周期操作统一收敛到 `ComponentGroup`，避免上层直接操作 `DeviceBlockPool` / `HostBlockPool` / `DiskBlockPool`。

---

## 1. 背景与问题

当前 block 生命周期管理分散在多个类中：

- `BlockTreeEvictor` 直接分配和释放 Host/Disk target block，并在 `setTargetSlot` 中手动 `incRef`。
- `BlockTreeCache::performLoadBack` 直接从 Host pool 分配临时 block，并直接释放。
- `BlockTreeCache::releaseMatchedBlocks` 直接访问 Full group 的 device pools，并依赖 `i % num_pools` 推导 pool index。
- `ComponentGroup` 目前只提供 device blocks 的 `referenceDeviceBlocks` / `releaseDeviceBlocks`，Host/Disk 没有统一接口。
- `GroupSlot` 状态更新、heap membership 更新、pool refcount 更新不在同一个职责边界内完成。

这些问题导致：

- slot 中 block 是否已经有 cache 引用不够清晰。
- match 临时引用和 cache 持有引用容易混淆。
- insert 遇到已有节点时，未消费的 input blocks 可能泄漏。
- eviction copy 失败 rollback 路径需要分别理解 target block、source block、heap 状态。

---

## 2. 设计原则

### 2.1 ComponentGroup 是唯一 block 生命周期入口

除 pool 初始化、CopyEngine 只读访问外，上层不得直接调用：

- `DeviceBlockPool::malloc/free/incRef/releaseRef`
- `HostBlockPool::malloc/free/incRef/releaseRef`
- `DiskBlockPool::malloc/free/incRef/releaseRef`

所有分配、释放、引用、解引用必须通过 `ComponentGroup`。

### 2.2 cache 引用的建立时机分两类

device pool 与推理运行时 `KVCacheAllocator` 共享，因此 device block 的生命周期同时存在“运行时请求引用”和“cache 持有引用”两方。cache 引用按 block 来源分两类建立：

1. **运行时新算 KV（insert 路径）**：block 由运行时自己的 allocator 分配并持有引用，cache 不 malloc。只有在 `insert` 真正把 block 写入新建节点时，cache 才对这些 block `incRef`，建立 cache 持有引用。这样即使发起请求结束、运行时释放自己那份引用，cache 那份仍在，block 不回 freelist。

2. **cache 自分配（load_back 目标 / demotion 目标 / temp host）**：这些 block 不属于任何运行时请求，语义为：

```text
malloc block -> incRef(block) -> 返回给调用者
```

从分配时刻即由 cache 持有，refcount 至少为 1，即使尚未写入 `GroupSlot` 也不会回 freelist。

> 关键点：heap 成员资格只决定“能否被选为淘汰候选”，cache 引用决定“block 存活”，两者正交。所有被缓存的 block（含不在任何 heap 的非叶子节点）都必须有一份 cache 引用。

### 2.3 eviction heap 只管理树上可见数据

新 malloc 的 block 不加入 tree，也不加入 eviction heap，因此不会被 eviction 选中。

只有 `insert` / copy complete / load_back complete 将 block 写入 `GroupSlot` 后，才根据 group 规则加入对应 tier heap。

### 2.4 match 引用是额外临时引用

一个 device block 的 refcount 可能同时来自三方：**运行时 active 请求引用**、**cache 持有引用（0 或 1）**、**match/path-lock 临时保护引用（≥0）**。slot 中被缓存且无请求在用的 block，cache 持有引用为 1。match 返回给上层使用时，必须额外 `incRef`（+1）。

evictability 判据是“仅 cache 持有”，即 refcount 恰为 cache 持有的那 1 份；只要还有运行时请求引用或 match 保护引用，block 就不可淘汰。eviction/match 的释放都通过 `releaseRef` 表达：最后一份归零时才回 freelist，因此三方以任意顺序释放都安全。

### 2.5 slot 更新和 refcount 维护分离但同属 ComponentGroup

`ComponentGroup` 同时提供：

- pool 级 block 生命周期接口；
- slot tier blocks 的读写/清理接口；
- heap membership 更新接口。

调用方可以清楚表达“先 copy，再 release source cache 引用（用保存的块 id 副本），再清 source slot，再设置 target slot”，但具体 pool/refcount 操作不外泄。

---

## 3. 核心不变量

1. `GroupSlot` 中出现的非空 block 必须已经有 cache 持有引用。
2. device/insert 路径的 cache 持有引用在 `insert` 写入新建节点时建立（对 `inserted_mask` 为 true 的节点 `incRef`）；cache 自分配路径（load_back/demotion/temp）的 cache 持有引用在 `allocateBlocks()`（malloc+incRef）时建立。
3. `ComponentGroup::releaseBlocks()` 释放 cache 持有引用；当 refcount 从 1 到 0 时，block 返回 freelist。
4. `ComponentGroup::referenceBlocks()` / `unreferenceBlocks()` 只用于 match 等临时保护引用。
5. eviction 只能从 heap 中选择 tree 上可见的 cached block。
6. 未写入 slot 的 target/temp block 不在 heap 中，不会被 eviction 选择。
7. cache 引用与 heap 成员资格正交：不在任何 heap 的非叶子节点同样持有 cache 引用。
8. `BlockTreeCache` 和 `BlockTreeEvictor` 不直接操作 pool refcount。

---

## 4. 最终接口设计

### 4.1 ComponentGroup block 生命周期接口

在 `ComponentGroup` 中新增统一接口：

```cpp
// 统一块集合结构：某个 component group 在某个 tier 上的一批 block。
// 布局自描述：外层按 cache_key(树节点)，内层下标 == device pool 下标，
// 解引用/释放无需 i % num_pools 取模。
struct GroupBlockSet {
    int  component_group_id{-1};
    Tier tier{Tier::DEVICE};
    // per_node[k]     对应第 k 个 cache_key(树节点)
    // per_node[k][p]  属于 device_pools_[p]；HOST/DISK 时内层 size == 1
    // per_node[k] 可直接赋给/来自 GroupSlot::device_blocks
    std::vector<std::vector<BlockIdxType>> per_node;
};

class ComponentGroup {
public:
    // ---- Structured block allocation / release (device multi-pool & match) ----
    // count = cache_key 数量；DEVICE 时为每个 pool 各分配 count 个块，组织进 per_node[k][p]。
    GroupBlockSet allocateBlocks(Tier tier, size_t count);
    void          releaseBlocks(const GroupBlockSet& set);

    // ---- Temporary external references, mainly for match result protection ----
    void referenceBlocks(const GroupBlockSet& set);
    void unreferenceBlocks(const GroupBlockSet& set);

    // ---- Scalar single-block helpers for single-pool tiers (HOST/DISK); DEVICE is multi-pool, use allocateBlocks ----
    BlockIdxType allocateSingleBlock(Tier tier);        // malloc + incRef
    void         releaseSingleBlock(Tier tier, BlockIdxType block) const;

    // ---- Slot helpers ----
    std::vector<BlockIdxType> getBlocks(const GroupSlot& slot, Tier tier) const;
    void setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks);
    void evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);

    // ---- Evictability ----
    bool isSlotEvictable(const GroupSlot& slot, Tier tier) const;
};
```

### 4.2 allocateBlocks 语义

`count` = **cache_key 数量**（即要分配的树节点数）。返回统一的 `GroupBlockSet`。

#### DEVICE

`DEVICE` 是多 pool 结构。为**每个 device pool 各分配 count 个块**，按 `per_node[k][p]` 组织（外层 cache_key，内层 pool 下标）。

```cpp
GroupBlockSet ComponentGroup::allocateBlocks(Tier::DEVICE, size_t count) {
    GroupBlockSet set{component_group_id, Tier::DEVICE};
    set.per_node.resize(count);
    for (size_t k = 0; k < count; ++k) {
        set.per_node[k].resize(device_pools_.size(), NULL_BLOCK_IDX);
        for (size_t p = 0; p < device_pools_.size(); ++p) {
            auto b = device_pools_[p]->malloc();
            if (!b) { releaseBlocks(set); return {}; }   // 失败回滚已分配
            device_pools_[p]->incRef(*b);
            set.per_node[k][p] = *b;
        }
    }
    return set;
}
```

失败返回空 `GroupBlockSet`，并保证已分配 block 被释放。约束：`per_node[k].size() == device_pools_.size()`，`per_node[k][p]` 严格对应 `device_pools_[p]`。

#### HOST / DISK

`HOST` / `DISK` 为单 pool，`per_node[k]` 内层 size==1。单块场景可直接用 `allocateSingleBlock(tier)`（malloc+incRef）。

### 4.3 releaseBlocks 语义

`releaseBlocks(set)` 释放 cache 持有引用，**内层下标即 pool 下标，无需取模**：

```cpp
void ComponentGroup::releaseBlocks(const GroupBlockSet& set) {
    for (const auto& node_blocks : set.per_node) {
        for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
            if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
                device_pools_[p]->releaseRef(node_blocks[p]);
            }
        }
    }
}
```

HOST/DISK 同理（单 pool）。统一用 `releaseRef`，不再使用 `free()` 作为上层释放入口；`free()` 只作为 pool 内部实现细节。

### 4.4 referenceBlocks / unreferenceBlocks 语义

接受同一 `GroupBlockSet`，用于 match 返回值保护：

- `referenceBlocks(set)`: 对 set 内每块额外 `incRef`。
- `unreferenceBlocks(set)`: 对 set 内每块 `releaseRef`，释放额外引用。

遍历方式与 `releaseBlocks` 相同（内层下标即 pool 下标）。这组接口不修改 `GroupSlot`，不修改 heap。

### 4.5 getBlocks / setBlocks / evictFromTier 语义

```cpp
getBlocks(slot, DEVICE) -> slot.device_blocks
getBlocks(slot, HOST)   -> {slot.host_block} or {}
getBlocks(slot, DISK)   -> {slot.disk_slot} or {}
```

`setBlocks` 只写 slot，不额外 incRef。原因是 block 已经由 `allocateBlocks` 或上层等价路径持有 cache 引用。

`evictFromTier`：

- 清空对应 tier 的 slot 字段；
- invalidate 对应 tier heap 并清理 `in_*_heap` 标记（已内含 clearHeapFlag）；
- 不释放 refcount。调用方须先 `getBlocks` 保存 blocks 副本，再 `releaseBlocks`。

顺序规范：**先 `releaseBlocks`（用保存的块 id 副本，不读 slot）再 `evictFromTier`（清 slot）**，避免"先清空后释放"的语义误解：

```cpp
auto source_blocks = group->getBlocks(slot, source_tier);
group->releaseBlocks(GroupBlockSet{gid, source_tier, {source_blocks}});
group->evictFromTier(node, slot, source_tier);
group->setBlocks(slot, target_tier, target_blocks);
```

---

## 5. BlockTree 插入接口调整

### 5.1 当前问题

当前 `BlockTree::insertNode` 行为：

- 已存在节点：复用，不覆盖 `group_slots`。
- 新建节点：设置 `slots[i]`。
- 返回 leaf。

因此 `BlockTreeCache::insert` 无法知道哪些 input slots 被实际消费。如果 input blocks 已经通过 `allocateBlocks` 拿到 refcount=1，而节点已存在导致 slot 未被消费，就必须释放这些未消费 blocks，否则泄漏。

### 5.2 新接口

将 `BlockTree::insertNode` 改为返回插入结果：

```cpp
struct BlockTreeInsertedNode {
    TreeNode* node{nullptr};
    size_t input_index{0};
};

struct BlockTreeInsertResult {
    TreeNode* leaf{nullptr};
    std::vector<BlockTreeInsertedNode> inserted_nodes;
    std::vector<bool> inserted_mask; // size == cache_keys.size()
};

BlockTreeInsertResult insertNode(TreeNode* parent,
                                 const CacheKeysType& cache_keys,
                                 const std::vector<std::vector<GroupSlot>>& slots);
```

`inserted_mask[i] == true` 表示 `slots[i]` 被写入新建节点。否则该 input slot 未被消费。

### 5.3 BlockTreeCache::insert 流程

device block 由运行时自己的 allocator 分配并持有引用（cache 不 malloc）。cache 持有引用在 `insert` 时对**新建节点**建立。

```cpp
void BlockTreeCache::insert(TreeNode* parent,
                            const CacheKeysType& keys,
                            const std::vector<std::vector<GroupSlot>>& slots) {
    lock(mutex_);

    auto result = tree_->insertNode(parent, keys, slots);

    // 1. 只对实际新建的节点：incRef 建立 cache 持有引用，并尝试加入 heap。
    for (const auto& inserted : result.inserted_nodes) {
        TreeNode* node = inserted.node;
        for (auto& group : component_groups_) {
            auto  gid    = group->component_group_id;
            auto& slot   = node->group_slots[gid];
            auto  blocks = group->getBlocks(slot, Tier::DEVICE);
            if (!blocks.empty()) {
                // incRef 建立 cache 持有引用（由淘汰时 releaseBlocks 平衡）
                group->referenceBlocks(GroupBlockSet{gid, Tier::DEVICE, {blocks}});
            }
            group->tryAddToDeviceHeap(node);   // 仅叶子会真正入 heap
        }
    }

    // 2. 对已有 path 做 heat 更新。
    updateOnInsertOverlap(...);

    checkWatermark();
}
```

要点：
- cache 只对新建节点 `incRef`。命中的既有节点的 input blocks **无需任何处理**——cache 从未对它们 `incRef`，运行时会按普通请求块正常释放。**因此原“释放未消费 input blocks”逻辑整段删除。**
- 分配后放弃 insert 也无需 cache 清理（同理）。
- cache 引用对**所有新建节点**建立（含非叶子中间节点），与是否入 heap 无关；heap 只影响能否被选为淘汰候选。

---

## 6. Device block 分配接口

device block 有两个来源，分别对应两条接口：

### 6.1 insert 路径（运行时新算 KV）

运行时通过自己的 allocator 分配 device block、算 KV，然后 `insert`。cache **不分配**，只在 `insert` 时对新建节点 `incRef`（见 §5.3）。上层无需调用 cache 的分配接口，也无需在放弃 insert 时通知 cache。

### 6.2 cache 自分配路径（load_back 目标等）

无 `BlockTreeCache` 层包装，直接调用目标 group 的结构化接口，`count` = **cache_key 数量**：

```cpp
auto set = component_groups_[gid]->allocateBlocks(Tier::DEVICE, count);  // malloc + incRef
component_groups_[set.component_group_id]->releaseBlocks(set);           // 未使用时回滚
```

约定：DEVICE 为每个 pool 各分配 count 个块，组织进 `per_node[k][p]`；`per_node[k]` 可直接写入第 k 个节点的 `GroupSlot::device_blocks`。若最终未使用，调用 `releaseBlocks(set)` 释放。

---

## 7. match / releaseMatchedBlocks 设计

### 7.1 Match result 使用统一 GroupBlockSet

match 结果直接用 §4.1 的 `GroupBlockSet`，每个 group 一份，彻底替代原 `MatchedBlockRef` + 平铺 `block_indices` 的取模推导：

```cpp
struct BlockTreeMatchResult {
    ...
    size_t                     matched_blocks{0};    // 连续命中块数
    BlockIndicesType           block_indices;        // FULL group 命中 device 块的扁平有序列表
    std::vector<GroupBlockSet> matched_block_sets;   // 每个 group 一份，结构化
};
```

`block_indices` 是面向 runtime 的**输出契约**：按前缀顺序收集 FULL group 命中节点的 device 块（含 load_back 目标块），供上层直接填 block table 复用；它**不作为释放依据**。释放依据是 `matched_block_sets`（结构化、含所有 group）。若需替代 `KVCacheMemoryConnector::asyncMatch`，其上层消费“命中块数 + 有序 device 块”与 `block_indices` 直接对齐。

### 7.2 match 引用流程

按 group 收集每个匹配节点的 device 块，组织进 `GroupBlockSet.per_node`（外层节点、内层 pool 下标），并 `referenceBlocks`：

```cpp
for each group:
    GroupBlockSet set{group_id, Tier::DEVICE};
    for each matched node:
        auto blocks = group->getBlocks(slot, Tier::DEVICE); // 长度 == device_pools_.size()
        set.per_node.push_back(blocks);
    group->referenceBlocks(set);            // 每块 incRef（match 保护）
    result.matched_block_sets.push_back(set);
```

### 7.3 releaseMatchedBlocks 新接口

```cpp
void releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets);
```

实现（reference 时记下什么，release 时原样传回，group_id / tier / pool 下标全部对齐）：

```cpp
for (const auto& set : sets) {
    component_groups_[set.component_group_id]->unreferenceBlocks(set);
}
```

这样彻底消除旧 `releaseMatchedBlocks(block_indices)` 只处理 FULL group、且用 `i % num_pools` 推导 pool 的泄漏 bug。旧扁平 `releaseMatchedBlocks(block_indices)` 已删除，仅保留结构化接口。

---

## 8. reclaimBlocks 直接淘汰流程

`reclaimBlocks` 语义：直接释放指定 tier，不触发降级 copy。

流程：

```cpp
auto move = group->driveEviction(1, source_tier);
move.target_tier = Tier::NONE;
submitEvictionLocked(move);
```

complete 阶段：

```cpp
auto& group = component_groups_[gid];
auto& slot = node->group_slots[gid];

auto source_blocks = group->getBlocks(slot, source_tier);
group->releaseBlocks(GroupBlockSet{gid, source_tier, {source_blocks}});
group->evictFromTier(node, slot, source_tier);

finalizeEviction(tree, node);
```

被选中淘汰的块一定是“仅 cache 持有”（refcount==1，见 §2.4 evictability 判据），`releaseBlocks` 后直接回 freelist；被 match 或运行时引用的块不会进入淘汰候选，因此不存在“降到 1 等待 release”的情况。

---

## 9. checkWatermark 降级淘汰流程

### 9.1 EvictionMove 数据结构

将 `blocks_to_release` / `target_block` 改为明确的 source/target blocks：

```cpp
struct EvictionMove {
    TreeNode* node{nullptr};
    int component_group_id{-1};
    Tier source_tier{Tier::NONE};
    Tier target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;
};
```

### 9.2 prepare 阶段

在 `BlockTreeEvictor::prepareMove` 中：

```cpp
auto& group = component_groups_[gid];
auto& slot = node->group_slots[gid];

move.source_blocks = group->getBlocks(slot, move.source_tier);
reserveSourceHeap(move);

if (move.target_tier != Tier::NONE) {
    // 降级目标为单 block（HOST/DISK 单 pool 打包）
    BlockIdxType tb = group->allocateSingleBlock(move.target_tier);  // malloc + incRef
    if (isNullBlockIdx(tb)) {
        restoreSourceHeap(move);
        return false;
    }
    move.target_blocks = {tb};
}
```

target block 已经 refcount=1（cache 自分配路径），但没有写入 slot，不在 heap 中。

### 9.3 copy 阶段

CopyEngine descriptor 使用：

- `DEVICE -> HOST`: `source_blocks` + `target_blocks[0]`
- `HOST -> DISK`: `source_blocks[0]` + `target_blocks[0]`
- `DISK -> HOST`: `source_blocks[0]` + `target_blocks[0]`

### 9.4 complete 成功

```cpp
auto source_blocks = group->getBlocks(slot, move.source_tier);
// 先 release source cache-hold（用保存的块 id）再清 slot
group->releaseBlocks(GroupBlockSet{move.component_group_id, move.source_tier, {source_blocks}});
group->evictFromTier(node, slot, move.source_tier);

if (move.target_tier != Tier::NONE) {
    group->setBlocks(slot, move.target_tier, move.target_blocks);
    group->tryAddToHeap(node, move.target_tier);   // 虚派发到子类 Leaf 判定
}
```

### 9.5 complete 失败 / rollback

```cpp
group->releaseSingleBlock(move.target_tier, move.target_blocks[0]);  // 单块目标
restoreSourceHeap(move);
```

因为 target blocks 已经有 refcount=1，失败时通过 `releaseSingleBlock` 释放即可。

---

## 10. load_back 流程

### 10.1 LoadBackItem

```cpp
struct LoadBackItem {
    TreeNode* node{nullptr};
    int group_id{-1};
    Tier source_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_device_blocks;
};
```

### 10.2 prepareMatchedLoadBack

在 match 持锁阶段：

```cpp
if (!slot.has_device_value() && slot.has_host_or_disk_value()) {
    auto source_blocks = group->getBlocks(slot, source_tier);
    group->referenceBlocks(GroupBlockSet{group_id, source_tier, {source_blocks}});

    auto set = group->allocateBlocks(Tier::DEVICE, 1);   // 单节点，count=1
    if (set.per_node.empty()) {
        group->unreferenceBlocks(GroupBlockSet{group_id, source_tier, {source_blocks}});
        continue;
    }

    // target device 有两份引用：
    // 1. allocateBlocks 建立 cache 持有引用，成功后写入 slot；
    // 2. referenceBlocks 建立 match 返回保护引用，由 releaseMatchedBlocks 释放。
    group->referenceBlocks(set);
    result.matched_block_sets.push_back(set);

    lb_items.push_back({node, group_id, source_tier, source_blocks, set.per_node[0]});
}
```

不要提前写 `slot.device_blocks`。

### 10.3 performLoadBack copy

HOST -> DEVICE：

```cpp
copy item.source_blocks[0] -> item.target_device_blocks
```

DISK -> DEVICE：

```cpp
auto temp_host = group->allocateSingleBlock(Tier::HOST);
copy item.source_blocks[0] -> temp_host;
copy temp_host -> item.target_device_blocks;
group->releaseSingleBlock(Tier::HOST, temp_host);
```

copy 结束后无论成功失败，都释放 source tier 的临时保护引用：

```cpp
group->unreferenceBlocks(GroupBlockSet{group_id, item.source_tier, {item.source_blocks}});
```

### 10.4 load_back 成功提交

重新持有 cache mutex：

```cpp
group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
// 先 release source cache-hold（用保存的 source block ids）再 evictFromTier 清 source slot
group->releaseBlocks(GroupBlockSet{group_id, source_tier, {item.source_blocks}});
group->evictFromTier(node, slot, source_tier);
group->tryAddToDeviceHeap(node);

// load_back 目标块的 cache 持有引用已由 allocateBlocks 建立；
// match 保护引用已在 prepareMatchedLoadBack 阶段建立并写入 result.matched_block_sets。
```

策略更新：load_back 成功后**释放 source(host/disk) 层**——清空 source slot 并 `releaseBlocks`，使每个逻辑块同一时刻只属一个 tier、cache-hold 恰一份，避免后续 device→host 降级覆盖旧 host_block 导致的泄漏。

注意：match 的 load_back 是**异步**执行的，调用方必须在 match 返回后 await `async_context` 完成，才能使用回填到 device 的数据。

### 10.5 load_back 失败

```cpp
group->releaseBlocks(GroupBlockSet{group_id, Tier::DEVICE, {item.target_device_blocks}});
```

如果分配了 temp host，也必须释放：

```cpp
group->releaseSingleBlock(Tier::HOST, temp_host);
```

如果线程池提交失败，必须释放所有已准备 item：

```cpp
group->unreferenceBlocks(GroupBlockSet{group_id, item.source_tier, {item.source_blocks}});
group->releaseBlocks(GroupBlockSet{group_id, Tier::DEVICE, {item.target_device_blocks}});
```

---

## 11. BlockTreeEvictor 修改点

删除或改为不再直接访问 pool 的函数：

- `allocateBlock`
- `releaseBlocks`
- `releaseTargetBlock`
- `hostPoolForGroup`
- `diskPoolForGroup`
- `setTargetSlot` 中直接 `incRef`

保留 Evictor 负责：

- victim 选择；
- cascade group 选择；
- plan 构建；
- CopyEngine descriptor 构建；
- complete / rollback 的流程编排。

所有 block 生命周期操作调用 `ComponentGroup`。

---

## 12. BlockTreeCache 修改点

### 12.1 insert

- 使用新的 `BlockTreeInsertResult`。
- **对新建节点的 device blocks `incRef`（建立 cache 持有引用），由淘汰时 `releaseBlocks` 平衡。**
- 命中的既有节点的 input blocks 无需处理（cache 从未 incRef，运行时正常释放）；**删除原“释放未消费 input blocks”逻辑。**
- 只对 leaf 尝试加入 device heap（叶子才真正入 FULL heap）；非叶节点在其子 leaf 被淘汰后经 parent 提升入堆，堆始终跟踪从叶到根的可淘汰前沿。

### 12.2 match

- 将 `referenceMatchedDeviceBlocks` 改为填充 `matched_block_sets`（`std::vector<GroupBlockSet>`）。
- 引用通过 `ComponentGroup::referenceBlocks(const GroupBlockSet&)`。

### 12.3 releaseMatchedBlocks

- 仅保留结构化接口 `releaseMatchedBlocks(const std::vector<GroupBlockSet>&)`。
- 旧 flat `releaseMatchedBlocks(block_indices)` 已删除。

### 12.4 performLoadBack

- 分配 target device 通过 group。
- 不提前写 slot。
- `prepareMatchedLoadBack` 阶段建立 source 临时保护引用和 target match 保护引用。
- 成功后 `setBlocks(DEVICE) + releaseBlocks(source) + evictFromTier(source) + tryAddToDeviceHeap`：释放 source cache-hold 并清空 source slot（先 release 再 evict），不在异步线程里补充 match 引用。
- 失败后 `releaseBlocks`。
- temp host 通过 group 用 `allocateSingleBlock/releaseSingleBlock` 分配和释放。
- load_back 异步执行，调用方须 await match 返回的 `async_context` 后才能使用 device 数据。

### 12.5 submitEvictionLocked / performEvictionCopy

- plan 中使用 `source_blocks/target_blocks`。
- complete / rollback 只调用 Evictor，Evictor 内部再调用 group。

---

## 13. ComponentGroup 修改点

### 13.1 原有接口替换

保留但内部改造：

- `referenceDeviceBlocks` / `releaseDeviceBlocks` -> 统一为 `referenceBlocks(GroupBlockSet)` / `releaseBlocks(GroupBlockSet)`。
- Host/Disk 单块场景用 `allocateSingleBlock/releaseSingleBlock`（DEVICE 多 pool 不支持 scalar）。

`GroupBlockSet` 的内层下标即 pool 下标，替代 `i % num_pools` 取模。

### 13.2 heap helper

新增通用 helper：

```cpp
void tryAddToHeap(TreeNode* node, Tier tier);
void invalidateHeap(TreeNode* node, Tier tier);
void clearHeapFlag(GroupSlot& slot, Tier tier);
```

`tryAddToDeviceHeap` 仍由子类实现，Host/Disk 复用基类实现。

---

## 14. 开发迁移步骤

1. 在 `ComponentGroup` 增加统一 block 接口，先保留旧接口兼容。
2. 修改 `EvictionMove` 为 `source_blocks/target_blocks`。
3. 改造 `BlockTreeEvictor`，移除直接 pool 操作。
4. 修改 `BlockTree::insertNode` 返回 `BlockTreeInsertResult`。
5. 改造 `BlockTreeCache::insert`：对新建节点 incRef 建立 cache 持有引用，删除未消费 blocks 释放逻辑。
6. 增加 `GroupBlockSet`，改造 allocate/match/release/reference 全部走统一结构。
7. 改造 load_back，禁止提前写 `slot.device_blocks`。
8. 收紧 `ComponentGroup` pool getter 使用范围，仅 CopyEngine / factory 可读访问。
9. 补齐测试。

---

## 15. 必要测试

### 15.1 allocate/release

- `allocateBlocks(DEVICE)` 后 refcount 为 1。
- `releaseBlocks(DEVICE)` 后 refcount 为 0 且回 freelist。
- Host/Disk 同样验证 refcount 和 freelist。

### 15.2 insert

- 新建节点：insert 时对 device blocks incRef，建立 cache 持有引用（运行时释放后 block 仍存活）。
- 插入已有路径：命中既有节点时 cache 不 incRef 新 input blocks，运行时正常释放，无泄漏。
- 只有 leaf 被加入 device heap；非叶子新节点也被 incRef 保护。

### 15.3 match

- match 后 refcount 从 1 到 2。
- reclaim matched block 后 refcount 从 2 到 1，不回 freelist。
- releaseMatchedBlocks 后 refcount 从 1 到 0，回 freelist。

### 15.4 reclaimBlocks

- direct reclaim 不分配 target。
- slot 被清理，heap 被 invalidate。
- source blocks 调用 `releaseBlocks`。

### 15.5 watermark demotion

- DEVICE -> HOST 成功：source release，target 写入 host slot，host heap 更新。
- copy 失败：target release，source heap restore，slot 不变。
- HOST -> DISK 成功/失败同理。

### 15.6 load_back

- copy 前 target device 不写入 slot。
- 成功后写 device slot、release source cache-hold 并清空 source slot、加入 device heap。
- 失败后 release target device。
- Disk -> Device temp host 必须 release。

---

## 16. 最终职责边界

| 类 | 职责 |
|---|---|
| `BlockTreeCache` | 树锁、公开 insert/match/reclaim 接口、任务调度、watermark 触发 |
| `BlockTreeEvictor` | victim 选择、cascade plan、copy 执行、complete/rollback 编排 |
| `ComponentGroup` | pool 分配/释放/refcount、slot tier 状态、heap membership |
| `BlockTree` | 纯树结构，返回新建节点信息，不理解 block refcount |
| `CopyEngine` | 只读 pool/layout，执行跨 tier copy，不管理 block 生命周期 |

---

## 17. 总结

本设计不引入复杂 ownership 对象，而是统一采用简单规则：

```text
cache 自分配(load_back/demotion/temp) = malloc + incRef
insert(运行时块)                       = insert 时对新建节点 incRef
releaseBlocks                          = releaseRef
match                                  = referenceBlocks / unreferenceBlocks
slot update                            = setBlocks / evictFromTier
块传递                                  = 统一 GroupBlockSet（内层下标即 pool 下标）
```

这样可以保证：

- cache 自分配的 block 即使尚未进入树，也不会回 freelist。
- device block 与运行时共享池：cache 引用在 insert 时建立，请求结束后 block 仍被 cache 持有而存活。
- cache 引用与 heap 成员资格正交：非叶子节点同样被引用保护。
- match / eviction 的并发安全通过 refcount 自然表达；GroupBlockSet 消除取模推导的泄漏 bug。
- 上层不再直接操作 pool，职责边界清晰。
