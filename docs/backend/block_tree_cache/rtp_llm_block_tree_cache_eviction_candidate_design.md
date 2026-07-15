# BlockTreeCache 淘汰候选集收敛设计

> 状态：设计方案
>
> 本文定义 `rtp_llm/cpp/cache/block_tree_cache/` 中淘汰候选集的统一设计。目标是让 insert、match、evict、load_back 共用一套候选管理：候选集精确、无冗余条目；LRU/LFU/FIFO 排序只由真实访问驱动，不被引用变化、回滚、拓扑提升等控制流扰动；异步 tier 迁移期间候选资格与块所有权明确、无竞争。

---

## 1. 设计目标与结论

1. `BlockTreeEvictor` 私有持有所有 `EvictionHeap`，是唯一执行 heap `upsert/erase/takeBest` 的类。
2. `BlockTreeCache` 对 heap 完全透明：它只上报插入、真实访问、引用变化、tier 迁移开始/完成、节点删除、拓扑变化等缓存语义事件。
3. `ComponentGroup` 不持有 heap；它只提供 FULL/SWA/LINEAR 的统一可淘汰性判断，以及 block/slot 生命周期操作。
4. **每个 `GroupSlot` 只存一份 `CandidateMeta`**，对应该 slot 当前对外服务的 tier。基于“淘汰后源块清理、稳态单 tier 服务”的不变量，不需要按 tier 存多份。
5. 候选状态由 slot 数据、heap 成员关系、引用计数和 `transfer_state` 推导，不维护额外的候选状态枚举，也不维护 `in_*_heap` 标志。
6. `EvictionHeap` 采用**精确更新**：任一节点在堆中至多一个条目，`upsert/erase/takeBest` 均为 O(log N)，物理大小恒等于当前 ready 候选数。
7. 异步 tier 迁移用 `GroupSlot::transfer_state` 表达真实的数据传输状态；迁移期间源块由 cache hold 保活、目标块由 staged-hold 保活，二者所有权明确。
8. match 采用**惰性引用判断**：match 只增加块引用、不主动操作 heap；淘汰时 `takeBest` 遇到被引用（`refcount > 1`）的堆顶节点即摘除并继续取下一个；引用释放后经 `refreshCandidate` 复位候选资格。
9. `EvictionHeap` 的选定实现为 **`std::set + node 索引`**；候选堆只覆盖 DEVICE/HOST/DISK 三层，`REMOTE` tier 不参与候选堆、不分配 `candidate_meta`（远端由 StorageBackend 自管 GC/TTL）。

---

## 2. 职责划分

| 类 | 负责 | 不负责 |
|---|---|---|
| `BlockTreeCache` | 树插入/匹配、发起 load_back、上报语义事件 | heap、候选排序、候选资格 |
| `ComponentGroup` | block 生命周期、slot 读写、group 特有可淘汰性判断 | heap 操作、候选恢复 |
| `BlockTreeEvictor` | heap 所有权与排序、候选刷新、迁移开始/完成/回滚、节点删除与父节点提升、排序元数据读写 | 直接管理 block pool 引用 |
| `EvictionHeap` | 一个 `(group, tier)` 的精确有序容器 | slot/拓扑/refcount 判断 |

`BlockTreeEvictor` 在 heap 收敛后同时承担 `GroupSlot` 排序元数据的读写；`ComponentGroup` 退化为“池 + 资格谓词”的载体。`BlockTreeCache` 只调用一类语义接口（见 §8.2），不再出现 `device_heap/host_heap/disk_heap`、`tryAddToHeap`、`invalidateHeap` 或 `in_*_heap`。候选数量通过 `evictor_.candidateStats()` 聚合。

---

## 3. 数据所有权与状态

### 3.1 单份 CandidateMeta 放在 GroupSlot

关键不变量：**demotion 或 load_back 完成后，源 tier 的块被立即清理**（`unreferenceBlocks(source)` 后 `evictFromTier(source)`），因此稳态下一个 `GroupSlot` 只在唯一 tier 持有数据并对外服务。唯一的“双 tier 并存”是 copy 进行中的瞬态，而那期间 `transfer_state != IDLE` 已将节点排除出所有 heap（见 §4.3），排序元数据不会被读取。

因此排序元数据只需一份，随数据所在 tier 移动而“跟随”，无需按 tier 存多份：

```cpp
struct CandidateMeta {
    uint64_t last_access_seq{0};  // LRU：最近一次真实 match 的逻辑时钟
    uint64_t admission_seq{0};    // FIFO：进入“当前服务 tier”的逻辑时钟
    uint64_t hit_count{0};        // LFU：累计真实命中次数
};

enum class SlotTransferState : uint8_t {
    IDLE,
    DEMOTING,     // Device -> Host，或 Host -> Disk
    LOADING_BACK  // Host/Disk -> Device
};

struct GroupSlot {
    std::vector<BlockIdxType> device_blocks;              // L1: 每个独立 Device 池一块
    BlockIdxType              host_block{NULL_BLOCK_IDX}; // L2
    BlockIdxType              disk_slot{NULL_BLOCK_IDX};  // L3

    SlotTransferState transfer_state{SlotTransferState::IDLE};
    CandidateMeta     candidate_meta;  // 单份，对应当前对外服务 tier

    bool has_device_value() const;
    bool has_host_value() const;
    bool has_disk_value() const;
    bool has_any_value() const;
    bool is_empty() const;
};
```

`candidate_meta` 是候选排序的唯一事实来源。heap 条目只保存它在入堆/刷新时刻的排序键快照；键一旦因真实访问变化，必须通过 `upsert` 同步到堆（见 §5.2）。

不同 tier 可用不同策略（如 host 用 LRU、disk 用 FIFO）。单份 meta 同时保存 LRU/FIFO/LFU 所需字段，每个 tier 的比较器只取用它需要的字段，互不干扰。

### 3.2 所有权图

```text
BlockTree owns TreeNode
  TreeNode owns vector<GroupSlot>          // 按 component_group_id 索引
    GroupSlot owns candidate_meta          // 单份，跟随服务 tier
    GroupSlot owns transfer_state

BlockTreeEvictor owns heaps_[group_id][tier]
  EvictionHeap owns 有序索引 + node -> 位置索引
  heap 条目只持排序键快照，不拥有 CandidateMeta

EvictionMove / LoadBackItem 描述一次 copy：
  node、group、source/target tier、source/target block 快照
  不拥有 block 引用，不保存候选状态字段
```

节点删除前，evictor 必须从其所在的所有 heap 精确 `erase`；随后 `GroupSlot` 与元数据随树节点销毁，无需全局 node→metadata 映射。

### 3.3 候选状态由推导得出

不保存独立的候选状态枚举。候选是否 ready 可完全推导：

| 条件 | 含义 |
|---|---|
| `heap.contains(node)` 且 `transfer_state == IDLE` 且 `refcount == 1` | ready 候选 |
| slot 无当前 tier 数据 | 不可候选 |
| slot 有数据但 heap 不含 node | 被拓扑、refcount、tier enable 或引用保护排除 |
| `transfer_state != IDLE` | 正在真实迁移，必须不在 heap |

> 由于采用惰性引用判断（§5.4），被引用的节点可能短暂停留在 heap 中；`takeBest` 取出后会检查 `refcount`，被引用的堆顶节点会被立即摘除，因此“堆成员”不是“可淘汰”的充要条件，最终 evictability 由 `takeBest` 阶段的 `isSlotEvictable` 判断确定。

### 3.4 为什么 transfer_state 不能用 refcount 替代

`refcount > 1` 可能表示运行时请求或 match 临时保护，它不能回答“是否已有异步迁移在执行”。若仅靠 refcount，第二个 match 可能对同一 source 重复发起 load_back。因此：

- refcount 用于内存安全与 evictability：`isSlotEvictable() == false` 时该节点不作为 victim。
- `transfer_state` 用于禁止同一 slot 重复发起异步迁移，并在迁移期间将其排除出候选。
- 稳态单 tier 服务 + 迁移瞬态出堆，使**单个 `transfer_state` 枚举即可覆盖该 slot 的全部迁移语义**。
- 同步 `reclaimBlocks()` 全程在 cache mutex 下完成，不设 `transfer_state`。

---

## 4. EvictionHeap：精确更新

### 4.1 数据结构

每个 heap 对应固定 `(group_id, tier)`，key 只需 `TreeNode*`：

```cpp
struct EvictionEntry {
    TreeNode*    node{nullptr};
    uint64_t     primary_key{0};
    uint64_t     secondary_key{0};
    CacheKeyType cache_key{0};   // 确定性 tie-breaker
};

class EvictionHeap {
public:
    void                         upsert(TreeNode* node, const CandidateMeta& meta);
    void                         erase(TreeNode* node);
    std::optional<EvictionEntry> takeBest();
    bool                         contains(TreeNode* node) const;
    size_t                       size() const;

private:
    std::set<EvictionEntry, EntryLess>                                         ordered_;
    std::unordered_map<TreeNode*, std::set<EvictionEntry, EntryLess>::iterator> index_;
};
```

**选定实现**：`std::set<EvictionEntry, EntryLess> ordered_` 保存有序条目，`unordered_map<TreeNode*, ordered_::iterator> index_` 保存 `node -> 迭代器` 索引。`std::set` 的迭代器在其它元素插入/删除后保持有效，因此索引可直接存迭代器。

- `upsert`：若 `index_` 已含该 node，先按旧迭代器 `ordered_.erase`，再插入新排序键并更新索引；否则直接插入。必须检查 `std::set::insert` 的 `inserted` 结果，只有成功插入有序容器后才写 node 索引。O(log N)。
- `erase`：按 node 查 `index_` 得迭代器，精确从 `ordered_` 与 `index_` 删除。O(log N)。
- `takeBest`：取 `ordered_.begin()`，从两个容器删除。O(log N)，无失效条目扫描。

关键性质：
- 任一 node 在堆中至多一个条目，`ordered_.size() == index_.size()` 恒成立。
- 物理大小恒等于当前 ready 候选数，不随热节点访问次数增长；无失效条目、无 `takeBest` 空转。

**为何不用惰性删除优先队列**：`std::priority_queue` 不支持改键与按 node 删除，只能用惰性删除（push 新副本 + 标记旧失效），会让热节点高频 `upsert` 堆积 O(N) 失效条目、`takeBest` 反复 pop 空条目，正是本设计要消除的问题。

**后续优化项**：若实测候选集很大、淘汰极频繁，使红黑树逐结点分配与 cache 不连续成为瓶颈，可替换为数组存储的索引二叉堆（连续数组 + `node->下标` 映射，O(log N) update-key）；`upsert/erase/takeBest/contains` 接口不变，可无缝切换。

### 4.2 排序规则与逻辑时钟

逻辑时钟 `access_seq_` / `admission_seq_` 由 `BlockTreeEvictor` 在 `BlockTreeCache::mutex_` 下单调递增，不使用 `steady_clock`（避免同一时刻并列与重推重置时间戳）：

| 策略 | primary key | secondary key | 何时变化 |
|---|---|---|---|
| LRU | `last_access_seq` 升序 | `admission_seq` 升序 | 仅真实 match |
| LFU | `hit_count` 升序 | `last_access_seq` 升序 | 仅真实 match |
| FIFO | `admission_seq` 升序 | 0 | 进入“当前服务 tier”时 |

最后以 `cache_key` 与 node 地址做确定性 tie-breaker；指针使用 `std::less<TreeNode*>` 比较，避免对无关对象直接使用内建 `<`（node 地址仅保证**本地进程内**确定性）。refcount 变化、候选临时退出、回滚、父节点提升都不写排序字段。

### 4.3 唯一资格入口：refreshCandidate

所有 heap 成员关系只经由 `refreshCandidate` 更新：

```cpp
void BlockTreeEvictor::refreshCandidate(ComponentGroup& group,
                                        TreeNode*       node,
                                        Tier            tier) {
    auto& slot = node->group_slots[group.component_group_id];
    auto& heap = heapFor(group.component_group_id, tier);

    if (slot.transfer_state != SlotTransferState::IDLE ||
        !group.isSlotEvictable(*node, tier)) {
        heap.erase(node);
        return;
    }
    heap.upsert(node, slot.candidate_meta);
}
```

`isSlotEvictable` 是统一资格入口：基类要求 slot 持有该 tier 数据，且对应块 refcount 恰为 1（仅 cache hold）；FULL override 先要求该节点为 tier 叶子（无子节点在同 tier 持数据），再调用基类完成数据与引用检查。SWA/LINEAR 直接复用基类实现。

---

## 5. match、访问与惰性引用判断

### 5.1 访问排序与引用保护分离

match 有两个独立效果：

1. **真实访问**：所有命中的 node 都应更新排序元数据。即使 FULL 的中间节点当前不在 heap，也要记录真实访问历史，未来成为叶子时 LRU 才正确。
2. **引用保护**：命中范围内的块通过 `referenceBlocks` 增加引用计数，使 `refcount > 1`，从而在被 `takeBest` 选中时判定为不可淘汰。

因此 `onMatched(path)` 处理第一件事；`referenceBlocks/unreferenceBlocks` 处理第二件事。二者不合并为“对所有 node 统一处理引用”的循环，也不需要 match 阶段主动操作 heap 成员关系。

### 5.2 onMatched 必须刷新在堆节点

真实访问既要写 slot 元数据，也要让**已在 heap 的命中节点**同步刷新排序键，否则堆顺序会与元数据不一致（尤其 SWA：匹配路径可能长于引用窗口，窗口外的命中节点仍在堆中）：

```cpp
void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t seq = ++access_seq_;
    for (TreeNode* node : path) {
        for (auto& group : component_groups_) {
            auto& slot = node->group_slots[group->component_group_id];
            const Tier tier = group->getTopTier(slot);  // 当前服务 tier
            if (tier == Tier::NONE)
                continue;
            slot.candidate_meta.last_access_seq = seq;
            ++slot.candidate_meta.hit_count;
            auto& heap = heapFor(group->component_group_id, tier);
            if (heap.contains(node)) {
                heap.upsert(node, slot.candidate_meta);
            }
        }
    }
}
```

### 5.3 元数据始终对应当前服务 tier

由于单 tier 服务不变量，`onMatched` 只需按 `getTopTier(slot)` 定位当前服务 tier，更新并刷新对应 heap。数据下沉到 host/disk 后，同一份 meta 继续在该 tier 生效，host/disk 层的 LRU/FIFO 因此也能反映真实访问历史。

### 5.4 惰性引用判断：takeBest 跳过摘除 + release 复位

match 期间不主动 suspend 候选、也不为整条 path 做 heap 操作；被引用的节点是否可淘汰完全由 `takeBest` 阶段的 `refcount` 检查裁决：

- **match 时**：只 `referenceBlocks(set)` 增加引用。被引用的块 `refcount > 1`，其所在节点即使仍在 heap 中也不会被误淘汰。命中节点保持在原有 heap 位置，无需 erase。
- **淘汰时**：`chooseVictim` 反复 `takeBest()` 取堆顶；对取出的节点调用 `isSlotEvictable`（`refcount == 1`）。若不可淘汰（仍被引用），该节点已随 `takeBest` 从 heap 移除，直接**丢弃并继续取下一个**，不重新入堆、不改排序键。
- **release 时**：`unreferenceBlocks(set)` 后，对本次释放涉及的节点逐个 `refreshCandidate`。refcount 降回 cache hold（==1）且拓扑/tier 合格的节点被重新 `upsert` 回 heap，排序键沿用其原 `candidate_meta`，不因引用与释放而变化。

该方案消除了 `collectReferenceCandidateNodes`、`suspendReferencedCandidates`、`resumeReferencedCandidates` 等冗余逻辑：

- match/release 是最热路径，不再对引用范围内的每个候选做 O(R·log N) 的 heap erase/insert。
- 刚被 match 的节点位于 LRU 的 MRU 端，几乎不会在被引用期间被 `takeBest` 选中，因此“跳过摘除”极少真正触发。
- 排序正确性由精确更新 heap（§4.1）与逻辑时钟（§4.2）保障，不依赖 match 阶段的主动出堆/入堆。

### 5.5 并发 match 的正确性

两个请求同时命中同一 FULL leaf（cache hold + 两次 match 引用，`refcount == 3`）：

1. 两个请求各自 `referenceBlocks`，leaf 仍留在 heap（惰性，不主动 erase）。
2. 若淘汰在此期间发生：`takeBest` 取到该 leaf，`isSlotEvictable` 返回 false（`refcount > 1`），leaf 被丢弃、退出 heap，`chooseVictim` 继续取下一个。
3. 请求 A 释放：`refcount` 降为 2，`refreshCandidate` 判定不可淘汰，`heap.erase(node)`（若仍在堆则移除，否则幂等无操作）。
4. 请求 B 释放：`refcount` 恢复为 1（仅 cache hold），`refreshCandidate` 判定可淘汰，按原 `candidate_meta` 重新 `upsert` 入堆。

`refreshCandidate` 自身的资格检查（transfer_state + 拓扑 + evictability）幂等且自洽，无论节点当前是否在堆、被引用了几次，都能得到正确的最终成员关系。

不变量：**持有 match 引用期间，节点存活且非空**（refcount > cache hold ⇒ 不可淘汰其块，slot 非空 ⇒ 不被删除），保证 release 时 `refreshCandidate` 操作的节点仍然有效。

### 5.6 调用顺序

```text
match:
  1. evictor.onMatched(match_path)     // 更新所有命中 node 的访问历史并刷新在堆节点
  2. group.referenceBlocks(set)        // 增加临时 block 引用；被引用块 refcount > 1

releaseMatchedBlocks:
  1. group.unreferenceBlocks(set)
  2. evictor.refreshCandidatesAfterRelease(set)  // 对释放涉及的节点逐个 refreshCandidate
```

两步都在 cache mutex 下执行。引用未释放时，相关 node 因 refcount 不满足 `isSlotEvictable()` 而不会被选为 victim；释放后是否恢复候选完全由 `refreshCandidate()` 决定，排序字段不变。

---

## 6. insert 与元数据初始化

insert 通过 `onInsertCommitted(insert_result)` 上报。它必须完成两件事，缺一不可：

1. **初始化新节点的排序元数据**：新节点首次入堆前设置 `last_access_seq = ++access_seq_`、`admission_seq = ++admission_seq_`。否则默认值 0 会让新节点在 LRU 升序中被当作“最旧”而被优先淘汰。
2. **刷新拓扑受影响的候选**：通过 `BlockTreeInsertResult::inserted_nodes` 找出所有新节点，并对每个新节点按全部 group 调用 `refreshCandidate`。`inserted_nodes` 只记录新建后缀，因此当新后缀挂在一个既有 FULL 叶子下时，还必须额外刷新第一个新节点的**直接父节点**；该父节点不在 `inserted_nodes` 中。更高祖先的直接 children 未变化，无需刷新；若直接父节点是 root 则跳过。FULL 由 override 后的 `isSlotEvictable` 自动过滤非叶子，SWA/LINEAR 则接纳所有持有对应 tier 数据且 ready 的节点。完全 overlap、没有新节点的 insert 不产生拓扑刷新。

insert overlap 命中已存在前缀**不视为访问**，不更新 `last_access_seq/hit_count`——排序只由真实 match 驱动（§4.2）。

---

## 7. 异步 tier 迁移

### 7.1 源与目标块所有权

- **源块**：迁移期间由 slot 中原有的 **cache hold** 保活。调用方约束：copy 开始到完成前，不释放源 cache hold、不清源 slot；成功后在 mutex 下先写目标 slot，再释放源 cache hold 并清源 slot（先 `unreferenceBlocks` 用提前保存的块 ID 副本、后 `evictFromTier` 清 slot）；失败保持源 slot 与源 cache hold 原样。
- **目标块**：分配 target 时的 `incRef` 作为 **staged-hold** 保活，target 在写入 slot 前不进入 heap。`finishTierMove` 成功时 staged-hold 转为该 slot 的 cache hold；失败时 `releaseSingleBlock` 释放 target。

在该约束下，refcount 负责内存安全，`transfer_state`（出堆 + 禁重复迁移）负责候选竞争，源与目标都不需要额外的临时保护引用。

### 7.2 demotion

```text
chooseVictim:
  循环 heap.takeBest() -> 检查 isSlotEvictable：
    可淘汰 -> 返回 EvictionMove(source block 快照)
    不可淘汰（refcount > 1）-> 丢弃（已出堆），继续取下一个

beginDemotion（与 takeBest 同一临界区，保证原子）:
  校验 transfer_state == IDLE
  分配 staged target（staged-hold）
  slot.transfer_state = DEMOTING

copy success:
  在 mutex 下写 target slot（staged-hold 转 cache hold）
  unreferenceBlocks(source 副本) -> evictFromTier(source)
  slot.transfer_state = IDLE
  meta 规则见 7.5
  refreshCandidate(target tier)

copy failure:
  releaseSingleBlock(staged target)
  slot.transfer_state = IDLE
  refreshCandidate(source tier)   // 元数据不变，按原排序恢复
```

`reclaimBlocks()` 的 target 为 `NONE` 且同步完成，不设置 `transfer_state`：`takeBest` 出堆后直接 `unreferenceBlocks + evictFromTier`，全程在 mutex 下。

### 7.3 load_back

```text
beginLoadBack:
  仅允许 transfer_state == IDLE
  若该 slot 已有在飞 load_back（transfer_state == LOADING_BACK），
    返回其现有 AsyncContext 供本请求 await 复用，不再发起新迁移
  heap.erase(source)
  分配 staged Device target（staged-hold）
  slot.transfer_state = LOADING_BACK

copy success:
  写 Device slot（staged-hold 转 cache hold）
  unreferenceBlocks(source 副本) -> evictFromTier(source)
  transfer_state = IDLE
  meta 规则见 7.5
  refreshCandidate(Device)

copy failure:
  releaseSingleBlock(staged Device target)
  保留 source slot
  transfer_state = IDLE
  refreshCandidate(source)
```

并发正确性：第二个命中同一 `LOADING_BACK` 源的请求，必须能拿到最终落到 Device 的数据。为此 `beginLoadBack` 在检测到在飞迁移时返回已有 `AsyncContext`；请求 await 该 context 完成后复用已写入的 Device slot，而不是被“跳过”导致返回不完整的 `block_indices`。

### 7.4 级联迁移

一次 victim 选择可能级联到同一节点的多个 group（如 FULL 叶子级联到 SWA/LINEAR，或反向叶子级联）。级联迁移规则：

- 每个参与迁移的 group slot **各自** 校验并置 `transfer_state`（各 slot 独立，互不影响）。
- `finishTierMove` 按 per-move `copy_ok` 分别提交或回滚；单条级联失败只回滚该条（`releaseSingleBlock` target + `refreshCandidate` source），不影响已成功的其它条。
- primary 失败则整体回滚。
- 父节点提升（`onTopologyChanged`）在所有 move 完成后统一触发一次。

### 7.5 tier 迁移的元数据规则

迁移成功写入目标 tier 时，单份 `candidate_meta` 跟随数据：

- **保留** `last_access_seq` 与 `hit_count`（延续 LRU/LFU 访问历史）。
- **刷新** `admission_seq = ++admission_seq_`（标记“进入新 tier 的先后”，供目标 tier 的 FIFO 排序使用）。

迁移失败时元数据完全不变，源按原排序恢复。

---

## 8. 接口定义

### 8.1 ComponentGroup

```cpp
// 统一检查 tier 数据、refcount 与 group 特有拓扑规则。
virtual bool isSlotEvictable(const TreeNode& node, Tier tier) const;
Tier getTopTier(const GroupSlot& slot) const;  // 当前服务 tier

std::vector<BlockIdxType> getBlocks(const GroupSlot&, Tier) const;
void                      setBlocks(GroupSlot&, Tier, const std::vector<BlockIdxType>&);
void                      evictFromTier(TreeNode*, GroupSlot&, Tier);

// block 生命周期（多 Device 池 + HOST/DISK 单池）
GroupBlockSet allocateBlocks(Tier tier, size_t count);
void          unreferenceBlocks(const GroupBlockSet& set) const;
void          referenceBlocks(const GroupBlockSet& set) const;
void          unreferenceBlocks(const GroupBlockSet& set) const;
BlockIdxType  allocateSingleBlock(Tier tier);
void          releaseSingleBlock(Tier tier, BlockIdxType block) const;

virtual size_t computeReuseBlockCount(size_t matched_block_count,
                                      const std::vector<TreeNode*>& path) const = 0;
```

`ComponentGroup` 不再持有 `device_heap/host_heap/disk_heap`，也不再暴露 `tryAddTo*Heap/invalidateHeap/heapForTier`；`GroupSlot` 不再有 `in_*_heap` 标志。惰性引用判断下不需要 `collectReferenceCandidateNodes`——候选是否退出/复位由 `takeBest` 的 refcount 检查与 release 的 `refreshCandidate` 共同裁决。

`GroupBlockSet` 只需记录本次引用/迁移涉及的块，不再附带 `candidate_nodes` 子集：

```cpp
struct GroupBlockSet {
    int  component_group_id{-1};
    Tier tier{Tier::DEVICE};
    // per_node[k]    -> 第 k 个 cache_key（树节点）
    // per_node[k][p] -> device_pools_[p]；HOST/DISK 内层大小为 1
    std::vector<std::vector<BlockIdxType>> per_node;
};
```

### 8.2 BlockTreeEvictor

```cpp
// 私有 heap 所有权；按 group 声明的 component_group_id 索引
struct GroupTierHeaps {
    std::unique_ptr<EvictionHeap> device;
    std::unique_ptr<EvictionHeap> host;
    std::unique_ptr<EvictionHeap> disk;
};
std::vector<GroupTierHeaps> heaps_;  // heaps_[gid], gid == component_group_id
uint64_t access_seq_{0};
uint64_t admission_seq_{0};

// policy 来自 BlockTreeCacheConfig；非法/null/重复 gid 记录日志并返回 false
bool init(const std::vector<Component>&,
          EvictionPolicy device_policy,
          EvictionPolicy host_policy,
          EvictionPolicy disk_policy);

// 对外语义接口
void onInsertCommitted(const BlockTreeInsertResult&);
void onMatched(const std::vector<TreeNode*>& path);
void refreshCandidatesAfterRelease(const GroupBlockSet&);  // release 后逐节点 refreshCandidate
void onTopologyChanged(TreeNode* parent);
void onNodeAboutToRemove(TreeNode*);

std::optional<EvictionMove> chooseVictim(Tier);            // 内部 takeBest + 跳过摘除
bool                        beginDemotion(EvictionMove&);
LoadBackHandle              beginLoadBack(LoadBackItem&);   // 返回新建或已有 AsyncContext
void                        finishTierMove(EvictionMove&, bool copy_ok);
void                        finishLoadBack(LoadBackItem&, bool copy_ok);
CandidateStats              candidateStats() const;

// 私有：唯一候选资格入口
void refreshCandidate(ComponentGroup&, TreeNode*, Tier);
```

`EvictionMove` 与 `LoadBackItem` 用已有 node/group/tier/block 快照描述一次 copy，不携带候选状态字段。

---

## 9. 并发不变量

所有会修改 tree/group/slot/heap 状态的操作在 `BlockTreeCache::mutex_` 下执行；`performCopy` 与 `writeRemoteThrough` 是无锁 I/O 阶段。必须固化的不变量：

1. `transfer_state != IDLE` ⇒ 源 slot 至少保留源 tier 数据 ⇒ 节点非空 ⇒ 不可被 `onNodeAboutToRemove` 删除。
2. 持有 match 引用 ⇒ 节点存活且非空（`refcount > cache hold` 且 slot 非空），保证 release 时 `refreshCandidate` 操作的节点仍有效。
3. `access_seq_ / admission_seq_` 只在 `mutex_` 下读写；copy 阶段不访问。
4. `chooseVictim` 内 `takeBest`+`isSlotEvictable` 检查、以及随后的 `beginDemotion/beginLoadBack`（置 `transfer_state`）必须在同一临界区内原子完成，不得出现“已出堆但未置迁移标记”的中间态。
5. 节点删除前先 `onNodeAboutToRemove`（从所有 heap `erase`），再销毁；`removeEmptyAncestors()` 返回连续清理后第一个未删除的祖先，后续拓扑刷新只能使用该返回值，不能继续解引用原始 parent。

---

## 10. 性能与内存优化

1. **单份 candidate_meta**：每 `GroupSlot` 只存一份排序元数据（约 24B）而非按 tier 存多份，大前缀树 + 多 group 场景下显著降低常驻内存。
2. **精确更新 heap**：物理大小恒等于 ready 候选数，热节点高频 match 不产生失效条目；`takeBest` 无空转扫描。
3. **`std::set + node 索引`**：精确更新，`ordered_.size() == index_.size()` 恒成立、无失效条目堆积；迭代器稳定性使 `node->iterator` 索引成立。相比惰性删除优先队列，消除 stale entry 与 `takeBest` 空转扫描（索引二叉堆可作为后续 cache 局部性优化，接口不变可无缝替换）。
4. **惰性引用判断**：match/release 热路径不做逐候选 suspend/resume 的 heap 操作；被引用节点只在极少被 `takeBest` 选中时才摘除，正确性由精确更新 heap 与逻辑时钟保障。
5. **确定性 tie-break**：`cache_key` + node 地址仅保证本地进程内确定性；若需跨进程可复现排序，应改用稳定标识。

---

## 11. 迁移步骤

1. 将 `EvictionHeap` 实现为 `std::set + node 索引`，提供 exact `upsert/erase/takeBest/contains`（索引二叉堆作为后续性能优化项）。
2. `GroupSlot` 增加单份 `candidate_meta` 与 `SlotTransferState`；删除 `in_*_heap`。
3. 三个 heap 从各 group 迁移到 `BlockTreeEvictor` 私有 `heaps_`。`init()` 遍历 `component_groups_`，读取 group 声明的 gid 并使用 `heaps_[gid]`；null、越界或重复 gid 记录错误日志、清理部分初始化状态并返回 false，由 `BlockTreeCache`/工厂上层处理，不在 evictor 内抛异常。
4. Device/Host/Disk 的 `EvictionPolicy` 归属 `BlockTreeCacheConfig`，由 `BlockTreeCache` 传给 evictor `init()` 创建各 group 的三层 heap；当前只使用默认配置，`ComponentGroup` 不再保存淘汰策略。将 tier 数据、refcount 与 FULL 叶子规则统一收敛到虚函数 `isSlotEvictable()`；实现 `refreshCandidate()`，所有 heap 成员关系经它更新。
5. insert 改为 `onInsertCommitted()`（含元数据初始化 + 拓扑刷新）；match 改为 `onMatched()`（含在堆节点刷新）。
6. match 只调用 `referenceBlocks()`，不主动操作 heap；`chooseVictim` 内 `takeBest` 增加 `isSlotEvictable` 跳过摘除；release 后调用 `refreshCandidatesAfterRelease()`。
7. 淘汰主流程改为 `chooseVictim/beginDemotion/finishTierMove`；级联按 §7.4 per-move 处理。
8. load_back 按 §7.3 迁移，含在飞 `AsyncContext` 复用。
9. 节点删除前 `onNodeAboutToRemove()`，父拓扑变化后 `onTopologyChanged()`。
10. `CacheStats` 改为读取 `candidateStats()`。

每步保持既有 direct reclaim、watermark、Device->Host、Host->Disk 与混合 group 测试可运行。

---

## 12. 测试与验收

### 12.1 EvictionHeap

1. 单一热节点连续 100000 次 upsert 后，物理条目数恒为 1。
2. LRU 中 A/B/C 的真实访问顺序决定淘汰顺序；refcount 临时变化、回滚、父节点提升不改变排序。
3. FIFO 只在进入目标 tier 时按 `admission_seq` 变化，不受 match/回滚影响。
4. `erase()` 后有序索引与 node 索引同步删除，无失效残留。
5. 相同排序键用 `cache_key` 与 node 地址稳定排序。

### 12.2 访问、insert 与惰性引用

1. `onMatched` 后，在堆的命中节点排序键被刷新；窗口外命中的 SWA 节点顺序随之更新。
2. 所有新插入节点的 `last_access_seq` 为当前时钟；FULL 仅叶子入堆，SWA/LINEAR 的 ready 新节点全部入堆；若第一个新节点挂在既有非 root 父节点下，只重新判定该直接父节点，不遍历更高祖先。
3. FULL 长 path 命中：中间节点只更新访问历史，不产生 heap 操作。
4. 被 match 引用的节点若被 `takeBest` 选中，因 `refcount > 1` 被跳过摘除，不作为 victim。
5. 两个并发 match 命中同一叶子：首次释放不恢复，末次释放后按原排序经 `refreshCandidate` 恢复。
6. insert overlap 不更新访问历史。

### 12.3 异步迁移

1. demotion 开始后 source 出堆，`transfer_state == DEMOTING`；成功后源清理、目标按 §7.5 元数据规则入堆。
2. demotion 失败后 source 按原排序恢复、target 释放。
3. load_back 期间 source 不再参与 eviction；第二个命中 `LOADING_BACK` 的请求复用同一 `AsyncContext` 并拿到完整数据。
4. load_back 成功后 target 在 copy 完成、slot 稳定、外部引用释放后才进入 Device heap。
5. 级联迁移单条失败只回滚该条，成功条不受影响；父节点提升在末尾触发一次。
6. direct reclaim 同步完成，不设置 `transfer_state`。

### 12.4 性能

构造“少量冷候选 + 大量高频 match 热节点”：

1. heap 物理大小始终等于当前 ready 候选数，不随访问次数增长。
2. 一次 victim 选择无失效条目扫描；被引用的热节点至多触发一次跳过摘除。
3. 冷节点仍按 LRU 优先被选中；引用、回滚、父节点提升不改变其它节点相对顺序。

---

## 13. 待确认项

1. `LOADING_BACK` 下第二请求复用 `AsyncContext` 的等待/超时语义由 load_back 调度层细化；本设计保证不重复迁移且请求最终可拿到数据。
2. 单份 `candidate_meta` + `transfer_state` 的常驻内存增量应在大前缀树压测中记录实际数值。
3. 跨 group victim 选择按 group 遍历顺序进行；全局 LRU 是独立策略，不在本设计范围内。

---

## 14. 结论

heap 收敛到 `BlockTreeEvictor`，`BlockTreeCache` 完全不感知 heap，`ComponentGroup` 只描述候选资格与 block/slot 语义。基于“淘汰后源块清理、稳态单 tier 服务”的不变量，排序元数据只需一份、随数据 tier 移动而跟随。`EvictionHeap` 选定 `std::set + node 索引` 精确更新实现，消除失效条目与空转扫描；逻辑时钟保证排序只由真实访问驱动。`transfer_state` 表达真实异步迁移状态，配合 cache hold 与 staged-hold 使源、目标块所有权明确、无候选竞争。match 采用惰性引用判断：只增加引用、不主动操作 heap，被引用节点在 `takeBest` 阶段按 refcount 跳过摘除、在引用释放后经 `refreshCandidate` 复位候选资格——既保持 FULL 只处理叶子的效率，又将 match/release 热路径的 heap 开销降到最低。
