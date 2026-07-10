# BlockTreeCache C++ 代码参考

> 本文件是 [BlockTreeCache 设计文档](rtp_llm_block_tree_cache_design.md) 的配套参考。
> 包含所有核心 C++ 数据结构定义、接口签名和配置代码，供实现时查阅。

---

📚 **BlockTreeCache 设计文档系列**
| [设计文档（主）](rtp_llm_block_tree_cache_design.md) | **C++ 代码参考** | [示例与详细图示](rtp_llm_block_tree_cache_examples.md) |

---

## 目录

1. [基础类型](#1-基础类型)
2. [核心数据结构](#2-核心数据结构)
3. [淘汰机制数据结构](#3-淘汰机制数据结构)
4. [传输与存储接口](#4-传输与存储接口)
5. [集成接口](#5-集成接口)
6. [配置与 API](#6-配置与-api)

---

## 1. 基础类型

### MemoryBlockLayerTagSlot

`MemoryBlockLayerTagSlot` **（Memory/Disk 层独有）** 描述 memory block 内部一个 (layer, group) 槽位的位置和大小。与现有 `KVCacheMemoryConnector::layerTagSlots()` 语义一致。Device 层不需要此概念——每个 device block 直接对应一个独立的 Device BlockPool。

```cpp
struct MemoryBlockLayerTagSlot {
    int         layer_id{-1};
    std::string tag;             // group tag，如 "csa_kv", "hca_kv", "swa_kv"
    size_t      stride_bytes{0}; // 该槽位在 memory block 中占的字节数
};
```

`MemoryBlockLayerTagSlot` 列表由 Component 持有，用于指导 D2H/H2D 拷贝时 memory block 内的 byte offset 计算（与现有 `prepareCopyBuffers` 逻辑一致）。Tree 本身不感知 `MemoryBlockLayerTagSlot` 细节。

---

## 2. 核心数据结构

### MatchValidator

匹配验证器接口，定义在 ComponentGroup.h 中，作为顶层类。

```cpp
class MatchValidator {
public:
    virtual ~MatchValidator() = default;
    virtual bool validate(const TreeNode* node, const GroupSlot& slot) = 0;
};
```

### ComponentGroup（主动管理实体）

主动管理实体，替代原 ComponentGroupInfo + ITreeComponent。同 group 所有 Component 生命周期完全一致，ComponentGroup 统一驱动行为。基类提供默认实现（virtual，非纯虚），子类按需覆盖。

共定义 **7 个**行为钩子：Match 2 个（`createMatchValidator` + `finalizeMatchResult`）、Insert 2 个（`commitInsertData` + `updateOnInsertOverlap`）、Evict 2 个（`evictFromTier` + `driveEviction`）、Transfer 1 个（`buildTransfer`）。

```cpp
class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    // ---- 静态元数据（构造时确定）----
    int                     component_group_id;
    CacheGroupType          group_type;          // FULL / SWA / LINEAR（决定淘汰候选规则和匹配语义）
    std::vector<int>        component_indices;   // 该 group 包含的 Component 索引
    size_t                  host_block_size;     // host block 大小 = Σ stride_bytes

    // ---- 三层淘汰堆（unique_ptr，构造时创建）----
    std::unique_ptr<EvictionHeap> device_heap;   // Device 层淘汰候选
    std::unique_ptr<EvictionHeap> host_heap;     // Host 层淘汰候选
    std::unique_ptr<EvictionHeap> disk_heap;     // Disk 层淘汰候选

    // ---- Match（2 个）----
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;
    virtual void finalizeMatchResult(BlockTreeMatchResult& result);  // 非 const，允许修改结果

    // ---- Insert（2 个）----
    virtual void commitInsertData(TreeNode* node, GroupSlot& slot,
                                  const std::vector<BlockIdxType>& block_indices);
    virtual void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot);

    // ---- Evict（2 个）----
    // evictFromTier: 对指定节点执行本 group 的淘汰（释放该层数据，降级或直接释放）
    virtual void evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);
    // driveEviction: 从本 group 的 tier-heap 中弹出候选并执行淘汰，返回 EvictionResult
    virtual std::optional<EvictionResult> driveEviction(int num_blocks, Tier tier);

    // ---- Transfer（1 个）----
    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    // ---- Heap 管理 ----
    virtual void tryAddToDeviceHeap(TreeNode* node) = 0;
    virtual void tryAddToHostHeap(TreeNode* node);   // 基类提供默认实现
    virtual void tryAddToDiskHeap(TreeNode* node);   // 基类提供默认实现
};

// 具体实现（各自定义不同的堆候选规则和匹配语义）：
class FullComponentGroup : public ComponentGroup { ... };     // Leaf-based heaps
class SWAComponentGroup : public ComponentGroup { ... };      // Any-node heaps
class LinearComponentGroup : public ComponentGroup { ... };   // Any-node heaps
```

### GroupSlot（多层数据位置）

每个 `GroupSlot` 对应一个 ComponentGroup 在一个 TreeNode 上的数据位置。

```cpp
struct GroupSlot {
    // L1: GPU Device — 每个独立 Device BlockPool 一个 block（同 group 的每个 Component 各一个）
    std::vector<BlockIdxType> device_blocks;  // 长度 = 该 group 的独立 Device BlockPool 数量（= component_indices.size()）
    // L2: CPU Host — 同 group 共享一个 host block（仅 REUSABLE group 有效）
    BlockIdxType  host_block{NULL_BLOCK_IDX};
    // L3: Disk — 同 group 共享一个 disk slot（仅 REUSABLE group 有效）
    BlockIdxType    disk_slot{NULL_BLOCK_IDX};  // 与 host_block 类型统一，由 DiskBlockPool 管理
    // L4: Remote 不在 GroupSlot 中（由 StorageBackend 自管 key 映射）

    // 驱逐状态（per-group，因为淘汰以 group 为单位）
    bool in_device_heap{false};
    bool in_host_heap{false};
    bool in_disk_heap{false};

    // 状态查询
    bool has_device_value() const {
        return std::any_of(device_blocks.begin(), device_blocks.end(),
                           [](auto b) { return !isNullBlockIdx(b); });
    }
    bool has_host_value() const   { return host_block != NULL_BLOCK_IDX; }
    bool has_disk_value() const   { return disk_slot != NULL_BLOCK_IDX; }
    bool is_empty() const {
        return !has_device_value() && !has_host_value() && !has_disk_value();
    }
};
```

### TreeNode

树节点，包含树结构（cache_key、children、parent）和多层数据位置（group_slots）。

```cpp
struct TreeNode {
    // ---- 树结构 ----
    CacheKeyType                          cache_key{0};     // block 内容的 hash（= 现有 CacheKeyType，树操作实际使用此字段）
    std::vector<int>                      token_ids;        // block 原始 token 序列（长度 = seq_size_per_block，调试/校验用，不参与树操作）
    std::unordered_map<CacheKeyType, TreeNode*> children;   // 子节点（cache_key → child）
    TreeNode*                             parent{nullptr};

    // ---- 多层数据位置 ----
    std::vector<GroupSlot>                group_slots;      // 按 component_group_id 位置索引（group_slots[i] 对应 component_groups_[i]）
};
```

### Component（纯描述符）

纯数据描述符，无行为。持有 `MemoryBlockLayerTagSlot` 列表供 CopyEngine 计算偏移。

```cpp
struct Component {
    int                                     component_id;
    int                                     component_group_id;
    CacheGroupType                          type;           // FULL / SWA / LINEAR
    std::vector<MemoryBlockLayerTagSlot>    memory_block_layer_tag_slots;  // memory block 布局
    int                                     device_pool_index;  // 对应哪个独立 Device BlockPool
};
```

### BlockTreeMatchResult

匹配结果，包含最佳匹配节点、匹配 block 数、异步加载上下文和分层加载统计。

```cpp
struct BlockTreeMatchResult {
    TreeNode*       matched_node{nullptr};           // 最佳匹配节点
    size_t          matched_blocks{0};               // 匹配的 block 数
    BlockIndicesType block_indices;                  // 匹配到的 block 索引序列（match 完成后均已位于 GPU）

    std::shared_ptr<AsyncContext> async_context;     // 异步加载上下文（如 match 内部触发了 load_back/prefetch）
    size_t          load_back_blocks{0};             // 本次 match 内部加载的 block 数（0 = 无加载）

    // 分层统计（用于上层监控和汇报）
    size_t          host_load_back_blocks{0};        // 本次 match 从 Host 加载的 block 数
    size_t          disk_load_back_blocks{0};        // 本次 match 从 Disk 加载的 block 数
    size_t          remote_load_back_blocks{0};      // 本次 match 从 Remote 加载的 block 数
};
```

---

## 3. 淘汰机制数据结构

### EvictionEntry（淘汰堆元素）

为适配多种淘汰策略（LRU/LFU/FIFO/Priority），堆中每个元素维护统一的状态字段，不同策略使用不同子集。

```cpp
struct EvictionEntry {
    TreeNode*       node{nullptr};
    int             component_group_id{-1};  // 操作的 group

    // 淘汰策略所需的状态字段（不同策略使用不同子集）：
    uint64_t        last_access_time{0};   // LRU: 时间戳（steady_clock 纳秒，match/insert 时由 Component 更新，支持 lifetime 统计）
    int             hit_count{0};         // LFU: 命中计数
    int             priority{0};          // Priority: 自定义优先级
    uint64_t        insert_seq{0};        // FIFO: 插入序号（插入时设置，之后不更新）
};
```

### EvictionPolicy（淘汰策略枚举）

```cpp
enum class EvictionPolicy {
    LRU,      // 按 last_access_time 升序（最久未访问优先淘汰）
    LFU,      // 按 hit_count 升序（最少命中优先淘汰）
    FIFO,     // 按 insert_seq 升序（最早插入优先淘汰）
    PRIORITY  // 按 priority 升序（最低优先级优先淘汰）
};
```

### EvictionHeap（淘汰堆）

由底层容器（`std::priority_queue`）+ 辅助索引（`entry_map_`）组成，采用失效标记策略（Lazy Deletion）。

```cpp
class EvictionHeap {
public:
    explicit EvictionHeap(EvictionPolicy policy);

    // 添加节点到堆
    void push(TreeNode* node, int group_id);

    // 弹出最应淘汰的节点（自动跳过失效条目）
    std::optional<EvictionEntry> pop();

    // 标记节点失效（逻辑删除，实际在 pop 时跳过）
    void invalidate(TreeNode* node);

    // 节点被访问时更新状态（match/insert 时调用）
    void onAccess(TreeNode* node);

    // 查询
    bool contains(TreeNode* node) const;
    bool empty() const;
    size_t size() const;

private:
    EvictionPolicy policy_;
    // 底层容器：std::priority_queue，比较器根据 policy_ 选择排序字段
    PriorityQueue heap_;
    // 辅助索引：TreeNode* → EvictionEntry，用于 O(1) 查找和失效检查
    std::unordered_map<TreeNode*, EvictionEntry> entry_map_;
};
```

### isSlotEvictable（引用计数检查）

evictability 由 `ComponentGroup::isSlotEvictable(slot, tier)` 判定，`driveEviction` 弹出堆顶后调用它跳过不可淘汰候选。块 refCount == 1 表示仅 BlockTreeCache 持有 cache-hold，可淘汰；> 1 表示还有 request / match 保护 / CopyEngine 在用，不可淘汰。按 tier 只检查对应层的块（DEVICE 遍历多 pool，HOST/DISK 单块）；块为空或无 pool 拥有时视为可淘汰。

```cpp
bool ComponentGroup::isSlotEvictable(const GroupSlot& slot, Tier tier) const {
    auto pool_evictable = [](const auto& pool, BlockIdxType block) {
        if (isNullBlockIdx(block) || !pool) {
            return true;
        }
        return pool->isAllocated(block) && pool->refCount(block) == 1;
    };
    switch (tier) {
        case Tier::DEVICE:
            for (size_t i = 0; i < slot.device_blocks.size(); ++i) {
                const auto& pool = i < device_pools_.size() ? device_pools_[i] : nullptr;
                if (!pool_evictable(pool, slot.device_blocks[i])) {
                    return false;
                }
            }
            return true;
        case Tier::HOST:
            return !slot.has_host_value() || pool_evictable(host_pool_, slot.host_block);
        case Tier::DISK:
            return !slot.has_disk_value() || pool_evictable(disk_pool_, slot.disk_slot);
        default:
            return false;
    }
}
```

### EvictionResult（淘汰结果）

Phase 1 选择候选时产生，包含传输描述和需要释放/分配的 block 信息。

```cpp
struct EvictionResult {
    TreeNode*               node;
    int                     component_group_id;
    Tier                    source_tier;         // 从哪层淘汰
    Tier                    target_tier;         // 降级到哪层（NONE 表示直接释放）
    TransferDescriptor      transfer;            // 传输描述（CopyEngine 使用）
    std::vector<BlockIdxType> blocks_to_release; // 需要释放的源层级 block
    BlockIdxType            target_block;        // 需要分配的目标层级 block
};
```

### EvictionTask（异步淘汰任务状态机）

纯头文件的状态机结构体（`EvictionTask.h`），无 `run()` 方法，任务执行逻辑在 `BlockTreeCache` 中。

```cpp
enum class EvictionTaskState : int8_t {
    PENDING   = 0,  // 任务已创建，未开始
    RUNNING   = 1,  // 拷贝进行中（无锁）
    COMPLETED = 2,  // 拷贝成功，onEvictionComplete 已完成
    FAILED    = 3,  // 拷贝失败，回滚已执行
};

struct EvictionTask {
    EvictionTaskState state{EvictionTaskState::PENDING};
    std::string       error_message;

    bool canTransition(EvictionTaskState from, EvictionTaskState to) const;
    bool transition(EvictionTaskState new_state);
    bool isTerminal() const;
};
```

---

## 4. 传输与存储接口

### TransferDescriptor（传输描述符）

以 component group 为操作单位。D2H 降级时收集同 group 的所有 `device_blocks`，通过 Component 的 `MemoryBlockLayerTagSlot` 信息计算 byte offset，打包写入一个 `host_block`。

```cpp
struct TransferDescriptor {
    // Tier 枚举定义在 TreeNode.h 顶层（enum class Tier { DEVICE, HOST, DISK, REMOTE, NONE }）
    Tier                source_tier{Tier::NONE};
    Tier                target_tier{Tier::NONE};
    int                 component_group_id{-1};  // ★ 操作的 component group
    std::vector<TreeNode*> nodes;

    // 通用源/目标数据，具体含义由 source_tier/target_tier 决定：
    //   D2H: source = device_blocks（多个）, target = host_block（一个）
    //   H2D: source = host_block（一个）, target = device_blocks（多个）
    //   H2Disk/Disk2H: source/target 包含 disk_slot
    std::vector<std::vector<BlockIdxType>> source_blocks;  // [node_idx][block_idx]
    std::vector<BlockIdxType>              target_blocks;  // [node_idx]

    // Remote
    std::vector<std::string> storage_keys;    // 存储 key（Remote 用）
};
```

### StorageBackend（远端存储接口）

所有远端存储通过统一的 `StorageBackend` 接口抽象（替代旧 `RemoteConnector`）。

```cpp
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    // 异步批量读：从 Remote 拉取数据到 Host 缓冲区
    virtual std::shared_ptr<AsyncContext>
    batchRead(const std::vector<std::string>& keys,
              std::vector<std::vector<char>>& results) = 0;

    // 异步批量写：将数据写入 Remote
    virtual std::shared_ptr<AsyncContext>
    batchWrite(const std::vector<std::pair<std::string, std::vector<char>>>& items) = 0;

    // 批量查询：检查远端是否存在数据
    virtual std::shared_ptr<AsyncContext>
    batchExists(const std::vector<std::string>& keys,
                std::vector<bool>& results) = 0;

    // 批量删除
    virtual std::shared_ptr<AsyncContext>
    batchDelete(const std::vector<std::string>& keys) = 0;
};
```

---

## 5. 集成接口

### KVCacheAllocator 修改

`KVCacheAllocator` 的公开接口不变，内部替换 `SharedBlockCachePtr` 为 `BlockTreeCachePtr`。

```cpp
class KVCacheAllocator {
    // ... 现有接口不变 ...

private:
    // 替换: SharedBlockCachePtr shared_block_cache_;
    BlockTreeCachePtr block_tree_cache_;  // ★ 新成员（淘汰流程协调者）
    // BlockPool 继续作为 GPU block 分配器，不引入额外 Allocator

    // initMalloc 中的 match 逻辑:
    //   旧: shared_block_cache_->match(cache_key)
    //   新: block_tree_cache_->match(cache_keys)
    //
    // insertIntoCache 中的 put 逻辑:
    //   旧: shared_block_cache_->put(cache_key, slots, is_resident)
    //   新: block_tree_cache_->insert(cache_keys, slots)
    //
    // popBlocksFromCache 中的驱逐:
    //   旧: shared_block_cache_->selectAndEvict(min_blocks)
    //   新: block_tree_cache_->evict(min_blocks, Tier::DEVICE)
};
```

---

## 6. 配置与 API

### BlockTreeCacheConfig（配置结构体）

从 `KVCacheConfig` 提取 L2/L3 层相关字段，定义在 `BlockTreeCache.h` 中。

```cpp
struct BlockTreeCacheConfig {
    int64_t memory_cache_size_mb{0};          // L2 Host pool 容量（0 = 禁用）
    int64_t memory_cache_disk_size_mb{0};     // L3 Disk pool 容量（0 = 禁用）
    std::string memory_cache_disk_path;       // 磁盘文件路径
    bool    memory_cache_disk_buffered_io{true}; // 是否使用缓冲 I/O
    size_t  block_size_bytes{0};              // block 大小（从 CacheConfig 获取）
    bool    enable_device_cache{true};        // L1 开关
    bool    enable_memory_cache{false};       // L2 开关
    bool    enable_disk_cache{false};         // L3 开关
    bool    enable_remote_cache{false};       // L4 开关
    int     eviction_thread_pool_size{2};     // 淘汰线程池大小

    size_t hostBlockCount() const;            // 计算 Host block 数
    size_t diskPoolSizeBytes() const;         // 计算 Disk pool 总字节数
};
```

### BlockTreeCache 构造函数

接受已构建的组件（tree、component_groups 等），由工厂函数组装。

```cpp
class BlockTreeCache {
public:
    BlockTreeCache(std::unique_ptr<BlockTree> tree,
                   std::vector<ComponentGroupPtr> component_groups,
                   std::vector<Component> components,
                   BlockPoolPtr host_pool = nullptr,
                   std::shared_ptr<DiskBlockPool> disk_pool = nullptr,
                   int eviction_thread_pool_size = 2,
                   std::shared_ptr<StorageBackend> storage_backend = nullptr,
                   bool enable_device_cache = true,
                   bool enable_memory_cache = false,
                   bool enable_disk_cache = false,
                   bool enable_remote_cache = false,
                   std::shared_ptr<BroadcastManager> broadcast_manager = nullptr);
};
```

### createBlockTreeCache 工厂函数

负责从 `CacheConfig` + `KVCacheConfig` 推导所有组件并组装 `BlockTreeCache`，定义在 `BlockTreeCacheFactory.h` 中。

```cpp
BlockTreeCachePtr createBlockTreeCache(
    const CacheConfig& cache_config,
    const KVCacheConfig& kv_cache_config,
    const std::shared_ptr<KVCacheAllocator>& allocator,
    const SWAGroupConfig& swa_configs = {},
    std::shared_ptr<StorageBackend> storage_backend = nullptr,
    std::shared_ptr<BroadcastManager> broadcast_manager = nullptr);
```

### BlockTreeCache 对外接口

三个核心接口（线程安全）：`match` / `insert` / `evict`，外加查询和控制接口。

```cpp
class BlockTreeCache {
public:
    // 核心接口：match / insert / evict
    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void insert(TreeNode* parent,
                const CacheKeysType& cache_keys,
                const std::vector<std::vector<GroupSlot>>& slots);
    int evict(size_t num_blocks, Tier tier = Tier::DEVICE);

    // 查询接口
    bool isEvictable(TreeNode* node, int group_id) const;  // 通过 BlockPool.refCount() 查询指定 group 是否可驱逐
    CacheStats getStats() const;

    // 等待所有 pending 任务完成（shutdown / 测试用）
    void waitForPendingTasks();
};
```
