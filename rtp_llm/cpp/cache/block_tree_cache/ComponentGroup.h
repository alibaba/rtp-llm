#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

class AsyncContext;

// Pure-descriptor component: describes one device pool's layout.
struct Component {
    int                                  component_id{-1};
    int                                  component_group_id{-1};
    CacheGroupType                       type{CacheGroupType::FULL};
    std::vector<MemoryBlockLayerTagSlot> memory_block_layer_tag_slots;
    int                                  device_pool_index{-1};
};

// Unified block set: a batch of blocks owned by one component group at one tier.
// Self-describing layout: outer index is the cache_key (tree node), inner index
// equals the device pool index, so release/reference needs no i % num_pools.
struct GroupBlockSet {
    int  component_group_id{-1};
    Tier tier{Tier::DEVICE};
    // per_node[k]    -> the k-th cache_key (tree node)
    // per_node[k][p] -> device_pools_[p]; inner size == 1 for HOST/DISK.
    std::vector<std::vector<BlockIdxType>> per_node;
};

// Match result returned by BlockTreeCache::match().
struct BlockTreeMatchResult {
    TreeNode*        matched_node{nullptr};
    size_t           matched_blocks{0};
    BlockIndicesType block_indices;

    // Per-component-group device block indices along the matched prefix.
    // key = component_group_id, value = the group's device blocks in match order
    // (NULL_BLOCK_IDX dropped). Whole-sequence consumers (allocators) pick reuse
    // blocks per group from here; block_indices keeps the FULL aggregate for the
    // legacy load_back path.
    std::unordered_map<int, BlockIndicesType> group_block_indices;

    // Structured match-protection references, one entry per group. Release basis.
    std::vector<GroupBlockSet> matched_block_sets;

    std::shared_ptr<AsyncContext> async_context;
    size_t                        load_back_blocks{0};
    size_t                        host_load_back_blocks{0};
    size_t                        disk_load_back_blocks{0};
    size_t                        remote_load_back_blocks{0};
};

// Match validator interface.
class MatchValidator {
public:
    virtual ~MatchValidator()                                          = default;
    virtual bool validate(const TreeNode* node, const GroupSlot& slot) = 0;
};

// Transfer type for buildTransfer.
enum class TransferType {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_DISK,
    DISK_TO_HOST,
    DEVICE_TO_REMOTE,
    HOST_TO_REMOTE,
    REMOTE_TO_DEVICE,
};

// A single eviction unit selected from a component group. The same descriptor is
// used for primary eviction and cascade eviction moves.
struct EvictionMove {
    TreeNode*                 node{nullptr};
    int                       component_group_id{-1};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;
};

// ComponentGroup: active management entity.
class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    // ---- Static metadata ----
    int              component_group_id{-1};
    CacheGroupType   group_type{CacheGroupType::FULL};
    std::vector<int> component_indices;
    size_t           host_block_size{0};

    // ---- Three-tier eviction heaps ----
    std::unique_ptr<EvictionHeap> device_heap;
    std::unique_ptr<EvictionHeap> host_heap;
    std::unique_ptr<EvictionHeap> disk_heap;

    // ---- Match ----
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;
    virtual void                            finalizeMatchResult(BlockTreeMatchResult& result) {}

    // ---- Insert (base class provides default; subclasses may override) ----
    virtual void commitInsertData(TreeNode* node, GroupSlot& slot, const std::vector<BlockIdxType>& block_indices);
    virtual void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot);

    // ---- Evict (base class provides default; subclasses may override) ----
    virtual void                        evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);
    virtual std::optional<EvictionMove> driveEviction(int num_blocks, Tier tier);

    // ---- Transfer (base class provides default; subclasses may override) ----
    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    // ---- Heap management ----
    virtual void tryAddToDeviceHeap(TreeNode* node) = 0;

    // Shared across all subclasses (implemented in ComponentGroup.cc).
    // Virtual: FullComponentGroup overrides to add Leaf checks.
    virtual void tryAddToHostHeap(TreeNode* node);
    virtual void tryAddToDiskHeap(TreeNode* node);

    // ---- Leaf checks (used by FullComponentGroup and base tryAddTo*Heap) ----
    bool isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const;

    // ---- Reference count for path lock/load_back (per-group strategy) ----
    // Returns number of nodes from path tail to process.
    virtual size_t computeReferenceCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const = 0;

    // ---- Heap access helpers ----
    EvictionHeap* heapForTier(Tier tier) {
        switch (tier) {
            case Tier::DEVICE:
                return device_heap.get();
            case Tier::HOST:
                return host_heap.get();
            case Tier::DISK:
                return disk_heap.get();
            default:
                return nullptr;
        }
    }

    // ---- Per-group pool injection and access ----
    void setDevicePools(std::vector<DeviceBlockPoolPtr> pools) {
        device_pools_ = std::move(pools);
    }
    void setHostPool(std::shared_ptr<HostBlockPool> pool) {
        host_pool_ = std::move(pool);
    }
    void setDiskPool(std::shared_ptr<DiskBlockPool> pool) {
        disk_pool_ = std::move(pool);
    }

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<DiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    // ---- Per-group DeviceKVCacheGroup (single-pool sequence-allocation entity) ----
    // Each DeviceKVCacheGroup wraps exactly one DeviceBlockPool (like the legacy
    // KVCacheGroup). Allocators reach these via BlockTreeCache::deviceKVGroup().
    void setDeviceKVGroups(std::vector<DeviceKVCacheGroupPtr> groups) {
        device_kv_groups_ = std::move(groups);
    }
    const std::vector<DeviceKVCacheGroupPtr>& deviceKVGroups() const {
        return device_kv_groups_;
    }
    DeviceKVCacheGroupPtr deviceKVGroup(size_t pool_index = 0) const {
        return pool_index < device_kv_groups_.size() ? device_kv_groups_[pool_index] : nullptr;
    }

    // Device pool usage queries for watermark checking
    size_t devicePoolCount() const {
        return device_pools_.size();
    }

    // Returns true if any device pool's usage exceeds capacity * ratio
    bool anyDevicePoolExceedsRatio(double ratio) const {
        for (const auto& pool : device_pools_) {
            if (!pool)
                continue;
            size_t capacity = pool->totalBlocksNum();
            if (capacity == 0)
                continue;
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold)
                return true;
        }
        return false;
    }

    // Returns the maximum excess across all device pools
    size_t devicePoolMaxExcess(double ratio) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            if (!pool)
                continue;
            size_t capacity = pool->totalBlocksNum();
            if (capacity == 0)
                continue;
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold)
                max_excess = std::max(max_excess, used - threshold);
        }
        return max_excess;
    }

    // Host/Disk pool usage queries for watermark checking
    size_t hostPoolUsed() const {
        return host_pool_ ? (host_pool_->totalBlocksNum() - host_pool_->freeBlocksNum()) : 0;
    }
    size_t hostPoolCapacity() const {
        return host_pool_ ? host_pool_->totalBlocksNum() : 0;
    }
    size_t diskPoolUsed() const {
        return disk_pool_ ? (disk_pool_->totalBlocksNum() - disk_pool_->freeBlocksNum()) : 0;
    }
    size_t diskPoolCapacity() const {
        return disk_pool_ ? disk_pool_->totalBlocksNum() : 0;
    }

    // ---- Device block reference counting via pools ----
    // A cache-category holder is added with incRef() and released with releaseRef(), which
    // returns capacity only when the refcount reaches 0.
    void referenceDeviceBlocks(const std::vector<BlockIdxType>& device_blocks) const {
        for (size_t i = 0; i < device_blocks.size() && i < device_pools_.size(); ++i) {
            if (device_pools_[i] && !isNullBlockIdx(device_blocks[i])) {
                device_pools_[i]->incRef(device_blocks[i]);
            }
        }
    }
    void releaseDeviceBlocks(const std::vector<BlockIdxType>& device_blocks) const {
        for (size_t i = 0; i < device_blocks.size() && i < device_pools_.size(); ++i) {
            if (device_pools_[i] && !isNullBlockIdx(device_blocks[i])) {
                device_pools_[i]->releaseRef(device_blocks[i]);
            }
        }
    }

    // ---- Unified structured block lifecycle (device multi-pool & match) ----
    // count = number of cache_keys. DEVICE allocates count blocks per pool,
    // organized into per_node[k][p]. Returns empty set on failure (rolled back).
    GroupBlockSet allocateBlocks(Tier tier, size_t count);
    // Release cache-holding references; inner index is the pool index.
    void releaseBlocks(const GroupBlockSet& set) const;
    // Temporary external references, mainly for match result protection.
    void referenceBlocks(const GroupBlockSet& set) const;
    void unreferenceBlocks(const GroupBlockSet& set) const;

    // Scalar helpers for single-pool tiers (HOST/DISK); DEVICE is multi-pool, use allocateBlocks.
    BlockIdxType allocateSingleBlock(Tier tier);
    void         releaseSingleBlock(Tier tier, BlockIdxType block) const;

    // ---- Slot helpers ----
    std::vector<BlockIdxType> getBlocks(const GroupSlot& slot, Tier tier) const;
    void                      setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks);

    // ---- Evictability: block held only by the single cache reference ----
    bool isSlotEvictable(const GroupSlot& slot, Tier tier) const;

    // ---- Generic heap helpers (dispatch by tier) ----
    void tryAddToHeap(TreeNode* node, Tier tier);
    void invalidateHeap(TreeNode* node, Tier tier);
    void clearHeapFlag(GroupSlot& slot, Tier tier);

protected:
    std::vector<DeviceBlockPoolPtr> device_pools_;
    std::shared_ptr<HostBlockPool>  host_pool_;
    std::shared_ptr<DiskBlockPool>  disk_pool_;

    // One DeviceKVCacheGroup per device pool (single-pool). Populated by the factory.
    std::vector<DeviceKVCacheGroupPtr> device_kv_groups_;
};

using ComponentGroupPtr = std::shared_ptr<ComponentGroup>;

}  // namespace rtp_llm
