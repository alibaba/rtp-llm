#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

class AsyncContext;

struct PendingLoadBackItem {
    TreeNode*                 node{nullptr};
    int                       group_id{-1};
    Tier                      source_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
};

class LoadBackTicket {
public:
    using CommitCallback = std::function<std::shared_ptr<AsyncContext>(const std::vector<PendingLoadBackItem>& items)>;
    using AbortCallback  = std::function<void(const std::vector<PendingLoadBackItem>& items)>;

    LoadBackTicket(CommitCallback commit_callback, AbortCallback abort_callback):
        commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}
    ~LoadBackTicket() {
        if (!committed_ && !items_.empty() && abort_callback_) {
            abort_callback_(items_);
        }
    }

    LoadBackTicket(const LoadBackTicket&)            = delete;
    LoadBackTicket& operator=(const LoadBackTicket&) = delete;
    LoadBackTicket(LoadBackTicket&&)                 = delete;
    LoadBackTicket& operator=(LoadBackTicket&&)      = delete;

    std::shared_ptr<AsyncContext> commit() {
        if (committed_ || items_.empty() || !commit_callback_) {
            committed_ = true;
            return nullptr;
        }
        committed_ = true;
        return commit_callback_(items_);
    }

    bool empty() const {
        return items_.empty();
    }

    std::vector<PendingLoadBackItem>& items() {
        return items_;
    }

private:
    CommitCallback                   commit_callback_;
    AbortCallback                    abort_callback_;
    std::vector<PendingLoadBackItem> items_;
    bool                             committed_{false};
};

struct Component {
    int                                  component_id{-1};
    int                                  component_group_id{-1};
    CacheGroupType                       type{CacheGroupType::FULL};
    std::vector<MemoryBlockLayerTagSlot> memory_block_layer_tag_slots;
    int                                  device_pool_index{-1};
};

struct GroupBlockSet {
    int                                    component_group_id{-1};
    Tier                                   tier{Tier::DEVICE};
    std::vector<std::vector<BlockIdxType>> per_node;
};

struct BlockTreeMatchResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};

    // Keyed by allocator-facing per-tag gid.
    std::unordered_map<int, BlockIndicesType> group_block_indices;

    std::vector<GroupBlockSet> matched_block_sets;

    std::shared_ptr<AsyncContext> async_context;
    size_t                        load_back_blocks{0};
    size_t                        host_load_back_blocks{0};
    size_t                        disk_load_back_blocks{0};
    size_t                        remote_load_back_blocks{0};

    std::shared_ptr<LoadBackTicket> load_back_ticket;
};

class MatchValidator {
public:
    virtual ~MatchValidator()                                          = default;
    virtual bool validate(const TreeNode* node, const GroupSlot& slot) = 0;
};

enum class TransferType {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_DISK,
    DISK_TO_HOST,
    DEVICE_TO_REMOTE,
    HOST_TO_REMOTE,
    REMOTE_TO_DEVICE,
};

struct EvictionMove {
    TreeNode*                 node{nullptr};
    int                       component_group_id{-1};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;
};

class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    int              component_group_id{-1};
    CacheGroupType   group_type{CacheGroupType::FULL};
    std::vector<int> component_indices;
    size_t           host_block_size{0};

    std::unique_ptr<EvictionHeap> device_heap;
    std::unique_ptr<EvictionHeap> host_heap;
    std::unique_ptr<EvictionHeap> disk_heap;

    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot);

    virtual void                        evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);
    virtual std::optional<EvictionMove> driveEviction(int num_blocks, Tier tier);

    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    virtual void tryAddToDeviceHeap(TreeNode* node) = 0;

    virtual void tryAddToHostHeap(TreeNode* node);
    virtual void tryAddToDiskHeap(TreeNode* node);

    bool isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const;

    virtual size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const = 0;

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

    size_t devicePoolCount() const {
        return device_pools_.size();
    }

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
                device_pools_[i]->decRef(device_blocks[i]);
            }
        }
    }

    GroupBlockSet allocateBlocks(Tier tier, size_t count);
    void          referenceBlocks(const GroupBlockSet& set) const;
    void          unreferenceBlocks(const GroupBlockSet& set) const;

    BlockIdxType allocateSingleBlock(Tier tier);
    void         releaseSingleBlock(Tier tier, BlockIdxType block) const;

    std::vector<BlockIdxType> getBlocks(const GroupSlot& slot, Tier tier) const;
    void                      setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks);

    bool isSlotEvictable(const GroupSlot& slot, Tier tier) const;

    void tryAddToHeap(TreeNode* node, Tier tier);
    void invalidateHeap(TreeNode* node, Tier tier);
    void clearHeapFlag(GroupSlot& slot, Tier tier);

protected:
    std::vector<DeviceBlockPoolPtr> device_pools_;
    std::shared_ptr<HostBlockPool>  host_pool_;
    std::shared_ptr<DiskBlockPool>  disk_pool_;

};

using ComponentGroupPtr = std::shared_ptr<ComponentGroup>;

}  // namespace rtp_llm
