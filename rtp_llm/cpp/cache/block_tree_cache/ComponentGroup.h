#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class LoadBackTicket;

struct Component {
    int                 component_id{-1};
    int                 component_group_id{-1};
    CacheGroupType      type{CacheGroupType::FULL};
    std::string         tag;
    std::vector<int>    model_layer_ids;
    std::vector<size_t> layer_bytes;

    size_t layerCount() const {
        return layer_bytes.size();
    }
    size_t layerBytes(size_t layer_idx) const {
        return layer_bytes[layer_idx];
    }
};

struct GroupBlockSet {
    int                                    component_group_id{-1};
    Tier                                   tier{Tier::DEVICE};
    std::vector<std::vector<BlockIdxType>> per_node;
    // Optional: tree nodes aligned with per_node, populated for match-protection
    // sets so release can drive candidate refresh. Empty when not needed.
    std::vector<TreeNode*> nodes;
};

struct BlockTreeMatchResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};

    // Keyed by stable declarative topology tag.
    std::unordered_map<std::string, BlockIndicesType> group_block_indices;

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
    int64_t                   source_tier_enter_time_us{0};
};

class ComponentGroupLayout {
public:
    struct Slice {
        size_t component_idx{0};
        size_t layer_idx{0};  // Component-local, not model-global.
        size_t offset_bytes{0};
    };

    static std::optional<ComponentGroupLayout> create(const std::vector<std::vector<size_t>>& component_layer_bytes);

    const std::vector<Slice>& slices() const {
        return slices_;
    }
    // Slices are emitted in canonical component order, so the last slice's
    // component_idx + 1 is the component count. Not stored separately.
    size_t componentCount() const {
        return slices_.empty() ? 0 : slices_.back().component_idx + 1;
    }
    size_t payloadBytes() const {
        return payload_bytes_;
    }

private:
    std::vector<Slice> slices_;
    size_t             payload_bytes_{0};
};

class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    int              component_group_id{-1};
    CacheGroupType   group_type{CacheGroupType::FULL};
    CacheEvictPolicy evict_policy{CacheEvictPolicy::CHAIN};

    bool finalizeLayout(std::vector<int> component_indices, const std::vector<Component>& components);

    const std::vector<int>& componentIndices() const {
        return component_indices_;
    }

    bool hasLayout() const {
        return layout_.has_value();
    }
    const ComponentGroupLayout& layout() const;

    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual void evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);

    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    bool isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const;

    virtual size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const = 0;

    void setDevicePools(std::vector<DeviceBlockPoolPtr> pools, std::vector<std::string> tags = {});
    const std::vector<std::string>& tags() const {
        return tags_;
    }
    void setHostPool(std::shared_ptr<HostBlockPool> pool) {
        host_pool_ = std::move(pool);
    }
    void setDiskPool(std::shared_ptr<BlockTreeDiskBlockPool> pool) {
        disk_pool_ = std::move(pool);
    }

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    bool hasCompleteDeviceValue(const GroupSlot& slot) const;
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<BlockTreeDiskBlockPool> diskPool() const {
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

    // A tree-node eviction releases one block from every physical device pool
    // in this component group. Return the exact number of node evictions needed
    // so every pool retains at least min_free_blocks (clamped to its capacity).
    size_t devicePoolMaxExcessForMinFree(size_t min_free_blocks) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            if (!pool) {
                continue;
            }
            const size_t capacity = pool->totalBlocksNum();
            if (capacity == 0) {
                continue;
            }
            const size_t used      = capacity - pool->freeBlocksNum();
            const size_t min_free  = std::min(min_free_blocks, capacity);
            const size_t threshold = capacity - min_free;
            if (used > threshold) {
                max_excess = std::max(max_excess, used - threshold);
            }
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

    GroupBlockSet allocateBlocks(Tier tier, size_t count, BlockRefType ref_type);
    void          referenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const;
    void          unreferenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const;

    BlockIdxType allocateSingleBlock(Tier tier, BlockRefType ref_type);
    void         releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const;

    std::vector<BlockIdxType> getBlocks(const GroupSlot& slot, Tier tier) const;
    void                      setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks);
    // Highest tier (DEVICE > HOST > DISK) holding this slot's data, else NONE.
    Tier getTopTier(const GroupSlot& slot) const;

    virtual bool isSlotEvictable(const TreeNode& node, Tier tier) const;

protected:
    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::vector<std::string>                tags_;
    std::shared_ptr<HostBlockPool>          host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool_;

private:
    std::vector<int>                    component_indices_;
    std::optional<ComponentGroupLayout> layout_;
};

using ComponentGroupPtr = std::shared_ptr<ComponentGroup>;

}  // namespace rtp_llm
