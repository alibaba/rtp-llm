#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class LoadTicket;

struct MultiNodeResource {
    size_t                                 group_set_id{0};
    Tier                                   tier{Tier::DEVICE};
    std::vector<std::vector<BlockIdxType>> per_node;
    // Optional: tree nodes aligned with per_node, populated for match-protection
    // sets so release can drive candidate refresh. Empty when not needed.
    std::vector<TreeNode*> tree_nodes;
};

struct BlockTreeMatchResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};

    std::vector<MultiNodeResource> matched_resources;

    std::shared_ptr<AsyncContext> async_context;
    size_t                        load_blocks{0};
    size_t                        host_load_blocks{0};
    size_t                        disk_load_blocks{0};
    size_t                        remote_load_blocks{0};

    std::shared_ptr<LoadTicket> load_ticket;
};

class MatchValidator {
public:
    virtual ~MatchValidator() = default;
    virtual bool validate(const GroupSetResource& resource) = 0;
};

struct EvictionMove {
    TreeNode*                 node{nullptr};
    size_t                    group_set_id{0};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;
    int64_t                   source_tier_enter_time_us{0};
};

class GroupSet {
public:
    virtual ~GroupSet() = default;

    void initialize(size_t                               group_set_id,
                    std::shared_ptr<const CacheTopology> topology,
                    std::vector<size_t>                  group_ids,
                    std::vector<DeviceBlockPoolPtr>      device_pools);

    size_t                                      groupSetId() const;
    const std::shared_ptr<const CacheTopology>& topology() const {
        return topology_;
    }
    const std::vector<size_t>& groupIds() const {
        return group_ids_;
    }
    const GroupBase& groupAt(size_t member_index) const;
    size_t           payloadBytes() const {
        return payload_bytes_;
    }
    CacheGroupType groupType() const {
        return groupAt(0).policy.group_type;
    }
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual void evictFromTier(GroupSetResource& resource, Tier tier);

    virtual size_t computeReuseBlockCount(size_t matched_block_count) const = 0;

    void setHostPool(std::shared_ptr<HostBlockPool> pool) {
        host_pool_ = std::move(pool);
    }
    void setDiskPool(std::shared_ptr<BlockTreeDiskBlockPool> pool) {
        disk_pool_ = std::move(pool);
    }

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    bool                           hasCompleteDeviceValue(const GroupSetResource& resource) const;
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<BlockTreeDiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    bool hasAllocatedDeviceBlocks(const std::vector<BlockIdxType>& blocks) const;

    bool anyDevicePoolExceedsRatio(double ratio) const {
        for (const auto& pool : device_pools_) {
            size_t capacity  = pool->totalBlocksNum();
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold) {
                return true;
            }
        }
        return false;
    }

    size_t devicePoolMaxExcess(double ratio) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            size_t capacity  = pool->totalBlocksNum();
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold) {
                max_excess = std::max(max_excess, used - threshold);
            }
        }
        return max_excess;
    }

    // A tree-node eviction releases one block from every physical device pool
    // in this group set. Return the exact number of node evictions needed
    // so every pool retains at least min_free_blocks (clamped to its capacity).
    size_t devicePoolMaxExcessForMinFree(size_t min_free_blocks) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            const size_t capacity  = pool->totalBlocksNum();
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

    MultiNodeResource allocateBlocks(Tier tier, size_t count, BlockRefType ref_type);
    void              referenceBlocks(const MultiNodeResource& set, BlockRefType ref_type) const;
    void              unreferenceBlocks(const MultiNodeResource& set, BlockRefType ref_type) const;

    BlockIdxType allocateSingleBlock(Tier tier, BlockRefType ref_type);
    void         releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const;

    std::vector<BlockIdxType> getBlocks(const GroupSetResource& resource, Tier tier) const;
    void                      setBlocks(GroupSetResource& resource, Tier tier, const std::vector<BlockIdxType>& blocks);
    // Highest tier (DEVICE > HOST > DISK) holding this resource's data, else NONE.
    Tier getTopTier(const GroupSetResource& resource) const;

    virtual bool isEvictable(const GroupSetResource& resource, Tier tier) const;

protected:
    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::shared_ptr<HostBlockPool>          host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool_;

private:
    void validateBlockResource(const MultiNodeResource& resource) const;

    size_t                               group_set_id_{0};
    std::shared_ptr<const CacheTopology> topology_;
    std::vector<size_t>                  group_ids_;
    size_t                               payload_bytes_{0};
};

using GroupSetPtr = std::shared_ptr<GroupSet>;

}  // namespace rtp_llm
