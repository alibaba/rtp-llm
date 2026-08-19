#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct TreeNode;

struct MultiNodeResource {
    size_t                                                       group_set_id{0};
    Tier                                                         tier{Tier::DEVICE};
    std::vector<std::pair<TreeNode*, std::vector<BlockIdxType>>> node_blocks;
};

class MatchValidator {
public:
    virtual ~MatchValidator() = default;
    virtual bool validate(const GroupSetResource& resource) = 0;
};

class GroupSet {
public:
    GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
             std::shared_ptr<HostBlockPool>  host_pool,
             BlockTreeDiskBlockPoolPtr       disk_pool);

    virtual ~GroupSet() = default;

    void initialize(size_t                               group_set_id,
                    std::shared_ptr<const CacheTopology> topology,
                    std::vector<size_t>                  group_ids);

    size_t groupSetId() const {
        return group_set_id_;
    }
    const std::vector<size_t>& groupIds() const {
        return group_ids_;
    }
    std::shared_ptr<const CacheTopology> topologyPtr() const {
        return topology_;
    }
    const GroupBase& groupAt(size_t member_group_id) const {
        return topology_->groupById(group_ids_[member_group_id]);
    }
    size_t payloadBytes() const {
        return payload_bytes_;
    }
    CacheGroupType groupType() const {
        return groupAt(0).policy.group_type;
    }
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual size_t computeReuseBlockCount(size_t matched_block_count) const = 0;

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<BlockTreeDiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    bool hasAllocatedDeviceBlocks(const std::vector<BlockIdxType>& blocks) const;

    void      mapDeviceBlocksToTreeNode(const MultiNodeResource& resource);
    void      unmapDeviceBlocksFromTreeNode(const MultiNodeResource& resource);
    TreeNode* findTreeNodeByDeviceBlock(size_t member_group_id, BlockIdxType block_id) const;
    bool      areBlockToNodeMapsEmpty() const;

    void referenceBlocks(const MultiNodeResource& resource, BlockRefType ref_type) const;
    void unreferenceBlocks(const MultiNodeResource& resource, BlockRefType ref_type) const;

    BlockIdxType allocateSingleBlock(Tier tier, BlockRefType ref_type);
    void         releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const;

private:
    using DeviceBlockToTreeNodeMap = std::unordered_map<BlockIdxType, TreeNode*>;

    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::shared_ptr<HostBlockPool>          host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool_;
    std::vector<DeviceBlockToTreeNodeMap>   block_to_node_maps_;
    size_t                                  group_set_id_{0};
    std::shared_ptr<const CacheTopology>    topology_;
    std::vector<size_t>                     group_ids_;
    size_t                                  payload_bytes_{0};
};

using GroupSetPtr = std::shared_ptr<GroupSet>;

}  // namespace rtp_llm
