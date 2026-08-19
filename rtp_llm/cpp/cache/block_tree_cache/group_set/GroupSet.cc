#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

#include <algorithm>

namespace rtp_llm {

GroupSet::GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
                   std::shared_ptr<HostBlockPool>  host_pool,
                   BlockTreeDiskBlockPoolPtr       disk_pool):
    device_pools_(std::move(device_pools)),
    host_pool_(std::move(host_pool)),
    disk_pool_(std::move(disk_pool)),
    block_to_node_maps_(device_pools_.size()) {}

void GroupSet::initialize(size_t                               group_set_id,
                          std::shared_ptr<const CacheTopology> topology,
                          std::vector<size_t>                  group_ids) {
    size_t payload_bytes = 0;
    for (size_t group_id : group_ids) {
        const auto& group = topology->groupById(group_id);
        payload_bytes += group.layer_ids.size() * (group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
    }

    group_set_id_  = group_set_id;
    topology_      = std::move(topology);
    group_ids_     = std::move(group_ids);
    payload_bytes_ = payload_bytes;
}

bool GroupSet::hasAllocatedDeviceBlocks(const std::vector<BlockIdxType>& blocks) const {
    if (blocks.size() != device_pools_.size()) {
        return false;
    }
    for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
        if (!device_pools_[pool_index]->isAllocated(blocks[pool_index])) {
            return false;
        }
    }
    return true;
}

void GroupSet::mapDeviceBlocksToTreeNode(const MultiNodeResource& resource) {
    for (const auto& [node, blocks] : resource.node_blocks) {
        for (size_t member_group_id = 0; member_group_id < blocks.size(); ++member_group_id) {
            block_to_node_maps_[member_group_id].emplace(blocks[member_group_id], node);
        }
    }
}

void GroupSet::unmapDeviceBlocksFromTreeNode(const MultiNodeResource& resource) {
    for (const auto& [_, blocks] : resource.node_blocks) {
        for (size_t member_group_id = 0; member_group_id < blocks.size(); ++member_group_id) {
            block_to_node_maps_[member_group_id].erase(blocks[member_group_id]);
        }
    }
}

TreeNode* GroupSet::findTreeNodeByDeviceBlock(size_t member_group_id, BlockIdxType block_id) const {
    const DeviceBlockToTreeNodeMap& block_to_node_map = block_to_node_maps_[member_group_id];
    const auto                      it                = block_to_node_map.find(block_id);
    return it == block_to_node_map.end() ? nullptr : it->second;
}

bool GroupSet::areBlockToNodeMapsEmpty() const {
    return std::all_of(block_to_node_maps_.begin(),
                       block_to_node_maps_.end(),
                       [](const DeviceBlockToTreeNodeMap& block_to_node_map) { return block_to_node_map.empty(); });
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource, BlockRefType ref_type) const {
    switch (resource.tier) {
        case Tier::DEVICE:
            for (const auto& [_, blocks] : resource.node_blocks) {
                for (size_t p = 0; p < blocks.size(); ++p) {
                    device_pools_[p]->incRef(blocks[p], ref_type);
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& [_, blocks] : resource.node_blocks)
                    for (auto b : blocks)
                        host_pool_->incRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& [_, blocks] : resource.node_blocks)
                    for (auto b : blocks)
                        disk_pool_->incRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource, BlockRefType ref_type) const {
    switch (resource.tier) {
        case Tier::DEVICE:
            for (const auto& [_, blocks] : resource.node_blocks) {
                for (size_t p = 0; p < blocks.size(); ++p) {
                    device_pools_[p]->decRef(blocks[p], ref_type);
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& [_, blocks] : resource.node_blocks)
                    for (auto b : blocks)
                        host_pool_->decRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& [_, blocks] : resource.node_blocks)
                    for (auto b : blocks)
                        disk_pool_->decRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

BlockIdxType GroupSet::allocateSingleBlock(Tier tier, BlockRefType ref_type) {
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = host_pool_.get();
    } else if (tier == Tier::DISK) {
        pool = disk_pool_.get();
    }
    if (!pool)
        return NULL_BLOCK_IDX;
    auto b = pool->malloc();
    if (!b.has_value())
        return NULL_BLOCK_IDX;
    pool->incRef(*b, ref_type);
    return *b;
}

void GroupSet::releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decRef(block, ref_type);
    }
}

}  // namespace rtp_llm
