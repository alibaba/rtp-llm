#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

namespace {

std::vector<BlockIdList> collectDeviceBlocks(const MultiNodeResource& resource, size_t pool_count) {
    std::vector<BlockIdList> blocks_by_pool(pool_count);
    for (const auto& [_, blocks] : resource.node_blocks) {
        for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
            blocks_by_pool[pool_index].push_back(blocks[pool_index]);
        }
    }
    return blocks_by_pool;
}

BlockIdList collectBlocks(const MultiNodeResource& resource) {
    BlockIdList result;
    for (const auto& [_, blocks] : resource.node_blocks) {
        result.insert(result.end(), blocks.begin(), blocks.end());
    }
    return result;
}

template<typename Operation>
void applyToResourcePools(const MultiNodeResource&               resource,
                          const std::vector<DeviceBlockPoolPtr>& device_pools,
                          const std::shared_ptr<HostBlockPool>&  host_pool,
                          const BlockTreeDiskBlockPoolPtr&       disk_pool,
                          Operation&&                            operation) {
    if (resource.tier == Tier::DEVICE) {
        const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools.size());
        for (size_t pool_index = 0; pool_index < device_pools.size(); ++pool_index) {
            operation(*device_pools[pool_index], blocks_by_pool[pool_index]);
        }
    } else if (resource.tier == Tier::HOST && host_pool) {
        operation(*host_pool, collectBlocks(resource));
    } else if (resource.tier == Tier::DISK && disk_pool) {
        operation(*disk_pool, collectBlocks(resource));
    }
}

}  // namespace

GroupSet::GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
                   std::shared_ptr<HostBlockPool>  host_pool,
                   BlockTreeDiskBlockPoolPtr       disk_pool):
    device_pools_(std::move(device_pools)),
    host_pool_(std::move(host_pool)),
    disk_pool_(std::move(disk_pool)) {}

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

void GroupSet::referenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
    for (size_t pool_index = 0; pool_index < device_pools_.size(); ++pool_index) {
        device_pools_[pool_index]->incRef(blocks_by_pool[pool_index]);
    }
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
    for (size_t pool_index = 0; pool_index < device_pools_.size(); ++pool_index) {
        device_pools_[pool_index]->decRef(blocks_by_pool[pool_index]);
    }
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, device_pools_, host_pool_, disk_pool_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.incTreeRef(blocks, ref_type);
        });
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, device_pools_, host_pool_, disk_pool_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.decTreeRef(blocks, ref_type);
        });
}

BlockIdxType GroupSet::allocateSingleBlock(Tier tier, BlockTreeRefType ref_type) {
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
    pool->incTreeRef(*b, ref_type);
    return *b;
}

void GroupSet::releaseSingleBlock(Tier tier, BlockIdxType block, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decTreeRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decTreeRef(block, ref_type);
    }
}

}  // namespace rtp_llm
