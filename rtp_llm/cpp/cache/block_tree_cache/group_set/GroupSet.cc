#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

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
    device_pools_(std::move(device_pools)), host_pool_(std::move(host_pool)), disk_pool_(std::move(disk_pool)) {}

void GroupSet::initialize(size_t                               group_set_id,
                          std::shared_ptr<const CacheTopology> topology,
                          std::vector<std::string>             group_tags,
                          BackingLayout                        backing_layout) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "GroupSet requires a topology");
    RTP_LLM_CHECK_WITH_INFO(!group_tags.empty(), "GroupSet requires at least one member tag");
    RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() || device_pools_.size() == group_tags.size(),
                            "GroupSet tag/pool count mismatch: tags=%zu pools=%zu",
                            group_tags.size(),
                            device_pools_.size());
    std::vector<ModuleGroupKey> keys;
    for (auto& tag : group_tags) {
        keys.push_back({CacheModuleId::main(), std::move(tag)});
    }
    initializeMembers(
        group_set_id, {{CacheModuleId::main(), std::move(topology)}}, std::move(keys), backing_layout);
}

void GroupSet::initialize(size_t                      group_set_id,
                          const CacheDomainConfig&    domain,
                          std::vector<ModuleGroupKey> group_keys,
                          BackingLayout               backing_layout) {
    std::unordered_map<CacheModuleId, std::shared_ptr<const CacheTopology>> topologies;
    for (const auto& key : group_keys) {
        (void)domain.group(key);
        RTP_LLM_CHECK_WITH_INFO(domain.isActive(key.module_id),
                                "inactive cache module=%zu cannot enter a reusable GroupSet",
                                key.module_id.value());
        topologies.emplace(key.module_id, domain.module(key.module_id).topologyPtr());
    }
    initializeMembers(group_set_id, std::move(topologies), std::move(group_keys), backing_layout);
}

void GroupSet::initializeMembers(size_t                                                                  group_set_id,
                                 std::unordered_map<CacheModuleId, std::shared_ptr<const CacheTopology>> topologies,
                                 std::vector<ModuleGroupKey>                                             group_keys,
                                 BackingLayout                                                           backing_layout) {
    RTP_LLM_CHECK_WITH_INFO(!group_keys.empty(), "GroupSet requires at least one member");
    RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() || device_pools_.size() == group_keys.size(),
                            "GroupSet member/pool count mismatch: members=%zu pools=%zu",
                            group_keys.size(),
                            device_pools_.size());
    std::unordered_set<ModuleGroupKey> seen;
    std::vector<std::string>           main_tags;
    for (size_t index = 0; index < group_keys.size(); ++index) {
        const auto& key = group_keys[index];
        RTP_LLM_CHECK_WITH_INFO(seen.emplace(key).second, "GroupSet duplicate member %s", key.debugString().c_str());
        const auto it = topologies.find(key.module_id);
        RTP_LLM_CHECK_WITH_INFO(
            it != topologies.end() && it->second, "GroupSet missing topology for %s", key.debugString().c_str());
        (void)it->second->group(key.tag);
        RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() || device_pools_[index],
                                "GroupSet null device pool for %s",
                                key.debugString().c_str());
        if (key.module_id.isMain()) {
            main_tags.push_back(key.tag);
        }
    }
    size_t physical_payload_bytes   = backing_layout.payload_bytes;
    const bool uses_physical_payload = physical_payload_bytes > 0;
    if (!uses_physical_payload) {
        for (const auto& key : group_keys) {
            const size_t bytes = topologies.at(key.module_id)->blockSizeBytesForGroup(key.tag);
            RTP_LLM_CHECK_WITH_INFO(bytes <= std::numeric_limits<size_t>::max() - physical_payload_bytes,
                                    "GroupSet payload size overflow at %s",
                                    key.debugString().c_str());
            physical_payload_bytes += bytes;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(physical_payload_bytes > 0, "GroupSet physical payload must be positive");
    group_set_id_                   = group_set_id;
    topologies_                     = std::move(topologies);
    group_keys_                     = std::move(group_keys);
    group_tags_                     = std::move(main_tags);
    payload_bytes_ = physical_payload_bytes;
    enable_crc_                     = backing_layout.crc_enabled;
    storage_bytes_                  = backing_layout.storage_bytes == 0 ? physical_payload_bytes :
                                                                          backing_layout.storage_bytes;
    uses_physical_payload_geometry_ = uses_physical_payload;
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
    auto blocks = allocateBlocks(1, tier, ref_type);
    return blocks.has_value() ? blocks->front() : NULL_BLOCK_IDX;
}

std::optional<BlockIdList> GroupSet::allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type) {
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = host_pool_.get();
    } else if (tier == Tier::DISK) {
        pool = disk_pool_.get();
    }
    if (!pool)
        return std::nullopt;
    auto blocks = pool->malloc(n);
    if (!blocks.has_value())
        return std::nullopt;
    pool->incTreeRef(*blocks, ref_type);
    return blocks;
}

void GroupSet::releaseBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decTreeRef(blocks, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decTreeRef(blocks, ref_type);
    }
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
