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

bool groupSetSemanticsCompatible(const GroupBase& lhs, const GroupBase& rhs) {
    if (lhs.policy.group_type != rhs.policy.group_type || lhs.seqSizePerBlock() != rhs.seqSizePerBlock()
        || lhs.policy.cp_mapping != rhs.policy.cp_mapping) {
        return false;
    }
    if (lhs.policy.group_type == CacheGroupType::SWA
        && lhs.policy.sliding_window_size != rhs.policy.sliding_window_size) {
        return false;
    }
    if ((lhs.policy.group_type == CacheGroupType::SWA || lhs.policy.group_type == CacheGroupType::LINEAR)
        && lhs.policy.active_tail_blocks != rhs.policy.active_tail_blocks) {
        return false;
    }
    return true;
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

void GroupSet::initialize(size_t                             group_set_id,
                          std::shared_ptr<const CacheConfig> cache_config,
                          std::vector<std::string>           group_tags,
                          BackingLayout                      backing_layout) {
    RTP_LLM_CHECK_WITH_INFO(cache_config != nullptr, "GroupSet requires a cache config");
    RTP_LLM_CHECK_WITH_INFO(!group_tags.empty(), "GroupSet requires at least one member tag");
    RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() || device_pools_.size() == group_tags.size(),
                            "GroupSet tag/pool count mismatch: tags=%zu pools=%zu",
                            group_tags.size(),
                            device_pools_.size());

    std::unordered_set<std::string> seen_tags;
    const GroupBase*                first_group = nullptr;
    for (size_t member_index = 0; member_index < group_tags.size(); ++member_index) {
        const auto& tag = group_tags[member_index];
        RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second, "GroupSet has duplicate member tag=%s", tag.c_str());
        const auto& group = cache_config->groupByTagGlobal(tag);
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "GroupSet member tag=%s has null cache spec", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            !cache_config->moduleConfig(cache_config->moduleOfTag(tag)).layerIdsForGroup(tag).empty(),
            "GroupSet member tag=%s has no local layers",
            tag.c_str());
        if (first_group == nullptr) {
            first_group = &group;
        } else {
            RTP_LLM_CHECK_WITH_INFO(groupSetSemanticsCompatible(*first_group, group),
                                    "GroupSet has incompatible reuse geometry for tags=%s and %s",
                                    group_tags.front().c_str(),
                                    tag.c_str());
        }
        if (!device_pools_.empty()) {
            RTP_LLM_CHECK_WITH_INFO(
                device_pools_[member_index] != nullptr, "GroupSet has null device pool for tag=%s", tag.c_str());
        }
    }

    size_t physical_payload_bytes = backing_layout.payload_bytes;
    if (physical_payload_bytes == 0) {
        for (const auto& tag : group_tags) {
            const size_t group_bytes =
                cache_config->moduleConfig(cache_config->moduleOfTag(tag)).blockSizeBytesForGroup(tag);
            RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - physical_payload_bytes,
                                    "GroupSet payload size overflow at tag=%s",
                                    tag.c_str());
            physical_payload_bytes += group_bytes;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(physical_payload_bytes > 0, "GroupSet physical payload must be positive");

    group_set_id_  = group_set_id;
    cache_config_  = std::move(cache_config);
    group_tags_    = std::move(group_tags);
    payload_bytes_ = physical_payload_bytes;
    enable_crc_    = backing_layout.crc_enabled;
    storage_bytes_ = backing_layout.storage_bytes == 0 ? physical_payload_bytes : backing_layout.storage_bytes;
    uses_physical_payload_geometry_ = backing_layout.payload_bytes > 0;
}

const KVCacheSpec& GroupSet::groupSpec(std::string_view tag) const {
    const auto& member = group(tag);
    RTP_LLM_CHECK_WITH_INFO(member.spec != nullptr,
                            "GroupSet member tag=%.*s has null cache spec",
                            static_cast<int>(tag.size()),
                            tag.data());
    return *member.spec;
}

std::vector<int> GroupSet::localLayerIdsForMember(std::string_view tag) const {
    return cache_config_->moduleConfig(cache_config_->moduleOfTag(tag)).layerIdsForGroup(tag);
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
