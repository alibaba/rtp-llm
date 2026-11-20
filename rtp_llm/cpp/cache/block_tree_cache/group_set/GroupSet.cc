#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

template<typename Operation>
void applyToBlockBatches(const BlockIdList& input, Operation&& operation) {
    if (std::all_of(input.begin(), input.end(), [&](auto block) { return block == input.front(); })) {
        const BlockIdList block{input.front()};
        for (size_t i = 0; i < input.size(); ++i) {
            operation(block);
        }
        return;
    }
    BlockIdList blocks = input;
    std::sort(blocks.begin(), blocks.end());
    if (std::adjacent_find(blocks.begin(), blocks.end()) == blocks.end()) {
        operation(blocks);
        return;
    }
    BlockIdList batch;
    batch.reserve(blocks.size());
    std::vector<std::pair<BlockIdxType, size_t>> repeated;
    for (size_t begin = 0; begin < blocks.size();) {
        size_t end = begin + 1;
        while (end < blocks.size() && blocks[end] == blocks[begin]) {
            ++end;
        }
        batch.push_back(blocks[begin]);
        if (end - begin > 1) {
            repeated.emplace_back(blocks[begin], end - begin - 1);
        }
        begin = end;
    }
    operation(batch);
    while (!repeated.empty()) {
        batch.clear();
        size_t remaining = 0;
        for (auto [block, count] : repeated) {
            batch.push_back(block);
            if (--count > 0) {
                repeated[remaining++] = {block, count};
            }
        }
        operation(batch);
        repeated.resize(remaining);
    }
}

std::vector<BlockIdList> collectDeviceBlocks(const MultiNodeResource& resource, size_t pool_count) {
    std::vector<BlockIdList> blocks_by_pool(pool_count);
    for (auto& blocks : blocks_by_pool) {
        blocks.reserve(resource.node_blocks.size());
    }
    for (const auto& [_, blocks] : resource.node_blocks) {
        for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
            blocks_by_pool[pool_index].push_back(blocks[pool_index]);
        }
    }
    return blocks_by_pool;
}

BlockIdList collectBlocks(const MultiNodeResource& resource) {
    BlockIdList result;
    result.reserve(resource.node_blocks.size());
    for (const auto& [_, blocks] : resource.node_blocks) {
        result.insert(result.end(), blocks.begin(), blocks.end());
    }
    return result;
}

template<typename Operation>
void applyToResourcePools(const MultiNodeResource&               resource,
                          const std::vector<DeviceBlockPoolPtr>& device_pools,
                          const std::shared_ptr<HostBlockPool>&  host_pool,
                          const DiskBlockPoolPtr&                disk_pool,
                          Operation&&                            operation) {
    auto apply_to_pool = [&](auto& pool) {
        if (resource.node_blocks.size() == 1) {
            operation(pool, resource.node_blocks.front().second);
        } else {
            operation(pool, collectBlocks(resource));
        }
    };
    if (resource.tier == Tier::DEVICE) {
        if (device_pools.size() == 1) {
            apply_to_pool(*device_pools.front());
        } else {
            auto blocks_by_pool = collectDeviceBlocks(resource, device_pools.size());
            for (size_t pool_index = 0; pool_index < device_pools.size(); ++pool_index) {
                operation(*device_pools[pool_index], blocks_by_pool[pool_index]);
            }
        }
    } else if (resource.tier == Tier::HOST && host_pool) {
        apply_to_pool(*host_pool);
    } else if (resource.tier == Tier::DISK && disk_pool) {
        apply_to_pool(*disk_pool);
    }
}

template<typename Operation>
void applyToResources(const std::vector<GroupSetPtr>&       group_sets,
                      const std::vector<MultiNodeResource>& resources,
                      Operation&&                           operation) {
    for (const auto& resource : resources) {
        const auto& group = *group_sets[resource.group_set_id];
        applyToResourcePools(
            resource,
            group.devicePools(),
            group.hostPool(),
            group.diskPool(),
            [&](auto& pool, const BlockIdList& blocks) {
                if (std::adjacent_find(blocks.begin(), blocks.end(), std::greater_equal<BlockIdxType>())
                    == blocks.end()) {
                    operation(pool, blocks);
                } else {
                    applyToBlockBatches(blocks, [&](const BlockIdList& batch) { operation(pool, batch); });
                }
            });
    }
}

}  // namespace

GroupSet::GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
                   std::shared_ptr<HostBlockPool>  host_pool,
                   DiskBlockPoolPtr                disk_pool):
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
    std::unordered_set<std::string> seen_tags;
    for (size_t member_index = 0; member_index < group_tags.size(); ++member_index) {
        const auto& tag = group_tags[member_index];
        RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second, "GroupSet has duplicate member tag=%s", tag.c_str());
        (void)topology->group(tag);
        if (!device_pools_.empty()) {
            RTP_LLM_CHECK_WITH_INFO(
                device_pools_[member_index] != nullptr, "GroupSet has null device pool for tag=%s", tag.c_str());
        }
    }
    size_t physical_payload_bytes   = backing_layout.payload_bytes;
    uses_physical_payload_geometry_ = physical_payload_bytes > 0;
    if (physical_payload_bytes == 0) {
        for (const auto& tag : group_tags) {
            const size_t group_bytes = topology->blockSizeBytesForGroup(tag);
            RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - physical_payload_bytes,
                                    "GroupSet payload size overflow at tag=%s",
                                    tag.c_str());
            physical_payload_bytes += group_bytes;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(physical_payload_bytes > 0, "GroupSet physical payload must be positive");

    group_set_id_  = group_set_id;
    topology_      = std::move(topology);
    group_tags_    = std::move(group_tags);
    payload_bytes_ = physical_payload_bytes;
    enable_crc_    = backing_layout.crc_enabled;
    storage_bytes_ = backing_layout.storage_bytes == 0 ? physical_payload_bytes : backing_layout.storage_bytes;
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
    for (size_t pool_index = 0; pool_index < device_pools_.size(); ++pool_index) {
        device_pools_[pool_index]->incRef(blocks_by_pool[pool_index]);
    }
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
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

void GroupSet::unreferenceBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decTreeRef(blocks, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decTreeRef(blocks, ref_type);
    }
}

void appendBlocks(std::vector<MultiNodeResource>& resources,
                  size_t                          group_set_id,
                  Tier                            tier,
                  const BlockIdList&              blocks) {
    auto resource = std::find_if(resources.rbegin(), resources.rend(), [&](const auto& resource) {
        return resource.group_set_id == group_set_id && resource.tier == tier;
    });
    if (resource == resources.rend()) {
        resources.push_back({group_set_id, tier, {}});
        resource = resources.rbegin();
    }
    resource->node_blocks.emplace_back(nullptr, blocks);
}

void referenceBlocks(const std::vector<GroupSetPtr>& group_sets, const std::vector<MultiNodeResource>& resources) {
    applyToResources(group_sets, resources, [](auto& pool, const BlockIdList& blocks) {
        if constexpr (std::is_same_v<std::decay_t<decltype(pool)>, DeviceBlockPool>) {
            pool.incRef(blocks);
        }
    });
}

void unreferenceBlocks(const std::vector<GroupSetPtr>& group_sets, const std::vector<MultiNodeResource>& resources) {
    applyToResources(group_sets, resources, [](auto& pool, const BlockIdList& blocks) {
        if constexpr (std::is_same_v<std::decay_t<decltype(pool)>, DeviceBlockPool>) {
            pool.decRef(blocks);
        }
    });
}

void referenceBlocks(const std::vector<GroupSetPtr>&       group_sets,
                     const std::vector<MultiNodeResource>& resources,
                     BlockTreeRefType                      ref_type) {
    applyToResources(group_sets, resources, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
        pool.incTreeRef(blocks, ref_type);
    });
}

void unreferenceBlocks(const std::vector<GroupSetPtr>&       group_sets,
                       const std::vector<MultiNodeResource>& resources,
                       BlockTreeRefType                      ref_type) {
    applyToResources(group_sets, resources, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
        pool.decTreeRef(blocks, ref_type);
    });
}

}  // namespace rtp_llm
