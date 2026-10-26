#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

#include <algorithm>
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
                          BackingLayout                        backing_layout,
                          bool                                 local_only) {
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
    local_only_    = local_only;
    payload_bytes_ = physical_payload_bytes;
    enable_crc_    = backing_layout.crc_enabled;
    storage_bytes_ = backing_layout.storage_bytes == 0 ? physical_payload_bytes : backing_layout.storage_bytes;
}

std::vector<size_t> GroupSet::requiredPathIndices(const std::vector<TreeNode*>& path, size_t token_end) const {
    std::vector<size_t> indices;
    if (path.empty() || token_end == 0) {
        return indices;
    }
    const GroupBase& cache_group = group(groupTags().front());
    size_t           start       = 0;
    if (cache_group.policy.group_type == CacheGroupType::LINEAR) {
        start = token_end - 1;
    } else if (cache_group.policy.group_type == CacheGroupType::SWA) {
        const size_t window = static_cast<size_t>(std::max(0, cache_group.policy.sliding_window_size));
        if (local_only_) {
            RTP_LLM_CHECK_WITH_INFO(window > 0, "local HCA requires a positive compression window");
            const size_t pending = token_end % window;
            if (pending == 0) {
                return indices;
            }
            start = token_end - pending;
        } else if (window > 1 && token_end >= window) {
            // A continuation at G reads the previous window-1 tokens.
            start = token_end - (window - 1);
        }
    }
    for (size_t i = 0; i < path.size(); ++i) {
        const size_t node_start = path[i]->parent == nullptr ? 0 : path[i]->parent->token_end;
        if (path[i]->token_end > start && node_start < token_end) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool GroupSet::hasZeroStateDemand(size_t token_end) const {
    if (!local_only_ || groupType() != CacheGroupType::SWA) {
        return false;
    }
    const int window = group(group_tags_.front()).policy.sliding_window_size;
    return window > 0 && token_end % static_cast<size_t>(window) == 0;
}

bool GroupSet::requiresExactStateEndpoint() const {
    if (groupType() == CacheGroupType::LINEAR || local_only_) {
        return true;
    }
    if (groupType() != CacheGroupType::SWA) {
        return false;
    }
    for (const auto& tag : group_tags_) {
        const auto& member = group(tag);
        if (member.spec->type == KVCacheSpecType::OpaqueState) {
            const auto* state = dynamic_cast<const FixedStateCacheSpec*>(member.spec.get());
            // The packed-byte SWA ABI stores token-addressed KV rows. Entry
            // projected compressor state (including CP1) stores accumulator
            // history, whose entry count is not a token KV coverage window.
            if (state == nullptr || member.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
                return true;
            }
        }
    }
    return false;
}

bool GroupSet::hasDeviceCoverage(const std::vector<TreeNode*>& path, size_t token_end) const {
    if (!hasPhysicalCoverage(path, token_end)) {
        return false;
    }
    for (const size_t index : requiredPathIndices(path, token_end)) {
        const auto& resource = path[index]->group_set_resources[group_set_id_];
        if (!resource.isMatchUsable() || !resource.hasCompleteDeviceValue()
            || resource.device_blocks.size() != group_tags_.size()) {
            return false;
        }
    }
    return true;
}

bool GroupSet::hasPhysicalCoverage(const std::vector<TreeNode*>& path, size_t token_end) const {
    if (path.empty() || token_end == 0 || token_end > path.back()->token_end) {
        return false;
    }
    if (hasZeroStateDemand(token_end)) {
        return true;
    }
    if (requiresExactStateEndpoint() && token_end != path.back()->token_end) {
        return false;
    }
    for (const size_t index : requiredPathIndices(path, token_end)) {
        const auto* node = path[index];
        if (!coversTokenInterval(node->parent == nullptr ? 0 : node->parent->token_end, node->token_end, token_end)) {
            return false;
        }
    }
    return true;
}

bool GroupSet::hasCanonicalCoverage(size_t count, size_t canonical_size_per_block) const {
    RTP_LLM_CHECK(canonical_size_per_block > 0
                  && count <= std::numeric_limits<size_t>::max() / canonical_size_per_block);
    if (groupType() != CacheGroupType::SWA || local_only_ || count == 0) {
        return true;
    }
    const size_t token_end = count * canonical_size_per_block;
    const size_t window    = static_cast<size_t>(std::max(0, group(group_tags_.front()).policy.sliding_window_size));
    const size_t start     = window > 1 && token_end >= window ? token_end - (window - 1) : 0;
    for (size_t index = start / canonical_size_per_block; index < count; ++index) {
        if (!coversTokenInterval(index * canonical_size_per_block, (index + 1) * canonical_size_per_block, token_end)) {
            return false;
        }
    }
    return true;
}

bool GroupSet::coversTokenInterval(size_t node_start, size_t node_end, size_t token_end) const {
    if (groupType() != CacheGroupType::SWA || local_only_) {
        return true;
    }
    for (const auto& tag : group_tags_) {
        const auto& member = group(tag);
        if (member.spec->type == KVCacheSpecType::OpaqueState) {
            const auto* state = dynamic_cast<const FixedStateCacheSpec*>(member.spec.get());
            if (state == nullptr || member.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
                // Exactness applies to the selected endpoint, not each
                // ancestor. Keep required ancestor resources for overlap
                // across physical blocks; do not reinterpret their small
                // compressor ring as the model's SWA token window.
                continue;
            }
        }
        const size_t window         = static_cast<size_t>(std::max(0, member.policy.sliding_window_size));
        const size_t window_start   = window > 1 && token_end >= window ? token_end - (window - 1) : 0;
        const size_t required_start = std::max(window_start, node_start);
        // A canonical CP node can span more than one nonsharded SWA
        // physical slot. Its single resource only owns the last slot.
        // BLOCK_ROUND_ROBIN IDs instead collectively represent all ranks'
        // portions of that canonical interval.
        const size_t span = member.seqSizePerBlock();
        if (span == 0
            || (member.policy.cp_mapping != CpBlockMappingMode::BLOCK_ROUND_ROBIN
                && required_start < ((node_end - 1) / span) * span)) {
            return false;
        }
        if (member.spec->type != KVCacheSpecType::OpaqueState) {
            continue;
        }
        const auto* state = dynamic_cast<const FixedStateCacheSpec*>(member.spec.get());
        if (state == nullptr || member.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            return false;
        }
        // Packed-byte CP slices collectively retain the complete token KV
        // ring. Compressor entry-projected states were handled above.
        const size_t entries        = state->entriesPerBlock();
        const size_t retained_start = node_end > entries ? node_end - entries : 0;
        if (entries == 0 || required_start < retained_start) {
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
