#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

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
                          const GroupSetPools&                    pools,
                          Operation&&                            operation) {
    if (resource.tier == Tier::DEVICE) {
        const auto blocks_by_pool = collectDeviceBlocks(resource, pools.device.size());
        for (size_t pool_index = 0; pool_index < pools.device.size(); ++pool_index) {
            operation(*pools.device[pool_index], blocks_by_pool[pool_index]);
        }
    } else if (resource.tier == Tier::HOST && pools.host) {
        operation(*pools.host, collectBlocks(resource));
    } else if (resource.tier == Tier::DISK && pools.disk) {
        operation(*pools.disk, collectBlocks(resource));
    }
}

}  // namespace

GroupSet::GroupSet(GroupSetPools pools): pools_(std::move(pools)) {}

void GroupSet::initialize(size_t group_set_id, GroupSetGeometry geometry, std::vector<std::string> group_tags,
                          BackingLayout backing_layout) {
    if (group_tags.empty() || !geometry.page_tokens || (!pools_.device.empty() && pools_.device.size() != group_tags.size())) {
        throw std::invalid_argument("GroupSet requires a geometry and one tag per device member");
    }
    std::unordered_set<std::string> seen_tags;
    for (size_t member_index = 0; member_index < group_tags.size(); ++member_index) {
        const auto& tag = group_tags[member_index];
        if (tag.empty() || !seen_tags.emplace(tag).second || (!pools_.device.empty() && !pools_.device[member_index])) {
            throw std::invalid_argument("GroupSet has an invalid or duplicated device member: " + tag);
        }
    }
    size_t physical_payload_bytes   = backing_layout.payload_bytes;
    uses_physical_payload_geometry_ = physical_payload_bytes > 0;
    if (physical_payload_bytes == 0) {
        for (const auto& pool : pools_.device) {
            const size_t group_bytes = pool->blockSizeBytes();
            if (group_bytes > std::numeric_limits<size_t>::max() - physical_payload_bytes) {
                throw std::overflow_error("GroupSet payload exceeds the size limit");
            }
            physical_payload_bytes += group_bytes;
        }
    }
    if (!physical_payload_bytes && (backing_layout.crc_enabled || backing_layout.storage_bytes || backing_layout.pool_stride)) {
        throw std::invalid_argument("GroupSet backing layout requires a positive physical payload");
    }

    group_set_id_  = group_set_id;
    geometry_      = geometry;
    group_tags_    = std::move(group_tags);
    payload_bytes_ = physical_payload_bytes;
    enable_crc_    = backing_layout.crc_enabled;
    storage_bytes_ = backing_layout.storage_bytes == 0 ? physical_payload_bytes : backing_layout.storage_bytes;
    storage_stride_bytes_ = backing_layout.pool_stride == 0 ? storage_bytes_ : backing_layout.pool_stride;
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    applyToResourcePools(resource, pools_, [](IBlockPool& pool, const BlockIdList& blocks) {
        pool.referenceRequestBlocks(blocks);
    });
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    applyToResourcePools(resource, pools_, [](IBlockPool& pool, const BlockIdList& blocks) {
        pool.unreferenceRequestBlocks(blocks);
    });
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, pools_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.incTreeRef(blocks, ref_type);
        });
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, pools_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.decTreeRef(blocks, ref_type);
        });
}

bool GroupSet::canPublishDevice(const BlockIdList& blocks) const {
    if (blocks.size() != pools_.device.size()) {
        return false;
    }
    for (size_t member = 0; member < blocks.size(); ++member) {
        if (!pools_.device[member]->canPublish(blocks[member])) {
            return false;
        }
    }
    return true;
}

BlockIdxType GroupSet::allocateSingleBlock(Tier tier, BlockTreeRefType ref_type) {
    auto blocks = allocateBlocks(1, tier, ref_type);
    return blocks.has_value() ? blocks->front() : NULL_BLOCK_IDX;
}

std::optional<BlockIdList> GroupSet::allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type) {
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = pools_.host.get();
    } else if (tier == Tier::DISK) {
        pool = pools_.disk.get();
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
        if (pools_.host)
            pools_.host->decTreeRef(blocks, ref_type);
    } else if (tier == Tier::DISK) {
        if (pools_.disk)
            pools_.disk->decTreeRef(blocks, ref_type);
    }
}

void GroupSet::releaseSingleBlock(Tier tier, BlockIdxType block, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (pools_.host)
            pools_.host->decTreeRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (pools_.disk)
            pools_.disk->decTreeRef(block, ref_type);
    }
}

}  // namespace rtp_llm
