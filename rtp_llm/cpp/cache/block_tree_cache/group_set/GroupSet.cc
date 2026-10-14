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
    if (topology_ != nullptr && topology_ == topology && group_tags_ == group_tags
        && uses_physical_payload_geometry_ == (backing_layout.payload_bytes > 0)
        && (backing_layout.payload_bytes == 0 || backing_layout.payload_bytes == payload_bytes_)
        && enable_crc_ == backing_layout.crc_enabled
        && storage_bytes_ == (backing_layout.storage_bytes == 0 ? payload_bytes_ : backing_layout.storage_bytes)) {
        group_set_id_ = group_set_id;
        return;
    }
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

    group_set_id_      = group_set_id;
    topology_          = std::move(topology);
    group_tags_        = std::move(group_tags);
    payload_bytes_     = physical_payload_bytes;
    enable_crc_        = backing_layout.crc_enabled;
    storage_bytes_     = backing_layout.storage_bytes == 0 ? physical_payload_bytes : backing_layout.storage_bytes;
    copy_3d_templates_ = initializeCopy3DTemplates();
}

std::vector<DeviceHostCopyTemplate> GroupSet::initializeCopy3DTemplates() const {
    std::vector<DeviceHostCopyTemplate> templates;
    if (topology_ == nullptr || group_tags_.empty() || group_tags_.size() != device_pools_.size()) {
        return {};
    }
    size_t host_offset  = 0;
    int    device_index = -1;
    for (size_t member = 0; member < device_pools_.size(); ++member) {
        const auto& pool       = device_pools_[member];
        const auto& group_base = group(group_tags_[member]);
        const auto  layer_ids  = topology_->layerIdsForGroup(group_tags_[member]);
        if (!pool || !pool->getBaseAddress() || pool->deviceIndex() < 0 || group_base.kvScaleStrideBytes() != 0) {
            return {};
        }
        if (device_index < 0) {
            device_index = pool->deviceIndex();
        } else if (device_index != pool->deviceIndex()) {
            return {};
        }
        const size_t width    = group_base.kvBlockStrideBytes();
        const size_t capacity = pool->getTotalSizeBytes();
        if (reinterpret_cast<uintptr_t>(pool->getBaseAddress()) > UINTPTR_MAX - capacity) {
            return {};
        }
        size_t layer_begin = 0;
        for (const auto& layout : pool->memoryLayouts()) {
            if (width == 0 || layout.block_num == 0 || layout.layer_num == 0 || layout.kv_block_stride_bytes != width
                || layout.kv_scale_stride_bytes != 0 || layout.hasScale()
                || layout.block_num > std::numeric_limits<size_t>::max() / width || layer_begin > layer_ids.size()
                || layout.layer_num > layer_ids.size() - layer_begin) {
                return {};
            }
            const size_t pitch  = layout.block_num * width;
            const size_t offset = layout.kv_cache_offset_bytes;
            if (offset > capacity || layout.layer_num > (capacity - offset) / pitch
                || layout.layer_num > layout.kv_block_pool_size_bytes / pitch || host_offset > payload_bytes_
                || layout.layer_num > (payload_bytes_ - host_offset) / width) {
                return {};
            }
            auto* base = static_cast<uint8_t*>(pool->getBaseAddress()) + offset;
            templates.push_back({member, base, host_offset, width, layout.layer_num, pitch});
            host_offset += width * layout.layer_num;
            layer_begin += layout.layer_num;
        }
        if (layer_begin != layer_ids.size()) {
            return {};
        }
    }
    if (host_offset != payload_bytes_) {
        return {};
    }
    return templates;
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
