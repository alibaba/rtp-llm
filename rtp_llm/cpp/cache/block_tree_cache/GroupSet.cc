#include "rtp_llm/cpp/cache/block_tree_cache/GroupSet.h"

#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void GroupSet::initialize(size_t                               group_set_id,
                          std::shared_ptr<const CacheTopology> topology,
                          std::vector<size_t>                   group_ids,
                          std::vector<DeviceBlockPoolPtr>      device_pools) {
    RTP_LLM_CHECK_WITH_INFO(topology_ == nullptr && group_ids_.empty() && device_pools_.empty(),
                            "GroupSet %zu identity and membership are immutable",
                            group_set_id);
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "GroupSet %zu topology is null", group_set_id);
    RTP_LLM_CHECK_WITH_INFO(!group_ids.empty(), "GroupSet %zu group_ids must not be empty", group_set_id);
    RTP_LLM_CHECK_WITH_INFO(group_ids.size() == device_pools.size(),
                            "GroupSet %zu group/pool count mismatch: groups=%zu pools=%zu",
                            group_set_id,
                            group_ids.size(),
                            device_pools.size());

    const auto& topology_groups = topology->groups();
    RTP_LLM_CHECK_WITH_INFO(group_ids.front() < topology_groups.size(),
                            "GroupSet %zu invalid group_id=%zu topology_size=%zu",
                            group_set_id,
                            group_ids.front(),
                            topology_groups.size());
    const auto& first = topology->groupById(group_ids.front());
    std::unordered_set<size_t> unique_group_ids;
    size_t                     payload_bytes = 0;
    for (size_t local_group_index = 0; local_group_index < group_ids.size(); ++local_group_index) {
        const size_t group_id = group_ids[local_group_index];
        RTP_LLM_CHECK_WITH_INFO(group_id < topology_groups.size(),
                                "GroupSet %zu invalid group_id=%zu topology_size=%zu",
                                group_set_id,
                                group_id,
                                topology_groups.size());
        RTP_LLM_CHECK_WITH_INFO(unique_group_ids.emplace(group_id).second,
                                "GroupSet %zu duplicate group_id=%zu",
                                group_set_id,
                                group_id);
        RTP_LLM_CHECK_WITH_INFO(device_pools[local_group_index] != nullptr,
                                "GroupSet %zu device pool[%zu] is null",
                                group_set_id,
                                local_group_index);

        const auto& group = topology->groupById(group_id);
        RTP_LLM_CHECK_WITH_INFO(group.policy.enable_prefix_reuse,
                                "GroupSet %zu contains non-reusable group_id=%zu tag=%s",
                                group_set_id,
                                group_id,
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(!group.layer_ids.empty(),
                                "GroupSet %zu group_id=%zu has no layers",
                                group_set_id,
                                group_id);
        RTP_LLM_CHECK_WITH_INFO(
            CacheConfig::samePolicy(group.policy, first.policy) && group.block_num == first.block_num
                && group.local_kv_head_num == first.local_kv_head_num
                && group.seq_size_per_block == first.seq_size_per_block
                && group.kernel_seq_size_per_block == first.kernel_seq_size_per_block
                && group.kv_block_stride_bytes == first.kv_block_stride_bytes
                && group.kv_scale_stride_bytes == first.kv_scale_stride_bytes
                && (group.spec == nullptr) == (first.spec == nullptr)
                && (group.spec == nullptr || group.spec->type == first.spec->type),
            "GroupSet %zu incompatible group_id=%zu tag=%s",
            group_set_id,
            group_id,
            group.tag.c_str());

        RTP_LLM_CHECK_WITH_INFO(
            group.kv_block_stride_bytes <= std::numeric_limits<size_t>::max() - group.kv_scale_stride_bytes,
            "GroupSet %zu group_id=%zu layer payload overflow",
            group_set_id,
            group_id);
        const size_t layer_bytes = group.kv_block_stride_bytes + group.kv_scale_stride_bytes;
        RTP_LLM_CHECK_WITH_INFO(layer_bytes > 0,
                                "GroupSet %zu group_id=%zu has zero logical layer payload",
                                group_set_id,
                                group_id);
        RTP_LLM_CHECK_WITH_INFO(group.layer_ids.size() <= std::numeric_limits<size_t>::max() / layer_bytes,
                                "GroupSet %zu group_id=%zu payload multiply overflow",
                                group_set_id,
                                group_id);
        const size_t group_bytes = group.layer_ids.size() * layer_bytes;
        RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - payload_bytes,
                                "GroupSet %zu payload add overflow at group_id=%zu",
                                group_set_id,
                                group_id);
        payload_bytes += group_bytes;
    }

    group_set_id_  = group_set_id;
    topology_      = std::move(topology);
    group_ids_     = std::move(group_ids);
    device_pools_  = std::move(device_pools);
    payload_bytes_ = payload_bytes;
}

size_t GroupSet::groupSetId() const {
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "GroupSet identity requested before initialization");
    return group_set_id_;
}

const GroupBase& GroupSet::groupAt(size_t local_group_index) const {
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr && local_group_index < group_ids_.size(),
                            "GroupSet %zu invalid local_group_index=%zu size=%zu",
                            group_set_id_,
                            local_group_index,
                            group_ids_.size());
    return topology_->groupById(group_ids_[local_group_index]);
}

std::vector<std::string> GroupSet::groupTags() const {
    std::vector<std::string> tags;
    tags.reserve(group_ids_.size());
    for (size_t local_group_index = 0; local_group_index < group_ids_.size(); ++local_group_index) {
        tags.push_back(groupAt(local_group_index).tag);
    }
    return tags;
}

void GroupSet::evictFromTier(TreeNode* node, GroupSetResource& slot, Tier tier) {
    // Clear only the tier's block fields; heap membership is owned by BlockTreeEvictor.
    switch (tier) {
        case Tier::DEVICE:
            for (auto& block : slot.device_blocks) {
                block = NULL_BLOCK_IDX;
            }
            break;
        case Tier::HOST:
            slot.host_block = NULL_BLOCK_IDX;
            break;
        case Tier::DISK:
            slot.disk_slot = NULL_BLOCK_IDX;
            break;
        default:
            break;
    }
}

TransferDescriptor GroupSet::buildTransfer(TreeNode* node, TransferType type) {
    auto& slot = node->group_set_resources[groupSetId()];

    switch (type) {
        case TransferType::DEVICE_TO_HOST:
            return TransferDescriptor::deviceToHost(groupSetId(), slot.device_blocks, NULL_BLOCK_IDX);
        case TransferType::HOST_TO_DEVICE:
            return TransferDescriptor::hostToDevice(groupSetId(), slot.host_block, slot.device_blocks);
        case TransferType::HOST_TO_DISK:
            return TransferDescriptor::hostToDisk(groupSetId(), slot.host_block, NULL_BLOCK_IDX);
        case TransferType::DISK_TO_HOST:
            return TransferDescriptor::diskToHost(groupSetId(), slot.disk_slot, NULL_BLOCK_IDX);
        default:
            return {};
    }
}

bool GroupSet::isLeafAtTier(const TreeNode* node, Tier tier) const {
    const size_t group_set_id = groupSetId();
    RTP_LLM_CHECK_WITH_INFO(node != nullptr && group_set_id < node->group_set_resources.size(),
                            "GroupSet::isLeafAtTier invalid node/group_set_id=%zu",
                            group_set_id);
    auto& slot = node->group_set_resources[group_set_id];

    bool has_value = false;
    switch (tier) {
        case Tier::DEVICE:
            has_value = hasCompleteDeviceValue(slot);
            break;
        case Tier::HOST:
            has_value = slot.hasTier(Tier::HOST);
            break;
        case Tier::DISK:
            has_value = slot.hasTier(Tier::DISK);
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false,
                                    "GroupSet::isLeafAtTier invalid tier=%d group_set_id=%zu",
                                    static_cast<int>(tier),
                                    group_set_id);
    }
    if (!has_value) {
        return false;
    }

    for (const auto& [key, child] : node->children) {
        (void)key;
        RTP_LLM_CHECK_WITH_INFO(child != nullptr
                                    && group_set_id < child->group_set_resources.size(),
                                "GroupSet::isLeafAtTier invalid child/group_set_id=%zu",
                                group_set_id);
        auto& child_slot = child->group_set_resources[group_set_id];
        if (child_slot.hasTier(tier)) {
            return false;
        }
    }
    return true;
}

bool GroupSet::hasCompleteDeviceValue(const GroupSetResource& slot) const {
    return slot.device_blocks.size() == device_pools_.size()
           && std::all_of(slot.device_blocks.begin(), slot.device_blocks.end(), [](BlockIdxType block) {
                  return !isNullBlockIdx(block);
              });
}

bool GroupSet::isValidSteadyState(const GroupSetResource& slot) const {
    return slot.isValidSteadyState() && (slot.hasTier(Tier::DEVICE) == hasCompleteDeviceValue(slot));
}

bool GroupSet::hasAllocatedDeviceBlocks(const std::vector<BlockIdxType>& blocks) const {
    if (blocks.size() != device_pools_.size()) {
        return false;
    }
    for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
        if (isNullBlockIdx(blocks[pool_index]) || !device_pools_[pool_index]->isAllocated(blocks[pool_index])) {
            return false;
        }
    }
    return true;
}

// ---- Unified structured block lifecycle ----

MultiNodeResource GroupSet::allocateBlocks(Tier tier, size_t count, BlockRefType ref_type) {
    MultiNodeResource set{groupSetId(), tier};
    if (tier == Tier::DEVICE) {
        set.per_node.assign(count, std::vector<BlockIdxType>(device_pools_.size(), NULL_BLOCK_IDX));
        for (size_t p = 0; p < device_pools_.size(); ++p) {
            auto blocks = device_pools_[p]->malloc(count);
            if (!blocks.has_value()) {
                unreferenceBlocks(set, ref_type);
                return {};
            }
            device_pools_[p]->incRef(*blocks, ref_type);
            for (size_t k = 0; k < count; ++k) {
                set.per_node[k][p] = (*blocks)[k];
            }
        }
        return set;
    }

    set.per_node.assign(count, std::vector<BlockIdxType>{NULL_BLOCK_IDX});
    for (size_t k = 0; k < count; ++k) {
        BlockIdxType b = allocateSingleBlock(tier, ref_type);
        if (isNullBlockIdx(b)) {
            unreferenceBlocks(set, ref_type);
            return {};
        }
        set.per_node[k][0] = b;
    }
    return set;
}

void GroupSet::validateBlockResource(const MultiNodeResource& resource) const {
    RTP_LLM_CHECK_WITH_INFO(resource.group_set_id == groupSetId(),
                            "GroupSet resource id mismatch: expected=%zu actual=%zu",
                            groupSetId(),
                            resource.group_set_id);
    RTP_LLM_CHECK_WITH_INFO(resource.tree_nodes.empty() || resource.tree_nodes.size() == resource.per_node.size(),
                            "GroupSet %zu resource tree-node alignment mismatch: nodes=%zu blocks=%zu",
                            groupSetId(),
                            resource.tree_nodes.size(),
                            resource.per_node.size());
    const size_t expected_width = resource.tier == Tier::DEVICE ? device_pools_.size() : 1;
    RTP_LLM_CHECK_WITH_INFO(resource.tier == Tier::DEVICE || resource.tier == Tier::HOST
                                || resource.tier == Tier::DISK,
                            "GroupSet %zu resource has invalid tier=%d",
                            groupSetId(),
                            static_cast<int>(resource.tier));
    for (const auto& node_blocks : resource.per_node) {
        RTP_LLM_CHECK_WITH_INFO(node_blocks.size() == expected_width,
                                "GroupSet %zu resource width mismatch: tier=%s expected=%zu actual=%zu",
                                groupSetId(),
                                tierName(resource.tier),
                                expected_width,
                                node_blocks.size());
    }
}

void GroupSet::referenceBlocks(const MultiNodeResource& set, BlockRefType ref_type) const {
    validateBlockResource(set);
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size(); ++p) {
                    if (!isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->incRef(node_blocks[p], ref_type);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->incRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->incRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& set, BlockRefType ref_type) const {
    validateBlockResource(set);
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size(); ++p) {
                    if (!isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->decRef(node_blocks[p], ref_type);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->decRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->decRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

BlockIdxType GroupSet::allocateSingleBlock(Tier tier, BlockRefType ref_type) {
    // DEVICE spans multiple pools and has no scalar block: use allocateBlocks.
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
    if (isNullBlockIdx(block))
        return;
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decRef(block, ref_type);
    }
}

std::vector<BlockIdxType> GroupSet::getBlocks(const GroupSetResource& slot, Tier tier) const {
    if (!slot.hasTier(tier)) {
        return {};
    }
    switch (tier) {
        case Tier::DEVICE:
            return slot.device_blocks;
        case Tier::HOST:
            return {slot.host_block};
        case Tier::DISK:
            return {slot.disk_slot};
        default:
            return {};
    }
}

Tier GroupSet::getTopTier(const GroupSetResource& slot) const {
    if (slot.hasTier(Tier::DEVICE)) {
        return Tier::DEVICE;
    }
    if (slot.hasTier(Tier::HOST)) {
        return Tier::HOST;
    }
    if (slot.hasTier(Tier::DISK)) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

void GroupSet::setBlocks(GroupSetResource& slot, Tier tier, const std::vector<BlockIdxType>& blocks) {
    switch (tier) {
        case Tier::DEVICE:
            slot.device_blocks = blocks;
            break;
        case Tier::HOST:
            slot.host_block = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        case Tier::DISK:
            slot.disk_slot = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        default:
            break;
    }
}

bool GroupSet::isSlotEvictable(const TreeNode& node, Tier tier) const {
    const size_t group_set_id = groupSetId();
    if (group_set_id >= node.group_set_resources.size()) {
        return false;
    }
    const auto& slot = node.group_set_resources[group_set_id];

    // A block is evictable only when its sole holder is the cache reference
    // (refCount == 1). When no pool owns the block, treat it as evictable.
    auto pool_evictable = [](const auto& pool, BlockIdxType block) {
        if (isNullBlockIdx(block) || !pool) {
            return true;
        }
        return pool->isAllocated(block) && pool->refCount(block) == 1;
    };

    switch (tier) {
        case Tier::DEVICE:
            if (!hasCompleteDeviceValue(slot)) {
                return false;
            }
            for (size_t i = 0; i < slot.device_blocks.size(); ++i) {
                const auto& pool = i < device_pools_.size() ? device_pools_[i] : nullptr;
                if (!pool_evictable(pool, slot.device_blocks[i])) {
                    return false;
                }
            }
            return true;
        case Tier::HOST:
            return slot.hasTier(Tier::HOST) && pool_evictable(host_pool_, slot.host_block);
        case Tier::DISK:
            return slot.hasTier(Tier::DISK) && pool_evictable(disk_pool_, slot.disk_slot);
        default:
            return false;
    }
}

}  // namespace rtp_llm
