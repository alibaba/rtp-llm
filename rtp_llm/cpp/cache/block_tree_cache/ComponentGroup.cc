#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

#include <limits>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void ComponentGroup::setDevicePools(std::vector<DeviceBlockPoolPtr> pools, std::vector<std::string> tags) {
    RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() && tags_.empty(), "component group device mapping is immutable");
    device_pools_ = std::move(pools);
    tags_         = std::move(tags);
}

std::optional<ComponentGroupLayout>
ComponentGroupLayout::create(const std::vector<std::vector<size_t>>& component_layer_bytes) {
    if (component_layer_bytes.empty()) {
        RTP_LLM_LOG_WARNING("ComponentGroupLayout: components must not be empty");
        return std::nullopt;
    }

    ComponentGroupLayout layout;
    size_t               offset = 0;
    for (size_t component_idx = 0; component_idx < component_layer_bytes.size(); ++component_idx) {
        const auto& layer_bytes = component_layer_bytes[component_idx];
        if (layer_bytes.empty()) {
            RTP_LLM_LOG_WARNING("ComponentGroupLayout: component=%zu has no layer binding", component_idx);
            return std::nullopt;
        }
        if (layer_bytes.size() > std::numeric_limits<size_t>::max() - layout.slices_.size()) {
            RTP_LLM_LOG_WARNING("ComponentGroupLayout: total layer count overflow at component=%zu", component_idx);
            return std::nullopt;
        }
        layout.slices_.reserve(layout.slices_.size() + layer_bytes.size());
        for (size_t layer_idx = 0; layer_idx < layer_bytes.size(); ++layer_idx) {
            const size_t bytes = layer_bytes[layer_idx];
            if (bytes == 0) {
                RTP_LLM_LOG_WARNING(
                    "ComponentGroupLayout: component=%zu layer=%zu has zero packed bytes", component_idx, layer_idx);
                return std::nullopt;
            }
            if (bytes > std::numeric_limits<size_t>::max() - offset) {
                RTP_LLM_LOG_WARNING("ComponentGroupLayout: payload offset overflow at component=%zu layer=%zu",
                                    component_idx,
                                    layer_idx);
                return std::nullopt;
            }
            layout.slices_.push_back(Slice{component_idx, layer_idx, offset});
            offset += bytes;
        }
    }

    layout.payload_bytes_ = offset;
    return layout;
}

bool ComponentGroup::setLayout(std::vector<int> component_indices, ComponentGroupLayout layout) {
    if (layout_.has_value()) {
        RTP_LLM_LOG_ERROR("group %d layout is already sealed", component_group_id);
        return false;
    }
    component_indices_ = std::move(component_indices);
    layout_            = std::move(layout);
    return true;
}

const ComponentGroupLayout& ComponentGroup::layout() const {
    RTP_LLM_CHECK_WITH_INFO(layout_.has_value(), "ComponentGroup %d layout has not been finalized", component_group_id);
    return *layout_;
}

void ComponentGroup::evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) {
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

TransferDescriptor ComponentGroup::buildTransfer(TreeNode* node, TransferType type) {
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];

    switch (type) {
        case TransferType::DEVICE_TO_HOST:
            return TransferDescriptor::deviceToHost(component_group_id, slot.device_blocks, NULL_BLOCK_IDX);
        case TransferType::HOST_TO_DEVICE:
            return TransferDescriptor::hostToDevice(component_group_id, slot.host_block, slot.device_blocks);
        case TransferType::HOST_TO_DISK:
            return TransferDescriptor::hostToDisk(component_group_id, slot.host_block, NULL_BLOCK_IDX);
        case TransferType::DISK_TO_HOST:
            return TransferDescriptor::diskToHost(component_group_id, slot.disk_slot, NULL_BLOCK_IDX);
        default:
            return {};
    }
}

bool ComponentGroup::isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const {
    if (node == nullptr || group_id < 0 || static_cast<size_t>(group_id) >= node->group_slots.size())
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];

    bool has_value = false;
    switch (tier) {
        case Tier::DEVICE:
            has_value = hasCompleteDeviceValue(slot);
            break;
        case Tier::HOST:
            has_value = slot.has_value(Tier::HOST);
            break;
        case Tier::DISK:
            has_value = slot.has_value(Tier::DISK);
            break;
        default:
            return false;
    }
    if (!has_value)
        return false;

    for (const auto& [key, child] : node->children) {
        (void)key;
        if (child == nullptr || static_cast<size_t>(group_id) >= child->group_slots.size()) {
            return false;
        }
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        if (child_slot.has_value(tier)) {
            return false;
        }
    }
    return true;
}

bool ComponentGroup::hasCompleteDeviceValue(const GroupSlot& slot) const {
    return !device_pools_.empty() && slot.device_blocks.size() == device_pools_.size()
           && std::all_of(slot.device_blocks.begin(), slot.device_blocks.end(), [](BlockIdxType block) {
                  return !isNullBlockIdx(block);
              });
}

// ---- Unified structured block lifecycle ----

GroupBlockSet ComponentGroup::allocateBlocks(Tier tier, size_t count, BlockRefType ref_type) {
    GroupBlockSet set{component_group_id, tier};
    if (tier == Tier::DEVICE) {
        set.per_node.assign(count, std::vector<BlockIdxType>(device_pools_.size(), NULL_BLOCK_IDX));
        for (size_t p = 0; p < device_pools_.size(); ++p) {
            if (!device_pools_[p]) {
                unreferenceBlocks(set, ref_type);
                return {};
            }
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

    set.per_node.resize(count);
    for (size_t k = 0; k < count; ++k) {
        BlockIdxType b = allocateSingleBlock(tier, ref_type);
        if (isNullBlockIdx(b)) {
            unreferenceBlocks(set, ref_type);
            return {};
        }
        set.per_node[k] = {b};
    }
    return set;
}

void ComponentGroup::referenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
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

void ComponentGroup::unreferenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
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

BlockIdxType ComponentGroup::allocateSingleBlock(Tier tier, BlockRefType ref_type) {
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

void ComponentGroup::releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const {
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

std::vector<BlockIdxType> ComponentGroup::getBlocks(const GroupSlot& slot, Tier tier) const {
    if (!slot.has_value(tier)) {
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

Tier ComponentGroup::getTopTier(const GroupSlot& slot) const {
    if (slot.has_value(Tier::DEVICE)) {
        return Tier::DEVICE;
    }
    if (slot.has_value(Tier::HOST)) {
        return Tier::HOST;
    }
    if (slot.has_value(Tier::DISK)) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

void ComponentGroup::setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks) {
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

bool ComponentGroup::isSlotEvictable(const TreeNode& node, Tier tier) const {
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= node.group_slots.size()) {
        return false;
    }
    const auto& slot = node.group_slots[static_cast<size_t>(component_group_id)];

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
            return slot.has_value(Tier::HOST) && pool_evictable(host_pool_, slot.host_block);
        case Tier::DISK:
            return slot.has_value(Tier::DISK) && pool_evictable(disk_pool_, slot.disk_slot);
        default:
            return false;
    }
}

}  // namespace rtp_llm
