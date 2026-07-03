#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {

FullComponentGroup::FullComponentGroup(EvictionPolicy device_policy,
                                       EvictionPolicy host_policy,
                                       EvictionPolicy disk_policy) {
    group_type  = CacheGroupType::FULL;
    device_heap = std::make_unique<EvictionHeap>(device_policy);
    host_heap   = std::make_unique<EvictionHeap>(host_policy);
    disk_heap   = std::make_unique<EvictionHeap>(disk_policy);
}

std::unique_ptr<MatchValidator> FullComponentGroup::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

void FullComponentGroup::finalizeMatchResult(BlockTreeMatchResult& result) {
    // Handled at BlockTreeCache level.
}

void FullComponentGroup::commitInsertData(TreeNode*                        node,
                                          GroupSlot&                       slot,
                                          const std::vector<BlockIdxType>& block_indices) {
    slot.device_blocks  = block_indices;
    slot.in_device_heap = false;
}

void FullComponentGroup::updateOnInsertOverlap(TreeNode* node, GroupSlot& slot) {
    if (slot.in_device_heap && device_heap && device_heap->contains(node)) {
        device_heap->onAccess(node);
    }
}

void FullComponentGroup::evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) {
    switch (tier) {
        case Tier::DEVICE: {
            for (auto& block : slot.device_blocks) {
                block = NULL_BLOCK_IDX;
            }
            slot.in_device_heap = false;
            if (device_heap) {
                device_heap->invalidate(node);
            }
            break;
        }
        case Tier::HOST: {
            slot.host_block   = NULL_BLOCK_IDX;
            slot.in_host_heap = false;
            if (host_heap) {
                host_heap->invalidate(node);
            }
            break;
        }
        case Tier::DISK: {
            slot.disk_slot    = NULL_BLOCK_IDX;
            slot.in_disk_heap = false;
            if (disk_heap) {
                disk_heap->invalidate(node);
            }
            break;
        }
        default:
            break;
    }
}

std::optional<EvictionResult> FullComponentGroup::driveEviction(int num_blocks, Tier tier) {
    auto* heap = heapForTier(tier);
    if (!heap || heap->empty()) {
        return std::nullopt;
    }

    auto entry = heap->pop();
    if (!entry.has_value()) {
        return std::nullopt;
    }

    EvictionResult result;
    result.node               = entry->node;
    result.component_group_id = component_group_id;
    result.source_tier        = tier;

    switch (tier) {
        case Tier::DEVICE: {
            result.target_tier       = (reuse_policy == CacheReusePolicy::NON_REUSABLE) ? Tier::NONE : Tier::HOST;
            auto& slot               = entry->node->group_slots[static_cast<size_t>(component_group_id)];
            result.blocks_to_release = slot.device_blocks;
            result.transfer          = buildTransfer(entry->node, TransferType::DEVICE_TO_HOST);
            break;
        }
        case Tier::HOST: {
            result.target_tier       = Tier::DISK;
            auto& slot               = entry->node->group_slots[static_cast<size_t>(component_group_id)];
            result.blocks_to_release = {slot.host_block};
            result.transfer          = buildTransfer(entry->node, TransferType::HOST_TO_DISK);
            break;
        }
        case Tier::DISK: {
            result.target_tier       = Tier::NONE;
            auto& slot               = entry->node->group_slots[static_cast<size_t>(component_group_id)];
            result.blocks_to_release = {slot.disk_slot};
            break;
        }
        default:
            return std::nullopt;
    }

    return result;
}

TransferDescriptor FullComponentGroup::buildTransfer(TreeNode* node, TransferType type) {
    TransferDescriptor desc;
    desc.component_group_id = component_group_id;
    desc.nodes              = {node};
    auto& slot              = node->group_slots[static_cast<size_t>(component_group_id)];

    switch (type) {
        case TransferType::DEVICE_TO_HOST:
            desc.source_tier   = Tier::DEVICE;
            desc.target_tier   = Tier::HOST;
            desc.source_blocks = {slot.device_blocks};
            break;
        case TransferType::HOST_TO_DEVICE:
            desc.source_tier   = Tier::HOST;
            desc.target_tier   = Tier::DEVICE;
            desc.source_blocks = {{slot.host_block}};
            break;
        case TransferType::HOST_TO_DISK:
            desc.source_tier   = Tier::HOST;
            desc.target_tier   = Tier::DISK;
            desc.source_blocks = {{slot.host_block}};
            break;
        case TransferType::DISK_TO_HOST:
            desc.source_tier   = Tier::DISK;
            desc.target_tier   = Tier::HOST;
            desc.source_blocks = {{slot.disk_slot}};
            break;
        default:
            break;
    }
    return desc;
}

bool FullComponentGroup::isDeviceLeaf(const TreeNode* node, int group_id) const {
    if (node == nullptr)
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];
    if (!slot.has_device_value())
        return false;

    for (const auto& [key, child] : node->children) {
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        if (child_slot.has_device_value()) {
            return false;
        }
    }
    return true;
}

bool FullComponentGroup::isHostLeaf(const TreeNode* node, int group_id) const {
    if (node == nullptr)
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];
    if (!slot.has_host_value())
        return false;

    for (const auto& [key, child] : node->children) {
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        if (child_slot.has_host_value()) {
            return false;
        }
    }
    return true;
}

bool FullComponentGroup::isDiskLeaf(const TreeNode* node, int group_id) const {
    if (node == nullptr)
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];
    if (!slot.has_disk_value())
        return false;

    for (const auto& [key, child] : node->children) {
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        if (child_slot.has_disk_value()) {
            return false;
        }
    }
    return true;
}

void FullComponentGroup::tryAddToDeviceHeap(TreeNode* node) {
    if (!device_heap)
        return;
    if (isDeviceLeaf(node, component_group_id)) {
        auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
        if (!slot.in_device_heap) {
            device_heap->push(node, component_group_id);
            slot.in_device_heap = true;
        }
    }
}

// FullMatchValidator
bool FullMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    return slot.has_any_value();
}

}  // namespace rtp_llm
