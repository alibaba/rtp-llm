#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// ---- Base class default implementations (shared by Full/SWA/Linear) ----

void ComponentGroup::commitInsertData(TreeNode* node, GroupSlot& slot, const std::vector<BlockIdxType>& block_indices) {
    slot.device_blocks  = block_indices;
    slot.in_device_heap = false;
}

void ComponentGroup::updateOnInsertOverlap(TreeNode* node, GroupSlot& slot) {
    if (slot.in_device_heap && device_heap && device_heap->contains(node)) {
        device_heap->onAccess(node);
    }
}

void ComponentGroup::evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) {
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

std::optional<EvictionResult> ComponentGroup::driveEviction(int num_blocks, Tier tier) {
    auto* heap = heapForTier(tier);
    if (!heap || heap->empty()) {
        return std::nullopt;
    }

    while (!heap->empty()) {
        auto entry = heap->pop();
        if (!entry.has_value()) {
            return std::nullopt;
        }

        // Check evictability (reference count) — skip if not evictable
        if (is_block_evictable_) {
            auto& slot      = entry->node->group_slots[static_cast<size_t>(component_group_id)];
            bool  evictable = true;
            switch (tier) {
                case Tier::DEVICE:
                    for (auto block : slot.device_blocks) {
                        if (block != NULL_BLOCK_IDX && !is_block_evictable_(block)) {
                            evictable = false;
                            break;
                        }
                    }
                    break;
                case Tier::HOST:
                    if (slot.has_host_value() && !is_block_evictable_(slot.host_block)) {
                        evictable = false;
                    }
                    break;
                case Tier::DISK:
                    if (slot.has_disk_value() && !is_block_evictable_(slot.disk_slot)) {
                        evictable = false;
                    }
                    break;
                default:
                    break;
            }
            if (!evictable) {
                continue;  // skip this candidate, try next
            }
        }

        EvictionResult result;
        result.node               = entry->node;
        result.component_group_id = component_group_id;
        result.source_tier        = tier;

        switch (tier) {
            case Tier::DEVICE: {
                result.target_tier       = Tier::HOST;
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

    return std::nullopt;
}

TransferDescriptor ComponentGroup::buildTransfer(TreeNode* node, TransferType type) {
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

bool ComponentGroup::isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const {
    if (node == nullptr)
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];

    bool has_value = false;
    switch (tier) {
        case Tier::DEVICE:
            has_value = slot.has_device_value();
            break;
        case Tier::HOST:
            has_value = slot.has_host_value();
            break;
        case Tier::DISK:
            has_value = slot.has_disk_value();
            break;
        default:
            return false;
    }
    if (!has_value)
        return false;

    for (const auto& [key, child] : node->children) {
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        switch (tier) {
            case Tier::DEVICE:
                if (child_slot.has_device_value())
                    return false;
                break;
            case Tier::HOST:
                if (child_slot.has_host_value())
                    return false;
                break;
            case Tier::DISK:
                if (child_slot.has_disk_value())
                    return false;
                break;
            default:
                break;
        }
    }
    return true;
}

void ComponentGroup::tryAddToHostHeap(TreeNode* node) {
    if (!host_heap)
        return;
    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= node->group_slots.size())
        return;
    auto& slot = node->group_slots[gid];
    if (!slot.has_device_value() && slot.has_host_value() && !slot.in_host_heap) {
        host_heap->push(node, component_group_id);
        slot.in_host_heap = true;
    }
}

void ComponentGroup::tryAddToDiskHeap(TreeNode* node) {
    if (!disk_heap)
        return;
    auto gid = static_cast<size_t>(component_group_id);
    if (gid >= node->group_slots.size())
        return;
    auto& slot = node->group_slots[gid];
    if (!slot.has_device_value() && !slot.has_host_value() && slot.has_disk_value() && !slot.in_disk_heap) {
        disk_heap->push(node, component_group_id);
        slot.in_disk_heap = true;
    }
}

}  // namespace rtp_llm
