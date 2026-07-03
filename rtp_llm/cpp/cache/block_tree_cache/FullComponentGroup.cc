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

void FullComponentGroup::tryAddToDeviceHeap(TreeNode* node) {
    if (!device_heap)
        return;
    if (isLeafAtTier(node, component_group_id, Tier::DEVICE)) {
        auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
        if (!slot.in_device_heap) {
            device_heap->push(node, component_group_id);
            slot.in_device_heap = true;
        }
    }
}

void FullComponentGroup::tryAddToHostHeap(TreeNode* node) {
    if (!host_heap)
        return;
    if (reuse_policy == CacheReusePolicy::NON_REUSABLE)
        return;
    // Full-specific: only HostLeaf nodes enter heap
    if (!isLeafAtTier(node, component_group_id, Tier::HOST))
        return;
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
    if (!slot.in_host_heap) {
        host_heap->push(node, component_group_id);
        slot.in_host_heap = true;
    }
}

void FullComponentGroup::tryAddToDiskHeap(TreeNode* node) {
    if (!disk_heap)
        return;
    if (reuse_policy == CacheReusePolicy::NON_REUSABLE)
        return;
    // Full-specific: only DiskLeaf nodes enter heap
    if (!isLeafAtTier(node, component_group_id, Tier::DISK))
        return;
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
    if (!slot.in_disk_heap) {
        disk_heap->push(node, component_group_id);
        slot.in_disk_heap = true;
    }
}

// FullMatchValidator
bool FullMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    return slot.has_any_value();
}

}  // namespace rtp_llm
