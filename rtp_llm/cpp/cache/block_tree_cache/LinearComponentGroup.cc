#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"

namespace rtp_llm {

LinearComponentGroup::LinearComponentGroup(EvictionPolicy device_policy,
                                           EvictionPolicy host_policy,
                                           EvictionPolicy disk_policy) {
    group_type  = CacheGroupType::LINEAR;
    device_heap = std::make_unique<EvictionHeap>(device_policy);
    host_heap   = std::make_unique<EvictionHeap>(host_policy);
    disk_heap   = std::make_unique<EvictionHeap>(disk_policy);
}

std::unique_ptr<MatchValidator> LinearComponentGroup::createMatchValidator() {
    return std::make_unique<LinearMatchValidator>();
}

size_t LinearComponentGroup::computeReferenceCount(size_t matched_block_count, const std::vector<TreeNode*>&) const {
    return matched_block_count == 0 ? 0 : 1;
}

void LinearComponentGroup::tryAddToDeviceHeap(TreeNode* node) {
    if (!device_heap)
        return;
    // LINEAR: any node with device data can enter heap (no Leaf requirement)
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
    if (slot.has_value(Tier::DEVICE) && !slot.in_device_heap) {
        device_heap->push(node, component_group_id);
        slot.in_device_heap = true;
    }
}

bool LinearMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    return !slot.is_empty();
}

}  // namespace rtp_llm
