#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

void ComponentGroup::tryAddToHostHeap(TreeNode* node) {
    if (!host_heap)
        return;
    if (reuse_policy == CacheReusePolicy::NON_REUSABLE)
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
    if (reuse_policy == CacheReusePolicy::NON_REUSABLE)
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
