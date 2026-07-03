#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"

namespace rtp_llm {

SWAComponentGroup::SWAComponentGroup(int            sliding_window_size,
                                     int            seq_size_per_block,
                                     EvictionPolicy device_policy,
                                     EvictionPolicy host_policy,
                                     EvictionPolicy disk_policy):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {
    group_type  = CacheGroupType::SWA;
    device_heap = std::make_unique<EvictionHeap>(device_policy);
    host_heap   = std::make_unique<EvictionHeap>(host_policy);
    disk_heap   = std::make_unique<EvictionHeap>(disk_policy);
}

std::unique_ptr<MatchValidator> SWAComponentGroup::createMatchValidator() {
    return std::make_unique<SWAMatchValidator>(sliding_window_size_, seq_size_per_block_);
}

void SWAComponentGroup::tryAddToDeviceHeap(TreeNode* node) {
    if (!device_heap)
        return;
    // SWA: any node with device data can enter heap (no Leaf requirement)
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];
    if (slot.has_device_value() && !slot.in_device_heap) {
        device_heap->push(node, component_group_id);
        slot.in_device_heap = true;
    }
}

size_t SWAComponentGroup::computeReferenceCount(size_t matched_blocks, const std::vector<TreeNode*>& path) const {
    if (sliding_window_size_ <= 0)
        return matched_blocks;  // No window configured → full path
    size_t count       = 0;
    size_t accumulated = 0;
    auto   gid         = static_cast<size_t>(component_group_id);
    for (int i = static_cast<int>(matched_blocks) - 1; i >= 0; --i) {
        if (gid < path[static_cast<size_t>(i)]->group_slots.size()
            && path[static_cast<size_t>(i)]->group_slots[gid].has_device_value()) {
            count++;
            accumulated += static_cast<size_t>(seq_size_per_block_);
            if (accumulated >= static_cast<size_t>(sliding_window_size_))
                break;
        }
    }
    return count;
}

// SWAMatchValidator
SWAMatchValidator::SWAMatchValidator(int sliding_window_size, int seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

bool SWAMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    bool has_swa_data = slot.has_any_value();

    if (has_swa_data) {
        accumulated_length_ += static_cast<size_t>(seq_size_per_block_);
        return true;  // SWA validator tracks state only; match validity is gated by FULL group
    }
    // SWA data missing: break connection, reset accumulated length
    connected_to_root_  = false;
    accumulated_length_ = 0;
    return true;  // SWA validator doesn't fail the match when data is missing (allows FULL-only matches)
}

}  // namespace rtp_llm
