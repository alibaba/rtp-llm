#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"

namespace rtp_llm {

SWAComponentGroup::SWAComponentGroup(size_t sliding_window_size, size_t seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {
    group_type = CacheGroupType::SWA;
}

std::unique_ptr<MatchValidator> SWAComponentGroup::createMatchValidator() {
    return std::make_unique<SWAMatchValidator>(sliding_window_size_, seq_size_per_block_);
}

size_t SWAComponentGroup::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const {
    if (sliding_window_size_ == 0) {
        return matched_block_count;  // No window configured, full path
    }
    const size_t group_id    = component_group_id;
    size_t       count       = 0;
    size_t       accumulated = 0;
    for (size_t i = matched_block_count; i > 0; --i) {
        const TreeNode* node = path[i - 1];
        if (group_id < node->group_slots.size() && !node->group_slots[group_id].is_empty()) {
            count++;
            accumulated += seq_size_per_block_;
            if (accumulated >= sliding_window_size_) {
                break;
            }
        }
    }
    return count;
}

// SWAMatchValidator
SWAMatchValidator::SWAMatchValidator(size_t sliding_window_size, size_t seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

bool SWAMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    const bool has_swa_data = !slot.is_empty();

    if (!has_swa_data) {
        connected_to_root_  = false;
        accumulated_length_ = 0;
        return false;
    }

    accumulated_length_ += seq_size_per_block_;
    if (connected_to_root_ || sliding_window_size_ == 0) {
        return true;
    }
    return accumulated_length_ >= sliding_window_size_;
}

}  // namespace rtp_llm
