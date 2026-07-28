#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

namespace rtp_llm {

SWAGroupSet::SWAGroupSet(size_t sliding_window_size, size_t seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

std::unique_ptr<MatchValidator> SWAGroupSet::createMatchValidator() {
    return std::make_unique<SWAMatchValidator>(sliding_window_size_, seq_size_per_block_);
}

size_t SWAGroupSet::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const {
    if (sliding_window_size_ == 0) {
        return matched_block_count;  // No window configured, full path
    }
    const size_t group_set_index = groupSetId();
    size_t       count           = 0;
    size_t       accumulated     = 0;
    for (size_t i = matched_block_count; i > 0; --i) {
        const TreeNode* node = path[i - 1];
        if (group_set_index < node->group_set_resources.size()
            && node->group_set_resources[group_set_index].isMatchUsable()
            && !node->group_set_resources[group_set_index].is_empty()) {
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

bool SWAMatchValidator::validate(const TreeNode* node, const GroupSetResource& slot) {
    // A busy slot (DEMOTING/LOAD_PENDING) counts as a hole: it cannot serve
    // this match, so the window accumulation restarts behind it.
    const bool has_swa_data = slot.isMatchUsable() && !slot.is_empty();

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
