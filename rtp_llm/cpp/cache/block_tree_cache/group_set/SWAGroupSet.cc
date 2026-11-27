#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/PrefixCoverage.h"

#include <limits>

namespace rtp_llm {

std::unique_ptr<MatchValidator> SWAGroupSet::createMatchValidator() {
    return std::make_unique<SWAMatchValidator>(slidingWindowSize(), seqSizePerBlock());
}

size_t SWAGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return reusableBlockCount(groupType(), pageTokens(), windowTokens(), matched_block_count);
}

// SWAMatchValidator
SWAMatchValidator::SWAMatchValidator(size_t sliding_window_size, size_t seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

bool SWAMatchValidator::validate(const GroupSetResource& resource) {
    if (!resource.isMatchUsable()) {
        connected_to_root_  = false;
        accumulated_length_ = 0;
        return false;
    }

    accumulated_length_ += std::min(seq_size_per_block_, std::numeric_limits<size_t>::max() - accumulated_length_);
    return connected_to_root_ || (sliding_window_size_ > 0 && accumulated_length_ >= sliding_window_size_);
}

}  // namespace rtp_llm
