#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// Logical positions describe the canonical ownership row. A forward may map a
// different execution position to an owned slot, without acquiring a new ref.
class LinearPrefillPlan {
public:
    struct Write {
        int64_t execution_position;
        int64_t owner_position;
        int64_t tokens;
        bool    retained;
    };

    struct Round {
        std::optional<int64_t> input_position;
        std::optional<int64_t> input_owner_position;
        std::vector<Write>     writes;
        int64_t                output_owner_position;
        int64_t                end_tokens;
    };

    LinearPrefillPlan(int64_t prompt_tokens,
                      int64_t page_tokens,
                      int64_t checkpoint_tokens,
                      int64_t tail_checkpoints,
                      int64_t reused_tokens = 0):
        prompt_tokens_(prompt_tokens),
        page_tokens_(page_tokens),
        checkpoint_tokens_(checkpoint_tokens),
        reused_tokens_(reused_tokens) {
        if (prompt_tokens <= 0 || page_tokens <= 0 || checkpoint_tokens <= 0 || tail_checkpoints <= 0
            || checkpoint_tokens % page_tokens != 0 || reused_tokens < 0 || reused_tokens >= prompt_tokens
            || reused_tokens % checkpoint_tokens != 0) {
            throw std::invalid_argument("invalid LINEAR prefill checkpoint geometry or reuse boundary");
        }
        const auto complete = prompt_tokens / checkpoint_tokens;
        const auto count    = std::min(tail_checkpoints, complete);
        first_checkpoint_   = complete - count + 1;
        for (int64_t index = first_checkpoint_; index <= complete; ++index) {
            checkpoint_positions_.push_back(index * checkpoint_tokens / page_tokens - 1);
        }
    }

    int64_t promptTokens() const {
        return prompt_tokens_;
    }
    int64_t reusedTokens() const {
        return reused_tokens_;
    }
    int64_t pageTokens() const {
        return page_tokens_;
    }
    int64_t checkpointTokens() const {
        return checkpoint_tokens_;
    }
    int64_t borrowedPosition() const {
        return (prompt_tokens_ - 1) / page_tokens_;
    }
    bool hasSeparateTail() const {
        return prompt_tokens_ % checkpoint_tokens_ != 0;
    }
    const std::vector<int64_t>& checkpointPositions() const {
        return checkpoint_positions_;
    }

    // Existing prefix state is an independent read-only hold. Missing slots in
    // the uncomputed prefix never become allocations merely to fill Last-N.
    std::vector<int64_t> newPositions() const {
        std::vector<int64_t> positions;
        for (const auto position : checkpoint_positions_) {
            if ((position + 1) * page_tokens_ > reused_tokens_) {
                positions.push_back(position);
            }
        }
        if (hasSeparateTail()) {
            positions.push_back(borrowedPosition());
        }
        return positions;
    }

    bool isRetainedBoundary(int64_t tokens) const {
        return tokens > 0 && tokens % checkpoint_tokens_ == 0 && tokens / checkpoint_tokens_ >= first_checkpoint_
               && tokens <= prompt_tokens_;
    }

    // Called with the scheduler's actual grant, not its configured maximum.
    // Intermediate grants must leave an unambiguous base-page input slot.
    Round round(int64_t prefix_tokens, int64_t query_tokens) const {
        if (prefix_tokens < reused_tokens_ || prefix_tokens >= prompt_tokens_ || prefix_tokens % page_tokens_ != 0
            || query_tokens <= 0 || query_tokens > prompt_tokens_ - prefix_tokens) {
            throw std::invalid_argument("invalid LINEAR prefill chunk");
        }
        const auto end = prefix_tokens + query_tokens;
        if (end != prompt_tokens_ && end % page_tokens_ != 0) {
            throw std::invalid_argument("intermediate LINEAR chunk must end on a base-page boundary");
        }
        Round result;
        result.end_tokens = end;
        if (prefix_tokens > 0) {
            result.input_position       = (prefix_tokens - 1) / page_tokens_;
            result.input_owner_position = prefix_tokens == reused_tokens_ || isRetainedBoundary(prefix_tokens) ?
                                              *result.input_position :
                                              borrowedPosition();
        }
        for (const auto position : checkpoint_positions_) {
            const auto tokens = (position + 1) * page_tokens_;
            if (tokens > prefix_tokens && tokens <= end) {
                result.writes.push_back({position, position, tokens, true});
            }
        }
        if (isRetainedBoundary(end)) {
            result.output_owner_position = (end - 1) / page_tokens_;
        } else {
            result.output_owner_position = borrowedPosition();
            result.writes.push_back({(end - 1) / page_tokens_, borrowedPosition(), end, false});
        }
        return result;
    }

    // A preallocated final slot can still hold an earlier chunk's temporary
    // state. Publication must check the actual completed boundary as well.
    bool canPublish(int64_t owner_position, int64_t completed_tokens) const {
        return std::binary_search(checkpoint_positions_.begin(), checkpoint_positions_.end(), owner_position)
               && (owner_position + 1) * page_tokens_ > reused_tokens_
               && (owner_position + 1) * page_tokens_ <= completed_tokens;
    }

private:
    int64_t              prompt_tokens_;
    int64_t              page_tokens_;
    int64_t              checkpoint_tokens_;
    int64_t              reused_tokens_;
    int64_t              first_checkpoint_;
    std::vector<int64_t> checkpoint_positions_;
};

}  // namespace rtp_llm
