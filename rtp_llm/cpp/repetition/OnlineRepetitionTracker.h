#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

struct OnlineRepetitionConfig {
    int min_repeats = 3;
    int min_duplicate_tokens = 32;
    int max_period = 512;
};

struct OnlineRepetitionResult {
    bool hit = false;
    int repeat_unit_size = 0;
    int repeat_count = 0;
    int partial_tail_tokens = 0;
    int covered_token_count = 0;
    int duplicate_token_count = 0;
    int start_index = 0;
    int end_index = 0;
    int first_detect_index = 0;
};

class OnlineRepetitionTracker {
public:
    explicit OnlineRepetitionTracker(OnlineRepetitionConfig config = {});

    void reset();
    bool update(int token_id);
    bool updateMany(const std::vector<int>& token_ids);
    bool considerFinalTail();

    const OnlineRepetitionResult& result() const {
        return result_;
    }

    int tokenCount() const {
        return token_count_;
    }

private:
    struct RecentPositions {
        std::vector<int> values;
        std::size_t first = 0;
    };

    OnlineRepetitionConfig config_;
    int token_count_ = 0;
    OnlineRepetitionResult result_;
    std::unordered_map<int, RecentPositions> positions_by_token_;
    std::vector<int> match_len_by_period_;
    std::vector<int> last_match_index_by_period_;

    bool considerMatch(int period, int match_len, int token_index);
    bool considerCandidate(int period, int covered, int token_index, bool include_partial_tail);
};

OnlineRepetitionResult detectOnlineRepetitionHitOnly(
    const std::vector<int>& token_ids,
    OnlineRepetitionConfig config = {});

OnlineRepetitionResult detectOnlineRepetitionMax(
    const std::vector<int>& token_ids,
    OnlineRepetitionConfig config = {});

}  // namespace rtp_llm
