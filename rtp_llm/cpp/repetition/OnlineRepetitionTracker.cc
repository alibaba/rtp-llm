#include "rtp_llm/cpp/repetition/OnlineRepetitionTracker.h"

#include <algorithm>
#include <tuple>

namespace rtp_llm {

namespace {

OnlineRepetitionConfig normalizeConfig(OnlineRepetitionConfig config) {
    config.min_repeats = std::max(3, config.min_repeats);
    config.min_duplicate_tokens = std::max(0, config.min_duplicate_tokens);
    config.max_period = std::max(1, config.max_period);
    return config;
}

bool betterResult(const OnlineRepetitionResult& lhs, const OnlineRepetitionResult& rhs) {
    if (!rhs.hit) {
        return lhs.hit;
    }
    if (!lhs.hit) {
        return false;
    }
    return std::make_tuple(lhs.duplicate_token_count, lhs.covered_token_count, -lhs.repeat_unit_size) >
           std::make_tuple(rhs.duplicate_token_count, rhs.covered_token_count, -rhs.repeat_unit_size);
}

OnlineRepetitionResult normalizeResultForEnd(const OnlineRepetitionResult& result, int token_count) {
    if (!result.hit) {
        return result;
    }
    OnlineRepetitionResult normalized = result;
    const bool reaches_final_end = token_count >= 0 && result.end_index == token_count;
    if (reaches_final_end) {
        normalized.duplicate_token_count =
            normalized.covered_token_count - normalized.repeat_unit_size;
        return normalized;
    }

    normalized.partial_tail_tokens = 0;
    normalized.covered_token_count =
        normalized.repeat_count * normalized.repeat_unit_size;
    normalized.duplicate_token_count =
        normalized.covered_token_count - normalized.repeat_unit_size;
    normalized.end_index = normalized.start_index + normalized.covered_token_count;
    return normalized;
}

}  // namespace

OnlineRepetitionTracker::OnlineRepetitionTracker(OnlineRepetitionConfig config):
    config_(normalizeConfig(config)),
    match_len_by_period_(static_cast<std::size_t>(config_.max_period) + 1, 0),
    last_match_index_by_period_(static_cast<std::size_t>(config_.max_period) + 1, -2) {}

void OnlineRepetitionTracker::reset() {
    token_count_ = 0;
    result_ = OnlineRepetitionResult();
    positions_by_token_.clear();
    std::fill(match_len_by_period_.begin(), match_len_by_period_.end(), 0);
    std::fill(last_match_index_by_period_.begin(), last_match_index_by_period_.end(), -2);
}

bool OnlineRepetitionTracker::considerCandidate(int period, int covered, int token_index, bool include_partial_tail) {
    const int repeat_count = covered / period;
    if (repeat_count < config_.min_repeats) {
        return false;
    }

    const int complete_covered = repeat_count * period;
    const int duplicate_tokens = (include_partial_tail ? covered : complete_covered) - period;
    if (duplicate_tokens < config_.min_duplicate_tokens) {
        return false;
    }

    OnlineRepetitionResult candidate;
    candidate.hit = true;
    candidate.repeat_unit_size = period;
    candidate.repeat_count = repeat_count;
    candidate.partial_tail_tokens = covered % period;
    candidate.covered_token_count = covered;
    candidate.duplicate_token_count = duplicate_tokens;
    candidate.start_index = token_index - covered + 1;
    candidate.end_index = token_index + 1;
    candidate.first_detect_index = token_index;

    if (!result_.hit || betterResult(candidate, result_)) {
        result_ = candidate;
    }
    return true;
}

bool OnlineRepetitionTracker::considerMatch(int period, int match_len, int token_index) {
    return considerCandidate(period, match_len + period, token_index, false);
}

bool OnlineRepetitionTracker::update(int token_id) {
    const int token_index = token_count_++;
    auto& positions = positions_by_token_[token_id];
    const int oldest_kept = token_index - config_.max_period;
    while (positions.first < positions.values.size() &&
           positions.values[positions.first] < oldest_kept) {
        ++positions.first;
    }

    bool hit_now = false;
    for (std::size_t pos_index = positions.values.size(); pos_index > positions.first;) {
        --pos_index;
        const int previous_index = positions.values[pos_index];
        const int period = token_index - previous_index;
        if (period <= 0 || period > config_.max_period) {
            continue;
        }

        int match_len = 1;
        if (last_match_index_by_period_[period] == token_index - 1) {
            match_len = match_len_by_period_[period] + 1;
        }
        match_len_by_period_[period] = match_len;
        last_match_index_by_period_[period] = token_index;

        hit_now = considerMatch(period, match_len, token_index) || hit_now;
    }

    positions.values.push_back(token_index);
    return result_.hit || hit_now;
}

bool OnlineRepetitionTracker::considerFinalTail() {
    if (token_count_ <= 0) {
        return result_.hit;
    }
    const int token_index = token_count_ - 1;
    bool hit_now = false;
    const int max_period = std::min(config_.max_period, token_count_ - 1);
    for (int period = 1; period <= max_period; ++period) {
        if (last_match_index_by_period_[period] != token_index) {
            continue;
        }
        const int covered = std::min(match_len_by_period_[period] + period, token_count_);
        hit_now = considerCandidate(period, covered, token_index, true) || hit_now;
    }
    return result_.hit || hit_now;
}

bool OnlineRepetitionTracker::updateMany(const std::vector<int>& token_ids) {
    bool hit = result_.hit;
    for (int token_id : token_ids) {
        hit = update(token_id) || hit;
    }
    return hit;
}

OnlineRepetitionResult detectOnlineRepetitionHitOnly(
    const std::vector<int>& token_ids,
    OnlineRepetitionConfig config) {
    OnlineRepetitionTracker tracker(config);
    for (int token_id : token_ids) {
        if (tracker.update(token_id)) {
            return normalizeResultForEnd(tracker.result(), -1);
        }
    }
    tracker.considerFinalTail();
    return normalizeResultForEnd(tracker.result(), -1);
}

OnlineRepetitionResult detectOnlineRepetitionMax(
    const std::vector<int>& token_ids,
    OnlineRepetitionConfig config) {
    OnlineRepetitionTracker tracker(config);
    tracker.updateMany(token_ids);
    tracker.considerFinalTail();
    return normalizeResultForEnd(tracker.result(), static_cast<int>(token_ids.size()));
}

}  // namespace rtp_llm
