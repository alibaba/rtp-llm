#include "rtp_llm/cpp/models/logits_processor/xgrammar/ReasoningGate.h"

#include <stdexcept>

namespace rtp_llm {

namespace {

// KMP failure-function (longest proper prefix that is also a suffix).
std::vector<size_t> buildKmpFailureTable(const std::vector<int>& pattern) {
    std::vector<size_t> lps(pattern.size(), 0);
    size_t              k = 0;
    for (size_t i = 1; i < pattern.size(); ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = lps[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        lps[i] = k;
    }
    return lps;
}

}  // namespace

ReasoningGate::ReasoningGate(std::vector<int> end_token_ids, bool in_think_body, size_t max_history):
    end_token_ids_(std::move(end_token_ids)),
    tokens_after_end_(in_think_body ? -1 : 0),
    match_pos_(0),
    max_history_(max_history == 0 ? 1 : max_history) {
    if (end_token_ids_.empty()) {
        throw std::invalid_argument("ReasoningGate requires non-empty end_token_ids");
    }
    end_lps_ = buildKmpFailureTable(end_token_ids_);
}

void ReasoningGate::observe(int32_t token_id, bool forwarded_to_matcher) {
    history_.push_back({tokens_after_end_, match_pos_, !inPassthrough()});
    // Cap rollback window so long sessions don't grow history_ unboundedly.
    // Matches the matcher's own max_rollback_tokens contract; older snapshots
    // are unrecoverable anyway because xgrammar has dropped them.
    while (history_.size() > max_history_) {
        history_.pop_front();
    }
    if (inPassthrough()) {
        // Parser frozen; advance the KMP scanner instead.
        match_pos_ = nextEndMatchPos(token_id);
        if (match_pos_ == end_token_ids_.size()) {
            tokens_after_end_ = 0;
            match_pos_        = 0;
        }
        return;
    }
    (void)forwarded_to_matcher;  // recorded via was_active in the snapshot.
    ++tokens_after_end_;
}

int ReasoningGate::rollback(int n) {
    int matcher_steps = 0;
    for (int i = 0; i < n; ++i) {
        if (history_.empty()) {
            break;
        }
        const Snapshot prev = history_.back();
        history_.pop_back();
        if (prev.was_active) {
            ++matcher_steps;
        }
        tokens_after_end_ = prev.tokens_after_end;
        match_pos_        = prev.match_pos;
    }
    return matcher_steps;
}

size_t ReasoningGate::nextEndMatchPos(int32_t token_id) const noexcept {
    size_t k = match_pos_;
    while (k > 0 && end_token_ids_[k] != token_id) {
        k = end_lps_[k - 1];
    }
    if (end_token_ids_[k] == token_id) {
        ++k;
    }
    return k;
}

}  // namespace rtp_llm
