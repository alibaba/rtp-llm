#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

std::vector<size_t> buildKmpFailureTable(const std::vector<int>& pattern) {
    std::vector<size_t> lps(pattern.size(), 0);
    size_t              len = 0;
    for (size_t i = 1; i < pattern.size();) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len > 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

}  // namespace

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     bool                                       require_reasoning,
                                     std::optional<std::vector<int>>            think_end_token_ids,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)),
    think_end_token_ids_(think_end_token_ids.value_or(std::vector<int>{})),
    require_reasoning_(require_reasoning && !think_end_token_ids_.empty()) {
    if (require_reasoning && think_end_token_ids_.empty()) {
        RTP_LLM_LOG_WARNING("grammar reasoning requested but think_end_token_ids is missing; using plain grammar mode");
    }
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }
    think_end_lps_ = buildKmpFailureTable(think_end_token_ids_);

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
        *compiled_, std::move(override_stop_tokens), terminate_without_stop_token, max_rollback_tokens);
}

void RtpGrammarMatcher::initReasoning(bool in_think_body) {
    if (require_reasoning_) {
        tokens_after_think_end_ = in_think_body ? -1 : 0;
        think_end_match_pos_    = 0;
        reasoner_state_history_.clear();
    }
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    if (isPassthroughForMask()) {
        // Parser frozen inside think body; track reasoner KMP state and absorb the token.
        transferReasonerState(token_id);
        ++num_accepted_;
        return true;
    }

    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
        // Spec-verify DFS reacts on the bool; keep at DEBUG to avoid log floods.
        RTP_LLM_LOG_DEBUG("RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, terminated=%d",
                          token_id,
                          num_accepted_,
                          static_cast<int>(matcher_->IsTerminated()));
        return false;
    }
    transferReasonerState(token_id);
    ++num_accepted_;
    return true;
}

bool RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        if (!acceptToken(token_id)) {
            return false;
        }
    }
    return true;
}

bool RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    if (isPassthroughForMask()) {
        return false;
    }
    return matcher_->FillNextTokenBitmask(bitmask, idx);
}

bool RtpGrammarMatcher::isTerminated() const {
    return matcher_->IsTerminated();
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }

    // Only the post-think-end portion of the history corresponds to xgrammar accepts;
    // pre-think-end tokens are absorbed by the reasoner state alone.
    const int active_steps = require_reasoning_ ? std::min<int>(n, std::max<int>(0, tokens_after_think_end_)) : n;
    if (active_steps > 0) {
        matcher_->Rollback(active_steps);
    }
    for (int i = 0; i < n; ++i) {
        rollbackReasonerState();
    }
    num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
}

void RtpGrammarMatcher::transferReasonerState(int32_t token_id) noexcept {
    if (!require_reasoning_) {
        return;
    }
    reasoner_state_history_.push_back({tokens_after_think_end_, think_end_match_pos_});
    if (tokens_after_think_end_ < 0) {
        think_end_match_pos_ = nextThinkEndMatchPos(token_id);
        if (think_end_match_pos_ == think_end_token_ids_.size()) {
            tokens_after_think_end_ = 0;
            think_end_match_pos_    = 0;
        }
    } else {
        ++tokens_after_think_end_;
    }
}

void RtpGrammarMatcher::rollbackReasonerState() noexcept {
    if (!require_reasoning_ || reasoner_state_history_.empty()) {
        return;
    }
    auto prev = reasoner_state_history_.back();
    reasoner_state_history_.pop_back();
    tokens_after_think_end_ = prev.tokens_after_think_end;
    think_end_match_pos_    = prev.think_end_match_pos;
}

size_t RtpGrammarMatcher::nextThinkEndMatchPos(int32_t token_id) const noexcept {
    if (think_end_token_ids_.empty()) {
        return 0;
    }
    size_t pos = think_end_match_pos_;
    if (pos >= think_end_token_ids_.size()) {
        pos = think_end_lps_.empty() ? 0 : think_end_lps_.back();
    }
    while (pos > 0 && token_id != think_end_token_ids_[pos]) {
        pos = think_end_lps_[pos - 1];
    }
    if (token_id == think_end_token_ids_[pos]) {
        return pos + 1;
    }
    return token_id == think_end_token_ids_.front() ? 1 : 0;
}

}  // namespace rtp_llm
