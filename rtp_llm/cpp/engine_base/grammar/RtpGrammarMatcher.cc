#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr int32_t kDeepSeekNewlineTokenId   = 201;
constexpr int32_t kDeepSeekBlankLineTokenId = 271;
constexpr int32_t kQwenGlmNewlineTokenId    = 198;

bool isBoundaryPaddingToken(int32_t token_id) {
    return token_id == kDeepSeekNewlineTokenId || token_id == kDeepSeekBlankLineTokenId
           || token_id == kQwenGlmNewlineTokenId;
}

std::vector<int> normalizeThinkEndTokenIds(const std::vector<int>& token_ids) {
    if (token_ids.size() <= 1) {
        return token_ids;
    }

    size_t begin = 0;
    size_t end   = token_ids.size();
    while (begin + 1 < end && isBoundaryPaddingToken(token_ids[begin])) {
        ++begin;
    }
    while (begin + 1 < end && isBoundaryPaddingToken(token_ids[end - 1])) {
        --end;
    }
    return std::vector<int>(token_ids.begin() + begin, token_ids.begin() + end);
}

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
                                     std::optional<std::vector<int>>            override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)),
    think_end_token_ids_(normalizeThinkEndTokenIds(think_end_token_ids.value_or(std::vector<int>{}))),
    require_reasoning_(require_reasoning && !think_end_token_ids_.empty()),
    override_stop_tokens_(std::move(override_stop_tokens)),
    terminate_without_stop_token_(terminate_without_stop_token),
    max_rollback_tokens_(max_rollback_tokens) {
    if (require_reasoning && think_end_token_ids_.empty()) {
        RTP_LLM_LOG_WARNING("grammar reasoning requested but think_end_token_ids is missing; using plain grammar mode");
    }
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }
    think_end_lps_ = buildKmpFailureTable(think_end_token_ids_);

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
        *compiled_, override_stop_tokens_, terminate_without_stop_token_, max_rollback_tokens_);
}

std::shared_ptr<RtpGrammarMatcher> RtpGrammarMatcher::clone() const {
    auto cloned = std::make_shared<RtpGrammarMatcher>(compiled_,
                                                      require_reasoning_,
                                                      think_end_token_ids_,
                                                      override_stop_tokens_,
                                                      terminate_without_stop_token_,
                                                      max_rollback_tokens_);
    if (reasoning_initialized_) {
        cloned->initReasoning(initial_in_think_body_);
    }
    for (const int32_t token_id : accepted_tokens_history_) {
        if (!cloned->acceptToken(token_id)) {
            throw std::runtime_error("RtpGrammarMatcher clone replay failed at token " + std::to_string(token_id));
        }
    }
    cloned->finished_ = finished_;
    return cloned;
}

void RtpGrammarMatcher::initReasoning(bool in_think_body) {
    reasoning_initialized_ = true;
    initial_in_think_body_ = in_think_body;
    if (require_reasoning_) {
        tokens_after_think_end_ = in_think_body ? -1 : 0;
        think_end_match_pos_    = 0;
        reasoner_state_history_.clear();
        accepted_tokens_history_.clear();
        num_accepted_ = 0;
        finished_     = false;
    }
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    if (isPassthroughForMask()) {
        transferReasonerState(token_id);
        accepted_tokens_history_.push_back(token_id);
        ++num_accepted_;
        return true;
    }

    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
        return false;
    }
    transferReasonerState(token_id);
    accepted_tokens_history_.push_back(token_id);
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

int32_t RtpGrammarMatcher::vocabSize() const {
    return compiled_->GetTokenizerInfo().GetVocabSize();
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }

    const int active_steps = require_reasoning_ ? std::min<int>(n, std::max<int>(0, tokens_after_think_end_)) : n;
    if (active_steps > 0) {
        matcher_->Rollback(active_steps);
    }
    for (int i = 0; i < n; ++i) {
        rollbackReasonerState();
    }
    if (n >= static_cast<int>(accepted_tokens_history_.size())) {
        accepted_tokens_history_.clear();
    } else {
        accepted_tokens_history_.resize(accepted_tokens_history_.size() - n);
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
