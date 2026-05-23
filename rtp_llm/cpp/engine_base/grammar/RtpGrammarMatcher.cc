#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     bool                                       require_reasoning,
                                     std::optional<int32_t>                     think_end_id,
                                     std::optional<std::vector<int>>            override_stop_tokens,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)),
    require_reasoning_(require_reasoning && think_end_id.has_value()),
    think_end_id_(think_end_id) {
    if (require_reasoning && !think_end_id.has_value()) {
        RTP_LLM_LOG_WARNING("grammar reasoning requested but think_end_id is missing; using plain grammar mode");
    }
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(*compiled_,
                                                          std::move(override_stop_tokens),
                                                          /*terminate_without_stop_token=*/false,
                                                          max_rollback_tokens);
}

void RtpGrammarMatcher::initReasoning(bool in_think_body) {
    if (require_reasoning_) {
        tokens_after_think_end_ = in_think_body ? -1 : 0;
    }
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    if (isPassthroughForMask()) {
        transferReasonerState(token_id);
        ++num_accepted_;
        return true;
    }

    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
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
    num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
}

void RtpGrammarMatcher::transferReasonerState(int32_t token_id) noexcept {
    if (!require_reasoning_) {
        return;
    }
    if (tokens_after_think_end_ < 0) {
        if (think_end_id_.has_value() && token_id == *think_end_id_) {
            tokens_after_think_end_ = 0;
        }
    } else {
        ++tokens_after_think_end_;
    }
}

void RtpGrammarMatcher::rollbackReasonerState() noexcept {
    if (!require_reasoning_) {
        return;
    }
    if (tokens_after_think_end_ == 0) {
        tokens_after_think_end_ = -1;
    } else if (tokens_after_think_end_ > 0) {
        --tokens_after_think_end_;
    }
}

}  // namespace rtp_llm
