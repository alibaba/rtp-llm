#include "rtp_llm/cpp/models/logits_processor/ThinkModeStateMachine.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace think_state_machine {

namespace {

bool dfaClosing(const StreamThinkInfo& info) {
    return info.dfa_ptr && info.dfa_ptr->status() > 0;
}

bool budgetExhausted(const StreamThinkInfo& info, int tokens_emitted) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && tokens_emitted >= info.max_thinking_tokens;
}

bool transitionToAfterThinkIfClosed(StreamThinkInfo& info) {
    if (!info.dfa_ptr || !info.dfa_ptr->isFinished()) {
        return false;
    }
    info.process_state = ThinkProcessState::AFTER_THINK;
    return true;
}

ThinkDecision forceEndOrFallback(StreamThinkInfo& info) {
    ThinkDecision d;
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        d.action = ThinkDecisionAction::MaskBoundaries;
        return d;
    }
    const auto next_idx = info.dfa_ptr->status();
    if (next_idx >= info.end_think_token_ids.size()) {
        d.action = ThinkDecisionAction::MaskBoundaries;
        return d;
    }
    d.action       = ThinkDecisionAction::ForceEndToken;
    d.forced_token = info.end_think_token_ids[next_idx];
    return d;
}

}  // namespace

bool isActiveThinkState(const StreamThinkInfo& info) {
    return info.process_state == ThinkProcessState::IN_THINK || info.process_state == ThinkProcessState::CLOSING_THINK;
}

int32_t firstTokenOrInvalid(const std::vector<int>& token_ids) {
    return token_ids.empty() ? -1 : token_ids.front();
}

bool forcedTokenInVocab(int32_t token_id, size_t vocab_size) {
    return token_id >= 0 && static_cast<size_t>(token_id) < vocab_size;
}

bool forcedEndTokenUsable(int32_t forced_token, size_t vocab_size, std::atomic<bool>& warned_once, const char* who) {
    if (forcedTokenInVocab(forced_token, vocab_size)) {
        return true;
    }
    if (!warned_once.exchange(true, std::memory_order_relaxed)) {
        RTP_LLM_LOG_WARNING("%sforceThinkEndToken: end_think_token_id=%d out of vocab_size=%zu; "
                            "downgrade to boundary masking",
                            who,
                            forced_token,
                            vocab_size);
    }
    return false;
}

ThinkDecision decideThinkMask(StreamThinkInfo& info, int tokens_emitted) {
    ThinkDecision d;
    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            d.action = ThinkDecisionAction::OutsideThink;
            return d;
        case ThinkProcessState::IN_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                d.action = ThinkDecisionAction::OutsideThink;
                return d;
            }
            if (dfaClosing(info) || budgetExhausted(info, tokens_emitted)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                return forceEndOrFallback(info);
            }
            d.action = ThinkDecisionAction::InsideThink;
            return d;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                d.action = ThinkDecisionAction::OutsideThink;
                return d;
            }
            return forceEndOrFallback(info);
        }
    }
    d.action = ThinkDecisionAction::OutsideThink;
    return d;
}

bool drainPendingForcedEnd(StreamThinkInfo& info, int32_t token_id, PendingMismatchPolicy policy, const char* who) {
    auto& pending = info.pending_forced_think_end_token_ids;
    if (pending.empty()) {
        return false;
    }
    const int32_t expected = pending.front();
    if (expected == token_id) {
        pending.erase(pending.begin());
        return true;
    }
    if (policy == PendingMismatchPolicy::KeepPending) {
        return false;
    }
    pending.erase(pending.begin());
    RTP_LLM_LOG_WARNING("%sforced think end token mismatch, expected=%d actual=%d, trust precommitted state",
                        who ? who : "",
                        expected,
                        token_id);
    return true;
}

void advanceThinkDfa(StreamThinkInfo& info, int32_t token_id) {
    // NOTE: no max_thinking_tokens guard. ThinkMode only builds dfa_ptr when
    // max_thinking_tokens > 0, so the !dfa_ptr check already covers it there.
    // ReasoningGrammar builds dfa_ptr whenever end_think tokens exist (budget or
    // not) and relies on advancing it to detect the natural </think> and switch to
    // grammar routing; gating on max_thinking_tokens would strand it in think mode.
    if (!isActiveThinkState(info) || !info.dfa_ptr) {
        return;
    }
    info.dfa_ptr->next(token_id);
    if (info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    } else if (dfaClosing(info)) {
        info.process_state = ThinkProcessState::CLOSING_THINK;
    } else if (info.process_state == ThinkProcessState::CLOSING_THINK) {
        info.process_state = ThinkProcessState::IN_THINK;
    }
}

}  // namespace think_state_machine
}  // namespace rtp_llm
