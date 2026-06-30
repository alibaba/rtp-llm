#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

namespace rtp_llm {
namespace think_state_machine {

// Abstract state-machine action shared by ThinkModeLogitsProcessor (boundary-only
// masking) and ReasoningGrammarLogitsProcessor (grammar-route + boundary-only).
// Each caller picks the concrete token set: e.g. ThinkMode masks {begin,end} on
// OutsideThink, ReasoningGrammar routes to the grammar matcher there instead.
enum class ThinkDecisionAction {
    OutsideThink,    // NO_THINK or AFTER_THINK
    InsideThink,     // IN_THINK and close-in-progress not yet started
    ForceEndToken,   // CLOSING_THINK with a valid next end-think token
    MaskBoundaries,  // CLOSING_THINK fallback (dfa finished early, idx OOR, etc)
};

struct ThinkDecision {
    ThinkDecisionAction action;
    int32_t             forced_token = -1;  // valid iff action == ForceEndToken
};

// True when info is mid-think (process_state ∈ {IN_THINK, CLOSING_THINK}).
bool isActiveThinkState(const StreamThinkInfo& info);

// First element of token_ids, or -1 if empty.
int32_t firstTokenOrInvalid(const std::vector<int>& token_ids);

// Forced-end token guard: caller falls back to MaskBoundaries when this is false,
// otherwise the all-zero row from forceTokenInBitmask would silently kill sampling.
bool forcedTokenInVocab(int32_t token_id, size_t vocab_size);

// Forced-end OOV check + warn-once. Returns true when forced_token is usable;
// false means the caller must downgrade to boundary masking (a forced OOV token
// would mask the entire row). Logs at most once per `warned_once` flag; `who` is a
// short prefix to disambiguate the warning source (e.g. "" or "reasoning grammar ").
bool forcedEndTokenUsable(int32_t forced_token, size_t vocab_size, std::atomic<bool>& warned_once, const char* who);

// Pure transition. May mutate info.process_state (IN_THINK → CLOSING_THINK,
// CLOSING_THINK → AFTER_THINK). Does NOT advance the DFA.
ThinkDecision decideThinkMask(StreamThinkInfo& info, int tokens_emitted);

// Mismatch handling when the committed token differs from the pending forced-end token.
//   PopAndWarn  : ThinkMode — trust the precommitted state, pop the entry, warn once-ish.
//   KeepPending : ReasoningGrammar — leave the entry in place (caller treats token normally).
enum class PendingMismatchPolicy {
    PopAndWarn,
    KeepPending,
};

// Drains at most one pending forced-end token. Returns true iff `token_id` was consumed as the
// pending forced-end (caller should then SKIP its normal commit/advance for this token).
// Behavior:
//   empty pending           -> false
//   front matches token_id  -> pop, return true
//   front mismatches token  -> policy-dependent: PopAndWarn pops+warns+true; KeepPending false.
// `who` is a short log prefix used only by PopAndWarn (e.g. "" or "reasoning grammar ").
bool drainPendingForcedEnd(StreamThinkInfo& info, int32_t token_id, PendingMismatchPolicy policy, const char* who);

// Advance the think DFA by one committed token and transition process_state.
// No-ops for inactive states / missing DFA. Intentionally NOT gated on
// max_thinking_tokens: ReasoningGrammar advances the DFA to detect the natural
// </think> boundary even without a thinking budget (ThinkMode only creates a DFA
// when max_thinking_tokens > 0, so the missing-DFA check covers it there).
// Callers manage current_output_length and pending_forced_think_end_token_ids
// themselves (the two processors disagree on warn-vs-silent on pending mismatch).
void advanceThinkDfa(StreamThinkInfo& info, int32_t token_id);

}  // namespace think_state_machine
}  // namespace rtp_llm
