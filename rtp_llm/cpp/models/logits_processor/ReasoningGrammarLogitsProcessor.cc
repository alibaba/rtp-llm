#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include <algorithm>
#include <atomic>
#include <limits>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeStateMachine.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

namespace tsm = ::rtp_llm::think_state_machine;

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    if (!inputs.input_lengths.defined() || !inputs.sequence_lengths.defined()) {
        return 0;
    }
    const int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

}  // namespace

ThinkRouter::ThinkRouter(int              max_thinking_tokens,
                         std::vector<int> begin_think_token_ids,
                         std::vector<int> end_think_token_ids,
                         int32_t          input_length,
                         int64_t          eos_token_id):
    eos_token_id_(eos_token_id) {
    info_.in_think_mode         = true;
    info_.max_thinking_tokens   = max_thinking_tokens;
    info_.begin_think_token_ids = std::move(begin_think_token_ids);
    info_.end_think_token_ids   = end_think_token_ids;
    info_.input_length          = input_length;
    info_.current_output_length = 0;
    info_.is_beam_search        = false;
    if (!end_think_token_ids.empty()) {
        info_.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
    }
    info_.process_state = info_.dfa_ptr ? ThinkProcessState::IN_THINK : ThinkProcessState::NO_THINK;
}

// Maps the shared think state-machine decision onto the reasoning-grammar action set.
// Distinct name from tsm::decideThinkMask to avoid the two reading as the same call.
ReasoningGrammarDecision ThinkRouter::decide(StreamThinkInfo& state, int64_t eos_token_id, int tokens_emitted) {
    ReasoningGrammarDecision d;
    d.begin_token = tsm::firstTokenOrInvalid(state.begin_think_token_ids);
    d.eos_token   = static_cast<int32_t>(eos_token_id);

    const auto base = tsm::decideThinkMask(state, tokens_emitted);
    switch (base.action) {
        case tsm::ThinkDecisionAction::OutsideThink:
            d.action = ReasoningGrammarAction::GrammarRoute;
            return d;
        case tsm::ThinkDecisionAction::InsideThink:
            d.action = ReasoningGrammarAction::MaskBoundaries;
            return d;
        case tsm::ThinkDecisionAction::ForceEndToken:
            d.action       = ReasoningGrammarAction::ForceEndToken;
            d.forced_token = base.forced_token;
            return d;
        case tsm::ThinkDecisionAction::MaskBoundaries:
            d.action = ReasoningGrammarAction::MaskBoundaries;
            return d;
    }
    d.action = ReasoningGrammarAction::GrammarRoute;
    return d;
}

bool ThinkRouter::tokenBelongsToGrammar(const StreamThinkInfo& state) {
    return state.process_state == ThinkProcessState::AFTER_THINK || state.process_state == ThinkProcessState::NO_THINK;
}

void ThinkRouter::advanceForVerify(StreamThinkInfo& state, int32_t token_id) {
    // ReasoningGrammar drains only on equality (KeepPending); mismatch leaves the pending token.
    if (tsm::drainPendingForcedEnd(state, token_id, tsm::PendingMismatchPolicy::KeepPending, /*who=*/"")) {
        return;
    }
    state.current_output_length += 1;
    tsm::advanceThinkDfa(state, token_id);
}

void ThinkRouter::commitForcedEnd(int32_t forced_token) {
    info_.dfa_ptr->next(forced_token);
    info_.pending_forced_think_end_token_ids.push_back(forced_token);
    info_.current_output_length += 1;
    info_.process_state =
        info_.dfa_ptr->isFinished() ? ThinkProcessState::AFTER_THINK : ThinkProcessState::CLOSING_THINK;
}

ThinkRouter::CommitRoute ThinkRouter::commitToken(int32_t token_id) {
    if (tsm::drainPendingForcedEnd(info_, token_id, tsm::PendingMismatchPolicy::KeepPending, /*who=*/"")) {
        return CommitRoute::ConsumedByThink;
    }
    info_.current_output_length += 1;
    if (tsm::isActiveThinkState(info_)) {
        tsm::advanceThinkDfa(info_, token_id);
        return CommitRoute::ConsumedByThink;
    }
    return CommitRoute::RouteToGrammar;
}

ThinkRouter::VerifyCursor::VerifyCursor(StreamThinkInfo state, int64_t eos_token_id):
    state_(std::move(state)), eos_token_id_(eos_token_id) {}

ReasoningGrammarDecision ThinkRouter::VerifyCursor::decideForMask() {
    return ThinkRouter::decide(state_, eos_token_id_, state_.current_output_length);
}

bool ThinkRouter::VerifyCursor::tokenBelongsToGrammar() const {
    return ThinkRouter::tokenBelongsToGrammar(state_);
}

void ThinkRouter::VerifyCursor::commitThinkToken(int32_t token_id) {
    ThinkRouter::advanceForVerify(state_, token_id);
}

void ThinkRouter::VerifyCursor::commitGrammarToken() {
    state_.current_output_length += 1;
}

ThinkRouter::VerifyCursor ThinkRouter::verifyCursor() const {
    return VerifyCursor(info_.copy(), eos_token_id_);
}

ReasoningGrammarLogitsProcessor::ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                                                 int64_t                            eos_token_id,
                                                                 int                                max_thinking_tokens,
                                                                 std::vector<int> begin_think_token_ids,
                                                                 std::vector<int> end_think_token_ids,
                                                                 int32_t          input_length):
    mask_core_(std::move(matcher), eos_token_id),
    eos_token_id_(eos_token_id),
    think_(max_thinking_tokens,
           std::move(begin_think_token_ids),
           std::move(end_think_token_ids),
           input_length,
           eos_token_id) {}

void ReasoningGrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (hasError()) {
        return;
    }
    if (!mask_core_.matcher()) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        setError(ErrorCode::INVALID_PARAMS,
                 "reasoning grammar logits processor only supports single sequence decoding");
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        applyMaskLocked(inputs, start_idx, local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
    }
}

void ReasoningGrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (hasError()) {
        return;
    }
    if (num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        setError(ErrorCode::INVALID_PARAMS, "reasoning grammar accept expects one row with num_new_tokens columns");
        return;
    }

    auto tokens_cpu       = new_tokens.is_cuda() ? new_tokens.cpu() : new_tokens;
    tokens_cpu            = tokens_cpu.to(torch::kInt32).contiguous();
    const auto* token_ptr = tokens_cpu.data_ptr<int32_t>();

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!mask_core_.matcher() || mask_core_.finished()) {
            return;
        }

        for (int32_t i = 0; i < num_new_tokens; ++i) {
            const int32_t token_id = token_ptr[i];
            if (think_.commitToken(token_id) == ThinkRouter::CommitRoute::ConsumedByThink) {
                continue;
            }
            acceptCommittedGrammarTokenLocked(token_id, local_err);
            if (local_err.hasError()) {
                break;
            }
        }
    }
    if (local_err.hasError()) {
        setError(local_err);
    }
}

bool ReasoningGrammarLogitsProcessor::isSpecVerifyEligible() const {
    return mask_core_.matcher() != nullptr;
}

int ReasoningGrammarLogitsProcessor::runSpecVerifyLocked(const SpecLogitsProcessorRequest& request,
                                                         ErrorInfo&                        out_err) {
    if (auto err = mask_core_.validateMatcherInvariantsLocked(request, think_.endThinkTokenIds()); err.hasError()) {
        out_err = err;
        return 0;
    }

    const int  P      = request.propose_step;
    const auto W      = request.bitmask_size_int32;
    auto       verify = think_.verifyCursor();

    // Reuse GrammarMaskCore::RowState directly: the verify loop below only distinguishes
    // Failed / {Terminated,Finished} / non-terminal, so think-routed "allow-all" rows
    // (boundary mask, force-end, inside-think) all map to RowState::Active = non-terminal.
    using RowState = GrammarMaskCore::RowState;

    auto fill_grammar_row = [&](int32_t* row) -> RowState {
        const RowState gs = mask_core_.fillGrammarRowLocked(row, W, request.vocab_size);
        if (gs == RowState::Active) {
            clearTokenFromBitmask(row, W, tsm::firstTokenOrInvalid(verify.beginThinkTokenIds()));
            clearTokenFromBitmask(row, W, tsm::firstTokenOrInvalid(verify.endThinkTokenIds()));
        }
        return gs;
    };

    auto fill_row = [&](int32_t* row) -> RowState {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        const ReasoningGrammarDecision d = verify.decideForMask();
        switch (d.action) {
            case ReasoningGrammarAction::GrammarRoute:
                return fill_grammar_row(row);
            case ReasoningGrammarAction::AllowAll:
                return RowState::Active;
            case ReasoningGrammarAction::ForceEndToken:
                forceTokenInBitmask(row, W, d.forced_token);
                return RowState::Active;
            case ReasoningGrammarAction::MaskBoundaries:
                clearTokenFromBitmask(row, W, d.begin_token);
                clearTokenFromBitmask(row, W, d.eos_token);
                return RowState::Active;
        }
        return RowState::Active;
    };

    // Shared snapshot/rollback/exception scaffolding lives in GrammarMaskCore; here we only
    // supply the think-routed per-offset walk. grammar_accepted_prefix counts only grammar
    // accepts (think-DFA advances run on the local VerifyCursor and need no rollback).
    return mask_core_.runSpecVerifyGuarded(
        request.bitmask_cpu_out,
        W,
        "reasoning grammar",
        [&](int& grammar_accepted_prefix, ErrorInfo& walk_err) -> int {
            int cap = P;
            for (int offset = 0; offset <= P; ++offset) {
                int32_t*       row       = request.bitmask_cpu_out + offset * W;
                const RowState row_state = fill_row(row);
                if (row_state == RowState::Failed) {
                    // fillGrammarRowLocked already markFinished'd the matcher and forced eos into
                    // the grammar row; force eos across the full model vocab and surface the error.
                    forceTokenInBitmask(row, W, eos_token_id_);
                    walk_err = ErrorInfo(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                                         "reasoning grammar matcher fillBitmask failed during MTP verify; "
                                         "matcher state corrupted");
                    cap      = offset;
                    break;
                }
                if (offset == P) {
                    break;
                }
                if (row_state == RowState::Terminated || row_state == RowState::Finished) {
                    cap = offset;
                    break;
                }

                const int32_t draft_token = request.draft_tokens[offset];
                if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
                    || !bitmaskAllowsToken(row, W, draft_token)) {
                    cap = offset;
                    break;
                }

                if (verify.tokenBelongsToGrammar()) {
                    if (!mask_core_.acceptToken(draft_token)) {
                        cap = offset;
                        break;
                    }
                    ++grammar_accepted_prefix;
                    verify.commitGrammarToken();
                } else {
                    verify.commitThinkToken(draft_token);
                }
            }
            return cap;
        },
        out_err);
}

int ReasoningGrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (hasError()) {
        return 0;
    }
    if (!mask_core_.matcher() || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return static_cast<int>(request.propose_step);
    }
    if (auto err = mask_core_.preflightSpecRequest(request); err.hasError()) {
        setError(err);
        return 0;
    }

    ErrorInfo local_err;
    int       cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cap_out = runSpecVerifyLocked(request, local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
        return 0;
    }
    return cap_out;
}

int64_t ReasoningGrammarLogitsProcessor::committedOutputLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Stream output length = think-phase tokens + grammar-phase tokens. Intentionally tracks the
    // think router's output counter (not mask_core_.acceptedTokenLen()), because think-phase
    // tokens are committed to the stream but never fed to the grammar matcher.
    return think_.committedOutputLen();
}

void ReasoningGrammarLogitsProcessor::applyMaskLocked(const SamplerInputs& inputs,
                                                      size_t               batch_idx,
                                                      ErrorInfo&           out_err) {
    auto logits = inputs.logits[batch_idx];

    const int tokens_emitted =
        std::max(generatedTokens(inputs, batch_idx), static_cast<int>(think_.committedOutputLen()));
    ReasoningGrammarDecision d = think_.decideForMask(tokens_emitted);

    // OOV forced_token → MaskBoundaries; do NOT advance dfa_ptr / push pending (would corrupt DFA).
    static std::atomic<bool> warned_force_oob{false};
    if (d.action == ReasoningGrammarAction::ForceEndToken
        && !tsm::forcedEndTokenUsable(d.forced_token, inputs.vocab_size, warned_force_oob, "reasoning grammar ")) {
        d.action = ReasoningGrammarAction::MaskBoundaries;
    }

    switch (d.action) {
        case ReasoningGrammarAction::GrammarRoute:
            applyGrammarMaskLocked(logits, out_err);
            return;
        case ReasoningGrammarAction::AllowAll:
            return;
        case ReasoningGrammarAction::ForceEndToken: {
            // Two-phase force-close protocol (paired with updateStatus):
            //   phase 1 (here, mask time): forceToken locks the sampler to forced_token, so we
            //     can already pre-advance the think DFA + state and push the token to pending.
            //     This is required because the *next* step's mask decision depends on the closing
            //     progress being reflected now, before the token is actually committed.
            //   phase 2 (updateStatus): ThinkRouter::commitToken drains this pending entry for
            //     reconciliation. The two phases MUST stay paired — do not move this advance into
            //     updateStatus, or the next mask step would route on stale think state.
            GrammarMaskCore::forceToken(logits, d.forced_token);
            think_.commitForcedEnd(d.forced_token);
            return;
        }
        case ReasoningGrammarAction::MaskBoundaries:
            GrammarMaskCore::maskToken(logits, d.begin_token);
            GrammarMaskCore::maskToken(logits, d.eos_token);
            return;
    }
}

void ReasoningGrammarLogitsProcessor::applyGrammarMaskLocked(const torch::Tensor& logits, ErrorInfo& out_err) {
    if (!mask_core_.matcher() || mask_core_.finished()) {
        return;
    }
    if (mask_core_.isTerminated()) {
        GrammarMaskCore::forceToken(logits, eos_token_id_);
        return;
    }
    if (mask_core_.isPassthroughForMask()) {
        return;
    }

    mask_core_.applyMaskLocked(logits, out_err);
    GrammarMaskCore::maskToken(logits, tsm::firstTokenOrInvalid(think_.beginThinkTokenIds()));
    GrammarMaskCore::maskToken(logits, tsm::firstTokenOrInvalid(think_.endThinkTokenIds()));
}

void ReasoningGrammarLogitsProcessor::acceptCommittedGrammarTokenLocked(int32_t token_id, ErrorInfo& out_err) {
    mask_core_.acceptCommittedLocked(&token_id, 1, out_err);
}

}  // namespace rtp_llm
