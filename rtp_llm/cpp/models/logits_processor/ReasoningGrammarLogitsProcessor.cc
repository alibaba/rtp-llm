#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include <algorithm>
#include <limits>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorException.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr int32_t kInvalidTokenId = -1;

int32_t firstTokenOrInvalid(const std::vector<int>& token_ids) {
    return token_ids.empty() ? kInvalidTokenId : token_ids.front();
}

bool isActiveThinkState(const StreamThinkInfo& info) {
    return info.process_state == ThinkProcessState::IN_THINK || info.process_state == ThinkProcessState::CLOSING_THINK;
}

bool transitionToAfterThinkIfClosed(StreamThinkInfo& info) {
    if (!info.dfa_ptr || !info.dfa_ptr->isFinished()) {
        return false;
    }
    info.process_state = ThinkProcessState::AFTER_THINK;
    return true;
}

bool thinkEndCloseInProgress(const StreamThinkInfo& info) {
    return info.dfa_ptr && info.dfa_ptr->status() > 0;
}

bool budgetExhausted(const StreamThinkInfo& info, int tokens_emitted) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && tokens_emitted >= info.max_thinking_tokens;
}

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    if (!inputs.input_lengths.defined() || !inputs.sequence_lengths.defined()) {
        return 0;
    }
    const int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

bool drainPendingForcedThinkEnd(StreamThinkInfo& info, int32_t token_id) {
    if (info.pending_forced_think_end_token_ids.empty()
        || info.pending_forced_think_end_token_ids.front() != token_id) {
        return false;
    }
    info.pending_forced_think_end_token_ids.erase(info.pending_forced_think_end_token_ids.begin());
    return true;
}

// One think-state decision shared by sampler and spec paths. May transition
// info.process_state IN_THINK -> CLOSING_THINK / AFTER_THINK before returning.
// Concretely four outcomes:
//   - GrammarRoute   : non-think state, caller falls through to grammar mask
//   - AllowAll       : passthrough, leave the row untouched
//   - ForceEndToken  : forced_token must be the only allowed id
//   - MaskBoundaries : drop begin_think + eos
enum class ThinkAction {
    GrammarRoute,
    AllowAll,
    ForceEndToken,
    MaskBoundaries,
};

struct ThinkDecision {
    ThinkAction action       = ThinkAction::GrammarRoute;
    int32_t     forced_token = kInvalidTokenId;
    int32_t     begin_token  = kInvalidTokenId;
    int32_t     eos_token    = kInvalidTokenId;
};

ThinkDecision decideThinkMask(StreamThinkInfo& info, int tokens_emitted, int64_t eos_token_id) {
    ThinkDecision d;
    d.begin_token = firstTokenOrInvalid(info.begin_think_token_ids);
    d.eos_token   = static_cast<int32_t>(eos_token_id);

    auto force_or_boundaries = [&]() -> ThinkDecision {
        if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
            d.action = ThinkAction::MaskBoundaries;
            return d;
        }
        const auto next_idx = info.dfa_ptr->status();
        if (next_idx >= info.end_think_token_ids.size()) {
            d.action = ThinkAction::MaskBoundaries;
            return d;
        }
        d.action       = ThinkAction::ForceEndToken;
        d.forced_token = info.end_think_token_ids[next_idx];
        return d;
    };

    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            d.action = ThinkAction::GrammarRoute;
            return d;
        case ThinkProcessState::IN_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                d.action = ThinkAction::GrammarRoute;
                return d;
            }
            if (thinkEndCloseInProgress(info) || budgetExhausted(info, tokens_emitted)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                return force_or_boundaries();
            }
            d.action = ThinkAction::MaskBoundaries;
            return d;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                d.action = ThinkAction::GrammarRoute;
                return d;
            }
            return force_or_boundaries();
        }
    }
    d.action = ThinkAction::GrammarRoute;
    return d;
}

// Push a committed token through the DFA. Same logic the sampler path runs in
// updateStatus; spec path calls this on the cloned state after a successful
// accept of a non-grammar token.
void commitThinkToken(StreamThinkInfo& info, int32_t token_id) {
    if (!isActiveThinkState(info) || !info.dfa_ptr) {
        return;
    }
    info.dfa_ptr->next(token_id);
    if (info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    } else if (thinkEndCloseInProgress(info)) {
        info.process_state = ThinkProcessState::CLOSING_THINK;
    } else if (info.process_state == ThinkProcessState::CLOSING_THINK) {
        info.process_state = ThinkProcessState::IN_THINK;
    }
}

}  // namespace

ReasoningGrammarLogitsProcessor::ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                                                 int64_t          eos_token_id,
                                                                 int              max_thinking_tokens,
                                                                 std::vector<int> begin_think_token_ids,
                                                                 std::vector<int> end_think_token_ids,
                                                                 int32_t          input_length):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id) {
    think_info_.in_think_mode         = true;
    think_info_.max_thinking_tokens   = max_thinking_tokens;
    think_info_.begin_think_token_ids = std::move(begin_think_token_ids);
    think_info_.end_think_token_ids   = end_think_token_ids;
    think_info_.input_length          = input_length;
    think_info_.current_output_length = 0;
    think_info_.is_beam_search        = false;
    if (!end_think_token_ids.empty()) {
        think_info_.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
    }
    think_info_.process_state = think_info_.dfa_ptr ? ThinkProcessState::IN_THINK : ThinkProcessState::NO_THINK;
}

void ReasoningGrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       "reasoning grammar logits processor only supports single sequence decoding");
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    applyMaskLocked(inputs, start_idx);
}

void ReasoningGrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

void ReasoningGrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       "reasoning grammar accept expects one row with num_new_tokens columns");
    }

    auto tokens_cpu       = new_tokens.is_cuda() ? new_tokens.cpu() : new_tokens;
    tokens_cpu            = tokens_cpu.to(torch::kInt32).contiguous();
    const auto* token_ptr = tokens_cpu.data_ptr<int32_t>();

    std::lock_guard<std::mutex> lock(mutex_);
    if (!matcher_ || matcher_->finished()) {
        return;
    }

    for (int32_t i = 0; i < num_new_tokens; ++i) {
        const int32_t token_id = token_ptr[i];
        if (drainPendingForcedThinkEnd(think_info_, token_id)) {
            continue;
        }
        think_info_.current_output_length += 1;
        if (isActiveThinkState(think_info_)) {
            commitThinkToken(think_info_, token_id);
            continue;
        }
        acceptCommittedGrammarTokenLocked(token_id);
    }
}

bool ReasoningGrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr;
}

int ReasoningGrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return static_cast<int>(request.propose_step);
    }
    if (request.bitmask_size_int32 < static_cast<size_t>((request.vocab_size + 31) / 32)) {
        throw LogitsProcessorException(
            ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
            "reasoning grammar MTP verify: bitmask buffer smaller than model vocab (words="
                + std::to_string(request.bitmask_size_int32)
                + ", vocab=" + std::to_string(request.vocab_size) + ")");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const int  P                       = request.propose_step;
    const auto W                       = request.bitmask_size_int32;
    int        grammar_accepted_prefix = 0;
    int        cap                     = P;
    auto       think_state             = think_info_.copy();

    {
        const int32_t grammar_vocab_size = matcher_->vocabSize();
        if (grammar_vocab_size > 0 && SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
            matcher_->markFinished();
            throw LogitsProcessorException(
                ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB,
                "reasoning grammar vocab exceeds model vocab in MTP verify (grammar="
                    + std::to_string(grammar_vocab_size) + ", model_words=" + std::to_string(W) + ")");
        }
    }

    // forceTokenInBitmask asserts on out-of-range token_id. Validate eos and every
    // configured end-think token here so a misconfigured generate_config surfaces
    // as a stream error instead of aborting the worker mid-verify.
    auto token_in_range = [W](int64_t t) { return t >= 0 && static_cast<size_t>(t / 32) < W; };
    if (!token_in_range(eos_token_id_)) {
        matcher_->markFinished();
        throw LogitsProcessorException(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                                       "reasoning grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id_)
                                           + ") out of model vocab bitmask (words=" + std::to_string(W) + ")");
    }
    for (int t : think_info_.end_think_token_ids) {
        if (!token_in_range(t)) {
            matcher_->markFinished();
            throw LogitsProcessorException(
                ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                "reasoning grammar MTP verify: end_think_token_id (" + std::to_string(t)
                    + ") out of model vocab bitmask (words=" + std::to_string(W) + ")");
        }
    }

    enum class GrammarRowState {
        Active,
        AllowAll,
        Finished,
        Terminated,
        Failed,
    };

    auto fill_grammar_row = [&](int32_t* row) -> GrammarRowState {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return GrammarRowState::Finished;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return GrammarRowState::Terminated;
        }
        if (matcher_->isPassthroughForMask()) {
            return GrammarRowState::AllowAll;
        }

        const int32_t grammar_vocab_size = matcher_->vocabSize();
        const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

        int64_t  dl_shape[2];
        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
        if (!matcher_->fillBitmask(&dl, 0)) {
            return GrammarRowState::Failed;
        }
        clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
        clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.begin_think_token_ids));
        clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.end_think_token_ids));
        return GrammarRowState::Active;
    };

    auto fill_row = [&](int32_t* row) -> GrammarRowState {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        const ThinkDecision d = decideThinkMask(think_state, think_state.current_output_length, eos_token_id_);
        switch (d.action) {
            case ThinkAction::GrammarRoute:
                return fill_grammar_row(row);
            case ThinkAction::AllowAll:
                return GrammarRowState::AllowAll;
            case ThinkAction::ForceEndToken:
                forceTokenInBitmask(row, W, d.forced_token);
                return GrammarRowState::AllowAll;
            case ThinkAction::MaskBoundaries:
                clearTokenFromBitmask(row, W, d.begin_token);
                clearTokenFromBitmask(row, W, d.eos_token);
                return GrammarRowState::AllowAll;
        }
        return GrammarRowState::AllowAll;
    };

    const auto reasoner_snapshot    = matcher_->reasonerSnapshot();
    auto       rollback_provisional = [&]() noexcept {
        if (grammar_accepted_prefix > 0) {
            matcher_->rollback(grammar_accepted_prefix);
        }
        matcher_->restoreReasoner(reasoner_snapshot);
    };

    std::string verify_exception_what;
    bool        fill_bitmask_failed = false;

    try {
        for (int offset = 0; offset <= P; ++offset) {
            int32_t*              row       = request.bitmask_cpu_out + offset * W;
            const GrammarRowState row_state = fill_row(row);
            if (row_state == GrammarRowState::Failed) {
                fill_bitmask_failed = true;
                cap                 = offset;
                forceTokenInBitmask(row, W, eos_token_id_);
                break;
            }
            if (offset == P) {
                break;
            }
            if (row_state == GrammarRowState::Terminated || row_state == GrammarRowState::Finished) {
                cap = offset;
                break;
            }

            const int32_t draft_token = request.draft_tokens[offset];
            if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
                || !bitmaskAllowsToken(row, W, draft_token)) {
                cap = offset;
                break;
            }

            const bool token_belongs_to_grammar = think_state.process_state == ThinkProcessState::AFTER_THINK
                                                  || think_state.process_state == ThinkProcessState::NO_THINK;
            if (token_belongs_to_grammar) {
                if (!matcher_->acceptToken(draft_token)) {
                    cap = offset;
                    break;
                }
                ++grammar_accepted_prefix;
                think_state.current_output_length += 1;
            } else {
                if (drainPendingForcedThinkEnd(think_state, draft_token)) {
                    continue;
                }
                think_state.current_output_length += 1;
                commitThinkToken(think_state, draft_token);
            }
        }
    } catch (const std::exception& e) {
        verify_exception_what = e.what();
    } catch (...) {
        verify_exception_what = "unknown";
    }

    rollback_provisional();

    if (!verify_exception_what.empty()) {
        matcher_->markFinished();
        if (request.bitmask_cpu_out != nullptr && request.bitmask_size_int32 > 0) {
            forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id_);
        }
        throw LogitsProcessorException(ErrorCode::GRAMMAR_VERIFY_EXCEPTION,
                                       "reasoning grammar MTP verify exception: " + verify_exception_what);
    }
    if (fill_bitmask_failed) {
        matcher_->markFinished();
        throw LogitsProcessorException(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                                       "reasoning grammar matcher fillBitmask failed during MTP verify; "
                                       "matcher state corrupted");
    }
    return static_cast<int>(cap);
}

int64_t ReasoningGrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return think_info_.current_output_length;
}

void ReasoningGrammarLogitsProcessor::applyMaskLocked(const SamplerInputs& inputs, size_t batch_idx) {
    auto logits = inputs.logits[batch_idx];

    const int tokens_emitted =
        std::max(generatedTokens(inputs, batch_idx), think_info_.current_output_length);
    const ThinkDecision d = decideThinkMask(think_info_, tokens_emitted, eos_token_id_);

    switch (d.action) {
        case ThinkAction::GrammarRoute:
            applyGrammarMaskLocked(logits);
            return;
        case ThinkAction::AllowAll:
            return;
        case ThinkAction::ForceEndToken: {
            forceToken(logits, d.forced_token);
            think_info_.dfa_ptr->next(d.forced_token);
            think_info_.pending_forced_think_end_token_ids.push_back(d.forced_token);
            think_info_.current_output_length += 1;
            think_info_.process_state =
                think_info_.dfa_ptr->isFinished() ? ThinkProcessState::AFTER_THINK : ThinkProcessState::CLOSING_THINK;
            return;
        }
        case ThinkAction::MaskBoundaries:
            maskToken(logits, d.begin_token);
            maskToken(logits, d.eos_token);
            return;
    }
}

void ReasoningGrammarLogitsProcessor::applyGrammarMaskLocked(const torch::Tensor& logits) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    if (matcher_->isTerminated()) {
        forceToken(logits, eos_token_id_);
        return;
    }
    if (matcher_->isPassthroughForMask()) {
        // Reasoner-mode passthrough (require_reasoning_ && </think> not yet crossed):
        // matcher is frozen and fillBitmask returns false BY DESIGN.
        return;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        return;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    int64_t       dl_shape[2];
    DLTensor      dl = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        matcher_->markFinished();
        forceToken(logits, eos_token_id_);
        throw LogitsProcessorException(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                                       "reasoning grammar matcher fillBitmask failed; matcher state corrupted");
    }

    auto mask_options = torch::TensorOptions().dtype(torch::kBool);
    if (logits.device().is_cuda()) {
        mask_options = mask_options.pinned_memory(true);
    }
    auto           vocab_mask  = torch::empty({grammar_vocab_size}, mask_options);
    bool*          mask_ptr    = vocab_mask.data_ptr<bool>();
    const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, static_cast<size_t>(words), token_id);
    }

    auto mask = vocab_mask;
    if (mask.device() != logits.device()) {
        mask = mask.to(logits.device(), /*non_blocking=*/true);
    }
    const int64_t mask_vocab_size = std::min<int64_t>(logits.size(0), mask.size(0));
    if (mask_vocab_size > 0) {
        logits.narrow(0, 0, mask_vocab_size)
            .masked_fill_(mask.narrow(0, 0, mask_vocab_size), BaseLogitsProcessor::neg_inf);
    }
    if (mask.size(0) < logits.size(0)) {
        logits.narrow(0, mask.size(0), logits.size(0) - mask.size(0)).fill_(BaseLogitsProcessor::neg_inf);
    }

    maskToken(logits, firstTokenOrInvalid(think_info_.begin_think_token_ids));
    maskToken(logits, firstTokenOrInvalid(think_info_.end_think_token_ids));
}

void ReasoningGrammarLogitsProcessor::acceptCommittedGrammarTokenLocked(int32_t token_id) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    if (matcher_->isTerminated()) {
        if (token_id != eos_token_id_) {
            matcher_->markFinished();
            throw LogitsProcessorException(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                           "reasoning grammar received non-EOS token after terminal state "
                                               + std::to_string(token_id));
        }
        matcher_->markFinished();
        return;
    }
    if (!matcher_->acceptToken(token_id)) {
        matcher_->markFinished();
        throw LogitsProcessorException(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                                       "reasoning grammar accept_token error: parser rejected token "
                                           + std::to_string(token_id));
    }
}

void ReasoningGrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 0.0f;
}

void ReasoningGrammarLogitsProcessor::maskToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits[token_id] = BaseLogitsProcessor::neg_inf;
}

}  // namespace rtp_llm
