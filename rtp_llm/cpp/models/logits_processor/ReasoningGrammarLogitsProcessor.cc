#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include <algorithm>
#include <limits>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/grammar/BitmaskUtils.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
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

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    if (!inputs.input_lengths.defined() || !inputs.sequence_lengths.defined()) {
        return 0;
    }
    const int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

bool thinkBudgetExhausted(const SamplerInputs& inputs, size_t batch_idx, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.end_think_token_ids.empty() || info.max_thinking_tokens <= 0) {
        return false;
    }
    const int observed_output_tokens = std::max(generatedTokens(inputs, batch_idx), info.current_output_length);
    return observed_output_tokens >= info.max_thinking_tokens;
}

bool specThinkBudgetExhausted(const StreamThinkInfo& info) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && info.current_output_length >= info.max_thinking_tokens;
}

// Drains the head of the pending-forced-think-end queue iff `token_id` is the
// expected next token. Mismatches fall through (no warn): the matcher's KMP
// state will reject naturally, surfacing the bug instead of masking it.
bool drainPendingForcedThinkEnd(StreamThinkInfo& info, int32_t token_id) {
    if (info.pending_forced_think_end_token_ids.empty()
        || info.pending_forced_think_end_token_ids.front() != token_id) {
        return false;
    }
    info.pending_forced_think_end_token_ids.erase(info.pending_forced_think_end_token_ids.begin());
    return true;
}

bool forceThinkEndTokenInBitmask(int32_t* row, size_t words, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }
    forceTokenInBitmask(row, words, info.end_think_token_ids[next_token_idx]);
    return true;
}

void applyThinkSpecRowMask(int32_t* row, size_t words, StreamThinkInfo& info, int64_t eos_token_id) {
    std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);

    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            return;
        case ThinkProcessState::IN_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                return;
            }
            if (thinkEndCloseInProgress(info) || specThinkBudgetExhausted(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                if (forceThinkEndTokenInBitmask(row, words, info)) {
                    return;
                }
            }
            clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
            clearTokenFromBitmask(row, words, eos_token_id);
            return;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                return;
            }
            if (forceThinkEndTokenInBitmask(row, words, info)) {
                return;
            }
            clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
            clearTokenFromBitmask(row, words, eos_token_id);
            return;
        }
    }
}

void advanceThinkStateForSpec(StreamThinkInfo& info, int32_t token_id) {
    if (drainPendingForcedThinkEnd(info, token_id)) {
        return;
    }
    info.current_output_length += 1;
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
                                                                 int64_t                            eos_token_id,
                                                                 int                                max_thinking_tokens,
                                                                 std::vector<int> begin_think_token_ids,
                                                                 std::vector<int> end_think_token_ids,
                                                                 int32_t          input_length,
                                                                 LogitsProcessorFactory::ErrorReporter error_reporter):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id) {
    if (error_reporter) {
        setErrorReporter(std::move(error_reporter));
    }
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
        reportErrorOnce(ErrorCode::INVALID_PARAMS,
                        "reasoning grammar logits processor only supports single sequence decoding",
                        /*stream_lock_held=*/false);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        applyMaskLocked(inputs, start_idx);
    }
    flushError(/*stream_lock_held=*/false);
}

void ReasoningGrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

void ReasoningGrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS,
                        "reasoning grammar accept expects one row with num_new_tokens columns",
                        /*stream_lock_held=*/true);
        return;
    }

    auto tokens_cpu       = new_tokens.is_cuda() ? new_tokens.cpu() : new_tokens;
    tokens_cpu            = tokens_cpu.to(torch::kInt32).contiguous();
    const auto* token_ptr = tokens_cpu.data_ptr<int32_t>();

    {
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
                if (think_info_.dfa_ptr) {
                    think_info_.dfa_ptr->next(token_id);
                    if (think_info_.dfa_ptr->isFinished()) {
                        think_info_.process_state = ThinkProcessState::AFTER_THINK;
                    } else if (thinkEndCloseInProgress(think_info_)) {
                        think_info_.process_state = ThinkProcessState::CLOSING_THINK;
                    } else if (think_info_.process_state == ThinkProcessState::CLOSING_THINK) {
                        think_info_.process_state = ThinkProcessState::IN_THINK;
                    }
                }
                continue;
            }

            acceptCommittedGrammarTokenLocked(token_id);
            if (reported_error_.load(std::memory_order_relaxed)) {
                break;
            }
        }
    }
    flushError(/*stream_lock_held=*/true);
}

bool ReasoningGrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int ReasoningGrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }
    if (reported_error_.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (request.bitmask_size_int32 < static_cast<size_t>((request.vocab_size + 31) / 32)) {
        RTP_LLM_LOG_WARNING("[grammar] reasoning tryAcceptAndFillBitmask: bitmask buffer too small "
                            "(words=%zu vocab=%zu); skipping verify",
                            request.bitmask_size_int32,
                            request.vocab_size);
        reported_error_.store(true, std::memory_order_relaxed);
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "reasoning grammar MTP verify: bitmask buffer smaller than model vocab",
                               /*stream_lock_held=*/false);
        return 0;
    }

    const int return_value = [&]() -> int {
        std::lock_guard<std::mutex> lock(mutex_);

        const int  P                       = request.propose_step;
        const auto W                       = request.bitmask_size_int32;
        int        grammar_accepted_prefix = 0;
        int        cap                     = P;
        auto       think_state             = think_info_.copy();

        {
            const int32_t grammar_vocab_size = matcher_->vocabSize();
            if (grammar_vocab_size > 0 && SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
                RTP_LLM_LOG_WARNING("[grammar] reasoning MTP verify: grammar vocab (%d) exceeds "
                                    "model vocab bitmask (%zu words)",
                                    grammar_vocab_size,
                                    W);
                reported_error_.store(true, std::memory_order_relaxed);
                matcher_->markFinished();
                pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
                pending_error_msg_  = "reasoning grammar vocab exceeds model vocab in MTP verify";
                return 0;
            }
        }

        // forceTokenInBitmask asserts on out-of-range token_id. Validate eos and every
        // configured end-think token here so a misconfigured generate_config surfaces
        // as a stream error instead of aborting the worker mid-verify.
        auto token_in_range = [W](int64_t t) { return t >= 0 && static_cast<size_t>(t / 32) < W; };
        if (!token_in_range(eos_token_id_)) {
            RTP_LLM_LOG_WARNING("[grammar] reasoning MTP verify: eos_token_id (%ld) out of bitmask range "
                                "(words=%zu)",
                                static_cast<long>(eos_token_id_),
                                W);
            reported_error_.store(true, std::memory_order_relaxed);
            matcher_->markFinished();
            pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
            pending_error_msg_  = "reasoning grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id_)
                                 + ") out of model vocab bitmask";
            return 0;
        }
        for (int t : think_info_.end_think_token_ids) {
            if (!token_in_range(t)) {
                RTP_LLM_LOG_WARNING("[grammar] reasoning MTP verify: end_think_token_id (%d) out of bitmask range "
                                    "(words=%zu)",
                                    t,
                                    W);
                reported_error_.store(true, std::memory_order_relaxed);
                matcher_->markFinished();
                pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
                pending_error_msg_  = "reasoning grammar MTP verify: end_think_token_id (" + std::to_string(t)
                                     + ") out of model vocab bitmask";
                return 0;
            }
        }

        enum class GrammarRowState {
            Active,
            AllowAll,  // passthrough / non-grammar state; row stays allow-all
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
                // Reasoner passthrough: parser frozen, allow-all is the correct grammar row.
                return GrammarRowState::AllowAll;
            }

            const int32_t grammar_vocab_size = matcher_->vocabSize();
            const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

            int64_t  dl_shape[2];
            DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
            if (!matcher_->fillBitmask(&dl, 0)) {
                // Indeterminate matcher state: surface the failure with EOS-only row.
                return GrammarRowState::Failed;
            }
            clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
            clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.begin_think_token_ids));
            clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.end_think_token_ids));
            return GrammarRowState::Active;
        };

        auto fill_row = [&](int32_t* row) -> GrammarRowState {
            applyThinkSpecRowMask(row, W, think_state, eos_token_id_);
            if (think_state.process_state == ThinkProcessState::AFTER_THINK
                || think_state.process_state == ThinkProcessState::NO_THINK) {
                return fill_grammar_row(row);
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
        auto fail_with = [&](std::string what) {
            rollback_provisional();
            matcher_->markFinished();
            reported_error_.store(true, std::memory_order_relaxed);
            if (request.bitmask_cpu_out != nullptr && request.bitmask_size_int32 > 0) {
                forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id_);
            }
            pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
            pending_error_msg_  = "reasoning grammar MTP verify exception: " + std::move(what);
        };

        try {
            for (int offset = 0; offset <= P; ++offset) {
                int32_t*              row       = request.bitmask_cpu_out + offset * W;
                const GrammarRowState row_state = fill_row(row);
                if (row_state == GrammarRowState::Failed) {
                    rollback_provisional();
                    matcher_->markFinished();
                    reported_error_.store(true, std::memory_order_relaxed);
                    forceTokenInBitmask(row, W, eos_token_id_);
                    pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
                    pending_error_msg_  = "reasoning grammar matcher fillBitmask failed during MTP verify; "
                                          "matcher state corrupted";
                    return offset;
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
                    advanceThinkStateForSpec(think_state, draft_token);
                }
            }
        } catch (const std::exception& e) {
            fail_with(e.what());
            return 0;
        } catch (...) {
            fail_with("unknown");
            return 0;
        }
        // Verify never accumulates state on the matcher; commit happens via updateStatus.
        rollback_provisional();
        return cap;
    }();
    flushError(/*stream_lock_held=*/false);
    return return_value;
}

int64_t ReasoningGrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return think_info_.current_output_length;
}

bool ReasoningGrammarLogitsProcessor::applyMaskLocked(const SamplerInputs& inputs, size_t batch_idx) {
    auto logits = inputs.logits[batch_idx];

    switch (think_info_.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            return applyGrammarMaskLocked(logits);
        case ThinkProcessState::IN_THINK: {
            if (transitionToAfterThinkIfClosed(think_info_)) {
                return applyGrammarMaskLocked(logits);
            }
            if (thinkEndCloseInProgress(think_info_) || thinkBudgetExhausted(inputs, batch_idx, think_info_)) {
                think_info_.process_state = ThinkProcessState::CLOSING_THINK;
                return forceThinkEndTokenLocked(logits);
            }
            maskToken(logits, firstTokenOrInvalid(think_info_.begin_think_token_ids));
            maskToken(logits, eos_token_id_);
            return true;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (transitionToAfterThinkIfClosed(think_info_)) {
                return applyGrammarMaskLocked(logits);
            }
            if (forceThinkEndTokenLocked(logits)) {
                return true;
            }
            maskToken(logits, firstTokenOrInvalid(think_info_.begin_think_token_ids));
            maskToken(logits, eos_token_id_);
            return true;
        }
    }
    return true;
}

bool ReasoningGrammarLogitsProcessor::applyGrammarMaskLocked(const torch::Tensor& logits) {
    if (!matcher_ || matcher_->finished()) {
        return false;
    }
    if (matcher_->isTerminated()) {
        forceToken(logits, eos_token_id_);
        return true;
    }
    if (matcher_->isPassthroughForMask()) {
        // Reasoner-mode passthrough (require_reasoning_ && </think> not yet crossed):
        // matcher is frozen and fillBitmask returns false BY DESIGN. Aligns with the
        // spec-verify path (fill_grammar_row → AllowAll) and prevents the false-positive
        // EXECUTION_EXCEPTION below in case think_state and matcher state desync at the
        // boundary (e.g. AFTER_THINK transition observed by think_state but the bonus
        // </think> token not yet consumed by the matcher).
        return false;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        return false;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    int64_t       dl_shape[2];
    DLTensor      dl = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        // Indeterminate matcher state: collapse to EOS-only instead of leaking allow-all logits.
        if (!reported_error_.exchange(true)) {
            matcher_->markFinished();
            pending_error_code_ = ErrorCode::EXECUTION_EXCEPTION;
            pending_error_msg_  = "reasoning grammar matcher fillBitmask failed; matcher state corrupted";
        }
        forceToken(logits, eos_token_id_);
        return true;
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
    return true;
}

bool ReasoningGrammarLogitsProcessor::forceThinkEndTokenLocked(const torch::Tensor& logits) {
    if (!think_info_.dfa_ptr || think_info_.dfa_ptr->isFinished() || think_info_.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = think_info_.dfa_ptr->status();
    if (next_token_idx >= think_info_.end_think_token_ids.size()) {
        return false;
    }

    const int32_t token_id = think_info_.end_think_token_ids[next_token_idx];
    forceToken(logits, token_id);
    if (reported_error_.load(std::memory_order_relaxed)) {
        return false;
    }

    think_info_.dfa_ptr->next(token_id);
    think_info_.pending_forced_think_end_token_ids.push_back(token_id);
    think_info_.current_output_length += 1;
    think_info_.process_state =
        think_info_.dfa_ptr->isFinished() ? ThinkProcessState::AFTER_THINK : ThinkProcessState::CLOSING_THINK;
    return true;
}

void ReasoningGrammarLogitsProcessor::acceptCommittedGrammarTokenLocked(int32_t token_id) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    if (matcher_->isTerminated()) {
        if (token_id != eos_token_id_) {
            matcher_->markFinished();
            reportErrorOnce(ErrorCode::INVALID_PARAMS,
                            "reasoning grammar received non-EOS token after terminal state " + std::to_string(token_id),
                            /*stream_lock_held=*/true);
            return;
        }
        matcher_->markFinished();
        return;
    }
    if (!matcher_->acceptToken(token_id)) {
        matcher_->markFinished();
        reportErrorOnce(ErrorCode::INVALID_PARAMS,
                        "reasoning grammar accept_token error: parser rejected token " + std::to_string(token_id),
                        /*stream_lock_held=*/true);
    }
}

void ReasoningGrammarLogitsProcessor::reportErrorOnce(ErrorCode code, const std::string& msg, bool stream_lock_held) {
    if (reported_error_.exchange(true)) {
        return;
    }
    if (error_reporter_) {
        error_reporter_(code, msg, stream_lock_held);
        return;
    }
    RTP_LLM_LOG_WARNING("%s", msg.c_str());
}

void ReasoningGrammarLogitsProcessor::flushError(bool stream_lock_held) {
    ErrorCode   code;
    std::string msg;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_error_msg_.empty()) {
            return;
        }
        code = pending_error_code_;
        msg  = std::move(pending_error_msg_);
    }
    if (error_reporter_) {
        error_reporter_(code, msg, stream_lock_held);
    } else {
        RTP_LLM_LOG_WARNING("%s", msg.c_str());
    }
}

void ReasoningGrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        if (!reported_error_.exchange(true)) {
            pending_error_code_ = ErrorCode::INVALID_PARAMS;
            pending_error_msg_  = "reasoning grammar forced token is out of logits vocab range";
        }
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
