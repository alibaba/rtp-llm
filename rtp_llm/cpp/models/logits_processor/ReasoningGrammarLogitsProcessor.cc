#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include <algorithm>
#include <limits>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
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
    info.markAfterThink();
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
    return observed_output_tokens >= info.bodyTokenBudget();
}

bool specThinkBudgetExhausted(const StreamThinkInfo& info) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && info.current_output_length >= info.bodyTokenBudget();
}

bool consumePendingForcedThinkEndToken(StreamThinkInfo& info, int32_t current_token_id) {
    if (info.pending_forced_think_end_token_ids.empty()) {
        return false;
    }
    const int32_t expected_token_id = info.pending_forced_think_end_token_ids.front();
    info.pending_forced_think_end_token_ids.erase(info.pending_forced_think_end_token_ids.begin());
    if (current_token_id != expected_token_id) {
        RTP_LLM_LOG_WARNING("forced think end token mismatch, expected=%d actual=%d, trust precommitted state",
                            expected_token_id,
                            current_token_id);
    }
    return true;
}

DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words) {
    DLTensor dl;
    dl.data   = data;
    dl.device = DLDevice{kDLCPU, 0};
    dl.ndim   = 2;
    dl.dtype  = DLDataType{kDLInt, 32, 1};
    static thread_local int64_t shape[2];
    shape[0]       = 1;
    shape[1]       = words;
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

bool bitmaskAllowsToken(const int32_t* bitmask, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    const uint32_t word = static_cast<uint32_t>(bitmask[token_id / 32]);
    return (word & (1u << (token_id % 32))) != 0u;
}

void clearTokenFromBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] &= ~(1u << (token_id % 32));
}

void forceTokenInBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    std::fill_n(bitmask, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] |= (1u << (token_id % 32));
}

void clearBitmaskTokenRange(int32_t* bitmask, size_t words, int64_t begin_token, int64_t end_token) {
    if (begin_token < 0 || end_token <= begin_token) {
        return;
    }
    for (int64_t token_id = begin_token; token_id < end_token; ++token_id) {
        clearTokenFromBitmask(bitmask, words, token_id);
    }
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
    if (consumePendingForcedThinkEndToken(info, token_id)) {
        return;
    }

    info.current_output_length += 1;
    if (!isActiveThinkState(info) || !info.dfa_ptr) {
        return;
    }

    info.dfa_ptr->next(token_id);
    if (info.dfa_ptr->isFinished()) {
        info.markAfterThink();
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
                                                                 ErrorReporter    error_reporter):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id), error_reporter_(std::move(error_reporter)) {
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
                        false);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = reinterpret_cast<const bool*>(inputs.finished_mask.data_ptr());
        if (finished[start_idx]) {
            return;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    applyReasoningOrGrammarMaskLocked(inputs, start_idx);
}

void ReasoningGrammarLogitsProcessor::processSpeculative(const SamplerInputs&        inputs,
                                                         size_t                      start_idx,
                                                         size_t                      finish_idx,
                                                         const std::vector<int32_t>& draft_prefix) {
    if (draft_prefix.empty()) {
        process(inputs, start_idx, finish_idx);
        return;
    }
    reportErrorOnce(
        ErrorCode::INVALID_PARAMS, "reasoning grammar speculative path requires precomputed MTP verify bitmask", false);
}

void ReasoningGrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    (void)src_batch_indices;
}

void ReasoningGrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        reportErrorOnce(
            ErrorCode::INVALID_PARAMS, "reasoning grammar accept expects one row with num_new_tokens columns", true);
        return;
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
        if (consumePendingForcedThinkEndToken(think_info_, token_id)) {
            continue;
        }

        think_info_.current_output_length += 1;
        if (isActiveThinkState(think_info_)) {
            if (think_info_.dfa_ptr) {
                think_info_.dfa_ptr->next(token_id);
                if (think_info_.dfa_ptr->isFinished()) {
                    think_info_.markAfterThink();
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
            return;
        }
    }
}

bool ReasoningGrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int ReasoningGrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const int  P                       = request.propose_step;
    const auto W                       = request.bitmask_size_int32;
    int        grammar_accepted_prefix = 0;
    int        cap                     = P;
    auto       think_state             = think_info_.copy();

    auto fill_grammar_row = [&](int32_t* row) {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            return;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return;
        }

        const int32_t grammar_vocab_size = matcher_->vocabSize();
        const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);
        RTP_LLM_CHECK_WITH_INFO(grammar_words <= W, "grammar vocab bitmask exceeds model vocab bitmask in MTP verify");

        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words));
        if (!matcher_->fillBitmask(&dl, 0)) {
            return;
        }
        clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
        clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.begin_think_token_ids));
        clearTokenFromBitmask(row, W, firstTokenOrInvalid(think_state.end_think_token_ids));
    };

    auto fill_row = [&](int32_t* row) {
        applyThinkSpecRowMask(row, W, think_state, eos_token_id_);
        if (think_state.process_state == ThinkProcessState::AFTER_THINK
            || think_state.process_state == ThinkProcessState::NO_THINK) {
            fill_grammar_row(row);
        }
    };

    for (int offset = 0; offset <= P; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        fill_row(row);
        if (offset == P) {
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

    matcher_->rollback(grammar_accepted_prefix);
    return cap;
}

int64_t ReasoningGrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return think_info_.current_output_length;
}

int64_t ReasoningGrammarLogitsProcessor::finishedThinkOutputLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return think_info_.finishedThinkOutputLen();
}

bool ReasoningGrammarLogitsProcessor::applyReasoningOrGrammarMaskLocked(const SamplerInputs& inputs, size_t batch_idx) {
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

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        return false;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    DLTensor      dl      = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words);
    if (!matcher_->fillBitmask(&dl, 0)) {
        maskToken(logits, firstTokenOrInvalid(think_info_.begin_think_token_ids));
        maskToken(logits, firstTokenOrInvalid(think_info_.end_think_token_ids));
        return false;
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
    if (think_info_.dfa_ptr->isFinished()) {
        think_info_.markAfterThink();
    } else {
        think_info_.process_state = ThinkProcessState::CLOSING_THINK;
    }
    return true;
}

void ReasoningGrammarLogitsProcessor::acceptCommittedGrammarTokenLocked(int32_t token_id) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    if (matcher_->isTerminated()) {
        if (token_id != eos_token_id_) {
            reportErrorOnce(ErrorCode::INVALID_PARAMS,
                            "reasoning grammar received non-EOS token after terminal state " + std::to_string(token_id),
                            true);
            return;
        }
        matcher_->markFinished();
        return;
    }
    if (!matcher_->acceptToken(token_id)) {
        matcher_->markFinished();
        reportErrorOnce(ErrorCode::INVALID_PARAMS,
                        "reasoning grammar accept_token error: parser rejected token " + std::to_string(token_id),
                        true);
    }
}

void ReasoningGrammarLogitsProcessor::reportErrorOnce(ErrorCode          error_code,
                                                      const std::string& error_msg,
                                                      bool               stream_lock_held) {
    if (reported_error_.exchange(true)) {
        return;
    }
    if (error_reporter_) {
        error_reporter_(error_code, error_msg, stream_lock_held);
        return;
    }
    RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
}

void ReasoningGrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        reportErrorOnce(
            ErrorCode::INVALID_PARAMS, "reasoning grammar forced token is out of logits vocab range", false);
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 1;
}

void ReasoningGrammarLogitsProcessor::maskToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits[token_id] = BaseLogitsProcessor::neg_inf;
}

}  // namespace rtp_llm
