#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include <algorithm>
#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr int32_t kInvalidTokenId = -1;

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

void clearTokensFromBitmask(int32_t* bitmask, size_t words, const std::vector<int>& token_ids) {
    for (const int token_id : token_ids) {
        clearTokenFromBitmask(bitmask, words, token_id);
    }
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
    const int32_t token_id = info.nextThinkEndToken();
    if (token_id == kInvalidTokenId) {
        return false;
    }
    forceTokenInBitmask(row, words, token_id);
    return true;
}

enum class SpecRowMode { THINK, GRAMMAR };

SpecRowMode applyThinkSpecRowMask(int32_t* row, size_t words, StreamThinkInfo& info) {
    std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            return SpecRowMode::GRAMMAR;
        case ThinkProcessState::IN_THINK: {
            if (info.thinkEndCloseInProgress() || specThinkBudgetExhausted(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                if (forceThinkEndTokenInBitmask(row, words, info)) {
                    return SpecRowMode::THINK;
                }
            }
            clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
            clearTokensFromBitmask(row, words, info.masked_stop_token_ids);
            return SpecRowMode::THINK;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (!info.pending_forced_think_end_token_ids.empty()) {
                forceTokenInBitmask(row, words, info.pending_forced_think_end_token_ids.front());
                return SpecRowMode::THINK;
            }
            if (forceThinkEndTokenInBitmask(row, words, info)) {
                return SpecRowMode::THINK;
            }
            clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
            clearTokensFromBitmask(row, words, info.masked_stop_token_ids);
            return SpecRowMode::THINK;
        }
    }
    return SpecRowMode::THINK;
}

}  // namespace

ReasoningGrammarLogitsProcessor::ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                                                 int64_t                            eos_token_id,
                                                                 int                                max_thinking_tokens,
                                                                 std::vector<int> begin_think_token_ids,
                                                                 std::vector<int> end_think_token_ids,
                                                                 int32_t          input_length,
                                                                 ErrorReporter    error_reporter,
                                                                 const std::vector<std::vector<int>>& stop_words_list,
                                                                 std::string model_type):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id), error_reporter_(std::move(error_reporter)) {
    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
    if (!end_think_token_ids.empty()) {
        dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
    }
    think_info_ = StreamThinkInfo(true,
                                  max_thinking_tokens,
                                  std::move(begin_think_token_ids),
                                  end_think_token_ids,
                                  input_length,
                                  0,
                                  false,
                                  std::move(dfa_ptr),
                                  makeMaskedStopTokenIds(stop_words_list, eos_token_id, end_think_token_ids),
                                  std::move(model_type));
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
        if (think_info_.updateState(token_id)) {
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
        clearTokenFromBitmask(row, W, think_state.thinkBoundaryToMask(think_state.begin_think_token_ids));
        clearTokenFromBitmask(row, W, think_state.thinkBoundaryToMask(think_state.end_think_token_ids));
    };

    auto fill_row = [&](int32_t* row) {
        const auto row_mode = applyThinkSpecRowMask(row, W, think_state);
        if (row_mode == SpecRowMode::GRAMMAR) {
            fill_grammar_row(row);
        }
        return row_mode;
    };

    for (int offset = 0; offset <= P; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        const auto row_mode = fill_row(row);
        if (offset == P) {
            break;
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
            || !bitmaskAllowsToken(row, W, draft_token)) {
            cap = offset;
            break;
        }

        if (row_mode == SpecRowMode::GRAMMAR) {
            if (!matcher_->acceptToken(draft_token)) {
                cap = offset;
                break;
            }
            ++grammar_accepted_prefix;
        }
        think_state.updateState(draft_token);
    }

    matcher_->rollback(grammar_accepted_prefix);
    return cap;
}

int64_t ReasoningGrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return think_info_.current_output_length;
}

bool ReasoningGrammarLogitsProcessor::applyReasoningOrGrammarMaskLocked(const SamplerInputs& inputs, size_t batch_idx) {
    auto logits = inputs.logits[batch_idx];

    switch (think_info_.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            return applyGrammarMaskLocked(logits);
        case ThinkProcessState::IN_THINK: {
            if (think_info_.thinkEndCloseInProgress() || thinkBudgetExhausted(inputs, batch_idx, think_info_)) {
                think_info_.process_state = ThinkProcessState::CLOSING_THINK;
                return forceThinkEndTokenLocked(logits);
            }
            maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.begin_think_token_ids));
            maskStopTokens(logits);
            return true;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (forceThinkEndTokenLocked(logits)) {
                return true;
            }
            maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.begin_think_token_ids));
            maskStopTokens(logits);
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
        maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.begin_think_token_ids));
        maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.end_think_token_ids));
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

    maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.begin_think_token_ids));
    maskToken(logits, think_info_.thinkBoundaryToMask(think_info_.end_think_token_ids));
    return true;
}

bool ReasoningGrammarLogitsProcessor::forceThinkEndTokenLocked(const torch::Tensor& logits) {
    const int32_t token_id = think_info_.nextThinkEndToken();
    if (token_id == kInvalidTokenId) {
        return false;
    }

    RTP_LLM_LOG_INFO("sampler enforce think end token: token_id=%d", token_id);
    forceToken(logits, token_id);
    if (reported_error_.load(std::memory_order_relaxed)) {
        return false;
    }

    think_info_.precommitThinkEndToken(token_id);
    return true;
}

void ReasoningGrammarLogitsProcessor::maskStopTokens(const torch::Tensor& logits) {
    for (const int token_id : think_info_.masked_stop_token_ids) {
        maskToken(logits, token_id);
    }
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
