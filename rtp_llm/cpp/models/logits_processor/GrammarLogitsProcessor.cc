#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <limits>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

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

bool bitmaskAllowsToken(const int32_t* bitmask, int32_t token_id) {
    const int32_t word = bitmask[token_id / 32];
    return (static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u;
}

}  // namespace

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                               int64_t                            eos_token_id,
                                               ErrorReporter                      error_reporter):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id), error_reporter_(std::move(error_reporter)) {}

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        reportErrorOnce(
            ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding", false);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = reinterpret_cast<const bool*>(inputs.finished_mask.data_ptr());
        if (finished[start_idx]) {
            return;
        }
    }

    auto logits = inputs.logits.narrow(0, start_idx, 1);
    if (matcher_->isTerminated()) {
        forceToken(logits[0], eos_token_id_);
        return;
    }
    if (matcher_->isPassthroughForMask()) {
        maskToken(logits[0], eos_token_id_);
        return;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    const int64_t logits_vocab_size  = logits.size(1);
    const int64_t mask_vocab_size    = std::min<int64_t>(logits_vocab_size, grammar_vocab_size);
    if (mask_vocab_size <= 0) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS, "grammar vocab size is empty", false);
        return;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    DLTensor      dl      = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words);
    if (!matcher_->fillBitmask(&dl, 0)) {
        return;
    }

    auto           vocab_mask  = at::ones({1, mask_vocab_size}, at::dtype(at::kByte));
    uint8_t*       mask_ptr    = vocab_mask.data_ptr<uint8_t>();
    const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
    for (int32_t token_id = 0; token_id < mask_vocab_size; ++token_id) {
        if (bitmaskAllowsToken(bitmask_ptr, token_id)) {
            mask_ptr[token_id] = 0;
        }
    }

    auto target_logits = logits.narrow(1, 0, mask_vocab_size);
    auto mask          = vocab_mask.to(torch::kBool);
    if (mask.device() != target_logits.device()) {
        mask = mask.to(target_logits.device(), true);
    }
    target_logits.masked_fill_(mask, BaseLogitsProcessor::neg_inf);

    if (grammar_vocab_size < logits_vocab_size) {
        logits.narrow(1, grammar_vocab_size, logits_vocab_size - grammar_vocab_size)
            .fill_(BaseLogitsProcessor::neg_inf);
    }
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    (void)src_batch_indices;
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_ || matcher_->finished() || num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS, "grammar accept expects one row with num_new_tokens columns", true);
        return;
    }

    auto tokens_cpu       = new_tokens.is_cuda() ? new_tokens.cpu() : new_tokens;
    tokens_cpu            = tokens_cpu.to(torch::kInt32).contiguous();
    const auto* token_ptr = tokens_cpu.data_ptr<int32_t>();
    for (int32_t i = 0; i < num_new_tokens; ++i) {
        const int32_t token_id = token_ptr[i];
        if (!matcher_->acceptToken(token_id)) {
            matcher_->markFinished();
            reportErrorOnce(ErrorCode::INVALID_PARAMS,
                            "grammar accept_token error: parser rejected token " + std::to_string(token_id),
                            true);
            return;
        }
        if (matcher_->isTerminated()) {
            return;
        }
    }
}

void GrammarLogitsProcessor::reportErrorOnce(ErrorCode          error_code,
                                             const std::string& error_msg,
                                             bool               stream_lock_held) {
    if (reported_error_) {
        return;
    }
    reported_error_ = true;
    if (error_reporter_) {
        error_reporter_(error_code, error_msg, stream_lock_held);
        return;
    }
    RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
}

void GrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS, "grammar terminal token is out of logits vocab range", false);
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 1;
}

void GrammarLogitsProcessor::maskToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits[token_id] = BaseLogitsProcessor::neg_inf;
}

}  // namespace rtp_llm
