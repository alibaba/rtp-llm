#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id):
    mask_core_(std::move(matcher), eos_token_id) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
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
        RTP_LLM_LOG_WARNING("grammar logits processor unexpected batch_size=%zu", batch_size);
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
        std::lock_guard<std::mutex> lock(state_mutex_);
        mask_core_.applyMaskLocked(inputs.logits[start_idx], local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
    }
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (hasError()) {
        return;
    }
    if (!mask_core_.matcher() || mask_core_.finished()) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int            batch_size = static_cast<int>(new_tokens.size(0));
    const int            stride     = static_cast<int>(new_tokens.size(1));
    const auto*          data       = new_tokens.data_ptr<int32_t>();
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(batch_size * num_new_tokens));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_new_tokens; ++j) {
            tokens.push_back(data[i * stride + j]);
        }
    }

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        mask_core_.acceptCommittedLocked(tokens.data(), tokens.size(), local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
    }
}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return mask_core_.matcher() != nullptr;
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
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
        std::lock_guard<std::mutex> lock(state_mutex_);
        cap_out = mask_core_.runSpecVerifyLocked(request, local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
        return 0;
    }
    return cap_out;
}

}  // namespace rtp_llm
