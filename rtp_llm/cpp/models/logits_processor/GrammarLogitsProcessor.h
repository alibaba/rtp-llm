#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarMaskCore.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

namespace rtp_llm {

class GrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id = 0);

    ~GrammarLogitsProcessor() override;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void updateMultiSeqStatus(const std::vector<int>& /*src_batch_indices*/) override {}

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    int64_t committedOutputLen() const override {
        return mask_core_.acceptedTokenLen();
    }

    bool hasError() const override {
        return has_error_.load(std::memory_order_acquire);
    }
    ErrorInfo error() const override {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return error_info_;
    }

private:
    void setError(ErrorCode code, std::string msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!has_error_.load(std::memory_order_relaxed)) {
            error_info_ = ErrorInfo(code, std::move(msg));
            has_error_.store(true, std::memory_order_release);
        }
    }
    void setError(const ErrorInfo& info) {
        if (!info.hasError()) {
            return;
        }
        setError(info.code(), info.ToString());
    }

    GrammarMaskCore    mask_core_;
    mutable std::mutex state_mutex_;
    std::atomic<bool>  has_error_{false};
    ErrorInfo          error_info_;
};

}  // namespace rtp_llm
