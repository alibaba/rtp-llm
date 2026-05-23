#pragma once

#include <functional>
#include <memory>
#include <string>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class RtpGrammarMatcher;

class GrammarLogitsProcessor: public BaseLogitsProcessor {
public:
    using ErrorReporter = std::function<void(ErrorCode, const std::string&, bool)>;

    GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                           int64_t                            eos_token_id,
                           ErrorReporter                      error_reporter = nullptr);
    ~GrammarLogitsProcessor() override = default;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

private:
    void reportErrorOnce(ErrorCode error_code, const std::string& error_msg, bool stream_lock_held);
    void forceToken(const torch::Tensor& logits, int64_t token_id);
    void maskToken(const torch::Tensor& logits, int64_t token_id);

private:
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_;
    ErrorReporter                      error_reporter_;
    bool                               reported_error_ = false;
};

using GrammarLogitsProcessorPtr = std::shared_ptr<GrammarLogitsProcessor>;

}  // namespace rtp_llm
