#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class RtpGrammarMatcher;

class ReasoningGrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    using ErrorReporter = std::function<void(ErrorCode, const std::string&, bool)>;

    ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    int64_t                            eos_token_id,
                                    int                                max_thinking_tokens,
                                    std::vector<int>                   begin_think_token_ids,
                                    std::vector<int>                   end_think_token_ids,
                                    int32_t                            input_length,
                                    ErrorReporter                      error_reporter = nullptr);
    ~ReasoningGrammarLogitsProcessor() override = default;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void processSpeculative(const SamplerInputs&        inputs,
                            size_t                      start_idx,
                            size_t                      finish_idx,
                            const std::vector<int32_t>& draft_prefix) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool isStateful() const override {
        return true;
    }
    int64_t acceptedTokenLen() const override;
    int64_t finishedThinkOutputLen() const override;

private:
    bool applyReasoningOrGrammarMaskLocked(const SamplerInputs& inputs, size_t batch_idx);
    bool applyGrammarMaskLocked(const torch::Tensor& logits);
    bool forceThinkEndTokenLocked(const torch::Tensor& logits);
    void acceptCommittedGrammarTokenLocked(int32_t token_id);
    void reportErrorOnce(ErrorCode error_code, const std::string& error_msg, bool stream_lock_held);
    void forceToken(const torch::Tensor& logits, int64_t token_id);
    void maskToken(const torch::Tensor& logits, int64_t token_id);

private:
    mutable std::mutex                 mutex_;
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_;
    StreamThinkInfo                    think_info_;
    ErrorReporter                      error_reporter_;
    std::atomic_bool                   reported_error_ = false;
};

using ReasoningGrammarLogitsProcessorPtr = std::shared_ptr<ReasoningGrammarLogitsProcessor>;

}  // namespace rtp_llm
