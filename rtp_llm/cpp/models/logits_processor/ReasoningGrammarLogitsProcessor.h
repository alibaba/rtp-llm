#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class GenerateInput;
struct SamplerInputs;

// In-think + grammar combined processor. The matcher carries the reasoner
// passthrough state internally; this processor layers the ThinkMode budget /
// force-close machinery on top of it. Token routing per state:
//   - NO_THINK / AFTER_THINK : matcher accepts every committed token; grammar
//     mask drives logits.
//   - IN_THINK / CLOSING_THINK : think mask drives logits (force end-token or
//     suppress begin/eos); committed tokens advance the DFA but the matcher's
//     internal passthrough swallows them without parser advance.
class ReasoningGrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    int64_t                            eos_token_id,
                                    int                                max_thinking_tokens,
                                    std::vector<int>                   begin_think_token_ids,
                                    std::vector<int>                   end_think_token_ids,
                                    int32_t                            input_length);
    ~ReasoningGrammarLogitsProcessor() override = default;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    int64_t acceptedTokenLen() const override;

private:
    void applyMaskLocked(const SamplerInputs& inputs, size_t batch_idx);
    void applyGrammarMaskLocked(const torch::Tensor& logits);
    void acceptCommittedGrammarTokenLocked(int32_t token_id);
    void forceToken(const torch::Tensor& logits, int64_t token_id);
    void maskToken(const torch::Tensor& logits, int64_t token_id);

    mutable std::mutex                 mutex_;
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_;
    StreamThinkInfo                    think_info_;
};

using ReasoningGrammarLogitsProcessorPtr = std::shared_ptr<ReasoningGrammarLogitsProcessor>;

}  // namespace rtp_llm
