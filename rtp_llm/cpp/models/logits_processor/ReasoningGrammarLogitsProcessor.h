#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

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
    ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher>    matcher,
                                    int64_t                               eos_token_id,
                                    int                                   max_thinking_tokens,
                                    std::vector<int>                      begin_think_token_ids,
                                    std::vector<int>                      end_think_token_ids,
                                    int32_t                               input_length,
                                    LogitsProcessorFactory::ErrorReporter error_reporter = {});
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
    bool applyMaskLocked(const SamplerInputs& inputs, size_t batch_idx);
    bool applyGrammarMaskLocked(const torch::Tensor& logits);
    bool forceThinkEndTokenLocked(const torch::Tensor& logits);
    void acceptCommittedGrammarTokenLocked(int32_t token_id);
    void forceToken(const torch::Tensor& logits, int64_t token_id);
    void maskToken(const torch::Tensor& logits, int64_t token_id);
    void reportErrorOnce(ErrorCode code, const std::string& msg, bool stream_lock_held);

    // Lock invariant: error_reporter_ takes stream.mutex_, so it must be
    // invoked outside our mutex_. Errors raised under mutex_ write to
    // pending_error_*; flushError() reads-and-clears them outside the lock.
    void flushError(bool stream_lock_held);

    mutable std::mutex                 mutex_;
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_;
    StreamThinkInfo                    think_info_;
    std::atomic_bool                   reported_error_{false};
    ErrorCode                          pending_error_code_{};
    std::string                        pending_error_msg_;  // guarded by mutex_
};

using ReasoningGrammarLogitsProcessorPtr = std::shared_ptr<ReasoningGrammarLogitsProcessor>;

}  // namespace rtp_llm
