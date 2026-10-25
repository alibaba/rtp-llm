#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include "kmonitor/client/MetricsReporter.h"

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class RtpGrammarMatcher;

class GrammarLogitsProcessor final: public BaseLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    int64_t                            eos_token_id      = 0,
                                    int                                update_timeout_ms = 10,
                                    kmonitor::MetricsReporterPtr       metrics_reporter  = nullptr,
                                    std::string                        stream_log_tag    = {});

    ~GrammarLogitsProcessor() override;

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void                     updateMultiSeqStatus(const std::vector<int>& /*src_batch_indices*/) override {}

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, {}};
    }

    // Grammar state advances per committed token. It does not publish its own
    // device-side next-step state, so the normal-decode async device-state
    // fast path must keep the pre-sampler wait for streams carrying it.
    bool isStateful() const override {
        return !updateTimedOut();
    }

    bool updateTimedOut() const;

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override;

    std::optional<int64_t> committedOutputLen() const override;

private:
    class DecodeMaskBuilder;

    ErrorInfo acceptCommittedLocked(const int32_t* tokens, size_t n);

    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_         = 0;
    int64_t                            committed_output_len_ = 0;
    int                                update_timeout_ms_;
    bool                               update_timed_out_ = false;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
    const std::string                  stream_log_tag_;
    std::unique_ptr<DecodeMaskBuilder> decode_mask_builder_;

    mutable std::mutex state_mutex_;
};

}  // namespace rtp_llm
