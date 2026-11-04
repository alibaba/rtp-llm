#pragma once

#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

// Acceptance-only teacher forcing. Both operator environment variables and a
// reserved task_id are required; ordinary requests never read a fixture.
class FixedContinuationLogitsProcessor final: public BaseLogitsProcessor {
public:
    static ErrorResult<BaseLogitsProcessorPtr> fromGenerateInput(const std::shared_ptr<GenerateInput>& input);

    FixedContinuationLogitsProcessor(std::vector<int32_t> input_ids,
                                     std::vector<int32_t> continuation,
                                     int32_t              vocab_size,
                                     std::string          output_directory,
                                     std::string          case_id,
                                     int64_t              request_id);

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void updateMultiSeqStatus(const std::vector<int>&) override {}

    bool isStateful() const override {
        return true;
    }
    bool requiresHostTokenHistory() const override {
        return true;
    }
    std::optional<int64_t> committedOutputLen() const override {
        return committed_;
    }

private:
    const std::vector<int32_t> input_ids_;
    const std::vector<int32_t> continuation_;
    const int32_t              vocab_size_;
    const std::string          output_directory_;
    const std::string          case_id_;
    const int64_t              request_id_;
    int64_t                    committed_{0};
    int                        source_scalar_type_{-1};
    torch::Tensor              pending_raw_logits_;
};

}  // namespace rtp_llm
