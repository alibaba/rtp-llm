#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class CodebookLogitsProcessor final: public BaseLogitsProcessor {
public:
    CodebookLogitsProcessor(const std::vector<std::vector<int64_t>>& groups, size_t vocab_size, size_t batch_size);
    CodebookLogitsProcessor(torch::Tensor masks, size_t batch_size);

    static std::optional<std::string> validateGroups(const std::vector<std::vector<int64_t>>& groups,
                                                     const std::vector<int64_t>&              ids,
                                                     int64_t                                  eos_token_id);

    // Build once per engine; processors share the immutable mask storage.
    static torch::Tensor createMasks(const std::vector<std::vector<int64_t>>& groups, size_t vocab_size);

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void                     updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    std::optional<int64_t> committedOutputLen() const override;
    bool                   isStateful() const override;
    bool                   supportsNormalAsyncDeviceState() const override;

private:
    torch::Tensor maskFor(const torch::Device& device);

private:
    size_t        vocab_size_;
    size_t        batch_size_;
    size_t        committed_steps_ = 0;
    torch::Tensor masks_;
};

}  // namespace rtp_llm
