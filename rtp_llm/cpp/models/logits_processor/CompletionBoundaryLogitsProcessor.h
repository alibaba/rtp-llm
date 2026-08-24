#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

struct CompletionBoundaryState {
    std::vector<int32_t> boundary_token_ids;
    std::vector<size_t>  prefix_table;
    size_t               boundary_status = 0;
    int32_t              input_length = 0;
    int32_t              current_output_length = 0;
    bool                 is_beam_search = false;

    CompletionBoundaryState() = default;
    CompletionBoundaryState(std::vector<int32_t> boundary,
                            int32_t              input_length,
                            bool                 is_beam_search);

    bool finished() const;
    void advance(int32_t token_id);
};

// A compatibility guard for pre-tokenized chat RPCs that no longer carry the
// original tool schema. It does not constrain normal generation: only EOS and
// single-token stop words are masked until the model emits its complete message
// boundary. Request-aware structural grammars remain the stronger path.
class CompletionBoundaryLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    CompletionBoundaryLogitsProcessor(std::vector<CompletionBoundaryState> states,
                                      std::vector<int32_t>                 guarded_stop_token_ids,
                                      int64_t                              request_id = 0,
                                      std::string                          trace_id = "");

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    bool    isSpecVerifyEligible() const override;
    int     tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool    isStateful() const override;
    int64_t acceptedTokenLen() const override;

    std::vector<size_t> boundaryStatus() const;

private:
    void maskGuardedStops(const torch::Tensor& logits, size_t vocab_size) const;

private:
    std::vector<CompletionBoundaryState> states_;
    std::vector<int32_t>                 guarded_stop_token_ids_;
    int64_t                              request_id_;
    std::string                          trace_id_;
    mutable std::mutex                   mutex_;
};

using CompletionBoundaryLogitsProcessorPtr = std::shared_ptr<CompletionBoundaryLogitsProcessor>;

}  // namespace rtp_llm
