#pragma once

#include <optional>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

// Engine stream outputs contain token deltas, including non-streaming PD requests.
// Keep token tensors until completion and concatenate once; retain optional outputs
// that may only be published in an earlier step (loss, prompt logits, hidden states).
class BatchStreamOutputCollector {
public:
    explicit BatchStreamOutputCollector(std::optional<int> logits_index = std::nullopt): logits_index_(logits_index) {}

    ErrorInfo       add(GenerateOutputs output);
    GenerateOutputs finish();
    bool            empty() const {
        return tokens_.empty();
    }

private:
    std::optional<int>                       logits_index_;
    std::vector<std::optional<torch::Tensor>> selected_logits_;
    GenerateOutputs                         output_;
    std::vector<std::vector<torch::Tensor>>   tokens_;
    std::vector<std::vector<torch::Tensor>>   softmax_probs_;
};

}  // namespace rtp_llm
