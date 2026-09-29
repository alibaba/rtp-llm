#pragma once

#include <vector>

#include <ATen/Generator.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

/**
 * Request-level sampling state and retained sampling results.
 * Retained tensors are stored on CPU. cum_log_probs contains one float32 value per batch row.
 */
struct SamplingState {
    std::vector<BaseLogitsProcessorPtr> logits_processors;
    at::Generator                       generator;

    torch::Tensor cum_log_probs;
    torch::Tensor all_probs;
    torch::Tensor softmax_probs;

    ReturnAllProbsMode return_all_probs = ReturnAllProbsMode::NONE;
};

}  // namespace rtp_llm
