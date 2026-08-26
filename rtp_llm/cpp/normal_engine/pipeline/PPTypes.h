#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

struct RequestLogitsProcessorConfig {
    std::string grammar_type;
    std::string grammar_value;

    int                           combo_token_size = 0;
    std::vector<std::vector<int>> banned_combo_token_ids;
    std::vector<int>              end_think_token_ids;
};

struct PPSamplingData {
    std::vector<std::optional<int>>           random_seeds;
    std::vector<RequestLogitsProcessorConfig> logits_processor_configs;
    bool                                      need_cum_log_probs = false;

    torch::Tensor request_ids;

    torch::Tensor token_ids;
    torch::Tensor input_lengths;
    torch::Tensor sequence_lengths;

    torch::Tensor top_k;
    torch::Tensor top_p;
    torch::Tensor temperature;
    torch::Tensor repetition_penalty;
    torch::Tensor presence_penalty;
    torch::Tensor frequency_penalty;
    torch::Tensor no_repeat_ngram_size;
    torch::Tensor do_sample;
    torch::Tensor finished_mask;
};

struct PPExecutionPlan {
    GptModelInputs model_input;
    PPSamplingData sampling;
};

struct PPIntermediateTensors {
    std::map<std::string, torch::Tensor> tensors;
};

struct PPSampleError {
    int64_t     request_id = 0;
    int32_t     error_code = 0;
    std::string message;
};

struct PPSampleResult {
    torch::Tensor              request_ids;     // [batch_size]
    torch::Tensor              new_token_ids;   // [batch_size, 1]
    torch::Tensor              sample_success;  // [batch_size]
    torch::Tensor              cum_log_probs;
    std::vector<PPSampleError> errors;
};

}  // namespace rtp_llm
