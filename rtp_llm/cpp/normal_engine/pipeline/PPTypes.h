#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

struct RequestLogitsProcessorConfig {
    std::string grammar_type;
    std::string grammar_value;

    int                           combo_token_size = 0;
    std::vector<std::vector<int>> banned_combo_token_ids;
    std::vector<int>              end_think_token_ids;
};

struct PPSamplingPlan {
    std::vector<std::optional<int>>           random_seeds;
    std::vector<RequestLogitsProcessorConfig> logits_processor_configs;

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

/** Batch-aggregated output configuration for the lm-head stage. */
struct PPOutputConfig {
    bool               return_logits            = false;
    bool               return_softmax_probs     = false;
    bool               return_cum_log_probs     = false;
    bool               calculate_loss           = false;
    bool               return_hidden_states     = false;
    bool               return_all_hidden_states = false;
    ReturnAllProbsMode return_all_probs         = ReturnAllProbsMode::NONE;
};

struct PPExecutionPlan {
    GptModelInputs       model_input;
    PPSamplingPlan       sampling_plan;
    PPOutputConfig       output_config;
    std::vector<int64_t> finished_request_ids;
};

struct PPIntermediateTensors {
    std::map<std::string, torch::Tensor> tensors;
};

/** Final per-request outputs produced by the lm-head stage TP root. */
struct PPExecutionResult {
    torch::Tensor request_ids;     // [batch_size]
    torch::Tensor new_token_ids;   // [batch_size, 1]
    torch::Tensor sample_success;  // [batch_size]

    torch::Tensor logits;
    torch::Tensor softmax_probs;
    torch::Tensor cum_log_probs;
    torch::Tensor all_probs;

    torch::Tensor loss;

    torch::Tensor hidden_states;
    torch::Tensor all_hidden_states;

    std::vector<std::optional<ErrorInfo>> processor_errors;
};

}  // namespace rtp_llm
