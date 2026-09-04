#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

struct RequestLogitsProcessorConfig {
    std::string grammar_type;   // scalar per stream
    std::string grammar_value;  // scalar per stream

    int                           combo_token_size = 0;                   // scalar per stream
    std::vector<std::vector<int>> banned_combo_token_ids;                 // [banned_combo_count, combo_token_size]
    std::vector<int>              end_think_token_ids;                    // [end_think_token_count]
    bool                          enable_cross_sequence_ban     = false;  // scalar per stream
    int                           cross_seq_diverge_start_combo = 0;      // scalar per stream
};

struct PPSamplingPlan {
    std::vector<std::optional<int>>           random_seeds;              // [stream_count]
    std::vector<RequestLogitsProcessorConfig> logits_processor_configs;  // [stream_count]
    std::vector<int32_t>                      num_return_sequences;      // [stream_count]

    torch::Tensor request_ids;  // [stream_count]

    torch::Tensor token_ids;         // [total_batch_size, max_sequence_length + 1]
    torch::Tensor input_lengths;     // [total_batch_size]
    torch::Tensor sequence_lengths;  // [total_batch_size]

    torch::Tensor top_k;                 // [total_batch_size]
    torch::Tensor top_p;                 // [total_batch_size]
    torch::Tensor temperature;           // [total_batch_size]
    torch::Tensor repetition_penalty;    // [total_batch_size]
    torch::Tensor presence_penalty;      // [total_batch_size]
    torch::Tensor frequency_penalty;     // [total_batch_size]
    torch::Tensor no_repeat_ngram_size;  // [total_batch_size]
    torch::Tensor do_sample;             // [total_batch_size]
    torch::Tensor finished_mask;         // [total_batch_size]
};

struct PPPromptLogitsRequest {
    bool enabled               = false;
    int  top_k                 = 64;
    int  start                 = -1;
    int  end                   = -1;
    bool return_target_logprob = true;
};

/** Output configuration for the lm-head stage. */
struct PPOutputConfig {
    bool               return_logits            = false;
    bool               return_softmax_probs     = false;
    bool               return_cum_log_probs     = false;
    bool               calculate_loss           = false;
    bool               return_hidden_states     = false;
    bool               return_all_hidden_states = false;
    ReturnAllProbsMode return_all_probs         = ReturnAllProbsMode::NONE;

    /** Per-stream configuration, aligned with PPSamplingPlan::request_ids. */
    std::vector<PPPromptLogitsRequest> prompt_logits_requests;
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

/** Final outputs produced by the lm-head stage TP root. */
struct PPExecutionResult {
    torch::Tensor request_ids;     // [stream_count]
    torch::Tensor new_token_ids;   // [total_batch_size, 1]
    torch::Tensor sample_success;  // [total_batch_size]

    torch::Tensor logits;         // optional [total_batch_size, vocab_size]
    torch::Tensor softmax_probs;  // optional [total_batch_size, 1]
    torch::Tensor cum_log_probs;  // optional [total_batch_size]
    torch::Tensor all_probs;      // optional [total_batch_size, vocab_size]

    torch::Tensor loss;  // optional [loss_token_count]

    torch::Tensor hidden_states;      // optional [total_batch_size, hidden_size]
    torch::Tensor all_hidden_states;  // optional [executed_token_count, hidden_size]

    std::vector<std::optional<PromptLogitsOutput>> prompt_logits;     // [stream_count]
    std::vector<std::optional<ErrorInfo>>          processor_errors;  // [total_batch_size]
};

}  // namespace rtp_llm
