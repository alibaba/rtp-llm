#pragma once

#include "maga_transformer/cpp/dataclass/Query.h"

namespace th = torch;

namespace rtp_llm {

class MergedGenerateConfig {
public:
    int64_t batch_size;
    int64_t beam_size = 1;
    th::optional<th::Tensor> runtime_top_k;
    th::optional<th::Tensor> runtime_top_p;
    th::optional<th::Tensor> temperature;
    th::optional<th::Tensor> repetition_penalty;
    th::optional<th::Tensor> presence_penalty;
    th::optional<th::Tensor> min_length;
    th::optional<th::Tensor> len_penalty;
    th::optional<th::Tensor> beam_search_diversity_rate;
    th::optional<th::Tensor> random_seed;
    th::optional<th::Tensor> top_p_decay;
    th::optional<th::Tensor> top_p_min;
    th::optional<th::Tensor> top_p_reset_ids;
};

class ModelRequest {
public:
    uint generate_batch_size;
    uint context_batch_size;
    th::Tensor merged_ids;
    th::Tensor combo_tokens;
    th::Tensor input_lengths;
    th::Tensor sequence_lengths;
    th::Tensor prefix_lengths;
    th::Tensor count_length;
    th::Tensor lora_ids;
    bool return_hidden_state;
    bool calculate_loss;
    th::Tensor kv_cache_blocks;
    th::Tensor kv_cache_scales;
};

class ModelOutput {
public:
    th::Tensor logits;
    th::optional<th::Tensor> attentions;
    th::optional<th::Tensor> all_hidden_states;
    th::optional<th::Tensor> last_hidden_states;
    th::optional<th::Tensor> loss;
};

class SamplerRequest {
public:
    int64_t batch_size;
    int32_t step;                 // same as max_input_length
    std::shared_ptr<MergedGenerateConfig> generate_config;
    bool need_setup = false;      // set to true when batch changes

    // GPU tensors
    th::Tensor input_lengths;     // [batch_size * beam_size]
    th::Tensor sequence_lengths;  // [batch_size * beam_size]

    // CPU tensors
    th::Tensor finished;          // [batch_size * beam_size]
    th::Tensor token_ids;         // [batch_size * beam_size, step + 1]
    th::Tensor cum_log_probs;     // [batch_size * beam_size]
};

class SamplerOutput {
public:
    th::Tensor next_tokens;
    th::Tensor finished;
    th::Tensor cum_log_probs;
    th::Tensor sequence_lengths;
};

class MergedRequest {
public:
    ModelRequest model_request;
    SamplerRequest sampler_request;
};

} // namespace rtp_llm
