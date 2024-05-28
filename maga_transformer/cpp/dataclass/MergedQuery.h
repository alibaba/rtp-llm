#pragma once
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/core/Buffer.h"
#include <memory>
#include <optional>

namespace ft = fastertransformer;

namespace rtp_llm {

class MergedGenerateConfig {
public:
    int64_t                      batch_size;
    int64_t                      beam_size = 1;
    std::optional<ft::BufferPtr> runtime_top_k;
    std::optional<ft::BufferPtr> runtime_top_p;
    std::optional<ft::BufferPtr> temperature;
    std::optional<ft::BufferPtr> repetition_penalty;
    std::optional<ft::BufferPtr> presence_penalty;
    std::optional<ft::BufferPtr> min_length;
    std::optional<ft::BufferPtr> len_penalty;
    std::optional<ft::BufferPtr> beam_search_diversity_rate;
    std::optional<ft::BufferPtr> random_seed;
    std::optional<ft::BufferPtr> top_p_decay;
    std::optional<ft::BufferPtr> top_p_min;
    std::optional<ft::BufferPtr> top_p_reset_ids;
};

class ModelInput {
public:
    uint generate_batch_size;
    uint context_batch_size;
    // ft::BufferPtr merged_ids;
    ft::BufferPtr    combo_tokens;
    std::vector<int> input_lengths;
    std::vector<int> sequence_lengths;
    std::vector<int> prefix_lengths;
    ft::BufferPtr    count_length;
    ft::BufferPtr    lora_ids;
    bool             return_hidden_state;
    bool             calculate_loss;
    ft::BufferPtr    kv_cache_blocks;
    ft::BufferPtr    kv_cache_scales;
};

class ModelOutput {
public:
    ft::BufferPtr                logits;
};

struct ModelRequest {
public:
    int generate_batch_size;
    int context_batch_size;
    ft::BufferPtr combo_tokens;                // [cumulated_seq_len]
    ft::BufferPtr combo_token_type_ids;        // [cumulated_seq_len]
    ft::BufferPtr combo_position_ids;          // [cumulated_seq_len]
    ft::BufferPtr input_lengths;               // [batch_size]
    ft::BufferPtr sequence_lengths;            // [decoder_batch_size]
    ft::BufferPtr prefix_lengths;              // [batch_size, seq_len]
    ft::BufferPtr kv_cache_blocks;             // [layer_id, batch_size, 2, block_num_per_seq]
    ft::BufferPtr kv_cache_scales;             // [layer_id, batch_size, 2, block_num_per_seq]
    ft::BufferPtr attention_mask;              // [batch_size, max_seq_len, max_seq_len + max_reuse_len]
    ft::BufferPtr lora_ids;                    // [batch_size]
};

struct MergedInput {
public:
    GptModelInputs model_input;
    SamplerInputs  sampler_input;
};

struct MergedOutput {
public:
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
};

}  // namespace rtp_llm
