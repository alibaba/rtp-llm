#pragma once
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/core/Buffer.h"
#include <memory>
#include <optional>



namespace rtp_llm {

class MergedGenerateConfig {
public:
    int64_t                      batch_size;
    int64_t                      beam_size = 1;
    std::optional<rtp_llm::BufferPtr> runtime_top_k;
    std::optional<rtp_llm::BufferPtr> runtime_top_p;
    std::optional<rtp_llm::BufferPtr> temperature;
    std::optional<rtp_llm::BufferPtr> repetition_penalty;
    std::optional<rtp_llm::BufferPtr> presence_penalty;
    std::optional<rtp_llm::BufferPtr> min_length;
    std::optional<rtp_llm::BufferPtr> len_penalty;
    std::optional<rtp_llm::BufferPtr> no_repeat_ngram_size;
    std::optional<rtp_llm::BufferPtr> beam_search_diversity_rate;
    std::optional<rtp_llm::BufferPtr> random_seed;
    std::optional<rtp_llm::BufferPtr> top_p_decay;
    std::optional<rtp_llm::BufferPtr> top_p_min;
    std::optional<rtp_llm::BufferPtr> top_p_reset_ids;
};

class ModelInput {
public:
    uint generate_batch_size;
    uint context_batch_size;
    // rtp_llm::BufferPtr merged_ids;
    rtp_llm::BufferPtr    combo_tokens;
    std::vector<int> input_lengths;
    std::vector<int> sequence_lengths;
    std::vector<int> prefix_lengths;
    rtp_llm::BufferPtr    count_length;
    rtp_llm::BufferPtr    lora_ids;
    bool             return_hidden_state;
    bool             calculate_loss;
    rtp_llm::BufferPtr    kv_cache_blocks;
    rtp_llm::BufferPtr    kv_cache_scales;
};

class ModelOutput {
public:
    rtp_llm::BufferPtr                logits;
};

struct ModelRequest {
public:
    int generate_batch_size;
    int context_batch_size;
    rtp_llm::BufferPtr combo_tokens;                // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_token_type_ids;        // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_position_ids;          // [cumulated_seq_len]
    rtp_llm::BufferPtr input_lengths;               // [batch_size]
    rtp_llm::BufferPtr sequence_lengths;            // [decoder_batch_size]
    rtp_llm::BufferPtr prefix_lengths;              // [batch_size]
    rtp_llm::BufferPtr max_prefix_length;           // [1]
    rtp_llm::BufferPtr kv_cache_blocks;             // [layer_id, batch_size, 2, block_num_per_seq]
    rtp_llm::BufferPtr kv_cache_scales;             // [layer_id, batch_size, 2, block_num_per_seq]
    rtp_llm::BufferPtr attention_mask;              // [batch_size, max_seq_len, max_seq_len + max_reuse_len]
    rtp_llm::BufferPtr lora_ids;                    // [batch_size]
    rtp_llm::BufferPtr lora_input_lengths;          // [batch_size]

    rtp_llm::BufferPtr kv_cache_block_id;    // [batch_size, block_nums], kv cache block block id
    rtp_llm::BufferPtr k_cache_buffer;       // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    rtp_llm::BufferPtr v_cache_buffer;       // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    rtp_llm::BufferPtr k_scale_buffer;       // [layer_num, block_nums, head, seq_size_per_block]
    rtp_llm::BufferPtr v_scale_buffer;       // [layer_num, block_nums, head, seq_size_per_block]
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
