#pragma once

#include <torch/python.h>

namespace rtp_llm {

struct ModelRequest {
public:
    int           generate_batch_size;
    int           context_batch_size;
    torch::Tensor combo_tokens;          // [cumulated_seq_len]
    torch::Tensor combo_token_type_ids;  // [cumulated_seq_len]
    torch::Tensor combo_position_ids;    // [cumulated_seq_len]
    torch::Tensor input_lengths;         // [batch_size]
    torch::Tensor sequence_lengths;      // [decoder_batch_size]
    torch::Tensor prefix_lengths;        // [batch_size]
    torch::Tensor max_prefix_length;     // [1]
    torch::Tensor kv_cache_blocks;       // [layer_id, batch_size, 2, block_num_per_seq]
    torch::Tensor kv_cache_scales;       // [layer_id, batch_size, 2, block_num_per_seq]
    torch::Tensor attention_mask;        // [batch_size, max_seq_len, max_seq_len + max_reuse_len]

    torch::Tensor kv_cache_block_id;  // [batch_size, block_nums], kv cache block block id
    torch::Tensor kv_cache_buffer;    // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    torch::Tensor kv_scale_buffer;    // [layer_num, block_nums, head, seq_size_per_block]
};

}  // namespace rtp_llm
