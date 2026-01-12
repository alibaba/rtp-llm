#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

struct ModelRequest {
public:
    int                generate_batch_size;
    int                context_batch_size;
    rtp_llm::BufferPtr combo_tokens;          // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_token_type_ids;  // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_position_ids;    // [cumulated_seq_len]
    rtp_llm::BufferPtr input_lengths;         // [batch_size]
    rtp_llm::BufferPtr sequence_lengths;      // [decoder_batch_size]
    rtp_llm::BufferPtr prefix_lengths;        // [batch_size]
    rtp_llm::BufferPtr max_prefix_length;     // [1]
    rtp_llm::BufferPtr kv_cache_blocks;       // [layer_id, batch_size, 2, block_num_per_seq]
    rtp_llm::BufferPtr kv_cache_scales;       // [layer_id, batch_size, 2, block_num_per_seq]
    rtp_llm::BufferPtr attention_mask;        // [batch_size, max_seq_len, max_seq_len + max_reuse_len]
    rtp_llm::BufferPtr lora_ids;              // [batch_size]
    rtp_llm::BufferPtr lora_input_lengths;    // [batch_size]

    rtp_llm::BufferPtr kv_cache_block_id;  // [batch_size, block_nums], kv cache block block id
    rtp_llm::BufferPtr kv_cache_buffer;     // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    rtp_llm::BufferPtr kv_scale_buffer;     // [layer_num, block_nums, head, seq_size_per_block]
};

} // namespace rtp_llm
