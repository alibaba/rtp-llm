#pragma once
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/core/allocator.h"

// namespace th = torch;

namespace fastertransformer {

struct GptCommonInputs {
    uint   generate_batch_size;
    uint   context_batch_size;
    uint   max_context_seq_length;
    uint   max_generate_seq_length;
    Tensor position_ids;
    Tensor attention_mask;
    Tensor linear_bias_slopes;
    Tensor input_lengths;
    Tensor sequence_lengths;
    Tensor padding_offset;
    Tensor cu_seqlens;
    Tensor kv_cache_blocks;
    Tensor kv_cache_scales;
    // KVBlockArray kv_cache;
    Tensor prefix_prompt_lengths;
    bool   count_prefix_length;
    uint   max_prefix_length;
    Tensor lora_ids;
    Tensor lora_input_lengths;
};

}  // namespace fastertransformer
