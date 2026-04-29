#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"

namespace rtp_llm {

// DeepSeek-V4 C4/C128 compression kernel launch functions

// C4 Compression (4:1 ratio)
void invokeFlashCompress4Decode(
    void* kv_score_buffer,          // [num_indices, 8, head_dim * 4]
    const void* kv_score_input,     // [batch_size, head_dim * 4]
    void* kv_compressed_output,     // [batch_size, head_dim]
    const void* score_bias,         // [8, head_dim] (ape)
    const int32_t* indices,         // [batch_size]
    const int32_t* seq_lens,        // [batch_size]
    const int32_t* extra,           // [batch_size, 1] or nullptr (ring buffer mode)
    uint32_t batch_size,
    int64_t head_dim,
    hipStream_t stream);

void invokeFlashCompress4Prefill(
    void* kv_score_buffer,          // [num_indices, 8, head_dim * 4]
    const void* kv_score_input,     // [num_q_tokens, head_dim * 4]
    void* kv_compressed_output,     // [num_q_tokens, head_dim]
    const void* score_bias,         // [8, head_dim] (ape)
    const int32_t* indices,         // [batch_size]
    const int32_t* compress_plan,   // [num_compress, 4] (uint32_t per field)
    const int32_t* write_plan,      // [num_write, 4] (uint32_t per field)
    const int32_t* extra,           // [batch_size, 4] or nullptr
    uint32_t num_compress,
    uint32_t num_write,
    int64_t head_dim,
    hipStream_t stream);

// C128 Compression (128:1 ratio)
void invokeFlashCompress128Decode(
    void* kv_score_buffer,          // [num_indices, 128, head_dim * 2]
    const void* kv_score_input,     // [batch_size, head_dim * 2]
    void* kv_compressed_output,     // [batch_size, head_dim]
    const void* score_bias,         // [128, head_dim] (ape)
    const int32_t* indices,         // [batch_size]
    const int32_t* seq_lens,        // [batch_size]
    uint32_t batch_size,
    int64_t head_dim,
    hipStream_t stream);

void invokeFlashCompress128Prefill(
    void* kv_score_buffer,          // [num_indices, 128, head_dim * 2]
    const void* kv_score_input,     // [num_q_tokens, head_dim * 2]
    void* kv_compressed_output,     // [num_q_tokens, head_dim]
    const void* score_bias,         // [128, head_dim] (ape)
    const int32_t* indices,         // [batch_size]
    const int32_t* compress_plan,   // [num_compress, 4]
    const int32_t* write_plan,      // [num_write, 4]
    uint32_t num_compress,
    uint32_t num_write,
    int64_t head_dim,
    hipStream_t stream);

}  // namespace rtp_llm
