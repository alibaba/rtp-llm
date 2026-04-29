#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace rtp_llm {

// DeepSeek-V4 TopK512 radix histogram kernel

void invokeTopK512(
    const float* scores,            // [batch_size, score_stride]
    const int32_t* seq_lens,        // [batch_size]
    const int32_t* page_table,      // [batch_size, page_table_stride]
    int32_t* page_indices,          // [batch_size, 512]
    int32_t* raw_indices,           // [batch_size, 512] or nullptr
    uint32_t batch_size,
    int64_t score_stride,
    int64_t page_table_stride,
    uint32_t page_size,             // must be power of 2
    hipStream_t stream);

}  // namespace rtp_llm
