#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace rtp_llm {

// DeepSeek-V4 FusedStoreCache kernels

// FlashMLA variant: input [num_tokens, 512], cache stores FP8 values + UE8M0 scales
void invokeFusedStoreCacheFlashMLA(
    const void* input,              // [num_tokens, 512] (float or half)
    void* cache,                    // uint8_t cache buffer
    const void* indices,            // [num_tokens] (int32_t or int64_t)
    uint32_t num_tokens,
    uint32_t page_size,             // power of 2
    hipStream_t stream);

// Indexer variant: input [num_tokens, 128], cache stores FP8 values + FP32 scales
void invokeFusedStoreCacheIndexer(
    const void* input,              // [num_tokens, 128] (float or half)
    void* cache,                    // uint8_t cache buffer
    const void* indices,            // [num_tokens] (int32_t or int64_t)
    uint32_t num_tokens,
    uint32_t page_size,             // power of 2
    hipStream_t stream);

}  // namespace rtp_llm
