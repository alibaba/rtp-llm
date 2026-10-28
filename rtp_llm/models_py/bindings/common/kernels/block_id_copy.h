#pragma once

#include <cstddef>
#include <cstdint>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace rtp_llm::kernels {

// One layer's KV or scale storage. Stride can include padding between blocks.
struct BlockCopyPlane {
    uint8_t* base;
    uint64_t block_stride;
    uint64_t copy_bytes;
    uint32_t tokens_per_block;
    uint32_t token_bytes;
};

// mappings is a device int32 [count, 4]: (group=0, src, dst, valid tokens).
// A zero valid count or an opaque plane retains whole-block copy semantics.
// Destinations must be unique and must not overwrite another entry's source.
void invokeBlockIdCopy(const BlockCopyPlane* planes,
                       int                   plane_count,
                       const int32_t*        mappings,
                       int                   count,
#if USING_ROCM
                       hipStream_t stream);
#else
                       cudaStream_t stream);
#endif

}  // namespace rtp_llm::kernels
