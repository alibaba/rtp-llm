#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm::kernels {

// One layer's KV or scale storage. Stride can include padding between blocks.
struct BlockCopyPlane {
    uint8_t* base;
    uint64_t block_stride;
    uint64_t copy_bytes;
};

// mappings is a device int32 [count, 3] array: (group=0, src block, dst block).
// Destinations must be unique and must not overwrite another entry's source.
void invokeBlockIdCopy(const BlockCopyPlane* planes,
                       int                   plane_count,
                       const int32_t*        mappings,
                       int                   count,
                       cudaStream_t          stream);

}  // namespace rtp_llm::kernels
