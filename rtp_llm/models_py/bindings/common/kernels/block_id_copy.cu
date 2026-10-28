#include "rtp_llm/models_py/bindings/common/kernels/block_id_copy.h"

namespace rtp_llm::kernels {
namespace {

__device__ inline void copyVector(uint4* dst, const uint4* src) {
#if USING_ROCM
    *dst = *src;
#else
    const auto value = __ldcs(src);
    __stcs(dst, value);
#endif
}

__global__ void blockIdCopyKernel(const BlockCopyPlane* __restrict__ planes, const int32_t* __restrict__ mappings) {
    const int src_id = mappings[4 * blockIdx.x + 1];
    const int dst_id = mappings[4 * blockIdx.x + 2];
    if (src_id == dst_id) {
        return;
    }
    const auto     plane        = planes[blockIdx.y];
    const auto*    src          = plane.base + static_cast<uint64_t>(src_id) * plane.block_stride;
    auto*          dst          = plane.base + static_cast<uint64_t>(dst_id) * plane.block_stride;
    const uint32_t valid_tokens = mappings[4 * blockIdx.x + 3];
    if (plane.tokens_per_block && valid_tokens && valid_tokens < plane.tokens_per_block) {
        // HND is [K/V, head, token, dim]. The valid prefix repeats for every
        // K/V head; copying just the first fraction of the page would corrupt V.
        const uint32_t head_stride = plane.tokens_per_block * plane.token_bytes;
        const uint32_t live_bytes  = valid_tokens * plane.token_bytes;
        const uint32_t heads       = plane.copy_bytes / head_stride;
        if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) | head_stride | live_bytes) & 15)
            == 0) {
            const uint32_t live_vectors = live_bytes / sizeof(uint4);
            const uint32_t head_vectors = head_stride / sizeof(uint4);
            for (uint32_t i = threadIdx.x; i < heads * live_vectors; i += blockDim.x) {
                const uint32_t offset = (i / live_vectors) * head_vectors + i % live_vectors;
                copyVector(reinterpret_cast<uint4*>(dst) + offset, reinterpret_cast<const uint4*>(src) + offset);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < heads * live_bytes; i += blockDim.x) {
                const uint32_t offset = (i / live_bytes) * head_stride + i % live_bytes;
                dst[offset]           = src[offset];
            }
        }
        return;
    }
    if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
        const uint64_t vectors = plane.copy_bytes / sizeof(uint4);
        for (uint64_t i = threadIdx.x; i < vectors; i += blockDim.x) {
            copyVector(reinterpret_cast<uint4*>(dst) + i, reinterpret_cast<const uint4*>(src) + i);
        }
        for (uint64_t i = vectors * sizeof(uint4) + threadIdx.x; i < plane.copy_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        for (uint64_t i = threadIdx.x; i < plane.copy_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

}  // namespace

void invokeBlockIdCopy(const BlockCopyPlane* planes,
                       int                   plane_count,
                       const int32_t*        mappings,
                       int                   count,
#if USING_ROCM
                       hipStream_t stream) {
#else
                       cudaStream_t stream) {
#endif
    if (count != 0 && plane_count != 0) {
        blockIdCopyKernel<<<dim3(count, plane_count), 256, 0, stream>>>(planes, mappings);
    }
}

}  // namespace rtp_llm::kernels
