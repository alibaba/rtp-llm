#include "rtp_llm/models_py/bindings/common/kernels/block_id_copy.h"

namespace rtp_llm::kernels {
namespace {

__global__ void blockIdCopyKernel(const BlockCopyPlane* __restrict__ planes,
                                  const int32_t* __restrict__ mappings) {
    const int src_id = mappings[3 * blockIdx.x + 1];
    const int dst_id = mappings[3 * blockIdx.x + 2];
    if (src_id == dst_id) {
        return;
    }
    const auto plane = planes[blockIdx.y];
    const auto* src = plane.base + static_cast<uint64_t>(src_id) * plane.block_stride;
    auto* dst = plane.base + static_cast<uint64_t>(dst_id) * plane.block_stride;
    if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
        const uint64_t vectors = plane.copy_bytes / sizeof(uint4);
        for (uint64_t i = threadIdx.x; i < vectors; i += blockDim.x) {
            const auto value = __ldcs(reinterpret_cast<const uint4*>(src) + i);
            __stcs(reinterpret_cast<uint4*>(dst) + i, value);
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
                       cudaStream_t          stream) {
    if (count != 0 && plane_count != 0) {
        blockIdCopyKernel<<<dim3(count, plane_count), 256, 0, stream>>>(planes, mappings);
    }
}

}  // namespace rtp_llm::kernels
