#include "rtp_llm/cpp/utils/PrefillOptimization.h"
#include <cstdint>
#include <cstddef>
#include <algorithm>

#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

namespace rtp_llm {

static constexpr int FUSED_COPY_BLOCKS_PER_TASK = 8;
static constexpr int FUSED_COPY_THREADS         = 256;

__global__ void fusedCopyKernel(FusedD2DCopyParams params) {
    const int copy_idx = blockIdx.y;
    if (copy_idx >= params.num_copies)
        return;

    const size_t total_bytes   = params.size[copy_idx];
    const size_t global_tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t global_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    const auto src_addr = reinterpret_cast<uintptr_t>(params.src[copy_idx]);
    const auto dst_addr = reinterpret_cast<uintptr_t>(params.dst[copy_idx]);

    if ((src_addr % sizeof(int4) == 0) && (dst_addr % sizeof(int4) == 0)) {
        // Fast path: 16-byte vectorized bulk copy
        const int4*  src = reinterpret_cast<const int4*>(src_addr);
        int4*        dst = reinterpret_cast<int4*>(dst_addr);
        const size_t n16 = total_bytes / sizeof(int4);

        for (size_t i = global_tid; i < n16; i += global_stride) {
            dst[i] = src[i];
        }

        if (blockIdx.x == 0) {
            const size_t rem_start = n16 * sizeof(int4);
            const char*  src_byte  = reinterpret_cast<const char*>(src_addr);
            char*        dst_byte  = reinterpret_cast<char*>(dst_addr);
            for (size_t i = rem_start + threadIdx.x; i < total_bytes; i += blockDim.x) {
                dst_byte[i] = src_byte[i];
            }
        }
    } else {
        // Slow path: byte-by-byte copy for unaligned pointers
        const char* src_byte = reinterpret_cast<const char*>(src_addr);
        char*       dst_byte = reinterpret_cast<char*>(dst_addr);
        for (size_t i = global_tid; i < total_bytes; i += global_stride) {
            dst_byte[i] = src_byte[i];
        }
    }
}

void invokeFusedCopy(const FusedD2DCopyParams& params, cudaStream_t stream) {
    if (params.num_copies <= 0)
        return;
    if (!rtp_llm::prefillOptimizationsEnabled()) {
        dim3 grid(FUSED_COPY_BLOCKS_PER_TASK, params.num_copies);
        fusedCopyKernel<<<grid, FUSED_COPY_THREADS, 0, stream>>>(params);
        return;
    }
    // Small metadata copies retain their launch shape. Large prefill hidden
    // states need more than eight CTAs to use device bandwidth. The size list
    // is already host-resident; this adds no device query or synchronization.
    size_t max_bytes = 0;
    for (int i = 0; i < params.num_copies; ++i) {
        max_bytes = std::max(max_bytes, params.size[i]);
    }
    constexpr size_t bytes_per_block = 128 * 1024;
    const size_t     wanted          = max_bytes / bytes_per_block + (max_bytes % bytes_per_block != 0);
    // Bound empty work when a single large tensor is mixed with many tiny
    // metadata tensors. Preserve the existing minimum for each copy.
    const int cap    = std::max(FUSED_COPY_BLOCKS_PER_TASK, std::min(256, 1024 / params.num_copies));
    const int blocks = static_cast<int>(std::max<size_t>(FUSED_COPY_BLOCKS_PER_TASK, std::min<size_t>(cap, wanted)));
    dim3      grid(blocks, params.num_copies);
    fusedCopyKernel<<<grid, FUSED_COPY_THREADS, 0, stream>>>(params);
}

__global__ void fusedStridedCopyKernel(FusedStridedCopyParams params) {
    const int copy_idx = blockIdx.y;
    if (copy_idx >= params.num_copies)
        return;

    const size_t nrows      = params.num_rows[copy_idx];
    const size_t rbytes     = params.row_bytes[copy_idx];
    const size_t src_stride = params.src_row_stride[copy_idx];
    const size_t dst_stride = params.dst_row_stride[copy_idx];
    const char*  src        = reinterpret_cast<const char*>(params.src[copy_idx]);
    char*        dst        = reinterpret_cast<char*>(params.dst[copy_idx]);

    const size_t total      = nrows * rbytes;
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride     = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = global_tid; idx < total; idx += stride) {
        const size_t row            = idx / rbytes;
        const size_t col            = idx % rbytes;
        dst[row * dst_stride + col] = src[row * src_stride + col];
    }
}

void invokeFusedStridedCopy(const FusedStridedCopyParams& params, cudaStream_t stream) {
    if (params.num_copies <= 0)
        return;
    dim3 grid(FUSED_COPY_BLOCKS_PER_TASK, params.num_copies);
    fusedStridedCopyKernel<<<grid, FUSED_COPY_THREADS, 0, stream>>>(params);
}

}  // namespace rtp_llm
