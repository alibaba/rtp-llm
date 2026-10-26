#include <cstdint>
#include <cstddef>

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
    dim3 grid(FUSED_COPY_BLOCKS_PER_TASK, params.num_copies);
    fusedCopyKernel<<<grid, FUSED_COPY_THREADS, 0, stream>>>(params);
}

static constexpr size_t STRIDED_COPY_TILE_BYTES = 16 * 1024;
static constexpr int    STRIDED_COPY_MAX_BLOCKS = 256;

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
    if (nrows == 0 || rbytes == 0)
        return;

    const size_t tiles_per_row = (rbytes - 1) / STRIDED_COPY_TILE_BYTES + 1;
    const bool   aligned       = (reinterpret_cast<uintptr_t>(src) % sizeof(int4) == 0)
                         && (reinterpret_cast<uintptr_t>(dst) % sizeof(int4) == 0) && (src_stride % sizeof(int4) == 0)
                         && (dst_stride % sizeof(int4) == 0);
    for (size_t tile = blockIdx.x; tile < nrows * tiles_per_row; tile += gridDim.x) {
        const size_t row        = tile / tiles_per_row;
        const size_t offset     = (tile % tiles_per_row) * STRIDED_COPY_TILE_BYTES;
        const size_t remaining  = rbytes - offset;
        const size_t copy_bytes = remaining < STRIDED_COPY_TILE_BYTES ? remaining : STRIDED_COPY_TILE_BYTES;
        const char*  src_row    = src + row * src_stride + offset;
        char*        dst_row    = dst + row * dst_stride + offset;

        if (aligned) {
            const size_t n16   = copy_bytes / sizeof(int4);
            const int4*  src16 = reinterpret_cast<const int4*>(src_row);
            int4*        dst16 = reinterpret_cast<int4*>(dst_row);
            for (size_t i = threadIdx.x; i < n16; i += blockDim.x)
                dst16[i] = src16[i];
            for (size_t i = n16 * sizeof(int4) + threadIdx.x; i < copy_bytes; i += blockDim.x)
                dst_row[i] = src_row[i];
        } else {
            for (size_t i = threadIdx.x; i < copy_bytes; i += blockDim.x)
                dst_row[i] = src_row[i];
        }
    }
}

void invokeFusedStridedCopy(const FusedStridedCopyParams& params, cudaStream_t stream) {
    if (params.num_copies <= 0)
        return;
    int grid_x = FUSED_COPY_BLOCKS_PER_TASK;
    for (int i = 0; i < params.num_copies; ++i) {
        const size_t row_bytes     = params.row_bytes[i];
        const size_t tiles_per_row = row_bytes == 0 ? 0 : (row_bytes - 1) / STRIDED_COPY_TILE_BYTES + 1;
        const size_t blocks        = params.num_rows[i] * tiles_per_row;
        if (blocks > static_cast<size_t>(grid_x))
            grid_x = static_cast<int>(blocks < STRIDED_COPY_MAX_BLOCKS ? blocks : STRIDED_COPY_MAX_BLOCKS);
    }
    dim3 grid(grid_x, params.num_copies);
    fusedStridedCopyKernel<<<grid, FUSED_COPY_THREADS, 0, stream>>>(params);
}

}  // namespace rtp_llm
