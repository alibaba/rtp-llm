#include "batch_copy.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {
namespace kernels {

#define DISPATCH_BOOL(BOOL, BOOL_EXPR, ...)                                                                            \
    do {                                                                                                               \
        if (BOOL_EXPR) {                                                                                               \
            constexpr bool BOOL = true;                                                                                \
            (__VA_ARGS__)();                                                                                           \
        } else {                                                                                                       \
            constexpr bool BOOL = false;                                                                               \
            (__VA_ARGS__)();                                                                                           \
        }                                                                                                              \
    } while (0)

static inline int getMultiProcessorCount() {
    int nSM{0};
    int deviceID{0};
    check_cuda_value(cudaGetDevice(&deviceID));
    check_cuda_value(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
    return nSM;
}

static constexpr size_t WARP_SIZE = 32;
static constexpr size_t SEG_SIZE  = WARP_SIZE * sizeof(uint4);
using CopyUnit                    = uint4;

template<bool NeedsCleanup>
__global__ void
batchCopyRowAlignedKernel(char* const* __restrict__ dst, char const* const* __restrict__ src, size_t uniform_size) {
    const size_t batch_idx = blockIdx.x;

    const char* __restrict__ base_src_char = src[batch_idx];
    char* __restrict__ base_dst_char       = dst[batch_idx];
    const size_t total_aligned_rows        = uniform_size / SEG_SIZE;

    const int lane_id            = threadIdx.x % WARP_SIZE;
    const int warp_id            = threadIdx.x / WARP_SIZE;
    const int num_warps_in_block = blockDim.x / WARP_SIZE;

#pragma unroll 4
    for (size_t row_idx = warp_id; row_idx < total_aligned_rows; row_idx += num_warps_in_block) {
        const CopyUnit* __restrict__ row_src = reinterpret_cast<const CopyUnit*>(base_src_char + row_idx * SEG_SIZE);

        CopyUnit* __restrict__ row_dst = reinterpret_cast<CopyUnit*>(base_dst_char + row_idx * SEG_SIZE);

#if USING_CUDA
        CopyUnit temp = __ldcs(&row_src[lane_id]);
        __stcs(&row_dst[lane_id], temp);
#elif USING_ROCM
        CopyUnit temp = row_src[lane_id];
        row_dst[lane_id] = temp;
#endif
    }

    if constexpr (NeedsCleanup) {
        const size_t remaining_bytes = uniform_size % SEG_SIZE;
        if (remaining_bytes > 0) {
            if (warp_id == 0) {
                const size_t aligned_offset = total_aligned_rows * SEG_SIZE;
                for (int i = lane_id; i < remaining_bytes; i += WARP_SIZE) {
                    const size_t offset   = aligned_offset + i;
                    base_dst_char[offset] = base_src_char[offset];
                }
            }
        }
    }
}

template<bool IsAligned>
static __global__ void batchCopy(char* __restrict__ const* __restrict__ dst,
                                 char const* __restrict__ const* __restrict__ src,
                                 size_t* __restrict__ bytes,
                                 size_t batch_size) {
    const size_t seg_idx_stride = blockDim.x * gridDim.x / WARP_SIZE;

    int batch_idx = -1, cur_bytes = 0;

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id      = tid % WARP_SIZE;
    int       seg_idx      = tid / WARP_SIZE;
    int       from_seg_idx = 0, to_seg_idx = 0;

    while (true) {
        // schedude copy segments
        do {
            ++batch_idx;
            if (batch_idx >= batch_size) {
                return;
            }

            from_seg_idx = to_seg_idx;
            cur_bytes    = bytes[batch_idx];
            if constexpr (IsAligned) {
                to_seg_idx += cur_bytes / SEG_SIZE;
            } else {
                to_seg_idx += (cur_bytes + SEG_SIZE - 1) / SEG_SIZE;
            }
        } while (to_seg_idx <= seg_idx);

        // start copying
        char const* const cur_src = src[batch_idx];
        char* const       cur_dst = dst[batch_idx];

        for (; seg_idx < to_seg_idx; seg_idx += seg_idx_stride) {
            const int from_seg_offset = (seg_idx - from_seg_idx) * SEG_SIZE;
            const int offset          = from_seg_offset + lane_id * sizeof(uint4);

            char const* const seg_src = cur_src + offset;
            char* const       seg_dst = cur_dst + offset;

            if constexpr (IsAligned) {
#if USING_CUDA
                const auto tmp = __ldcs(reinterpret_cast<uint4 const*>(seg_src));
                __stcs(reinterpret_cast<uint4*>(seg_dst), tmp);
#elif USING_ROCM
                const auto tmp = *reinterpret_cast<uint4 const*>(seg_src);
                *reinterpret_cast<uint4*>(seg_dst) = tmp;
#endif
            } else {
                const int seg_offset = from_seg_offset + SEG_SIZE;
                if (seg_offset <= cur_bytes) {
                    // aligned to segment size
#if USING_CUDA
                    const auto tmp = __ldcs(reinterpret_cast<uint4 const*>(seg_src));
                    __stcs(reinterpret_cast<uint4*>(seg_dst), tmp);
#elif USING_ROCM
                    const auto tmp = *reinterpret_cast<uint4 const*>(seg_src);
                    *reinterpret_cast<uint4*>(seg_dst) = tmp;
#endif
                } else {
                    // not aligned to segment size
                    if (offset + sizeof(uint4) <= cur_bytes) {
                        // aligned to sizeof(uint4)
#if USING_CUDA
                        const auto tmp = __ldcs(reinterpret_cast<uint4 const*>(seg_src));
                        __stcs(reinterpret_cast<uint4*>(seg_dst), tmp);
#elif USING_ROCM
                        const auto tmp = *reinterpret_cast<uint4 const*>(seg_src);
                        *reinterpret_cast<uint4*>(seg_dst) = tmp;
#endif
                    }

                    const int rest_bytes = cur_bytes % sizeof(uint4);
                    if (rest_bytes != 0) {
                        // not aligned to sizeof(uint4)
                        const int byte_offset = cur_bytes - rest_bytes + lane_id;
                        if (byte_offset < cur_bytes) {
#if USING_CUDA
                            const auto tmp = __ldcs(cur_src + byte_offset);
                            __stcs(cur_dst + byte_offset, tmp);
#elif USING_ROCM
                            const auto tmp = *(cur_src + byte_offset);
                            *(cur_dst + byte_offset) = tmp;
#endif
                        }
                    }
#if USING_CUDA
                    __syncwarp();
#elif USING_ROCM
                    __syncthreads();
#endif
                }
            }
        }
    };
}

void invokeBatchCopy(void* const*           dst,
                     void const* const*     src,
                     size_t*                bytes,
                     size_t                 batch_size,
                     const BatchCopyConfig& config,
                     cudaStream_t           stream) {

    RTP_LLM_LOG_DEBUG("Batch copy config: uniform_size=%zu, is_fully_aligned=%s",
                      config.uniform_size,
                      (config.is_fully_aligned ? "true" : "false"));

    if (batch_size == 0) {
        return;
    }

    if (config.uniform_size > 0) {
        const bool has_remaining_bytes = !config.is_fully_aligned;

        const int     grid_size  = batch_size;
        constexpr int block_size = 512;

        DISPATCH_BOOL(NeedsCleanup, has_remaining_bytes, [&]() {
            batchCopyRowAlignedKernel<NeedsCleanup><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<char* const*>(dst), reinterpret_cast<char const* const*>(src), config.uniform_size);
        });
    } else {
        int sm_count = getMultiProcessorCount();
        DISPATCH_BOOL(IsAligned, config.is_fully_aligned, [&]() {
            constexpr auto kernel           = batchCopy<IsAligned>;
            int            block_num_per_sm = 0;
            constexpr int  block_size       = 1024;
            constexpr int  smem_size        = 0;
            check_cuda_value(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_num_per_sm, kernel, block_size, smem_size));
            kernel<<<sm_count * block_num_per_sm, block_size, smem_size, stream>>>(
                reinterpret_cast<char* const*>(dst), reinterpret_cast<char const* const*>(src), bytes, batch_size);
        });
    }

    check_cuda_error();
}

BatchCopyConfig getBatchCopyConfig(const size_t* bytes_host, size_t batch_size) {
    if (batch_size == 0) {
        return {0, false};
    }

    const size_t first_size            = bytes_host[0];
    bool         all_sizes_are_uniform = true;

    if (first_size == 0) {
        all_sizes_are_uniform = false;
    } else {
        for (size_t i = 1; i < batch_size; ++i) {
            if (bytes_host[i] != first_size) {
                all_sizes_are_uniform = false;
                break;
            }
        }
    }

    if (all_sizes_are_uniform) {
        bool is_aligned = (first_size % SEG_SIZE == 0);
        return {first_size, is_aligned};
    } else {
        bool all_are_seg_aligned = true;
        for (size_t i = 0; i < batch_size; ++i) {
            if (bytes_host[i] % SEG_SIZE != 0) {
                all_are_seg_aligned = false;
                break;
            }
        }
        return {0, all_are_seg_aligned};
    }
}
}  // namespace kernels
}  // namespace rtp_llm
