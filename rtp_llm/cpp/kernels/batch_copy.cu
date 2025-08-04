#include "batch_copy.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"

namespace rtp_llm
{
namespace kernels
{

#define DISPATCH_BOOL(BOOL, BOOL_EXPR, ...) do { \
    if (BOOL_EXPR) {                             \
        constexpr bool BOOL = true;              \
        (__VA_ARGS__)();                         \
    } else {                                     \
        constexpr bool BOOL = false;             \
        (__VA_ARGS__)();                         \
    }                                            \
} while(0)

static inline int getMultiProcessorCount()
{
    int nSM{0};
    int deviceID{0};
    check_cuda_value(cudaGetDevice(&deviceID));
    check_cuda_value(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
    return nSM;
}

static constexpr size_t WARP_SIZE = 32;
// each warp handles a segement (32 * uint4 = 512 bytes = 4 sectors)
static constexpr size_t SEG_SIZE = WARP_SIZE * sizeof(uint4);
static constexpr size_t ROW_SIZE_BYTES = 512;

using CopyUnit = uint4;

template<size_t ROW_SIZE_BYTES_T = ROW_SIZE_BYTES>
__global__ void batchCopyRowAlignedKernel(char* const* __restrict__ dst, 
                                          char const* const* __restrict__ src, 
                                          const size_t* __restrict__ bytes, 
                                          size_t batch_size) 
{
    const size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const char* __restrict__ base_src_char = src[batch_idx];
    char*       __restrict__ base_dst_char = dst[batch_idx];
    const size_t total_rows = bytes[batch_idx] / ROW_SIZE_BYTES_T;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps_in_block = blockDim.x / 32;

    #pragma unroll 4
    for (size_t row_idx = warp_id; row_idx < total_rows; row_idx += num_warps_in_block) {
        const CopyUnit* __restrict__ row_src = 
            reinterpret_cast<const CopyUnit*>(base_src_char + row_idx * ROW_SIZE_BYTES_T);
        
        CopyUnit* __restrict__ row_dst = 
            reinterpret_cast<CopyUnit*>(base_dst_char + row_idx * ROW_SIZE_BYTES_T);


        CopyUnit temp = __ldcs(&row_src[lane_id]); 
        __stcs(&row_dst[lane_id], temp);          
    }
}


template<bool IsAligned>
static __global__ void batchCopy(char *__restrict__ const*__restrict__ dst, 
                                 char const*__restrict__ const*__restrict__ src, 
                                 size_t *__restrict__ bytes, 
                                 size_t batch_size) {
    const size_t seg_idx_stride = blockDim.x * gridDim.x / WARP_SIZE;

    int batch_idx = -1, cur_bytes = 0;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    int seg_idx = tid / WARP_SIZE;
    int from_seg_idx = 0, to_seg_idx = 0;

    while (true) {
        // schedude copy segments
        do {
            ++batch_idx;
            if (batch_idx >= batch_size) {
                return;
            }

            from_seg_idx = to_seg_idx;
            cur_bytes = bytes[batch_idx];
            if constexpr (IsAligned) {
                to_seg_idx += cur_bytes / SEG_SIZE;
            } else {
                to_seg_idx += (cur_bytes + SEG_SIZE - 1) / SEG_SIZE;
            }
        } while (to_seg_idx <= seg_idx);

        // start copying
        char const * const cur_src = src[batch_idx];
        char * const cur_dst       = dst[batch_idx];

        for (; seg_idx < to_seg_idx; seg_idx += seg_idx_stride) {
            const int from_seg_offset = (seg_idx - from_seg_idx) * SEG_SIZE;
            const int offset          = from_seg_offset + lane_id * sizeof(uint4);

            char const * const seg_src = cur_src + offset;
            char * const seg_dst       = cur_dst + offset;

            if constexpr (IsAligned) {
                const auto tmp = __ldcs(reinterpret_cast<uint4 const *>(seg_src));
                __stcs(reinterpret_cast<uint4 *>(seg_dst), tmp);
            } else {
                const int seg_offset = from_seg_offset + SEG_SIZE;
                if (seg_offset <= cur_bytes) {
                    // aligned to segment size
                    const auto tmp = __ldcs(reinterpret_cast<uint4 const *>(seg_src));
                    __stcs(reinterpret_cast<uint4 *>(seg_dst), tmp);
                } else {
                    // not aligned to segment size
                    if (offset + sizeof(uint4) <= cur_bytes) {
                        // aligned to sizeof(uint4)
                        const auto tmp = __ldcs(reinterpret_cast<uint4 const *>(seg_src));
                        __stcs(reinterpret_cast<uint4 *>(seg_dst), tmp);
                    }

                    const int rest_bytes = cur_bytes % sizeof(uint4);
                    if (rest_bytes != 0) {
                        // not aligned to sizeof(uint4)
                        const int byte_offset = cur_bytes - rest_bytes + lane_id;
                        if (byte_offset < cur_bytes) {
                            const auto tmp = __ldcs(cur_src + byte_offset);
                            __stcs(cur_dst + byte_offset, tmp);
                        }
                    }
                    __syncwarp();
                }
            }
        }
    };
}

void invokeBatchCopy(void * const* dst, 
                     void const* const* src, 
                     size_t * bytes, 
                     size_t batch_size, 
                     const BatchCopyConfig &config,
                     cudaStream_t stream) {


    RTP_LLM_LOG_DEBUG("Batch copy config: aligned_copy=%s, rowAligned=%s",
                      (config.aligned_copy ? "true" : "false"),
                      (config.rowAligned ? "true" : "false"));

    if (config.rowAligned)
    {
        if (batch_size == 0) {
            return;
        }
        const int grid_size = batch_size;
        constexpr int block_size = 512; 
        batchCopyRowAlignedKernel<><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<char* const*>(dst), 
            reinterpret_cast<char const* const*>(src), 
            bytes,
            batch_size);
    }
    else
    {
        int sm_count = getMultiProcessorCount();
        DISPATCH_BOOL(IsAligned, config.aligned_copy, [&](){
            constexpr auto kernel = batchCopy<IsAligned>; 
            int block_num_per_sm = 0;
            constexpr int block_size = 1024;
            constexpr int smem_size = 0;
            check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_num_per_sm, kernel, block_size, smem_size));
            kernel<<<sm_count * block_num_per_sm, block_size, smem_size, stream>>>(
                reinterpret_cast<char * const*>(dst), 
                reinterpret_cast<char const* const*>(src), 
                bytes, 
                batch_size);
        });
    }

    check_cuda_error();
}

BatchCopyConfig getBatchCopyConfig(const size_t * bytes_host, size_t batch_size) {
    bool is_row_aligned = true; 
    for (size_t i = 0; i < batch_size; ++i) {
        if (bytes_host[i] % ROW_SIZE_BYTES != 0) {
            is_row_aligned = false;
            break; 
        }
    }
    
    bool is_seg_aligned = is_row_aligned;
    if (!is_row_aligned) {
        is_seg_aligned = true;
        for (size_t i = 0; i < batch_size; ++i) {
            if (bytes_host[i] % SEG_SIZE != 0) {
                is_seg_aligned = false;
                break;
            }
        }
    }

    return {is_seg_aligned, is_row_aligned};
}
}
}