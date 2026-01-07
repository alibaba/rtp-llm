#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
#include "rtp_llm/cpp/cuda/launch_utils.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {
template<typename T>
__global__ void euqal_expert_balance_kernel(T*         experts_ids,
                                            int*       log_stats,
                                            const int* log2phy,
                                            const int* logic_expert_cnt,
                                            int        max_exp_num,
                                            int        total_tokens,
                                            int        ep_rank) {
    // experts_ids: [total_tokens]
    // log_stats: [log_exp_num]
    // log2phy: [log_exp_num * max_exp_num] *value range: [-1, phy_exp_num)
    // logic_expert_cnt: [log_exp_num]      *value range: [1, max_exp_num]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_tokens) {
        return;
    }

    int log_exp_id = experts_ids[i];

    int cnt        = logic_expert_cnt[log_exp_id];
    int idx        = log_exp_id * max_exp_num + (i + ep_rank) % cnt;
    int phy_exp_id = log2phy[idx];
    experts_ids[i] = phy_exp_id;

    if (log_stats != nullptr) {
        atomicAdd(&log_stats[log_exp_id], 1);
    }
}

__global__ void
update_gpu_loads_kernel(int* experts_ids, int* gpu_loads, int total_token_num, int experts_per_gpu, int ep_rank) {
    // use shared memory to reduce global memory access
    __shared__ int shared_sum[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    int contribute = 0;
    if (i < total_token_num) {
        int gpu_id = experts_ids[i] / experts_per_gpu;
        contribute = (gpu_id == ep_rank) ? 1 : 0;
    }

    // save the contribution to shared memory
    shared_sum[tid] = contribute;
    __syncthreads();

    // reduce the contributions in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // write the result to global memory
    if (tid == 0) {
        atomicAdd(&gpu_loads[ep_rank], shared_sum[0]);
    }
}

__global__ void update_gpu_loads_deepep_kernel(int64_t* experts_ids, int* gpu_loads, int total_token_num, int ep_rank) {
    // use shared memory to reduce global memory access
    __shared__ int shared_sum[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    int contribute = 0;
    if (i < total_token_num) {
        contribute = (experts_ids[i] >= 0) ? 1 : 0;
    }

    // save the contribution to shared memory
    shared_sum[tid] = contribute;
    __syncthreads();

    // reduce the contributions in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // write the result to global memory
    if (tid == 0) {
        atomicAdd(&gpu_loads[ep_rank], shared_sum[0]);
    }
}

__global__ void update_gpu_loads_ll(int* experts_cnts, int* gpu_loads, int local_experts_num, int ep_rank) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // since the local_experts_num is small, we can use atomicAdd to update the gpu_loads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_experts_num) {
        return;
    }
    atomicAdd(&gpu_loads[ep_rank], experts_cnts[i]);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template<typename T>
void launch_equal_expert_balance(T*           experts_ids,
                                 int*         log_stats,
                                 const int*   log2phy,
                                 const int*   logic_expert_cnt,
                                 int          log_exp_num,
                                 int          phy_exp_num,
                                 int          total_tokens,
                                 int          ep_rank,
                                 cudaStream_t stream) {
    int max_exp_num = phy_exp_num - log_exp_num + 1;
    int block_size  = 256;
    int grid_size   = (total_tokens + block_size - 1) / block_size;
    if (grid_size > 0) {
        euqal_expert_balance_kernel<T><<<grid_size, block_size, 0, stream>>>(
            experts_ids, log_stats, log2phy, logic_expert_cnt, max_exp_num, total_tokens, ep_rank);
    }
}

template void launch_equal_expert_balance(int64_t*     experts_ids,
                                          int*         log_stats,
                                          const int*   log2phy,
                                          const int*   logic_expert_cnt,
                                          int          log_exp_num,
                                          int          phy_exp_num,
                                          int          total_tokens,
                                          int          ep_rank,
                                          cudaStream_t stream);

template void launch_equal_expert_balance(int*         experts_ids,
                                          int*         log_stats,
                                          const int*   log2phy,
                                          const int*   logic_expert_cnt,
                                          int          log_exp_num,
                                          int          phy_exp_num,
                                          int          total_tokens,
                                          int          ep_rank,
                                          cudaStream_t stream);

void launch_update_gpu_loads(int*         experts_ids,
                             int*         gpu_loads,
                             int          total_token_num,
                             int          phy_exp_num,
                             int          ep_rank,
                             int          ep_size,
                             cudaStream_t stream) {
    int block_size      = 256;
    int experts_per_gpu = phy_exp_num / ep_size;
    int grid_size       = (total_token_num + block_size - 1) / block_size;
    if (grid_size > 0) {
        update_gpu_loads_kernel<<<grid_size, block_size, sizeof(int) * block_size, stream>>>(
            experts_ids, gpu_loads, total_token_num, experts_per_gpu, ep_rank);
    }
}

void update_gpu_loads_deepep_kernel(
    int64_t* experts_ids, int* gpu_loads, int total_token_num, int ep_rank, cudaStream_t stream) {
    int block_size = 256;
    int grid_size  = (total_token_num + block_size - 1) / block_size;
    if (grid_size > 0) {
        update_gpu_loads_deepep_kernel<<<grid_size, block_size, sizeof(int) * block_size, stream>>>(
            experts_ids, gpu_loads, total_token_num, ep_rank);
    }
}

void launch_update_gpu_loads_ll(
    int* experts_cnts, int* gpu_loads, int local_experts_num, int ep_rank, cudaStream_t stream) {
    int block_size = 128;
    int grid_size  = (local_experts_num + block_size - 1) / block_size;
    LAUNCH_KERNEL_WITH_PDL(
        update_gpu_loads_ll, grid_size, block_size, 0, stream, experts_cnts, gpu_loads, local_experts_num, ep_rank);
}

};  // namespace rtp_llm
