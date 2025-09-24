#pragma once

#include <stdint.h>

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

template<typename T>
void launch_equal_expert_balance(T*           experts_ids,
                                 int*         log_stats,
                                 const int*   log2phy,
                                 const int*   logic_expert_cnt,
                                 int          log_exp_num,
                                 int          phy_exp_num,
                                 int          total_tokens,
                                 int          ep_rank,
#if USING_CUDA
                                 cudaStream_t stream);
#elif USING_ROCM
                                 hipStream_t stream);
#endif

void launch_update_gpu_loads(int*         experts_ids,
                             int*         gpu_loads,
                             int          total_token_num,
                             int          phy_exp_num,
                             int          ep_rank,
                             int          ep_size,
#if USING_CUDA
                             cudaStream_t stream);
#elif USING_ROCM
                             hipStream_t stream);
#endif

void update_gpu_loads_deepep_kernel(
    int64_t* experts_ids, int* gpu_loads, int total_token_num, int ep_rank, 
#if USING_CUDA
    cudaStream_t stream);
#elif USING_ROCM
    hipStream_t stream);
#endif

void launch_update_gpu_loads_ll(
    int* experts_cnts, int* gpu_loads, int local_experts_num, int ep_rank, 
#if USING_CUDA
    cudaStream_t stream);
#elif USING_ROCM
    hipStream_t stream);
#endif

}  // namespace rtp_llm
