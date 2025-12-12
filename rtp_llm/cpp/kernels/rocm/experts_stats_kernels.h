#pragma once

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#else
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

namespace rtp_llm {

void launch_equal_expert_balance(int*         experts_ids,
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
                             cudaStream_t stream);

void update_gpu_loads_deepep_kernel(
    int64_t* experts_ids, int* gpu_loads, int total_token_num, int ep_rank, cudaStream_t stream);

void launch_update_gpu_loads_ll(
    int* experts_cnts, int* gpu_loads, int local_experts_num, int ep_rank, cudaStream_t stream);

}  // namespace rtp_llm
