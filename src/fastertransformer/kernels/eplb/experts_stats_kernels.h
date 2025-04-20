#pragma once
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace fastertransformer {

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

void launch_update_gpu_loads_ll(
    int* experts_cnts, int* gpu_loads, int local_experts_num, int ep_rank, cudaStream_t stream);

}  // namespace fastertransformer