#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <iostream>
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/fp8_group_gemm.h"

constexpr uint64_t THREADS_PER_EXPERT = 512;
// threshold must match the dispatch logic in run_cutlass_moe_mm_sm90()
constexpr int SWAP_AB_THRESHOLD = 64;

__global__ void compute_problem_sizes(const int32_t* __restrict__ topk_ids,
                                      int32_t*   problem_sizes1,
                                      int32_t*   problem_sizes2,
                                      int32_t*   atomic_buffer,
                                      const int  topk_length,
                                      const int  n,
                                      const int  k,
                                      const bool problem_1_swap_ab,
                                      const bool problem_2_swap_ab) {
    int expert_id = blockIdx.x;

    int occurrences = 0;
    for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        occurrences += (topk_ids[i] == expert_id);
    }
    atomicAdd(&atomic_buffer[expert_id], occurrences);
    __syncthreads();

    if (threadIdx.x == 0) {
        int final_occurrences = atomic_buffer[expert_id];
        if (!problem_1_swap_ab) {
            problem_sizes1[expert_id * 3]     = final_occurrences;
            problem_sizes1[expert_id * 3 + 1] = 2 * n;
            problem_sizes1[expert_id * 3 + 2] = k;
        } else {
            problem_sizes1[expert_id * 3]     = 2 * n;
            problem_sizes1[expert_id * 3 + 1] = final_occurrences;
            problem_sizes1[expert_id * 3 + 2] = k;
        }
        if (!problem_2_swap_ab) {
            problem_sizes2[expert_id * 3]     = final_occurrences;
            problem_sizes2[expert_id * 3 + 1] = k;
            problem_sizes2[expert_id * 3 + 2] = n;
        } else {
            problem_sizes2[expert_id * 3]     = k;
            problem_sizes2[expert_id * 3 + 1] = final_occurrences;
            problem_sizes2[expert_id * 3 + 2] = n;
        }
    }
}

__global__ void compute_expert_offsets(const int32_t* __restrict__ problem_sizes1,
                                       int32_t*   expert_offsets,
                                       int32_t*   atomic_buffer,
                                       const int  num_experts,
                                       const bool swap_ab) {
    int32_t tot_offset = 0;
    expert_offsets[0]  = 0;
    for (int i = 0; i < num_experts; ++i) {
        atomic_buffer[i] = tot_offset;
        tot_offset += swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
        expert_offsets[i + 1] = tot_offset;
    }
}

__global__ void compute_expert_blockscale_offsets(const int32_t* __restrict__ problem_sizes1,
                                                  int32_t*   expert_offsets,
                                                  int32_t*   blockscale_offsets,
                                                  int32_t*   atomic_buffer,
                                                  const int  num_experts,
                                                  const bool swap_ab) {
    int32_t tot_offset       = 0;
    int32_t tot_offset_round = 0;
    expert_offsets[0]        = 0;
    blockscale_offsets[0]    = 0;
    for (int i = 0; i < num_experts; ++i) {
        int32_t cur_offset = swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
        atomic_buffer[i]   = tot_offset;
        tot_offset += cur_offset;
        expert_offsets[i + 1] = tot_offset;
        tot_offset_round += (cur_offset + (128 - 1)) / 128 * 128;
        blockscale_offsets[i + 1] = tot_offset_round;
    }
}

__global__ void compute_arg_sorts(const int32_t* __restrict__ topk_ids,
                                  const int32_t* __restrict__ expert_offsets,
                                  int32_t*  input_permutation,
                                  int32_t*  output_permutation,
                                  int32_t*  atomic_buffer,
                                  const int topk_length,
                                  const int topk) {
    int const     blk_expert_id = blockIdx.x;
    int const     num_experts   = gridDim.x;
    int32_t const num_tokens    = expert_offsets[num_experts];

    for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        int const expert_id = topk_ids[i];
        if (expert_id == -1 && blockIdx.x == 0) {
            // output_permutation is used to re-order the moe outputs. It is
            // used as c2 = c2[c_map], where c2 is a torch.tensor that is the
            // output of the cutlass kernels and c_map is the output_permutation.
            // c2 is initialized to zeros, therefore by setting the output_permutation
            // to num_tokens, we are guaranteed to fill the moe outputs to zero
            // for "invalid" topk_ids.
            output_permutation[i] = num_tokens;
        } else if (expert_id == blk_expert_id) {
            int start                = atomicAdd(&atomic_buffer[expert_id], 1);
            input_permutation[start] = i / topk;
            output_permutation[i]    = start;
        }
    }
}

__global__ void compute_batched_problem_data(int32_t* expert_offsets,
                                             int32_t* problem_sizes1,
                                             int32_t* problem_sizes2,
                                             const int32_t* __restrict__ expert_num_tokens,
                                             const int  padded_m,
                                             const int  n,
                                             const int  k,
                                             const bool problem_1_swap_ab,
                                             const bool problem_2_swap_ab) {
    int expert_idx             = threadIdx.x;
    expert_offsets[expert_idx] = expert_idx * padded_m;

    if (!problem_1_swap_ab) {
        problem_sizes1[expert_idx * 3]     = expert_num_tokens[expert_idx];
        problem_sizes1[expert_idx * 3 + 1] = 2 * n;
        problem_sizes1[expert_idx * 3 + 2] = k;
    } else {
        problem_sizes1[expert_idx * 3]     = 2 * n;
        problem_sizes1[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
        problem_sizes1[expert_idx * 3 + 2] = k;
    }
    if (!problem_2_swap_ab) {
        problem_sizes2[expert_idx * 3]     = expert_num_tokens[expert_idx];
        problem_sizes2[expert_idx * 3 + 1] = k;
        problem_sizes2[expert_idx * 3 + 2] = n;
    } else {
        problem_sizes2[expert_idx * 3]     = k;
        problem_sizes2[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
        problem_sizes2[expert_idx * 3 + 2] = n;
    }
}

void get_cutlass_batched_moe_mm_data_caller(torch::Tensor&       expert_offsets,
                                            torch::Tensor&       problem_sizes1,
                                            torch::Tensor&       problem_sizes2,
                                            const torch::Tensor& expert_num_tokens,
                                            const int64_t        num_local_experts,
                                            const int64_t        padded_m,
                                            const int64_t        n,
                                            const int64_t        k,
                                            const bool           problem_1_swap_ab,
                                            const bool           problem_2_swap_ab) {
    auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());

    compute_batched_problem_data<<<1, num_local_experts, 0, stream>>>(
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(problem_sizes2.data_ptr()),
        static_cast<const int32_t*>(expert_num_tokens.data_ptr()),
        padded_m,
        n,
        k,
        problem_1_swap_ab,
        problem_2_swap_ab);
}

void rtp_llm::get_cutlass_batched_moe_mm_data(torch::Tensor&       expert_offsets,
                                              torch::Tensor&       problem_sizes1,
                                              torch::Tensor&       problem_sizes2,
                                              const torch::Tensor& expert_num_tokens,
                                              const int64_t        num_local_experts,
                                              const int64_t        padded_m,
                                              const int64_t        n,
                                              const int64_t        k,
                                              const bool           problem_1_swap_ab,
                                              const bool           problem_2_swap_ab) {
    int32_t version_num = get_sm_version_num();
    // #if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) || \
//     (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100)
    get_cutlass_batched_moe_mm_data_caller(expert_offsets,
                                           problem_sizes1,
                                           problem_sizes2,
                                           expert_num_tokens,
                                           num_local_experts,
                                           padded_m,
                                           n,
                                           k,
                                           problem_1_swap_ab,
                                           problem_2_swap_ab);
    return;
    // #endif
    //   TORCH_CHECK_NOT_IMPLEMENTED(
    //       false,
    //       "No compiled get_cutlass_pplx_moe_mm_data: no cutlass_scaled_mm kernel "
    //       "for CUDA device capability: ",
    //       version_num, ". Required capability: 90 or 100");
}

void rtp_llm::get_cutlass_moe_mm_without_permute_info(const torch::Tensor&                topk_ids,
                                                      torch::Tensor&                      expert_offsets,
                                                      torch::Tensor&                      problem_sizes1,
                                                      torch::Tensor&                      problem_sizes2,
                                                      const int64_t                       num_experts,
                                                      const int64_t                       n,
                                                      const int64_t                       k,
                                                      const bool                          problem_1_swap_ab,
                                                      const bool                          problem_2_swap_ab,
                                                      const std::optional<torch::Tensor>& blockscale_offsets) {
    auto          stream        = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
    auto          options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
    torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

    int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());

    const int32_t* topk_ptr   = static_cast<const int32_t*>(topk_ids.data_ptr());
    int32_t*       ps1_ptr    = static_cast<int32_t*>(problem_sizes1.data_ptr());
    int32_t*       ps2_ptr    = static_cast<int32_t*>(problem_sizes2.data_ptr());
    int32_t*       atomic_ptr = static_cast<int32_t*>(atomic_buffer.data_ptr());

    compute_problem_sizes<<<num_experts, num_threads, 0, stream>>>(topk_ptr,
                                                                   ps1_ptr,
                                                                   ps2_ptr,
                                                                   atomic_ptr,
                                                                   static_cast<int>(topk_ids.numel()),
                                                                   static_cast<int>(n),
                                                                   static_cast<int>(k),
                                                                   problem_1_swap_ab,
                                                                   problem_2_swap_ab);

    compute_expert_offsets<<<1, 1, 0, stream>>>(static_cast<const int32_t*>(problem_sizes1.data_ptr()),
                                                static_cast<int32_t*>(expert_offsets.data_ptr()),
                                                static_cast<int32_t*>(atomic_buffer.data_ptr()),
                                                static_cast<int>(num_experts),
                                                problem_1_swap_ab);
}
