// copy from src/fastertransformer/cutlass/cutlass_kernels/moe_gemm/moe_kernels.h
// but remove all moeGemm part

#pragma once
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif
#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif
namespace fastertransformer {

/**
 * \brief Describes what parallelism mode the MoE is using
 *
 * Tensor Parallelism refers to the mode where the weight matrices for each expert are sliced up between nodes.
 * Each node will handle part of each expert, the final result is achieved by summing the result.
 * The inter_size dimension should be divided by the number of nodes prior to passing it to the MoE plugin, only the
 * required slice of the weights should be provided to the plugin FC1 is a ColumnLinear and FC2 is a RowLinear, see
 * tensorrt_llm/mlp/mlp.py for an example of how this works for a single MLP
 *
 * NOTE: The bias for fc2 is only applied on rank 0. If we added it on all nodes the allreduce() would contain multiple
 * copies of the bias. The bias on other node will be ignored, and may be set to nullptr
 *
 * Expert Parallelism refers to the mode where experts are divided between the nodes. Each node will handle only the
 * tokens that are routed to the experts it is assigned to. Only the weights for the node's experts should be provided
 * to the plugin For example, with #experts = 8, expert parallelism = 2: Node 0 would handle experts 0-3, and node 1
 * would handle experts 4-7
 *
 * Regardless of parallelism mode:
 *  * The input routing values must be the complete routing for all tokens/experts (required for softmax)
 *  * An allreduce must be run on the result to combine the results from different nodes if parallelism > 1
 */
struct MOEParallelismConfig {
    constexpr static MOEParallelismConfig TensorParallelism(int tp_size, int tp_rank) {
        return {tp_size, tp_rank, 1, 0};
    }

    constexpr static MOEParallelismConfig ExpertParallelism(int ep_size, int ep_rank) {
        return {1, 0, ep_size, ep_rank};
    }

    const int tp_size = 1;
    const int tp_rank = 0;
    const int ep_size = 1;
    const int ep_rank = 0;
};
/*
  Launches the topk gating softmax required for the MoE layers.

  Params:
  input - a [num_rows x num_experts]
  finished - [num_rows] vector with 1 if the sentence at this row is done translating and 0 otherwise.
  output - a buffer of shape [num_rows x k] containing the top-k values of the softmax for each row.
  indices - a matrix of shape [num_rows x k] containing the top-k experts each row should get routed to.
  source_rows - a matrix of shape [num_rows x k] used internally for permuting. source_rows[row][k] =  k * num_rows +
  row. It is constructed like this so we can track where each of the original rows end up in order to perform the
                "k-way" reduction later in the routing.

  num_rows - The number of rows in the matrix
  num_experts - The number of expert layers present
  k - k value in topk
*/
void topkGatingSoftmax_KL(const float* input,
                          const bool*  finished,
                          float*       softmax_temp_output,
                          float*       topk_scales,
                          int*         topk_expertID,
                          int*         topk_rowColID,
                          const int    num_rows,
                          const int    num_experts,
                          const int    num_cols,
                          const int    start_expert,
                          const int    end_expert,
                          cudaStream_t stream);

void sort_KL(const int*   keys_in,
             const int*   values_in,
             int*         keys_out,
             int*         values_out,
             const size_t num_key_value_pairs,
             size_t       num_bits,
             cudaStream_t stream);

void computeTotalRowsBeforeExpert_KL(const int*   sorted_indices,
                                     const int    total_indices,
                                     const int    num_experts,
                                     int*         total_rows_before_expert,
                                     cudaStream_t stream);

template<typename T>
void permutInputRows_KL(const T*       unpermuted_input,
                        T*             permuted_output,
                        const int*     expanded_dest_row_to_expanded_source_row,
                        int*           expanded_source_row_to_expanded_dest_row,
                        const int      num_rows,
                        const int      num_cols,
                        const int64_t* num_valid_tokens_ptr,
                        const int      k,
                        cudaStream_t   stream);

template<typename T>
void finalizeMoeRoutingKernelLauncher(const T*       expanded_permuted_rows,
                                      T*             reduced_unpermuted_output,
                                      const T*       skip_1,
                                      const T*       skip_2,
                                      const T*       bias,
                                      const float*   scales,
                                      const int*     expanded_source_row_to_expanded_dest_row,
                                      const int*     expert_for_source_row,
                                      const int      num_rows,
                                      const int      cols,
                                      const int      k,
                                      const int64_t* num_valid_ptr,
                                      const int      tp_rank,
                                      int            normalization_mode,
                                      cudaStream_t   stream);

}  // namespace fastertransformer
