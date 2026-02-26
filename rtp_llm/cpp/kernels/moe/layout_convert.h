#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/all.h>

namespace rtp_llm {

/**
 * Convert contiguous layout to masked layout for FP16.
 *
 * @param contiguous_data Input tensor [total_tokens, hidden_dim]
 * @param grouped_layout Expert ID for each token [total_tokens]
 * @param masked_data Output tensor [num_experts, max_tokens, hidden_dim]
 * @param mask Output: number of tokens per expert [num_experts]
 * @param total_tokens Total number of tokens
 * @param hidden_dim Hidden dimension size
 * @param max_tokens Maximum tokens per expert
 * @param num_experts Number of experts
 * @param stream CUDA stream
 */
void contiguous_to_masked_fp16(const half*  contiguous_data,
                               const int*   grouped_layout,
                               half*        masked_data,
                               int*         mask,
                               int          total_tokens,
                               int          hidden_dim,
                               int          max_tokens,
                               int          num_experts,
                               cudaStream_t stream = 0);

/**
 * Convert contiguous layout to masked layout for BF16.
 */
void contiguous_to_masked_bf16(const __nv_bfloat16* contiguous_data,
                               const int*           grouped_layout,
                               __nv_bfloat16*       masked_data,
                               int*                 mask,
                               int                  total_tokens,
                               int                  hidden_dim,
                               int                  max_tokens,
                               int                  num_experts,
                               cudaStream_t         stream = 0);

/**
 * Convert contiguous layout to masked layout for FP8.
 */
void contiguous_to_masked_fp8(const __nv_fp8_e4m3* contiguous_data,
                              const int*           grouped_layout,
                              __nv_fp8_e4m3*       masked_data,
                              int*                 mask,
                              int                  total_tokens,
                              int                  hidden_dim,
                              int                  max_tokens,
                              int                  num_experts,
                              cudaStream_t         stream = 0);

/**
 * Convert contiguous layout to masked layout - Torch interface.
 *
 * @param contiguous_data Input tensor [total_tokens, hidden_dim]
 * @param grouped_layout Expert ID for each token [total_tokens]
 * @param num_experts Number of experts
 * @param max_tokens_per_expert Maximum tokens per expert
 * @return Tuple of (masked_data, mask)
 */
std::tuple<torch::Tensor, torch::Tensor> convert_contiguous_to_masked_torch(const torch::Tensor& contiguous_data,
                                                                            const torch::Tensor& grouped_layout,
                                                                            int                  num_experts,
                                                                            int                  max_tokens_per_expert);

/**
 * Convert masked layout back to contiguous layout - Torch interface.
 *
 * @param masked_data Input tensor [num_experts, max_tokens, hidden_dim]
 * @param grouped_layout Expert ID for each token [total_tokens]
 * @param mask Number of tokens per expert [num_experts]
 * @return Contiguous tensor [total_tokens, hidden_dim]
 */
torch::Tensor convert_masked_to_contiguous_torch(const torch::Tensor& masked_data,
                                                 const torch::Tensor& grouped_layout,
                                                 const torch::Tensor& mask);

}  // namespace rtp_llm
