#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4TopK8Bf16(const __nv_bfloat16* input,
                           __nv_bfloat16*       output_values,
                           int64_t*             output_indices,
                           int64_t              tokens,
                           cudaStream_t         stream);

void invokeGemma4WeightedReorderBf16(const __nv_bfloat16* expert_output,
                                     const __nv_bfloat16* sorted_weight,
                                     const int64_t*       inverse_permutation,
                                     __nv_bfloat16*       output,
                                     int64_t              rows,
                                     int32_t              hidden_size,
                                     cudaStream_t         stream);

void invokeGemma4GatherSortedExpertInputBf16(const __nv_bfloat16* input,
                                             const int64_t*       permutation,
                                             __nv_bfloat16*       output,
                                             int64_t              rows,
                                             int32_t              top_k,
                                             int32_t              hidden_size,
                                             cudaStream_t         stream);

void invokeGemma4PrepareGroupedMoe(const int64_t*       expert_ids,
                                   const __nv_bfloat16* weights,
                                   int64_t*             permutation,
                                   int64_t*             inverse_permutation,
                                   __nv_bfloat16*       sorted_weights,
                                   int32_t*             offsets,
                                   int32_t*             scratch,
                                   int64_t              rows,
                                   int32_t              num_experts,
                                   cudaStream_t         stream);

void invokeGemma4Top8SumBf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t tokens, int32_t hidden_size, cudaStream_t stream);

void invokeGemma4FinalizeRouterWeightsBf16(const __nv_bfloat16* top_weights,
                                           const int64_t*       top_indices,
                                           const __nv_bfloat16* expert_scales,
                                           __nv_bfloat16*       output,
                                           int64_t              tokens,
                                           cudaStream_t         stream);

}  // namespace rtp_llm
