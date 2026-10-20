#pragma once

#include <cstdint>

#ifdef USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace rtp_llm {

template<typename DType, typename IdType>
cudaError_t invokeRejectionSampling(DType*       draft_probs,
                                    IdType*      draft_token_ids,
                                    DType*       uniform_samples,
                                    DType*       target_probs,
                                    IdType*      target_token_ids,
                                    int          target_token_stride,
                                    IdType*      output_token_ids,
                                    IdType*      output_accepted_token_num,
                                    bool*        do_sample,
                                    int          batch_size,
                                    int          num_speculative_tokens,
                                    int          target_vocab_size,
                                    cudaStream_t stream,
                                    bool         draft_probs_point_mass = false);

cudaError_t invokeSplitRejectionSampling(const float* draft_probs,
                                         const int*   draft_ids,
                                         const float* uniforms,
                                         const float* target_probs,
                                         const int*   target_ids,
                                         int          target_stride,
                                         int*         output_ids,
                                         int*         accepted,
                                         const bool*  do_sample,
                                         int          batch,
                                         int          steps,
                                         int          vocab,
                                         bool         point_mass,
                                         float*       tile_sums,
                                         int*         rejected_positions,
                                         cudaStream_t stream);
}  // namespace rtp_llm