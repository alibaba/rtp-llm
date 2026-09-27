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

// Bias is the already-rounded GEMM output. Base may be a strided gamma slice.
template<typename BiasT>
cudaError_t invokeDSparkCombineLogits(const float* base,
                                      const BiasT* bias,
                                      const float* temperature,
                                      float*       output,
                                      int64_t      batch,
                                      int64_t      vocab,
                                      int64_t      base_row_stride,
                                      cudaStream_t stream);

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
                                    bool         deterministic_draft,
                                    int          batch_size,
                                    int          num_speculative_tokens,
                                    int          target_vocab_size,
                                    cudaStream_t stream,
                                    bool         sampled_draft = false);
}  // namespace rtp_llm
