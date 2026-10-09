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

#if USING_CUDA
size_t dsparkSoftmaxWorkspaceBytes(const float* logits, const float* output, int64_t rows, int64_t vocab,
                                  int major, int minor);
cudaError_t invokeDSparkSoftmax(float* logits, float* output, int64_t rows, int64_t vocab,
                               int major, int minor, int sm_count, void* workspace, size_t workspace_bytes,
                               cudaStream_t stream);
#endif

constexpr int kRejectionValidationTileSize = 8192;
// Two probability planes, each with one partial mass and invalid flag.
inline int64_t rejectionValidationWorkspaceElements(int batch, int steps, int vocab) {
    return static_cast<int64_t>(batch) * (steps + 1)
           * ((static_cast<int64_t>(vocab) + kRejectionValidationTileSize - 1) / kRejectionValidationTileSize) * 4;
}

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

// Compute sigmoid(float(BF16(W_h * hidden + W_m * markov(prev_token) + bias)))
// without materializing the [hidden, markov] feature concatenation. The raw
// BF16 projection boundary matches the confidence head's BF16 Linear output.
cudaError_t invokeDSparkConfidence(const __nv_bfloat16* hidden,
                                   const int32_t*       anchors,
                                   const int32_t*       sampled_tokens,
                                   const __nv_bfloat16* markov_w1,
                                   const __nv_bfloat16* confidence_w,
                                   const __nv_bfloat16* confidence_b,
                                   float*               output,
                                   int64_t              batch,
                                   int64_t              gamma,
                                   int64_t              hidden_dim,
                                   int64_t              markov_rank,
                                   cudaStream_t         stream);

// Select a batch-wide number of extra target-verify rows from conditional
// confidence.  verify_lengths contains target rows per request, including the
// mandatory anchor row.  compact_to_dense maps the packed rows back into the
// logical [batch, gamma + 1] layout used by rejection sampling and commit.
cudaError_t invokeDSparkVerifyPlan(const float* confidence,
                                   int32_t*     verify_lengths,
                                   int32_t*     compact_to_dense,
                                   int64_t      batch,
                                   int64_t      gamma,
                                   int64_t      extra_budget,
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
                                    bool         sampled_draft         = false,
                                    bool*        success               = nullptr,
                                    const bool*  target_success        = nullptr,
                                    const int*   active_verify_lengths = nullptr,
                                    float*       validation_workspace  = nullptr);
}  // namespace rtp_llm
