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
                                    bool         draft_probs_point_mass  = false,
                                    bool         target_probs_point_mass = false,
                                    // Optional post-rejection finalization folded into the kernel:
                                    // force_mask (nullable, [batch_size] device bool) overrides
                                    // forced rows with all draft tokens + the target bonus token
                                    // and a full accept count; finalize_outputs additionally resets
                                    // every -1 padding slot to 0. With finalize_outputs the kernel
                                    // writes every element of both output buffers on every path, so
                                    // callers may pass uninitialized outputs. Defaults keep the
                                    // legacy kernel contract byte-for-byte.
                                    const bool* force_mask       = nullptr,
                                    bool        finalize_outputs = false,
                                    // Optional [batch_size] CUDA int32 lengths. Physical strides
                                    // stay num_speculative_tokens / num_speculative_tokens+1.
                                    // A row verifies only its prefix L and uses target row L for
                                    // the bonus. Values are clamped to [0, num_speculative_tokens]
                                    // on device. nullptr preserves the full-width behavior.
                                    const int32_t* valid_draft_lengths = nullptr);
}  // namespace rtp_llm
