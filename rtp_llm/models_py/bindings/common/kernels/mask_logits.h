#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invokeMaskLogits(T*             logits_batch,
                      const uint8_t* mask_batch,
                      const int      batch_size,
                      const int      vocab_size,
#if USING_CUDA
                      cudaStream_t stream);
#elif USING_ROCM
                      hipStream_t stream);
#endif

// Applies compact packed allow-masks to selected logits rows. Each int32 mask
// word encodes 32 tokens (bit=1 allowed); row_indices maps compact mask rows to
// rows in logits_batch. A null row_indices pointer means identity mapping.
#if USING_CUDA
template<typename T>
void invokePackedMaskLogits(T*             logits_batch,
                            const int32_t* packed_allow_mask,
                            const int32_t* row_indices,
                            const int      mask_rows,
                            const int      logits_rows,
                            const int      logits_row_stride,
                            const int      vocab_size,
                            const int      bitmask_row_stride,
                            const int      bitmask_words,
                            cudaStream_t   stream);

// Applies a single batched grammar mask/tail/force-token step across active rows.
// packed_buf layout (int32):
//   [0 .. active_rows-1]:                 row_indices
//   [active_rows .. 2*active_rows-1]:     row_modes (>=0 force token id; -1 tail-only; <=-2 bitmask row -2-mode)
//   [2*active_rows .. 3*active_rows-1]:   grammar_vocab_limits
//   [bitmask_header_words .. ...]:        packed bitmasks [num_bitmask_rows, bitmask_row_stride]
template<typename T>
void invokeBatchGrammarMaskLogits(T*             logits_batch,
                                  const int32_t* packed_buf,
                                  const int      bitmask_header_words,
                                  const int      active_rows,
                                  const int      logits_rows,
                                  const int      logits_row_stride,
                                  const int      vocab_size,
                                  const int      bitmask_row_stride,
                                  cudaStream_t   stream);
#endif

}  // namespace rtp_llm
