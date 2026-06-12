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
                            cudaStream_t stream);
#endif

}  // namespace rtp_llm
