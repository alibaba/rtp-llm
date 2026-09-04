#pragma once

#include <stdint.h>

#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

// Applies a CSR candidate row directly to each logits row. col_idx must be
// sorted inside every row. Invalid and terminal states mask the whole row so a
// request can never escape the constraint and resume free generation.
template<typename T>
void invokeCSRMaskLogits(T*         logits,
                         const int* states,
                         const int* row_ptr,
                         const int* col_idx,
                         int        batch_size,
                         int        vocab_size,
                         int        state_count,
#if USING_CUDA
                         cudaStream_t stream);
#elif USING_ROCM
                         hipStream_t stream);
#endif

}  // namespace rtp_llm
