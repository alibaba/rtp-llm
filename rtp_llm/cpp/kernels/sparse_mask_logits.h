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
void invokeSparseMaskLogits(T*         logits_batch,
                            const int* batch_idx,
                            const int* mask_idx,
                            const int  batch_size,
                            const int  vocab_size,
                            const int  mask_size,
#if USING_CUDA
                            cudaStream_t stream);
#elif USING_ROCM
                            hipStream_t stream);
#endif

}  // namespace rtp_llm
