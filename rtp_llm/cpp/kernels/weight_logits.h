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
void invokeWeightLogits(T*         logits_batch,
                        const int* batch_idx,
                        const int* vocab_idx,
                        float*     weight_batch,
                        const int  batch_size,
                        const int  vocab_size,
                        const int  weight_size,
#if USING_CUDA
                        cudaStream_t stream);
#elif USING_ROCM
                        hipStream_t stream);
#endif

}  // namespace rtp_llm
