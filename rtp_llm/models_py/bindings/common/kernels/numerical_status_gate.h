#pragma once

#include <stdint.h>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invokeNumericalStatusGate(T* logits,
                               const int32_t* status,
                               const int32_t* row_to_status,
                               uint8_t* failure_mask,
                               int rows,
                               int status_rows,
                               int vocab_size,
                               int status_scope,
#if USING_CUDA
                               cudaStream_t stream);
#elif USING_ROCM
                               hipStream_t stream);
#endif

}  // namespace rtp_llm
