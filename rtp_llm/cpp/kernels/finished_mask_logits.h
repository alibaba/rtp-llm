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
void invokeFinishedMaskLogits(T*             logits_batch,
                              const uint8_t* finished_mask,
                              int            batch_size,
                              int            vocab_size,
                              int            end_token_id,
#if USING_CUDA
                              cudaStream_t stream);
#elif USING_ROCM
                              hipStream_t stream);
#endif

}  // namespace rtp_llm
