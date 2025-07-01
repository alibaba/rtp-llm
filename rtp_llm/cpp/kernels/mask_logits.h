#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invokeMaskLogits(T* logits_batch, const uint8_t* mask_batch, const int batch_size, const int vocab_size, cudaStream_t stream);

}  // namespace rtp_llm
