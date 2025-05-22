#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invokeMaskLogits(T* logits, const uint8_t* mask, const int vector_len, cudaStream_t stream);

}  // namespace rtp_llm
