#pragma once

#if USEING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "src/fastertransformer/rocm/cuda_shims.h"
#endif

namespace fastertransformer {

template<typename T>
void invokeHelloWorld(const T* a, const T* b, T* c, const int vector_len, cudaStream_t stream);

}  // namespace fastertransformer
