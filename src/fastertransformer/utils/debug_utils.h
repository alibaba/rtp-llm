#pragma once

#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/string_utils.h"

namespace fastertransformer {

template<typename T>
std::string valueToString(const T* value, const size_t idx = 0);

template<typename T>
std::string printCudaMemoryToString(const void* ptr, int size = 10);

template<typename T>
std::string printCudaMemoryStrided(const void* ptr, int stride_length, int stride_num = 1, int size_per_stride = 10);

}  // namespace fastertransformer
