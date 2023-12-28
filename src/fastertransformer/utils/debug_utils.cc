#include "src/fastertransformer/utils/debug_utils.h"

namespace fastertransformer {

template <typename T>
std::string valueToString(const T* value, const size_t idx) {
    return std::to_string(value[idx]);
}

template <>
std::string valueToString<__half>(const __half* value, const size_t idx) {
    return std::to_string((float)value[idx]);
}

template <typename T>
std::string printCudaMemoryToString(const void* ptr, int size) {
    const auto buffer_size = size * sizeof(T);
    auto* cpu_mem = reinterpret_cast<T*>(malloc(buffer_size));
    cudaDeviceSynchronize();
    check_cuda_error(cudaMemcpy(cpu_mem, ptr, buffer_size, cudaMemcpyDeviceToHost));
    std::string result = "[";
    for (int i = 0; i < size; ++i) {
        result += valueToString(cpu_mem, i);
        if (i != size - 1) {
            result += ", ";
        }
    }
    result += "]";
    free(cpu_mem);
    return result;
}

template <typename T>
std::string printCudaMemoryStrided(const void* ptr, int stride_length, int stride_num, int size_per_stride) {
    std::string result = "";
    for (int i = 0; i < stride_num; ++i) {
        result += printCudaMemoryToString<T>(reinterpret_cast<const T*>(ptr) + i * stride_length, size_per_stride);
        if (i != stride_num - 1) {
            result += "\n, ";
        }
    }
    return result;
}

template std::string printCudaMemoryToString<__half>(const void* ptr, int size);
template std::string printCudaMemoryToString<int>(const void* ptr, int size);
template std::string printCudaMemoryStrided<__half>(const void* ptr, int stride_length, int stride_num, int size_per_stride);

} // namespace fastertransformer
