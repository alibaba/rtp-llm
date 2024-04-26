#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include <torch/extension.h>

namespace fastertransformer {

template <typename T>
BufferPtr vector2Buffer(const std::vector<T>& vec) {
    const auto& shape = std::vector{vec.size()};
    const auto& dtype = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return std::make_unique<Buffer>(memory_type, dtype, shape, vec.data());
};

inline void bufferCopy(const BufferPtr& src, torch::Tensor& dst, size_t numberOfElements) {
    assert(dst.device().is_cpu());
    size_t copySize = src->typeSize() * numberOfElements;
    memcpy((void*)(dst.data_ptr<int>()), src->data(), copySize);
}

template<typename T>
inline void bufferIndexSelect(BufferPtr& dst, const BufferPtr& src, std::vector<size_t>& select_index) {
    assert(src->size() == select_index.size());
    assert(src->type() == dst->type());
    T* src_data = (T*)src->data();
    T* dst_data = (T*)dst->data();
    for (size_t i = 0; i < select_index.size(); i++) {
        src_data[i] = dst_data[select_index[i]];
    }
}

template<typename T>
std::vector<T> buffer2vector(const BufferPtr& src, size_t num) {
    assert(num <= src->size());
    assert(sizeof(T) == src->typeSize());
    // assert(src->where() == MemoryType::MEMORY_CPU || src->where() == MemoryType::MEMORY_CPU_PINNED);
    std::vector<T> dst;
    auto           size = num * sizeof(T);
    dst.resize(num);
    memcpy(dst.data(), src->data(), size);
    return dst;
}

template<typename T>
std::vector<T> buffer2vector(const BufferPtr& src) {
    return buffer2vector<T>(src, src->size());
}

}  // namespace fastertransformer
