#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include <torch/extension.h>

namespace fastertransformer {

template <typename T>
Buffer vector2Buffer(const std::vector<T>& vec) {
    const auto& shape = {vec.size()};
    const auto& dtype = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return Buffer(memory_type, dtype, shape, vec.data());
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

inline void bufferSliceCopy(BufferPtr& dst, const BufferPtr& src, int dim, int from, int to) {
    assert(dst->type() == src->type());
    auto width = src->typeSize();
    assert(dst->dim() == 2 && src->dim() == 2);
    // assert(dst->where() == MemoryType::MEMORY_CPU && src->where() == MemoryType::MEMORY_CPU);
    if (dim == 0) {
        assert(dst->shape()[1] == src->shape()[1]);
        assert(dst->shape()[0] >= src->shape()[0]);
        memcpy(dst->data(), src->data(), src->sizeBytes());
    } else if (dim == 1) {
        size_t pre_dims = 1;
        for (int i = 0; i < dim; ++i) {
            pre_dims *= dst->shape()[i];
        }
        size_t post_dims = src->size() / src->shape()[0];

        auto cp_size = src->sizeBytes() / src->shape()[0];
        assert(dst->size() / dst->shape()[dim] / pre_dims == post_dims);
        assert(dst->shape()[dim] == src->shape()[0]);
        for (int i = from; i < to; ++i) {
            memcpy((char*)dst->data() + i * dst->sizeBytes() / dst->shape()[0] + from * width,
                   (char*)src->data() + i * cp_size,
                   cp_size);
        }
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
