#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/utils/logger.h"

#include <cassert>
#include <cstring>

namespace fastertransformer {

inline DataType QBufferDtype2BufferDtype(DataType dtype) {
    if (dtype == DataType::TYPE_QINT8) {
        return DataType::TYPE_INT8;
    } else {
        return DataType::TYPE_INVALID;
    }
}

template <typename T>
BufferPtr vector2Buffer(const std::vector<T>& vec) {
    const auto& shape = std::vector{vec.size()};
    const auto& dtype = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return std::make_shared<Buffer>(memory_type, dtype, shape, vec.data());
};

template<typename T>
inline void bufferIndexSelect(const BufferPtr& dst, const BufferPtr& src, std::vector<int>& select_index) {
    assert(src->size() == select_index.size());
    assert(src->type() == dst->type());
    T* src_data = (T*)src->data();
    T* dst_data = (T*)dst->data();
    for (size_t i = 0; i < select_index.size(); i++) {
        src_data[i] = dst_data[select_index[i]];
    }
}

template<typename T>
std::vector<T> buffer2vector(const Buffer& src, size_t num) {
    assert(num <= src.size());
    assert(sizeof(T) == src.typeSize());
    std::vector<T> dst;
    auto           size = num * sizeof(T);
    dst.resize(num);
    memcpy(dst.data(), src.data(), size);
    return dst;
}

template<typename T>
std::vector<T> buffer2vector(const Buffer& src) {
    return buffer2vector<T>(src, src.size());
}

}  // namespace fastertransformer
