#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"

#include <cassert>
#include <cstring>


namespace fastertransformer {

inline BufferPtr convertBuffer2Ptr(const Buffer& buffer) {
    return std::make_unique<Buffer>(buffer.where(), buffer.type(), buffer.shape(), buffer.data(), buffer.deleter());
}

template <typename T>
BufferPtr vector2Buffer(const std::vector<T>& vec) {
    const auto& shape = std::vector{vec.size()};
    const auto& dtype = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return std::make_unique<Buffer>(memory_type, dtype, shape, vec.data());
};

inline void bufferCopy(const BufferPtr& src, const BufferPtr& dst, size_t numberOfElements) {
    assert(numberOfElements <= src->size());
    size_t copySize = src->typeSize() * numberOfElements;
    memcpy(dst->data(), src->data(), copySize);
}


inline void bufferConcat(const BufferPtr& src1, const BufferPtr& src2, const BufferPtr& dst) {
    assert(src1->type() == src2->type());
    assert(src1->type() == dst->type());
    bufferCopy(src1, dst, src1->size());
    auto newDst = dst->view(src1->size(), dst->size() - src1->size());
    auto dstPtr = convertBuffer2Ptr(newDst);
    bufferCopy(src2, dstPtr, src2->size());
}

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
