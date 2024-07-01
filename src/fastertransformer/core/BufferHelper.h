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
std::vector<T> buffer2vector(const Buffer& src, size_t num) {
    FT_CHECK_WITH_INFO((num <= src.size()),
        "buffer num[%d] is less than num", src.size(), num);
    FT_CHECK_WITH_INFO((sizeof(T) == src.typeSize()),
        "Buffer type size %d is not equal to %d", src.typeSize(), sizeof(T));
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

inline void BUFFER_DTYPE_CHECK(const Buffer& buffer, std::vector<DataType> dtypes) {
    FT_CHECK_WITH_INFO(
        (std::find(dtypes.begin(), dtypes.end(), buffer.type()) != dtypes.end()),
        "buffer type[%d] is invalid", buffer.type());
}

#define BUFFER_GET_SCALE_IF_Q_BUFFER(buf) \
    ((buf)->isQBuffer() ? dynamic_cast<const QBuffer*>(buf.get())->scalesData() : nullptr)
#define OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(buf) \
    ((buf) ? buf->data() : nullptr)

#define WEIGHT_MAY_GET_BIAS(weights) \
    (weights) ? weights->bias : nullptr

template <typename T>
inline std::optional<std::reference_wrapper<T>> mayGetRef(const std::shared_ptr<T>& ptr) {
    return ptr ? std::optional<std::reference_wrapper<T>>(*ptr) : std::nullopt;
}

}  // namespace fastertransformer
