#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cassert>
#include <cstring>
#include <optional>

namespace rtp_llm {

template<typename T>
BufferPtr vector2Buffer(const std::vector<T>& vec) {
    const auto& shape       = std::vector{vec.size()};
    const auto& dtype       = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return std::make_shared<Buffer>(memory_type, dtype, shape, vec.data());
};

template<typename T>
std::vector<T> buffer2vector(const Buffer& src, size_t size) {
    RTP_LLM_CHECK_WITH_INFO((src.size() >= size), "buffer size [%d] is less than size [%d]", src.size(), size);
    RTP_LLM_CHECK_WITH_INFO(
        (src.typeSize() == sizeof(T)), "Buffer type size %d is not equal to size of T: %d", src.typeSize(), sizeof(T));
    std::vector<T> dst;
    auto           total_size = size * sizeof(T);
    dst.resize(size);
    memcpy(dst.data(), src.data(), total_size);
    return dst;
}

template<typename T>
std::vector<T> buffer2vector(const Buffer& src) {
    return buffer2vector<T>(src, src.size());
}

inline void BUFFER_DTYPE_CHECK(const Buffer& buffer, std::vector<DataType> dtypes) {
    RTP_LLM_CHECK_WITH_INFO((std::find(dtypes.begin(), dtypes.end(), buffer.type()) != dtypes.end()),
                            "buffer type[%d] is invalid",
                            buffer.type());
}

#define BUFFER_GET_SCALE_IF_Q_BUFFER(buf)                                                                              \
    ((buf)->isQBuffer() ? dynamic_cast<const QBuffer*>(buf.get())->scalesData() : nullptr)
#define BUFFER_GET_ZERO_IF_Q_BUFFER(buf)                                                                               \
    ((buf)->isQBuffer() ? dynamic_cast<const QBuffer*>(buf.get())->zerosData() : nullptr)
#define OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(buf) ((buf) ? buf->data() : nullptr)

#define WEIGHT_MAY_GET_BIAS(weights) (weights) ? weights->bias : nullptr

#define GET_TYPED_VALUE_FROM_OPT_REF(optional_buf_ref, type)                                                           \
    optional_buf_ref.has_value() ? optional_buf_ref.value().get().data<type>() : nullptr

template<typename T>
inline std::optional<std::reference_wrapper<T>> mayGetRef(const std::shared_ptr<T>& ptr) {
    return ptr ? std::optional<std::reference_wrapper<T>>(*ptr) : std::nullopt;
}

}  // namespace rtp_llm
