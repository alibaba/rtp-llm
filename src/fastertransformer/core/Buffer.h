#pragma once

#include "Types.h"
#include "src/fastertransformer/utils/assert_utils.h"

#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace fastertransformer {

class QBuffer;

enum class BufferLifecycleType {
    SHORT,
    LONG
};

enum class SpaceComplexityType {
    UNKNOWN,
    CONSTANT,
    LINEAR,
    QUADRATIC
};

struct BufferHints {
    BufferHints(const std::string& tag = "",
                BufferLifecycleType lifecycle = BufferLifecycleType::SHORT,
                SpaceComplexityType space_complexity = SpaceComplexityType::UNKNOWN)
    : tag(tag), lifecycle(lifecycle), space_complexity(space_complexity) {}

    std::string tag;
    BufferLifecycleType lifecycle;
    SpaceComplexityType space_complexity;
};

// Buffer is similar to Tensor, but with more limited functionality.
// It only includes a pointer with metadata.
class Buffer {
public:
    typedef std::function<void(Buffer *)> DeleterFuncType;
    Buffer(const MemoryType where,
           const DataType type,
           const std::vector<size_t>& shape,
           const void* data,
           const DeleterFuncType deleter = nullptr);
    virtual ~Buffer();

    static Buffer emptyBuffer();

    Buffer(const Buffer& buffer)            = delete;
    Buffer(Buffer&& buffer)                 = delete;
    Buffer& operator=(const Buffer& buffer) = delete;
    Buffer& operator=(Buffer&& buffer)      = delete;

    bool operator==(const Buffer& other);

    MemoryType                 where() const;
    DataType                   type() const;
    const std::vector<size_t>& shape() const;
    void*                      data() const;
    void*                      dataWithOffset(size_t offset) const;

    template<typename T>
    inline T* data() const {
        static_assert(!std::is_same<T, void>::value);
        FT_CHECK_WITH_INFO(
            type_ == getTensorType<T>(),
            "get data type %d not match buffer type %d", getTensorType<T>(), type_);
        return static_cast<T*>(data_);
    }

    template<typename T>
    inline T* dataWithOffset(size_t offset) const {
        return data<T>() + offset;
    }

    size_t typeSize() const;
    size_t size() const;
    size_t sizeBytes() const;
    size_t dim() const;

    bool isFloat() const {
        return  (type_ == DataType::TYPE_BF16) ||
                (type_ == DataType::TYPE_FP16) ||
                (type_ == DataType::TYPE_FP32) ||
                (type_ == DataType::TYPE_FP64);
    }

    bool isQuantify() const {
        return (type_ == DataType::TYPE_QINT8);
    }

    void updateShape(const std::vector<size_t>& shape);
    Buffer reshape(const std::vector<size_t>& shape) const;
    Buffer view(size_t offset, size_t size) const; // only from 0-d
    Buffer operator[](size_t offset) const;

    std::string debugString() const {
        return debugStringMeta();
    }

    template<typename T>
    std::string debugStringWithData() const {
        return debugStringMeta() + ", " + debugDataString<T>(size());
    }

    template<typename T>
    std::string debugStringWithData(size_t count) const {
        return debugStringMeta() + ", " + debugDataString<T>(count);
    }

    template<typename T>
    std::string debugDataString(size_t count) const {
        auto base = data<T>();
        auto total_size = size();
        std::ostringstream oss;
        auto data_size = std::min(count, total_size);
        for (size_t i = 0; i < data_size; i++) {
            oss << base[i] << ", ";
        }
        if (data_size != total_size) {
            oss << "...... ";
            for (size_t i = total_size - data_size; i < total_size; i++) {
                oss << base[i] << ", ";
            }
        }
        return "BufferData Detail(" + oss.str() + ")";
    }

    std::string debugStringMeta() const;

private:
    DeleterFuncType getSubBufferDeleter() const ;

protected:
    MemoryType          where_;
    DataType            type_;
    std::vector<size_t> shape_;
    void*               data_;
    DeleterFuncType     deleter_ = nullptr;
    mutable size_t      view_count_ = 0;

friend class QBuffer;

};

using ConstBufferPtr = std::shared_ptr<const Buffer>;
using BufferPtr = std::shared_ptr<Buffer>;

} // namespace fastertransformer

