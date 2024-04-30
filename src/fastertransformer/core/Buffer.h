#pragma once

#include "Types.h"
#include "src/fastertransformer/utils/assert_utils.h"

#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace fastertransformer {

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
    ~Buffer();

    static Buffer emptyBuffer();

    Buffer(const Buffer& buffer) = delete;
    Buffer(Buffer&& buffer)      = delete;
    Buffer& operator=(const Buffer& buffer) = delete;
    Buffer& operator=(Buffer&& buffer) = delete;

    bool operator==(const Buffer& other);

    MemoryType                 where() const;
    DataType                   type() const;
    const std::vector<size_t>& shape() const;
    void*                      data() const;
    void*                      dataWithOffset(size_t offset) const;
    const DeleterFuncType&     deleter() const;

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
    size_t                     typeSize() const;

    size_t size() const;
    size_t sizeBytes() const;
    size_t dim() const;

    void reshape(std::vector<size_t>& shape);
    Buffer view(size_t offset, size_t size) const; // only from 0-d
    Buffer operator[](size_t offset) const;

    std::string debugString() const;

private:
    MemoryType          where_;
    DataType            type_;
    std::vector<size_t> shape_;
    void*               data_;
    DeleterFuncType     deleter_ = nullptr;
};

using ConstBufferPtr = std::unique_ptr<const Buffer>;
using BufferPtr = std::unique_ptr<Buffer>;

} // namespace fastertransformer

