#pragma once

#include "Types.h"

#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace fastertransformer {

// Buffer is similar to Tensor, but with more limited functionality.
// It only includes a pointer with metadata.
class Buffer {
public:
    Buffer(const MemoryType where,
           const DataType type,
           const std::vector<size_t>& shape,
           const void* data,
           const std::function<void(Buffer *)> deleter = nullptr);
    ~Buffer();

    Buffer(const Buffer& buffer) = delete;
    Buffer(Buffer&& buffer)      = delete;
    Buffer& operator=(const Buffer& buffer) = delete;
    Buffer& operator=(Buffer&& buffer) = delete;

    MemoryType                 where() const;
    DataType                   type() const;
    const std::vector<size_t>& shape() const;
    void*                      data() const;

    size_t size() const;
    size_t sizeBytes() const;

    std::string debugString() const;

private:
    MemoryType          where_;
    DataType            type_;
    std::vector<size_t> shape_;
    void*               data_;
    std::function<void(Buffer *)> deleter_;
};

using ConstBufferPtr = std::unique_ptr<const Buffer>;
using BufferPtr = std::unique_ptr<Buffer>;

} // namespace fastertransformer

