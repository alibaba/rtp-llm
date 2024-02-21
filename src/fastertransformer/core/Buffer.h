#pragma once

#include "Types.h"

#include <vector>
#include <string>

namespace fastertransformer {

// Buffer is similar to Tensor, but with more limited functionality.
// It only includes a pointer with metadata.
class Buffer {
public:
    Buffer(const MemoryType where,
           const DataType type,
           const std::vector<size_t>& shape,
           const void* data);
    ~Buffer();

    Buffer(const Buffer& tensor) = delete;
    Buffer(Buffer&& tensor)      = delete;
    Buffer& operator=(const Buffer& tensor) = delete;
    Buffer& operator=(Buffer&& tensor) = delete;

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
    void*               data_   = nullptr;
};

} // namespace fastertransformer

