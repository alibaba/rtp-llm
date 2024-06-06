#pragma once

#include "Buffer.h"

namespace fastertransformer {

class QBuffer final : public Buffer {

public:
    
    QBuffer(const MemoryType where,
            const DataType type,
            const std::vector<size_t>& shape,
            void* data,
            Buffer scales,
            Buffer zeros,
            const std::function<void(Buffer *)> deleter = nullptr);
    
    QBuffer(Buffer kernel,
            Buffer scales,
            Buffer zeros);

    ~QBuffer() = default;

    QBuffer(const Buffer& buffer)               = delete;
    QBuffer(Buffer&& buffer)                    = delete;
    QBuffer& operator=(const QBuffer& buffer)   = delete;
    QBuffer& operator=(QBuffer&& buffer)        = delete;

    BufferPtr       scales()                const;
    BufferPtr       zeros()                 const;
    BufferPtr       kernel()                const;
    void*           scales_data()           const;
    void*           zeros_data()            const;
    DataType        scales_type()           const;
    DataType        zeros_type()            const;
    size_t          scales_sizeBytes()      const;
    size_t          zeros_sizeBytes()       const;

private:
    Buffer                      scales_;
    Buffer                      zeros_;
};

using ConstQBufferPtr = std::shared_ptr<const QBuffer>;
using QBufferPtr = std::shared_ptr<QBuffer>;




} // namespace fastertransformer