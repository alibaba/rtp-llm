#pragma once

#include "Buffer.h"

namespace fastertransformer {

class QBuffer final : public Buffer {

public:
    
    QBuffer(BufferPtr kernel,
            BufferPtr scales,
            BufferPtr zeros);

    ~QBuffer() = default;

    QBuffer(const Buffer& buffer)               = delete;
    QBuffer(Buffer&& buffer)                    = delete;
    QBuffer& operator=(const QBuffer& buffer)   = delete;
    QBuffer& operator=(QBuffer&& buffer)        = delete;

    Buffer          scales()                const;
    Buffer          zeros()                 const;
    Buffer          kernel()                const;
    void*           scales_data()           const;
    void*           zeros_data()            const;
    DataType        scales_type()           const;
    DataType        zeros_type()            const;
    size_t          scales_sizeBytes()      const;
    size_t          zeros_sizeBytes()       const;

private:
    BufferPtr                      scales_;
    BufferPtr                      zeros_;

};

using ConstQBufferPtr = std::shared_ptr<const QBuffer>;
using QBufferPtr = std::shared_ptr<QBuffer>;




} // namespace fastertransformer