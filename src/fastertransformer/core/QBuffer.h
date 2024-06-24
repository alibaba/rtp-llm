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
    void*           scalesData()           const;
    void*           zerosData()            const;
    DataType        scalesType()           const;
    DataType        zerosType()            const;
    size_t          scalesSizebytes()      const;
    size_t           zerosSizebytes()       const;

    template<typename T>
    inline T* scalesData() const {
        return scales_->data<T>();
    }

    template<typename T>
    inline T* zerosData() const {
        return zeros_->data<T>();
    }

private:
    BufferPtr                      scales_;
    BufferPtr                      zeros_;

};

using ConstQBufferPtr = std::shared_ptr<const QBuffer>;
using QBufferPtr = std::shared_ptr<QBuffer>;




} // namespace fastertransformer