#pragma once

#include "Buffer.h"

namespace fastertransformer {

enum QScheme : size_t {
    NoQuantize = 0,
    Qint8PerToken,
    Qint8PerTensor,
    Qfp8PerTensor
};

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
    size_t          zerosSizebytes()       const;

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

inline DataType QBufferDtype2BufferDtype(DataType dtype) {
    if (dtype == DataType::TYPE_QINT8 || dtype == DataType::TYPE_INT8) {
        return DataType::TYPE_INT8;
    } else if (dtype == TYPE_QINT4X2 || dtype == DataType::TYPE_INT4X2) {
        return DataType::TYPE_INT4X2;
    } else if (dtype == DataType::TYPE_FP8_E4M3 || dtype == DataType::TYPE_QFP8_E4M3) {
        return DataType::TYPE_FP8_E4M3;
    } else {
        return DataType::TYPE_INVALID;
    }
}

inline DataType BufferDtype2QBufferDtype(DataType dtype) {
    if (dtype == DataType::TYPE_QINT8 || dtype == DataType::TYPE_INT8) {
        return DataType::TYPE_QINT8;
    } else if (dtype == TYPE_QINT4X2 || dtype == DataType::TYPE_INT4X2) {
        return DataType::TYPE_QINT4X2;
    } else if (dtype == DataType::TYPE_QFP8_E4M3 || dtype == DataType::TYPE_FP8_E4M3) {
        return DataType::TYPE_QFP8_E4M3;
    } else {
        return DataType::TYPE_INVALID;
    }
}


} // namespace fastertransformer
