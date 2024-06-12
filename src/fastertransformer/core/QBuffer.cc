#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/core/BufferHelper.h"

namespace fastertransformer {

QBuffer::QBuffer(BufferPtr kernel,
                 BufferPtr scales,
                 BufferPtr zeros) :
                 Buffer(kernel->where(),
                        kernel->type(),
                        kernel->shape(),
                        kernel->data(),
                        std::move(kernel->deleter_))
{
    FT_CHECK_WITH_INFO(
       (kernel.use_count() == 1 &&
        scales.use_count() == 1 && 
        zeros.use_count() == 1),
        "kerenl[%d], scales[%d] and zeros[%d] buffers need to have no ref cout.",
        kernel.use_count(),
        scales.use_count(),
        zeros.use_count()
    );
    // std::static_pointer_cast<QBuffer>(kernel)->deleter_ = nullptr;
    scales_.swap(scales);
    zeros_.swap(zeros);
    kernel.reset();
    scales.reset();
    zeros.reset();

    FT_CHECK_WITH_INFO(
        (type_ == DataType::TYPE_INT8 ||
         type_ == DataType::TYPE_QINT8),
        "kerenl buffer datatype[%d] must be int8.", type_
    );
    if(type_ == DataType::TYPE_INT8) {
        type_ = DataType::TYPE_QINT8;
    }

    FT_CHECK_WITH_INFO(
        (scales_->dim() == 1 && zeros_->dim() == 1),
        "dim of scales[%d] and zero_points[%d] must be 1.",
        scales_->dim(), zeros_->dim()
    );

    FT_CHECK_WITH_INFO(
        (scales_->size() == zeros_->size()),
        "number of elements in scales[] and zero_points[] must match.",
        scales_->size(), zeros_->size()
    );

    FT_CHECK_WITH_INFO(
        (scales_->where() == zeros_->where() &&
         scales_->where() == where()),
        "scales[%d] and zeros[%d] must in same memory.",
        scales_->where(), zeros_->where()
    );
                    
};


// NOTE: new Buffer from kernel()/scales()/zeros() has the same data pointer as the original buffer,
//       and also shorter life time than the original buffer.
//       user has the responsibility to keep the original buffer alive.
Buffer QBuffer::scales() const {
    return Buffer(scales_->where(),
                  scales_->type(),
                  scales_->shape(),
                  scales_->data(),
                  nullptr);
}

Buffer QBuffer::zeros() const {
    return Buffer(zeros_->where(),
                  zeros_->type(),
                  zeros_->shape(),
                  zeros_->data(),
                  nullptr);
}

Buffer QBuffer::kernel() const {
    return Buffer(where(),
                  QBufferDtype2BufferDtype(type()),
                  shape(),
                  data(),
                  nullptr);
}

void* QBuffer::scales_data() const {
    return scales_->data();
}

void* QBuffer::zeros_data() const {
    return zeros_->data();
}

DataType QBuffer::scales_type() const {
    return scales_->type();
}

DataType QBuffer::zeros_type() const {
    return zeros_->type();
}

size_t QBuffer::scales_size_bytes() const {
    return scales_->sizeBytes();
}

size_t QBuffer::zeros_size_bytes() const {
    return zeros_->sizeBytes();
}

} // namespace fastertransformer

