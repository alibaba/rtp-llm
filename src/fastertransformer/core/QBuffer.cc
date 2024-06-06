#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/core/BufferHelper.h"

namespace fastertransformer {

QBuffer::QBuffer(const MemoryType where,
                 const DataType type,
                 const std::vector<size_t>& shape,
                 void* data,
                 Buffer scales,
                 Buffer zeros,
                 const std::function<void(Buffer *)> deleter) :
                 Buffer(where, type, shape, data, deleter),
                 scales_(std::move(scales)),
                 zeros_(std::move(zeros))
{
    FT_CHECK_WITH_INFO(
        (type == DataType::TYPE_QINT8),
        "type must be qtype"
    );

    FT_CHECK_WITH_INFO(
        (scales_.dim() == 1 && zeros_.dim() == 1),
        "dim of scales and zero_points must be 1"
    );
    FT_CHECK_WITH_INFO(
        (scales_.size() == zeros_.size()),
        "number of elements in scales and zero_points must match"
    );
                    
};

QBuffer::QBuffer(Buffer kernel,
                 Buffer scales,
                 Buffer zeros) :
                 Buffer(std::move(kernel)),
                 scales_(std::move(scales)),
                 zeros_(std::move(zeros))
{
    FT_CHECK_WITH_INFO(
        (type_ == DataType::TYPE_INT8 ||
         type_ == DataType::TYPE_QINT8),
        "kerenl buffer datatype[%d] must be int8.", type_
    );
    if(type_ == DataType::TYPE_INT8) {
        type_ = DataType::TYPE_QINT8;
    }

    FT_CHECK_WITH_INFO(
        (scales_.dim() == 1 && zeros_.dim() == 1),
        "dim of scales and zero_points must be 1."
    );

    FT_CHECK_WITH_INFO(
        (scales_.size() == zeros_.size()),
        "number of elements in scales and zero_points must match."
    );

    FT_CHECK_WITH_INFO(
        (scales_.where() == zeros_.where() &&
         scales_.where() == where()),
        "scales and zeros must in same memory."
    );
                    
};


// NOTE: new Buffer from kernel()/scales()/zeros() has the same data pointer as the original buffer,
//       and also shorter life time than the original buffer.
//       user has the responsibility to keep the original buffer alive.
BufferPtr QBuffer::scales() const {
    return convertBuffer2Ptr(Buffer(scales_.where(),
                                    scales_.type(),
                                    scales_.shape(),
                                    scales_.data(),
                                    nullptr));
}

BufferPtr QBuffer::zeros() const {
    return convertBuffer2Ptr(Buffer(zeros_.where(),
                                    zeros_.type(),
                                    zeros_.shape(),
                                    zeros_.data(),
                                    nullptr));
}

BufferPtr QBuffer::kernel() const {
    return convertBuffer2Ptr(Buffer(where_,
                                    QBufferDtype2BufferDtype(type_),
                                    shape_,
                                    data_,
                                    nullptr));
}

void* QBuffer::scales_data() const {
    return scales_.data();
}

void* QBuffer::zeros_data() const {
    return zeros_.data();
}

DataType QBuffer::scales_type() const {
    return scales_.type();
}

DataType QBuffer::zeros_type() const {
    return zeros_.type();
}

size_t QBuffer::scales_sizeBytes() const {
    return scales_.sizeBytes();
}

size_t QBuffer::zeros_sizeBytes() const {
    return zeros_.sizeBytes();
}

} // namespace fastertransformer

