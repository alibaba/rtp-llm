#include "rtp_llm/cpp/core/QBuffer.h"
#include "Buffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include <memory>

namespace rtp_llm {

QBuffer::QBuffer(BufferPtr kernel, BufferPtr scales, BufferPtr zeros):
    Buffer(kernel->where(), kernel->type(), kernel->shape(), kernel->data(), std::move(kernel->deleter_)) {
    RTP_LLM_CHECK_WITH_INFO((kernel.use_count() == 1 && scales.use_count() == 1 && zeros.use_count() == 1),
                            "kerenl[%d], scales[%d] and zeros[%d] buffers need to have no ref cout.",
                            kernel.use_count(),
                            scales.use_count(),
                            zeros.use_count());

    scales_.swap(scales);
    zeros_.swap(zeros);
    kernel.reset();
    scales.reset();
    zeros.reset();

    type_ = BufferDtype2QBufferDtype(type_);
    RTP_LLM_CHECK_WITH_INFO(
        (type_ != DataType::TYPE_INVALID), "kerenl buffer datatype[%d] must be int8 or int4x2.", type_);

    RTP_LLM_CHECK_WITH_INFO(((scales_->dim() == 1 || scales_->dim() == 2 || scales_->dim() == 3 || scales_->dim() == 4)
                             && (zeros_->dim() == 1 || zeros_->dim() == 2 || zeros_->dim() == 3 || zeros_->dim() == 4)),
                            "dim of scales[%d] and zero_points[%d] must be 1, 2, 3, 4.",
                            scales_->dim(),
                            zeros_->dim());

    RTP_LLM_CHECK_WITH_INFO((scales_->where() == zeros_->where() && scales_->where() == where()),
                            "scales[%d] and zeros[%d] must in same memory.",
                            scales_->where(),
                            zeros_->where());
};

// NOTE: new Buffer from kernel()/scales()/zeros() has the same data pointer as the original buffer,
//       and also shorter life time than the original buffer.
//       user has the responsibility to keep the original buffer alive.
Buffer QBuffer::scales() const {
    return Buffer(scales_->where(), scales_->type(), scales_->shape(), scales_->data(), nullptr);
}

BufferPtr QBuffer::scalesPtr() const {
    return std::make_shared<Buffer>(scales_->where(), scales_->type(), scales_->shape(), scales_->data(), nullptr);
}

Buffer QBuffer::zeros() const {
    return Buffer(zeros_->where(), zeros_->type(), zeros_->shape(), zeros_->data(), nullptr);
}

Buffer QBuffer::kernel() const {
    return Buffer(where(), QBufferDtype2BufferDtype(type()), shape(), data(), nullptr);
}

BufferPtr QBuffer::kernelPtr() const {
    return std::make_shared<Buffer>(where(), QBufferDtype2BufferDtype(type()), shape(), data(), nullptr);
}

std::shared_ptr<QBuffer> QBuffer::qslice(size_t offset, size_t size) const {
    BufferPtr new_kernel = slice(offset, size);
    BufferPtr new_scales = scales_ ? scales_->slice(offset, size) : nullptr;
    BufferPtr new_zeros  = nullptr;
    if (zeros_ && zeros_->shape()[0] > (offset + size)) {
        new_zeros = zeros_->slice(offset, size);
    } else {
        new_zeros = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr));
    }
    return std::make_shared<QBuffer>(std::move(new_kernel), std::move(new_scales), std::move(new_zeros));
}

std::shared_ptr<QBuffer> QBuffer::qslicePerTensor(size_t offset, size_t size) const {
    BufferPtr new_kernel = slice(offset, size);
    BufferPtr new_scales = scales_ ? scales_->slice(0, scales_->shape()[0]) : nullptr;
    BufferPtr new_zeros  = nullptr;
    if (zeros_ && zeros_->shape()[0] > (offset + size)) {
        new_zeros = zeros_->slice(offset, size);
    } else {
        new_zeros = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr));
    }
    return std::make_shared<QBuffer>(std::move(new_kernel), std::move(new_scales), std::move(new_zeros));
}

void* QBuffer::scalesData() const {
    return scales_->data();
}

void* QBuffer::zerosData() const {
    return zeros_->data();
}

DataType QBuffer::scalesType() const {
    return scales_->type();
}

DataType QBuffer::zerosType() const {
    return zeros_->type();
}

size_t QBuffer::scalesSizebytes() const {
    return scales_->sizeBytes();
}

size_t QBuffer::zerosSizebytes() const {
    return zeros_->sizeBytes();
}

}  // namespace rtp_llm
