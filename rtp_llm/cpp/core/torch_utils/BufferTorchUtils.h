#pragma once

#include <torch/extension.h>
#include <torch/all.h>

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StackTrace.h"

#include <array>

namespace rtp_llm {

inline at::ScalarType getScalarType(const std::string& data_type) {
    at::ScalarType scalar_type;
    if (data_type == "fp16") {
        scalar_type = at::ScalarType::Half;

    } else if (data_type == "bf16") {
        scalar_type = at::ScalarType::BFloat16;

    } else if (data_type == "fp32") {
        scalar_type = at::ScalarType::Float;

    } else {
        RTP_LLM_LOG_ERROR("datatype not implemented %s", data_type.c_str());
    }
    return scalar_type;
}

using BufferShapeType = std::remove_cv<std::remove_reference<decltype(std::declval<Buffer>().shape())>::type>::type;

inline BufferShapeType torchShapeToBufferShape(const c10::IntArrayRef& sizes) {
    BufferShapeType shape;
    for (int i = 0; i < int(sizes.size()); i++) {
        shape.push_back(sizes[i]);
    }
    // when tensor only one element, sizes will be empty
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
}

inline std::vector<int64_t> bufferShapeToTorchShape(const Buffer& buffer) {
    std::vector<int64_t> tensor_shape(buffer.shape().size());
    std::transform(
        buffer.shape().begin(), buffer.shape().end(), tensor_shape.begin(), [](size_t x) { return (int64_t)x; });
    return tensor_shape;
}

#if USING_ROCM
#define TORCH_FP8_E4M3_TYPE torch::kFloat8_e4m3fnuz
#else
#define TORCH_FP8_E4M3_TYPE torch::kFloat8_e4m3fn
#endif

#define FOREACH_BUFFER_TORCH_TYPE_MAP(F)                                                                               \
    F(TYPE_UINT8, torch::kByte)                                                                                        \
    F(TYPE_INT8, torch::kChar)                                                                                         \
    F(TYPE_INT16, torch::kShort)                                                                                       \
    F(TYPE_INT32, torch::kInt)                                                                                         \
    F(TYPE_INT64, torch::kLong)                                                                                        \
    F(TYPE_FP16, torch::kHalf)                                                                                         \
    F(TYPE_FP32, torch::kFloat)                                                                                        \
    F(TYPE_FP64, torch::kDouble)                                                                                       \
    F(TYPE_BOOL, torch::kBool)                                                                                         \
    F(TYPE_BF16, torch::kBFloat16)                                                                                     \
    F(TYPE_FP8_E4M3, TORCH_FP8_E4M3_TYPE)

inline DataType torchDTypeToDataType(caffe2::TypeMeta dtype) {
#define TYPE_CASE(type, torch_type)                                                                                    \
    case torch_type: {                                                                                                 \
        return type;                                                                                                   \
    }

    switch (dtype.toScalarType()) {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
        default:
            printStackTrace();
            RTP_LLM_LOG_ERROR("Unsupported data type: [%d]%s", dtype.toScalarType(), dtype.name().data());
            throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

inline c10::ScalarType dataTypeToTorchType(DataType data_type) {
#define TYPE_CASE(type, torch_type)                                                                                    \
    case type: {                                                                                                       \
        return torch_type;                                                                                             \
    }

    switch (data_type) {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
        case TYPE_UINT32:
            return torch::kInt;
        case TYPE_BYTES:
            return torch::kByte;
        case TYPE_QFP8_E4M3:
            return TORCH_FP8_E4M3_TYPE;
        default:
            printStackTrace();
            RTP_LLM_LOG_ERROR("Unsupported data type: [%d]", data_type);
            throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)data_type));
    }

#undef TYPE_CASE
}

#undef FOREACH_BUFFER_TORCH_TYPE_MAP

inline MemoryType torchDeviceToMemoryType(const c10::Device& device) {
    return device.is_cuda() ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

inline c10::Device memoryTypeToTorchDevice(const MemoryType& memory_type) {
    return memory_type == MemoryType::MEMORY_GPU ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
}

inline BufferPtr torchTensor2Buffer(const torch::Tensor& tensor) {
    const auto& data        = tensor.data_ptr();
    const auto& shape       = torchShapeToBufferShape(tensor.sizes());
    const auto& dtype       = torchDTypeToDataType(tensor.dtype());
    const auto& memory_type = torchDeviceToMemoryType(tensor.device());
    return std::make_shared<Buffer>(memory_type, dtype, shape, data);
}

inline std::vector<BufferPtr> torchTensorVec2BufferVec(const std::vector<torch::Tensor>& tensor_vec) {
    std::vector<BufferPtr> res;
    for (auto& tensor : tensor_vec) {
        res.emplace_back(torchTensor2Buffer(tensor));
    }
    return res;
}

inline BufferPtr
torchTensor2Buffer(const torch::Tensor& tensor, const torch::Tensor& scales, const torch::Tensor& zeros) {
    return BufferPtr(new QBuffer(std::move(torchTensor2Buffer(tensor)),
                                 std::move(torchTensor2Buffer(scales)),
                                 std::move(torchTensor2Buffer(zeros))));
}

inline BufferPtr torchTensor2BufferWithDstType(const torch::Tensor& tensor, c10::ScalarType dst_type) {
    auto typed_tensor = tensor.toType(dst_type);

    const auto& data        = typed_tensor.data_ptr();
    const auto& shape       = torchShapeToBufferShape(typed_tensor.sizes());
    const auto& dtype       = torchDTypeToDataType(typed_tensor.dtype());
    const auto& memory_type = torchDeviceToMemoryType(typed_tensor.device());
    return std::make_shared<Buffer>(memory_type, dtype, shape, data, [typed_tensor](Buffer* data) {
        // do nothing, just for typed_tensor will not release before buffer release
    });
}

inline torch::Tensor Buffer2torchTensor(const Buffer& buf, bool copyData = true) {
    if (buf.isQBuffer()) {
        throw std::runtime_error("not support qbuffer!");
    }
    auto option =
        torch::dtype(dataTypeToTorchType(buf.type())).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);
    if (copyData) {
        torch::Tensor out = torch::zeros(bufferShapeToTorchShape(buf), option);
        if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
            memcpy(out.data_ptr(), buf.data(), buf.sizeBytes());
        } else {
            throw std::runtime_error("Buffer2torchTensor gpu copy not implemented");
        }
        return out;
    } else {
        return torch::from_blob(buf.data(), bufferShapeToTorchShape(buf), option);
    }
}

inline torch::Tensor Buffer2torchTensor(const ConstBufferPtr& buf, bool copyData = true) {
    if (buf == nullptr) {
        throw std::runtime_error("buf is nullptr");
    }
    if (buf->size() == 0) {
        auto option = torch::dtype(dataTypeToTorchType(buf->type()))
                          .device(memoryTypeToTorchDevice(buf->where()))
                          .requires_grad(false);
        return torch::empty(bufferShapeToTorchShape(*buf), option);
    }
    return Buffer2torchTensor(*buf, copyData);
}

inline torch::Tensor Buffer2torchTensorWithDstType(const Buffer& buf, bool copyData, c10::ScalarType dst_type) {
    if (buf.isQBuffer()) {
        throw std::runtime_error("not support qbuffer!");
    }
    if (buf.size() == 0) {
        auto option = torch::dtype(dst_type).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);
        return torch::empty(bufferShapeToTorchShape(buf), option);
    }
    auto tensor = Buffer2torchTensor(buf, copyData);
    if (tensor.dtype() == dst_type) {
        return tensor;
    }
    return tensor.toType(dst_type);
}

inline torch::Tensor Buffer2torchTensorWithDstType(const ConstBufferPtr& buf, bool copyData, c10::ScalarType dst_type) {
    if (buf == nullptr) {
        throw std::runtime_error("buf is nullptr");
    }
    return Buffer2torchTensorWithDstType(*buf, copyData, dst_type);
}

inline torch::Tensor
Buffer2torchTensorWithStride(const Buffer& buf, const std::vector<int64_t>& new_shape, const size_t offset = 0) {
    if (buf.isQBuffer()) {
        throw std::runtime_error("not support qbuffer!");
    }
    if (buf.shape().size() != new_shape.size()) {
        throw std::runtime_error("strides size not match new_shape size");
    }
    for (int i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] > buf.shape()[i]) {
            throw std::runtime_error("new_shape not match buf.shape()");
        }
    }
    auto strides = buf.strides();
    auto option =
        torch::dtype(dataTypeToTorchType(buf.type())).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);

    // 直接从现有内存创建带stride的tensor
    return torch::from_blob(buf.dataWithOffset(offset), new_shape, buf.strides(), option);
}

inline std::array<torch::Tensor, 3> QBuffer2torchTensor(const ConstQBufferPtr& buf, bool copyData = true) {
    if (!buf->isQBuffer()) {
        throw std::runtime_error("only support qbuffer!");
    }

    return {Buffer2torchTensor(std::move(BufferPtr(new Buffer(
                buf->kernel().where(), buf->kernel().type(), buf->kernel().shape(), buf->kernel().data(), nullptr)))),
            Buffer2torchTensor(std::move(BufferPtr(new Buffer(
                buf->scales().where(), buf->scales().type(), buf->scales().shape(), buf->scales().data(), nullptr)))),
            Buffer2torchTensor(std::move(BufferPtr(new Buffer(
                buf->zeros().where(), buf->zeros().type(), buf->zeros().shape(), buf->zeros().data(), nullptr))))};
}

}  // namespace rtp_llm
