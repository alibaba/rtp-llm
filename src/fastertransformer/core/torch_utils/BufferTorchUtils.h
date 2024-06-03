#pragma once

#include <torch/extension.h>
#include <torch/all.h>

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"


namespace fastertransformer {

using BufferShapeType = std::remove_cv<std::remove_reference<
        decltype(std::declval<Buffer>().shape())
    >::type>::type;

inline BufferShapeType torchShapeToBufferShape(const c10::IntArrayRef& sizes) {
    BufferShapeType shape;
    for (int i = 0; i < sizes.size(); i++) {
        shape.push_back(sizes[i]);
    }
    return shape;
}

inline std::vector<int64_t> bufferShapeToTorchShape(const Buffer& buffer) {
    std::vector<int64_t> tensor_shape(buffer.shape().size());
    std::transform(
        buffer.shape().begin(), buffer.shape().end(), tensor_shape.begin(), [](size_t x) { return (int64_t)x;});
    return tensor_shape;
}

inline size_t calcTensorBytes(torch::Tensor tensor) {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

#define FOREACH_BUFFER_TORCH_TYPE_MAP(F)    \
    F(TYPE_UINT8, torch::kByte)             \
    F(TYPE_INT8, torch::kChar)              \
    F(TYPE_INT16, torch::kShort)            \
    F(TYPE_INT32, torch::kInt)              \
    F(TYPE_INT64, torch::kLong)             \
    F(TYPE_FP16, torch::kHalf)              \
    F(TYPE_FP32, torch::kFloat)             \
    F(TYPE_FP64, torch::kDouble)            \
    F(TYPE_BOOL, torch::kBool)              \
    F(TYPE_BF16, torch::kBFloat16)          \
    F(TYPE_FP8_E4M3, torch::kFloat8_e4m3fn)

inline DataType torchDTypeToDataType(caffe2::TypeMeta dtype) {
#define TYPE_CASE(type, torch_type) \
    case torch_type: { \
        return type;   \
    }

    switch (dtype.toScalarType()) {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        FT_LOG_ERROR("Unsupported data type: [%d]%s", dtype.toScalarType(), dtype.name().data());
        throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

inline c10::ScalarType dataTypeToTorchType(DataType data_type) {
#define TYPE_CASE(type, torch_type) \
    case type: { \
        return torch_type;   \
    }

    switch (data_type) {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        FT_LOG_ERROR("Unsupported data type: [%d]", data_type);
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
    const auto& data = tensor.data_ptr();
    const auto& shape = torchShapeToBufferShape(tensor.sizes());
    const auto& dtype = torchDTypeToDataType(tensor.dtype());
    const auto& memory_type = torchDeviceToMemoryType(tensor.device());
    return std::make_unique<Buffer>(memory_type, dtype, shape, data);
}

inline torch::Tensor Buffer2torchTensor(const ConstBufferPtr& buf, bool copyData = true) {
    auto option = torch::dtype(dataTypeToTorchType(buf->type())).device(memoryTypeToTorchDevice(buf->where())).requires_grad(false);
    if (copyData) {
        torch::Tensor out = torch::zeros(bufferShapeToTorchShape(*buf), option);
        if (buf->where() == MemoryType::MEMORY_CPU || buf->where() == MemoryType::MEMORY_CPU_PINNED) {
            memcpy(out.data_ptr(), buf->data(), buf->sizeBytes());
        } else {
            throw std::runtime_error("not implemented");
        }
        return out;
    } else {
        return torch::from_blob(buf->data(), bufferShapeToTorchShape(*buf), option);        
    }
}

inline void bufferCopy(const BufferPtr& src, torch::Tensor& dst, size_t numberOfElements) {
    assert(dst.device().is_cpu());
    size_t copySize = src->typeSize() * numberOfElements;
    memcpy((void*)(dst.data_ptr<int>()), src->data(), copySize);
}

} // namespace fastertransformer
