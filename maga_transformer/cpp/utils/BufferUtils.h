#pragma once

#include <torch/extension.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"

namespace ft = fastertransformer;

namespace rtp_llm {

using BufferShapeType = std::remove_cv<std::remove_reference<
        decltype(std::declval<ft::Buffer>().shape())
    >::type>::type;

inline BufferShapeType torchShapeToBufferShape(const c10::IntArrayRef& sizes) {
    BufferShapeType shape;
    for (int i = 0; i < sizes.size(); i++) {
        shape.push_back(sizes[i]);
    }
    return shape;
}

inline ft::DataType torchDTypeToDataType(caffe2::TypeMeta dtype) {
    switch (dtype.toScalarType()) {
    case torch::ScalarType::Byte:
        return ft::DataType::TYPE_UINT8;
    case torch::ScalarType::Char:
        return ft::DataType::TYPE_INT8;
    case torch::ScalarType::Short:
        return ft::DataType::TYPE_INT16;
    case torch::ScalarType::Int:
        return ft::DataType::TYPE_INT32;
    case torch::ScalarType::Long:
        return ft::DataType::TYPE_INT64;
    case torch::ScalarType::Half:
        return ft::DataType::TYPE_FP16;
    case torch::ScalarType::Float:
        return ft::DataType::TYPE_FP32;
    case torch::ScalarType::Double:
        return ft::DataType::TYPE_FP64;
    case torch::ScalarType::Bool:
        return ft::DataType::TYPE_BOOL;
    case torch::ScalarType::BFloat16:
        return ft::DataType::TYPE_BF16;
    case torch::ScalarType::Float8_e4m3fn:
        return ft::DataType::TYPE_FP8_E4M3;
    default:
        FT_LOG_ERROR("Unsupported data type: [%d]%s", dtype.toScalarType(), dtype.name().data());
        throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }
}

ft::MemoryType torchDeviceToMemoryType(const c10::Device& device ) {
    return device.is_cuda() ? ft::MemoryType::MEMORY_GPU : ft::MemoryType::MEMORY_CPU;
}

ft::ConstBufferPtr torchTensor2Buffer(const torch::Tensor& tensor) {
    const auto& data = tensor.data_ptr();
    const auto& shape = torchShapeToBufferShape(tensor.sizes());
    const auto& dtype = torchDTypeToDataType(tensor.dtype());
    const auto& memory_type = torchDeviceToMemoryType(tensor.device());
    return std::make_unique<const ft::Buffer>(memory_type, dtype, shape, data);
}

}

