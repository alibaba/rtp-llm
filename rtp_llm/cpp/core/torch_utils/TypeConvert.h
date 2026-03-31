#pragma once

#include <torch/extension.h>
#include "rtp_llm/cpp/core/Types.h"

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
        throw std::runtime_error("datatype not implemented: " + data_type);
    }
    return scalar_type;
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
            throw std::runtime_error("Unsupported torch dtype: " + std::to_string((int8_t)(dtype.toScalarType())));
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
        case TYPE_UINT64:
            return torch::kLong;
        case TYPE_BYTES:
            return torch::kByte;
        case TYPE_QFP8_E4M3:
            return TORCH_FP8_E4M3_TYPE;
        default:
            throw std::runtime_error("Unsupported data type: " + std::to_string((int8_t)data_type));
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

}  // namespace rtp_llm
