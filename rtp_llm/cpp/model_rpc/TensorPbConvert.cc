#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <limits>
#include <stdexcept>

namespace rtp_llm {

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    const std::string*   payload      = nullptr;
    c10::ScalarType      scalar_type  = torch::kFloat32;
    size_t               element_size = 0;
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            payload      = &tensor_pb.fp32_data();
            scalar_type  = torch::kFloat32;
            element_size = sizeof(float);
            break;
        }
        case TensorPB::INT32: {
            payload      = &tensor_pb.int32_data();
            scalar_type  = torch::kInt32;
            element_size = sizeof(int32_t);
            break;
        }
        case TensorPB::FP16: {
            payload      = &tensor_pb.fp16_data();
            scalar_type  = torch::kFloat16;
            element_size = sizeof(c10::Half);
            break;
        }
        case TensorPB::BF16: {
            payload      = &tensor_pb.bf16_data();
            scalar_type  = torch::kBFloat16;
            element_size = sizeof(c10::BFloat16);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type.");
    }

    if (shape.empty() && payload->empty()) {
        return torch::empty({0}, torch::TensorOptions().dtype(scalar_type));
    }

    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("TensorPB shape contains a negative dimension.");
        }
        const size_t unsigned_dim = static_cast<size_t>(dim);
        if (unsigned_dim > 0 && numel > std::numeric_limits<size_t>::max() / unsigned_dim) {
            throw std::runtime_error("TensorPB element count overflows.");
        }
        numel *= unsigned_dim;
    }
    if (element_size > 0 && numel > std::numeric_limits<size_t>::max() / element_size) {
        throw std::runtime_error("TensorPB byte size overflows.");
    }
    const size_t expected_bytes = numel * element_size;
    if (payload->size() != expected_bytes) {
        throw std::runtime_error("TensorPB payload size does not match shape and dtype.");
    }

    void* data_ptr = const_cast<char*>(payload->data());
    return torch::from_blob(data_ptr, shape, torch::TensorOptions().dtype(scalar_type)).clone();
}

void TensorPbConvert::torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32:
            tensor_pb->set_data_type(TensorPB::FP32);
            break;
        case torch::kInt32:
            tensor_pb->set_data_type(TensorPB::INT32);
            break;
        case torch::kFloat16:
            tensor_pb->set_data_type(TensorPB::FP16);
            break;
        case torch::kBFloat16:
            tensor_pb->set_data_type(TensorPB::BF16);
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    auto shape = tensor.sizes();
    for (auto dim : shape) {
        tensor_pb->add_shape(dim);
    }
    torch::Tensor contiguous_tensor = tensor.contiguous();
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(float);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_fp32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kInt32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(int32_t);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_int32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::Half);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_fp16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kBFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::BFloat16);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_bf16_data(data_ptr, num_bytes);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
}

}  // namespace rtp_llm
