#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

namespace {

size_t checkedElementCount(const TensorPB& tensor_pb) {
    size_t numel = 1;
    for (int i = 0; i < tensor_pb.shape_size(); ++i) {
        const int64_t dim = tensor_pb.shape(i);
        if (dim < 0) {
            throw std::runtime_error("TensorPB shape dim must be non-negative.");
        }
        const auto dim_size = static_cast<size_t>(dim);
        if (dim_size != 0 && numel > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error("TensorPB shape element count overflow.");
        }
        numel *= dim_size;
    }
    return numel;
}

void validateTensorPayload(const TensorPB& tensor_pb, const std::string& active_data, size_t dtype_size) {
    const size_t numel = checkedElementCount(tensor_pb);
    if (numel > std::numeric_limits<size_t>::max() / dtype_size) {
        throw std::runtime_error("TensorPB payload byte size overflow.");
    }
    const size_t expected_bytes = numel * dtype_size;
    if (active_data.size() != expected_bytes) {
        throw std::runtime_error("TensorPB payload byte size does not match shape and dtype.");
    }
}

void validateInactivePayloadsAreEmpty(const TensorPB& tensor_pb) {
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            if (!tensor_pb.int32_data().empty() || !tensor_pb.fp16_data().empty() || !tensor_pb.bf16_data().empty()) {
                throw std::runtime_error("TensorPB contains payload for inactive dtype field.");
            }
            break;
        case TensorPB::INT32:
            if (!tensor_pb.fp32_data().empty() || !tensor_pb.fp16_data().empty() || !tensor_pb.bf16_data().empty()) {
                throw std::runtime_error("TensorPB contains payload for inactive dtype field.");
            }
            break;
        case TensorPB::FP16:
            if (!tensor_pb.fp32_data().empty() || !tensor_pb.int32_data().empty() || !tensor_pb.bf16_data().empty()) {
                throw std::runtime_error("TensorPB contains payload for inactive dtype field.");
            }
            break;
        case TensorPB::BF16:
            if (!tensor_pb.fp32_data().empty() || !tensor_pb.int32_data().empty() || !tensor_pb.fp16_data().empty()) {
                throw std::runtime_error("TensorPB contains payload for inactive dtype field.");
            }
            break;
        default:
            break;
    }
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    validateInactivePayloadsAreEmpty(tensor_pb);
    void* data_ptr = nullptr;
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            validateTensorPayload(tensor_pb, tensor_pb.fp32_data(), sizeof(float));
            data_ptr     = const_cast<char*>(tensor_pb.fp32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::INT32: {
            validateTensorPayload(tensor_pb, tensor_pb.int32_data(), sizeof(int32_t));
            data_ptr     = const_cast<char*>(tensor_pb.int32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::FP16: {
            validateTensorPayload(tensor_pb, tensor_pb.fp16_data(), sizeof(c10::Half));
            data_ptr     = const_cast<char*>(tensor_pb.fp16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::BF16: {
            validateTensorPayload(tensor_pb, tensor_pb.bf16_data(), sizeof(c10::BFloat16));
            data_ptr     = const_cast<char*>(tensor_pb.bf16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        default:
            throw std::runtime_error("Unsupported data type.");
    }
}

void TensorPbConvert::torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    tensor_pb->clear_shape();
    tensor_pb->clear_fp32_data();
    tensor_pb->clear_int32_data();
    tensor_pb->clear_fp16_data();
    tensor_pb->clear_bf16_data();
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
