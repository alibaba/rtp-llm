#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <cstring>
#include <stdexcept>

namespace rtp_llm {

namespace {

torch::Tensor copyPbTensor(
    const void* data, size_t data_bytes, const std::vector<int64_t>& shape, torch::ScalarType dtype, bool pinned) {
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    if (pinned) {
        options = options.pinned_memory(true);
    }
    auto result = torch::empty(shape, options);
    if (result.nbytes() != static_cast<int64_t>(data_bytes)) {
        throw std::runtime_error("TensorPB data size does not match shape and dtype");
    }
    if (data_bytes > 0) {
        std::memcpy(result.data_ptr(), data, data_bytes);
    }
    return result;
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    void*                data_ptr = nullptr;
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            data_ptr     = const_cast<char*>(tensor_pb.fp32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::INT32: {
            data_ptr     = const_cast<char*>(tensor_pb.int32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::FP16: {
            data_ptr     = const_cast<char*>(tensor_pb.fp16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::BF16: {
            data_ptr     = const_cast<char*>(tensor_pb.bf16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        default:
            throw std::runtime_error("Unsupported data type.");
    }
}

torch::Tensor TensorPbConvert::pbToPinnedTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            return copyPbTensor(
                tensor_pb.fp32_data().data(), tensor_pb.fp32_data().size(), shape, torch::kFloat32, true);
        case TensorPB::INT32:
            return copyPbTensor(
                tensor_pb.int32_data().data(), tensor_pb.int32_data().size(), shape, torch::kInt32, true);
        case TensorPB::FP16:
            return copyPbTensor(
                tensor_pb.fp16_data().data(), tensor_pb.fp16_data().size(), shape, torch::kFloat16, true);
        case TensorPB::BF16:
            return copyPbTensor(
                tensor_pb.bf16_data().data(), tensor_pb.bf16_data().size(), shape, torch::kBFloat16, true);
        default:
            throw std::runtime_error("Unsupported data type.");
    }
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
