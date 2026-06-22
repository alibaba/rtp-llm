#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <limits>
#include <stdexcept>

namespace rtp_llm {

namespace {

size_t dtypeSize(TensorPB::DataType dtype) {
    switch (dtype) {
        case TensorPB::FP32:
            return sizeof(float);
        case TensorPB::INT32:
            return sizeof(int32_t);
        case TensorPB::FP16:
            return sizeof(c10::Half);
        case TensorPB::BF16:
            return sizeof(c10::BFloat16);
        default:
            throw std::runtime_error("Unsupported TensorPB data type.");
    }
}

size_t dataBytes(const TensorPB& tensor_pb) {
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            return tensor_pb.fp32_data().size();
        case TensorPB::INT32:
            return tensor_pb.int32_data().size();
        case TensorPB::FP16:
            return tensor_pb.fp16_data().size();
        case TensorPB::BF16:
            return tensor_pb.bf16_data().size();
        default:
            throw std::runtime_error("Unsupported TensorPB data type.");
    }
}

torch::ScalarType pbDtypeToTorch(TensorPB::DataType dtype) {
    switch (dtype) {
        case TensorPB::FP32:
            return torch::kFloat32;
        case TensorPB::INT32:
            return torch::kInt32;
        case TensorPB::FP16:
            return torch::kFloat16;
        case TensorPB::BF16:
            return torch::kBFloat16;
        default:
            throw std::runtime_error("Unsupported TensorPB data type.");
    }
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());

    int64_t numel = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("TensorPB shape dimension must be non-negative, got "
                                     + std::to_string(dim));
        }
        if (dim != 0 && numel > std::numeric_limits<int64_t>::max() / dim) {
            throw std::runtime_error("TensorPB shape numel overflow");
        }
        numel *= dim;
    }

    // Default-constructed / empty TensorPB has no shape and no data. Do not
    // treat it as a scalar; return an empty 1-D tensor. Also handle explicitly
    // zero-volume tensors (e.g., shape {0}).
    if (shape.empty() && dataBytes(tensor_pb) == 0) {
        return torch::empty({0});
    }
    if (numel == 0) {
        auto options = torch::TensorOptions().dtype(pbDtypeToTorch(tensor_pb.data_type()));
        return torch::empty(shape, options);
    }

    // Validate that the declared payload size matches shape * dtype size before
    // reading the data pointer.
    const size_t expected_bytes = static_cast<size_t>(numel) * dtypeSize(tensor_pb.data_type());
    const size_t actual_bytes   = dataBytes(tensor_pb);
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error("TensorPB data size mismatch: expected "
                                 + std::to_string(expected_bytes) + " bytes, got "
                                 + std::to_string(actual_bytes) + " bytes for shape ["
                                 + [&shape]() {
                                       std::string s;
                                       for (size_t i = 0; i < shape.size(); ++i) {
                                           if (i) s += ", ";
                                           s += std::to_string(shape[i]);
                                       }
                                       return s;
                                   }()
                                 + "] and dtype " + std::to_string(tensor_pb.data_type()) + ".");
    }

    void*                data_ptr = nullptr;
    auto                 options  = torch::TensorOptions().dtype(pbDtypeToTorch(tensor_pb.data_type()));
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            data_ptr = const_cast<char*>(tensor_pb.fp32_data().data());
            break;
        }
        case TensorPB::INT32: {
            data_ptr = const_cast<char*>(tensor_pb.int32_data().data());
            break;
        }
        case TensorPB::FP16: {
            data_ptr = const_cast<char*>(tensor_pb.fp16_data().data());
            break;
        }
        case TensorPB::BF16: {
            data_ptr = const_cast<char*>(tensor_pb.bf16_data().data());
            break;
        }
        default:
            throw std::runtime_error("Unsupported TensorPB data type.");
    }
    return torch::from_blob(data_ptr, shape, options).clone();
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
