#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

namespace {

struct TensorPayload {
    c10::ScalarType    dtype;
    const std::string* bytes;
    size_t             element_size;
};

TensorPayload selectPayload(const TensorPB& tensor_pb) {
    const auto reject_unexpected_payload = [](const std::string& payload, const char* field_name) {
        if (!payload.empty()) {
            throw std::invalid_argument(std::string("TensorPB contains unexpected payload field: ") + field_name);
        }
    };

    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kFloat32, &tensor_pb.fp32_data(), sizeof(float)};
        case TensorPB::INT32:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kInt32, &tensor_pb.int32_data(), sizeof(int32_t)};
        case TensorPB::FP16:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kFloat16, &tensor_pb.fp16_data(), sizeof(c10::Half)};
        case TensorPB::BF16:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            return {torch::kBFloat16, &tensor_pb.bf16_data(), sizeof(c10::BFloat16)};
        default:
            throw std::invalid_argument("TensorPB has an unsupported data type");
    }
}

size_t checkedPayloadSize(const TensorPB& tensor_pb, size_t element_size) {
    size_t element_count = 1;
    for (const int64_t dimension : tensor_pb.shape()) {
        if (dimension < 0) {
            throw std::invalid_argument("TensorPB shape contains a negative dimension");
        }
        const size_t unsigned_dimension = static_cast<size_t>(dimension);
        if (unsigned_dimension != 0 && element_count > std::numeric_limits<size_t>::max() / unsigned_dimension) {
            throw std::invalid_argument("TensorPB shape element count overflows size_t");
        }
        element_count *= unsigned_dimension;
        if (element_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("TensorPB shape element count exceeds the tensor limit");
        }
    }
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        throw std::invalid_argument("TensorPB payload byte count overflows size_t");
    }
    return element_count * element_size;
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    const auto           payload       = selectPayload(tensor_pb);
    const size_t         expected_size = checkedPayloadSize(tensor_pb, payload.element_size);
    if (payload.bytes->size() != expected_size) {
        throw std::invalid_argument("TensorPB payload size does not match its shape and data type");
    }

    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(payload.dtype));
    if (expected_size != 0) {
        std::memcpy(tensor.data_ptr(), payload.bytes->data(), expected_size);
    }
    return tensor;
}

void TensorPbConvert::torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    if (tensor_pb == nullptr) {
        throw std::invalid_argument("TensorPB output must not be null");
    }
    TensorPB::DataType data_type;
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32:
            data_type = TensorPB::FP32;
            break;
        case torch::kInt32:
            data_type = TensorPB::INT32;
            break;
        case torch::kFloat16:
            data_type = TensorPB::FP16;
            break;
        case torch::kBFloat16:
            data_type = TensorPB::BF16;
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    TensorPB converted;
    converted.set_data_type(data_type);
    auto shape = tensor.sizes();
    for (auto dim : shape) {
        converted.add_shape(dim);
    }
    torch::Tensor contiguous_tensor = tensor.contiguous();
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(float);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_fp32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kInt32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(int32_t);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_int32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::Half);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_fp16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kBFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::BFloat16);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_bf16_data(data_ptr, num_bytes);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    tensor_pb->Swap(&converted);
}

}  // namespace rtp_llm
