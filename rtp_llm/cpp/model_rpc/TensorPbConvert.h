#pragma once

#include <ATen/ATen.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

/// TensorPB ↔ at::Tensor 转换，与 QueryConverter::transTensor / transTensorPB 逻辑一致，供多处复用。
struct TensorPbConvert {
    // Validate dtype, dimensions, active payload size and inactive payload fields
    // without allocating a tensor or borrowing a typed pointer into protobuf bytes.
    static void       validate(const TensorPB& tensor_pb);
    static at::Tensor pbToTorch(const TensorPB& tensor_pb);
    static void       torchToPb(TensorPB* tensor_pb, const at::Tensor& tensor);
};

}  // namespace rtp_llm
