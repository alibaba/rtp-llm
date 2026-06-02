#pragma once

#include <torch/extension.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

/// TensorPB ↔ torch::Tensor 转换，与 QueryConverter::transTensor / transTensorPB 逻辑一致，供多处复用。
struct TensorPbConvert {
    static torch::Tensor pbToTorch(const TensorPB& tensor_pb);
    static void          torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor);
};

}  // namespace rtp_llm
