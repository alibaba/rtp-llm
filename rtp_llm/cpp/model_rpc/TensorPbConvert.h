#pragma once

#include <torch/extension.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

/// TensorPB ↔ torch::Tensor 转换，与 QueryConverter::transTensor / transTensorPB 逻辑一致，供多处复用。
struct TensorPbConvert {
    static torch::Tensor pbToTorch(const TensorPB& tensor_pb);
    /// Contract: accepts a tensor on any device. If the tensor is not on CPU, this
    /// function implicitly copies it to host (and makes it contiguous) before
    /// serialization, so callers do NOT need to call .cpu() beforehand.
    /// Note: the implicit device-to-host copy synchronizes only with the calling
    /// thread's current stream; callers must ensure the tensor's producing
    /// computation has completed (or been synchronized) before passing it in.
    static void          torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor);
};

}  // namespace rtp_llm
