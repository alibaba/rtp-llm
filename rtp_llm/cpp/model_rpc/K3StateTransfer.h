#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {
inline bool supportsK3StateTransfer(const GenerateRequestPB& request, DataType decode_dtype) {
    const auto storage = request.prefill_ssm_state_dtype();
    const auto wire = (request.prefill_ssm_transfer_dtype_presence_case() == GenerateRequestPB::kPrefillSsmTransferDtype) ? request.prefill_ssm_transfer_dtype() : storage;
    return (storage == static_cast<int32_t>(DataType::TYPE_FP32)
            || storage == static_cast<int32_t>(DataType::TYPE_BF16))
           && wire == static_cast<int32_t>(DataType::TYPE_FP32)
           && decode_dtype == DataType::TYPE_FP32;
}
}  // namespace rtp_llm
