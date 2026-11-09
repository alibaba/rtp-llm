#pragma once

#include <cstdint>
#include <grpcpp/support/status.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Preserve client input errors across the ViT -> Prefill RPC boundary. The
// caller only invokes this for failed RPCs; unknown failures stay internal.
inline ErrorCode multimodalRpcErrorCode(grpc::StatusCode status) {
    switch (status) {
        case grpc::StatusCode::CANCELLED:
            return ErrorCode::CANCELLED;
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return ErrorCode::CONCURRENCY_LIMIT_ERROR;
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return ErrorCode::GENERATE_TIMEOUT;
        case grpc::StatusCode::INVALID_ARGUMENT:
            return ErrorCode::MM_WRONG_FORMAT_ERROR;
        default:
            return ErrorCode::MM_PROCESS_ERROR;
    }
}

inline bool isMultimodalRpcErrorCode(int64_t code) {
    return code >= static_cast<int64_t>(ErrorCode::MM_LONG_PROMPT_ERROR)
           && code <= static_cast<int64_t>(ErrorCode::UNSAFE_OUTPUT_CONTENT);
}

inline ErrorCode multimodalRpcErrorCode(const grpc::Status& status) {
    ErrorDetailsPB details;
    if (details.ParseFromString(status.error_details())) {
        const auto code = details.error_code();
        // Accept the ViT application codes only. Unknown, absent, or malformed
        // details use the existing gRPC transport mapping.
        if (isMultimodalRpcErrorCode(code)) {
            return static_cast<ErrorCode>(code);
        }
    }
    return multimodalRpcErrorCode(status.error_code());
}

}  // namespace rtp_llm
