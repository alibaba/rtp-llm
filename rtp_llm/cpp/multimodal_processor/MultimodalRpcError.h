#pragma once

#include <grpcpp/support/status_code_enum.h>
#include <string>
#include <utility>

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

}  // namespace rtp_llm
