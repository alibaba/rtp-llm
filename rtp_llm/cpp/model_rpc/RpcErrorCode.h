#pragma once

#include <optional>
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"

namespace rtp_llm {

inline grpc::StatusCode transErrorCodeToGrpc(ErrorCode error_code) {
    const static std::unordered_map<ErrorCode, grpc::StatusCode> error_code_map = {
        {ErrorCode::CANCELLED, grpc::StatusCode::CANCELLED},
        {ErrorCode::MALLOC_FAILED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::DECODE_MALLOC_FAILED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::GENERATE_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::OUT_OF_VOCAB_RANGE, grpc::StatusCode::OUT_OF_RANGE},
        {ErrorCode::LONG_PROMPT_ERROR, grpc::StatusCode::OUT_OF_RANGE},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return grpc::StatusCode::INTERNAL;
    }
}

inline ErrorCode transRPCErrorCode(ErrorCodePB error_code) {
    const static std::unordered_map<ErrorCodePB, ErrorCode> error_code_map = {
        {ErrorCodePB::NONE_ERROR, ErrorCode::NONE_ERROR},
        {ErrorCodePB::UNKNOWN_ERROR, ErrorCode::UNKNOWN_ERROR},
        {ErrorCodePB::CANCELLED, ErrorCode::CANCELLED},
        {ErrorCodePB::LOAD_CACHE_TIMEOUT, ErrorCode::LOAD_CACHE_TIMEOUT},
        {ErrorCodePB::CACHE_STORE_LOAD_CONNECT_FAILED, ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_SEND_REQUEST_FAILED, ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED},
        {ErrorCodePB::CACHE_STORE_CALL_PREFILL_TIMEOUT, ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT},
        {ErrorCodePB::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED, ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_RDMA_WRITE_FAILED, ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_BUFFER_TIMEOUT, ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return ErrorCode::UNKNOWN_ERROR;
    }
}

inline ErrorCodePB transErrorCodeToRPC(ErrorCode error_code) {
    const static std::unordered_map<ErrorCode, ErrorCodePB> error_code_map = {
        {ErrorCode::NONE_ERROR, ErrorCodePB::NONE_ERROR},
        {ErrorCode::UNKNOWN_ERROR, ErrorCodePB::UNKNOWN_ERROR},
        {ErrorCode::CANCELLED, ErrorCodePB::CANCELLED},
        {ErrorCode::LOAD_CACHE_TIMEOUT, ErrorCodePB::LOAD_CACHE_TIMEOUT},
        {ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED, ErrorCodePB::CACHE_STORE_LOAD_CONNECT_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED, ErrorCodePB::CACHE_STORE_LOAD_SEND_REQUEST_FAILED},
        {ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT, ErrorCodePB::CACHE_STORE_CALL_PREFILL_TIMEOUT},
        {ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED, ErrorCodePB::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED, ErrorCodePB::CACHE_STORE_LOAD_RDMA_WRITE_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, ErrorCodePB::CACHE_STORE_LOAD_BUFFER_TIMEOUT},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return ErrorCodePB::UNKNOWN_ERROR;
    }
}

// Extract ErrorInfo from grpc::Status, preserving original error code from error_details if available
inline std::optional<ErrorInfo> extractErrorInfoFromGrpcStatus(const grpc::Status& status) {
    if (status.ok()) {
        return std::nullopt;
    }

    // Try to extract error details from grpc status
    ErrorDetailsPB error_details;
    if (error_details.ParseFromString(status.error_details())) {
        // Successfully extracted error details, use original error code and message
        return ErrorInfo(static_cast<ErrorCode>(error_details.error_code()), error_details.error_message());
    }

    // Fallback: error_details not available, return nullopt to indicate extraction failed
    return std::nullopt;
}
}  // namespace rtp_llm
