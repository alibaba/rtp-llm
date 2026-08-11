#pragma once

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

inline ErrorInfo transGrpcStatusToErrorInfo(const grpc::Status& status, ErrorCode fallback) {
    if (status.ok()) {
        return ErrorInfo::OkStatus();
    }

    ErrorDetailsPB details;
    const bool has_details = !status.error_details().empty() && details.ParseFromString(status.error_details())
                             && details.error_code() != static_cast<int>(ErrorCode::NONE_ERROR);
    if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT,
                         has_details && !details.error_message().empty() ? details.error_message() :
                                                                         status.error_message());
    }
    if (has_details) {
        return ErrorInfo(static_cast<ErrorCode>(details.error_code()), details.error_message());
    }

    switch (status.error_code()) {
        case grpc::StatusCode::CANCELLED:
            return ErrorInfo(ErrorCode::CANCELLED, status.error_message());
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::DECODE_MALLOC_FAILED, status.error_message());
        default:
            return ErrorInfo(fallback, status.error_message());
    }
}

inline bool shouldRetryGenerateFailure(ErrorCode error_code, grpc::StatusCode status_code) {
    switch (error_code) {
        case ErrorCode::GENERATE_TIMEOUT:
        case ErrorCode::CANCELLED:
        case ErrorCode::LONG_PROMPT_ERROR:
        case ErrorCode::ERROR_GENERATE_CONFIG_FORMAT:
        case ErrorCode::INVALID_PARAMS:
        case ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN:
        case ErrorCode::OUT_OF_VOCAB_RANGE:
            return false;
        default:
            break;
    }

    switch (status_code) {
        case grpc::StatusCode::CANCELLED:
        case grpc::StatusCode::DEADLINE_EXCEEDED:
        case grpc::StatusCode::INVALID_ARGUMENT:
        case grpc::StatusCode::FAILED_PRECONDITION:
        case grpc::StatusCode::OUT_OF_RANGE:
        case grpc::StatusCode::UNAUTHENTICATED:
        case grpc::StatusCode::PERMISSION_DENIED:
        case grpc::StatusCode::UNIMPLEMENTED:
            return false;
        default:
            return true;
    }
}

inline ErrorCode transRemoteLoadGrpcStatus(grpc::StatusCode status_code, bool deadline_reached) noexcept {
    if (deadline_reached || status_code == grpc::StatusCode::DEADLINE_EXCEEDED) {
        return ErrorCode::LOAD_CACHE_TIMEOUT;
    }
    if (status_code == grpc::StatusCode::CANCELLED) {
        return ErrorCode::CANCELLED;
    }
    return ErrorCode::LOAD_KV_CACHE_FAILED;
}

inline ErrorCode mergeRemoteLoadErrorCode(ErrorCode current, ErrorCode candidate) noexcept {
    auto priority = [](ErrorCode code) {
        switch (code) {
            case ErrorCode::LOAD_CACHE_TIMEOUT:
                return 3;
            case ErrorCode::CANCELLED:
                return 2;
            case ErrorCode::NONE_ERROR:
                return 0;
            default:
                return 1;
        }
    };
    return priority(candidate) > priority(current) ? candidate : current;
}

inline ErrorCode transRPCErrorCode(ErrorCodePB error_code) {
    const static std::unordered_map<ErrorCodePB, ErrorCode> error_code_map = {
        {ErrorCodePB::NONE_ERROR, ErrorCode::NONE_ERROR},
        {ErrorCodePB::UNKNOWN_ERROR, ErrorCode::UNKNOWN_ERROR},
        {ErrorCodePB::CANCELLED, ErrorCode::CANCELLED},
        {ErrorCodePB::GENERATE_TIMEOUT, ErrorCode::GENERATE_TIMEOUT},
        {ErrorCodePB::LOAD_CACHE_TIMEOUT, ErrorCode::LOAD_CACHE_TIMEOUT},
        {ErrorCodePB::CACHE_STORE_LOAD_CONNECT_FAILED, ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_SEND_REQUEST_FAILED, ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED},
        {ErrorCodePB::CACHE_STORE_CALL_PREFILL_TIMEOUT, ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT},
        {ErrorCodePB::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED, ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_RDMA_WRITE_FAILED, ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED},
        {ErrorCodePB::CACHE_STORE_LOAD_BUFFER_TIMEOUT, ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT},
        {ErrorCodePB::P2P_CONNECTOR_CALL_PREFILL_FAILED, ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED, ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
         ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
         ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED,
         ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED, ErrorCode::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT, ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED,
         ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED,
         ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED,
         ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH, ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT,
         ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_FAILED, ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_CANCELED, ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TIMEOUT, ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT},
        {ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
         ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE},
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
        {ErrorCode::GENERATE_TIMEOUT, ErrorCodePB::GENERATE_TIMEOUT},
        {ErrorCode::LOAD_CACHE_TIMEOUT, ErrorCodePB::LOAD_CACHE_TIMEOUT},
        {ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED, ErrorCodePB::CACHE_STORE_LOAD_CONNECT_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED, ErrorCodePB::CACHE_STORE_LOAD_SEND_REQUEST_FAILED},
        {ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT, ErrorCodePB::CACHE_STORE_CALL_PREFILL_TIMEOUT},
        {ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED, ErrorCodePB::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED, ErrorCodePB::CACHE_STORE_LOAD_RDMA_WRITE_FAILED},
        {ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, ErrorCodePB::CACHE_STORE_LOAD_BUFFER_TIMEOUT},
        {ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED, ErrorCodePB::P2P_CONNECTOR_CALL_PREFILL_FAILED},
        {ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED, ErrorCodePB::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED},
        {ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
         ErrorCodePB::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED},
        {ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
         ErrorCodePB::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED},
        {ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED,
         ErrorCodePB::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED},
        {ErrorCode::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED, ErrorCodePB::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT, ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED,
         ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED,
         ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED,
         ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH, ErrorCodePB::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT,
         ErrorCodePB::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED, ErrorCodePB::P2P_CONNECTOR_WORKER_READ_FAILED},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED, ErrorCodePB::P2P_CONNECTOR_WORKER_READ_CANCELED},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT, ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TIMEOUT},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
         ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return ErrorCodePB::UNKNOWN_ERROR;
    }
}
}  // namespace rtp_llm
