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
        {ErrorCode::INVALID_PARAMS, grpc::StatusCode::INVALID_ARGUMENT},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return grpc::StatusCode::INTERNAL;
    }
}

// Reverse of transErrorCodeToGrpc: extract a representative ErrorCode from a
// grpc::StatusCode received over the wire. Lossy by design — multiple
// ErrorCodes share the same grpc status (e.g. MALLOC_FAILED and
// DECODE_MALLOC_FAILED both → RESOURCE_EXHAUSTED), so we pick the most
// common producer for each status. Used by PrefillServerCallerContext
// when wrapping prefill's gRPC failures back into ErrorInfo for downstream
// consumers, so they don't all collapse to UNKNOWN_ERROR.
inline ErrorCode transGrpcStatusToErrorCode(grpc::StatusCode status_code) {
    const static std::unordered_map<grpc::StatusCode, ErrorCode> status_code_map = {
        {grpc::StatusCode::CANCELLED, ErrorCode::CANCELLED},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, ErrorCode::MALLOC_FAILED},
        {grpc::StatusCode::DEADLINE_EXCEEDED, ErrorCode::GENERATE_TIMEOUT},
        {grpc::StatusCode::OUT_OF_RANGE, ErrorCode::LONG_PROMPT_ERROR},
        {grpc::StatusCode::INVALID_ARGUMENT, ErrorCode::INVALID_PARAMS},
    };
    auto it = status_code_map.find(status_code);
    if (it != status_code_map.end()) {
        return it->second;
    }
    // grpc::StatusCode::INTERNAL and any other status fall here. INTERNAL is
    // a catch-all on the prefill side (transErrorCodeToGrpc default), so the
    // original ErrorCode is genuinely lost — keep UNKNOWN_ERROR.
    return ErrorCode::UNKNOWN_ERROR;
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
