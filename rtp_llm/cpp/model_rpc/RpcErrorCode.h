#pragma once

#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <unordered_map>

namespace rtp_llm {

inline grpc::StatusCode transErrorCodeToGrpc(ErrorCode error_code) {
    const static std::unordered_map<ErrorCode, grpc::StatusCode> error_code_map = {
        {ErrorCode::CANCELLED, grpc::StatusCode::CANCELLED},
        {ErrorCode::CONCURRENCY_LIMIT_ERROR, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::INVALID_PARAMS, grpc::StatusCode::INVALID_ARGUMENT},
        {ErrorCode::MALLOC_FAILED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::DECODE_MALLOC_FAILED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        // AutoTPM Cancel: victim terminal maps to
        // RESOURCE_EXHAUSTED (HTTP 429 upstream); exact 8429 travels in
        // ErrorDetailsPB trailing metadata.
        {ErrorCode::PRIORITY_PREEMPTED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::GRAMMAR_COMPILE_OVERLOADED, grpc::StatusCode::RESOURCE_EXHAUSTED},
        {ErrorCode::GENERATE_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::CONNECT_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::DEADLINE_EXCEEDED, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::WAIT_TO_RUN_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::KEEP_ALIVE_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::LOAD_CACHE_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT, grpc::StatusCode::DEADLINE_EXCEEDED},
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
        {grpc::StatusCode::UNAVAILABLE, ErrorCode::CONNECT_FAILED},
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

namespace rpc_error_detail {

inline constexpr std::pair<ErrorCodePB, ErrorCode> kErrorCodePairs[] = {
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
    {ErrorCodePB::INVALID_PARAMS, ErrorCode::INVALID_PARAMS},
    {ErrorCodePB::EXECUTION_EXCEPTION, ErrorCode::EXECUTION_EXCEPTION},
    {ErrorCodePB::GENERATE_TIMEOUT, ErrorCode::GENERATE_TIMEOUT},
    {ErrorCodePB::GET_CONNECTION_FAILED, ErrorCode::GET_CONNECTION_FAILED},
    {ErrorCodePB::RPC_FINISH_FAILED, ErrorCode::RPC_FINISH_FAILED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_BUILD_REQUEST_FAILED, ErrorCode::P2P_CONNECTOR_TRANSFER_BUILD_REQUEST_FAILED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_CONNECTION_FAILED, ErrorCode::P2P_CONNECTOR_TRANSFER_CONNECTION_FAILED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_RPC_FAILED, ErrorCode::P2P_CONNECTOR_TRANSFER_RPC_FAILED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_NOT_INITIALIZED, ErrorCode::P2P_CONNECTOR_TRANSFER_NOT_INITIALIZED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_QUEUE_REJECTED, ErrorCode::P2P_CONNECTOR_TRANSFER_QUEUE_REJECTED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_COPY_FAILED, ErrorCode::P2P_CONNECTOR_TRANSFER_COPY_FAILED},
    {ErrorCodePB::P2P_CONNECTOR_TRANSFER_CONTEXT_DESTROYED, ErrorCode::P2P_CONNECTOR_TRANSFER_CONTEXT_DESTROYED},
    {ErrorCodePB::LONG_PROMPT_ERROR, ErrorCode::LONG_PROMPT_ERROR},
    {ErrorCodePB::CONCURRENCY_LIMIT_ERROR, ErrorCode::CONCURRENCY_LIMIT_ERROR},
    {ErrorCodePB::PRIORITY_PREEMPTED, ErrorCode::PRIORITY_PREEMPTED},
    {ErrorCodePB::MALLOC_FAILED, ErrorCode::MALLOC_FAILED},
    {ErrorCodePB::ERROR_GENERATE_CONFIG_FORMAT, ErrorCode::ERROR_GENERATE_CONFIG_FORMAT},
    {ErrorCodePB::EXCEEDS_KV_CACHE_MAX_LEN, ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN},
    {ErrorCodePB::MM_LONG_PROMPT_ERROR, ErrorCode::MM_LONG_PROMPT_ERROR},
    {ErrorCodePB::MM_WRONG_FORMAT_ERROR, ErrorCode::MM_WRONG_FORMAT_ERROR},
    {ErrorCodePB::MM_PROCESS_ERROR, ErrorCode::MM_PROCESS_ERROR},
    {ErrorCodePB::MM_EMPTY_ENGINE_ERROR, ErrorCode::MM_EMPTY_ENGINE_ERROR},
    {ErrorCodePB::MM_NOT_SUPPORTED_ERROR, ErrorCode::MM_NOT_SUPPORTED_ERROR},
    {ErrorCodePB::MM_DOWNLOAD_FAILED, ErrorCode::MM_DOWNLOAD_FAILED},
    {ErrorCodePB::OUT_OF_VOCAB_RANGE, ErrorCode::OUT_OF_VOCAB_RANGE},
    {ErrorCodePB::OUTPUT_QUEUE_FULL, ErrorCode::OUTPUT_QUEUE_FULL},
    {ErrorCodePB::OUTPUT_QUEUE_IS_EMPTY, ErrorCode::OUTPUT_QUEUE_IS_EMPTY},
    {ErrorCodePB::FINISHED, ErrorCode::FINISHED},
    {ErrorCodePB::GET_HOST_FAILED, ErrorCode::GET_HOST_FAILED},
    {ErrorCodePB::CONNECT_FAILED, ErrorCode::CONNECT_FAILED},
    {ErrorCodePB::CONNECT_TIMEOUT, ErrorCode::CONNECT_TIMEOUT},
    {ErrorCodePB::DEADLINE_EXCEEDED, ErrorCode::DEADLINE_EXCEEDED},
    {ErrorCodePB::CONNECTION_RESET_BY_PEER, ErrorCode::CONNECTION_RESET_BY_PEER},
    {ErrorCodePB::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED, ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED},
    {ErrorCodePB::REMOTE_ALLOCATE_RESOURCE_READ_FAILED, ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED},
    {ErrorCodePB::REMOTE_LOAD_KV_CACHE_FAILED, ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED},
    {ErrorCodePB::REMOTE_GENERATE_FAILED, ErrorCode::REMOTE_GENERATE_FAILED},
    {ErrorCodePB::DECODE_MALLOC_FAILED, ErrorCode::DECODE_MALLOC_FAILED},
    {ErrorCodePB::LOAD_KV_CACHE_FAILED, ErrorCode::LOAD_KV_CACHE_FAILED},
    {ErrorCodePB::WAIT_TO_RUN_TIMEOUT, ErrorCode::WAIT_TO_RUN_TIMEOUT},
    {ErrorCodePB::KEEP_ALIVE_TIMEOUT, ErrorCode::KEEP_ALIVE_TIMEOUT},
    {ErrorCodePB::CACHE_STORE_PUSH_ITEM_FAILED, ErrorCode::CACHE_STORE_PUSH_ITEM_FAILED},
    {ErrorCodePB::CACHE_STORE_LOAD_UNKNOWN_ERROR, ErrorCode::CACHE_STORE_LOAD_UNKNOWN_ERROR},
    {ErrorCodePB::CACHE_STORE_STORE_FAILED, ErrorCode::CACHE_STORE_STORE_FAILED},
    {ErrorCodePB::GET_PART_NODE_STATUS_FAILED, ErrorCode::GET_PART_NODE_STATUS_FAILED},
    {ErrorCodePB::GET_ALL_NODE_STATUS_FAILED, ErrorCode::GET_ALL_NODE_STATUS_FAILED},
};

}  // namespace rpc_error_detail

inline ErrorCode transRPCErrorCode(ErrorCodePB error_code) {
    static const auto error_code_map = [] {
        std::unordered_map<ErrorCodePB, ErrorCode> map;
        for (const auto& [rpc_code, local_code] : rpc_error_detail::kErrorCodePairs) {
            map.emplace(rpc_code, local_code);
        }
        return map;
    }();
    const auto it = error_code_map.find(error_code);
    return it != error_code_map.end() ? it->second : ErrorCode::UNKNOWN_ERROR;
}

inline ErrorCodePB transErrorCodeToRPC(ErrorCode error_code) {
    static const auto error_code_map = [] {
        std::unordered_map<ErrorCode, ErrorCodePB> map;
        for (const auto& [rpc_code, local_code] : rpc_error_detail::kErrorCodePairs) {
            map.emplace(local_code, rpc_code);
        }
        return map;
    }();
    const auto it = error_code_map.find(error_code);
    return it != error_code_map.end() ? it->second : ErrorCodePB::UNKNOWN_ERROR;
}

// Preserve the application code/message when present; transport status is the
// first observable cause only when the peer supplied no structured error.
inline ErrorInfo errorInfoFromGrpcStatus(const grpc::Status& status, const std::string& location = "") {
    if (status.ok())
        return ErrorInfo::OkStatus();
    ErrorDetailsPB details;
    if (!status.error_details().empty() && details.ParseFromString(status.error_details())
        && details.error_code() != 0) {
        const std::string message = details.error_message().empty() ? status.error_message() : details.error_message();
        return ErrorInfo(static_cast<ErrorCode>(details.error_code()),
                         location.empty() ? message :
                                            message + " [received_at " + location + " grpc_code="
                                                + std::to_string(static_cast<int>(status.error_code())) + "]");
    }
    return ErrorInfo(transGrpcStatusToErrorCode(status.error_code()),
                     (location.empty() ? "grpc_code=" : location + " grpc_code=")
                         + std::to_string(static_cast<int>(status.error_code())) + ": "
                         + status.error_message());
}

inline grpc::Status grpcStatusFromErrorInfo(const ErrorInfo& error) {
    if (error.ok())
        return grpc::Status::OK;
    ErrorDetailsPB details;
    details.set_error_code(static_cast<int>(error.code()));
    details.set_error_message(error.ToString());
    return grpc::Status(transErrorCodeToGrpc(error.code()), error.ToString(), details.SerializeAsString());
}
}  // namespace rtp_llm
