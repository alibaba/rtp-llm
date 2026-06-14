#include "rtp_llm/cpp/api_server/Exception.h"

namespace rtp_llm {

HttpApiServerException::Type transErrorCodeToHttpExceptionType(ErrorCode code) {
    switch (code) {
        case ErrorCode::LONG_PROMPT_ERROR:
        case ErrorCode::MM_LONG_PROMPT_ERROR:
        case ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN:
            return HttpApiServerException::LONG_PROMPT_ERROR;
        case ErrorCode::MALLOC_FAILED:
        case ErrorCode::DECODE_MALLOC_FAILED:
            return HttpApiServerException::MALLOC_ERROR;
        case ErrorCode::GENERATE_TIMEOUT:
        case ErrorCode::WAIT_TO_RUN_TIMEOUT:
        case ErrorCode::LOAD_CACHE_TIMEOUT:
            return HttpApiServerException::GENERATE_TIMEOUT_ERROR;
        case ErrorCode::CANCELLED:
            return HttpApiServerException::CANCELLED_ERROR;
        case ErrorCode::GET_HOST_FAILED:
            return HttpApiServerException::GET_HOST_ERROR;
        case ErrorCode::GET_CONNECTION_FAILED:
            return HttpApiServerException::GET_CONNECTION_ERROR;
        case ErrorCode::CONNECT_FAILED:
        case ErrorCode::CONNECT_TIMEOUT:
        case ErrorCode::DEADLINE_EXCEEDED:
        case ErrorCode::KEEP_ALIVE_TIMEOUT:
        case ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED:
        case ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED:
            return HttpApiServerException::CONNECT_ERROR;
        case ErrorCode::CONNECTION_RESET_BY_PEER:
            return HttpApiServerException::CONNECTION_RESET_BY_PEER_ERROR;
        case ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED:
        case ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED:
            return HttpApiServerException::REMOTE_ALLOCATE_RESOURCE_ERROR;
        case ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED:
        case ErrorCode::LOAD_KV_CACHE_FAILED:
        case ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED:
        case ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT:
        case ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED:
        case ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT:
        case ErrorCode::CACHE_STORE_LOAD_UNKNOWN_ERROR:
        case ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED:
        case ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED:
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED:
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED:
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED:
        case ErrorCode::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED:
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT:
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED:
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH:
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT:
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE:
            return HttpApiServerException::REMOTE_LOAD_KV_CACHE_ERROR;
        case ErrorCode::REMOTE_GENERATE_FAILED:
        case ErrorCode::RPC_FINISH_FAILED:
            return HttpApiServerException::REMOTE_GENERATE_ERROR;
        default:
            return HttpApiServerException::UNKNOWN_ERROR;
    }
}

}  // namespace rtp_llm
