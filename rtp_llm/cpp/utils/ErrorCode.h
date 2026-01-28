#pragma once

namespace rtp_llm {

enum class ErrorCode {
    NONE_ERROR                   = 0,
    LONG_PROMPT_ERROR            = 511,
    UNKNOWN_ERROR                = 514,
    MALLOC_FAILED                = 602,
    GENERATE_TIMEOUT             = 603,
    ERROR_GENERATE_CONFIG_FORMAT = 604,
    INVALID_PARAMS               = 605,
    EXECUTION_EXCEPTION          = 606,
    EXCEEDS_KV_CACHE_MAX_LEN     = 607,

    // multimodal error
    MM_LONG_PROMPT_ERROR   = 901,
    MM_WRONG_FORMAT_ERROR  = 902,
    MM_PROCESS_ERROR       = 903,
    MM_EMPTY_ENGINE_ERROR  = 904,
    MM_NOT_SUPPORTED_ERROR = 905,
    MM_DOWNLOAD_FAILED     = 906,

    // Error codes starting from 8000 can be retried
    CANCELLED             = 8100,
    OUT_OF_VOCAB_RANGE    = 8101,
    OUTPUT_QUEUE_FULL     = 8102,
    OUTPUT_QUEUE_IS_EMPTY = 8103,
    FINISHED              = 8104,

    // rpc error
    GET_HOST_FAILED                       = 8200,
    GET_CONNECTION_FAILED                 = 8201,
    CONNECT_FAILED                        = 8202,
    CONNECT_TIMEOUT                       = 8203,
    DEADLINE_EXCEEDED                     = 8204,
    CONNECTION_RESET_BY_PEER              = 8205,
    REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED = 8206,
    REMOTE_ALLOCATE_RESOURCE_READ_FAILED  = 8207,
    REMOTE_LOAD_KV_CACHE_FAILED           = 8208,
    REMOTE_GENERATE_FAILED                = 8209,
    RPC_FINISH_FAILED                     = 8210,
    DECODE_MALLOC_FAILED                  = 8211,
    LOAD_KV_CACHE_FAILED                  = 8212,
    WAIT_TO_RUN_TIMEOUT                   = 8213,
    KEEP_ALIVE_TIMEOUT                    = 8214,

    // cache store error
    LOAD_CACHE_TIMEOUT                   = 8300,
    CACHE_STORE_PUSH_ITEM_FAILED         = 8301,
    CACHE_STORE_LOAD_CONNECT_FAILED      = 8302,
    CACHE_STORE_LOAD_SEND_REQUEST_FAILED = 8303,
    CACHE_STORE_CALL_PREFILL_TIMEOUT     = 8304,
    CACHE_STORE_LOAD_RDMA_CONNECT_FAILED = 8305,
    CACHE_STORE_LOAD_RDMA_WRITE_FAILED   = 8306,
    CACHE_STORE_LOAD_BUFFER_TIMEOUT      = 8307,
    CACHE_STORE_LOAD_UNKNOWN_ERROR       = 8308,
    CACHE_STORE_STORE_FAILED             = 8309,

    // p2p connector error
    P2P_CONNECTOR_CALL_PREFILL_FAILED                 = 8310,
    P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED            = 8311,
    P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED        = 8312,
    P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED    = 8313,
    P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED      = 8314,
    P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED         = 8315,
    P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT          = 8316,
    P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED        = 8317,
    P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED  = 8318,
    P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED    = 8319,
    P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH         = 8320,
    P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT = 8321,
    P2P_CONNECTOR_WORKER_READ_FAILED                  = 8322,
    P2P_CONNECTOR_WORKER_READ_CANCELLED               = 8323,
    P2P_CONNECTOR_WORKER_READ_TIMEOUT                 = 8324,

    // load balance error
    GET_PART_NODE_STATUS_FAILED = 8400,
    GET_ALL_NODE_STATUS_FAILED  = 8401,
};

inline std::string ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::NONE_ERROR:
            return "NONE_ERROR";
        case ErrorCode::LONG_PROMPT_ERROR:
            return "LONG_PROMPT_ERROR";
        case ErrorCode::UNKNOWN_ERROR:
            return "UNKNOWN_ERROR";
        case ErrorCode::MALLOC_FAILED:
            return "MALLOC_FAILED";
        case ErrorCode::GENERATE_TIMEOUT:
            return "GENERATE_TIMEOUT";
        case ErrorCode::ERROR_GENERATE_CONFIG_FORMAT:
            return "ERROR_GENERATE_CONFIG_FORMAT";
        case ErrorCode::INVALID_PARAMS:
            return "INVALID_PARAMS";
        case ErrorCode::EXECUTION_EXCEPTION:
            return "EXECUTION_EXCEPTION";
        case ErrorCode::CANCELLED:
            return "CANCELLED";
        case ErrorCode::OUT_OF_VOCAB_RANGE:
            return "OUT_OF_VOCAB_RANGE";
        case ErrorCode::OUTPUT_QUEUE_FULL:
            return "OUTPUT_QUEUE_FULL";
        case ErrorCode::OUTPUT_QUEUE_IS_EMPTY:
            return "OUTPUT_QUEUE_IS_EMPTY";
        case ErrorCode::FINISHED:
            return "FINISHED";
        case ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN:
            return "EXCEEDS_KV_CACHE_MAX_LEN";
        case ErrorCode::GET_HOST_FAILED:
            return "GET_HOST_FAILED";
        case ErrorCode::GET_CONNECTION_FAILED:
            return "GET_CONNECTION_FAILED";
        case ErrorCode::CONNECT_FAILED:
            return "CONNECT_FAILED";
        case ErrorCode::CONNECT_TIMEOUT:
            return "CONNECT_TIMEOUT";
        case ErrorCode::DEADLINE_EXCEEDED:
            return "DEADLINE_EXCEEDED";
        case ErrorCode::CONNECTION_RESET_BY_PEER:
            return "CONNECTION_RESET_BY_PEER";
        case ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED:
            return "REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED";
        case ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED:
            return "REMOTE_ALLOCATE_RESOURCE_READ_FAILED";
        case ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED:
            return "REMOTE_LOAD_KV_CACHE_FAILED";
        case ErrorCode::WAIT_TO_RUN_TIMEOUT:
            return "WAIT_TO_RUN_TIMEOUT";
        case ErrorCode::KEEP_ALIVE_TIMEOUT:
            return "KEEP_ALIVE_TIMEOUT";
        case ErrorCode::REMOTE_GENERATE_FAILED:
            return "REMOTE_GENERATE_FAILED";
        case ErrorCode::RPC_FINISH_FAILED:
            return "RPC_FINISH_FAILED";
        case ErrorCode::DECODE_MALLOC_FAILED:
            return "DECODE_MALLOC_FAILED";
        case ErrorCode::LOAD_KV_CACHE_FAILED:
            return "LOAD_KV_CACHE_FAILED";
        case ErrorCode::LOAD_CACHE_TIMEOUT:
            return "LOAD_CACHE_TIMEOUT";
        case ErrorCode::CACHE_STORE_PUSH_ITEM_FAILED:
            return "CACHE_STORE_PUSH_ITEM_FAILED";
        case ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED:
            return "CACHE_STORE_LOAD_CONNECT_FAILED";
        case ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED:
            return "CACHE_STORE_LOAD_SEND_REQUEST_FAILED";
        case ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT:
            return "CACHE_STORE_CALL_PREFILL_TIMEOUT";
        case ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED:
            return "CACHE_STORE_LOAD_RDMA_CONNECT_FAILED";
        case ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED:
            return "CACHE_STORE_LOAD_RDMA_WRITE_FAILED";
        case ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT:
            return "CACHE_STORE_LOAD_BUFFER_TIMEOUT";
        case ErrorCode::CACHE_STORE_LOAD_UNKNOWN_ERROR:
            return "CACHE_STORE_LOAD_UNKNOWN_ERROR";
        case ErrorCode::CACHE_STORE_STORE_FAILED:
            return "CACHE_STORE_STORE_FAILED";
        case ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED:
            return "P2P_CONNECTOR_CALL_PREFILL_FAILED";
        case ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED:
            return "P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED";
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED:
            return "P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED";
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED:
            return "P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED";
        case ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED:
            return "P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED";
        case ErrorCode::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED:
            return "P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED";
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT:
            return "P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT";
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED:
            return "P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED";
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED:
            return "P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED:
            return "P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH:
            return "P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH";
        case ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT:
            return "P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED:
            return "P2P_CONNECTOR_WORKER_READ_FAILED";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED:
            return "P2P_CONNECTOR_WORKER_READ_CANCELLED";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT:
            return "P2P_CONNECTOR_WORKER_READ_TIMEOUT";
        case ErrorCode::MM_LONG_PROMPT_ERROR:
            return "MM_LONG_PROMPT_ERROR";
        case ErrorCode::MM_WRONG_FORMAT_ERROR:
            return "MM_WRONG_FORMAT_ERROR";
        case ErrorCode::MM_PROCESS_ERROR:
            return "MM_PROCESS_ERROR";
        case ErrorCode::MM_EMPTY_ENGINE_ERROR:
            return "MM_EMPTY_ENGINE_ERROR";
        case ErrorCode::MM_NOT_SUPPORTED_ERROR:
            return "MM_NOT_SUPPORTED_ERROR";
        case ErrorCode::GET_PART_NODE_STATUS_FAILED:
            return "GET_PART_NODE_STATUS_FAILED";
        case ErrorCode::GET_ALL_NODE_STATUS_FAILED:
            return "GET_ALL_NODE_STATUS_FAILED";
        default:
            return "Error: Unrecognized ErrorCode";
    }
}

class ErrorInfo {
public:
    ErrorInfo() {}
    ErrorInfo(ErrorCode code, const std::string& message): code_(code), message_(message) {}
    ErrorInfo(const ErrorInfo& other): code_(other.code_), message_(other.message_) {}
    ErrorInfo& operator=(const ErrorInfo& other) {
        if (this != &other) {
            code_    = other.code_;
            message_ = other.message_;
        }
        return *this;
    }

    const std::string& ToString() const {
        return message_;
    }

    static ErrorInfo OkStatus() {
        return ErrorInfo(ErrorCode::NONE_ERROR, "");
    }

    bool ok() const {
        return code_ == ErrorCode::NONE_ERROR;
    }

    bool hasError() const {
        return !ok();
    }

    ErrorCode code() const {
        return code_;
    }

    void setErrorCode(ErrorCode ec) {
        code_ = ec;
    }

private:
    ErrorCode   code_ = ErrorCode::NONE_ERROR;
    std::string message_;
};

template<typename T>
class ErrorResult {
public:
    ErrorResult(ErrorCode code, std::string message): status_(code, message) {}

    ErrorResult(ErrorInfo status): status_(status) {}

    ErrorResult(T&& value): value_(std::move(value)) {}

    bool ok() const {
        return status_.ok();
    }

    const ErrorInfo& status() const {
        return status_;
    }

    const T& value() const {
        return value_;
    }

    T& value() {
        return value_;
    }

    void setStatus(const ErrorInfo& new_status) {
        status_ = new_status;
    }

private:
    ErrorInfo status_;
    T         value_;
};

}  // namespace rtp_llm
