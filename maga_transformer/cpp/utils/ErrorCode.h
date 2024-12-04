#pragma once

namespace rtp_llm {

enum class ErrorCode {
    NONE_ERROR = 0,
    LONG_PROMPT_ERROR = 511,
    EXCEEDS_KV_CACHE_MAX_LEN = 513,
    UNKNOWN_ERROR = 514,
    MALLOC_FAILED = 602,
    GENERATE_TIMEOUT = 603,
    CANCELLED = 604,
    OUT_OF_VOCAB_RANGE = 605,
    OUTPUT_QUEUE_FULL = 606,
    OUTPUT_QUEUE_IS_EMPTY = 607,
    FINISHED = 608,

    // rpc error
    GET_HOST_FAILED = 704,
    GET_CONNECTION_FAILED = 705,
    CONNECT_FAILED = 706,
    CONNECTION_RESET_BY_PEER = 707,
    REMOTE_ALLOCATE_RESOURCE_FAILED = 708,
    REMOTE_LOAD_KV_CACHE_FAILED = 709,
    REMOTE_GENERATE_FAILED = 710,
    RPC_FINISH_FAILED = 711,
    DECODE_MALLOC_FAILED = 712,
    LOAD_KV_CACHE_FAILED = 713,

    // load cache error
    LOAD_CACHE_TIMEOUT = 810,
    CACHE_STORE_LOAD_CONNECT_FAILED = 811,
    CACHE_STORE_LOAD_SEND_REQUEST_FAILED = 812,
    CACHE_STORE_CALL_PREFILL_TIMEOUT = 813,
    CACHE_STORE_LOAD_RDMA_CONNECT_FAILED = 814,
    CACHE_STORE_LOAD_RDMA_WRITE_FAILED = 815,
    CACHE_STORE_LOAD_BUFFER_TIMEOUT = 816,

    // multimodal error
    MM_LONG_PROMPT_ERROR = 901,
    MM_WRONG_FORMAT_ERROR = 902,
    MM_PROCESS_ERROR = 903,
    MM_EMPTY_ENGINE_ERROR = 904,
    MM_NOT_SUPPORTED_ERROR = 905,
};

inline std::string ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::NONE_ERROR:
            return "NONE_ERROR";
        case ErrorCode::LONG_PROMPT_ERROR:
            return "LONG_PROMPT_ERROR";
        case ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN:
            return "EXCEEDS_KV_CACHE_MAX_LEN";
        case ErrorCode::UNKNOWN_ERROR:
            return "UNKNOWN_ERROR";
        case ErrorCode::MALLOC_FAILED:
            return "MALLOC_FAILED";
        case ErrorCode::GENERATE_TIMEOUT:
            return "GENERATE_TIMEOUT";
        case ErrorCode::CANCELLED:
            return "CANCELLED";
        case ErrorCode::GET_HOST_FAILED:
            return "GET_HOST_FAILED";
        case ErrorCode::GET_CONNECTION_FAILED:
            return "GET_CONNECTION_FAILED";
        case ErrorCode::CONNECT_FAILED:
            return "CONNECT_FAILED";
        case ErrorCode::CONNECTION_RESET_BY_PEER:
            return "CONNECTION_RESET_BY_PEER";
        case ErrorCode::REMOTE_ALLOCATE_RESOURCE_FAILED:
            return "REMOTE_ALLOCATE_RESOURCE_FAILED";
        case ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED:
            return "REMOTE_LOAD_KV_CACHE_FAILED";
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
        default:
            return "Error: Unrecognized ErrorCode";
    }
}

class ErrorInfo {
public:
    ErrorInfo() {}
    ErrorInfo(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    ErrorInfo(const ErrorInfo& other)
        : code_(other.code_), message_(other.message_) {}
    ErrorInfo& operator=(const ErrorInfo& other) {
        if (this != &other) {
            code_ = other.code_;
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

private:
    ErrorCode code_ = ErrorCode::NONE_ERROR;
    std::string message_;
};

template<typename T>
class ErrorResult {
public:
    ErrorResult(ErrorCode code, std::string message)
        : status_(code, message), value_(value) {}

    ErrorResult(ErrorInfo status)
        : status_(status) {}

    ErrorResult(T&& value)
        : value_(std::move(value)) {}

    bool ok() const {
        return status_.ok();
    }

    const ErrorInfo& status() const {
        return status_;
    }

    const T& value() const {
        return value_;
    }

private:
    ErrorInfo status_;
    T value_;
};

}
