#pragma once

namespace rtp_llm {

enum class ErrorCode {
    NONE = 0,
    UNKNOWN = 514,
    MALLOC_FAILED = 602,
    GENERATE_TIMEOUT = 603,
    GET_HOST_FAILED = 604,
    GET_CONNECTION_FAILED = 605,
    CONNECT_FAILED = 606,
    CONNECTION_RESET_BY_PEER = 607,
    REMOTE_ALLOCATE_RESOURCE_FAILED = 608,
    REMOTE_LOAD_KV_CACHE_FAILED = 609,
    REMOTE_GENERATE_FAILED = 610,
};

inline std::string toString(ErrorCode code) {
    switch (code) {
        case ErrorCode::NONE:
            return "NONE";
        case ErrorCode::UNKNOWN:
            return "UNKNOWN";
        case ErrorCode::MALLOC_FAILED:
            return "MALLOC_FAILED";
        case ErrorCode::GENERATE_TIMEOUT:
            return "GENERATE_TIMEOUT";
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
        default:
            return "Error: Unrecognized ErrorCode";
    }
}

// TODO: not use absl::status
inline ErrorCode transErrorCode(absl::StatusCode code) {
    const static std::unordered_map<absl::StatusCode, ErrorCode> error_code_map = {
        {absl::StatusCode::kResourceExhausted, ErrorCode::MALLOC_FAILED},
        {absl::StatusCode::kDeadlineExceeded, ErrorCode::GENERATE_TIMEOUT},
        {absl::StatusCode::kInternal, ErrorCode::UNKNOWN}
    };
    auto it = error_code_map.find(code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return ErrorCode::UNKNOWN;
    }
}

}
