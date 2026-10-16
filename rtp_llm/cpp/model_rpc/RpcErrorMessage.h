#pragma once

#include <algorithm>
#include <string>

#include "google/protobuf/stubs/common.h"

namespace rtp_llm {

// gRPC error_details is arbitrary bytes, NOT a protobuf string. Bound diagnostic
// expansion so a failed RPC cannot produce an unbounded status/history message.
inline std::string rpcErrorDetailsHex(const std::string& bytes) {
    constexpr size_t kMaxBytes = 256;
    constexpr char   kHex[]    = "0123456789abcdef";
    std::string      result;
    for (size_t i = 0; i < std::min(bytes.size(), kMaxBytes); ++i) {
        const auto byte = static_cast<unsigned char>(bytes[i]);
        result += kHex[byte >> 4];
        result += kHex[byte & 15];
    }
    if (bytes.size() > kMaxBytes) {
        result += "...[truncated,total_bytes=" + std::to_string(bytes.size()) + "]";
    }
    return result;
}

// Also sanitize on OUTPUT: old finished-task history may already contain bytes
// produced before this fix. Preserve task identity/code/completion and sanitize
// only its diagnostic text. Never let truncation split a UTF-8 code point.
inline std::string safeRpcErrorMessage(const std::string& message, size_t max_bytes = 4096) {
    auto text = message.substr(0, max_bytes);
    for (int trim = 0; trim < 4; ++trim) {
        if (google::protobuf::internal::IsStructurallyValidUTF8(text)) {
            return message.size() > max_bytes ? text + "...[truncated]" : text;
        }
        if (message.size() <= max_bytes || text.empty()) {
            break;
        }
        text.pop_back();
    }
    return "[invalid UTF-8; hex=" + rpcErrorDetailsHex(message) + "]";
}

// A grpc::Status carries the text twice: percent-encoded grpc-message (up to
// 3x) and base64-encoded ErrorDetailsPB (about 4/3x). Keep both plus the other
// trailers comfortably below the default 8 KiB metadata limit. Protobuf body
// fields/history keep the larger safeRpcErrorMessage budget above.
inline std::string safeGrpcErrorMessage(const std::string& message) {
    return safeRpcErrorMessage(message, 1024);
}

template<typename ErrorMessagePB>
inline void setSafeRpcErrorMessage(ErrorMessagePB* error, const std::string& message) {
    error->set_error_message(safeRpcErrorMessage(message));
}

}  // namespace rtp_llm
