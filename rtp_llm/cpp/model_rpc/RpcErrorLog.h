#pragma once

#include <cstring>
#include <string>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Error/debug text can come from a peer. Bound it, remove known credentials,
// and keep each RPC error on a single engine-log line.
inline std::string sanitizeRpcError(const char* message) {
    std::string text(message ? message : "", message ? strnlen(message, 4096) : 0);
    for (const char* key : {"OSS_ACCESS_ID=", "OSS_ACCESS_KEY=", "Bearer ", "bearer ", "sk-"}) {
        size_t pos = 0;
        while ((pos = text.find(key, pos)) != std::string::npos) {
            const size_t begin = pos + std::strlen(key);
            size_t end = text.find_first_of("& \t\r\n\"'\\,;}", begin);
            if (end == std::string::npos) end = text.size();
            text.replace(begin, end - begin, "<redacted>");
            pos = begin + std::strlen("<redacted>");
        }
    }
    for (char& c : text) {
        if (static_cast<unsigned char>(c) < 32 || c == 127) c = ' ';
    }
    return text;
}

// Call only after Finish completion (or a synchronous call returning). Like
// Frontend's RpcError.details(), these are returned RPC details, not trace logs.
inline void logKvRpcFailure(const std::string& request_key, const std::string& attempt,
                            size_t worker_index, const grpc::Status& status,
                            const grpc::ClientContext& context) noexcept {
    if (status.ok()) return;
    try {
        RTP_LLM_LOG_INFO("[KV_RPC] RPC_FAILED request=[%s] attempt=%s worker_index=%zu grpc_code=%d "
                         "status=[%.2048s] grpc_debug=[%.4096s]",
                         request_key.c_str(), attempt.c_str(), worker_index, status.error_code(),
                         sanitizeRpcError(status.error_message().c_str()).c_str(),
                         sanitizeRpcError(context.debug_error_string().c_str()).c_str());
        // Keep the final diagnostic visible before the caller may abort.
        Logger::getEngineLogger().flush();
    } catch (...) {
        // Diagnostics must not change cancellation, cleanup or abort behavior.
    }
}

}  // namespace rtp_llm
