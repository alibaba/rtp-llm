#pragma once

#include <string>

namespace rtp_llm {

// Request-level K3 PD trace control. A request whose trace_id starts with
// kK3PdTraceIdMarker (settable via the OpenAI "trace_id" field or the
// "trace_id" HTTP header) enables per-request reconciliation logging on the
// decode side. The decode side forwards the marker to the cache store by
// prefixing the store requestid with kK3PdTraceRequestIdPrefix, so the store
// side enables the same per-request logging without any env-var or redeploy
// dependency.
inline constexpr const char* kK3PdTraceIdMarker        = "k3_pd_trace";
inline constexpr const char* kK3PdTraceRequestIdPrefix = "k3_pd_trace:";

inline bool k3PdTraceEnabledForTraceId(const std::string& trace_id) {
    return trace_id.rfind(kK3PdTraceIdMarker, 0) == 0;
}

inline bool k3PdTraceMarkedRequestId(const std::string& requestid) {
    return requestid.rfind(kK3PdTraceRequestIdPrefix, 0) == 0;
}

inline std::string makeK3PdTraceRequestId(const std::string& request_id) {
    return std::string(kK3PdTraceRequestIdPrefix) + request_id;
}

}  // namespace rtp_llm
