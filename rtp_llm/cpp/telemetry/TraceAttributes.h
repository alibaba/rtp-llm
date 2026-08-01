#pragma once

// Centralized C++ span attribute registry, mirroring the Python-side schema in
// rtp_llm/telemetry/attributes.py; keep the two in sync. Grouping follows the
// same three-layer annotation: OTel official candidate / ARMS-Unitrace extension
// / rtp_llm.* internal. Resource attributes such as host.ip and service.name are
// process identity and are intentionally not listed here.

namespace rtp_llm {
namespace telemetry {

// Bailian Unitrace indexes the string request_id key for span search. The
// numeric rtp_llm.request_id twin remains available for internal correlation.
inline constexpr const char* kAttrRequestId       = "request_id";
inline constexpr const char* kAttrRtpLlmRequestId = "rtp_llm.request_id";

// OTel GenAI usage attributes plus the legacy aliases consumed by Unitrace.
inline constexpr const char* kAttrGenAiUsageInputTokens      = "gen_ai.usage.input_tokens";
inline constexpr const char* kAttrGenAiUsageOutputTokens     = "gen_ai.usage.output_tokens";
inline constexpr const char* kAttrGenAiUsagePromptTokens     = "gen_ai.usage.prompt_tokens";
inline constexpr const char* kAttrGenAiUsageCompletionTokens = "gen_ai.usage.completion_tokens";
inline constexpr const char* kAttrGenAiUsageTotalTokens      = "gen_ai.usage.total_tokens";

// RPC semantic convention attributes for real gRPC boundary spans. Keep the
// legacy rpc.system key for the published platform contract; migration to
// rpc.system.name requires an explicit duplicate-write/cutover, not a rename.
inline constexpr const char* kAttrRpcSystem             = "rpc.system";
inline constexpr const char* kAttrRpcMethod             = "rpc.method";
inline constexpr const char* kAttrRpcResponseStatusCode = "rpc.response.status_code";
inline constexpr const char* kValRpcSystemGrpc          = "grpc";

// OTel stable server endpoint attributes on CLIENT spans.
inline constexpr const char* kAttrServerAddress = "server.address";
inline constexpr const char* kAttrServerPort    = "server.port";

// Low-cardinality error/status attributes.
inline constexpr const char* kAttrErrorType            = "error.type";
inline constexpr const char* kAttrRtpLlmGrpcStatusCode = "rtp_llm.grpc_status_code";
// Stable application error identity on the operation that directly observed
// the failure. Never populate these from a raw ErrorInfo message.
inline constexpr const char* kAttrRtpLlmErrorCode   = "rtp_llm.error.code";
inline constexpr const char* kAttrRtpLlmErrorReason = "rtp_llm.error.reason";

// RTP-LLM internal span attributes.
inline constexpr const char* kAttrRtpLlmPdSep = "rtp_llm.pd_sep";
// Zero-based retries already performed before this physical RPC.
inline constexpr const char* kAttrRtpLlmRetryAttempt         = "rtp_llm.retry_attempt";
inline constexpr const char* kAttrRtpLlmAllocateRtUs         = "rtp_llm.allocate_rt_us";
inline constexpr const char* kAttrRtpLlmPollLocalOutputRtUs  = "rtp_llm.poll_local_output_rt_us";
inline constexpr const char* kAttrRtpLlmPollRemoteOutputRtUs = "rtp_llm.poll_remote_output_rt_us";
inline constexpr const char* kAttrRtpLlmPhaseTruncated       = "rtp_llm.phase.truncated";

}  // namespace telemetry
}  // namespace rtp_llm
