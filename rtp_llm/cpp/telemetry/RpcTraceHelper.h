#pragma once

#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_startoptions.h"

#include "rtp_llm/cpp/telemetry/GrpcTraceCarrier.h"
#include "rtp_llm/cpp/telemetry/RequestSpanGuard.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include "rtp_llm/cpp/telemetry/TraceAttributes.h"

namespace rtp_llm {
namespace telemetry {

// Low-cardinality gRPC status code name for error.type: enum names only, never
// raw error messages, which may carry endpoints or request data.
inline const char* grpcStatusCodeName(grpc::StatusCode code) noexcept {
    switch (code) {
        case grpc::StatusCode::OK:
            return "OK";
        case grpc::StatusCode::CANCELLED:
            return "Cancelled";
        case grpc::StatusCode::UNKNOWN:
            return "Unknown";
        case grpc::StatusCode::INVALID_ARGUMENT:
            return "InvalidArgument";
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return "DeadlineExceeded";
        case grpc::StatusCode::NOT_FOUND:
            return "NotFound";
        case grpc::StatusCode::ALREADY_EXISTS:
            return "AlreadyExists";
        case grpc::StatusCode::PERMISSION_DENIED:
            return "PermissionDenied";
        case grpc::StatusCode::UNAUTHENTICATED:
            return "Unauthenticated";
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return "ResourceExhausted";
        case grpc::StatusCode::FAILED_PRECONDITION:
            return "FailedPrecondition";
        case grpc::StatusCode::ABORTED:
            return "Aborted";
        case grpc::StatusCode::OUT_OF_RANGE:
            return "OutOfRange";
        case grpc::StatusCode::UNIMPLEMENTED:
            return "Unimplemented";
        case grpc::StatusCode::INTERNAL:
            return "Internal";
        case grpc::StatusCode::UNAVAILABLE:
            return "Unavailable";
        case grpc::StatusCode::DATA_LOSS:
            return "DataLoss";
        default:
            return "UnknownCode";
    }
}

// gRPC semconv requires the canonical enum spelling on
// rpc.response.status_code. Keep this separate from the established CamelCase
// error.type values above for platform compatibility.
inline const char* grpcStatusCodeValue(grpc::StatusCode code) noexcept {
    switch (code) {
        case grpc::StatusCode::OK:
            return "OK";
        case grpc::StatusCode::CANCELLED:
            return "CANCELLED";
        case grpc::StatusCode::UNKNOWN:
            return "UNKNOWN";
        case grpc::StatusCode::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return "DEADLINE_EXCEEDED";
        case grpc::StatusCode::NOT_FOUND:
            return "NOT_FOUND";
        case grpc::StatusCode::ALREADY_EXISTS:
            return "ALREADY_EXISTS";
        case grpc::StatusCode::PERMISSION_DENIED:
            return "PERMISSION_DENIED";
        case grpc::StatusCode::UNAUTHENTICATED:
            return "UNAUTHENTICATED";
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return "RESOURCE_EXHAUSTED";
        case grpc::StatusCode::FAILED_PRECONDITION:
            return "FAILED_PRECONDITION";
        case grpc::StatusCode::ABORTED:
            return "ABORTED";
        case grpc::StatusCode::OUT_OF_RANGE:
            return "OUT_OF_RANGE";
        case grpc::StatusCode::UNIMPLEMENTED:
            return "UNIMPLEMENTED";
        case grpc::StatusCode::INTERNAL:
            return "INTERNAL";
        case grpc::StatusCode::UNAVAILABLE:
            return "UNAVAILABLE";
        case grpc::StatusCode::DATA_LOSS:
            return "DATA_LOSS";
        default:
            return "UNKNOWN";
    }
}

// Predictable human-readable status descriptions for Trace UIs. These are
// deliberately derived from the closed gRPC status enum, never from
// grpc::Status::error_message(), which may contain endpoints, request data, or
// other high-cardinality runtime details.
inline const char* grpcStatusDescription(grpc::StatusCode code) noexcept {
    switch (code) {
        case grpc::StatusCode::CANCELLED:
            return "RPC request was cancelled";
        case grpc::StatusCode::UNKNOWN:
            return "RPC request failed for an unknown reason";
        case grpc::StatusCode::INVALID_ARGUMENT:
            return "RPC request contained invalid arguments";
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return "RPC deadline was exceeded";
        case grpc::StatusCode::NOT_FOUND:
            return "RPC resource was not found";
        case grpc::StatusCode::ALREADY_EXISTS:
            return "RPC resource already exists";
        case grpc::StatusCode::PERMISSION_DENIED:
            return "RPC permission was denied";
        case grpc::StatusCode::UNAUTHENTICATED:
            return "RPC request was not authenticated";
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return "RPC request exhausted available resources";
        case grpc::StatusCode::FAILED_PRECONDITION:
            return "RPC precondition failed";
        case grpc::StatusCode::ABORTED:
            return "RPC request was aborted";
        case grpc::StatusCode::OUT_OF_RANGE:
            return "RPC request was out of range";
        case grpc::StatusCode::UNIMPLEMENTED:
            return "RPC operation is not implemented";
        case grpc::StatusCode::INTERNAL:
            return "RPC request failed because of an internal error";
        case grpc::StatusCode::UNAVAILABLE:
            return "RPC service was unavailable";
        case grpc::StatusCode::DATA_LOSS:
            return "RPC request failed because data was lost";
        case grpc::StatusCode::OK:
            return "";
        default:
            return "RPC request failed";
    }
}

// Starts a gRPC SERVER span with the remote parent extracted from server
// metadata (W3C traceparent). Returns an empty shared_ptr when telemetry is
// inactive so callers pay near-zero cost on the disabled path; never throws.
// NOTE: `return {}` (not `return nullptr`) — nostd::shared_ptr has no
// nullptr_t conversion under hip-clang (AMD CI build failure).
inline opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>
startRpcServerSpan(const std::string&         name,
                   const grpc::ServerContext* server_context,
                   bool                       pd_separation = false,
                   const std::string&         rpc_method    = {}) {
    if (!TelemetryRuntime::isActive()) {
        return {};
    }
    try {
        auto                                   parent_context = extractContextFromServerMetadata(server_context);
        opentelemetry::trace::StartSpanOptions options;
        options.parent = parent_context;
        options.kind   = opentelemetry::trace::SpanKind::kServer;
        // Initial attributes are visible to head samplers. Attributes added
        // after StartSpan are exporter-only and cannot influence sampling.
        using SpanAttribute = std::pair<opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue>;
        std::vector<SpanAttribute> attributes;
        attributes.reserve(3);
        attributes.emplace_back(kAttrRpcSystem, opentelemetry::nostd::string_view(kValRpcSystemGrpc));
        if (!rpc_method.empty()) {
            attributes.emplace_back(kAttrRpcMethod, opentelemetry::nostd::string_view(rpc_method));
        }
        if (pd_separation) {
            attributes.emplace_back(kAttrRtpLlmPdSep, true);
        }
        return TelemetryRuntime::tracer()->StartSpan(name, attributes, options);
    } catch (...) {
        return {};
    }
}

inline int64_t retryAttemptFromExecutionCount(int64_t execution_count) noexcept {
    return execution_count > 0 ? execution_count - 1 : 0;
}

// Starts a gRPC CLIENT span as child of the given local span (e.g. the
// Prefill SERVER span for the P->D RemoteGenerate call). Returns an empty
// shared_ptr when telemetry is inactive or the parent is null; never throws.
inline opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>
startChildClientSpan(const std::string&                                                  name,
                     const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                     const std::string&                                                  server_address = {},
                     int64_t                                                             server_port    = 0,
                     int64_t                                                             request_id     = -1,
                     int64_t                                                             retry_attempt  = 0,
                     const std::string&                                                  rpc_method     = {}) {
    if (!TelemetryRuntime::isActive() || parent_span == nullptr) {
        return {};
    }
    try {
        opentelemetry::trace::StartSpanOptions options;
        options.parent = parent_span->GetContext();
        options.kind   = opentelemetry::trace::SpanKind::kClient;

        std::string canonical_address = server_address;
        bool        endpoint_valid    = !canonical_address.empty() && canonical_address.find("://") == std::string::npos
                              && server_port >= 1 && server_port <= 65535;
        if (!canonical_address.empty()) {
            const bool has_left_bracket  = canonical_address.front() == '[';
            const bool has_right_bracket = canonical_address.back() == ']';
            if (has_left_bracket || has_right_bracket) {
                if (has_left_bracket && has_right_bracket && canonical_address.size() > 2) {
                    canonical_address = canonical_address.substr(1, canonical_address.size() - 2);
                } else {
                    endpoint_valid = false;
                }
            }
        }
        endpoint_valid = endpoint_valid && !canonical_address.empty();

        using SpanAttribute = std::pair<opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue>;
        std::vector<SpanAttribute> attributes;
        attributes.reserve(7);
        attributes.emplace_back(kAttrRpcSystem, opentelemetry::nostd::string_view(kValRpcSystemGrpc));
        if (!rpc_method.empty()) {
            attributes.emplace_back(kAttrRpcMethod, opentelemetry::nostd::string_view(rpc_method));
        }
        std::string request_id_string;
        if (request_id >= 0) {
            request_id_string = std::to_string(request_id);
            attributes.emplace_back(kAttrRequestId, opentelemetry::nostd::string_view(request_id_string));
            attributes.emplace_back(kAttrRtpLlmRequestId, request_id);
            attributes.emplace_back(kAttrRtpLlmRetryAttempt, retry_attempt);
        }
        if (endpoint_valid) {
            attributes.emplace_back(kAttrServerAddress, opentelemetry::nostd::string_view(canonical_address));
            attributes.emplace_back(kAttrServerPort, server_port);
        }
        return TelemetryRuntime::tracer()->StartSpan(name, attributes, options);
    } catch (...) {
        return {};
    }
}

// Injects the span's context into gRPC client metadata. Must run after the
// ClientContext is (re)created and before the RPC is initiated, which covers
// every retry re-creation; never throws.
inline void injectSpanToClientContext(grpc::ClientContext* client_context,
                                      const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span) {
    if (client_context == nullptr || span == nullptr) {
        return;
    }
    try {
        opentelemetry::context::Context base_context{};
        auto                            context_with_span = opentelemetry::trace::SetSpan(
            base_context, const_cast<opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>&>(span));
        injectContextToClientMetadata(client_context, context_with_span);
    } catch (...) {}
}

// Writes the gen_ai.usage token attributes on a per-role gRPC SERVER span
// (per-hop token breakdown beyond the frontend HTTP span). Five-key
// double-write follows the cross-language usage contract: current semconv
// input/output, legacy prompt/completion aliases (some platform views only read
// those older names) and total_tokens (the LLM view aggregates it). Values <= 0
// are skipped: partial usage is worse than none for platform aggregation.
template<typename SpanLike>
inline void setUsageTokenAttributes(SpanLike& span_like, int64_t input_tokens, int64_t output_tokens) noexcept {
    try {
        if (input_tokens <= 0 || output_tokens <= 0) {
            return;
        }
        span_like.setAttribute(kAttrGenAiUsageInputTokens, input_tokens);
        span_like.setAttribute(kAttrGenAiUsageOutputTokens, output_tokens);
        span_like.setAttribute(kAttrGenAiUsagePromptTokens, input_tokens);
        span_like.setAttribute(kAttrGenAiUsageCompletionTokens, output_tokens);
        span_like.setAttribute(kAttrGenAiUsageTotalTokens, input_tokens + output_tokens);
    } catch (...) {}
}

// RAII span finisher bound to a grpc::Status owned by the enclosing handler.
// On destruction it maps the final RPC status to span status + low-cardinality
// error.type, then ends the span exactly once. MUST be declared AFTER the
// status owner (e.g. GenerateContext) so it destructs first and the pointed-to
// status is still alive. Covers CHECK_ERROR_STATUS / EXECUTE_STAGE_FUNC early
// returns and exceptions via stack unwinding.
class GrpcStatusSpanGuard {
public:
    GrpcStatusSpanGuard(opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span,
                        const grpc::Status*                                          final_status):
        guard_(std::move(span)), final_status_(final_status), uncaught_exceptions_(std::uncaught_exceptions()) {}

    GrpcStatusSpanGuard(const GrpcStatusSpanGuard&)            = delete;
    GrpcStatusSpanGuard& operator=(const GrpcStatusSpanGuard&) = delete;

    ~GrpcStatusSpanGuard() noexcept {
        finish();
    }

    bool valid() const {
        return guard_.valid();
    }

    void setAttribute(opentelemetry::nostd::string_view            key,
                      const opentelemetry::common::AttributeValue& value) noexcept {
        guard_.setAttribute(key, value);
    }

    void addEvent(opentelemetry::nostd::string_view name, int64_t epoch_us) noexcept {
        guard_.addEvent(name, epoch_us);
    }

    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> sharedSpan() const {
        return guard_.sharedSpan();
    }

    void finish() noexcept {
        try {
            if (!guard_.valid()) {
                return;
            }
            if (final_status_ != nullptr && !final_status_->ok()) {
                guard_.setAttribute(kAttrRpcResponseStatusCode, grpcStatusCodeValue(final_status_->error_code()));
                guard_.setAttribute(kAttrErrorType, grpcStatusCodeName(final_status_->error_code()));
                guard_.setAttribute(kAttrRtpLlmGrpcStatusCode, (int64_t)final_status_->error_code());
                guard_.finish(opentelemetry::trace::StatusCode::kError,
                              grpcStatusDescription(final_status_->error_code()));
            } else if (std::uncaught_exceptions() > uncaught_exceptions_) {
                guard_.setAttribute(kAttrErrorType, "Exception");
                guard_.finish(opentelemetry::trace::StatusCode::kError, "RPC handler raised an exception");
            } else {
                guard_.setAttribute(kAttrRpcResponseStatusCode, grpcStatusCodeValue(grpc::StatusCode::OK));
                guard_.finish(opentelemetry::trace::StatusCode::kOk);
            }
        } catch (...) {}
    }

private:
    RequestSpanGuard    guard_;
    const grpc::Status* final_status_;
    int                 uncaught_exceptions_;
};

}  // namespace telemetry
}  // namespace rtp_llm
