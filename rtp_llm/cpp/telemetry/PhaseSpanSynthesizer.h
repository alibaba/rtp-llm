#pragma once

#include <chrono>
#include <cstring>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <utility>

#include "opentelemetry/common/timestamp.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_metadata.h"
#include "opentelemetry/trace/span_startoptions.h"

#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include "rtp_llm/cpp/telemetry/TraceAttributes.h"

namespace rtp_llm {
namespace telemetry {

// One coherent GenerateStream progress snapshot plus the handler's absolute
// synthesis endpoint. All timestamps are epoch microseconds.
struct PhaseTiming {
    int64_t     begin_time_us           = 0;
    bool        running_started         = false;
    int64_t     running_started_time_us = 0;
    bool        first_token_committed   = false;
    int64_t     first_token_time_us     = 0;
    bool        generation_done         = false;
    int64_t     generation_done_time_us = 0;
    int64_t     synthesis_end_time_us   = 0;
    int64_t     request_id              = -1;  // <0 = unknown (attribute skipped)
    const char* error_type              = nullptr;
};

// Deployment role determines which child spans are synthesized:
//   fusion  -> wait + prefill + decode
//   prefill -> wait + prefill
//   decode  -> wait + decode (decode starts at scheduled, covers prefill+decode)
enum class PhaseRole {
    Fusion,
    Prefill,
    Decode,
};

namespace detail {

inline bool errorTypeIs(const char* error_type, const char* expected) noexcept {
    return error_type != nullptr && std::strcmp(error_type, expected) == 0;
}

// Phase descriptions are a closed vocabulary for human diagnosis in Trace
// UIs. Never interpolate raw ErrorInfo or transport messages here.
inline const char*
phaseErrorDescription(const std::string& name, const char* error_type, const char* error_reason) noexcept {
    if (name == "load_cache") {
        if (error_reason != nullptr && std::strcmp(error_reason, "CACHE_STORE_LOAD_BUFFER_TIMEOUT") == 0) {
            return "KV cache loading timed out while waiting for a buffer";
        }
        if (errorTypeIs(error_type, "Cancelled")) {
            return "KV cache loading was cancelled";
        }
        return "KV cache loading failed";
    }
    if (name == "wait") {
        if (errorTypeIs(error_type, "Cancelled")) {
            return "Request processing was cancelled while waiting for execution";
        }
        if (errorTypeIs(error_type, "DeadlineExceeded")) {
            return "Request deadline was exceeded while waiting for execution";
        }
        if (errorTypeIs(error_type, "DependencyFailure")) {
            return "Request stopped before execution because a dependency failed";
        }
        return "Request failed while waiting for execution";
    }
    if (name == "prefill") {
        if (errorTypeIs(error_type, "Cancelled")) {
            return "Prefill was cancelled";
        }
        if (errorTypeIs(error_type, "DeadlineExceeded")) {
            return "Prefill exceeded the request deadline";
        }
        return "Prefill failed";
    }
    if (name == "decode") {
        if (errorTypeIs(error_type, "Cancelled")) {
            return "Decode was cancelled";
        }
        if (errorTypeIs(error_type, "DeadlineExceeded")) {
            return "Decode exceeded the request deadline";
        }
        return "Decode failed";
    }
    return "Request phase failed";
}

// Creates a single INTERNAL child span under `parent_span` with explicit
// post-hoc timestamps. The span is created and immediately ended so it
// carries the correct absolute start time and duration without requiring
// real-time instrumentation in the scheduler loop.
//
// Technique: start_system_time places the span on the absolute timeline;
// start_steady_time + end_steady_time encode the duration. Both steady
// timestamps are synthetic (anchored at now()) since we only need the
// delta for duration computation.
//
// Never throws; returns silently on any failure (fail-open contract).
inline void synthesizeChildSpan(const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                                const std::string&                                                  name,
                                int64_t                                                             start_epoch_us,
                                int64_t                                                             duration_us,
                                int64_t                                                             request_id,
                                opentelemetry::trace::StatusCode status       = opentelemetry::trace::StatusCode::kOk,
                                bool                             truncated    = false,
                                const char*                      error_type   = nullptr,
                                int64_t                          error_code   = -1,
                                const char*                      error_reason = nullptr) {
    if (parent_span == nullptr || duration_us <= 0) {
        return;
    }
    try {
        auto tracer = TelemetryRuntime::tracer();

        opentelemetry::trace::StartSpanOptions options;
        options.parent = parent_span->GetContext();
        options.kind   = opentelemetry::trace::SpanKind::kInternal;
        // Absolute placement on the timeline (system clock, epoch-based).
        options.start_system_time = opentelemetry::common::SystemTimestamp(std::chrono::microseconds(start_epoch_us));
        // Synthetic steady anchor for duration computation.
        auto steady_now           = std::chrono::steady_clock::now();
        options.start_steady_time = opentelemetry::common::SteadyTimestamp(steady_now);

        auto span = tracer->StartSpan(name, options);
        if (span == nullptr) {
            return;
        }
        // The platform only indexes spans carrying a string `request_id`
        // attribute; without it these child spans stay unsearchable in Trace
        // Analysis (visible in the detail waterfall only).
        if (request_id >= 0) {
            span->SetAttribute(kAttrRequestId, std::to_string(request_id));
            span->SetAttribute(kAttrRtpLlmRequestId, request_id);
        }
        if (status == opentelemetry::trace::StatusCode::kError) {
            span->SetStatus(status, phaseErrorDescription(name, error_type, error_reason));
            if (error_type != nullptr && error_type[0] != '\0') {
                span->SetAttribute(kAttrErrorType, error_type);
            }
            if (error_code >= 0) {
                span->SetAttribute(kAttrRtpLlmErrorCode, error_code);
            }
            if (error_reason != nullptr && error_reason[0] != '\0') {
                span->SetAttribute(kAttrRtpLlmErrorReason, error_reason);
            }
        } else {
            span->SetStatus(status);
        }
        if (truncated) {
            span->SetAttribute(kAttrRtpLlmPhaseTruncated, true);
        }
        // End with explicit steady time so duration = duration_us exactly.
        opentelemetry::trace::EndSpanOptions end_options;
        end_options.end_steady_time =
            opentelemetry::common::SteadyTimestamp(steady_now + std::chrono::microseconds(duration_us));
        span->End(end_options);
    } catch (...) {
        // fail-open: telemetry must never break inference
    }
}

}  // namespace detail

// Synthesizes the decode-node `load_cache` INTERNAL child span covering
// [load_begin_us, load_end_us): the window where decode waits for the KV
// cache to arrive via cache_store. NOTE this window runs IN PARALLEL with the
// prefill computation — prefill sends the LOAD message right after leaving
// the WAITING state (PrefillRpcServer::remoteLoadCacheStart), so the span's
// tail beyond the prefill node's `prefill` child span is the pure transfer
// overhead. Unlike phase spans, callers invoke this on BOTH success and
// failure exits (KV load timeout is the classic PD failure mode and must stay
// visible on the waterfall); `ok=false` marks the span kError.
// Never throws (fail-open).
inline void synthesizeKvLoadSpan(const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                                 int64_t                                                             load_begin_us,
                                 int64_t                                                             load_end_us,
                                 int64_t                                                             request_id,
                                 bool                                                                ok,
                                 const char* error_type   = nullptr,
                                 int64_t     error_code   = -1,
                                 const char* error_reason = nullptr) {
    if (parent_span == nullptr || !TelemetryRuntime::isActive()) {
        return;
    }
    if (load_begin_us <= 0 || load_end_us <= load_begin_us) {
        return;
    }
    detail::synthesizeChildSpan(parent_span,
                                "load_cache",
                                load_begin_us,
                                load_end_us - load_begin_us,
                                request_id,
                                ok ? opentelemetry::trace::StatusCode::kOk : opentelemetry::trace::StatusCode::kError,
                                /*truncated=*/false,
                                error_type,
                                error_code,
                                error_reason);
}

// Synthesizes wait/prefill/decode INTERNAL child spans under the given
// SERVER parent span, using post-hoc timing data from the engine.
//
// Completed milestones define natural OK boundaries. On request failure, only
// the currently active phase is cut at synthesis_end and marked truncated;
// failures after GenerateDone never repaint completed child phases. Invalid or
// out-of-order boundaries are skipped instead of clamped into plausible spans.
// Decode intentionally ignores first_token_time because its remote first token
// may predate the reset local begin timestamp. Never throws (fail-open).
inline void synthesizePhaseSpans(const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                                 const PhaseTiming&                                                  timing,
                                 PhaseRole                                                           role,
                                 bool                                                                request_ok) {
    if (parent_span == nullptr || !TelemetryRuntime::isActive()) {
        return;
    }
    const int64_t begin = timing.begin_time_us;
    const int64_t end   = timing.synthesis_end_time_us;
    if (begin <= 0 || end <= begin) {
        return;
    }

    constexpr auto kError = opentelemetry::trace::StatusCode::kError;
    if (!timing.running_started) {
        if (!request_ok) {
            detail::synthesizeChildSpan(parent_span,
                                        "wait",
                                        begin,
                                        end - begin,
                                        timing.request_id,
                                        kError,
                                        /*truncated=*/true,
                                        timing.error_type);
        }
        return;
    }

    const int64_t running = timing.running_started_time_us;
    if (running < begin || running > end) {
        return;
    }
    if (running > begin) {
        detail::synthesizeChildSpan(parent_span, "wait", begin, running - begin, timing.request_id);
    }

    switch (role) {
        case PhaseRole::Fusion: {
            if (!timing.first_token_committed) {
                if (!timing.generation_done && !request_ok) {
                    detail::synthesizeChildSpan(parent_span,
                                                "prefill",
                                                running,
                                                end - running,
                                                timing.request_id,
                                                kError,
                                                /*truncated=*/true,
                                                timing.error_type);
                }
                break;
            }
            const int64_t first_token = timing.first_token_time_us;
            if (first_token < running || first_token > end) {
                break;
            }
            if (first_token > running) {
                detail::synthesizeChildSpan(parent_span, "prefill", running, first_token - running, timing.request_id);
            }
            if (timing.generation_done) {
                const int64_t done = timing.generation_done_time_us;
                if (done >= first_token && done <= end && done > first_token) {
                    detail::synthesizeChildSpan(
                        parent_span, "decode", first_token, done - first_token, timing.request_id);
                }
            } else if (!request_ok && end > first_token) {
                detail::synthesizeChildSpan(parent_span,
                                            "decode",
                                            first_token,
                                            end - first_token,
                                            timing.request_id,
                                            kError,
                                            /*truncated=*/true,
                                            timing.error_type);
            }
            break;
        }
        case PhaseRole::Prefill: {
            if (!timing.first_token_committed) {
                if (!timing.generation_done && !request_ok) {
                    detail::synthesizeChildSpan(parent_span,
                                                "prefill",
                                                running,
                                                end - running,
                                                timing.request_id,
                                                kError,
                                                /*truncated=*/true,
                                                timing.error_type);
                }
                break;
            }
            const int64_t first_token = timing.first_token_time_us;
            if (first_token > running && first_token <= end) {
                detail::synthesizeChildSpan(parent_span, "prefill", running, first_token - running, timing.request_id);
            }
            break;
        }
        case PhaseRole::Decode: {
            if (timing.generation_done) {
                const int64_t done = timing.generation_done_time_us;
                if (done > running && done <= end) {
                    detail::synthesizeChildSpan(parent_span, "decode", running, done - running, timing.request_id);
                }
            } else if (!request_ok) {
                detail::synthesizeChildSpan(parent_span,
                                            "decode",
                                            running,
                                            end - running,
                                            timing.request_id,
                                            kError,
                                            /*truncated=*/true,
                                            timing.error_type);
            }
            break;
        }
    }
}

class PhaseSpanSynthesisScope {
public:
    explicit PhaseSpanSynthesisScope(std::function<void(bool)> callback):
        callback_(std::move(callback)), uncaught_exceptions_(std::uncaught_exceptions()) {}

    PhaseSpanSynthesisScope(const PhaseSpanSynthesisScope&)            = delete;
    PhaseSpanSynthesisScope& operator=(const PhaseSpanSynthesisScope&) = delete;

    ~PhaseSpanSynthesisScope() noexcept {
        try {
            if (callback_) {
                callback_(std::uncaught_exceptions() > uncaught_exceptions_);
            }
        } catch (...) {
            // fail-open: telemetry must never break inference
        }
    }

private:
    std::function<void(bool)> callback_;
    int                       uncaught_exceptions_;
};

}  // namespace telemetry
}  // namespace rtp_llm
