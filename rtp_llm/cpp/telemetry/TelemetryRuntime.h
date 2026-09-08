#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/trace/tracer.h"

namespace rtp_llm {
namespace telemetry {

// Trace telemetry configuration resolved from environment variables.
// All values fall back to safe defaults on invalid input (fail-open).
struct TelemetryConfig {
    // Master switch, off unless set. Env: RTP_LLM_OTEL_TRACE_ENABLE
    bool enabled = false;
    // OTLP/HTTP endpoint resolved by priority:
    //   1. OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (signal-specific, used as-is)
    //   2. OTEL_EXPORTER_OTLP_ENDPOINT (generic, "/v1/traces" appended)
    //   3. empty -> telemetry disabled with warning
    std::string endpoint;
    // Local root sampling ratio for ParentBased delegate, default 1.0.
    // Env: RTP_LLM_OTEL_TRACE_SAMPLER_RATIO
    double sampler_ratio = 1.0;
    // Bounded BSP defaults: conservative enough that a stalled collector can
    // never grow unbounded memory, all overridable per deployment.
    // Env: RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE / RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS /
    //      RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE / RTP_LLM_OTEL_HTTP_TIMEOUT_MS
    size_t  max_queue_size        = 2048;
    int64_t schedule_delay_ms     = 5000;
    size_t  max_export_batch_size = 512;
    int64_t http_timeout_ms       = 3000;
    // Env: RTP_LLM_OTEL_SERVICE_NAME. Empty means "derive from role during
    // initialization" (rtp_llm_<role>, or plain "rtp_llm" when role is empty).
    // Both entry points resolve this identically in initInternal(), so a
    // directly constructed config behaves exactly like the production one.
    std::string service_name;

    // Process identity set by caller, not env.
    std::string role;  // frontend / prefill / decode / pdfusion
    int64_t     tp_rank = 0;
    // DP-deployment identity (rtp_llm.dp_rank / rtp_llm.world_rank resource
    // attributes): every DP group's tp_rank0 produces spans, so these are the
    // only semantic keys distinguishing replicas on the platform.
    int64_t dp_rank    = 0;
    int64_t world_rank = 0;

    static TelemetryConfig fromEnv();
};

// Telemetry runtime health states, queryable for self monitoring.
enum class TelemetryState {
    UNINITIALIZED = 0,
    DISABLED      = 1,  // switch off / non rank0 / missing endpoint / invalid config
    ACTIVE        = 2,
    INIT_FAILURE  = 3,
    SHUTDOWN      = 4,
};

// Process-level trace runtime. Thread-safe, idempotent init, fail-open:
// any failure disables telemetry and never throws into business code.
// Init order: Resource -> Sampler(ParentBased) -> BSP -> TracerProvider
// -> global W3C TraceContext propagator.
class TelemetryRuntime {
public:
    // Initialize from env config. Only tp_rank==0 actually enables span
    // production; dp_rank/world_rank ride along as replica-identity resource
    // attributes.
    // Returns true iff telemetry becomes ACTIVE. Never throws.
    static bool init(const std::string& role, int64_t tp_rank, int64_t dp_rank = 0, int64_t world_rank = 0);

    // Test-only: initialize with an injected exporter regardless of env switch.
    // Caller must ensure prior state is SHUTDOWN/UNINITIALIZED (no lock re-entry).
    static bool initWithExporter(std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter,
                                 const TelemetryConfig&                                   config);

    // Bounded shutdown: flush/shutdown runs on a detached thread; on deadline
    // expiry the provider is intentionally leaked so business exit never hangs.
    // Idempotent.
    static bool shutdown(int64_t deadline_ms = 2000);

    static bool           isActive();
    static TelemetryState state();

    // Returns tracer when ACTIVE, otherwise a no-op tracer. Never null.
    static opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> tracer();

private:
    static bool initInternal(std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter,
                             const TelemetryConfig&                                   config);
};

}  // namespace telemetry
}  // namespace rtp_llm
