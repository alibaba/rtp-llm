#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <utility>

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/exporters/otlp/otlp_http.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"
#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/batch_span_processor_options.h"
#include "opentelemetry/sdk/trace/provider.h"
#include "opentelemetry/sdk/trace/samplers/parent.h"
#include "opentelemetry/sdk/trace/samplers/trace_id_ratio.h"
#include "opentelemetry/sdk/trace/tracer_provider.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/noop.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace telemetry {

namespace {

namespace trace_api = opentelemetry::trace;
namespace trace_sdk = opentelemetry::sdk::trace;
namespace otlp      = opentelemetry::exporter::otlp;
namespace otel_ctx  = opentelemetry::context;
namespace resource  = opentelemetry::sdk::resource;
namespace nostd     = opentelemetry::nostd;

struct RuntimeGlobals {
    std::mutex     mutex;
    TelemetryState state = TelemetryState::UNINITIALIZED;
    // Lock-free fast path mirror of (state == ACTIVE) for per-request checks
    // on hot RPC paths (avoids a process-wide mutex hotspot).
    std::atomic<bool>                          active{false};
    std::shared_ptr<trace_sdk::TracerProvider> provider;
};

RuntimeGlobals& globals() {
    static RuntimeGlobals instance;
    return instance;
}

// Export-failure accounting wrapper around the OTLP exporter. Queue overflow
// is already reported by the upstream BSP; this wrapper covers the separate
// exporter-side failure channel with rate-limited cumulative diagnostics.
class DiagnosticSpanExporter final: public trace_sdk::SpanExporter {
public:
    explicit DiagnosticSpanExporter(std::unique_ptr<trace_sdk::SpanExporter> inner): inner_(std::move(inner)) {}

    std::unique_ptr<trace_sdk::Recordable> MakeRecordable() noexcept override {
        return inner_->MakeRecordable();
    }

    opentelemetry::sdk::common::ExportResult
    Export(const nostd::span<std::unique_ptr<trace_sdk::Recordable>>& spans) noexcept override {
        auto result = inner_->Export(spans);
        total_batches_.fetch_add(1, std::memory_order_relaxed);
        if (result != opentelemetry::sdk::common::ExportResult::kSuccess) {
            auto failed_batches = failed_batches_.fetch_add(1, std::memory_order_relaxed) + 1;
            auto failed_spans =
                failed_spans_.fetch_add((int64_t)spans.size(), std::memory_order_relaxed) + (int64_t)spans.size();
            int64_t now_s =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            int64_t last_s = last_log_s_.load(std::memory_order_relaxed);
            if (now_s - last_s >= kLogIntervalS
                && last_log_s_.compare_exchange_strong(last_s, now_s, std::memory_order_relaxed)) {
                RTP_LLM_LOG_WARNING("telemetry span export failing: result=%d (cumulative %ld/%ld batches failed, "
                                    "%ld spans lost; next warning suppressed for %lds)",
                                    (int)result,
                                    (long)failed_batches,
                                    (long)total_batches_.load(std::memory_order_relaxed),
                                    (long)failed_spans,
                                    (long)kLogIntervalS);
            }
        }
        return result;
    }

    bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
        return inner_->ForceFlush(timeout);
    }

    bool Shutdown(std::chrono::microseconds timeout) noexcept override {
        auto failed_batches = failed_batches_.load(std::memory_order_relaxed);
        if (failed_batches > 0) {
            RTP_LLM_LOG_WARNING("telemetry exporter shutdown: %ld/%ld batches failed, %ld spans lost",
                                (long)failed_batches,
                                (long)total_batches_.load(std::memory_order_relaxed),
                                (long)failed_spans_.load(std::memory_order_relaxed));
        }
        return inner_->Shutdown(timeout);
    }

private:
    static constexpr int64_t                 kLogIntervalS = 60;
    std::unique_ptr<trace_sdk::SpanExporter> inner_;
    std::atomic<int64_t>                     total_batches_{0};
    std::atomic<int64_t>                     failed_batches_{0};
    std::atomic<int64_t>                     failed_spans_{0};
    std::atomic<int64_t>                     last_log_s_{0};
};

std::string getEnvString(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::string(value);
}

bool getEnvBool(const char* name, bool default_value) {
    std::string value = getEnvString(name, "");
    if (value.empty()) {
        return default_value;
    }
    return value == "1" || value == "true" || value == "TRUE" || value == "True" || value == "on" || value == "ON";
}

// Parses positive integer; falls back to default on invalid input (fail-open).
int64_t getEnvPositiveInt(const char* name, int64_t default_value) {
    std::string value = getEnvString(name, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        int64_t parsed = std::stoll(value);
        if (parsed <= 0) {
            RTP_LLM_LOG_WARNING(
                "telemetry env %s=%s invalid (must be > 0), fallback %ld", name, value.c_str(), (long)default_value);
            return default_value;
        }
        return parsed;
    } catch (const std::exception&) {
        RTP_LLM_LOG_WARNING("telemetry env %s=%s parse failed, fallback %ld", name, value.c_str(), (long)default_value);
        return default_value;
    }
}

double getEnvRatio(const char* name, double default_value) {
    std::string value = getEnvString(name, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        size_t parsed_length = 0;
        double parsed        = std::stod(value, &parsed_length);
        if (parsed_length != value.size() || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
            RTP_LLM_LOG_WARNING("telemetry env %s=%s out of [0,1], fallback %f", name, value.c_str(), default_value);
            return default_value;
        }
        return parsed;
    } catch (const std::exception&) {
        RTP_LLM_LOG_WARNING("telemetry env %s=%s parse failed, fallback %f", name, value.c_str(), default_value);
        return default_value;
    }
}

void setGlobalNoop() {
    std::shared_ptr<trace_api::TracerProvider> none;
    trace_sdk::Provider::SetTracerProvider(none);
}

}  // namespace

TelemetryConfig TelemetryConfig::fromEnv() {
    TelemetryConfig config;
    config.enabled = getEnvBool("RTP_LLM_OTEL_TRACE_ENABLE", false);

    // Endpoint priority: signal-specific > generic (+ /v1/traces) > empty.
    std::string signal_endpoint  = getEnvString("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "");
    std::string generic_endpoint = getEnvString("OTEL_EXPORTER_OTLP_ENDPOINT", "");
    if (!signal_endpoint.empty()) {
        config.endpoint = signal_endpoint;
    } else if (!generic_endpoint.empty()) {
        // OTLP spec: for HTTP the generic endpoint gets the signal path appended.
        std::string base = generic_endpoint;
        while (!base.empty() && base.back() == '/') {
            base.pop_back();
        }
        config.endpoint = base + "/v1/traces";
    }

    config.sampler_ratio         = getEnvRatio("RTP_LLM_OTEL_TRACE_SAMPLER_RATIO", 1.0);
    config.max_queue_size        = (size_t)getEnvPositiveInt("RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE", 2048);
    config.schedule_delay_ms     = getEnvPositiveInt("RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS", 5000);
    config.max_export_batch_size = (size_t)getEnvPositiveInt("RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE", 512);
    config.http_timeout_ms       = getEnvPositiveInt("RTP_LLM_OTEL_HTTP_TIMEOUT_MS", 3000);
    // Empty means "derive from role at init()" (role-split components).
    config.service_name = getEnvString("RTP_LLM_OTEL_SERVICE_NAME", "");
    // BSP requires max_export_batch_size <= max_queue_size.
    if (config.max_export_batch_size > config.max_queue_size) {
        RTP_LLM_LOG_WARNING("telemetry max_export_batch_size %zu > max_queue_size %zu, clamp",
                            config.max_export_batch_size,
                            config.max_queue_size);
        config.max_export_batch_size = config.max_queue_size;
    }
    return config;
}

bool TelemetryRuntime::initInternal(std::unique_ptr<trace_sdk::SpanExporter> exporter, const TelemetryConfig& config) {
    // Caller holds globals().mutex.
    auto& g = globals();
    // Single derivation point shared by BOTH entry points (init() and
    // initWithExporter()). Role-split components: an empty service_name derives
    // from the deployment role (rtp_llm_prefill / rtp_llm_decode /
    // rtp_llm_pdfusion) so the Unitrace topology shows P/D as separate nodes;
    // an explicit RTP_LLM_OTEL_SERVICE_NAME still overrides globally. Resolving
    // here rather than in init() keeps the test entry on the same contract as
    // production, so a broken derivation can no longer pass the tests.
    const std::string service_name = !config.service_name.empty() ?
                                         config.service_name :
                                         (config.role.empty() ? std::string("rtp_llm") : "rtp_llm_" + config.role);
    try {
        // 1. Resource: service.instance.id carries per-process
        // identity; host.ip is written ONLY when a real pod IP is available and
        // is never synthesized from hostname-pid, which would misrepresent the
        // node address to topology views.
        resource::ResourceAttributes attributes{
            {"service.name", service_name},
            {"service.instance.id", getEnvString("HOSTNAME", "unknown") + "-" + std::to_string(getpid())},
            {"process.pid", (int64_t)getpid()},
            {"rtp_llm.role", config.role},
            // rtp_llm.tp_rank is intentionally NOT a resource attribute: the
            // rank0-only gate makes it constantly 0 on every exported span
            // (zero information). tp_rank stays an init() gate parameter only.
            // Replica identity: every DP group's tp_rank0 produces spans with
            // otherwise identical resources (same role/tp_rank/ip), so dp_rank
            // and world_rank are the semantic keys telling replicas apart.
            {"rtp_llm.dp_rank", config.dp_rank},
            {"rtp_llm.world_rank", config.world_rank},
        };
        std::string pod_ip = getEnvString("POD_IP", "");
        if (!pod_ip.empty()) {
            attributes.SetAttribute("host.ip", opentelemetry::nostd::string_view(pod_ip));
        }
        auto res = resource::Resource::Create(attributes);

        // 2. Sampler: ParentBased(TraceIdRatio) trusts the upstream sampled flag.
        auto delegate = std::make_shared<trace_sdk::TraceIdRatioBasedSampler>(config.sampler_ratio);
        auto sampler  = std::unique_ptr<trace_sdk::Sampler>(new trace_sdk::ParentBasedSampler(delegate));

        // 3. Bounded BSP, queue-full drops silently (fail-open).
        trace_sdk::BatchSpanProcessorOptions bsp_options;
        bsp_options.max_queue_size        = config.max_queue_size;
        bsp_options.schedule_delay_millis = std::chrono::milliseconds(config.schedule_delay_ms);
        bsp_options.max_export_batch_size = config.max_export_batch_size;
        auto processor = trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), bsp_options);

        // 4. TracerProvider
        auto provider_unique = trace_sdk::TracerProviderFactory::Create(std::move(processor), res, std::move(sampler));
        g.provider           = std::shared_ptr<trace_sdk::TracerProvider>(provider_unique.release());
        std::shared_ptr<trace_api::TracerProvider> api_provider = g.provider;
        trace_sdk::Provider::SetTracerProvider(api_provider);

        // 5. Global W3C TraceContext propagator, set explicitly: without it the
        // default no-op propagator silently drops cross-process context.
        otel_ctx::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
            nostd::shared_ptr<otel_ctx::propagation::TextMapPropagator>(
                new trace_api::propagation::HttpTraceContext()));

        g.state = TelemetryState::ACTIVE;
        g.active.store(true, std::memory_order_release);
        RTP_LLM_LOG_INFO("telemetry runtime active: role=%s tp_rank=%ld service=%s",
                         config.role.c_str(),
                         (long)config.tp_rank,
                         service_name.c_str());
        return true;
    } catch (const std::exception&) {
        g.state = TelemetryState::INIT_FAILURE;
        g.provider.reset();
        setGlobalNoop();
        RTP_LLM_LOG_ERROR("telemetry init failed (telemetry disabled, inference unaffected)");
        return false;
    } catch (...) {
        g.state = TelemetryState::INIT_FAILURE;
        g.provider.reset();
        setGlobalNoop();
        RTP_LLM_LOG_ERROR("telemetry init failed with unknown exception (telemetry disabled)");
        return false;
    }
}

bool TelemetryRuntime::init(const std::string& role, int64_t tp_rank, int64_t dp_rank, int64_t world_rank) {
    auto&                       g = globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (g.state == TelemetryState::ACTIVE) {
        return true;
    }

    TelemetryConfig config = TelemetryConfig::fromEnv();
    config.role            = role;
    config.tp_rank         = tp_rank;
    config.dp_rank         = dp_rank;
    config.world_rank      = world_rank;
    // service_name stays as resolved by fromEnv(): an empty value is derived
    // from the role inside initInternal(), the single point both entry points
    // share.

    if (!config.enabled) {
        g.state = TelemetryState::DISABLED;
        // Log once so a missing RTP_LLM_OTEL_TRACE_ENABLE in this process is
        // diagnosable (live PD run 2026-07-26: the silent branch made "env not
        // propagated" indistinguishable from "init never called").
        RTP_LLM_LOG_INFO("telemetry disabled: RTP_LLM_OTEL_TRACE_ENABLE not set (role=%s)", role.c_str());
        return false;
    }
    // Only tp_rank==0 owns request spans.
    if (tp_rank != 0) {
        g.state = TelemetryState::DISABLED;
        RTP_LLM_LOG_INFO("telemetry disabled on tp_rank %ld (only rank0 produces spans)", (long)tp_rank);
        return false;
    }
    if (config.endpoint.empty()) {
        g.state = TelemetryState::DISABLED;
        RTP_LLM_LOG_ERROR("telemetry enabled but no OTLP endpoint configured "
                          "(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT / OTEL_EXPORTER_OTLP_ENDPOINT), telemetry disabled");
        return false;
    }

    std::unique_ptr<trace_sdk::SpanExporter> exporter;
    try {
        otlp::OtlpHttpExporterOptions exporter_options;
        exporter_options.url          = config.endpoint;
        exporter_options.content_type = otlp::HttpRequestContentType::kBinary;
        exporter_options.timeout      = std::chrono::milliseconds(config.http_timeout_ms);
        // HTTPS endpoints need a CA bundle for libcurl; without one every C++
        // span is silently dropped with "error setting certificate verify
        // locations" (hit on the live 2026-07-26 run). The options constructor
        // already honors OTEL_EXPORTER_OTLP(_TRACES)_CERTIFICATE; only when
        // nothing is configured, fall back to the common system bundles.
        if (config.endpoint.rfind("https", 0) == 0 && exporter_options.ssl_ca_cert_path.empty()
            && exporter_options.ssl_ca_cert_string.empty()) {
            static const char* kCaBundleCandidates[] = {
                "/etc/pki/tls/certs/ca-bundle.crt",    // RHEL / Alibaba Cloud Linux
                "/etc/ssl/certs/ca-certificates.crt",  // Debian / Ubuntu
                "/etc/ssl/certs/ca-bundle.crt",
                "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
            };
            for (const char* candidate : kCaBundleCandidates) {
                if (access(candidate, R_OK) == 0) {
                    exporter_options.ssl_ca_cert_path = candidate;
                    RTP_LLM_LOG_INFO("telemetry auto-detected system CA bundle for HTTPS export: %s", candidate);
                    break;
                }
            }
            if (exporter_options.ssl_ca_cert_path.empty()) {
                RTP_LLM_LOG_WARNING("telemetry HTTPS endpoint but no CA bundle found; export will likely fail "
                                    "(set OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE)");
            }
        }
        exporter = otlp::OtlpHttpExporterFactory::Create(exporter_options);
        // Production wire path only; initWithExporter (test injection) stays
        // unwrapped so tests assert against their exporter directly.
        exporter = std::make_unique<DiagnosticSpanExporter>(std::move(exporter));
    } catch (const std::exception&) {
        g.state = TelemetryState::INIT_FAILURE;
        RTP_LLM_LOG_ERROR("telemetry exporter create failed (telemetry disabled)");
        return false;
    }
    return initInternal(std::move(exporter), config);
}

bool TelemetryRuntime::initWithExporter(std::unique_ptr<trace_sdk::SpanExporter> exporter,
                                        const TelemetryConfig&                   config) {
    auto&                       g = globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (g.state == TelemetryState::ACTIVE) {
        RTP_LLM_LOG_WARNING("telemetry initWithExporter called while ACTIVE, call shutdown first");
        return false;
    }
    return initInternal(std::move(exporter), config);
}

bool TelemetryRuntime::shutdown(int64_t deadline_ms) {
    std::shared_ptr<trace_sdk::TracerProvider> provider;
    {
        auto&                       g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        if (g.state != TelemetryState::ACTIVE) {
            g.state =
                (g.state == TelemetryState::UNINITIALIZED) ? TelemetryState::UNINITIALIZED : TelemetryState::SHUTDOWN;
            return true;
        }
        provider = std::move(g.provider);
        g.provider.reset();
        g.state = TelemetryState::SHUTDOWN;
        g.active.store(false, std::memory_order_release);
        // Detach the old provider while holding the same mutex used by init().
        // Otherwise a concurrent init can publish a new provider here and then
        // be overwritten by the old shutdown's delayed no-op assignment.
        setGlobalNoop();
    }

    // Bounded shutdown: flush + shutdown on a detached thread. BSP destruction
    // joins its worker thread indefinitely, so on deadline expiry we leak the
    // provider intentionally and let the detached thread block forever; the OS
    // reclaims it at process exit.
    auto done   = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    std::thread([provider, done, deadline_ms]() mutable {
        try {
            provider->ForceFlush(std::chrono::microseconds(deadline_ms * 1000));
            provider->Shutdown();
        } catch (...) {
            // swallow everything: telemetry must never break shutdown
        }
        provider.reset();
        done->set_value();
    }).detach();

    if (future.wait_for(std::chrono::milliseconds(deadline_ms)) != std::future_status::ready) {
        RTP_LLM_LOG_WARNING("telemetry shutdown exceeded deadline %ld ms, remaining spans dropped", (long)deadline_ms);
        return false;
    }
    return true;
}

bool TelemetryRuntime::isActive() {
    // Lock-free: called once per request on RPC hot paths even when telemetry
    // is disabled; the mutex-guarded state stays authoritative for init/shutdown.
    return globals().active.load(std::memory_order_acquire);
}

TelemetryState TelemetryRuntime::state() {
    auto&                       g = globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    return g.state;
}

nostd::shared_ptr<trace_api::Tracer> TelemetryRuntime::tracer() {
    {
        auto&                       g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        if (g.state == TelemetryState::ACTIVE && g.provider) {
            // Scope version is injected by the Python launcher (same env
            // inheritance path as OTEL_EXPORTER_OTLP_TRACES_*).
            static const std::string scope_version = getEnvString("RTP_LLM_OTEL_SCOPE_VERSION", "");
            return g.provider->GetTracer("rtp_llm", scope_version);
        }
    }
    static nostd::shared_ptr<trace_api::TracerProvider> noop_provider(new trace_api::NoopTracerProvider());
    return noop_provider->GetTracer("rtp_llm_noop", "");
}

}  // namespace telemetry
}  // namespace rtp_llm
