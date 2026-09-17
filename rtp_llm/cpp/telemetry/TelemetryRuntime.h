#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <map>
#include <string>

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/trace/tracer.h"

namespace rtp_llm {
namespace telemetry {

// Python 校验后的 Trace 配置，独立于引擎调试配置，禁止输出凭证。
struct TelemetryConfig {
    bool                               enabled = false;
    std::string                        endpoint;
    std::map<std::string, std::string> headers;
    std::string                        certificate;
    std::string                        scope_version;
    std::string                        source;
    // 仅根 span 使用比例采样；有父上下文时跟随父节点。
    double  sampler_ratio         = 1.0;
    size_t  max_queue_size        = 2048;
    int64_t schedule_delay_ms     = 5000;
    size_t  max_export_batch_size = 512;
    int64_t http_timeout_ms       = 3000;
    // 空值按运行时角色派生 service.name。
    std::string service_name;

    // Process identity set by caller, not env.
    std::string role;  // frontend / prefill / decode / pdfusion
    int64_t     tp_rank = 0;
    // DP-deployment identity (rtp_llm.dp_rank / rtp_llm.world_rank resource
    // attributes): every DP group's tp_rank0 produces spans, so these are the
    // only semantic keys distinguishing replicas on the platform.
    int64_t dp_rank    = 0;
    int64_t world_rank = 0;
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
    // 初始化显式配置，实际 rank 由引擎传入；不读取 Trace 环境变量。
    static bool init(const TelemetryConfig& config,
                     const std::string&     role,
                     int64_t                tp_rank,
                     int64_t                dp_rank    = 0,
                     int64_t                world_rank = 0);

    // Test-only: initialize with an injected exporter regardless of env switch.
    // Caller must ensure prior state is SHUTDOWN/UNINITIALIZED (no lock re-entry).
    static bool initWithExporter(std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter,
                                 const TelemetryConfig&                                   config);

    // Bounded shutdown: flush/shutdown runs on a detached thread; on deadline
    // expiry the provider is intentionally leaked so business exit never hangs.
    // Idempotent.
    static bool shutdown(int64_t deadline_ms = 2000);
    // 仅供测试隔离使用；生产进程不重读配置。
    static bool resetForTest();

    static bool           isActive();
    static TelemetryState state();

    // Engine identity behind the gen_ai.engine.index phase span attribute. Only
    // meaningful while ACTIVE; returns 0 otherwise, which is indistinguishable
    // from a genuine rank 0 and is why callers must gate on isActive() first
    // (every phase span factory already does).
    static int64_t worldRank();

    // Returns tracer when ACTIVE, otherwise a no-op tracer. Never null.
    static opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> tracer();

private:
    static bool initInternal(std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter,
                             const TelemetryConfig&                                   config);
};

}  // namespace telemetry
}  // namespace rtp_llm
