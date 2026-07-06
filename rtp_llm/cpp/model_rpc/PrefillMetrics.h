#pragma once

#include <atomic>
#include <cstddef>
#include <string>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"

namespace rtp_llm {

class RecentCacheKeyWindow;

// Pool-level health metrics, reported periodically.
struct PoolMetrics {
    std::atomic<size_t> active     = 0;  // currently executing tasks
    std::atomic<size_t> queued     = 0;  // tasks waiting in queue
    std::atomic<size_t> completed  = 0;  // total finished since creation
    std::atomic<size_t> rejected   = 0;  // pushTask refused (pool full)
    std::atomic<size_t> fallback   = 0;  // fallback to detached thread
    size_t              thread_max = 0;  // configured thread count (set once in initThreadPools)
    size_t              queue_max  = 0;  // configured queue depth (set once in initThreadPools)
};

// ---- Trace logging (prefill single-request failure tracing) ----
// Whether verbose prefill request tracing is enabled (env-gated, cached).
bool prefillTraceLogEnabled();
// Whether prefill cache-key debug logging is enabled (currently aliases the trace flag).
bool prefillCacheDebugLogEnabled();
// Human-readable name for a prefill execution stage.
const char* prefillStageName(PrefillStatInfo::ExecuteStage stage);
// Emit a one-line WARNING describing a prefill failure (no-op unless trace logging is enabled).
void logPrefillFailureTrace(const char* event, PrefillGenerateContext& prefill_context);

// ---- Pool metrics reporting ----
// Report one thread-pool's PoolMetrics snapshot to kmonitor under tag pool_name.
void reportPoolMetricsToKmonitor(const kmonitor::MetricsReporterPtr& metrics_reporter,
                                 const std::string&                  pool_name,
                                 const PoolMetrics&                  metrics);

// ---- Recent-cache-key / theory-hit metrics ----
// Compute full-block cache keys for the request, record them into the recent-cache-key window,
// update theory-hit statistics, optionally append the theory-hit log line, and report to kmonitor.
// Callers are responsible for the "report only once" / enabled gating; this does the work.
void reportPrefillRecentCacheKeyMetrics(RecentCacheKeyWindow*               window,
                                        const kmonitor::MetricsReporterPtr& metrics_reporter,
                                        PrefillGenerateContext&             prefill_context,
                                        int                                 seq_size_per_block);

}  // namespace rtp_llm
