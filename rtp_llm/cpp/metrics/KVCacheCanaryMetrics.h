#pragma once

// R6 DEV-zeta — canary §1.0 item 7 G5 gauges.
//
// This file declares the six metric paths required by Phase 5 canary §1.0
// item 7 (TASKS.md G5 gate, PHASE5_CANARY_PROCEDURE.md §1 + §3):
//
//   1. kv_cache.hit_rate           — G5a (aliases rtp_llm_kv_cache_hit_rate)
//   2. kv_cache.hbm_used_blocks    — G5b (NEW; derived from total - free)
//   3. inference.ttft_ms           — G5e (aliases rtp_llm_first_token_latency_us / 1000)
//   4. inference.tpot_ms           — G5f (derived from total_latency / iterate_count)
//   5. engine.error_count          — G5c (aliases rtp_llm_framework_error_qps as counter)
//   6. engine.oom_count            — G5c (aliases rtp_llm_malloc_failed_times)
//   7. pd.peer.refused_total       — G5d (NEW; fires at validatePeerHandshake refusal)
//
// The G5 gate count is six (G5a..G5f) but ttft/tpot share G5e/G5f for p99/p999,
// and engine.error_count + engine.oom_count share G5c.  We register seven
// distinct metric paths so dashboards can graph each axis independently.
//
// Wiring strategy:
//   - For the four reuse cases (hit_rate, ttft, error, oom) we tick the canary
//     metric path at the existing report site (GenerateStream::reportMetric)
//     so the canary gauge stays bit-for-bit consistent with the legacy gauge
//     it aliases.  See the "G5 canary alias" comments at the call sites.
//   - hbm_used_blocks is computed inside KVCacheManager::reportMetricsLoop
//     from (total_blocks - free_blocks) which is the exact "used" semantic
//     the canary doc expects (canary G5b threshold is "≤ 2 blocks vs baseline").
//   - pd.peer.refused_total is a process-wide atomic counter incremented in
//     KVCacheConnectorCoordinator::validatePeerHandshake on the same path
//     that already calls recordPdSaltMismatchSkipped.  Today salt mismatch
//     is the only refusal source, so the counter is a clean super-set hook
//     for any future refusal reasons (forward-compat for G5d gate).
//
// Tests (KVCacheCanaryMetricsTest) read the atomic counters via the
// canaryXxxCount() accessors so they don't need a kmonitor mock.

#include <atomic>
#include <cstdint>

#include "kmonitor/client/MetricsReporter.h"

namespace kmonitor {
class MetricsTags;
class MutableMetric;
}  // namespace kmonitor

namespace rtp_llm {

// ---- Collector ------------------------------------------------------------

class KVCacheCanaryMetricsCollector final {
public:
    // Set the field that is being reported on this tick; leave the others at
    // their sentinel default (-1 or 0) and the report() function will skip
    // them.  Each helper recordCanaryXxx() builds a one-field collector so
    // gauges fire independently without cross-tick smearing.
    float   kv_cache_hit_rate    = -1.0f;       // negative → not set this tick
    int64_t kv_cache_hbm_used_blocks = -1;      // negative → not set this tick
    int64_t inference_ttft_ms    = -1;
    int64_t inference_tpot_ms    = -1;
    bool    engine_error_event   = false;
    bool    engine_oom_event     = false;
    bool    pd_peer_refused      = false;
};

// ---- MetricsGroup ---------------------------------------------------------

class KVCacheCanaryMetrics: public kmonitor::MetricsGroup {
public:
    ~KVCacheCanaryMetrics() = default;

    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, KVCacheCanaryMetricsCollector* collector);

private:
    kmonitor::MutableMetric* hit_rate_metric_         = nullptr;  // kv_cache.hit_rate
    kmonitor::MutableMetric* hbm_used_blocks_metric_  = nullptr;  // kv_cache.hbm_used_blocks
    kmonitor::MutableMetric* ttft_ms_metric_          = nullptr;  // inference.ttft_ms
    kmonitor::MutableMetric* tpot_ms_metric_          = nullptr;  // inference.tpot_ms
    kmonitor::MutableMetric* error_count_metric_      = nullptr;  // engine.error_count (QPS)
    kmonitor::MutableMetric* oom_count_metric_        = nullptr;  // engine.oom_count (QPS)
    kmonitor::MutableMetric* peer_refused_metric_     = nullptr;  // pd.peer.refused_total (QPS)
};

// ---- Process-wide counters (test-readable) --------------------------------
//
// These exist so unit tests can assert "the helper actually fired" without a
// kmonitor mock.  They are NOT the production data path — production reads
// the gauges via kmonitor.  Each helper increments its counter unconditionally
// and ALSO fires the kmonitor gauge if a non-null reporter is supplied.

uint64_t canaryHitRateTickCount();
uint64_t canaryHbmUsedBlocksTickCount();
uint64_t canaryTtftTickCount();
uint64_t canaryTpotTickCount();
uint64_t canaryErrorEventCount();
uint64_t canaryOomEventCount();
uint64_t canaryPeerRefusedCount();

void resetCanaryCountersForTest();

// ---- Helpers --------------------------------------------------------------
//
// Each helper is safe to call with a null reporter (no-op except for the
// process-wide counter increment, which is always cheap and lock-free).

void recordCanaryHitRate(const kmonitor::MetricsReporterPtr& reporter, float hit_rate_percent);
void recordCanaryHbmUsedBlocks(const kmonitor::MetricsReporterPtr& reporter, int64_t used_blocks);
void recordCanaryTtftMs(const kmonitor::MetricsReporterPtr& reporter, int64_t ttft_ms);
void recordCanaryTpotMs(const kmonitor::MetricsReporterPtr& reporter, int64_t tpot_ms);
void recordCanaryErrorEvent(const kmonitor::MetricsReporterPtr& reporter);
void recordCanaryOomEvent(const kmonitor::MetricsReporterPtr& reporter);
void recordCanaryPeerRefused(const kmonitor::MetricsReporterPtr& reporter);

}  // namespace rtp_llm
