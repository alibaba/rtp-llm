#include "rtp_llm/cpp/metrics/KVCacheCanaryMetrics.h"

#include <atomic>

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

namespace {

std::atomic<uint64_t> g_canary_hit_rate_ticks{0};
std::atomic<uint64_t> g_canary_hbm_used_blocks_ticks{0};
std::atomic<uint64_t> g_canary_ttft_ticks{0};
std::atomic<uint64_t> g_canary_tpot_ticks{0};
std::atomic<uint64_t> g_canary_error_events{0};
std::atomic<uint64_t> g_canary_oom_events{0};
std::atomic<uint64_t> g_canary_peer_refused{0};

}  // namespace

bool KVCacheCanaryMetrics::init(kmonitor::MetricsGroupManager* manager) {
    // Metric path names match canary §1.0 item 7 verbatim — DO NOT rename
    // without updating PHASE5_CANARY_PROCEDURE.md and the corresponding
    // dashboard JSON in the whale-biz-operation skill.
    REGISTER_GAUGE_MUTABLE_METRIC(hit_rate_metric_, "kv_cache.hit_rate");
    REGISTER_GAUGE_MUTABLE_METRIC(hbm_used_blocks_metric_, "kv_cache.hbm_used_blocks");
    REGISTER_GAUGE_MUTABLE_METRIC(ttft_ms_metric_, "inference.ttft_ms");
    REGISTER_GAUGE_MUTABLE_METRIC(tpot_ms_metric_, "inference.tpot_ms");
    REGISTER_QPS_MUTABLE_METRIC(error_count_metric_, "engine.error_count");
    REGISTER_QPS_MUTABLE_METRIC(oom_count_metric_, "engine.oom_count");
    REGISTER_QPS_MUTABLE_METRIC(peer_refused_metric_, "pd.peer.refused_total");
    return true;
}

void KVCacheCanaryMetrics::report(const kmonitor::MetricsTags* tags, KVCacheCanaryMetricsCollector* collector) {
    // ``tags`` is captured by the REPORT_* macros from MetricMacro.h via
    // identifier-name lookup, so the parameter MUST stay named ``tags``
    // — do not rename or comment-out without rewriting the macros.
    (void)tags;
    if (collector == nullptr) {
        return;
    }
    if (collector->kv_cache_hit_rate >= 0.0f) {
        REPORT_MUTABLE_METRIC(hit_rate_metric_, collector->kv_cache_hit_rate);
    }
    if (collector->kv_cache_hbm_used_blocks >= 0) {
        REPORT_MUTABLE_METRIC(hbm_used_blocks_metric_, collector->kv_cache_hbm_used_blocks);
    }
    if (collector->inference_ttft_ms >= 0) {
        REPORT_MUTABLE_METRIC(ttft_ms_metric_, collector->inference_ttft_ms);
    }
    if (collector->inference_tpot_ms >= 0) {
        REPORT_MUTABLE_METRIC(tpot_ms_metric_, collector->inference_tpot_ms);
    }
    if (collector->engine_error_event) {
        REPORT_MUTABLE_QPS(error_count_metric_);
    }
    if (collector->engine_oom_event) {
        REPORT_MUTABLE_QPS(oom_count_metric_);
    }
    if (collector->pd_peer_refused) {
        REPORT_MUTABLE_QPS(peer_refused_metric_);
    }
}

// ---- Test accessors -------------------------------------------------------

uint64_t canaryHitRateTickCount() {
    return g_canary_hit_rate_ticks.load(std::memory_order_relaxed);
}
uint64_t canaryHbmUsedBlocksTickCount() {
    return g_canary_hbm_used_blocks_ticks.load(std::memory_order_relaxed);
}
uint64_t canaryTtftTickCount() {
    return g_canary_ttft_ticks.load(std::memory_order_relaxed);
}
uint64_t canaryTpotTickCount() {
    return g_canary_tpot_ticks.load(std::memory_order_relaxed);
}
uint64_t canaryErrorEventCount() {
    return g_canary_error_events.load(std::memory_order_relaxed);
}
uint64_t canaryOomEventCount() {
    return g_canary_oom_events.load(std::memory_order_relaxed);
}
uint64_t canaryPeerRefusedCount() {
    return g_canary_peer_refused.load(std::memory_order_relaxed);
}

void resetCanaryCountersForTest() {
    g_canary_hit_rate_ticks.store(0, std::memory_order_relaxed);
    g_canary_hbm_used_blocks_ticks.store(0, std::memory_order_relaxed);
    g_canary_ttft_ticks.store(0, std::memory_order_relaxed);
    g_canary_tpot_ticks.store(0, std::memory_order_relaxed);
    g_canary_error_events.store(0, std::memory_order_relaxed);
    g_canary_oom_events.store(0, std::memory_order_relaxed);
    g_canary_peer_refused.store(0, std::memory_order_relaxed);
}

// ---- Helpers --------------------------------------------------------------

void recordCanaryHitRate(const kmonitor::MetricsReporterPtr& reporter, float hit_rate_percent) {
    g_canary_hit_rate_ticks.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.kv_cache_hit_rate = hit_rate_percent;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryHbmUsedBlocks(const kmonitor::MetricsReporterPtr& reporter, int64_t used_blocks) {
    g_canary_hbm_used_blocks_ticks.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.kv_cache_hbm_used_blocks = used_blocks < 0 ? 0 : used_blocks;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryTtftMs(const kmonitor::MetricsReporterPtr& reporter, int64_t ttft_ms) {
    g_canary_ttft_ticks.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.inference_ttft_ms = ttft_ms < 0 ? 0 : ttft_ms;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryTpotMs(const kmonitor::MetricsReporterPtr& reporter, int64_t tpot_ms) {
    g_canary_tpot_ticks.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.inference_tpot_ms = tpot_ms < 0 ? 0 : tpot_ms;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryErrorEvent(const kmonitor::MetricsReporterPtr& reporter) {
    g_canary_error_events.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.engine_error_event = true;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryOomEvent(const kmonitor::MetricsReporterPtr& reporter) {
    g_canary_oom_events.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.engine_oom_event = true;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

void recordCanaryPeerRefused(const kmonitor::MetricsReporterPtr& reporter) {
    g_canary_peer_refused.fetch_add(1, std::memory_order_relaxed);
    if (reporter) {
        KVCacheCanaryMetricsCollector c;
        c.pd_peer_refused = true;
        reporter->report<KVCacheCanaryMetrics, KVCacheCanaryMetricsCollector>(nullptr, &c);
    }
}

}  // namespace rtp_llm
