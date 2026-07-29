#include "rtp_llm/cpp/model_rpc/PrefillMetrics.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <strings.h>

#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/cache/RecentCacheKeyWindow.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

namespace {

bool envValueIsTrue(const char* value) {
    return value != nullptr
           && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
               || strcasecmp(value, "yes") == 0);
}

double theoryHitRatio(int64_t hit_count, int64_t total_count) {
    return total_count > 0 ? static_cast<double>(hit_count) / static_cast<double>(total_count) : 0.0;
}

struct TheoryHitStatsSnapshot {
    int64_t all_hit_count   = 0;
    int64_t all_total_count = 0;
    double  all_hit_ratio   = 0.0;
};

class TheoryHitStats {
public:
    TheoryHitStatsSnapshot record(int64_t hit_count, int64_t total_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t               safe_hit   = std::max<int64_t>(0, hit_count);
        const int64_t               safe_total = std::max<int64_t>(0, total_count);

        if (safe_total > 0) {
            all_hit_count_ += safe_hit;
            all_total_count_ += safe_total;
        }

        TheoryHitStatsSnapshot snapshot;
        snapshot.all_hit_count   = all_hit_count_;
        snapshot.all_total_count = all_total_count_;
        snapshot.all_hit_ratio   = theoryHitRatio(all_hit_count_, all_total_count_);
        return snapshot;
    }

private:
    int64_t    all_hit_count_   = 0;
    int64_t    all_total_count_ = 0;
    std::mutex mutex_;
};

std::string cacheKeyPreview(const std::vector<CacheKeyType>& keys, size_t limit = 6) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < keys.size() && i < limit; ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << keys[i];
    }
    if (keys.size() > limit) {
        oss << ",...";
    }
    oss << "]";
    return oss.str();
}

std::string cacheKeysToString(const std::vector<CacheKeyType>& keys) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << keys[i];
    }
    oss << "]";
    return oss.str();
}

std::string cacheKeyDigest(const std::vector<CacheKeyType>& keys) {
    uint64_t digest = 14695981039346656037ULL;
    for (const auto cache_key : keys) {
        uint64_t value = static_cast<uint64_t>(cache_key);
        digest ^= value;
        digest *= 1099511628211ULL;
        digest ^= value >> 32;
        digest *= 1099511628211ULL;
    }
    return std::to_string(digest);
}

std::vector<CacheKeyType> buildFullBlockCacheKeys(torch::Tensor input_ids, int seq_size_per_block) {
    std::vector<CacheKeyType> cache_keys;
    if (seq_size_per_block <= 0 || !input_ids.defined() || input_ids.numel() <= 0) {
        return cache_keys;
    }

    if (!input_ids.device().is_cpu()) {
        input_ids = input_ids.cpu();
    }
    if (!input_ids.is_contiguous()) {
        input_ids = input_ids.contiguous();
    }
    if (input_ids.scalar_type() != torch::kInt32) {
        input_ids = input_ids.to(torch::kInt32);
    }

    const int64_t token_num   = input_ids.numel();
    const int64_t block_count = token_num / seq_size_per_block;
    if (block_count <= 0) {
        return cache_keys;
    }
    cache_keys.reserve(static_cast<size_t>(block_count));

    auto*   token_ids    = input_ids.data_ptr<int32_t>();
    int64_t rolling_hash = 0;
    for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
        const int64_t pos = block_idx * seq_size_per_block;
        rolling_hash      = rtp_llm::hashInt64Array(
            rolling_hash, token_ids + pos, token_ids + pos + static_cast<int64_t>(seq_size_per_block));
        cache_keys.push_back(static_cast<CacheKeyType>(rolling_hash));
    }
    return cache_keys;
}

void fillPrefillRecentCacheKeyMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                               const RecentCacheKeyWindow::Snapshot&  snapshot) {
    collector.has_value                  = true;
    collector.request_count              = true;
    collector.empty_request_count        = snapshot.request_occurrences == 0;
    collector.hit_count                  = snapshot.request_hit_occurrences;
    collector.total_count                = snapshot.request_occurrences;
    collector.hit_ratio                  = snapshot.request_hit_ratio;
    collector.retained_occurrences       = snapshot.retained_occurrences;
    collector.retained_unique_cache_keys = static_cast<int64_t>(snapshot.retained_unique_cache_keys);
    collector.time_window_ms             = snapshot.time_window_ms;
}

void fillPrefillTheoryHitMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                          const TheoryHitStatsSnapshot&          snapshot) {
    if (snapshot.all_total_count <= 0) {
        return;
    }
    collector.theory_has_value       = true;
    collector.theory_all_hit_count   = snapshot.all_hit_count;
    collector.theory_all_total_count = snapshot.all_total_count;
    collector.theory_all_hit_ratio   = snapshot.all_hit_ratio;
}

}  // namespace

bool prefillTraceLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_TRACE_LOG_ENABLE");
        if (value == nullptr) {
            value = std::getenv("PREFILL_CACHE_DEBUG_LOG");
        }
        if (value == nullptr) {
            value = std::getenv("KV_CACHE_DEBUG_LOG");
        }
        return envValueIsTrue(value);
    }();
    return enabled;
}

bool prefillCacheDebugLogEnabled() {
    return prefillTraceLogEnabled();
}

const char* prefillStageName(PrefillStatInfo::ExecuteStage stage) {
    switch (stage) {
        case PrefillStatInfo::start:
            return "start";
        case PrefillStatInfo::getRpcConnection:
            return "getRpcConnection";
        case PrefillStatInfo::multimodalProcess:
            return "multimodalProcess";
        case PrefillStatInfo::remoteAllocateResource:
            return "remoteAllocateResource";
        case PrefillStatInfo::enqueueRequest:
            return "enqueueRequest";
        case PrefillStatInfo::remoteLoadCacheStart:
            return "remoteLoadCacheStart";
        case PrefillStatInfo::pollLocalOutput:
            return "pollLocalOutput";
        case PrefillStatInfo::remoteLoadCacheEnd:
            return "remoteLoadCacheEnd";
        case PrefillStatInfo::RemoteGenerate:
            return "RemoteGenerate";
        case PrefillStatInfo::pollRemoteOutput:
            return "pollRemoteOutput";
        case PrefillStatInfo::finish:
            return "finish";
        default:
            return "unknown";
    }
}

void logPrefillFailureTrace(const char* event, PrefillGenerateContext& prefill_context) {
    if (!prefillTraceLogEnabled()) {
        return;
    }
    RTP_LLM_LOG_WARNING("Prefill request trace: event=%s request_id=%ld request_key=%s stage=%s retry_times=%ld "
                        "retry_cost_time_ms=%ld execute_time_ms=%ld decode_addr=%s grpc_code=%d grpc_message=%s "
                        "error_code=%d error_message=%s",
                        event,
                        prefill_context.request_id,
                        prefill_context.request_key.c_str(),
                        prefillStageName(prefill_context.stat_info.stage),
                        prefill_context.retry_times,
                        prefill_context.retry_cost_time_ms,
                        prefill_context.executeTimeMs(),
                        prefill_context.decode_addr.c_str(),
                        static_cast<int>(prefill_context.error_status.error_code()),
                        prefill_context.error_status.error_message().c_str(),
                        static_cast<int>(prefill_context.error_info.code()),
                        prefill_context.error_info.ToString().c_str());
}

void reportPoolMetricsToKmonitor(const kmonitor::MetricsReporterPtr& metrics_reporter,
                                 const std::string&                  pool_name,
                                 const PoolMetrics&                  metrics) {
    if (!metrics_reporter) {
        return;
    }
    PrefillPoolMetricsCollector collector;
    collector.active     = static_cast<int64_t>(metrics.active.load());
    collector.queued     = static_cast<int64_t>(metrics.queued.load());
    collector.completed  = static_cast<int64_t>(metrics.completed.load());
    collector.rejected   = static_cast<int64_t>(metrics.rejected.load());
    collector.fallback   = static_cast<int64_t>(metrics.fallback.load());
    collector.thread_max = static_cast<int64_t>(metrics.thread_max);
    collector.queue_max  = static_cast<int64_t>(metrics.queue_max);
    kmonitor::MetricsTags tags("pool_name", pool_name);
    metrics_reporter->report<PrefillPoolMetrics, PrefillPoolMetricsCollector>(&tags, &collector);
}

void reportPrefillRecentCacheKeyMetrics(RecentCacheKeyWindow*               window,
                                        const kmonitor::MetricsReporterPtr& metrics_reporter,
                                        PrefillGenerateContext&             prefill_context,
                                        int                                 seq_size_per_block) {
    if (!window || !prefill_context.generate_input) {
        return;
    }

    auto cache_keys = buildFullBlockCacheKeys(prefill_context.generate_input->input_ids, seq_size_per_block);
    auto snapshot   = window->record(cache_keys);
    static TheoryHitStats theory_stats;
    auto theory_snapshot = theory_stats.record(snapshot.request_hit_occurrences, snapshot.request_occurrences);

    if (metrics_reporter) {
        PrefillRecentCacheKeyMetricsCollector collector;
        fillPrefillRecentCacheKeyMetricsCollector(collector, snapshot);
        fillPrefillTheoryHitMetricsCollector(collector, theory_snapshot);
        metrics_reporter->report<PrefillRecentCacheKeyMetrics, PrefillRecentCacheKeyMetricsCollector>(nullptr,
                                                                                                      &collector);
    }

    if (prefillCacheDebugLogEnabled()) {
        auto key_digest = cacheKeyDigest(cache_keys);
        auto key_text   = cacheKeysToString(cache_keys);
        RTP_LLM_LOG_INFO("Prefill cache-key trace: request_id=%ld request_key=%s token_num=%ld seq_size_per_block=%d "
                         "key_count=%zu hit_count=%ld total_count=%ld hit_ratio=%.6f cache_key_digest=%s "
                         "retained_occurrences=%ld retained_unique_cache_keys=%zu window_ms=%ld cache_keys=%s",
                         prefill_context.request_id,
                         prefill_context.request_key.c_str(),
                         prefill_context.generate_input->input_ids.numel(),
                         seq_size_per_block,
                         cache_keys.size(),
                         snapshot.request_hit_occurrences,
                         snapshot.request_occurrences,
                         snapshot.request_hit_ratio,
                         key_digest.c_str(),
                         snapshot.retained_occurrences,
                         snapshot.retained_unique_cache_keys,
                         snapshot.time_window_ms,
                         key_text.c_str());
        RTP_LLM_LOG_INFO("Prefill cache-key preview trace: request_id=%ld cache_key_digest=%s keys_preview=%s",
                         prefill_context.request_id,
                         key_digest.c_str(),
                         cacheKeyPreview(cache_keys).c_str());
    }
}

}  // namespace rtp_llm
