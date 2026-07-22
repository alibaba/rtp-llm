#include "rtp_llm/cpp/model_rpc/PrefillMetrics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <strings.h>

#include "autil/TimeUtility.h"
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

bool envValueIsFalse(const char* value) {
    return value != nullptr
           && (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0
               || strcasecmp(value, "no") == 0);
}

bool prefillTheoryHitLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_THEORY_HIT_LOG_ENABLED");
        if (value == nullptr || value[0] == 0) {
            value = std::getenv("PREFILL_THEORY_HIT_LOG_ENABLE");
        }
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        return !envValueIsFalse(value);
    }();
    return enabled;
}

const char* prefillTheoryHitLogPath() {
    const char* value = std::getenv("PREFILL_THEORY_HIT_LOG_PATH");
    if (value == nullptr || value[0] == 0) {
        return "/home/admin/logs/prefill_theory_hit.log";
    }
    return value;
}

double theoryHitRatio(int64_t hit_count, int64_t total_count) {
    return total_count > 0 ? static_cast<double>(hit_count) / static_cast<double>(total_count) : 0.0;
}

struct TheoryHitWindowSnapshot {
    const char* label       = "";
    int64_t     window_ms   = 0;
    int64_t     hit_count   = 0;
    int64_t     total_count = 0;
    double      hit_ratio   = 0.0;
};

struct TheoryHitStatsSnapshot {
    int64_t                 now_ms              = 0;
    int64_t                 request_hit_count   = 0;
    int64_t                 request_total_count = 0;
    double                  request_hit_ratio   = 0.0;
    int64_t                 all_hit_count       = 0;
    int64_t                 all_total_count     = 0;
    double                  all_hit_ratio       = 0.0;
    TheoryHitWindowSnapshot window_1m;
    TheoryHitWindowSnapshot window_5m;
    TheoryHitWindowSnapshot window_10m;
    TheoryHitWindowSnapshot window_15m;
};

class TheoryHitStats {
public:
    TheoryHitStats() {
        bucket_seconds_.fill(std::numeric_limits<int64_t>::min());
        bucket_hit_counts_.fill(0);
        bucket_total_counts_.fill(0);
    }

    TheoryHitStatsSnapshot record(int64_t hit_count, int64_t total_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t               now_ms         = autil::TimeUtility::currentTimeInMilliSeconds();
        const int64_t               current_second = now_ms / 1000;
        const int64_t               safe_hit       = std::max<int64_t>(0, hit_count);
        const int64_t               safe_total     = std::max<int64_t>(0, total_count);

        if (safe_total > 0) {
            const size_t index = static_cast<size_t>(current_second % kBucketCount);
            if (bucket_seconds_[index] != current_second) {
                bucket_seconds_[index]      = current_second;
                bucket_hit_counts_[index]   = 0;
                bucket_total_counts_[index] = 0;
            }
            bucket_hit_counts_[index] += safe_hit;
            bucket_total_counts_[index] += safe_total;
            all_hit_count_ += safe_hit;
            all_total_count_ += safe_total;
        }

        TheoryHitStatsSnapshot snapshot;
        snapshot.now_ms              = now_ms;
        snapshot.request_hit_count   = safe_hit;
        snapshot.request_total_count = safe_total;
        snapshot.request_hit_ratio   = theoryHitRatio(safe_hit, safe_total);
        snapshot.all_hit_count       = all_hit_count_;
        snapshot.all_total_count     = all_total_count_;
        snapshot.all_hit_ratio       = theoryHitRatio(all_hit_count_, all_total_count_);
        snapshot.window_1m           = windowSnapshot("1m", 60 * 1000, current_second);
        snapshot.window_5m           = windowSnapshot("5m", 5 * 60 * 1000, current_second);
        snapshot.window_10m          = windowSnapshot("10m", 10 * 60 * 1000, current_second);
        snapshot.window_15m          = windowSnapshot("15m", 15 * 60 * 1000, current_second);
        return snapshot;
    }

private:
    static constexpr size_t kBucketCount = 15 * 60 + 2;

    TheoryHitWindowSnapshot windowSnapshot(const char* label, int64_t window_ms, int64_t current_second) const {
        TheoryHitWindowSnapshot snapshot;
        snapshot.label               = label;
        snapshot.window_ms           = window_ms;
        const int64_t window_seconds = window_ms / 1000;
        for (size_t i = 0; i < kBucketCount; ++i) {
            const int64_t age_seconds = current_second - bucket_seconds_[i];
            if (age_seconds >= 0 && age_seconds < window_seconds) {
                snapshot.hit_count += bucket_hit_counts_[i];
                snapshot.total_count += bucket_total_counts_[i];
            }
        }
        snapshot.hit_ratio = theoryHitRatio(snapshot.hit_count, snapshot.total_count);
        return snapshot;
    }

private:
    std::array<int64_t, kBucketCount> bucket_seconds_;
    std::array<int64_t, kBucketCount> bucket_hit_counts_;
    std::array<int64_t, kBucketCount> bucket_total_counts_;
    int64_t                           all_hit_count_   = 0;
    int64_t                           all_total_count_ = 0;
    std::mutex                        mutex_;
};

std::string formatTheoryTimestampMs(int64_t timestamp_ms) {
    const time_t seconds = static_cast<time_t>(timestamp_ms / 1000);
    struct tm    local_time;
    if (localtime_r(&seconds, &local_time) == nullptr) {
        return std::to_string(timestamp_ms);
    }

    char date_buffer[64];
    if (strftime(date_buffer, sizeof(date_buffer), "%Y-%m-%dT%H:%M:%S", &local_time) == 0) {
        return std::to_string(timestamp_ms);
    }

    char offset_buffer[16];
    if (strftime(offset_buffer, sizeof(offset_buffer), "%z", &local_time) == 0) {
        offset_buffer[0] = '\0';
    }

    char output[96];
    snprintf(output, sizeof(output), "%s.%03ld%s", date_buffer, static_cast<long>(timestamp_ms % 1000), offset_buffer);
    return output;
}

void appendPrefillTheoryHitLogLine(const std::string& line) {
    if (!prefillTheoryHitLogEnabled()) {
        return;
    }
    static std::mutex    log_mutex;
    static std::ofstream log_file;
    static bool          open_failed = false;

    std::lock_guard<std::mutex> lock(log_mutex);
    if (!log_file.is_open() && !open_failed) {
        const char* path = prefillTheoryHitLogPath();
        log_file.open(path, std::ios::out | std::ios::app);
        if (!log_file.is_open()) {
            open_failed = true;
            RTP_LLM_LOG_WARNING("Failed to open prefill theory hit log path: %s", path);
            return;
        }
        RTP_LLM_LOG_INFO("Prefill theory hit log path: %s", path);
    }
    if (!log_file.is_open()) {
        return;
    }
    log_file << line << '\n';
    log_file.flush();
}

std::string formatPrefillTheoryHitLogLine(PrefillGenerateContext&       prefill_context,
                                          int64_t                       token_num,
                                          int                           seq_size_per_block,
                                          const TheoryHitStatsSnapshot& snapshot) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "time=" << formatTheoryTimestampMs(snapshot.now_ms)
        << " ts_ms=" << snapshot.now_ms << " source=prefill"
        << " request_id=" << prefill_context.request_id << " request_key=" << prefill_context.request_key
        << " token_num=" << token_num << " seq_size_per_block=" << seq_size_per_block
        << " request_hit=" << snapshot.request_hit_count << " request_total=" << snapshot.request_total_count
        << " request_ratio=" << snapshot.request_hit_ratio << " all_hit=" << snapshot.all_hit_count
        << " all_total=" << snapshot.all_total_count << " all_ratio=" << snapshot.all_hit_ratio
        << " win1m_hit=" << snapshot.window_1m.hit_count << " win1m_total=" << snapshot.window_1m.total_count
        << " win1m_ratio=" << snapshot.window_1m.hit_ratio << " win5m_hit=" << snapshot.window_5m.hit_count
        << " win5m_total=" << snapshot.window_5m.total_count << " win5m_ratio=" << snapshot.window_5m.hit_ratio
        << " win10m_hit=" << snapshot.window_10m.hit_count << " win10m_total=" << snapshot.window_10m.total_count
        << " win10m_ratio=" << snapshot.window_10m.hit_ratio << " win15m_hit=" << snapshot.window_15m.hit_count
        << " win15m_total=" << snapshot.window_15m.total_count << " win15m_ratio=" << snapshot.window_15m.hit_ratio;
    return oss.str();
}

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
    if (theory_snapshot.request_total_count > 0) {
        appendPrefillTheoryHitLogLine(formatPrefillTheoryHitLogLine(
            prefill_context, prefill_context.generate_input->input_ids.numel(), seq_size_per_block, theory_snapshot));
    }

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
