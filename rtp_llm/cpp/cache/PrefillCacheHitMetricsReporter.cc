#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <strings.h>
#include <utility>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

bool envValueIsFalse(const char* value) {
    return value != nullptr
           && (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0
               || strcasecmp(value, "no") == 0);
}

bool envValueIsTrue(const char* value) {
    return value != nullptr
           && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
               || strcasecmp(value, "yes") == 0);
}

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

int64_t theoryBlockTokens(int seq_size_per_block, const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    if (cp_mapper && cp_mapper->isSharded()) {
        return static_cast<int64_t>(cp_mapper->virtualBlockSize());
    }
    return static_cast<int64_t>(std::max(seq_size_per_block, 0));
}

int64_t theoryHitTokens(int64_t hit_key_count, int64_t input_token_count, int64_t block_tokens) {
    if (hit_key_count <= 0 || input_token_count <= 0 || block_tokens <= 0) {
        return 0;
    }
    return std::min(input_token_count, hit_key_count * block_tokens);
}

struct TheoryHitStatsSnapshot {
    int64_t now_ms              = 0;
    int64_t request_hit_count   = 0;
    int64_t request_total_count = 0;
    double  request_hit_ratio   = 0.0;
    int64_t all_hit_count       = 0;
    int64_t all_total_count     = 0;
    double  all_hit_ratio       = 0.0;
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

std::string formatPrefillTheoryHitLogLine(int64_t                       request_id,
                                          int64_t                       token_num,
                                          int                           seq_size_per_block,
                                          const TheoryHitStatsSnapshot& snapshot) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "time=" << formatTheoryTimestampMs(snapshot.now_ms)
        << " ts_ms=" << snapshot.now_ms << " source=prefill"
        << " request_id=" << request_id << " token_num=" << token_num << " seq_size_per_block=" << seq_size_per_block
        << " request_hit_tokens=" << snapshot.request_hit_count
        << " request_input_tokens=" << snapshot.request_total_count << " request_ratio=" << snapshot.request_hit_ratio
        << " all_hit_tokens=" << snapshot.all_hit_count << " all_input_tokens=" << snapshot.all_total_count
        << " all_ratio=" << snapshot.all_hit_ratio;
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

void fillPrefillRecentCacheKeyMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                               const RecentCacheKeyWindow::Snapshot&  snapshot,
                                               int64_t                                hit_token_count,
                                               int64_t                                input_token_count) {
    collector.has_value                  = true;
    collector.request_count              = true;
    collector.empty_request_count        = snapshot.request_occurrences == 0;
    collector.hit_count                  = hit_token_count;
    collector.total_count                = input_token_count;
    collector.hit_ratio                  = theoryHitRatio(hit_token_count, input_token_count);
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

struct PrefillCacheHitMetricsReporter::TheoryHitStats {
    TheoryHitStatsSnapshot record(int64_t hit_count, int64_t total_count) {
        std::lock_guard<std::mutex> lock(mutex);
        const int64_t               now_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
        const int64_t               safe_hit   = std::max<int64_t>(0, hit_count);
        const int64_t               safe_total = std::max<int64_t>(0, total_count);

        if (safe_total > 0) {
            all_hit_count += safe_hit;
            all_total_count += safe_total;
        }

        TheoryHitStatsSnapshot snapshot;
        snapshot.now_ms              = now_ms;
        snapshot.request_hit_count   = safe_hit;
        snapshot.request_total_count = safe_total;
        snapshot.request_hit_ratio   = theoryHitRatio(safe_hit, safe_total);
        snapshot.all_hit_count       = all_hit_count;
        snapshot.all_total_count     = all_total_count;
        snapshot.all_hit_ratio       = theoryHitRatio(all_hit_count, all_total_count);
        return snapshot;
    }

private:
    int64_t    all_hit_count   = 0;
    int64_t    all_total_count = 0;
    std::mutex mutex;
};

bool PrefillCacheHitMetricsReporter::enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_CACHE_HIT_METRIC_ENABLE");
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        return !envValueIsFalse(value);
    }();
    return enabled;
}

PrefillCacheHitMetricsReporter::PrefillCacheHitMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter):
    metrics_reporter_(std::move(metrics_reporter)), theory_stats_(std::make_unique<TheoryHitStats>()) {}

PrefillCacheHitMetricsReporter::~PrefillCacheHitMetricsReporter() = default;

CacheKeysType buildPrefillTheoryWindowKeys(const BatchKVCacheResource&          resource,
                                           const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    CacheKeysType theory_keys;
    const int     cp_size   = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    const int     cp_offset = cp_size - 1;

    for (int batch_id = 0; batch_id < resource.batchSize(); ++batch_id) {
        CacheKeysType full_keys = resource.cacheKeys(batch_id);
        if (!resource.lastBlockAligned() && !full_keys.empty()) {
            full_keys.pop_back();
        }

        if (cp_size <= 1) {
            theory_keys.insert(theory_keys.end(), full_keys.begin(), full_keys.end());
            continue;
        }

        for (int i = cp_offset; i < static_cast<int>(full_keys.size()); i += cp_size) {
            theory_keys.push_back(full_keys[static_cast<size_t>(i)]);
        }
    }
    return theory_keys;
}

void PrefillCacheHitMetricsReporter::record(const BatchKVCacheResource&          resource,
                                            const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                            int64_t                              request_id,
                                            int64_t                              token_num,
                                            int                                  seq_size_per_block) {
    if (!enabled()) {
        return;
    }

    const auto    cache_keys        = buildPrefillTheoryWindowKeys(resource, cp_mapper);
    auto          snapshot          = recent_window_.record(cache_keys);
    const int64_t input_token_count = std::max<int64_t>(token_num, 0);
    const int64_t hit_token_count   = theoryHitTokens(
        snapshot.request_hit_occurrences, input_token_count, theoryBlockTokens(seq_size_per_block, cp_mapper));
    auto theory_snapshot = theory_stats_->record(hit_token_count, input_token_count);
    if (theory_snapshot.request_total_count > 0) {
        appendPrefillTheoryHitLogLine(
            formatPrefillTheoryHitLogLine(request_id, token_num, seq_size_per_block, theory_snapshot));
    }

    if (metrics_reporter_) {
        PrefillRecentCacheKeyMetricsCollector collector;
        fillPrefillRecentCacheKeyMetricsCollector(collector, snapshot, hit_token_count, input_token_count);
        fillPrefillTheoryHitMetricsCollector(collector, theory_snapshot);
        metrics_reporter_->report<PrefillRecentCacheKeyMetrics, PrefillRecentCacheKeyMetricsCollector>(nullptr,
                                                                                                       &collector);
    }

    if (prefillTraceLogEnabled()) {
        auto key_digest = cacheKeyDigest(cache_keys);
        auto key_text   = cacheKeysToString(cache_keys);
        RTP_LLM_LOG_INFO("Prefill cache-key trace: request_id=%ld token_num=%ld seq_size_per_block=%d "
                         "key_count=%zu hit_key_count=%ld total_key_count=%ld key_hit_ratio=%.6f "
                         "hit_tokens=%ld input_tokens=%ld token_hit_ratio=%.6f cache_key_digest=%s "
                         "retained_occurrences=%ld retained_unique_cache_keys=%zu window_ms=%ld cache_keys=%s",
                         request_id,
                         token_num,
                         seq_size_per_block,
                         cache_keys.size(),
                         snapshot.request_hit_occurrences,
                         snapshot.request_occurrences,
                         snapshot.request_hit_ratio,
                         hit_token_count,
                         input_token_count,
                         theoryHitRatio(hit_token_count, input_token_count),
                         key_digest.c_str(),
                         snapshot.retained_occurrences,
                         snapshot.retained_unique_cache_keys,
                         snapshot.time_window_ms,
                         key_text.c_str());
        RTP_LLM_LOG_INFO("Prefill cache-key preview trace: request_id=%ld cache_key_digest=%s keys_preview=%s",
                         request_id,
                         key_digest.c_str(),
                         cacheKeyPreview(cache_keys).c_str());
    }
}

}  // namespace rtp_llm
