#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

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
    const auto    cache_keys        = buildPrefillTheoryWindowKeys(resource, cp_mapper);
    auto          snapshot          = recent_window_.record(cache_keys);
    const int64_t input_token_count = std::max<int64_t>(token_num, 0);
    const int64_t hit_token_count   = theoryHitTokens(
        snapshot.request_hit_occurrences, input_token_count, theoryBlockTokens(seq_size_per_block, cp_mapper));
    auto theory_snapshot = theory_stats_->record(hit_token_count, input_token_count);

    if (metrics_reporter_) {
        PrefillRecentCacheKeyMetricsCollector collector;
        fillPrefillRecentCacheKeyMetricsCollector(collector, snapshot, hit_token_count, input_token_count);
        fillPrefillTheoryHitMetricsCollector(collector, theory_snapshot);
        metrics_reporter_->report<PrefillRecentCacheKeyMetrics, PrefillRecentCacheKeyMetricsCollector>(nullptr,
                                                                                                       &collector);
    }
}

}  // namespace rtp_llm
