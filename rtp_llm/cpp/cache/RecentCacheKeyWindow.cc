#include "rtp_llm/cpp/cache/RecentCacheKeyWindow.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

constexpr const char* kPrefillCacheHitTimeWindowMsEnv = "PREFILL_CACHE_HIT_TIME_WINDOW_MS";
constexpr const char* kCacheHitTimeWindowMsEnv        = "CACHE_HIT_TIME_WINDOW_MS";

int64_t normalizeTimeWindowMs(int64_t candidate_ms) {
    if (candidate_ms > 0) {
        return candidate_ms;
    }
    RTP_LLM_LOG_WARNING("Invalid cache hit time window ms: %ld, fallback to default: %ld",
                        candidate_ms,
                        RecentCacheKeyWindow::DEFAULT_TIME_WINDOW_MS);
    return RecentCacheKeyWindow::DEFAULT_TIME_WINDOW_MS;
}

bool parseEnvMs(const char* env_name, int64_t& value) {
    const char* raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    try {
        value = std::stoll(std::string(raw));
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Invalid %s=%s, fallback to default: %ld, error=%s",
                            env_name,
                            raw,
                            RecentCacheKeyWindow::DEFAULT_TIME_WINDOW_MS,
                            e.what());
        value = RecentCacheKeyWindow::DEFAULT_TIME_WINDOW_MS;
        return true;
    }
}

}  // namespace

RecentCacheKeyWindow::RecentCacheKeyWindow():
    RecentCacheKeyWindow(resolveTimeWindowMsFromEnv(), []() { return currentTimeMs(); }) {}

RecentCacheKeyWindow::RecentCacheKeyWindow(int64_t time_window_ms, NowSupplier now_supplier):
    time_window_ms_(normalizeTimeWindowMs(time_window_ms)), now_supplier_(std::move(now_supplier)) {
    if (!now_supplier_) {
        now_supplier_ = []() { return currentTimeMs(); };
    }
}

RecentCacheKeyWindow::Snapshot RecentCacheKeyWindow::record(const std::vector<CacheKeyType>& cache_keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t               now_ms = now_supplier_();
    evictExpiredLocked(now_ms);

    if (cache_keys.empty()) {
        return snapshotLocked(0, 0);
    }

    std::unordered_map<CacheKeyType, int64_t> entry_counts;
    entry_counts.reserve(cache_keys.size());

    int64_t request_occurrences     = 0;
    int64_t request_hit_occurrences = 0;
    for (const auto cache_key : cache_keys) {
        ++request_occurrences;
        if (cache_key_counts_.find(cache_key) != cache_key_counts_.end()) {
            ++request_hit_occurrences;
        }
        ++entry_counts[cache_key];
    }

    if (entry_counts.empty()) {
        return snapshotLocked(0, 0);
    }

    WindowEntry entry;
    entry.timestamp_ms     = now_ms;
    entry.cache_key_counts = std::move(entry_counts);

    for (const auto& [cache_key, count] : entry.cache_key_counts) {
        cache_key_counts_[cache_key] += count;
        retained_occurrences_ += count;
    }
    window_entries_.push_back(std::move(entry));

    return snapshotLocked(request_occurrences, request_hit_occurrences);
}

RecentCacheKeyWindow::Snapshot RecentCacheKeyWindow::snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    evictExpiredLocked(now_supplier_());
    return snapshotLocked(0, 0);
}

int64_t RecentCacheKeyWindow::timeWindowMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return time_window_ms_;
}

int64_t RecentCacheKeyWindow::resolveTimeWindowMsFromEnv() {
    int64_t value = 0;
    if (parseEnvMs(kPrefillCacheHitTimeWindowMsEnv, value) || parseEnvMs(kCacheHitTimeWindowMsEnv, value)) {
        return normalizeTimeWindowMs(value);
    }
    return DEFAULT_TIME_WINDOW_MS;
}

void RecentCacheKeyWindow::evictExpiredLocked(int64_t now_ms) {
    const int64_t expire_before_or_at = now_ms - time_window_ms_;
    while (!window_entries_.empty() && window_entries_.front().timestamp_ms <= expire_before_or_at) {
        auto entry = std::move(window_entries_.front());
        window_entries_.pop_front();
        for (const auto& [cache_key, expired_count] : entry.cache_key_counts) {
            auto it = cache_key_counts_.find(cache_key);
            if (it == cache_key_counts_.end()) {
                continue;
            }
            const int64_t decrement = std::min(it->second, expired_count);
            retained_occurrences_ -= decrement;
            if (it->second <= expired_count) {
                cache_key_counts_.erase(it);
            } else {
                it->second -= expired_count;
            }
        }
    }
}

RecentCacheKeyWindow::Snapshot RecentCacheKeyWindow::snapshotLocked(int64_t request_occurrences,
                                                                    int64_t request_hit_occurrences) const {
    Snapshot snapshot;
    snapshot.time_window_ms          = time_window_ms_;
    snapshot.request_occurrences     = request_occurrences;
    snapshot.request_hit_occurrences = request_hit_occurrences;
    snapshot.request_hit_ratio =
        request_occurrences > 0 ? static_cast<double>(request_hit_occurrences) / request_occurrences : 0.0;
    snapshot.retained_occurrences       = retained_occurrences_;
    snapshot.retained_unique_cache_keys = cache_key_counts_.size();
    return snapshot;
}

}  // namespace rtp_llm
