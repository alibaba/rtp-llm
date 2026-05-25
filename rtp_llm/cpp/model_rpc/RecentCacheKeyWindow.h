#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class RecentCacheKeyWindow {
public:
    static constexpr int64_t DEFAULT_TIME_WINDOW_MS = 30LL * 60LL * 1000LL;

    struct Snapshot {
        int64_t time_window_ms             = DEFAULT_TIME_WINDOW_MS;
        int64_t request_occurrences        = 0;
        int64_t request_hit_occurrences    = 0;
        double  request_hit_ratio          = 0.0;
        int64_t retained_occurrences       = 0;
        size_t  retained_unique_cache_keys = 0;
    };

    using NowSupplier = std::function<int64_t()>;

public:
    RecentCacheKeyWindow();
    RecentCacheKeyWindow(int64_t time_window_ms, NowSupplier now_supplier);

    Snapshot record(const std::vector<CacheKeyType>& cache_keys);
    Snapshot snapshot();

    int64_t timeWindowMs() const;

    static int64_t resolveTimeWindowMsFromEnv();

private:
    struct WindowEntry {
        int64_t                                      timestamp_ms = 0;
        std::unordered_map<CacheKeyType, int64_t> cache_key_counts;
    };

    void     evictExpiredLocked(int64_t now_ms);
    Snapshot snapshotLocked(int64_t request_occurrences, int64_t request_hit_occurrences) const;

private:
    int64_t                                      time_window_ms_;
    NowSupplier                                  now_supplier_;
    std::deque<WindowEntry>                      window_entries_;
    std::unordered_map<CacheKeyType, int64_t> cache_key_counts_;
    int64_t                                      retained_occurrences_ = 0;
    mutable std::mutex                           mutex_;
};

}  // namespace rtp_llm
