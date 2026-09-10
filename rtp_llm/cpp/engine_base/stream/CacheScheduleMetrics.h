#pragma once

#include <cstdint>
#include <optional>

#include "rtp_llm/cpp/cache/CacheLoadMetrics.h"

namespace rtp_llm {

struct SchedulerRoundContext {
    uint64_t owner_id = 0;
    uint64_t round_id = 0;
};

struct CacheScheduleSnapshot {
    bool     has_async_cache_dependency = false;
    int64_t  enqueue_to_canrun_us       = 0;
    int64_t  canrun_to_running_us       = 0;
    int64_t  loading_latency_us         = 0;
    int64_t  load_done_to_running_us    = 0;
    int64_t  ready_wait_us              = 0;
    uint64_t schedule_rounds            = 0;
};

// All methods run under GenerateStream::mutex_. Copies have no owner rights.
// Times are supplied by the caller so the legacy and metrics boundaries coincide.
class CacheScheduleMetrics {
public:
    CacheScheduleMetrics() = default;
    CacheScheduleMetrics(const CacheScheduleMetrics&) {}
    CacheScheduleMetrics& operator=(const CacheScheduleMetrics&);
    void                  activate(uint64_t owner);
    bool                  active() const {
        return owner_ != 0;
    }
    bool frozen() const {
        return snapshot_.has_value();
    }
    void invalidate() {
        if (active() && !frozen()) {
            invalid_ = true;
        }
    }
    void reset();
    void enqueue(int64_t time);
    void canRun(int64_t time, const SchedulerRoundContext* round);
    void loadingStart(int64_t time);
    void loadingDone(int64_t time);
    void running(int64_t time, const SchedulerRoundContext* round);
    void beginAttempt(bool has_async_cache_dependency, bool success);
    void absorb(bool has_async_cache_dependency);
    void observe(const std::optional<CacheLoadTerminalSnapshot>& terminal, int64_t time);
    std::optional<CacheScheduleSnapshot> takeReport();

private:
    bool recording() const {
        return active() && !frozen();
    }
    void                    freeze();
    std::optional<uint64_t> checkedRound(const SchedulerRoundContext* round);

    uint64_t                             owner_                      = 0;
    bool                                 reported_                   = false;
    bool                                 invalid_                    = false;
    bool                                 loading_interval_valid_     = true;
    bool                                 had_attempt_                = false;
    bool                                 has_async_cache_dependency_ = false;
    std::optional<int64_t>               te_, tc_, ts_, tb_, to_, td_, tr_;
    std::optional<uint64_t>              rc_, rr_;
    std::optional<CacheScheduleSnapshot> snapshot_;
};

}  // namespace rtp_llm
