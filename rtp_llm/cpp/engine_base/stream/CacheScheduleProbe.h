#pragma once

#include <cstdint>
#include <optional>

#include "rtp_llm/cpp/cache/CacheLoadProbe.h"

namespace rtp_llm {

struct SchedulerRoundContext {
    uint64_t owner_id = 0;
    uint64_t round_id = 0;
};

enum class CacheSampleStatus {
    SINGLE_PASS,
    RECOVERED,
    EXCLUDED
};
inline const char* cacheSampleStatusName(CacheSampleStatus status) {
    return status == CacheSampleStatus::SINGLE_PASS ? "single_pass" :
           status == CacheSampleStatus::RECOVERED   ? "recovered" :
                                                      "excluded";
}

struct CacheScheduleSnapshot {
    CacheDependency   dependency              = CacheDependency::UNKNOWN;
    bool              loading_entered         = false;
    CacheSampleStatus sample_status           = CacheSampleStatus::EXCLUDED;
    bool              numeric_valid           = false;
    int64_t           enqueue_to_canrun_us    = 0;
    int64_t           canrun_to_running_us    = 0;
    int64_t           loading_latency_us      = 0;
    int64_t           load_done_to_running_us = 0;
    int64_t           ready_wait_us           = 0;
    uint64_t          schedule_rounds         = 0;
};

// All methods run under GenerateStream::mutex_. Copies have no owner rights.
// Times are supplied by the caller so the legacy and probe boundaries coincide.
class CacheScheduleProbe {
public:
    CacheScheduleProbe() = default;
    CacheScheduleProbe(const CacheScheduleProbe&) {}
    CacheScheduleProbe& operator=(const CacheScheduleProbe&);
    void                activate(uint64_t owner);
    bool                active() const {
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
    void beginAttempt(const CacheDependencyEvidence& evidence, bool success);
    void absorb(const CacheDependencyEvidence& evidence);
    void observe(const std::optional<CacheLoadTerminalSnapshot>& terminal, int64_t time);
    std::optional<CacheScheduleSnapshot> takeReport();

private:
    bool recording() const {
        return active() && !frozen();
    }
    void                    freeze(bool reached_running);
    std::optional<uint64_t> checkedRound(const SchedulerRoundContext* round);

    uint64_t                             owner_              = 0;
    bool                                 reported_           = false;
    bool                                 invalid_            = false;
    bool                                 reset_invalid_      = false;
    bool                                 had_attempt_        = false;
    bool                                 had_retry_          = false;
    bool                                 unresolved_history_ = false;
    bool                                 loading_entered_    = false;
    bool                                 terminal_success_   = false;
    CacheDependencyEvidence              evidence_;
    std::optional<int64_t>               te_, tc_, ts_, tb_, to_, td_, tr_;
    std::optional<uint64_t>              rc_, rr_;
    std::optional<CacheScheduleSnapshot> snapshot_;
};

}  // namespace rtp_llm
