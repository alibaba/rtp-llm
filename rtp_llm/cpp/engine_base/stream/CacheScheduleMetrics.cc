#include "rtp_llm/cpp/engine_base/stream/CacheScheduleMetrics.h"

#include <algorithm>
#include <new>

namespace rtp_llm {

CacheScheduleMetrics& CacheScheduleMetrics::operator=(const CacheScheduleMetrics& other) {
    if (this != &other) {
        this->~CacheScheduleMetrics();
        new (this) CacheScheduleMetrics();
    }
    return *this;
}

void CacheScheduleMetrics::activate(uint64_t owner) {
    if (!active() && !reported_) {
        owner_ = owner;
    }
}

void CacheScheduleMetrics::reset() {
    if (!recording()) {
        return;
    }
    // Preserve first enqueue/CanRun across retries; loading attempts may no longer match.
    loading_interval_valid_ = false;
}

void CacheScheduleMetrics::enqueue(int64_t time) {
    if (recording() && !te_) {
        te_ = time;
    }
}

std::optional<uint64_t> CacheScheduleMetrics::checkedRound(const SchedulerRoundContext* round) {
    if (round == nullptr || round->owner_id != owner_ || round->round_id == 0) {
        return std::nullopt;
    }
    return round->round_id;
}

void CacheScheduleMetrics::canRun(int64_t time, const SchedulerRoundContext* round) {
    if (!recording() || tc_) {
        return;
    }
    tc_ = time;
    rc_ = checkedRound(round);
}

void CacheScheduleMetrics::loadingStart(int64_t time) {
    if (!recording()) {
        return;
    }
    if (!ts_) {
        ts_ = time;
    }
}

void CacheScheduleMetrics::loadingDone(int64_t time) {
    if (recording() && !td_) {
        td_ = time;
    }
}

void CacheScheduleMetrics::absorb(bool has_async_cache_dependency) {
    if (!recording()) {
        return;
    }
    has_async_cache_dependency_ |= has_async_cache_dependency;
}

void CacheScheduleMetrics::beginAttempt(bool has_async_cache_dependency, bool success) {
    if (!recording()) {
        return;
    }
    loading_interval_valid_ &= !had_attempt_ && success;
    had_attempt_ = true;
    absorb(has_async_cache_dependency);
}

void CacheScheduleMetrics::observe(const std::optional<CacheLoadTerminalSnapshot>& terminal, int64_t time) {
    if (!recording()) {
        return;
    }
    if (terminal) {
        absorb(terminal->has_async_cache_dependency);
    }
    if (to_) {
        return;
    }
    to_                         = time;
    const bool terminal_success = terminal && terminal->terminal_time_us && terminal->success;
    if (terminal_success) {
        tb_ = terminal->terminal_time_us;
    } else {
        loading_interval_valid_ = false;
    }
}

void CacheScheduleMetrics::running(int64_t time, const SchedulerRoundContext* round) {
    if (!recording()) {
        return;
    }
    tr_ = time;
    rr_ = checkedRound(round);
    freeze();
}

void CacheScheduleMetrics::freeze() {
    CacheScheduleSnapshot result;
    result.has_async_cache_dependency = has_async_cache_dependency_;
    auto ordered                      = [](const std::optional<int64_t>& start, const std::optional<int64_t>& end) {
        return start && end && *start >= 0 && *end >= *start;
    };
    if (!invalid_) {
        // Each metric owns its validity and denominator. Missing RUNNING must
        // not discard a completed enqueue or loading interval.
        if (ordered(te_, tc_)) {
            result.enqueue_to_canrun_us = *tc_ - *te_;
        }
        if (ordered(tc_, tr_)) {
            result.canrun_to_running_us = *tr_ - *tc_;
        }
        if (ordered(tc_, tr_) && rc_ && rr_ && *rr_ >= *rc_) {
            result.schedule_rounds = *rr_ - *rc_ + 1;
        }
        if (loading_interval_valid_ && tb_) {
            if (ordered(ts_, td_)) {
                result.loading_latency_us = *td_ - *ts_;
            }
            if (ordered(td_, tr_)) {
                result.load_done_to_running_us = *tr_ - *td_;
            }
            if (ordered(ts_, to_) && ordered(tb_, to_)) {
                result.ready_wait_us = *to_ - std::max(*ts_, *tb_);
            }
        }
    }
    snapshot_ = result;
}

std::optional<CacheScheduleSnapshot> CacheScheduleMetrics::takeReport() {
    if (!active() || reported_) {
        return std::nullopt;
    }
    if (!frozen()) {
        freeze();
    }
    reported_ = true;
    return snapshot_;
}

}  // namespace rtp_llm
