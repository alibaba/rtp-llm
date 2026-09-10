#include "rtp_llm/cpp/engine_base/stream/CacheScheduleProbe.h"

#include <algorithm>
#include <new>

namespace rtp_llm {

CacheScheduleProbe& CacheScheduleProbe::operator=(const CacheScheduleProbe& other) {
    if (this != &other) {
        this->~CacheScheduleProbe();
        new (this) CacheScheduleProbe();
    }
    return *this;
}

void CacheScheduleProbe::activate(uint64_t owner) {
    if (!active() && !reported_) {
        owner_ = owner;
    }
}

void CacheScheduleProbe::reset() {
    if (!recording()) {
        return;
    }
    reset_invalid_ = true;
    te_.reset();
    tc_.reset();
    ts_.reset();
    tb_.reset();
    to_.reset();
    td_.reset();
    tr_.reset();
    rc_.reset();
    rr_.reset();
}

void CacheScheduleProbe::enqueue(int64_t time) {
    if (recording() && !te_) {
        te_ = time;
    }
}

std::optional<uint64_t> CacheScheduleProbe::checkedRound(const SchedulerRoundContext* round) {
    if (round == nullptr || round->owner_id != owner_ || round->round_id == 0) {
        invalid_ = true;
        return std::nullopt;
    }
    return round->round_id;
}

void CacheScheduleProbe::canRun(int64_t time, const SchedulerRoundContext* round) {
    if (!recording() || tc_) {
        return;
    }
    tc_ = time;
    rc_ = checkedRound(round);
}

void CacheScheduleProbe::loadingStart(int64_t time) {
    if (!recording()) {
        return;
    }
    loading_entered_ = true;
    if (!ts_) {
        ts_ = time;
    }
}

void CacheScheduleProbe::loadingDone(int64_t time) {
    if (recording() && !td_) {
        td_ = time;
    }
}

void CacheScheduleProbe::absorb(const CacheDependencyEvidence& evidence) {
    if (!recording()) {
        return;
    }
    evidence_.all_sources_resolved = evidence.all_sources_resolved;
    evidence_.has_data_dependency |= evidence.has_data_dependency;
    evidence_.async_lookup_started |= evidence.async_lookup_started;
    evidence_.had_error_or_fallback |= evidence.had_error_or_fallback;
}

void CacheScheduleProbe::beginAttempt(const CacheDependencyEvidence& evidence, bool success) {
    if (!recording()) {
        return;
    }
    if (had_attempt_) {
        had_retry_ = true;
        unresolved_history_ |= !evidence_.all_sources_resolved;
    }
    had_attempt_ = true;
    had_retry_ |= !success;
    absorb(evidence);
}

void CacheScheduleProbe::observe(const std::optional<CacheLoadTerminalSnapshot>& terminal, int64_t time) {
    if (!recording()) {
        return;
    }
    if (terminal) {
        absorb(terminal->evidence);
    }
    if (to_) {
        return;
    }
    to_ = time;
    if (!terminal || !terminal->terminal || !terminal->terminal_time_us) {
        invalid_ = true;
        return;
    }
    terminal_success_ = terminal->success;
    tb_               = terminal->terminal_time_us;
    had_retry_ |= !terminal->success;
}

void CacheScheduleProbe::running(int64_t time, const SchedulerRoundContext* round) {
    if (!recording()) {
        return;
    }
    tr_ = time;
    rr_ = checkedRound(round);
    freeze(true);
}

void CacheScheduleProbe::freeze(bool reached_running) {
    CacheScheduleSnapshot result;
    auto                  evidence = evidence_;
    evidence.all_sources_resolved &= !unresolved_history_;
    result.dependency      = evidence.dependency();
    result.loading_entered = loading_entered_;
    if (!reached_running || reset_invalid_) {
        snapshot_ = result;
        return;
    }
    if (had_retry_ || evidence.had_error_or_fallback) {
        result.sample_status = CacheSampleStatus::RECOVERED;
        snapshot_            = result;
        return;
    }
    bool valid = !invalid_ && result.dependency != CacheDependency::UNKNOWN && te_ && tc_ && tr_ && rc_ && rr_;
    if (valid) {
        valid = *te_ >= 0 && *te_ <= *tc_ && *tc_ <= *tr_ && *rr_ >= *rc_;
    }
    if (valid && loading_entered_) {
        valid = ts_ && tb_ && to_ && td_ && terminal_success_;
        if (valid) {
            valid = *tb_ >= 0 && *tc_ <= *ts_ && *ts_ <= *to_ && *to_ <= *td_ && *td_ <= *tr_
                    && *to_ >= std::max(*ts_, *tb_);
        }
    }
    if (valid) {
        result.enqueue_to_canrun_us = *tc_ - *te_;
        result.canrun_to_running_us = *tr_ - *tc_;
        result.schedule_rounds      = *rr_ - *rc_ + 1;
        if (loading_entered_) {
            result.loading_latency_us      = *td_ - *ts_;
            result.load_done_to_running_us = *tr_ - *td_;
            result.ready_wait_us           = *to_ - std::max(*ts_, *tb_);
        }
        valid = result.schedule_rounds >= 1 && result.ready_wait_us <= result.loading_latency_us
                && result.loading_latency_us + result.load_done_to_running_us <= result.canrun_to_running_us;
    }
    result.numeric_valid = valid;
    if (valid) {
        result.sample_status = CacheSampleStatus::SINGLE_PASS;
    }
    snapshot_ = result;
}

std::optional<CacheScheduleSnapshot> CacheScheduleProbe::takeReport() {
    if (!active() || reported_) {
        return std::nullopt;
    }
    if (!frozen()) {
        freeze(false);
    }
    reported_ = true;
    return snapshot_;
}

}  // namespace rtp_llm
