#include "rtp_llm/cpp/normal_engine/PressureStepRecorder.h"
#include <algorithm>
#include <stdexcept>

namespace rtp_llm {
int64_t pressureClockNs(clockid_t clock) {
    timespec value{};
    if (clock_gettime(clock, &value) != 0) {
        throw std::runtime_error("pressure clock read failed");
    }
    return int64_t(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

PressureStepRecorder::PressureStepRecorder(size_t capacity): capacity_(capacity) {
    if (!capacity || capacity > kMaxCapacity) {
        throw std::invalid_argument("pressure recorder capacity must be in [1, 65536]");
    }
}

void PressureStepRecorder::enable(const std::string& session_id, size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session_id.empty() || session_id.size() > 256 || capacity > kMaxCapacity) {
        throw std::invalid_argument("invalid pressure session or recorder capacity");
    }
    if (enabled_) {
        if (session_id != session_id_ || (capacity && capacity != capacity_)) {
            throw std::invalid_argument("conflicting pressure diagnostics session");
        }
        return;
    }
    if (session_id == session_id_) {
        throw std::invalid_argument("pressure session already completed");
    }
    if (capacity) {
        capacity_ = capacity;
    }
    session_id_ = session_id;
    phase_.clear();
    records_.clear();
    active_.reset();
    sequence_ = step_id_ = 0;
    enabled_             = true;
}

bool PressureStepRecorder::disable(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session_id.empty() || session_id != session_id_) {
        return false;
    }
    enabled_ = false;
    active_.reset();
    return true;
}

void PressureStepRecorder::markPhase(const std::string& session_id, const std::string& phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || session_id != session_id_ || phase.empty() || phase.size() > 256) {
        throw std::invalid_argument("pressure phase requires matching active session and nonempty phase");
    }
    phase_ = phase;
}

std::pair<std::string, std::string> PressureStepRecorder::controlIdentity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {session_id_, phase_};
}

bool PressureStepRecorder::matchesControl(const std::string& session_id, const std::string& phase) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && session_id == session_id_ && phase == phase_;
}

uint64_t PressureStepRecorder::candidateId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ ? step_id_ + 1 : 0;
}

void PressureStepRecorder::begin(PressureStepRecord record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !record.step_id || !(record.context_batch + record.decode_batch) || !record.tokens) {
        return;
    }
    record.session_id = session_id_;
    record.phase      = phase_;
    record.sequence   = 0;  // Active state never advances the completed-record cursor.
    record.complete   = false;
    step_id_          = record.step_id;
    active_           = std::move(record);
}

void PressureStepRecorder::append(PressureStepRecord record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !record.step_id || !(record.context_batch + record.decode_batch) || !record.complete) {
        return;
    }
    record.session_id = session_id_;
    record.phase      = active_ ? active_->phase : phase_;
    record.sequence   = ++sequence_;
    step_id_          = record.step_id;
    records_.push_back(std::move(record));
    if (records_.size() > capacity_) {
        records_.pop_front();
    }
    active_.reset();
}

void PressureStepRecorder::abort() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_.reset();
}

PressureStepPage PressureStepRecorder::readLocked(uint64_t after_sequence, size_t limit) const {
    PressureStepPage page;
    page.available     = !session_id_.empty();
    page.next_sequence = after_sequence;
    page.active        = active_;
    page.lost          = !records_.empty() && after_sequence < records_.front().sequence - 1;
    limit              = std::min(limit, kPageCap);
    for (const auto& record : records_) {
        if (record.sequence > after_sequence && page.records.size() < limit) {
            page.records.push_back(record);
            page.next_sequence = record.sequence;
        }
    }
    return page;
}

PressureStepPage PressureStepRecorder::read(uint64_t after_sequence, size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return readLocked(after_sequence, limit);
}

PressureStepPage
PressureStepRecorder::read(const std::string& session_id, uint64_t after_sequence, size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session_id.empty() || session_id != session_id_) {
        return {};
    }
    return readLocked(after_sequence, limit);
}
}  // namespace rtp_llm
