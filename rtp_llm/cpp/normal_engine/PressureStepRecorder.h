#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

namespace rtp_llm {

int64_t pressureClockNs(clockid_t clock);

struct PressureStepRecord {
    uint64_t                 sequence{0}, step_id{0};
    int64_t                  begin_ns{0}, forward_enter_ns{0}, forward_return_ns{0}, end_ns{0};
    int64_t                  forward_enter_cpu_ns{0}, forward_return_cpu_ns{0};
    uint64_t                 context_batch{0}, decode_batch{0}, tokens{0};
    std::vector<std::string> request_ids;
    std::vector<int64_t>     stream_ids;
    bool                     complete{false};
    std::string              session_id, phase;
};

struct PressureStepPage {
    std::vector<PressureStepRecord>   records;
    uint64_t                          next_sequence{0};
    bool                              lost{false};
    bool                              available{false};
    std::optional<PressureStepRecord> active;
};

class PressureStepRecorder {
public:
    static constexpr size_t kMaxCapacity = 65536;
    static constexpr size_t kPageCap     = 1024;
    explicit PressureStepRecorder(size_t capacity);
    void enable(const std::string& session_id, size_t capacity = 0);
    bool disable(const std::string& session_id);
    void markPhase(const std::string& session_id, const std::string& phase);
    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }
    uint64_t                            candidateId() const;
    std::pair<std::string, std::string> controlIdentity() const;
    bool                                matchesControl(const std::string& session_id, const std::string& phase) const;
    void                                begin(PressureStepRecord record);
    void                                append(PressureStepRecord record);
    void                                abort();
    PressureStepPage                    read(uint64_t after_sequence, size_t limit) const;
    PressureStepPage read(const std::string& session_id, uint64_t after_sequence, size_t limit) const;

private:
    PressureStepPage                  readLocked(uint64_t after_sequence, size_t limit) const;
    mutable std::mutex                mutex_;
    size_t                            capacity_;
    std::atomic<bool>                 enabled_{false};
    std::string                       session_id_, phase_;
    uint64_t                          sequence_{0}, step_id_{0};
    std::deque<PressureStepRecord>    records_;
    std::optional<PressureStepRecord> active_;
};
}  // namespace rtp_llm
