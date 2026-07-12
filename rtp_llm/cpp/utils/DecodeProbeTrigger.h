#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace rtp_llm {

struct DecodeProbeTriggerEvent {
    uint64_t    generation{0};
    uint64_t    timestamp_us{0};
    int64_t     observed_sequence_length{-1};
    std::string trace_id;
    std::string reason;
    uint64_t    required_rank_mask{0};
    uint64_t    ready_rank_mask{0};
    uint64_t    ack_rank_mask{0};
    uint64_t    failure_rank_mask{0};
};

namespace detail {

constexpr uint32_t kDecodeProbeTriggerRecordMagic   = 0x44505452;
constexpr uint32_t kDecodeProbeTriggerRecordVersion = 2;

struct DecodeProbeTriggerSharedRecord {
    uint32_t              magic{0};
    uint32_t              version{0};
    std::atomic<uint64_t> generation{0};
    uint64_t              timestamp_us{0};
    int64_t               observed_sequence_length{-1};
    char                  trace_id[256]{};
    char                  reason[64]{};
    uint64_t              required_rank_mask{0};
    std::atomic<uint64_t> ready_rank_mask{0};
    std::atomic<uint64_t> ack_rank_mask{0};
    std::atomic<uint64_t> failure_rank_mask{0};
};

}  // namespace detail

// A directly configurable mapping used by tests and specialized callers.
class DecodeProbeTriggerRegistry {
public:
    DecodeProbeTriggerRegistry(const char* shm_name,
                               bool        enabled,
                               uint64_t    expiry_us = 30ULL * 1000ULL * 1000ULL) noexcept;
    ~DecodeProbeTriggerRegistry() noexcept;

    DecodeProbeTriggerRegistry(const DecodeProbeTriggerRegistry&)            = delete;
    DecodeProbeTriggerRegistry& operator=(const DecodeProbeTriggerRegistry&) = delete;

    bool publish(const DecodeProbeTriggerEvent& event) noexcept;
    bool peek(DecodeProbeTriggerEvent& event) const noexcept;
    bool arrive(uint64_t generation, uint32_t rank) noexcept;
    bool acknowledge(uint64_t generation, uint32_t rank, bool failed = false) noexcept;
    bool enabled() const noexcept;

private:
    bool initialize(const char* shm_name) noexcept;
    void disableUnlocked() noexcept;

    mutable std::mutex                       mutex_;
    int                                      fd_{-1};
    detail::DecodeProbeTriggerSharedRecord* record_{nullptr};
    uint64_t                                 expiry_us_{0};
    bool                                     enabled_{false};
};

// Production facade. It is disabled unless RTPLLM_RETROSPECTIVE_PROBE_DEBUG is true.
class DecodeProbeTrigger {
public:
    static bool publish(const DecodeProbeTriggerEvent& event) noexcept;
    static bool peek(DecodeProbeTriggerEvent& event) noexcept;
    static bool arrive(uint64_t generation) noexcept;
    static bool acknowledge(uint64_t generation, bool failed = false) noexcept;
    static bool enabled() noexcept;
};

}  // namespace rtp_llm
