#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace autil {
class DataBuffer;
}

namespace navi_sched {

// One immutable SchedulerOutput span, before Native advances that request again.
struct NaviSchedNativePPSpan {
    uint64_t requestId = 0;
    int64_t computedTokens = 0;
    int64_t scheduledTokens = 0;
    bool valid() const;
    bool operator==(const NaviSchedNativePPSpan &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

struct NaviSchedNativePPSlot {
    int64_t ordinal = 0;
    int64_t dispatchAgeUs = 0;
    double expectedForwardMs = 0.0;
    // -1 means the last-stage host reply has not been observed. This is not a
    // worker execution clock, and a ready Future does not invent this timestamp.
    int64_t lastStageReplyAgeUs = -1;
    std::array<double, 8> features{};
    std::vector<NaviSchedNativePPSpan> spans;
    bool valid() const;
    bool operator==(const NaviSchedNativePPSlot &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

struct NaviSchedNativePPResult {
    uint64_t schedulerOutputId = 0;
    int64_t dispatchAgeUs = 0;
    // Native _Step.slot, not execute/sample Future aliasing, defines compute.
    // -1 denotes an actual metadata-only call; it still occupies FIFO capacity.
    int64_t slotOrdinal = -1;
    bool executeReplyReady = false;
    bool resultReady = false;
    // Sorted request IDs whose KV load metadata this exact Native call carries.
    std::vector<uint64_t> loadRequestIds;
    bool valid() const;
    bool operator==(const NaviSchedNativePPResult &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

enum class NativePPQueue : int32_t {
    RUNNING = 0,
    SKIPPED_WAITING = 1,
    WAITING = 2,
    NOT_QUEUED = 3,
    PREPARED = 4
};

enum class NativePPRequestPhase : int32_t {
    UNKNOWN = 0,
    RUNNING = 1,
    PREEMPTED = 2,
    KV_LOADING = 3,
    LOOKUP_SKIPPED = 4,
    QUERY_PENDING = 5,
    QUERY_READY_NOT_CONSUMED = 6,
    QUERY_CONSUMED = 7,
    QUERY_NOT_STARTED = 8,
    CACHE_KNOWN = 9,
    CACHE_UNKNOWN = 10,
    NOT_REGISTERED = 11,
};

enum class NativePPPrefillSlotRelease : int32_t {
    UNKNOWN = 0,
    AFTER_PREFILL_RESULT = 1,
    RETAINS_SLOT = 2,
};

inline bool validNativePrefillSlotRelease(NativePPPrefillSlotRelease value)
{
    return value == NativePPPrefillSlotRelease::UNKNOWN ||
           value == NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT ||
           value == NativePPPrefillSlotRelease::RETAINS_SLOT;
}

struct NaviSchedNativePPRequest {
    uint64_t requestId = 0;
    NativePPQueue queue = NativePPQueue::NOT_QUEUED;
    int64_t queueIndex = -1;
    NativePPRequestPhase phase = NativePPRequestPhase::UNKNOWN;
    int64_t promptTokens = 0;
    // Native's raw cursor includes already scheduled, unfinished spans. The
    // predictor subtracts reservations from those immutable spans, once.
    int64_t computedTokens = 0;
    int64_t cacheHitTokens = -1;
    int64_t sharedPrefixBoundary = 0;
    int64_t queryAgeUs = -1;
    // Time since the Native call carrying KV load metadata was dispatched,
    // not V6D's query/prepare timer. -1 until dispatch is observed for this
    // request's current query generation; worker completion is a separate fact.
    int64_t loadAgeUs = -1;
    NativePPPrefillSlotRelease prefillSlotRelease = NativePPPrefillSlotRelease::UNKNOWN;
    bool valid() const;
    bool operator==(const NaviSchedNativePPRequest &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

struct NaviSchedNativePPDecision {
    uint64_t schedulerOutputId = 0;
    std::array<double, 8> features{};
    std::vector<NaviSchedNativePPSpan> spans;
    std::vector<uint64_t> loadRequestIds;
    bool valid() const;
    bool operator==(const NaviSchedNativePPDecision &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

// Rank0 EngineCore PP4 state. All ages share the original observation boundary;
// WorkloadSnapshot carries its age at reporting separately. No cross-host
// monotonic timestamp, fixed execution policy, tensor or block hash is sent here.
// Fixed stages/padding/splitting policy belongs to the connection handshake.
struct NaviSchedNativePPState {
    uint64_t observationId = 0;
    uint64_t burst = 0;
    bool waveOpen = false;
    int64_t lastRealOrdinal = -1;
    double measuredForwardMs = 0.0;
    // Native admission counts running requests and paused streaming sessions.
    // Prepared requests have not entered Native and do not occupy these slots.
    int64_t occupiedRequestSlotCount = -1;
    std::vector<NaviSchedNativePPSlot> slots;
    std::vector<NaviSchedNativePPResult> results;
    // Native queue order, independent of request-id order.
    std::vector<NaviSchedNativePPRequest> requests;
    // A SchedulerOutput can already be decided when a schedule-reason workload
    // report is emitted, before Native has assigned it a slot or sent it.
    std::optional<NaviSchedNativePPDecision> pendingDecision;
    bool valid() const;
    bool operator==(const NaviSchedNativePPState &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
};

} // namespace navi_sched
