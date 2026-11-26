#pragma once

#include <deque>
#include <map>
#include <optional>
#include <vector>

#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"
#include "aios/navi_sched/common/TPUPrefillPP4Timing.h"

namespace navi_sched {

struct TPUPrefillPP4HostResult {
    // Native _Step.slot determines compute identity, independently of whether
    // its execute and sample Futures alias. Only compute results have ordinals.
    std::optional<int64_t> slotOrdinal;
    bool ready = false;
    // Predicted control-call readiness; used only when slotOrdinal is absent.
    double controlReadyTimeMs = 0.0;
};

struct TPUPrefillPP4ForecastSpan {
    uint64_t requestId = 0;
    int64_t computedTokens = 0;
    int64_t scheduledTokens = 0;
};

// A real SchedulerOutput already chosen by Native, before its host dispatch.
// Its work is reserved in the request cursor; it is never scheduled a second time.
struct TPUPrefillPP4PendingDispatch {
    double dispatchTimeMs = 0.0;
    double expectedForwardMs = 0.0;
    double controlReadyTimeMs = 0.0;
    std::array<double, 8> features{};
    std::vector<TPUPrefillPP4ForecastSpan> spans;
};

struct TPUPrefillPP4State {
    double nowMs = 0.0;
    double measuredForwardMs = 0.0;
    int64_t lastRealOrdinal = -1;
    std::vector<TPUPrefillPipelineSlot> slots;
    std::map<int64_t, double> expectedForwardMs;
    TPUPrefillPP4Timing::ReplyTimes observedReplies;
    TPUPrefillPP4Timing::ReplyTimes replyLowerBounds;
    std::deque<TPUPrefillPP4HostResult> results;
    std::optional<TPUPrefillPP4PendingDispatch> pendingDispatch;
    // Query-service features are evaluated at the simulated Native issue time:
    // [1, log1p(own blocks), active count, log1p(active blocks),
    // log1p(sum of active query ages in ms)]. No worker-load/FIFO time here.
    std::optional<std::array<double, 5>> queryServiceParameters;
    int64_t maxRunningRequestCount = 0;
    // Non-prefill/paused requests occupy Native request slots too. These are
    // explicit predicted release events, not additional PP compute steps.
    std::vector<double> otherRequestReleaseTimesMs;
    // Final prefill results for already scheduled requests. They release a
    // Native slot only when that request permits release after prefill.
    // Ordinals may refer to the pending decided dispatch.
    std::map<uint64_t, int64_t> requestCompletionSlots;
};

enum class TPUPrefillForecastQueue { RUNNING, SKIPPED_WAITING, WAITING };

struct TPUPrefillKVReceiveDependency {
    int64_t carrierOrdinal = -1;
    std::optional<int64_t> reporterOrdinal;
    // A pending query cannot attach new load metadata to an already decided
    // SchedulerOutput. Its next carrier must be beyond this immutable fence.
    std::optional<int64_t> decidedProposalFence;
};

// KV-load metadata can accompany compute or a control-only call. localLoadMs
// starts at its carrier dispatch and ends at worker load completion. A later
// Native result still has to deliver that notification. controlReplyMs predicts
// control-only polling when compute is unavailable, never a PP handoff.
struct TPUPrefillMetadataKVLoad {
    double localLoadMs = 0.0;
    double controlReplyMs = 0.0;
};

struct TPUPrefillPP4Request {
    uint64_t requestId = 0;
    int64_t contextTokens = 0;
    int64_t remainingTokens = 0;
    int64_t reservedTokens = 0;
    int64_t sharedPrefixBoundary = 0;
    TPUPrefillForecastQueue queue = TPUPrefillForecastQueue::WAITING;
    bool owned = false;
    // A completed prefill can pause a resumable session or continue decoding.
    // Its Native request slot is released here only with an explicit true fact.
    std::optional<bool> releaseSlotAfterPrefill;
    bool needsFirstQuery = false;
    // With an unassigned load carrier, this is query readiness. Otherwise
    // it is cache readiness; an issued load still waits for its FIFO reporter.
    double readyTimeMs = 0.0;
    std::optional<TPUPrefillKVReceiveDependency> receiveDependency;
    // Predicted notification in the original result FIFO, including metadata-only
    // calls. The pending decided output, if any, has index results.size().
    std::optional<size_t> receiveHostResultIndex;
    // Timing also applies when Native attaches this load to a compute output.
    std::optional<TPUPrefillMetadataKVLoad> metadataOnlyLoad;
    // Service begins when Native first polls this unissued query. Until then
    // readyTimeMs is its earliest issue/arrival gate, not a completion deadline.
    // An absent value retains the existing absolute-readiness input.
    std::optional<double> firstQueryServiceMs;
    // Exact observed query arguments or a caller-supplied causal estimate.
    // Total predicted cache hit is not the physical HBM query prefix.
    std::optional<int64_t> queryBlockCount;
    // Only for a query already issued and pending at observation. Its ready
    // completion is separate from readyTimeMs (which may include a KV load).
    // Both fields are required together; no query is reissued.
    std::optional<double> pendingQueryAgeMs;
    std::optional<double> pendingQueryReadyTimeMs;
};

struct TPUPrefillPP4ForecastDispatch {
    int64_t ordinal = 0;
    double timeMs = 0.0;
    bool real = false;
    std::vector<TPUPrefillPP4ForecastSpan> spans;
    // Requests whose pending load metadata is attached to this real output.
    std::vector<uint64_t> loadMetadataRequestIds;
    std::vector<uint64_t> finishedLoadRequestIds;
};

struct TPUPrefillPP4ForecastOutput {
    double queueMs = 0.0;
    double executionMs = 0.0;
    double totalMs = 0.0;
    size_t realSteps = 0;
    size_t ownedSteps = 0;
    size_t realPrefixSteps = 0;
    std::map<uint64_t, double> firstScheduledTimeMs;
    std::map<uint64_t, int64_t> scheduledTokens;
    std::vector<TPUPrefillPP4ForecastDispatch> dispatches;
    struct ControlDispatch {
        double timeMs = 0.0;
        double readyTimeMs = 0.0;
        int64_t ppOrdinal = -1;
        std::vector<uint64_t> loadRequestIds;
        std::vector<uint64_t> finishedLoadRequestIds;
    };
    std::vector<ControlDispatch> controlDispatches;
    // Allocated only when captureDispatches is requested.
    std::map<uint64_t, double> firstQueryTimeMs;
    std::map<uint64_t, double> firstQueryReadyTimeMs;
    std::map<uint64_t, std::array<double, 5>> firstQueryServiceFeatures;
};

// A forecast of continuous PP4 prefill, never an instruction to the engine.
// Inputs are one immutable observation plus predicted cache-ready events.
// Each call owns its request/FIFO copy. The caller derives maxTransitions from
// request/policy bounds; exhaustion throws rather than returning a partial plan.
class TPUPrefillPP4Forecast {
public:
    static TPUPrefillPP4ForecastOutput predict(
        const TPUPrefillPP4State &state,
        const std::vector<TPUPrefillPP4Request> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &policy,
        const TPUPrefillPP4Timing::Parameters &parameters, size_t maxTransitions,
        bool captureDispatches = false);
};

} // namespace navi_sched
