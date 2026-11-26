#pragma once

#include <map>
#include <optional>

#include "aios/navi_sched/common/TPUPrefillPP4Forecast.h"
#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedNativePPState.h"

namespace navi_sched {

struct TPUPrefillPP4RequestPrediction {
    std::optional<int64_t> cacheHitTokens;
    // Prepared requests precede Native admission and prompt normalization.
    // Their caller supplies the normalized count rather than guessing a drop.
    std::optional<int64_t> nativePromptTokens;
    // Prepared work has not entered Native; the normalized request supplies this.
    std::optional<bool> releaseSlotAfterPrefill;
    std::optional<double> readyAfterObservationMs;
    std::optional<bool> needsFirstQuery;
    std::optional<TPUPrefillKVReceiveDependency> receiveDependency;
    std::optional<uint64_t> receiveSchedulerOutputId;
    std::optional<TPUPrefillMetadataKVLoad> metadataOnlyLoad;
    std::optional<double> firstQueryServiceMs;
    std::optional<int64_t> queryBlockCount;
    // Background Future completion only; excludes worker load and FIFO receipt.
    std::optional<double> queryReadyAfterObservationMs;
};

struct TPUPrefillPP4ObservationPredictions {
    std::optional<std::array<double, 5>> queryServiceParameters;
    std::map<uint64_t, TPUPrefillPP4RequestPrediction> requests;
    // Only unresolved metadata-only calls need a control readiness prediction.
    std::map<uint64_t, double> controlReadyAfterObservationMs;
    std::optional<double> pendingDispatchAfterObservationMs;
    // Normalized work not yet in Native: earlier unreported reservations have
    // owned=false, newly selected requests have owned=true. Order is preserved.
    // readyTimeMs is relative to the original observation, as above.
    std::vector<TPUPrefillPP4Request> incoming;
    std::vector<double> otherRequestReleaseAfterObservationMs;
};

struct TPUPrefillPP4ObservedCapacity {
    int64_t maxRunningRequestCount = 0;
    // Actual Native occupancy outside the observed prefill RUNNING rows,
    // including paused streaming sessions. Its release times are predictions.
    int64_t otherOccupiedRequestCount = 0;
};

struct TPUPrefillPP4ObservationInput {
    TPUPrefillPP4State state;
    std::vector<TPUPrefillPP4Request> requests;
};

// Convert one received observation without advancing its unfinished-Future
// lower bounds to the newer evaluation clock. Missing readiness/cache facts
// require explicit predictions; unknown cache is never silently changed to zero.
class TPUPrefillPP4Observation {
public:
    static TPUPrefillPP4ObservationInput prepare(
        const NaviSchedWorkloadSnapshot &workload,
        const TPUPrefillPP4ObservationPredictions &predictions);

    static TPUPrefillPP4ObservationInput prepare(
        const NaviSchedNativePPState &observed, int64_t observationAgeUs,
        const TPUPrefillPP4ObservedCapacity &capacity,
        const TPUPrefillPP4ObservationPredictions &predictions);

private:
    static TPUPrefillPP4ObservationInput prepare(
        const NaviSchedNativePPState &observed, int64_t observationAgeUs,
        const TPUPrefillPP4ObservedCapacity &capacity,
        const TPUPrefillPP4ObservationPredictions &predictions,
        const std::vector<NaviSchedNativeCacheQuery> *cacheQueries);
};

} // namespace navi_sched
