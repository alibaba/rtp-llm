#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace autil {
class DataBuffer;
}

namespace navi_sched {

// Resolved runtime values. In particular, schedulerBlockSize is the effective
// logical scheduler block, not necessarily the physical cache allocation block.
struct TPUPrefillExecutionPolicyConfig {
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    int64_t pcpSize = 0;
    int64_t interleaveSize = 0;
    std::vector<int64_t> compileSizes;
    int64_t maxScheduledTokens = 0;
    int64_t longPrefillTokenThreshold = 0; // Zero disables the threshold.
    int64_t hashBlockSize = 0;
    int64_t schedulerBlockSize = 0;
    // Effective trailing-block-drop/back-off flag; r50 MTP may keep lookahead
    // enabled while this is false. External lookup has its own flag below.
    bool useEagle = false;
    bool mambaHasPrefillCheckpointBlocks = false;
    bool mambaPartialCacheHit = false;
    bool asyncKVLookupEnabled = false;
    int64_t asyncKVLookupHashBlockSize = 0;
    bool asyncKVLookupUsesEagle = false;
    // Immutable fixed-plan policy. Runtime queue sizes are deliberately absent.
    bool fixedPrefillChunkPlan = false;
    bool mambaBlockAlignedSplit = true;
    int64_t attentionScheduleCapacity = 0;
    int64_t attentionQueryTile = 0;
    int64_t attentionKvTile = 0;
    int64_t prefillLookahead = 0;
    int64_t maxInputTokens = 0; // Zero selects maxScheduledTokens.
    int64_t draftSlotsPerRequest = 0;
    int64_t inputTokenBudget() const
    {
        return maxInputTokens > 0 ? maxInputTokens : maxScheduledTokens;
    }
};

// Diagnostic snapshot captured under the same pool lock as cache matching.
// It never participates in geometry scoring or serialized model features.
struct TPUPrefillCacheObservation {
    uint64_t revision = 0;
    bool stateReady = false;
    uint64_t cachedBlocks = 0;
    uint64_t fullReadyBlocks = 0;
    uint64_t tailReadyBlocks = 0;
    uint64_t cachedChecksum = 0;
    uint64_t fullReadyChecksum = 0;
    uint64_t tailReadyChecksum = 0;
};

struct TPUPrefillCacheObservationBatch {
    std::optional<TPUPrefillCacheObservation> pool = std::nullopt;
    // Includes pending-cache forecast, before query drop/tail intersection.
    std::vector<int64_t> fullPrefixBlocksBeforeQueryDrop;
};

struct TPUPrefillRequestInput {
    int64_t promptTokens = 0;
    int64_t computedTokens = 0;
    int64_t sharedPrefixBoundary = 0;
    // Resolved V6D reads, excluding resident block-pool hits. Runtime-only.
    int64_t externalReadTokens = -1;
    // Explicit runtime coordinates for resumed prefill/MTP. Zero uses prompt
    // coordinates, as in a newly submitted Master prefill request.
    int64_t prefillEndTokens = 0;
    int64_t numTokens = 0;
    int64_t prefillEnd() const
    {
        return prefillEndTokens > 0 ? prefillEndTokens : promptTokens;
    }
    int64_t tokenEnd() const { return numTokens > 0 ? numTokens : promptTokens; }
};

struct TPUPrefillStepGeometry {
    int64_t scheduledTokens = 0;
    int64_t maxLocalTokens = 0;
    int64_t localPaddedTokens = 0;
    int64_t globalPaddedTokens = 0;
};

struct TPUPrefillScheduledRequest {
    size_t requestIndex = 0;
    int64_t computedStart = 0;
    int64_t scheduledTokens = 0;
};

struct TPUPrefillExecutionStep {
    std::vector<TPUPrefillScheduledRequest> requests;
    TPUPrefillStepGeometry geometry;
};

struct TPUPrefillExecutionProjection {
    int64_t stepCount = 0;
    int64_t globalPaddedTokenSum = 0;
    std::vector<TPUPrefillExecutionStep> steps;
    int64_t attentionPairSum = 0;
};

// Pure arithmetic only: these functions send no requests and change no runtime
// scheduler state. Every failure clears the output; no partial plan is usable.
class TPUPrefillExecutionPolicy {
public:
    static int64_t asyncLookupPromptBoundary(
        const TPUPrefillExecutionPolicyConfig &config)
    {
        return config.asyncKVLookupEnabled && config.asyncKVLookupHashBlockSize > 0
                   ? config.asyncKVLookupHashBlockSize -
                         (config.asyncKVLookupUsesEagle ? 0 : 1)
                   : std::numeric_limits<int64_t>::max();
    }

    static bool hasAsyncLookupOpportunity(int64_t promptTokens,
                                          const TPUPrefillExecutionPolicyConfig &config)
    {
        return promptTokens > asyncLookupPromptBoundary(config);
    }

    // One immutable request chunk. Batch token/attention space may defer this
    // whole chunk, but never changes its length.
    static bool planNextPrefillChunk(const TPUPrefillRequestInput &request,
                                     const TPUPrefillExecutionPolicyConfig &config,
                                     int64_t &tokens, int64_t &attentionPairs,
                                     std::string *error = nullptr);
    static bool countAttentionPairs(int64_t tokens, int64_t prefix,
                                    const TPUPrefillExecutionPolicyConfig &config,
                                    int64_t &pairs, std::string *error = nullptr);

    static bool validate(const TPUPrefillExecutionPolicyConfig &config,
                         std::string *error = nullptr);

    // Model features only: async lookup-eligible requests form independent
    // groups; requests without a lookup share one group. This is a forecast,
    // never an instruction to change the runtime scheduler's ready queues.
    static bool projectLatencyFeatures(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error = nullptr);

    // ONE runner chunk. Request-major query spans use cumulative owner offsets:
    // [0, q0, q0+q1, ...], so the rank maximum depends on their total. Ownership
    // resets between runner chunks, never independently for each request.
    // The optional DP override must be resolved by the caller, not guessed.
    static bool projectStepGeometry(const std::vector<int64_t> &requestScheduledTokens,
                                    const TPUPrefillExecutionPolicyConfig &config,
                                    TPUPrefillStepGeometry &output,
                                    std::optional<int64_t> dpTargetBucket = std::nullopt,
                                    std::string *error = nullptr);

    // Prefill-only ALL-CACHE-RESOLVED hypothesis in the supplied request order.
    // num_tokens is promptTokens; this does not model decode or resumed output.
    // Members start waiting in supplied order; admission preserves the waiting
    // head and admitted members retain running priority in subsequent rounds.
    // External queue state, actual async-KV eligibility and scheduler partial-
    // prefill constraints are not inferred. A forecast is not an execution plan.
    //
    // maxProjectionSteps is solely a computational resource bound. The caller
    // must derive it from runtime model/request bounds and step configuration;
    // there is deliberately no hard-coded default, environment setting, or
    // influence on request execution. Exhaustion fails without partial geometry.
    static bool projectCacheResolvedBatch(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error = nullptr);
    // Aggregate-only scoring uses the same arithmetic without materializing steps.
    static bool projectLatencyFeaturesSummary(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error = nullptr);
    static bool projectCacheResolvedBatchSummary(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error = nullptr);

private:
    static bool projectLatencyFeaturesImpl(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error, bool collectSteps);
    static bool projectCacheResolvedBatchImpl(
        const std::vector<TPUPrefillRequestInput> &orderedRequests,
        const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
        TPUPrefillExecutionProjection &output, std::string *error, bool collectSteps);
};

} // namespace navi_sched
