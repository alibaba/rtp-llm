#include <autil/DataBuffer.h>
#include <algorithm>
#include <limits>
#include <utility>

#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"

namespace navi_sched {

void TPUPrefillExecutionPolicyConfig::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(pcpSize);
    dataBuffer.write(interleaveSize);
    dataBuffer.write(compileSizes);
    dataBuffer.write(maxScheduledTokens);
    dataBuffer.write(longPrefillTokenThreshold);
    dataBuffer.write(hashBlockSize);
    dataBuffer.write(schedulerBlockSize);
    dataBuffer.write(useEagle);
    dataBuffer.write(mambaHasPrefillCheckpointBlocks);
    dataBuffer.write(mambaPartialCacheHit);
    dataBuffer.write(asyncKVLookupEnabled);
    dataBuffer.write(asyncKVLookupHashBlockSize);
    dataBuffer.write(asyncKVLookupUsesEagle);
    dataBuffer.write(fixedPrefillChunkPlan);
    dataBuffer.write(mambaBlockAlignedSplit);
    dataBuffer.write(attentionScheduleCapacity);
    dataBuffer.write(attentionQueryTile);
    dataBuffer.write(attentionKvTile);
    dataBuffer.write(prefillLookahead);
    dataBuffer.write(maxInputTokens);
    dataBuffer.write(draftSlotsPerRequest);
}

void TPUPrefillExecutionPolicyConfig::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(pcpSize);
    dataBuffer.read(interleaveSize);
    dataBuffer.read(compileSizes);
    dataBuffer.read(maxScheduledTokens);
    dataBuffer.read(longPrefillTokenThreshold);
    dataBuffer.read(hashBlockSize);
    dataBuffer.read(schedulerBlockSize);
    dataBuffer.read(useEagle);
    dataBuffer.read(mambaHasPrefillCheckpointBlocks);
    dataBuffer.read(mambaPartialCacheHit);
    dataBuffer.read(asyncKVLookupEnabled);
    dataBuffer.read(asyncKVLookupHashBlockSize);
    dataBuffer.read(asyncKVLookupUsesEagle);
    dataBuffer.read(fixedPrefillChunkPlan);
    dataBuffer.read(mambaBlockAlignedSplit);
    dataBuffer.read(attentionScheduleCapacity);
    dataBuffer.read(attentionQueryTile);
    dataBuffer.read(attentionKvTile);
    dataBuffer.read(prefillLookahead);
    dataBuffer.read(maxInputTokens);
    dataBuffer.read(draftSlotsPerRequest);
}

namespace {

bool fail(std::string *error, const char *message)
{
    if (error) {
        *error = message;
    }
    return false;
}

bool checkedAdd(int64_t left, int64_t right, int64_t &result)
{
    if (left < 0 || right < 0 || right > std::numeric_limits<int64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checkedMultiply(int64_t left, int64_t right, int64_t &result)
{
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    result = left * right;
    return true;
}

// Mirrors the captured runtime _mamba_block_aligned_split for prefill requests
// (num_tokens == num_prompt_tokens). Config values determine every boundary.
int64_t clipPrefillChunk(const TPUPrefillRequestInput &request, int64_t newTokens,
                         const TPUPrefillExecutionPolicyConfig &config,
                         int64_t requestTokenLimit = 0)
{
    const auto start = request.computedTokens;
    const auto block = config.schedulerBlockSize;
    auto end = start + newTokens; // Caller bounds newTokens by prompt-start.
    auto lastCachePosition = request.tokenEnd() / block * block;
    if (config.useEagle) {
        lastCachePosition = std::max<int64_t>(lastCachePosition - block, 0);
    }
    const auto internalCheckpoint =
        config.mambaHasPrefillCheckpointBlocks && start % block == 0;
    if (internalCheckpoint) {
        lastCachePosition = 0;
    }
    if (end < request.prefillEnd() && !internalCheckpoint) {
        auto maxPrefillTokens = config.maxScheduledTokens;
        if (config.longPrefillTokenThreshold > 0) {
            maxPrefillTokens =
                std::min(maxPrefillTokens, config.longPrefillTokenThreshold);
        }
        if (requestTokenLimit > 0) {
            maxPrefillTokens = std::min(maxPrefillTokens, requestTokenLimit);
        }
        const auto alignedEnd = end / block * block;
        if (alignedEnd > start || block <= maxPrefillTokens) {
            end = alignedEnd;
        }
    }
    const auto proposedEnd = end;
    const auto considerStop = [&](int64_t stop) {
        if (start < stop && stop < end) {
            end = stop;
        }
    };
    const auto remainder = start % block;
    if (remainder != 0) {
        const auto distance = block - remainder;
        // A mathematical next boundary beyond INT64_MAX cannot be inside end.
        if (distance < end - start) {
            considerStop(start + distance);
        }
    }
    considerStop(lastCachePosition);
    const auto tailBoundary =
        config.mambaPartialCacheHit
            ? request.promptTokens / config.hashBlockSize * config.hashBlockSize
            : 0;
    if (lastCachePosition < tailBoundary && tailBoundary < request.promptTokens) {
        considerStop(tailBoundary);
    }
    if (start < request.sharedPrefixBoundary &&
        request.sharedPrefixBoundary < proposedEnd) {
        considerStop(start + (request.sharedPrefixBoundary - start) / block * block);
    }
    return std::max<int64_t>(end - start, 0);
}

// Callers validate the policy and bound the positive total before entering.
bool projectStepGeometryTotal(int64_t total,
                              const TPUPrefillExecutionPolicyConfig &config,
                              TPUPrefillStepGeometry &output,
                              std::optional<int64_t> dpTargetBucket, std::string *error)
{
    const auto cycle = config.pcpSize * config.interleaveSize;
    const auto cycles = total / cycle;
    const auto remainder = total % cycle;
    int64_t required = 0;
    if (!checkedMultiply(cycles, config.interleaveSize, required) ||
        !checkedAdd(required, std::min(remainder, config.interleaveSize), required)) {
        return fail(error, "PCP rank token count exceeds int64 range");
    }
    const auto bucket = std::lower_bound(config.compileSizes.begin(),
                                         config.compileSizes.end(), required);
    if (bucket == config.compileSizes.end()) {
        return fail(error, "PCP rank work exceeds configured compile buckets");
    }
    auto padded = *bucket;
    if (dpTargetBucket) {
        if (*dpTargetBucket <= 0 || *dpTargetBucket > config.compileSizes.back()) {
            return fail(error, "invalid or unavailable DP target bucket");
        }
        padded = std::max(padded, *dpTargetBucket);
    }
    int64_t globalPadded = 0;
    if (!checkedMultiply(config.pcpSize, padded, globalPadded)) {
        return fail(error, "global padded token count exceeds int64 range");
    }
    output = {total, required, padded, globalPadded};
    return true;
}

// Exact Euclidean floor sum; 128-bit intermediates cover int64 coordinates.
using Wide = unsigned __int128;
Wide floorSum(Wide n, Wide m, Wide a, Wide b)
{
    if (n == 0) {
        return 0;
    }
    Wide result = 0;
    while (true) {
        if (a >= m) {
            result += n * (n - 1) / 2 * (a / m);
            a %= m;
        }
        if (b >= m) {
            result += n * (b / m);
            b %= m;
        }
        const auto top = a * n + b;
        if (top < m) {
            return result;
        }
        n = top / m;
        b = top % m;
        std::swap(m, a);
    }
}

} // namespace

bool TPUPrefillExecutionPolicy::validate(const TPUPrefillExecutionPolicyConfig &config,
                                         std::string *error)
{
    if (error) {
        error->clear();
    }
    if (config.pcpSize <= 0 || (config.pcpSize == 1 && !config.fixedPrefillChunkPlan) ||
        config.interleaveSize <= 0 || config.maxScheduledTokens <= 0 ||
        config.longPrefillTokenThreshold < 0 || config.hashBlockSize <= 0 ||
        config.schedulerBlockSize <= 0 || config.prefillLookahead < 0 ||
        config.maxInputTokens < 0 || config.draftSlotsPerRequest < 0 ||
        config.draftSlotsPerRequest >= config.inputTokenBudget() ||
        config.compileSizes.empty()) {
        return fail(error, "missing or invalid resolved streaming PCP policy");
    }
    if (config.asyncKVLookupEnabled && config.asyncKVLookupHashBlockSize <= 0) {
        return fail(error, "async lookup hash block size must be positive");
    }
    if (config.fixedPrefillChunkPlan &&
        (config.longPrefillTokenThreshold <= 0 ||
         config.longPrefillTokenThreshold > config.maxScheduledTokens)) {
        return fail(error,
                    "fixed chunk cap must be positive and fit the step token budget");
    }
    const bool hasAttention = config.attentionScheduleCapacity != 0 ||
                              config.attentionQueryTile != 0 ||
                              config.attentionKvTile != 0;
    if (hasAttention && (config.attentionScheduleCapacity <= 0 ||
                         config.attentionQueryTile <= 0 || config.attentionKvTile <= 0)) {
        return fail(error, "incomplete attention schedule constraints");
    }
    int64_t previous = 0;
    for (auto bucket : config.compileSizes) {
        if (bucket <= previous) {
            return fail(error, "compile sizes must be strictly increasing and positive");
        }
        previous = bucket;
    }
    int64_t unused = 0;
    if (!checkedMultiply(config.pcpSize, config.interleaveSize, unused) ||
        !checkedMultiply(config.pcpSize, config.compileSizes.back(), unused)) {
        return fail(error, "PCP geometry exceeds int64 range");
    }
    return true;
}

bool TPUPrefillExecutionPolicy::countAttentionPairs(
    int64_t tokens, int64_t prefix, const TPUPrefillExecutionPolicyConfig &config,
    int64_t &pairs, std::string *error)
{
    pairs = 0;
    if (tokens < 0 || prefix < 0 ||
        tokens > std::numeric_limits<int64_t>::max() - prefix) {
        return fail(error, "invalid attention query coordinates");
    }
    if (tokens == 0 || config.attentionScheduleCapacity == 0) {
        return true;
    }
    if (config.attentionQueryTile <= 0 || config.attentionKvTile <= 0) {
        return fail(error, "invalid attention tiles");
    }
    const Wide q = static_cast<uint64_t>(tokens), p = static_cast<uint64_t>(prefix);
    const Wide bq = static_cast<uint64_t>(config.attentionQueryTile);
    const Wide bkv = static_cast<uint64_t>(config.attentionKvTile);
    const Wide full = (q - 1) / bq;
    const Wide total =
        (full ? floorSum(full, bkv, bq, p + bq + bkv - 1) : 0) + (p + q + bkv - 1) / bkv;
    if (total > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return fail(error, "attention pair count exceeds int64 range");
    }
    pairs = static_cast<int64_t>(total);
    return true;
}

bool TPUPrefillExecutionPolicy::planNextPrefillChunk(
    const TPUPrefillRequestInput &request, const TPUPrefillExecutionPolicyConfig &config,
    int64_t &tokens, int64_t &attentionPairs, std::string *error)
{
    tokens = attentionPairs = 0;
    if (!validate(config, error) || !config.fixedPrefillChunkPlan ||
        config.prefillLookahead < 0 || request.promptTokens <= 0 ||
        request.prefillEndTokens < 0 || request.numTokens < 0 ||
        request.prefillEnd() < request.promptTokens ||
        request.tokenEnd() < request.prefillEnd() || request.computedTokens < 0 ||
        request.computedTokens >= request.prefillEnd() ||
        request.sharedPrefixBoundary < 0 ||
        request.sharedPrefixBoundary > request.prefillEnd()) {
        return fail(error, "invalid fixed prefill plan input");
    }
    auto limit = std::min(config.longPrefillTokenThreshold,
                          request.tokenEnd() - request.computedTokens);
    limit = std::min(limit, config.inputTokenBudget() - config.draftSlotsPerRequest);
    if (config.attentionScheduleCapacity > 0) {
        int64_t low = 0, high = limit;
        while (low < high) {
            const auto middle = low + (high - low) / 2 + 1;
            int64_t pairs = 0;
            if (countAttentionPairs(middle, request.computedTokens, config, pairs,
                                    nullptr) &&
                pairs <= config.attentionScheduleCapacity) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        limit = low;
    }
    auto candidate = config.mambaBlockAlignedSplit
                         ? clipPrefillChunk(request, limit, config, limit)
                         : limit;
    // r64 multi-module MTP: leave the whole immutable lookahead tail, or
    // finish prefill. This mirrors _reserve_prefill_lookahead after Mamba.
    const auto tail = request.tokenEnd() - request.computedTokens - candidate;
    if (tail > 0 && tail < config.prefillLookahead) {
        candidate -= config.prefillLookahead - tail;
    }
    int64_t pairs = 0;
    if (candidate <= 0 ||
        !countAttentionPairs(candidate, request.computedTokens, config, pairs, error)) {
        return fail(error, "request cannot advance within fixed plan constraints");
    }
    tokens = candidate;
    attentionPairs = pairs;
    return true;
}

bool TPUPrefillExecutionPolicy::projectStepGeometry(
    const std::vector<int64_t> &requestScheduledTokens,
    const TPUPrefillExecutionPolicyConfig &config, TPUPrefillStepGeometry &output,
    std::optional<int64_t> dpTargetBucket, std::string *error)
{
    output = {};
    if (!validate(config, error)) {
        return false;
    }
    if (requestScheduledTokens.empty()) {
        return fail(error, "runner chunk must contain positive scheduled work");
    }
    int64_t total = 0;
    for (auto tokens : requestScheduledTokens) {
        if (tokens <= 0 || !checkedAdd(total, tokens, total)) {
            return fail(error, "invalid or overflowing scheduled token sum");
        }
    }
    return projectStepGeometryTotal(total, config, output, dpTargetBucket, error);
}

bool TPUPrefillExecutionPolicy::projectCacheResolvedBatchImpl(
    const std::vector<TPUPrefillRequestInput> &orderedRequests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
    TPUPrefillExecutionProjection &output, std::string *error, bool collectSteps)
{
    output = {};
    if (!validate(config, error)) {
        return false;
    }
    auto requests = orderedRequests;
    for (const auto &request : requests) {
        if (request.promptTokens <= 0 || request.computedTokens < 0 ||
            request.computedTokens > request.prefillEnd() ||
            request.sharedPrefixBoundary < 0 ||
            request.sharedPrefixBoundary > request.prefillEnd()) {
            return fail(error, "invalid prefill request coordinates");
        }
    }
    TPUPrefillExecutionProjection candidate;
    // A deferred whole chunk depends only on immutable policy/request state.
    // Reuse it until this request advances, including across full rounds.
    std::vector<int64_t> nextTokens, nextPairs;
    // A cached prefix does not make a newly submitted request RUNNING.
    // Admission is a waiting prefix, so original order also keeps running
    // members ahead of the remaining waiting queue in subsequent rounds.
    std::vector<bool> admitted;
    if (config.fixedPrefillChunkPlan) {
        nextTokens.assign(requests.size(), -1);
        nextPairs.assign(requests.size(), 0);
        admitted.assign(requests.size(), false);
    }
    while (true) {
        const auto workRemains =
            std::any_of(requests.begin(), requests.end(), [](const auto &request) {
                return request.computedTokens < request.prefillEnd();
            });
        if (!workRemains) {
            output = std::move(candidate);
            return true;
        }
        if (static_cast<uint64_t>(candidate.stepCount) >= maxProjectionSteps ||
            candidate.stepCount == std::numeric_limits<int64_t>::max()) {
            return fail(error, "projection computation step bound exceeded");
        }
        TPUPrefillExecutionStep step;
        int64_t scheduledTotal = 0;
        int64_t stepAttentionPairs = 0;
        auto budget = config.maxScheduledTokens;
        auto inputBudget = config.inputTokenBudget();
        for (size_t index = 0; index < requests.size(); ++index) {
            auto &request = requests[index];
            const auto remaining = request.prefillEnd() - request.computedTokens;
            if (remaining <= 0) {
                continue;
            }
            int64_t tokens = 0, pairs = 0;
            if (config.fixedPrefillChunkPlan) {
                if (budget == 0 || inputBudget <= config.draftSlotsPerRequest ||
                    (config.attentionScheduleCapacity > 0 &&
                     stepAttentionPairs == config.attentionScheduleCapacity)) {
                    break;
                }
                if (nextTokens[index] < 0 &&
                    !planNextPrefillChunk(request, config, nextTokens[index],
                                          nextPairs[index], error)) {
                    return false;
                }
                tokens = nextTokens[index];
                pairs = nextPairs[index];
                if (tokens > budget ||
                    tokens > inputBudget - config.draftSlotsPerRequest ||
                    (config.attentionScheduleCapacity > 0 &&
                     pairs > config.attentionScheduleCapacity - stepAttentionPairs)) {
                    // Runtime may defer a running member and try later running
                    // work, but never skips an unfit waiting head to admit its tail.
                    if (!admitted[index]) {
                        break;
                    }
                    continue;
                }
            } else {
                tokens = std::min(remaining, budget);
            }
            if (!config.fixedPrefillChunkPlan && config.longPrefillTokenThreshold > 0) {
                tokens = std::min(tokens, config.longPrefillTokenThreshold);
            }
            if (tokens <= 0) {
                break;
            }
            if (!config.fixedPrefillChunkPlan) {
                tokens = clipPrefillChunk(request, tokens, config);
            }
            if (tokens <= 0) {
                break;
            }
            if (collectSteps) {
                step.requests.push_back({index, request.computedTokens, tokens});
            }
            stepAttentionPairs +=
                pairs;                // Each admitted chunk fits the remaining capacity.
            scheduledTotal += tokens; // Bounded by the original step budget.
            request.computedTokens += tokens; // tokens <= prompt-computed.
            if (config.fixedPrefillChunkPlan) {
                nextTokens[index] = -1;
                admitted[index] = true;
            }
            budget -= tokens;
            if (config.fixedPrefillChunkPlan) {
                inputBudget -= tokens + config.draftSlotsPerRequest;
            }
        }
        if (scheduledTotal == 0) {
            return fail(error, "all-ready hypothesis cannot schedule positive work");
        }
        if (!projectStepGeometryTotal(scheduledTotal, config, step.geometry, std::nullopt,
                                      error) ||
            !checkedAdd(candidate.globalPaddedTokenSum, step.geometry.globalPaddedTokens,
                        candidate.globalPaddedTokenSum)) {
            if (error && error->empty()) {
                *error = "projected padded token sum exceeds int64 range";
            }
            return false;
        }
        if (!checkedAdd(candidate.attentionPairSum, stepAttentionPairs,
                        candidate.attentionPairSum)) {
            return fail(error, "projected attention pair sum exceeds int64 range");
        }
        ++candidate.stepCount;
        if (collectSteps) {
            candidate.steps.emplace_back(std::move(step));
        }
    }
}

bool TPUPrefillExecutionPolicy::projectLatencyFeaturesImpl(
    const std::vector<TPUPrefillRequestInput> &orderedRequests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxProjectionSteps,
    TPUPrefillExecutionProjection &output, std::string *error, bool collectSteps)
{
    // A singleton has identical geometry in either lookup group; use the same projector
    // directly.
    if (!config.asyncKVLookupEnabled || orderedRequests.size() <= 1) {
        return projectCacheResolvedBatchImpl(orderedRequests, config, maxProjectionSteps,
                                             output, error, collectSteps);
    }
    output = {};
    if (!validate(config, error)) {
        return false;
    }
    std::vector<size_t> shared;
    std::vector<std::vector<size_t>> groups;
    for (size_t index = 0; index < orderedRequests.size(); ++index) {
        const auto &request = orderedRequests[index];
        if (request.promptTokens <= 0 || request.computedTokens < 0 ||
            request.computedTokens > request.prefillEnd() ||
            request.sharedPrefixBoundary < 0 ||
            request.sharedPrefixBoundary > request.prefillEnd()) {
            return fail(error, "invalid prefill request coordinates");
        }
        if (hasAsyncLookupOpportunity(request.promptTokens, config)) {
            groups.push_back({index});
        } else {
            shared.push_back(index);
        }
    }
    if (!shared.empty()) {
        groups.insert(groups.begin(), std::move(shared));
    }
    TPUPrefillExecutionProjection candidate;
    for (const auto &indices : groups) {
        std::vector<TPUPrefillRequestInput> requests;
        for (auto index : indices) {
            requests.push_back(orderedRequests[index]);
        }
        TPUPrefillExecutionProjection group;
        if (candidate.stepCount < 0 ||
            static_cast<uint64_t>(candidate.stepCount) > maxProjectionSteps ||
            !projectCacheResolvedBatchImpl(
                requests, config,
                maxProjectionSteps - static_cast<size_t>(candidate.stepCount), group,
                error, collectSteps) ||
            !checkedAdd(candidate.stepCount, group.stepCount, candidate.stepCount) ||
            !checkedAdd(candidate.globalPaddedTokenSum, group.globalPaddedTokenSum,
                        candidate.globalPaddedTokenSum) ||
            !checkedAdd(candidate.attentionPairSum, group.attentionPairSum,
                        candidate.attentionPairSum)) {
            if (error && error->empty()) {
                *error = "lookup-group geometry exceeds bounds";
            }
            return false;
        }
        for (auto &step : group.steps) {
            for (auto &request : step.requests) {
                request.requestIndex = indices[request.requestIndex];
            }
            candidate.steps.push_back(std::move(step));
        }
    }
    output = std::move(candidate);
    return true;
}

bool TPUPrefillExecutionPolicy::projectLatencyFeatures(
    const std::vector<TPUPrefillRequestInput> &requests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxSteps,
    TPUPrefillExecutionProjection &output, std::string *error)
{
    return projectLatencyFeaturesImpl(requests, config, maxSteps, output, error, true);
}

bool TPUPrefillExecutionPolicy::projectLatencyFeaturesSummary(
    const std::vector<TPUPrefillRequestInput> &requests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxSteps,
    TPUPrefillExecutionProjection &output, std::string *error)
{
    return projectLatencyFeaturesImpl(requests, config, maxSteps, output, error, false);
}

bool TPUPrefillExecutionPolicy::projectCacheResolvedBatch(
    const std::vector<TPUPrefillRequestInput> &requests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxSteps,
    TPUPrefillExecutionProjection &output, std::string *error)
{
    return projectCacheResolvedBatchImpl(requests, config, maxSteps, output, error, true);
}

bool TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
    const std::vector<TPUPrefillRequestInput> &requests,
    const TPUPrefillExecutionPolicyConfig &config, size_t maxSteps,
    TPUPrefillExecutionProjection &output, std::string *error)
{
    return projectCacheResolvedBatchImpl(requests, config, maxSteps, output, error,
                                         false);
}

} // namespace navi_sched
