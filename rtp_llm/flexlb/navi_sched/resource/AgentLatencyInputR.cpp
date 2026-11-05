#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "autil/TimeUtility.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <unordered_set>

namespace navi_sched {

const std::string AgentLatencyInputR::RESOURCE_ID = "navi_sched.agent_latency_input_r";

void AgentLatencyInput::appendFeatures(NaviSchedLatencyFeatures &features,
                                       const NaviSchedRequestStats &query)
{
    features.appendRequest(query.cacheHitTokenCount,
                           query.totalTokenCount - query.cacheHitTokenCount);
}

NaviSchedLatencySample AgentLatencyInput::makeLatencySample() const
{
    NaviSchedLatencySample sample;
    if (unifiedLatencySample && precedingFeatures.usesCapacitySplit()) {
        sample.features = precedingFeatures;
    } else if (unifiedLatencySample) {
        sample.features.append(precedingFeatures);
        sample.features.appendBatchCount(1);
        for (const auto &query : queries) {
            appendFeatures(sample.features, query);
        }
    } else {
        sample.features.appendBatchCount(1);
        for (const auto &query : queries) {
            appendFeatures(sample.features, query);
        }
    }
    if (geometryFeatureOriginVersion == PROJECTED_GEOMETRY_FEATURE_VERSION) {
        sample.features.setExecutionGeometry(projectedExecutionStepCount,
                                             projectedGlobalPaddedTokens);
    } else if (executionStepCount > 0 && globalPaddedTokens > 0) {
        sample.features.setExecutionGeometry(executionStepCount, globalPaddedTokens);
    }
    if (!unifiedLatencySample && !queries.empty()) {
        int64_t readRequests = 0, maxReadTokens = 0;
        bool known = true;
        for (const auto &query : queries) {
            if (query.executionExternalReadTokens < 0 ||
                query.executionExternalReadTokens > query.cacheHitTokenCount) {
                known = false;
                break;
            }
            readRequests += query.executionExternalReadTokens > 0;
            maxReadTokens = std::max(maxReadTokens, query.executionExternalReadTokens);
        }
        if (known) {
            sample.features.setExecutionCacheSource(readRequests, maxReadTokens);
        }
    }
    sample.latencyMs = unifiedLatencySample ? modelLatencyMs : latencyMs;
    sample.eventTimeUs = eventTimeUs;
    return sample;
}

bool AgentLatencyInput::hasCompatibleGeometryFeatureOrigin(
    int32_t modelBinaryVersion) const
{
    if (NonLinearPrefillModelR::usesProjectedGeometryForVersion(modelBinaryVersion)) {
        return geometryFeatureOriginVersion == PROJECTED_GEOMETRY_FEATURE_VERSION &&
               projectedExecutionStepCount > 0 &&
               projectedGlobalPaddedTokens >= projectedExecutionStepCount;
    }
    return geometryFeatureOriginVersion != PROJECTED_GEOMETRY_FEATURE_VERSION;
}

NaviSchedLatencyTrainingPhase AgentLatencyInput::getLatencyTrainingPhase() const
{
    for (const auto &query : queries) {
        if (NaviSchedLatencyTrainingPhase::NORMAL == query.latencyTrainingPhase) {
            return NaviSchedLatencyTrainingPhase::NORMAL;
        }
    }
    return queries.empty() ? NaviSchedLatencyTrainingPhase::NORMAL
                           : NaviSchedLatencyTrainingPhase::RAMP;
}

AgentLatencyInputR::AgentLatencyInputR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

bool AgentLatencyInputR::submitBatchReady(const NaviSchedBatchPtr &batch,
                                          int32_t batchWaitingRequests,
                                          int32_t queuedBatchRequests,
                                          int32_t runningRequests, int64_t eventTimeUs)
{
    if (!batch || batchWaitingRequests < 0 || queuedBatchRequests < 0 ||
        runningRequests < 0) {
        return false;
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::BATCH_READY;
    input.batch = batch;
    input.batchId = batch->batchId;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitBatchCacheResolved(const NaviSchedBatchPtr &batch,
                                                  const AgentLatencyInput &executionState)
{
    if (!batch || batch->batchId <= 0 || executionState.eventTimeUs <= 0 ||
        executionState.batchWaitingRequests < 0 ||
        executionState.queuedBatchRequests < 0 || executionState.runningRequests < 0) {
        return false;
    }
    auto input = executionState;
    input.type = AgentLatencyInputType::BATCH_CACHE_RESOLVED;
    input.batch = batch;
    input.batchId = batch->batchId;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitBatchStarted(int64_t batchId, int32_t batchWaitingRequests,
                                            int32_t queuedBatchRequests,
                                            int32_t runningRequests, int64_t eventTimeUs)
{
    if (batchId <= 0 || batchWaitingRequests < 0 || queuedBatchRequests < 0 ||
        runningRequests < 0) {
        return false;
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::BATCH_STARTED;
    input.batchId = batchId;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitBatchFinished(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    double backendQueueLatencyMs, double predictedQueueLatencyMs,
    double predictedExecutionLatencyMs, int64_t predictedCacheHitTokenCount,
    int64_t actualCacheHitTokenCount, int32_t cacheHitMismatchRequestCount,
    int64_t cacheHitAbsoluteErrorTokenCount, int64_t cacheHitMaxAbsoluteErrorTokenCount,
    int64_t batchId, int32_t inflightRequests, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests, int64_t eventTimeUs,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    bool latencySampleValid, bool latencyValidationValid,
    const NaviSchedLocalBatchInfo &localBatch)
{
    return submitBatchFinishedImpl(
        requests, latencyMs, backendQueueLatencyMs, predictedQueueLatencyMs,
        predictedExecutionLatencyMs, predictedCacheHitTokenCount,
        actualCacheHitTokenCount, cacheHitMismatchRequestCount,
        cacheHitAbsoluteErrorTokenCount, cacheHitMaxAbsoluteErrorTokenCount, batchId,
        inflightRequests, batchWaitingRequests, queuedBatchRequests, runningRequests,
        eventTimeUs, precedingFeatures, unifiedLatencyMs, latencySampleValid,
        latencyValidationValid, localBatch, 0, 0);
}

bool AgentLatencyInputR::submitBatchFinishedWithExecutionGeometry(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    double backendQueueLatencyMs, double predictedQueueLatencyMs,
    double predictedExecutionLatencyMs, int64_t predictedCacheHitTokenCount,
    int64_t actualCacheHitTokenCount, int32_t cacheHitMismatchRequestCount,
    int64_t cacheHitAbsoluteErrorTokenCount, int64_t cacheHitMaxAbsoluteErrorTokenCount,
    int64_t batchId, int32_t inflightRequests, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests, int64_t eventTimeUs,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    bool latencySampleValid, bool latencyValidationValid,
    const NaviSchedLocalBatchInfo &localBatch, int64_t executionStepCount,
    int64_t globalPaddedTokens)
{
    if (executionStepCount <= 0 || globalPaddedTokens <= 0 || unifiedLatencyMs != -1.0) {
        return false;
    }
    return submitBatchFinishedImpl(
        requests, latencyMs, backendQueueLatencyMs, predictedQueueLatencyMs,
        predictedExecutionLatencyMs, predictedCacheHitTokenCount,
        actualCacheHitTokenCount, cacheHitMismatchRequestCount,
        cacheHitAbsoluteErrorTokenCount, cacheHitMaxAbsoluteErrorTokenCount, batchId,
        inflightRequests, batchWaitingRequests, queuedBatchRequests, runningRequests,
        eventTimeUs, precedingFeatures, unifiedLatencyMs, latencySampleValid,
        latencyValidationValid, localBatch, executionStepCount, globalPaddedTokens);
}

bool AgentLatencyInputR::submitBatchFinishedWithProjectedExecutionGeometry(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    double backendQueueLatencyMs, double predictedQueueLatencyMs,
    double predictedExecutionLatencyMs, int64_t predictedCacheHitTokenCount,
    int64_t actualCacheHitTokenCount, int32_t cacheHitMismatchRequestCount,
    int64_t cacheHitAbsoluteErrorTokenCount, int64_t cacheHitMaxAbsoluteErrorTokenCount,
    int64_t batchId, int32_t inflightRequests, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests, int64_t eventTimeUs,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    bool latencySampleValid, bool latencyValidationValid,
    const NaviSchedLocalBatchInfo &localBatch, int64_t executionStepCount,
    int64_t globalPaddedTokens, int64_t projectedExecutionStepCount,
    int64_t projectedGlobalPaddedTokens)
{
    if (executionStepCount <= 0 || globalPaddedTokens < 0 || unifiedLatencyMs != -1.0 ||
        projectedExecutionStepCount <= 0 ||
        projectedGlobalPaddedTokens < projectedExecutionStepCount) {
        return false;
    }
    return submitBatchFinishedImpl(
        requests, latencyMs, backendQueueLatencyMs, predictedQueueLatencyMs,
        predictedExecutionLatencyMs, predictedCacheHitTokenCount,
        actualCacheHitTokenCount, cacheHitMismatchRequestCount,
        cacheHitAbsoluteErrorTokenCount, cacheHitMaxAbsoluteErrorTokenCount, batchId,
        inflightRequests, batchWaitingRequests, queuedBatchRequests, runningRequests,
        eventTimeUs, precedingFeatures, unifiedLatencyMs, latencySampleValid,
        latencyValidationValid, localBatch, executionStepCount, globalPaddedTokens,
        projectedExecutionStepCount, projectedGlobalPaddedTokens);
}

bool AgentLatencyInputR::submitBatchFinishedImpl(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    double backendQueueLatencyMs, double predictedQueueLatencyMs,
    double predictedExecutionLatencyMs, int64_t predictedCacheHitTokenCount,
    int64_t actualCacheHitTokenCount, int32_t cacheHitMismatchRequestCount,
    int64_t cacheHitAbsoluteErrorTokenCount, int64_t cacheHitMaxAbsoluteErrorTokenCount,
    int64_t batchId, int32_t inflightRequests, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests, int64_t eventTimeUs,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    bool latencySampleValid, bool latencyValidationValid,
    const NaviSchedLocalBatchInfo &localBatch, int64_t executionStepCount,
    int64_t globalPaddedTokens, int64_t projectedExecutionStepCount,
    int64_t projectedGlobalPaddedTokens)
{
    if (batchId <= 0 || requests.empty() || !std::isfinite(latencyMs) ||
        latencyMs < 0.0 || !std::isfinite(backendQueueLatencyMs) ||
        backendQueueLatencyMs < 0.0 || !std::isfinite(predictedQueueLatencyMs) ||
        predictedQueueLatencyMs < 0.0 || !std::isfinite(predictedExecutionLatencyMs) ||
        predictedExecutionLatencyMs < 0.0 || predictedCacheHitTokenCount < 0 ||
        actualCacheHitTokenCount < 0 || cacheHitMismatchRequestCount < 0 ||
        cacheHitAbsoluteErrorTokenCount < 0 || cacheHitMaxAbsoluteErrorTokenCount < 0 ||
        inflightRequests < 0 || batchWaitingRequests < 0 || queuedBatchRequests < 0 ||
        runningRequests < 0 || (!localBatch.empty() && !localBatch.valid()) ||
        (unifiedLatencyMs != -1.0 &&
         (!std::isfinite(unifiedLatencyMs) || unifiedLatencyMs < 0.0))) {
        return false;
    }
    AgentLatencyInput input;
    input.executionStepCount = executionStepCount;
    input.globalPaddedTokens = globalPaddedTokens;
    input.projectedExecutionStepCount = projectedExecutionStepCount;
    input.projectedGlobalPaddedTokens = projectedGlobalPaddedTokens;
    input.geometryFeatureOriginVersion =
        projectedExecutionStepCount > 0
            ? AgentLatencyInput::PROJECTED_GEOMETRY_FEATURE_VERSION
            : (executionStepCount > 0 && globalPaddedTokens > 0 ? 26 : 0);
    input.type = AgentLatencyInputType::BATCH_FINISHED;
    input.queries = requests;
    input.batchId = batchId;
    input.latencyMs = latencyMs;
    input.unifiedLatencySample = unifiedLatencyMs >= 0.0;
    input.latencySampleValid = latencySampleValid;
    input.latencyValidationValid = latencyValidationValid;
    input.modelLatencyMs = input.unifiedLatencySample ? unifiedLatencyMs : latencyMs;
    input.precedingFeatures = precedingFeatures;
    input.localBatch = localBatch;
    input.backendQueueLatencyMs = backendQueueLatencyMs;
    input.predictedQueueLatencyMs = predictedQueueLatencyMs;
    input.predictedExecutionLatencyMs = predictedExecutionLatencyMs;
    input.predictedCacheHitTokenCount = predictedCacheHitTokenCount;
    input.actualCacheHitTokenCount = actualCacheHitTokenCount;
    input.cacheHitMismatchRequestCount = cacheHitMismatchRequestCount;
    input.cacheHitAbsoluteErrorTokenCount = cacheHitAbsoluteErrorTokenCount;
    input.cacheHitMaxAbsoluteErrorTokenCount = cacheHitMaxAbsoluteErrorTokenCount;
    input.inflightRequests = inflightRequests;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.completedRequestCount = static_cast<int32_t>(requests.size());
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitBatchAborted(int32_t completedRequestCount,
                                            int64_t batchId, int32_t inflightRequests,
                                            int32_t batchWaitingRequests,
                                            int32_t queuedBatchRequests,
                                            int32_t runningRequests, int64_t eventTimeUs,
                                            const NaviSchedLocalBatchInfo &localBatch)
{
    if (batchId <= 0 || completedRequestCount <= 0 || inflightRequests < 0 ||
        batchWaitingRequests < 0 || queuedBatchRequests < 0 || runningRequests < 0 ||
        (!localBatch.empty() && !localBatch.valid())) {
        return false;
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::BATCH_ABORTED;
    input.batchId = batchId;
    input.localBatch = localBatch;
    input.inflightRequests = inflightRequests;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.completedRequestCount = completedRequestCount;
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitBatchRequestsAborted(
    const std::vector<NaviSchedRequestStats> &requests,
    const NaviSchedBatchPtr &remainingBatch, int64_t batchId, int32_t inflightRequests,
    int32_t batchWaitingRequests, int32_t queuedBatchRequests, int32_t runningRequests,
    int64_t eventTimeUs, const NaviSchedLocalBatchInfo &localBatch)
{
    if (requests.empty() || !remainingBatch || batchId <= 0 ||
        remainingBatch->batchId != batchId || remainingBatch->requestIds.empty() ||
        !remainingBatch->makeLocalBatchInfo().valid() || inflightRequests < 0 ||
        batchWaitingRequests < 0 || queuedBatchRequests < 0 || runningRequests < 0 ||
        !localBatch.valid() || localBatch.localBatchId != batchId ||
        localBatch.scheduleBatchId != remainingBatch->scheduleBatchId ||
        localBatch.requestIds.empty() ||
        localBatch.requestCount != static_cast<int32_t>(requests.size())) {
        return false;
    }
    std::unordered_set<uint64_t> abortedRequestIds;
    abortedRequestIds.reserve(requests.size());
    for (const auto &request : requests) {
        if (!request.valid() || !abortedRequestIds.emplace(request.requestId).second) {
            return false;
        }
    }
    if (abortedRequestIds != std::unordered_set<uint64_t>(localBatch.requestIds.begin(),
                                                          localBatch.requestIds.end())) {
        return false;
    }
    const auto &remainingRequests = AgentBatchR::getRequestStats(*remainingBatch);
    if (remainingRequests.size() != remainingBatch->requestIds.size() ||
        remainingRequests.size() + requests.size() >
            static_cast<size_t>(remainingBatch->expectedBatchSize)) {
        return false;
    }
    std::unordered_set<uint64_t> remainingRequestIds;
    remainingRequestIds.reserve(remainingRequests.size());
    for (const auto &request : remainingRequests) {
        if (!request.valid() || abortedRequestIds.count(request.requestId) != 0 ||
            !remainingRequestIds.emplace(request.requestId).second) {
            return false;
        }
    }
    if (remainingRequestIds !=
        std::unordered_set<uint64_t>(remainingBatch->requestIds.begin(),
                                     remainingBatch->requestIds.end())) {
        return false;
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::BATCH_REQUESTS_ABORTED;
    input.queries = requests;
    input.batch = remainingBatch;
    input.batchId = batchId;
    input.localBatch = localBatch;
    input.inflightRequests = inflightRequests;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.completedRequestCount = static_cast<int32_t>(requests.size());
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitObservedBatchFinished(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    int32_t inflightRequests, int64_t eventTimeUs)
{
    if (requests.empty() || !std::isfinite(latencyMs) || latencyMs < 0.0 ||
        inflightRequests < 0) {
        return false;
    }
    for (const auto &request : requests) {
        if (!request.valid()) {
            return false;
        }
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::OBSERVED_BATCH_FINISHED;
    input.queries = requests;
    input.latencyMs = latencyMs;
    input.inflightRequests = inflightRequests;
    input.completedRequestCount = static_cast<int32_t>(requests.size());
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

bool AgentLatencyInputR::submitExecutionLoad(int32_t batchWaitingRequests,
                                             int32_t queuedBatchRequests,
                                             int32_t runningRequests, int64_t eventTimeUs)
{
    if (batchWaitingRequests < 0 || queuedBatchRequests < 0 || runningRequests < 0) {
        return false;
    }
    AgentLatencyInput input;
    input.type = AgentLatencyInputType::EXECUTION_LOAD;
    input.batchWaitingRequests = batchWaitingRequests;
    input.queuedBatchRequests = queuedBatchRequests;
    input.runningRequests = runningRequests;
    input.eventTimeUs = eventTimeUs;
    return submit(std::move(input));
}

size_t AgentLatencyInputR::getPendingInputCount() const
{
    autil::ScopedLock lock(_lock);
    return _inputs.size();
}

AgentLatencyInputR::Stats AgentLatencyInputR::getStats() const
{
    Stats stats;
    stats.submittedInputCount = _submittedInputCount.load(std::memory_order_relaxed);
    stats.rejectedInputCount = _rejectedInputCount.load(std::memory_order_relaxed);
    stats.processedInputCount = _processedInputCount.load(std::memory_order_relaxed);
    stats.submittedFinishedRequestCount =
        _submittedFinishedRequestCount.load(std::memory_order_relaxed);
    stats.processedFinishedRequestCount =
        _processedFinishedRequestCount.load(std::memory_order_relaxed);
    stats.submittedAbortedRequestCount =
        _submittedAbortedRequestCount.load(std::memory_order_relaxed);
    stats.processedAbortedRequestCount =
        _processedAbortedRequestCount.load(std::memory_order_relaxed);
    return stats;
}

AgentLatencyInputR::ProcessingLagStats
AgentLatencyInputR::takeBatchStartedProcessingLagWindowStats() const
{
    ProcessingLagStats stats;
    int64_t processingLagUs = 0;
    while (_batchStartedProcessingLagQueue.Pop(&processingLagUs)) {
        ++stats.sampleCount;
        stats.totalTimeUs += processingLagUs;
        stats.maxTimeUs = std::max(stats.maxTimeUs, processingLagUs);
    }
    return stats;
}

bool AgentLatencyInputR::submit(AgentLatencyInput &&input)
{
    const auto type = input.type;
    const auto completedRequestCount = input.completedRequestCount;
    autil::ScopedLock lock(_lock);
    if (_finishRequested) {
        _rejectedInputCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (input.eventTimeUs <= 0) {
        input.eventTimeUs = autil::TimeUtility::currentTime();
    }
    _inputs.emplace_back(std::move(input));
    if (!_inputPipe ||
        navi::EC_NONE == _inputPipe->setDataOverride(nullptr, true, true)) {
        _submittedInputCount.fetch_add(1, std::memory_order_relaxed);
        if (AgentLatencyInputType::BATCH_FINISHED == type) {
            _submittedFinishedRequestCount.fetch_add(completedRequestCount,
                                                     std::memory_order_relaxed);
        } else if (AgentLatencyInputType::BATCH_ABORTED == type ||
                   AgentLatencyInputType::BATCH_REQUESTS_ABORTED == type) {
            _submittedAbortedRequestCount.fetch_add(completedRequestCount,
                                                    std::memory_order_relaxed);
        }
        return true;
    }
    _inputPipe.reset();
    _submittedInputCount.fetch_add(1, std::memory_order_relaxed);
    if (AgentLatencyInputType::BATCH_FINISHED == type) {
        _submittedFinishedRequestCount.fetch_add(completedRequestCount,
                                                 std::memory_order_relaxed);
    } else if (AgentLatencyInputType::BATCH_ABORTED == type ||
               AgentLatencyInputType::BATCH_REQUESTS_ABORTED == type) {
        _submittedAbortedRequestCount.fetch_add(completedRequestCount,
                                                std::memory_order_relaxed);
    }
    return true;
}

void AgentLatencyInputR::take(std::vector<AgentLatencyInput> &inputs)
{
    std::deque<AgentLatencyInput> pending;
    {
        autil::ScopedLock lock(_lock);
        pending.swap(_inputs);
    }
    inputs.reserve(inputs.size() + pending.size());
    while (!pending.empty()) {
        const auto &input = pending.front();
        if (!input.graphRecovery && AgentLatencyInputType::BATCH_STARTED == input.type) {
            const auto lagUs = std::max<int64_t>(
                0, autil::TimeUtility::currentTime() - input.eventTimeUs);
            _batchStartedProcessingLagQueue.Push(lagUs);
        }
        if (!input.graphRecovery) {
            _processedInputCount.fetch_add(1, std::memory_order_relaxed);
            if (AgentLatencyInputType::BATCH_FINISHED == input.type) {
                _processedFinishedRequestCount.fetch_add(input.completedRequestCount,
                                                         std::memory_order_relaxed);
            } else if (AgentLatencyInputType::BATCH_ABORTED == input.type ||
                       AgentLatencyInputType::BATCH_REQUESTS_ABORTED == input.type) {
                _processedAbortedRequestCount.fetch_add(input.completedRequestCount,
                                                        std::memory_order_relaxed);
            }
        }
        inputs.emplace_back(std::move(pending.front()));
        pending.pop_front();
    }
}

void AgentLatencyInputR::replaceGraphRecoveryInputs(std::vector<AgentLatencyInput> inputs)
{
    autil::ScopedLock lock(_lock);
    _inputs.erase(std::remove_if(
                      _inputs.begin(), _inputs.end(),
                      [](const AgentLatencyInput &input) { return input.graphRecovery; }),
                  _inputs.end());
    for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
        it->graphRecovery = true;
        _inputs.emplace_front(std::move(*it));
    }
}

void AgentLatencyInputR::prepareGraphStart()
{
    autil::ScopedLock lock(_lock);
    _finishRequested = false;
    _inputPipe.reset();
    _inputPipeOwnerGeneration = 0;
}

uint64_t AgentLatencyInputR::beginInputPipeGeneration()
{
    autil::ScopedLock lock(_lock);
    if (_finishRequested) {
        return 0;
    }
    const auto generation =
        _inputPipeGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
    _inputPipeOwnerGeneration = generation;
    _inputPipe.reset();
    return generation;
}

bool AgentLatencyInputR::setInputPipe(uint64_t generation,
                                      const navi::AsyncPipePtr &inputPipe)
{
    if (0 == generation || !inputPipe) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    if (_inputPipeOwnerGeneration != generation ||
        (_inputPipe && _inputPipe != inputPipe)) {
        return false;
    }
    _inputPipe = inputPipe;
    if (!_inputs.empty() &&
        navi::EC_NONE != _inputPipe->setDataOverride(nullptr, true, true)) {
        return false;
    }
    if (_finishRequested &&
        (navi::EC_NONE != _inputPipe->setDataOverride(nullptr, true, true) ||
         navi::EC_NONE != _inputPipe->setEof())) {
        return false;
    }
    return true;
}

bool AgentLatencyInputR::waitInputPipe(uint64_t generation, int64_t timeoutMs) const
{
    if (0 == generation || timeoutMs < 0) {
        return false;
    }
    constexpr int64_t READY_POLL_INTERVAL_US = 10 * 1000;
    const auto deadlineUs = autil::TimeUtility::monotonicTimeUs() + timeoutMs * 1000;
    while (true) {
        {
            autil::ScopedLock lock(_lock);
            if (_finishRequested || _inputPipeOwnerGeneration != generation) {
                return false;
            }
            if (_inputPipe) {
                return true;
            }
        }
        const auto remainingUs = deadlineUs - autil::TimeUtility::monotonicTimeUs();
        if (remainingUs <= 0) {
            return false;
        }
        std::this_thread::sleep_for(
            std::chrono::microseconds(std::min(remainingUs, READY_POLL_INTERVAL_US)));
    }
}

void AgentLatencyInputR::clearInputPipe(uint64_t generation)
{
    autil::ScopedLock lock(_lock);
    if (_inputPipeOwnerGeneration != generation) {
        return;
    }
    _inputPipe.reset();
    _inputPipeOwnerGeneration = 0;
}

bool AgentLatencyInputR::finish()
{
    autil::ScopedLock lock(_lock);
    _finishRequested = true;
    return !_inputPipe ||
           (navi::EC_NONE == _inputPipe->setDataOverride(nullptr, true, true) &&
            navi::EC_NONE == _inputPipe->setEof());
}

uint64_t AgentLatencyInputR::getInputPipeGeneration() const
{
    return _inputPipeGeneration.load(std::memory_order_relaxed);
}

navi::SummaryErrorCode AgentLatencyInputR::getSummary(const std::string &query,
                                                      navi::SummaryBuilder &builder) const
{
    autil::ScopedLock lock(_lock);
    auto table = builder.newTable("NaviSchedAgentLatencyInputOverview");
    TABLE_ADD_COLUMN_VALUE(table, "pending_input_count", _inputs.size());
    TABLE_ADD_COLUMN_VALUE(table, "finish_requested", _finishRequested);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentLatencyInputR);

}
