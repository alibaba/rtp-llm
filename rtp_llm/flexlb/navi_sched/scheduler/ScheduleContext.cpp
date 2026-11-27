#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "aios/navi_sched/scheduler/ScheduleContextInternal.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"

#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/MasterScheduleRequestData.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "autil/HashUtil.h"
#include "autil/legacy/json.h"

#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

namespace navi_sched {


int64_t NaviSchedScheduleContext::getScheduleTimeUs() const { return _scheduleTimeUs; }

NaviSchedScheduleNode::NaviSchedScheduleNode(NaviSchedNodeConstPtr node,
                                             NaviSchedNodeSnapshot snapshot,
                                             int64_t scheduleTimeUs,
                                             double timeCompression)
    : _node(std::move(node))
    , _prefillExecutionPolicy(std::move(snapshot.prefillExecutionPolicy))
    , _queueLookupPromptBoundary(
          _prefillExecutionPolicy ? TPUPrefillExecutionPolicy::asyncLookupPromptBoundary(
                                        *_prefillExecutionPolicy)
                                  : std::numeric_limits<int64_t>::max())
    , _peerCacheReadAvailable(snapshot.peerCacheReadAvailable)
    , _peerCacheReadRevision(snapshot.peerCacheReadRevision)
    , _info(std::move(snapshot.info))
    , _latencyModel(std::move(snapshot.latencyModel))
    , _residualRuntimeConfig(snapshot.residualRuntimeConfig)
    , _totalBlocks(snapshot.load.totalBlocks)
    , _inflightRequests(snapshot.load.inflightRequests)
    , _inflightPredictedLatencySumMs(snapshot.load.inflightPredictedLatencySumMs)
    , _batchWaitingRequests(snapshot.load.batchWaitingRequests)
    , _queuedBatchRequests(snapshot.load.queuedBatchRequests)
    , _runningRequests(snapshot.load.runningRequests)
    , _batchTimeoutCount(snapshot.load.batchTimeoutCount)
    , _abnormalBatchFinishCount(snapshot.load.abnormalBatchFinishCount)
    , _queueTotalPredictedLatencyMs(snapshot.load.queueTotalPredictedLatencyMs)
    , _queueDecayHeadRemainingMs(snapshot.load.queueDecayHeadRemainingMs)
    , _queueModel(snapshot.load.queueModel)
    , _queueReportReceiveTimeUs(snapshot.load.queueReportReceiveTimeUs)
    , _scheduleTimeUs(scheduleTimeUs)
    , _timeCompression(timeCompression)
    , _unifiedLatencyPrediction(snapshot.unifiedLatencyPrediction)
    , _workload(std::move(snapshot.workload))
    , _nativePrefillReservations(std::move(snapshot.nativePrefillReservations))
    , _workloadFeatures(std::move(snapshot.workloadFeatures))
    , _reportedWorkloadFeatures(std::move(snapshot.reportedWorkloadFeatures))
    , _reservedWorkloadFeatures(std::move(snapshot.reservedWorkloadFeatures))
    , _maxRunningRequestCount(snapshot.maxRunningRequestCount)
    , _workloadRunningCapacitySlotCount(snapshot.workloadRunningCapacitySlotCount)
    , _reservedRunningCapacitySlotCount(snapshot.reservedRunningCapacitySlotCount)
    , _workloadReportReceiveTimeUs(snapshot.workloadReportReceiveTimeUs)
    , _agentStartTimeUs(snapshot.agentStartTimeUs)
{
    if (_latencyModel &&
        (_latencyModel->binaryVersion == NonLinearPrefillModelR::PD_BINARY_VERSION ||
         NonLinearPrefillModelR::usesGeometryForVersion(_latencyModel->binaryVersion))) {
        // Smooth proposal coordinates only. Binary26 exact predictions below
        // always use versioned parameters and complete-batch geometry.
        _legacyLatencyParameters = NonLinearPrefillModelR::parametersForLegacyPrediction(
            _latencyModel->parameters, _latencyModel->binaryVersion);
    }
    if (_unifiedLatencyPrediction && _latencyModel &&
        _latencyModel->binaryVersion == NonLinearPrefillModelR::BINARY_VERSION &&
        NonLinearPrefillModelR::usesCapacitySplitParameters(_latencyModel->parameters) &&
        _info && NativePrefillStepForecast::declaredBy(_info->latencyModelFingerprint)) {
        if (!_prefillExecutionPolicy || _prefillExecutionPolicy->maxScheduledTokens <= 0)
            throw std::invalid_argument("Native step model requires reported token budget");
        _nativeStepTokenBudget = _prefillExecutionPolicy->maxScheduledTokens;
        // The optimizer uses continuous steps so its request-cost sum and gradient
        // remain additive. Final predictions use the corresponding whole-step count.
        _legacyLatencyParameters = _latencyModel->parameters;
        constexpr size_t stepParameterIndex = 6;
        constexpr size_t runningMissParameterIndex = 3;
        constexpr size_t waitingMissParameterIndex = 12;
        const double stepCost = _legacyLatencyParameters[stepParameterIndex] *
                                NonLinearPrefillModelR::getParameterScale(stepParameterIndex);
        const double stepsPerKToken = 1024.0 / _nativeStepTokenBudget;
        for (const size_t index : {runningMissParameterIndex, waitingMissParameterIndex}) {
            _legacyLatencyParameters[index] += stepCost * stepsPerKToken /
                NonLinearPrefillModelR::getParameterScale(index);
        }
        _legacyLatencyParameters[stepParameterIndex] = 0.0;
    }
    if (_unifiedLatencyPrediction && _workloadReportReceiveTimeUs > 0) {
        const auto workloadReportAgeMs = getWorkloadReportAgeMs();
        const bool supportsRunningElapsed =
            _latencyModel && NonLinearPrefillModelR::usesCapacitySplitParameters(
                                 _latencyModel->parameters);
        if (supportsRunningElapsed && _workload.runningCapacityPrefillRequestCount > 0) {
            const auto elapsedRequestCount = _workload.runningElapsedRequestCount >= 0
                                                 ? _workload.runningElapsedRequestCount
                                                 : _workload.runningCapacityPrefillRequestCount;
            const auto runningElapsedDeltaMs = workloadReportAgeMs * elapsedRequestCount;
            auto reportedFeatures = _reportedWorkloadFeatures;
            auto workloadFeatures = _workloadFeatures;
            if (reportedFeatures.appendRunningElapsedTimeMs(runningElapsedDeltaMs) &&
                ((_info && _info->workloadFeatureSnapshotValid) ||
                 workloadFeatures.appendRunningElapsedTimeMs(runningElapsedDeltaMs))) {
                _reportedWorkloadFeatures = std::move(reportedFeatures);
                _workloadFeatures = std::move(workloadFeatures);
                _runningElapsedProgressApplied = true;
            }
        }
    }
    if (_unifiedLatencyPrediction && _workload.executionLoadTracked &&
        _latencyModel && _latencyModel->binaryVersion==NonLinearPrefillModelR::BINARY_VERSION &&
        NonLinearPrefillModelR::usesCapacitySplitParameters(_latencyModel->parameters)) {
        if (_info && _info->workloadFeatureSnapshotValid) {
            // A replay/explicit snapshot already contains the frozen projection,
            // including a legitimate zero probability with no extra component.
            NaviSchedExpectedKVWorkload reported;
            if (const auto full = _workloadFeatures.getExpectedKVWorkload()) reported = *full;
            if (const auto reserved = _reservedWorkloadFeatures.getExpectedKVWorkload()) {
                if (!reported.subtract(*reserved))
                    throw std::invalid_argument("inconsistent frozen reserved KV projection");
            }
            if (!_reportedWorkloadFeatures.setExpectedKVWorkload(reported))
                throw std::invalid_argument("invalid frozen KV projection");
        } else {
        if(!snapshot.kvPreparationEstimates.empty() && snapshot.kvPreparationSnapshotMonoUs<=0)
            throw std::invalid_argument("missing KV preparation snapshot clock");
        NaviSchedLatencyFeatures floor;
        if (!getUnifiedDecayFloorFeatures(floor)) throw std::invalid_argument("invalid ready workload floor");
        // One-pass horizon fixed for the scheduling round. Candidate assignment
        // gradients do not refit time or feed a changed horizon back into p.
        const double horizon=std::max(0.0,predictLatency(_workloadFeatures)-predictLatency(floor));
        NaviSchedExpectedKVWorkload expected;
        size_t at=0;
        for (const auto &request:_workload.executionLoadRequests) {
            if(request.state!=NativeKVExecutionState::PREPARING) continue;
            MasterKVPreparationForecasts::Estimate estimate;
            if(!snapshot.kvPreparationEstimates.empty()) {
                if(at>=snapshot.kvPreparationEstimates.size()) throw std::invalid_argument("missing KV preparation estimate identity");
                estimate=snapshot.kvPreparationEstimates[at];
                if(estimate.receivedMonoUs>snapshot.kvPreparationSnapshotMonoUs)
                    throw std::invalid_argument("KV preparation clock regression");
                if(estimate.requestId!=request.requestId || estimate.generation!=request.loadGeneration)
                    throw std::invalid_argument("KV preparation estimate generation mismatch");
            }
            ++at;
            const double p=estimate.weight(snapshot.kvPreparationSnapshotMonoUs,horizon);
            NaviSchedKVPreparationProbabilityAudit audit;
            audit.requestId = request.requestId;
            audit.generation = request.loadGeneration;
            audit.snapshotMonoUs = snapshot.kvPreparationSnapshotMonoUs;
            audit.reportReceiveMonoUs = estimate.receivedMonoUs;
            audit.observedPreparationMs = estimate.observedPreparationMs;
            audit.elapsedAtSnapshotMs = estimate.observedPreparationMs +
                std::max<int64_t>(0, snapshot.kvPreparationSnapshotMonoUs -
                                       estimate.receivedMonoUs) / 1000.0;
            audit.queueHorizonMs = horizon;
            audit.frozenTotalMs = estimate.totalMs;
            audit.probability = p;
            _kvPreparationProbabilityAudits.push_back(audit);
            expected.precedingRequests+=p;
            if(request.computeTokens>0) {
                auto &moments=request.runningCapacity ? expected.running : expected.waiting;
                const double h=request.contextTokens,n=request.computeTokens;
                moments[0]+=p;moments[1]+=p*h;moments[2]+=p*n;moments[3]+=p*n*n;moments[4]+=p*h*n;
            }
        }
        if(!snapshot.kvPreparationEstimates.empty() && at!=snapshot.kvPreparationEstimates.size())
            throw std::invalid_argument("unmatched KV preparation estimate");
        // Preserve the existing logical batch/step axis in this isolation change.
        // No per-request ceil(tokens/chunk) approximation is introduced here.
        auto completeExpected = expected;
        if (const auto reserved = _reservedWorkloadFeatures.getExpectedKVWorkload()) {
            if (!completeExpected.append(*reserved))
                throw std::invalid_argument("invalid reserved KV workload moments");
        }
        if(!_reportedWorkloadFeatures.setExpectedKVWorkload(expected) ||
           !_workloadFeatures.setExpectedKVWorkload(completeExpected))
            throw std::invalid_argument("invalid expected KV workload moments");
        }
    }
    initializeNativeStepPackingRequests();
}

void NaviSchedScheduleNode::initializeNativeStepPackingRequests()
{
    if (_nativeStepTokenBudget <= 0 || !_prefillExecutionPolicy) return;
    if (_info && _info->workloadFeatureSnapshotValid) {
        _nativeStepPackingRequests = _info->nativeStepPackingRequests;
        return;
    }
    if (!_workload.executionLoadTracked) return;
    std::unordered_map<uint64_t, const MasterBatchReservation::UnfinishedRequest *> metadata;
    for (const auto &entry : _nativePrefillReservations)
        if (!metadata.emplace(entry.request.requestId, &entry).second) return;
    std::unordered_set<uint64_t> reported;
    std::vector<NativePrefillStepRequest> requests;
    for (const auto &record : _workload.executionLoadRequests) {
        if (record.state == NativeKVExecutionState::PREPARING) return;
        if (!reported.insert(record.requestId).second) return;
        if (record.state == NativeKVExecutionState::FAILED || record.computeTokens == 0)
            continue;
        if (record.state != NativeKVExecutionState::READY &&
            record.state != NativeKVExecutionState::EXECUTING) return;
        const auto found = metadata.find(record.requestId);
        if (found == metadata.end()) return;
        const auto &request = found->second->request;
        if (request.predictedSharedPrefixBoundary < 0 ||
            record.contextTokens < 0 || record.computeTokens < 0 ||
            record.contextTokens > request.totalTokenCount ||
            record.computeTokens != request.totalTokenCount - record.contextTokens) return;
        requests.push_back({record.requestId,
                           {request.totalTokenCount, record.contextTokens,
                            request.predictedSharedPrefixBoundary},
                           record.state == NativeKVExecutionState::EXECUTING, true});
    }
    for (const auto &entry : _nativePrefillReservations) {
        const auto &request = entry.request;
        if (reported.count(request.requestId) ||
            entry.state != MasterBatchReservation::RequestState::WAITING ||
            request.totalTokenCount == request.predictedCacheHitTokenCount) continue;
        if (request.executionLoadIncluded != std::optional<bool>(true) ||
            request.predictedSharedPrefixBoundary < 0) return;
        requests.push_back({request.requestId,
                           {request.totalTokenCount, request.predictedCacheHitTokenCount,
                            request.predictedSharedPrefixBoundary}, false, false});
    }
    // Reports are sorted by request ID for serialization, not by Native queue.
    // Executing members retain priority; order within each group is a forecast.
    std::sort(requests.begin(), requests.end(), [](const auto &a, const auto &b) {
        if (a.executing != b.executing) return a.executing > b.executing;
        return a.requestId < b.requestId;
    });
    _nativeStepPackingRequests = std::move(requests);
}

namespace {

void supplyNativeRelease(TPUPrefillPP4RequestPrediction &prediction,
                         NativePPPrefillSlotRelease release)
{
    if (release == NativePPPrefillSlotRelease::UNKNOWN) return;
    if (!validNativePrefillSlotRelease(release))
        throw std::invalid_argument("invalid reserved Native completion condition");
    const bool afterPrefill = release == NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    if (prediction.releaseSlotAfterPrefill &&
        *prediction.releaseSlotAfterPrefill != afterPrefill)
        throw std::invalid_argument("Native completion prediction contradicts request");
    prediction.releaseSlotAfterPrefill = afterPrefill;
}

void supplyNoAsyncQueryPrediction(TPUPrefillPP4RequestPrediction &prediction,
                                  NativePPRequestPhase phase, int64_t prompt,
                                  const TPUPrefillExecutionPolicyConfig &policy,
                                  std::optional<int64_t> hbmPrefix = std::nullopt)
{
    // A pending query, issued load or host call needs its own service estimate.
    // The absence of async lookup is sufficient only before a query is issued.
    if (TPUPrefillExecutionPolicy::hasAsyncLookupOpportunity(prompt, policy)) {
        // Full HBM hash blocks leave no external suffix to query.
        const bool completeHBMPrefix =
            hbmPrefix && *hbmPrefix >= 0 && *hbmPrefix < prompt &&
            policy.schedulerBlockSize > 0 && policy.asyncKVLookupHashBlockSize > 0 &&
            *hbmPrefix % policy.schedulerBlockSize == 0 &&
            *hbmPrefix >= (prompt - (policy.asyncKVLookupUsesEagle ? 1 : 0)) /
                              policy.asyncKVLookupHashBlockSize *
                              policy.asyncKVLookupHashBlockSize;
        if (!completeHBMPrefix) return;
    }
    if (phase != NativePPRequestPhase::NOT_REGISTERED &&
        phase != NativePPRequestPhase::QUERY_NOT_STARTED &&
        phase != NativePPRequestPhase::CACHE_UNKNOWN &&
        phase != NativePPRequestPhase::PREEMPTED &&
        phase != NativePPRequestPhase::UNKNOWN)
        return;
    if (!prediction.needsFirstQuery) prediction.needsFirstQuery = false;
    if (!prediction.readyAfterObservationMs) prediction.readyAfterObservationMs = 0.0;
}

TPUPrefillPP4Request unobservedNativeRequest(
    uint64_t id, int64_t prompt, int64_t checkpoint, NativePPPrefillSlotRelease release,
    int64_t expectedHit, bool owned, const TPUPrefillExecutionPolicyConfig &policy,
    TPUPrefillPP4RequestPrediction prediction,
    std::optional<int64_t> hbmPrefix = std::nullopt)
{
    supplyNativeRelease(prediction, release);
    supplyNoAsyncQueryPrediction(prediction, NativePPRequestPhase::NOT_REGISTERED, prompt,
                                 policy, hbmPrefix);
    if (prompt <= 0 || !prediction.releaseSlotAfterPrefill ||
        !prediction.needsFirstQuery || !prediction.readyAfterObservationMs ||
        prediction.receiveSchedulerOutputId || prediction.queryReadyAfterObservationMs ||
        (prediction.nativePromptTokens && *prediction.nativePromptTokens != prompt))
        throw std::invalid_argument(
            "unobserved Native request needs explicit prediction inputs");
    const auto hit = prediction.cacheHitTokens.value_or(expectedHit);
    if (hit < 0 || hit > prompt || checkpoint < 0 || checkpoint > prompt)
        throw std::invalid_argument("invalid unobserved Native cache prediction");
    TPUPrefillPP4Request value;
    value.requestId = id;
    value.contextTokens = std::min(prompt - 1, hit);
    value.remainingTokens = prompt - value.contextTokens;
    value.owned = owned;
    value.releaseSlotAfterPrefill = prediction.releaseSlotAfterPrefill;
    value.needsFirstQuery = *prediction.needsFirstQuery;
    value.readyTimeMs = *prediction.readyAfterObservationMs;
    value.receiveDependency = prediction.receiveDependency;
    value.metadataOnlyLoad = prediction.metadataOnlyLoad;
    value.firstQueryServiceMs = prediction.firstQueryServiceMs;
    value.queryBlockCount = prediction.queryBlockCount;
    if (checkpoint > 0)
        value.sharedPrefixBoundary =
            checkpoint / policy.schedulerBlockSize * policy.schedulerBlockSize;
    return value;
}

} // namespace

TPUPrefillPP4ObservationInput NaviSchedScheduleNode::prepareNativePrefillObservation(
    TPUPrefillPP4ObservationPredictions predictions) const
{
    if (!NonLinearPrefillModelR::usesNativePP4ForVersion(getLatencyBinaryVersion()) ||
        !_prefillExecutionPolicy || _prefillExecutionPolicy->schedulerBlockSize <= 0 ||
        !_workload.nativePPState || !_workload.valid())
        throw std::invalid_argument(
            "Native Master prediction requires its typed observation and policy");
    std::unordered_map<uint64_t, const NaviSchedNativePPRequest *> observed;
    for (const auto &request : _workload.nativePPState->requests)
        observed.emplace(request.requestId, &request);
    std::unordered_set<uint64_t> reservations;
    std::unordered_map<uint64_t, int64_t> preparedCheckpoints;
    std::vector<TPUPrefillPP4Request> prior;
    for (const auto &entry : _nativePrefillReservations) {
        const auto &request = entry.request;
        if (!reservations.insert(request.requestId).second)
            throw std::invalid_argument("duplicate Native Master reservation");
        const auto found = observed.find(request.requestId);
        if (found == observed.end()) {
            // DISPATCHED can outlive Prefill, including paused streaming sessions.
            // Only an unreported WAITING request adds work absent from Native.
            if (entry.state != MasterBatchReservation::RequestState::WAITING) continue;
            auto prediction = predictions.requests.extract(request.requestId);
            prior.push_back(unobservedNativeRequest(
                request.requestId, request.totalTokenCount,
                request.explicitCacheCheckpointPosition, request.nativePrefillSlotRelease,
                request.predictedCacheHitTokenCount, false, *_prefillExecutionPolicy,
                prediction.empty() ? TPUPrefillPP4RequestPrediction()
                                   : std::move(prediction.mapped())));
            continue;
        }
        auto &prediction = predictions.requests[request.requestId];
        const auto &native = *found->second;
        if (native.queue == NativePPQueue::PREPARED) {
            if (prediction.nativePromptTokens &&
                *prediction.nativePromptTokens != request.totalTokenCount)
                throw std::invalid_argument("conflicting normalized Native prompt");
            prediction.nativePromptTokens = request.totalTokenCount;
            if (request.explicitCacheCheckpointPosition > 0) {
                if (request.explicitCacheCheckpointPosition > request.totalTokenCount)
                    throw std::invalid_argument(
                        "prepared checkpoint exceeds Native prompt");
                preparedCheckpoints.emplace(
                    request.requestId, request.explicitCacheCheckpointPosition /
                                           _prefillExecutionPolicy->schedulerBlockSize *
                                           _prefillExecutionPolicy->schedulerBlockSize);
            }
            if (!prediction.cacheHitTokens)
                prediction.cacheHitTokens = request.predictedCacheHitTokenCount;
            supplyNoAsyncQueryPrediction(prediction, native.phase,
                                         request.totalTokenCount,
                                         *_prefillExecutionPolicy);
        } else {
            if (native.promptTokens != request.totalTokenCount)
                throw std::invalid_argument(
                    "registered Native prompt differs from Master request");
            if (!prediction.cacheHitTokens)
                prediction.cacheHitTokens = request.predictedCacheHitTokenCount;
            supplyNoAsyncQueryPrediction(prediction, native.phase,
                                         request.totalTokenCount,
                                         *_prefillExecutionPolicy);
        }
        supplyNativeRelease(prediction, request.nativePrefillSlotRelease);
    }
    prior.insert(prior.end(), predictions.incoming.begin(), predictions.incoming.end());
    predictions.incoming = std::move(prior);
    auto workload = _workload;
    const long double age = workload.nativePPObservationAgeUs +
                            static_cast<long double>(getWorkloadReportAgeMs()) * 1000.0L;
    if (age < 0 || age > std::numeric_limits<int64_t>::max())
        throw std::overflow_error("Native observation age exceeds representable range");
    workload.nativePPObservationAgeUs = static_cast<int64_t>(age);
    auto result = TPUPrefillPP4Observation::prepare(workload, predictions);
    for (auto &request : result.requests) {
        const auto checkpoint = preparedCheckpoints.find(request.requestId);
        if (checkpoint != preparedCheckpoints.end() && checkpoint->second > 0)
            request.sharedPrefixBoundary = checkpoint->second;
    }
    return result;
}

std::shared_ptr<const TPUPrefillPP4ObservationInput>
NaviSchedScheduleNode::getNativePrefillObservation() const
{
    auto published = std::atomic_load(&_nativePrefillObservation);
    if (published) return published;
    std::shared_ptr<const TPUPrefillPP4ObservationInput> prepared =
        std::make_shared<const TPUPrefillPP4ObservationInput>(
            prepareNativePrefillObservation({}));
    std::atomic_compare_exchange_strong(&_nativePrefillObservation, &published, prepared);
    return published ? published : prepared;
}

void NaviSchedScheduleNode::refreshExistingUnifiedLatency()
{
    _existingUnifiedLatencyMs = 0.0;
    NaviSchedLatencyFeatures floorFeatures;
    if (usesUnifiedLatencyPrediction() && hasLatencyModel() &&
        getUnifiedDecayFloorFeatures(floorFeatures)) {
        if (_nativeStepTokenBudget > 0) {
            // This cached value is the optimizer's baseline. Mixing a rounded
            // baseline with continuous candidate costs would invent negative work.
            const auto predictOptimizationLatency = [this](const NaviSchedLatencyFeatures &features) {
                return applyLatencyResidual(NonLinearPrefillModelR::predict(
                    features, getLatencyParameters()));
            };
            _existingUnifiedLatencyMs = decayUnifiedLatency(
                predictOptimizationLatency(_workloadFeatures),
                predictOptimizationLatency(floorFeatures));
        } else {
            _existingUnifiedLatencyMs = decayUnifiedLatency(predictLatency(_workloadFeatures),
                                                            predictLatency(floorFeatures));
        }
    }
}

uint64_t NaviSchedScheduleNode::getAgentIdHash() const { return _info->agentIdHash; }

const std::string &NaviSchedScheduleNode::getAgentId() const { return _info->agentId; }

int64_t NaviSchedScheduleNode::getTotalBlocks() const { return _totalBlocks; }

int32_t NaviSchedScheduleNode::getInflightRequests() const { return _inflightRequests; }

double NaviSchedScheduleNode::getInflightPredictedLatencySumMs() const
{
    return _inflightPredictedLatencySumMs;
}

int32_t NaviSchedScheduleNode::getBatchWaitingRequests() const
{
    return _batchWaitingRequests;
}

int32_t NaviSchedScheduleNode::getQueuedBatchRequests() const
{
    return _queuedBatchRequests;
}

int32_t NaviSchedScheduleNode::getRunningRequests() const { return _runningRequests; }

uint64_t NaviSchedScheduleNode::getBatchTimeoutCount() const
{
    return _batchTimeoutCount;
}

uint64_t NaviSchedScheduleNode::getAbnormalBatchFinishCount() const
{
    return _abnormalBatchFinishCount;
}

double NaviSchedScheduleNode::getEstimatedQueueWaitMs(
    double hitMemberWeight, double batchWeight, double lookupMemberWeight,
    double *hitDerivative, double *batchDerivative, double *lookupDerivative) const
{
    return NaviSchedQueueModel::predictMs(
        getEstimatedQueueBaseMsAt(_scheduleTimeUs),
        usesUnifiedLatencyPrediction() ? 0.0 : _queueModel.cacheMemberWeightMs,
        usesUnifiedLatencyPrediction() ? 0.0 : _queueModel.lookupMemberWeightMs,
        hitMemberWeight, lookupMemberWeight, batchWeight, hitDerivative, lookupDerivative,
        batchDerivative);
}

double NaviSchedScheduleNode::getEstimatedQueueWaitMsAt(int64_t predictionTimeUs,
                                                        double hitMemberWeight,
                                                        double batchWeight,
                                                        double lookupMemberWeight) const
{
    return NaviSchedQueueModel::predictMs(
        getEstimatedQueueBaseMsAt(predictionTimeUs),
        usesUnifiedLatencyPrediction() ? 0.0 : _queueModel.cacheMemberWeightMs,
        usesUnifiedLatencyPrediction() ? 0.0 : _queueModel.lookupMemberWeightMs,
        hitMemberWeight, lookupMemberWeight, batchWeight);
}

double NaviSchedScheduleNode::getEstimatedQueueBaseMsAt(int64_t predictionTimeUs) const
{
    if (usesUnifiedLatencyPrediction()) {
        return 0.0;
    }
    const auto elapsedMs =
        _queueReportReceiveTimeUs > 0
            ? std::max<int64_t>(0, predictionTimeUs - _queueReportReceiveTimeUs) / 1000.0
            : 0.0;
    const auto decayMs =
        std::min(_queueDecayHeadRemainingMs, elapsedMs * _timeCompression);
    return _queueModel.decayedBaseMs(_queueTotalPredictedLatencyMs, decayMs);
}

double NaviSchedScheduleNode::getEstimatedBacklogLatencyMs() const
{
    if (!usesUnifiedLatencyPrediction()) {
        return getEstimatedQueueWaitMs();
    }
    // Accepted requests own workload before their first batch becomes ready.
    if (!hasLatencyModel() || (_workloadFeatures.getBatchCount() == 0 &&
                               _workloadFeatures.getBatchSize() == 0 &&
                               _workloadFeatures.getPrecedingRequestCount() == 0)) {
        return 0.0;
    }
    if (_nativeStepTokenBudget > 0)
        return std::max(0.0, getExistingDiscreteUnifiedLatencyMs());
    const auto latencyWithReportedWorkloadMs = predictLatency(_workloadFeatures);
    NaviSchedLatencyFeatures floorFeatures;
    if (!getUnifiedDecayFloorFeatures(floorFeatures)) {
        return 0.0;
    }
    const auto latencyWithoutReportedWorkloadMs =
        floorFeatures.getBatchCount() == 0 &&
                floorFeatures.getPrecedingRequestCount() == 0
            ? 0.0
            : predictLatency(floorFeatures);
    return std::max(0.0, decayUnifiedLatency(latencyWithReportedWorkloadMs,
                                             latencyWithoutReportedWorkloadMs));
}

double NaviSchedScheduleNode::getExecutionUtilization() const
{
    return _node->getExecutionUtilization();
}

double NaviSchedScheduleNode::getExecutionBusyUtilization() const
{
    return _node->getExecutionBusyUtilization();
}

int32_t NaviSchedScheduleNode::getExecutionWaitingRequests() const
{
    const auto readyQueued = _info->pdSeparated && !_unifiedLatencyPrediction
        ? _queuedBatchRequests : 0;
    return static_cast<int32_t>(std::min<int64_t>(std::numeric_limits<int32_t>::max(),
        int64_t(_workload.waitingRequestCount) + readyQueued));
}

bool NaviSchedScheduleNode::hasLatencyModel() const { return bool(_latencyModel); }

bool NaviSchedScheduleNode::hasValidLatencyModel() const
{
    return _latencyModel &&
           (!NonLinearPrefillModelR::usesGeometryForVersion(
                _latencyModel->binaryVersion) ||
            (_info && _info->pdSeparated && !usesUnifiedLatencyPrediction() &&
             _prefillExecutionPolicy &&
             TPUPrefillExecutionPolicy::validate(*_prefillExecutionPolicy))) &&
           NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL == _latencyModel->formula &&
           NonLinearPrefillModelR::supportsBinaryVersion(_latencyModel->binaryVersion) &&
           NonLinearPrefillModelR::validateParametersForVersion(
               _latencyModel->parameters, _latencyModel->binaryVersion) &&
           (_residualRuntimeConfig.learningRate == 0.0 ||
            (LatencyResidualModel::VERSION == _latencyModel->residualVersion &&
             LatencyResidualModel::validateParameters(
                 _latencyModel->residualParameters) &&
             _latencyModel->residualLastUpdateTimeUs >= 0));
}

int32_t NaviSchedScheduleNode::getLatencyFormula() const
{
    return _latencyModel ? _latencyModel->formula : 0;
}

int64_t NaviSchedScheduleNode::getLatencyBinaryVersion() const
{
    return _latencyModel ? _latencyModel->binaryVersion : 0;
}

int64_t NaviSchedScheduleNode::getLatencyModelRevision() const
{
    return _latencyModel ? _latencyModel->revision : 0;
}

uint64_t NaviSchedScheduleNode::getLatencySessionSampleCount() const
{
    return _latencyModel ? _latencyModel->sessionSampleCount : 0;
}

int32_t NaviSchedScheduleNode::getLatencyResidualVersion() const
{
    return _latencyModel ? _latencyModel->residualVersion : 0;
}

int64_t NaviSchedScheduleNode::getLatencyResidualLastUpdateTimeUs() const
{
    return _latencyModel ? _latencyModel->residualLastUpdateTimeUs : 0;
}

const std::string &NaviSchedScheduleNode::getLatencyModelFingerprint() const
{
    return _info->latencyModelFingerprint;
}

bool NaviSchedScheduleNode::isLatencyModelTrusted() const
{
    if (!hasValidLatencyModel() || !_latencyModel->overloadPredictionTrusted) {
        return false;
    }
    if (!NonLinearPrefillModelR::usesCapacitySplitParameters(_latencyModel->parameters) ||
        !_workloadFeatures.usesCapacitySplit()) {
        return true;
    }
    return NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(
        _latencyModel->parameters, _workloadFeatures);
}

bool NaviSchedScheduleNode::isBacklogPredictionTrusted() const
{
    return !usesUnifiedLatencyPrediction() || isLatencyModelTrusted();
}

const std::vector<double> &NaviSchedScheduleNode::getLatencyParameters() const
{
    static const std::vector<double> EMPTY_PARAMETERS;
    return !_legacyLatencyParameters.empty()
               ? _legacyLatencyParameters
               : (_latencyModel ? _latencyModel->parameters : EMPTY_PARAMETERS);
}

double NaviSchedScheduleNode::applyLatencyResidual(double basePredictionMs,
                                                   double *derivativeFactor) const
{
    if (derivativeFactor) {
        *derivativeFactor = 1.0;
    }
    if (!_latencyModel) {
        return basePredictionMs;
    }
    const auto result = LatencyResidualModel::evaluate(
        basePredictionMs, _latencyModel->residualParameters,
        _latencyModel->residualLastUpdateTimeUs, _scheduleTimeUs, _residualRuntimeConfig);
    if (derivativeFactor) {
        *derivativeFactor = result.derivativeFactor;
    }
    return result.predictionMs;
}

double NaviSchedScheduleNode::predictBaseLatency(
    const NaviSchedLatencyFeatures &features) const
{
    if (!_latencyModel) {
        return 0.0;
    }
    if (_nativeStepTokenBudget > 0) {
        auto predictionFeatures = features;
        if (!prepareLatencyPredictionFeatures(predictionFeatures))
            throw std::invalid_argument("invalid Native step prediction features");
        return NonLinearPrefillModelR::predict(predictionFeatures, _latencyModel->parameters);
    }
    if (NonLinearPrefillModelR::usesGeometryForVersion(_latencyModel->binaryVersion)) {
        return NonLinearPrefillModelR::predictForVersion(
            features, _latencyModel->parameters, _latencyModel->binaryVersion);
    }
    // Preserve the legacy prediction path and its numerical coordinates.
    return NonLinearPrefillModelR::predict(features, getLatencyParameters());
}

bool NaviSchedScheduleNode::prepareLatencyPredictionFeatures(
    NaviSchedLatencyFeatures &features) const
{
    if (_nativeStepTokenBudget == 0) return true;
    const long double missTokens = features.getPlanningCacheMissTokenCount();
    if (!std::isfinite(missTokens) || missTokens < 0) return false;
    long double steps = std::ceil(missTokens / _nativeStepTokenBudget);
    const auto expected = features.getExpectedKVWorkload();
    if (steps == 0 && (features.getBatchSize() > 0 ||
                      (expected && expected->precedingRequests > 0))) steps = 1;
    if (steps > std::numeric_limits<int64_t>::max()) return false;
    if (expected && expected->batchCount != 0.0) {
        auto nativeExpected = *expected;
        nativeExpected.batchCount = 0.0;
        if (!features.setExpectedKVWorkload(nativeExpected)) return false;
    }
    return features.setBatchCount(static_cast<int64_t>(steps));
}

double NaviSchedScheduleNode::predictFrozenBaseLatency(
    const NaviSchedLatencyFeatures &features) const
{
    if (!_latencyModel) return 0.0;
    if (_nativeStepTokenBudget > 0)
        return NonLinearPrefillModelR::predict(features, _latencyModel->parameters);
    return predictBaseLatency(features);
}

double NaviSchedScheduleNode::predictFrozenLatency(
    const NaviSchedLatencyFeatures &features) const
{
    if (!_latencyModel) return 0.0;
    return applyLatencyResidual(predictFrozenBaseLatency(features));
}

double NaviSchedScheduleNode::getExistingDiscreteUnifiedLatencyMs() const
{
    if (_nativeStepTokenBudget <= 0) return getExistingUnifiedLatencyMs();
    auto with = _workloadFeatures;
    NaviSchedLatencyFeatures floor;
    if (!getUnifiedDecayFloorFeatures(floor) ||
        !prepareLatencyPredictionFeatures(with) ||
        !prepareLatencyPredictionFeatures(floor))
        return std::numeric_limits<double>::quiet_NaN();
    if (_nativeStepPackingRequests && _prefillExecutionPolicy) {
        std::vector<NativePrefillStepRequest> reserved;
        for (const auto &request : *_nativeStepPackingRequests)
            if (!request.reported) reserved.push_back(request);
        auto packedWith = with, packedFloor = floor;
        if (NativePrefillStepForecast::prepare(
                packedWith, *_nativeStepPackingRequests, *_prefillExecutionPolicy) &&
            NativePrefillStepForecast::prepare(
                packedFloor, reserved, *_prefillExecutionPolicy)) {
            with = std::move(packedWith);
            floor = std::move(packedFloor);
        }
    }
    return decayUnifiedLatency(predictFrozenLatency(with),
                               predictFrozenLatency(floor));
}

double NaviSchedScheduleNode::predictLatency(
    const NaviSchedLatencyFeatures &features) const
{
    if (!_latencyModel) {
        return 0.0;
    }
    return applyLatencyResidual(predictBaseLatency(features));
}

const std::optional<TPUPrefillExecutionPolicyConfig> &
NaviSchedScheduleNode::getPrefillExecutionPolicy() const
{
    return _prefillExecutionPolicy;
}

bool NaviSchedScheduleNode::usesUnifiedLatencyPrediction() const
{
    return _unifiedLatencyPrediction;
}

int64_t NaviSchedScheduleNode::getAgentStartTimeUs() const { return _agentStartTimeUs; }

const NaviSchedLatencyFeatures &NaviSchedScheduleNode::getWorkloadFeatures() const
{
    return _workloadFeatures;
}

const NaviSchedLatencyFeatures &NaviSchedScheduleNode::getReportedWorkloadFeatures() const
{
    return _reportedWorkloadFeatures;
}

const NaviSchedLatencyFeatures &NaviSchedScheduleNode::getReservedWorkloadFeatures() const
{
    return _reservedWorkloadFeatures;
}

bool NaviSchedScheduleNode::getUnifiedDecayFloorFeatures(
    NaviSchedLatencyFeatures &features) const
{
    auto floorFeatures = _reservedWorkloadFeatures;
    // Even an empty reservation floor must use the same feature family as the
    // ready workload. Otherwise predict() silently returns zero for this floor.
    const bool preparingKVProjection = _unifiedLatencyPrediction &&
        _workload.executionLoadTracked &&
        getLatencyBinaryVersion()==NonLinearPrefillModelR::BINARY_VERSION;
    if (preparingKVProjection &&
        NonLinearPrefillModelR::usesCapacitySplitParameters(getLatencyParameters()) &&
        !floorFeatures.usesCapacitySplit() &&
        !floorFeatures.appendCapacitySplitAggregate(0,0,0,0,0,0,0,0,0,0,0)) return false;
    if (NonLinearPrefillModelR::usesCapacitySplitParameters(getLatencyParameters()) &&
        !floorFeatures.appendPrecedingRequestCount(
            _reportedWorkloadFeatures.getPrecedingRequestCount())) {
        return false;
    }
    if (const auto expected=_reportedWorkloadFeatures.getExpectedKVWorkload()) {
        NaviSchedExpectedKVWorkload floorExpected;
        if (const auto reserved=floorFeatures.getExpectedKVWorkload()) floorExpected=*reserved;
        floorExpected.precedingRequests+=expected->precedingRequests;
        if(!floorFeatures.usesCapacitySplit() &&
           !floorFeatures.appendCapacitySplitAggregate(0,0,0,0,0,0,0,0,0,0,0)) return false;
        if(!floorFeatures.setExpectedKVWorkload(floorExpected)) return false;
    }
    features = std::move(floorFeatures);
    return true;
}

int32_t NaviSchedScheduleNode::getMaxRunningRequestCount() const
{
    return _maxRunningRequestCount;
}

int32_t NaviSchedScheduleNode::getWorkloadRunningCapacitySlotCount() const
{
    return _workloadRunningCapacitySlotCount;
}

int32_t NaviSchedScheduleNode::getReservedRunningCapacitySlotCount() const
{
    return _reservedRunningCapacitySlotCount;
}

double NaviSchedScheduleNode::getExistingUnifiedLatencyMs() const
{
    return _existingUnifiedLatencyMs;
}

double NaviSchedScheduleNode::getReportedRunningElapsedTimeMs() const
{
    return _workload.runningElapsedTimeMs;
}

int32_t NaviSchedScheduleNode::getRunningCapacityPrefillRequestCount() const
{
    return _workload.runningCapacityPrefillRequestCount;
}

uint64_t NaviSchedScheduleNode::getWorkloadRevision() const { return _workload.revision; }

int64_t NaviSchedScheduleNode::getWorkloadEventTimeUs() const
{
    return _workload.eventTimeUs;
}

int64_t NaviSchedScheduleNode::getWorkloadReportReceiveTimeUs() const
{
    return _workloadReportReceiveTimeUs;
}

double NaviSchedScheduleNode::getWorkloadReportAgeMs() const
{
    if (_workloadReportReceiveTimeUs <= 0) {
        return 0.0;
    }
    return std::max<int64_t>(0, _scheduleTimeUs - _workloadReportReceiveTimeUs) / 1000.0 *
           _timeCompression;
}

bool NaviSchedScheduleNode::usesRunningElapsedProgress() const
{
    return _runningElapsedProgressApplied;
}

const LatencyResidualModel::RuntimeConfig &
NaviSchedScheduleNode::getLatencyResidualRuntimeConfig() const
{
    return _residualRuntimeConfig;
}

double NaviSchedScheduleNode::decayUnifiedLatency(double latencyWithReportedWorkloadMs,
                                                  double latencyWithoutReportedWorkloadMs,
                                                  bool *usesReportedWorkload) const
{
    if (usesReportedWorkload) {
        *usesReportedWorkload = true;
    }
    if (!usesUnifiedLatencyPrediction() || _workloadReportReceiveTimeUs <= 0 ||
        (_reportedWorkloadFeatures.getBatchCount() == 0 && !_reportedWorkloadFeatures.getExpectedKVWorkload())) {
        return latencyWithReportedWorkloadMs;
    }
    // An elapsed-aware model has already advanced running prefill progress.
    // Applying wall-clock decay as well would count the same progress twice.
    const auto decayedLatencyMs =
        (_runningElapsedProgressApplied || _reportedWorkloadFeatures.getExpectedKVWorkload())
            ? latencyWithReportedWorkloadMs
            : latencyWithReportedWorkloadMs - getWorkloadReportAgeMs();
    if (decayedLatencyMs <= latencyWithoutReportedWorkloadMs) {
        if (usesReportedWorkload) {
            *usesReportedWorkload = false;
        }
        return latencyWithoutReportedWorkloadMs;
    }
    return decayedLatencyMs;
}

bool NaviSchedScheduleContext::valid() const { return _prepared; }

bool NaviSchedScheduleContext::prepare()
{
    clearScheduleView();
    _coldStartRoutingStats = {};
    _prepared = false;
    _resultPrepared = false;
    if (_requests.empty()) {
        return false;
    }
    for (auto &node : _scheduleNodes) {
        if (!node._info || !node._node) {
            return false;
        }
        node.refreshExistingUnifiedLatency();
    }
    preparePeerCacheCounts();
    if (!finishNativePeerCacheInputs()) return false;
    _decisions.assign(requestCount(), NaviSchedDecision());
    _selectedNodeIndexes.assign(requestCount(), INVALID_NODE_INDEX);
    _prepared = true;
    return true;
}

int64_t NaviSchedPeerCacheSuffixLookup::extend(size_t requestIndex,
                                               int64_t localBlocks) const
{
    if (localBlocks < 0 || alignmentBlocks <= 0 || localBlocks % alignmentBlocks != 0 ||
        requestIndex >= endBlocks.size()) {
        return localBlocks;
    }
    const auto &row = endBlocks[requestIndex];
    return static_cast<uint64_t>(localBlocks) < row.size()
               ? std::max(localBlocks, row[localBlocks])
               : localBlocks;
}

std::shared_ptr<const NaviSchedPeerCacheSuffixLookup>
NaviSchedPeerCacheSuffixLookup::build(const std::vector<NaviSchedScheduleItem> &requests,
                                      const std::vector<std::vector<size_t>> &keyIndexes,
                                      const std::vector<uint8_t> &readyStates,
                                      const NaviSchedCachePolicy &policy)
{
    if (policy.blockSize <= 0 || policy.linearStep <= 0 ||
        policy.fullAttentionQueryDropTokens < 0 || requests.size() != keyIndexes.size()) {
        return nullptr;
    }
    auto result = std::make_shared<NaviSchedPeerCacheSuffixLookup>();
    result->alignmentBlocks = policy.linearStep;
    result->endBlocks.resize(requests.size());
    for (size_t r = 0; r < requests.size(); ++r) {
        const auto &request = requests[r].request;
        if (request.blockSize != policy.blockSize) {
            return nullptr;
        }
        const auto &indexes = keyIndexes[r];
        const auto matchable =
            request.promptTokenCount > policy.fullAttentionQueryDropTokens
                ? (request.promptTokenCount - policy.fullAttentionQueryDropTokens) /
                      policy.blockSize
                : 0;
        const auto limit = std::min<size_t>(indexes.size(), matchable);
        std::vector<int64_t> tailEnd(limit + 1, 0);
        for (size_t i = 0; i < limit; ++i) {
            if (indexes[i] >= readyStates.size()) {
                return nullptr;
            }
            tailEnd[i + 1] =
                ((i + 1) % policy.linearStep == 0 && (readyStates[indexes[i]] & 2))
                    ? i + 1
                    : tailEnd[i];
        }
        auto &row = result->endBlocks[r];
        row.resize(limit + 1);
        row[limit] = limit;
        int64_t nextGap = limit;
        for (size_t i = limit; i-- > 0;) {
            if (!(readyStates[indexes[i]] & 1)) {
                nextGap = i;
            }
            auto end = nextGap - nextGap % policy.linearStep;
            if (policy.hybridTailRequired) {
                end = tailEnd[end];
            }
            row[i] = std::max<int64_t>(i, end);
        }
    }
    return result;
}

bool NaviSchedScheduleContext::finishNativePeerCacheInputs()
{
    // Preparation is single threaded. Publish the final peer suffix before Cost
    // reads the cache matrix or Native forecast memos in parallel.
    for (size_t n = 0; n < _scheduleNodes.size(); ++n) {
        auto &node = _scheduleNodes[n];
        if (!node._peerCacheSuffixLookup ||
            !NonLinearPrefillModelR::usesNativePP4ForVersion(
                node.getLatencyBinaryVersion()))
            continue;
        const auto before = std::atomic_load(&node._executionCacheInputs);
        const auto inputs = getExecutionCacheInputs(n);
        if (!inputs) return false;
        const auto final = std::atomic_load(&node._executionCacheInputs);
        auto counts = std::make_shared<std::vector<int64_t>>();
        counts->reserve(inputs->size());
        for (size_t i = 0; i < inputs->size(); ++i)
            counts->push_back((*inputs)[i].computedTokens /
                              _requests[i].request.blockSize);
        std::shared_ptr<const std::vector<int64_t>> immutableCounts = std::move(counts);
        std::atomic_store(&node._matchedBlockCounts, immutableCounts);
        if (before != final) {
            std::atomic_store(&node._nativeContinuousInput,
                              std::shared_ptr<const NaviSchedNativeContinuousInput>());
            std::atomic_store(&node._nativeFinalPrediction,
                              std::shared_ptr<const NaviSchedNativeFinalPrediction>());
        }
        if (!_replay || n >= _replayNativeCacheInputs.size() ||
            !_replayNativeCacheInputs[n] || _replayMatchRequestIndexes.empty())
            continue;
        const auto local = std::atomic_load(&node._localCacheBlockCounts);
        // Admission may already have compacted the live arrays. Keep the replay's
        // original rows and update consumed rows by stable request ID.
        auto recorded = std::make_shared<NaviSchedExecutionCacheInputCapture>(
            *_replayNativeCacheInputs[n]);
        if (final->observation.readyOnlyHbmBlockCounts &&
            (!recorded->observation.readyOnlyHbmBlockCounts ||
             recorded->observation.readyOnlyHbmBlockCounts->size() != recorded->inputs.size()))
            return false;
        if (final->observation.beforeTairBlockCounts &&
            (!recorded->observation.beforeTairBlockCounts ||
             recorded->observation.beforeTairBlockCounts->size() != recorded->inputs.size()))
            return false;
        auto recordedLocal = std::make_shared<std::vector<int64_t>>();
        const auto previousLocal = _replayNativeLocalCacheCounts[n];
        if (previousLocal) {
            *recordedLocal = *previousLocal;
        } else {
            recordedLocal->reserve(recorded->inputs.size());
            for (size_t i = 0; i < recorded->inputs.size(); ++i)
                recordedLocal->push_back(recorded->inputs[i].computedTokens /
                                         _replay->requests[i].request.blockSize);
        }
        for (size_t i = 0; i < _requests.size(); ++i) {
            const auto row =
                _replayMatchRequestIndexes.at(_requests[i].request.requestId);
            recorded->inputs.at(row) = (*inputs)[i];
            if (!final->observation.fullPrefixBlocksBeforeQueryDrop.empty())
                recorded->observation.fullPrefixBlocksBeforeQueryDrop.at(row) =
                    final->observation.fullPrefixBlocksBeforeQueryDrop[i];
            if (final->observation.readyOnlyHbmBlockCounts)
                recorded->observation.readyOnlyHbmBlockCounts->at(row) =
                    final->observation.readyOnlyHbmBlockCounts->at(i);
            if (final->observation.beforeTairBlockCounts)
                recorded->observation.beforeTairBlockCounts->at(row) =
                    final->observation.beforeTairBlockCounts->at(i);
            if (local) recordedLocal->at(row) = (*local)[i];
            _replay->nodeMatches[row][n].matchedBlockCount = (*immutableCounts)[i];
        }
        recorded->peerCacheMerged = true;
        _replayNativeCacheInputs[n] = std::move(recorded);
        if (local) _replayNativeLocalCacheCounts[n] = std::move(recordedLocal);
    }
    return true;
}

void NaviSchedScheduleContext::preparePeerCacheCounts()
{
    _peerCacheSupplierGroups.clear();
    _peerCacheSupplierStates.clear();
    _peerCacheRequestKeyIndexes.clear();
    if (!_peerCacheEnabled || _requests.empty()) {
        return;
    }
    NaviSchedBlockHashes queryKeys;
    std::unordered_map<uint64_t, size_t> keyIndex;
    std::vector<std::vector<size_t>> requestKeyIndexes(_requests.size());
    for (size_t i = 0; i < _requests.size(); ++i) {
        if (!_requests[i].request.blockHashes) {
            continue;
        }
        auto &indexes = requestKeyIndexes[i];
        const auto &request = _requests[i].request;
        const auto limit =
            request.blockSize > 0
                ? std::min<size_t>(
                      request.blockHashes->size(),
                      std::max<int64_t>(0, request.promptTokenCount) / request.blockSize)
                : 0;
        indexes.reserve(limit);
        for (size_t h = 0; h < limit; ++h) {
            const auto hash = (*request.blockHashes)[h];
            const auto entry = keyIndex.emplace(hash, queryKeys.size());
            if (entry.second) {
                queryKeys.push_back(hash);
            }
            indexes.push_back(entry.first->second);
        }
    }
    struct ReadyGroup {
        NaviSchedCachePolicy policy;
        std::vector<uint8_t> states;
        std::shared_ptr<const NaviSchedPeerCacheSuffixLookup> lookup;
    };
    std::unordered_map<std::string, ReadyGroup> groups;
    std::vector<std::string> keys(_scheduleNodes.size());
    _peerCacheSupplierStates.resize(_scheduleNodes.size());
    std::vector<uint8_t> states;
    for (size_t n = 0; n < _scheduleNodes.size(); ++n) {
        auto &node = _scheduleNodes[n];
        NaviSchedCachePolicy p;
        if (!node._info->online || !node._info->peerCacheEnabled ||
            (_peerCacheReadyAgents && !_peerCacheReadyAgents->empty() &&
             !_peerCacheReadyAgents->count(node.getAgentId())) ||
            node.getLatencyModelFingerprint().empty() ||
            !node._node->getReadyExternalCacheKeyStates(queryKeys, states, p) ||
            states.size() != queryKeys.size() || p.blockSize <= 0 || p.linearStep <= 0 ||
            std::any_of(_requests.begin(), _requests.end(), [&](const auto &item) {
                return item.request.blockSize != p.blockSize;
            })) {
            continue;
        }
        // Compare fingerprints as opaque identities, with identical cache policies.
        auto key = node.getLatencyModelFingerprint() + ":" + std::to_string(p.blockSize) +
                   ":" + std::to_string(p.linearStep) + ":" +
                   std::to_string(p.hybridTailRequired) + ":" +
                   std::to_string(p.mambaReplayStopShift) + ":" +
                   std::to_string(p.mambaRetentionInterval) + ":" +
                   std::to_string(p.mambaAlign) + ":" +
                   std::to_string(p.fullAttentionQueryDropTokens);
        keys[n] = key;
        _peerCacheSupplierStates[n] = states;
        auto it = groups.find(key);
        if (it == groups.end()) {
            groups.emplace(key, ReadyGroup{p, states, nullptr});
        } else {
            // The SDK may fetch different READY keys/groups from different peers.
            // Union only published physical observations, never pending estimates.
            for (size_t i = 0; i < states.size(); ++i) {
                it->second.states[i] |= states[i];
            }
        }
    }
    for (auto &entry : groups) {
        auto &group = entry.second;
        group.lookup = NaviSchedPeerCacheSuffixLookup::build(_requests, requestKeyIndexes,
                                                             group.states, group.policy);
    }
    for (size_t n = 0; n < _scheduleNodes.size(); ++n) {
        // A receiver's read failure does not remove its healthy resident objects
        // from the supplier pool. Gate only cross-node consumption.
        if (!keys[n].empty() && _scheduleNodes[n]._peerCacheReadAvailable) {
            _scheduleNodes[n]._peerCacheSuffixLookup = groups.at(keys[n]).lookup;
        }
    }
    // Freeze the single local observation plus READY suffix before optimization.
    for (size_t n = 0; n < _scheduleNodes.size(); ++n) {
        if (_scheduleNodes[n]._peerCacheSuffixLookup) {
            getCacheHitBlockCountsSnapshot(n);
        }
    }
    _peerCacheSupplierGroups = std::move(keys);
    _peerCacheRequestKeyIndexes = std::move(requestKeyIndexes);
}

std::vector<NaviSchedTarget> NaviSchedScheduleContext::getCacheTouchTargets(
    size_t requestIndex, int64_t hitTokenCount) const
{
    std::vector<NaviSchedTarget> targets;
    if (!_cacheTouchEnabled || requestIndex >= _selectedNodeIndexes.size() ||
        requestIndex >= _peerCacheRequestKeyIndexes.size() || hitTokenCount <= 0) {
        return targets;
    }
    const auto receiver = _selectedNodeIndexes[requestIndex];
    if (receiver >= _scheduleNodes.size() ||
        receiver >= _peerCacheSupplierGroups.size() ||
        !_scheduleNodes[receiver]._peerCacheSuffixLookup) {
        return targets;
    }
    const auto local = std::atomic_load(&_scheduleNodes[receiver]._localCacheBlockCounts);
    const auto blockSize = _requests[requestIndex].request.blockSize;
    if (!local || requestIndex >= local->size() || blockSize <= 0) {
        return targets;
    }
    const auto &indexes = _peerCacheRequestKeyIndexes[requestIndex];
    const auto end = std::min<size_t>(hitTokenCount / blockSize, indexes.size());
    const auto begin = static_cast<size_t>(std::max<int64_t>(0, (*local)[requestIndex]));
    if (begin >= end) {
        return targets;
    }
    const auto &group = _peerCacheSupplierGroups[receiver];
    for (size_t n = 0; n < _scheduleNodes.size(); ++n) {
        if (_peerCacheSupplierGroups[n] != group) {
            continue;
        }
        const auto &states = _peerCacheSupplierStates[n];
        bool contributes = indexes[end - 1] < states.size() &&
                           (states[indexes[end - 1]] & 2) != 0;
        for (size_t h = begin; h < end && !contributes; ++h) {
            contributes = indexes[h] < states.size() && (states[indexes[h]] & 1) != 0;
        }
        const auto &info = _scheduleNodes[n]._info;
        if (contributes && info && !info->bizName.empty() &&
            !info->collectorAddress.empty()) {
            NaviSchedTarget target;
            target.bizName = info->bizName;
            target.address = info->serverAddress;
            target.agentAddress = info->collectorAddress;
            targets.push_back(std::move(target));
        }
    }
    return targets;
}

size_t NaviSchedScheduleContext::requestCount() const
{
    return _scheduleViewActive ? _scheduleViewRequestIndexes.size() : _requests.size();
}

size_t NaviSchedScheduleContext::nodeCount() const
{
    return _scheduleViewActive ? _scheduleViewNodeIndexes.size() : _scheduleNodes.size();
}

const NaviSchedScheduleItem &NaviSchedScheduleContext::getRequest(
    size_t requestIndex) const
{
    return _requests[resolveRequestIndex(requestIndex)];
}

const NaviSchedScheduleNode &NaviSchedScheduleContext::getNode(size_t nodeIndex) const
{
    return _scheduleNodes[resolveNodeIndex(nodeIndex)];
}

std::shared_ptr<const NaviSchedTairKvcmConfig>
NaviSchedScheduleNode::getTairKvcmConfig() const
{
    return _info && _info->online ? _info->tairKvcmConfig : nullptr;
}

bool NaviSchedScheduleContext::setRemoteCacheReadyBlocks(
    size_t nodeIndex, std::shared_ptr<const NaviSchedExternalCacheReadyBlocks> blocks)
{
    if (_prepared || _scheduleViewActive || nodeIndex >= _scheduleNodes.size())
        return false;
    auto &node = _scheduleNodes[nodeIndex];
    if (std::atomic_load(&node._executionCacheInputs) ||
        std::atomic_load(&node._matchedBlockCounts)) return false;
    if (blocks && (!node._info || !node._info->online || !node._info->tairKvcmConfig))
        return false;
    node._remoteCacheReadyBlocks = std::move(blocks);
    return true;
}

std::shared_ptr<const std::vector<int64_t>>
NaviSchedScheduleContext::getCacheHitBlockCountsSnapshot(size_t nodeIndex) const
{
    if (nodeIndex >= nodeCount()) {
        return nullptr;
    }
    const auto storageNode = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNode];
    auto published = std::atomic_load(&node._matchedBlockCounts);
    if (published) {
        return published;
    }
    auto counts = std::make_shared<std::vector<int64_t>>(_requests.size());
    bool recordedBatch = !_requests.empty() && !_replayMatchRequestIndexes.empty();
    for (size_t index = 0; recordedBatch && index < _requests.size(); ++index) {
        recordedBatch = getReplayMatch(index, storageNode, (*counts)[index]);
    }
    if (recordedBatch) {
        std::shared_ptr<const std::vector<int64_t>> immutable = counts;
        std::atomic_compare_exchange_strong(&node._matchedBlockCounts, &published, immutable);
        return published ? published : immutable;
    }
    if (node._remoteCacheReadyBlocks || node.classifiesNewRequestKVLoad() ||
        NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion()) ||
        NonLinearPrefillModelR::usesNativePP4ForVersion(node.getLatencyBinaryVersion())) {
        const auto inputs = getExecutionCacheInputs(storageNode);
        if (!inputs) {
            return nullptr;
        }
        for (size_t i = 0; i < _requests.size(); ++i) {
            const auto block = _requests[i].request.blockSize;
            if (block <= 0) {
                return nullptr;
            }
            (*counts)[i] = (*inputs)[i].computedTokens / block;
        }
    } else if (!_requests.empty()) {
        // Ordinary and geometry paths publish one immutable batch observation.
        // Keep ordinary Full-prefix semantics in the existing batch matcher.
        if ((!_scheduleViewActive || _scheduleViewUsesAllRequests) &&
            _cacheMatchCommonPrefixes.size() == _requests.size()) {
            node._node->getMatchedBlockCounts(_requests, counts->data(), _cacheMatchCommonPrefixes);
        } else {
            node._node->getMatchedBlockCounts(_requests, counts->data());
        }
    }
    if (node._peerCacheSuffixLookup && !node.classifiesNewRequestKVLoad() &&
        !(NonLinearPrefillModelR::usesGeometryForVersion(
                                             node.getLatencyBinaryVersion()) ||
                                         NonLinearPrefillModelR::usesNativePP4ForVersion(
                                             node.getLatencyBinaryVersion()))) {
        std::shared_ptr<const std::vector<int64_t>> local =
            std::make_shared<const std::vector<int64_t>>(*counts);
        std::atomic_store(&node._localCacheBlockCounts, local);
        for (size_t i = 0; i < counts->size(); ++i) {
            (*counts)[i] = node._peerCacheSuffixLookup->extend(i, (*counts)[i]);
        }
    }
    std::shared_ptr<const std::vector<int64_t>> immutable = counts;
    std::atomic_compare_exchange_strong(&node._matchedBlockCounts, &published, immutable);
    return published ? published : immutable;
}

int64_t NaviSchedScheduleContext::getCacheHitBlockCount(size_t requestIndex,
                                                        size_t nodeIndex) const
{
    if (requestIndex >= requestCount() || nodeIndex >= nodeCount()) {
        return 0;
    }
    const auto storageRequestIndex = resolveRequestIndex(requestIndex);
    const auto storageNodeIndex = resolveNodeIndex(nodeIndex);
    int64_t matchedBlockCount = 0;
    if (getReplayMatch(storageRequestIndex, storageNodeIndex, matchedBlockCount)) {
        return matchedBlockCount;
    }
    const auto counts = getCacheHitBlockCountsSnapshot(nodeIndex);
    return counts ? getCacheHitBlockCount(*counts, requestIndex) : 0;
}

void NaviSchedScheduleContext::prepareCacheMatchPrefixes()
{
    _cacheMatchCommonPrefixes.assign(requestCount(), 0);
    for (size_t index = 1; index < requestCount(); ++index) {
        const auto &previous = getRequest(index - 1).request.blockHashes;
        const auto &current = getRequest(index).request.blockHashes;
        if (!previous || !current) {
            continue;
        }
        const auto limit = std::min(previous->size(), current->size());
        auto &common = _cacheMatchCommonPrefixes[index];
        if (previous == current) {
            common = limit;
        } else {
            while (common < limit && (*previous)[common] == (*current)[common]) {
                ++common;
            }
        }
    }
}

bool NaviSchedScheduleContext::getCacheHitBlockCounts(
    size_t nodeIndex, std::vector<int64_t> &matchedBlockCounts) const
{
    if (nodeIndex >= nodeCount()) {
        return false;
    }
    matchedBlockCounts.resize(requestCount());
    const auto storageNodeIndex = resolveNodeIndex(nodeIndex);
    if (!_replayMatchRequestIndexes.empty()) {
        bool complete = true;
        for (size_t requestIndex = 0; requestIndex < requestCount(); ++requestIndex) {
            if (!getReplayMatch(resolveRequestIndex(requestIndex), storageNodeIndex,
                                matchedBlockCounts[requestIndex])) {
                complete = false;
                break;
            }
        }
        if (complete) {
            return true;
        }
    }
    const auto &node = _scheduleNodes[storageNodeIndex];
    if (std::atomic_load(&node._matchedBlockCounts) || node.classifiesNewRequestKVLoad() ||
        NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion()) ||
        node._peerCacheSuffixLookup) {
        const auto counts = getCacheHitBlockCountsSnapshot(nodeIndex);
        if (!counts) return false;
        for (size_t index = 0; index < requestCount(); ++index) {
            matchedBlockCounts[index] = getCacheHitBlockCount(*counts, index);
        }
        return true;
    }
    if (!_scheduleViewActive || _scheduleViewUsesAllRequests) {
        _scheduleNodes[storageNodeIndex]._node->getMatchedBlockCounts(
            _requests, matchedBlockCounts.data(), _cacheMatchCommonPrefixes);
        return true;
    }
    _scheduleNodes[storageNodeIndex]._node->getMatchedBlockCounts(
        _scheduleViewRequests, matchedBlockCounts.data(), _cacheMatchCommonPrefixes);
    return true;
}

bool NaviSchedScheduleContext::isCollectingReplay() const { return bool(_replay); }

void NaviSchedScheduleContext::setSchedulerInfo(
    autil::legacy::json::JsonMap schedulerInfo)
{
    if (_replay) {
        const auto pending = _replay->schedulerInfo.find("pending_state");
        if (pending != _replay->schedulerInfo.end()) {
            schedulerInfo["pending_state"] = pending->second;
        }
        _replay->schedulerInfo = std::move(schedulerInfo);
    }
}

void NaviSchedScheduleContext::recordDiscreteSearchLosses(double initialLoss,
                                                          double finalLoss) const
{
    _discreteSearchLosses.reset();
    if (std::isfinite(initialLoss) && std::isfinite(finalLoss)) {
        _discreteSearchLosses.emplace(initialLoss, finalLoss);
    }
}

std::optional<double> NaviSchedScheduleContext::discreteSearchImprovementPct() const
{
    if (!_discreteSearchLosses) {
        return std::nullopt;
    }
    const auto [initialLoss, finalLoss] = *_discreteSearchLosses;
    if (initialLoss == 0.0) {
        return finalLoss == 0.0 ? std::optional<double>(0.0) : std::nullopt;
    }
    // Normalize by |initial| so lower losses always mean positive improvement.
    // Widen before subtraction to keep finite large losses from overflowing.
    const auto initial = static_cast<long double>(initialLoss);
    const auto improvement = static_cast<double>(
        (initial - static_cast<long double>(finalLoss)) / std::abs(initial) * 100.0L);
    return std::isfinite(improvement) ? std::optional<double>(improvement) : std::nullopt;
}

bool NaviSchedScheduleContext::setScheduleLoss(double loss)
{
    if (!std::isfinite(loss)) {
        return false;
    }
    _scheduleLoss = loss;
    return true;
}



std::shared_ptr<const std::vector<TPUPrefillRequestInput>>
NaviSchedScheduleContext::getExecutionCacheInputs(size_t storageNodeIndex) const
{
    const auto inputFailure = [this, storageNodeIndex](const char *reason)
        -> std::shared_ptr<const std::vector<TPUPrefillRequestInput>> {
        if (_scheduleId < 10 || _scheduleId % 100 == 0)
            std::fprintf(stderr, "[navi-master-cache-input-failure] schedule_id=%lu node=%lu reason=%s requests=%lu\n",
                         _scheduleId, storageNodeIndex, reason, _requests.size());
        return nullptr;
    };
    const auto &node = _scheduleNodes[storageNodeIndex];
    auto published = std::atomic_load(&node._executionCacheInputs);
    const bool nativePP4 =
        NonLinearPrefillModelR::usesNativePP4ForVersion(node.getLatencyBinaryVersion());
    const bool tracksKVLoad = node.classifiesNewRequestKVLoad();
    if (published &&
        (!(nativePP4 || tracksKVLoad) || published->peerCacheMerged || !node._peerCacheSuffixLookup)) {
        return {published, &published->inputs};
    }
    // Extend the original local observation once; replay collection must not
    // cause a second physical lookup at a different point in time.
    auto capture = published
                       ? std::make_shared<NaviSchedExecutionCacheInputCapture>(*published)
                       : std::make_shared<NaviSchedExecutionCacheInputCapture>();
    auto inputs =
        std::shared_ptr<std::vector<TPUPrefillRequestInput>>(capture, &capture->inputs);
    // Retain physical HBM separately from merged V6D/Tair hits in every
    // captured cache match, including unified prediction diagnostics.
    if (!published)
        capture->observation.readyOnlyHbmBlockCounts.emplace();
    if (!published && node._remoteCacheReadyBlocks)
        capture->observation.beforeTairBlockCounts.emplace();
    if (!published && !_requests.empty() &&
        !node._node->getExecutionCacheInputs(_requests, *inputs, nullptr,
                                             &capture->observation, node._remoteCacheReadyBlocks.get(),
                                             nativePP4 || NonLinearPrefillModelR::usesGeometryForVersion(
                                                 node.getLatencyBinaryVersion()),
                                             (!_scheduleViewActive || _scheduleViewUsesAllRequests) &&
                                                     _cacheMatchCommonPrefixes.size() == _requests.size()
                                                 ? &_cacheMatchCommonPrefixes : nullptr)) {
        return inputFailure("node_cache_lookup");
    }
    if (inputs->size() != _requests.size()) {
        return inputFailure("input_size");
    }
    if (node._peerCacheSuffixLookup) {
        auto local = std::make_shared<std::vector<int64_t>>(_requests.size());
        for (size_t i = 0; i < inputs->size(); ++i) {
            const auto block = _requests[i].request.blockSize;
            if (block <= 0 || (*inputs)[i].computedTokens < 0 ||
                ((*inputs)[i].computedTokens > (*inputs)[i].promptTokens ||
                 (!(nativePP4 || tracksKVLoad) &&
                  (*inputs)[i].computedTokens == (*inputs)[i].promptTokens)) ||
                (*inputs)[i].computedTokens % block != 0) {
                return inputFailure("peer_input_validation");
            }
            const auto resident =
                (*inputs)[i].externalReadTokens >= 0
                    ? (*inputs)[i].computedTokens - (*inputs)[i].externalReadTokens
                    : -1;
            (*local)[i] = (*inputs)[i].computedTokens / block;
            if (capture->observation.beforeTairBlockCounts) {
                auto &before = capture->observation.beforeTairBlockCounts->at(i);
                before = node._peerCacheSuffixLookup->extend(i, before);
            }
            (*inputs)[i].computedTokens =
                node._peerCacheSuffixLookup->extend(i, (*local)[i]) * block;
            if (resident >= 0) {
                (*inputs)[i].externalReadTokens = (*inputs)[i].computedTokens - resident;
            }
        }
        std::shared_ptr<const std::vector<int64_t>> immutableLocal = local;
        std::atomic_store(&node._localCacheBlockCounts, immutableLocal);
        capture->peerCacheMerged = true;
    }
    for (size_t i = 0; i < _requests.size(); ++i) {
        const auto &request = _requests[i].request;
        const auto &input = (*inputs)[i];
        if (request.blockSize <= 0 || input.promptTokens <= 0 ||
            input.promptTokens != request.promptTokenCount || input.computedTokens < 0 ||
            (input.computedTokens > input.promptTokens ||
             (!(nativePP4 || tracksKVLoad) && input.computedTokens == input.promptTokens)) ||
            input.computedTokens % request.blockSize != 0 ||
            (NonLinearPrefillModelR::usesCacheSourceForVersion(
                 node.getLatencyBinaryVersion()) &&
             (input.externalReadTokens < 0 ||
              input.externalReadTokens > input.computedTokens)) ||
            input.sharedPrefixBoundary < 0 ||
            input.sharedPrefixBoundary > input.promptTokens) {
            return inputFailure("final_input_validation");
        }
    }
    std::shared_ptr<const NaviSchedExecutionCacheInputCapture> immutable = capture;
    // Publish the observation and inputs as one immutable object.
    const bool installed = std::atomic_compare_exchange_strong(
        &node._executionCacheInputs, &published, immutable);
    const auto winner = installed ? immutable : published;
    return {winner, &winner->inputs};
}

const NaviSchedGeometryOptimizationInput *
NaviSchedScheduleContext::getGeometryOptimizationInput(size_t nodeIndex,
                                                       size_t requestIndex) const
{
    if (nodeIndex >= nodeCount() || requestIndex >= requestCount()) {
        return nullptr;
    }
    const auto inputs = getGeometryOptimizationInputs(nodeIndex);
    return inputs ? getGeometryOptimizationInput(*inputs, requestIndex) : nullptr;
}

const NaviSchedGeometryOptimizationInput *
NaviSchedScheduleContext::getGeometryOptimizationInput(
    const std::vector<NaviSchedGeometryOptimizationInput> &inputs,
    size_t requestIndex) const
{
    if (requestIndex >= requestCount()) {
        return nullptr;
    }
    const auto storageRequest = resolveRequestIndex(requestIndex);
    return storageRequest < inputs.size() ? &inputs[storageRequest] : nullptr;
}

std::shared_ptr<const std::vector<NaviSchedGeometryOptimizationInput>>
NaviSchedScheduleContext::getGeometryOptimizationInputs(size_t nodeIndex) const
{
    if (nodeIndex >= nodeCount()) {
        return nullptr;
    }
    const auto storageNode = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNode];
    auto published = std::atomic_load(&node._geometryOptimizationInputs);
    if (!published) {
        const auto &policy = node.getPrefillExecutionPolicy();
        if (!policy || !node.hasValidLatencyModel() ||
            !NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            return nullptr;
        }
        const auto inputs = getExecutionCacheInputs(storageNode);
        if (!inputs) {
            return nullptr;
        }
        auto coordinates =
            std::make_shared<std::vector<NaviSchedGeometryOptimizationInput>>();
        coordinates->reserve(inputs->size());
        const auto &parameters = node.getLatencyParameters();
        const auto paddingCoefficient =
            parameters[3] * NonLinearPrefillModelR::getParameterScale(3) / 1024.0;
        const auto stepCoefficient =
            parameters[6] * NonLinearPrefillModelR::getParameterScale(6);
        for (const auto &input : *inputs) {
            NaviSchedLatencyFeatures features;
            if (!features.appendBatchCount(1)) {
                return nullptr;
            }
            const auto remaining = input.promptTokens - input.computedTokens;
            features.appendRequest(input.computedTokens, remaining);
            NaviSchedGeometryOptimizationInput value;
            value.cacheHit = input.computedTokens > 0;
            value.externalReadTokens = std::max(int64_t{0}, input.externalReadTokens);
            // Remove the old raw-compute and constant-batch coordinates; the
            // continuous group projection below supplies padding and steps.
            value.linearCost =
                NonLinearPrefillModelR::calculateLinearCost(parameters, features) -
                paddingCoefficient * remaining - stepCoefficient;
            const auto independent =
                NonLinearPrefillModelR::usesProjectedGeometryForVersion(
                    node.getLatencyBinaryVersion()) &&
                policy->asyncKVLookupEnabled &&
                (input.promptTokens - (policy->asyncKVLookupUsesEagle ? 1 : 0)) /
                        policy->asyncKVLookupHashBlockSize >
                    0;
            value.shared = !independent;
            if (independent) {
                TPUPrefillExecutionProjection projection;
                if (!TPUPrefillExecutionPolicy::projectLatencyFeaturesSummary(
                        {input}, *policy, static_cast<size_t>(remaining), projection,
                        nullptr)) {
                    return nullptr;
                }
                value.linearCost += paddingCoefficient * projection.globalPaddedTokenSum +
                                    stepCoefficient * projection.stepCount;
            } else {
                value.sharedTokens = remaining;
                if (policy->fixedPrefillChunkPlan) {
                    TPUPrefillExecutionProjection projection;
                    if (!TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
                            {input}, *policy, static_cast<size_t>(remaining), projection,
                            nullptr)) {
                        return nullptr;
                    }
                    value.fixedChunkCount = projection.stepCount;
                    value.fixedAttentionPairs = projection.attentionPairSum;
                }
            }
            coordinates->emplace_back(value);
        }
        std::shared_ptr<const std::vector<NaviSchedGeometryOptimizationInput>> immutable =
            coordinates;
        std::atomic_compare_exchange_strong(&node._geometryOptimizationInputs, &published,
                                            immutable);
        if (!published) {
            published = std::move(immutable);
        }
    }
    return published;
}

bool NaviSchedScheduleContext::buildExecutionGeometryFeatures(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    NaviSchedLatencyFeatures &features) const
{
    features = NaviSchedLatencyFeatures();
    if (nodeIndex >= nodeCount() || requestIndexes.empty()) {
        return false;
    }
    const auto storageNode = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNode];
    const auto &policy = node.getPrefillExecutionPolicy();
    if (!node.hasValidLatencyModel() || !policy ||
        !NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion()) ||
        node.usesUnifiedLatencyPrediction()) {
        return false;
    }
    const auto inputs = getExecutionCacheInputs(storageNode);
    if (!inputs) {
        return false;
    }
    NaviSchedLatencyFeatures logical;
    if (!logical.appendBatchCount(1)) {
        return false;
    }
    std::vector<TPUPrefillRequestInput> ordered;
    ordered.reserve(requestIndexes.size());
    if (std::is_sorted(requestIndexes.begin(), requestIndexes.end())) {
        if (std::adjacent_find(requestIndexes.begin(), requestIndexes.end()) !=
            requestIndexes.end()) {
            return false;
        }
    } else {
        auto sorted = requestIndexes;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return false;
        }
    }
    // Whole-chunk deferral makes geometry order-sensitive. Use the same
    // canonical batch order as native forwarding and queued reservations,
    // independent of context views, search iteration and RPC arrival order.
    auto canonicalIndexes = requestIndexes;
    for (const auto index : canonicalIndexes) {
        if (index >= requestCount()) {
            return false;
        }
    }
    std::stable_sort(canonicalIndexes.begin(), canonicalIndexes.end(),
                     [this](size_t left, size_t right) {
                         return getRequest(left).request.requestId <
                                getRequest(right).request.requestId;
                     });
    size_t maxSteps = 0;
    int64_t totalPrompt = 0;
    for (const auto index : canonicalIndexes) {
        const auto storageIndex = resolveRequestIndex(index);
        const auto &input = (*inputs)[storageIndex];
        const auto remaining = input.promptTokens - input.computedTokens;
        if (input.promptTokens > std::numeric_limits<int64_t>::max() - totalPrompt ||
            static_cast<uint64_t>(remaining) >
                std::numeric_limits<size_t>::max() - maxSteps) {
            return false;
        }
        totalPrompt += input.promptTokens;
        maxSteps += static_cast<size_t>(remaining);
        logical.appendRequest(input.computedTokens, remaining);
        ordered.emplace_back(input);
    }
    TPUPrefillExecutionProjection projection;
    const auto projector =
        NonLinearPrefillModelR::usesProjectedGeometryForVersion(
            node.getLatencyBinaryVersion())
            ? TPUPrefillExecutionPolicy::projectLatencyFeaturesSummary
            : TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary;
    if (!projector(ordered, *policy, maxSteps, projection, nullptr) ||
        !logical.setExecutionGeometry(projection.stepCount,
                                      projection.globalPaddedTokenSum)) {
        return false;
    }
    if (NonLinearPrefillModelR::usesCacheSourceForVersion(
            node.getLatencyBinaryVersion())) {
        int64_t reads = 0, maximum = 0;
        for (const auto &input : ordered) {
            reads += input.externalReadTokens > 0;
            maximum = std::max(maximum, input.externalReadTokens);
        }
        if (!logical.setExecutionCacheSource(reads, maximum)) {
            return false;
        }
    }
    features = std::move(logical);
    return true;
}

namespace {
size_t nativeTransitionLimit(const TPUPrefillPP4ObservationInput &input)
{
    size_t units = input.requests.size() + input.state.results.size() + 1;
    for (const auto &request : input.requests) {
        if (request.remainingTokens < 0 || uint64_t(request.remainingTokens) >
                                               std::numeric_limits<size_t>::max() - units)
            throw std::overflow_error("Native forecast size exceeds representable bound");
        units += static_cast<size_t>(request.remainingTokens);
    }
    if (units > std::numeric_limits<size_t>::max() / 8)
        throw std::overflow_error("Native transition bound exceeds representable range");
    return units * 8;
}
}

TPUPrefillPP4ObservationInput NaviSchedScheduleContext::prepareNativeForecastInput(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    double arrivalDelayMs) const
{
    if (nodeIndex >= nodeCount())
        throw std::invalid_argument("invalid Native node index");
    const auto storageNode = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNode];
    const auto base = node.getNativePrefillObservation();
    const auto inputs = getExecutionCacheInputs(storageNode);
    if (!inputs || !node.getPrefillExecutionPolicy())
        throw std::invalid_argument("missing Native execution inputs");
    const auto capture = std::atomic_load(&node._executionCacheInputs);
    if (!capture || !capture->observation.readyOnlyHbmBlockCounts ||
        capture->observation.readyOnlyHbmBlockCounts->size() != inputs->size())
        throw std::invalid_argument("missing Native physical HBM observation");
    auto result = *base;
    std::vector<TPUPrefillPP4Request> incoming;
    for (const auto index : requestIndexes) {
        if (index >= requestCount())
            throw std::invalid_argument("invalid Native request index");
        const auto storage = resolveRequestIndex(index);
        const auto &request = _requests[storage].request;
        const auto &input = (*inputs)[storage];
        const auto readyHbmBlocks = (*capture->observation.readyOnlyHbmBlockCounts)[storage];
        if (readyHbmBlocks < -1 || request.blockSize <= 0 ||
            readyHbmBlocks > input.promptTokens / request.blockSize)
            throw std::invalid_argument("invalid Native physical HBM observation");
        auto value = unobservedNativeRequest(
            request.requestId, input.promptTokens,
            request.explicitCacheCheckpointPosition, request.nativePrefillSlotRelease,
            input.computedTokens, true, *node.getPrefillExecutionPolicy(), {},
            readyHbmBlocks >= 0
                ? std::optional<int64_t>(readyHbmBlocks * request.blockSize)
                : std::nullopt);
        if (request.explicitCacheCheckpointPosition <= 0)
            value.sharedPrefixBoundary = input.sharedPrefixBoundary;
        value.readyTimeMs = base->state.nowMs + arrivalDelayMs;
        incoming.push_back(std::move(value));
    }
    std::sort(incoming.begin(), incoming.end(), [](const auto &left, const auto &right) {
        return left.requestId < right.requestId;
    });
    result.requests.insert(result.requests.end(), incoming.begin(), incoming.end());
    return result;
}

std::shared_ptr<const NaviSchedNativeContinuousInput>
NaviSchedScheduleContext::getNativeContinuousInput(size_t nodeIndex) const
{
    if (nodeIndex >= nodeCount() || requestCount() == 0) return nullptr;
    const auto &node = _scheduleNodes[resolveNodeIndex(nodeIndex)];
    auto published = std::atomic_load(&node._nativeContinuousInput);
    if (published) return published;
    try {
        if (!node.hasValidLatencyModel() ||
            !NonLinearPrefillModelR::usesNativePP4ForVersion(
                node.getLatencyBinaryVersion()))
            return nullptr;
        std::vector<size_t> indexes;
        std::unordered_map<uint64_t, size_t> requestIndexes;
        for (size_t index = 0; index < requestCount(); ++index) {
            indexes.push_back(index);
            requestIndexes.emplace(getRequest(index).request.requestId, index);
        }
        const auto input = prepareNativeForecastInput(nodeIndex, indexes);
        auto prepared = std::make_shared<NaviSchedNativeContinuousInput>();
        prepared->state = input.state;
        prepared->slots = input.state.slots;
        std::copy(node.getLatencyParameters().begin(), node.getLatencyParameters().end(),
                  prepared->parameters.begin());
        const auto &parameters = prepared->parameters;
        const auto &policy = *node.getPrefillExecutionPolicy();
        const auto fixedCost = [&](const std::array<double, 8> &features) {
            long double sum = 0.0L;
            for (size_t i = 0; i < 8; ++i) sum += features[i] * parameters[i + 1];
            return static_cast<double>(sum);
        };
        for (const auto &slot : prepared->slots) {
            NaviSchedNativeContinuousSlot cost;
            cost.fixed = true;
            cost.fixedCostMs = fixedCost(slot.features);
            prepared->costs.push_back(std::move(cost));
        }
        if (input.state.pendingDispatch && !input.state.pendingDispatch->spans.empty()) {
            const auto &pending = *input.state.pendingDispatch;
            TPUPrefillPipelineSlot slot;
            slot.ordinal =
                prepared->slots.empty() ? 0 : prepared->slots.back().ordinal + 1;
            slot.dispatchTimeMs = pending.dispatchTimeMs;
            slot.features = pending.features;
            prepared->slots.push_back(slot);
            NaviSchedNativeContinuousSlot cost;
            cost.fixed = true;
            cost.fixedCostMs = fixedCost(slot.features);
            prepared->costs.push_back(cost);
        }
        const auto forecast = TPUPrefillPP4Forecast::predict(
            input.state, input.requests, policy, parameters, nativeTransitionLimit(input),
            true);
        prepared->referenceTotalMs = forecast.totalMs;
        const double minimumPadding =
            double(policy.pcpSize) * policy.compileSizes.front();
        for (const auto &dispatch : forecast.dispatches) {
            TPUPrefillPipelineSlot slot;
            slot.ordinal = dispatch.ordinal;
            slot.dispatchTimeMs = dispatch.timeMs;
            NaviSchedNativeContinuousSlot cost;
            if (!dispatch.real)
                cost.fixed = true;
            else {
                std::vector<int64_t> counts;
                int64_t scheduled = 0;
                for (const auto &span : dispatch.spans) {
                    counts.push_back(span.scheduledTokens);
                    scheduled += span.scheduledTokens;
                }
                TPUPrefillStepGeometry geometry;
                if (!TPUPrefillExecutionPolicy::projectStepGeometry(counts, policy,
                                                                    geometry))
                    return nullptr;
                const double extraPadding = geometry.globalPaddedTokens - minimumPadding;
                if (scheduled <= 0 || extraPadding < 0) return nullptr;
                cost.sharedStepCostMs =
                    parameters[1] + parameters[3] * minimumPadding / 1024.0;
                for (const auto &span : dispatch.spans) {
                    const auto found = requestIndexes.find(span.requestId);
                    NaviSchedNativeContinuousSpan contribution;
                    if (found == requestIndexes.end())
                        cost.hasFixedRequest = true;
                    else {
                        contribution.requestIndex = found->second;
                        prepared->targetOrdinal = dispatch.ordinal;
                    }
                    const double c = span.computedTokens, q = span.scheduledTokens;
                    contribution.contextKTokens = c / 1024.0;
                    contribution.linearCostMs =
                        parameters[2] * q / 1024.0 +
                        parameters[3] * extraPadding * q / scheduled / 1024.0 +
                        parameters[4] * c * q / 1048576.0 + parameters[5] * c / 1024.0 +
                        parameters[6] * q * q / 1048576.0 + parameters[7];
                    cost.spans.push_back(contribution);
                }
            }
            prepared->slots.push_back(slot);
            prepared->costs.push_back(std::move(cost));
        }
        if (prepared->targetOrdinal < 0 ||
            prepared->slots.size() != prepared->costs.size())
            return nullptr;
        if (!CostSchedulerKernelInternal::initializeNativeContinuousReference(
                *prepared, requestCount()))
            return nullptr;
        std::shared_ptr<const NaviSchedNativeContinuousInput> immutable = prepared;
        std::atomic_compare_exchange_strong(&node._nativeContinuousInput, &published,
                                            immutable);
        return published ? published : immutable;
    } catch (const std::logic_error &) {
        return nullptr;
    } catch (const std::runtime_error &) {
        return nullptr;
    }
}

double NaviSchedNativeFinalPrediction::recomputeBaseLatencyMs(
    const TPUPrefillPP4Timing::Parameters &weights) const
{
    return TPUPrefillPP4Forecast::predict(input.state, input.requests, policy, weights,
                                          maxTransitions, false)
               .totalMs -
           arrivalDelayMs;
}

double NaviSchedScheduleContext::estimateBatchExecutionLatencyMs(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    NaviSchedLatencyPrediction *latencyPrediction) const
{
    return estimateBatchExecutionLatencyMs(nodeIndex, requestIndexes, latencyPrediction,
                                           nullptr);
}

double NaviSchedScheduleContext::estimateBatchWorkCompletionLatencyMs(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes) const
{
    return estimateBatchExecutionLatencyMs(nodeIndex, requestIndexes, nullptr,
                                           nullptr, 0, false);
}

double NaviSchedScheduleContext::estimateBatchExecutionLatencyMs(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    NaviSchedLatencyPrediction *latencyPrediction,
    const std::vector<int64_t> *matchedBlockCounts, int64_t predictionTimeUs,
    bool targetCompletionScope) const
{
    if (latencyPrediction) {
        *latencyPrediction = NaviSchedLatencyPrediction();
    }
    if (nodeIndex >= nodeCount() || requestIndexes.empty()) {
        return 0.0;
    }
    const auto storageNodeIndex = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNodeIndex];
    if (NonLinearPrefillModelR::usesNativePP4ForVersion(node.getLatencyBinaryVersion())) {
        if (latencyPrediction) {
            std::atomic_store(&node._nativeFinalPrediction,
                              std::shared_ptr<const NaviSchedNativeFinalPrediction>());
        }
        if (!node.hasValidLatencyModel()) return 0.0;
        try {
            // Keep observed replies and unfinished bounds on their original
            // clock. New work is constructed at result preparation's later time.
            const double arrivalDelayMs =
                predictionTimeUs > 0 && predictionTimeUs > _scheduleTimeUs
                    ? static_cast<double>(
                          (static_cast<long double>(predictionTimeUs) - _scheduleTimeUs) /
                          1000.0L)
                    : 0.0;
            auto input =
                prepareNativeForecastInput(nodeIndex, requestIndexes, arrivalDelayMs);
            const double arrivalTimeMs = input.state.nowMs + arrivalDelayMs;
            TPUPrefillPP4Timing::Parameters parameters;
            std::copy(node.getLatencyParameters().begin(),
                      node.getLatencyParameters().end(), parameters.begin());
            const auto maxTransitions = nativeTransitionLimit(input);
            const auto forecast = TPUPrefillPP4Forecast::predict(
                input.state, input.requests, *node.getPrefillExecutionPolicy(),
                parameters, maxTransitions, false);
            const double remainingMs =
                forecast.totalMs - (arrivalTimeMs - input.state.nowMs);
            const double finalMs = node.applyLatencyResidual(remainingMs);
            if (latencyPrediction) {
                auto captured = std::make_shared<NaviSchedNativeFinalPrediction>();
                captured->input = std::move(input);
                captured->policy = *node.getPrefillExecutionPolicy();
                captured->parameters = parameters;
                captured->maxTransitions = maxTransitions;
                captured->predictionTimeUs =
                    predictionTimeUs > 0 ? std::max(_scheduleTimeUs, predictionTimeUs)
                                         : _scheduleTimeUs;
                captured->arrivalDelayMs = arrivalDelayMs;
                captured->baseExecutionMs = remainingMs;
                captured->finalExecutionMs = finalMs;
                std::shared_ptr<const NaviSchedNativeFinalPrediction> immutable =
                    std::move(captured);
                std::atomic_store(&node._nativeFinalPrediction, std::move(immutable));
            }
            return finalMs;
        } catch (const std::logic_error &failure) {
            if (_scheduleId < 10 || _scheduleId % 100 == 0)
                std::fprintf(stderr, "[navi-master-native-forecast-failure] schedule_id=%lu node=%lu error=%s\n",
                             _scheduleId, nodeIndex, failure.what());
            return std::numeric_limits<double>::quiet_NaN();
        } catch (const std::runtime_error &failure) {
            if (_scheduleId < 10 || _scheduleId % 100 == 0)
                std::fprintf(stderr, "[navi-master-native-forecast-failure] schedule_id=%lu node=%lu error=%s\n",
                             _scheduleId, nodeIndex, failure.what());
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion())) {
        NaviSchedLatencyFeatures geometryFeatures;
        if (!buildExecutionGeometryFeatures(nodeIndex, requestIndexes,
                                            geometryFeatures)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto prediction = node.predictLatency(geometryFeatures);
        if (latencyPrediction && std::isfinite(prediction)) {
            latencyPrediction->features = geometryFeatures;
            latencyPrediction->recomputable = true;
        }
        return prediction;
    }
    if (!node.hasValidLatencyModel()) {
        return 0.0;
    }
    const auto capacitySplit =
        NonLinearPrefillModelR::usesCapacitySplitParameters(node.getLatencyParameters());
    NaviSchedLatencyFeatures requestFeatures;
    NaviSchedLatencyFeatures requestFeaturesWithoutReportedWorkload;
    auto workloadSlots = node._workloadRunningCapacitySlotCount;
    auto reservedSlots = node._reservedRunningCapacitySlotCount;
    if (capacitySplit) {
        if (!requestFeatures.appendCapacitySplitAggregate(1, 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                                                          0.0, 0.0) ||
            !requestFeaturesWithoutReportedWorkload.appendCapacitySplitAggregate(
                1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0)) {
            return 0.0;
        }
    } else if (!requestFeatures.appendBatchCount(1)) {
        return 0.0;
    }
    for (auto requestIndex : requestIndexes) {
        if (requestIndex >= requestCount()) {
            return 0.0;
        }
        const auto storageRequestIndex = resolveRequestIndex(requestIndex);
        const auto &request = _requests[storageRequestIndex].request;
        auto promptTokenCount = request.promptTokenCount;
        auto blockSize = request.blockSize;
        auto matchedBlockCount = matchedBlockCounts
                                     ? (*matchedBlockCounts)[requestIndex]
                                     : getCacheHitBlockCount(requestIndex, nodeIndex);
        auto matchedTokenCount = matchedBlockCount * blockSize;
        const auto computeTokenCount = promptTokenCount - matchedTokenCount;
        if (!capacitySplit) {
            requestFeatures.appendRequest(matchedTokenCount, computeTokenCount);
            continue;
        }
        if (workloadSlots < node._maxRunningRequestCount) {
            requestFeatures.appendRequest(matchedTokenCount, computeTokenCount);
            ++workloadSlots;
        } else {
            requestFeatures.appendWaitingRequest(matchedTokenCount, computeTokenCount);
        }
        if (reservedSlots < node._maxRunningRequestCount) {
            requestFeaturesWithoutReportedWorkload.appendRequest(matchedTokenCount,
                                                                 computeTokenCount);
            ++reservedSlots;
        } else {
            requestFeaturesWithoutReportedWorkload.appendWaitingRequest(
                matchedTokenCount, computeTokenCount);
        }
    }
    if (!node.usesUnifiedLatencyPrediction()) {
        if (!node.prepareLatencyPredictionFeatures(requestFeatures)) return 0.0;
        const auto prediction = node.predictLatency(requestFeatures);
        if (latencyPrediction) {
            latencyPrediction->features = requestFeatures;
            latencyPrediction->recomputable = true;
        }
        return prediction;
    }
    auto withReportedWorkload = requestFeatures;
    if (!withReportedWorkload.append(node.getWorkloadFeatures())) {
        return 0.0;
    }
    auto withoutReportedWorkload =
        capacitySplit ? requestFeaturesWithoutReportedWorkload : requestFeatures;
    NaviSchedLatencyFeatures floorFeatures;
    if (!node.getUnifiedDecayFloorFeatures(floorFeatures) ||
        !withoutReportedWorkload.append(floorFeatures)) {
        return 0.0;
    }
    if (!node.prepareLatencyPredictionFeatures(withReportedWorkload) ||
        !node.prepareLatencyPredictionFeatures(withoutReportedWorkload)) return 0.0;
    const auto unpackedSteps = withReportedWorkload.getBatchCount();
    bool packedSteps = false;
    bool completionScoped = false;
    if (node._nativeStepTokenBudget > 0 && node._nativeStepPackingRequests &&
        node._prefillExecutionPolicy) {
        auto withRequests = *node._nativeStepPackingRequests;
        std::vector<NativePrefillStepRequest> floorRequests;
        std::vector<uint64_t> targetRequestIds;
        for (const auto &request : withRequests)
            if (!request.reported) floorRequests.push_back(request);
        const auto inputs = getExecutionCacheInputs(nodeIndex);
        bool complete = inputs && inputs->size() == _requests.size();
        for (auto requestIndex : requestIndexes) {
            if (!complete) break;
            const auto storageIndex = resolveRequestIndex(requestIndex);
            const auto &input = (*inputs)[storageIndex];
            const auto &request = _requests[storageIndex].request;
            const auto matched = (matchedBlockCounts ? (*matchedBlockCounts)[requestIndex]
                                                    : getCacheHitBlockCount(requestIndex, nodeIndex)) * request.blockSize;
            if (matched != input.computedTokens || request.promptTokenCount != input.promptTokens) {
                complete = false;
                break;
            }
            if (input.promptTokens > input.computedTokens) {
                NativePrefillStepRequest member{request.requestId, input, false, false};
                withRequests.push_back(member);
                floorRequests.push_back(member);
                targetRequestIds.push_back(request.requestId);
            }
        }
        auto packedWith = withReportedWorkload;
        auto packedFloor = withoutReportedWorkload;
        if (complete && NativePrefillStepForecast::prepare(
                packedWith, withRequests, *node._prefillExecutionPolicy) &&
            NativePrefillStepForecast::prepare(
                packedFloor, floorRequests, *node._prefillExecutionPolicy)) {
            if (targetCompletionScope) {
                auto scopedWith = packedWith, scopedFloor = packedFloor;
                if (NativePrefillStepForecast::prepareUntilTargets(
                        scopedWith, withRequests, *node._prefillExecutionPolicy, targetRequestIds) &&
                    NativePrefillStepForecast::prepareUntilTargets(
                        scopedFloor, floorRequests, *node._prefillExecutionPolicy, targetRequestIds)) {
                    packedWith = std::move(scopedWith);
                    packedFloor = std::move(scopedFloor);
                    completionScoped = true;
                }
            }
            withReportedWorkload = std::move(packedWith);
            withoutReportedWorkload = std::move(packedFloor);
            packedSteps = true;
        }
    }
    if (node._nativeStepTokenBudget > 0 && latencyPrediction && _scheduleId % 20 == 0)
        std::fprintf(stderr, "[navi-master-native-step-packing] schedule_id=%lu node=%lu applied=%d metadata_available=%d old_steps=%ld new_steps=%ld floor_steps=%ld completion_scoped=%d\n",
                     _scheduleId, nodeIndex, int(packedSteps),
                     int(node._nativeStepPackingRequests.has_value()), unpackedSteps,
                     withReportedWorkload.getBatchCount(), withoutReportedWorkload.getBatchCount(),
                     int(completionScoped));
    const auto withReportedPrediction = node.predictFrozenLatency(withReportedWorkload);
    const auto withoutReportedPrediction = node.predictFrozenLatency(withoutReportedWorkload);
    const auto prediction =
        node.decayUnifiedLatency(withReportedPrediction, withoutReportedPrediction);
    if (latencyPrediction) {
        latencyPrediction->features = std::move(withReportedWorkload);
        latencyPrediction->floorFeatures = std::move(withoutReportedWorkload);
        latencyPrediction->hasFloor = node._workloadReportReceiveTimeUs > 0 &&
                                      (node._reportedWorkloadFeatures.getBatchCount() > 0 ||
                                       node._reportedWorkloadFeatures.getExpectedKVWorkload());
        if (latencyPrediction->hasFloor && !node._runningElapsedProgressApplied &&
            !node._reportedWorkloadFeatures.getExpectedKVWorkload()) {
            latencyPrediction->fixedLatencyMs = -node.getWorkloadReportAgeMs();
        }
        latencyPrediction->recomputable = true;
    }
    return prediction;
}

bool NaviSchedScheduleContext::selectNode(
    size_t requestIndex, size_t nodeIndex, const std::string &reason,
    NaviSchedLatencyTrainingPhase latencyTrainingPhase)
{
    if (!_prepared || requestIndex >= requestCount() || nodeIndex >= nodeCount() ||
        reason.empty() || !validLatencyTrainingPhase(latencyTrainingPhase)) {
        return false;
    }
    requestIndex = resolveRequestIndex(requestIndex);
    nodeIndex = resolveNodeIndex(nodeIndex);
    auto &decision = _decisions[requestIndex];
    decision.reason = reason;
    decision.latencyTrainingPhase = latencyTrainingPhase;
    _selectedNodeIndexes[requestIndex] = nodeIndex;
    return true;
}

bool NaviSchedScheduleContext::reject(size_t requestIndex, const std::string &reason)
{
    if (!_prepared || requestIndex >= requestCount() || reason.empty()) {
        return false;
    }
    requestIndex = resolveRequestIndex(requestIndex);
    _decisions[requestIndex] = NaviSchedDecision();
    _decisions[requestIndex].reason = reason;
    _selectedNodeIndexes[requestIndex] = INVALID_NODE_INDEX;
    return true;
}

void NaviSchedScheduleContext::addRequest(const MasterScheduleRequestDataPtr &request)
{
    _responseRequests.emplace_back(request);
}

void NaviSchedScheduleContext::setNodes(std::vector<NaviSchedNodeConstPtr> nodes,
                                        std::vector<NaviSchedNodeSnapshot> nodeSnapshots)
{
    clearScheduleView();
    _scheduleNodes.clear();
    if (nodes.size() != nodeSnapshots.size()) {
        return;
    }
    _scheduleNodes.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        _scheduleNodes.push_back(NaviSchedScheduleNode(
            std::move(nodes[index]), std::move(nodeSnapshots[index]), _scheduleTimeUs,
            _timeCompression));
    }
}

bool NaviSchedScheduleContext::activateScheduleView(
    const std::vector<size_t> &requestIndexes, const std::vector<size_t> &nodeIndexes)
{
    if (_scheduleViewActive || requestIndexes.empty() || nodeIndexes.empty()) {
        return false;
    }
    std::vector<bool> requestSeen(_requests.size(), false);
    for (const auto requestIndex : requestIndexes) {
        if (requestIndex >= _requests.size() || requestSeen[requestIndex]) {
            return false;
        }
        requestSeen[requestIndex] = true;
    }
    std::vector<bool> nodeSeen(_scheduleNodes.size(), false);
    for (const auto nodeIndex : nodeIndexes) {
        if (nodeIndex >= _scheduleNodes.size() || nodeSeen[nodeIndex]) {
            return false;
        }
        nodeSeen[nodeIndex] = true;
    }
    _scheduleViewRequestIndexes = requestIndexes;
    _scheduleViewNodeIndexes = nodeIndexes;
    _scheduleViewUsesAllRequests = requestIndexes.size() == _requests.size();
    for (size_t index = 0; _scheduleViewUsesAllRequests && index < requestIndexes.size();
         ++index) {
        _scheduleViewUsesAllRequests = requestIndexes[index] == index;
    }
    if (!_scheduleViewUsesAllRequests) {
        _scheduleViewRequests.reserve(requestIndexes.size());
        for (const auto requestIndex : requestIndexes) {
            _scheduleViewRequests.emplace_back(_requests[requestIndex]);
        }
    }
    _scheduleViewActive = true;
    for (auto &node : _scheduleNodes)
        std::atomic_store(&node._nativeContinuousInput,
                          std::shared_ptr<const NaviSchedNativeContinuousInput>());
    _cacheMatchCommonPrefixes.clear();
    return true;
}

void NaviSchedScheduleContext::clearScheduleView()
{
    for (auto &node : _scheduleNodes)
        std::atomic_store(&node._nativeContinuousInput,
                          std::shared_ptr<const NaviSchedNativeContinuousInput>());
    _cacheMatchCommonPrefixes.clear();
    _scheduleViewRequestIndexes.clear();
    _scheduleViewNodeIndexes.clear();
    _scheduleViewRequests.clear();
    _scheduleViewUsesAllRequests = false;
    _scheduleViewActive = false;
}

size_t NaviSchedScheduleContext::resolveRequestIndex(size_t requestIndex) const
{
    return _scheduleViewActive ? _scheduleViewRequestIndexes[requestIndex] : requestIndex;
}

size_t NaviSchedScheduleContext::resolveNodeIndex(size_t nodeIndex) const
{
    return _scheduleViewActive ? _scheduleViewNodeIndexes[nodeIndex] : nodeIndex;
}

void NaviSchedScheduleContext::setReplay(
    const std::shared_ptr<NaviSchedScheduleReplay> &replay)
{
    _replay = replay;
    _replayNativeCacheInputs.clear();
    _replayNativeLocalCacheCounts.clear();
    _replayMatchRequestIndexes.clear();
    if (!replay || replay->nodes.size() != _scheduleNodes.size() ||
        replay->requests.size() != replay->nodeMatches.size()) {
        return;
    }
    for (size_t index = 0; index < replay->nodes.size(); ++index) {
        if (!_scheduleNodes[index]._info ||
            replay->nodes[index].agentId != _scheduleNodes[index]._info->agentId) {
            return;
        }
    }
    for (size_t index = 0; index < replay->requests.size(); ++index) {
        if (replay->nodeMatches[index].size() != _scheduleNodes.size() ||
            !_replayMatchRequestIndexes
                 .emplace(replay->requests[index].request.requestId, index)
                 .second) {
            _replayMatchRequestIndexes.clear();
            return;
        }
    }
    if (replay->requests.size() != _requests.size()) return;
    for (size_t i = 0; i < _requests.size(); ++i)
        if (replay->requests[i].request.requestId != _requests[i].request.requestId)
            return;
    for (size_t i = 0; i < _scheduleNodes.size(); ++i) {
        const auto &node = _scheduleNodes[i];
        if (!NonLinearPrefillModelR::usesNativePP4ForVersion(
                node.getLatencyBinaryVersion()))
            continue;
        if (_replayNativeCacheInputs.empty()) {
            _replayNativeCacheInputs.resize(_scheduleNodes.size());
            _replayNativeLocalCacheCounts.resize(_scheduleNodes.size());
        }
        _replayNativeCacheInputs[i] = std::atomic_load(&node._executionCacheInputs);
        _replayNativeLocalCacheCounts[i] = std::atomic_load(&node._localCacheBlockCounts);
    }
}

void NaviSchedScheduleContext::retainNativeCacheInputs(
    const std::vector<size_t> &requestIndexes)
{
    const auto select =
        [&requestIndexes](const std::shared_ptr<const std::vector<int64_t>> &values) {
            if (!values) return values;
            std::vector<int64_t> selected;
            selected.reserve(requestIndexes.size());
            for (const auto index : requestIndexes) selected.push_back(values->at(index));
            return std::make_shared<const std::vector<int64_t>>(std::move(selected));
        };
    for (size_t i = 0; i < _scheduleNodes.size(); ++i) {
        auto &node = _scheduleNodes[i];
        if (!NonLinearPrefillModelR::usesNativePP4ForVersion(
                node.getLatencyBinaryVersion()))
            continue;
        // Resolve once while all original request identities are still available.
        // Later scoring reuses the selected rows instead of reading cache again.
        getExecutionCacheInputs(i);
        const auto cache = std::atomic_load(&node._executionCacheInputs);
        if (cache) {
            auto selected = std::make_shared<NaviSchedExecutionCacheInputCapture>();
            selected->observation.pool = cache->observation.pool;
            if (cache->observation.readyOnlyHbmBlockCounts)
                selected->observation.readyOnlyHbmBlockCounts.emplace();
            if (cache->observation.beforeTairBlockCounts)
                selected->observation.beforeTairBlockCounts.emplace();
            selected->peerCacheMerged = cache->peerCacheMerged;
            for (const auto index : requestIndexes) {
                selected->inputs.push_back(cache->inputs.at(index));
                if (cache->observation.beforeTairBlockCounts)
                    selected->observation.beforeTairBlockCounts->push_back(
                        cache->observation.beforeTairBlockCounts->at(index));
                if (!cache->observation.fullPrefixBlocksBeforeQueryDrop.empty())
                    selected->observation.fullPrefixBlocksBeforeQueryDrop.push_back(
                        cache->observation.fullPrefixBlocksBeforeQueryDrop.at(index));
                if (cache->observation.readyOnlyHbmBlockCounts)
                    selected->observation.readyOnlyHbmBlockCounts->push_back(
                        cache->observation.readyOnlyHbmBlockCounts->at(index));
            }
            std::shared_ptr<const NaviSchedExecutionCacheInputCapture> immutable =
                std::move(selected);
            std::atomic_store(&node._executionCacheInputs, std::move(immutable));
        }
        std::atomic_store(&node._matchedBlockCounts,
                          select(std::atomic_load(&node._matchedBlockCounts)));
        std::atomic_store(&node._localCacheBlockCounts,
                          select(std::atomic_load(&node._localCacheBlockCounts)));
        std::atomic_store(&node._nativeContinuousInput,
                          std::shared_ptr<const NaviSchedNativeContinuousInput>());
        std::atomic_store(&node._nativeFinalPrediction,
                          std::shared_ptr<const NaviSchedNativeFinalPrediction>());
    }
    if (!_peerCacheRequestKeyIndexes.empty()) {
        decltype(_peerCacheRequestKeyIndexes) selected;
        selected.reserve(requestIndexes.size());
        for (const auto index : requestIndexes)
            selected.push_back(_peerCacheRequestKeyIndexes.at(index));
        _peerCacheRequestKeyIndexes = std::move(selected);
    }
}

bool NaviSchedScheduleContext::getReplayMatch(size_t storageRequestIndex,
                                              size_t storageNodeIndex,
                                              int64_t &matchedBlockCount) const
{
    const auto it =
        _replayMatchRequestIndexes.find(_requests[storageRequestIndex].request.requestId);
    if (it == _replayMatchRequestIndexes.end()) {
        return false;
    }
    // Queue-overload filtering compacts _requests after replay capture.
    // Retain the original row by request ID so costs use the recorded snapshot.
    matchedBlockCount =
        _replay->nodeMatches[it->second][storageNodeIndex].matchedBlockCount;
    return true;
}

bool NaviSchedScheduleContext::prepareResult(int64_t queuePredictionTimeUs)
{
    // The Master reserves this round after the first result preparation.  Reusing
    // that result prevents the response from matching its own reservation.
    if (_resultPrepared) {
        return true;
    }
    const auto preparationFailure = [this](const char *reason) {
        if (_scheduleId < 10 || _scheduleId % 100 == 0)
            std::fprintf(stderr, "[navi-master-prepare-result-failure] schedule_id=%lu reason=%s requests=%lu nodes=%lu decisions=%lu selections=%lu\n",
                         _scheduleId, reason, requestCount(), nodeCount(), _decisions.size(), _selectedNodeIndexes.size());
        return false;
    };
    _queuePredictionTimeUs = std::max(_scheduleTimeUs, queuePredictionTimeUs);
    const auto requestSize = requestCount();
    const auto nodeSize = nodeCount();
    if (_decisions.size() != requestSize || _selectedNodeIndexes.size() != requestSize) {
        return preparationFailure("decision_size");
    }
    struct NodeBatchState {
        size_t requestCount = 0;
        uint64_t batchHash = 0;
        int64_t batchId = 0;
        std::vector<size_t> requestIndexes;
        double predictedQueueLatencyMs = 0.0;
        double predictedExecutionLatencyMs = 0.0;
        NaviSchedLatencyPredictionConstPtr latencyPrediction;
        std::shared_ptr<const NaviSchedExecutionCacheInputCapture> cacheCapture;
    };
    std::vector<NodeBatchState> nodeBatchStates(nodeSize);
    std::vector<size_t> selectedRequestCounts(nodeSize, 0);
    std::vector<int64_t> matchedBlockCounts(requestSize, 0);
    for (size_t requestIndex = 0; requestIndex < requestSize; ++requestIndex) {
        auto nodeIndex = _selectedNodeIndexes[requestIndex];
        if (INVALID_NODE_INDEX == nodeIndex) {
            continue;
        }
        if (nodeIndex >= nodeSize) {
            return preparationFailure("node_index");
        }
        const auto inflightRequests =
            std::max<int32_t>(0, _scheduleNodes[nodeIndex]._inflightRequests);
        const auto remainingCapacity =
            inflightRequests < NAVI_SCHED_MAX_INFLIGHT_REQUEST_COUNT
                ? static_cast<size_t>(NAVI_SCHED_MAX_INFLIGHT_REQUEST_COUNT -
                                      inflightRequests)
                : 0;
        if (selectedRequestCounts[nodeIndex] >= remainingCapacity) {
            _selectedNodeIndexes[requestIndex] = INVALID_NODE_INDEX;
            _decisions[requestIndex].reason = "rate_limited";
            continue;
        }
        ++selectedRequestCounts[nodeIndex];
        auto &batchState = nodeBatchStates[nodeIndex];
        ++batchState.requestCount;
        batchState.requestIndexes.emplace_back(requestIndex);
        if (1 == batchState.requestCount) {
            autil::HashUtil::combineHash(batchState.batchHash, _scheduleTimeUs);
            autil::HashUtil::combineHash(batchState.batchHash, nodeIndex);
            autil::HashUtil::combineHash(batchState.batchHash,
                                         _requests[requestIndex].request.requestId);
            continue;
        }
        autil::HashUtil::combineHash(batchState.batchHash,
                                     _requests[requestIndex].request.requestId);
    }
    for (size_t nodeIndex = 0; nodeIndex < nodeSize; ++nodeIndex) {
        auto &batchState = nodeBatchStates[nodeIndex];
        if (batchState.requestIndexes.empty()) {
            continue;
        }
        const auto cacheCounts = getCacheHitBlockCountsSnapshot(nodeIndex);
        if (!cacheCounts) {
            return preparationFailure("cache_inputs");
        }
        if (_scheduleNodes[nodeIndex].classifiesNewRequestKVLoad())
            batchState.cacheCapture = std::atomic_load(&_scheduleNodes[nodeIndex]._executionCacheInputs);
        for (const auto requestIndex : batchState.requestIndexes) {
            matchedBlockCounts[requestIndex] = getCacheHitBlockCount(*cacheCounts, requestIndex);
        }
        size_t hitMembers = 0, lookupMembers = 0;
        for (const auto requestIndex : batchState.requestIndexes) {
            hitMembers += getCacheHitBlockCount(*cacheCounts, requestIndex) > 0;
            lookupMembers += _scheduleNodes[nodeIndex].hasQueueLookupOpportunity(
                _requests[requestIndex].request.promptTokenCount);
        }
        batchState.predictedQueueLatencyMs =
            _scheduleNodes[nodeIndex].getEstimatedQueueWaitMsAt(
                _queuePredictionTimeUs, hitMembers, batchState.requestIndexes.size(),
                lookupMembers);
        auto latencyPrediction = std::make_shared<NaviSchedLatencyPrediction>();
        batchState.predictedExecutionLatencyMs = estimateBatchExecutionLatencyMs(
            nodeIndex, batchState.requestIndexes, latencyPrediction.get(),
            &matchedBlockCounts, _queuePredictionTimeUs);
        if ((NonLinearPrefillModelR::usesGeometryForVersion(
                 _scheduleNodes[nodeIndex].getLatencyBinaryVersion()) ||
             NonLinearPrefillModelR::usesNativePP4ForVersion(
                 _scheduleNodes[nodeIndex].getLatencyBinaryVersion())) &&
            !std::isfinite(batchState.predictedExecutionLatencyMs)) {
            return preparationFailure("nonfinite_execution_prediction");
        }
        latencyPrediction->fixedLatencyMs += batchState.predictedQueueLatencyMs;
        if (latencyPrediction->valid()) {
            batchState.latencyPrediction = std::move(latencyPrediction);
        }
    }
    for (size_t requestIndex = 0; requestIndex < requestSize; ++requestIndex) {
        auto &decision = _decisions[requestIndex];
        decision.requestId = _requests[requestIndex].request.requestId;
        decision.scheduleId = _scheduleId;
        decision.scheduleLoss = _scheduleLoss;
        decision.scheduleLatencyUs = _scheduleLatencyUs;
        decision.scheduleTimeUs = _scheduleTimeUs;
        auto nodeIndex = _selectedNodeIndexes[requestIndex];
        if (INVALID_NODE_INDEX == nodeIndex) {
            if (decision.reason.empty()) {
                decision.reason = "not_scheduled";
            }
            continue;
        }
        const auto &node = _scheduleNodes[nodeIndex]._info;
        if (!node) {
            return preparationFailure("missing_node_info");
        }
        decision.target.bizName = node->bizName;
        decision.target.address = node->serverAddress;
        decision.target.agentAddress = node->collectorAddress;
        auto &batchState = nodeBatchStates[nodeIndex];
        if (0 == batchState.batchId) {
            batchState.batchId = makeBatchId(batchState.batchHash);
        }
        decision.batchId = batchState.batchId;
        decision.expectedBatchSize = batchState.requestCount;
        decision.predictedCacheHitTokenCount =
            matchedBlockCounts[requestIndex] * _requests[requestIndex].request.blockSize;
        decision.executionLoadIncluded.reset();
        if (_scheduleNodes[nodeIndex].classifiesNewRequestKVLoad()) {
            if (decision.predictedCacheHitTokenCount == 0) {
                decision.executionLoadIncluded = true;
            } else {
                const auto &capture = batchState.cacheCapture;
                const auto storage = resolveRequestIndex(requestIndex);
                if (!capture || !capture->observation.readyOnlyHbmBlockCounts ||
                    storage >= capture->observation.readyOnlyHbmBlockCounts->size() ||
                    capture->observation.readyOnlyHbmBlockCounts->at(storage) < 0)
                    return preparationFailure("missing_hbm_load_classification");
                const auto hbmBlocks = capture->observation.readyOnlyHbmBlockCounts->at(storage);
                // Compare blocks to avoid overflow, retaining the same frozen
                // physical HBM observation that produced the selected hit.
                decision.executionLoadIncluded =
                    matchedBlockCounts[requestIndex] <= hbmBlocks;
            }
        }
        decision.predictedQueueLatencyMs = batchState.predictedQueueLatencyMs;
        decision.predictedExecutionLatencyMs = batchState.predictedExecutionLatencyMs;
        decision.latencyPrediction = batchState.latencyPrediction;
    }
    _resultPrepared = true;
    return true;
}

int64_t NaviSchedScheduleContext::makeBatchId(uint64_t hash)
{
    auto batchId = static_cast<int64_t>(
        hash & static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    return 0 == batchId ? 1 : batchId;
}

bool NaviSchedScheduleContext::finish()
{
    clearScheduleView();
    _responseDeliveryFailedIndexes.clear();
    if (_responseRequests.size() != requestCount()) {
        for (size_t index = 0; index < requestCount(); ++index) {
            _responseDeliveryFailedIndexes.emplace_back(index);
        }
        return false;
    }
    if (!prepareResult(_scheduleTimeUs)) {
        for (size_t index = 0; index < requestCount(); ++index) {
            _responseDeliveryFailedIndexes.emplace_back(index);
        }
        return false;
    }
    for (auto &response : _deferredResponses) {
        response.decision.scheduleLoss = _scheduleLoss;
        response.decision.scheduleLatencyUs = _scheduleLatencyUs;
    }
    if (_replay) {
        _replay->decisions = _decisions;
        for (const auto &response : _deferredResponses) {
            _replay->decisions.emplace_back(response.decision);
        }
        for (auto &decision : _replay->decisions) {
            decision.target.bizName.clear();
        }
    }
    bool success = true;
    if (!_replay) {
        for (size_t index = 0; index < requestCount(); ++index) {
            auto response = makeResponse(_decisions[index], index);
            if (!_responseRequests[index] || !_responseRequests[index]->valid() ||
                !_responseRequests[index]->finish(response)) {
                _responseDeliveryFailedIndexes.emplace_back(index);
                success = false;
            }
        }
        for (const auto &deferred : _deferredResponses) {
            success = deferred.request && deferred.request->valid() &&
                      deferred.request->finish(makeResponse(deferred.decision)) &&
                      success;
        }
        return success;
    }
    // Finish non-debug requests before returning the larger replay value.
    for (size_t index = 0; index < requestCount(); ++index) {
        const auto &responseRequest = _responseRequests[index];
        if (!responseRequest || !responseRequest->valid()) {
            _responseDeliveryFailedIndexes.emplace_back(index);
            success = false;
            continue;
        }
        if (responseRequest->request->collectReplay) {
            continue;
        }
        auto response = makeResponse(_decisions[index], index);
        if (!responseRequest->finish(response)) {
            _responseDeliveryFailedIndexes.emplace_back(index);
            success = false;
        }
    }
    for (size_t index = 0; index < requestCount(); ++index) {
        const auto &responseRequest = _responseRequests[index];
        if (!responseRequest || !responseRequest->valid() ||
            !responseRequest->request->collectReplay) {
            continue;
        }
        auto response = makeResponse(_decisions[index], index);
        response->replay = _replay;
        if (!responseRequest->finish(response)) {
            _responseDeliveryFailedIndexes.emplace_back(index);
            success = false;
        }
    }
    for (const auto &deferred : _deferredResponses) {
        auto response = makeResponse(deferred.decision);
        if (deferred.request && deferred.request->valid() &&
            deferred.request->request->collectReplay) {
            response->replay = _replay;
        }
        success = deferred.request && deferred.request->valid() &&
                  deferred.request->finish(response) && success;
    }
    return success;
}

void NaviSchedScheduleContext::fail(const std::string &reason)
{
    clearScheduleView();
    _decisions.assign(requestCount(), NaviSchedDecision());
    _selectedNodeIndexes.assign(requestCount(), INVALID_NODE_INDEX);
    for (auto &decision : _decisions) {
        decision.reason = reason;
    }
}

void NaviSchedScheduleContext::attachLatencyModelGlobals(
    const MasterLatencyModelR &latencyModels)
{
    _latencyModelGlobals.clear();
    for (const auto nodeIndex : _selectedNodeIndexes) {
        if (nodeIndex >= _scheduleNodes.size()) {
            continue;
        }
        const auto &node = _scheduleNodes[nodeIndex];
        if (!node._info || !node._latencyModel || node._info->collectorAddress.empty()) {
            continue;
        }
        auto global = latencyModels.getGlobalZeroPoint(
            node._latencyModel->formula, node._latencyModel->binaryVersion,
            node._latencyModel->parameters.size());
        if (global.valid()) {
            _latencyModelGlobals[node._info->collectorAddress] = global;
        }
    }
}

MasterResponseToClientDataPtr NaviSchedScheduleContext::makeResponse(
    const NaviSchedDecision &decision, size_t requestIndex) const
{
    auto response = std::make_shared<MasterResponseToClientData>();
    response->version = decision.version;
    response->requestId = decision.requestId;
    response->scheduleId = decision.scheduleId;
    response->scheduleLoss = decision.scheduleLoss;
    response->scheduleLatencyUs = decision.scheduleLatencyUs;
    response->scheduleTimeUs = decision.scheduleTimeUs;
    response->reason = decision.reason.empty() ? "not_scheduled" : decision.reason;
    response->batchId = decision.batchId;
    response->expectedBatchSize = decision.expectedBatchSize;
    response->executionLoadIncluded = decision.executionLoadIncluded;
    response->predictedCacheHitTokenCount = decision.predictedCacheHitTokenCount;
    response->predictedQueueLatencyMs = decision.predictedQueueLatencyMs;
    response->predictedExecutionLatencyMs = decision.predictedExecutionLatencyMs;
    response->latencyPrediction = decision.latencyPrediction;
    response->latencyTrainingPhase = decision.latencyTrainingPhase;
    response->target = decision.target;
    response->cacheTouchTargets =
        getCacheTouchTargets(requestIndex, decision.predictedCacheHitTokenCount);
    if (requestIndex < _dispatchHandles.size()) {
        response->dispatchHandle = _dispatchHandles[requestIndex];
    }
    auto global = _latencyModelGlobals.find(decision.target.agentAddress);
    if (global != _latencyModelGlobals.end()) {
        response->latencyModelGlobal = global->second;
    }
    return response;
}

} // namespace navi_sched
