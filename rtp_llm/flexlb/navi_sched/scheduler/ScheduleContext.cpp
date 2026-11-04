#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContextInternal.h"

#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterScheduleRequestData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterLatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "autil/HashUtil.h"

#include <cmath>

namespace navi_sched {

int64_t NaviSchedScheduleContext::getScheduleTimeUs() const
{
    return _scheduleTimeUs;
}

NaviSchedScheduleNode::NaviSchedScheduleNode(NaviSchedNodeConstPtr node,
                                             NaviSchedNodeSnapshot snapshot,
                                             int64_t scheduleTimeUs,
                                             double timeCompression)
    : _node(std::move(node))
    , _info(std::move(snapshot.info))
    , _latencyModel(std::move(snapshot.latencyModel))
    , _residualRuntimeConfig(snapshot.residualRuntimeConfig)
    , _totalBlocks(snapshot.load.totalBlocks)
    , _inflightRequests(snapshot.load.inflightRequests)
    , _inflightPredictedLatencySumMs(
          snapshot.load.inflightPredictedLatencySumMs)
    , _batchWaitingRequests(snapshot.load.batchWaitingRequests)
    , _queuedBatchRequests(snapshot.load.queuedBatchRequests)
    , _runningRequests(snapshot.load.runningRequests)
    , _batchTimeoutCount(snapshot.load.batchTimeoutCount)
    , _abnormalBatchFinishCount(snapshot.load.abnormalBatchFinishCount)
    , _queueTotalPredictedLatencyMs(snapshot.load.queueTotalPredictedLatencyMs)
    , _queueDecayHeadRemainingMs(snapshot.load.queueDecayHeadRemainingMs)
    , _queueReportReceiveTimeUs(snapshot.load.queueReportReceiveTimeUs)
    , _scheduleTimeUs(scheduleTimeUs)
    , _timeCompression(timeCompression)
    , _unifiedLatencyPrediction(snapshot.unifiedLatencyPrediction)
    , _workload(std::move(snapshot.workload))
    , _workloadFeatures(std::move(snapshot.workloadFeatures))
    , _reportedWorkloadFeatures(std::move(snapshot.reportedWorkloadFeatures))
    , _reservedWorkloadFeatures(std::move(snapshot.reservedWorkloadFeatures))
    , _maxRunningRequestCount(snapshot.maxRunningRequestCount)
    , _workloadRunningCapacitySlotCount(
          snapshot.workloadRunningCapacitySlotCount)
    , _reservedRunningCapacitySlotCount(
          snapshot.reservedRunningCapacitySlotCount)
    , _workloadReportReceiveTimeUs(snapshot.workloadReportReceiveTimeUs)
    , _agentStartTimeUs(snapshot.agentStartTimeUs)
{
    if (_unifiedLatencyPrediction && _workloadReportReceiveTimeUs > 0) {
        const auto workloadReportAgeMs = getWorkloadReportAgeMs();
        const bool supportsRunningElapsed =
            _latencyModel && NonLinearPrefillModelR::usesCapacitySplitParameters(
                                 _latencyModel->parameters);
        if (supportsRunningElapsed &&
            _workload.runningCapacityPrefillRequestCount > 0) {
            const auto runningElapsedDeltaMs =
                workloadReportAgeMs *
                _workload.runningCapacityPrefillRequestCount;
            auto reportedFeatures = _reportedWorkloadFeatures;
            auto workloadFeatures = _workloadFeatures;
            if (reportedFeatures.appendRunningElapsedTimeMs(
                    runningElapsedDeltaMs) &&
                workloadFeatures.appendRunningElapsedTimeMs(
                    runningElapsedDeltaMs)) {
                _reportedWorkloadFeatures = std::move(reportedFeatures);
                _workloadFeatures = std::move(workloadFeatures);
                _runningElapsedProgressApplied = true;
            }
        }
    }
}

void NaviSchedScheduleNode::refreshExistingUnifiedLatency()
{
    _existingUnifiedLatencyMs = 0.0;
    NaviSchedLatencyFeatures floorFeatures;
    if (usesUnifiedLatencyPrediction() && hasLatencyModel() &&
        getUnifiedDecayFloorFeatures(floorFeatures)) {
        _existingUnifiedLatencyMs = decayUnifiedLatency(
            predictLatency(_workloadFeatures), predictLatency(floorFeatures));
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

double NaviSchedScheduleNode::getEstimatedQueueWaitMs() const
{
    return getEstimatedQueueWaitMsAt(_scheduleTimeUs);
}

double NaviSchedScheduleNode::getEstimatedQueueWaitMsAt(int64_t predictionTimeUs) const
{
    if (usesUnifiedLatencyPrediction()) {
        return 0.0;
    }
    const auto elapsedMs =
        _queueReportReceiveTimeUs > 0
            ? std::max<int64_t>(0, predictionTimeUs - _queueReportReceiveTimeUs) /
                  1000.0
            : 0.0;
    const auto decayMs =
        std::min(_queueDecayHeadRemainingMs, elapsedMs * _timeCompression);
    return std::max(0.0, _queueTotalPredictedLatencyMs - decayMs);
}

double NaviSchedScheduleNode::getEstimatedBacklogLatencyMs() const
{
    if (!usesUnifiedLatencyPrediction()) {
        return getEstimatedQueueWaitMs();
    }
    // Accepted requests own workload before their first batch becomes ready.
    if (!hasLatencyModel() ||
        (_workloadFeatures.getBatchCount() == 0 &&
         _workloadFeatures.getBatchSize() == 0 &&
         _workloadFeatures.getPrecedingRequestCount() == 0)) {
        return 0.0;
    }
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
    return std::max(
        0.0, decayUnifiedLatency(latencyWithReportedWorkloadMs,
                                 latencyWithoutReportedWorkloadMs));
}

bool NaviSchedScheduleNode::hasLatencyModel() const { return bool(_latencyModel); }

bool NaviSchedScheduleNode::hasValidLatencyModel() const
{
    return _latencyModel &&
           NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL ==
               _latencyModel->formula &&
           NonLinearPrefillModelR::BINARY_VERSION ==
               _latencyModel->binaryVersion &&
           NonLinearPrefillModelR::validateParameters(
               _latencyModel->parameters) &&
           LatencyResidualModel::VERSION == _latencyModel->residualVersion &&
           LatencyResidualModel::validateParameters(
               _latencyModel->residualParameters) &&
           _latencyModel->residualLastUpdateTimeUs >= 0;
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
    if (!hasValidLatencyModel() ||
        !_latencyModel->overloadPredictionTrusted) {
        return false;
    }
    if (!NonLinearPrefillModelR::usesCapacitySplitParameters(
            _latencyModel->parameters) ||
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
    return _latencyModel ? _latencyModel->parameters : EMPTY_PARAMETERS;
}

double NaviSchedScheduleNode::applyLatencyResidual(
    double basePredictionMs, double *derivativeFactor) const
{
    if (derivativeFactor) {
        *derivativeFactor = 1.0;
    }
    if (!_latencyModel) {
        return basePredictionMs;
    }
    const auto result = LatencyResidualModel::evaluate(
        basePredictionMs, _latencyModel->residualParameters,
        _latencyModel->residualLastUpdateTimeUs, _scheduleTimeUs,
        _residualRuntimeConfig);
    if (derivativeFactor) {
        *derivativeFactor = result.derivativeFactor;
    }
    return result.predictionMs;
}

double NaviSchedScheduleNode::predictLatency(
    const NaviSchedLatencyFeatures &features) const
{
    return applyLatencyResidual(
        NonLinearPrefillModelR::predict(features, getLatencyParameters()));
}

bool NaviSchedScheduleNode::usesUnifiedLatencyPrediction() const
{
    return _unifiedLatencyPrediction;
}

int64_t NaviSchedScheduleNode::getAgentStartTimeUs() const
{
    return _agentStartTimeUs;
}

const NaviSchedLatencyFeatures &NaviSchedScheduleNode::getWorkloadFeatures() const
{
    return _workloadFeatures;
}

const NaviSchedLatencyFeatures &
NaviSchedScheduleNode::getReportedWorkloadFeatures() const
{
    return _reportedWorkloadFeatures;
}

const NaviSchedLatencyFeatures &
NaviSchedScheduleNode::getReservedWorkloadFeatures() const
{
    return _reservedWorkloadFeatures;
}

bool NaviSchedScheduleNode::getUnifiedDecayFloorFeatures(
    NaviSchedLatencyFeatures &features) const
{
    auto floorFeatures = _reservedWorkloadFeatures;
    if (NonLinearPrefillModelR::usesCapacitySplitParameters(
            getLatencyParameters()) &&
        !floorFeatures.appendPrecedingRequestCount(
            _reportedWorkloadFeatures.getPrecedingRequestCount())) {
        return false;
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

uint64_t NaviSchedScheduleNode::getWorkloadRevision() const
{
    return _workload.revision;
}

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
    return std::max<int64_t>(0, _scheduleTimeUs - _workloadReportReceiveTimeUs) /
           1000.0 * _timeCompression;
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

double NaviSchedScheduleNode::decayUnifiedLatency(
    double latencyWithReportedWorkloadMs,
    double latencyWithoutReportedWorkloadMs,
    bool *usesReportedWorkload) const
{
    if (usesReportedWorkload) {
        *usesReportedWorkload = true;
    }
    if (!usesUnifiedLatencyPrediction() || _workloadReportReceiveTimeUs <= 0 ||
        _reportedWorkloadFeatures.getBatchCount() == 0) {
        return latencyWithReportedWorkloadMs;
    }
    // An elapsed-aware model has already advanced running prefill progress.
    // Applying wall-clock decay as well would count the same progress twice.
    const auto decayedLatencyMs = _runningElapsedProgressApplied
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
    _decisions.assign(requestCount(), NaviSchedDecision());
    _selectedNodeIndexes.assign(requestCount(), INVALID_NODE_INDEX);
    _prepared = true;
    return true;
}

size_t NaviSchedScheduleContext::requestCount() const
{
    return _scheduleViewActive ? _scheduleViewRequestIndexes.size()
                               : _requests.size();
}

size_t NaviSchedScheduleContext::nodeCount() const
{
    return _scheduleViewActive ? _scheduleViewNodeIndexes.size()
                               : _scheduleNodes.size();
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
    return _scheduleNodes[storageNodeIndex]._node->getMatchedBlockCountForRequest(
        _requests[storageRequestIndex].request);
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
    if (!_scheduleViewActive || _scheduleViewUsesAllRequests) {
        _scheduleNodes[storageNodeIndex]._node->getMatchedBlockCounts(
            _requests, matchedBlockCounts.data(), _cacheMatchCommonPrefixes);
        return true;
    }
    _scheduleNodes[storageNodeIndex]._node->getMatchedBlockCounts(
        _scheduleViewRequests, matchedBlockCounts.data(), _cacheMatchCommonPrefixes);
    return true;
}

bool NaviSchedScheduleContext::isCollectingReplay() const
{
    return bool(_replay);
}

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

void NaviSchedScheduleContext::recordDiscreteSearchLosses(
    double initialLoss, double finalLoss) const
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
    return std::isfinite(improvement) ? std::optional<double>(improvement)
                                     : std::nullopt;
}

bool NaviSchedScheduleContext::setScheduleLoss(double loss)
{
    if (!std::isfinite(loss)) {
        return false;
    }
    _scheduleLoss = loss;
    return true;
}

double NaviSchedScheduleContext::estimateBatchExecutionLatencyMs(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    NaviSchedLatencyPrediction *latencyPrediction) const
{
    return estimateBatchExecutionLatencyMs(
        nodeIndex, requestIndexes, latencyPrediction, nullptr);
}

double NaviSchedScheduleContext::estimateBatchExecutionLatencyMs(
    size_t nodeIndex, const std::vector<size_t> &requestIndexes,
    NaviSchedLatencyPrediction *latencyPrediction,
    const std::vector<int64_t> *matchedBlockCounts) const
{
    if (latencyPrediction) {
        *latencyPrediction = NaviSchedLatencyPrediction();
    }
    if (nodeIndex >= nodeCount() || requestIndexes.empty()) {
        return 0.0;
    }
    const auto storageNodeIndex = resolveNodeIndex(nodeIndex);
    const auto &node = _scheduleNodes[storageNodeIndex];
    if (!node.hasValidLatencyModel()) {
        return 0.0;
    }
    const auto capacitySplit = NonLinearPrefillModelR::usesCapacitySplitParameters(
        node.getLatencyParameters());
    NaviSchedLatencyFeatures requestFeatures;
    NaviSchedLatencyFeatures requestFeaturesWithoutReportedWorkload;
    auto workloadSlots = node._workloadRunningCapacitySlotCount;
    auto reservedSlots = node._reservedRunningCapacitySlotCount;
    if (capacitySplit) {
        if (!requestFeatures.appendCapacitySplitAggregate(
                1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0) ||
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
            requestFeatures.appendWaitingRequest(matchedTokenCount,
                                                 computeTokenCount);
        }
        if (reservedSlots < node._maxRunningRequestCount) {
            requestFeaturesWithoutReportedWorkload.appendRequest(
                matchedTokenCount, computeTokenCount);
            ++reservedSlots;
        } else {
            requestFeaturesWithoutReportedWorkload.appendWaitingRequest(
                matchedTokenCount, computeTokenCount);
        }
    }
    if (!node.usesUnifiedLatencyPrediction()) {
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
    const auto withReportedPrediction = node.predictLatency(withReportedWorkload);
    const auto withoutReportedPrediction =
        node.predictLatency(withoutReportedWorkload);
    const auto prediction = node.decayUnifiedLatency(
        withReportedPrediction, withoutReportedPrediction);
    if (latencyPrediction) {
        latencyPrediction->features = std::move(withReportedWorkload);
        latencyPrediction->floorFeatures = std::move(withoutReportedWorkload);
        latencyPrediction->hasFloor =
            node._workloadReportReceiveTimeUs > 0 &&
            node._reportedWorkloadFeatures.getBatchCount() > 0;
        if (latencyPrediction->hasFloor &&
            !node._runningElapsedProgressApplied) {
            latencyPrediction->fixedLatencyMs = -node.getWorkloadReportAgeMs();
        }
        latencyPrediction->recomputable = true;
    }
    return prediction;
}

bool NaviSchedScheduleContext::selectNode(size_t requestIndex, size_t nodeIndex,
                                          const std::string &reason,
                                          NaviSchedLatencyTrainingPhase
                                              latencyTrainingPhase)
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
    const std::vector<size_t> &requestIndexes,
    const std::vector<size_t> &nodeIndexes)
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
    for (size_t index = 0; _scheduleViewUsesAllRequests && index < requestIndexes.size(); ++index) {
        _scheduleViewUsesAllRequests = requestIndexes[index] == index;
    }
    if (!_scheduleViewUsesAllRequests) {
        _scheduleViewRequests.reserve(requestIndexes.size());
        for (const auto requestIndex : requestIndexes) {
            _scheduleViewRequests.emplace_back(_requests[requestIndex]);
        }
    }
    _scheduleViewActive = true;
    _cacheMatchCommonPrefixes.clear();
    return true;
}

void NaviSchedScheduleContext::clearScheduleView()
{
    _cacheMatchCommonPrefixes.clear();
    _scheduleViewRequestIndexes.clear();
    _scheduleViewNodeIndexes.clear();
    _scheduleViewRequests.clear();
    _scheduleViewUsesAllRequests = false;
    _scheduleViewActive = false;
}

size_t NaviSchedScheduleContext::resolveRequestIndex(size_t requestIndex) const
{
    return _scheduleViewActive ? _scheduleViewRequestIndexes[requestIndex]
                               : requestIndex;
}

size_t NaviSchedScheduleContext::resolveNodeIndex(size_t nodeIndex) const
{
    return _scheduleViewActive ? _scheduleViewNodeIndexes[nodeIndex]
                               : nodeIndex;
}

void NaviSchedScheduleContext::setReplay(
    const std::shared_ptr<NaviSchedScheduleReplay> &replay)
{
    _replay = replay;
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
                 .emplace(replay->requests[index].request.requestId, index).second) {
            _replayMatchRequestIndexes.clear();
            return;
        }
    }
}

bool NaviSchedScheduleContext::getReplayMatch(size_t storageRequestIndex,
                                             size_t storageNodeIndex,
                                             int64_t &matchedBlockCount) const
{
    const auto it = _replayMatchRequestIndexes.find(
        _requests[storageRequestIndex].request.requestId);
    if (it == _replayMatchRequestIndexes.end()) {
        return false;
    }
    // Queue-overload filtering compacts _requests after replay capture.
    // Retain the original row by request ID so costs use the recorded snapshot.
    matchedBlockCount = _replay->nodeMatches[it->second][storageNodeIndex]
                            .matchedBlockCount;
    return true;
}

bool NaviSchedScheduleContext::prepareResult(int64_t queuePredictionTimeUs)
{
    // The Master reserves this round after the first result preparation.  Reusing
    // that result prevents the response from matching its own reservation.
    if (_resultPrepared) {
        return true;
    }
    _queuePredictionTimeUs = std::max(_scheduleTimeUs, queuePredictionTimeUs);
    const auto requestSize = requestCount();
    const auto nodeSize = nodeCount();
    if (_decisions.size() != requestSize || _selectedNodeIndexes.size() != requestSize) {
        return false;
    }
    struct NodeBatchState {
        size_t requestCount = 0;
        uint64_t batchHash = 0;
        int64_t batchId = 0;
        std::vector<size_t> requestIndexes;
        double predictedQueueLatencyMs = 0.0;
        double predictedExecutionLatencyMs = 0.0;
        NaviSchedLatencyPredictionConstPtr latencyPrediction;
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
            return false;
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
        // Use one cache observation for both execution prediction and response.
        for (auto requestIndex : batchState.requestIndexes) {
            matchedBlockCounts[requestIndex] = getCacheHitBlockCount(requestIndex, nodeIndex);
        }
        batchState.predictedQueueLatencyMs =
            _scheduleNodes[nodeIndex].getEstimatedQueueWaitMsAt(_queuePredictionTimeUs);
        auto latencyPrediction = std::make_shared<NaviSchedLatencyPrediction>();
        batchState.predictedExecutionLatencyMs = estimateBatchExecutionLatencyMs(
            nodeIndex, batchState.requestIndexes, latencyPrediction.get(), &matchedBlockCounts);
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
            return false;
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
            matchedBlockCounts[requestIndex] *
            _requests[requestIndex].request.blockSize;
        decision.predictedQueueLatencyMs = batchState.predictedQueueLatencyMs;
        decision.predictedExecutionLatencyMs =
            batchState.predictedExecutionLatencyMs;
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
        if (!node._info || !node._latencyModel ||
            node._info->collectorAddress.empty()) {
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
    response->predictedCacheHitTokenCount =
        decision.predictedCacheHitTokenCount;
    response->predictedQueueLatencyMs = decision.predictedQueueLatencyMs;
    response->predictedExecutionLatencyMs =
        decision.predictedExecutionLatencyMs;
    response->latencyPrediction = decision.latencyPrediction;
    response->latencyTrainingPhase = decision.latencyTrainingPhase;
    response->target = decision.target;
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
