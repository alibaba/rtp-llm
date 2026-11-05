#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include <autil/DataBuffer.h>
#include <autil/legacy/fast_jsonizable.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace navi_sched {

int64_t getNaviSchedRpcTimeoutMs(int64_t entryTimeoutMs)
{
    if (entryTimeoutMs <= 1) {
        return entryTimeoutMs;
    }
    constexpr int64_t MAX_ERROR_RESERVE_MS = 30;
    const auto reserveMs =
        std::min(MAX_ERROR_RESERVE_MS, std::max(int64_t(1), entryTimeoutMs / 10));
    return entryTimeoutMs - reserveMs;
}

bool NaviSchedBatchRequest::valid() const
{
    return requestId != 0 && predictedCacheHitTokenCount >= 0 &&
           (cachePublicationTokenCount == -1 ||
            cachePublicationTokenCount >= totalTokenCount) &&
           predictedCacheHitTokenCount <= totalTokenCount &&
           std::isfinite(predictedQueueLatencyMs) && predictedQueueLatencyMs >= 0.0 &&
           std::isfinite(predictedExecutionLatencyMs) &&
           predictedExecutionLatencyMs >= 0.0 &&
           validLatencyTrainingPhase(latencyTrainingPhase) &&
           (!latencyPrediction || latencyPrediction->valid()) &&
           BlockHashUtil::validTokenBlocks(totalTokenCount, blockSize, blockHashes);
}

bool NaviSchedLatencyPrediction::valid() const
{
    return recomputable && std::isfinite(fixedLatencyMs);
}

void NaviSchedLatencyPrediction::serialize(autil::DataBuffer &dataBuffer) const
{
    features.serializeWithExecutionCacheSource(dataBuffer);
    floorFeatures.serializeWithExecutionCacheSource(dataBuffer);
    dataBuffer.write(fixedLatencyMs);
    dataBuffer.write(hasFloor);
    dataBuffer.write(recomputable);
}

void NaviSchedLatencyPrediction::deserialize(autil::DataBuffer &dataBuffer)
{
    if (!features.deserializeWithExecutionCacheSource(dataBuffer) ||
        !floorFeatures.deserializeWithExecutionCacheSource(dataBuffer)) {
        AUTIL_LEGACY_THROW(autil::DataBufferCorruptedException,
                           "invalid latency prediction execution features");
    }
    dataBuffer.read(fixedLatencyMs);
    dataBuffer.read(hasFloor);
    dataBuffer.read(recomputable);
}

bool NaviSchedRequestStats::valid() const
{
    return requestId != 0 && totalTokenCount >= 0 && cacheHitTokenCount >= 0 &&
           executionExternalReadTokens >= -1 &&
           executionExternalReadTokens <= cacheHitTokenCount &&
           cacheHitTokenCount <= totalTokenCount && predictedCacheHitTokenCount >= 0 &&
           predictedCacheHitTokenCount <= totalTokenCount &&
           std::isfinite(predictedQueueLatencyMs) && predictedQueueLatencyMs >= 0.0 &&
           std::isfinite(predictedExecutionLatencyMs) &&
           predictedExecutionLatencyMs >= 0.0 &&
           validLatencyTrainingPhase(latencyTrainingPhase);
}

NaviSchedLocalBatchInfo NaviSchedBatch::makeLocalBatchInfo() const
{
    NaviSchedLocalBatchInfo info;
    if (scheduleBatchId <= 0 || batchId <= 0 || requestIds.empty() ||
        requestIds.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return info;
    }
    info.scheduleBatchId = scheduleBatchId;
    info.localBatchId = batchId;
    info.requestCount = static_cast<int32_t>(requestIds.size());
    info.pendingCacheTracked = pendingCacheTracked;
    if (static_cast<int64_t>(requestIds.size()) != expectedBatchSize) {
        info.requestIds = requestIds;
    }
    return info;
}

bool NaviSchedLocalBatchInfo::empty() const
{
    return scheduleBatchId == 0 && localBatchId == 0 && requestCount == 0 &&
           !pendingCacheTracked && requestIds.empty();
}

bool NaviSchedLocalBatchInfo::valid() const
{
    if (scheduleBatchId <= 0 || localBatchId <= 0 || requestCount <= 0) {
        return false;
    }
    if (requestIds.empty()) {
        return true;
    }
    if (requestIds.size() != static_cast<size_t>(requestCount)) {
        return false;
    }
    std::unordered_set<uint64_t> uniqueRequestIds;
    uniqueRequestIds.reserve(requestIds.size());
    for (auto requestId : requestIds) {
        if (requestId == 0 || !uniqueRequestIds.emplace(requestId).second) {
            return false;
        }
    }
    return true;
}

void NaviSchedLocalBatchInfo::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(scheduleBatchId);
    dataBuffer.write(localBatchId);
    dataBuffer.write(requestCount);
    dataBuffer.write(pendingCacheTracked);
    dataBuffer.write(requestIds);
}

void NaviSchedLocalBatchInfo::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(scheduleBatchId);
    dataBuffer.read(localBatchId);
    dataBuffer.read(requestCount);
    dataBuffer.read(pendingCacheTracked);
    dataBuffer.read(requestIds);
}

bool NaviSchedWorkloadSnapshot::valid() const
{
    const bool capacitySplitValid =
        prefillCapacitySplit
            ? (maxRunningRequestCount >= 0 && runningCapacitySlotCount >= 0 &&
               runningCapacitySlotCount <= maxRunningRequestCount &&
               runningCapacityPrefillRequestCount >= 0 &&
               runningCapacityPrefillRequestCount <= runningCapacitySlotCount &&
               waitingCapacityPrefillRequestCount >= 0 &&
               (waitingCapacityPrefillRequestCount == 0 ||
                runningCapacitySlotCount == maxRunningRequestCount) &&
               runningCapacityPrefillContextTokenCount >= 0 &&
               waitingCapacityPrefillContextTokenCount >= 0 &&
               runningCapacityPrefillTokenCount >= 0 &&
               waitingCapacityPrefillTokenCount >= 0 &&
               std::isfinite(runningCapacityPrefillTokenSquareSum) &&
               runningCapacityPrefillTokenSquareSum >= 0.0 &&
               std::isfinite(waitingCapacityPrefillTokenSquareSum) &&
               waitingCapacityPrefillTokenSquareSum >= 0.0 &&
               std::isfinite(runningCapacityContextRemainingPrefillTokenProductSum) &&
               runningCapacityContextRemainingPrefillTokenProductSum >= 0.0 &&
               std::isfinite(waitingCapacityContextRemainingPrefillTokenProductSum) &&
               waitingCapacityContextRemainingPrefillTokenProductSum >= 0.0 &&
               remainingPrefillRequestCount == runningCapacityPrefillRequestCount +
                                                   waitingCapacityPrefillRequestCount &&
               remainingPrefillContextTokenCount ==
                   runningCapacityPrefillContextTokenCount +
                       waitingCapacityPrefillContextTokenCount &&
               remainingPrefillTokenCount ==
                   runningCapacityPrefillTokenCount + waitingCapacityPrefillTokenCount &&
               remainingPrefillTokenSquareSum ==
                   runningCapacityPrefillTokenSquareSum +
                       waitingCapacityPrefillTokenSquareSum &&
               contextRemainingPrefillTokenProductSum ==
                   runningCapacityContextRemainingPrefillTokenProductSum +
                       waitingCapacityContextRemainingPrefillTokenProductSum)
            : (maxRunningRequestCount == 0 && runningCapacitySlotCount == 0 &&
               runningCapacityPrefillRequestCount == 0 &&
               runningCapacityPrefillContextTokenCount == 0 &&
               runningCapacityPrefillTokenCount == 0 &&
               runningCapacityPrefillTokenSquareSum == 0.0 &&
               runningCapacityContextRemainingPrefillTokenProductSum == 0.0 &&
               waitingCapacityPrefillRequestCount == 0 &&
               waitingCapacityPrefillContextTokenCount == 0 &&
               waitingCapacityPrefillTokenCount == 0 &&
               waitingCapacityPrefillTokenSquareSum == 0.0 &&
               waitingCapacityContextRemainingPrefillTokenProductSum == 0.0);
    return revision > 0 && eventTimeUs > 0 && batchCount >= 0 &&
           batchCount <= requestCount && requestCount >= 0 &&
           remainingPrefillRequestCount >= 0 &&
           remainingPrefillRequestCount <= requestCount && runningRequestCount >= 0 &&
           std::isfinite(runningElapsedTimeMs) && runningElapsedTimeMs >= 0.0 &&
           (runningRequestCount > 0 || runningElapsedTimeMs == 0.0) &&
           (!prefillCapacitySplit
                ? (runningDecodeRequestCount == 0 &&
                   runningDecodeGeneratedTokenCount == 0 &&
                   runningDecodeRemainingTokenBudget == 0)
                : (runningDecodeRequestCount >= 0 &&
                   runningDecodeRequestCount <= runningRequestCount &&
                   runningDecodeGeneratedTokenCount >= runningDecodeRequestCount &&
                   runningDecodeRemainingTokenBudget >= 0 &&
                   (runningDecodeRequestCount > 0 ||
                    (runningDecodeGeneratedTokenCount == 0 &&
                     runningDecodeRemainingTokenBudget == 0)))) &&
           waitingRequestCount >= 0 && remainingPrefillContextTokenCount >= 0 &&
           remainingPrefillTokenCount >= 0 &&
           std::isfinite(remainingPrefillTokenSquareSum) &&
           remainingPrefillTokenSquareSum >= 0.0 &&
           std::isfinite(contextRemainingPrefillTokenProductSum) &&
           contextRemainingPrefillTokenProductSum >= 0.0 && capacitySplitValid &&
           remainingTotalContextTokenCount >= remainingPrefillContextTokenCount &&
           remainingTotalTokenCount >= remainingPrefillTokenCount &&
           std::isfinite(remainingTotalTokenSquareSum) &&
           remainingTotalTokenSquareSum >= remainingPrefillTokenSquareSum &&
           std::isfinite(contextRemainingTotalTokenProductSum) &&
           contextRemainingTotalTokenProductSum >=
               contextRemainingPrefillTokenProductSum &&
           totalPromptTokenCount >= 0 && inFlightScheduledTokenCount >= 0 &&
           (admittedBatch.empty() || admittedBatch.valid());
}

bool NaviSchedWorkloadSnapshot::sameWorkload(const NaviSchedWorkloadSnapshot &other) const
{
    return batchCount == other.batchCount && requestCount == other.requestCount &&
           remainingPrefillRequestCount == other.remainingPrefillRequestCount &&
           runningRequestCount == other.runningRequestCount &&
           runningElapsedTimeMs == other.runningElapsedTimeMs &&
           runningDecodeRequestCount == other.runningDecodeRequestCount &&
           runningDecodeGeneratedTokenCount == other.runningDecodeGeneratedTokenCount &&
           runningDecodeRemainingTokenBudget == other.runningDecodeRemainingTokenBudget &&
           waitingRequestCount == other.waitingRequestCount &&
           remainingPrefillContextTokenCount == other.remainingPrefillContextTokenCount &&
           remainingPrefillTokenCount == other.remainingPrefillTokenCount &&
           remainingPrefillTokenSquareSum == other.remainingPrefillTokenSquareSum &&
           contextRemainingPrefillTokenProductSum ==
               other.contextRemainingPrefillTokenProductSum &&
           prefillCapacitySplit == other.prefillCapacitySplit &&
           maxRunningRequestCount == other.maxRunningRequestCount &&
           runningCapacitySlotCount == other.runningCapacitySlotCount &&
           runningCapacityPrefillRequestCount ==
               other.runningCapacityPrefillRequestCount &&
           runningCapacityPrefillContextTokenCount ==
               other.runningCapacityPrefillContextTokenCount &&
           runningCapacityPrefillTokenCount == other.runningCapacityPrefillTokenCount &&
           runningCapacityPrefillTokenSquareSum ==
               other.runningCapacityPrefillTokenSquareSum &&
           runningCapacityContextRemainingPrefillTokenProductSum ==
               other.runningCapacityContextRemainingPrefillTokenProductSum &&
           waitingCapacityPrefillRequestCount ==
               other.waitingCapacityPrefillRequestCount &&
           waitingCapacityPrefillContextTokenCount ==
               other.waitingCapacityPrefillContextTokenCount &&
           waitingCapacityPrefillTokenCount == other.waitingCapacityPrefillTokenCount &&
           waitingCapacityPrefillTokenSquareSum ==
               other.waitingCapacityPrefillTokenSquareSum &&
           waitingCapacityContextRemainingPrefillTokenProductSum ==
               other.waitingCapacityContextRemainingPrefillTokenProductSum &&
           remainingTotalContextTokenCount == other.remainingTotalContextTokenCount &&
           remainingTotalTokenCount == other.remainingTotalTokenCount &&
           remainingTotalTokenSquareSum == other.remainingTotalTokenSquareSum &&
           contextRemainingTotalTokenProductSum ==
               other.contextRemainingTotalTokenProductSum &&
           totalPromptTokenCount == other.totalPromptTokenCount &&
           inFlightScheduledTokenCount == other.inFlightScheduledTokenCount;
}

void NaviSchedWorkloadSnapshot::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(revision);
    dataBuffer.write(eventTimeUs);
    dataBuffer.write(batchCount);
    dataBuffer.write(requestCount);
    dataBuffer.write(remainingPrefillRequestCount);
    dataBuffer.write(runningRequestCount);
    dataBuffer.write(runningElapsedTimeMs);
    dataBuffer.write(runningDecodeRequestCount);
    dataBuffer.write(runningDecodeGeneratedTokenCount);
    dataBuffer.write(runningDecodeRemainingTokenBudget);
    dataBuffer.write(waitingRequestCount);
    dataBuffer.write(remainingPrefillContextTokenCount);
    dataBuffer.write(remainingPrefillTokenCount);
    dataBuffer.write(remainingPrefillTokenSquareSum);
    dataBuffer.write(contextRemainingPrefillTokenProductSum);
    dataBuffer.write(prefillCapacitySplit);
    dataBuffer.write(maxRunningRequestCount);
    dataBuffer.write(runningCapacitySlotCount);
    dataBuffer.write(runningCapacityPrefillRequestCount);
    dataBuffer.write(runningCapacityPrefillContextTokenCount);
    dataBuffer.write(runningCapacityPrefillTokenCount);
    dataBuffer.write(runningCapacityPrefillTokenSquareSum);
    dataBuffer.write(runningCapacityContextRemainingPrefillTokenProductSum);
    dataBuffer.write(waitingCapacityPrefillRequestCount);
    dataBuffer.write(waitingCapacityPrefillContextTokenCount);
    dataBuffer.write(waitingCapacityPrefillTokenCount);
    dataBuffer.write(waitingCapacityPrefillTokenSquareSum);
    dataBuffer.write(waitingCapacityContextRemainingPrefillTokenProductSum);
    dataBuffer.write(remainingTotalContextTokenCount);
    dataBuffer.write(remainingTotalTokenCount);
    dataBuffer.write(remainingTotalTokenSquareSum);
    dataBuffer.write(contextRemainingTotalTokenProductSum);
    dataBuffer.write(totalPromptTokenCount);
    dataBuffer.write(inFlightScheduledTokenCount);
    dataBuffer.write(admittedBatch);
}

void NaviSchedWorkloadSnapshot::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(revision);
    dataBuffer.read(eventTimeUs);
    dataBuffer.read(batchCount);
    dataBuffer.read(requestCount);
    dataBuffer.read(remainingPrefillRequestCount);
    dataBuffer.read(runningRequestCount);
    dataBuffer.read(runningElapsedTimeMs);
    dataBuffer.read(runningDecodeRequestCount);
    dataBuffer.read(runningDecodeGeneratedTokenCount);
    dataBuffer.read(runningDecodeRemainingTokenBudget);
    dataBuffer.read(waitingRequestCount);
    dataBuffer.read(remainingPrefillContextTokenCount);
    dataBuffer.read(remainingPrefillTokenCount);
    dataBuffer.read(remainingPrefillTokenSquareSum);
    dataBuffer.read(contextRemainingPrefillTokenProductSum);
    dataBuffer.read(prefillCapacitySplit);
    dataBuffer.read(maxRunningRequestCount);
    dataBuffer.read(runningCapacitySlotCount);
    dataBuffer.read(runningCapacityPrefillRequestCount);
    dataBuffer.read(runningCapacityPrefillContextTokenCount);
    dataBuffer.read(runningCapacityPrefillTokenCount);
    dataBuffer.read(runningCapacityPrefillTokenSquareSum);
    dataBuffer.read(runningCapacityContextRemainingPrefillTokenProductSum);
    dataBuffer.read(waitingCapacityPrefillRequestCount);
    dataBuffer.read(waitingCapacityPrefillContextTokenCount);
    dataBuffer.read(waitingCapacityPrefillTokenCount);
    dataBuffer.read(waitingCapacityPrefillTokenSquareSum);
    dataBuffer.read(waitingCapacityContextRemainingPrefillTokenProductSum);
    dataBuffer.read(remainingTotalContextTokenCount);
    dataBuffer.read(remainingTotalTokenCount);
    dataBuffer.read(remainingTotalTokenSquareSum);
    dataBuffer.read(contextRemainingTotalTokenProductSum);
    dataBuffer.read(totalPromptTokenCount);
    dataBuffer.read(inFlightScheduledTokenCount);
    dataBuffer.read(admittedBatch);
}

bool NaviSchedServerInfo::valid() const { return port > 0 && port <= 65535; }

std::string NaviSchedServerInfo::getAddress(const std::string &defaultIp) const
{
    if (!valid()) {
        return "";
    }
    const auto &serverIp = ip.empty() ? defaultIp : ip;
    if (serverIp.empty()) {
        return "";
    }
    return serverIp + ":" + std::to_string(port);
}

void NaviSchedServerInfo::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(ip);
    dataBuffer.write(port);
}

void NaviSchedServerInfo::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(ip);
    dataBuffer.read(port);
}

void NaviSchedTarget::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(bizName);
    dataBuffer.write(address);
    dataBuffer.write(agentAddress);
}

void NaviSchedTarget::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    dataBuffer.read(bizName);
    dataBuffer.read(address);
    dataBuffer.read(agentAddress);
}

void NaviSchedDecision::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(requestId);
    dataBuffer.write(scheduleId);
    dataBuffer.write(scheduleLoss);
    dataBuffer.write(scheduleLatencyUs);
    dataBuffer.write(scheduleTimeUs);
    dataBuffer.write(reason);
    dataBuffer.write(batchId);
    dataBuffer.write(expectedBatchSize);
    dataBuffer.write(predictedCacheHitTokenCount);
    dataBuffer.write(predictedQueueLatencyMs);
    dataBuffer.write(predictedExecutionLatencyMs);
    dataBuffer.write(bool(latencyPrediction));
    if (latencyPrediction) {
        dataBuffer.write(*latencyPrediction);
    }
    dataBuffer.write(static_cast<int8_t>(latencyTrainingPhase));
    dataBuffer.write(target);
}

void NaviSchedDecision::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    dataBuffer.read(requestId);
    dataBuffer.read(scheduleId);
    dataBuffer.read(scheduleLoss);
    dataBuffer.read(scheduleLatencyUs);
    dataBuffer.read(scheduleTimeUs);
    dataBuffer.read(reason);
    dataBuffer.read(batchId);
    dataBuffer.read(expectedBatchSize);
    dataBuffer.read(predictedCacheHitTokenCount);
    dataBuffer.read(predictedQueueLatencyMs);
    dataBuffer.read(predictedExecutionLatencyMs);
    bool hasLatencyPrediction = false;
    dataBuffer.read(hasLatencyPrediction);
    if (hasLatencyPrediction) {
        auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
        dataBuffer.read(*prediction);
        latencyPrediction = std::move(prediction);
    } else {
        latencyPrediction.reset();
    }
    int8_t serializedTrainingPhase = 0;
    dataBuffer.read(serializedTrainingPhase);
    latencyTrainingPhase =
        static_cast<NaviSchedLatencyTrainingPhase>(serializedTrainingPhase);
    dataBuffer.read(target);
}

void NaviSchedRequestTiming::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(batchWaitMs);
    dataBuffer.write(queueWaitMs);
    dataBuffer.write(executionLatencyMs);
    dataBuffer.write(clientTotalLatencyMs);
}

void NaviSchedRequestTiming::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(batchWaitMs);
    dataBuffer.read(queueWaitMs);
    dataBuffer.read(executionLatencyMs);
    dataBuffer.read(clientTotalLatencyMs);
}

void NaviSchedLatencyValidationStats::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(sampleCount);
    dataBuffer.write(unmatchedBatchCount);
    dataBuffer.write(batchRequestCountMean);
    dataBuffer.write(predictedCacheHitTokensMean);
    dataBuffer.write(actualCacheHitTokensMean);
    dataBuffer.write(cacheHitTokensBiasMean);
    dataBuffer.write(cacheHitTokensMae);
    dataBuffer.write(masterPredictedQueueMeanMs);
    dataBuffer.write(actualQueueMeanMs);
    dataBuffer.write(backendQueueMeanMs);
    dataBuffer.write(queueBiasMeanMs);
    dataBuffer.write(queueMaeMs);
    dataBuffer.write(queueRawPredictedMeanMs);
    dataBuffer.write(queueRawBiasMeanMs);
    dataBuffer.write(queueRawMaeMs);
    dataBuffer.write(agentPredictedQueueMeanMs);
    dataBuffer.write(agentQueueBiasMeanMs);
    dataBuffer.write(agentQueueMaeMs);
    dataBuffer.write(queueFeatureBatchCountMean);
    dataBuffer.write(queueModelWeights);
    dataBuffer.write(queueModelFitStep);
    dataBuffer.write(zeroPredictedQueueSampleCount);
    dataBuffer.write(zeroPredictedQueueActualMeanMs);
    dataBuffer.write(zeroPredictedQueueMaeMs);
    dataBuffer.write(positivePredictedQueueSampleCount);
    dataBuffer.write(positivePredictedQueueMeanMs);
    dataBuffer.write(positivePredictedQueueActualMeanMs);
    dataBuffer.write(positivePredictedQueueBiasMeanMs);
    dataBuffer.write(positivePredictedQueueMaeMs);
    dataBuffer.write(masterPredictedExecutionMeanMs);
    dataBuffer.write(agentPredictedExecutionMeanMs);
    dataBuffer.write(actualExecutionMeanMs);
    dataBuffer.write(masterExecutionBiasMeanMs);
    dataBuffer.write(masterExecutionMaeMs);
    dataBuffer.write(masterExecutionMapePct);
    dataBuffer.write(agentExecutionBiasMeanMs);
    dataBuffer.write(agentExecutionMaeMs);
    dataBuffer.write(agentExecutionMapePct);
    dataBuffer.write(masterPredictedTotalMeanMs);
    dataBuffer.write(actualTotalMeanMs);
    dataBuffer.write(totalBiasMeanMs);
    dataBuffer.write(totalMaeMs);
    dataBuffer.write(totalMapePct);
    dataBuffer.write(executionLatencyBuckets);
}

void NaviSchedLatencyValidationStats::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(sampleCount);
    dataBuffer.read(unmatchedBatchCount);
    dataBuffer.read(batchRequestCountMean);
    dataBuffer.read(predictedCacheHitTokensMean);
    dataBuffer.read(actualCacheHitTokensMean);
    dataBuffer.read(cacheHitTokensBiasMean);
    dataBuffer.read(cacheHitTokensMae);
    dataBuffer.read(masterPredictedQueueMeanMs);
    dataBuffer.read(actualQueueMeanMs);
    dataBuffer.read(backendQueueMeanMs);
    dataBuffer.read(queueBiasMeanMs);
    dataBuffer.read(queueMaeMs);
    dataBuffer.read(queueRawPredictedMeanMs);
    dataBuffer.read(queueRawBiasMeanMs);
    dataBuffer.read(queueRawMaeMs);
    dataBuffer.read(agentPredictedQueueMeanMs);
    dataBuffer.read(agentQueueBiasMeanMs);
    dataBuffer.read(agentQueueMaeMs);
    dataBuffer.read(queueFeatureBatchCountMean);
    dataBuffer.read(queueModelWeights);
    dataBuffer.read(queueModelFitStep);
    dataBuffer.read(zeroPredictedQueueSampleCount);
    dataBuffer.read(zeroPredictedQueueActualMeanMs);
    dataBuffer.read(zeroPredictedQueueMaeMs);
    dataBuffer.read(positivePredictedQueueSampleCount);
    dataBuffer.read(positivePredictedQueueMeanMs);
    dataBuffer.read(positivePredictedQueueActualMeanMs);
    dataBuffer.read(positivePredictedQueueBiasMeanMs);
    dataBuffer.read(positivePredictedQueueMaeMs);
    dataBuffer.read(masterPredictedExecutionMeanMs);
    dataBuffer.read(agentPredictedExecutionMeanMs);
    dataBuffer.read(actualExecutionMeanMs);
    dataBuffer.read(masterExecutionBiasMeanMs);
    dataBuffer.read(masterExecutionMaeMs);
    dataBuffer.read(masterExecutionMapePct);
    dataBuffer.read(agentExecutionBiasMeanMs);
    dataBuffer.read(agentExecutionMaeMs);
    dataBuffer.read(agentExecutionMapePct);
    dataBuffer.read(masterPredictedTotalMeanMs);
    dataBuffer.read(actualTotalMeanMs);
    dataBuffer.read(totalBiasMeanMs);
    dataBuffer.read(totalMaeMs);
    dataBuffer.read(totalMapePct);
    dataBuffer.read(executionLatencyBuckets);
}

void NaviSchedExecutionLatencyBucketStats::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(sampleCount);
    dataBuffer.write(masterExecutionMaeMs);
    dataBuffer.write(masterExecutionMapePct);
    dataBuffer.write(agentExecutionMaeMs);
    dataBuffer.write(agentExecutionMapePct);
}

void NaviSchedExecutionLatencyBucketStats::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(sampleCount);
    dataBuffer.read(masterExecutionMaeMs);
    dataBuffer.read(masterExecutionMapePct);
    dataBuffer.read(agentExecutionMaeMs);
    dataBuffer.read(agentExecutionMapePct);
}

void NaviSchedNodeInfo::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(agentIdHash);
    dataBuffer.write(agentId);
    dataBuffer.write(bizName);
    dataBuffer.write(collectorAddress);
    dataBuffer.write(serverInfo);
    dataBuffer.write(serverAddress);
    dataBuffer.write(totalBlocks);
    dataBuffer.write(inflightStatsRevision);
    dataBuffer.write(inflightRequests);
    dataBuffer.write(inflightPredictedLatencySumMs);
    dataBuffer.write(batchWaitingRequests);
    dataBuffer.write(queuedBatchRequests);
    dataBuffer.write(runningRequests);
    dataBuffer.write(pdSeparated);
    dataBuffer.write(unifiedLatencyPrediction);
    dataBuffer.write(workload);
    dataBuffer.write(reservedWorkloadFeatures);
    dataBuffer.write(workloadFeatureSnapshotValid);
    dataBuffer.write(workloadFeatures);
    dataBuffer.write(maxRunningRequestCount);
    dataBuffer.write(workloadRunningCapacitySlotCount);
    dataBuffer.write(reservedRunningCapacitySlotCount);
    dataBuffer.write(workloadReportReceiveTimeUs);
    dataBuffer.write(workloadStaleUpdateCount);
    dataBuffer.write(batchTimeoutCount);
    dataBuffer.write(abnormalBatchFinishCount);
    dataBuffer.write(lastRequestLatencyMs);
    dataBuffer.write(queueTotalPredictedLatencyMs);
    dataBuffer.write(queueDecayHeadRemainingMs);
    dataBuffer.write(queueModel);
    dataBuffer.write(queueReportReceiveTimeUs);
    dataBuffer.write(online);
    dataBuffer.write(lastEventType);
    dataBuffer.write(lastEventTimeUs);
    dataBuffer.write(lastRequestEventType);
    dataBuffer.write(lastRequestEventTimeUs);
    dataBuffer.write(lastReceiveTimeUs);
    dataBuffer.write(latencyFormula);
    dataBuffer.write(latencyBinaryVersion);
    dataBuffer.write(latencyModelRevision);
    dataBuffer.write(latencyParameters);
    dataBuffer.write(latencyResidualVersion);
    dataBuffer.write(latencyResidualParameters);
    dataBuffer.write(latencyResidualLastUpdateTimeUs);
    dataBuffer.write(latencyModelFingerprint);
    dataBuffer.write(latencyModelTrusted);
    dataBuffer.write(latencyModelWarmStarted);
    dataBuffer.write(latencyModelConsecutiveRejectedCount);
    dataBuffer.write(latencyModelLastRejectReason);
    dataBuffer.write(schedulerConfigGeneration);
    dataBuffer.write(schedulerConfigHash);
    dataBuffer.write(schedulerConfigJson);
    dataBuffer.write(latencyValidation);
    dataBuffer.write(peerCacheEnabled);
}

void NaviSchedNodeInfo::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(agentIdHash);
    dataBuffer.read(agentId);
    dataBuffer.read(bizName);
    dataBuffer.read(collectorAddress);
    dataBuffer.read(serverInfo);
    dataBuffer.read(serverAddress);
    dataBuffer.read(totalBlocks);
    dataBuffer.read(inflightStatsRevision);
    dataBuffer.read(inflightRequests);
    dataBuffer.read(inflightPredictedLatencySumMs);
    dataBuffer.read(batchWaitingRequests);
    dataBuffer.read(queuedBatchRequests);
    dataBuffer.read(runningRequests);
    dataBuffer.read(pdSeparated);
    dataBuffer.read(unifiedLatencyPrediction);
    dataBuffer.read(workload);
    dataBuffer.read(reservedWorkloadFeatures);
    dataBuffer.read(workloadFeatureSnapshotValid);
    dataBuffer.read(workloadFeatures);
    dataBuffer.read(maxRunningRequestCount);
    dataBuffer.read(workloadRunningCapacitySlotCount);
    dataBuffer.read(reservedRunningCapacitySlotCount);
    dataBuffer.read(workloadReportReceiveTimeUs);
    dataBuffer.read(workloadStaleUpdateCount);
    dataBuffer.read(batchTimeoutCount);
    dataBuffer.read(abnormalBatchFinishCount);
    dataBuffer.read(lastRequestLatencyMs);
    dataBuffer.read(queueTotalPredictedLatencyMs);
    dataBuffer.read(queueDecayHeadRemainingMs);
    dataBuffer.read(queueModel);
    dataBuffer.read(queueReportReceiveTimeUs);
    dataBuffer.read(online);
    dataBuffer.read(lastEventType);
    dataBuffer.read(lastEventTimeUs);
    dataBuffer.read(lastRequestEventType);
    dataBuffer.read(lastRequestEventTimeUs);
    dataBuffer.read(lastReceiveTimeUs);
    dataBuffer.read(latencyFormula);
    dataBuffer.read(latencyBinaryVersion);
    dataBuffer.read(latencyModelRevision);
    dataBuffer.read(latencyParameters);
    dataBuffer.read(latencyResidualVersion);
    dataBuffer.read(latencyResidualParameters);
    dataBuffer.read(latencyResidualLastUpdateTimeUs);
    dataBuffer.read(latencyModelFingerprint);
    dataBuffer.read(latencyModelTrusted);
    dataBuffer.read(latencyModelWarmStarted);
    dataBuffer.read(latencyModelConsecutiveRejectedCount);
    dataBuffer.read(latencyModelLastRejectReason);
    dataBuffer.read(schedulerConfigGeneration);
    dataBuffer.read(schedulerConfigHash);
    dataBuffer.read(schedulerConfigJson);
    dataBuffer.read(latencyValidation);
    dataBuffer.read(peerCacheEnabled);
}

bool NaviSchedScheduleRequest::valid() const
{
    return version == NAVI_SCHED_DATA_VERSION && requestId != 0 && !bizName.empty() &&
           timeoutMs > 0 && explicitCacheCheckpointPosition >= 0 &&
           explicitCacheCheckpointPosition <= promptTokenCount &&
           BlockHashUtil::validTokenBlocks(promptTokenCount, blockSize, blockHashes);
}

void NaviSchedScheduleRequest::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(requestId);
    dataBuffer.write(bizName);
    dataBuffer.write(promptTokenCount);
    dataBuffer.write(explicitCacheCheckpointPosition);
    dataBuffer.write(blockSize);
    BlockHashUtil::write(dataBuffer, blockHashes);
    dataBuffer.write(timeoutMs);
    dataBuffer.write(routeOnly);
}

void NaviSchedScheduleRequest::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    if (version != NAVI_SCHED_DATA_VERSION) {
        return;
    }
    dataBuffer.read(requestId);
    dataBuffer.read(bizName);
    dataBuffer.read(promptTokenCount);
    dataBuffer.read(explicitCacheCheckpointPosition);
    dataBuffer.read(blockSize);
    BlockHashUtil::read(dataBuffer, blockHashes);
    dataBuffer.read(timeoutMs);
    dataBuffer.read(routeOnly);
}

void NaviSchedScheduleItem::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(request);
    dataBuffer.write(waitTimeMs);
    dataBuffer.write(remainingTimeoutMs);
    dataBuffer.write(timeoutTriggered);
}

void NaviSchedScheduleItem::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(request);
    dataBuffer.read(waitTimeMs);
    dataBuffer.read(remainingTimeoutMs);
    dataBuffer.read(timeoutTriggered);
}

void NaviSchedReplayNodeMatch::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(matchedBlockCount);
}

void NaviSchedReplayNodeMatch::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(matchedBlockCount);
}

void NaviSchedScheduleReplay::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(scheduleTimeUs);
    dataBuffer.write(bizName);
    dataBuffer.write(requests);
    dataBuffer.write(nodes);
    dataBuffer.write(nodeMatches);
    dataBuffer.write(decisions);
    dataBuffer.write(autil::legacy::FastToJsonString(schedulerInfo));
}

void NaviSchedScheduleReplay::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    if (version != NAVI_SCHED_DATA_VERSION) {
        return;
    }
    dataBuffer.read(scheduleTimeUs);
    dataBuffer.read(bizName);
    dataBuffer.read(requests);
    dataBuffer.read(nodes);
    dataBuffer.read(nodeMatches);
    dataBuffer.read(decisions);
    std::string schedulerInfoJson;
    dataBuffer.read(schedulerInfoJson);
    schedulerInfo.clear();
    if (!schedulerInfoJson.empty()) {
        autil::legacy::FastFromJsonString(schedulerInfo, schedulerInfoJson);
    }
}

}
