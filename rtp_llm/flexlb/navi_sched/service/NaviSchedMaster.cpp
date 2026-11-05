#include "aios/navi_sched/service/NaviSchedMaster.h"
#include "aios/navi_sched/graph/MasterCollectorManager.h"
#include "aios/navi_sched/graph/MasterScheduleGraph.h"
#include "aios/navi_sched/kernel/SchedulerConfigSyncKernel.h"
#include "aios/navi_sched/kernel/ReplayControlSyncKernel.h"
#include "aios/navi_sched/kernel/ReplayMasterControlKernel.h"
#include "aios/navi_sched/kernel/ReplayMasterHistoryKernel.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/resource/RouteModeR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "aios/navi_sched/resource/ReplayControlR.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleResultGraph.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/service/NaviSchedMasterSummary.h"
#include "autil/legacy/jsonizable.h"
#include "autil/legacy/fast_jsonizable_dec.h"
#include "kmonitor/client/core/MetricsTags.h"
#include "navi/resource/GigClientR.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>

namespace navi_sched {
AUTIL_LOG_SETUP(navi_sched, NaviSchedMaster);
namespace {

struct ValidationWindowTotals {
    uint64_t sampleCount = 0;
    uint64_t unmatchedBatchCount = 0;
    uint64_t zeroQueueSampleCount = 0;
    uint64_t positiveQueueSampleCount = 0;
    double batchRequestCount = 0.0;
    double predictedCacheHitTokens = 0.0;
    double actualCacheHitTokens = 0.0;
    double predictedQueueMs = 0.0;
    double actualQueueMs = 0.0;
    double queueRawPredictedMs = 0.0;
    double agentPredictedQueueMs = 0.0;
    double queueFeatureBatchCount = 0.0;
    double zeroQueueActualMs = 0.0;
    double positivePredictedQueueMs = 0.0;
    double positiveActualQueueMs = 0.0;
    double masterPredictedExecutionMs = 0.0;
    double agentPredictedExecutionMs = 0.0;
    double actualExecutionMs = 0.0;
    double predictedTotalMs = 0.0;
    double actualTotalMs = 0.0;

    static double deltaSum(double currentMean, uint64_t currentCount, double previousMean,
                           uint64_t previousCount)
    {
        return currentMean * currentCount - previousMean * previousCount;
    }

    void add(const NaviSchedLatencyValidationStats &current,
             const NaviSchedLatencyValidationStats &previous)
    {
        if (current.sampleCount < previous.sampleCount) {
            return;
        }
        const auto count = current.sampleCount;
        const auto previousCount = previous.sampleCount;
        sampleCount += count - previousCount;
        if (current.unmatchedBatchCount >= previous.unmatchedBatchCount) {
            unmatchedBatchCount +=
                current.unmatchedBatchCount - previous.unmatchedBatchCount;
        }
        const auto addSampleSum =
            [count, previousCount](double currentMean, double previousMean, double &sum) {
                sum += deltaSum(currentMean, count, previousMean, previousCount);
            };
        addSampleSum(current.batchRequestCountMean, previous.batchRequestCountMean,
                     batchRequestCount);
        addSampleSum(current.predictedCacheHitTokensMean,
                     previous.predictedCacheHitTokensMean, predictedCacheHitTokens);
        addSampleSum(current.actualCacheHitTokensMean, previous.actualCacheHitTokensMean,
                     actualCacheHitTokens);
        addSampleSum(current.masterPredictedQueueMeanMs,
                     previous.masterPredictedQueueMeanMs, predictedQueueMs);
        addSampleSum(current.actualQueueMeanMs, previous.actualQueueMeanMs,
                     actualQueueMs);
        addSampleSum(current.queueRawPredictedMeanMs, previous.queueRawPredictedMeanMs,
                     queueRawPredictedMs);
        addSampleSum(current.agentPredictedQueueMeanMs,
                     previous.agentPredictedQueueMeanMs, agentPredictedQueueMs);
        addSampleSum(current.queueFeatureBatchCountMean,
                     previous.queueFeatureBatchCountMean, queueFeatureBatchCount);
        addSampleSum(current.masterPredictedExecutionMeanMs,
                     previous.masterPredictedExecutionMeanMs, masterPredictedExecutionMs);
        addSampleSum(current.agentPredictedExecutionMeanMs,
                     previous.agentPredictedExecutionMeanMs, agentPredictedExecutionMs);
        addSampleSum(current.actualExecutionMeanMs, previous.actualExecutionMeanMs,
                     actualExecutionMs);
        addSampleSum(current.masterPredictedTotalMeanMs,
                     previous.masterPredictedTotalMeanMs, predictedTotalMs);
        addSampleSum(current.actualTotalMeanMs, previous.actualTotalMeanMs,
                     actualTotalMs);

        if (current.zeroPredictedQueueSampleCount >=
            previous.zeroPredictedQueueSampleCount) {
            const auto zeroCount = current.zeroPredictedQueueSampleCount;
            const auto previousZeroCount = previous.zeroPredictedQueueSampleCount;
            zeroQueueSampleCount += zeroCount - previousZeroCount;
            zeroQueueActualMs +=
                deltaSum(current.zeroPredictedQueueActualMeanMs, zeroCount,
                         previous.zeroPredictedQueueActualMeanMs, previousZeroCount);
        }
        if (current.positivePredictedQueueSampleCount >=
            previous.positivePredictedQueueSampleCount) {
            const auto positiveCount = current.positivePredictedQueueSampleCount;
            const auto previousPositiveCount = previous.positivePredictedQueueSampleCount;
            positiveQueueSampleCount += positiveCount - previousPositiveCount;
            positivePredictedQueueMs +=
                deltaSum(current.positivePredictedQueueMeanMs, positiveCount,
                         previous.positivePredictedQueueMeanMs, previousPositiveCount);
            positiveActualQueueMs += deltaSum(
                current.positivePredictedQueueActualMeanMs, positiveCount,
                previous.positivePredictedQueueActualMeanMs, previousPositiveCount);
        }
    }
};

}

std::map<std::string, uint64_t> NaviSchedMaster::makeCacheStateSummary(
    const MasterStateR::CacheSyncStats &stats)
{
    return {
        {"cached_block_count", stats.masterCachedBlockCount},
        {"full_ready_block_count", stats.masterFullReadyBlockCount},
        {"tail_ready_block_count", stats.masterTailReadyBlockCount},
        {"pending_cache_batch_count", stats.pendingCacheBatchCount},
        {"pending_cache_agent_batch_count", stats.pendingCacheAgentBatchCount},
        {"pending_cache_scheduled_batch_count", stats.pendingCacheScheduledBatchCount},
        {"pending_cache_block_count", stats.pendingCacheBlockCount},
        {"pending_cache_tail_ready_block_count", stats.pendingCacheTailReadyBlockCount},
        {"cache_revision_gap_total", stats.revisionGapCount},
        {"cache_state_mismatch_total", stats.stateMismatchCount},
        {"pending_cache_revision_gap_total", stats.pendingCacheRevisionGapCount},
        {"pending_cache_state_mismatch_total", stats.pendingCacheStateMismatchCount},
        {"pending_cache_resync_trigger_total", stats.pendingCacheResyncTriggerCount},
        {"cache_ready_node_count", stats.cacheStateReady},
        {"pending_cache_ready_node_count", stats.pendingCacheSnapshotReady},
    };
}

NaviSchedMasterNodeSummary NaviSchedMaster::makeNodeSummary(
    const NaviSchedNodeInfo &node, const MasterStateR::CacheSyncStats &cacheStats,
    uint64_t sentRequestCount, uint64_t sentBatchCount, double timeCompression)
{
    NaviSchedMasterNodeSummary summary;
    summary.agentIdHash = node.agentIdHash;
    summary.agentId = node.agentId;
    summary.bizName = node.bizName;
    summary.collectorAddress = node.collectorAddress;
    summary.serverIp = node.serverInfo.ip;
    summary.serverPort = node.serverInfo.port;
    summary.serverAddress = node.serverAddress;
    summary.online = node.online;
    summary.pdSeparated = node.pdSeparated;
    summary.peerCacheEnabled = node.peerCacheEnabled;
    summary.peerCacheReadAvailable = node.peerCacheReadAvailable;
    summary.peerCacheReadRevision = node.peerCacheReadRevision;
    summary.peerCacheReadAgentStartTimeUs = node.peerCacheReadAgentStartTimeUs;
    summary.totalBlocks = node.totalBlocks;
    summary.cachedBlockCount = cacheStats.masterCachedBlockCount;
    summary.agentCacheRevision = cacheStats.agentRevision;
    summary.agentCachedBlockCount = cacheStats.agentCachedBlockCount;
    summary.agentCacheHashChecksum = cacheStats.agentHashChecksum;
    summary.agentFullReadyBlockCount = cacheStats.agentFullReadyBlockCount;
    summary.agentFullReadyHashChecksum = cacheStats.agentFullReadyHashChecksum;
    summary.agentTailReadyBlockCount = cacheStats.agentTailReadyBlockCount;
    summary.agentTailReadyHashChecksum = cacheStats.agentTailReadyHashChecksum;
    summary.masterCacheHashChecksum = cacheStats.masterHashChecksum;
    summary.masterFullReadyBlockCount = cacheStats.masterFullReadyBlockCount;
    summary.masterFullReadyHashChecksum = cacheStats.masterFullReadyHashChecksum;
    summary.masterTailReadyBlockCount = cacheStats.masterTailReadyBlockCount;
    summary.masterTailReadyHashChecksum = cacheStats.masterTailReadyHashChecksum;
    summary.rtpCachedBlockCount = cacheStats.rtpCachedBlockCount;
    summary.cacheRevisionGapCount = cacheStats.revisionGapCount;
    summary.cacheStateMismatchCount = cacheStats.stateMismatchCount;
    summary.rtpCacheMirrorSampleCount = cacheStats.rtpMirrorSampleCount;
    summary.rtpCacheMirrorMismatchCount = cacheStats.rtpMirrorMismatchCount;
    summary.pendingCacheBatchCount = cacheStats.pendingCacheBatchCount;
    summary.pendingCacheBlockCount = cacheStats.pendingCacheBlockCount;
    summary.pendingCacheScheduledBatchCount = cacheStats.pendingCacheScheduledBatchCount;
    summary.pendingCacheAgentBatchCount = cacheStats.pendingCacheAgentBatchCount;
    summary.pendingCacheAgentRevision = cacheStats.pendingCacheAgentRevision;
    summary.pendingCacheAgentChecksum = cacheStats.pendingCacheAgentChecksum;
    summary.pendingCacheRevisionGapCount = cacheStats.pendingCacheRevisionGapCount;
    summary.pendingCacheStateMismatchCount = cacheStats.pendingCacheStateMismatchCount;
    summary.pendingCacheResyncTriggerCount = cacheStats.pendingCacheResyncTriggerCount;
    summary.pendingCacheSnapshotReady = cacheStats.pendingCacheSnapshotReady;
    summary.inflightStatsRevision = node.inflightStatsRevision;
    summary.inflightRequests = node.inflightRequests;
    summary.inflightPredictedLatencySumMs = node.inflightPredictedLatencySumMs;
    summary.batchWaitingRequests = node.batchWaitingRequests;
    summary.queuedBatchRequests = node.queuedBatchRequests;
    summary.runningRequests = node.runningRequests;
    summary.batchTimeoutCount = node.batchTimeoutCount;
    summary.abnormalBatchFinishCount = node.abnormalBatchFinishCount;
    summary.masterSentRequestCount = sentRequestCount;
    summary.masterSentBatchCount = sentBatchCount;
    summary.masterSentRequestsPerBatchMean =
        sentBatchCount > 0 ? static_cast<double>(sentRequestCount) / sentBatchCount : 0.0;
    summary.lastRequestLatencyMs = node.lastRequestLatencyMs;
    summary.queueTotalPredictedLatencyMs = node.queueTotalPredictedLatencyMs;
    summary.queueRawDecayFactor = node.queueModel.rawDecayFactor;
    summary.queueReportReceiveTimeUs = node.queueReportReceiveTimeUs;
    summary.workload = node.workload;
    summary.workloadReportReceiveTimeUs = node.workloadReportReceiveTimeUs;
    summary.workloadEffectiveRunningElapsedTimeMs = node.workload.runningElapsedTimeMs;
    if (node.workloadReportReceiveTimeUs > 0 &&
        node.workload.runningCapacityPrefillRequestCount > 0) {
        summary.workloadEffectiveRunningElapsedTimeMs +=
            std::max<int64_t>(
                0, autil::TimeUtility::currentTime() - node.workloadReportReceiveTimeUs) /
            1000.0 * node.workload.runningCapacityPrefillRequestCount * timeCompression;
    }
    summary.workloadStaleUpdateCount = node.workloadStaleUpdateCount;
    summary.lastEventType = node.lastEventType;
    summary.lastEventTimeUs = node.lastEventTimeUs;
    summary.lastRequestEventType = node.lastRequestEventType;
    summary.lastRequestEventTimeUs = node.lastRequestEventTimeUs;
    summary.lastReceiveTimeUs = node.lastReceiveTimeUs;
    summary.latencyFormula = node.latencyFormula;
    summary.latencyBinaryVersion = node.latencyBinaryVersion;
    summary.latencyModelRevision = node.latencyModelRevision;
    summary.latencyParameterCount = node.latencyParameters.size();
    summary.latencyParameters = node.latencyParameters;
    summary.latencyResidualVersion = node.latencyResidualVersion;
    summary.latencyResidualParameters = node.latencyResidualParameters;
    summary.latencyResidualLastUpdateTimeUs = node.latencyResidualLastUpdateTimeUs;
    summary.latencyModelFingerprint = node.latencyModelFingerprint;
    summary.latencyModelTrusted = node.latencyModelTrusted;
    summary.latencyModelWarmStarted = node.latencyModelWarmStarted;
    summary.latencyModelConsecutiveRejectedCount =
        node.latencyModelConsecutiveRejectedCount;
    summary.latencyModelLastRejectReason = node.latencyModelLastRejectReason;
    summary.schedulerConfigGeneration = node.schedulerConfigGeneration;
    summary.schedulerConfigHash = node.schedulerConfigHash;
    summary.schedulerConfigJson = node.schedulerConfigJson;
    summary.latencyValidation = node.latencyValidation;
    return summary;
}

NaviSchedMaster::NaviSchedMaster(const NaviSchedMasterConfig &config)
    : NaviSchedRole(config.service)
    , _masterBizName(config.runtime.bizName)
    , _masterConfig(config.runtime)
    , _collectorThreadNum(config.collectorThreadNum)
    , _collectorQueueCount(config.collectorQueueCount)
    , _configuredMaxWaitScheduleCount(config.maxWaitScheduleCount)
    , _configuredMaxWaitScheduleTimeMs(config.maxWaitScheduleTimeMs)
    , _configuredQueueOverloadStartCycleMultiplier(
          config.queueOverloadStartCycleMultiplier)
    , _configuredQueueOverloadFullCycleMultiplier(config.queueOverloadFullCycleMultiplier)
    , _configuredSchedulerName(config.schedulerName)
    , _configuredSchedulerConfig(config.schedulerConfig)
    , _configuredTimeCompression(config.timeCompression)
    , _configuredLatencyModelPoolStateFile(config.latencyModelPoolStateFile)
    , _configuredLatencyModelPoolSourceMinRevision(
          config.latencyModelPoolSourceMinRevision)
    , _configuredLatencyModelPoolMaxCandidates(config.latencyModelPoolMaxCandidates)
    , _configuredLatencyModelPoolPersistIntervalMs(
          config.latencyModelPoolPersistIntervalMs)
{
}

NaviSchedMaster::~NaviSchedMaster() { stop(); }

bool NaviSchedMaster::init()
{
    if (!NaviSchedRole::init()) {
        return false;
    }
    const auto &config = _masterConfig;
    if ((_collectorThreadNum == 0) != (_collectorQueueCount == 0) ||
        (_collectorThreadNum > 0 && (_collectorThreadNum >= config.threadNum ||
                                     _collectorQueueCount > _collectorThreadNum))) {
        AUTIL_LOG(ERROR,
                  "invalid collector queues, total threads [%lu], collector threads "
                  "[%lu], queues [%lu]",
                  config.threadNum, _collectorThreadNum, _collectorQueueCount);
        stop();
        return false;
    }
    auto runtimeConfig = config;
    runtimeConfig.threadNum -= _collectorThreadNum;
    std::map<std::string, size_t> collectorTaskQueues;
    _collectorTaskQueueNames.clear();
    for (size_t index = 0; index < _collectorQueueCount; ++index) {
        const auto name = "nsc" + std::to_string(index);
        const size_t threads = _collectorThreadNum / _collectorQueueCount +
                               (index < _collectorThreadNum % _collectorQueueCount);
        collectorTaskQueues.emplace(name, threads);
        _collectorTaskQueueNames.emplace_back(name);
    }
    const bool queueOverloadLimitDisabled =
        _configuredQueueOverloadStartCycleMultiplier == 0 ||
        _configuredQueueOverloadFullCycleMultiplier == 0;
    if (0 == _configuredMaxWaitScheduleCount || _configuredMaxWaitScheduleTimeMs < 0 ||
        !std::isfinite(_configuredTimeCompression) || _configuredTimeCompression <= 0.0 ||
        _configuredQueueOverloadStartCycleMultiplier < 0 ||
        _configuredQueueOverloadFullCycleMultiplier < 0 ||
        (!queueOverloadLimitDisabled &&
         _configuredQueueOverloadFullCycleMultiplier <=
             _configuredQueueOverloadStartCycleMultiplier) ||
        _configuredLatencyModelPoolSourceMinRevision <= 1 ||
        _configuredLatencyModelPoolMaxCandidates == 0 ||
        _configuredLatencyModelPoolPersistIntervalMs <= 0) {
        stop();
        return false;
    }
    if (_configuredSchedulerName.empty()) {
        stop();
        return false;
    }
    _schedulerKernelId = "navi_sched." + _configuredSchedulerName + "_scheduler_k";
    autil::legacy::json::JsonMap schedulerConfig;
    try {
        if (!_configuredSchedulerConfig.empty()) {
            schedulerConfig = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(_configuredSchedulerConfig));
        }
    } catch (...) {
        stop();
        return false;
    }
    _maxWaitScheduleTimeMs = std::max<int64_t>(10, _configuredMaxWaitScheduleTimeMs);
    if (CostSchedulerKernel::KERNEL_ID == _schedulerKernelId) {
        CostSchedulerConfig costConfig;
        try {
            autil::legacy::FromJson(costConfig, autil::legacy::Any(schedulerConfig));
        } catch (...) {
            stop();
            return false;
        }
        if (costConfig.parallelism == 0) {
            schedulerConfig["parallelism"] = static_cast<int32_t>(std::max<size_t>(
                1, std::min<size_t>(runtimeConfig.threadNum,
                                    CostSchedulerConfig::MAX_PARALLELISM)));
        }
        if (_maxWaitScheduleTimeMs > std::numeric_limits<int64_t>::max() / 1000) {
            stop();
            return false;
        }
        schedulerConfig["optimization_time_budget_us"] = _maxWaitScheduleTimeMs * 800;
    }
    autil::legacy::json::JsonMap contextConfig;
    contextConfig["max_wait_schedule_count"] = _configuredMaxWaitScheduleCount;
    contextConfig["max_wait_schedule_time_ms"] = _maxWaitScheduleTimeMs;
    contextConfig["time_compression"] = _configuredTimeCompression;
    _contextKernelConfig = autil::legacy::ToJsonString(contextConfig, true);
    _schedulerKernelConfig = autil::legacy::ToJsonString(schedulerConfig, true);
    _maxWaitScheduleCount = _configuredMaxWaitScheduleCount;
    std::vector<std::string> kernels = {
        "navi_sched.master_*",
        "navi_sched.schedule_k",
        "navi_sched.route_mode_k",
        "navi_sched.scheduler_config_k",
        SchedulerConfigSyncKernel::KERNEL_ID,
        ReplayControlSyncKernel::KERNEL_ID,
        ReplayMasterControlKernel::KERNEL_ID,
        ReplayMasterHistoryKernel::KERNEL_ID,
        MasterScheduleContextKernel::KERNEL_ID,
        MasterScheduleResultKernel::KERNEL_ID,
        MasterScheduleResultPrepareKernel::KERNEL_ID,
        MasterScheduleResultWorkerKernel::KERNEL_ID,
        MasterScheduleResultBarrierKernel::KERNEL_ID,
        _schedulerKernelId,
    };
    if (CostSchedulerKernel::KERNEL_ID == _schedulerKernelId) {
        kernels.emplace_back("navi_sched.cost_scheduler_*");
    }
    SchedulerConfigR::Config dynamicConfig;
    dynamicConfig.maxWaitScheduleCount = _maxWaitScheduleCount;
    dynamicConfig.maxWaitScheduleTimeMs = _maxWaitScheduleTimeMs;
    dynamicConfig.queueOverloadStartCycleMultiplier =
        _configuredQueueOverloadStartCycleMultiplier;
    dynamicConfig.queueOverloadFullCycleMultiplier =
        _configuredQueueOverloadFullCycleMultiplier;
    dynamicConfig.overloadControlStrategy =
        (_configuredQueueOverloadStartCycleMultiplier > 0 ||
         _configuredQueueOverloadFullCycleMultiplier > 0)
            ? "latency"
            : "execution_utilization";
    dynamicConfig.agent =
        AgentSchedulerConfig::fromServiceConfig(getConfig(), _configuredTimeCompression);
    if (CostSchedulerKernel::KERNEL_ID == _schedulerKernelId) {
        try {
            autil::legacy::FromJsonString(dynamicConfig, _schedulerKernelConfig);
            const auto admission = schedulerConfig.find("admission_timeout_ms");
            if (admission != schedulerConfig.end()) {
                autil::legacy::FromJson(dynamicConfig.admissionTimeoutMs,
                                        admission->second);
            }
        } catch (...) {
            stop();
            return false;
        }
    }
    autil::legacy::json::JsonMap dynamicResourceConfig;
    dynamicResourceConfig["config_json"] = SchedulerConfigR::toJson(dynamicConfig);
    autil::legacy::json::JsonMap latencyModelPoolConfig;
    latencyModelPoolConfig["state_file"] = _configuredLatencyModelPoolStateFile;
    latencyModelPoolConfig["trust_min_revision"] =
        dynamicConfig.agent.latencyModelTrustMinRevision;
    latencyModelPoolConfig["source_min_revision"] =
        _configuredLatencyModelPoolSourceMinRevision;
    latencyModelPoolConfig["max_candidates"] = _configuredLatencyModelPoolMaxCandidates;
    latencyModelPoolConfig["persist_interval_ms"] =
        _configuredLatencyModelPoolPersistIntervalMs;
    auto configStr = NaviSchedConfigBuilder::build(
        runtimeConfig, kernels, true,
        {
            {MasterScheduleContextKernel::KERNEL_ID, _contextKernelConfig},
            {_schedulerKernelId, _schedulerKernelConfig},
        },
        {{SchedulerConfigR::RESOURCE_ID,
          autil::legacy::ToJsonString(dynamicResourceConfig, true)},
         {ReplayControlR::RESOURCE_ID, "{}"},
         {MasterLatencyModelR::RESOURCE_ID,
          autil::legacy::ToJsonString(latencyModelPoolConfig, true)}},
        {RouteModeR::RESOURCE_ID, SchedulerConfigR::RESOURCE_ID,
         ReplayControlR::RESOURCE_ID, MasterLatencyModelR::RESOURCE_ID},
        getConfig().gig.grpcIoThreadNum, collectorTaskQueues);
    if (!configStr.empty() && update(configStr, std::make_shared<navi::ResourceMap>())) {
        return true;
    }
    stop();
    return false;
}

bool NaviSchedMaster::enableHttpRpc() const { return true; }

void NaviSchedMaster::stop()
{
    stopSnapshot();
    if (_replayControlR) {
        _replayControlR->stop();
    }
    if (_scheduleGraph) {
        _scheduleGraph->stop();
        _scheduleGraph.reset();
    }
    if (_masterScheduleR) {
        _masterScheduleR->stop();
        _masterScheduleR.reset();
    }
    if (_collectorManager) {
        _collectorManager->stop();
        _collectorManager.reset();
    }
    if (_masterLatencyModelR) {
        _masterLatencyModelR->save();
        _masterLatencyModelR.reset();
    }
    _masterStateR.reset();
    _schedulerConfigR.reset();
    _replayControlR.reset();
    _lastValidationStats.clear();
    NaviSchedRole::stop();
}

std::map<std::string, NaviSchedNodeInfo> NaviSchedMaster::getNodes() const
{
    return _masterStateR ? _masterStateR->getNodes()
                         : std::map<std::string, NaviSchedNodeInfo>();
}

NaviSchedMasterStats NaviSchedMaster::getStats() const
{
    NaviSchedMasterStats stats;
    if (!_masterStateR) {
        return stats;
    }
    auto requestStats = _masterStateR->getAgentRequestStats();
    stats.agentRequestReceivedCount = requestStats.receivedCount;
    stats.agentRequestFinishedCount = requestStats.finishedCount;
    stats.agentRequestCacheHitTokens = requestStats.cacheHitTokens;
    stats.agentRequestCacheMissTokens = requestStats.cacheMissTokens;
    stats.agentRequestTotalLatencyUs = requestStats.totalLatencyUs;
    return stats;
}

void NaviSchedMaster::onUpdating()
{
    if (_replayControlR) {
        _replayControlR->stop();
    }
    if (_scheduleGraph) {
        _scheduleGraph->stop();
        _scheduleGraph.reset();
    }
    if (_masterScheduleR) {
        _masterScheduleR->stop();
        _masterScheduleR.reset();
    }
    if (_collectorManager) {
        _collectorManager->stop();
        _collectorManager.reset();
    }
    if (_masterLatencyModelR) {
        _masterLatencyModelR->save();
        _masterLatencyModelR.reset();
    }
    _masterStateR.reset();
    _schedulerConfigR.reset();
    _replayControlR.reset();
}

bool NaviSchedMaster::onUpdated()
{
    navi::ResourceMap resourceMap;
    std::set<std::string> resources = {
        navi::GigClientR::RESOURCE_ID, MasterScheduleR::RESOURCE_ID,
        MasterStateR::RESOURCE_ID,     MasterLatencyModelR::RESOURCE_ID,
        SchedulerConfigR::RESOURCE_ID, ReplayControlR::RESOURCE_ID,
    };
    const auto &navi = getNavi();
    if (!navi->createResource(_masterBizName, 1, 0, resources, resourceMap)) {
        return false;
    }
    auto stateIt = resourceMap.getMap().find(MasterStateR::RESOURCE_ID);
    auto latencyModelIt = resourceMap.getMap().find(MasterLatencyModelR::RESOURCE_ID);
    auto scheduleIt = resourceMap.getMap().find(MasterScheduleR::RESOURCE_ID);
    auto configIt = resourceMap.getMap().find(SchedulerConfigR::RESOURCE_ID);
    auto replayIt = resourceMap.getMap().find(ReplayControlR::RESOURCE_ID);
    if (resourceMap.getMap().end() == stateIt ||
        resourceMap.getMap().end() == latencyModelIt ||
        resourceMap.getMap().end() == scheduleIt ||
        resourceMap.getMap().end() == configIt ||
        resourceMap.getMap().end() == replayIt) {
        return false;
    }
    _masterStateR = std::dynamic_pointer_cast<MasterStateR>(stateIt->second);
    _masterLatencyModelR =
        std::dynamic_pointer_cast<MasterLatencyModelR>(latencyModelIt->second);
    _masterScheduleR = std::dynamic_pointer_cast<MasterScheduleR>(scheduleIt->second);
    _schedulerConfigR = std::dynamic_pointer_cast<SchedulerConfigR>(configIt->second);
    _replayControlR = std::dynamic_pointer_cast<ReplayControlR>(replayIt->second);
    if (!_masterStateR || !_masterLatencyModelR || !_masterScheduleR ||
        !_schedulerConfigR || !_replayControlR) {
        return false;
    }
    _scheduleGraph = std::make_shared<MasterScheduleGraph>(
        navi, _masterScheduleR, _schedulerConfigR, _masterBizName, _contextKernelConfig,
        _schedulerKernelId, _schedulerKernelConfig, getConfig().enableVis);
    if (!_scheduleGraph->start()) {
        _scheduleGraph.reset();
        return false;
    }
    _collectorManager = std::make_shared<MasterCollectorManager>(
        navi, _masterStateR, _masterBizName, getConfig().enableVis,
        _collectorTaskQueueNames);
    auto gigIt = resourceMap.getMap().find(navi::GigClientR::RESOURCE_ID);
    if (resourceMap.getMap().end() == gigIt) {
        return false;
    }
    auto gigClientR = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    return gigClientR &&
           _collectorManager->startAutoCollect(gigClientR->getSearchService());
}

std::string NaviSchedMaster::makeSnapshot() const
{
    NaviSchedMasterSummary summary;
    static_cast<NaviSchedRoleSummary &>(summary) = *makeSnapshotSummary();
    summary.role = "master";
    summary.masterBizName = _masterBizName;
    summary.collectorCount = _collectorManager ? _collectorManager->size() : 0;
    summary.scheduleGraphRunning = _scheduleGraph ? _scheduleGraph->running() : false;
    summary.scheduleGraphState =
        _scheduleGraph ? _scheduleGraph->getStateName() : "stopped";
    summary.scheduleGraphRestartCount =
        _scheduleGraph ? _scheduleGraph->getRestartCount() : 0;
    summary.scheduleSubmitCount =
        _masterScheduleR ? _masterScheduleR->getSubmitCount() : 0;
    summary.scheduleRequestPipeGeneration =
        _masterScheduleR ? _masterScheduleR->getRequestPipeGeneration() : 0;
    const auto schedulePendingRequestCount =
        _masterScheduleR ? _masterScheduleR->getPendingRequestCount() : 0;
    summary.schedulePendingRequestCount = schedulePendingRequestCount;
    summary.scheduleProcessingRequestCount =
        _masterScheduleR ? _masterScheduleR->getSchedulingRequestCount() : 0;
    summary.activeRequestCount =
        summary.schedulePendingRequestCount + summary.scheduleProcessingRequestCount;
    const auto fillQueueOverloadSummary = [this, &summary]() {
        if (!_masterScheduleR) {
            return;
        }
        const auto execution = _masterScheduleR->getExecutionUtilizationStats();
        summary.executionUtilizationControlEnabled = execution.enabled;
        summary.executionUtilizationAverage = execution.average;
        summary.executionUtilizationRawAverage = execution.rawAverage;
        summary.executionUtilizationRunning = execution.running;
        summary.executionUtilizationWaiting = execution.waiting;
        summary.executionUtilizationGlobalScaling = execution.globalScaling;
        summary.executionUtilizationIntegral = execution.integral;
        summary.executionUtilizationReportingNodes = execution.reportingNodes;
        const auto overloadStats = _masterScheduleR->getQueueOverloadStats();
        summary.queueOverloadEvaluatedRoundCount = overloadStats.evaluatedRoundCount;
        summary.queueOverloadEvaluatedRequestCount = overloadStats.evaluatedRequestCount;
        summary.queueOverloadRejectedRequestCount = overloadStats.rejectedRequestCount;
        summary.queueOverloadLatestAveragePredictedLatencyMs =
            overloadStats.latestAveragePredictedLatencyMs;
        summary.queueOverloadLatestDropProbability = overloadStats.latestDropProbability;
    };
    auto timeCompression = _configuredTimeCompression;
    if (_schedulerConfigR) {
        const auto schedulerConfig = _schedulerConfigR->get();
        summary.maxWaitScheduleCount = schedulerConfig.config.maxWaitScheduleCount;
        summary.maxWaitScheduleTimeMs = schedulerConfig.config.maxWaitScheduleTimeMs;
        summary.queueOverloadStartCycleMultiplier =
            schedulerConfig.config.queueOverloadStartCycleMultiplier;
        summary.queueOverloadFullCycleMultiplier =
            schedulerConfig.config.queueOverloadFullCycleMultiplier;
        summary.schedulerConfigGeneration = schedulerConfig.generation;
        summary.schedulerConfigHash = schedulerConfig.agentConfigHash;
        summary.schedulerConfigJson = SchedulerConfigR::toJson(schedulerConfig.config);
        timeCompression = schedulerConfig.config.agent.timeCompression;
    } else {
        summary.maxWaitScheduleCount = _maxWaitScheduleCount;
        summary.maxWaitScheduleTimeMs = _maxWaitScheduleTimeMs;
        summary.queueOverloadStartCycleMultiplier =
            _configuredQueueOverloadStartCycleMultiplier;
        summary.queueOverloadFullCycleMultiplier =
            _configuredQueueOverloadFullCycleMultiplier;
    }
    if (!_masterStateR) {
        fillQueueOverloadSummary();
        return summary.format();
    }
    auto scheduleStats = _masterStateR->getScheduleRoundStats();
    summary.scheduleRoundCount = scheduleStats.roundCount;
    summary.scheduleRoundRequestCount = scheduleStats.requestCount;
    summary.scheduleRoundRequestsMean =
        scheduleStats.roundCount > 0
            ? static_cast<double>(scheduleStats.requestCount) / scheduleStats.roundCount
            : 0.0;
    summary.scheduleRoundRequestsMin = scheduleStats.minRequestCount;
    summary.scheduleRoundRequestsMax = scheduleStats.maxRequestCount;
    for (const auto &[requestCount, roundCount] : scheduleStats.requestCountHistogram) {
        summary.scheduleRoundRequestCountHistogram.emplace(std::to_string(requestCount),
                                                           roundCount);
    }
    auto requestStats = _masterStateR->getAgentRequestStats();
    summary.agentRequestReceivedCount = requestStats.receivedCount;
    summary.agentRequestFinishedCount = requestStats.finishedCount;
    summary.agentWorkloadUpdateCount = requestStats.workloadUpdateCount;
    summary.agentRequestCacheHitTokens = requestStats.cacheHitTokens;
    summary.agentRequestCacheMissTokens = requestStats.cacheMissTokens;
    summary.agentRequestTotalLatencyUs = requestStats.totalLatencyUs;
    for (const auto &state : _masterStateR->getNodeStates()) {
        const auto dispatchStats = state->getDispatchStats();
        const auto reservationStats = state->getReservationStats();
        const auto info = state->getInfo();
        auto nodeSummary =
            makeNodeSummary(info, state->getCacheSyncStats(), dispatchStats.requestCount,
                            dispatchStats.batchCount, timeCompression);
        NaviSchedCachePolicy v6dPolicy;
        if (state->getCachePolicy(v6dPolicy, NaviSchedCacheKeyType::V6D) &&
            v6dPolicy.valid(NaviSchedCacheKeyType::V6D)) {
            nodeSummary.v6dCacheStats = makeCacheStateSummary(
                state->getCacheSyncStats(NaviSchedCacheKeyType::V6D));
        }
        nodeSummary.reservationBatchCount = reservationStats.batchCount;
        nodeSummary.executionUtilization = state->getExecutionUtilization();
        nodeSummary.executionBusyUtilization = state->getExecutionBusyUtilization();
        nodeSummary.reservationRequestCount = reservationStats.requestCount;
        summary.activeRequestCount += reservationStats.waitingRequestCount +
                                      reservationStats.dispatchedRequestCount;
        nodeSummary.reservationWaitingRequestCount = reservationStats.waitingRequestCount;
        nodeSummary.reservationDispatchedRequestCount =
            reservationStats.dispatchedRequestCount;
        nodeSummary.dispatchWaiterCount = reservationStats.dispatchWaiterCount;
        nodeSummary.liveDispatchWaiterCount = reservationStats.liveDispatchWaiterCount;
        summary.nodes.emplace_back(std::move(nodeSummary));
    }
    fillQueueOverloadSummary();
    return summary.format();
}

void NaviSchedMaster::reportMetrics() const
{
    NaviSchedRole::reportMetrics();
    reportMetric("master.collector_count",
                 _collectorManager ? _collectorManager->size() : 0);
    reportMetric("master.schedule_submit_total",
                 _masterScheduleR ? _masterScheduleR->getSubmitCount() : 0);
    reportMetric("master.schedule_graph_running",
                 _scheduleGraph && _scheduleGraph->running() ? 1 : 0);
    reportMetric("master.schedule_graph_restart_total",
                 _scheduleGraph ? _scheduleGraph->getRestartCount() : 0);
    reportMetric("master.schedule_request_pipe_generation",
                 _masterScheduleR ? _masterScheduleR->getRequestPipeGeneration() : 0);
    const auto schedulePendingRequestCount =
        _masterScheduleR ? _masterScheduleR->getPendingRequestCount() : 0;
    reportMetric("master.schedule_pending_requests", schedulePendingRequestCount);
    if (_masterScheduleR) {
        const auto backlog = _masterScheduleR->getBacklogStats();
        reportMetric("master.schedule_pipe_pending_requests", backlog.pipePending);
        reportMetric("master.schedule_fifo_pending_requests", backlog.kernelPending);
        reportMetric("master.schedule_dequeue_wait_ms", backlog.dequeueWaitMs);
        reportMetric("master.schedule_dequeue_max_wait_ms", backlog.dequeueMaxWaitMs);
        reportMetric("master.schedule_expired_request_total", backlog.expired);
        reportMetric("master.schedule_cancelled_request_total", backlog.cancelled);
        reportCounterQpsMetric("master.schedule_expired_request_qps", backlog.expired);
    }
    if (_masterLatencyModelR) {
        const auto stats = _masterLatencyModelR->getStats();
        reportMetric("master.latency_model_pool_fingerprint_count",
                     stats.fingerprintCount);
        reportMetric("master.latency_model_pool_candidate_count", stats.candidateCount);
        reportMetric("master.latency_model_pool_accepted_update_total",
                     stats.acceptedUpdateCount);
        reportMetric("master.latency_model_pool_rejected_update_total",
                     stats.rejectedUpdateCount);
        reportMetric("master.latency_model_pool_warmup_response_total",
                     stats.warmupResponseCount);
        reportMetric("master.latency_model_pool_warmup_notification_total",
                     stats.warmupNotificationCount);
        reportMetric("master.latency_model_pool_warmup_notification_failure_total",
                     stats.warmupNotificationFailureCount);
        reportMetric("master.latency_model_pool_warmup_subscriber_count",
                     stats.warmupSubscriberCount);
        reportMetric("master.latency_model_pool_persist_success_total",
                     stats.persistSuccessCount);
        reportMetric("master.latency_model_pool_persist_failure_total",
                     stats.persistFailureCount);
        reportMetric("master.latency_model_global_zero_point_live_node_count",
                     stats.globalZeroPointLiveNodeCount);
        reportMetric("master.latency_model_global_zero_point_group_count",
                     stats.globalZeroPointGroupCount);
        reportMetric("master.latency_model_global_zero_point_refresh_total",
                     stats.globalZeroPointRefreshCount);
        for (const auto parameterCount :
             {NonLinearPrefillModelR::PARAMETER_COUNT,
              NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT}) {
            const auto diagnostics = _masterLatencyModelR->getGlobalZeroPointDiagnostics(
                NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL,
                NonLinearPrefillModelR::BINARY_VERSION, parameterCount);
            const auto suffix = std::to_string(parameterCount);
            reportMetric("master.latency_model_global_zero_point_available_" + suffix,
                         diagnostics.available() ? 1 : 0);
            reportMetric(
                "master.latency_model_global_zero_point_invalid_reason_" + suffix,
                static_cast<int>(diagnostics.invalidReason));
            reportMetric(
                "master.latency_model_global_zero_point_group_live_node_count_" + suffix,
                diagnostics.groupLiveNodeCount);
            reportMetric(
                "master.latency_model_global_zero_point_group_trusted_node_count_" +
                    suffix,
                diagnostics.groupTrustedLiveNodeCount);
            reportMetric("master.latency_model_global_zero_point_query_formula_" + suffix,
                         NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL);
            reportMetric(
                "master.latency_model_global_zero_point_query_binary_version_" + suffix,
                NonLinearPrefillModelR::BINARY_VERSION);
            reportMetric(
                "master.latency_model_global_zero_point_query_parameter_count_" + suffix,
                parameterCount);
            if (!diagnostics.available()) {
                continue;
            }
            reportMetric("master.latency_model_global_zero_point_" + suffix,
                         diagnostics.data.globalZeroPoint);
            reportMetric("master.latency_model_global_zero_point_version_" + suffix,
                         diagnostics.data.version);
        }
    }
    const auto reportQueueOverloadMetrics = [this]() {
        if (!_masterScheduleR) {
            return;
        }
        const auto execution = _masterScheduleR->getExecutionUtilizationStats();
        reportMetric("master.execution_utilization_average_pct",
                     execution.average * 100.0);
        reportMetric("master.execution_utilization_raw_average_pct",
                     execution.rawAverage * 100.0);
        reportMetric("master.execution_utilization_running_requests", execution.running);
        reportMetric("master.execution_utilization_waiting_requests", execution.waiting);
        reportMetric("master.execution_utilization_global_scaling",
                     execution.globalScaling ? 1 : 0);
        reportMetric("master.execution_utilization_integral", execution.integral);
        reportMetric("master.execution_utilization_reporting_nodes",
                     execution.reportingNodes);
        reportMetric("master.execution_utilization_control_enabled",
                     execution.enabled ? 1 : 0);
        const auto overloadStats = _masterScheduleR->getQueueOverloadStats();
        reportMetric("master.queue_overload_average_predicted_latency_ms",
                     overloadStats.latestAveragePredictedLatencyMs);
        reportMetric("master.queue_overload_drop_probability",
                     overloadStats.latestDropProbability);
        reportMetric("master.queue_overload_active", overloadStats.active ? 1 : 0);
        reportMetric("master.queue_overload_evaluated_round_total",
                     overloadStats.evaluatedRoundCount);
        reportMetric("master.queue_overload_evaluated_request_total",
                     overloadStats.evaluatedRequestCount);
        reportMetric("master.queue_overload_rejected_request_total",
                     overloadStats.rejectedRequestCount);
        reportCounterQpsMetric("master.queue_overload_evaluated_request_qps",
                               overloadStats.evaluatedRequestCount);
        reportCounterQpsMetric("master.queue_overload_rejected_request_qps",
                               overloadStats.rejectedRequestCount);
    };
    if (_masterScheduleR) {
        const auto coldStartStats = _masterScheduleR->getColdStartRoutingStats();
        reportMetric("master.cold_start_latest_target_inflight",
                     coldStartStats.latestTargetInflight);
        reportMetric("master.cold_start_latest_trusted_node_count",
                     coldStartStats.latestTrustedNodeCount);
        reportMetric("master.cold_start_latest_untrusted_node_count",
                     coldStartStats.latestUntrustedNodeCount);
        reportMetric("master.cold_start_latest_untrusted_capacity",
                     coldStartStats.latestUntrustedCapacity);
        reportMetric("master.cold_start_latest_ramping_node_count",
                     coldStartStats.latestRampingNodeCount);
        reportMetric("master.cold_start_latest_unsafe_waiting_marginal_node_count",
                     coldStartStats.latestUnsafeWaitingMarginalNodeCount);
        reportMetric("master.cold_start_unsafe_waiting_marginal_node_observation_total",
                     coldStartStats.unsafeWaitingMarginalNodeObservationCount);
        reportCounterQpsMetric(
            "master.cold_start_unsafe_waiting_marginal_node_observation_qps",
            coldStartStats.unsafeWaitingMarginalNodeObservationCount);
        reportMetric("master.cold_start_latest_minimum_ramp_progress",
                     coldStartStats.latestMinimumRampProgress);
        reportMetric("master.cold_start_latest_minimum_ramped_target_inflight",
                     coldStartStats.latestMinimumRampedTargetInflight);
        reportMetric("master.cold_start_evaluated_round_total",
                     coldStartStats.evaluatedRoundCount);
        reportMetric("master.cold_start_trusted_request_total",
                     coldStartStats.trustedRequestCount);
        reportMetric("master.cold_start_untrusted_request_total",
                     coldStartStats.untrustedRequestCount);
        reportMetric("master.cold_start_rejected_request_total",
                     coldStartStats.rejectedRequestCount);
        reportCounterQpsMetric("master.cold_start_trusted_request_qps",
                               coldStartStats.trustedRequestCount);
        reportCounterQpsMetric("master.cold_start_untrusted_request_qps",
                               coldStartStats.untrustedRequestCount);
        reportCounterQpsMetric("master.cold_start_rejected_request_qps",
                               coldStartStats.rejectedRequestCount);
    }
    if (!_masterStateR) {
        reportQueueOverloadMetrics();
        return;
    }

    const auto scheduleStats = _masterStateR->getScheduleRoundStats();
    const auto clientRequestCount = _masterStateR->getClientRequestCount();
    reportMetric("master.client_request_received_total", clientRequestCount);
    reportMetric("master.schedule_round_total", scheduleStats.roundCount);
    reportMetric("master.schedule_round_request_total", scheduleStats.requestCount);
    reportCounterQpsMetric("master.schedule_round_qps", scheduleStats.roundCount);
    reportCounterQpsMetric("master.schedule_request_qps", scheduleStats.requestCount);
    const auto requestStats = _masterStateR->getAgentRequestStats();
    const auto cacheTokenReportTimeUs = autil::TimeUtility::monotonicTimeUs();
    const auto cacheHitTokens =
        reportCounterQpsMetricAt("master.kv_cache_hit_tokens_qps",
                                 requestStats.cacheHitTokens, cacheTokenReportTimeUs);
    const auto cacheMissTokens =
        reportCounterQpsMetricAt("master.kv_cache_miss_tokens_qps",
                                 requestStats.cacheMissTokens, cacheTokenReportTimeUs);
    const auto cachePromptTokens = static_cast<double>(cacheHitTokens) + cacheMissTokens;
    if (cachePromptTokens > 0.0) {
        reportSampleMetric("master.kv_cache_hit_rate_pct",
                           100.0 * cacheHitTokens / cachePromptTokens);
    }
    reportMetric("master.kv_cache_hit_tokens_total", requestStats.cacheHitTokens);
    reportMetric("master.kv_cache_miss_tokens_total", requestStats.cacheMissTokens);
    reportMetric("master.agent_request_received_total", requestStats.receivedCount);
    reportMetric("master.agent_request_finished_total", requestStats.finishedCount);
    reportCounterQpsMetric("master.agent_request_received_qps",
                           requestStats.receivedCount);
    reportCounterQpsMetric("master.agent_request_finished_qps",
                           requestStats.finishedCount);
    reportCounterQpsMetric("master.agent_batch_finished_qps",
                           requestStats.batchFinishedCount);
    reportCounterQpsMetric("master.agent_request_abnormal_finished_qps",
                           requestStats.abnormalFinishedCount);
    reportMetric("master.agent_report_received_total", requestStats.reportCount);
    reportMetric("master.agent_handshake_received_total", requestStats.handshakeCount);
    reportMetric("master.agent_stopping_received_total", requestStats.stoppingCount);
    reportMetric("master.agent_execution_load_received_total",
                 requestStats.executionLoadCount);
    reportMetric("master.agent_batch_finished_received_total",
                 requestStats.batchFinishedCount);
    reportMetric("master.agent_request_abnormal_finished_total",
                 requestStats.abnormalFinishedCount);
    reportMetric("master.agent_cache_update_received_total",
                 requestStats.cacheUpdateCount);
    reportMetric("master.agent_latency_model_update_received_total",
                 requestStats.latencyModelUpdateCount);
    reportMetric("master.agent_workload_update_received_total",
                 requestStats.workloadUpdateCount);
    reportCounterQpsMetric("master.agent_workload_update_received_qps",
                           requestStats.workloadUpdateCount);

    int64_t onlineNodes = 0;
    int64_t trustedLatencyModelNodes = 0;
    int64_t untrustedLatencyModelNodes = 0;
    int64_t inflightRequests = 0;
    int64_t queuedBatchRequests = 0;
    int64_t runningRequests = 0;
    double predictedQueueLatencyMs = 0.0;
    uint64_t validationSamples = 0;
    uint64_t unmatchedBatches = 0;
    uint64_t zeroQueueSamples = 0;
    uint64_t positiveQueueSamples = 0;
    double predictedCacheHitTokens = 0.0;
    double actualCacheHitTokens = 0.0;
    double predictedQueueMs = 0.0;
    double actualQueueMs = 0.0;
    double queueRawPredictedMs = 0.0;
    double agentPredictedQueueMs = 0.0;
    double queueFeatureBatchCount = 0.0;
    double zeroQueueActualMs = 0.0;
    double positivePredictedQueueMs = 0.0;
    double positiveActualQueueMs = 0.0;
    double masterPredictedExecutionMs = 0.0;
    double agentPredictedExecutionMs = 0.0;
    double actualExecutionMs = 0.0;
    double predictedTotalMs = 0.0;
    double actualTotalMs = 0.0;
    double batchRequestCount = 0.0;
    int64_t reportLagUs = 0;
    int64_t maxReportLagUs = 0;
    uint64_t reportLagCount = 0;
    uint64_t cacheRevisionGapCount = 0;
    uint64_t cacheStateMismatchCount = 0;
    uint64_t cacheDivergedNodeCount = 0;
    uint64_t rtpCacheMirrorSampleCount = 0;
    uint64_t rtpCacheMirrorMismatchCount = 0;
    uint64_t rtpCacheMirrorDivergedNodeCount = 0;
    std::map<std::string, uint64_t> v6dCacheTotals;
    uint64_t v6dCacheNodeCount = 0;
    uint64_t pendingCacheBatchCount = 0;
    uint64_t pendingCacheBlockCount = 0;
    uint64_t pendingCacheRevisionGapCount = 0;
    uint64_t pendingCacheStateMismatchCount = 0;
    uint64_t pendingCacheResyncTriggerCount = 0;
    uint64_t pendingCacheNotReadyNodeCount = 0;
    uint64_t reservationBatchCount = 0;
    uint64_t reservationRequestCount = 0;
    uint64_t reservationWaitingRequestCount = 0;
    uint64_t reservationDispatchedRequestCount = 0;
    uint64_t dispatchWaiterCount = 0;
    uint64_t liveDispatchWaiterCount = 0;
    uint64_t workloadStaleUpdateCount = 0;
    uint64_t workloadReadyNodeCount = 0;
    uint64_t pdSeparatedNodeCount = 0;
    uint64_t peerCacheEnabledNodeCount = 0;
    int64_t workloadBatchCount = 0;
    int64_t workloadRequestCount = 0;
    int64_t workloadRemainingPrefillRequestCount = 0;
    int64_t workloadRunningRequestCount = 0;
    double workloadRunningElapsedTimeMs = 0.0;
    int64_t workloadRunningDecodeRequestCount = 0;
    int64_t workloadRunningDecodeGeneratedTokens = 0;
    int64_t workloadRunningDecodeRemainingTokenBudget = 0;
    int64_t workloadWaitingRequestCount = 0;
    int64_t workloadRemainingPrefillContextTokens = 0;
    int64_t workloadRemainingPrefillTokens = 0;
    double workloadRemainingPrefillTokenSquareSum = 0.0;
    double workloadContextRemainingPrefillProductSum = 0.0;
    uint64_t workloadCapacitySplitNodeCount = 0;
    int64_t workloadMaxRunningRequestCount = 0;
    int64_t workloadRunningCapacitySlotCount = 0;
    int64_t workloadRunningCapacityPrefillRequestCount = 0;
    int64_t workloadRunningCapacityPrefillContextTokens = 0;
    int64_t workloadRunningCapacityPrefillTokens = 0;
    double workloadRunningCapacityPrefillTokenSquareSum = 0.0;
    double workloadRunningCapacityPrefillProductSum = 0.0;
    int64_t workloadWaitingCapacityPrefillRequestCount = 0;
    int64_t workloadWaitingCapacityPrefillContextTokens = 0;
    int64_t workloadWaitingCapacityPrefillTokens = 0;
    double workloadWaitingCapacityPrefillTokenSquareSum = 0.0;
    double workloadWaitingCapacityPrefillProductSum = 0.0;
    int64_t workloadRemainingTotalContextTokens = 0;
    int64_t workloadRemainingTotalTokens = 0;
    double workloadRemainingTotalTokenSquareSum = 0.0;
    double workloadContextRemainingTotalProductSum = 0.0;
    int64_t workloadTotalPromptTokens = 0;
    int64_t workloadInFlightScheduledTokens = 0;
    int64_t workloadAgeUs = 0;
    int64_t workloadMinAgeUs = std::numeric_limits<int64_t>::max();
    int64_t workloadMaxAgeUs = std::numeric_limits<int64_t>::min();
    int64_t workloadTransportLagUs = 0;
    int64_t workloadMinTransportLagUs = std::numeric_limits<int64_t>::max();
    int64_t workloadMaxTransportLagUs = std::numeric_limits<int64_t>::min();
    const auto reportTimeUs = autil::TimeUtility::currentTime();
    const auto timeCompression =
        _schedulerConfigR ? _schedulerConfigR->get().config.agent.timeCompression
                          : _configuredTimeCompression;
    ValidationWindowTotals validationWindow;
    std::unordered_set<std::string> activeAgentIds;
    for (const auto &state : _masterStateR->getNodeStates()) {
        const auto info = state->getInfo();
        const auto utilization = state->getExecutionUtilization();
        if (info.online && utilization >= 0.0) {
            kmonitor::MetricsTags tags("agent_id", info.agentId);
            tags.AddTag("host_ip", info.serverInfo.ip);
            reportMetric("master.agent_execution_utilization_pct", utilization * 100.0,
                         tags);
        }
        if (info.online &&
            LatencyResidualModel::validateParameters(info.latencyResidualParameters)) {
            const auto residualDecay = LatencyResidualModel::decayFactor(
                info.latencyResidualLastUpdateTimeUs, reportTimeUs,
                state->getResidualRuntimeConfig());
            kmonitor::MetricsTags residualNodeTags("agent_id", info.agentId);
            residualNodeTags.AddTag("host_ip", info.serverInfo.ip);
            const auto residualAgeMs =
                info.latencyResidualLastUpdateTimeUs > 0
                    ? std::max<int64_t>(
                          0, reportTimeUs - info.latencyResidualLastUpdateTimeUs) /
                          1000.0
                    : 0.0;
            reportMetric("master.latency_residual_age_ms", residualAgeMs,
                         residualNodeTags);
            reportMetric("master.latency_residual_decay_factor", residualDecay,
                         residualNodeTags);
            reportMetric("master.latency_residual_model_revision",
                         info.latencyModelRevision, residualNodeTags);
            for (size_t index = 0; index < info.latencyResidualParameters.size();
                 ++index) {
                auto residualBucketTags = residualNodeTags;
                residualBucketTags.AddTag(
                    "residual_bucket",
                    std::to_string(LatencyResidualModel::MIN_BUCKET + index));
                reportMetric("master.latency_residual_parameter_ms",
                             info.latencyResidualParameters[index], residualBucketTags);
                reportMetric("master.latency_residual_effective_parameter_ms",
                             info.latencyResidualParameters[index] * residualDecay,
                             residualBucketTags);
            }
        }
        if (info.online) {
            const auto validLatencyModel =
                NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL == info.latencyFormula &&
                NonLinearPrefillModelR::supportsBinaryVersion(
                    info.latencyBinaryVersion) &&
                info.latencyModelRevision > 0 &&
                NonLinearPrefillModelR::validateParametersForVersion(
                    info.latencyParameters, info.latencyBinaryVersion);
            if (validLatencyModel && info.latencyModelTrusted) {
                ++trustedLatencyModelNodes;
            } else {
                ++untrustedLatencyModelNodes;
            }
        }
        NaviSchedCachePolicy v6dPolicy;
        if (info.online && state->getCachePolicy(v6dPolicy, NaviSchedCacheKeyType::V6D) &&
            v6dPolicy.valid(NaviSchedCacheKeyType::V6D)) {
            ++v6dCacheNodeCount;
            for (const auto &[name, value] : makeCacheStateSummary(
                     state->getCacheSyncStats(NaviSchedCacheKeyType::V6D))) {
                v6dCacheTotals[name] += value;
            }
        }
        const auto cacheStats = state->getCacheSyncStats();
        cacheRevisionGapCount += cacheStats.revisionGapCount;
        cacheStateMismatchCount += cacheStats.stateMismatchCount;
        rtpCacheMirrorSampleCount += cacheStats.rtpMirrorSampleCount;
        rtpCacheMirrorMismatchCount += cacheStats.rtpMirrorMismatchCount;
        pendingCacheBatchCount += cacheStats.pendingCacheBatchCount;
        pendingCacheBlockCount += cacheStats.pendingCacheBlockCount;
        pendingCacheRevisionGapCount += cacheStats.pendingCacheRevisionGapCount;
        pendingCacheStateMismatchCount += cacheStats.pendingCacheStateMismatchCount;
        pendingCacheResyncTriggerCount += cacheStats.pendingCacheResyncTriggerCount;
        pendingCacheNotReadyNodeCount +=
            info.online && !cacheStats.pendingCacheSnapshotReady ? 1 : 0;
        const auto reservationStats = state->getReservationStats();
        reservationBatchCount += reservationStats.batchCount;
        reservationRequestCount += reservationStats.requestCount;
        reservationWaitingRequestCount += reservationStats.waitingRequestCount;
        reservationDispatchedRequestCount += reservationStats.dispatchedRequestCount;
        dispatchWaiterCount += reservationStats.dispatchWaiterCount;
        liveDispatchWaiterCount += reservationStats.liveDispatchWaiterCount;
        workloadStaleUpdateCount += info.workloadStaleUpdateCount;
        pdSeparatedNodeCount += info.pdSeparated ? 1 : 0;
        peerCacheEnabledNodeCount += info.online && info.peerCacheEnabled ? 1 : 0;
        if (info.workload.valid()) {
            ++workloadReadyNodeCount;
            workloadBatchCount += info.workload.batchCount;
            workloadRequestCount += info.workload.requestCount;
            workloadRemainingPrefillRequestCount +=
                info.workload.remainingPrefillRequestCount;
            workloadRunningRequestCount += info.workload.runningRequestCount;
            workloadRunningElapsedTimeMs += info.workload.runningElapsedTimeMs;
            workloadRunningDecodeRequestCount += info.workload.runningDecodeRequestCount;
            workloadRunningDecodeGeneratedTokens +=
                info.workload.runningDecodeGeneratedTokenCount;
            workloadRunningDecodeRemainingTokenBudget +=
                info.workload.runningDecodeRemainingTokenBudget;
            if (info.workloadReportReceiveTimeUs > 0 &&
                info.workload.runningCapacityPrefillRequestCount > 0) {
                workloadRunningElapsedTimeMs +=
                    std::max<int64_t>(0,
                                      reportTimeUs - info.workloadReportReceiveTimeUs) /
                    1000.0 * info.workload.runningCapacityPrefillRequestCount *
                    timeCompression;
            }
            workloadWaitingRequestCount += info.workload.waitingRequestCount;
            workloadRemainingPrefillContextTokens +=
                info.workload.remainingPrefillContextTokenCount;
            workloadRemainingPrefillTokens += info.workload.remainingPrefillTokenCount;
            workloadRemainingPrefillTokenSquareSum +=
                info.workload.remainingPrefillTokenSquareSum;
            workloadContextRemainingPrefillProductSum +=
                info.workload.contextRemainingPrefillTokenProductSum;
            workloadCapacitySplitNodeCount += info.workload.prefillCapacitySplit ? 1 : 0;
            workloadMaxRunningRequestCount += info.workload.maxRunningRequestCount;
            workloadRunningCapacitySlotCount += info.workload.runningCapacitySlotCount;
            workloadRunningCapacityPrefillRequestCount +=
                info.workload.runningCapacityPrefillRequestCount;
            workloadRunningCapacityPrefillContextTokens +=
                info.workload.runningCapacityPrefillContextTokenCount;
            workloadRunningCapacityPrefillTokens +=
                info.workload.runningCapacityPrefillTokenCount;
            workloadRunningCapacityPrefillTokenSquareSum +=
                info.workload.runningCapacityPrefillTokenSquareSum;
            workloadRunningCapacityPrefillProductSum +=
                info.workload.runningCapacityContextRemainingPrefillTokenProductSum;
            workloadWaitingCapacityPrefillRequestCount +=
                info.workload.waitingCapacityPrefillRequestCount;
            workloadWaitingCapacityPrefillContextTokens +=
                info.workload.waitingCapacityPrefillContextTokenCount;
            workloadWaitingCapacityPrefillTokens +=
                info.workload.waitingCapacityPrefillTokenCount;
            workloadWaitingCapacityPrefillTokenSquareSum +=
                info.workload.waitingCapacityPrefillTokenSquareSum;
            workloadWaitingCapacityPrefillProductSum +=
                info.workload.waitingCapacityContextRemainingPrefillTokenProductSum;
            workloadRemainingTotalContextTokens +=
                info.workload.remainingTotalContextTokenCount;
            workloadRemainingTotalTokens += info.workload.remainingTotalTokenCount;
            workloadRemainingTotalTokenSquareSum +=
                info.workload.remainingTotalTokenSquareSum;
            workloadContextRemainingTotalProductSum +=
                info.workload.contextRemainingTotalTokenProductSum;
            workloadTotalPromptTokens += info.workload.totalPromptTokenCount;
            workloadInFlightScheduledTokens += info.workload.inFlightScheduledTokenCount;
            const auto ageUs = reportTimeUs - info.workload.eventTimeUs;
            workloadAgeUs += ageUs;
            workloadMinAgeUs = std::min(workloadMinAgeUs, ageUs);
            workloadMaxAgeUs = std::max(workloadMaxAgeUs, ageUs);
            const auto transportLagUs =
                info.workloadReportReceiveTimeUs - info.workload.eventTimeUs;
            workloadTransportLagUs += transportLagUs;
            workloadMinTransportLagUs =
                std::min(workloadMinTransportLagUs, transportLagUs);
            workloadMaxTransportLagUs =
                std::max(workloadMaxTransportLagUs, transportLagUs);
        }
        rtpCacheMirrorDivergedNodeCount +=
            cacheStats.rtpCachedBlockCount >= 0 &&
                    cacheStats.rtpCachedBlockCount != cacheStats.agentCachedBlockCount
                ? 1
                : 0;
        cacheDivergedNodeCount +=
            cacheStats.agentCachedBlockCount >= 0 &&
                    (cacheStats.agentCachedBlockCount !=
                         static_cast<int64_t>(cacheStats.masterCachedBlockCount) ||
                     cacheStats.agentHashChecksum != cacheStats.masterHashChecksum ||
                     (cacheStats.agentFullReadyBlockCount >= 0 &&
                      (cacheStats.agentFullReadyBlockCount !=
                           static_cast<int64_t>(cacheStats.masterFullReadyBlockCount) ||
                       cacheStats.agentFullReadyHashChecksum !=
                           cacheStats.masterFullReadyHashChecksum)) ||
                     (cacheStats.agentTailReadyBlockCount >= 0 &&
                      (cacheStats.agentTailReadyBlockCount !=
                           static_cast<int64_t>(cacheStats.masterTailReadyBlockCount) ||
                       cacheStats.agentTailReadyHashChecksum !=
                           cacheStats.masterTailReadyHashChecksum)))
                ? 1
                : 0;
        activeAgentIds.emplace(info.agentId);
        onlineNodes += info.online ? 1 : 0;
        inflightRequests += info.inflightRequests;
        queuedBatchRequests += info.queuedBatchRequests;
        runningRequests += info.runningRequests;
        predictedQueueLatencyMs += info.queueTotalPredictedLatencyMs;
        const auto &validation = info.latencyValidation;
        auto [previousIt, inserted] =
            _lastValidationStats.emplace(info.agentId, validation);
        if (!inserted) {
            validationWindow.add(validation, previousIt->second);
            previousIt->second = validation;
        }
        const auto samples = validation.sampleCount;
        validationSamples += samples;
        unmatchedBatches += validation.unmatchedBatchCount;
        predictedCacheHitTokens += validation.predictedCacheHitTokensMean * samples;
        actualCacheHitTokens += validation.actualCacheHitTokensMean * samples;
        predictedQueueMs += validation.masterPredictedQueueMeanMs * samples;
        actualQueueMs += validation.actualQueueMeanMs * samples;
        queueRawPredictedMs += validation.queueRawPredictedMeanMs * samples;
        agentPredictedQueueMs += validation.agentPredictedQueueMeanMs * samples;
        queueFeatureBatchCount += validation.queueFeatureBatchCountMean * samples;
        batchRequestCount += validation.batchRequestCountMean * samples;
        masterPredictedExecutionMs += validation.masterPredictedExecutionMeanMs * samples;
        agentPredictedExecutionMs += validation.agentPredictedExecutionMeanMs * samples;
        actualExecutionMs += validation.actualExecutionMeanMs * samples;
        predictedTotalMs += validation.masterPredictedTotalMeanMs * samples;
        actualTotalMs += validation.actualTotalMeanMs * samples;
        if (info.lastEventTimeUs > 0 && info.lastReceiveTimeUs >= info.lastEventTimeUs) {
            const auto lagUs = info.lastReceiveTimeUs - info.lastEventTimeUs;
            reportLagUs += lagUs;
            maxReportLagUs = std::max(maxReportLagUs, lagUs);
            ++reportLagCount;
        }

        const auto zeroSamples = validation.zeroPredictedQueueSampleCount;
        zeroQueueSamples += zeroSamples;
        zeroQueueActualMs += validation.zeroPredictedQueueActualMeanMs * zeroSamples;
        const auto positiveSamples = validation.positivePredictedQueueSampleCount;
        positiveQueueSamples += positiveSamples;
        positivePredictedQueueMs +=
            validation.positivePredictedQueueMeanMs * positiveSamples;
        positiveActualQueueMs +=
            validation.positivePredictedQueueActualMeanMs * positiveSamples;
    }
    for (auto it = _lastValidationStats.begin(); it != _lastValidationStats.end();) {
        if (activeAgentIds.count(it->first) == 0) {
            it = _lastValidationStats.erase(it);
        } else {
            ++it;
        }
    }
    const bool hasValidationWindowData =
        validationWindow.sampleCount > 0 || validationWindow.unmatchedBatchCount > 0;
    reportQueueOverloadMetrics();
    if (v6dCacheNodeCount > 0) {
        reportMetric("master.v6d_cache_node_count", v6dCacheNodeCount);
        for (const auto &[name, value] : v6dCacheTotals) {
            reportMetric("master.v6d_" + name, value);
        }
    }
    reportMetric("master.online_node_count", onlineNodes);
    reportMetric("master.latency_model_trusted_node_count", trustedLatencyModelNodes);
    reportMetric("master.latency_model_untrusted_node_count", untrustedLatencyModelNodes);
    reportMetric("master.requests_inflight", inflightRequests);
    reportMetric("master.requests_queued_for_execution", queuedBatchRequests);
    reportMetric("master.requests_running", runningRequests);
    reportMetric("master.predicted_queue_latency_ms", predictedQueueLatencyMs);
    reportMetric("master.cache_sync_diverged_node_count", cacheDivergedNodeCount);
    reportMetric("master.cache_sync_revision_gap_total", cacheRevisionGapCount);
    reportMetric("master.cache_sync_state_mismatch_total", cacheStateMismatchCount);
    reportMetric("master.rtp_cache_mirror_mismatch_total", rtpCacheMirrorMismatchCount);
    reportMetric("master.rtp_cache_mirror_sample_total", rtpCacheMirrorSampleCount);
    reportMetric(
        "master.rtp_cache_mirror_mismatch_ratio",
        rtpCacheMirrorSampleCount > 0
            ? static_cast<double>(rtpCacheMirrorMismatchCount) / rtpCacheMirrorSampleCount
            : 0.0);
    reportMetric("master.rtp_cache_mirror_diverged_node_count",
                 rtpCacheMirrorDivergedNodeCount);
    reportMetric("master.pending_cache_batch_count", pendingCacheBatchCount);
    reportMetric("master.pending_cache_unique_block_count", pendingCacheBlockCount);
    reportMetric("master.pending_cache_sync_revision_gap_total",
                 pendingCacheRevisionGapCount);
    reportMetric("master.pending_cache_sync_state_mismatch_total",
                 pendingCacheStateMismatchCount);
    reportMetric("master.pending_cache_sync_resync_trigger_total",
                 pendingCacheResyncTriggerCount);
    reportMetric("master.pending_cache_sync_not_ready_node_count",
                 pendingCacheNotReadyNodeCount);
    reportMetric("master.reservation_batch_count", reservationBatchCount);
    reportMetric("master.reservation_request_count", reservationRequestCount);
    const auto schedulingRequestCount =
        _masterScheduleR ? _masterScheduleR->getSchedulingRequestCount() : 0;
    reportMetric("master.schedule_processing_requests", schedulingRequestCount);
    const auto reservationActiveRequestCount =
        reservationWaitingRequestCount + reservationDispatchedRequestCount;
    reportMetric("master.reservation_active_request_count",
                 reservationActiveRequestCount);
    reportMetric("master.requests_active", schedulePendingRequestCount +
                                               schedulingRequestCount +
                                               reservationActiveRequestCount);
    reportMetric("master.reservation_waiting_request_count",
                 reservationWaitingRequestCount);
    reportMetric("master.reservation_dispatched_request_count",
                 reservationDispatchedRequestCount);
    reportMetric("master.dispatch_waiter_count", dispatchWaiterCount);
    reportMetric("master.live_dispatch_waiter_count", liveDispatchWaiterCount);
    reportMetric("master.pd_separated_node_count", pdSeparatedNodeCount);
    reportMetric("master.peer_cache_enabled_node_count", peerCacheEnabledNodeCount);
    reportMetric("master.workload_ready_node_count", workloadReadyNodeCount);
    reportMetric("master.workload_stale_update_total", workloadStaleUpdateCount);
    reportMetric("master.workload_batch_count", workloadBatchCount);
    reportMetric("master.workload_request_count", workloadRequestCount);
    reportMetric("master.workload_remaining_prefill_request_count",
                 workloadRemainingPrefillRequestCount);
    reportMetric("master.workload_running_request_count", workloadRunningRequestCount);
    reportMetric("master.workload_running_elapsed_time_ms", workloadRunningElapsedTimeMs);
    reportMetric("master.workload_running_decode_request_count",
                 workloadRunningDecodeRequestCount);
    reportMetric("master.workload_running_decode_generated_tokens",
                 workloadRunningDecodeGeneratedTokens);
    reportMetric("master.workload_running_decode_remaining_token_budget",
                 workloadRunningDecodeRemainingTokenBudget);
    reportMetric("master.workload_waiting_request_count", workloadWaitingRequestCount);
    reportMetric("master.workload_remaining_prefill_context_tokens",
                 workloadRemainingPrefillContextTokens);
    reportMetric("master.workload_remaining_prefill_tokens",
                 workloadRemainingPrefillTokens);
    reportMetric("master.workload_remaining_prefill_token_square_sum",
                 workloadRemainingPrefillTokenSquareSum);
    reportMetric("master.workload_context_remaining_prefill_product_sum",
                 workloadContextRemainingPrefillProductSum);
    reportMetric("master.workload_capacity_split_node_count",
                 workloadCapacitySplitNodeCount);
    reportMetric("master.workload_max_running_request_count_sum",
                 workloadMaxRunningRequestCount);
    reportMetric("master.workload_running_capacity_slot_count",
                 workloadRunningCapacitySlotCount);
    reportMetric("master.workload_running_capacity_prefill_request_count",
                 workloadRunningCapacityPrefillRequestCount);
    reportMetric("master.workload_running_capacity_prefill_context_tokens",
                 workloadRunningCapacityPrefillContextTokens);
    reportMetric("master.workload_running_capacity_prefill_tokens",
                 workloadRunningCapacityPrefillTokens);
    reportMetric("master.workload_running_capacity_prefill_token_square_sum",
                 workloadRunningCapacityPrefillTokenSquareSum);
    reportMetric("master.workload_running_capacity_prefill_product_sum",
                 workloadRunningCapacityPrefillProductSum);
    reportMetric("master.workload_waiting_capacity_prefill_request_count",
                 workloadWaitingCapacityPrefillRequestCount);
    reportMetric("master.workload_waiting_capacity_prefill_context_tokens",
                 workloadWaitingCapacityPrefillContextTokens);
    reportMetric("master.workload_waiting_capacity_prefill_tokens",
                 workloadWaitingCapacityPrefillTokens);
    reportMetric("master.workload_waiting_capacity_prefill_token_square_sum",
                 workloadWaitingCapacityPrefillTokenSquareSum);
    reportMetric("master.workload_waiting_capacity_prefill_product_sum",
                 workloadWaitingCapacityPrefillProductSum);
    reportMetric("master.workload_remaining_total_context_tokens",
                 workloadRemainingTotalContextTokens);
    reportMetric("master.workload_remaining_total_tokens", workloadRemainingTotalTokens);
    reportMetric("master.workload_remaining_total_token_square_sum",
                 workloadRemainingTotalTokenSquareSum);
    reportMetric("master.workload_context_remaining_total_product_sum",
                 workloadContextRemainingTotalProductSum);
    reportMetric("master.workload_total_prompt_tokens", workloadTotalPromptTokens);
    reportMetric("master.workload_inflight_scheduled_tokens",
                 workloadInFlightScheduledTokens);
    reportMetric("master.workload_age_mean_ms",
                 workloadReadyNodeCount > 0
                     ? workloadAgeUs / 1000.0 / workloadReadyNodeCount
                     : 0.0);
    reportMetric("master.workload_age_min_ms",
                 workloadReadyNodeCount > 0 ? workloadMinAgeUs / 1000.0 : 0.0);
    reportMetric("master.workload_age_max_ms",
                 workloadReadyNodeCount > 0 ? workloadMaxAgeUs / 1000.0 : 0.0);
    reportMetric("master.workload_transport_lag_mean_ms",
                 workloadReadyNodeCount > 0
                     ? workloadTransportLagUs / 1000.0 / workloadReadyNodeCount
                     : 0.0);
    reportMetric("master.workload_transport_lag_min_ms",
                 workloadReadyNodeCount > 0 ? workloadMinTransportLagUs / 1000.0 : 0.0);
    reportMetric("master.workload_transport_lag_max_ms",
                 workloadReadyNodeCount > 0 ? workloadMaxTransportLagUs / 1000.0 : 0.0);
    const auto reportValidationMetric = [this, hasValidationWindowData](
                                            const std::string &name, double value) {
        if (hasValidationWindowData) {
            reportSampleMetric(name, value);
        }
    };
    reportValidationMetric("master.validation_samples", validationSamples);
    reportValidationMetric("master.validation_unmatched_batches", unmatchedBatches);
    const auto validationCount = static_cast<double>(validationSamples);
    const auto validationMean = [validationCount](double sum) {
        return validationCount > 0.0 ? sum / validationCount : 0.0;
    };
    reportValidationMetric("master.validation_cache_hit_tokens_predicted_mean",
                           validationMean(predictedCacheHitTokens));
    reportValidationMetric("master.validation_cache_hit_tokens_actual_mean",
                           validationMean(actualCacheHitTokens));
    reportValidationMetric("master.validation_queue_predicted_mean_ms",
                           validationMean(predictedQueueMs));
    reportValidationMetric("master.validation_queue_actual_mean_ms",
                           validationMean(actualQueueMs));
    reportValidationMetric("master.queue_model_raw_prediction_mean_ms",
                           validationMean(queueRawPredictedMs));
    reportValidationMetric("master.queue_model_prediction_mean_ms",
                           validationMean(agentPredictedQueueMs));
    reportValidationMetric("master.queue_model_preceding_batch_count_mean",
                           validationMean(queueFeatureBatchCount));
    reportValidationMetric("master.validation_batch_request_count_mean",
                           validationMean(batchRequestCount));
    reportValidationMetric("master.validation_execution_predicted_mean_ms",
                           validationMean(masterPredictedExecutionMs));
    reportValidationMetric("master.validation_execution_agent_predicted_mean_ms",
                           validationMean(agentPredictedExecutionMs));
    reportValidationMetric("master.validation_execution_actual_mean_ms",
                           validationMean(actualExecutionMs));
    reportValidationMetric("master.validation_total_predicted_mean_ms",
                           validationMean(predictedTotalMs));
    reportValidationMetric("master.validation_total_actual_mean_ms",
                           validationMean(actualTotalMs));
    reportMetric("master.agent_report_lag_mean_ms",
                 reportLagCount > 0
                     ? reportLagUs / 1000.0 / static_cast<double>(reportLagCount)
                     : 0.0);
    reportMetric("master.agent_report_lag_max_ms", maxReportLagUs / 1000.0);
    reportValidationMetric("master.validation_queue_zero_samples", zeroQueueSamples);
    const auto zeroQueueCount = static_cast<double>(zeroQueueSamples);
    reportValidationMetric(
        "master.validation_queue_zero_actual_mean_ms",
        zeroQueueCount > 0.0 ? zeroQueueActualMs / zeroQueueCount : 0.0);
    reportValidationMetric("master.validation_queue_positive_samples",
                           positiveQueueSamples);
    const auto positiveQueueCount = static_cast<double>(positiveQueueSamples);
    reportValidationMetric(
        "master.validation_queue_positive_predicted_mean_ms",
        positiveQueueCount > 0.0 ? positivePredictedQueueMs / positiveQueueCount : 0.0);
    reportValidationMetric(
        "master.validation_queue_positive_actual_mean_ms",
        positiveQueueCount > 0.0 ? positiveActualQueueMs / positiveQueueCount : 0.0);

    if (!hasValidationWindowData) {
        return;
    }
    const auto reportWindowMetric = [this](const std::string &name, double value) {
        reportSampleMetric(name, value);
    };
    const auto windowCount = static_cast<double>(validationWindow.sampleCount);
    const auto windowMean = [windowCount](double sum) {
        return windowCount > 0.0 ? sum / windowCount : 0.0;
    };
    reportWindowMetric("master.validation_window_samples", validationWindow.sampleCount);
    reportWindowMetric("master.validation_window_unmatched_batches",
                       validationWindow.unmatchedBatchCount);
    reportWindowMetric("master.validation_window_batch_request_count_mean",
                       windowMean(validationWindow.batchRequestCount));
    reportWindowMetric("master.validation_window_cache_hit_tokens_predicted_mean",
                       windowMean(validationWindow.predictedCacheHitTokens));
    reportWindowMetric("master.validation_window_cache_hit_tokens_actual_mean",
                       windowMean(validationWindow.actualCacheHitTokens));
    reportWindowMetric("master.validation_window_queue_predicted_mean_ms",
                       windowMean(validationWindow.predictedQueueMs));
    reportWindowMetric("master.validation_window_queue_actual_mean_ms",
                       windowMean(validationWindow.actualQueueMs));
    reportWindowMetric("master.queue_model_window_raw_prediction_mean_ms",
                       windowMean(validationWindow.queueRawPredictedMs));
    reportWindowMetric("master.queue_model_window_prediction_mean_ms",
                       windowMean(validationWindow.agentPredictedQueueMs));
    reportWindowMetric("master.queue_model_window_preceding_batch_count_mean",
                       windowMean(validationWindow.queueFeatureBatchCount));
    reportWindowMetric("master.validation_window_execution_predicted_mean_ms",
                       windowMean(validationWindow.masterPredictedExecutionMs));
    reportWindowMetric("master.validation_window_execution_agent_predicted_mean_ms",
                       windowMean(validationWindow.agentPredictedExecutionMs));
    reportWindowMetric("master.validation_window_execution_actual_mean_ms",
                       windowMean(validationWindow.actualExecutionMs));
    reportWindowMetric("master.validation_window_total_predicted_mean_ms",
                       windowMean(validationWindow.predictedTotalMs));
    reportWindowMetric("master.validation_window_total_actual_mean_ms",
                       windowMean(validationWindow.actualTotalMs));
    const auto windowZeroCount =
        static_cast<double>(validationWindow.zeroQueueSampleCount);
    reportWindowMetric("master.validation_window_queue_zero_samples",
                       validationWindow.zeroQueueSampleCount);
    reportWindowMetric("master.validation_window_queue_zero_actual_mean_ms",
                       windowZeroCount > 0.0
                           ? validationWindow.zeroQueueActualMs / windowZeroCount
                           : 0.0);
    const auto windowPositiveCount =
        static_cast<double>(validationWindow.positiveQueueSampleCount);
    reportWindowMetric("master.validation_window_queue_positive_samples",
                       validationWindow.positiveQueueSampleCount);
    reportWindowMetric(
        "master.validation_window_queue_positive_predicted_mean_ms",
        windowPositiveCount > 0.0
            ? validationWindow.positivePredictedQueueMs / windowPositiveCount
            : 0.0);
    reportWindowMetric("master.validation_window_queue_positive_actual_mean_ms",
                       windowPositiveCount > 0.0
                           ? validationWindow.positiveActualQueueMs / windowPositiveCount
                           : 0.0);
}

}
