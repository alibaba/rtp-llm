#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "autil/HashAlgorithm.h"
#include "autil/NetUtil.h"
#include "autil/TimeUtility.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace navi_sched {
namespace {

ClientRequestToMasterDataConstPtr makeCanonicalRequest(
    const ClientRequestToMasterData &request, uint64_t requestId)
{
    auto output = std::make_shared<ClientRequestToMasterData>();
    output->version = request.version;
    output->bizName = request.bizName;
    output->timeoutMs = request.timeoutMs;
    output->promptTokenCount = request.promptTokenCount;
    output->blockSize = request.blockSize;
    output->requestId = requestId;
    output->collectReplay = request.collectReplay;
    output->blockHashes = request.blockHashes;
    output->forwardPayload = request.forwardPayload;
    return output;
}

}

const std::string MasterScheduleR::RESOURCE_ID = "navi_sched.master_schedule_r";

MasterScheduleR::~MasterScheduleR() { stop(); }

void MasterScheduleR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode MasterScheduleR::init(navi::ResourceInitContext &ctx)
{
    autil::ScopedWriteLock lock(_lock);
    _stopped = false;
    _nextCanonicalRequestId.store(
        makeRequestIdSeed(autil::NetUtil::getBindIp(),
                          autil::TimeUtility::currentTimeInMicroSeconds()),
        std::memory_order_relaxed);
    return navi::EC_NONE;
}

uint64_t MasterScheduleR::makeRequestIdSeed(const std::string &bindIp,
                                            int64_t startupTimeUs)
{
    const auto source = bindIp + ":" + std::to_string(startupTimeUs);
    const auto seed = autil::HashAlgorithm::hashString64(source.data(), source.size());
    return seed == 0 ? 1 : seed;
}

uint64_t MasterScheduleR::nextRequestId()
{
    auto requestId = _nextCanonicalRequestId.fetch_add(1, std::memory_order_relaxed);
    if (requestId == 0) {
        requestId = _nextCanonicalRequestId.fetch_add(1, std::memory_order_relaxed);
    }
    return requestId;
}

uint64_t MasterScheduleR::beginRequestPipeGeneration()
{
    uint64_t generation = 0;
    {
        autil::ScopedWriteLock lock(_lock);
        if (_stopped) {
            return 0;
        }
        generation =
            _requestPipeGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
        _requestPipeOwnerGeneration = generation;
        _kernelPendingRequestCount = 0;
        _requestPipe.reset();
    }
    return generation;
}

bool MasterScheduleR::setRequestPipe(
    uint64_t generation, const navi::AsyncPipePtr &requestPipe)
{
    if (0 == generation || !requestPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject invalid Master schedule request pipe, generation [%lu], "
                        "pipe [%p]",
                        generation, requestPipe.get());
        return false;
    }
    {
        autil::ScopedWriteLock lock(_lock);
        if (_stopped || _requestPipeOwnerGeneration != generation ||
            (_requestPipe && _requestPipe != requestPipe)) {
            NAVI_KERNEL_LOG(ERROR,
                            "reject stale Master schedule request pipe, generation [%lu], "
                            "ownerGeneration [%lu], stopped [%d], currentPipe [%p], "
                            "incomingPipe [%p]",
                            generation, _requestPipeOwnerGeneration,
                            static_cast<int>(_stopped), _requestPipe.get(),
                            requestPipe.get());
            return false;
        }
        _requestPipe = requestPipe;
    }
    return true;
}

bool MasterScheduleR::waitRequestPipe(uint64_t generation,
                                      int64_t timeoutMs) const
{
    if (0 == generation || timeoutMs < 0) {
        return false;
    }
    constexpr int64_t READY_POLL_INTERVAL_US = 10 * 1000;
    const auto deadlineUs =
        autil::TimeUtility::monotonicTimeUs() + timeoutMs * 1000;
    while (true) {
        {
            autil::ScopedReadLock lock(_lock);
            if (_stopped || _requestPipeOwnerGeneration != generation) {
                return false;
            }
            if (_requestPipe) {
                return true;
            }
        }
        const auto remainingUs =
            deadlineUs - autil::TimeUtility::monotonicTimeUs();
        if (remainingUs <= 0) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(
            std::min(remainingUs, READY_POLL_INTERVAL_US)));
    }
}

void MasterScheduleR::clearRequestPipe(uint64_t generation)
{
    autil::ScopedWriteLock lock(_lock);
    if (_requestPipeOwnerGeneration != generation) {
        NAVI_KERNEL_LOG(INFO,
                        "ignore stale Master schedule pipe cleanup, generation [%lu], "
                        "ownerGeneration [%lu]",
                        generation, _requestPipeOwnerGeneration);
        return;
    }
    _requestPipe.reset();
    _requestPipeOwnerGeneration = 0;
    _kernelPendingRequestCount = 0;
}

void MasterScheduleR::rejectUnavailable(
    const ClientRequestToMasterDataConstPtr &request,
    const navi::AsyncPipePtr &responsePipe,
    const MasterScheduleRequestDataPtr &scheduleRequest)
{
    auto response = std::make_shared<MasterResponseToClientData>();
    response->requestId = request ? request->requestId : 0;
    response->reason = "master_restarting";
    if (scheduleRequest) {
        if (!scheduleRequest->finish(response)) {
            NAVI_KERNEL_LOG(ERROR,
                            "finish unavailable Master schedule request failed, "
                            "requestId [%lu]",
                            response->requestId);
        }
        return;
    }
    if (!responsePipe || navi::EC_NONE != responsePipe->setData(response)) {
        NAVI_KERNEL_LOG(ERROR,
                        "publish unavailable Master schedule response failed, "
                        "requestId [%lu], responsePipe [%p]",
                        response->requestId, responsePipe.get());
    }
    if (responsePipe) {
        responsePipe->setEof();
    }
}

MasterScheduleRequestDataPtr MasterScheduleR::submit(const ClientRequestToMasterDataConstPtr &request,
                             const navi::AsyncPipePtr &responsePipe)
{
    if (!request || !responsePipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject invalid Master schedule submit, request [%p], "
                        "responsePipe [%p]",
                        request.get(), responsePipe.get());
        return nullptr;
    }
    const auto canonicalRequestId = nextRequestId();
    auto canonicalRequest = makeCanonicalRequest(*request, canonicalRequestId);
    auto data = std::make_shared<MasterScheduleRequestData>(canonicalRequest,
                                                            responsePipe);
    navi::AsyncPipePtr failedRequestPipe;
    uint64_t failedRequestPipeGeneration = 0;
    bool rejectedWhileStopped = false;
    bool rejectedWithoutPipe = false;
    {
        autil::ScopedReadLock lock(_lock);
        if (_stopped) {
            rejectedWhileStopped = true;
        } else if (!_requestPipe) {
            rejectedWithoutPipe = true;
        } else {
            if (navi::EC_NONE != _requestPipe->setData(data)) {
                failedRequestPipe = _requestPipe;
                failedRequestPipeGeneration = _requestPipeOwnerGeneration;
            } else {
                _submitCount.fetch_add(1, std::memory_order_relaxed);
                return data;
            }
        }
    }
    if (rejectedWhileStopped || rejectedWithoutPipe) {
        NAVI_KERNEL_LOG(WARN,
                        "reject Master schedule request, requestId [%lu], "
                        "reason [%s]",
                        request->requestId,
                        rejectedWhileStopped ? "resource_stopped"
                                             : "no_active_pipe");
        rejectUnavailable(request, responsePipe, data);
        return nullptr;
    }

    uint64_t ownerGeneration = 0;
    bool stoppedAfterPipeFailure = false;
    {
        autil::ScopedWriteLock lock(_lock);
        stoppedAfterPipeFailure = _stopped;
        if (!stoppedAfterPipeFailure && failedRequestPipe &&
            _requestPipe == failedRequestPipe &&
            _requestPipeOwnerGeneration == failedRequestPipeGeneration) {
            _requestPipe.reset();
        }
        if (!stoppedAfterPipeFailure && _requestPipe &&
            navi::EC_NONE == _requestPipe->setData(data)) {
            _submitCount.fetch_add(1, std::memory_order_relaxed);
            return data;
        }
        if (!stoppedAfterPipeFailure && _requestPipe) {
            _requestPipe.reset();
        }
        ownerGeneration = _requestPipeOwnerGeneration;
    }
    NAVI_KERNEL_LOG(WARN,
                    "reject Master schedule request after pipe failure, "
                    "requestId [%lu], failedGeneration [%lu], ownerGeneration [%lu], "
                    "stopped [%d]",
                    request->requestId, failedRequestPipeGeneration,
                    ownerGeneration, static_cast<int>(stoppedAfterPipeFailure));
    rejectUnavailable(request, responsePipe, data);
    return nullptr;
}

bool MasterScheduleR::stopped() const
{
    autil::ScopedReadLock lock(_lock);
    return _stopped;
}

void MasterScheduleR::stop()
{
    navi::AsyncPipePtr requestPipe;
    {
        autil::ScopedWriteLock lock(_lock);
        _stopped = true;
        requestPipe = _requestPipe;
    }
    if (requestPipe) {
        requestPipe->setDataOverride(nullptr, false, true);
    }
}

uint64_t MasterScheduleR::getSubmitCount() const
{
    return _submitCount.load(std::memory_order_relaxed);
}

uint64_t MasterScheduleR::getRequestPipeGeneration() const
{
    return _requestPipeGeneration.load(std::memory_order_relaxed);
}

size_t MasterScheduleR::getPendingRequestCount() const
{
    autil::ScopedReadLock lock(_lock);
    return _kernelPendingRequestCount +
           (_requestPipe ? _requestPipe->getDataItemCnt() : 0);
}

size_t MasterScheduleR::getSchedulingRequestCount() const
{
    return _schedulingRequestCount->load(std::memory_order_relaxed);
}

std::shared_ptr<void> MasterScheduleR::trackSchedulingRequests(size_t count)
{
    _schedulingRequestCount->fetch_add(count, std::memory_order_relaxed);
    return std::shared_ptr<void>(nullptr, [counter = _schedulingRequestCount, count](void *) {
        counter->fetch_sub(count, std::memory_order_relaxed);
    });
}

void MasterScheduleR::recordKernelPendingRequests(uint64_t generation, size_t count)
{
    autil::ScopedWriteLock lock(_lock);
    if (_requestPipeOwnerGeneration == generation) {
        _kernelPendingRequestCount = count;
    }
}

MasterScheduleR::BacklogStats MasterScheduleR::getBacklogStats() const
{
    BacklogStats stats;
    {
        autil::ScopedReadLock lock(_lock);
        stats.kernelPending = _kernelPendingRequestCount;
        stats.pipePending = _requestPipe ? _requestPipe->getDataItemCnt() : 0;
    }
    stats.expired = _expiredRequestCount.load(std::memory_order_relaxed);
    stats.cancelled = _cancelledRequestCount.load(std::memory_order_relaxed);
    stats.dequeueWaitMs = _dequeueWaitMs.load(std::memory_order_relaxed);
    stats.dequeueMaxWaitMs = _dequeueMaxWaitMs.load(std::memory_order_relaxed);
    return stats;
}

void MasterScheduleR::recordRetiredRequest(bool expired)
{
    (expired ? _expiredRequestCount : _cancelledRequestCount)
        .fetch_add(1, std::memory_order_relaxed);
}

void MasterScheduleR::recordDequeueWait(int64_t totalUs, int64_t maxUs, size_t count)
{
    if (count) {
        _dequeueWaitMs.store(static_cast<double>(totalUs) / count / 1000.0,
                             std::memory_order_relaxed);
        _dequeueMaxWaitMs.store(maxUs / 1000.0, std::memory_order_relaxed);
    }
}

void MasterScheduleR::recordQueueOverload(size_t requestCount,
                                          size_t rejectedRequestCount,
                                          double averagePredictedLatencyMs,
                                          double dropProbability)
{
    _queueOverloadEvaluatedRoundCount.fetch_add(1, std::memory_order_relaxed);
    _queueOverloadEvaluatedRequestCount.fetch_add(requestCount,
                                                  std::memory_order_relaxed);
    _queueOverloadRejectedRequestCount.fetch_add(rejectedRequestCount,
                                                 std::memory_order_relaxed);
    _queueOverloadLatestAveragePredictedLatencyMs.store(
        averagePredictedLatencyMs, std::memory_order_relaxed);
    _queueOverloadLatestDropProbability.store(dropProbability,
                                              std::memory_order_relaxed);
}

void MasterScheduleR::recordExecutionUtilization(double average, double integral, size_t reportingNodes, bool enabled,
    double rawAverage, uint64_t running, uint64_t waiting, bool globalScaling)
{
    _executionRawAverage.store(rawAverage, std::memory_order_relaxed);
    _executionRunning.store(running, std::memory_order_relaxed);
    _executionWaiting.store(waiting, std::memory_order_relaxed);
    _executionGlobalScaling.store(globalScaling, std::memory_order_relaxed);
    _executionUtilizationAverage.store(average, std::memory_order_relaxed);
    _executionUtilizationIntegral.store(integral, std::memory_order_relaxed);
    _executionUtilizationReportingNodes.store(reportingNodes, std::memory_order_relaxed);
    _executionUtilizationEnabled.store(enabled, std::memory_order_relaxed);
}

void MasterScheduleR::recordExecutionUtilizationGains(double ingressQps, double latencySeconds,
                                                      double kp, double kiPerSecond)
{
    _executionIngressQps.store(ingressQps, std::memory_order_relaxed);
    _executionLatencySeconds.store(latencySeconds, std::memory_order_relaxed);
    _executionKp.store(kp, std::memory_order_relaxed);
    _executionKiPerSecond.store(kiPerSecond, std::memory_order_relaxed);
}

MasterScheduleR::ExecutionUtilizationStats MasterScheduleR::getExecutionUtilizationStats() const
{
    ExecutionUtilizationStats stats;
    stats.rawAverage = _executionRawAverage.load(std::memory_order_relaxed);
    stats.running = _executionRunning.load(std::memory_order_relaxed);
    stats.waiting = _executionWaiting.load(std::memory_order_relaxed);
    stats.globalScaling = _executionGlobalScaling.load(std::memory_order_relaxed);
    stats.average = _executionUtilizationAverage.load(std::memory_order_relaxed);
    stats.integral = _executionUtilizationIntegral.load(std::memory_order_relaxed);
    stats.ingressQps = _executionIngressQps.load(std::memory_order_relaxed);
    stats.latencySeconds = _executionLatencySeconds.load(std::memory_order_relaxed);
    stats.kp = _executionKp.load(std::memory_order_relaxed);
    stats.kiPerSecond = _executionKiPerSecond.load(std::memory_order_relaxed);
    stats.reportingNodes = _executionUtilizationReportingNodes.load(std::memory_order_relaxed);
    stats.enabled = _executionUtilizationEnabled.load(std::memory_order_relaxed);
    return stats;
}

MasterScheduleR::QueueOverloadStats MasterScheduleR::getQueueOverloadStats() const
{
    QueueOverloadStats stats;
    stats.evaluatedRoundCount =
        _queueOverloadEvaluatedRoundCount.load(std::memory_order_relaxed);
    stats.evaluatedRequestCount =
        _queueOverloadEvaluatedRequestCount.load(std::memory_order_relaxed);
    stats.rejectedRequestCount =
        _queueOverloadRejectedRequestCount.load(std::memory_order_relaxed);
    stats.latestAveragePredictedLatencyMs =
        _queueOverloadLatestAveragePredictedLatencyMs.load(
            std::memory_order_relaxed);
    stats.latestDropProbability =
        _queueOverloadLatestDropProbability.load(std::memory_order_relaxed);
    stats.active = stats.evaluatedRoundCount > 0;
    return stats;
}

void MasterScheduleR::recordColdStartRouting(
    const NaviSchedColdStartRoutingStats &stats)
{
    if (!stats.evaluated) {
        return;
    }
    _coldStartEvaluatedRoundCount.fetch_add(1, std::memory_order_relaxed);
    _coldStartTrustedRequestCount.fetch_add(
        stats.trustedRequestCount, std::memory_order_relaxed);
    _coldStartUntrustedRequestCount.fetch_add(
        stats.untrustedRequestCount, std::memory_order_relaxed);
    _coldStartRejectedRequestCount.fetch_add(
        stats.rejectedRequestCount, std::memory_order_relaxed);
    _coldStartLatestTargetInflight.store(stats.targetInflight,
                                         std::memory_order_relaxed);
    _coldStartLatestTrustedNodeCount.store(stats.trustedNodeCount,
                                           std::memory_order_relaxed);
    _coldStartLatestUntrustedNodeCount.store(stats.untrustedNodeCount,
                                             std::memory_order_relaxed);
    _coldStartLatestUntrustedCapacity.store(stats.untrustedCapacity,
                                             std::memory_order_relaxed);
    _coldStartLatestRampingNodeCount.store(stats.rampingNodeCount,
                                           std::memory_order_relaxed);
    _coldStartLatestUnsafeWaitingMarginalNodeCount.store(
        stats.unsafeWaitingMarginalNodeCount, std::memory_order_relaxed);
    _coldStartUnsafeWaitingMarginalNodeObservationCount.fetch_add(
        stats.unsafeWaitingMarginalNodeCount, std::memory_order_relaxed);
    _coldStartLatestMinimumRampProgress.store(stats.minimumRampProgress,
                                               std::memory_order_relaxed);
    _coldStartLatestMinimumRampedTargetInflight.store(
        stats.minimumRampedTargetInflight, std::memory_order_relaxed);
}

MasterScheduleR::ColdStartRoutingStats
MasterScheduleR::getColdStartRoutingStats() const
{
    ColdStartRoutingStats stats;
    stats.evaluatedRoundCount =
        _coldStartEvaluatedRoundCount.load(std::memory_order_relaxed);
    stats.trustedRequestCount =
        _coldStartTrustedRequestCount.load(std::memory_order_relaxed);
    stats.untrustedRequestCount =
        _coldStartUntrustedRequestCount.load(std::memory_order_relaxed);
    stats.rejectedRequestCount =
        _coldStartRejectedRequestCount.load(std::memory_order_relaxed);
    stats.latestTargetInflight =
        _coldStartLatestTargetInflight.load(std::memory_order_relaxed);
    stats.latestTrustedNodeCount =
        _coldStartLatestTrustedNodeCount.load(std::memory_order_relaxed);
    stats.latestUntrustedNodeCount =
        _coldStartLatestUntrustedNodeCount.load(std::memory_order_relaxed);
    stats.latestUntrustedCapacity =
        _coldStartLatestUntrustedCapacity.load(std::memory_order_relaxed);
    stats.latestRampingNodeCount =
        _coldStartLatestRampingNodeCount.load(std::memory_order_relaxed);
    stats.latestUnsafeWaitingMarginalNodeCount =
        _coldStartLatestUnsafeWaitingMarginalNodeCount.load(
            std::memory_order_relaxed);
    stats.unsafeWaitingMarginalNodeObservationCount =
        _coldStartUnsafeWaitingMarginalNodeObservationCount.load(
            std::memory_order_relaxed);
    stats.latestMinimumRampProgress =
        _coldStartLatestMinimumRampProgress.load(std::memory_order_relaxed);
    stats.latestMinimumRampedTargetInflight =
        _coldStartLatestMinimumRampedTargetInflight.load(
            std::memory_order_relaxed);
    return stats;
}

navi::SummaryErrorCode MasterScheduleR::getSummary(const std::string &query,
                                                   navi::SummaryBuilder &builder) const
{
    bool ready = false;
    {
        autil::ScopedReadLock lock(_lock);
        ready = bool(_requestPipe);
    }
    auto table = builder.newTable("NaviSchedMasterScheduleOverview");
    TABLE_ADD_COLUMN_VALUE(table, "ready", ready);
    TABLE_ADD_COLUMN_VALUE(table, "submit_count", getSubmitCount());
    TABLE_ADD_COLUMN_VALUE(table, "request_pipe_generation",
                           getRequestPipeGeneration());
    TABLE_ADD_COLUMN_VALUE(table, "pending_request_count", getPendingRequestCount());
    const auto overloadStats = getQueueOverloadStats();
    TABLE_ADD_COLUMN_VALUE(table, "queue_overload_evaluated_round_count",
                           overloadStats.evaluatedRoundCount);
    TABLE_ADD_COLUMN_VALUE(table, "queue_overload_evaluated_request_count",
                           overloadStats.evaluatedRequestCount);
    TABLE_ADD_COLUMN_VALUE(table, "queue_overload_rejected_request_count",
                           overloadStats.rejectedRequestCount);
    TABLE_ADD_COLUMN_VALUE(
        table, "queue_overload_latest_average_predicted_latency_ms",
        overloadStats.latestAveragePredictedLatencyMs);
    TABLE_ADD_COLUMN_VALUE(table, "queue_overload_latest_drop_probability",
                           overloadStats.latestDropProbability);
    const auto executionUtilization = getExecutionUtilizationStats();
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_control_enabled",
                           executionUtilization.enabled);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_raw_average", executionUtilization.rawAverage);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_running", executionUtilization.running);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_waiting", executionUtilization.waiting);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_global_scaling", executionUtilization.globalScaling);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_average",
                           executionUtilization.average);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_integral",
                           executionUtilization.integral);
    TABLE_ADD_COLUMN_VALUE(table, "execution_utilization_reporting_nodes",
                           executionUtilization.reportingNodes);
    const auto coldStartStats = getColdStartRoutingStats();
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_evaluated_round_count",
                           coldStartStats.evaluatedRoundCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_trusted_request_count",
                           coldStartStats.trustedRequestCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_untrusted_request_count",
                           coldStartStats.untrustedRequestCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_rejected_request_count",
                           coldStartStats.rejectedRequestCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_target_inflight",
                           coldStartStats.latestTargetInflight);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_trusted_node_count",
                           coldStartStats.latestTrustedNodeCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_untrusted_node_count",
                           coldStartStats.latestUntrustedNodeCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_untrusted_capacity",
                           coldStartStats.latestUntrustedCapacity);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_ramping_node_count",
                           coldStartStats.latestRampingNodeCount);
    TABLE_ADD_COLUMN_VALUE(
        table, "cold_start_latest_unsafe_waiting_marginal_node_count",
        coldStartStats.latestUnsafeWaitingMarginalNodeCount);
    TABLE_ADD_COLUMN_VALUE(
        table, "cold_start_unsafe_waiting_marginal_node_observation_count",
        coldStartStats.unsafeWaitingMarginalNodeObservationCount);
    TABLE_ADD_COLUMN_VALUE(table, "cold_start_latest_minimum_ramp_progress",
                           coldStartStats.latestMinimumRampProgress);
    TABLE_ADD_COLUMN_VALUE(
        table, "cold_start_latest_minimum_ramped_target_inflight",
        coldStartStats.latestMinimumRampedTargetInflight);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(MasterScheduleR);

}
