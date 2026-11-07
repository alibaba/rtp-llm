#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"

#include "aios/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextData.h"
#include "aios/navi_sched/scheduler/MasterScheduleResultGraph.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "autil/legacy/json.h"
#include "autil/TimeUtility.h"
#include "autil/Scope.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_set>
#include <unordered_map>

namespace navi_sched {

const std::string MasterScheduleContextKernel::KERNEL_ID =
    "navi_sched.master_schedule_context_k";
const std::string MasterScheduleResultKernel::KERNEL_ID =
    "navi_sched.master_schedule_result_k";

MasterScheduleContextKernel::MasterScheduleContextKernel()
    : _randomGenerator(autil::TimeUtility::currentTimeInMicroSeconds())
{
    (void)_randomGenerator.get();
}

void MasterScheduleContextKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}

bool MasterScheduleContextKernel::config(navi::KernelConfigContext &ctx)
{
    const auto graphAttrs = ctx.getBinaryAttrs();
    const auto scheduler = graphAttrs.find("scheduler_kernel_id");
    _schedulerKernelId =
        scheduler == graphAttrs.end() ? std::string() : scheduler->second;
    NAVI_JSONIZE(ctx, "max_wait_schedule_count", _maxWaitScheduleCount,
                 _maxWaitScheduleCount);
    NAVI_JSONIZE(ctx, "max_wait_schedule_time_ms", _maxWaitScheduleTimeMs,
                 _maxWaitScheduleTimeMs);
    NAVI_JSONIZE(ctx, "time_compression", _timeCompression, _timeCompression);
    NAVI_JSONIZE(ctx, "request_pipe_generation", _requestPipeGeneration,
                 _requestPipeGeneration);
    _maxWaitScheduleTimeMs = std::max<int64_t>(10, _maxWaitScheduleTimeMs);
    return _maxWaitScheduleCount > 0 && std::isfinite(_timeCompression) &&
           _timeCompression > 0.0 && _requestPipeGeneration > 0;
}

navi::ErrorCode MasterScheduleContextKernel::init(navi::KernelInitContext &ctx)
{
    _requestPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _roundCompletionPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_requestPipe || !_roundCompletionPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Master schedule request pipe failed, generation [%ld]",
                        _requestPipeGeneration);
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!_masterScheduleR->setRequestPipe(static_cast<uint64_t>(_requestPipeGeneration),
                                          _requestPipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "install Master schedule request pipe failed, generation [%ld], "
                        "resourceGeneration [%lu], stopped [%d]",
                        _requestPipeGeneration,
                        _masterScheduleR->getRequestPipeGeneration(),
                        static_cast<int>(_masterScheduleR->stopped()));
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode MasterScheduleContextKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_roundCompletionPipe) {
        navi::DataPtr wake;
        bool wakeEof = false;
        _roundCompletionPipe->getData(wake, wakeEof);
    }
    if (_schedulerConfigR) {
        const auto schedulerConfig = _schedulerConfigR->get();
        _executionUtilizationLimitEnabled =
            schedulerConfig.config.overloadControlStrategy == "execution_utilization";
        if (_executionUtilizationConfig.globalScaling !=
            schedulerConfig.config.executionUtilization.globalScaling) {
            _executionUtilizationController.reset();
        }
        _executionUtilizationConfig = schedulerConfig.config.executionUtilization;
        _maxWaitScheduleCount = schedulerConfig.config.maxWaitScheduleCount;
        _maxWaitScheduleTimeMs = schedulerConfig.config.maxWaitScheduleTimeMs;
        _queueOverloadStartCycleMultiplier =
            schedulerConfig.config.queueOverloadStartCycleMultiplier;
        _queueOverloadFullCycleMultiplier =
            schedulerConfig.config.queueOverloadFullCycleMultiplier;
        _timeCompression = schedulerConfig.config.agent.timeCompression;
        _peerCacheEnabled = schedulerConfig.config.peerCacheEnabled;
        if (_peerCacheReadyConfigGeneration != schedulerConfig.generation) {
            const auto &agents = schedulerConfig.config.peerCacheReadyAgents;
            _peerCacheReadyAgents =
                std::make_shared<const std::unordered_set<std::string>>(agents.begin(),
                                                                        agents.end());
            _peerCacheReadyConfigGeneration = schedulerConfig.generation;
        }
    }
    if (!drainRequests()) {
        NAVI_KERNEL_LOG(ERROR, "drain Master schedule requests failed");
        return navi::EC_ABORT;
    }
    auto stopped = _masterScheduleR->stopped();
    if (stopped && !drainRequests()) {
        NAVI_KERNEL_LOG(ERROR, "drain Master schedule requests during stop failed");
        return navi::EC_ABORT;
    }
    _requestEof = _requestEof || stopped;
    auto currentTimeUs = autil::TimeUtility::monotonicTimeUs();
    _masterScheduleR->recordKernelPendingRequests(_requestPipeGeneration,
                                                  _pendingRequests.size());
    retireCancelledRequests();
    if (_readyContexts.empty() && _roundInFlight->load(std::memory_order_acquire)) {
        stopScheduleTimer(ctx);
        return navi::EC_NONE;
    }
    if (_readyContexts.empty() && shouldBuildContext(currentTimeUs)) {
        const auto activationTimeUs = autil::TimeUtility::monotonicTimeUs();
        buildReadyContexts(activationTimeUs);
    }
    if (!_readyContexts.empty()) {
        auto context = std::move(_readyContexts.front());
        _readyContexts.pop_front();
        auto eof = _requestEof && _pendingRequests.empty() && _readyContexts.empty();
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), context, eof);
        if (eof) {
            stopScheduleTimer(ctx);
        } else if (!activate(ctx)) {
            NAVI_KERNEL_LOG(ERROR,
                            "activate next Master schedule round failed, "
                            "pendingRequests [%lu], readyContexts [%lu]",
                            _pendingRequests.size(), _readyContexts.size());
            return navi::EC_ABORT;
        }
        return navi::EC_NONE;
    }
    if (_requestEof) {
        stopScheduleTimer(ctx);
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
        return navi::EC_NONE;
    }
    if (!updateScheduleTimer(ctx, currentTimeUs)) {
        NAVI_KERNEL_LOG(ERROR,
                        "update Master schedule timer failed, pendingRequests [%lu], "
                        "maxWaitScheduleCount [%lu], maxWaitScheduleTimeMs [%ld]",
                        _pendingRequests.size(), _maxWaitScheduleCount,
                        _maxWaitScheduleTimeMs);
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

std::shared_ptr<void> MasterScheduleContextKernel::beginScheduleRound()
{
    _roundInFlight->store(true, std::memory_order_release);
    return std::shared_ptr<void>(
        nullptr, [inFlight = _roundInFlight, pipe = _roundCompletionPipe](void *) {
            inFlight->store(false, std::memory_order_release);
            if (pipe) {
                pipe->setDataOverride(nullptr, false, true);
            }
        });
}

void MasterScheduleContextKernel::retireCancelledRequests()
{
    const auto nowUs = autil::TimeUtility::monotonicTimeUs();
    _pendingRequests.erase(
        std::remove_if(
            _pendingRequests.begin(), _pendingRequests.end(),
            [&](const auto &request) { return retireRequest(request, nowUs); }),
        _pendingRequests.end());
    _masterScheduleR->recordKernelPendingRequests(_requestPipeGeneration,
                                                  _pendingRequests.size());
}

bool MasterScheduleContextKernel::retireRequest(
    const MasterScheduleRequestDataPtr &request, int64_t currentTimeUs)
{
    if (!request->pending()) {
        _masterScheduleR->recordRetiredRequest(false);
        return true;
    }
    const auto waitUs = std::max<int64_t>(0, currentTimeUs - request->enqueueTimeUs);
    if (request->request->timeoutMs > 0 && waitUs / 1000 >= request->request->timeoutMs) {
        auto response = std::make_shared<MasterResponseToClientData>();
        response->requestId = request->request->requestId;
        response->reason = "timeout";
        request->finish(response);
        _masterScheduleR->recordRetiredRequest(true);
        return true;
    }
    return false;
}

bool MasterScheduleContextKernel::drainRequests()
{
    while (true) {
        navi::DataPtr data;
        bool eof = false;
        if (!_requestPipe->getData(data, eof)) {
            return true;
        }
        _requestEof |= eof;
        if (data) {
            auto request = std::dynamic_pointer_cast<MasterScheduleRequestData>(data);
            if (!request || !request->valid()) {
                NAVI_KERNEL_LOG(
                    ERROR, "invalid Master schedule request, data [%p], request [%p]",
                    data.get(), request.get());
                return false;
            }
            if (!retireRequest(request, autil::TimeUtility::monotonicTimeUs())) {
                _pendingRequests.emplace_back(std::move(request));
            }
        }
        if (eof) {
            return true;
        }
    }
}

bool MasterScheduleContextKernel::shouldBuildContext(int64_t currentTimeUs) const
{
    if (_pendingRequests.empty() || _roundInFlight->load(std::memory_order_acquire)) {
        return false;
    }
    if (_requestEof || 0 == _maxWaitScheduleTimeMs ||
        _pendingRequests.size() >= _maxWaitScheduleCount) {
        return true;
    }
    return getNextScheduleDelayMs(currentTimeUs) <= 0;
}

int64_t MasterScheduleContextKernel::getNextScheduleDelayMs(int64_t currentTimeUs) const
{
    auto maxWaitUs = _maxWaitScheduleTimeMs * 1000;
    auto nextDelayUs = maxWaitUs;
    for (const auto &request : _pendingRequests) {
        auto waitTimeUs = std::max<int64_t>(0, currentTimeUs - request->enqueueTimeUs);
        auto waitDelayUs = maxWaitUs - waitTimeUs;
        auto timeoutDelayUs = request->request->timeoutMs * 1000 - waitTimeUs - maxWaitUs;
        nextDelayUs = std::min(nextDelayUs, std::min(waitDelayUs, timeoutDelayUs));
    }
    return nextDelayUs <= 0 ? 0 : (nextDelayUs + 999) / 1000;
}

bool MasterScheduleContextKernel::stopScheduleTimer(navi::KernelComputeContext &ctx)
{
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(_requestPipe, activatedCount, remainingCount,
                                            periodMs, nextTickTimeMs)) {
        return true;
    }
    if (nextTickTimeMs < 0) {
        return true;
    }
    if (!ctx.stopPeriodicAsyncPipe(_requestPipe)) {
        NAVI_KERNEL_LOG(WARN, "stop Master schedule timer failed");
    }
    return true;
}

bool MasterScheduleContextKernel::updateScheduleTimer(navi::KernelComputeContext &ctx,
                                                      int64_t currentTimeUs)
{
    if (_pendingRequests.empty() || _maxWaitScheduleTimeMs <= 0) {
        return stopScheduleTimer(ctx);
    }
    auto delayMs = getNextScheduleDelayMs(currentTimeUs);
    if (delayMs <= 0) {
        return activate(ctx);
    }
    if (!ctx.registerPeriodicPipe(_requestPipe, delayMs, 1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "register Master schedule timer failed, delayMs [%ld], "
                        "pendingRequests [%lu]",
                        delayMs, _pendingRequests.size());
        return false;
    }
    return true;
}

void MasterScheduleContextKernel::buildReadyContexts(int64_t activationTimeUs)
{
    auto completion = beginScheduleRound();
    auto scheduleTimeUs = autil::TimeUtility::currentTime();
    auto currentTimeUs = autil::TimeUtility::monotonicTimeUs();
    auto count = std::min(_pendingRequests.size(), _maxWaitScheduleCount);
    std::unordered_map<std::string, MasterScheduleContextDataPtr> contexts;
    contexts.reserve(count);
    int64_t totalWaitUs = 0;
    int64_t maxWaitUs = 0;
    size_t scheduledCount = 0;
    for (size_t index = 0; index < count; ++index) {
        auto request = std::move(_pendingRequests.front());
        _pendingRequests.pop_front();
        if (retireRequest(request, autil::TimeUtility::monotonicTimeUs())) {
            continue;
        }
        auto &context = contexts[request->request->bizName];
        if (!context) {
            context = std::make_shared<MasterScheduleContextData>();
            context->context->_scheduleBeginTimeUs = activationTimeUs;
            context->context->_scheduleDeadlineUs =
                activationTimeUs + _maxWaitScheduleTimeMs * 800;
            context->context->_scheduleTimeUs = scheduleTimeUs;
            context->context->_scheduleId = _nextScheduleId++;
            context->context->_timeCompression = _timeCompression;
            context->context->_peerCacheEnabled = _peerCacheEnabled;
            context->context->_peerCacheReadyAgents = _peerCacheReadyAgents;
        }
        auto waitTimeUs = std::max<int64_t>(0, currentTimeUs - request->enqueueTimeUs);
        totalWaitUs += waitTimeUs;
        maxWaitUs = std::max(maxWaitUs, waitTimeUs);
        ++scheduledCount;
        context->context->_requests.emplace_back(
            MasterStateR::makeScheduleItem(*request->request, (waitTimeUs + 999) / 1000));
        context->context->addRequest(request);
    }
    _masterScheduleR->recordDequeueWait(totalWaitUs, maxWaitUs, scheduledCount);
    _masterScheduleR->recordKernelPendingRequests(_requestPipeGeneration,
                                                  _pendingRequests.size());
    for (auto &[bizName, context] : contexts) {
        context->context->_scheduleRequestCompletion =
            _masterScheduleR->trackSchedulingRequests(context->context->requestCount());
        const auto snapshotBeginUs = autil::TimeUtility::monotonicTimeUs();
        _masterStateR->snapshotScheduleNodes(bizName, *context->context);
        const auto snapshotEndUs = autil::TimeUtility::monotonicTimeUs();
        for (const auto &request : context->context->_responseRequests) {
            if (request && request->request && request->request->collectReplay) {
                const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
                context->context->setReplay(
                    _masterStateR->makeScheduleReplay(*context->context));
                context->context->_replayPrepareUs =
                    autil::TimeUtility::monotonicTimeUs() - replayBeginUs;
                context->context->addReplayCollectionTimeUs(
                    context->context->_replayPrepareUs);
                break;
            }
        }
        const auto admissionBeginUs = autil::TimeUtility::monotonicTimeUs();
        const auto accepted = prepareAdmission(context);
        const auto admissionEndUs = autil::TimeUtility::monotonicTimeUs();
        context->context->_snapshotPrepareUs = snapshotEndUs - snapshotBeginUs;
        context->context->_admissionPrepareUs =
            admissionEndUs - admissionBeginUs - context->context->_contextPrepareUs;
        if (!accepted) {
            context->context->_scheduleLatencyUs = std::max<int64_t>(
                0, autil::TimeUtility::monotonicTimeUs() - activationTimeUs);
            MasterScheduleResultKernel::recordReplayAdmission(*context->context);
            if (!context->context->finish()) {
                NAVI_KERNEL_LOG(ERROR,
                                "finish fully rate-limited schedule round failed, "
                                "scheduleId [%lu]",
                                context->context->_scheduleId);
            }
            if (context->context->_replay) {
                NAVI_KERNEL_LOG(
                    DEBUG, "[navi-sched-replay] schedule_id=%lu json=%s",
                    context->context->_scheduleId,
                    NaviSchedScheduleReplayJson::toJsonString(*context->context->_replay)
                        .c_str());
            }
            continue;
        }
        _masterStateR->recordScheduleRound(scheduleRoundRequestCount(*context->context));
        if (!context->context->_prepared && !context->context->prepare()) {
            context->context->fail("scheduler_error");
        }
        const auto preparedUs = autil::TimeUtility::monotonicTimeUs();
        context->context->_contextPrepareUs += preparedUs - admissionEndUs;
        context->context->_masterPrepareUs = preparedUs - activationTimeUs;
        if (context->context->_scheduleId < 10 ||
            context->context->_scheduleId % 100 == 0) {
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-prepare-timing] schedule_id=%lu requests=%lu "
                "candidates=%lu activation_to_snapshot_us=%ld snapshot_us=%ld "
                "replay_setup_us=%ld admission_us=%ld context_prepare_us=%ld "
                "activation_to_output_us=%ld snapshot_runtime_lock_wait_us=%ld "
                "snapshot_runtime_lock_hold_us=%ld",
                context->context->_scheduleId, context->context->requestCount(),
                context->context->nodeCount(), snapshotBeginUs - activationTimeUs,
                snapshotEndUs - snapshotBeginUs, context->context->_replayPrepareUs,
                context->context->_admissionPrepareUs,
                context->context->_contextPrepareUs, preparedUs - activationTimeUs,
                context->context->_snapshotRuntimeLockWaitUs,
                context->context->_snapshotRuntimeLockHoldUs);
        }
        context->context->_scheduleRoundCompletion = completion;
        _readyContexts.emplace_back(std::move(context));
    }
}

double MasterScheduleContextKernel::calculateQueueOverloadDropProbability(
    double averagePredictedLatencyMs, int64_t scheduleCycleMs,
    int64_t startCycleMultiplier, int64_t fullCycleMultiplier)
{
    if (!std::isfinite(averagePredictedLatencyMs) || averagePredictedLatencyMs <= 0.0 ||
        scheduleCycleMs < 10 || startCycleMultiplier == 0 || fullCycleMultiplier == 0 ||
        startCycleMultiplier < 0 || fullCycleMultiplier <= startCycleMultiplier) {
        return 0.0;
    }
    const auto startLatencyMs =
        static_cast<double>(scheduleCycleMs) * startCycleMultiplier;
    const auto fullLatencyMs = static_cast<double>(scheduleCycleMs) * fullCycleMultiplier;
    if (averagePredictedLatencyMs <= startLatencyMs) {
        return 0.0;
    }
    if (averagePredictedLatencyMs >= fullLatencyMs) {
        return 1.0;
    }
    return (averagePredictedLatencyMs - startLatencyMs) /
           (fullLatencyMs - startLatencyMs);
}

bool MasterScheduleContextKernel::shouldDropRequest(
    double dropProbability, uint64_t *sample, const std::function<uint64_t()> &randomDraw)
{
    if (sample) {
        *sample = 0;
    }
    if (dropProbability <= 0.0) {
        return false;
    }
    if (dropProbability >= 1.0) {
        return true;
    }
    const auto randomValue = randomDraw ? randomDraw() : _randomGenerator.get();
    if (sample) {
        *sample = randomValue;
    }
    const auto randomSample =
        static_cast<double>(randomValue) / std::numeric_limits<uint64_t>::max();
    return randomSample < dropProbability;
}

size_t MasterScheduleContextKernel::scheduleRoundRequestCount(
    const NaviSchedScheduleContext &context)
{
    const auto &coldStart = context._coldStartRoutingStats;
    // All-cold admission already removed rejected requests before routing.
    const auto rejectedAfterRouting = context._allColdUtilizationAdmission
                                          ? 0
                                          : context._queueOverloadRejectedRequestCount;
    return coldStart.evaluated
               ? context._requests.size() - coldStart.rejectedRequestCount -
                     rejectedAfterRouting
               : context.requestCount();
}

bool MasterScheduleContextKernel::prepareAdmission(
    const MasterScheduleContextDataPtr &contextData,
    const std::function<uint64_t()> &randomDraw)
{
    auto &context = *contextData->context;
    context._contextPrepareUs = 0;
    if (_schedulerKernelId == CostSchedulerKernel::KERNEL_ID && context.nodeCount() > 0) {
        const auto beginUs = autil::TimeUtility::monotonicTimeUs();
        const auto prepared = context.prepare();
        context._contextPrepareUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
        if (!prepared) {
            context.fail("scheduler_error");
            return false;
        }
        const bool allCold = !CostSchedulerColdStartStrategy::hasTrustedNode(context);
        context._allColdUtilizationAdmission = allCold;
        if (allCold) {
            const bool accepted = applyQueueOverloadLimit(contextData, randomDraw, true);
            // Admission compacts requests before any routing decisions exist.
            context._decisions.resize(context.requestCount());
            context._selectedNodeIndexes.resize(context.requestCount());
            if (!accepted) {
                return false;
            }
        }
        CostSchedulerColdStartStrategy::Result result;
        if (!CostSchedulerColdStartStrategy::apply(context, _randomGenerator, result,
                                                   randomDraw, !allCold)) {
            context.fail("scheduler_error");
            return false;
        }
        if (allCold) {
            return true;
        }
    }
    return applyQueueOverloadLimit(contextData, randomDraw);
}

void MasterScheduleContextKernel::updateExecutionUtilization(
    const NaviSchedScheduleContext &context, bool enabled)
{
    long double sum = 0.0;
    ClusterExecutionUtilization cluster;
    autil::legacy::json::JsonArray utilizationNodes;
    _executionUtilizationReportingNodes = 0;
    // Admission observes all ready nodes, including cold-start routes.
    for (const auto &node : context._scheduleNodes) {
        const auto busy = node.getExecutionBusyUtilization();
        if (node.getWorkloadRevision() > 0) {
            cluster.add(busy, node.getExecutionRunningRequests(),
                        node.getExecutionWaitingRequests());
        }
        const auto value = node.getExecutionUtilization();
        if (context._replay) {
            autil::legacy::json::JsonMap sample;
            sample["effective"] = value;
            sample["busy"] = busy;
            utilizationNodes.emplace_back(std::move(sample));
        }
        if (value >= 0.0) {
            sum += value;
            ++_executionUtilizationReportingNodes;
        }
    }
    if (context._replay) {
        context._replay->schedulerInfo["admission_utilization_nodes"] =
            std::move(utilizationNodes);
    }
    _averageExecutionUtilization =
        _executionUtilizationReportingNodes > 0
            ? static_cast<double>(sum / _executionUtilizationReportingNodes)
            : 0.0;
    _clusterExecutionUtilization = cluster;
    if (_executionUtilizationConfig.globalScaling) {
        _averageExecutionUtilization = cluster.effective();
        _executionUtilizationReportingNodes = cluster.nodes;
    }
    if (enabled) {
        _executionUtilizationController.update(
            _executionUtilizationConfig, _averageExecutionUtilization,
            _executionUtilizationReportingNodes, autil::TimeUtility::monotonicTimeUs());
    } else {
        _executionUtilizationController.reset();
    }
    _masterScheduleR->recordExecutionUtilization(
        _averageExecutionUtilization, _executionUtilizationController.integral(),
        _executionUtilizationReportingNodes, enabled,
        _clusterExecutionUtilization.average(), _clusterExecutionUtilization.running,
        _clusterExecutionUtilization.waiting, _executionUtilizationConfig.globalScaling);
}

bool MasterScheduleContextKernel::applyQueueOverloadLimit(
    const MasterScheduleContextDataPtr &contextData,
    const std::function<uint64_t()> &randomDraw, bool forceExecutionUtilization)
{
    auto &context = *contextData->context;
    const bool executionUtilizationMode =
        forceExecutionUtilization || _executionUtilizationLimitEnabled;
    context._executionUtilizationAdmission = executionUtilizationMode;
    updateExecutionUtilization(context, executionUtilizationMode);
    context._queueOverloadRandomDecisions.clear();
    context._queueOverloadEvaluated = false;
    context._queueOverloadAveragePredictedLatencyMs = 0.0;
    context._queueOverloadDropProbability = 0.0;
    context._queueOverloadScheduleCycleMs = _maxWaitScheduleTimeMs;
    context._queueOverloadStartCycleMultiplier = _queueOverloadStartCycleMultiplier;
    context._queueOverloadFullCycleMultiplier = _queueOverloadFullCycleMultiplier;
    const auto routed = context._coldStartRoutingStats.evaluated;
    context._queueOverloadRequestCount =
        routed && !context._scheduleViewActive ? 0 : context.requestCount();
    context._queueOverloadRejectedRequestCount = 0;
    if (routed && !context._scheduleViewActive) {
        return true;
    }
    if (0 == context.requestCount() || 0 == context.nodeCount()) {
        return context.requestCount() > 0;
    }
    const auto coldStartAnalysis = executionUtilizationMode
                                       ? CostSchedulerColdStartStrategy::Analysis()
                                       : CostSchedulerColdStartStrategy::analyze(context);
    const auto averagePredictedLatencyMs =
        coldStartAnalysis.trustedBacklogInflight > 0
            ? static_cast<double>(coldStartAnalysis.trustedBacklogLatencyWeightedSumMs /
                                  coldStartAnalysis.trustedBacklogInflight)
            : 0.0;
    auto dropProbability =
        executionUtilizationMode
            ? 0.0
            : calculateQueueOverloadDropProbability(
                  averagePredictedLatencyMs, _maxWaitScheduleTimeMs,
                  _queueOverloadStartCycleMultiplier, _queueOverloadFullCycleMultiplier);
    if (executionUtilizationMode) {
        dropProbability = _executionUtilizationController.probability();
        if (dropProbability < 0.001) dropProbability = 0.0;
    }
    context._queueOverloadEvaluated = true;
    context._queueOverloadAveragePredictedLatencyMs = averagePredictedLatencyMs;
    context._queueOverloadDropProbability = dropProbability;
    const auto requestCount = context.requestCount();
    if (dropProbability <= 0.0) {
        _masterScheduleR->recordQueueOverload(requestCount, 0, averagePredictedLatencyMs,
                                              dropProbability);
        return true;
    }

    std::vector<NaviSchedScheduleItem> acceptedRequests;
    std::vector<MasterScheduleRequestDataPtr> acceptedResponseRequests;
    std::vector<size_t> acceptedIndexes;
    if (routed) {
        acceptedIndexes.reserve(requestCount);
    } else {
        acceptedRequests.reserve(requestCount);
        acceptedResponseRequests.reserve(requestCount);
    }
    size_t rejectedRequestCount = 0;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        const auto storageIndex = context.resolveRequestIndex(requestIndex);
        auto &request = context._requests[storageIndex];
        auto &responseRequest = context._responseRequests[storageIndex];
        uint64_t randomSample = 0;
        const auto dropped =
            shouldDropRequest(dropProbability, &randomSample, randomDraw);
        if (context.isCollectingReplay() && dropProbability > 0.0 &&
            dropProbability < 1.0) {
            context._queueOverloadRandomDecisions.push_back(
                {storageIndex, randomSample, dropProbability, !dropped});
        }
        if (!dropped) {
            if (routed) {
                acceptedIndexes.emplace_back(storageIndex);
            } else {
                acceptedRequests.emplace_back(std::move(request));
                acceptedResponseRequests.emplace_back(std::move(responseRequest));
            }
            continue;
        }
        ++rejectedRequestCount;
        if (routed) {
            if (!context.reject(requestIndex, "rate_limited")) {
                context.fail("scheduler_error");
                return false;
            }
            context._decisions[storageIndex].predictedQueueLatencyMs =
                averagePredictedLatencyMs;
            continue;
        }
        NaviSchedScheduleContext::DeferredResponse deferred;
        deferred.request = std::move(responseRequest);
        deferred.decision.requestId = request.request.requestId;
        deferred.decision.scheduleId = context._scheduleId;
        deferred.decision.scheduleTimeUs = context._scheduleTimeUs;
        deferred.decision.reason = "rate_limited";
        deferred.decision.predictedQueueLatencyMs = averagePredictedLatencyMs;
        context._deferredResponses.emplace_back(std::move(deferred));
    }
    if (routed) {
        // Preserve cold-start selections and original response ownership. Only
        // shrink the trusted view consumed by Cost, including an empty remainder.
        auto nodeIndexes = context._scheduleViewNodeIndexes;
        context.clearScheduleView();
        if (!acceptedIndexes.empty() &&
            !context.activateScheduleView(acceptedIndexes, nodeIndexes)) {
            context.fail("scheduler_error");
            return false;
        }
    } else {
        context._requests.swap(acceptedRequests);
        context._responseRequests.swap(acceptedResponseRequests);
    }
    _masterScheduleR->recordQueueOverload(requestCount, rejectedRequestCount,
                                          averagePredictedLatencyMs, dropProbability);
    context._queueOverloadRejectedRequestCount = rejectedRequestCount;
    NAVI_KERNEL_LOG(DEBUG,
                    "overload limit evaluated, executionUtilizationMode [%d], "
                    "averagePredictedLatencyMs [%.3f], dropProbability [%.6f], "
                    "requests [%lu], rejected [%lu]",
                    static_cast<int>(executionUtilizationMode), averagePredictedLatencyMs,
                    dropProbability, requestCount, rejectedRequestCount);
    return routed || !context._requests.empty();
}

bool MasterScheduleContextKernel::activate(navi::KernelComputeContext &ctx)
{
    stopScheduleTimer(ctx);
    const auto ec = _requestPipe->setDataOverride(nullptr, false, true);
    if (navi::EC_NONE != ec) {
        NAVI_KERNEL_LOG(ERROR,
                        "activate Master schedule request pipe failed, errorCode [%d], "
                        "pendingRequests [%lu], readyContexts [%lu]",
                        static_cast<int>(ec), _pendingRequests.size(),
                        _readyContexts.size());
        return false;
    }
    return true;
}

void MasterScheduleResultKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", MASTER_SCHEDULE_RESULT_TYPE_ID, navi::IT_REQUIRE,
                    navi::IT_REQUIRE)
        .output("finished", "");
}

MasterScheduleResultKernel::AdmissionPredictionAudit
MasterScheduleResultKernel::getAdmissionPredictionAudit(
    const NaviSchedScheduleContext &context, size_t requestIndex,
    bool includeFrozenExecutionInputs)
{
    AdmissionPredictionAudit audit;
    if (requestIndex >= context._decisions.size() ||
        requestIndex >= context._selectedNodeIndexes.size()) {
        return audit;
    }
    const auto nodeIndex = context._selectedNodeIndexes[requestIndex];
    const auto &decision = context._decisions[requestIndex];
    if (nodeIndex >= context._scheduleNodes.size() || !decision.latencyPrediction) {
        return audit;
    }
    const auto &node = context._scheduleNodes[nodeIndex];
    const auto &prediction = *decision.latencyPrediction;
    const auto &parameters = node.getLatencyParameters();
    if (!prediction.valid() || !NonLinearPrefillModelR::validateParameters(parameters)) {
        return audit;
    }

    audit.withWorkloadBaseMs = node.predictBaseLatency(prediction.features);
    audit.withWorkloadFinalMs = node.applyLatencyResidual(audit.withWorkloadBaseMs);
    audit.baseExecutionMs = audit.withWorkloadBaseMs;
    audit.finalExecutionMs = audit.withWorkloadFinalMs;
    if (node.usesUnifiedLatencyPrediction() && prediction.hasFloor) {
        audit.floorBaseMs = node.predictBaseLatency(prediction.floorFeatures);
        audit.floorFinalMs = node.applyLatencyResidual(audit.floorBaseMs);
        const auto executionFixedLatencyMs =
            prediction.fixedLatencyMs - decision.predictedQueueLatencyMs;
        audit.baseExecutionMs = std::max(
            audit.withWorkloadBaseMs + executionFixedLatencyMs, audit.floorBaseMs);
        audit.finalExecutionMs = std::max(
            audit.withWorkloadFinalMs + executionFixedLatencyMs, audit.floorFinalMs);
    }
    audit.correctionMs = audit.finalExecutionMs - audit.baseExecutionMs;
    audit.valid = std::isfinite(audit.baseExecutionMs) &&
                  std::isfinite(audit.finalExecutionMs) &&
                  std::isfinite(audit.correctionMs);
    if (audit.valid && includeFrozenExecutionInputs && decision.batchId > 0) {
        // Copy only the already-published forecast inputs. Logging must not
        // perform another cache lookup against a later node state.
        const auto inputs = std::atomic_load(&node._executionCacheInputs);
        if (inputs) {
            audit.cacheObservation = inputs->observation.pool;
            for (size_t member = 0; member < context._decisions.size(); ++member) {
                if (context._decisions[member].batchId != decision.batchId) {
                    continue;
                }
                const auto storage = context.resolveRequestIndex(member);
                if (storage < inputs->inputs.size()) {
                    const auto &prefixes =
                        inputs->observation.fullPrefixBlocksBeforeQueryDrop;
                    audit.frozenExecutionInputs.push_back(
                        {context._decisions[member].requestId, inputs->inputs[storage],
                         storage < prefixes.size() ? prefixes[storage] : -1});
                }
            }
        }
    }
    return audit;
}

void MasterScheduleResultKernel::logAdmissionPredictions(
    const NaviSchedScheduleContext &context) const
{
    std::unordered_set<int64_t> loggedBatchIds;
    for (size_t requestIndex = 0; requestIndex < context._decisions.size();
         ++requestIndex) {
        const auto &decision = context._decisions[requestIndex];
        if (decision.batchId <= 0 || !loggedBatchIds.emplace(decision.batchId).second ||
            requestIndex >= context._selectedNodeIndexes.size()) {
            continue;
        }
        const auto nodeIndex = context._selectedNodeIndexes[requestIndex];
        if (nodeIndex >= context._scheduleNodes.size() || !decision.latencyPrediction) {
            continue;
        }
        const auto audit = getAdmissionPredictionAudit(context, requestIndex, true);
        if (!audit.valid) {
            NAVI_KERNEL_LOG(
                WARN,
                "skip invalid Master admission prediction audit, scheduleId [%lu], "
                "scheduleBatchId [%ld], requestId [%lu]",
                decision.scheduleId, decision.batchId, decision.requestId);
            continue;
        }
        const auto &node = context._scheduleNodes[nodeIndex];
        for (const auto &entry : audit.frozenExecutionInputs) {
            const auto &input = entry.input;
            const auto observation =
                audit.cacheObservation.value_or(TPUPrefillCacheObservation{});
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-frozen-prefill-input] schedule_time_us=%ld "
                "schedule_id=%lu schedule_batch_id=%ld request_id=%lu "
                "agent=%s model_revision=%ld prompt_tokens=%ld computed_tokens=%ld "
                "shared_prefix_boundary=%ld external_read_tokens=%ld "
                "cache_observation_valid=%d block_pool_revision=%lu "
                "block_pool_state_ready=%d "
                "block_pool_cached_blocks=%lu block_pool_full_ready_blocks=%lu "
                "block_pool_tail_ready_blocks=%lu block_pool_cached_checksum=%lu "
                "block_pool_full_ready_checksum=%lu block_pool_tail_ready_checksum=%lu "
                "full_prefix_before_query_drop_blocks=%ld",
                context.getScheduleTimeUs(), decision.scheduleId, decision.batchId,
                entry.requestId, node.getAgentId().c_str(),
                node.getLatencyModelRevision(), input.promptTokens, input.computedTokens,
                input.sharedPrefixBoundary, input.externalReadTokens,
                int(audit.cacheObservation.has_value()), observation.revision,
                int(observation.stateReady), observation.cachedBlocks,
                observation.fullReadyBlocks, observation.tailReadyBlocks,
                observation.cachedChecksum, observation.fullReadyChecksum,
                observation.tailReadyChecksum, entry.fullPrefixBlocksBeforeQueryDrop);
        }
        const auto localCache = std::atomic_load(&node._localCacheBlockCounts);
        const auto localHitTokens =
            localCache ? (*localCache)[requestIndex] *
                             context._requests[requestIndex].request.blockSize
                       : decision.predictedCacheHitTokenCount;
        const auto peerExtraTokens =
            std::max<int64_t>(0, decision.predictedCacheHitTokenCount - localHitTokens);
        const auto &prediction = *decision.latencyPrediction;
        const auto &features = prediction.features;
        const auto &floor = prediction.floorFeatures;
        const char *decayBranch = !prediction.hasFloor ? "none"
                                  : node.usesRunningElapsedProgress()
                                      ? "running_elapsed_progress"
                                      : "wall_clock_decay";
        const auto geometrySuffix =
            NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion())
                ? std::string(" model_binary_version=") +
                      std::to_string(node.getLatencyBinaryVersion()) +
                      " execution_geometry_valid=" +
                      (features.hasExecutionGeometry() ? "1" : "0") +
                      " execution_step_count=" +
                      std::to_string(features.getExecutionStepCount()) +
                      " global_padded_tokens=" +
                      std::to_string(features.getGlobalPaddedTokens()) +
                      " execution_cache_source_valid=" +
                      (features.hasExecutionCacheSource() ? "1" : "0") +
                      " external_read_request_count=" +
                      std::to_string(features.getExternalReadRequestCount()) +
                      " max_external_read_tokens=" +
                      std::to_string(features.getMaxExternalReadTokens())
                : std::string();
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-master-admission-prediction] schedule_time_us=%ld "
            "schedule_id=%lu schedule_batch_id=%ld request_id=%lu "
            "request_count=%ld agent=%s collector=%s unified=%d "
            "peer_cache_read_available=%d peer_cache_read_revision=%lu "
            "peer_cache_read_agent_start_time_us=%ld "
            "local_hit_tokens=%ld peer_extra_hit_tokens=%ld total_hit_tokens=%ld "
            "model_revision=%ld session_sample_count=%lu trusted=%d "
            "residual_version=%d residual_last_update_time_us=%ld "
            "base_execution_ms=%.9f residual_correction_ms=%.9f "
            "final_execution_ms=%.9f decision_execution_ms=%.9f "
            "queue_prediction_ms=%.9f queue_prediction_time_us=%ld "
            "queue_report_receive_time_us=%ld queue_snapshot_total_ms=%.9f "
            "queue_snapshot_head_remaining_ms=%.9f with_base_ms=%.9f with_final_ms=%.9f "
            "floor_base_ms=%.9f floor_final_ms=%.9f has_floor=%d "
            "fixed_execution_ms=%.9f running_elapsed_time_ms=%.9f "
            "reported_running_elapsed_time_ms=%.9f "
            "advanced_running_elapsed_time_ms=%.9f report_age_ms=%.9f "
            "workload_revision=%lu workload_event_time_us=%ld "
            "workload_report_receive_time_us=%ld "
            "running_prefill_request_count=%d decay_branch=%s "
            "batch_count=%ld batch_size=%ld "
            "preceding_request_count=%ld hit_tokens=%ld miss_tokens=%ld "
            "miss_square_sum=%.0f hit_miss_sum=%.0f capacity_split=%d "
            "running_batch_size=%ld running_hit_tokens=%ld "
            "running_miss_tokens=%ld running_miss_square_sum=%.0f "
            "running_hit_miss_sum=%.0f "
            "running_decode_request_count=%ld "
            "running_decode_generated_tokens=%ld "
            "running_decode_remaining_token_budget=%ld "
            "reported_running_decode_request_count=%ld "
            "reported_running_decode_generated_tokens=%ld "
            "reported_running_decode_remaining_token_budget=%ld "
            "waiting_batch_size=%ld "
            "waiting_hit_tokens=%ld waiting_miss_tokens=%ld "
            "waiting_miss_square_sum=%.0f waiting_hit_miss_sum=%.0f "
            "floor_batch_count=%ld floor_batch_size=%ld "
            "floor_preceding_request_count=%ld floor_hit_tokens=%ld "
            "floor_miss_tokens=%ld floor_miss_square_sum=%.0f "
            "floor_hit_miss_sum=%.0f floor_capacity_split=%d "
            "floor_running_batch_size=%ld floor_running_hit_tokens=%ld "
            "floor_running_miss_tokens=%ld "
            "floor_running_miss_square_sum=%.0f "
            "floor_running_hit_miss_sum=%.0f "
            "floor_running_decode_request_count=%ld "
            "floor_running_decode_generated_tokens=%ld "
            "floor_running_decode_remaining_token_budget=%ld "
            "floor_waiting_batch_size=%ld floor_waiting_hit_tokens=%ld "
            "floor_waiting_miss_tokens=%ld "
            "floor_waiting_miss_square_sum=%.0f "
            "floor_waiting_hit_miss_sum=%.0f%s",
            decision.scheduleTimeUs, decision.scheduleId, decision.batchId,
            decision.requestId, decision.expectedBatchSize, node.getAgentId().c_str(),
            decision.target.agentAddress.c_str(),
            node.usesUnifiedLatencyPrediction() ? 1 : 0,
            int(node._peerCacheReadAvailable), node._peerCacheReadRevision,
            node.getAgentStartTimeUs(), localHitTokens, peerExtraTokens,
            decision.predictedCacheHitTokenCount, node.getLatencyModelRevision(),
            node.getLatencySessionSampleCount(), node.isLatencyModelTrusted() ? 1 : 0,
            node.getLatencyResidualVersion(), node.getLatencyResidualLastUpdateTimeUs(),
            audit.baseExecutionMs, audit.correctionMs, audit.finalExecutionMs,
            decision.predictedExecutionLatencyMs, decision.predictedQueueLatencyMs,
            context._queuePredictionTimeUs, node._queueReportReceiveTimeUs,
            node._queueTotalPredictedLatencyMs, node._queueDecayHeadRemainingMs,
            audit.withWorkloadBaseMs, audit.withWorkloadFinalMs, audit.floorBaseMs,
            audit.floorFinalMs, prediction.hasFloor ? 1 : 0,
            prediction.fixedLatencyMs - decision.predictedQueueLatencyMs,
            features.getRunningElapsedTimeMs(), node.getReportedRunningElapsedTimeMs(),
            node.getReportedWorkloadFeatures().getRunningElapsedTimeMs(),
            node.getWorkloadReportAgeMs(), node.getWorkloadRevision(),
            node.getWorkloadEventTimeUs(), node.getWorkloadReportReceiveTimeUs(),
            node.getRunningCapacityPrefillRequestCount(), decayBranch,
            features.getBatchCount(), features.getBatchSize(),
            features.getPrecedingRequestCount(), features.getCacheHitTokenCount(),
            features.getCacheMissTokenCount(), features.getCacheMissTokenSquareSum(),
            features.getCacheHitMissTokenProductSum(),
            features.usesCapacitySplit() ? 1 : 0, features.getRunningBatchSize(),
            features.getRunningCacheHitTokenCount(),
            features.getRunningCacheMissTokenCount(),
            features.getRunningCacheMissTokenSquareSum(),
            features.getRunningCacheHitMissTokenProductSum(),
            features.getRunningDecodeRequestCount(),
            features.getRunningDecodeGeneratedTokenCount(),
            features.getRunningDecodeRemainingTokenBudget(),
            node.getReportedWorkloadFeatures().getRunningDecodeRequestCount(),
            node.getReportedWorkloadFeatures().getRunningDecodeGeneratedTokenCount(),
            node.getReportedWorkloadFeatures().getRunningDecodeRemainingTokenBudget(),
            features.getWaitingBatchSize(), features.getWaitingCacheHitTokenCount(),
            features.getWaitingCacheMissTokenCount(),
            features.getWaitingCacheMissTokenSquareSum(),
            features.getWaitingCacheHitMissTokenProductSum(), floor.getBatchCount(),
            floor.getBatchSize(), floor.getPrecedingRequestCount(),
            floor.getCacheHitTokenCount(), floor.getCacheMissTokenCount(),
            floor.getCacheMissTokenSquareSum(), floor.getCacheHitMissTokenProductSum(),
            floor.usesCapacitySplit() ? 1 : 0, floor.getRunningBatchSize(),
            floor.getRunningCacheHitTokenCount(), floor.getRunningCacheMissTokenCount(),
            floor.getRunningCacheMissTokenSquareSum(),
            floor.getRunningCacheHitMissTokenProductSum(),
            floor.getRunningDecodeRequestCount(),
            floor.getRunningDecodeGeneratedTokenCount(),
            floor.getRunningDecodeRemainingTokenBudget(), floor.getWaitingBatchSize(),
            floor.getWaitingCacheHitTokenCount(), floor.getWaitingCacheMissTokenCount(),
            floor.getWaitingCacheMissTokenSquareSum(),
            floor.getWaitingCacheHitMissTokenProductSum(), geometrySuffix.c_str());
        if (std::abs(audit.finalExecutionMs - decision.predictedExecutionLatencyMs) >
            1e-6) {
            NAVI_KERNEL_LOG(
                ERROR,
                "Master admission prediction audit mismatch, scheduleId [%lu], "
                "scheduleBatchId [%ld], audited [%.9f], decision [%.9f]",
                decision.scheduleId, decision.batchId, audit.finalExecutionMs,
                decision.predictedExecutionLatencyMs);
        }
    }
}

void MasterScheduleResultKernel::reportPredictionMetrics(
    const NaviSchedScheduleContext &context) const
{
    if (!_metricsReporterR || !_metricsReporterR->getReporter()) {
        return;
    }
    const auto &reporter = _metricsReporterR->getReporter();
    const auto report = [&reporter](const char *name, double value) {
        reporter->report(value, name, kmonitor::MetricType::GAUGE, nullptr,
                         kmonitor::MetricLevel::FATAL);
    };
    report("master.schedule_compute_ms", context._scheduleLatencyUs / 1000.0);
    const auto reportStage = [&report](const char *name, int64_t elapsedUs) {
        if (elapsedUs >= 0) report(name, elapsedUs / 1000.0);
    };
    reportStage("master.schedule_prepare_ms", context._masterPrepareUs);
    reportStage("master.schedule_snapshot_ms", context._snapshotPrepareUs);
    reportStage("master.schedule_snapshot_runtime_lock_wait_ms",
                context._snapshotRuntimeLockWaitUs);
    reportStage("master.schedule_snapshot_runtime_lock_hold_ms",
                context._snapshotRuntimeLockHoldUs);
    if (context.isCollectingReplay()) {
        reportStage("master.schedule_replay_prepare_ms", context._replayPrepareUs);
    }
    reportStage("master.schedule_admission_ms", context._admissionPrepareUs);
    reportStage("master.schedule_context_prepare_ms", context._contextPrepareUs);
    reportStage("master.cost_prepare_ms", context._costPrepareUs);
    if (context._costGradientIterationCount > 0) {
        report("master.cost_optimizer_gradient_iteration_ms",
               context._costGradientIterationTotalUs /
                   (1000.0 * context._costGradientIterationCount));
        reportStage("master.cost_optimizer_gradient_iteration_max_ms",
                    context._costGradientIterationMaxUs);
    }
    if (context._costOptimizerLoopCount >= 0) {
        report("master.cost_optimizer_loop_count", context._costOptimizerLoopCount);
        report("master.cost_optimizer_max_loop_count",
               context._costOptimizerMaxLoopCount);
        report("master.cost_optimizer_time_budget_exhausted",
               context._costOptimizerTimeBudgetExhausted ? 1 : 0);
        if (context._costOptimizerTimeBudgetExhausted) {
            reporter->report(1, "master.cost_optimizer_time_budget_stop_qps",
                             kmonitor::MetricType::QPS, nullptr,
                             kmonitor::MetricLevel::FATAL);
        }
    }
    if (context.isCollectingReplay()) {
        reportStage("master.schedule_replay_collection_ms",
                    context.getReplayCollectionTimeUs());
    }
    if (context._discreteSearchLosses) {
        const auto [initialLoss, finalLoss] = *context._discreteSearchLosses;
        report("master.discrete_search_initial_loss", initialLoss);
        report("master.discrete_search_final_loss", finalLoss);
        if (const auto improvement = context.discreteSearchImprovementPct()) {
            report("master.discrete_search_improvement_pct", *improvement);
        }
    }
    if (context._discreteSearchElapsedUs >= 0) {
        report("master.discrete_search_ms", context._discreteSearchElapsedUs / 1000.0);
        report("master.discrete_search_evaluations", context._discreteSearchEvaluations);
        report("master.discrete_search_moves", context._discreteSearchMoves);
        report("master.discrete_search_effective",
               context._discreteSearchMoves > 0 ? 1 : 0);
        reporter->report(1, "master.discrete_search_qps", kmonitor::MetricType::QPS,
                         nullptr, kmonitor::MetricLevel::FATAL);
        if (context._discreteSearchMoves > 0) {
            reporter->report(1, "master.discrete_search_effective_qps",
                             kmonitor::MetricType::QPS, nullptr,
                             kmonitor::MetricLevel::FATAL);
        }
    }
    report(
        "master.schedule_snapshot_to_result_ms",
        std::max<int64_t>(0, context._queuePredictionTimeUs - context._scheduleTimeUs) /
            1000.0);
    for (const auto &request : context._requests) {
        report("master.schedule_request_wait_ms", request.waitTimeMs);
    }
    std::unordered_set<int64_t> batches;
    const NaviSchedScheduleNode *oldestNode = nullptr;
    int64_t oldestReportUs = 0;
    int64_t oldestAgeUs = 0;
    for (size_t index = 0; index < context._decisions.size(); ++index) {
        const auto &decision = context._decisions[index];
        if (decision.batchId <= 0 || !batches.insert(decision.batchId).second ||
            index >= context._selectedNodeIndexes.size() ||
            context._selectedNodeIndexes[index] >= context._scheduleNodes.size()) {
            continue;
        }
        const auto &node = context._scheduleNodes[context._selectedNodeIndexes[index]];
        const auto reportReceiveTimeUs = node.usesUnifiedLatencyPrediction()
                                             ? node._workloadReportReceiveTimeUs
                                             : node._queueReportReceiveTimeUs;
        if (reportReceiveTimeUs > 0) {
            const auto ageUs = std::max<int64_t>(
                0, context._queuePredictionTimeUs - reportReceiveTimeUs);
            const auto ageAtStartUs =
                std::max<int64_t>(0, context._scheduleTimeUs - reportReceiveTimeUs);
            report("master.queue_prediction_snapshot_age_ms", ageUs / 1000.0);
            report("master.queue_prediction_snapshot_age_at_schedule_start_ms",
                   ageAtStartUs / 1000.0);
            report("master.queue_prediction_snapshot_age_during_schedule_ms",
                   (ageUs - ageAtStartUs) / 1000.0);
            if (ageUs > oldestAgeUs) {
                oldestAgeUs = ageUs;
                oldestReportUs = reportReceiveTimeUs;
                oldestNode = &node;
            }
        }
        if (node.usesUnifiedLatencyPrediction()) {
            continue;
        }
        const auto before = node.getEstimatedQueueWaitMs();
        report("master.queue_prediction_at_schedule_start_ms", before);
        report("master.queue_prediction_at_result_ms", decision.predictedQueueLatencyMs);
        report("master.queue_prediction_decay_during_schedule_ms",
               std::max(0.0, before - decision.predictedQueueLatencyMs));
        report("master.queue_prediction_snapshot_head_remaining_ms",
               node._queueDecayHeadRemainingMs);
        report("master.scheduled_batch_backlog_pct", before > 0.0 ? 100.0 : 0.0);
    }
    if (oldestNode && oldestAgeUs >= 1'000'000 &&
        (context._scheduleId < 10 || context._scheduleId % 100 == 0)) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-master-prediction-age] schedule_id=%lu agent=%s unified=%d "
            "report_receive_time_us=%ld schedule_time_us=%ld prediction_time_us=%ld "
            "age_at_start_us=%ld schedule_elapsed_us=%ld total_age_us=%ld "
            "workload_revision=%lu workload_event_time_us=%ld",
            context._scheduleId, oldestNode->_info->agentId.c_str(),
            int(oldestNode->usesUnifiedLatencyPrediction()), oldestReportUs,
            context._scheduleTimeUs, context._queuePredictionTimeUs,
            std::max<int64_t>(0, context._scheduleTimeUs - oldestReportUs),
            context._queuePredictionTimeUs - context._scheduleTimeUs, oldestAgeUs,
            oldestNode->getWorkloadRevision(), oldestNode->getWorkloadEventTimeUs());
    }
}

void MasterScheduleResultKernel::reportResultMetrics(
    const NaviSchedScheduleContext &context) const
{
    if (!_metricsReporterR || !_metricsReporterR->getReporter()) {
        return;
    }
    const auto &reporter = _metricsReporterR->getReporter();
    const auto report = [&reporter](const char *name, int64_t elapsedUs) {
        if (elapsedUs >= 0) {
            reporter->report(elapsedUs / 1000.0, name, kmonitor::MetricType::GAUGE,
                             nullptr, kmonitor::MetricLevel::FATAL);
        }
    };
    report("master.schedule_result_prepare_ms", context._resultPrepareUs);
    report("master.schedule_result_audit_ms", context._resultAuditUs);
    report("master.schedule_result_reserve_ms", context._resultReserveUs);
    report("master.schedule_result_report_replay_ms", context._resultReportReplayUs);
    report("master.schedule_result_delivery_ms", context._resultDeliveryUs);
    report("master.schedule_result_replay_log_ms", context._resultReplayLogUs);
    report("master.schedule_result_failure_cleanup_ms", context._resultFailureCleanupUs);
    report("master.schedule_result_partition_ms", context._resultPartitionUs);
    report("master.schedule_result_failure_collect_ms", context._resultFailureCollectUs);
    report("master.schedule_result_merge_ms", context._resultMergeUs);
    report("master.schedule_result_prediction_report_ms",
           context._resultPredictionReportUs);
    report("master.schedule_result_reserve_stage_ms", context._resultReserveStageUs);
    report("master.schedule_result_delivery_stage_ms", context._resultDeliveryStageUs);
    report("master.schedule_result_cleanup_stage_ms", context._resultCleanupStageUs);
    report("master.schedule_result_reserve_dispatch_delay_ms",
           context._resultReserveDispatchDelayUs);
    report("master.schedule_result_delivery_dispatch_delay_ms",
           context._resultDeliveryDispatchDelayUs);
    report("master.schedule_result_cleanup_dispatch_delay_ms",
           context._resultCleanupDispatchDelayUs);
    report("master.schedule_result_finish_ms", context._resultFinishUs);
    report("master.schedule_round_total_ms", context._roundTotalUs);
}

void MasterScheduleResultKernel::reportResultTiming(NaviSchedScheduleContext &context,
                                                    int64_t resultBeginUs) const
{
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    context._resultFinishUs = endUs - resultBeginUs;
    if (context._scheduleBeginTimeUs > 0) {
        context._roundTotalUs = endUs - context._scheduleBeginTimeUs;
    }
    reportResultMetrics(context);
    if (context._scheduleId < 10 || context._scheduleId % 100 == 0 ||
        context._resultFinishUs >= 100'000) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-result-timing] schedule_id=%lu requests=%lu "
                        "prepare_us=%ld audit_us=%ld reserve_us=%ld report_replay_us=%ld "
                        "delivery_us=%ld replay_log_us=%ld failure_cleanup_us=%ld "
                        "result_finish_us=%ld "
                        "round_total_us=%ld collect_replay=%d failed_deliveries=%lu "
                        "result_workers=%lu "
                        "delivery_workers=%lu",
                        context._scheduleId, context.requestCount(),
                        context._resultPrepareUs, context._resultAuditUs,
                        context._resultReserveUs, context._resultReportReplayUs,
                        context._resultDeliveryUs, context._resultReplayLogUs,
                        context._resultFailureCleanupUs, context._resultFinishUs,
                        context._roundTotalUs, static_cast<int>(bool(context._replay)),
                        context._responseDeliveryFailedIndexes.size(),
                        context._resultWorkerCount, context._resultDeliveryWorkerCount);
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-result-stage-timing] schedule_id=%lu "
                        "partition_us=%ld reserve_stage_us=%ld delivery_stage_us=%ld "
                        "cleanup_stage_us=%ld failure_collect_us=%ld merge_us=%ld "
                        "prediction_report_us=%ld reserve_dispatch_delay_us=%ld "
                        "delivery_dispatch_delay_us=%ld cleanup_dispatch_delay_us=%ld",
                        context._scheduleId, context._resultPartitionUs,
                        context._resultReserveStageUs, context._resultDeliveryStageUs,
                        context._resultCleanupStageUs, context._resultFailureCollectUs,
                        context._resultMergeUs, context._resultPredictionReportUs,
                        context._resultReserveDispatchDelayUs,
                        context._resultDeliveryDispatchDelayUs,
                        context._resultCleanupDispatchDelayUs);
    }
}

bool MasterScheduleResultKernel::prepareContext(NaviSchedScheduleContext &context,
                                                int64_t resultBeginUs)
{
    if (!context.prepareResult(autil::TimeUtility::currentTime())) {
        context._resultPrepareUs = autil::TimeUtility::monotonicTimeUs() - resultBeginUs;
        if (context._replay) {
            context._replay->decisions = context._decisions;
            auto &schedulerInfo = context._replay->schedulerInfo;
            schedulerInfo["audit_version"] = int64_t(1);
            schedulerInfo["schedule_id"] = context._scheduleId;
            schedulerInfo["schedule_begin_time_us"] = context._scheduleBeginTimeUs;
            schedulerInfo["schedule_latency_us"] = context._scheduleLatencyUs;
            schedulerInfo["finish_error"] = std::string("prepare_result_failed");
            NAVI_KERNEL_LOG(
                DEBUG, "[navi-sched-replay] schedule_id=%lu json=%s", context._scheduleId,
                NaviSchedScheduleReplayJson::toJsonString(*context._replay).c_str());
        }
        return false;
    }
    const auto preparedUs = autil::TimeUtility::monotonicTimeUs();
    context._resultPrepareUs = preparedUs - resultBeginUs;
    logAdmissionPredictions(context);
    _masterScheduleR->recordColdStartRouting(context._coldStartRoutingStats);
    context.attachLatencyModelGlobals(*_masterLatencyModelR);
    context._scheduleRequestCompletion.reset();
    context._resultAuditUs = autil::TimeUtility::monotonicTimeUs() - preparedUs;
    return true;
}

void MasterScheduleResultKernel::recordReplayAdmission(NaviSchedScheduleContext &context)
{
    if (context._replay) {
        auto &schedulerInfo = context._replay->schedulerInfo;
        schedulerInfo["audit_version"] = int64_t(1);
        schedulerInfo["admission_after_cold_start"] =
            context._coldStartRoutingStats.evaluated &&
            !context._allColdUtilizationAdmission;
        schedulerInfo["admission_all_cold_utilization"] =
            context._allColdUtilizationAdmission;
        schedulerInfo["admission_execution_utilization"] =
            context._executionUtilizationAdmission;
        schedulerInfo["schedule_id"] = context._scheduleId;
        schedulerInfo["schedule_begin_time_us"] = context._scheduleBeginTimeUs;
        schedulerInfo["schedule_latency_us"] = context._scheduleLatencyUs;
        schedulerInfo["time_compression"] = context._timeCompression;
        if (context._queuePredictionTimeUs >= context._scheduleTimeUs) {
            schedulerInfo["queue_prediction_time_us"] = context._queuePredictionTimeUs;
        }
        schedulerInfo["cold_start_target_inflight"] =
            context._coldStartRoutingStats.targetInflight;
        schedulerInfo["cold_start_trusted_node_count"] =
            context._coldStartRoutingStats.trustedNodeCount;
        schedulerInfo["cold_start_untrusted_node_count"] =
            context._coldStartRoutingStats.untrustedNodeCount;
        schedulerInfo["cold_start_trusted_request_count"] =
            context._coldStartRoutingStats.trustedRequestCount;
        schedulerInfo["cold_start_untrusted_request_count"] =
            context._coldStartRoutingStats.untrustedRequestCount;
        schedulerInfo["cold_start_rejected_request_count"] =
            context._coldStartRoutingStats.rejectedRequestCount;
        schedulerInfo["cold_start_untrusted_capacity"] =
            context._coldStartRoutingStats.untrustedCapacity;
        schedulerInfo["cold_start_ramping_node_count"] =
            context._coldStartRoutingStats.rampingNodeCount;
        schedulerInfo["cold_start_unsafe_waiting_marginal_node_count"] =
            context._coldStartRoutingStats.unsafeWaitingMarginalNodeCount;
        schedulerInfo["cold_start_minimum_ramp_progress"] =
            context._coldStartRoutingStats.minimumRampProgress;
        schedulerInfo["cold_start_minimum_ramped_target_inflight"] =
            context._coldStartRoutingStats.minimumRampedTargetInflight;
        schedulerInfo["cold_start_evaluated"] = context._coldStartRoutingStats.evaluated;
        schedulerInfo["queue_overload_evaluated"] = context._queueOverloadEvaluated;
        schedulerInfo["queue_overload_average_predicted_latency_ms"] =
            context._queueOverloadAveragePredictedLatencyMs;
        schedulerInfo["queue_overload_drop_probability"] =
            context._queueOverloadDropProbability;
        schedulerInfo["queue_overload_schedule_cycle_ms"] =
            context._queueOverloadScheduleCycleMs;
        schedulerInfo["queue_overload_start_cycle_multiplier"] =
            context._queueOverloadStartCycleMultiplier;
        schedulerInfo["queue_overload_full_cycle_multiplier"] =
            context._queueOverloadFullCycleMultiplier;
        schedulerInfo["queue_overload_request_count"] =
            int64_t(context._queueOverloadRequestCount);
        schedulerInfo["queue_overload_rejected_request_count"] =
            int64_t(context._queueOverloadRejectedRequestCount);
        autil::legacy::json::JsonArray randomDecisions;
        randomDecisions.reserve(context._coldStartRoutingStats.randomDecisions.size());
        for (const auto &decision : context._coldStartRoutingStats.randomDecisions) {
            autil::legacy::json::JsonMap randomDecision;
            randomDecision["request_index"] = int64_t(decision.requestIndex);
            randomDecision["sample"] = std::to_string(decision.sample);
            randomDecision["probability"] = decision.probability;
            randomDecision["selected"] = decision.selected;
            randomDecisions.emplace_back(std::move(randomDecision));
        }
        schedulerInfo["cold_start_random_decisions"] = std::move(randomDecisions);
        autil::legacy::json::JsonArray targetRandomDecisions;
        targetRandomDecisions.reserve(
            context._coldStartRoutingStats.targetRandomDecisions.size());
        for (const auto &decision :
             context._coldStartRoutingStats.targetRandomDecisions) {
            autil::legacy::json::JsonMap record;
            record["node_index"] = int64_t(decision.nodeIndex);
            record["target_inflight"] = decision.targetInflight;
            record["admitted_target_inflight"] = decision.admittedTargetInflight;
            record["sample"] = std::to_string(decision.sample);
            record["probability"] = decision.probability;
            record["rounded_up"] = decision.roundedUp;
            targetRandomDecisions.emplace_back(std::move(record));
        }
        schedulerInfo["cold_start_target_random_decisions"] =
            std::move(targetRandomDecisions);
        autil::legacy::json::JsonArray queueOverloadRandomDecisions;
        queueOverloadRandomDecisions.reserve(
            context._queueOverloadRandomDecisions.size());
        for (const auto &decision : context._queueOverloadRandomDecisions) {
            autil::legacy::json::JsonMap randomDecision;
            randomDecision["request_index"] = int64_t(decision.requestIndex);
            randomDecision["sample"] = std::to_string(decision.sample);
            randomDecision["probability"] = decision.probability;
            randomDecision["accepted"] = decision.accepted;
            queueOverloadRandomDecisions.emplace_back(std::move(randomDecision));
        }
        schedulerInfo["queue_overload_random_decisions"] =
            std::move(queueOverloadRandomDecisions);
        autil::legacy::json::JsonArray nodeRuntime;
        nodeRuntime.reserve(context._scheduleNodes.size());
        for (size_t nodeIndex = 0; nodeIndex < context._scheduleNodes.size();
             ++nodeIndex) {
            const auto &node = context._scheduleNodes[nodeIndex];
            autil::legacy::json::JsonMap runtime;
            runtime["node_index"] = int64_t(nodeIndex);
            runtime["latency_session_sample_count"] =
                int64_t(node.getLatencySessionSampleCount());
            runtime["agent_start_time_us"] = node.getAgentStartTimeUs();
            runtime["ramp_finished"] = CostSchedulerColdStartStrategy::isRampFinished(
                context._scheduleTimeUs, node.getAgentStartTimeUs(),
                node.getLatencySessionSampleCount());
            runtime["backlog_prediction_trusted"] = node.isBacklogPredictionTrusted();
            runtime["estimated_backlog_latency_ms"] = node.getEstimatedBacklogLatencyMs();
            runtime["reported_running_elapsed_time_ms"] =
                node.getReportedRunningElapsedTimeMs();
            runtime["workload_report_age_ms"] = node.getWorkloadReportAgeMs();
            const auto &residualConfig = node.getLatencyResidualRuntimeConfig();
            runtime["residual_learning_rate"] = residualConfig.learningRate;
            runtime["residual_update_error_floor_ms"] = residualConfig.updateErrorFloorMs;
            runtime["residual_update_error_ratio"] = residualConfig.updateErrorRatio;
            runtime["residual_idle_decay_start_us"] = residualConfig.idleDecayStartUs;
            runtime["residual_idle_decay_end_us"] = residualConfig.idleDecayEndUs;
            nodeRuntime.emplace_back(std::move(runtime));
        }
        schedulerInfo["replay_node_runtime"] = std::move(nodeRuntime);
    }
}

bool MasterScheduleResultKernel::finishContext(NaviSchedScheduleContext &context)
{
    const auto resultBeginUs = autil::TimeUtility::monotonicTimeUs();
    autil::ScopeGuard reportTimings([this, &context, resultBeginUs]() {
        reportResultTiming(context, resultBeginUs);
    });
    if (!prepareContext(context, resultBeginUs)) {
        return false;
    }
    const auto auditEndUs = autil::TimeUtility::monotonicTimeUs();
    _masterStateR->reserveScheduleBatches(context);
    const auto reservedUs = autil::TimeUtility::monotonicTimeUs();
    context._resultReserveUs = reservedUs - auditEndUs;
    reportPredictionMetrics(context);
    context._resultPredictionReportUs =
        autil::TimeUtility::monotonicTimeUs() - reservedUs;
    recordReplayAdmission(context);
    if (context._replay) {
        context._replay->decisions = context._decisions;
    }
    const auto deliveryBeginUs = autil::TimeUtility::monotonicTimeUs();
    context._resultReportReplayUs = deliveryBeginUs - reservedUs;
    const auto delivered = context.finish();
    const auto deliveryEndUs = autil::TimeUtility::monotonicTimeUs();
    context._resultDeliveryUs = deliveryEndUs - deliveryBeginUs;
    if (context._replay) {
        NAVI_KERNEL_LOG(
            DEBUG, "[navi-sched-replay] schedule_id=%lu json=%s", context._scheduleId,
            NaviSchedScheduleReplayJson::toJsonString(*context._replay).c_str());
    }
    const auto replayLogEndUs = autil::TimeUtility::monotonicTimeUs();
    context._resultReplayLogUs = replayLogEndUs - deliveryEndUs;
    std::vector<MasterDispatchHandle> failedHandles;
    failedHandles.reserve(context._responseDeliveryFailedIndexes.size());
    for (auto requestIndex : context._responseDeliveryFailedIndexes) {
        if (requestIndex < context._dispatchHandles.size()) {
            failedHandles.push_back(context._dispatchHandles[requestIndex]);
        }
    }
    MasterDispatchHandle::fail(failedHandles);
    context._resultFailureCleanupUs =
        autil::TimeUtility::monotonicTimeUs() - replayLogEndUs;
    return delivered;
}

navi::ErrorCode MasterScheduleResultKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::GroupDatas inputs;
    if (!ctx.getGroupInput(0, inputs)) {
        return navi::EC_NONE;
    }
    std::shared_ptr<MasterScheduleResultState> state;
    for (const auto &input : inputs) {
        if (!input.hasValue || !input.data) {
            continue;
        }
        const auto data = std::dynamic_pointer_cast<MasterScheduleResultData>(input.data);
        if (!data || !data->state || (state && state != data->state)) {
            return navi::EC_ABORT;
        }
        state = data->state;
    }
    if (state) {
        if (!state->serialFinished) {
            state->merge();
            const auto reportBeginUs = autil::TimeUtility::monotonicTimeUs();
            reportPredictionMetrics(*state->context);
            state->context->_resultPredictionReportUs =
                autil::TimeUtility::monotonicTimeUs() - reportBeginUs;
            reportResultTiming(*state->context, state->beginUs);
            for (const auto &worker : state->workers) {
                if (!worker.success) {
                    NAVI_KERNEL_LOG(WARN,
                                    "finish parallel master schedule result failed");
                    break;
                }
            }
        }
        state->completion.reset();
    }
    if (inputs.eof()) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(MasterScheduleContextKernel);
REGISTER_KERNEL(MasterScheduleResultKernel);

}
