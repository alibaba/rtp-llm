#include "aios/navi_sched/scheduler/MasterSchedulePrepareKernel.h"
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
#include <optional>
#include <tuple>
#include <unordered_set>
#include <unordered_map>

namespace navi_sched {
const std::string MasterSchedulePrepareKernel::KERNEL_ID = "navi_sched.master_schedule_prepare_k";

MasterSchedulePrepareKernel::MasterSchedulePrepareKernel()
    : _randomGenerator(autil::TimeUtility::currentTimeInMicroSeconds())
{
    (void)_randomGenerator.get();
}

void MasterSchedulePrepareKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).input("queried_context", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
        .output("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}

bool MasterSchedulePrepareKernel::config(navi::KernelConfigContext &ctx)
{
    const auto attrs = ctx.getBinaryAttrs();
    const auto scheduler = attrs.find("scheduler_kernel_id");
    _schedulerKernelId = scheduler == attrs.end() ? std::string() : scheduler->second;
    NAVI_JSONIZE(ctx, "max_wait_schedule_time_ms", _maxWaitScheduleTimeMs, _maxWaitScheduleTimeMs);
    return _maxWaitScheduleTimeMs >= 10;
}

void MasterSchedulePrepareKernel::useConfig(const SchedulerConfigR::Snapshot &snapshot)
{
    const auto &config = snapshot.config;
    _executionUtilizationLimitEnabled = config.overloadControlStrategy == "execution_utilization";
    _useMinimumBacklogForAdmission = config.overloadControlStrategy == "min_latency";
    _useSelectedQueueForAdmission = config.overloadControlStrategy == "post_cost_latency";
    if (_executionUtilizationConfig.globalScaling != config.executionUtilization.globalScaling)
        _executionUtilizationController.reset();
    _executionUtilizationConfig = config.executionUtilization;
    _maxWaitScheduleTimeMs = config.maxWaitScheduleTimeMs;
    _queueOverloadStartCycleMultiplier = config.queueOverloadStartCycleMultiplier;
    _queueOverloadFullCycleMultiplier = config.queueOverloadFullCycleMultiplier;
}

navi::ErrorCode MasterSchedulePrepareKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr input;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof))
        return navi::EC_NONE;
    if (!input) {
        return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, eof)
                   ? navi::EC_NONE : navi::EC_ABORT;
    }
    auto data = std::dynamic_pointer_cast<MasterScheduleContextData>(input);
    if (!data || !data->context) return navi::EC_ABORT;
    if (!prepareContext(data)) {
        if (eof && !ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true))
            return navi::EC_ABORT;
        return navi::EC_NONE;
    }
    return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)
               ? navi::EC_NONE : navi::EC_ABORT;
}

bool MasterSchedulePrepareKernel::prepareContext(
    const std::shared_ptr<MasterScheduleContextData> &data)
{
    if (data->schedulerConfig) useConfig(*data->schedulerConfig);
    auto &context = *data->context;
    for (const auto &request : context._responseRequests) {
        if (!request || !request->request || !request->request->collectReplay) continue;
        const auto beginUs = autil::TimeUtility::monotonicTimeUs();
        context.setReplay(_masterStateR->makeScheduleReplay(context));
        context._replayPrepareUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
        context.addReplayCollectionTimeUs(context._replayPrepareUs);
        break;
    }
    const auto admissionBeginUs = autil::TimeUtility::monotonicTimeUs();
    const auto accepted = prepareAdmission(data);
    const auto admissionEndUs = autil::TimeUtility::monotonicTimeUs();
    context._admissionPrepareUs =
        admissionEndUs - admissionBeginUs - context._contextPrepareUs;
    if (!accepted) {
        context._scheduleLatencyUs = std::max<int64_t>(
            0, admissionEndUs - context._scheduleBeginTimeUs);
        MasterScheduleResultKernel::recordReplayAdmission(context);
        if (!context.finish())
            NAVI_KERNEL_LOG(ERROR, "finish fully rate-limited schedule round failed, scheduleId [%lu]",
                            context._scheduleId);
        if (context._replay)
            NAVI_KERNEL_LOG(DEBUG, "[navi-sched-replay] schedule_id=%lu json=%s",
                            context._scheduleId,
                            NaviSchedScheduleReplayJson::toJsonString(*context._replay).c_str());
        return false;
    }
    _masterStateR->recordScheduleRound(scheduleRoundRequestCount(context));
    if (!context._prepared && !context.prepare()) context.fail("scheduler_error");
    const auto preparedUs = autil::TimeUtility::monotonicTimeUs();
    context._contextPrepareUs += preparedUs - admissionEndUs;
    context._masterPrepareUs = preparedUs - context._scheduleBeginTimeUs;
    if (context._scheduleId < 10 || context._scheduleId % 100 == 0) {
        NAVI_KERNEL_LOG(
            INFO, "[navi-master-prepare-timing] schedule_id=%lu requests=%lu candidates=%lu "
            "snapshot_us=%ld replay_setup_us=%ld admission_us=%ld context_prepare_us=%ld "
            "activation_to_output_us=%ld snapshot_runtime_lock_wait_us=%ld "
            "snapshot_runtime_lock_hold_us=%ld",
            context._scheduleId, context.requestCount(), context.nodeCount(),
            context._snapshotPrepareUs, context._replayPrepareUs, context._admissionPrepareUs,
            context._contextPrepareUs, context._masterPrepareUs,
            context._snapshotRuntimeLockWaitUs, context._snapshotRuntimeLockHoldUs);
    }
    return true;
}

double MasterSchedulePrepareKernel::calculateQueueOverloadDropProbability(
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

bool MasterSchedulePrepareKernel::shouldDropRequest(
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

size_t MasterSchedulePrepareKernel::scheduleRoundRequestCount(
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

bool MasterSchedulePrepareKernel::prepareAdmission(
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
        if (_useSelectedQueueForAdmission) {
            updateExecutionUtilization(context, false);
            context._queueOverloadAfterCost = true;
            context._queueOverloadScheduleCycleMs = _maxWaitScheduleTimeMs;
            context._queueOverloadStartCycleMultiplier = _queueOverloadStartCycleMultiplier;
            context._queueOverloadFullCycleMultiplier = _queueOverloadFullCycleMultiplier;
            return true;
        }
    }
    return applyQueueOverloadLimit(contextData, randomDraw);
}

void MasterSchedulePrepareKernel::updateExecutionUtilization(
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

bool MasterSchedulePrepareKernel::applyQueueOverloadLimit(
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
    context._queueOverloadAdmissionPredictedLatencyMs = 0.0;
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
    const auto minimumPredictedLatencyMs = coldStartAnalysis.minimumTrustedBacklogLatencyMs;
    const auto admissionPredictedLatencyMs =
        _useMinimumBacklogForAdmission && std::isfinite(minimumPredictedLatencyMs)
            ? minimumPredictedLatencyMs
            : averagePredictedLatencyMs;
    auto dropProbability =
        executionUtilizationMode
            ? 0.0
            : calculateQueueOverloadDropProbability(
                  admissionPredictedLatencyMs, _maxWaitScheduleTimeMs,
                  _queueOverloadStartCycleMultiplier, _queueOverloadFullCycleMultiplier);
    if (executionUtilizationMode) {
        dropProbability = _executionUtilizationController.probability();
        if (dropProbability < 0.001) dropProbability = 0.0;
    }
    context._queueOverloadEvaluated = true;
    context._queueOverloadAveragePredictedLatencyMs = averagePredictedLatencyMs;
    context._queueOverloadAdmissionPredictedLatencyMs = admissionPredictedLatencyMs;
    if (context._replay && _useMinimumBacklogForAdmission && !executionUtilizationMode) {
        context._replay->schedulerInfo["queue_overload_minimum_latency"] = true;
    }
    context._queueOverloadDropProbability = dropProbability;
    const auto requestCount = context.requestCount();
    if (dropProbability <= 0.0) {
        _masterScheduleR->recordQueueOverload(requestCount, 0, averagePredictedLatencyMs,
                                              dropProbability);
        return true;
    }

    const bool compactNativeInputs =
        !routed && std::any_of(context._scheduleNodes.begin(),
                               context._scheduleNodes.end(), [](const auto &node) {
                                   return NonLinearPrefillModelR::usesNativePP4ForVersion(
                                       node.getLatencyBinaryVersion());
                               });
    std::vector<NaviSchedScheduleItem> acceptedRequests;
    std::vector<MasterScheduleRequestDataPtr> acceptedResponseRequests;
    std::vector<size_t> acceptedIndexes;
    if (routed) {
        acceptedIndexes.reserve(requestCount);
    } else {
        acceptedRequests.reserve(requestCount);
        acceptedResponseRequests.reserve(requestCount);
        if (compactNativeInputs) acceptedIndexes.reserve(requestCount);
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
            } else if (compactNativeInputs) {
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
                admissionPredictedLatencyMs;
            continue;
        }
        NaviSchedScheduleContext::DeferredResponse deferred;
        deferred.request = std::move(responseRequest);
        deferred.decision.requestId = request.request.requestId;
        deferred.decision.scheduleId = context._scheduleId;
        deferred.decision.scheduleTimeUs = context._scheduleTimeUs;
        deferred.decision.reason = "rate_limited";
        deferred.decision.predictedQueueLatencyMs = admissionPredictedLatencyMs;
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
        if (compactNativeInputs) {
            context.retainNativeCacheInputs(acceptedIndexes);
            for (const auto index : acceptedIndexes) {
                acceptedRequests.emplace_back(std::move(context._requests[index]));
                acceptedResponseRequests.emplace_back(
                    std::move(context._responseRequests[index]));
            }
        }
        context._requests.swap(acceptedRequests);
        context._responseRequests.swap(acceptedResponseRequests);
    }
    _masterScheduleR->recordQueueOverload(requestCount, rejectedRequestCount,
                                          averagePredictedLatencyMs, dropProbability);
    context._queueOverloadRejectedRequestCount = rejectedRequestCount;
    NAVI_KERNEL_LOG(DEBUG,
                    "overload limit evaluated, executionUtilizationMode [%d], "
                    "averagePredictedLatencyMs [%.3f], admissionPredictedLatencyMs [%.3f], dropProbability [%.6f], "
                    "requests [%lu], rejected [%lu]",
                    static_cast<int>(executionUtilizationMode), averagePredictedLatencyMs,
                    admissionPredictedLatencyMs, dropProbability, requestCount, rejectedRequestCount);
    return routed || !context._requests.empty();
}


REGISTER_KERNEL(MasterSchedulePrepareKernel);

} // namespace navi_sched
