#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "aios/navi_sched/scheduler/MasterSchedulePrepareKernel.h"

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

const std::string MasterScheduleContextKernel::KERNEL_ID =
    "navi_sched.master_schedule_context_k";
const std::string MasterScheduleResultKernel::KERNEL_ID =
    "navi_sched.master_schedule_result_k";

MasterScheduleContextKernel::MasterScheduleContextKernel() = default;

void MasterScheduleContextKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}

bool MasterScheduleContextKernel::config(navi::KernelConfigContext &ctx)
{
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
        if (!_schedulerConfigSnapshot ||
            _schedulerConfigSnapshot->generation != schedulerConfig.generation)
            _schedulerConfigSnapshot =
                std::make_shared<const SchedulerConfigR::Snapshot>(schedulerConfig);
        _maxWaitScheduleCount = schedulerConfig.config.maxWaitScheduleCount;
        _maxWaitScheduleTimeMs = schedulerConfig.config.maxWaitScheduleTimeMs;
        _timeCompression = schedulerConfig.config.agent.timeCompression;
        _peerCacheEnabled = schedulerConfig.config.peerCacheEnabled;
        _cacheTouchEnabled = schedulerConfig.config.cacheTouchEnabled;
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
            context->context->_cacheTouchEnabled = _cacheTouchEnabled;
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
        context->context->_snapshotPrepareUs = snapshotEndUs - snapshotBeginUs;
        context->schedulerConfig = _schedulerConfigSnapshot;
        context->context->_scheduleRoundCompletion = completion;
        _readyContexts.emplace_back(std::move(context));
    }
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
    if (nodeIndex >= context._scheduleNodes.size()) {
        return audit;
    }
    const auto &node = context._scheduleNodes[nodeIndex];
    if (NonLinearPrefillModelR::usesNativePP4ForVersion(node.getLatencyBinaryVersion())) {
        audit.nativePrediction = std::atomic_load(&node._nativeFinalPrediction);
        if (!audit.nativePrediction ||
            audit.nativePrediction->predictionTimeUs != context._queuePredictionTimeUs) {
            return audit;
        }
        audit.withWorkloadBaseMs = audit.nativePrediction->baseExecutionMs;
        audit.withWorkloadFinalMs = audit.nativePrediction->finalExecutionMs;
        audit.baseExecutionMs = audit.withWorkloadBaseMs;
        audit.finalExecutionMs = audit.withWorkloadFinalMs;
    } else {
        if (!decision.latencyPrediction) return audit;
        const auto &prediction = *decision.latencyPrediction;
        const auto &parameters = node.getLatencyParameters();
        if (!prediction.valid() ||
            !NonLinearPrefillModelR::validateParameters(parameters)) {
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
                         storage < prefixes.size() ? prefixes[storage] : -1,
                         inputs->observation.beforeTairBlockCounts &&
                                 storage < inputs->observation.beforeTairBlockCounts->size()
                             ? inputs->observation.beforeTairBlockCounts->at(storage) *
                                   context._requests[member].request.blockSize
                             : inputs->inputs[storage].computedTokens,
                         inputs->observation.readyOnlyHbmBlockCounts &&
                                 storage < inputs->observation.readyOnlyHbmBlockCounts->size() &&
                                 inputs->observation.readyOnlyHbmBlockCounts->at(storage) >= 0
                             ? inputs->observation.readyOnlyHbmBlockCounts->at(storage) *
                                   context._requests[storage].request.blockSize
                             : -1});
                }
            }
        }
    }
    return audit;
}

void MasterScheduleResultKernel::logAdmissionPredictions(
    const NaviSchedScheduleContext &context, const char *admissionPhase) const
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
        if (nodeIndex >= context._scheduleNodes.size()) {
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
                "[navi-master-frozen-prefill-input] admission_phase=%s schedule_time_us=%ld "
                "schedule_id=%lu schedule_batch_id=%ld request_id=%lu "
                "agent=%s model_revision=%ld prompt_tokens=%ld computed_tokens=%ld "
                "shared_prefix_boundary=%ld external_read_tokens=%ld "
                "cache_observation_valid=%d block_pool_revision=%lu "
                "block_pool_state_ready=%d "
                "block_pool_cached_blocks=%lu block_pool_full_ready_blocks=%lu "
                "block_pool_tail_ready_blocks=%lu block_pool_cached_checksum=%lu "
                "block_pool_full_ready_checksum=%lu block_pool_tail_ready_checksum=%lu "
                "full_prefix_before_query_drop_blocks=%ld "
                "tair_query_status=%s before_tair_hit_tokens=%ld after_tair_hit_tokens=%ld "
                "ready_only_hbm_hit_tokens=%ld",
                admissionPhase, context.getScheduleTimeUs(), decision.scheduleId, decision.batchId,
                entry.requestId, node.getAgentId().c_str(),
                node.getLatencyModelRevision(), input.promptTokens, input.computedTokens,
                input.sharedPrefixBoundary, input.externalReadTokens,
                int(audit.cacheObservation.has_value()), observation.revision,
                int(observation.stateReady), observation.cachedBlocks,
                observation.fullReadyBlocks, observation.tailReadyBlocks,
                observation.cachedChecksum, observation.fullReadyChecksum,
                observation.tailReadyChecksum, entry.fullPrefixBlocksBeforeQueryDrop,
                nodeIndex < context._tairKvcmQueryStatuses.size()
                    ? context._tairKvcmQueryStatuses[nodeIndex].c_str() : "NOT_CONFIGURED",
                entry.beforeTairTokens, input.computedTokens, entry.readyOnlyHbmTokens);
        }
        const auto localCache = std::atomic_load(&node._localCacheBlockCounts);
        const auto localHitTokens =
            localCache ? (*localCache)[requestIndex] *
                             context._requests[requestIndex].request.blockSize
                       : decision.predictedCacheHitTokenCount;
        const auto peerExtraTokens =
            std::max<int64_t>(0, decision.predictedCacheHitTokenCount - localHitTokens);
        if (audit.nativePrediction) {
            const auto &native = *audit.nativePrediction;
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-native-admission-prediction] admission_phase=%s "
                "schedule_time_us=%ld queue_prediction_time_us=%ld schedule_id=%lu "
                "schedule_batch_id=%ld request_id=%lu agent=%s model_binary_version=%d "
                "model_revision=%ld base_execution_ms=%.9f residual_correction_ms=%.9f "
                "final_execution_ms=%.9f decision_execution_ms=%.9f "
                "queue_prediction_ms=%.9f "
                "observation_now_ms=%.9f incoming_delay_ms=%.9f request_count=%lu "
                "result_count=%lu slot_count=%lu max_running_requests=%ld "
                "max_transitions=%lu local_hit_tokens=%ld peer_extra_hit_tokens=%ld "
                "total_hit_tokens=%ld",
                admissionPhase, context.getScheduleTimeUs(), native.predictionTimeUs,
                decision.scheduleId, decision.batchId, decision.requestId,
                node.getAgentId().c_str(), node.getLatencyBinaryVersion(),
                node.getLatencyModelRevision(), audit.baseExecutionMs, audit.correctionMs,
                audit.finalExecutionMs, decision.predictedExecutionLatencyMs,
                decision.predictedQueueLatencyMs, native.input.state.nowMs,
                native.arrivalDelayMs, native.input.requests.size(),
                native.input.state.results.size(), native.input.state.slots.size(),
                native.input.state.maxRunningRequestCount, native.maxTransitions,
                localHitTokens, peerExtraTokens, decision.predictedCacheHitTokenCount);
            continue;
        }
        const auto &prediction = *decision.latencyPrediction;
        const auto &features = prediction.features;
        const auto &floor = prediction.floorFeatures;
        const char *decayBranch = !prediction.hasFloor ? "none"
                                  : node.usesRunningElapsedProgress()
                                      ? "running_elapsed_progress"
                                      : "wall_clock_decay";
        const auto expectedMoments=[](const NaviSchedLatencyFeatures &value) {
            const auto expected=value.getExpectedKVWorkload();
            if(!expected) return std::string("[]");
            std::vector<double> moments(expected->running.begin(),expected->running.end());
            moments.insert(moments.end(),expected->waiting.begin(),expected->waiting.end());
            moments.push_back(expected->precedingRequests);
            moments.push_back(expected->batchCount);
            return autil::legacy::ToJsonString(moments,true);
        };
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
        const auto workloadSuffix=geometrySuffix +
            " native_step_token_budget=" + std::to_string(node.getNativeStepTokenBudget()) +
            " kv_preparing_expected_moments="+expectedMoments(features)+
            " kv_preparing_floor_expected_moments="+expectedMoments(floor);
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-master-admission-prediction] admission_phase=%s schedule_time_us=%ld "
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
            admissionPhase, decision.scheduleTimeUs, decision.scheduleId, decision.batchId,
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
            floor.getWaitingCacheHitMissTokenProductSum(), workloadSuffix.c_str());
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
    std::unordered_set<size_t> probabilityReportedNodes;
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
        // One record per selected node and round, outside candidate evaluation.
        // The stored probability is the value used in features, never recomputed here.
        if (probabilityReportedNodes.insert(context._selectedNodeIndexes[index]).second) {
            for (const auto &audit : node.getKVPreparationProbabilityAudits()) {
                report("master.kv_preparation_probability", audit.probability);
                report("master.kv_preparation_probability_elapsed_ms", audit.elapsedAtSnapshotMs);
                report("master.kv_preparation_probability_queue_horizon_ms", audit.queueHorizonMs);
                report("master.kv_preparation_probability_forecast_known", audit.frozenTotalMs ? 1 : 0);
                if (audit.frozenTotalMs) {
                    report("master.kv_preparation_probability_frozen_total_ms", *audit.frozenTotalMs);
                }
                const auto scheduleElapsedMs = std::max<int64_t>(
                    0, context._queuePredictionTimeUs - context._scheduleTimeUs) / 1000.0;
                report("master.kv_preparation_probability_schedule_elapsed_ms", scheduleElapsedMs);
                reporter->report(1, "master.kv_preparation_probability_qps",
                                 kmonitor::MetricType::QPS, nullptr,
                                 kmonitor::MetricLevel::FATAL);
                NAVI_KERNEL_LOG(
                    INFO,
                    "[navi-master-kv-preparation-probability] schedule_id=%lu "
                    "schedule_batch_id=%ld agent=%s request_id=%lu load_generation=%lu "
                    "snapshot_mono_us=%ld report_receive_mono_us=%ld "
                    "observed_preparation_ms=%.9f elapsed_at_snapshot_ms=%.9f "
                    "queue_horizon_ms=%.9f forecast_known=%d frozen_total_ms=%.9f "
                    "probability=%.9f schedule_elapsed_ms=%.9f",
                    context._scheduleId, decision.batchId, node.getAgentId().c_str(),
                    audit.requestId, audit.generation, audit.snapshotMonoUs,
                    audit.reportReceiveMonoUs, audit.observedPreparationMs,
                    audit.elapsedAtSnapshotMs, audit.queueHorizonMs,
                    int(audit.frozenTotalMs.has_value()), audit.frozenTotalMs.value_or(-1.0),
                    audit.probability, scheduleElapsedMs);
            }
        }
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
    if (context._queueOverloadAfterCost) {
        // Retain bounded pre-drop predictions before rejected decisions lose their target.
        if (context._scheduleId < 10 || context._scheduleId % 100 == 0) {
            logAdmissionPredictions(context, "pre_drop_sample");
        }
        applySelectedQueueAdmission(context, *_masterScheduleR);
    }
    logAdmissionPredictions(context);
    _masterScheduleR->recordColdStartRouting(context._coldStartRoutingStats);
    context.attachLatencyModelGlobals(*_masterLatencyModelR);
    context._scheduleRequestCompletion.reset();
    context._resultAuditUs = autil::TimeUtility::monotonicTimeUs() - preparedUs;
    return true;
}

void MasterScheduleResultKernel::applySelectedQueueAdmission(
    NaviSchedScheduleContext &context, MasterScheduleR &scheduleResource,
    const std::function<uint64_t()> &randomDraw)
{
    if (!context._queueOverloadAfterCost || context._queueOverloadEvaluated) {
        return;
    }
    struct BatchAdmission {
        double probability;
        bool rejected;
    };
    std::vector<std::optional<BatchAdmission>> admissionByNode(context.nodeCount());
    size_t evaluated = 0;
    size_t rejected = 0;
    double totalQueueMs = 0.0;
    double totalProbability = 0.0;
    for (size_t index = 0; index < context._decisions.size(); ++index) {
        auto &decision = context._decisions[index];
        const auto nodeIndex = context._selectedNodeIndexes[index];
        if (decision.reason != "cost_selected" || nodeIndex >= admissionByNode.size()) {
            continue;
        }
        const auto queueMs = decision.predictedQueueLatencyMs;
        auto &admission = admissionByNode[nodeIndex];
        if (!admission) {
            const auto probability = MasterSchedulePrepareKernel::calculateQueueOverloadDropProbability(
                queueMs, context._queueOverloadScheduleCycleMs,
                context._queueOverloadStartCycleMultiplier, context._queueOverloadFullCycleMultiplier);
            uint64_t sample = 0;
            bool drop = probability >= 1.0;
            if (probability > 0.0 && probability < 1.0) {
                thread_local multi_call::RandomGenerator randomGenerator(
                    autil::TimeUtility::currentTimeInMicroSeconds());
                sample = randomDraw ? randomDraw() : randomGenerator.get();
                drop = static_cast<double>(sample) / std::numeric_limits<uint64_t>::max() < probability;
            }
            // One dispatch batch per node shares a queue prediction and a batch ID.
            admission = BatchAdmission{probability, drop};
            if (context._replay && probability > 0.0 && probability < 1.0) {
                context._queueOverloadRandomDecisions.push_back(
                    {index, sample, probability, !drop});
            }
        }
        ++evaluated;
        if (std::isfinite(queueMs)) {
            totalQueueMs += queueMs;
        }
        totalProbability += admission->probability;
        if (admission->rejected) {
            ++rejected;
            context._selectedNodeIndexes[index] = NaviSchedScheduleContext::INVALID_NODE_INDEX;
            decision.reason = "rate_limited";
            decision.target = {};
            decision.batchId = 0;
            decision.expectedBatchSize = 0;
            decision.predictedCacheHitTokenCount = 0;
            decision.predictedExecutionLatencyMs = 0.0;
            decision.latencyPrediction.reset();
        }
    }
    if (evaluated == 0) {
        return;
    }
    context._queueOverloadEvaluated = true;
    context._queueOverloadRequestCount = evaluated;
    context._queueOverloadRejectedRequestCount = rejected;
    context._queueOverloadAveragePredictedLatencyMs = totalQueueMs / evaluated;
    context._queueOverloadAdmissionPredictedLatencyMs = totalQueueMs / evaluated;
    context._queueOverloadDropProbability = totalProbability / evaluated;
    scheduleResource.recordQueueOverload(evaluated, rejected,
        context._queueOverloadAdmissionPredictedLatencyMs, context._queueOverloadDropProbability);
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
        schedulerInfo["queue_overload_after_cost"] = context._queueOverloadAfterCost;
        schedulerInfo["queue_overload_average_predicted_latency_ms"] =
            context._queueOverloadAveragePredictedLatencyMs;
        schedulerInfo["queue_overload_admission_predicted_latency_ms"] =
            context._queueOverloadAdmissionPredictedLatencyMs;
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
            if (NonLinearPrefillModelR::usesNativePP4ForVersion(
                    node.getLatencyBinaryVersion())) {
                autil::legacy::json::JsonArray requests;
                for (const auto &entry : node.getNativePrefillReservations()) {
                    const auto &r = entry.request;
                    requests.emplace_back(autil::legacy::json::JsonMap{
                        {"request_id", r.requestId},
                        {"total_token_count", r.totalTokenCount},
                        {"block_size", r.blockSize},
                        {"cache_hit_token_count", r.predictedCacheHitTokenCount},
                        {"predicted_latency_ms", r.predictedLatencyMs},
                        {"explicit_cache_checkpoint_position",
                         r.explicitCacheCheckpointPosition},
                        {"native_prefill_slot_release",
                         int32_t(r.nativePrefillSlotRelease)},
                        {"has_block_hashes", bool(r.blockHashes)},
                        {"block_hashes",
                         autil::legacy::ToJson(r.blockHashes ? *r.blockHashes
                                                             : NaviSchedBlockHashes())},
                        {"state", int32_t(entry.state)}});
                }
                runtime["native_prefill_requests"] = std::move(requests);
                const auto cache = nodeIndex < context._replayNativeCacheInputs.size()
                                       ? context._replayNativeCacheInputs[nodeIndex]
                                       : nullptr;
                runtime["native_cache_inputs_present"] = bool(cache);
                if (cache) {
                    autil::legacy::json::JsonArray inputs;
                    for (size_t i = 0; i < cache->inputs.size(); ++i) {
                        const auto &r = cache->inputs[i];
                        inputs.emplace_back(autil::legacy::json::JsonMap{
                            {"request_id",
                             context._replay->requests[i].request.requestId},
                            {"prompt_tokens", r.promptTokens},
                            {"computed_tokens", r.computedTokens},
                            {"shared_prefix_boundary", r.sharedPrefixBoundary},
                            {"external_read_tokens", r.externalReadTokens},
                            {"prefill_end_tokens", r.prefillEndTokens},
                            {"num_tokens", r.numTokens}});
                    }
                    runtime["native_cache_inputs"] = std::move(inputs);
                    runtime["native_full_prefix_blocks_before_query_drop"] =
                        autil::legacy::ToJson(
                            cache->observation.fullPrefixBlocksBeforeQueryDrop);
                    runtime["native_ready_hbm_counts_present"] =
                        bool(cache->observation.readyOnlyHbmBlockCounts);
                    if (cache->observation.readyOnlyHbmBlockCounts)
                        runtime["native_ready_hbm_block_counts"] = autil::legacy::ToJson(
                            *cache->observation.readyOnlyHbmBlockCounts);
                    runtime["native_cache_pool_present"] = bool(cache->observation.pool);
                    if (cache->observation.pool) {
                        const auto &p = *cache->observation.pool;
                        runtime["native_cache_pool"] = autil::legacy::json::JsonMap{
                            {"revision", p.revision},
                            {"state_ready", p.stateReady},
                            {"cached_blocks", p.cachedBlocks},
                            {"full_ready_blocks", p.fullReadyBlocks},
                            {"tail_ready_blocks", p.tailReadyBlocks},
                            {"cached_checksum", p.cachedChecksum},
                            {"full_ready_checksum", p.fullReadyChecksum},
                            {"tail_ready_checksum", p.tailReadyChecksum}};
                    }
                    const auto local =
                        nodeIndex < context._replayNativeLocalCacheCounts.size()
                            ? context._replayNativeLocalCacheCounts[nodeIndex]
                            : nullptr;
                    runtime["native_local_cache_counts_present"] = bool(local);
                    if (local)
                        runtime["native_local_cache_counts"] =
                            autil::legacy::ToJson(*local);
                }
            }
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
