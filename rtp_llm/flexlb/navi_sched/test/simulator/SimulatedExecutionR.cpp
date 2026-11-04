#include "rtp_llm/flexlb/navi_sched/test/simulator/SimulatedExecutionR.h"

#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <autil/HashAlgorithm.h>
#include <autil/HashUtil.h>
#include <autil/TimeUtility.h>
#include <navi/log/NaviLogger.h>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace navi_sched::test {
const std::vector<double> SIMULATED_FORMULA_PARAMETERS = [] {
    // Preserve the simulator's physical 280 + 50 * (u + sqrt(u*u + 4))
    // workload, expressed in the current zero-point/curvature coordinates.
    const double zeroLinear = -964.0 / 320.0;
    const double zeroShape = 4.0 / (std::sqrt(zeroLinear * zeroLinear + 4.0) - zeroLinear);
    std::vector<double> parameters = {
        -964.0 / 30.0, 10.0 / 30.0, 1.4 / 30.0, 20.0 / 30.0,
        0.1 / 30.0, 0.09 / 30.0, 0.0, 280.0 + 50.0 * zeroShape,
        4.0 / 900.0, 0.0,
    };
    for (size_t index = 0; index < parameters.size(); ++index) {
        parameters[index] /= NonLinearPrefillModelR::getParameterScale(index);
    }
    return parameters;
}();

const std::string SimulatedExecutionR::RESOURCE_ID = "navi_sched.simulate_execution_r";

const std::string SimulatedExecutionBridgeR::RESOURCE_ID =
    "navi_sched.simulate_execution_bridge_r";

SimulatedExecutionBridgeR::SimulatedExecutionBridgeR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_RUN_GRAPH_EXTERNAL)
{
}

void SimulatedExecutionBridgeR::setExecutionR(
    const std::shared_ptr<SimulatedExecutionR> &resource)
{
    autil::ScopedLock lock(_condition);
    _executionR = resource;
    _condition.broadcast();
}

void SimulatedExecutionBridgeR::clearExecutionR(const SimulatedExecutionR *resource)
{
    autil::ScopedLock lock(_condition);
    if (_executionR.get() == resource) {
        _executionR.reset();
    }
}

std::shared_ptr<SimulatedExecutionR> SimulatedExecutionBridgeR::waitExecutionR(
    int64_t timeoutUs)
{
    autil::ScopedLock lock(_condition);
    while (!_executionR) {
        if (ETIMEDOUT == _condition.wait(timeoutUs)) {
            return nullptr;
        }
    }
    return _executionR;
}

const std::vector<double> &SimulatedExecutionR::getFormulaParameters()
{
    return SIMULATED_FORMULA_PARAMETERS;
}

void SimulatedExecutionR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_KERNEL);
}

bool SimulatedExecutionR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "cache_capacity_bytes", _config.cacheCapacityBytes,
                 _config.cacheCapacityBytes);
    NAVI_JSONIZE(ctx, "token_cache_bytes", _config.tokenCacheBytes,
                 _config.tokenCacheBytes);
    NAVI_JSONIZE(ctx, "block_size", _config.blockSize, _config.blockSize);
    NAVI_JSONIZE(ctx, "formula_parameters", _config.formulaParameters,
                 _config.formulaParameters);
    NAVI_JSONIZE(ctx, "noise_stddev_ratio", _config.noiseStddevRatio,
                 _config.noiseStddevRatio);
    NAVI_JSONIZE(ctx, "random_seed", _config.randomSeed, _config.randomSeed);
    NAVI_JSONIZE(ctx, "time_compression", _config.timeCompression,
                 _config.timeCompression);
    NAVI_JSONIZE(ctx, "queue_trace_output_path", _config.queueTraceOutputPath,
                 _config.queueTraceOutputPath);
    if (_config.formulaParameters.empty()) {
        _config.formulaParameters = SIMULATED_FORMULA_PARAMETERS;
    }
    return _config.timeCompression > 0.0 && _config.blockSize > 0 &&
           NonLinearPrefillModelR::validateParameters(_config.formulaParameters);
}

navi::ErrorCode SimulatedExecutionR::init(navi::ResourceInitContext &)
{
    if (!_executionBridgeR) {
        return navi::EC_INIT_RESOURCE;
    }
    _agentRequestR->setCompletionHandler(
        [this](const NaviSchedForwardRequest &request,
               const navi::AsyncPipePtr &completionPipe) {
            return submit(request, completionPipe);
        });
    _agentRequestR->setFinishHandler([this]() { return finishGraph(); });
    _executionBridgeR->setExecutionR(
        std::dynamic_pointer_cast<SimulatedExecutionR>(shared_from_this()));
    return navi::EC_NONE;
}

SimulatedExecutionR::~SimulatedExecutionR()
{
    if (_executionBridgeR) {
        _executionBridgeR->clearExecutionR(this);
    }
    if (_agentRequestR) {
        _agentRequestR->setCompletionHandler({});
        _agentRequestR->setFinishHandler({});
    }
}

bool SimulatedExecutionR::submit(SimulatedRequest request)
{
    request.receiveTimeUs = autil::TimeUtility::currentTime();
    return submitRequest(std::move(request));
}

bool SimulatedExecutionR::submit(const NaviSchedForwardRequest &forwardRequest,
                                 const navi::AsyncPipePtr &completionPipe)
{
    SimulatedRequest request;
    request.requestId = forwardRequest.request.requestId;
    request.blockHashes = forwardRequest.request.blockHashes;
    request.prefillTokenCount = forwardRequest.request.promptTokenCount;
    request.completionPipe = completionPipe;
    const auto &decision = forwardRequest.decision;
    request.batchId = decision.batchId;
    request.expectedBatchSize = decision.expectedBatchSize;
    request.predictedQueueLatencyMs = decision.predictedQueueLatencyMs;
    request.predictedExecutionLatencyMs = decision.predictedExecutionLatencyMs;
    request.latencyPrediction = decision.latencyPrediction;
    request.predictedCacheHitTokenCount = decision.predictedCacheHitTokenCount;
    request.latencyTrainingPhase = decision.latencyTrainingPhase;
    request.receiveTimeUs = autil::TimeUtility::currentTime();
    return submitRequest(std::move(request));
}

bool SimulatedExecutionR::submitRequest(SimulatedRequest request)
{
    if (0 == request.requestId) {
        return false;
    }
    if (!request.blockHashes) {
        request.blockHashes = calculateBlockHashes(request.promptTokenIds);
    }
    auto batchId = request.batchId;
    auto expectedBatchSize = request.expectedBatchSize;
    return submitBatchRequest(std::move(request), batchId, expectedBatchSize);
}

bool SimulatedExecutionR::submitBatchRequest(SimulatedRequest request, int64_t batchId,
                                             int64_t expectedBatchSize)
{
    {
        autil::ScopedLock lock(_stateLock);
        if (_finishRequested ||
            !_batchRequests.emplace(request.requestId, request).second) {
            return false;
        }
        ++_waitingRequestCount;
    }
    NaviSchedBatchRequest batchRequest;
    batchRequest.requestId = request.requestId;
    batchRequest.totalTokenCount = request.prefillTokenCount > 0
                                       ? request.prefillTokenCount
                                       : request.promptTokenIds.size();
    batchRequest.blockHashes = request.blockHashes;
    batchRequest.predictedQueueLatencyMs = request.predictedQueueLatencyMs;
    batchRequest.predictedExecutionLatencyMs = request.predictedExecutionLatencyMs;
    batchRequest.latencyPrediction = request.latencyPrediction;
    batchRequest.latencyTrainingPhase = request.latencyTrainingPhase;
    batchRequest.predictedCacheHitTokenCount = request.predictedCacheHitTokenCount;
    auto requestStats = _agentReportR->makeRequestStats(batchRequest);
    RequestLatencyTracker::Snapshot inflight;
    const auto predictedLatencyMs =
        batchRequest.predictedQueueLatencyMs + batchRequest.predictedExecutionLatencyMs;
    if (!_inflightState.add(batchRequest.requestId, predictedLatencyMs,
                            batchRequest.latencyPrediction, inflight)) {
        NAVI_KERNEL_LOG(ERROR,
                        "claim simulated inflight request failed, requestId [%lu], "
                        "scheduleBatchId [%ld]",
                        request.requestId, batchId);
        autil::ScopedLock lock(_stateLock);
        if (_batchRequests.erase(request.requestId) > 0) {
            --_waitingRequestCount;
        }
        return false;
    }
    if (!_agentReportR->setInflightStats(inflight)) {
        RequestLatencyTracker::Snapshot rollback;
        if (!_inflightState.remove(batchRequest.requestId, rollback)) {
            NAVI_KERNEL_LOG(ERROR,
                            "rollback simulated inflight claim failed, "
                            "requestId [%lu], scheduleBatchId [%ld]",
                            request.requestId, batchId);
        } else if (!_agentReportR->setInflightStats(rollback)) {
            NAVI_KERNEL_LOG(ERROR,
                            "publish simulated inflight rollback failed, "
                            "requestId [%lu], revision [%lu]",
                            request.requestId, rollback.revision);
        }
        autil::ScopedLock lock(_stateLock);
        if (_batchRequests.erase(request.requestId) > 0) {
            --_waitingRequestCount;
        }
        return false;
    }
    bool acceptedCallbackStarted = false;
    try {
        if (_agentBatchR->addRequest(
                requestStats, batchId, expectedBatchSize,
                [this](const NaviSchedBatchPtr &batch) {
                    AgentBatchR::ExecutionState state;
                    _agentBatchR->getExecutionState(state);
                    if (!_agentLatencyInputR->submitBatchReady(
                            batch, state.batchWaitingRequests,
                            state.queuedBatchRequests, state.runningRequests,
                            state.eventTimeUs)) {
                        NAVI_KERNEL_LOG(
                            ERROR,
                            "submit simulated ready batch failed, "
                            "scheduleBatchId [%ld], localBatchId [%ld], "
                            "requestCount [%lu]",
                            batch->scheduleBatchId, batch->batchId,
                            batch->requestIds.size());
                        throw std::runtime_error(
                            "submit simulated ready batch failed");
                    }
                    if (!enqueueBatch(batch)) {
                        rejectBatch(batch);
                    }
                },
                [this, batchRequest, &acceptedCallbackStarted]() {
                    acceptedCallbackStarted = true;
                    reportRequestReceived(batchRequest);
                },
                AgentBatchR::CacheKeyAudit(),
                [this](const AgentBatchR::ExecutionState &state) {
                    handleAbortedBatch(state, "accepted_batch_failed");
                })) {
            return true;
        }
    } catch (const std::exception &e) {
        NAVI_KERNEL_LOG(ERROR,
                        "accept simulated request failed, requestId [%lu], "
                        "scheduleBatchId [%ld], error [%s]",
                        request.requestId, batchId, e.what());
    } catch (...) {
        NAVI_KERNEL_LOG(ERROR,
                        "accept simulated request failed with unknown exception, "
                        "requestId [%lu], scheduleBatchId [%ld]",
                        request.requestId, batchId);
    }
    if (!acceptedCallbackStarted) {
        removeInflightRequests({request.requestId}, "simulated_batch_admission_failed");
    }
    {
        autil::ScopedLock lock(_stateLock);
        if (_batchRequests.erase(request.requestId) > 0) {
            --_waitingRequestCount;
        }
    }
    return false;
}

bool SimulatedExecutionR::enqueueBatch(const NaviSchedBatchPtr &batch)
{
    if (!batch) {
        return false;
    }
    navi::AsyncPipePtr requestPipe;
    uint64_t missingRequestId = 0;
    {
        autil::ScopedLock lock(_stateLock);
        requestPipe = _requestPipe;
        if (requestPipe &&
            navi::EC_NONE != requestPipe->setDataOverride(nullptr, true, true)) {
            return false;
        }
        auto batchReadyTimeUs = autil::TimeUtility::currentTime();
        for (const auto &requestId : batch->requestIds) {
            if (_batchRequests.end() == _batchRequests.find(requestId)) {
                missingRequestId = requestId;
                break;
            }
        }
        if (0 == missingRequestId) {
            for (const auto &requestId : batch->requestIds) {
                auto it = _batchRequests.find(requestId);
                it->second.batchId = batch->batchId;
                it->second.batchReadyTimeUs = batchReadyTimeUs;
                _submitted.emplace_back(std::move(it->second));
                _batchRequests.erase(it);
            }
        }
    }
    if (0 != missingRequestId) {
        NAVI_KERNEL_LOG(ERROR,
                        "ready simulated batch lost request ownership, "
                        "batchId [%ld], requestId [%lu]",
                        batch->batchId, missingRequestId);
        return false;
    }
    return true;
}

void SimulatedExecutionR::rejectBatch(const NaviSchedBatchPtr &batch)
{
    if (!batch) {
        return;
    }
    AgentBatchR::ExecutionState state;
    if (_agentBatchR->abortBatch(batch->batchId, state)) {
        handleAbortedBatch(state, "enqueue_batch_failed");
    } else {
        NAVI_KERNEL_LOG(ERROR, "reject simulated batch failed, batchId [%ld]",
                        batch->batchId);
    }
}

void SimulatedExecutionR::reportRequestReceived(const NaviSchedBatchRequest &request)
{
    AgentBatchR::ExecutionState state;
    _agentBatchR->getExecutionState(state);
    const auto inflight = _inflightState.snapshot();
    _agentReportR->reportRequestReceived(
        request.requestId, request.totalTokenCount, inflight.requestCount,
        state.batchWaitingRequests, state.queuedBatchRequests, state.runningRequests,
        state.eventTimeUs);
}

bool SimulatedExecutionR::takeSubmitted(std::vector<SimulatedRequest> &requests)
{
    autil::ScopedLock lock(_stateLock);
    requests.reserve(requests.size() + _submitted.size());
    for (auto &request : _submitted) {
        requests.emplace_back(std::move(request));
    }
    _submitted.clear();
    return _finishRequested;
}

SimulatedBatchLatency SimulatedExecutionR::predictBatch(
    const std::vector<SimulatedRequest> &requests)
{
    SimulatedBatchLatency result;
    result.predictedLatencyMs = getPredictedLatencyMs(requests);
    result.executionLatencyMs = result.predictedLatencyMs;
    if (_config.noiseStddevRatio > 0.0) {
        uint64_t seed = _config.randomSeed;
        for (const auto &request : requests) {
            autil::HashUtil::combineHash(seed, request.requestId);
        }
        std::mt19937_64 random(seed);
        std::normal_distribution<double> noise(
            0.0, result.executionLatencyMs * _config.noiseStddevRatio);
        result.executionLatencyMs += noise(random);
    }
    result.executionLatencyMs = std::max(0.0, result.executionLatencyMs);
    return result;
}

bool SimulatedExecutionR::startBatch(std::vector<SimulatedRequest> &requests)
{
    if (requests.empty()) {
        return false;
    }
    std::vector<uint64_t> requestIds;
    requestIds.reserve(requests.size());
    for (const auto &request : requests) {
        requestIds.emplace_back(request.requestId);
    }
    AgentBatchR::ExecutionState executionState;
    if (!_agentBatchR->startBatch(requests.front().batchId, requestIds, executionState)) {
        NAVI_KERNEL_LOG(ERROR, "start simulated batch failed, batchId [%ld]",
                        requests.front().batchId);
        abortBatch(requests);
        return false;
    }
    _agentLatencyInputR->submitBatchStarted(
        executionState.batchId, executionState.batchWaitingRequests,
        executionState.queuedBatchRequests, executionState.runningRequests,
        executionState.eventTimeUs);
    auto startTimeUs = autil::TimeUtility::currentTime();
    auto firstReceiveTimeUs = requests.front().receiveTimeUs;
    for (const auto &request : requests) {
        firstReceiveTimeUs = std::min(firstReceiveTimeUs, request.receiveTimeUs);
    }
    size_t previousWaiting = 0;
    size_t authoritativeWaiting = 0;
    bool waitingMismatch = false;
    {
        autil::ScopedLock lock(_stateLock);
        previousWaiting = _waitingRequestCount;
        authoritativeWaiting = static_cast<size_t>(
            executionState.batchWaitingRequests + executionState.queuedBatchRequests);
        waitingMismatch = previousWaiting < requests.size() ||
                          previousWaiting - requests.size() != authoritativeWaiting;
        _waitingRequestCount = authoritativeWaiting;
    }
    if (waitingMismatch) {
        NAVI_KERNEL_LOG(ERROR,
                        "reconcile simulated waiting requests after start, "
                        "batchId [%ld], previous [%lu], started [%lu], "
                        "authoritative [%lu]",
                        requests.front().batchId, previousWaiting, requests.size(),
                        authoritativeWaiting);
    }
    _runningRequestCount.fetch_add(static_cast<int32_t>(requests.size()),
                                   std::memory_order_acq_rel);
    for (auto &request : requests) {
        request.batchWaitMs = getLogicalElapsedMs(
            std::max<int64_t>(0, request.batchReadyTimeUs - firstReceiveTimeUs) / 1000.0);
        request.queueWaitMs = getLogicalElapsedMs(
            std::max<int64_t>(0, startTimeUs - request.batchReadyTimeUs) / 1000.0);
        readCache(request);
        if (!_config.queueTraceOutputPath.empty()) {
            _queueTrace.push_back({request.requestId, request.batchId,
                                   request.batchReadyTimeUs, startTimeUs,
                                   request.prefillTokenCount,
                                   request.cacheHitBlocks * _config.blockSize});
        }
    }
    return true;
}

double SimulatedExecutionR::predictModelLatencyMs(
    const std::vector<SimulatedRequest> &requests) const
{
    return _latencyModelR->predict(makeLatencySample(requests, 0.0).features);
}

double SimulatedExecutionR::getWaitLatencyMs(double latencyMs) const
{
    return std::max(0.0, latencyMs) / _config.timeCompression;
}

double SimulatedExecutionR::getLogicalElapsedMs(double elapsedMs) const
{
    return std::max(0.0, elapsedMs) * _config.timeCompression;
}

void SimulatedExecutionR::finishRequest(const SimulatedRequest &request, double latencyMs)
{
    if (request.completionPipe) {
        auto ack = std::make_shared<AgentResponseToClientData>();
        ack->requestId = request.requestId;
        ack->kind = AgentResponseKind::ACK;
        ack->requestTiming.batchWaitMs = request.batchWaitMs;
        ack->requestTiming.queueWaitMs = request.queueWaitMs;
        ack->requestTiming.executionLatencyMs = latencyMs;
        request.completionPipe->setData(ack);

        auto terminal = std::make_shared<AgentResponseToClientData>();
        terminal->requestId = request.requestId;
        terminal->kind = AgentResponseKind::OUTPUT;
        terminal->frames.emplace_back("simulated_execution_complete");
        terminal->terminal = true;
        request.completionPipe->setData(terminal);
        request.completionPipe->setEof();
    }
}

RequestLatencyTracker::Snapshot SimulatedExecutionR::removeInflightRequests(
    const std::vector<uint64_t> &requestIds, const char *phase)
{
    RequestLatencyTracker::Snapshot snapshot;
    std::vector<uint64_t> missing;
    const auto removed = _inflightState.remove(requestIds, snapshot, &missing);
    if (removed != requestIds.size()) {
        NAVI_KERNEL_LOG(ERROR,
                        "remove simulated inflight requests incomplete, phase [%s], "
                        "requested [%lu], removed [%lu], missing [%lu], revision [%lu], "
                        "remaining [%d]",
                        phase ? phase : "unknown", requestIds.size(), removed,
                        missing.size(), snapshot.revision, snapshot.requestCount);
    }
    if (!_agentReportR->setInflightStats(snapshot)) {
        NAVI_KERNEL_LOG(ERROR,
                        "publish simulated inflight snapshot failed, phase [%s], "
                        "revision [%lu], count [%d], latencySumMs [%.6f]",
                        phase ? phase : "unknown", snapshot.revision,
                        snapshot.requestCount, snapshot.totalPredictedLatencyMs);
    }
    return snapshot;
}

void SimulatedExecutionR::handleAbortedBatch(const AgentBatchR::ExecutionState &state,
                                             const char *phase)
{
    std::vector<SimulatedRequest> requests;
    size_t previousWaiting = 0;
    size_t authoritativeWaiting = 0;
    bool waitingMismatch = false;
    {
        autil::ScopedLock lock(_stateLock);
        requests.reserve(state.requestIds.size());
        for (auto requestId : state.requestIds) {
            auto it = _batchRequests.find(requestId);
            if (_batchRequests.end() == it) {
                continue;
            }
            requests.emplace_back(std::move(it->second));
            _batchRequests.erase(it);
        }
        previousWaiting = _waitingRequestCount;
        authoritativeWaiting =
            static_cast<size_t>(state.batchWaitingRequests + state.queuedBatchRequests);
        waitingMismatch = previousWaiting < requests.size() ||
                          previousWaiting - requests.size() != authoritativeWaiting;
        _waitingRequestCount = authoritativeWaiting;
    }
    if (waitingMismatch) {
        NAVI_KERNEL_LOG(ERROR,
                        "reconcile simulated waiting requests after abort, "
                        "phase [%s], previous [%lu], removed [%lu], "
                        "authoritative [%lu]",
                        phase ? phase : "unknown", previousWaiting, requests.size(),
                        authoritativeWaiting);
    }
    for (const auto &request : requests) {
        if (request.completionPipe) {
            request.completionPipe->setEof();
        }
    }
    if (state.requestIds.size() != static_cast<size_t>(state.batchRequestCount)) {
        NAVI_KERNEL_LOG(ERROR,
                        "simulated aborted batch has inconsistent ownership, "
                        "phase [%s], batchId [%ld], requestCount [%d], "
                        "requestIdCount [%lu]",
                        phase ? phase : "unknown", state.batchId, state.batchRequestCount,
                        state.requestIds.size());
    }
    const auto inflight = removeInflightRequests(state.requestIds, phase);
    if (!_agentLatencyInputR->submitBatchAborted(
            state.batchRequestCount, state.batchId, inflight.requestCount,
            state.batchWaitingRequests, state.queuedBatchRequests, state.runningRequests,
            state.eventTimeUs, state.localBatch)) {
        NAVI_KERNEL_LOG(ERROR,
                        "report simulated aborted batch failed, phase [%s], "
                        "batchId [%ld], requestCount [%d]",
                        phase ? phase : "unknown", state.batchId,
                        state.batchRequestCount);
    }
}

void SimulatedExecutionR::abortBatch(const std::vector<SimulatedRequest> &requests)
{
    if (requests.empty()) {
        return;
    }
    AgentBatchR::ExecutionState executionState;
    if (!_agentBatchR->abortBatch(requests.front().batchId, executionState)) {
        NAVI_KERNEL_LOG(ERROR, "abort simulated batch failed, batchId [%ld]",
                        requests.front().batchId);
        return;
    }
    size_t previousWaiting = 0;
    size_t authoritativeWaiting = 0;
    bool waitingMismatch = false;
    {
        autil::ScopedLock lock(_stateLock);
        previousWaiting = _waitingRequestCount;
        authoritativeWaiting = static_cast<size_t>(
            executionState.batchWaitingRequests + executionState.queuedBatchRequests);
        waitingMismatch = previousWaiting < requests.size() ||
                          previousWaiting - requests.size() != authoritativeWaiting;
        _waitingRequestCount = authoritativeWaiting;
    }
    if (waitingMismatch) {
        NAVI_KERNEL_LOG(ERROR,
                        "reconcile simulated waiting requests after abort, "
                        "batchId [%ld], previous [%lu], removed [%lu], "
                        "authoritative [%lu]",
                        requests.front().batchId, previousWaiting, requests.size(),
                        authoritativeWaiting);
    }
    const auto inflight =
        removeInflightRequests(executionState.requestIds, "simulated_batch_aborted");
    for (const auto &request : requests) {
        finishRequest(request, 0.0);
    }
    _agentLatencyInputR->submitBatchAborted(
        executionState.batchRequestCount, executionState.batchId, inflight.requestCount,
        executionState.batchWaitingRequests, executionState.queuedBatchRequests,
        executionState.runningRequests, executionState.eventTimeUs,
        executionState.localBatch);
}

void SimulatedExecutionR::finishBatch(const std::vector<SimulatedRequest> &requests,
                                      double latencyMs)
{
    if (requests.empty()) {
        return;
    }
    int64_t eventTimeUs = 0;
    std::vector<NaviSchedRequestStats> queries;
    queries.reserve(requests.size());
    for (const auto &request : requests) {
        queries.emplace_back(makeRequestStats(request));
    }
    AgentBatchR::ExecutionState executionState;
    std::vector<NaviSchedRequestStats> completedQueries;
    if (!_agentBatchR->finishBatch(requests.front().batchId, queries, executionState,
                                   &completedQueries)) {
        NAVI_KERNEL_LOG(ERROR, "finish simulated batch failed, batchId [%ld]",
                        requests.front().batchId);
        if (_agentBatchR->abortBatch(requests.front().batchId, executionState)) {
            releaseRunningRequests(executionState.batchRequestCount,
                                   "simulated_finish_failed");
            const auto inflight = removeInflightRequests(executionState.requestIds,
                                                         "simulated_finish_failed");
            if (!_agentLatencyInputR->submitBatchAborted(
                    executionState.batchRequestCount, executionState.batchId,
                    inflight.requestCount, executionState.batchWaitingRequests,
                    executionState.queuedBatchRequests, executionState.runningRequests,
                    eventTimeUs, executionState.localBatch)) {
                NAVI_KERNEL_LOG(ERROR,
                                "report simulated finish failure abort failed, "
                                "batchId [%ld], requestCount [%d]",
                                executionState.batchId, executionState.batchRequestCount);
            }
        }
        return;
    }
    releaseRunningRequests(executionState.batchRequestCount, "simulated_batch_finished");
    std::unordered_set<uint64_t> addedSet;
    std::unordered_set<uint64_t> removedSet;
    NaviSchedCacheUpdate cacheUpdate;
    for (const auto &request : requests) {
        finishRequest(request, latencyMs);
        auto requestCacheUpdate = updateCache(request.blockHashes);
        for (auto hash : BlockHashUtil::get(requestCacheUpdate.addedBlockHashes)) {
            if (removedSet.erase(hash) == 0) {
                addedSet.emplace(hash);
            }
        }
        for (auto hash : BlockHashUtil::get(requestCacheUpdate.removedBlockHashes)) {
            if (addedSet.erase(hash) == 0) {
                removedSet.emplace(hash);
            }
        }
    }
    auto addedBlockHashes =
        std::make_shared<NaviSchedBlockHashes>(addedSet.begin(), addedSet.end());
    auto removedBlockHashes =
        std::make_shared<NaviSchedBlockHashes>(removedSet.begin(), removedSet.end());
    std::sort(addedBlockHashes->begin(), addedBlockHashes->end());
    std::sort(removedBlockHashes->begin(), removedBlockHashes->end());
    cacheUpdate.addedBlockHashes = std::move(addedBlockHashes);
    cacheUpdate.removedBlockHashes = std::move(removedBlockHashes);
    const auto inflight =
        removeInflightRequests(executionState.requestIds, "simulated_batch_finished");
    _agentReportR->reportCacheUpdated(cacheUpdate);
    _agentLatencyInputR->submitBatchFinished(
        completedQueries, latencyMs, executionState.backendQueueLatencyMs,
        executionState.predictedQueueLatencyMs,
        executionState.predictedExecutionLatencyMs,
        executionState.predictedCacheHitTokenCount,
        executionState.actualCacheHitTokenCount,
        executionState.cacheHitMismatchRequestCount,
        executionState.cacheHitAbsoluteErrorTokenCount,
        executionState.cacheHitMaxAbsoluteErrorTokenCount, executionState.batchId,
        inflight.requestCount, executionState.batchWaitingRequests,
        executionState.queuedBatchRequests, executionState.runningRequests, eventTimeUs);
}

void SimulatedExecutionR::releaseRunningRequests(int32_t requestCount, const char *phase)
{
    if (requestCount <= 0) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject invalid simulated running release, phase [%s], "
                        "requestCount [%d]",
                        phase ? phase : "unknown", requestCount);
        return;
    }
    auto current = _runningRequestCount.load(std::memory_order_acquire);
    while (true) {
        const auto next = std::max<int32_t>(0, current - requestCount);
        if (_runningRequestCount.compare_exchange_weak(
                current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            if (current < requestCount) {
                NAVI_KERNEL_LOG(ERROR,
                                "simulated running release exceeded ownership, "
                                "phase [%s], current [%d], released [%d]",
                                phase ? phase : "unknown", current, requestCount);
            }
            return;
        }
    }
}

bool SimulatedExecutionR::setRequestPipe(const navi::AsyncPipePtr &requestPipe)
{
    if (!requestPipe) {
        return false;
    }
    bool ready = false;
    bool finishRequested = false;
    {
        autil::ScopedLock lock(_stateLock);
        _requestPipe = requestPipe;
        ready = !_submitted.empty();
        finishRequested = _finishRequested;
    }
    if (finishRequested) {
        return navi::EC_NONE == requestPipe->setDataOverride(nullptr, true, true);
    }
    return !ready || navi::EC_NONE == requestPipe->setDataOverride(nullptr, true, true);
}

bool SimulatedExecutionR::finishGraph()
{
    navi::AsyncPipePtr requestPipe;
    {
        autil::ScopedLock lock(_stateLock);
        _finishRequested = true;
        requestPipe = _requestPipe;
    }
    return !requestPipe ||
           navi::EC_NONE == requestPipe->setDataOverride(nullptr, true, true);
}

bool SimulatedExecutionR::finishResidentGraph()
{
    if (!writeQueueTrace()) {
        NAVI_KERNEL_LOG(ERROR, "write simulated queue trace failed, path [%s]",
                        _config.queueTraceOutputPath.c_str());
    }
    auto batchFinished = _agentBatchR->finishGraph();
    auto latencyFinished = _agentLatencyInputR->finish();
    return batchFinished && latencyFinished;
}

bool SimulatedExecutionR::writeQueueTrace() const
{
    if (_config.queueTraceOutputPath.empty()) {
        return true;
    }
    std::ofstream output(_config.queueTraceOutputPath);
    if (!output.good()) {
        return false;
    }
    output << "request_id,batch_id,batch_ready_time_us,"
              "batch_start_time_us,total_token_count,"
              "cache_hit_token_count\n";
    for (const auto &entry : _queueTrace) {
        output << entry.requestId << ',' << entry.batchId << ',' << entry.batchReadyTimeUs
               << ',' << entry.batchStartTimeUs << ',' << entry.totalTokenCount << ','
               << entry.cacheHitTokenCount << '\n';
    }
    return output.good();
}

int32_t SimulatedExecutionR::getInflightRequestCount() const
{
    return _inflightState.snapshot().requestCount;
}

size_t SimulatedExecutionR::getCachedBlockCount() const
{
    autil::ScopedLock lock(_cacheLock);
    return _cacheEntries.size();
}

uint64_t SimulatedExecutionR::getEvictedBlockCount() const
{
    return _evictedBlockCount.load(std::memory_order_relaxed);
}

void SimulatedExecutionR::readCache(SimulatedRequest &request)
{
    autil::ScopedLock lock(_cacheLock);
    request.cacheHitBlocks = 0;
    for (auto hash : *request.blockHashes) {
        auto it = _cacheEntries.find(hash);
        if (_cacheEntries.end() == it) {
            break;
        }
        ++request.cacheHitBlocks;
        _cacheLru.splice(_cacheLru.end(), _cacheLru, it->second.position);
    }
    request.cacheMissBlocks = request.blockHashes->size() - request.cacheHitBlocks;
}

NaviSchedCacheUpdate SimulatedExecutionR::updateCache(
    const NaviSchedBlockHashesConstPtr &blockHashes)
{
    NaviSchedCacheUpdate update;
    auto blockBytes = _config.blockSize * _config.tokenCacheBytes;
    if (!blockHashes || blockBytes <= 0 || _config.cacheCapacityBytes < blockBytes) {
        return update;
    }
    auto addedBlockHashes = std::make_shared<NaviSchedBlockHashes>();
    auto removedBlockHashes = std::make_shared<NaviSchedBlockHashes>();
    autil::ScopedLock lock(_cacheLock);
    for (auto hash : *blockHashes) {
        auto it = _cacheEntries.find(hash);
        if (_cacheEntries.end() != it) {
            _cacheLru.splice(_cacheLru.end(), _cacheLru, it->second.position);
            continue;
        }
        while (_cacheUsedBytes + blockBytes > _config.cacheCapacityBytes &&
               !_cacheLru.empty()) {
            auto removedHash = _cacheLru.front();
            _cacheEntries.erase(removedHash);
            _cacheLru.pop_front();
            removedBlockHashes->emplace_back(removedHash);
            _cacheUsedBytes -= blockBytes;
        }
        _cacheLru.emplace_back(hash);
        _cacheEntries.emplace(hash, CacheEntry{std::prev(_cacheLru.end())});
        addedBlockHashes->emplace_back(hash);
        _cacheUsedBytes += blockBytes;
    }
    update.addedBlockHashes = std::move(addedBlockHashes);
    _evictedBlockCount.fetch_add(removedBlockHashes->size(), std::memory_order_relaxed);
    update.removedBlockHashes = std::move(removedBlockHashes);
    return update;
}

double SimulatedExecutionR::getPredictedLatencyMs(
    const std::vector<SimulatedRequest> &requests) const
{
    auto sample = makeLatencySample(requests, 0.0);
    return _nonLinearPrefillModelR->predict(sample.features, _config.formulaParameters);
}

NaviSchedLatencySample SimulatedExecutionR::makeLatencySample(
    const std::vector<SimulatedRequest> &requests, double latencyMs) const
{
    AgentLatencyInput input;
    input.queries.reserve(requests.size());
    for (const auto &request : requests) {
        input.queries.emplace_back(makeRequestStats(request));
    }
    input.latencyMs = latencyMs;
    return input.makeLatencySample();
}

NaviSchedRequestStats SimulatedExecutionR::makeRequestStats(
    const SimulatedRequest &request) const
{
    NaviSchedRequestStats query;
    query.requestId = request.requestId;
    query.totalTokenCount = request.prefillTokenCount;
    query.cacheHitTokenCount = request.cacheHitBlocks * _config.blockSize;
    query.latencyTrainingPhase = request.latencyTrainingPhase;
    return query;
}

NaviSchedBlockHashesConstPtr SimulatedExecutionR::calculateBlockHashes(
    const std::vector<int64_t> &tokenIds)
{
    auto hashes = std::make_shared<NaviSchedBlockHashes>();
    auto blockCount = tokenIds.size() / _config.blockSize;
    hashes->reserve(blockCount);
    size_t rollingHash = 0;
    for (size_t index = 0; index < blockCount * _config.blockSize; ++index) {
        autil::HashUtil::combineHash(rollingHash, tokenIds[index]);
        if (0 == (index + 1) % _config.blockSize) {
            hashes->emplace_back(static_cast<uint64_t>(rollingHash));
        }
    }
    return hashes;
}

REGISTER_RESOURCE(SimulatedExecutionR);
REGISTER_RESOURCE(SimulatedExecutionBridgeR);

} // namespace navi_sched::test
