#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "aios/navi_sched/bindings/NaviSchedBindings.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/service/NaviSchedClient.h"
#include "aios/navi_sched/resource/AsyncNotification.h"
#include "aios/navi_sched/service/NaviSchedMaster.h"
#include "aios/network/gig/multi_call/subscribe/VIPSubscribeService.h"
#include "autil/Log.h"
#include "alog/Configurator.h"
#include "alog/Logger.h"
#include "navi/engine/Navi.h"
#include "navi/pybind/NaviPybind.h"
#include <multi_call/rpc/GigRpcServer.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;

namespace navi_sched {
AUTIL_DECLARE_AND_SETUP_LOGGER(navi_sched, VLLMAgentBridgeRuntime);
namespace {

void shutdownAlogAtExit()
{
    multi_call::VIPSubscribeService::destroy();
    alog::Logger::flushAll();
    alog::Logger::shutdown();
}

void registerAlogExitHandler()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (std::atexit(shutdownAlogAtExit) != 0) {
            throw std::runtime_error("failed to register NaviSched ALOG exit handler");
        }
    });
}

class VLLMAgentBridge {
public:
    using BatchRequest = std::tuple<uint64_t, int64_t, int64_t, std::vector<uint64_t>>;
    using RequestStats = std::tuple<uint64_t, int64_t, int64_t>;

    struct ForwardBatch {
        int64_t scheduleBatchId = 0;
        int64_t localBatchId = 0;
        int64_t scheduleBatchSize = 0;
        bool pendingCacheTracked = false;
        int64_t readyTimeNs = 0;
        std::vector<uint64_t> requestIds;
        std::vector<std::string> payloads;
        std::vector<int64_t> predictedCacheHitTokenCounts;
        std::unordered_map<uint64_t, NaviSchedBlockHashesConstPtr> requestBlockHashes;
        std::mutex reporterMutex;
        NaviSchedBatchFinishReporterPtr reporter;
    };

    struct ForwardRequestTouch {
        uint64_t requestId = 0;
        int64_t acceptedTimeNs = 0;
        std::string payload;
    };

    explicit VLLMAgentBridge(const std::shared_ptr<NaviSchedAgent> &agent)
        : _agent(agent), _forwardState(std::make_shared<ForwardState>())
    {
    }

    bool enableForwarding()
    {
        if (!_agent) {
            return false;
        }
        const auto state = _forwardState;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->enabled) {
                return false;
            }
            state->enabled = true;
        }
        _agent->setForwardCancelHandler([state](uint64_t requestId) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->enabled) {
                return;
            }
            if (state->queuedTouchRequestIds.erase(requestId) > 0) {
                state->touchQueue.erase(
                    std::remove_if(
                        state->touchQueue.begin(), state->touchQueue.end(),
                        [requestId](const std::shared_ptr<ForwardRequestTouch> &touch) {
                            return touch && touch->requestId == requestId;
                        }),
                    state->touchQueue.end());
            }
            state->cancelledRequests.emplace_back(requestId);
            state->condition.notify_one();
            state->notification.notify();
        });
        const auto enabled = _agent->setForwardBatchHandler(
            [state](const std::vector<NaviSchedForwardedRequest> &requests,
                    int64_t batchId, const NaviSchedBatchFinishReporterPtr &reporter) {
                if (batchId <= 0 || requests.empty() || !reporter) {
                    return false;
                }
                auto batch = std::make_shared<ForwardBatch>();
                batch->scheduleBatchId = batchId;
                batch->localBatchId = requests.front().localBatchId;
                batch->scheduleBatchSize = requests.front().scheduleBatchSize;
                batch->pendingCacheTracked = requests.front().pendingCacheTracked;
                batch->readyTimeNs =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
                batch->reporter = reporter;
                batch->requestIds.reserve(requests.size());
                batch->payloads.reserve(requests.size());
                batch->predictedCacheHitTokenCounts.reserve(requests.size());
                std::unordered_set<uint64_t> uniqueRequestIds;
                for (const auto &request : requests) {
                    if (request.requestId == 0 || request.payload.empty() ||
                        batch->scheduleBatchId <= 0 || batch->localBatchId <= 0 ||
                        batch->scheduleBatchSize <
                            static_cast<int64_t>(requests.size()) ||
                        request.localBatchId != batch->localBatchId ||
                        request.scheduleBatchSize != batch->scheduleBatchSize ||
                        request.localBatchSize != static_cast<int64_t>(requests.size()) ||
                        request.pendingCacheTracked != batch->pendingCacheTracked ||
                        !uniqueRequestIds.emplace(request.requestId).second) {
                        return false;
                    }
                    batch->requestIds.emplace_back(request.requestId);
                    batch->requestBlockHashes.emplace(request.requestId,
                                                      request.blockHashes);
                    batch->payloads.emplace_back(request.payload);
                    batch->predictedCacheHitTokenCounts.emplace_back(
                        request.predictedCacheHitTokenCount);
                }
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!state->enabled) {
                    return false;
                }
                for (auto requestId : batch->requestIds) {
                    if (state->queuedForwardRequestIds.count(requestId) != 0) {
                        return false;
                    }
                }
                for (auto requestId : batch->requestIds) {
                    state->queuedForwardRequestIds.emplace(requestId);
                }
                state->forwardQueue.emplace_back(batch);
                state->condition.notify_one();
                state->notification.notify();
                return true;
            },
            true,
            [state](uint64_t requestId, const std::string &payload) {
                if (requestId == 0 || payload.empty()) {
                    return;
                }
                auto touch = std::make_shared<ForwardRequestTouch>();
                touch->requestId = requestId;
                touch->acceptedTimeNs =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
                touch->payload = payload;
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!state->enabled ||
                    !state->queuedTouchRequestIds.emplace(requestId).second) {
                    return;
                }
                state->touchQueue.emplace_back(std::move(touch));
                state->condition.notify_one();
            });
        if (!enabled) {
            _agent->setForwardCancelHandler({});
            std::lock_guard<std::mutex> lock(state->mutex);
            state->enabled = false;
            state->condition.notify_all();
        }
        return enabled;
    }

    bool disableForwarding()
    {
        const auto state = _forwardState;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->enabled) {
                return true;
            }
            state->enabled = false;
            state->condition.notify_all();
        }
        if (_agent) {
            _agent->setForwardCancelHandler({});
        }
        const auto disabled = _agent && _agent->setForwardBatchHandler({});
        std::vector<std::shared_ptr<ForwardBatch>> pending;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            pending.assign(state->forwardQueue.begin(), state->forwardQueue.end());
            state->forwardQueue.clear();
            state->touchQueue.clear();
            state->queuedForwardRequestIds.clear();
            state->queuedTouchRequestIds.clear();
            state->cancelledRequests.clear();
            state->condition.notify_all();
        }
        for (const auto &batch : pending) {
            releaseForwardBatch(batch);
        }
        return disabled;
    }

    std::shared_ptr<ForwardBatch> takeNextForwardBatch(int64_t timeoutMs)
    {
        const auto state = _forwardState;
        std::unique_lock<std::mutex> lock(state->mutex);
        if (state->forwardQueue.empty() && state->enabled && timeoutMs > 0) {
            state->condition.wait_for(
                lock, std::chrono::milliseconds(timeoutMs), [state] {
                    return !state->enabled || !state->forwardQueue.empty() ||
                           !state->cancelledRequests.empty();
                });
        }
        if (!state->enabled || state->forwardQueue.empty()) {
            return nullptr;
        }
        auto batch = state->forwardQueue.front();
        state->forwardQueue.pop_front();
        bool removeQueuedTouches = false;
        for (auto requestId : batch->requestIds) {
            state->queuedForwardRequestIds.erase(requestId);
            removeQueuedTouches =
                state->queuedTouchRequestIds.erase(requestId) > 0 || removeQueuedTouches;
        }
        if (removeQueuedTouches) {
            const std::unordered_set<uint64_t> requestIds(batch->requestIds.begin(),
                                                          batch->requestIds.end());
            state->touchQueue.erase(
                std::remove_if(
                    state->touchQueue.begin(), state->touchQueue.end(),
                    [&requestIds](const std::shared_ptr<ForwardRequestTouch> &touch) {
                        return touch && requestIds.count(touch->requestId) != 0;
                    }),
                state->touchQueue.end());
        }
        lock.unlock();
        if (!batch->reporter || !batch->reporter->beginExecution()) {
            releaseForwardBatch(batch);
            // A skipped batch must not hide later batches after the fd was drained.
            state->notification.notify();
            return nullptr;
        }
        return batch;
    }

    int notificationFd() { return _forwardState->notification.fd(); }

    std::vector<std::shared_ptr<ForwardRequestTouch>> takeForwardRequestTouches(
        size_t maxCount)
    {
        const auto state = _forwardState;
        std::lock_guard<std::mutex> lock(state->mutex);
        std::vector<std::shared_ptr<ForwardRequestTouch>> touches;
        const auto count = maxCount == 0 ? state->touchQueue.size()
                                         : std::min(maxCount, state->touchQueue.size());
        touches.reserve(count);
        while (!state->touchQueue.empty() && touches.size() < count) {
            auto touch = std::move(state->touchQueue.front());
            state->touchQueue.pop_front();
            if (!touch || state->queuedTouchRequestIds.erase(touch->requestId) == 0) {
                continue;
            }
            touches.emplace_back(std::move(touch));
        }
        return touches;
    }

    std::vector<uint64_t> takeCancelledRequests()
    {
        const auto state = _forwardState;
        std::lock_guard<std::mutex> lock(state->mutex);
        std::vector<uint64_t> requests;
        requests.reserve(state->cancelledRequests.size());
        while (!state->cancelledRequests.empty()) {
            requests.emplace_back(state->cancelledRequests.front());
            state->cancelledRequests.pop_front();
        }
        return requests;
    }

    bool publishForwardResponse(uint64_t requestId, std::vector<std::string> frames,
                                bool terminal) const
    {
        return _agent &&
               _agent->publishForwardResponse(requestId, std::move(frames), terminal);
    }

    bool failForwardResponse(uint64_t requestId, const std::string &errorMessage) const
    {
        return _agent && _agent->failForwardResponse(requestId, errorMessage);
    }

    static NaviSchedBatchFinishReporterPtr startForwardBatch(
        const std::shared_ptr<ForwardBatch> &batch,
        const std::vector<uint64_t> &requestIds)
    {
        if (!batch || requestIds.size() != batch->requestIds.size() ||
            std::unordered_set<uint64_t>(requestIds.begin(), requestIds.end()) !=
                std::unordered_set<uint64_t>(batch->requestIds.begin(),
                                             batch->requestIds.end())) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(batch->reporterMutex);
        if (!batch->reporter) {
            return nullptr;
        }
        return batch->reporter->startBatch(requestIds) ? batch->reporter : nullptr;
    }

    static bool abortForwardBatchRequests(const std::shared_ptr<ForwardBatch> &batch,
                                          const std::vector<uint64_t> &requestIds)
    {
        if (!batch || requestIds.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(batch->reporterMutex);
        if (!batch->reporter || requestIds.size() >= batch->requestIds.size() ||
            batch->requestIds.size() != batch->payloads.size() ||
            batch->requestIds.size() != batch->predictedCacheHitTokenCounts.size()) {
            return false;
        }
        std::unordered_set<uint64_t> abortedIds;
        abortedIds.reserve(requestIds.size());
        for (auto requestId : requestIds) {
            if (requestId == 0 || !abortedIds.emplace(requestId).second ||
                std::find(batch->requestIds.begin(), batch->requestIds.end(),
                          requestId) == batch->requestIds.end()) {
                return false;
            }
        }
        std::vector<uint64_t> remainingRequestIds;
        std::vector<std::string> remainingPayloads;
        std::vector<int64_t> remainingPredictedCacheHits;
        const auto remainingSize = batch->requestIds.size() - abortedIds.size();
        remainingRequestIds.reserve(remainingSize);
        remainingPayloads.reserve(remainingSize);
        remainingPredictedCacheHits.reserve(remainingSize);
        for (size_t index = 0; index < batch->requestIds.size(); ++index) {
            if (abortedIds.count(batch->requestIds[index]) != 0) {
                continue;
            }
            remainingRequestIds.emplace_back(batch->requestIds[index]);
            remainingPayloads.emplace_back(batch->payloads[index]);
            remainingPredictedCacheHits.emplace_back(
                batch->predictedCacheHitTokenCounts[index]);
        }
        if (!batch->reporter->abortRequests(requestIds)) {
            return false;
        }
        batch->requestIds.swap(remainingRequestIds);
        batch->payloads.swap(remainingPayloads);
        batch->predictedCacheHitTokenCounts.swap(remainingPredictedCacheHits);
        return true;
    }

    static void releaseForwardBatch(const std::shared_ptr<ForwardBatch> &batch)
    {
        if (batch) {
            std::lock_guard<std::mutex> lock(batch->reporterMutex);
            batch->reporter.reset();
        }
    }

    NaviSchedBatchFinishReporterPtr startBatch(
        int64_t batchId, const std::vector<BatchRequest> &requests) const
    {
        if (!_agent || batchId <= 0 || requests.empty()) {
            return nullptr;
        }
        auto readyState = std::make_shared<BatchReadyState>();
        const auto expectedBatchSize = static_cast<int64_t>(requests.size());
        std::vector<NaviSchedBatchRequest> batchRequests;
        batchRequests.reserve(requests.size());
        std::vector<uint64_t> requestIds;
        requestIds.reserve(requests.size());
        std::unordered_set<uint64_t> uniqueRequestIds;
        uniqueRequestIds.reserve(requests.size());
        for (const auto &value : requests) {
            if (0 == std::get<0>(value) ||
                !uniqueRequestIds.emplace(std::get<0>(value)).second) {
                return nullptr;
            }
        }
        for (const auto &value : requests) {
            NaviSchedBatchRequest request;
            request.requestId = std::get<0>(value);
            request.totalTokenCount = std::get<1>(value);
            request.blockSize = std::get<2>(value);
            request.blockHashes =
                std::make_shared<const NaviSchedBlockHashes>(std::get<3>(value));
            if (!request.valid()) {
                return nullptr;
            }
            requestIds.emplace_back(request.requestId);
            batchRequests.emplace_back(std::move(request));
        }
        for (const auto &request : batchRequests) {
            if (!_agent->addBatchRequest(
                    request, batchId, expectedBatchSize,
                    [readyState](const NaviSchedBatchPtr &batch) {
                        std::lock_guard<std::mutex> lock(readyState->mutex);
                        readyState->reporter = batch ? batch->finishReporter : nullptr;
                    })) {
                _agent->abortPendingBatch(batchId);
                return nullptr;
            }
        }
        NaviSchedBatchFinishReporterPtr reporter;
        {
            std::lock_guard<std::mutex> lock(readyState->mutex);
            reporter = readyState->reporter;
        }
        if (!reporter || !reporter->startBatch(requestIds)) {
            return nullptr;
        }
        return reporter;
    }

    static bool finishBatch(const NaviSchedBatchFinishReporterPtr &reporter,
                            const std::vector<RequestStats> &values)
    {
        if (!reporter || values.empty()) {
            return false;
        }
        std::vector<NaviSchedRequestStats> requests;
        requests.reserve(values.size());
        for (const auto &value : values) {
            NaviSchedRequestStats request;
            request.requestId = std::get<0>(value);
            request.totalTokenCount = std::get<1>(value);
            request.cacheHitTokenCount = std::get<2>(value);
            requests.emplace_back(request);
        }
        return reporter->finishBatch(requests);
    }

    static bool finishBatchWithExecution(
        const NaviSchedBatchFinishReporterPtr &reporter,
        const std::vector<RequestStats> &values, double wallTimeMs,
        double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
        int64_t decodeRequestStepCount, int64_t decodeTokenCount, double unifiedLatencyMs,
        int64_t precedingRequestCount, int64_t sampleBatchCount,
        int64_t runningRequestCount, int64_t runningCacheHitTokenCount,
        int64_t runningCacheMissTokenCount, double runningCacheMissTokenSquareSum,
        double runningCacheHitMissTokenProductSum, int64_t waitingRequestCount,
        int64_t waitingCacheHitTokenCount, int64_t waitingCacheMissTokenCount,
        double waitingCacheMissTokenSquareSum, double waitingCacheHitMissTokenProductSum,
        double runningElapsedTimeMs, int64_t runningDecodeRequestCount,
        int64_t runningDecodeGeneratedTokenCount,
        int64_t runningDecodeRemainingTokenBudget, double admissionToFirstExecutionMs,
        double nativePreparationMs)
    {
        if (!reporter || values.empty()) {
            return false;
        }
        std::vector<NaviSchedRequestStats> requests;
        requests.reserve(values.size());
        for (const auto &value : values) {
            NaviSchedRequestStats request;
            request.requestId = std::get<0>(value);
            request.totalTokenCount = std::get<1>(value);
            request.cacheHitTokenCount = std::get<2>(value);
            requests.emplace_back(request);
        }
        if (unifiedLatencyMs < 0.0) {
            return reporter->finishBatch(
                requests, wallTimeMs, executeModelTimeMs, executionStepCount,
                prefillTokenCount, decodeRequestStepCount, decodeTokenCount,
                admissionToFirstExecutionMs, nativePreparationMs);
        }
        NaviSchedLatencyFeatures sampleFeatures;
        if (!sampleFeatures.appendCapacitySplitAggregate(
                sampleBatchCount, runningRequestCount, runningCacheHitTokenCount,
                runningCacheMissTokenCount, runningCacheMissTokenSquareSum,
                runningCacheHitMissTokenProductSum, waitingRequestCount,
                waitingCacheHitTokenCount, waitingCacheMissTokenCount,
                waitingCacheMissTokenSquareSum, waitingCacheHitMissTokenProductSum) ||
            !sampleFeatures.appendPrecedingRequestCount(precedingRequestCount) ||
            !sampleFeatures.appendRunningElapsedTimeMs(runningElapsedTimeMs) ||
            !sampleFeatures.appendRunningDecodeProgress(
                runningDecodeRequestCount, runningDecodeGeneratedTokenCount,
                runningDecodeRemainingTokenBudget)) {
            return false;
        }
        return reporter->finishBatchWithUnifiedLatency(
            requests, wallTimeMs, executeModelTimeMs, executionStepCount,
            prefillTokenCount, decodeRequestStepCount, decodeTokenCount, sampleFeatures,
            unifiedLatencyMs, admissionToFirstExecutionMs, nativePreparationMs);
    }

    bool beginBackendExecution(int32_t requestCount) const
    {
        return _agent && _agent->beginObservedBackendExecution(requestCount);
    }

    bool endBackendExecution() const
    {
        return _agent && _agent->endObservedBackendExecution();
    }

    bool reportExecutionStep(double elapsedMs, int64_t prefillTokens,
                             int64_t decodeTokens) const
    {
        return _agent && _agent->reportObservedExecutionStep(elapsedMs, prefillTokens,
                                                             decodeTokens);
    }

    bool reportCoreStep(double elapsedMs, bool futureWait) const
    {
        return _agent && _agent->reportObservedCoreStep(elapsedMs, futureWait);
    }

    bool reportWorkload(
        int64_t eventTimeUs, int32_t batchCount, int32_t requestCount,
        int32_t remainingPrefillRequestCount, int32_t runningRequestCount,
        int32_t waitingRequestCount, int64_t remainingPrefillContextTokenCount,
        int64_t remainingPrefillTokenCount, double remainingPrefillTokenSquareSum,
        double contextRemainingPrefillTokenProductSum, int32_t maxRunningRequestCount,
        int32_t runningCapacitySlotCount, int32_t runningCapacityPrefillRequestCount,
        int64_t runningCapacityPrefillContextTokenCount,
        int64_t runningCapacityPrefillTokenCount,
        double runningCapacityPrefillTokenSquareSum,
        double runningCapacityContextRemainingPrefillTokenProductSum,
        int32_t waitingCapacityPrefillRequestCount,
        int64_t waitingCapacityPrefillContextTokenCount,
        int64_t waitingCapacityPrefillTokenCount,
        double waitingCapacityPrefillTokenSquareSum,
        double waitingCapacityContextRemainingPrefillTokenProductSum,
        int64_t remainingTotalContextTokenCount, int64_t remainingTotalTokenCount,
        double remainingTotalTokenSquareSum, double contextRemainingTotalTokenProductSum,
        int64_t totalPromptTokenCount, int64_t inFlightScheduledTokenCount,
        int64_t admittedScheduleBatchId, int64_t admittedLocalBatchId,
        int32_t admittedRequestCount, bool admittedPendingCacheTracked,
        const std::vector<uint64_t> &admittedRequestIds, double runningElapsedTimeMs,
        int32_t runningDecodeRequestCount, int64_t runningDecodeGeneratedTokenCount,
        int64_t runningDecodeRemainingTokenBudget,
        const std::vector<int64_t> &coveredNativeBatchIds) const
    {
        if (!_agent) {
            return false;
        }
        NaviSchedWorkloadSnapshot snapshot;
        snapshot.eventTimeUs = eventTimeUs;
        snapshot.batchCount = batchCount;
        snapshot.requestCount = requestCount;
        snapshot.remainingPrefillRequestCount = remainingPrefillRequestCount;
        snapshot.runningRequestCount = runningRequestCount;
        snapshot.runningElapsedTimeMs = runningElapsedTimeMs;
        snapshot.runningDecodeRequestCount = runningDecodeRequestCount;
        snapshot.runningDecodeGeneratedTokenCount = runningDecodeGeneratedTokenCount;
        snapshot.runningDecodeRemainingTokenBudget = runningDecodeRemainingTokenBudget;
        snapshot.waitingRequestCount = waitingRequestCount;
        snapshot.remainingPrefillContextTokenCount = remainingPrefillContextTokenCount;
        snapshot.remainingPrefillTokenCount = remainingPrefillTokenCount;
        snapshot.remainingPrefillTokenSquareSum = remainingPrefillTokenSquareSum;
        snapshot.contextRemainingPrefillTokenProductSum =
            contextRemainingPrefillTokenProductSum;
        snapshot.prefillCapacitySplit = true;
        snapshot.maxRunningRequestCount = maxRunningRequestCount;
        snapshot.runningCapacitySlotCount = runningCapacitySlotCount;
        snapshot.runningCapacityPrefillRequestCount = runningCapacityPrefillRequestCount;
        snapshot.runningCapacityPrefillContextTokenCount =
            runningCapacityPrefillContextTokenCount;
        snapshot.runningCapacityPrefillTokenCount = runningCapacityPrefillTokenCount;
        snapshot.runningCapacityPrefillTokenSquareSum =
            runningCapacityPrefillTokenSquareSum;
        snapshot.runningCapacityContextRemainingPrefillTokenProductSum =
            runningCapacityContextRemainingPrefillTokenProductSum;
        snapshot.waitingCapacityPrefillRequestCount = waitingCapacityPrefillRequestCount;
        snapshot.waitingCapacityPrefillContextTokenCount =
            waitingCapacityPrefillContextTokenCount;
        snapshot.waitingCapacityPrefillTokenCount = waitingCapacityPrefillTokenCount;
        snapshot.waitingCapacityPrefillTokenSquareSum =
            waitingCapacityPrefillTokenSquareSum;
        snapshot.waitingCapacityContextRemainingPrefillTokenProductSum =
            waitingCapacityContextRemainingPrefillTokenProductSum;
        snapshot.remainingTotalContextTokenCount = remainingTotalContextTokenCount;
        snapshot.remainingTotalTokenCount = remainingTotalTokenCount;
        snapshot.remainingTotalTokenSquareSum = remainingTotalTokenSquareSum;
        snapshot.contextRemainingTotalTokenProductSum =
            contextRemainingTotalTokenProductSum;
        snapshot.totalPromptTokenCount = totalPromptTokenCount;
        snapshot.inFlightScheduledTokenCount = inFlightScheduledTokenCount;
        snapshot.admittedBatch.scheduleBatchId = admittedScheduleBatchId;
        snapshot.admittedBatch.localBatchId = admittedLocalBatchId;
        snapshot.admittedBatch.requestCount = admittedRequestCount;
        snapshot.admittedBatch.pendingCacheTracked = admittedPendingCacheTracked;
        snapshot.admittedBatch.requestIds = admittedRequestIds;
        return _agent->reportObservedWorkload(snapshot, coveredNativeBatchIds);
    }

    void reportCacheInserted(const std::vector<uint64_t> &hashes) const
    {
        if (_agent && !hashes.empty()) {
            _agent->reportCacheInserted(
                std::make_shared<const NaviSchedBlockHashes>(hashes));
        }
    }

    void reportCacheRemoved(const std::vector<uint64_t> &hashes) const
    {
        if (_agent && !hashes.empty()) {
            _agent->reportCacheRemoved(
                std::make_shared<const NaviSchedBlockHashes>(hashes));
        }
    }

    void reportCacheUpdated(const std::vector<uint64_t> &added,
                            const std::vector<uint64_t> &removed,
                            const std::vector<uint64_t> &addedFullReady,
                            const std::vector<uint64_t> &removedFullReady,
                            const std::vector<uint64_t> &addedTailReady,
                            const std::vector<uint64_t> &removedTailReady) const
    {
        if (!_agent) {
            return;
        }
        const auto makeHashes = [](const std::vector<uint64_t> &values) {
            return values.empty() ? NaviSchedBlockHashesConstPtr()
                                  : std::make_shared<const NaviSchedBlockHashes>(values);
        };
        _agent->reportCacheUpdated(
            -1, -1, -1, makeHashes(added), makeHashes(removed),
            makeHashes(addedFullReady), makeHashes(removedFullReady),
            makeHashes(addedTailReady), makeHashes(removedTailReady), -1, 0);
    }

    void logCacheLifecycleAudit(const std::string &audit) const
    {
        AUTIL_LOG(INFO, "[navi-cache-lifecycle-audit] %s", audit.c_str());
    }

    void logRequestLifecycleAudit(const std::string &audit) const
    {
        AUTIL_LOG(INFO, "[navi-request-lifecycle-audit] %s", audit.c_str());
    }

    void logCacheTouchAudit(uint64_t requestCount, uint64_t promotedBlockCount,
                            uint64_t lastDrainRequestCount, double lastDrainMaxLagMs,
                            double lastDrainProcessingMs) const
    {
        AUTIL_LOG(INFO,
                  "[navi-cache-touch-audit] requests=%lu promoted_blocks=%lu "
                  "last_drain_requests=%lu last_drain_max_lag_ms=%.3f "
                  "last_drain_processing_ms=%.3f",
                  requestCount, promotedBlockCount, lastDrainRequestCount,
                  lastDrainMaxLagMs, lastDrainProcessingMs);
    }

private:
    struct ForwardState {
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<std::shared_ptr<ForwardBatch>> forwardQueue;
        std::deque<std::shared_ptr<ForwardRequestTouch>> touchQueue;
        std::deque<uint64_t> cancelledRequests;
        std::unordered_set<uint64_t> queuedForwardRequestIds;
        std::unordered_set<uint64_t> queuedTouchRequestIds;
        bool enabled = false;
        AsyncNotification notification;
    };

    struct BatchReadyState {
        std::mutex mutex;
        NaviSchedBatchFinishReporterPtr reporter;
    };

private:
    std::shared_ptr<NaviSchedAgent> _agent;
    std::shared_ptr<ForwardState> _forwardState;
};

NaviSchedBlockHashesConstPtr blockHashesFromPython(const py::iterable &values)
{
    auto hashes = std::make_shared<NaviSchedBlockHashes>();
    for (const auto &value : values) {
        hashes->emplace_back(py::cast<uint64_t>(value));
    }
    return hashes;
}

void bindNaviSchedData(py::module_ &m)
{
    py::class_<VLLMAgentBridge::ForwardBatch,
               std::shared_ptr<VLLMAgentBridge::ForwardBatch>>(m, "_VLLMForwardBatch")
        .def_readonly("batch_id", &VLLMAgentBridge::ForwardBatch::localBatchId)
        .def_readonly("schedule_batch_id",
                      &VLLMAgentBridge::ForwardBatch::scheduleBatchId)
        .def_readonly("schedule_batch_size",
                      &VLLMAgentBridge::ForwardBatch::scheduleBatchSize)
        .def_readonly("pending_cache_tracked",
                      &VLLMAgentBridge::ForwardBatch::pendingCacheTracked)
        .def_readonly("ready_time_ns", &VLLMAgentBridge::ForwardBatch::readyTimeNs)
        .def_readonly("request_ids", &VLLMAgentBridge::ForwardBatch::requestIds)
        .def_readonly("predicted_cache_hit_token_counts",
                      &VLLMAgentBridge::ForwardBatch::predictedCacheHitTokenCounts)
        .def("get_request_block_hashes",
             [](const VLLMAgentBridge::ForwardBatch &self,
                uint64_t requestId) -> const NaviSchedBlockHashes & {
                 const auto it = self.requestBlockHashes.find(requestId);
                 if (it == self.requestBlockHashes.end()) {
                     throw py::key_error("unknown forwarded request id");
                 }
                 return BlockHashUtil::get(it->second);
             })
        .def_property_readonly("payloads", [](const VLLMAgentBridge::ForwardBatch &self) {
            py::list result;
            for (const auto &payload : self.payloads) {
                result.append(py::bytes(payload));
            }
            return result;
        });
    py::class_<VLLMAgentBridge::ForwardRequestTouch,
               std::shared_ptr<VLLMAgentBridge::ForwardRequestTouch>>(
        m, "_VLLMForwardRequestTouch")
        .def_readonly("request_id", &VLLMAgentBridge::ForwardRequestTouch::requestId)
        .def_readonly("accepted_time_ns",
                      &VLLMAgentBridge::ForwardRequestTouch::acceptedTimeNs)
        .def_property_readonly("payload",
                               [](const VLLMAgentBridge::ForwardRequestTouch &self) {
                                   return py::bytes(self.payload);
                               });
    py::class_<NaviSchedBatchFinishReporter, NaviSchedBatchFinishReporterPtr>(
        m, "_VLLMBatchHandle");
    py::class_<VLLMAgentBridge>(m, "_VLLMAgentBridge")
        .def(py::init<const std::shared_ptr<NaviSchedAgent> &>())
        .def("start_batch", &VLLMAgentBridge::startBatch,
             py::call_guard<py::gil_scoped_release>())
        .def("enable_forwarding", &VLLMAgentBridge::enableForwarding,
             py::call_guard<py::gil_scoped_release>())
        .def("disable_forwarding", &VLLMAgentBridge::disableForwarding,
             py::call_guard<py::gil_scoped_release>())
        .def("take_next_forward_batch", &VLLMAgentBridge::takeNextForwardBatch,
             py::arg("timeout_ms") = 0, py::call_guard<py::gil_scoped_release>())
        .def("notification_fd", &VLLMAgentBridge::notificationFd)
        .def("take_forward_request_touches", &VLLMAgentBridge::takeForwardRequestTouches,
             py::arg("max_count") = 0, py::call_guard<py::gil_scoped_release>())
        .def("take_cancelled_requests", &VLLMAgentBridge::takeCancelledRequests,
             py::call_guard<py::gil_scoped_release>())
        .def("publish_forward_response", &VLLMAgentBridge::publishForwardResponse,
             py::arg("request_id"), py::arg("frames"), py::arg("terminal"),
             py::call_guard<py::gil_scoped_release>())
        .def("fail_forward_response", &VLLMAgentBridge::failForwardResponse,
             py::arg("request_id"), py::arg("error_message"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("start_forward_batch", &VLLMAgentBridge::startForwardBatch,
                    py::call_guard<py::gil_scoped_release>())
        .def_static("abort_forward_batch_requests",
                    &VLLMAgentBridge::abortForwardBatchRequests,
                    py::call_guard<py::gil_scoped_release>())
        .def_static("release_forward_batch", &VLLMAgentBridge::releaseForwardBatch,
                    py::call_guard<py::gil_scoped_release>())
        .def_static("finish_batch", &VLLMAgentBridge::finishBatch,
                    py::call_guard<py::gil_scoped_release>())
        .def_static(
            "finish_batch_with_execution", &VLLMAgentBridge::finishBatchWithExecution,
            py::arg("reporter"), py::arg("values"), py::arg("wall_time_ms"),
            py::arg("execute_model_time_ms"), py::arg("execution_step_count"),
            py::arg("prefill_token_count"), py::arg("decode_request_step_count"),
            py::arg("decode_token_count"), py::arg("unified_latency_ms"),
            py::arg("preceding_request_count"), py::arg("sample_batch_count"),
            py::arg("running_request_count"), py::arg("running_cache_hit_token_count"),
            py::arg("running_cache_miss_token_count"),
            py::arg("running_cache_miss_token_square_sum"),
            py::arg("running_cache_hit_miss_token_product_sum"),
            py::arg("waiting_request_count"), py::arg("waiting_cache_hit_token_count"),
            py::arg("waiting_cache_miss_token_count"),
            py::arg("waiting_cache_miss_token_square_sum"),
            py::arg("waiting_cache_hit_miss_token_product_sum"),
            py::arg("running_elapsed_time_ms"), py::arg("running_decode_request_count"),
            py::arg("running_decode_generated_token_count"),
            py::arg("running_decode_remaining_token_budget"),
            py::arg("admission_to_first_execution_ms") = -1.0,
            py::arg("native_preparation_ms") = 0.0,
            py::call_guard<py::gil_scoped_release>())
        .def("begin_backend_execution", &VLLMAgentBridge::beginBackendExecution,
             py::arg("request_count") = 0, py::call_guard<py::gil_scoped_release>())
        .def("end_backend_execution", &VLLMAgentBridge::endBackendExecution,
             py::call_guard<py::gil_scoped_release>())
        .def("report_workload", &VLLMAgentBridge::reportWorkload,
             py::arg("event_time_us"), py::arg("batch_count"), py::arg("request_count"),
             py::arg("remaining_prefill_request_count"), py::arg("running_request_count"),
             py::arg("waiting_request_count"),
             py::arg("remaining_prefill_context_token_count"),
             py::arg("remaining_prefill_token_count"),
             py::arg("remaining_prefill_token_square_sum"),
             py::arg("context_remaining_prefill_token_product_sum"),
             py::arg("max_running_request_count"), py::arg("running_capacity_slot_count"),
             py::arg("running_capacity_prefill_request_count"),
             py::arg("running_capacity_prefill_context_token_count"),
             py::arg("running_capacity_prefill_token_count"),
             py::arg("running_capacity_prefill_token_square_sum"),
             py::arg("running_capacity_context_remaining_prefill_token_product_sum"),
             py::arg("waiting_capacity_prefill_request_count"),
             py::arg("waiting_capacity_prefill_context_token_count"),
             py::arg("waiting_capacity_prefill_token_count"),
             py::arg("waiting_capacity_prefill_token_square_sum"),
             py::arg("waiting_capacity_context_remaining_prefill_token_product_sum"),
             py::arg("remaining_total_context_token_count"),
             py::arg("remaining_total_token_count"),
             py::arg("remaining_total_token_square_sum"),
             py::arg("context_remaining_total_token_product_sum"),
             py::arg("total_prompt_token_count"),
             py::arg("in_flight_scheduled_token_count"),
             py::arg("admitted_schedule_batch_id"), py::arg("admitted_local_batch_id"),
             py::arg("admitted_request_count"), py::arg("admitted_pending_cache_tracked"),
             py::arg("admitted_request_ids"), py::arg("running_elapsed_time_ms"),
             py::arg("running_decode_request_count"),
             py::arg("running_decode_generated_token_count"),
             py::arg("running_decode_remaining_token_budget"),
             py::arg("covered_native_batch_ids") = std::vector<int64_t>(),
             py::call_guard<py::gil_scoped_release>())
        .def("report_cache_inserted", &VLLMAgentBridge::reportCacheInserted,
             py::call_guard<py::gil_scoped_release>())
        .def("report_cache_removed", &VLLMAgentBridge::reportCacheRemoved,
             py::call_guard<py::gil_scoped_release>())
        .def("report_cache_updated", &VLLMAgentBridge::reportCacheUpdated,
             py::call_guard<py::gil_scoped_release>())
        .def("log_cache_lifecycle_audit", &VLLMAgentBridge::logCacheLifecycleAudit,
             py::call_guard<py::gil_scoped_release>())
        .def("log_request_lifecycle_audit", &VLLMAgentBridge::logRequestLifecycleAudit,
             py::call_guard<py::gil_scoped_release>())
        .def("report_execution_step", &VLLMAgentBridge::reportExecutionStep,
             py::arg("elapsed_ms"), py::arg("prefill_tokens"), py::arg("decode_tokens"),
             py::call_guard<py::gil_scoped_release>())
        .def("report_core_step", &VLLMAgentBridge::reportCoreStep, py::arg("elapsed_ms"),
             py::arg("future_wait"), py::call_guard<py::gil_scoped_release>())
        .def("log_cache_touch_audit", &VLLMAgentBridge::logCacheTouchAudit,
             py::call_guard<py::gil_scoped_release>());
    py::class_<NaviSchedTarget>(m, "NaviSchedTarget")
        .def(py::init<>())
        .def_readwrite("version", &NaviSchedTarget::version)
        .def_readwrite("biz_name", &NaviSchedTarget::bizName)
        .def_readwrite("address", &NaviSchedTarget::address)
        .def_readwrite("agent_address", &NaviSchedTarget::agentAddress);
    py::class_<NaviSchedScheduleRequest>(m, "NaviSchedScheduleRequest")
        .def(py::init<>())
        .def_readwrite("version", &NaviSchedScheduleRequest::version)
        .def_readwrite("request_id", &NaviSchedScheduleRequest::requestId)
        .def_readwrite("biz_name", &NaviSchedScheduleRequest::bizName)
        .def_property(
            "block_hashes",
            [](const NaviSchedScheduleRequest &self) {
                return BlockHashUtil::get(self.blockHashes);
            },
            [](NaviSchedScheduleRequest &self, const py::iterable &values) {
                self.blockHashes = blockHashesFromPython(values);
            })
        .def_readwrite("prompt_token_count", &NaviSchedScheduleRequest::promptTokenCount)
        .def_readwrite("block_size", &NaviSchedScheduleRequest::blockSize)
        .def_readwrite("timeout_ms", &NaviSchedScheduleRequest::timeoutMs)
        .def_property(
            "forward_payload",
            [](const NaviSchedScheduleRequest &self) {
                return py::bytes(self.forwardPayload);
            },
            [](NaviSchedScheduleRequest &self, const py::bytes &value) {
                self.forwardPayload = value;
            });
    py::class_<NaviSchedScheduleItem>(m, "NaviSchedScheduleItem")
        .def_readonly("request", &NaviSchedScheduleItem::request)
        .def_readonly("wait_time_ms", &NaviSchedScheduleItem::waitTimeMs)
        .def_readonly("remaining_timeout_ms", &NaviSchedScheduleItem::remainingTimeoutMs)
        .def_readonly("timeout_triggered", &NaviSchedScheduleItem::timeoutTriggered);
    py::class_<NaviSchedReplayNodeMatch>(m, "NaviSchedReplayNodeMatch")
        .def_readonly("matched_block_count",
                      &NaviSchedReplayNodeMatch::matchedBlockCount);
    m.def("get_latency_formula_name", &getNaviSchedLatencyFormulaName,
          py::arg("formula"));
    py::class_<NaviSchedRequestStats>(m, "NaviSchedRequestStats")
        .def(py::init<>())
        .def_readwrite("total_token_count", &NaviSchedRequestStats::totalTokenCount)
        .def_readwrite("cache_hit_token_count",
                       &NaviSchedRequestStats::cacheHitTokenCount)
        .def_readwrite("predicted_cache_hit_token_count",
                       &NaviSchedRequestStats::predictedCacheHitTokenCount)
        .def_readwrite("request_id", &NaviSchedRequestStats::requestId);
    py::class_<NaviSchedBatch, std::shared_ptr<NaviSchedBatch>>(m, "NaviSchedBatch")
        .def_readonly("batch_id", &NaviSchedBatch::batchId)
        .def_readonly("request_ids", &NaviSchedBatch::requestIds)
        .def_readonly("expected_batch_size", &NaviSchedBatch::expectedBatchSize)
        .def_readonly("timed_out", &NaviSchedBatch::timedOut);

    py::class_<NaviSchedDecision>(m, "NaviSchedDecision")
        .def_readonly("version", &NaviSchedDecision::version)
        .def_readonly("request_id", &NaviSchedDecision::requestId)
        .def_readonly("schedule_id", &NaviSchedDecision::scheduleId)
        .def_readonly("schedule_loss", &NaviSchedDecision::scheduleLoss)
        .def_readonly("schedule_latency_us", &NaviSchedDecision::scheduleLatencyUs)
        .def_readonly("schedule_time_us", &NaviSchedDecision::scheduleTimeUs)
        .def_readonly("target", &NaviSchedDecision::target)
        .def_readonly("batch_id", &NaviSchedDecision::batchId)
        .def_readonly("expected_batch_size", &NaviSchedDecision::expectedBatchSize)
        .def_readonly("predicted_cache_hit_tokens",
                      &NaviSchedDecision::predictedCacheHitTokenCount)
        .def_readonly("predicted_queue_latency_ms",
                      &NaviSchedDecision::predictedQueueLatencyMs)
        .def_readonly("predicted_execution_latency_ms",
                      &NaviSchedDecision::predictedExecutionLatencyMs)
        .def_readonly("reason", &NaviSchedDecision::reason);

    py::class_<NaviSchedRequestTiming>(m, "NaviSchedRequestTiming")
        .def_readonly("batch_wait_ms", &NaviSchedRequestTiming::batchWaitMs)
        .def_readonly("queue_wait_ms", &NaviSchedRequestTiming::queueWaitMs)
        .def_readonly("execution_latency_ms", &NaviSchedRequestTiming::executionLatencyMs)
        .def_readonly("client_total_latency_ms",
                      &NaviSchedRequestTiming::clientTotalLatencyMs);

    py::enum_<AgentResponseKind>(m, "AgentResponseKind")
        .value("ACK", AgentResponseKind::ACK)
        .value("OUTPUT", AgentResponseKind::OUTPUT)
        .value("ERROR", AgentResponseKind::ERROR);
    py::class_<AgentResponseToClientData, AgentResponseToClientDataPtr>(
        m, "AgentResponseToClientData")
        .def_readonly("request_id", &AgentResponseToClientData::requestId)
        .def_readonly("kind", &AgentResponseToClientData::kind)
        .def_property_readonly("frames",
                               [](const AgentResponseToClientData &self) {
                                   py::list result;
                                   for (const auto &frame : self.frames) {
                                       result.append(py::bytes(frame));
                                   }
                                   return result;
                               })
        .def_readonly("terminal", &AgentResponseToClientData::terminal)
        .def_readonly("error_message", &AgentResponseToClientData::errorMessage);

    py::class_<NaviSchedScheduleReplay>(m, "NaviSchedScheduleReplay")
        .def("to_json",
             [](const NaviSchedScheduleReplay &self) {
                 return NaviSchedScheduleReplayJson::toJsonString(self);
             })
        .def_readonly("version", &NaviSchedScheduleReplay::version)
        .def_readonly("schedule_time_us", &NaviSchedScheduleReplay::scheduleTimeUs)
        .def_readonly("biz_name", &NaviSchedScheduleReplay::bizName)
        .def_readonly("requests", &NaviSchedScheduleReplay::requests)
        .def_readonly("node_matches", &NaviSchedScheduleReplay::nodeMatches)
        .def_readonly("decisions", &NaviSchedScheduleReplay::decisions)
        .def_property_readonly(
            "scheduler_info", [](const NaviSchedScheduleReplay &self) -> py::object {
                if (self.schedulerInfo.empty()) {
                    return py::none();
                }
                return py::module_::import("json").attr("loads")(
                    autil::legacy::FastToJsonString(self.schedulerInfo));
            });

    py::class_<NaviSchedScheduleResult>(m, "NaviSchedScheduleResult")
        .def_readonly("error_code", &NaviSchedScheduleResult::errorCode)
        .def_readonly("error_message", &NaviSchedScheduleResult::errorMessage)
        .def_readonly("decision", &NaviSchedScheduleResult::decision)
        .def_property_readonly(
            "request_timing",
            [](const NaviSchedScheduleResult &self) { return self.requestTiming.get(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "replay",
            [](const NaviSchedScheduleResult &self) { return self.replay.get(); },
            py::return_value_policy::reference_internal);

    py::class_<NaviSchedGigConfig>(m, "NaviSchedGigConfig")
        .def(py::init<>())
        .def_readwrite("heartbeat_thread_num", &NaviSchedGigConfig::heartbeatThreadNum)
        .def_readwrite("heartbeat_queue_size", &NaviSchedGigConfig::heartbeatQueueSize)
        .def_readwrite("log_prefix", &NaviSchedGigConfig::logPrefix)
        .def_readwrite("snapshot_log_dir", &NaviSchedGigConfig::snapshotLogDir)
        .def_readwrite("http_port", &NaviSchedGigConfig::httpPort)
        .def_readwrite("arpc_port", &NaviSchedGigConfig::arpcPort)
        .def_readwrite("grpc_port", &NaviSchedGigConfig::grpcPort)
        .def_readwrite("grpc_io_thread_num", &NaviSchedGigConfig::grpcIoThreadNum);

    py::class_<NaviSchedConfig>(m, "NaviSchedConfig")
        .def(py::init<>())
        .def_readwrite("name", &NaviSchedConfig::name)
        .def_readwrite("log_file_name", &NaviSchedConfig::logFileName)
        .def_readwrite("log_level", &NaviSchedConfig::logLevel)
        .def_readwrite("enable_kmon", &NaviSchedConfig::enableKmon)
        .def_readwrite("kmon_service_name", &NaviSchedConfig::kmonServiceName)
        .def_readwrite("kmon_metrics_path", &NaviSchedConfig::kmonMetricsPath)
        .def_readwrite("kmon_tags", &NaviSchedConfig::kmonTags)
        .def_readwrite("kmon_monitor_name_prefix",
                       &NaviSchedConfig::kmonMonitorNamePrefix)
        .def_readwrite("kmon_report_interval_ms", &NaviSchedConfig::kmonReportIntervalMs)
        .def_readwrite("gig", &NaviSchedConfig::gig)
        .def_readwrite("wait_no_query", &NaviSchedConfig::waitNoQuery)
        .def_readwrite("batch_timeout_ms", &NaviSchedConfig::batchTimeoutMs)
        .def_readwrite("closed_batch_keep_ms", &NaviSchedConfig::closedBatchKeepMs)
        .def_readwrite("latency_fit_batch_size", &NaviSchedConfig::latencyFitBatchSize)
        .def_readwrite("latency_fit_interval", &NaviSchedConfig::latencyFitInterval)
        .def_readwrite("latency_learning_rate", &NaviSchedConfig::latencyLearningRate)
        .def_readwrite("latency_online_learning_rate_scale",
                       &NaviSchedConfig::latencyOnlineLearningRateScale)
        .def_readwrite("latency_current_sample_weight",
                       &NaviSchedConfig::latencyCurrentSampleWeight)
        .def_readwrite("latency_parameter_learning_rate_scales",
                       &NaviSchedConfig::latencyParameterLearningRateScales)
        .def_readwrite("latency_nonlinear_scale", &NaviSchedConfig::latencyNonlinearScale)
        .def_readwrite("queue_learning_rate", &NaviSchedConfig::queueLearningRate)
        .def_readwrite("latency_check_gradient", &NaviSchedConfig::latencyCheckGradient)
        .def_readwrite("latency_cold_start_fit_epochs",
                       &NaviSchedConfig::latencyColdStartFitEpochs)
        .def_readwrite("latency_history_limit", &NaviSchedConfig::latencyHistoryLimit)
        .def_readwrite("latency_history_bucket_capacity",
                       &NaviSchedConfig::latencyHistoryBucketCapacity)
        .def_readwrite("latency_persist_interval",
                       &NaviSchedConfig::latencyPersistInterval)
        .def_readwrite("latency_persist_interval_ms",
                       &NaviSchedConfig::latencyPersistIntervalMs)
        .def_readwrite("latency_warmup_sample_count",
                       &NaviSchedConfig::latencyWarmupSampleCount)
        .def_readwrite("latency_model_trust_min_revision",
                       &NaviSchedConfig::latencyModelTrustMinRevision)
        .def_readwrite("latency_zero_anchor_max_ms",
                       &NaviSchedConfig::latencyZeroAnchorMaxMs)
        .def_readwrite("latency_zero_anchor_min_samples",
                       &NaviSchedConfig::latencyZeroAnchorMinSamples)
        .def_readwrite("latency_validation_enabled",
                       &NaviSchedConfig::latencyValidationEnabled)
        .def_readwrite("residual_learning_rate", &NaviSchedConfig::residualLearningRate)
        .def_readwrite("residual_update_error_floor_ms",
                       &NaviSchedConfig::residualUpdateErrorFloorMs)
        .def_readwrite("residual_update_error_ratio",
                       &NaviSchedConfig::residualUpdateErrorRatio)
        .def_readwrite("residual_idle_decay_start_ms",
                       &NaviSchedConfig::residualIdleDecayStartMs)
        .def_readwrite("residual_idle_decay_end_ms",
                       &NaviSchedConfig::residualIdleDecayEndMs)
        .def_readwrite("latency_model_state_file",
                       &NaviSchedConfig::latencyModelStateFile)
        .def_readwrite("enable_snapshot", &NaviSchedConfig::enableSnapshot)
        .def_readwrite("snapshot_dir", &NaviSchedConfig::snapshotDir)
        .def_readwrite("snapshot_interval_ms", &NaviSchedConfig::snapshotIntervalMs)
        .def_readwrite("snapshot_keep_count", &NaviSchedConfig::snapshotKeepCount)
        .def_readwrite("enable_vis", &NaviSchedConfig::enableVis);

    auto runtimeConfigClass = py::class_<NaviSchedRuntimeConfig, navi::NaviRuntimeConfig>(
        m, "NaviSchedRuntimeConfig");
    runtimeConfigClass.def(py::init<>())
        .def(py::init([](const navi::NaviRuntimeConfig &runtime) {
                 NaviSchedRuntimeConfig config;
                 static_cast<navi::NaviRuntimeConfig &>(config) = runtime;
                 return config;
             }),
             py::arg("runtime_config"))
        .def_readwrite("biz_name", &NaviSchedRuntimeConfig::bizName);

    py::class_<NaviSchedAgentConfig>(m, "NaviSchedAgentConfig")
        .def(py::init<>())
        .def_readwrite("service", &NaviSchedAgentConfig::service)
        .def_readwrite("runtime", &NaviSchedAgentConfig::runtime)
        .def_readwrite("pd_separated", &NaviSchedAgentConfig::pdSeparated)
        .def_readwrite("unified_latency_prediction",
                       &NaviSchedAgentConfig::unifiedLatencyPrediction)
        .def_readwrite("time_compression", &NaviSchedAgentConfig::timeCompression)
        .def_readwrite("latency_model_fingerprint",
                       &NaviSchedAgentConfig::latencyModelFingerprint)
        .def_readwrite("latency_history_fingerprint",
                       &NaviSchedAgentConfig::latencyHistoryFingerprint)
        .def_readwrite("latency_history_binary_fingerprint",
                       &NaviSchedAgentConfig::latencyHistoryBinaryFingerprint);
    py::class_<NaviSchedClientConfig>(m, "NaviSchedClientConfig")
        .def(py::init<>())
        .def_readwrite("service", &NaviSchedClientConfig::service)
        .def_readwrite("runtime", &NaviSchedClientConfig::runtime)
        .def_readwrite("master_biz_name", &NaviSchedClientConfig::masterBizName)
        .def_readwrite("enable_sample_collection",
                       &NaviSchedClientConfig::enableSampleCollection)
        .def_readwrite("enable_agent_forward", &NaviSchedClientConfig::enableAgentForward)
        .def_readwrite("expect_agent_response_stream",
                       &NaviSchedClientConfig::expectAgentResponseStream);
    py::class_<NaviSchedMasterConfig>(m, "NaviSchedMasterConfig")
        .def(py::init<>())
        .def_readwrite("service", &NaviSchedMasterConfig::service)
        .def_readwrite("runtime", &NaviSchedMasterConfig::runtime)
        .def_readwrite("collector_thread_num", &NaviSchedMasterConfig::collectorThreadNum)
        .def_readwrite("collector_queue_count",
                       &NaviSchedMasterConfig::collectorQueueCount)
        .def_readwrite("max_wait_schedule_count",
                       &NaviSchedMasterConfig::maxWaitScheduleCount)
        .def_readwrite("max_wait_schedule_time_ms",
                       &NaviSchedMasterConfig::maxWaitScheduleTimeMs)
        .def_readwrite("queue_overload_start_cycle_multiplier",
                       &NaviSchedMasterConfig::queueOverloadStartCycleMultiplier)
        .def_readwrite("queue_overload_full_cycle_multiplier",
                       &NaviSchedMasterConfig::queueOverloadFullCycleMultiplier)
        .def_readwrite("scheduler_name", &NaviSchedMasterConfig::schedulerName)
        .def_readwrite("scheduler_config", &NaviSchedMasterConfig::schedulerConfig)
        .def_readwrite("time_compression", &NaviSchedMasterConfig::timeCompression)
        .def_readwrite("latency_model_pool_state_file",
                       &NaviSchedMasterConfig::latencyModelPoolStateFile)
        .def_readwrite("latency_model_pool_source_min_revision",
                       &NaviSchedMasterConfig::latencyModelPoolSourceMinRevision)
        .def_readwrite("latency_model_pool_max_candidates",
                       &NaviSchedMasterConfig::latencyModelPoolMaxCandidates)
        .def_readwrite("latency_model_pool_persist_interval_ms",
                       &NaviSchedMasterConfig::latencyModelPoolPersistIntervalMs);

    py::class_<NaviSchedClient, std::shared_ptr<NaviSchedClient>>(m, "NaviSchedClient")
        .def(py::init<const NaviSchedClientConfig &>(), py::arg("config"))
        .def("init", &NaviSchedClient::init, py::call_guard<py::gil_scoped_release>())
        .def("stop", &NaviSchedClient::stop, py::call_guard<py::gil_scoped_release>())
        .def("get_grpc_port", &NaviSchedClient::getGrpcPort)
        .def(
            "schedule",
            [](NaviSchedClient &self, const NaviSchedScheduleRequest &request,
               const std::string &masterBizName, bool collectReplay,
               const std::string &masterAddress) {
                py::gil_scoped_release release;
                return self.schedule(request, masterBizName, collectReplay,
                                     masterAddress);
            },
            py::arg("request"), py::arg("master_biz_name") = "",
            py::arg("collect_replay") = false, py::arg("master_address") = "")
        .def(
            "schedule_flexlb",
            [](NaviSchedClient &self, uint64_t requestId, const std::string &address,
               py::bytes payload, int64_t timeoutMs,
               const std::vector<std::pair<std::string, std::string>> &metadata) {
                std::string input = payload;
                std::tuple<int32_t, std::string, std::string> output;
                {
                    py::gil_scoped_release release;
                    output = self.scheduleFlexlb(requestId, address, input, timeoutMs,
                                                 metadata);
                }
                return py::make_tuple(std::get<0>(output), py::bytes(std::get<1>(output)),
                                      std::get<2>(output));
            },
            py::arg("request_id"), py::arg("address"), py::arg("payload"),
            py::arg("timeout_ms"),
            py::arg("metadata") = std::vector<std::pair<std::string, std::string>>{})
        .def("take_next_forward_response", &NaviSchedClient::takeNextForwardResponse,
             py::arg("timeout_ms") = 0, py::call_guard<py::gil_scoped_release>())
        .def("notification_fd", &NaviSchedClient::notificationFd)
        .def("submit_schedule", &NaviSchedClient::submitSchedule, py::arg("request"),
             py::arg("master_biz_name") = "", py::call_guard<py::gil_scoped_release>())
        .def("take_schedule_result", &NaviSchedClient::takeScheduleResult,
             py::arg("request_id"), py::call_guard<py::gil_scoped_release>())
        .def("abort_forward_request", &NaviSchedClient::abortForwardRequest,
             py::arg("request_id"), py::call_guard<py::gil_scoped_release>())
        .def("allocate_request_id", &NaviSchedClient::allocateRequestId)
        .def("has_master_biz", &NaviSchedClient::hasMasterBiz,
             py::arg("master_biz_name") = "", py::call_guard<py::gil_scoped_release>())
        .def("report_local_fallback", &NaviSchedClient::reportLocalFallback,
             py::call_guard<py::gil_scoped_release>())
        .def("get_request_received_count", &NaviSchedClient::getRequestReceivedCount);

    py::class_<NaviSchedMaster, std::shared_ptr<NaviSchedMaster>>(m, "NaviSchedMaster")
        .def(py::init<const NaviSchedMasterConfig &>(), py::arg("config"))
        .def("init", &NaviSchedMaster::init, py::call_guard<py::gil_scoped_release>())
        .def("stop", &NaviSchedMaster::stop, py::call_guard<py::gil_scoped_release>())
        .def("get_grpc_port", &NaviSchedMaster::getGrpcPort)
        .def("get_http_port", &NaviSchedMaster::getHttpPort);

    py::class_<NaviSchedAgent, std::shared_ptr<NaviSchedAgent>>(m, "NaviSchedAgent")
        .def(py::init<const NaviSchedAgentConfig &, int32_t, const std::string &,
                      int64_t>(),
             py::arg("agent_config"), py::arg("server_port") = 0,
             py::arg("server_ip") = "", py::arg("total_blocks") = 0)
        .def("init", &NaviSchedAgent::init, py::call_guard<py::gil_scoped_release>())
        .def("stop", &NaviSchedAgent::stop, py::call_guard<py::gil_scoped_release>())
        .def(
            "report_cache_status",
            [](const NaviSchedAgent &self, int64_t totalBlocks, int64_t blockSize,
               int64_t linearStep, bool hybridTailRequired) {
                self.reportCacheUpdated(totalBlocks, blockSize, linearStep, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, nullptr,
                                        hybridTailRequired ? 1 : 0, 0);
            },
            py::arg("total_blocks"), py::arg("block_size"), py::arg("linear_step"),
            py::arg("hybrid_tail_required"), py::call_guard<py::gil_scoped_release>())
        .def("get_grpc_port", &NaviSchedAgent::getGrpcPort);
}

}

void bindNaviSched(py::module_ &m) { bindNaviSchedData(m); }

void bindNaviSchedModule(py::module_ &module, const char *description)
{
    module.doc() = description;
    auto naviModule = module.def_submodule("navi", "Navi runtime configuration");
    registerNaviRuntimeConfigPybind(naviModule);
    module.def("_configure_alog", [](const std::string &config) {
        alog::Configurator::configureLoggerFromString(config.c_str());
        registerAlogExitHandler();
    });
    bindNaviSched(module);
}

}
