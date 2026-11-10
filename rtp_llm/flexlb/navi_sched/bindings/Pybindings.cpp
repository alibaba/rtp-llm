#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <openssl/sha.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "aios/navi_sched/bindings/NaviSchedBindings.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/service/NaviSchedClient.h"
#include "aios/navi_sched/resource/AsyncNotification.h"
#include "aios/navi_sched/service/NaviSchedMaster.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"
#include <limits>
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
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;

namespace navi_sched {
AUTIL_DECLARE_AND_SETUP_LOGGER(navi_sched, VLLMAgentBridgeRuntime);
namespace {

void appendCborValue(std::vector<uint8_t>& bytes, uint8_t majorType, uint64_t value) {
    const uint8_t prefix = majorType << 5;
    if (value < 24) {
        bytes.push_back(prefix | value);
        return;
    }
    unsigned width = 8;
    uint8_t tag = 27;
    if (value <= std::numeric_limits<uint8_t>::max()) {
        width = 1;
        tag = 24;
    } else if (value <= std::numeric_limits<uint16_t>::max()) {
        width = 2;
        tag = 25;
    } else if (value <= std::numeric_limits<uint32_t>::max()) {
        width = 4;
        tag = 26;
    }
    bytes.push_back(prefix | tag);
    while (width != 0) {
        --width;
        bytes.push_back(static_cast<uint8_t>(value >> (8 * width)));
    }
}

using VllmBlockDigest = std::array<uint8_t, SHA256_DIGEST_LENGTH>;

std::vector<VllmBlockDigest> hashVllmPlainTextBlockDigests(
    const py::buffer &tokenBuffer, size_t blockSize, VllmBlockDigest parent)
{
    const auto tokens = tokenBuffer.request();
    if (blockSize == 0 || tokens.ndim != 1 || tokens.format != "q"
        || tokens.itemsize != sizeof(int64_t) || tokens.strides[0] != sizeof(int64_t)) {
        throw std::invalid_argument("expected contiguous signed 64-bit tokens and a positive block size");
    }
    const auto *tokenBytes = static_cast<const uint8_t *>(tokens.ptr);
    const size_t fullBlocks = tokens.size / blockSize;
    std::vector<VllmBlockDigest> result(fullBlocks);

    // The caller owns its token buffer until this synchronous call returns.
    py::gil_scoped_release release;
    std::vector<uint8_t> encoded;
    for (size_t block = 0; block < fullBlocks; ++block) {
        encoded.clear();
        // The hashed tuple is (parent hash, current block tokens, extra keys).
        appendCborValue(encoded, 4, 3);
        appendCborValue(encoded, 2, parent.size());
        encoded.insert(encoded.end(), parent.begin(), parent.end());
        appendCborValue(encoded, 4, blockSize);
        const size_t first = block * blockSize;
        for (size_t offset = 0; offset < blockSize; ++offset) {
            int64_t token;
            std::memcpy(&token, tokenBytes + (first + offset) * sizeof(token), sizeof(token));
            if (token < 0) {
                appendCborValue(encoded, 1, static_cast<uint64_t>(-(token + 1)));
            } else {
                appendCborValue(encoded, 0, static_cast<uint64_t>(token));
            }
        }
        encoded.push_back(0xf6);  // No extra cache keys on the plain-text path.
        SHA256(encoded.data(), encoded.size(), parent.data());
        result[block] = parent;
    }
    return result;
}

std::vector<uint64_t> hashVllmPlainTextBlocks(const py::buffer &tokenBuffer, size_t blockSize)
{
    // The route hash uses vLLM's sha256_cbor seed for PYTHONHASHSEED=0.
    constexpr std::array<uint8_t, 2> seed{0x61, '0'};
    VllmBlockDigest parent{};
    SHA256(seed.data(), seed.size(), parent.data());
    const auto digests = hashVllmPlainTextBlockDigests(tokenBuffer, blockSize, parent);
    std::vector<uint64_t> result;
    result.reserve(digests.size());
    for (const auto &digest : digests) {
        uint64_t wireHash = 0;
        for (size_t i = digest.size() - sizeof(wireHash); i < digest.size(); ++i) {
            wireHash = (wireHash << 8) | digest[i];
        }
        result.push_back(wireHash);
    }
    return result;
}

py::list hashVllmPlainTextKvBlocks(
    const py::buffer &tokenBuffer, size_t blockSize, const py::bytes &parentHash)
{
    const std::string parentBytes = parentHash;
    VllmBlockDigest parent{};
    if (parentBytes.size() != parent.size()) {
        throw std::invalid_argument("expected a 32-byte parent hash");
    }
    std::memcpy(parent.data(), parentBytes.data(), parent.size());
    py::list result;
    for (const auto &digest : hashVllmPlainTextBlockDigests(tokenBuffer, blockSize, parent)) {
        result.append(py::bytes(reinterpret_cast<const char *>(digest.data()), digest.size()));
    }
    return result;
}

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

thread_local int64_t lastWorkloadReportDurationNs = 0;

class WorkloadReportTimer {
public:
    explicit WorkloadReportTimer(bool enabled)
        : _enabled(enabled)
        , _start(enabled ? Clock::now() : Clock::time_point())
    {
    }

    ~WorkloadReportTimer()
    {
        if (_enabled) {
            lastWorkloadReportDurationNs =
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - _start).count();
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    bool _enabled;
    Clock::time_point _start;
};

class VLLMAgentBridge {
public:
    using BatchRequest = std::tuple<uint64_t, int64_t, int64_t, std::vector<uint64_t>>;
    using RequestStats = std::tuple<uint64_t, int64_t, int64_t>;

    struct ForwardRequestTouch {
        uint64_t requestId = 0;
        int64_t acceptedTimeNs = 0;
        std::string payload;
    };

    struct ForwardBatch {
        int64_t scheduleBatchId = 0;
        int64_t localBatchId = 0;
        int64_t scheduleBatchSize = 0;
        bool pendingCacheTracked = false;
        int64_t readyTimeNs = 0;
        int64_t readyTimeUs = 0;
        std::vector<uint64_t> requestIds;
        std::vector<std::string> payloads;
        std::vector<int64_t> predictedCacheHitTokenCounts;
        std::unordered_map<uint64_t, NaviSchedBlockHashesConstPtr> requestBlockHashes;
        std::vector<std::shared_ptr<ForwardRequestTouch>> touches;
        std::mutex reporterMutex;
        NaviSchedBatchFinishReporterPtr reporter;
    };

    explicit VLLMAgentBridge(const std::shared_ptr<NaviSchedAgent> &agent,
                             bool pdSeparated = false)
        : _agent(agent)
        , _pdSeparated(pdSeparated)
        , _forwardState(std::make_shared<ForwardState>())
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
                batch->readyTimeUs = requests.front().batchReadyTimeUs;
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
            for (auto it = state->touchQueue.begin(); it != state->touchQueue.end();) {
                if (*it && requestIds.count((*it)->requestId) != 0) {
                    if (_pdSeparated) {
                        batch->touches.emplace_back(std::move(*it));
                    }
                    it = state->touchQueue.erase(it);
                } else {
                    ++it;
                }
            }
        }
        lock.unlock();
        if (!batch->reporter || !batch->reporter->beginExecution()) {
            releaseForwardBatch(batch);
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
        const std::vector<uint64_t> &requestIds, int64_t executionStartTimeUs)
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
        return batch->reporter->startBatch(requestIds, executionStartTimeUs)
                   ? batch->reporter
                   : nullptr;
    }

    static bool resolveForwardExecutionCache(
        const NaviSchedBatchFinishReporterPtr &reporter,
        const std::vector<std::tuple<uint64_t, int64_t, int64_t, int64_t, int64_t>>
            &values)
    {
        if (!reporter || values.empty()) {
            return false;
        }
        std::vector<NaviSchedRequestStats> requests;
        for (const auto &value : values) {
            NaviSchedRequestStats request;
            request.requestId = std::get<0>(value);
            request.totalTokenCount = std::get<1>(value);
            request.cacheHitTokenCount = std::get<2>(value);
            request.executionSharedPrefixBoundary = std::get<3>(value);
            request.executionExternalReadTokens = std::get<4>(value);
            requests.emplace_back(std::move(request));
        }
        return reporter->resolveExecutionCache(requests);
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
        int64_t batchId, const std::vector<BatchRequest> &requests,
        const std::map<uint64_t, int64_t> &prefixCachePromptTokens) const
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
            const auto publication = prefixCachePromptTokens.find(request.requestId);
            if (publication != prefixCachePromptTokens.end()) {
                request.cachePublicationTokenCount = publication->second;
            }
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
        double nativePreparationMs, int64_t executionFinishTimeUs)
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
                admissionToFirstExecutionMs, nativePreparationMs, executionFinishTimeUs);
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

    static bool finishBatchWithExecutionGeometry(
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
        double nativePreparationMs, int64_t geometryExecutionStepCount, int64_t globalPaddedTokens,
        int64_t executionFinishTimeUs)
    {
        if (!reporter || values.empty() || unifiedLatencyMs != -1.0 ||
            geometryExecutionStepCount <= 0 ||
            executionStepCount != geometryExecutionStepCount || globalPaddedTokens <= 0) {
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
        return reporter->finishBatchWithExecutionGeometry(
            requests, wallTimeMs, executeModelTimeMs, executionStepCount,
            prefillTokenCount, decodeRequestStepCount, decodeTokenCount,
            admissionToFirstExecutionMs, globalPaddedTokens, executionFinishTimeUs, nativePreparationMs);
    }
    static bool finishBatchWithProjectedExecutionGeometry(
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
        double nativePreparationMs, int64_t geometryExecutionStepCount, int64_t globalPaddedTokens,
        int64_t projectedExecutionStepCount, int64_t projectedGlobalPaddedTokens,
        int64_t executionFinishTimeUs)
    {
        if (!reporter || values.empty() || unifiedLatencyMs != -1.0 ||
            executionStepCount <= 0 ||
            !((geometryExecutionStepCount == 0 && globalPaddedTokens == 0) ||
              (geometryExecutionStepCount == executionStepCount &&
               globalPaddedTokens > 0)) ||
            projectedExecutionStepCount <= 0 ||
            projectedGlobalPaddedTokens < projectedExecutionStepCount) {
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
        return reporter->finishBatchWithProjectedExecutionGeometry(
            requests, wallTimeMs, executeModelTimeMs, executionStepCount,
            prefillTokenCount, decodeRequestStepCount, decodeTokenCount,
            admissionToFirstExecutionMs, globalPaddedTokens, projectedExecutionStepCount,
            projectedGlobalPaddedTokens, executionFinishTimeUs, nativePreparationMs);
    }

    static std::pair<int64_t, int64_t> fixedPrefillChunk(
        const TPUPrefillExecutionPolicyConfig &policy, int64_t prompt, int64_t prefillEnd,
        int64_t numTokens, int64_t computed, int64_t sharedBoundary)
    {
        int64_t tokens = 0, pairs = 0;
        std::string error;
        TPUPrefillRequestInput input{prompt, computed, sharedBoundary};
        input.prefillEndTokens = prefillEnd;
        input.numTokens = numTokens;
        if (!TPUPrefillExecutionPolicy::planNextPrefillChunk(input, policy, tokens, pairs,
                                                             &error)) {
            throw std::invalid_argument(error);
        }
        return {tokens, pairs};
    }

    static int64_t prefillAttentionPairs(const TPUPrefillExecutionPolicyConfig &policy,
                                         int64_t tokens, int64_t prefix)
    {
        int64_t pairs = 0;
        std::string error;
        if (!TPUPrefillExecutionPolicy::validate(policy, &error) ||
            !TPUPrefillExecutionPolicy::countAttentionPairs(tokens, prefix, policy, pairs,
                                                            &error)) {
            throw std::invalid_argument(error);
        }
        return pairs;
    }

    static std::pair<int64_t, int64_t> projectPrefillExecutionGeometry(
        const TPUPrefillExecutionPolicyConfig &policy,
        const std::vector<std::tuple<int64_t, int64_t, int64_t>> &requests)
    {
        if (!TPUPrefillExecutionPolicy::validate(policy) || requests.empty()) {
            throw std::invalid_argument(
                "invalid typed prefill projection policy or empty batch");
        }
        std::vector<TPUPrefillRequestInput> ordered;
        ordered.reserve(requests.size());
        size_t maxSteps = 0;
        int64_t totalPrompt = 0;
        for (const auto &request : requests) {
            const auto prompt = std::get<0>(request);
            const auto hit = std::get<1>(request);
            const auto boundary = std::get<2>(request);
            if (prompt <= 0 || hit < 0 || hit >= prompt || boundary < 0 ||
                boundary > prompt ||
                prompt > std::numeric_limits<int64_t>::max() - totalPrompt ||
                static_cast<uint64_t>(prompt - hit) >
                    std::numeric_limits<size_t>::max() - maxSteps) {
                throw std::invalid_argument(
                    "invalid prefill projection request coordinates");
            }
            totalPrompt += prompt;
            maxSteps += static_cast<size_t>(prompt - hit);
            ordered.push_back({prompt, hit, boundary});
        }
        TPUPrefillExecutionProjection projection;
        if (!TPUPrefillExecutionPolicy::projectLatencyFeatures(ordered, policy, maxSteps,
                                                               projection)) {
            throw std::invalid_argument("prefill policy projection failed");
        }
        return {projection.stepCount, projection.globalPaddedTokenSum};
    }

    // Offline diagnostics of the SAME native policy used by Master geometry.
    // The explicit resource bound is not a runtime execution instruction.
    using PrefillProjectionStepForReplay =
        std::tuple<int64_t, int64_t, int64_t, int64_t,
                   std::vector<std::tuple<size_t, int64_t, int64_t>>>;
    static std::vector<PrefillProjectionStepForReplay>
    projectPrefillExecutionStepsForReplay(
        const TPUPrefillExecutionPolicyConfig &policy,
        const std::vector<std::tuple<int64_t, int64_t, int64_t>> &requests,
        size_t maxProjectionSteps, bool allCacheResolved)
    {
        if (!TPUPrefillExecutionPolicy::validate(policy) || requests.empty() ||
            maxProjectionSteps == 0) {
            throw std::invalid_argument(
                "invalid replay prefill projection policy, batch or bound");
        }
        std::vector<TPUPrefillRequestInput> ordered;
        ordered.reserve(requests.size());
        int64_t totalPrompt = 0;
        for (const auto &request : requests) {
            const auto prompt = std::get<0>(request);
            const auto hit = std::get<1>(request);
            const auto boundary = std::get<2>(request);
            if (prompt <= 0 || hit < 0 || hit >= prompt || boundary < 0 ||
                boundary > prompt ||
                prompt > std::numeric_limits<int64_t>::max() - totalPrompt) {
                throw std::invalid_argument("invalid replay prefill request coordinates");
            }
            totalPrompt += prompt;
            ordered.push_back({prompt, hit, boundary});
        }
        TPUPrefillExecutionProjection projection;
        const bool ok = allCacheResolved
                            ? TPUPrefillExecutionPolicy::projectCacheResolvedBatch(
                                  ordered, policy, maxProjectionSteps, projection)
                            : TPUPrefillExecutionPolicy::projectLatencyFeatures(
                                  ordered, policy, maxProjectionSteps, projection);
        if (!ok || projection.stepCount < 0 ||
            static_cast<uint64_t>(projection.stepCount) != projection.steps.size()) {
            throw std::invalid_argument(
                "replay prefill projection failed without a complete plan");
        }
        std::vector<PrefillProjectionStepForReplay> result;
        result.reserve(projection.steps.size());
        for (const auto &step : projection.steps) {
            std::vector<std::tuple<size_t, int64_t, int64_t>> members;
            members.reserve(step.requests.size());
            for (const auto &request : step.requests) {
                members.emplace_back(request.requestIndex, request.computedStart,
                                     request.scheduledTokens);
            }
            result.emplace_back(step.geometry.scheduledTokens,
                                step.geometry.maxLocalTokens,
                                step.geometry.localPaddedTokens,
                                step.geometry.globalPaddedTokens, std::move(members));
        }
        return result;
    }

    bool setBackendExecutionPipelineStages(size_t stages) const
    {
        return _agent && _agent->setBackendExecutionPipelineStages(stages);
    }

    bool beginBackendExecution(int64_t eventTimeUs, int32_t requestCount) const
    {
        return _agent && _agent->beginObservedBackendExecution(eventTimeUs, requestCount);
    }

    bool endBackendExecution(int64_t eventTimeUs) const
    {
        return _agent && _agent->endObservedBackendExecution(eventTimeUs);
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

    bool reportPeerCacheReadAvailable(bool available) const
    {
        return _agent && _agent->reportPeerCacheReadAvailable(available);
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
        const std::vector<int64_t> &coveredNativeBatchIds,
        const std::optional<std::vector<uint64_t>> &waitingRequestIds,
        bool recordTiming) const
    {
        const WorkloadReportTimer timer(recordTiming);
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
        snapshot.prefillCapacitySplit = !_pdSeparated;
        if (!_pdSeparated) {
            snapshot.maxRunningRequestCount = maxRunningRequestCount;
            snapshot.runningCapacitySlotCount = runningCapacitySlotCount;
            snapshot.runningCapacityPrefillRequestCount =
                runningCapacityPrefillRequestCount;
            snapshot.runningCapacityPrefillContextTokenCount =
                runningCapacityPrefillContextTokenCount;
            snapshot.runningCapacityPrefillTokenCount = runningCapacityPrefillTokenCount;
            snapshot.runningCapacityPrefillTokenSquareSum =
                runningCapacityPrefillTokenSquareSum;
            snapshot.runningCapacityContextRemainingPrefillTokenProductSum =
                runningCapacityContextRemainingPrefillTokenProductSum;
            snapshot.waitingCapacityPrefillRequestCount =
                waitingCapacityPrefillRequestCount;
            snapshot.waitingCapacityPrefillContextTokenCount =
                waitingCapacityPrefillContextTokenCount;
            snapshot.waitingCapacityPrefillTokenCount = waitingCapacityPrefillTokenCount;
            snapshot.waitingCapacityPrefillTokenSquareSum =
                waitingCapacityPrefillTokenSquareSum;
            snapshot.waitingCapacityContextRemainingPrefillTokenProductSum =
                waitingCapacityContextRemainingPrefillTokenProductSum;
        }
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
        return (!_pdSeparated || (runningDecodeRequestCount == 0 &&
                                  runningDecodeGeneratedTokenCount == 0 &&
                                  runningDecodeRemainingTokenBudget == 0)) &&
               (waitingRequestIds
                    ? _agent->reportObservedWorkload(snapshot, coveredNativeBatchIds,
                                                     *waitingRequestIds)
                    : _agent->reportObservedWorkload(snapshot, coveredNativeBatchIds));
    }

    static NaviSchedCacheKeyType checkedCacheKeyType(int value)
    {
        if (value != static_cast<int>(NaviSchedCacheKeyType::BLOCK_POOL) &&
            value != static_cast<int>(NaviSchedCacheKeyType::V6D)) {
            throw std::invalid_argument("invalid cache key type");
        }
        return static_cast<NaviSchedCacheKeyType>(value);
    }

    void reportCacheInserted(const std::vector<uint64_t> &hashes, int cacheKeyType) const
    {
        const auto type = checkedCacheKeyType(cacheKeyType);
        if (_agent && !hashes.empty()) {
            _agent->reportCacheInserted(
                std::make_shared<const NaviSchedBlockHashes>(hashes), type);
        }
    }

    void reportCacheRemoved(const std::vector<uint64_t> &hashes, int cacheKeyType) const
    {
        const auto type = checkedCacheKeyType(cacheKeyType);
        if (_agent && !hashes.empty()) {
            _agent->reportCacheRemoved(
                std::make_shared<const NaviSchedBlockHashes>(hashes), type);
        }
    }

    void reportCacheUpdated(const std::vector<uint64_t> &added,
                            const std::vector<uint64_t> &removed,
                            const std::vector<uint64_t> &addedFullReady,
                            const std::vector<uint64_t> &removedFullReady,
                            const std::vector<uint64_t> &addedTailReady,
                            const std::vector<uint64_t> &removedTailReady,
                            int cacheKeyType) const
    {
        const auto type = checkedCacheKeyType(cacheKeyType);
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
            makeHashes(addedTailReady), makeHashes(removedTailReady), -1, 0, -1, type);
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
        AsyncNotification notification;
        std::deque<std::shared_ptr<ForwardBatch>> forwardQueue;
        std::deque<std::shared_ptr<ForwardRequestTouch>> touchQueue;
        std::deque<uint64_t> cancelledRequests;
        std::unordered_set<uint64_t> queuedForwardRequestIds;
        std::unordered_set<uint64_t> queuedTouchRequestIds;
        bool enabled = false;
    };

    struct BatchReadyState {
        std::mutex mutex;
        NaviSchedBatchFinishReporterPtr reporter;
    };

private:
    std::shared_ptr<NaviSchedAgent> _agent;
    bool _pdSeparated = false;
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
    m.attr("VLLM_WAITING_REQUEST_IDS_VERSION") = 1;
    m.def("native_wire_versions", []() {
        py::dict versions;
        versions["agent_update_protocol"] = NAVI_SCHED_AGENT_UPDATE_VERSION;
        versions["agent_update_batch_protocol"] = NAVI_SCHED_AGENT_UPDATE_BATCH_VERSION;
        versions["request_data"] = NAVI_SCHED_DATA_VERSION;
        return versions;
    });
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
        .def_readonly("ready_time_us", &VLLMAgentBridge::ForwardBatch::readyTimeUs)
        .def_readonly("request_ids", &VLLMAgentBridge::ForwardBatch::requestIds)
        .def_readonly("touches", &VLLMAgentBridge::ForwardBatch::touches)
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
        m, "_VLLMBatchHandle")
        .def(
            "report_cache_publication_failed",
            [](NaviSchedBatchFinishReporter &self, const std::vector<uint64_t> &full,
               const std::vector<uint64_t> &tail, int cacheKeyType) {
                return self.reportCachePublicationFailed(
                    VLLMAgentBridge::checkedCacheKeyType(cacheKeyType), full, tail);
            },
            py::arg("full_hashes"), py::arg("tail_hashes"), py::arg("cache_key_type"),
            py::call_guard<py::gil_scoped_release>())
        .def("make_cache_publication_reporter",
             [](const NaviSchedBatchFinishReporter &self) {
                 auto callback = self.makeCachePublicationFailureCallback();
                 return py::cpp_function([callback](const std::vector<uint64_t> &full,
                                                    const std::vector<uint64_t> &tail,
                                                    int type) {
                     const auto checked = VLLMAgentBridge::checkedCacheKeyType(type);
                     py::gil_scoped_release release;
                     return callback(checked, full, tail);
                 });
             });
    py::class_<VLLMAgentBridge>(m, "_VLLMAgentBridge")
        .def(py::init<const std::shared_ptr<NaviSchedAgent> &, bool>(), py::arg("agent"),
             py::arg("pd_separated") = false)
        .def("start_batch", &VLLMAgentBridge::startBatch, py::arg("batch_id"),
             py::arg("requests"),
             py::arg("prefix_cache_prompt_tokens") = std::map<uint64_t, int64_t>(),
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
                    py::arg("batch"), py::arg("request_ids"),
                    py::arg("execution_start_time_us") = 0,
                    py::call_guard<py::gil_scoped_release>())
        .def_static("resolve_forward_execution_cache",
                    &VLLMAgentBridge::resolveForwardExecutionCache,
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
            py::arg("execution_finish_time_us") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def_static(
            "finish_batch_with_execution_geometry",
            &VLLMAgentBridge::finishBatchWithExecutionGeometry, py::arg("reporter"),
            py::arg("values"), py::arg("wall_time_ms"), py::arg("execute_model_time_ms"),
            py::arg("execution_step_count"), py::arg("prefill_token_count"),
            py::arg("decode_request_step_count"), py::arg("decode_token_count"),
            py::arg("unified_latency_ms"), py::arg("preceding_request_count"),
            py::arg("sample_batch_count"), py::arg("running_request_count"),
            py::arg("running_cache_hit_token_count"),
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
            py::arg("admission_to_first_execution_ms"), py::arg("native_preparation_ms"),
            py::arg("geometry_execution_step_count"), py::arg("global_padded_tokens"),
            py::arg("execution_finish_time_us") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def_static(
            "finish_batch_with_projected_execution_geometry",
            &VLLMAgentBridge::finishBatchWithProjectedExecutionGeometry,
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
            py::arg("admission_to_first_execution_ms"), py::arg("native_preparation_ms"),
            py::arg("geometry_execution_step_count"), py::arg("global_padded_tokens"),
            py::arg("projected_execution_step_count"),
            py::arg("projected_global_padded_tokens"),
            py::arg("execution_finish_time_us") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def_static("fixed_prefill_chunk", &VLLMAgentBridge::fixedPrefillChunk)
        .def_static("prefill_attention_pairs", &VLLMAgentBridge::prefillAttentionPairs)
        .def_static("project_prefill_execution_geometry",
                    &VLLMAgentBridge::projectPrefillExecutionGeometry, py::arg("policy"),
                    py::arg("ordered_requests"), py::call_guard<py::gil_scoped_release>())
        .def_static("project_prefill_execution_steps_for_replay",
                    &VLLMAgentBridge::projectPrefillExecutionStepsForReplay,
                    py::arg("policy"), py::arg("ordered_requests"),
                    py::arg("max_projection_steps"), py::arg("all_cache_resolved"),
                    py::call_guard<py::gil_scoped_release>())
        .def("set_backend_execution_pipeline_stages",
             &VLLMAgentBridge::setBackendExecutionPipelineStages, py::arg("stages"),
             py::call_guard<py::gil_scoped_release>())
        .def("begin_backend_execution", &VLLMAgentBridge::beginBackendExecution,
             py::arg("event_time_us") = 0, py::arg("request_count") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("end_backend_execution", &VLLMAgentBridge::endBackendExecution,
             py::arg("event_time_us") = 0, py::call_guard<py::gil_scoped_release>())
        .def("report_peer_cache_read_available",
             &VLLMAgentBridge::reportPeerCacheReadAvailable, py::arg("available"))
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
             py::arg("waiting_request_ids") = py::none(),
             py::arg("record_timing") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("last_workload_report_duration_ns", [](const VLLMAgentBridge &) {
            return lastWorkloadReportDurationNs;
        })
        .def("report_cache_inserted", &VLLMAgentBridge::reportCacheInserted,
             py::arg("hashes"), py::arg("cache_key_type") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("report_cache_removed", &VLLMAgentBridge::reportCacheRemoved,
             py::arg("hashes"), py::arg("cache_key_type") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("report_cache_updated", &VLLMAgentBridge::reportCacheUpdated,
             py::arg("added"), py::arg("removed"), py::arg("added_full_ready"),
             py::arg("removed_full_ready"), py::arg("added_tail_ready"),
             py::arg("removed_tail_ready"), py::arg("cache_key_type") = 0,
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
        .def_readwrite("explicit_cache_checkpoint_position",
                       &NaviSchedScheduleRequest::explicitCacheCheckpointPosition)
        .def_readwrite("block_size", &NaviSchedScheduleRequest::blockSize)
        .def_readwrite("timeout_ms", &NaviSchedScheduleRequest::timeoutMs)
        .def_readwrite("route_only", &NaviSchedScheduleRequest::routeOnly)
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
        .def_readwrite("execution_external_read_tokens",
                       &NaviSchedRequestStats::executionExternalReadTokens)
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
        .def_readwrite("kmon_platform_discovery", &NaviSchedConfig::kmonPlatformDiscovery)
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
        .def_readwrite("latency_error_clip_ms", &NaviSchedConfig::latencyErrorClipMs)
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

    py::class_<TPUPrefillExecutionPolicyConfig>(m, "TPUPrefillExecutionPolicyConfig")
        .def(py::init<>())
        .def_readwrite("pcp_size", &TPUPrefillExecutionPolicyConfig::pcpSize)
        .def_readwrite("interleave_size",
                       &TPUPrefillExecutionPolicyConfig::interleaveSize)
        .def_readwrite("compile_sizes", &TPUPrefillExecutionPolicyConfig::compileSizes)
        .def_readwrite("max_scheduled_tokens",
                       &TPUPrefillExecutionPolicyConfig::maxScheduledTokens)
        .def_readwrite("long_prefill_token_threshold",
                       &TPUPrefillExecutionPolicyConfig::longPrefillTokenThreshold)
        .def_readwrite("hash_block_size", &TPUPrefillExecutionPolicyConfig::hashBlockSize)
        .def_readwrite("scheduler_block_size",
                       &TPUPrefillExecutionPolicyConfig::schedulerBlockSize)
        .def_readwrite("use_eagle", &TPUPrefillExecutionPolicyConfig::useEagle)
        .def_readwrite("mamba_has_prefill_checkpoint_blocks",
                       &TPUPrefillExecutionPolicyConfig::mambaHasPrefillCheckpointBlocks)
        .def_readwrite("mamba_partial_cache_hit",
                       &TPUPrefillExecutionPolicyConfig::mambaPartialCacheHit)
        .def_readwrite("async_kv_lookup_enabled",
                       &TPUPrefillExecutionPolicyConfig::asyncKVLookupEnabled)
        .def_readwrite("async_kv_lookup_hash_block_size",
                       &TPUPrefillExecutionPolicyConfig::asyncKVLookupHashBlockSize)
        .def_readwrite("async_kv_lookup_uses_eagle",
                       &TPUPrefillExecutionPolicyConfig::asyncKVLookupUsesEagle)
        .def_readwrite("fixed_prefill_chunk_plan",
                       &TPUPrefillExecutionPolicyConfig::fixedPrefillChunkPlan)
        .def_readwrite("mamba_block_aligned_split",
                       &TPUPrefillExecutionPolicyConfig::mambaBlockAlignedSplit)
        .def_readwrite("attention_schedule_capacity",
                       &TPUPrefillExecutionPolicyConfig::attentionScheduleCapacity)
        .def_readwrite("attention_query_tile",
                       &TPUPrefillExecutionPolicyConfig::attentionQueryTile)
        .def_readwrite("attention_kv_tile",
                       &TPUPrefillExecutionPolicyConfig::attentionKvTile)
        .def_readwrite("prefill_lookahead",
                       &TPUPrefillExecutionPolicyConfig::prefillLookahead)
        .def_readwrite("max_input_tokens",
                       &TPUPrefillExecutionPolicyConfig::maxInputTokens)
        .def_readwrite("draft_slots_per_request",
                       &TPUPrefillExecutionPolicyConfig::draftSlotsPerRequest);

    py::class_<NaviSchedAgentConfig>(m, "NaviSchedAgentConfig")
        .def(py::init<>())
        .def_readwrite("service", &NaviSchedAgentConfig::service)
        .def_readwrite("runtime", &NaviSchedAgentConfig::runtime)
        .def_readwrite("pd_separated", &NaviSchedAgentConfig::pdSeparated)
        .def_readwrite("execution_geometry_enabled",
                       &NaviSchedAgentConfig::executionGeometryEnabled)
        .def_readwrite("projected_execution_geometry_enabled",
                       &NaviSchedAgentConfig::projectedExecutionGeometryEnabled)
        .def_readwrite("execution_cache_source_enabled",
                       &NaviSchedAgentConfig::executionCacheSourceEnabled)
        .def_readwrite("peer_cache_enabled", &NaviSchedAgentConfig::peerCacheEnabled)
        .def_readwrite("prefill_execution_policy",
                       &NaviSchedAgentConfig::prefillExecutionPolicy)
        .def_readwrite("unified_latency_prediction",
                       &NaviSchedAgentConfig::unifiedLatencyPrediction)
        .def_readwrite("full_attention_query_drop_tokens",
                       &NaviSchedAgentConfig::fullAttentionQueryDropTokens)
        .def_readwrite("mamba_replay_stop_shift",
                       &NaviSchedAgentConfig::mambaReplayStopShift)
        .def_readwrite("mamba_retention_interval",
                       &NaviSchedAgentConfig::mambaRetentionInterval)
        .def_readwrite("mamba_align", &NaviSchedAgentConfig::mambaAlign)
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
                return self.schedule(request, masterBizName, collectReplay, masterAddress);
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
                    output = self.scheduleFlexlb(requestId, address, input, timeoutMs, metadata);
                }
                return py::make_tuple(std::get<0>(output), py::bytes(std::get<1>(output)),
                                      std::get<2>(output));
            },
            py::arg("request_id"), py::arg("address"), py::arg("payload"),
            py::arg("timeout_ms"),
            py::arg("metadata") = std::vector<std::pair<std::string, std::string>>{})
        .def("take_next_forward_response",
             &NaviSchedClient::takeNextForwardResponse,
             py::arg("timeout_ms") = 0,
             py::call_guard<py::gil_scoped_release>())
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
               int64_t linearStep, bool hybridTailRequired, int cacheKeyType,
               const std::optional<int64_t> &fullQueryDrop,
               const std::optional<bool> &replayStopShift,
               const std::optional<int64_t> &retentionInterval,
               const std::optional<bool> &mambaAlign) {
                const auto type = VLLMAgentBridge::checkedCacheKeyType(cacheKeyType);
                const bool anyPolicy =
                    fullQueryDrop || replayStopShift || retentionInterval || mambaAlign;
                const bool allPolicy =
                    fullQueryDrop && replayStopShift && retentionInterval && mambaAlign;
                if (anyPolicy && !allPolicy) {
                    throw std::invalid_argument(
                        "cache status requires all policy fields");
                }
                NaviSchedCachePolicy policy;
                if (allPolicy) {
                    policy.blockSize = blockSize;
                    policy.linearStep = linearStep;
                    policy.hybridTailRequired = hybridTailRequired;
                    policy.fullAttentionQueryDropTokens = *fullQueryDrop;
                    policy.mambaReplayStopShift = *replayStopShift;
                    policy.mambaRetentionInterval = *retentionInterval;
                    policy.mambaAlign = *mambaAlign;
                }
                self.reportCacheUpdated(totalBlocks, blockSize, linearStep, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, nullptr,
                                        hybridTailRequired ? 1 : 0, 0, -1, type,
                                        allPolicy ? &policy : nullptr);
            },
            py::arg("total_blocks"), py::arg("block_size"), py::arg("linear_step"),
            py::arg("hybrid_tail_required"), py::arg("cache_key_type") = 0,
            py::arg("full_attention_query_drop_tokens") = py::none(),
            py::arg("mamba_replay_stop_shift") = py::none(),
            py::arg("mamba_retention_interval") = py::none(),
            py::arg("mamba_align") = py::none(), py::call_guard<py::gil_scoped_release>())
        .def("get_grpc_port", &NaviSchedAgent::getGrpcPort);
}

}

void bindNaviSched(py::module_ &m) { bindNaviSchedData(m); }

void bindNaviSchedModule(py::module_ &module, const char *description)
{
    module.def("hash_vllm_plain_text_blocks", &hashVllmPlainTextBlocks,
               py::arg("token_buffer"), py::arg("block_size"));
    module.def("hash_vllm_plain_text_kv_blocks", &hashVllmPlainTextKvBlocks,
               py::arg("token_buffer"), py::arg("block_size"), py::arg("parent_hash"));
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
