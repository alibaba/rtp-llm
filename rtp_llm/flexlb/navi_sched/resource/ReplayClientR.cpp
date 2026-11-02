#include "aios/navi_sched/resource/ReplayClientR.h"

#include "navi/engine/NaviRegisterMacro.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <system_error>

namespace navi_sched {
namespace {

int64_t currentWallTimeUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t adjustedPacingEpochUs(int64_t pacingEpochUs,
                              int64_t clockOffsetUs)
{
    if (pacingEpochUs <= 0) {
        return pacingEpochUs;
    }
    if (clockOffsetUs > 0 &&
        pacingEpochUs > std::numeric_limits<int64_t>::max() - clockOffsetUs) {
        return std::numeric_limits<int64_t>::max();
    }
    if (clockOffsetUs < 0 &&
        pacingEpochUs < std::numeric_limits<int64_t>::min() - clockOffsetUs) {
        return std::numeric_limits<int64_t>::min();
    }
    return pacingEpochUs + clockOffsetUs;
}

std::chrono::microseconds fixedPacingOffset(double totalQps,
                                             uint64_t pacingIndex,
                                             uint64_t pacingClientCount)
{
    if (!(totalQps > 0.0) || !std::isfinite(totalQps) ||
        pacingClientCount == 0 || pacingIndex >= pacingClientCount) {
        return std::chrono::microseconds::zero();
    }
    const auto offsetUs =
        (static_cast<long double>(pacingIndex) + 0.5L) * 1000000.0L /
        static_cast<long double>(totalQps);
    if (!std::isfinite(offsetUs) ||
        offsetUs >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return std::chrono::microseconds::max();
    }
    return std::chrono::microseconds(
        std::max<int64_t>(0, static_cast<int64_t>(offsetUs)));
}

std::chrono::microseconds fixedPacingPeriod(double totalQps,
                                            double perClientQps,
                                            uint64_t pacingClientCount)
{
    long double periodUs = 0.0L;
    if (totalQps > 0.0 && std::isfinite(totalQps) &&
        pacingClientCount > 0) {
        periodUs = static_cast<long double>(pacingClientCount) * 1000000.0L /
                   static_cast<long double>(totalQps);
    } else if (perClientQps > 0.0 && std::isfinite(perClientQps)) {
        periodUs = 1000000.0L / static_cast<long double>(perClientQps);
    } else {
        return std::chrono::microseconds::zero();
    }
    if (!std::isfinite(periodUs) ||
        periodUs >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return std::chrono::microseconds::max();
    }
    return std::chrono::microseconds(
        std::max<int64_t>(1, static_cast<int64_t>(periodUs)));
}

std::chrono::steady_clock::time_point wallClockTimeToSteady(int64_t wallClockUs)
{
    const auto steadyNow = std::chrono::steady_clock::now();
    if (wallClockUs <= 0) {
        return steadyNow;
    }
    const auto systemNowUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
    const auto deltaUs = static_cast<long double>(wallClockUs) -
                         static_cast<long double>(systemNowUs);
    const auto boundedDeltaUs = std::clamp<long double>(
        deltaUs,
        static_cast<long double>(std::numeric_limits<int64_t>::min()),
        static_cast<long double>(std::numeric_limits<int64_t>::max()));
    return steadyNow + std::chrono::microseconds(
                           static_cast<int64_t>(boundedDeltaUs));
}

std::chrono::steady_clock::time_point nextFixedPacingTime(
    double totalQps, uint64_t pacingIndex, uint64_t pacingClientCount,
    int64_t pacingEpochUs)
{
    const auto steadyNow = std::chrono::steady_clock::now();
    const auto maxDelay = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::time_point::max() - steadyNow);
    if (pacingEpochUs <= 0) {
        return steadyNow + std::min(
                               fixedPacingOffset(totalQps, pacingIndex,
                                                 pacingClientCount),
                               maxDelay);
    }
    const auto systemNowUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
    const auto period = fixedPacingPeriod(totalQps, 0.0, pacingClientCount);
    if (period <= std::chrono::microseconds::zero()) {
        return steadyNow;
    }
    long double nextUs = static_cast<long double>(pacingEpochUs) +
                         (static_cast<long double>(pacingIndex) + 0.5L) *
                             1000000.0L /
                             static_cast<long double>(totalQps);
    if (nextUs <= static_cast<long double>(systemNowUs)) {
        const auto periods = std::floor(
                                 (static_cast<long double>(systemNowUs) - nextUs) /
                                 static_cast<long double>(period.count())) +
                             1.0L;
        nextUs += periods * static_cast<long double>(period.count());
    }
    const auto maxDelayUs = maxDelay.count();
    const auto delayUs = std::clamp<long double>(
        nextUs - static_cast<long double>(systemNowUs), 0.0L,
        static_cast<long double>(maxDelayUs));
    return steadyNow + std::chrono::microseconds(
                           static_cast<int64_t>(delayUs));
}

void advanceFixedPacingTime(std::chrono::steady_clock::time_point &target,
                            std::chrono::microseconds period)
{
    if (period <= std::chrono::microseconds::zero()) {
        return;
    }
    const auto maxAdvance = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::time_point::max() - target);
    if (period >= maxAdvance) {
        target = std::chrono::steady_clock::time_point::max();
        return;
    }
    target += period;
    const auto now = std::chrono::steady_clock::now();
    if (target > now) {
        return;
    }
    const auto lateUs =
        std::chrono::duration_cast<std::chrono::microseconds>(now - target).count();
    const auto periods = lateUs / period.count() + 1;
    const auto remainingUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::time_point::max() - target)
            .count();
    if (periods > remainingUs / period.count()) {
        target = std::chrono::steady_clock::time_point::max();
        return;
    }
    target += period * periods;
}

}

AUTIL_LOG_SETUP(navi_sched, ReplayClientR);

const std::string ReplayClientR::RESOURCE_ID = "navi_sched.replay_client_r";

ReplayClientR::ReplayClientR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

ReplayClientR::~ReplayClientR()
{
    stopReplay();
    stopHistoryWriter();
}

void ReplayClientR::setClientId(const std::string &clientId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _clientId = clientId;
}

bool ReplayClientR::configureHistory(const std::string &directory,
                                     size_t historyLimit,
                                     const std::string &legacyDirectory)
{
    return _historyStore.configure(directory, historyLimit, legacyDirectory);
}

void ReplayClientR::setScheduleCallback(ScheduleCallback callback)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _scheduleCallback = std::move(callback);
}

void ReplayClientR::setStopCallback(StopCallback callback)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _stopCallback = std::move(callback);
}

bool ReplayClientR::validEventTime(int64_t eventTimeUs)
{
    return eventTimeUs > 0;
}

bool ReplayClientR::recordNormalQuery(const NaviSchedScheduleRequest &request,
                                      int64_t eventTimeUs)
{
    if (!request.valid() || !validEventTime(eventTimeUs)) {
        return false;
    }
    return _historyStore.append(request, eventTimeUs);
}

size_t ReplayClientR::historyRequestBytes(const ClientRequestToMasterData &request)
{
    return sizeof(request) + request.bizName.capacity() + request.forwardPayload.capacity() +
           (request.blockHashes ? request.blockHashes->capacity() * sizeof(uint64_t) : 0);
}

void ReplayClientR::enqueueNormalQuery(
    const ClientRequestToMasterDataConstPtr &request, int64_t eventTimeUs)
{
    if (!request || !validEventTime(eventTimeUs)) {
        return;
    }
    const auto bytes = historyRequestBytes(*request);
    std::lock_guard<std::mutex> lock(_historyMutex);
    if (_historyStopRequested) {
        return;
    }
    if (bytes > HISTORY_QUEUE_BYTES_LIMIT) {
        ++_historyDroppedCount;
        return;
    }
    if (!_historyThread.joinable()) {
        _historyThread = std::thread(&ReplayClientR::runHistoryWriter, this);
    }
    while (_historyQueue.size() >= HISTORY_QUEUE_LIMIT ||
           _historyQueueBytes + bytes > HISTORY_QUEUE_BYTES_LIMIT) {
        _historyQueueBytes -= historyRequestBytes(*_historyQueue.front().first);
        _historyQueue.pop_front();
        ++_historyDroppedCount;
    }
    _historyQueue.emplace_back(request, eventTimeUs);
    _historyQueueBytes += bytes;
    _historyCondition.notify_one();
}

void ReplayClientR::runHistoryWriter()
{
    while (true) {
        std::pair<ClientRequestToMasterDataConstPtr, int64_t> item;
        {
            std::unique_lock<std::mutex> lock(_historyMutex);
            _historyCondition.wait(lock, [this] {
                return _historyStopRequested || !_historyQueue.empty();
            });
            if (_historyQueue.empty()) {
                return;
            }
            item = std::move(_historyQueue.front());
            _historyQueue.pop_front();
            _historyQueueBytes -= historyRequestBytes(*item.first);
        }
        const auto &input = *item.first;
        NaviSchedScheduleRequest request;
        request.requestId = input.requestId;
        request.bizName = input.bizName;
        request.promptTokenCount = input.promptTokenCount;
        request.blockSize = input.blockSize;
        request.blockHashes = input.blockHashes;
        request.timeoutMs = input.timeoutMs;
        request.forwardPayload = input.forwardPayload;
        if (!recordNormalQuery(request, item.second)) {
            AUTIL_LOG(WARN, "failed to record client history, requestId [%lu]",
                      input.requestId);
        }
    }
}

void ReplayClientR::stopHistoryWriter()
{
    {
        std::lock_guard<std::mutex> lock(_historyMutex);
        _historyStopRequested = true;
        _historyCondition.notify_all();
    }
    if (_historyThread.joinable()) {
        _historyThread.join();
    }
}

ReplayHistorySummary ReplayClientR::getSummary() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto summary = _historyStore.getSummary(_clientId);
    summary.selectionGeneration = _activeSelectionGeneration;
    summary.selectedSampleCount = _selectedSampleCount;
    summary.completedSelectionGeneration = _completedSelectionGeneration;
    summary.reportTimeUs = currentWallTimeUs();
    return summary;
}

std::vector<ReplayHistorySample> ReplayClientR::selectSamples(
    int64_t startTimeUs, int64_t endTimeUs) const
{
    std::vector<ReplayHistorySample> result;
    std::string error;
    _historyStore.loadTimeRange(startTimeUs, endTimeUs, result, error);
    return result;
}

std::vector<ReplayHistorySample> ReplayClientR::selectRecentSamples(size_t count) const
{
    std::vector<ReplayHistorySample> result;
    std::string error;
    _historyStore.loadRecent(count, result, error);
    return result;
}

bool ReplayClientR::subscribeHistory(const navi::AsyncPipePtr &pipe)
{
    if (!pipe) {
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    _historySubscribers.emplace_back(pipe);
    auto data = std::make_shared<ReplayClientCapabilityData>();
    data->clientId = _clientId;
    data->summary.clientId = _clientId;
    data->summary.reportTimeUs = currentWallTimeUs();
    return navi::EC_NONE == pipe->setData(data);
}

void ReplayClientR::unsubscribeHistory(const navi::AsyncPipePtr &pipe)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _historySubscribers.erase(
        std::remove_if(_historySubscribers.begin(), _historySubscribers.end(),
                       [&pipe](const auto &item) {
                           return !item.lock() || item.lock() == pipe;
                       }),
        _historySubscribers.end());
}

bool ReplayClientR::subscribeHistoryQueryResponse(const navi::AsyncPipePtr &pipe)
{
    if (!pipe) {
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    _historyQueryResponseSubscribers.emplace_back(pipe);
    return true;
}

void ReplayClientR::unsubscribeHistoryQueryResponse(const navi::AsyncPipePtr &pipe)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _historyQueryResponseSubscribers.erase(
        std::remove_if(_historyQueryResponseSubscribers.begin(),
                       _historyQueryResponseSubscribers.end(),
                       [&pipe](const auto &item) {
                           return !item.lock() || item.lock() == pipe;
                       }),
        _historyQueryResponseSubscribers.end());
}

bool ReplayClientR::answerHistoryQuery(uint64_t queryId)
{
    if (queryId == 0) {
        return false;
    }
    auto data = std::make_shared<ReplayHistoryQueryResponseData>();
    data->queryId = queryId;
    data->summary = getSummary();
    const auto &clientId = data->summary.clientId;
    std::lock_guard<std::mutex> lock(_mutex);
    bool delivered = false;
    for (auto it = _historyQueryResponseSubscribers.begin();
         it != _historyQueryResponseSubscribers.end();) {
        auto pipe = it->lock();
        if (!pipe || navi::EC_NONE != pipe->setData(data)) {
            it = _historyQueryResponseSubscribers.erase(it);
        } else {
            delivered = true;
            ++it;
        }
    }
    AUTIL_LOG(INFO,
              "prepared replay history response, query [%lu], client [%s], "
              "delivered [%d]",
              queryId, clientId.c_str(), delivered);
    return delivered;
}

bool ReplayClientR::applyCommand(const ReplayControlCommandData &command)
{
    std::string clientId;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        clientId = _clientId;
        if (_hasAppliedCommand && command.generation != 0 &&
            command.generation <= _lastAppliedCommandGeneration) {
            return true;
        }
        _hasAppliedCommand = true;
        _lastAppliedCommandGeneration = command.generation;
        if (isFixedQpsPacingUpdateLocked(command)) {
            _activeTotalQps = command.totalQps;
            _activePerClientQps = command.perClientQps;
            _activePacingIndex = command.pacingIndex;
            _activePacingClientCount = command.pacingClientCount;
            _activePacingEpochUs = command.pacingEpochUs;
            _activePacingClockOffsetUs = command.pacingClockOffsetUs;
            ++_pacingRevision;
            AUTIL_LOG(INFO,
                      "updated fixed replay pacing, client [%s], generation [%lu], "
                      "total qps [%f], per client qps [%f], pacing index [%lu], "
                      "client count [%lu], master epoch [%ld], clock offset [%ld]",
                      _clientId.c_str(), command.generation, command.totalQps,
                      command.perClientQps, command.pacingIndex,
                      command.pacingClientCount, command.pacingEpochUs,
                      command.pacingClockOffsetUs);
            _condition.notify_all();
            return true;
        }
        if (command.action == ReplayAction::START &&
            command.selectionGeneration != 0 && _activeSelectionCompleted &&
            isSameReplaySelectionLocked(command)) {
            return true;
        }
        if (_replaying && command.action == ReplayAction::START &&
            command.mode == ReplayMode::REALTIME &&
            isSameReplaySelectionLocked(command)) {
            return true;
        }
    }
    AUTIL_LOG(INFO,
              "replay command transition, client [%s], action [%d], "
              "generation [%lu], selection generation [%lu]",
              clientId.c_str(), static_cast<int>(command.action),
              command.generation, command.selectionGeneration);
    stopReplay();
    if (command.action == ReplayAction::STOP) {
        if (command.selectionGeneration != 0) {
            std::lock_guard<std::mutex> lock(_mutex);
            _activeSelectionGeneration = command.selectionGeneration;
            _selectedSampleCount = 0;
        }
        return true;
    }
    std::vector<ReplayHistorySample> samples;
    std::string loadError;
    const bool loaded = command.historyMode == ReplayHistoryMode::RECENT_COUNT
                            ? _historyStore.loadRecent(command.recentQueryCount,
                                                       samples, loadError)
                            : _historyStore.loadTimeRange(command.startTimeUs,
                                                          command.endTimeUs,
                                                          samples, loadError);
    if (!loaded) {
        AUTIL_LOG(ERROR, "load replay history failed: %s", loadError.c_str());
        publishSelectionResult(command.selectionGeneration, 0);
        return true;
    }
    if (samples.empty()) {
        publishSelectionResult(command.selectionGeneration, 0);
        return true;
    }
    const auto selectedSampleCount = samples.size();
    ScheduleCallback callback;
    uint64_t generation;
    auto commandCopy = std::make_shared<ReplayControlCommandData>();
    commandCopy->action = command.action;
    commandCopy->mode = command.mode;
    commandCopy->historyMode = command.historyMode;
    commandCopy->generation = command.generation;
    commandCopy->startTimeUs = command.startTimeUs;
    commandCopy->endTimeUs = command.endTimeUs;
    commandCopy->durationMs = command.durationMs;
    commandCopy->totalQps = command.totalQps;
    commandCopy->perClientQps = command.perClientQps;
    commandCopy->loopCount = command.loopCount;
    commandCopy->recentQueryCount = command.recentQueryCount;
    commandCopy->selectionGeneration = command.selectionGeneration;
    commandCopy->pacingIndex = command.pacingIndex;
    commandCopy->pacingClientCount = command.pacingClientCount;
    commandCopy->pacingEpochUs = command.pacingEpochUs;
    commandCopy->pacingClockOffsetUs = command.pacingClockOffsetUs;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        callback = _scheduleCallback;
        if (!callback) {
            return false;
        }
        _stopRequested = false;
        generation = ++_replayGeneration;
        setActiveCommandLocked(command);
        _selectedSampleCount = selectedSampleCount;
        ++_pacingRevision;
        _replaying = true;
        _loadedReplaySampleCount = samples.size();
        AUTIL_LOG(INFO,
                  "starting replay, client [%s], generation [%lu], mode [%d], "
                  "sample count [%lu], total qps [%f], per client qps [%f], "
                  "pacing index [%lu], client count [%lu], master epoch [%ld], "
                  "clock offset [%ld]",
                  _clientId.c_str(), command.generation,
                  static_cast<int>(command.mode), selectedSampleCount,
                  command.totalQps, command.perClientQps, command.pacingIndex,
                  command.pacingClientCount, command.pacingEpochUs,
                  command.pacingClockOffsetUs);
        try {
            _replayThread = std::thread(
                [this, commandCopy = std::move(commandCopy),
                 samples = std::move(samples), generation]() mutable {
                    runReplay(*commandCopy, std::move(samples));
                    bool completed = false;
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        if (_replayGeneration == generation) {
                            completed = !_stopRequested;
                            _activeSelectionCompleted = completed;
                            _loadedReplaySampleCount = 0;
                        }
                    }
                    if (completed) {
                        publishCompletionResult(
                            commandCopy->selectionGeneration);
                    }
                    std::lock_guard<std::mutex> lock(_mutex);
                    if (_replayGeneration == generation) {
                        _replaying = false;
                    }
                });
        } catch (const std::system_error &error) {
            _stopRequested = true;
            _replaying = false;
            _loadedReplaySampleCount = 0;
            AUTIL_LOG(ERROR, "start replay thread failed: %s", error.what());
            return false;
        }
    }
    publishSelectionResult(command.selectionGeneration, selectedSampleCount);
    return true;
}

bool ReplayClientR::isFixedQpsPacingUpdateLocked(
    const ReplayControlCommandData &command) const
{
    return _replaying && command.action == ReplayAction::START &&
           command.mode == ReplayMode::FIXED_QPS &&
           isSameReplaySelectionLocked(command) &&
           std::isfinite(command.perClientQps) && command.perClientQps > 0.0;
}

bool ReplayClientR::isSameReplaySelectionLocked(
    const ReplayControlCommandData &command) const
{
    return command.mode == _activeMode &&
           command.historyMode == _activeHistoryMode &&
           command.startTimeUs == _activeStartTimeUs &&
           command.endTimeUs == _activeEndTimeUs &&
           command.loopCount == _activeLoopCount &&
           command.recentQueryCount == _activeRecentQueryCount &&
           command.selectionGeneration == _activeSelectionGeneration;
}

void ReplayClientR::setActiveCommandLocked(
    const ReplayControlCommandData &command)
{
    _activeMode = command.mode;
    _activeHistoryMode = command.historyMode;
    _activeStartTimeUs = command.startTimeUs;
    _activeEndTimeUs = command.endTimeUs;
    _activeDurationMs = command.durationMs;
    _activeTotalQps = command.totalQps;
    _activePerClientQps = command.perClientQps;
    _activePacingIndex = command.pacingIndex;
    _activePacingClientCount = command.pacingClientCount;
    _activePacingEpochUs = command.pacingEpochUs;
    _activePacingClockOffsetUs = command.pacingClockOffsetUs;
    _activeLoopCount = command.loopCount;
    _activeRecentQueryCount = command.recentQueryCount;
    _activeSelectionGeneration = command.selectionGeneration;
    _activeSelectionCompleted = false;
    _completedSelectionGeneration = 0;
}

void ReplayClientR::publishSelectionResult(uint64_t selectionGeneration,
                                           size_t selectedSampleCount)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _activeSelectionGeneration = selectionGeneration;
    _selectedSampleCount = selectedSampleCount;
    auto data = std::make_shared<ReplayClientCapabilityData>();
    data->clientId = _clientId;
    data->summary = _historyStore.getSummary(_clientId);
    data->summary.selectionGeneration = selectionGeneration;
    data->summary.selectedSampleCount = selectedSampleCount;
    data->summary.completedSelectionGeneration =
        _completedSelectionGeneration;
    data->summary.reportTimeUs = currentWallTimeUs();
    for (auto it = _historySubscribers.begin();
         it != _historySubscribers.end();) {
        auto pipe = it->lock();
        if (!pipe || navi::EC_NONE != pipe->setData(data)) {
            it = _historySubscribers.erase(it);
        } else {
            ++it;
        }
    }
}

void ReplayClientR::publishCompletionResult(uint64_t selectionGeneration)
{
    if (selectionGeneration == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_activeSelectionCompleted ||
        _activeSelectionGeneration != selectionGeneration) {
        return;
    }
    _completedSelectionGeneration = selectionGeneration;
    auto data = std::make_shared<ReplayClientCapabilityData>();
    data->clientId = _clientId;
    data->summary = _historyStore.getSummary(_clientId);
    data->summary.selectionGeneration = _activeSelectionGeneration;
    data->summary.selectedSampleCount = _selectedSampleCount;
    data->summary.completedSelectionGeneration =
        _completedSelectionGeneration;
    data->summary.reportTimeUs = currentWallTimeUs();
    for (auto it = _historySubscribers.begin();
         it != _historySubscribers.end();) {
        auto pipe = it->lock();
        if (!pipe || navi::EC_NONE != pipe->setData(data)) {
            it = _historySubscribers.erase(it);
        } else {
            ++it;
        }
    }
}

bool ReplayClientR::waitUntil(std::chrono::steady_clock::time_point deadline,
                              uint64_t generation)
{
    std::unique_lock<std::mutex> lock(_mutex);
    return _condition.wait_until(lock, deadline, [this, generation] {
        return _stopRequested || _replayGeneration != generation;
    });
}

void ReplayClientR::runReplay(const ReplayControlCommandData &command,
                              std::vector<ReplayHistorySample> samples)
{
    const auto started = std::chrono::steady_clock::now();
    const auto duration = std::chrono::milliseconds(std::max<int64_t>(0, command.durationMs));
    uint64_t generation;
    ScheduleCallback callback;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        generation = _replayGeneration;
        callback = _scheduleCallback;
    }
    const auto maxEnd = command.durationMs > 0 ? started + duration
                                               : std::chrono::steady_clock::time_point::max();
    uint64_t completedLoops = 0;
    uint64_t pacingRevision = 0;
    auto activeTotalQps = command.totalQps;
    auto clientPacingEpochUs = adjustedPacingEpochUs(
        command.pacingEpochUs, command.pacingClockOffsetUs);
    auto pacingIndex = command.pacingIndex;
    auto pacingClientCount = command.pacingClientCount;
    auto nextFixedSendTime = nextFixedPacingTime(
        activeTotalQps, pacingIndex, pacingClientCount,
        clientPacingEpochUs);
    {
        std::lock_guard<std::mutex> lock(_mutex);
        pacingRevision = _pacingRevision;
    }
    const auto firstEventUs = samples.front().eventTimeUs;
    const auto lastEventUs = samples.back().eventTimeUs;
    const auto sourceOriginUs =
        command.historyMode == ReplayHistoryMode::TIME_RANGE
            ? command.startTimeUs
            : firstEventUs;
    const auto sourceSpanUs = std::max<int64_t>(
        0, command.historyMode == ReplayHistoryMode::TIME_RANGE
               ? command.endTimeUs - command.startTimeUs
               : lastEventUs - firstEventUs);
    auto realtimeLoopStart = command.pacingEpochUs > 0
                                 ? wallClockTimeToSteady(clientPacingEpochUs)
                                 : started;
    const auto realtimeExpiryCutoff = std::chrono::steady_clock::now();
    while (command.loopCount == 0 || completedLoops < command.loopCount) {
        const auto loopStart = command.mode == ReplayMode::REALTIME
                                   ? realtimeLoopStart
                                   : started;
        for (size_t index = 0; index < samples.size(); ++index) {
            auto target = loopStart;
            if (command.mode == ReplayMode::REALTIME) {
                const auto offsetUs = samples[index].eventTimeUs - sourceOriginUs;
                target += std::chrono::microseconds(std::max<int64_t>(0, offsetUs));
            } else {
                while (true) {
                    std::unique_lock<std::mutex> lock(_mutex);
                    if (_stopRequested || _replayGeneration != generation) {
                        return;
                    }
                    const auto qps = _activePerClientQps;
                    if (!(qps > 0.0) || !std::isfinite(qps)) {
                        return;
                    }
                    if (_pacingRevision != pacingRevision) {
                        pacingRevision = _pacingRevision;
                        const auto updatedEpochUs = adjustedPacingEpochUs(
                            _activePacingEpochUs,
                            _activePacingClockOffsetUs);
                        activeTotalQps = _activeTotalQps;
                        clientPacingEpochUs = updatedEpochUs;
                        pacingIndex = _activePacingIndex;
                        pacingClientCount = _activePacingClientCount;
                        nextFixedSendTime = nextFixedPacingTime(
                            activeTotalQps, pacingIndex,
                            pacingClientCount, clientPacingEpochUs);
                    }
                    const auto interrupted = _condition.wait_until(
                        lock, nextFixedSendTime,
                        [this, generation, pacingRevision] {
                            return _stopRequested ||
                                   _replayGeneration != generation ||
                                   _pacingRevision != pacingRevision;
                        });
                    if (_stopRequested || _replayGeneration != generation) {
                        return;
                    }
                    if (interrupted) {
                        continue;
                    }
                    target = nextFixedSendTime;
                    break;
                }
            }
            if (command.durationMs > 0 && target >= maxEnd) {
                return;
            }
            if (command.mode == ReplayMode::REALTIME &&
                command.pacingEpochUs > 0 &&
                target < realtimeExpiryCutoff) {
                continue;
            }
            if (command.mode == ReplayMode::REALTIME &&
                waitUntil(target, generation)) {
                return;
            }
            auto request = samples[index].request;
            callback(request);
            if (command.mode == ReplayMode::FIXED_QPS) {
                std::lock_guard<std::mutex> lock(_mutex);
                const auto qps = _activePerClientQps;
                if (!(qps > 0.0) || !std::isfinite(qps)) {
                    return;
                }
                advanceFixedPacingTime(
                    nextFixedSendTime,
                    fixedPacingPeriod(_activeTotalQps,
                                      _activePerClientQps,
                                      _activePacingClientCount));
            }
        }
        ++completedLoops;
        if (command.mode == ReplayMode::REALTIME) {
            realtimeLoopStart += std::chrono::microseconds(sourceSpanUs);
        }
        if (command.mode == ReplayMode::REALTIME && sourceSpanUs == 0) {
            if (command.durationMs > 0 && std::chrono::steady_clock::now() >= maxEnd) {
                return;
            }
        }
        if (command.durationMs > 0 && std::chrono::steady_clock::now() >= maxEnd) {
            return;
        }
    }
}

void ReplayClientR::stopReplay()
{
    std::thread thread;
    StopCallback stopCallback;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopRequested = true;
        ++_replayGeneration;
        _condition.notify_all();
        thread = std::move(_replayThread);
        _replaying = false;
        _activeSelectionCompleted = false;
        _activeSelectionGeneration = 0;
        _selectedSampleCount = 0;
        _completedSelectionGeneration = 0;
        stopCallback = _stopCallback;
    }
    if (thread.joinable()) {
        thread.join();
    }
    if (stopCallback) {
        stopCallback();
    }
    std::lock_guard<std::mutex> lock(_mutex);
    _loadedReplaySampleCount = 0;
}

bool ReplayClientR::replaying() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _replaying;
}

size_t ReplayClientR::loadedReplaySampleCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _loadedReplaySampleCount;
}

REGISTER_RESOURCE(ReplayClientR);

}
