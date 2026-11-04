#pragma once

#include "rtp_llm/flexlb/navi_sched/data/ReplayControlData.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/resource/ReplayHistoryStore.h"
#include "autil/Log.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"

#include <autil/Lock.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace navi_sched {

class ReplayClientR : public navi::ExternalResource {
public:
    ReplayClientR();
    using ScheduleCallback = std::function<void(const NaviSchedScheduleRequest &)>;
    using StopCallback = std::function<void()>;

    ~ReplayClientR() override;

    void setClientId(const std::string &clientId);
    bool configureHistory(const std::string &directory, size_t historyLimit = 20000,
                          const std::string &legacyDirectory = {});
    void setScheduleCallback(ScheduleCallback callback);
    void setStopCallback(StopCallback callback);
    bool recordNormalQuery(const NaviSchedScheduleRequest &request,
                           int64_t eventTimeUs);
    void enqueueNormalQuery(const ClientRequestToMasterDataConstPtr &request,
                            int64_t eventTimeUs);
    ReplayHistorySummary getSummary() const;
    std::vector<ReplayHistorySample> selectSamples(int64_t startTimeUs,
                                                   int64_t endTimeUs) const;
    std::vector<ReplayHistorySample> selectRecentSamples(size_t count) const;
    bool subscribeHistory(const navi::AsyncPipePtr &pipe);
    void unsubscribeHistory(const navi::AsyncPipePtr &pipe);
    bool subscribeHistoryQueryResponse(const navi::AsyncPipePtr &pipe);
    void unsubscribeHistoryQueryResponse(const navi::AsyncPipePtr &pipe);
    bool answerHistoryQuery(uint64_t queryId);
    bool applyCommand(const ReplayControlCommandData &command);
    void stopReplay();
    bool replaying() const;
    size_t loadedReplaySampleCount() const;

    static const std::string RESOURCE_ID;

private:
    void runHistoryWriter();
    void stopHistoryWriter();
    static size_t historyRequestBytes(const ClientRequestToMasterData &request);

    void runReplay(const ReplayControlCommandData &command,
                   std::vector<ReplayHistorySample> samples);
    bool waitUntil(std::chrono::steady_clock::time_point deadline,
                   uint64_t generation);
    bool isFixedQpsPacingUpdateLocked(
        const ReplayControlCommandData &command) const;
    bool isSameReplaySelectionLocked(
        const ReplayControlCommandData &command) const;
    void setActiveCommandLocked(const ReplayControlCommandData &command);
    void publishSelectionResult(uint64_t selectionGeneration,
                                size_t selectedSampleCount);
    void publishCompletionResult(uint64_t selectionGeneration);
    static bool validEventTime(int64_t eventTimeUs);

private:
    AUTIL_LOG_DECLARE();

    mutable std::mutex _mutex;
    std::condition_variable _condition;
    std::string _clientId;
    ReplayHistoryStore _historyStore;
    // Bound queued payloads independently of the existing on-disk history limit.
    static constexpr size_t HISTORY_QUEUE_LIMIT = 256;
    static constexpr size_t HISTORY_QUEUE_BYTES_LIMIT = 16 * 1024 * 1024;
    std::mutex _historyMutex;
    std::condition_variable _historyCondition;
    std::deque<std::pair<ClientRequestToMasterDataConstPtr, int64_t>> _historyQueue;
    size_t _historyQueueBytes = 0;
    uint64_t _historyDroppedCount = 0;
    bool _historyStopRequested = false;
    std::thread _historyThread;
    std::vector<std::weak_ptr<navi::AsyncPipe>> _historySubscribers;
    std::vector<std::weak_ptr<navi::AsyncPipe>> _historyQueryResponseSubscribers;
    ScheduleCallback _scheduleCallback;
    StopCallback _stopCallback;
    std::thread _replayThread;
    uint64_t _replayGeneration = 0;
    uint64_t _lastAppliedCommandGeneration = 0;
    bool _hasAppliedCommand = false;
    bool _stopRequested = false;
    bool _replaying = false;
    bool _activeSelectionCompleted = false;
    size_t _loadedReplaySampleCount = 0;
    ReplayMode _activeMode = ReplayMode::FIXED_QPS;
    ReplayHistoryMode _activeHistoryMode = ReplayHistoryMode::TIME_RANGE;
    int64_t _activeStartTimeUs = 0;
    int64_t _activeEndTimeUs = 0;
    int64_t _activeDurationMs = 0;
    double _activeTotalQps = 0.0;
    double _activePerClientQps = 0.0;
    uint64_t _activePacingIndex = 0;
    uint64_t _activePacingClientCount = 0;
    int64_t _activePacingEpochUs = 0;
    int64_t _activePacingClockOffsetUs = 0;
    uint64_t _activeLoopCount = 0;
    uint64_t _activeRecentQueryCount = 0;
    uint64_t _activeSelectionGeneration = 0;
    uint64_t _selectedSampleCount = 0;
    uint64_t _completedSelectionGeneration = 0;
    uint64_t _pacingRevision = 0;
};

NAVI_TYPEDEF_PTR(ReplayClientR);

} // namespace navi_sched
