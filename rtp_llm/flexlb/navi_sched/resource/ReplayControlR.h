#pragma once

#include "rtp_llm/flexlb/navi_sched/data/ReplayControlData.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include "navi/rpc_server/NaviRpcServerR.h"

#include <autil/Lock.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

class ReplayControlResponseData : public navi::Data {
public:
    ReplayControlResponseData();

    bool accepted = false;
    std::string error;
    uint64_t generation = 0;
    ReplayAction action = ReplayAction::STOP;
    ReplayMode mode = ReplayMode::FIXED_QPS;
    ReplayHistoryMode historyMode = ReplayHistoryMode::TIME_RANGE;
    size_t eligibleClientCount = 0;
    double perClientQps = 0.0;
    uint64_t recentQueryCount = 0;
    uint64_t historyQueryId = 0;
    size_t expectedClientCount = 0;
    size_t respondedClientCount = 0;
    std::vector<ReplayHistorySummary> clients;
};

using ReplayControlResponseDataPtr = std::shared_ptr<ReplayControlResponseData>;

class ReplayControlR : public navi::Resource {
public:
    struct Request {
        std::string action;
        std::string mode;
        std::string historyMode;
        int64_t startTimeUs = 0;
        int64_t endTimeUs = 0;
        int64_t durationMs = 0;
        double totalQps = 0.0;
        uint64_t loopCount = 0;
        uint64_t recentQueryCount = 0;
    };

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;

    bool processRequest(const Request &request,
                        ReplayControlResponseDataPtr &response);
    bool processRequestAsync(const Request &request,
                             const navi::AsyncPipePtr &responsePipe,
                             uint64_t &historyQueryId,
                             ReplayControlResponseDataPtr &immediateResponse);
    void timeoutHistoryQuery(uint64_t historyQueryId);

    bool subscribe(const std::string &clientId,
                   const navi::AsyncPipePtr &commandPipe,
                   const navi::AsyncPipePtr &historyQueryPipe);
    bool subscribe(const std::string &clientId,
                   const navi::AsyncPipePtr &pipe)
    {
        return subscribe(clientId, pipe, pipe);
    }
    void unsubscribe(const std::string &clientId,
                     const navi::AsyncPipePtr &commandPipe,
                     const navi::AsyncPipePtr &historyQueryPipe);
    void unsubscribe(const std::string &clientId,
                     const navi::AsyncPipePtr &pipe)
    {
        unsubscribe(clientId, pipe, pipe);
    }
    void completeHistoryQuery(const ReplayHistoryQueryResponseData &response);
    void updateHistory(const ReplayHistorySummary &summary);
    std::vector<ReplayHistorySummary> getHistory() const;
    void stop();

    static const std::string RESOURCE_ID;
    static const std::string RPC_REQUEST_NAME;

private:
    struct Subscriber {
        std::weak_ptr<navi::AsyncPipe> commandPipe;
        std::weak_ptr<navi::AsyncPipe> historyQueryPipe;
        ReplayHistorySummary summary;
        int64_t clockOffsetUs = 0;
        bool hasClockOffset = false;
    };

    struct PendingHistoryQuery {
        uint64_t queryId = 0;
        std::map<std::string, ReplayHistorySummary> responses;
        std::map<std::string, std::string> failures;
        std::vector<std::string> expectedClients;
        navi::AsyncPipePtr responsePipe;
    };

    struct CommandState {
        bool active = false;
        ReplayMode mode = ReplayMode::FIXED_QPS;
        ReplayHistoryMode historyMode = ReplayHistoryMode::TIME_RANGE;
        int64_t startTimeUs = 0;
        int64_t endTimeUs = 0;
        int64_t durationMs = 0;
        double totalQps = 0.0;
        uint64_t loopCount = 0;
        uint64_t recentQueryCount = 0;
        uint64_t generation = 0;
        uint64_t selectionGeneration = 0;
        int64_t deadlineTimeUs = 0;
        int64_t pacingEpochUs = 0;
    };

    bool validateStart(const Request &request, ReplayMode &mode,
                       ReplayHistoryMode &historyMode,
                       int64_t &endTimeUs, std::string &error) const;
    bool isEligible(const ReplayHistorySummary &summary,
                    const CommandState &command) const;
    bool isSelectionCandidate(const ReplayHistorySummary &summary,
                              const CommandState &command) const;
    ReplayControlCommandDataPtr makeCommandLocked(
        const ReplayHistorySummary &summary) const;
    ReplayControlCommandDataPtr makeCommandLocked(
        const ReplayHistorySummary &summary, size_t pacingIndex,
        size_t eligibleCount) const;
    void notifySubscribersLocked();
    void notifyClockOffsetLocked(const Subscriber &subscriber);
    bool historyQueryCompleteLocked() const;
    ReplayControlResponseDataPtr finishHistoryQueryLocked(
        const std::string &timeoutError);
    static void publishAsyncResponse(
        const navi::AsyncPipePtr &pipe,
        const ReplayControlResponseDataPtr &response);
    ReplayControlResponseDataPtr makeResponseLocked(bool accepted,
                                                     const std::string &error) const;
    bool expireCommandIfNeededLocked();
    bool finishCommandIfAllClientsCompletedLocked();
    int64_t remainingDurationMsLocked() const;
    void initializeReplayEpochLocked();
    void rebasePacingEpochLocked(double newTotalQps);
    void resetPacingEpochLocked();
    bool updateClockOffsetLocked(Subscriber &subscriber,
                                 int64_t clientReportTimeUs,
                                 int64_t masterReceiveTimeUs);
    void clearSelectionStateLocked();
    bool buildGraph(navi::GraphDef &graphDef) const;

private:
    mutable autil::ThreadMutex _mutex;
    std::map<std::string, Subscriber> _subscribers;
    CommandState _command;
    bool _stopped = false;
    uint64_t _nextHistoryQueryId = 0;
    std::shared_ptr<PendingHistoryQuery> _pendingHistoryQuery;
    std::string _bizName;
    std::string _initialConfigJson;
    navi::NaviRpcServerR *_naviRpcServerR = nullptr;
};

NAVI_TYPEDEF_PTR(ReplayControlR);

} // namespace navi_sched
