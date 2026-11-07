#pragma once

#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/resource/AsyncNotification.h"
#include "navi/engine/NaviUserResult.h"
#include "navi/engine/Resource.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace navi_sched {

class ClientResponseR : public navi::ExternalResource {
public:
    struct DecisionResult {
        navi::ErrorCode errorCode = navi::EC_NO_DATA;
        std::string errorMessage;
        MasterResponseToClientDataPtr decision;
        NaviSchedScheduleReplayPtr replay;
        int64_t decisionTimeUs = 0;
    };

    ClientResponseR();

public:
    bool begin(uint64_t requestId, bool expectReplay, bool expectResponseStream = true,
               bool hasClosureCompletion = false);
    bool attach(uint64_t requestId, const navi::NaviUserResultPtr &userResult);
    bool publishDecision(uint64_t requestId,
                         const MasterResponseToClientDataPtr &decision);
    bool publishReplay(uint64_t requestId, const MasterResponseToClientDataPtr &replay);
    bool publishResponse(uint64_t requestId,
                         const AgentResponseToClientDataPtr &response);
    DecisionResult waitDecision(uint64_t requestId, int64_t timeoutMs);
    std::optional<DecisionResult> takeDecision(uint64_t requestId);
    int notificationFd() { return _notification.fd(); }
    AgentResponseToClientDataPtr takeNextResponse(int64_t timeoutMs);
    bool abort(uint64_t requestId);
    DecisionResult finishGraph(uint64_t requestId, navi::ErrorCode errorCode,
                               const std::string &errorMessage,
                               bool consumeDecision = false);
    void finishClosureCompletion(uint64_t requestId);
    void finishAll();
    size_t requestCount() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    struct RequestState {
        std::mutex mutex;
        std::condition_variable condition;
        navi::NaviUserResultPtr userResult;
        MasterResponseToClientDataPtr decision;
        NaviSchedScheduleReplayPtr replay;
        navi::ErrorCode errorCode = navi::EC_NONE;
        std::string errorMessage;
        bool graphFinished = false;
        bool responseFinished = false;
        bool decisionConsumed = false;
        bool expectReplay = false;
        bool expectResponseStream = true;
        bool abortRequested = false;
        bool abortIssued = false;
        bool stopRequested = false;
        bool closureCompletionFinished = true;
        int64_t decisionTimeUs = 0;
    };

    using RequestStatePtr = std::shared_ptr<RequestState>;

    RequestStatePtr find(uint64_t requestId) const;

private:
    mutable std::mutex _mutex;
    std::unordered_map<uint64_t, RequestStatePtr> _states;
    std::condition_variable _responseCondition;
    std::condition_variable _finishCondition;
    std::deque<AgentResponseToClientDataPtr> _responses;
    bool _stopped = false;
    bool _finishCompleted = false;
    AsyncNotification _notification;
};

NAVI_TYPEDEF_PTR(ClientResponseR);

}
