#pragma once

#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "navi/engine/Navi.h"
#include <functional>

namespace navi_sched {

struct ClientGraphRunResult {
    navi::ErrorCode errorCode = navi::EC_NO_DATA;
    std::string errorMessage;
    MasterResponseToClientDataPtr decision;
    NaviSchedScheduleReplayPtr replay;
    int64_t masterRequestLatencyUs = 0;
};

class ClientGraphRunner {
public:
    using DecisionCallback = std::function<void(ClientGraphRunResult)>;

    ClientGraphRunner(const navi::NaviPtr &navi, bool enableVis = false,
                      const std::shared_ptr<class ClientResponseR> &responseR = nullptr,
                      bool expectAgentResponseStream = true);

public:
    ClientGraphRunResult schedule(const ClientRequestToMasterDataConstPtr &request,
                                  const std::string &masterBizName, int64_t timeoutMs,
                                  bool expectResponseStream = true);
    ClientGraphRunResult scheduleDetached(
        const ClientRequestToMasterDataConstPtr &request,
        const std::string &masterBizName, int64_t timeoutMs,
        DecisionCallback decisionCallback,
        bool expectResponseStream = false);
    void setClientBizName(const std::string &clientBizName);

private:
    static std::unique_ptr<navi::GraphDef> buildScheduleGraph(
        const std::string &clientBizName, const std::string &masterBizName,
        bool collectReplay, int64_t entryTimeoutMs,
        navi::GraphId &clientGraphId, navi::GraphId &masterGraphId);
    static bool addScheduleRequest(navi::RunGraphParams &params, navi::GraphId graphId,
                                   const ClientRequestToMasterDataConstPtr &request);
    ClientGraphRunResult runRequest(const ClientRequestToMasterDataConstPtr &request,
                                    const std::string &masterBizName,
                                    int64_t timeoutMs,
                                    bool expectResponseStream,
                                    DecisionCallback decisionCallback = {}) const;
    ClientGraphRunResult collectResult(const navi::NaviUserResultPtr &userResult,
                                       uint64_t requestId) const;

private:
    navi::NaviPtr _navi;
    std::string _clientBizName;
    bool _enableVis = false;
    std::shared_ptr<class ClientResponseR> _responseR;
    bool _expectAgentResponseStream = true;
};

}
