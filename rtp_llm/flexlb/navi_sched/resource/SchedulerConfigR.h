#pragma once

#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/scheduler/CostSchedulerConfig.h"
#include "aios/navi_sched/scheduler/ExecutionUtilizationController.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Data.h"
#include "navi/engine/Resource.h"
#include "navi/rpc_server/NaviRpcServerR.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

class SchedulerConfigResponseData : public navi::Data {
public:
    struct AgentResult {
        std::string ip;
        std::string hardware;
        bool updated = false;
        std::string error;
        std::string effectiveHash;
        std::string effectiveConfigJson;
    };

    SchedulerConfigResponseData();

    std::string configJson;
    std::string hardware;
    uint64_t generation = 0;
    bool accepted = false;
    std::string error;
    size_t targetAgentCount = 0;
    size_t updatedAgentCount = 0;
    size_t failedAgentCount = 0;
    std::vector<AgentResult> agentResults;
};

using SchedulerConfigResponseDataPtr = std::shared_ptr<SchedulerConfigResponseData>;

class SchedulerConfigR : public navi::Resource {
public:
    struct Config : autil::legacy::Jsonizable {
        void Jsonize(JsonWrapper &json) override;
        bool valid() const;

        CostSchedulerConfig cost;
        bool peerCacheEnabled = false;
        std::vector<std::string> peerCacheReadyAgents;
        size_t maxWaitScheduleCount = 30;
        int64_t maxWaitScheduleTimeMs = 30;
        int64_t queueOverloadStartCycleMultiplier = 10;
        int64_t queueOverloadFullCycleMultiplier = 30;
        std::string overloadControlStrategy = "latency";
        ExecutionUtilizationController::Config executionUtilization;
        int64_t admissionTimeoutMs = 1000;
        AgentSchedulerConfig agent;
    };

    struct Snapshot {
        Config config;
        uint64_t generation = 1;
        std::string agentConfigHash = AgentSchedulerConfigR::hashConfig(config.agent);
    };

    struct Request {
        bool update = false;
        std::string configJson;
        std::string hardware;
        navi::AsyncPipePtr responsePipe;
        uint64_t updateToken = 0;
    };

    struct PendingAgentUpdate {
        uint64_t requestGeneration = 0;
        std::string configHash;
        std::string configJson;
    };

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    Snapshot get() const;
    bool setSyncPipe(const navi::AsyncPipePtr &syncPipe);
    void clearSyncPipe(const navi::AsyncPipePtr &syncPipe);
    bool stopSyncPipe();
    bool submit(Request request);
    bool takeRequest(Request &request);
    bool processRequest(const Request &request, SchedulerConfigResponseDataPtr &response,
                        bool &waitingForAgents);
    bool finishAgentSync(bool timedOut, SchedulerConfigResponseDataPtr &response,
                         navi::AsyncPipePtr &responsePipe);
    bool hasActiveAgentSync() const;
    int64_t getAgentSyncRemainingTimeoutMs(int64_t currentTimeUs) const;
    bool subscribe(const std::string &subscriberId,
                   const navi::AsyncPipePtr &notificationPipe);
    bool activateSubscriber(const std::string &subscriberId, const std::string &agentId,
                            const std::string &agentIp, uint64_t configGeneration,
                            const std::string &configHash, const std::string &configJson,
                            const std::string &latencyModelFingerprint = "");
    void reportAgentConfig(const std::string &subscriberId, uint64_t requestGeneration,
                           uint64_t configGeneration, const std::string &configHash,
                           const std::string &configJson,
                           const std::string &errorMessage);
    bool getPendingAgentUpdate(const std::string &subscriberId,
                               PendingAgentUpdate &update);
    void unsubscribe(const std::string &subscriberId,
                     const navi::AsyncPipePtr &notificationPipe);
    static std::string toJson(const Config &config);
    static std::string toAgentJson(const Config &config);

public:
    static const std::string RESOURCE_ID;
    static const std::string RPC_REQUEST_NAME;

private:
    bool buildGraph(navi::GraphDef &graphDef) const;
    static bool applyPatch(const Config &current, const std::string &configJson,
                           Config &updated, std::string &errorMessage);
    static bool applyAgentPatch(const AgentSchedulerConfig &current,
                                const std::string &configJson,
                                AgentSchedulerConfig &updated, bool &hasAgentFields,
                                std::string &errorMessage);
    static bool mergeAgentPatch(const std::string &currentPatchJson,
                                const std::string &configJson,
                                std::string &updatedPatchJson, bool &hasAgentFields,
                                std::string &errorMessage);
    static std::string extractHardware(const std::string &latencyModelFingerprint);
    bool applyStoredAgentPatchesLocked(const AgentSchedulerConfig &current,
                                       const std::string &hardware,
                                       const std::string *globalPatchOverride,
                                       const std::string *hardwarePatchOverride,
                                       AgentSchedulerConfig &updated,
                                       std::string &errorMessage) const;

    struct Subscriber {
        std::weak_ptr<navi::AsyncPipe> notificationPipe;
        bool active = false;
        std::string agentId;
        std::string agentIp;
        std::string hardware;
        uint64_t configGeneration = 0;
        std::string configHash;
        std::string configJson;
        uint64_t pendingConfigGeneration = 0;
        std::string pendingConfigHash;
        std::string pendingConfigJson;
        bool pendingConfigSent = false;
    };

    struct AgentSyncTarget {
        std::string agentIp;
        std::string hardware;
        std::string expectedHash;
        std::string expectedJson;
        std::string effectiveHash;
        std::string effectiveJson;
        std::string errorMessage;
        bool sent = false;
        bool completed = false;
    };

    struct AgentSyncState {
        uint64_t updateToken = 0;
        uint64_t requestGeneration = 0;
        std::string hardware;
        int64_t deadlineTimeUs = 0;
        navi::AsyncPipePtr responsePipe;
        std::map<std::string, AgentSyncTarget> targets;
    };
    uint64_t acquireUpdateTokenLocked();
    void releaseUpdateTokenLocked(uint64_t updateToken);
    void logRejectedUpdateLocked(const std::string &configJson,
                                 const std::string &errorMessage) const;
    void logCompletedUpdateLocked(uint64_t generation, bool accepted, bool timedOut,
                                  size_t targetAgentCount, size_t updatedAgentCount,
                                  size_t failedAgentCount,
                                  const std::string &errorMessage) const;
    SchedulerConfigResponseDataPtr makeResponseLocked(
        bool accepted, const std::string &errorMessage) const;
    bool notifySyncPipe(const navi::AsyncPipePtr &syncPipe) const;

private:
    mutable autil::ThreadMutex _mutex;
    Snapshot _snapshot;
    std::deque<Request> _requests;
    navi::AsyncPipePtr _syncPipe;
    std::map<std::string, Subscriber> _subscribers;
    std::unique_ptr<AgentSyncState> _agentSync;
    uint64_t _activeUpdateToken = 0;
    uint64_t _nextUpdateToken = 1;
    std::string _agentConfigPatchJson;
    std::map<std::string, std::string> _hardwareAgentConfigPatchJson;
    uint64_t _agentConfigGeneration = 0;
    std::string _initialConfigJson;
    std::string _bizName;
    navi::NaviRpcServerR *_naviRpcServerR = nullptr;
};

NAVI_TYPEDEF_PTR(SchedulerConfigR);

}
