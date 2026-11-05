#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/service/NaviSchedRole.h"
#include "autil/LockFreeQueue.h"
#include "autil/Log.h"
#include "navi/common.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_set>

namespace multi_call {
class SearchService;
}

namespace navi_sched {

class ClientGraphRunner;
class FlexlbScheduleGraph;
class ReplayControlGraph;
class ClientResponseR;
class ReplayClientR;
class ClientRequestToMasterData;
class AgentResponseToClientData;
struct ClientGraphRunResult;
class MasterResponseToClientData;

class NaviSchedClient final : public NaviSchedRole {
public:
    explicit NaviSchedClient(const NaviSchedClientConfig &config);
    ~NaviSchedClient() override;

public:
    bool init();
    NaviSchedScheduleResult schedule(const NaviSchedScheduleRequest &request,
                                     const std::string &masterBizName,
                                     bool collectReplay = false,
                                     const std::string &masterAddress = "");
    std::shared_ptr<AgentResponseToClientData> takeNextForwardResponse(
        int64_t timeoutMs);
    bool abortForwardRequest(uint64_t requestId);
    std::tuple<int32_t, std::string, std::string> scheduleFlexlb(
        uint64_t requestId, const std::string &address, const std::string &payload,
        int64_t timeoutMs, const std::vector<std::pair<std::string, std::string>> &metadata);
    uint64_t allocateRequestId();
    bool hasMasterBiz(const std::string &masterBizName = "") const;
    void reportLocalFallback();
    uint64_t getRequestReceivedCount() const;
    void stop() override;

protected:
    std::string makeSnapshot() const override;
    void reportMetrics() const override;

private:
    enum class FailureType {
        TIMEOUT,
        TRANSPORT,
        ROUTING,
        NO_DECISION,
        OTHER,
    };

    struct MasterRequestLatencySample {
        int64_t latencyUs = 0;
        bool success = false;
        FailureType failureType = FailureType::OTHER;
    };

    NaviSchedScheduleResult scheduleRequest(
        const std::shared_ptr<const ClientRequestToMasterData> &request,
        const std::string &masterBizName, int64_t timeoutMs,
        bool expectResponseStream = true, const std::string &masterAddress = "");
    void scheduleReplayRequest(
        const std::shared_ptr<const ClientRequestToMasterData> &request,
        const std::string &masterBizName, int64_t timeoutMs);
    void abortReplayRequests();
    bool replayAbortOccurredSince(uint64_t generation) const;
    void abortReplayRequestLaunchedAcrossStop(uint64_t requestId,
                                              uint64_t generation);
    void recordScheduleResult(
        const std::shared_ptr<const ClientRequestToMasterData> &request,
        const std::string &masterBizName, ClientGraphRunResult &result,
        int64_t masterRequestLatencyUs);
    static FailureType classifyFailure(const ClientGraphRunResult &result);
    static void normalizeGraphResult(ClientGraphRunResult &result);
    static MasterRequestLatencySample makeMasterRequestLatencySample(
        int64_t latencyUs, const ClientGraphRunResult &result);
    void recordMasterFailure(const ClientRequestToMasterData &request,
                             const std::string &masterBizName,
                             const ClientGraphRunResult &result,
                             int64_t masterRequestLatencyUs);
    void logFailure(uint64_t requestId, const std::string &targetBizName,
                    const std::string &masterBizName, int64_t timeoutMs,
                    int64_t masterRequestLatencyUs, int32_t errorCode,
                    const std::string &errorMessage) const;
    static NaviSchedDecision makeDecision(const MasterResponseToClientData &input);
    static NaviSchedScheduleResult makeScheduleResult(const ClientGraphRunResult &input);

private:
    std::string _defaultMasterBizName;
    NaviSchedRuntimeConfig _clientConfig;
    bool _enableAgentForward = false;
    bool _enableSampleCollection = true;
    std::shared_ptr<ClientResponseR> _responseR;
    std::shared_ptr<ReplayClientR> _replayClientR;
    std::shared_ptr<ClientGraphRunner> _runner;
    std::mutex _flexlbGraphMutex;
    std::shared_ptr<FlexlbScheduleGraph> _flexlbGraph;
    std::shared_ptr<ReplayControlGraph> _replayControlGraph;
    mutable std::mutex _masterSearchServiceMutex;
    std::shared_ptr<multi_call::SearchService> _masterSearchService;
    mutable std::mutex _replayRequestsMutex;
    std::unordered_set<uint64_t> _activeReplayRequestIds;
    uint64_t _replayAbortGeneration = 0;
    std::atomic<uint64_t> _nextRequestId{1};
    std::atomic<uint64_t> _requestReceivedCount{0};
    std::atomic<uint64_t> _invalidRequestCount{0};
    std::atomic<uint64_t> _stoppedRequestCount{0};
    std::atomic<uint64_t> _scheduleCount{0};
    std::atomic<uint64_t> _scheduleSuccessCount{0};
    std::atomic<uint64_t> _scheduleErrorCount{0};
    std::atomic<uint64_t> _masterRateLimitedCount{0};
    std::atomic<uint64_t> _masterTimeoutCount{0};
    std::atomic<uint64_t> _masterTransportErrorCount{0};
    std::atomic<uint64_t> _masterRoutingErrorCount{0};
    std::atomic<uint64_t> _masterNoDecisionCount{0};
    std::atomic<uint64_t> _masterOtherErrorCount{0};
    std::atomic<uint64_t> _localFallbackCount{0};
    mutable autil::LockFreeQueue<MasterRequestLatencySample> _masterRequestLatencyQueue;
    mutable std::atomic<int64_t> _lastFailureLogTimeUs{0};
    mutable std::atomic<uint64_t> _suppressedFailureLogCount{0};

private:
    AUTIL_LOG_DECLARE();
};

}
