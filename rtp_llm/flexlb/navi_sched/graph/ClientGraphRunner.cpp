#include "aios/navi_sched/graph/ClientGraphRunner.h"
#include "aios/navi_sched/graph/GraphVisRecorder.h"
#include "aios/navi_sched/resource/ClientResponseR.h"
#include "autil/TimeUtility.h"
#include "autil/StringUtil.h"
#include "navi/builder/GraphBuilder.h"
#include <algorithm>

namespace navi_sched {
namespace {

const std::string SCHEDULE_NODE = "navi_sched_schedule";
const std::string CLIENT_GRAPH_VIS_FILE = "./client_graph.vis";
const std::string ADMISSION_TIMEOUT_ERROR = "NaviSched request admission timed out";
const std::string DECISION_WAIT_TIMEOUT_ERROR =
    "timed out waiting for NaviSched decision";

class ClientGraphClosure : public navi::NaviUserResultClosure {
public:
    ClientGraphClosure(std::shared_ptr<ClientResponseR> responseR, uint64_t requestId,
                       bool enableVis, int64_t beginTimeUs,
                       ClientGraphRunner::DecisionCallback completion = {})
        : _responseR(std::move(responseR))
        , _requestId(requestId)
        , _enableVis(enableVis)
        , _beginTimeUs(beginTimeUs)
        , _completion(std::move(completion))
    {
        waitDestruct = true;
    }

    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        bool eof = false;
        while (userResult && !eof) {
            navi::NaviUserData data;
            userResult->nextData(data, eof);
        }
        auto result = userResult ? userResult->getNaviResult() : nullptr;
        if (_enableVis) {
            GraphVisRecorder::save(result, CLIENT_GRAPH_VIS_FILE);
        }
        const auto errorCode =
            result && result->hasError() ? result->getErrorCode() : navi::EC_NONE;
        const auto errorMessage =
            result && result->hasError() ? result->getErrorMessage() : std::string();
        auto response = _responseR->finishGraph(_requestId, errorCode, errorMessage,
                                                static_cast<bool>(_completion));
        if (_completion) {
            ClientGraphRunResult result;
            result.errorCode = response.errorCode;
            result.errorMessage = std::move(response.errorMessage);
            result.decision = std::move(response.decision);
            result.replay = std::move(response.replay);
            const auto completedTimeUs = autil::TimeUtility::currentTime();
            const auto decisionTimeUs =
                response.decisionTimeUs > 0 ? response.decisionTimeUs : completedTimeUs;
            result.masterRequestLatencyUs =
                std::max(int64_t(0), decisionTimeUs - _beginTimeUs);
            if (result.decision) {
                if (!result.decision->requestTiming) {
                    result.decision->requestTiming =
                        std::make_shared<NaviSchedRequestTiming>();
                }
                result.decision->requestTiming->clientTotalLatencyMs =
                    result.masterRequestLatencyUs / 1000.0;
            }
            _completion(std::move(result));
            _responseR->finishClosureCompletion(_requestId);
        }
        delete this;
    }

private:
    std::shared_ptr<ClientResponseR> _responseR;
    uint64_t _requestId = 0;
    bool _enableVis = false;
    int64_t _beginTimeUs = 0;
    ClientGraphRunner::DecisionCallback _completion;
};

}

std::unique_ptr<navi::GraphDef> ClientGraphRunner::buildScheduleGraph(
    const std::string &clientBizName, const std::string &masterBizName,
    bool collectReplay, int64_t entryTimeoutMs, navi::GraphId &clientGraphId,
    navi::GraphId &masterGraphId, const std::string &masterAddress)
{
    const auto rpcTimeoutMs = getNaviSchedRpcTimeoutMs(entryTimeoutMs);
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    masterGraphId = builder.newSubGraph(masterBizName);
    builder.subGraph(masterGraphId).inlineMode(true).subGraphTimeoutMs(rpcTimeoutMs);
    if (masterAddress.empty()) {
        builder.subGraph(masterGraphId).location(masterBizName, 1, 0);
    } else {
        const auto separator = masterAddress.rfind(':');
        int32_t port = 0;
        if (separator == std::string::npos || separator == 0 ||
            masterAddress.find(':') != separator ||
            !autil::StringUtil::fromString(masterAddress.substr(separator + 1), port) ||
            port <= 0 || port > 65535) {
            return nullptr;
        }
        builder.subGraph(masterGraphId)
            .directLocation(masterBizName, masterAddress.substr(0, separator), port);
    }
    builder.subGraph(masterGraphId).gigProbe(false);
    auto scheduleNode =
        builder.node(SCHEDULE_NODE)
            .kernel("navi_sched.schedule_k")
            .integerAttr("dispatch_guard_enabled", clientBizName.empty() ? 0 : 1);
    if (clientBizName.empty()) {
        scheduleNode.out("decision").asGraphOutput("decision");
        if (collectReplay) {
            scheduleNode.out("replay").asGraphOutput("replay");
        }
        if (!builder.ok()) {
            return nullptr;
        }
        navi::GraphBuilder::fillHashKey(graphDef.get());
        return graphDef;
    }
    clientGraphId = builder.newSubGraph(clientBizName);
    builder.subGraph(clientGraphId).location(clientBizName, 1, 0).inlineMode(true);
    auto forwardNode = builder.node(clientGraphId, "navi_sched_forward")
                           .kernel("navi_sched.client_forward_k");
    auto responseNode = builder.node(clientGraphId, "navi_sched_response")
                            .kernel("navi_sched.client_response_k");
    scheduleNode.jsonAttrs("{\"fused_dispatch_guard\":true}")
        .integerAttr("admission_timeout_ms", rpcTimeoutMs);
    scheduleNode.out("decision").to(forwardNode.in("decision"));
    forwardNode.out("decision").to(responseNode.in("decision"));
    forwardNode.out("decision").to(scheduleNode.in("forward_decision")).require(false);
    forwardNode.out("response").to(responseNode.in("response"));
    if (collectReplay) {
        scheduleNode.out("replay").to(responseNode.in("replay"));
    }
    responseNode.out("finished").asGraphOutput("finished");
    scheduleNode.out("dispatch_finished").asGraphOutput("dispatch_finished");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

bool ClientGraphRunner::addScheduleRequest(
    navi::RunGraphParams &params, navi::GraphId masterGraphId,
    const ClientRequestToMasterDataConstPtr &request)
{
    if (!request) {
        return false;
    }
    navi::NamedData namedData;
    namedData.graphId = masterGraphId;
    namedData.name = CLIENT_REQUEST_TO_MASTER_DATA_NAME;
    namedData.data = std::const_pointer_cast<ClientRequestToMasterData>(request);
    return params.addNamedData(namedData);
}

namespace {

ClientRequestToMasterDataConstPtr withoutForwardPayload(
    const ClientRequestToMasterDataConstPtr &request)
{
    if (!request || request->forwardPayload.empty()) {
        return request;
    }
    auto copy = std::make_shared<ClientRequestToMasterData>();
    copy->version = request->version;
    copy->bizName = request->bizName;
    copy->timeoutMs = request->timeoutMs;
    copy->promptTokenCount = request->promptTokenCount;
    copy->explicitCacheCheckpointPosition = request->explicitCacheCheckpointPosition;
    copy->blockSize = request->blockSize;
    copy->requestId = request->requestId;
    copy->collectReplay = request->collectReplay;
    copy->routeOnly = request->routeOnly;
    copy->blockHashes = request->blockHashes;
    return copy;
}

}

ClientGraphRunner::ClientGraphRunner(const navi::NaviPtr &navi, bool enableVis,
                                     const std::shared_ptr<ClientResponseR> &responseR,
                                     bool expectAgentResponseStream)
    : _navi(navi)
    , _enableVis(enableVis)
    , _responseR(responseR ? responseR : std::make_shared<ClientResponseR>())
    , _expectAgentResponseStream(expectAgentResponseStream)
{
}

void ClientGraphRunner::setClientBizName(const std::string &clientBizName)
{
    _clientBizName = clientBizName;
}

ClientGraphRunResult ClientGraphRunner::schedule(
    const ClientRequestToMasterDataConstPtr &request, const std::string &masterBizName,
    int64_t timeoutMs, bool expectResponseStream, const std::string &masterAddress,
    bool waitForDecision)
{
    ClientGraphRunResult result;
    if (!_navi || !request || 0 == request->requestId || request->bizName.empty() ||
        masterBizName.empty() || timeoutMs <= 0) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "invalid navi sched client runner state";
        return result;
    }
    return runRequest(request, masterBizName, timeoutMs, expectResponseStream, {},
                      masterAddress, waitForDecision);
}

ClientGraphRunResult ClientGraphRunner::scheduleDetached(
    const ClientRequestToMasterDataConstPtr &request, const std::string &masterBizName,
    int64_t timeoutMs, DecisionCallback decisionCallback, bool expectResponseStream)
{
    ClientGraphRunResult result;
    if (!_navi || !request || 0 == request->requestId || request->bizName.empty() ||
        masterBizName.empty() || timeoutMs <= 0 || _clientBizName.empty() ||
        !decisionCallback) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "invalid detached navi sched client runner state";
        return result;
    }
    return runRequest(request, masterBizName, timeoutMs, expectResponseStream,
                      std::move(decisionCallback));
}

ClientGraphRunResult ClientGraphRunner::runRequest(
    const ClientRequestToMasterDataConstPtr &request, const std::string &masterBizName,
    int64_t timeoutMs, bool expectResponseStream, DecisionCallback decisionCallback,
    const std::string &masterAddress, bool waitForDecision) const
{
    ClientGraphRunResult result;
    auto beginTimeUs = autil::TimeUtility::currentTime();
    const bool detached = static_cast<bool>(decisionCallback);
    navi::GraphId masterGraphId = navi::INVALID_GRAPH_ID;
    navi::GraphId clientGraphId = navi::INVALID_GRAPH_ID;
    auto graphDef =
        buildScheduleGraph(_clientBizName, masterBizName, request->collectReplay,
                           timeoutMs, clientGraphId, masterGraphId, masterAddress);
    navi::RunGraphParams params;
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    params.setMaxInline(20);
    if (_clientBizName.empty()) {
        params.setTimeoutMs(timeoutMs);
    }
    params.setTraceLevel("INFO");
    if (_enableVis && !detached) {
        params.setCollectMetric(true);
        params.addParamTopoTreeStr("edge_key=*;edge_query=2");
    }
    auto masterRequest = withoutForwardPayload(request);
    if (!graphDef || !addScheduleRequest(params, masterGraphId, masterRequest) ||
        (navi::INVALID_GRAPH_ID != clientGraphId &&
         !addScheduleRequest(params, clientGraphId, request))) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "build navi sched request graph failed";
        return result;
    }
    if (_clientBizName.empty()) {
        result = collectResult(_navi->runGraph(graphDef.release(), params),
                               request->requestId);
    } else {
        if (!_responseR->begin(request->requestId, request->collectReplay,
                               _expectAgentResponseStream && expectResponseStream,
                               detached)) {
            result.errorCode = navi::EC_ABORT;
            result.errorMessage = "duplicate active NaviSched client request";
            return result;
        }
        navi::ResourceMap resources;
        if (!resources.addResource(_responseR)) {
            _responseR->finishGraph(request->requestId, navi::EC_ABORT,
                                    "failed to attach NaviSched client response resource",
                                    detached);
            if (detached) {
                _responseR->finishClosureCompletion(request->requestId);
            }
            result.errorCode = navi::EC_ABORT;
            result.errorMessage = "failed to attach NaviSched client response resource";
            return result;
        } else {
            auto userResult = _navi->runLocalGraphAsync(
                graphDef.release(), params, resources,
                new ClientGraphClosure(_responseR, request->requestId,
                                       _enableVis && !detached, beginTimeUs,
                                       std::move(decisionCallback)));
            if (userResult) {
                _responseR->attach(request->requestId, userResult);
            } else {
                _responseR->finishGraph(request->requestId, navi::EC_ABORT,
                                        "Navi.runLocalGraphAsync returned null",
                                        detached);
                if (detached) {
                    _responseR->finishClosureCompletion(request->requestId);
                }
                result.errorCode = navi::EC_ABORT;
                result.errorMessage = "Navi.runLocalGraphAsync returned null";
                return result;
            }
        }
        if (detached || !waitForDecision) {
            result.errorCode = navi::EC_NONE;
        } else {
            auto decision = _responseR->waitDecision(request->requestId,
                                                     getNaviSchedRpcTimeoutMs(timeoutMs));
            if (navi::EC_TIMEOUT == decision.errorCode &&
                DECISION_WAIT_TIMEOUT_ERROR == decision.errorMessage) {
                decision.errorMessage = ADMISSION_TIMEOUT_ERROR;
            }
            result.errorCode = decision.errorCode;
            result.errorMessage = std::move(decision.errorMessage);
            result.decision = std::move(decision.decision);
            result.replay = std::move(decision.replay);
            if (navi::EC_TIMEOUT == result.errorCode) {
                _responseR->abort(request->requestId);
            }
        }
    }
    if (result.decision) {
        if (!result.decision->requestTiming) {
            result.decision->requestTiming = std::make_shared<NaviSchedRequestTiming>();
        }
        result.decision->requestTiming->clientTotalLatencyMs =
            std::max(int64_t(0), autil::TimeUtility::currentTime() - beginTimeUs) /
            1000.0;
    }
    return result;
}

ClientGraphRunResult ClientGraphRunner::collectResult(
    const navi::NaviUserResultPtr &userResult, uint64_t requestId) const
{
    ClientGraphRunResult result;
    if (!userResult) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "Navi.runGraph returned null";
        return result;
    }
    while (true) {
        navi::NaviUserData userData;
        bool eof = false;
        if (userResult->nextData(userData, eof) && userData.data) {
            auto decision =
                std::dynamic_pointer_cast<MasterResponseToClientData>(userData.data);
            if (decision && decision->version == NAVI_SCHED_DATA_VERSION) {
                if (decision->replay) {
                    if (decision->replay->version == NAVI_SCHED_DATA_VERSION) {
                        result.replay = decision->replay;
                    }
                } else {
                    result.decision = decision->makeCopy();
                    result.decision->requestId = requestId;
                }
                result.errorCode = navi::EC_NONE;
            }
        }
        if (eof) {
            break;
        }
    }
    auto naviResult = userResult->getNaviResult();
    if (_enableVis) {
        GraphVisRecorder::save(naviResult, CLIENT_GRAPH_VIS_FILE);
    }
    if (naviResult && naviResult->hasError()) {
        result.errorCode = naviResult->getErrorCode();
        result.errorMessage = naviResult->getErrorMessage();
    } else if (navi::EC_NONE != result.errorCode) {
        result.errorCode = navi::EC_NO_DATA;
        result.errorMessage = "schedule graph returned no decision";
    }
    return result;
}

}
