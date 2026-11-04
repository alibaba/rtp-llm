#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayGraph.h"

#include "rtp_llm/flexlb/navi_sched/graph/GraphVisRecorder.h"
#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayR.h"
#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayStartData.h"
#include "autil/Lock.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviUserResult.h"
#include "navi/log/NaviLogger.h"
#include <cerrno>

namespace navi_sched::test {
namespace {

const int64_t TRAFFIC_REPLAY_GRAPH_TIMEOUT_MS = int64_t(10) * 365 * 24 * 60 * 60 * 1000;
const std::string TRAFFIC_REPLAY_GRAPH_VIS_FILE = "./traffic_replay_graph.vis";
const int64_t TRAFFIC_REPLAY_STOP_TIMEOUT_US = 6 * 1000 * 1000;

class TrafficReplayClosure : public navi::NaviUserResultClosure {
public:
    explicit TrafficReplayClosure(bool enableVis) : _enableVis(enableVis) {}

public:
    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        consumeToEof(userResult);
        autil::ScopedLock lock(_condition);
        _userResult = std::move(userResult);
        _finished = true;
        _condition.broadcast();
    }
    bool waitAndSave(int64_t timeoutUs)
    {
        navi::NaviUserResultPtr userResult;
        {
            autil::ScopedLock lock(_condition);
            while (!_finished) {
                if (ETIMEDOUT == _condition.wait(timeoutUs)) {
                    return false;
                }
            }
            userResult = _userResult;
        }
        if (_enableVis) {
            GraphVisRecorder::save(userResult ? userResult->getNaviResult() : nullptr,
                                   TRAFFIC_REPLAY_GRAPH_VIS_FILE);
        }
        return true;
    }

private:
    static void consumeToEof(const navi::NaviUserResultPtr &userResult)
    {
        if (!userResult) {
            return;
        }
        bool eof = false;
        while (!eof) {
            navi::NaviUserData data;
            userResult->nextData(data, eof);
        }
    }

private:
    bool _enableVis;
    autil::ThreadCond _condition;
    navi::NaviUserResultPtr _userResult;
    bool _finished = false;
};

class TrafficReplayCompletionClosure : public navi::NaviUserResultClosure {
public:
    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        bool eof = false;
        while (userResult && !eof) {
            navi::NaviUserData data;
            userResult->nextData(data, eof);
        }
        autil::ScopedLock lock(_condition);
        _completed = true;
        _condition.broadcast();
    }

    void wait()
    {
        autil::ScopedLock lock(_condition);
        while (!_completed) {
            _condition.wait();
        }
    }

private:
    autil::ThreadCond _condition;
    bool _completed = false;
};

std::unique_ptr<navi::GraphDef> buildGraph(const std::string &clientBizName,
                                           const std::string &resourceConfig)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    auto graphId = builder.newSubGraph(clientBizName);
    builder.subGraph(graphId).location(clientBizName, 1, 0);
    auto replay = builder.node("navi_sched_traffic_replay")
                      .kernel("navi_sched.client_simulate_traffic_replay_k")
                      .jsonAttrs(resourceConfig);
    replay.out("finished").asGraphOutput("finished");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

std::unique_ptr<navi::GraphDef> buildStartGraph(const TrafficReplayR &trafficReplayR,
                                                const std::string &clientBizName,
                                                navi::GraphId &masterGraphId)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    masterGraphId = builder.newSubGraph(trafficReplayR.getMasterBizName());
    builder.subGraph(masterGraphId).location(trafficReplayR.getMasterBizName(), 1, 0);
    auto clientGraphId = builder.newSubGraph(clientBizName);
    builder.subGraph(clientGraphId).location(clientBizName, 1, 0);
    auto start = builder.node(masterGraphId, "navi_sched_traffic_replay_start")
                     .kernel("navi_sched.master_traffic_replay_start_k");
    auto result = builder.node(clientGraphId, "navi_sched_traffic_replay_start_result")
                      .kernel("navi_sched.client_simulate_traffic_replay_start_result_k");
    start.out("start").to(result.in("start"));
    result.out("finished").asGraphOutput("finished");
    if (!builder.ok()) return nullptr;
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

} // namespace

class TrafficReplayGraph::Impl {
public:
    navi::NaviPtr navi;
    std::shared_ptr<TrafficReplayBridgeR> trafficReplayBridgeR;
    std::shared_ptr<TrafficReplayR> trafficReplayR;
    std::string clientBizName;
    std::string resourceConfig;
    bool enableVis = false;
    navi::NaviUserResultPtr userResult;
    navi::NaviUserResultPtr startUserResult;
    std::shared_ptr<TrafficReplayClosure> closure;
    std::shared_ptr<TrafficReplayCompletionClosure> startClosure;
};

TrafficReplayGraph::TrafficReplayGraph(
    const navi::NaviPtr &navi,
    const std::shared_ptr<TrafficReplayBridgeR> &trafficReplayBridgeR,
    const std::string &clientBizName, const std::string &resourceConfig, bool enableVis)
    : _impl(std::make_unique<Impl>())
{
    _impl->navi = navi;
    _impl->trafficReplayBridgeR = trafficReplayBridgeR;
    _impl->clientBizName = clientBizName;
    _impl->resourceConfig = resourceConfig;
    _impl->enableVis = enableVis;
}

TrafficReplayGraph::~TrafficReplayGraph() { stop(); }

bool TrafficReplayGraph::start()
{
    if (_impl->userResult) {
        return true;
    }
    if (!_impl->navi || !_impl->trafficReplayBridgeR || _impl->clientBizName.empty() ||
        _impl->resourceConfig.empty()) {
        return false;
    }
    auto graphDef = buildGraph(_impl->clientBizName, _impl->resourceConfig);
    if (!graphDef) {
        return false;
    }
    navi::RunGraphParams params;
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    params.setTimeoutMs(TRAFFIC_REPLAY_GRAPH_TIMEOUT_MS);
    params.setTraceLevel("ERROR");
    if (_impl->enableVis) {
        params.setCollectMetric(true);
        params.addParamTopoTreeStr("edge_key=*;edge_query=2");
    }
    _impl->closure = std::make_shared<TrafficReplayClosure>(_impl->enableVis);
    navi::ResourceMap resourceMap;
    if (!resourceMap.addResource(_impl->trafficReplayBridgeR)) {
        return false;
    }
    _impl->userResult = _impl->navi->runLocalGraphAsync(
        graphDef.release(), params, resourceMap, _impl->closure.get());
    if (!_impl->userResult) {
        _impl->closure.reset();
        return false;
    }
    _impl->trafficReplayR =
        _impl->trafficReplayBridgeR->waitTrafficReplayR(30 * 1000 * 1000);
    if (!_impl->trafficReplayR) {
        _impl->userResult->abort(navi::EC_ABORT);
        _impl->closure->waitAndSave(0);
        _impl->userResult.reset();
        _impl->closure.reset();
        return false;
    }
    _impl->trafficReplayR->setNavi(_impl->navi.get());
    navi::GraphId masterGraphId = navi::INVALID_GRAPH_ID;
    auto startGraphDef =
        buildStartGraph(*_impl->trafficReplayR, _impl->clientBizName, masterGraphId);
    navi::RunGraphParams startParams;
    startParams.setResourceBuildMode(navi::RBM_PARALLEL);
    startParams.setTimeoutMs(TRAFFIC_REPLAY_GRAPH_TIMEOUT_MS);
    if (!startGraphDef) {
        _impl->trafficReplayR->finish();
        return false;
    }
    auto request = std::make_shared<TrafficReplayStartRequestData>();
    request->replayGroup = _impl->trafficReplayR->getTargetBizName();
    request->clientIndex = _impl->trafficReplayR->getClientIndex();
    request->clientCount = _impl->trafficReplayR->getClientCount();
    request->firstRequestTimeMs = _impl->trafficReplayR->getFirstRequestTimeMs();
    request->startDelayMs = _impl->trafficReplayR->getStartDelayMs();
    navi::NamedData namedData;
    namedData.graphId = masterGraphId;
    namedData.name = TRAFFIC_REPLAY_START_REQUEST_DATA_NAME;
    namedData.data = request;
    if (!startParams.addNamedData(namedData)) {
        _impl->trafficReplayR->finish();
        return false;
    }
    _impl->startClosure = std::make_shared<TrafficReplayCompletionClosure>();
    _impl->startUserResult = _impl->navi->runLocalGraphAsync(
        startGraphDef.release(), startParams, resourceMap, _impl->startClosure.get());
    if (_impl->startUserResult) return true;
    _impl->trafficReplayR->finish();
    _impl->trafficReplayR->clearNavi(_impl->navi.get());
    _impl->startClosure.reset();
    _impl->closure.reset();
    _impl->userResult.reset();
    return false;
}

void TrafficReplayGraph::stop()
{
    auto userResult = std::move(_impl->userResult);
    auto startUserResult = std::move(_impl->startUserResult);
    auto closure = std::move(_impl->closure);
    auto startClosure = std::move(_impl->startClosure);
    if (startUserResult) {
        startUserResult->abort(navi::EC_ABORT);
    }
    if (startClosure) {
        startClosure->wait();
    }
    if (!closure) {
        return;
    }
    _impl->trafficReplayR->finish();
    if (!closure->waitAndSave(TRAFFIC_REPLAY_STOP_TIMEOUT_US) && userResult) {
        userResult->abort(navi::EC_ABORT);
        closure->waitAndSave(0);
    }
    _impl->trafficReplayR->clearNavi(_impl->navi.get());
    _impl->trafficReplayR.reset();
}

} // namespace navi_sched::test
