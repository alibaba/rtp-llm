#include "aios/navi_sched/graph/ReplayControlGraph.h"

#include "aios/navi_sched/kernel/ReplayClientControlKernel.h"
#include "aios/navi_sched/kernel/ReplayMasterControlKernel.h"
#include "aios/navi_sched/kernel/ReplayMasterHistoryKernel.h"
#include "autil/Log.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviUserResult.h"

#include <chrono>
#include <exception>

namespace navi_sched {

AUTIL_LOG_SETUP(navi_sched, ReplayControlGraph);

std::unique_ptr<navi::GraphDef> ReplayControlGraph::buildGraph(
    const std::string &clientBizName, const std::string &masterBizName)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    const auto masterGraphId = builder.newSubGraph(masterBizName);
    const auto clientGraphId = builder.newSubGraph(clientBizName);
    builder.subGraph(masterGraphId)
        .location(masterBizName, 1, 0)
        .subGraphAttr("navi.gig.enable_biz_stat", "false");
    builder.subGraph(clientGraphId).location(clientBizName, 1, 0);
    auto master = builder.node(masterGraphId, "replay_master_control")
                      .kernel(ReplayMasterControlKernel::KERNEL_ID);
    auto masterHistory = builder.node(masterGraphId, "replay_master_history")
                             .kernel(ReplayMasterHistoryKernel::KERNEL_ID);
    auto client = builder.node(clientGraphId, "replay_client_control")
                      .kernel(ReplayClientControlKernel::KERNEL_ID);
    client.out("capability").to(master.in("capability"));
    client.out("history_response").to(masterHistory.in("history_response"));
    master.out("command").asGraphOutput("command");
    master.out("history_query").asGraphOutput("history_query");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

ReplayControlGraph::ReplayControlGraph(const navi::NaviPtr &navi,
                                       const ReplayClientRPtr &clientResource,
                                       const std::string &clientBizName,
                                       const std::string &masterBizName)
    : _navi(navi)
    , _clientResource(clientResource)
    , _clientBizName(clientBizName)
    , _masterBizName(masterBizName)
    , _state(std::make_shared<State>())
{
}

ReplayControlGraph::~ReplayControlGraph() { stop(); }

bool ReplayControlGraph::start()
{
    if (!_navi || !_clientResource || _clientBizName.empty() ||
        _masterBizName.empty() || !_state ||
        _state->running.exchange(true)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->stopping.store(false, std::memory_order_release);
    }
    try {
        _supervisorThread = std::thread(&ReplayControlGraph::supervise, this);
    } catch (const std::exception &error) {
        AUTIL_LOG(ERROR, "start replay control supervisor failed: %s", error.what());
        _state->running.store(false);
        return false;
    }
    return true;
}

void ReplayControlGraph::supervise()
{
    constexpr auto RETRY_DELAY = std::chrono::seconds(1);
    while (!_state->stopping.load(std::memory_order_acquire)) {
        auto graph = buildGraph(_clientBizName, _masterBizName);
        if (!graph) {
            AUTIL_LOG(ERROR, "build replay control graph failed, retrying");
        } else {
            navi::ResourceMap resources;
            if (!resources.addResource(_clientResource)) {
                AUTIL_LOG(ERROR, "attach replay client resource failed, retrying");
            } else {
                navi::RunGraphParams params;
                params.setResourceBuildMode(navi::RBM_PARALLEL);
                params.setTimeoutMs(int64_t(10) * 365 * 24 * 60 * 60 * 1000);
                params.setTraceLevel("INFO");
                auto userResult = _navi->runLocalGraphAsync(
                    graph.release(), params, resources, nullptr);
                {
                    std::lock_guard<std::mutex> lock(_state->mutex);
                    _userResult = userResult;
                }
                if (!userResult) {
                    AUTIL_LOG(ERROR, "start replay control graph failed, retrying");
                } else {
                    bool eof = false;
                    while (!eof) {
                        if (_state->stopping.load(std::memory_order_acquire)) {
                            userResult->abort(navi::EC_ABORT);
                        }
                        navi::NaviUserData data;
                        if (userResult->nextData(data, eof) && data.data) {
                            auto command = std::dynamic_pointer_cast<
                                ReplayControlCommandData>(data.data);
                            auto historyQuery = std::dynamic_pointer_cast<
                                ReplayHistoryQueryData>(data.data);
                            if (command && !_clientResource->applyCommand(*command)) {
                                AUTIL_LOG(ERROR,
                                          "apply replay control graph output failed");
                                userResult->abort(navi::EC_ABORT);
                            } else if (historyQuery &&
                                       !_clientResource->answerHistoryQuery(
                                           historyQuery->queryId)) {
                                AUTIL_LOG(ERROR,
                                          "answer replay history query failed");
                                userResult->abort(navi::EC_ABORT);
                            } else if (historyQuery) {
                                AUTIL_LOG(INFO,
                                          "answered replay history query [%lu]",
                                          historyQuery->queryId);
                            } else if (!command && !historyQuery) {
                                AUTIL_LOG(ERROR,
                                          "invalid replay control graph output");
                                userResult->abort(navi::EC_ABORT);
                            }
                        }
                    }
                    const bool stopping = _state->stopping.load(
                        std::memory_order_acquire);
                    if (!stopping) {
                        auto result = userResult->getNaviResult();
                        AUTIL_LOG(WARN,
                                  "replay control graph exited; restarting%s%s",
                                  result && result->hasError() ? ": " : "",
                                  result && result->hasError()
                                      ? result->getErrorMessage().c_str()
                                      : "");
                    } else {
                        userResult->abort(navi::EC_ABORT);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(_state->mutex);
                    _userResult.reset();
                }
            }
        }
        if (_state->stopping.load(std::memory_order_acquire)) {
            break;
        }
        std::unique_lock<std::mutex> lock(_state->mutex);
        _state->condition.wait_for(lock, RETRY_DELAY, [this] {
            return _state->stopping.load(std::memory_order_acquire);
        });
    }
    _state->running.store(false, std::memory_order_release);
    _state->condition.notify_all();
}

void ReplayControlGraph::stop()
{
    if (!_state) {
        return;
    }
    _state->stopping.store(true, std::memory_order_release);
    navi::NaviUserResultPtr userResult;
    {
        std::lock_guard<std::mutex> lock(_state->mutex);
        userResult = _userResult;
    }
    if (userResult) {
        userResult->abort(navi::EC_ABORT);
    }
    _state->condition.notify_all();
    if (_supervisorThread.joinable()) {
        _supervisorThread.join();
    }
    {
        std::lock_guard<std::mutex> lock(_state->mutex);
        _userResult.reset();
    }
    _state->running.store(false, std::memory_order_release);
}

bool ReplayControlGraph::running() const
{
    return _state && _state->running.load(std::memory_order_acquire);
}

}
