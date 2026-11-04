#include "rtp_llm/flexlb/navi_sched/graph/AgentGraph.h"
#include "rtp_llm/flexlb/navi_sched/graph/GraphVisRecorder.h"
#include "rtp_llm/flexlb/navi_sched/kernel/AgentBatchKernel.h"
#include "rtp_llm/flexlb/navi_sched/kernel/AgentLatencyKernel.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentBatchR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentLatencyInputR.h"
#include "autil/Lock.h"
#include "autil/TimeUtility.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviUserResult.h"
#include <atomic>
#include <cerrno>
#include <thread>

namespace navi_sched {
namespace {

const int64_t AGENT_GRAPH_TIMEOUT_MS = int64_t(10) * 365 * 24 * 60 * 60 * 1000;
const int64_t AGENT_GRAPH_READY_TIMEOUT_MS = 30 * 1000;
const int64_t AGENT_GRAPH_RESTART_DELAY_US = 100 * 1000;
const int64_t AGENT_GRAPH_STOP_TIMEOUT_US = 5 * 1000 * 1000;
const std::string AGENT_GRAPH_VIS_FILE = "./agent_graph.vis";

class AgentGraphClosure : public navi::NaviUserResultClosure {
public:
    AgentGraphClosure(bool enableVis, const AgentBatchRPtr &agentBatchR,
                      uint64_t batchPipeGeneration,
                      const AgentLatencyInputRPtr &latencyInputR,
                      uint64_t latencyPipeGeneration)
        : _enableVis(enableVis)
        , _agentBatchR(agentBatchR)
        , _batchPipeGeneration(batchPipeGeneration)
        , _latencyInputR(latencyInputR)
        , _latencyPipeGeneration(latencyPipeGeneration)
    {
        waitDestruct = true;
    }

    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        consumeToEof(userResult);
        if (_agentBatchR) {
            _agentBatchR->clearGraphPipe(_batchPipeGeneration);
        }
        if (_latencyInputR) {
            _latencyInputR->clearInputPipe(_latencyPipeGeneration);
        }
        autil::ScopedLock lock(_condition);
        _userResult = std::move(userResult);
        _completed = true;
        _condition.broadcast();
    }

    navi::NaviUserResultPtr waitAndTake()
    {
        navi::NaviUserResultPtr userResult;
        {
            autil::ScopedLock lock(_condition);
            while (!_completed) {
                _condition.wait();
            }
            userResult = std::move(_userResult);
        }
        if (_enableVis) {
            GraphVisRecorder::save(userResult ? userResult->getNaviResult() : nullptr,
                                   AGENT_GRAPH_VIS_FILE);
        }
        return userResult;
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
    AgentBatchRPtr _agentBatchR;
    uint64_t _batchPipeGeneration;
    AgentLatencyInputRPtr _latencyInputR;
    uint64_t _latencyPipeGeneration;
    autil::ThreadCond _condition;
    navi::NaviUserResultPtr _userResult;
    bool _completed = false;
};

NAVI_TYPEDEF_PTR(AgentGraphClosure);

} // namespace

std::unique_ptr<navi::GraphDef> AgentGraph::buildGraph(
    const std::string &agentBizName, uint64_t batchPipeGeneration,
    uint64_t latencyPipeGeneration)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    auto graphId = builder.newSubGraph(agentBizName);
    builder.subGraph(graphId).location(agentBizName, 1, 0);
    AgentBatchKernel::buildGraph(builder, batchPipeGeneration);
    AgentLatencyKernel::buildGraph(builder, latencyPipeGeneration);
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

class AgentGraph::Impl {
public:
    bool valid() const
    {
        return navi && agentBatchR && latencyInputR && !agentBizName.empty();
    }

    size_t prepareRecoveryInputs()
    {
        const auto recovery = agentBatchR->getGraphRecoveryState();
        std::vector<AgentLatencyInput> inputs;
        inputs.reserve(recovery.batches.size() * 2);
        for (const auto &recoveryBatch : recovery.batches) {
            if (!recoveryBatch.batch) {
                continue;
            }
            AgentLatencyInput ready;
            ready.type = AgentLatencyInputType::BATCH_READY;
            ready.batch = recoveryBatch.batch;
            ready.batchId = recoveryBatch.batch->batchId;
            ready.batchWaitingRequests = recovery.batchWaitingRequests;
            ready.queuedBatchRequests = recovery.queuedBatchRequests;
            ready.runningRequests = recovery.runningRequests;
            ready.eventTimeUs = recoveryBatch.batch->readyTimeUs;
            inputs.emplace_back(std::move(ready));
            if (!recoveryBatch.running) {
                continue;
            }
            AgentLatencyInput started;
            started.type = AgentLatencyInputType::BATCH_STARTED;
            started.batchId = recoveryBatch.batch->batchId;
            started.batchWaitingRequests = recovery.batchWaitingRequests;
            started.queuedBatchRequests = recovery.queuedBatchRequests;
            started.runningRequests = recovery.runningRequests;
            started.eventTimeUs = recoveryBatch.startTimeUs;
            inputs.emplace_back(std::move(started));
        }
        const auto inputCount = inputs.size();
        latencyInputR->replaceGraphRecoveryInputs(std::move(inputs));
        return inputCount;
    }

    bool launchGraph()
    {
        const auto batchPipeGeneration =
            agentBatchR->beginGraphPipeGeneration();
        if (0 == batchPipeGeneration) {
            AUTIL_LOG(ERROR, "begin Agent batch pipe generation failed");
            return false;
        }
        const auto latencyPipeGeneration =
            latencyInputR->beginInputPipeGeneration();
        if (0 == latencyPipeGeneration) {
            AUTIL_LOG(ERROR,
                      "begin Agent latency pipe generation failed, "
                      "batchPipeGeneration [%lu]",
                      batchPipeGeneration);
            agentBatchR->clearGraphPipe(batchPipeGeneration);
            return false;
        }
        const auto recoveryInputCount = prepareRecoveryInputs();
        if (recoveryInputCount > 0) {
            AUTIL_LOG(INFO,
                      "prepared Agent graph recovery inputs, count [%lu], "
                      "batchPipeGeneration [%lu], latencyPipeGeneration [%lu]",
                      recoveryInputCount, batchPipeGeneration,
                      latencyPipeGeneration);
        }
        auto graphDef = AgentGraph::buildGraph(
            agentBizName, batchPipeGeneration, latencyPipeGeneration);
        if (!graphDef) {
            AUTIL_LOG(ERROR,
                      "build Agent graph failed, batchPipeGeneration [%lu], "
                      "latencyPipeGeneration [%lu]",
                      batchPipeGeneration, latencyPipeGeneration);
            agentBatchR->clearGraphPipe(batchPipeGeneration);
            latencyInputR->clearInputPipe(latencyPipeGeneration);
            return false;
        }
        navi::ResourceMap resourceMap;
        if (!resourceMap.addResource(agentBatchR) ||
            !resourceMap.addResource(latencyInputR)) {
            AUTIL_LOG(ERROR,
                      "add Agent graph external resources failed, "
                      "batchPipeGeneration [%lu], latencyPipeGeneration [%lu]",
                      batchPipeGeneration, latencyPipeGeneration);
            agentBatchR->clearGraphPipe(batchPipeGeneration);
            latencyInputR->clearInputPipe(latencyPipeGeneration);
            return false;
        }
        navi::RunGraphParams params;
        params.setResourceBuildMode(navi::RBM_PARALLEL);
        params.setTimeoutMs(AGENT_GRAPH_TIMEOUT_MS);
        if (enableVis) {
            params.setTraceLevel("INFO");
            params.setCollectMetric(true);
            params.addParamTopoTreeStr("edge_key=*;edge_query=2");
        }
        auto nextClosure = std::make_shared<AgentGraphClosure>(
            enableVis, agentBatchR, batchPipeGeneration, latencyInputR,
            latencyPipeGeneration);
        auto nextUserResult = navi->runLocalGraphAsync(
            graphDef.release(), params, resourceMap, nextClosure.get());
        if (!nextUserResult) {
            AUTIL_LOG(ERROR,
                      "run Agent graph failed, batchPipeGeneration [%lu], "
                      "latencyPipeGeneration [%lu]",
                      batchPipeGeneration, latencyPipeGeneration);
            agentBatchR->clearGraphPipe(batchPipeGeneration);
            latencyInputR->clearInputPipe(latencyPipeGeneration);
            return false;
        }
        const auto readyDeadlineUs =
            autil::TimeUtility::monotonicTimeUs() +
            AGENT_GRAPH_READY_TIMEOUT_MS * 1000;
        const auto waitReady = [&](const auto &wait) {
            const auto remainingUs =
                readyDeadlineUs - autil::TimeUtility::monotonicTimeUs();
            return remainingUs > 0 && wait((remainingUs + 999) / 1000);
        };
        const bool batchReady = waitReady([&](int64_t timeoutMs) {
            return agentBatchR->waitGraphPipe(batchPipeGeneration, timeoutMs);
        });
        const bool latencyReady = batchReady && waitReady([&](int64_t timeoutMs) {
            return latencyInputR->waitInputPipe(latencyPipeGeneration, timeoutMs);
        });
        if (!batchReady || !latencyReady) {
            AUTIL_LOG(ERROR,
                      "Agent graph did not install pipes within [%ld]ms, "
                      "batchReady [%d], latencyReady [%d], "
                      "batchPipeGeneration [%lu], latencyPipeGeneration [%lu]",
                      AGENT_GRAPH_READY_TIMEOUT_MS, static_cast<int>(batchReady),
                      static_cast<int>(latencyReady), batchPipeGeneration,
                      latencyPipeGeneration);
            nextUserResult->abort(navi::EC_ABORT);
            auto failedResult = nextClosure->waitAndTake();
            failedResult.reset();
            nextUserResult.reset();
            nextClosure.reset();
            agentBatchR->clearGraphPipe(batchPipeGeneration);
            latencyInputR->clearInputPipe(latencyPipeGeneration);
            return false;
        }
        bool abortForStop = false;
        {
            autil::ScopedLock lock(condition);
            if (stopping) {
                abortForStop = true;
            } else {
                closure = std::move(nextClosure);
                userResult = std::move(nextUserResult);
                state.store(AgentGraph::State::RUNNING,
                            std::memory_order_release);
                condition.broadcast();
                return true;
            }
        }
        if (abortForStop) {
            nextUserResult->abort(navi::EC_ABORT);
        }
        auto stoppedResult = nextClosure->waitAndTake();
        stoppedResult.reset();
        nextUserResult.reset();
        nextClosure.reset();
        agentBatchR->clearGraphPipe(batchPipeGeneration);
        latencyInputR->clearInputPipe(latencyPipeGeneration);
        return false;
    }

    bool waitRestartDelay()
    {
        autil::ScopedLock lock(condition);
        if (stopping) {
            return false;
        }
        condition.wait(AGENT_GRAPH_RESTART_DELAY_US);
        return !stopping;
    }

    void finishStopped()
    {
        autil::ScopedLock lock(condition);
        state.store(AgentGraph::State::STOPPED, std::memory_order_release);
        condition.broadcast();
    }

    void supervise()
    {
        while (true) {
            {
                autil::ScopedLock lock(condition);
                if (stopping) {
                    state.store(AgentGraph::State::STOPPED,
                                std::memory_order_release);
                    condition.broadcast();
                    return;
                }
            }
            if (!launchGraph()) {
                {
                    autil::ScopedLock lock(condition);
                    if (stopping) {
                        state.store(AgentGraph::State::STOPPED,
                                    std::memory_order_release);
                        condition.broadcast();
                        return;
                    }
                }
                const auto currentRestartCount =
                    restartCount.fetch_add(1, std::memory_order_relaxed) + 1;
                AUTIL_LOG(ERROR,
                          "launch Agent graph failed, retry after [%ld]us, "
                          "restartCount [%lu]",
                          AGENT_GRAPH_RESTART_DELAY_US, currentRestartCount);
                if (!waitRestartDelay()) {
                    finishStopped();
                    return;
                }
                continue;
            }

            AgentGraphClosurePtr completedClosure;
            {
                autil::ScopedLock lock(condition);
                completedClosure = closure;
            }
            auto completedResult = completedClosure->waitAndTake();
            navi::ErrorCode errorCode = navi::EC_ABORT;
            std::string errorMessage = "missing Navi result";
            if (completedResult) {
                auto naviResult = completedResult->getNaviResult();
                if (naviResult) {
                    errorCode = naviResult->getErrorCode();
                    errorMessage = naviResult->getErrorMessage();
                }
            }

            navi::NaviUserResultPtr retiredUserResult;
            bool shouldStop = false;
            {
                autil::ScopedLock lock(condition);
                shouldStop = stopping;
                state.store(shouldStop ? AgentGraph::State::STOPPING
                                       : AgentGraph::State::RESTARTING,
                            std::memory_order_release);
                retiredUserResult = std::move(userResult);
                closure.reset();
                condition.broadcast();
            }
            completedResult.reset();
            retiredUserResult.reset();
            completedClosure.reset();
            if (shouldStop) {
                finishStopped();
                return;
            }

            const auto currentRestartCount =
                restartCount.fetch_add(1, std::memory_order_relaxed) + 1;
            AUTIL_LOG(ERROR,
                      "Agent graph destructed unexpectedly, errorCode [%d], "
                      "errorMessage [%s], restartCount [%lu]",
                      static_cast<int>(errorCode), errorMessage.c_str(),
                      currentRestartCount);
            if (!waitRestartDelay()) {
                finishStopped();
                return;
            }
        }
    }

public:
    navi::NaviPtr navi;
    AgentBatchRPtr agentBatchR;
    AgentLatencyInputRPtr latencyInputR;
    std::string agentBizName;
    bool enableVis = false;
    navi::NaviUserResultPtr userResult;
    AgentGraphClosurePtr closure;
    mutable autil::ThreadCond condition;
    std::thread supervisor;
    std::atomic<AgentGraph::State> state{AgentGraph::State::STOPPED};
    std::atomic<uint64_t> restartCount{0};
    bool stopping = true;
};

AUTIL_LOG_SETUP(navi_sched, AgentGraph);

AgentGraph::AgentGraph(const navi::NaviPtr &navi, const AgentBatchRPtr &agentBatchR,
                       const AgentLatencyInputRPtr &latencyInputR,
                       const std::string &agentBizName, bool enableVis)
    : _impl(std::make_unique<Impl>())
{
    _impl->navi = navi;
    _impl->agentBatchR = agentBatchR;
    _impl->latencyInputR = latencyInputR;
    _impl->agentBizName = agentBizName;
    _impl->enableVis = enableVis;
}

AgentGraph::~AgentGraph() { stop(); }

bool AgentGraph::start()
{
    {
        autil::ScopedLock lock(_impl->condition);
        if (_impl->supervisor.joinable()) {
            return !_impl->stopping && State::STOPPED != getState();
        }
        if (!_impl->valid()) {
            return false;
        }
    }
    if (!_impl->agentBatchR->prepareGraphStart()) {
        AUTIL_LOG(ERROR, "prepare Agent batch resource for graph start failed");
        return false;
    }
    _impl->latencyInputR->prepareGraphStart();
    autil::ScopedLock lock(_impl->condition);
    if (_impl->supervisor.joinable()) {
        return !_impl->stopping && State::STOPPED != getState();
    }
    _impl->stopping = false;
    _impl->state.store(State::RESTARTING, std::memory_order_release);
    _impl->supervisor =
        std::thread([impl = _impl.get()]() { impl->supervise(); });
    while (!_impl->stopping && State::RESTARTING == getState()) {
        _impl->condition.wait();
    }
    return State::RUNNING == getState();
}

void AgentGraph::stop()
{
    navi::NaviUserResultPtr userResult;
    {
        autil::ScopedLock lock(_impl->condition);
        if (!_impl->supervisor.joinable()) {
            _impl->state.store(State::STOPPED, std::memory_order_release);
            return;
        }
        _impl->stopping = true;
        _impl->state.store(State::STOPPING, std::memory_order_release);
        userResult = _impl->userResult;
        _impl->condition.broadcast();
    }
    const auto batchFinished = _impl->agentBatchR->finishGraph();
    const auto latencyFinished = _impl->latencyInputR->finish();
    if ((!batchFinished || !latencyFinished) && userResult) {
        userResult->abort(navi::EC_ABORT);
    }

    bool stopTimedOut = false;
    {
        const auto deadlineUs = autil::TimeUtility::monotonicTimeUs() +
                                AGENT_GRAPH_STOP_TIMEOUT_US;
        autil::ScopedLock lock(_impl->condition);
        while (State::STOPPED != getState()) {
            const auto remainingUs =
                deadlineUs - autil::TimeUtility::monotonicTimeUs();
            if (remainingUs <= 0 ||
                ETIMEDOUT == _impl->condition.wait(remainingUs)) {
                stopTimedOut = true;
                userResult = _impl->userResult;
                break;
            }
        }
    }
    if (stopTimedOut && userResult) {
        AUTIL_LOG(ERROR, "stop Agent graph timed out, abort graph");
        userResult->abort(navi::EC_ABORT);
    }
    _impl->condition.broadcast();
    _impl->supervisor.join();
    {
        autil::ScopedLock lock(_impl->condition);
        _impl->userResult.reset();
        _impl->closure.reset();
        _impl->state.store(State::STOPPED, std::memory_order_release);
    }
}

bool AgentGraph::running() const { return State::RUNNING == getState(); }

AgentGraph::State AgentGraph::getState() const
{
    return _impl->state.load(std::memory_order_acquire);
}

std::string AgentGraph::getStateName() const
{
    switch (getState()) {
    case State::STOPPED:
        return "stopped";
    case State::RUNNING:
        return "running";
    case State::RESTARTING:
        return "restarting";
    case State::STOPPING:
        return "stopping";
    }
    return "unknown";
}

uint64_t AgentGraph::getRestartCount() const
{
    return _impl->restartCount.load(std::memory_order_relaxed);
}

bool AgentGraph::abortCurrentGraphForTest()
{
    navi::NaviUserResultPtr userResult;
    {
        autil::ScopedLock lock(_impl->condition);
        userResult = _impl->userResult;
    }
    if (!userResult) {
        return false;
    }
    userResult->abort(navi::EC_ABORT);
    return true;
}

} // namespace navi_sched
