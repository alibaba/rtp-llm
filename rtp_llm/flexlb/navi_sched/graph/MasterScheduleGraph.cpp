#include "aios/navi_sched/graph/MasterScheduleGraph.h"
#include "aios/navi_sched/graph/GraphVisRecorder.h"
#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "aios/navi_sched/scheduler/CostSchedulerGraph.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleResultGraph.h"
#include "aios/navi_sched/kernel/SchedulerConfigSyncKernel.h"
#include "autil/Lock.h"
#include "autil/legacy/json.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviUserResult.h"
#include <atomic>
#include <thread>

namespace navi_sched {
namespace {

const int64_t SCHEDULE_GRAPH_TIMEOUT_MS = int64_t(10) * 365 * 24 * 60 * 60 * 1000;
const int64_t SCHEDULE_GRAPH_RESTART_DELAY_US = 100 * 1000;
const int64_t SCHEDULE_GRAPH_READY_TIMEOUT_MS = 30 * 1000;
const std::string MASTER_SCHEDULE_GRAPH_VIS_FILE = "./master_schedule_graph.vis";

class ScheduleGraphClosure : public navi::NaviUserResultClosure {
public:
    ScheduleGraphClosure(bool enableVis, const MasterScheduleRPtr &masterScheduleR,
                         uint64_t requestPipeGeneration)
        : _enableVis(enableVis)
        , _masterScheduleR(masterScheduleR)
        , _requestPipeGeneration(requestPipeGeneration)
    {
        waitDestruct = true;
    }

    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        consumeToEof(userResult);
        if (_masterScheduleR) {
            _masterScheduleR->clearRequestPipe(_requestPipeGeneration);
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
                                   MASTER_SCHEDULE_GRAPH_VIS_FILE);
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
    MasterScheduleRPtr _masterScheduleR;
    uint64_t _requestPipeGeneration;
    autil::ThreadCond _condition;
    navi::NaviUserResultPtr _userResult;
    bool _completed = false;
};

NAVI_TYPEDEF_PTR(ScheduleGraphClosure);

class MasterScheduleGraphBuilder {
public:
    static bool addSchedulePath(
        navi::GraphBuilder &builder, const navi::N &context,
        const navi::N &result, const std::string &schedulerKernelId,
        const std::string &schedulerKernelConfig)
    {
        if (CostSchedulerKernel::KERNEL_ID != schedulerKernelId) {
            auto schedule = builder.node("navi_sched_master_schedule")
                                .kernel(schedulerKernelId)
                                .jsonAttrs(schedulerKernelConfig);
            context.out("context").to(schedule.in("context")).require(true);
            schedule.out("result").to(result.in("result")).require(true);
            return true;
        }
        return CostSchedulerGraphBuilder::add(
            builder, context, result, schedulerKernelConfig);
    }
};

}

std::unique_ptr<navi::GraphDef> MasterScheduleGraph::buildGraph(
    const std::string &masterBizName, const std::string &contextKernelConfig,
    const std::string &schedulerKernelId,
    const std::string &schedulerKernelConfig,
    uint64_t requestPipeGeneration)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    auto graphId = builder.newSubGraph(masterBizName);
    builder.subGraph(graphId).location(masterBizName, 1, 0);
    auto context = builder.node("navi_sched_master_schedule_context")
                       .kernel(MasterScheduleContextKernel::KERNEL_ID)
                       .jsonAttrs(contextKernelConfig)
                       .attr("scheduler_kernel_id", schedulerKernelId)
                       .integerAttr("request_pipe_generation",
                                    static_cast<int64_t>(requestPipeGeneration));
    auto result = builder.node("navi_sched_master_schedule_result")
                      .kernel(MasterScheduleResultKernel::KERNEL_ID);
    CostSchedulerConfig costConfig;
    try {
        autil::legacy::FastFromJsonString(costConfig, schedulerKernelConfig);
    } catch (const autil::legacy::ExceptionBase &) {
        return nullptr;
    }
    const auto workerCount = CostSchedulerGraphBuilder::getWorkerCount(costConfig.parallelism);
    auto resultPrepare = builder.node("navi_sched_master_schedule_result_prepare")
        .kernel(MasterScheduleResultPrepareKernel::KERNEL_ID)
        .jsonAttrs("{\"worker_count\":" + std::to_string(workerCount) + "}");
    auto deliveryBarrier = builder.node("navi_sched_master_schedule_delivery_barrier")
        .kernel(MasterScheduleResultBarrierKernel::KERNEL_ID)
        .jsonAttrs("{\"phase\":1,\"worker_count\":" + std::to_string(workerCount) + "}");
    auto cleanupBarrier = builder.node("navi_sched_master_schedule_cleanup_barrier")
        .kernel(MasterScheduleResultBarrierKernel::KERNEL_ID)
        .jsonAttrs("{\"phase\":2,\"worker_count\":" + std::to_string(workerCount) + "}");
    for (int32_t phase = 0; phase < 3; ++phase) {
        auto source = phase == 0 ? resultPrepare :
            (phase == 1 ? deliveryBarrier : cleanupBarrier);
        auto target = phase == 0 ? deliveryBarrier :
            (phase == 1 ? cleanupBarrier : result);
        for (size_t index = 0; index < workerCount; ++index) {
            auto worker = builder.node("navi_sched_master_schedule_result_worker_" +
                                       std::to_string(phase) + "_" + std::to_string(index))
                .kernel(MasterScheduleResultWorkerKernel::KERNEL_ID)
                .jsonAttrs("{\"phase\":" + std::to_string(phase) +
                           ",\"worker_index\":" + std::to_string(index) + "}");
            source.out("workers").asGroup().index(static_cast<navi::IndexType>(index))
                .to(worker.in("state")).require(true);
            worker.out("done").to(target.in("parts").asGroup()
                .index(static_cast<navi::IndexType>(index))).require(true);
        }
    }
    auto schedulerConfigSync =
        builder.node("navi_sched_scheduler_config_sync")
            .kernel(SchedulerConfigSyncKernel::KERNEL_ID);
    if (!MasterScheduleGraphBuilder::addSchedulePath(
            builder, context, resultPrepare, schedulerKernelId,
            schedulerKernelConfig)) {
        return nullptr;
    }
    result.out("finished").asGraphOutput("finished");
    schedulerConfigSync.out("finished")
        .asGraphOutput("scheduler_config_finished");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

class MasterScheduleGraph::Impl {
public:
    bool valid() const
    {
        return navi && masterScheduleR && schedulerConfigR &&
               !masterBizName.empty() && !contextKernelConfig.empty() &&
               !schedulerKernelId.empty() && !schedulerKernelConfig.empty();
    }

    bool launchGraph()
    {
        const auto requestPipeGeneration =
            masterScheduleR->beginRequestPipeGeneration();
        if (0 == requestPipeGeneration) {
            return false;
        }
        auto graphDef = MasterScheduleGraph::buildGraph(
            masterBizName, contextKernelConfig, schedulerKernelId,
            schedulerKernelConfig, requestPipeGeneration);
        if (!graphDef) {
            masterScheduleR->clearRequestPipe(requestPipeGeneration);
            return false;
        }
        navi::RunGraphParams params;
        params.setResourceBuildMode(navi::RBM_PARALLEL);
        params.setTimeoutMs(SCHEDULE_GRAPH_TIMEOUT_MS);
        if (enableVis) {
            params.setCollectMetric(true);
            params.addParamTopoTreeStr("edge_key=*;edge_query=2");
        }
        auto nextClosure =
            std::make_shared<ScheduleGraphClosure>(
                enableVis, masterScheduleR, requestPipeGeneration);
        auto nextUserResult = navi->runLocalGraphAsync(
            graphDef.release(), params, {}, nextClosure.get());
        if (!nextUserResult) {
            AUTIL_LOG(ERROR,
                      "run Master schedule graph failed, requestPipeGeneration [%lu]",
                      requestPipeGeneration);
            masterScheduleR->clearRequestPipe(requestPipeGeneration);
            return false;
        }
        if (!masterScheduleR->waitRequestPipe(
                requestPipeGeneration, SCHEDULE_GRAPH_READY_TIMEOUT_MS)) {
            AUTIL_LOG(ERROR,
                      "Master schedule graph did not install request pipe within [%ld]ms, "
                      "requestPipeGeneration [%lu], currentGeneration [%lu], stopped [%d]",
                      SCHEDULE_GRAPH_READY_TIMEOUT_MS, requestPipeGeneration,
                      masterScheduleR->getRequestPipeGeneration(),
                      static_cast<int>(masterScheduleR->stopped()));
            nextUserResult->abort(navi::EC_ABORT);
            auto failedResult = nextClosure->waitAndTake();
            if (failedResult) {
                const auto naviResult = failedResult->getNaviResult();
                if (naviResult) {
                    AUTIL_LOG(ERROR,
                              "failed Master schedule graph result, "
                              "errorCode [%d], errorMessage [%s]",
                              static_cast<int>(naviResult->getErrorCode()),
                              naviResult->getErrorMessage().c_str());
                }
            }
            failedResult.reset();
            nextUserResult.reset();
            nextClosure.reset();
            masterScheduleR->clearRequestPipe(requestPipeGeneration);
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
                state.store(MasterScheduleGraph::State::RUNNING,
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
        masterScheduleR->clearRequestPipe(requestPipeGeneration);
        return false;
    }

    bool waitRestartDelay()
    {
        autil::ScopedLock lock(condition);
        if (stopping) {
            return false;
        }
        condition.wait(SCHEDULE_GRAPH_RESTART_DELAY_US);
        return !stopping;
    }

    void finishStopped()
    {
        autil::ScopedLock lock(condition);
        state.store(MasterScheduleGraph::State::STOPPED,
                    std::memory_order_release);
        condition.broadcast();
    }

    void supervise()
    {
        while (true) {
            {
                autil::ScopedLock lock(condition);
                if (stopping) {
                    state.store(MasterScheduleGraph::State::STOPPED,
                                std::memory_order_release);
                    firstLaunchCompleted = true;
                    condition.broadcast();
                    return;
                }
            }
            if (!launchGraph()) {
                bool shouldStop = false;
                {
                    autil::ScopedLock lock(condition);
                    firstLaunchCompleted = true;
                    shouldStop = stopping;
                    condition.broadcast();
                }
                if (shouldStop) {
                    finishStopped();
                    return;
                }
                const auto currentRestartCount =
                    restartCount.fetch_add(1, std::memory_order_relaxed) + 1;
                AUTIL_LOG(ERROR,
                          "launch Master schedule graph failed, retry after [%ld]us, "
                          "restartCount [%lu]",
                          SCHEDULE_GRAPH_RESTART_DELAY_US,
                          currentRestartCount);
                if (!waitRestartDelay()) {
                    finishStopped();
                    return;
                }
                continue;
            }

            ScheduleGraphClosurePtr completedClosure;
            {
                autil::ScopedLock lock(condition);
                firstLaunchCompleted = true;
                completedClosure = closure;
                condition.broadcast();
            }
            const auto runningRestartCount =
                restartCount.load(std::memory_order_relaxed);
            if (runningRestartCount > 0) {
                AUTIL_LOG(INFO,
                          "Master schedule graph running, restartCount [%lu], "
                          "requestPipeGeneration [%lu]",
                          runningRestartCount,
                          masterScheduleR->getRequestPipeGeneration());
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
                state.store(shouldStop ? MasterScheduleGraph::State::STOPPING
                                       : MasterScheduleGraph::State::RESTARTING,
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
                      "Master schedule graph destructed unexpectedly, errorCode [%d], "
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
    MasterScheduleRPtr masterScheduleR;
    SchedulerConfigRPtr schedulerConfigR;
    std::string masterBizName;
    std::string contextKernelConfig;
    std::string schedulerKernelId;
    std::string schedulerKernelConfig;
    bool enableVis = false;
    navi::NaviUserResultPtr userResult;
    ScheduleGraphClosurePtr closure;
    mutable autil::ThreadCond condition;
    std::thread supervisor;
    std::atomic<MasterScheduleGraph::State> state{
        MasterScheduleGraph::State::STOPPED};
    std::atomic<uint64_t> restartCount{0};
    bool stopping = true;
    bool firstLaunchCompleted = false;
};

AUTIL_LOG_SETUP(navi_sched, MasterScheduleGraph);

MasterScheduleGraph::MasterScheduleGraph(const navi::NaviPtr &navi,
                                         const MasterScheduleRPtr &masterScheduleR,
                                         const SchedulerConfigRPtr &schedulerConfigR,
                                         const std::string &masterBizName,
                                         const std::string &contextKernelConfig,
                                         const std::string &schedulerKernelId,
                                         const std::string &schedulerKernelConfig,
                                         bool enableVis)
    : _impl(std::make_unique<Impl>())
{
    _impl->navi = navi;
    _impl->masterScheduleR = masterScheduleR;
    _impl->schedulerConfigR = schedulerConfigR;
    _impl->masterBizName = masterBizName;
    _impl->contextKernelConfig = contextKernelConfig;
    _impl->schedulerKernelId = schedulerKernelId;
    _impl->schedulerKernelConfig = schedulerKernelConfig;
    _impl->enableVis = enableVis;
}

MasterScheduleGraph::~MasterScheduleGraph() { stop(); }

bool MasterScheduleGraph::start()
{
    autil::ScopedLock lock(_impl->condition);
    if (_impl->supervisor.joinable()) {
        return !_impl->stopping;
    }
    if (!_impl->valid()) {
        return false;
    }
    _impl->stopping = false;
    _impl->firstLaunchCompleted = false;
    _impl->state.store(State::RESTARTING, std::memory_order_release);
    _impl->supervisor = std::thread([impl = _impl.get()]() { impl->supervise(); });
    while (!_impl->firstLaunchCompleted && !_impl->stopping) {
        _impl->condition.wait();
    }
    return !_impl->stopping;
}

void MasterScheduleGraph::stop()
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
    _impl->masterScheduleR->stop();
    const auto configSyncStopped =
        _impl->schedulerConfigR && _impl->schedulerConfigR->stopSyncPipe();
    if (!configSyncStopped && userResult) {
        userResult->abort(navi::EC_ABORT);
    }
    _impl->supervisor.join();
    {
        autil::ScopedLock lock(_impl->condition);
        _impl->userResult.reset();
        _impl->closure.reset();
        _impl->state.store(State::STOPPED, std::memory_order_release);
    }
}

bool MasterScheduleGraph::running() const
{
    return State::RUNNING == getState();
}

MasterScheduleGraph::State MasterScheduleGraph::getState() const
{
    return _impl->state.load(std::memory_order_acquire);
}

std::string MasterScheduleGraph::getStateName() const
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

uint64_t MasterScheduleGraph::getRestartCount() const
{
    return _impl->restartCount.load(std::memory_order_relaxed);
}

}
