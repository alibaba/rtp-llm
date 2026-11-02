#include "aios/navi_sched/test/simulator/SimulatedAgentAccess.h"

#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/service/NaviSchedAgentImpl.h"
#include "aios/navi_sched/test/simulator/SimulatedExecutionR.h"
#include "autil/Lock.h"
#include "autil/legacy/jsonizable.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviUserResult.h"

namespace navi_sched {
namespace {

const int64_t SIMULATED_AGENT_GRAPH_TIMEOUT_MS = int64_t(10) * 365 * 24 * 60 * 60 * 1000;

class SimulatedAgentClosure : public navi::NaviUserResultClosure {
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

std::string makeConfig(const test::SimulatedExecutionConfig &config)
{
    autil::legacy::json::JsonMap value;
    value["cache_capacity_bytes"] = config.cacheCapacityBytes;
    value["token_cache_bytes"] = config.tokenCacheBytes;
    value["block_size"] = config.blockSize;
    autil::legacy::json::JsonArray formulaParameters;
    formulaParameters.reserve(config.formulaParameters.size());
    for (auto parameter : config.formulaParameters) {
        formulaParameters.emplace_back(parameter);
    }
    value["formula_parameters"] = std::move(formulaParameters);
    value["noise_stddev_ratio"] = config.noiseStddevRatio;
    value["random_seed"] = config.randomSeed;
    value["time_compression"] = config.timeCompression;
    value["queue_trace_output_path"] = config.queueTraceOutputPath;
    return autil::legacy::ToJsonString(value, true);
}

std::unique_ptr<navi::GraphDef> buildGraph(const std::string &bizName,
                                           const std::string &config)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    auto graphId = builder.newSubGraph(bizName);
    builder.subGraph(graphId).location(bizName, 1, 0);
    builder.node("navi_sched_simulated_execution")
        .kernel("navi_sched.agent_simulate_execution_k")
        .jsonAttrs(config)
        .out("output")
        .asGraphOutput("finished");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

}

class SimulatedAgentAccess::Impl {
public:
    NaviSchedAgentImpl *agentImpl = nullptr;
    navi::NaviPtr navi;
    std::shared_ptr<test::SimulatedExecutionBridgeR> executionBridgeR;
    std::shared_ptr<test::SimulatedExecutionR> executionR;
    navi::NaviUserResultPtr userResult;
    std::shared_ptr<SimulatedAgentClosure> closure;
};

SimulatedAgentAccess::SimulatedAgentAccess(NaviSchedAgentImpl *impl)
    : _impl(std::make_unique<Impl>())
{
    _impl->agentImpl = impl;
}

SimulatedAgentAccess::~SimulatedAgentAccess() { stop(); }

std::shared_ptr<SimulatedAgentAccess> SimulatedAgentAccess::create(
    NaviSchedAgent &agent, const test::SimulatedExecutionConfig &simulateConfig)
{
    auto impl = NaviSchedAgentImpl::getImpl(agent);
    if (!impl || !impl->init()) {
        return nullptr;
    }
    auto access = std::shared_ptr<SimulatedAgentAccess>(new SimulatedAgentAccess(impl));
    return access->start(simulateConfig) ? access : nullptr;
}

bool SimulatedAgentAccess::start(const test::SimulatedExecutionConfig &simulateConfig)
{
    if (simulateConfig.timeCompression <= 0.0 || simulateConfig.blockSize <= 0 ||
        (!simulateConfig.formulaParameters.empty() &&
         !NonLinearPrefillModelR::validateParameters(simulateConfig.formulaParameters))) {
        return false;
    }
    _impl->navi = _impl->agentImpl->getRuntimeNavi();
    _impl->executionBridgeR = std::make_shared<test::SimulatedExecutionBridgeR>();
    auto graphDef =
        buildGraph(_impl->agentImpl->getRuntimeBizName(), makeConfig(simulateConfig));
    if (!_impl->navi || !graphDef) {
        return false;
    }
    navi::RunGraphParams params;
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    params.setTimeoutMs(SIMULATED_AGENT_GRAPH_TIMEOUT_MS);
    params.setTraceLevel("ERROR");
    _impl->closure = std::make_shared<SimulatedAgentClosure>();
    navi::ResourceMap resourceMap;
    if (!resourceMap.addResource(_impl->executionBridgeR)) {
        return false;
    }
    _impl->userResult = _impl->navi->runLocalGraphAsync(
        graphDef.release(), params, resourceMap, _impl->closure.get());
    if (!_impl->userResult) {
        _impl->closure.reset();
        return false;
    }
    _impl->executionR = _impl->executionBridgeR->waitExecutionR(30 * 1000 * 1000);
    if (!_impl->executionR) {
        _impl->userResult->abort(navi::EC_ABORT);
        _impl->closure->wait();
        _impl->userResult.reset();
        _impl->closure.reset();
        return false;
    }
    return true;
}

void SimulatedAgentAccess::stop()
{
    auto closure = std::move(_impl->closure);
    auto userResult = std::move(_impl->userResult);
    if (!closure) {
        return;
    }
    if (!_impl->executionR->finishGraph() && userResult) {
        userResult->abort(navi::EC_ABORT);
    }
    closure->wait();
    _impl->executionR.reset();
    _impl->executionBridgeR.reset();
    _impl->navi.reset();
}

bool SimulatedAgentAccess::submit(const NaviSchedForwardRequest &request,
                                  const navi::AsyncPipePtr &completionPipe)
{
    return _impl->executionR && _impl->executionR->submit(request, completionPipe);
}

bool SimulatedAgentAccess::reportBatch(NaviSchedAgent &agent, uint64_t requestId,
                                       int64_t totalTokenCount,
                                       int64_t cacheHitTokenCount)
{
    NaviSchedBatchRequest request;
    request.requestId = requestId;
    request.totalTokenCount = totalTokenCount;
    NaviSchedRequestStats stats;
    stats.requestId = requestId;
    stats.totalTokenCount = totalTokenCount;
    stats.cacheHitTokenCount = cacheHitTokenCount;
    return agent.addBatchRequest(request, 0, 0, [stats](const NaviSchedBatchPtr &batch) {
        if (!batch || !batch->finishReporter) {
            return;
        }
        if (batch->finishReporter->startBatch(batch->requestIds)) {
            batch->finishReporter->finishBatch({stats});
        }
    });
}

}
