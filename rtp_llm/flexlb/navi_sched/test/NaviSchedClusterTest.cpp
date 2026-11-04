#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterPendingCacheSyncToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "rtp_llm/flexlb/navi_sched/graph/ClientGraphRunner.h"
#include "rtp_llm/flexlb/navi_sched/graph/MasterCollectorManager.h"
#include "rtp_llm/flexlb/navi_sched/graph/MasterScheduleGraph.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentBatchR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentReportR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentInstanceR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentRequestR.h"
#include "rtp_llm/flexlb/navi_sched/resource/ClientResponseR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterScheduleR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterPendingCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterLatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MaxCacheHitSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerGraph.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleContextData.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleResultGraph.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/RandomSchedulerKernel.h"
#include "autil/HashAlgorithm.h"
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "autil/legacy/json.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/resource/GigClientR.h"
#include "navi/engine/Kernel.h"
#include "navi/engine/GraphResult.h"
#include "navi/engine/Graph.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/Node.h"
#include "navi/log/NaviLogger.h"
#include "navi/test_cluster/NaviTestCluster.h"
#include "unittest/unittest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace navi_sched;

namespace {

uint64_t hashString(const std::string &value)
{
    return autil::HashAlgorithm::hashString64(value.data(), value.size());
}

NaviSchedBlockHashesConstPtr makeBlockHashes(std::initializer_list<uint64_t> values)
{
    return std::make_shared<const NaviSchedBlockHashes>(values);
}

NaviSchedLatencyOptimizerState makeLatencyOptimizerState(size_t parameterCount,
                                                         int64_t step)
{
    NaviSchedLatencyOptimizerState optimizer;
    optimizer.adamMoment1.assign(parameterCount, 0.0);
    optimizer.adamMoment2.assign(parameterCount, 0.0);
    optimizer.step = step;
    return optimizer;
}

AgentUpdateToMasterDataPtr makeHandshakeUpdate(int32_t serverPort,
                                               bool rampCompleted = false)
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    update->eventTimeUs = autil::TimeUtility::currentTime();
    update->control = std::make_shared<AgentControlToMasterData>();
    update->control->serverInfo.port = serverPort;
    update->control->agentStartTimeUs =
        update->eventTimeUs - (rampCompleted ? 61'000'000 : 0);
    update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    update->requestEvent->inflightStatsRevision = 1;
    return update;
}

void attachTrustedLatencyModel(
    const AgentUpdateToMasterDataPtr &update,
    const std::vector<double> &parameters, int64_t modelRevision = 1024)
{
    if (update->eventTimeUs <= 0) {
        update->eventTimeUs = autil::TimeUtility::currentTime();
    }
    if (!update->requestEvent) {
        update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
        update->requestEvent->inflightStatsRevision = 1;
    }
    update->latency = std::make_shared<AgentLatencyToMasterData>();
    update->latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    update->latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    update->latency->modelRevision = modelRevision;
    update->latency->parameters = parameters;
    update->latency->optimizerState =
        makeLatencyOptimizerState(parameters.size(), modelRevision);
    update->latency->sessionSampleCount = 128;
    update->latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(parameters);
    update->latencyModelTrusted = true;
}

AgentUpdateToMasterDataPtr makeCachePolicyUpdate(int64_t blockSize,
                                                 int64_t linearStep = 1,
                                                 bool hybridTailRequired = false)
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::CACHE_UPDATED;
    update->cache = std::make_shared<AgentCacheToMasterData>();
    update->cache->blockSize = blockSize;
    update->cache->linearStep = linearStep;
    update->cache->hybridTailRequired = hybridTailRequired ? 1 : 0;
    update->cache->fullSnapshot = true;
    update->cache->revision = 1;
    update->cache->cachedBlockCount = 0;
    update->cache->fullReadyBlockCount = 0;
    update->cache->tailReadyBlockCount = 0;
    return update;
}

AgentUpdateToMasterDataPtr makeEmptyPendingCacheSnapshot()
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    update->pendingCache = std::make_shared<AgentPendingCacheToMasterData>();
    update->pendingCache->fullSnapshot = true;
    return update;
}

class ScheduleResponsePipe : public navi::AsyncPipe {
public:
    ScheduleResponsePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

public:
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        autil::ScopedLock lock(_condition);
        _data = data;
        _condition.broadcast();
        return navi::EC_NONE;
    }

    navi::ErrorCode setEof() override
    {
        autil::ScopedLock lock(_condition);
        _eof = true;
        _condition.broadcast();
        return navi::EC_NONE;
    }

    MasterResponseToClientDataPtr waitResponse(int64_t timeoutUs = 0)
    {
        autil::ScopedLock lock(_condition);
        while (!_eof) {
            if (ETIMEDOUT == _condition.wait(timeoutUs)) {
                return nullptr;
            }
        }
        return std::dynamic_pointer_cast<MasterResponseToClientData>(_data);
    }

    bool waitEof(int64_t timeoutUs = 0)
    {
        autil::ScopedLock lock(_condition);
        while (!_eof) {
            if (ETIMEDOUT == _condition.wait(timeoutUs)) {
                return false;
            }
        }
        return true;
    }

private:
    autil::ThreadCond _condition;
    navi::DataPtr _data;
    bool _eof = false;
};

class SchedulerConfigResponsePipe : public navi::AsyncPipe {
public:
    SchedulerConfigResponsePipe()
        : navi::AsyncPipe(nullptr, navi::AS_REQUIRED)
    {
    }

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        autil::ScopedLock lock(_condition);
        _response = std::dynamic_pointer_cast<SchedulerConfigResponseData>(data);
        _condition.broadcast();
        return navi::EC_NONE;
    }

    SchedulerConfigResponseDataPtr waitResponse(int64_t timeoutUs)
    {
        autil::ScopedLock lock(_condition);
        while (!_response) {
            if (ETIMEDOUT == _condition.wait(timeoutUs)) {
                return nullptr;
            }
        }
        return _response;
    }

private:
    autil::ThreadCond _condition;
    SchedulerConfigResponseDataPtr _response;
};

std::shared_ptr<ClientRequestToMasterData> makeClientRequest(
    uint64_t requestId, const std::string &bizName, int64_t promptTokenCount,
    const NaviSchedBlockHashesConstPtr &blockHashes)
{
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = requestId;
    request->bizName = bizName;
    request->promptTokenCount = promptTokenCount;
    request->blockHashes = blockHashes;
    request->timeoutMs = 1000;
    return request;
}

bool containsKernel(const navi::GraphVisNodeDef &visNode, const std::string &kernelName)
{
    if (visNode.has_result()) {
        for (const auto &subGraph : visNode.result().graph().sub_graphs()) {
            for (const auto &node : subGraph.nodes()) {
                if (kernelName == node.kernel_name()) {
                    return true;
                }
            }
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        if (containsKernel(pair.second, kernelName)) {
            return true;
        }
    }
    return false;
}

size_t countSummarizedDataEvents(const navi::GraphVisNodeDef &visNode,
                                 const std::string &typeId)
{
    size_t count = 0;
    if (visNode.has_result() && visNode.result().has_graph_metric()) {
        for (const auto &edge : visNode.result().graph_metric().edge_metrics()) {
            if ("2" != edge.summary_query()) {
                continue;
            }
            for (const auto &event : edge.set_events()) {
                if (event.success() && event.summary_hash() != 0 &&
                    typeId == event.type_id()) {
                    ++count;
                }
            }
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        count += countSummarizedDataEvents(pair.second, typeId);
    }
    return count;
}

size_t countSummarizedDataEvents(const navi::GraphVisDef &vis, const std::string &typeId)
{
    size_t count = 0;
    for (const auto &node : vis.nodes()) {
        count += countSummarizedDataEvents(node, typeId);
    }
    return count;
}

bool containsKernel(const navi::GraphVisDef &vis, const std::string &kernelName)
{
    for (const auto &node : vis.nodes()) {
        if (containsKernel(node, kernelName)) {
            return true;
        }
    }
    return false;
}

bool containsKernelMetric(const navi::GraphVisNodeDef &visNode,
                          const std::string &kernelName)
{
    if (visNode.has_result() && visNode.result().has_graph_metric()) {
        for (const auto &metric : visNode.result().graph_metric().metrics()) {
            if (kernelName == metric.kernel()) {
                return true;
            }
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        if (containsKernelMetric(pair.second, kernelName)) {
            return true;
        }
    }
    return false;
}

int64_t getKernelScheduleCount(const navi::GraphVisNodeDef &visNode,
                               const std::string &kernelName)
{
    if (visNode.has_result() && visNode.result().has_graph_metric()) {
        for (const auto &metric : visNode.result().graph_metric().metrics()) {
            if (kernelName == metric.kernel()) {
                return metric.schedule_count();
            }
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        auto count = getKernelScheduleCount(pair.second, kernelName);
        if (count >= 0) {
            return count;
        }
    }
    return -1;
}

int64_t getKernelScheduleCount(const navi::GraphVisDef &vis,
                               const std::string &kernelName)
{
    for (const auto &node : vis.nodes()) {
        auto count = getKernelScheduleCount(node, kernelName);
        if (count >= 0) {
            return count;
        }
    }
    return -1;
}

bool containsRemoteKernelMetric(const navi::GraphVisNodeDef &visNode,
                                const std::string &bizName,
                                const std::string &kernelName)
{
    if (visNode.location().type() == navi::NRT_RPC &&
        visNode.location().rpc_location().to_biz() == bizName &&
        containsKernelMetric(visNode, kernelName)) {
        return true;
    }
    for (const auto &pair : visNode.sub_results()) {
        if (containsRemoteKernelMetric(pair.second, bizName, kernelName)) {
            return true;
        }
    }
    return false;
}

bool containsRemoteKernelMetric(const navi::GraphVisDef &vis, const std::string &bizName,
                                const std::string &kernelName)
{
    for (const auto &node : vis.nodes()) {
        if (containsRemoteKernelMetric(node, bizName, kernelName)) {
            return true;
        }
    }
    return false;
}

void collectSubGraphTimeouts(const navi::GraphVisNodeDef &visNode,
                             const std::string &bizName,
                             std::vector<int64_t> &timeouts)
{
    if (visNode.has_result()) {
        for (const auto &subGraph : visNode.result().graph().sub_graphs()) {
            if (subGraph.location().biz_name() == bizName) {
                timeouts.emplace_back(subGraph.option().timeout_ms());
            }
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        collectSubGraphTimeouts(pair.second, bizName, timeouts);
    }
}

std::vector<int64_t> collectSubGraphTimeouts(const navi::GraphVisDef &vis,
                                             const std::string &bizName)
{
    std::vector<int64_t> timeouts;
    for (const auto &node : vis.nodes()) {
        collectSubGraphTimeouts(node, bizName, timeouts);
    }
    return timeouts;
}

bool containsGraphError(const navi::GraphVisNodeDef &visNode)
{
    if (visNode.has_result() && visNode.result().has_error() &&
        navi::EC_NONE != visNode.result().error().ec()) {
        return true;
    }
    for (const auto &pair : visNode.sub_results()) {
        if (containsGraphError(pair.second)) {
            return true;
        }
    }
    return false;
}

bool containsGraphError(const navi::GraphVisDef &vis)
{
    for (const auto &node : vis.nodes()) {
        if (containsGraphError(node)) {
            return true;
        }
    }
    return false;
}

class MasterPeriodicPipeTestKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.master_periodic_pipe_test_k").output("tick", "");
    }

    bool config(navi::KernelConfigContext &ctx) override
    {
        NAVI_JSONIZE(ctx, "repeat", _repeat, _repeat);
        NAVI_JSONIZE(ctx, "stop_after_repeat", _stopAfterRepeat, _stopAfterRepeat);
        return true;
    }

    navi::ErrorCode init(navi::KernelInitContext &ctx) override
    {
        _periodicPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
        auto repeat = _repeat > 0 ? 1 : _repeat;
        return _periodicPipe && ctx.registerPeriodicPipe(_periodicPipe, 1, repeat)
                   ? navi::EC_NONE
                   : navi::EC_INIT_SUB_GRAPH;
    }

    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (_repeat <= 0 && 0 != _periodicPipe->getDataItemCnt()) {
            return navi::EC_ABORT;
        }
        _periodicPipe->getData(data, eof);
        ++_triggerRepeat;
        int64_t activatedCount = 0;
        int64_t remainingCount = 0;
        int64_t periodMs = 0;
        int64_t nextTickTimeMs = 0;
        if (!ctx.getPeriodicPipeActivationCount(_periodicPipe, activatedCount,
                                                remainingCount, periodMs,
                                                nextTickTimeMs)) {
            return navi::EC_ABORT;
        }
        if (1 != periodMs ||
            (_repeat > 0 &&
             (1 != activatedCount || 0 != remainingCount || -1 != nextTickTimeMs)) ||
            (_repeat <= 0 && (activatedCount != _triggerRepeat ||
                              remainingCount != _repeat || nextTickTimeMs <= 0))) {
            return navi::EC_ABORT;
        }
        auto stopped = _stopAfterRepeat > 0 && _triggerRepeat >= _stopAfterRepeat;
        auto repeatFinished = _repeat > 0 && _triggerRepeat >= _repeat;
        if (stopped && _repeat <= 0 && !ctx.stopPeriodicAsyncPipe(_periodicPipe)) {
            return navi::EC_ABORT;
        }
        if (!stopped && !repeatFinished && _repeat > 0 &&
            !ctx.registerPeriodicPipe(_periodicPipe, 1, 1)) {
            return navi::EC_ABORT;
        }
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX),
                      std::make_shared<ClientRequestToMasterData>(),
                      eof || stopped || repeatFinished);
        return navi::EC_NONE;
    }

private:
    navi::AsyncPipePtr _periodicPipe;
    int64_t _repeat = 1;
    int64_t _stopAfterRepeat = 0;
    int64_t _triggerRepeat = 0;
};

REGISTER_KERNEL(MasterPeriodicPipeTestKernel);

class FailNMasterSchedulerKernel : public navi::Kernel {
public:
    static constexpr const char *KERNEL_ID =
        "navi_sched.fail_n_test_scheduler_k";

public:
    static void reset(size_t failureCount = 1, bool spreadRequests = false)
    {
        _remainingFailures.store(failureCount, std::memory_order_release);
        _spreadRequests.store(spreadRequests, std::memory_order_release);
    }

    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name(KERNEL_ID)
            .input("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
            .output("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
    }

    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_NONE;
        }
        if (data) {
            auto remaining = _remainingFailures.load(std::memory_order_acquire);
            while (remaining > 0) {
                if (_remainingFailures.compare_exchange_weak(
                        remaining, remaining - 1, std::memory_order_acq_rel)) {
                    return navi::EC_ABORT;
                }
            }
        }
        auto context = MasterScheduleContextData::getContext(data);
        if (context) {
            for (size_t requestIndex = 0;
                 requestIndex < context->requestCount(); ++requestIndex) {
                const auto nodeIndex = _spreadRequests.load(std::memory_order_acquire)
                    ? requestIndex % context->nodeCount() : 0;
                if (!context->selectNode(requestIndex, nodeIndex, "restarted_selected")) {
                    return navi::EC_ABORT;
                }
            }
        }
        return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)
                   ? navi::EC_NONE
                   : navi::EC_ABORT;
    }

private:
    static std::atomic<size_t> _remainingFailures;
    static std::atomic<bool> _spreadRequests;
};

std::atomic<size_t> FailNMasterSchedulerKernel::_remainingFailures{1};
std::atomic<bool> FailNMasterSchedulerKernel::_spreadRequests{false};

REGISTER_KERNEL(FailNMasterSchedulerKernel);

class AlwaysFailInitMasterSchedulerKernel : public navi::Kernel {
public:
    static constexpr const char *KERNEL_ID =
        "navi_sched.always_fail_init_test_scheduler_k";

public:
    static void reset(
        size_t failureCount = std::numeric_limits<size_t>::max())
    {
        _initCount.store(0, std::memory_order_release);
        _remainingFailures.store(failureCount, std::memory_order_release);
    }

    static size_t getInitCount()
    {
        return _initCount.load(std::memory_order_acquire);
    }

    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name(KERNEL_ID)
            .input("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
            .output("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
    }

    navi::ErrorCode init(navi::KernelInitContext &) override
    {
        _initCount.fetch_add(1, std::memory_order_acq_rel);
        auto remaining = _remainingFailures.load(std::memory_order_acquire);
        while (remaining > 0) {
            if (_remainingFailures.compare_exchange_weak(
                    remaining, remaining - 1, std::memory_order_acq_rel)) {
                return navi::EC_ABORT;
            }
        }
        return navi::EC_NONE;
    }

    navi::ErrorCode compute(navi::KernelComputeContext &) override
    {
        return navi::EC_NONE;
    }

private:
    static std::atomic<size_t> _initCount;
    static std::atomic<size_t> _remainingFailures;
};

std::atomic<size_t> AlwaysFailInitMasterSchedulerKernel::_initCount{0};
std::atomic<size_t> AlwaysFailInitMasterSchedulerKernel::_remainingFailures{
    std::numeric_limits<size_t>::max()};

REGISTER_KERNEL(AlwaysFailInitMasterSchedulerKernel);

} // namespace

class NaviSchedClusterTest : public TESTBASE {
public:
    void setUp() override
    {
        _pythonHome = std::make_unique<autil::EnvGuard>(
            "NAVI_PYTHON_HOME",
            "./aios/navi/config_loader/python:/opt/conda310/lib/python3.10");
    }

protected:
    struct ReservationState {
        size_t batchCount = 0;
        size_t dispatchWaiterCount = 0;
    };

    ReservationState getReservationState(
        const MasterStateR::NodeStatePtr &state) const
    {
        ReservationState result;
        if (!state) {
            return result;
        }
        autil::ScopedLock lock(state->_runtimeLock);
        result.batchCount = state->_pendingBatchReservations.size();
        for (const auto &reservation : state->_pendingBatchReservations) {
            result.dispatchWaiterCount += reservation->dispatchWaiters.size();
        }
        return result;
    }

    AgentReportRPtr loadAgentReportR(const std::string &host,
                                     const std::string &bizName) const
    {
        auto navi = _cluster.getNavi(host);
        if (!navi) {
            return nullptr;
        }
        navi::ResourceMap resources;
        if (!navi->createResource(bizName, 1, 0, {AgentReportR::RESOURCE_ID},
                                  resources)) {
            return nullptr;
        }
        auto it = resources.getMap().find(AgentReportR::RESOURCE_ID);
        if (resources.getMap().end() == it) {
            return nullptr;
        }
        auto agent = std::dynamic_pointer_cast<AgentReportR>(it->second);
        if (!agent) {
            return nullptr;
        }
        NaviSchedCacheUpdate cachePolicy;
        cachePolicy.blockSize = 256;
        cachePolicy.linearStep = 1;
        cachePolicy.hybridTailRequired = 0;
        agent->reportCacheUpdated(cachePolicy);
        return agent;
    }

    LatencyModelRPtr loadLatencyModelR(const std::string &host,
                                       const std::string &bizName) const
    {
        auto navi = _cluster.getNavi(host);
        if (!navi) {
            return nullptr;
        }
        navi::ResourceMap resources;
        if (!navi->createResource(bizName, 1, 0,
                                  {LatencyModelR::RESOURCE_ID}, resources)) {
            return nullptr;
        }
        auto it = resources.getMap().find(LatencyModelR::RESOURCE_ID);
        return resources.getMap().end() == it
                   ? nullptr
                   : std::dynamic_pointer_cast<LatencyModelR>(it->second);
    }

    AgentSchedulerConfigRPtr loadAgentSchedulerConfigR(
        const std::string &host, const std::string &bizName) const
    {
        auto navi = _cluster.getNavi(host);
        if (!navi) {
            return nullptr;
        }
        navi::ResourceMap resources;
        if (!navi->createResource(
                bizName, 1, 0, {AgentSchedulerConfigR::RESOURCE_ID}, resources)) {
            return nullptr;
        }
        auto it = resources.getMap().find(AgentSchedulerConfigR::RESOURCE_ID);
        return resources.getMap().end() == it
                   ? nullptr
                   : std::dynamic_pointer_cast<AgentSchedulerConfigR>(it->second);
    }

    void startMasterScheduleGraph(const std::string &host)
    {
        auto masterNavi = _cluster.getNavi(host);
        ASSERT_TRUE(masterNavi);
        _masterScheduleGraph = std::make_shared<MasterScheduleGraph>(
            masterNavi, _masterScheduleR, _schedulerConfigR, "master_biz",
            "{\"max_wait_schedule_count\":" +
                std::to_string(_maxWaitScheduleCount) +
                ",\"max_wait_schedule_time_ms\":" +
                std::to_string(_maxWaitScheduleTimeMs) + "}",
            _schedulerKernelId, _schedulerKernelConfig, true);
        ASSERT_TRUE(_masterScheduleGraph->start());
    }

    void loadMasterResources(const std::string &host,
                             bool startScheduleGraph = true)
    {
        auto masterNavi = _cluster.getNavi(host);
        navi::ResourceMap resources;
        ASSERT_TRUE(masterNavi->createResource(
            "master_biz", 1, 0,
            {MasterScheduleR::RESOURCE_ID, MasterPendingCacheR::RESOURCE_ID,
             MasterStateR::RESOURCE_ID, SchedulerConfigR::RESOURCE_ID,
             MasterLatencyModelR::RESOURCE_ID},
            resources));
        auto scheduleIt = resources.getMap().find(MasterScheduleR::RESOURCE_ID);
        auto pendingCacheIt =
            resources.getMap().find(MasterPendingCacheR::RESOURCE_ID);
        auto stateIt = resources.getMap().find(MasterStateR::RESOURCE_ID);
        auto configIt = resources.getMap().find(SchedulerConfigR::RESOURCE_ID);
        auto latencyModelIt =
            resources.getMap().find(MasterLatencyModelR::RESOURCE_ID);
        ASSERT_NE(resources.getMap().end(), scheduleIt);
        ASSERT_NE(resources.getMap().end(), pendingCacheIt);
        ASSERT_NE(resources.getMap().end(), stateIt);
        ASSERT_NE(resources.getMap().end(), configIt);
        ASSERT_NE(resources.getMap().end(), latencyModelIt);
        auto pendingCacheR =
            std::dynamic_pointer_cast<MasterPendingCacheR>(pendingCacheIt->second);
        _masterPendingCacheR = pendingCacheR;
        _masterStateR = std::dynamic_pointer_cast<MasterStateR>(stateIt->second);
        _masterScheduleR = std::dynamic_pointer_cast<MasterScheduleR>(scheduleIt->second);
        _schedulerConfigR =
            std::dynamic_pointer_cast<SchedulerConfigR>(configIt->second);
        _masterLatencyModelR = std::dynamic_pointer_cast<MasterLatencyModelR>(
            latencyModelIt->second);
        ASSERT_TRUE(pendingCacheR);
        ASSERT_TRUE(_masterScheduleR);
        ASSERT_TRUE(_masterStateR);
        ASSERT_TRUE(_schedulerConfigR);
        ASSERT_TRUE(_masterLatencyModelR);
        ASSERT_EQ(pendingCacheR.get(), _masterStateR->_pendingCacheR);
        const auto updateSchedulerConfig =
            [this](const std::string &configJson) {
            SchedulerConfigR::Request request;
            request.update = true;
            request.configJson = configJson;
            SchedulerConfigResponseDataPtr response;
            bool waitingForAgents = false;
            ASSERT_TRUE(_schedulerConfigR->processRequest(
                request, response, waitingForAgents));
            ASSERT_FALSE(waitingForAgents);
            ASSERT_TRUE(response);
            ASSERT_TRUE(response->accepted) << response->error;
        };
        auto schedulerConfigPatch =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(_schedulerKernelConfig));
        schedulerConfigPatch.erase("parallelism");
        schedulerConfigPatch.erase("optimization_time_budget_us");
        updateSchedulerConfig(
            autil::legacy::ToJsonString(schedulerConfigPatch, true));
        updateSchedulerConfig(
            "{\"max_wait_schedule_count\":" +
                std::to_string(_maxWaitScheduleCount) +
                ",\"max_wait_schedule_time_ms\":" +
                std::to_string(_maxWaitScheduleTimeMs) + "}");
        if (startScheduleGraph) {
            ASSERT_NO_FATAL_FAILURE(startMasterScheduleGraph(host));
        }
    }

    bool reportReadyAgent(const std::string &agentId, int32_t port,
                          bool rampCompleted = false)
    {
        return _masterStateR->reportAgent(
                   agentId, makeHandshakeUpdate(port, rampCompleted)) &&
               _masterStateR->reportAgent(agentId, makeCachePolicyUpdate(256)) &&
               _masterStateR->reportAgent(agentId,
                                          makeEmptyPendingCacheSnapshot());
    }

    void buildCluster(bool startScheduleGraph = true)
    {
        const std::string loader = "./aios/navi_sched/testdata/test_config_loader.py";
        const std::string configPath = "./aios/navi_sched/testdata/config";
        const std::string clientBizConfig = configPath + "/client_biz.py";
        const std::string masterBizConfig = configPath + "/master_biz.py";
        const std::string agentBizConfig = configPath + "/agent_biz.py";

        auto clientResources = std::make_shared<navi::ResourceMap>();
        auto masterResources = std::make_shared<navi::ResourceMap>();
        auto agent0Resources = std::make_shared<navi::ResourceMap>();
        ASSERT_TRUE(clientResources->addResource(_clientResponseR));
        ASSERT_TRUE(agent0Resources->addResource(_agent0ReportBridgeR));
        ASSERT_TRUE(agent0Resources->addResource(_agent0RequestR));
        ASSERT_TRUE(agent0Resources->addResource(_agent0BatchR));
        ASSERT_TRUE(_cluster.addServer("host_0", loader, configPath, clientResources));
        ASSERT_TRUE(_cluster.addServer("host_1", loader, configPath, masterResources));
        ASSERT_TRUE(_cluster.addServer("host_2", loader, configPath, agent0Resources));
        ASSERT_TRUE(_cluster.addBiz("host_0", "client_biz", 1, 0, clientBizConfig));
        ASSERT_TRUE(_cluster.addBiz("host_1", "master_biz", 1, 0, masterBizConfig));
        ASSERT_TRUE(_cluster.addBiz("host_2", "agent_biz", 1, 0, agentBizConfig));
        ASSERT_TRUE(_cluster.start());
        _agent0 = loadAgentReportR("host_2", "agent_biz");
        ASSERT_TRUE(_agent0);
        ASSERT_NO_FATAL_FAILURE(
            loadMasterResources("host_1", startScheduleGraph));
    }

    void buildPartitionCluster()
    {
        const std::string loader = "./aios/navi_sched/testdata/test_config_loader.py";
        const std::string configPath = "./aios/navi_sched/testdata/config";
        const std::string masterBizConfig = configPath + "/master_biz.py";
        const std::string agentBizConfig = configPath + "/agent_biz.py";
        auto masterResources = std::make_shared<navi::ResourceMap>();
        auto agent0Resources = std::make_shared<navi::ResourceMap>();
        auto agent1Resources = std::make_shared<navi::ResourceMap>();
        ASSERT_TRUE(agent0Resources->addResource(_agent0ReportBridgeR));
        ASSERT_TRUE(agent0Resources->addResource(_agent0RequestR));
        ASSERT_TRUE(agent0Resources->addResource(_agent0BatchR));
        ASSERT_TRUE(agent1Resources->addResource(_agent1ReportBridgeR));
        ASSERT_TRUE(agent1Resources->addResource(_agent1RequestR));
        ASSERT_TRUE(agent1Resources->addResource(_agent1BatchR));
        ASSERT_TRUE(_cluster.addServer("host_0", loader, configPath, masterResources));
        ASSERT_TRUE(_cluster.addServer("host_1", loader, configPath, agent0Resources));
        ASSERT_TRUE(_cluster.addServer("host_2", loader, configPath, agent1Resources));
        ASSERT_TRUE(_cluster.addBiz("host_0", "master_biz", 1, 0, masterBizConfig));
        ASSERT_TRUE(_cluster.addBiz("host_1", "prefill_biz_0", 1, 0, agentBizConfig));
        ASSERT_TRUE(_cluster.addBiz("host_2", "prefill_biz_1", 1, 0, agentBizConfig));
        ASSERT_TRUE(_cluster.start());
        _agent0 = loadAgentReportR("host_1", "prefill_biz_0");
        _agent1 = loadAgentReportR("host_2", "prefill_biz_1");
        ASSERT_TRUE(_agent0);
        ASSERT_TRUE(_agent1);
        ASSERT_NO_FATAL_FAILURE(loadMasterResources("host_0"));
    }

protected:
    std::shared_ptr<MasterPendingCacheR> _masterPendingCacheR;
    std::shared_ptr<MasterStateR> _masterStateR;
    std::shared_ptr<MasterScheduleR> _masterScheduleR;
    std::shared_ptr<SchedulerConfigR> _schedulerConfigR;
    MasterLatencyModelRPtr _masterLatencyModelR;
    AgentReportRPtr _agent0;
    AgentReportRPtr _agent1;
    AgentReportBridgeRPtr _agent0ReportBridgeR = std::make_shared<AgentReportBridgeR>();
    AgentReportBridgeRPtr _agent1ReportBridgeR = std::make_shared<AgentReportBridgeR>();
    std::shared_ptr<AgentRequestR> _agent0RequestR = std::make_shared<AgentRequestR>();
    std::shared_ptr<AgentRequestR> _agent1RequestR = std::make_shared<AgentRequestR>();
    std::shared_ptr<AgentBatchR> _agent0BatchR = std::make_shared<AgentBatchR>();
    std::shared_ptr<AgentBatchR> _agent1BatchR = std::make_shared<AgentBatchR>();
    std::shared_ptr<ClientResponseR> _clientResponseR =
        std::make_shared<ClientResponseR>();
    navi::NaviTestCluster _cluster;
    std::shared_ptr<MasterScheduleGraph> _masterScheduleGraph;
    size_t _maxWaitScheduleCount = 30;
    int64_t _maxWaitScheduleTimeMs = 10;
    std::string _schedulerKernelId = RandomSchedulerKernel::KERNEL_ID;
    std::string _schedulerKernelConfig = "{}";
    std::unique_ptr<autil::EnvGuard> _pythonHome;
};

TEST_F(NaviSchedClusterTest, testMasterScheduleGraphDoesNotRetainLifetimeTrace)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    navi::NaviLoggerPtr logger;
    {
        autil::ScopedReadLock lock(_masterScheduleR->_lock);
        ASSERT_TRUE(_masterScheduleR->_requestPipe);
        logger = _masterScheduleR->_requestPipe->_node->getLogger().logger;
    }
    ASSERT_TRUE(logger);
    EXPECT_EQ(navi::LOG_LEVEL_DISABLE, logger->getTraceLevel());
    navi::TraceCollector initial;
    logger->collectTrace(initial);
    for (size_t index = 0; index < 128; ++index) {
        NAVI_LOG_INTERNAL(logger, INFO, "Master lifetime trace regression %lu", index);
    }
    navi::TraceCollector retained;
    logger->collectTrace(retained);
    EXPECT_EQ(0, retained.eventCount());
    EXPECT_TRUE(_masterScheduleGraph->running());
}

TEST_F(NaviSchedClusterTest, testMasterCollectorDoesNotRetainLifetimeTrace)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    MasterCollectorManager manager(_cluster.getNavi("host_1"), _masterStateR,
                                   "master_biz", true);
    ASSERT_TRUE(manager.open("prefill_0", "agent_biz"));
    ASSERT_EQ(1, manager._collectors.size());
    auto result = manager._collectors.begin()->second.userResult->getGraphResult();
    ASSERT_TRUE(result);
    auto logger = result->_logger;
    ASSERT_TRUE(logger);
    EXPECT_EQ(navi::LOG_LEVEL_DISABLE, logger->getTraceLevel());
    navi::TraceCollector initial;
    logger->collectTrace(initial);
    for (size_t index = 0; index < 128; ++index) {
        NAVI_LOG_INTERNAL(logger, INFO, "Collector lifetime trace regression %lu", index);
    }
    navi::TraceCollector retained;
    logger->collectTrace(retained);
    EXPECT_EQ(0, retained.eventCount());
    EXPECT_EQ(1, manager.size());
}

TEST_F(NaviSchedClusterTest, testCollectorQueuesPreserveRemoteOrderReconnectAndDrain)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster(false));
    auto masterNavi = _cluster.getNavi("host_1");
    auto realMaster = masterNavi->getRealNavi();
    ASSERT_TRUE(realMaster);
    std::string configJson;
    ASSERT_TRUE(navi::NaviConfig::prettyDumpToStr(*realMaster->getSnapshot()->_config, configJson));
    navi::NaviConfig config;
    ASSERT_TRUE(config.loadConfig(configJson));
    config.engineConfig.extraTaskQueueMap.emplace("nsc0", navi::ConcurrencyConfig(2, 1000, 1000));
    config.engineConfig.extraTaskQueueMap.emplace("nsc1", navi::ConcurrencyConfig(2, 1000, 1000));
    ASSERT_TRUE(masterNavi->update(config, navi::ResourceMap()));
    ASSERT_NO_FATAL_FAILURE(loadMasterResources("host_1", false));
    const std::vector<std::string> queues{"nsc0", "nsc1"};
    const auto expectedQueue = queues[hashString("prefill_0") % queues.size()];
    MasterCollectorManager manager(masterNavi, _masterStateR, "master_biz", false, queues);
    const auto nodeHash = hashString("prefill_0");
    for (size_t round = 0; round < 2; ++round) {
        ASSERT_TRUE(manager.open("prefill_0", "agent_biz"));
        for (size_t retry = 0; retry < 200 && _agent0->getSubscriberCount() != 1; ++retry) {
            usleep(10000);
        }
        ASSERT_EQ(1, _agent0->getSubscriberCount());
        ASSERT_EQ(1, manager._collectors.size());
        const auto result = manager._collectors.begin()->second.userResult->getGraphResult();
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->_graph);
        const auto *param = result->_graph->getGraphParam();
        EXPECT_EQ(expectedQueue, param->runParams.getTaskQueueName());
        EXPECT_EQ(realMaster->getSnapshot()->getTaskQueue(expectedQueue)->getThreadPool(),
                  param->worker->getThreadPool());
        {
            autil::ScopedLock lock(_agent0->_mutex);
            auto pipe = _agent0->_subscribers.begin()->second->asyncPipe.lock();
            ASSERT_TRUE(pipe);
            const auto *remoteParam = pipe->_node->_graph->_param;
            EXPECT_EQ(expectedQueue, remoteParam->runParams.getTaskQueueName());
            // Agents need no new queues: unknown remote names use their builtin pool.
            auto agentSnapshot = _cluster.getNavi("host_2")->getRealNavi()->getSnapshot();
            ASSERT_TRUE(agentSnapshot->_extraTaskQueueMap.empty());
            EXPECT_EQ(agentSnapshot->getTaskQueue("")->getThreadPool(),
                      remoteParam->worker->getThreadPool());
        }
        for (size_t index = 1; index <= 128; ++index) {
            NaviSchedCacheUpdate update;
            update.totalBlocks = 1000;
            update.addedBlockHashes = makeBlockHashes({round * 128 + index});
            ASSERT_EQ(1, _agent0->reportCacheUpdated(update));
        }
        for (size_t retry = 0; retry < 200; ++retry) {
            auto node = _masterStateR->getNodeState(nodeHash);
            if (node && node->getCacheSyncStats().masterCachedBlockCount == (round + 1) * 128) {
                break;
            }
            usleep(10000);
        }
        auto node = _masterStateR->getNodeState(nodeHash);
        ASSERT_TRUE(node);
        const auto stats = node->getCacheSyncStats();
        EXPECT_EQ((round + 1) * 128, stats.masterCachedBlockCount);
        EXPECT_EQ(round == 0 ? 128 : 256, stats.masterHashChecksum);
        EXPECT_EQ(0, stats.revisionGapCount);
        EXPECT_EQ(0, stats.stateMismatchCount);
        ASSERT_TRUE(manager.close("prefill_0"));
        for (size_t retry = 0; retry < 200 && _agent0->getSubscriberCount() != 0; ++retry) {
            usleep(10000);
        }
        EXPECT_EQ(0, _agent0->getSubscriberCount());
        EXPECT_EQ(0, manager.size());
    }
    for (size_t index = 0; index < 8; ++index) {
        ASSERT_TRUE(manager.open("concurrent_" + std::to_string(index), "agent_biz"));
    }
    for (size_t retry = 0; retry < 200 && _agent0->getSubscriberCount() != 8; ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(8, _agent0->getSubscriberCount());
    std::set<std::string> usedQueues;
    for (const auto &[id, collector] : manager._collectors) {
        const auto result = collector.userResult->getGraphResult();
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->_graph);
        const auto queue = result->_graph->getGraphParam()->runParams.getTaskQueueName();
        EXPECT_EQ(queues[hashString(id) % queues.size()], queue);
        usedQueues.emplace(queue);
    }
    EXPECT_EQ(2, usedQueues.size());
    for (size_t index = 257; index <= 384; ++index) {
        NaviSchedCacheUpdate update;
        update.totalBlocks = 1000;
        update.addedBlockHashes = makeBlockHashes({index});
        ASSERT_EQ(8, _agent0->reportCacheUpdated(update));
    }
    for (size_t index = 0; index < 8; ++index) {
        const auto hash = hashString("concurrent_" + std::to_string(index));
        for (size_t retry = 0; retry < 200; ++retry) {
            auto node = _masterStateR->getNodeState(hash);
            if (node && node->getCacheSyncStats().masterCachedBlockCount == 384) {
                break;
            }
            usleep(10000);
        }
        auto node = _masterStateR->getNodeState(hash);
        ASSERT_TRUE(node);
        const auto stats = node->getCacheSyncStats();
        EXPECT_EQ(384, stats.masterCachedBlockCount);
        EXPECT_EQ(384, stats.masterHashChecksum);
        EXPECT_EQ(0, stats.revisionGapCount);
        EXPECT_EQ(0, stats.stateMismatchCount);
        EXPECT_EQ(0, _masterPendingCacheR->getStats(hash).agentBatchCount);
    }
    manager.stop();
    for (size_t retry = 0; retry < 200 && _agent0->getSubscriberCount() != 0; ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _agent0->getSubscriberCount());
    EXPECT_EQ(0, manager.size());
    EXPECT_EQ(0, _masterPendingCacheR->getStats(nodeHash).agentBatchCount);
}

TEST_F(NaviSchedClusterTest, testClientMasterAndAgentCollector)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(clientNavi);
    ASSERT_TRUE(masterNavi);

    MasterCollectorManager collectorManager(masterNavi, _masterStateR, "master_biz",
                                            true);
    ASSERT_TRUE(collectorManager.open("prefill_0", "agent_biz"));
    for (size_t retry = 0; retry < 100 && 0 == _agent0->getSubscriberCount(); ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(1, _agent0->getSubscriberCount());
    std::map<std::string, NaviSchedNodeInfo> agentNodes;
    for (size_t retry = 0; retry < 100; ++retry) {
        agentNodes = _masterStateR->getNodes();
        if (1 == agentNodes.size() &&
            "10.0.0.1:18000" == agentNodes.begin()->second.serverAddress) {
            break;
        }
        usleep(10000);
    }
    ASSERT_EQ(1, agentNodes.size());
    ASSERT_EQ("10.0.0.1:18000", agentNodes.begin()->second.serverAddress);

    auto invalidModel = std::make_shared<NaviSchedLatencyModelSnapshot>();
    invalidModel->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    invalidModel->revision = 1;
    invalidModel->parameters = {1.0};
    ASSERT_EQ(0, _agent0->reportLatencyModelUpdated(invalidModel));

    auto request = makeClientRequest(1, "agent_biz", 4, nullptr);
    ClientGraphRunner clientRunner(clientNavi, true);
    auto result = clientRunner.schedule(request, "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_EQ(1, result.decision->requestId);
    ASSERT_EQ("random_selected", result.decision->reason);
    ASSERT_EQ(1, _masterScheduleR->getSubmitCount());
    ASSERT_EQ(1, _masterStateR->getClientRequestCount());

    auto hashRequest =
        makeClientRequest(2, "agent_biz", 512, makeBlockHashes({101, 202}));
    auto hashResult = clientRunner.schedule(hashRequest, "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, hashResult.errorCode) << hashResult.errorMessage;
    ASSERT_EQ(2, _masterScheduleR->getSubmitCount());

    _agent0->reportRequestReceived(1, 4, 1, 1, 0, 0, 100);
    for (size_t retry = 0; retry < 100 && _masterStateR->getAgentReportCount() < 4;
         ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(4, _masterStateR->getAgentReportCount());
    ASSERT_EQ(1, _masterStateR->getAgentRequestStats().receivedCount);
    auto nodeHash = hashString("prefill_0");
    _agent0->reportBatchFinished(0, 0, 0, 0, 8.0, 0.0, 0.0,
                                 NaviSchedCacheUpdate(), 1, 0, 0, 0, 0, false, 120);
    NaviSchedCacheUpdate cacheUpdate;
    cacheUpdate.totalBlocks = 100;
    cacheUpdate.addedBlockHashes = makeBlockHashes({11, 22});
    cacheUpdate.eventTimeUs = 130;
    _agent0->reportCacheUpdated(cacheUpdate);
    ASSERT_EQ(1, _agent0->reportExecutionLoad(3, 5, 2, 27.5, 10.0, 9, 2, 135));
    for (size_t retry = 0; retry < 100; ++retry) {
        auto node = _masterStateR->getNodeState(nodeHash);
        auto info = node ? node->getInfo() : NaviSchedNodeInfo();
        if (node && 8.0 == info.lastRequestLatencyMs &&
            27.5 == info.queueTotalPredictedLatencyMs) {
            break;
        }
        usleep(10000);
    }
    auto activeNode = _masterStateR->getNodeState(nodeHash);
    ASSERT_TRUE(activeNode);
    auto activeNodeInfo = activeNode->getInfo();
    ASSERT_DOUBLE_EQ(8.0, activeNodeInfo.lastRequestLatencyMs);
    ASSERT_EQ(3, activeNodeInfo.batchWaitingRequests);
    ASSERT_EQ(5, activeNodeInfo.queuedBatchRequests);
    ASSERT_EQ(2, activeNodeInfo.runningRequests);
    ASSERT_EQ(9, activeNodeInfo.batchTimeoutCount);
    ASSERT_EQ(2, activeNodeInfo.abnormalBatchFinishCount);
    ASSERT_DOUBLE_EQ(27.5, activeNodeInfo.queueTotalPredictedLatencyMs);
    ASSERT_EQ(1, _agent0->beginStop(140));
    _agent0->finishAll();

    for (size_t retry = 0;
         retry < 100 && (_masterStateR->getAgentReportCount() < 8 ||
                         !_masterStateR->getNodes().empty());
         ++retry) {
        usleep(10000);
    }
    auto nodes = _masterStateR->getNodes();
    ASSERT_EQ(8, _masterStateR->getAgentReportCount());
    ASSERT_TRUE(nodes.empty());
    for (size_t retry = 0; retry < 100 && 0 != collectorManager.size(); ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(0, collectorManager.size());
    ASSERT_TRUE(collectorManager.close("prefill_0"));
    _masterScheduleGraph->stop();
    std::ifstream clientVis("./client_graph.vis", std::ios::binary);
    std::ifstream masterVis("./master_graph.vis", std::ios::binary);
    std::ifstream scheduleVis("./master_schedule_graph.vis", std::ios::binary);
    ASSERT_TRUE(clientVis.is_open());
    ASSERT_TRUE(masterVis.is_open());
    ASSERT_TRUE(scheduleVis.is_open());
    navi::GraphVisDef clientVisProto;
    navi::GraphVisDef masterVisProto;
    navi::GraphVisDef scheduleVisProto;
    ASSERT_TRUE(clientVisProto.ParseFromIstream(&clientVis));
    ASSERT_TRUE(masterVisProto.ParseFromIstream(&masterVis));
    ASSERT_TRUE(scheduleVisProto.ParseFromIstream(&scheduleVis));
    ASSERT_TRUE(containsKernel(clientVisProto, "navi_sched.schedule_k"));
    ASSERT_FALSE(containsKernel(clientVisProto, navi::PLACEHOLDER_KERNEL_NAME));
    ASSERT_TRUE(containsKernel(masterVisProto, "navi_sched.master_update_k"));
    ASSERT_TRUE(containsRemoteKernelMetric(masterVisProto, "agent_biz",
                                           "navi_sched.agent_report_source_k"));
    ASSERT_GE(countSummarizedDataEvents(masterVisProto, AGENT_UPDATE_TO_MASTER_TYPE_ID),
              6);
    auto masterVisString = masterVisProto.SerializeAsString();
    ASSERT_NE(std::string::npos, masterVisString.find("batch_finished"));
    ASSERT_NE(std::string::npos, masterVisString.find("AgentCacheToMasterData"));
    ASSERT_NE(std::string::npos, masterVisString.find("agent_stopping"));
    ASSERT_TRUE(containsKernel(scheduleVisProto, "navi_sched.master_schedule_context_k"));
    ASSERT_TRUE(containsKernel(scheduleVisProto, RandomSchedulerKernel::KERNEL_ID));
    ASSERT_TRUE(containsKernel(scheduleVisProto, "navi_sched.master_schedule_result_k"));
    ASSERT_FALSE(containsGraphError(scheduleVisProto));
}

TEST_F(NaviSchedClusterTest,
       testSchedulerConfigSyncsCurrentAgentAndCatchesUpNewAgent)
{
    ASSERT_NO_FATAL_FAILURE(buildPartitionCluster());
    auto masterNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(masterNavi);
    auto agent0Config = loadAgentSchedulerConfigR("host_1", "prefill_biz_0");
    auto agent1Config = loadAgentSchedulerConfigR("host_2", "prefill_biz_1");
    ASSERT_TRUE(agent0Config);
    ASSERT_TRUE(agent1Config);
    ASSERT_DOUBLE_EQ(0.0, agent0Config->get().config.residualLearningRate);
    ASSERT_DOUBLE_EQ(0.0, agent1Config->get().config.residualLearningRate);

    MasterCollectorManager collectorManager(masterNavi, _masterStateR,
                                            "master_biz", true);
    ASSERT_TRUE(collectorManager.open("agent_0", "prefill_biz_0"));
    for (size_t retry = 0;
         retry < 200 && 0 == _agent0->getSubscriberCount(); ++retry) {
        usleep(10 * 1000);
    }
    ASSERT_EQ(1, _agent0->getSubscriberCount());
    for (size_t retry = 0;
         retry < 200 && 1 != _masterStateR->getNodes().size(); ++retry) {
        usleep(10 * 1000);
    }
    ASSERT_EQ(1, _masterStateR->getNodes().size());
    const auto acknowledgedConfig = _masterStateR->getNodes().begin()->second;
    ASSERT_GT(acknowledgedConfig.schedulerConfigGeneration, 0);
    ASSERT_FALSE(acknowledgedConfig.schedulerConfigHash.empty());
    ASSERT_FALSE(acknowledgedConfig.schedulerConfigJson.empty());

    auto responsePipe = std::make_shared<SchedulerConfigResponsePipe>();
    SchedulerConfigR::Request request;
    request.update = true;
    request.configJson = R"({"residual_learning_rate":0.017})";
    request.responsePipe = responsePipe;
    ASSERT_TRUE(_schedulerConfigR->submit(std::move(request)));
    auto response = responsePipe->waitResponse(2 * 1000 * 1000);
    ASSERT_TRUE(response);
    ASSERT_TRUE(response->accepted) << response->error;
    EXPECT_EQ(1, response->targetAgentCount);
    EXPECT_EQ(1, response->updatedAgentCount);
    EXPECT_EQ(0, response->failedAgentCount);
    EXPECT_DOUBLE_EQ(0.017,
                     agent0Config->get().config.residualLearningRate);
    EXPECT_DOUBLE_EQ(0.0,
                     agent1Config->get().config.residualLearningRate);

    ASSERT_TRUE(collectorManager.open("agent_1", "prefill_biz_1"));
    for (size_t retry = 0; retry < 200; ++retry) {
        if (_agent1->getSubscriberCount() > 0 &&
            2 == _masterStateR->getNodes().size() &&
            0.017 == agent1Config->get().config.residualLearningRate) {
            break;
        }
        usleep(10 * 1000);
    }
    EXPECT_EQ(1, _agent1->getSubscriberCount());
    EXPECT_DOUBLE_EQ(0.017,
                     agent1Config->get().config.residualLearningRate);
    EXPECT_FALSE(_schedulerConfigR->hasActiveAgentSync());

    responsePipe = std::make_shared<SchedulerConfigResponsePipe>();
    SchedulerConfigR::Request disable;
    disable.update = true;
    disable.configJson = R"({"residual_learning_rate":0.0})";
    disable.responsePipe = responsePipe;
    ASSERT_TRUE(_schedulerConfigR->submit(std::move(disable)));
    response = responsePipe->waitResponse(2 * 1000 * 1000);
    ASSERT_TRUE(response);
    ASSERT_TRUE(response->accepted) << response->error;
    EXPECT_EQ(2, response->targetAgentCount);
    EXPECT_EQ(2, response->updatedAgentCount);
    EXPECT_EQ(0, response->failedAgentCount);
    EXPECT_DOUBLE_EQ(0.0, agent0Config->get().config.residualLearningRate);
    EXPECT_DOUBLE_EQ(0.0, agent1Config->get().config.residualLearningRate);
    EXPECT_FALSE(_schedulerConfigR->hasActiveAgentSync());
}

TEST_F(NaviSchedClusterTest,
       testFullyRateLimitedBacklogDoesNotAbortMasterScheduleGraph)
{
    _maxWaitScheduleCount = 1;
    _maxWaitScheduleTimeMs = 10;
    ASSERT_NO_FATAL_FAILURE(buildCluster(false));

    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000, true));
    auto latency = std::make_shared<AgentUpdateToMasterData>();
    latency->event = NaviSchedAgentEvent::LATENCY_MODEL_UPDATED;
    attachTrustedLatencyModel(
        latency,
        {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 0.0, 1.4, 1.0, 0.0});
    ASSERT_TRUE(_masterStateR->reportAgent("agent_0", latency));
    auto load = std::make_shared<AgentUpdateToMasterData>();
    load->event = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED;
    load->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    load->requestEvent->inflightStatsRevision = 2;
    load->requestEvent->inflightRequests = 1;
    load->requestEvent->inflightPredictedLatencySumMs = 1000.0;
    load->requestEvent->batchWaitingRequests = 1;
    load->requestEvent->queueTotalPredictedLatencyMs = 1000.0;
    load->requestEvent->queueDecayHeadRemainingMs = 1000.0;
    ASSERT_TRUE(_masterStateR->reportAgent("agent_0", load));

    ASSERT_NO_FATAL_FAILURE(startMasterScheduleGraph("host_1"));
    constexpr size_t requestCount = 4;
    std::vector<std::shared_ptr<ScheduleResponsePipe>> responsePipes;
    for (size_t index = 0; index < requestCount; ++index) {
        auto responsePipe = std::make_shared<ScheduleResponsePipe>();
        responsePipes.emplace_back(responsePipe);
        ASSERT_TRUE(_masterScheduleR->submit(
            makeClientRequest(1000 + index, "agent_biz", 4, nullptr),
            responsePipe));
    }
    uint64_t previousCanonicalRequestId = 0;
    for (size_t index = 0; index < requestCount; ++index) {
        auto response = responsePipes[index]->waitResponse(500 * 1000);
        ASSERT_TRUE(response) << "missing response for request " << index;
        EXPECT_EQ("rate_limited", response->reason);
        EXPECT_NE(0, response->requestId);
        EXPECT_NE(1000 + index, response->requestId);
        if (index > 0) {
            EXPECT_EQ(previousCanonicalRequestId + 1, response->requestId);
        }
        previousCanonicalRequestId = response->requestId;
    }
    EXPECT_TRUE(_masterScheduleGraph->running());
    EXPECT_EQ(0, _masterScheduleGraph->getRestartCount());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphRestartsAfterDestructAndReplacesPipe)
{
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    FailNMasterSchedulerKernel::reset();
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    auto failedResponsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1100, "agent_biz", 4, nullptr), failedResponsePipe));
    ASSERT_TRUE(failedResponsePipe->waitEof(500 * 1000));
    EXPECT_FALSE(failedResponsePipe->waitResponse());

    for (size_t retry = 0;
         retry < 100 &&
         (_masterScheduleGraph->getRestartCount() < 1 ||
          _masterScheduleR->getRequestPipeGeneration() < 2 ||
          !_masterScheduleGraph->running());
         ++retry) {
        usleep(10 * 1000);
    }
    ASSERT_EQ(1, _masterScheduleGraph->getRestartCount());
    ASSERT_EQ(2, _masterScheduleR->getRequestPipeGeneration());
    ASSERT_TRUE(_masterScheduleGraph->running());
    ASSERT_EQ("running", _masterScheduleGraph->getStateName());

    bool submittedToReplacementPipe = false;
    std::shared_ptr<ScheduleResponsePipe> replacementResponsePipe;
    for (size_t retry = 0; retry < 100 && !submittedToReplacementPipe; ++retry) {
        replacementResponsePipe = std::make_shared<ScheduleResponsePipe>();
        submittedToReplacementPipe = bool(_masterScheduleR->submit(
            makeClientRequest(1200 + retry, "agent_biz", 4, nullptr),
            replacementResponsePipe));
        if (!submittedToReplacementPipe) {
            usleep(10 * 1000);
        }
    }
    ASSERT_TRUE(submittedToReplacementPipe);
    auto response = replacementResponsePipe->waitResponse(500 * 1000);
    ASSERT_TRUE(response);
    EXPECT_EQ("restarted_selected", response->reason);
    EXPECT_TRUE(_masterScheduleGraph->running());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphRejectsRequestDuringRestartGap)
{
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    FailNMasterSchedulerKernel::reset();
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    auto failedResponsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1300, "agent_biz", 4, nullptr), failedResponsePipe));
    ASSERT_TRUE(failedResponsePipe->waitEof(500 * 1000));

    for (size_t retry = 0;
         retry < 100 && _masterScheduleGraph->running(); ++retry) {
        usleep(1000);
    }
    ASSERT_FALSE(_masterScheduleGraph->running());
    ASSERT_EQ("restarting", _masterScheduleGraph->getStateName());
    ASSERT_EQ(1, _masterScheduleR->getRequestPipeGeneration());

    auto gapResponsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_FALSE(_masterScheduleR->submit(
        makeClientRequest(1301, "agent_biz", 4, nullptr), gapResponsePipe));
    auto rejected = gapResponsePipe->waitResponse(100 * 1000);
    ASSERT_TRUE(rejected);
    EXPECT_EQ(1301, rejected->requestId);
    EXPECT_EQ("master_restarting", rejected->reason);
    EXPECT_EQ(0, _masterScheduleR->getPendingRequestCount());

    for (size_t retry = 0;
         retry < 100 && !_masterScheduleGraph->running(); ++retry) {
        usleep(10 * 1000);
    }
    EXPECT_EQ(1, _masterScheduleGraph->getRestartCount());
    EXPECT_EQ(2, _masterScheduleR->getRequestPipeGeneration());
    EXPECT_TRUE(_masterScheduleGraph->running());
    EXPECT_EQ("running", _masterScheduleGraph->getStateName());

    auto recoveredResponsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1302, "agent_biz", 4, nullptr),
        recoveredResponsePipe));
    auto recovered = recoveredResponsePipe->waitResponse(500 * 1000);
    ASSERT_TRUE(recovered);
    EXPECT_NE(0, recovered->requestId);
    EXPECT_NE(1302, recovered->requestId);
    EXPECT_EQ("restarted_selected", recovered->reason);
}

TEST_F(NaviSchedClusterTest,
       testClientGraphRejectsMasterScheduleRestartGap)
{
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    FailNMasterSchedulerKernel::reset();
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    auto failedResponsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1350, "agent_biz", 4, nullptr), failedResponsePipe));
    ASSERT_TRUE(failedResponsePipe->waitEof(500 * 1000));
    for (size_t retry = 0;
         retry < 100 && _masterScheduleGraph->running(); ++retry) {
        usleep(1000);
    }
    ASSERT_EQ("restarting", _masterScheduleGraph->getStateName());

    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    ClientGraphRunner client(clientNavi, true);
    auto result = client.schedule(
        makeClientRequest(1351, "agent_biz", 4, nullptr), "master_biz", 2000);

    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_EQ(1351, result.decision->requestId);
    EXPECT_EQ("master_restarting", result.decision->reason);
    EXPECT_EQ(0, _masterScheduleR->getPendingRequestCount());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphUsesFreshPipeForEveryRestart)
{
    constexpr size_t failureCount = 3;
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    FailNMasterSchedulerKernel::reset(failureCount);
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    for (size_t failureIndex = 0; failureIndex < failureCount; ++failureIndex) {
        auto responsePipe = std::make_shared<ScheduleResponsePipe>();
        ASSERT_TRUE(_masterScheduleR->submit(
            makeClientRequest(1400 + failureIndex, "agent_biz", 4, nullptr),
            responsePipe));
        ASSERT_TRUE(responsePipe->waitEof(500 * 1000));
        EXPECT_FALSE(responsePipe->waitResponse());

        const auto expectedGeneration = failureIndex + 2;
        for (size_t retry = 0;
             retry < 100 &&
             (_masterScheduleGraph->getRestartCount() < failureIndex + 1 ||
              _masterScheduleR->getRequestPipeGeneration() < expectedGeneration ||
              !_masterScheduleGraph->running());
             ++retry) {
            usleep(10 * 1000);
        }
        ASSERT_EQ(failureIndex + 1,
                  _masterScheduleGraph->getRestartCount());
        ASSERT_EQ(expectedGeneration,
                  _masterScheduleR->getRequestPipeGeneration());
        ASSERT_TRUE(_masterScheduleGraph->running());
    }

    auto responsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1500, "agent_biz", 4, nullptr), responsePipe));
    auto response = responsePipe->waitResponse(500 * 1000);
    ASSERT_TRUE(response);
    EXPECT_EQ("restarted_selected", response->reason);
    EXPECT_EQ(failureCount, _masterScheduleGraph->getRestartCount());
    EXPECT_EQ(failureCount + 1,
              _masterScheduleR->getRequestPipeGeneration());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphStopsWithoutRestartingDuringBackoff)
{
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    FailNMasterSchedulerKernel::reset();
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:18000", 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    auto responsePipe = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(
        makeClientRequest(1600, "agent_biz", 4, nullptr), responsePipe));
    ASSERT_TRUE(responsePipe->waitEof(500 * 1000));
    for (size_t retry = 0;
         retry < 100 && _masterScheduleGraph->running(); ++retry) {
        usleep(1000);
    }
    ASSERT_FALSE(_masterScheduleGraph->running());

    _masterScheduleGraph->stop();
    const auto restartCount = _masterScheduleGraph->getRestartCount();
    const auto pipeGeneration = _masterScheduleR->getRequestPipeGeneration();
    usleep(150 * 1000);
    EXPECT_FALSE(_masterScheduleGraph->running());
    EXPECT_EQ("stopped", _masterScheduleGraph->getStateName());
    EXPECT_EQ(restartCount, _masterScheduleGraph->getRestartCount());
    EXPECT_EQ(pipeGeneration, _masterScheduleR->getRequestPipeGeneration());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphBacksOffBetweenConsecutiveInitFailures)
{
    _schedulerKernelId = AlwaysFailInitMasterSchedulerKernel::KERNEL_ID;
    AlwaysFailInitMasterSchedulerKernel::reset();
    const auto startTimeUs = autil::TimeUtility::monotonicTimeUs();
    ASSERT_NO_FATAL_FAILURE(buildCluster());

    for (size_t retry = 0;
         retry < 100 && AlwaysFailInitMasterSchedulerKernel::getInitCount() < 3;
         ++retry) {
        usleep(10 * 1000);
    }
    const auto elapsedUs =
        autil::TimeUtility::monotonicTimeUs() - startTimeUs;
    ASSERT_GE(AlwaysFailInitMasterSchedulerKernel::getInitCount(), 3);
    EXPECT_GE(elapsedUs, 150 * 1000);
    EXPECT_LE(AlwaysFailInitMasterSchedulerKernel::getInitCount(), 4);
    EXPECT_GE(_masterScheduleGraph->getRestartCount(), 2);

    _masterScheduleGraph->stop();
    const auto initCount = AlwaysFailInitMasterSchedulerKernel::getInitCount();
    usleep(150 * 1000);
    EXPECT_EQ(initCount, AlwaysFailInitMasterSchedulerKernel::getInitCount());
    EXPECT_FALSE(_masterScheduleGraph->running());
    EXPECT_EQ("stopped", _masterScheduleGraph->getStateName());
}

TEST_F(NaviSchedClusterTest,
       testMasterScheduleGraphRecoversAfterInitialInitFailures)
{
    _schedulerKernelId = AlwaysFailInitMasterSchedulerKernel::KERNEL_ID;
    AlwaysFailInitMasterSchedulerKernel::reset(2);
    ASSERT_NO_FATAL_FAILURE(buildCluster());

    for (size_t retry = 0;
         retry < 100 && !_masterScheduleGraph->running(); ++retry) {
        usleep(10 * 1000);
    }
    EXPECT_TRUE(_masterScheduleGraph->running());
    EXPECT_EQ("running", _masterScheduleGraph->getStateName());
    EXPECT_EQ(3, AlwaysFailInitMasterSchedulerKernel::getInitCount());
    EXPECT_EQ(2, _masterScheduleGraph->getRestartCount());
    EXPECT_EQ(3, _masterScheduleR->getRequestPipeGeneration());
}

TEST_F(NaviSchedClusterTest, testMasterResultGraphRunsParallelAndSerialRounds)
{
    _maxWaitScheduleCount = 128;
    _maxWaitScheduleTimeMs = 1000;
    _schedulerKernelConfig = R"({"parallelism":4})";
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    for (size_t index = 0; index < 1; ++index) {
        const auto id = "parallel_agent_" + std::to_string(index);
        _masterStateR->registerAgent(id, "agent_biz",
            "127.0.0.1:" + std::to_string(18000 + index), 10000);
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeHandshakeUpdate(18000 + index, true)));
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeCachePolicyUpdate(256)));
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeEmptyPendingCacheSnapshot()));
    }
    uint64_t previousRound = 0;
    for (size_t roundSize : {128, 128, 3}) {
        std::vector<std::shared_ptr<ScheduleResponsePipe>> pipes;
        std::vector<uint64_t> canonicalIds;
        const auto firstId = previousRound * 1000 + 1001;
        for (size_t index = 0; index < roundSize; ++index) {
            auto request = makeClientRequest(firstId + index, "agent_biz", 512,
                                             makeBlockHashes({11, 100 + index}));
            request->timeoutMs = 10000;
            auto pipe = std::make_shared<ScheduleResponsePipe>();
            auto submitted = _masterScheduleR->submit(request, pipe);
            ASSERT_TRUE(submitted);
            canonicalIds.push_back(submitted->request->requestId);
            pipes.push_back(std::move(pipe));
        }
        std::set<std::string> targets;
        uint64_t roundId = 0;
        for (size_t index = 0; index < pipes.size(); ++index) {
            auto response = pipes[index]->waitResponse(10'000'000);
            ASSERT_TRUE(response);
            EXPECT_EQ(canonicalIds[index], response->requestId);
            EXPECT_EQ("random_selected", response->reason);
            EXPECT_TRUE(response->dispatchHandle.valid());
            EXPECT_FALSE(response->replay);
            if (!roundId) roundId = response->scheduleId;
            EXPECT_EQ(roundId, response->scheduleId);
            targets.insert(response->target.agentAddress);
        }
        EXPECT_GT(roundId, previousRound);
        EXPECT_EQ(1, targets.size());
        previousRound = roundId;
    }
    _masterScheduleGraph->stop();
    std::ifstream input("./master_schedule_graph.vis", std::ios::binary);
    navi::GraphVisDef vis;
    ASSERT_TRUE(vis.ParseFromIstream(&input));
    EXPECT_GE(getKernelScheduleCount(vis, MasterScheduleResultPrepareKernel::KERNEL_ID), 3);
    std::map<std::string, int64_t> workerSchedules;
    auto collectWorkers = [&](auto &&self, const navi::GraphVisNodeDef &node) -> void {
        for (const auto &metric : node.result().graph_metric().metrics()) {
            if (metric.kernel() == MasterScheduleResultWorkerKernel::KERNEL_ID) {
                workerSchedules[metric.node()] = metric.schedule_count();
            }
        }
        for (const auto &child : node.sub_results()) self(self, child.second);
    };
    for (const auto &node : vis.nodes()) collectWorkers(collectWorkers, node);
    ASSERT_EQ(12, workerSchedules.size());
    for (const auto &[name, count] : workerSchedules) {
        SCOPED_TRACE(name);
        EXPECT_GE(count, 3);
    }
    EXPECT_FALSE(containsGraphError(vis));
}

TEST_F(NaviSchedClusterTest, testMasterFullBatchesDoNotWaitForScheduleTimer)
{
    _maxWaitScheduleCount = 512;
    _maxWaitScheduleTimeMs = 100;
    _schedulerKernelId = FailNMasterSchedulerKernel::KERNEL_ID;
    _schedulerKernelConfig =
        R"({"queue_overload_start_cycle_multiplier":0,"queue_overload_full_cycle_multiplier":0})";
    FailNMasterSchedulerKernel::reset(0, true);
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    for (size_t index = 0; index < 64; ++index) {
        const auto id = "full_batch_agent_" + std::to_string(index);
        const auto port = 18000 + index;
        _masterStateR->registerAgent(id, "agent_biz", "127.0.0.1:" + std::to_string(port), 0);
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeHandshakeUpdate(port, true)));
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeCachePolicyUpdate(256)));
        ASSERT_TRUE(_masterStateR->reportAgent(id, makeEmptyPendingCacheSnapshot()));
    }

    constexpr size_t ROUND_COUNT = 20;
    std::vector<std::shared_ptr<ScheduleResponsePipe>> pipes;
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    for (size_t index = 0; index < ROUND_COUNT * 512; ++index) {
        auto request = makeClientRequest(1001 + index, "agent_biz", 256,
                                       makeBlockHashes({11}));
        request->timeoutMs = 10000;
        auto pipe = std::make_shared<ScheduleResponsePipe>();
        ASSERT_TRUE(_masterScheduleR->submit(request, pipe));
        pipes.push_back(std::move(pipe));
    }
    std::set<uint64_t> rounds;
    for (const auto &pipe : pipes) {
        auto response = pipe->waitResponse(5'000'000);
        ASSERT_TRUE(response);
        EXPECT_EQ("restarted_selected", response->reason);
        rounds.insert(response->scheduleId);
    }
    const auto elapsedUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
    EXPECT_EQ(ROUND_COUNT, rounds.size());
    // A timer-gated implementation needs 2 seconds for these twenty batches.
    EXPECT_LT(elapsedUs, 1'000'000);
    std::cout << "[master-full-batch-trigger] rounds=" << rounds.size()
              << " requests=" << pipes.size() << " elapsed_us=" << elapsedUs << std::endl;
}

TEST_F(NaviSchedClusterTest, testCostSchedulerRunsParallelGraph)
{
    _maxWaitScheduleCount = 4;
    _maxWaitScheduleTimeMs = 1000;
    _schedulerKernelId = CostSchedulerKernel::KERNEL_ID;
    _schedulerKernelConfig =
        R"({"parallelism":4,"max_loop_count":3,"latency_weight_concavity":0.0,"latency_token_normalization_power":0.25,"initial_weight_seed":20260823})";
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    const std::vector<double> parameters = {
        -4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 0.0, 1.4, 1.0, 0.0,
    };
    for (size_t nodeIndex = 0; nodeIndex < 4; ++nodeIndex) {
        auto agentId = "cost_agent_" + std::to_string(nodeIndex);
        _masterStateR->registerAgent(
            agentId, "agent_biz",
            "127.0.0.1:" + std::to_string(18000 + nodeIndex), 0);
        auto handshake = makeHandshakeUpdate(18000 + nodeIndex, true);
        attachTrustedLatencyModel(handshake, parameters);
        ASSERT_TRUE(_masterStateR->reportAgent(agentId, handshake));
        ASSERT_TRUE(_masterStateR->reportAgent(
            agentId, makeCachePolicyUpdate(256)));
        ASSERT_TRUE(_masterStateR->reportAgent(
            agentId, makeEmptyPendingCacheSnapshot()));
    }
    std::vector<std::shared_ptr<ScheduleResponsePipe>> responsePipes;
    for (size_t requestIndex = 0; requestIndex < 8; ++requestIndex) {
        auto request = makeClientRequest(
            500 + requestIndex, "agent_biz", 1024,
            makeBlockHashes({11, 22, 33, 44}));
        // Leave time for batching instead of triggering timeout-driven dispatch.
        request->timeoutMs = 5000;
        request->collectReplay = true;
        auto responsePipe = std::make_shared<ScheduleResponsePipe>();
        ASSERT_TRUE(_masterScheduleR->submit(request, responsePipe));
        responsePipes.emplace_back(std::move(responsePipe));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    uint64_t scheduleIds[2] = {0, 0};
    double scheduleLosses[2] = {0.0, 0.0};
    bool checkedEffectiveConfig = false;
    for (size_t index = 0; index < responsePipes.size(); ++index) {
        const auto &responsePipe = responsePipes[index];
        auto response = responsePipe->waitResponse();
        ASSERT_TRUE(response);
        EXPECT_EQ("cost_selected", response->reason);
        EXPECT_GT(response->scheduleId, 0);
        EXPECT_GT(response->scheduleLoss, 0.0);
        if (response->replay) {
            CostSchedulerKernelInternal::ReplayInfo replayInfo;
            autil::legacy::FromJson(
                replayInfo,
                autil::legacy::Any(response->replay->schedulerInfo));
            EXPECT_EQ(4, replayInfo.parallelism);
            EXPECT_DOUBLE_EQ(0.0, replayInfo.latencyWeightConcavity);
            EXPECT_DOUBLE_EQ(
                0.25, replayInfo.latencyTokenNormalizationPower);
            EXPECT_EQ(20260823, replayInfo.initialWeightSeed);
            checkedEffectiveConfig = true;
        }
        auto group = index / 4;
        if (0 == scheduleIds[group]) {
            scheduleIds[group] = response->scheduleId;
            scheduleLosses[group] = response->scheduleLoss;
        } else {
            EXPECT_EQ(scheduleIds[group], response->scheduleId);
            EXPECT_DOUBLE_EQ(
                scheduleLosses[group], response->scheduleLoss);
        }
    }
    EXPECT_NE(scheduleIds[0], scheduleIds[1]);
    EXPECT_TRUE(checkedEffectiveConfig);

    _masterScheduleGraph->stop();
    std::ifstream scheduleVis("./master_schedule_graph.vis",
                              std::ios::binary);
    ASSERT_TRUE(scheduleVis.is_open());
    navi::GraphVisDef scheduleVisProto;
    ASSERT_TRUE(scheduleVisProto.ParseFromIstream(&scheduleVis));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto, CostSchedulerPrepareKernel::KERNEL_ID));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto, CostSchedulerRowKernel::KERNEL_ID));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto,
        CostSchedulerGradientBarrierKernel::KERNEL_ID));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto, CostSchedulerProjectionKernel::KERNEL_ID));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto,
        CostSchedulerProjectionBarrierKernel::KERNEL_ID));
    EXPECT_TRUE(containsKernel(
        scheduleVisProto, CostSchedulerLoopKernel::KERNEL_ID));
    EXPECT_GE(getKernelScheduleCount(
        scheduleVisProto, CostSchedulerRowKernel::KERNEL_ID), 3);
    EXPECT_TRUE(containsKernel(
        scheduleVisProto, CostSchedulerFinalizeKernel::KERNEL_ID));
    EXPECT_FALSE(containsKernel(
        scheduleVisProto, CostSchedulerKernel::KERNEL_ID));
    EXPECT_FALSE(containsGraphError(scheduleVisProto));
}

TEST_F(NaviSchedClusterTest, testClientForwardStreamsResponsesOnSingleGraph)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto agentAddress = "127.0.0.1:" + std::to_string(agentGrpcPort);
    _masterStateR->registerAgent("agent_0", "agent_biz", agentAddress, 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ClientGraphRunner masterOnlyClient(clientNavi, false);
    auto masterOnlyResult = masterOnlyClient.schedule(
        makeClientRequest(6, "agent_biz", 4, nullptr), "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, masterOnlyResult.errorCode)
        << masterOnlyResult.errorMessage;
    ASSERT_TRUE(masterOnlyResult.decision);
    ASSERT_TRUE(masterOnlyResult.decision->requestTiming);
    EXPECT_GE(masterOnlyResult.decision->requestTiming->clientTotalLatencyMs, 0.0);
    EXPECT_DOUBLE_EQ(
        0.0, masterOnlyResult.decision->requestTiming->executionLatencyMs);

    std::atomic<uint64_t> canonicalRequestId{0};
    _agent0RequestR->setCompletionHandler([this, &canonicalRequestId](
        const NaviSchedForwardRequest &request,
        const navi::AsyncPipePtr &completionPipe) {
        if (!completionPipe || request.request.requestId == 0 ||
            request.request.requestId != request.decision.requestId) {
            return false;
        }
        canonicalRequestId.store(request.request.requestId,
                                 std::memory_order_relaxed);
        auto received = std::make_shared<AgentUpdateToMasterData>();
        received->event = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED;
        received->requestEvent =
            std::make_shared<AgentRequestEventToMasterData>();
        received->requestEvent->localBatch.scheduleBatchId =
            request.decision.batchId;
        received->requestEvent->localBatch.localBatchId =
            request.decision.batchId;
        received->requestEvent->localBatch.requestCount = 1;
        if (!_masterStateR->reportAgent("agent_0", received)) {
            return false;
        }
        auto ack = std::make_shared<AgentResponseToClientData>();
        ack->requestId = request.request.requestId;
        ack->kind = AgentResponseKind::ACK;
        ack->requestTiming.executionLatencyMs = 12.5;
        auto first = std::make_shared<AgentResponseToClientData>();
        first->requestId = request.request.requestId;
        first->kind = AgentResponseKind::OUTPUT;
        first->frames = {"first"};
        auto last = std::make_shared<AgentResponseToClientData>();
        last->requestId = request.request.requestId;
        last->kind = AgentResponseKind::OUTPUT;
        last->frames = {"last"};
        last->terminal = true;
        return navi::EC_NONE == completionPipe->setData(ack) &&
               navi::EC_NONE == completionPipe->setData(first) &&
               navi::EC_NONE == completionPipe->setData(last) &&
               navi::EC_NONE == completionPipe->setEof();
    });
    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    auto request = makeClientRequest(7, "agent_biz", 4, nullptr);
    request->collectReplay = true;
    constexpr int64_t entryTimeoutMs = 10000;
    auto result = client.schedule(request, "master_biz", entryTimeoutMs);
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_FALSE(result.decision->replay);
    ASSERT_EQ(7, result.decision->requestId);
    EXPECT_NE(0, canonicalRequestId.load(std::memory_order_relaxed));
    EXPECT_NE(7, canonicalRequestId.load(std::memory_order_relaxed));
    ASSERT_TRUE(result.decision->requestTiming);
    ASSERT_DOUBLE_EQ(12.5, result.decision->requestTiming->executionLatencyMs);
    EXPECT_GE(result.decision->requestTiming->clientTotalLatencyMs, 0.0);
    ASSERT_TRUE(result.replay);
    ASSERT_EQ(1, result.replay->requests.size());
    ASSERT_EQ(1, result.replay->decisions.size());
    EXPECT_EQ(result.decision->scheduleId,
              result.replay->decisions.front().scheduleId);
    auto first = _clientResponseR->takeNextResponse(1000);
    ASSERT_TRUE(first);
    EXPECT_EQ(AgentResponseKind::OUTPUT, first->kind);
    EXPECT_EQ(std::vector<std::string>({"first"}), first->frames);
    EXPECT_FALSE(first->terminal);
    auto last = _clientResponseR->takeNextResponse(1000);
    ASSERT_TRUE(last);
    EXPECT_EQ(AgentResponseKind::OUTPUT, last->kind);
    EXPECT_EQ(std::vector<std::string>({"last"}), last->frames);
    EXPECT_TRUE(last->terminal);
    for (size_t retry = 0; retry < 100 && _clientResponseR->requestCount() != 0;
         ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    std::ifstream clientVis("./client_graph.vis", std::ios::binary);
    ASSERT_TRUE(clientVis.is_open());
    navi::GraphVisDef clientVisProto;
    ASSERT_TRUE(clientVisProto.ParseFromIstream(&clientVis));
    EXPECT_TRUE(containsKernel(clientVisProto, "navi_sched.schedule_k"));
    EXPECT_TRUE(containsKernel(clientVisProto, "navi_sched.client_forward_k"));
    EXPECT_TRUE(containsKernel(clientVisProto, "navi_sched.client_response_k"));
    EXPECT_TRUE(containsKernel(clientVisProto,
                               "navi_sched.master_dispatch_guard_k"));
    EXPECT_FALSE(containsKernel(clientVisProto,
                                "navi_sched.client_dispatch_guard_k"));
    EXPECT_FALSE(containsKernel(clientVisProto,
                                "navi_sched.master_dispatch_failure_k"));
    EXPECT_TRUE(containsRemoteKernelMetric(clientVisProto, "agent_biz",
                                           "navi_sched.agent_request_k"));
    const auto expectedRpcTimeoutMs = getNaviSchedRpcTimeoutMs(entryTimeoutMs);
    const auto masterTimeouts = collectSubGraphTimeouts(clientVisProto, "master_biz");
    const auto agentTimeouts = collectSubGraphTimeouts(clientVisProto, "agent_biz");
    EXPECT_NE(masterTimeouts.end(),
              std::find(masterTimeouts.begin(), masterTimeouts.end(),
                        expectedRpcTimeoutMs));
    EXPECT_EQ(agentTimeouts.end(),
              std::find(agentTimeouts.begin(), agentTimeouts.end(),
                        expectedRpcTimeoutMs));
    EXPECT_EQ(2, countSummarizedDataEvents(
                     clientVisProto, AGENT_RESPONSE_TO_CLIENT_TYPE_ID));
    EXPECT_FALSE(containsGraphError(clientVisProto));
}

TEST_F(NaviSchedClusterTest,
       testLaterClientRequestCarriesPublishedGlobalZeroPointToSelectedAgent)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    auto agentLatencyModel = loadLatencyModelR("host_2", "agent_biz");
    ASSERT_TRUE(clientNavi);
    ASSERT_TRUE(agentLatencyModel);
    ASSERT_TRUE(_masterLatencyModelR);

    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    const auto agentAddress =
        "127.0.0.1:" + std::to_string(agentGrpcPort);
    _masterStateR->registerAgent("agent_0", "agent_biz", agentAddress, 0);

    const auto initial = agentLatencyModel->getSnapshot();
    ASSERT_TRUE(initial);
    auto handshake = makeHandshakeUpdate(18000);
    handshake->latency = std::make_shared<AgentLatencyToMasterData>();
    handshake->latency->formula = initial->formula;
    handshake->latency->binaryVersion = initial->binaryVersion;
    handshake->latency->modelRevision = 1024;
    handshake->latency->parameters = initial->parameters;
    handshake->latency->optimizerState = makeLatencyOptimizerState(
        handshake->latency->parameters.size(), handshake->latency->modelRevision);
    handshake->latency->parameters[
        NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 17.0;
    handshake->latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(handshake->latency->parameters);
    handshake->latencyModelFingerprint = "test-fingerprint";
    ASSERT_TRUE(_masterStateR->reportAgent("agent_0", handshake));
    ASSERT_TRUE(_masterStateR->reportAgent(
        "agent_0", makeCachePolicyUpdate(256)));
    ASSERT_TRUE(_masterStateR->reportAgent(
        "agent_0", makeEmptyPendingCacheSnapshot()));
    ASSERT_TRUE(_masterLatencyModelR->record(
        "agent_0", "test-fingerprint", *handshake->latency, true, 1,
        "agent_0"));

    LatencyModelGlobalData published;
    for (size_t retry = 0; retry < 300; ++retry) {
        published = _masterLatencyModelR->getGlobalZeroPoint(
            initial->formula, initial->binaryVersion, initial->parameters.size());
        if (published.valid()) {
            break;
        }
        usleep(10000);
    }
    ASSERT_TRUE(published.valid());
    ASSERT_DOUBLE_EQ(handshake->latency->zeroPoint, published.globalZeroPoint);

    _agent0RequestR->setCompletionHandler([](
        const NaviSchedForwardRequest &request,
        const navi::AsyncPipePtr &completionPipe) {
        if (!completionPipe) {
            return false;
        }
        auto ack = std::make_shared<AgentResponseToClientData>();
        ack->requestId = request.request.requestId;
        ack->kind = AgentResponseKind::ACK;
        ack->terminal = true;
        return navi::EC_NONE == completionPipe->setData(ack) &&
               navi::EC_NONE == completionPipe->setEof();
    });
    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    auto result = client.schedule(
        makeClientRequest(701, "agent_biz", 4, nullptr), "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_FALSE(result.decision->latencyModelGlobal.valid());

    const auto applied = agentLatencyModel->getGlobalZeroPoint();
    ASSERT_TRUE(applied.valid());
    EXPECT_EQ(published.masterEpoch, applied.masterEpoch);
    EXPECT_EQ(published.version, applied.version);
    EXPECT_DOUBLE_EQ(published.globalZeroPoint, applied.globalZeroPoint);
}

TEST_F(NaviSchedClusterTest,
       testScheduleKernelReleasesReservationWhenDispatchGuardNeverStarts)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto state = _masterStateR->registerAgent(
        "agent_0", "agent_biz",
        "127.0.0.1:" + std::to_string(agentGrpcPort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    const auto masterGraphId = builder.newSubGraph("master_biz");
    builder.subGraph(masterGraphId).location("master_biz", 1, 0);
    auto schedule = builder.node(masterGraphId, "unguarded_schedule")
                        .kernel("navi_sched.schedule_k")
                        .integerAttr("dispatch_guard_enabled", 1);
    schedule.out("decision").asGraphOutput("decision");
    ASSERT_TRUE(builder.ok());
    navi::GraphBuilder::fillHashKey(graphDef.get());

    auto request = makeClientRequest(
        87, "agent_biz", 256, makeBlockHashes({11}));
    navi::RunGraphParams params;
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    params.setTimeoutMs(1000);
    navi::NamedData namedData;
    namedData.graphId = masterGraphId;
    namedData.name = CLIENT_REQUEST_TO_MASTER_DATA_NAME;
    namedData.data = request;
    ASSERT_TRUE(params.addNamedData(namedData));

    auto userResult = clientNavi->runGraph(graphDef.release(), params);
    ASSERT_TRUE(userResult);
    bool receivedDecision = false;
    while (true) {
        navi::NaviUserData userData;
        bool eof = false;
        if (userResult->nextData(userData, eof) && userData.data) {
            receivedDecision = bool(std::dynamic_pointer_cast<
                                    MasterResponseToClientData>(userData.data));
        }
        if (eof) {
            break;
        }
    }
    ASSERT_TRUE(receivedDecision);
    for (size_t retry = 0;
         retry < 100 && getReservationState(state).batchCount != 0; ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, getReservationState(state).batchCount);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
}

TEST_F(NaviSchedClusterTest, testClientForwardResponseOutlivesAdmissionTimeout)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:" + std::to_string(agentGrpcPort), 0));
    auto handshake = makeHandshakeUpdate(18000);
    handshake->control->pdSeparated = false;
    handshake->control->unifiedLatencyPrediction = true;
    handshake->workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    handshake->workload->revision = 1;
    handshake->workload->eventTimeUs = autil::TimeUtility::currentTime();
    handshake->workload->prefillCapacitySplit = true;
    handshake->workload->maxRunningRequestCount = 32;
    ASSERT_TRUE(handshake->workload->valid());
    ASSERT_TRUE(_masterStateR->reportAgent("agent_0", handshake));
    ASSERT_TRUE(_masterStateR->reportAgent("agent_0", makeCachePolicyUpdate(256)));
    ASSERT_TRUE(_masterStateR->reportAgent(
        "agent_0", makeEmptyPendingCacheSnapshot()));

    std::promise<void> responseFutureReady;
    auto responseFutureReadyWaiter = responseFutureReady.get_future();
    std::future<void> responseFuture;
    _agent0RequestR->setCompletionHandler(
        [this, &responseFutureReady,
         &responseFuture](const NaviSchedForwardRequest &request,
                          const navi::AsyncPipePtr &completionPipe) {
            if (!completionPipe || request.request.requestId == 0 ||
                request.request.requestId != request.decision.requestId) {
                return false;
            }
            auto received = std::make_shared<AgentUpdateToMasterData>();
            received->event = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED;
            received->requestEvent =
                std::make_shared<AgentRequestEventToMasterData>();
            received->requestEvent->localBatch.scheduleBatchId =
                request.decision.batchId;
            received->requestEvent->localBatch.localBatchId =
                request.decision.batchId;
            received->requestEvent->localBatch.requestCount = 1;
            if (!_masterStateR->reportAgent("agent_0", received)) {
                return false;
            }
            responseFuture = std::async(
                std::launch::async, [requestId = request.request.requestId,
                                     completionPipe]() {
                    auto ack = std::make_shared<AgentResponseToClientData>();
                    ack->requestId = requestId;
                    ack->kind = AgentResponseKind::ACK;
                    ASSERT_EQ(navi::EC_NONE, completionPipe->setData(ack));

                    usleep(400000);
                    auto terminal = std::make_shared<AgentResponseToClientData>();
                    terminal->requestId = requestId;
                    terminal->kind = AgentResponseKind::OUTPUT;
                    terminal->frames = {"done"};
                    terminal->terminal = true;
                    ASSERT_EQ(navi::EC_NONE, completionPipe->setData(terminal));
                    ASSERT_EQ(navi::EC_NONE, completionPipe->setEof());
                });
            responseFutureReady.set_value();
            return true;
        });

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    constexpr int64_t entryTimeoutMs = 200;
    auto result = client.schedule(
        makeClientRequest(80, "agent_biz", 4, nullptr), "master_biz",
        entryTimeoutMs);
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    ASSERT_FALSE(result.decision->target.address.empty());
    ASSERT_EQ(std::future_status::ready,
              responseFutureReadyWaiter.wait_for(std::chrono::seconds(1)));
    ASSERT_TRUE(responseFuture.valid());

    auto terminal = _clientResponseR->takeNextResponse(1000);
    ASSERT_TRUE(terminal);
    EXPECT_EQ(AgentResponseKind::OUTPUT, terminal->kind);
    EXPECT_EQ(std::vector<std::string>({"done"}), terminal->frames);
    EXPECT_TRUE(terminal->terminal);
    responseFuture.get();
    for (size_t retry = 0; retry < 100 && _clientResponseR->requestCount() != 0;
         ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    std::ifstream clientVis("./client_graph.vis", std::ios::binary);
    ASSERT_TRUE(clientVis.is_open());
    navi::GraphVisDef clientVisProto;
    ASSERT_TRUE(clientVisProto.ParseFromIstream(&clientVis));
    EXPECT_FALSE(containsGraphError(clientVisProto));
}

TEST_F(NaviSchedClusterTest, testClientForwardSuppressedResponseConsumesFullStream)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    ASSERT_TRUE(_masterStateR->registerAgent(
        "agent_0", "agent_biz", "127.0.0.1:" + std::to_string(agentGrpcPort), 0));
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    std::atomic<uint64_t> canonicalRequestId{0};
    _agent0RequestR->setCompletionHandler(
        [this, &canonicalRequestId](const NaviSchedForwardRequest &request,
                                   const navi::AsyncPipePtr &completionPipe) {
            if (!completionPipe || request.request.requestId == 0 ||
                request.request.requestId != request.decision.requestId) {
                return false;
            }
            canonicalRequestId.store(request.request.requestId,
                                     std::memory_order_relaxed);
            auto received = std::make_shared<AgentUpdateToMasterData>();
            received->event = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED;
            received->requestEvent =
                std::make_shared<AgentRequestEventToMasterData>();
            received->requestEvent->localBatch.scheduleBatchId =
                request.decision.batchId;
            received->requestEvent->localBatch.localBatchId =
                request.decision.batchId;
            received->requestEvent->localBatch.requestCount = 1;
            if (!_masterStateR->reportAgent("agent_0", received)) {
                return false;
            }
            auto ack = std::make_shared<AgentResponseToClientData>();
            ack->requestId = request.request.requestId;
            ack->kind = AgentResponseKind::ACK;
            auto first = std::make_shared<AgentResponseToClientData>();
            first->requestId = request.request.requestId;
            first->kind = AgentResponseKind::OUTPUT;
            first->frames = {"replay-token-1"};
            auto second = std::make_shared<AgentResponseToClientData>();
            second->requestId = request.request.requestId;
            second->kind = AgentResponseKind::OUTPUT;
            second->frames = {"replay-token-2"};
            auto terminal = std::make_shared<AgentResponseToClientData>();
            terminal->requestId = request.request.requestId;
            terminal->kind = AgentResponseKind::OUTPUT;
            terminal->frames = {"replay-token-3"};
            terminal->terminal = true;
            return navi::EC_NONE == completionPipe->setData(ack) &&
                   navi::EC_NONE == completionPipe->setData(first) &&
                   navi::EC_NONE == completionPipe->setData(second) &&
                   navi::EC_NONE == completionPipe->setData(terminal) &&
                   navi::EC_NONE == completionPipe->setEof();
        });

    ClientGraphRunner client(clientNavi, false, _clientResponseR);
    client.setClientBizName("client_biz");
    std::promise<ClientGraphRunResult> completionPromise;
    auto completion = completionPromise.get_future();
    auto launchResult = client.scheduleDetached(
        makeClientRequest(86, "agent_biz", 4, nullptr), "master_biz", 10000,
        [&completionPromise](ClientGraphRunResult result) {
            completionPromise.set_value(std::move(result));
        },
        false);
    ASSERT_EQ(navi::EC_NONE, launchResult.errorCode) << launchResult.errorMessage;
    ASSERT_EQ(std::future_status::ready,
              completion.wait_for(std::chrono::seconds(5)));
    auto result = completion.get();
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_EQ(86, result.decision->requestId);
    EXPECT_GT(result.masterRequestLatencyUs, 0);
    EXPECT_NE(0, canonicalRequestId.load(std::memory_order_relaxed));
    EXPECT_NE(86, canonicalRequestId.load(std::memory_order_relaxed));
    EXPECT_FALSE(_clientResponseR->takeNextResponse(0));
    for (size_t retry = 0; retry < 100 && _clientResponseR->requestCount() != 0;
         ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
}

TEST_F(NaviSchedClusterTest, testClientForwardFailureReleasesMasterReservation)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);

    constexpr int32_t unavailablePort = 65534;
    auto state = _masterStateR->registerAgent(
        "unreachable_agent", "agent_biz",
        "127.0.0.1:" + std::to_string(unavailablePort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("unreachable_agent", unavailablePort));
    auto latency = std::make_shared<AgentUpdateToMasterData>();
    latency->event = NaviSchedAgentEvent::LATENCY_MODEL_UPDATED;
    latency->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    latency->requestEvent->inflightStatsRevision = 1;
    latency->latency = std::make_shared<AgentLatencyToMasterData>();
    latency->latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    latency->latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    latency->latency->modelRevision = 1;
    latency->latency->parameters = {
        30.0, 20.0, 10.0, 40.0, 5.0, 3.0, 0.0, 100.0, 50.0, 1.0,
    };
    latency->latency->optimizerState = makeLatencyOptimizerState(
        latency->latency->parameters.size(), latency->latency->modelRevision);
    latency->latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(latency->latency->parameters);
    ASSERT_TRUE(_masterStateR->reportAgent("unreachable_agent", latency));

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    auto result = client.schedule(
        makeClientRequest(8, "agent_biz", 512, makeBlockHashes({11, 22})),
        "master_biz", 1000);
    EXPECT_NE(navi::EC_NONE, result.errorCode);
    for (size_t retry = 0;
         retry < 100 && _clientResponseR->requestCount() != 0; ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("agent_biz", snapshot));
    EXPECT_DOUBLE_EQ(0.0, snapshot.load.queueTotalPredictedLatencyMs);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
    std::ifstream clientVis("./client_graph.vis", std::ios::binary);
    ASSERT_TRUE(clientVis.is_open());
    navi::GraphVisDef clientVisProto;
    ASSERT_TRUE(clientVisProto.ParseFromIstream(&clientVis));
    EXPECT_TRUE(containsKernel(clientVisProto,
                               "navi_sched.master_dispatch_guard_k"));
    EXPECT_FALSE(containsKernel(clientVisProto,
                                "navi_sched.client_dispatch_guard_k"));
    EXPECT_FALSE(containsKernel(clientVisProto,
                               "navi_sched.master_dispatch_failure_k"));
}

TEST_F(NaviSchedClusterTest,
       testInvalidAgentEndpointReleasesMasterReservation)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);

    auto state = _masterStateR->registerAgent(
        "invalid_agent", "agent_biz", "invalid-agent-endpoint", 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("invalid_agent", 65533));
    auto latency = std::make_shared<AgentUpdateToMasterData>();
    latency->event = NaviSchedAgentEvent::LATENCY_MODEL_UPDATED;
    latency->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    latency->requestEvent->inflightStatsRevision = 1;
    latency->latency = std::make_shared<AgentLatencyToMasterData>();
    latency->latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    latency->latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    latency->latency->modelRevision = 1;
    latency->latency->parameters = {
        30.0, 20.0, 10.0, 40.0, 5.0, 3.0, 0.0, 100.0, 50.0, 1.0,
    };
    latency->latency->optimizerState = makeLatencyOptimizerState(
        latency->latency->parameters.size(), latency->latency->modelRevision);
    latency->latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(latency->latency->parameters);
    ASSERT_TRUE(_masterStateR->reportAgent("invalid_agent", latency));

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    auto result = client.schedule(
        makeClientRequest(81, "agent_biz", 256, makeBlockHashes({11})),
        "master_biz", 1000);
    EXPECT_NE(navi::EC_NONE, result.errorCode);
    for (size_t retry = 0;
         retry < 100 && _clientResponseR->requestCount() != 0; ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("agent_biz", snapshot));
    EXPECT_DOUBLE_EQ(0.0, snapshot.load.queueTotalPredictedLatencyMs);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
}

TEST_F(NaviSchedClusterTest,
       testAgentAdmissionTimeoutReleasesMasterReservation)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto state = _masterStateR->registerAgent(
        "agent_0", "agent_biz",
        "127.0.0.1:" + std::to_string(agentGrpcPort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    std::atomic<int64_t> receivedRemainingMs{0};
    _agent0RequestR->setCompletionHandler(
        [&receivedRemainingMs](const NaviSchedForwardRequest &request,
           const navi::AsyncPipePtr &completionPipe) {
            receivedRemainingMs.store(request.request.timeoutMs);
            return request.request.requestId != 0 &&
                   request.request.requestId == request.decision.requestId &&
                   bool(completionPipe);
        });

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    constexpr int64_t entryTimeoutMs = 300;
    const auto rpcTimeoutMs = getNaviSchedRpcTimeoutMs(entryTimeoutMs);
    const auto beginTimeUs = autil::TimeUtility::currentTime();
    auto request = makeClientRequest(82, "agent_biz", 256, makeBlockHashes({11}));
    request->timeoutMs = entryTimeoutMs;
    auto result = client.schedule(request, "master_biz", entryTimeoutMs);
    const auto elapsedMs =
        (autil::TimeUtility::currentTime() - beginTimeUs) / 1000.0;
    EXPECT_EQ(navi::EC_TIMEOUT, result.errorCode);
    EXPECT_EQ("NaviSched request admission timed out", result.errorMessage);
    EXPECT_GE(elapsedMs, rpcTimeoutMs * 0.8);
    EXPECT_LT(elapsedMs, entryTimeoutMs + 50);
    EXPECT_GT(receivedRemainingMs.load(), 0);
    EXPECT_LE(receivedRemainingMs.load(), rpcTimeoutMs);
    for (size_t retry = 0;
         retry < 100 &&
         (_clientResponseR->requestCount() != 0 ||
          getReservationState(state).batchCount != 0);
         ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    auto reservationState = getReservationState(state);
    EXPECT_EQ(0, reservationState.batchCount);
    EXPECT_EQ(0, reservationState.dispatchWaiterCount);
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("agent_biz", snapshot));
    EXPECT_DOUBLE_EQ(0.0, snapshot.load.queueTotalPredictedLatencyMs);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
    EXPECT_EQ(0, _masterScheduleR->getPendingRequestCount());
}

TEST_F(NaviSchedClusterTest,
       testBlockedAgentAdmissionCanOutliveClientTimeout)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto state = _masterStateR->registerAgent(
        "agent_0", "agent_biz",
        "127.0.0.1:" + std::to_string(agentGrpcPort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    std::promise<int64_t> entered;
    auto enteredFuture = entered.get_future();
    std::promise<void> release;
    auto releaseFuture = release.get_future().share();
    std::promise<bool> returned;
    auto returnedFuture = returned.get_future();
    _agent0RequestR->setCompletionHandler(
        [&entered, releaseFuture, &returned](
            const NaviSchedForwardRequest &request,
            const navi::AsyncPipePtr &completionPipe) {
            entered.set_value(request.decision.batchId);
            // Hold actual Agent kernel initialization across Client cancellation.
            // The bound ensures a failed assertion cannot strand cluster teardown.
            const bool released = releaseFuture.wait_for(std::chrono::seconds(5)) ==
                                  std::future_status::ready;
            const bool accepted = released && completionPipe &&
                                  request.request.requestId != 0 &&
                                  request.decision.requestId == request.request.requestId;
            returned.set_value(accepted);
            return accepted;
        });

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    auto response = std::async(std::launch::async, [&client]() {
        return client.schedule(
            makeClientRequest(84, "agent_biz", 256, makeBlockHashes({11})),
            "master_biz", 300);
    });
    const auto enteredStatus = enteredFuture.wait_for(std::chrono::seconds(2));
    const auto responseStatus = response.wait_for(std::chrono::seconds(1));
    ReservationState cancelledState;
    if (responseStatus == std::future_status::ready) {
        for (size_t retry = 0;
             retry < 100 && getReservationState(state).batchCount != 0; ++retry) {
            usleep(1000);
        }
        cancelledState = getReservationState(state);
    }
    const auto beforeRelease = returnedFuture.wait_for(std::chrono::seconds(0));
    release.set_value();
    const auto result = response.get();
    const auto returnedStatus = returnedFuture.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(std::future_status::ready, enteredStatus);
    if (enteredStatus == std::future_status::ready) {
        EXPECT_GT(enteredFuture.get(), 0);
    }
    EXPECT_EQ(std::future_status::ready, responseStatus);
    EXPECT_EQ(navi::EC_TIMEOUT, result.errorCode);
    EXPECT_EQ("NaviSched request admission timed out", result.errorMessage);
    EXPECT_EQ(std::future_status::timeout, beforeRelease);
    EXPECT_EQ(0u, cancelledState.batchCount);
    EXPECT_EQ(0u, cancelledState.dispatchWaiterCount);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
    EXPECT_EQ(std::future_status::ready, returnedStatus);
    if (returnedStatus == std::future_status::ready) {
        EXPECT_TRUE(returnedFuture.get());
    }
}

TEST_F(NaviSchedClusterTest,
       testAgentAckWithoutMasterAdmissionReleasesReservation)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto state = _masterStateR->registerAgent(
        "agent_0", "agent_biz",
        "127.0.0.1:" + std::to_string(agentGrpcPort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    _agent0RequestR->setCompletionHandler(
        [](const NaviSchedForwardRequest &request,
           const navi::AsyncPipePtr &completionPipe) {
            if (!completionPipe || request.request.requestId == 0 ||
                request.request.requestId != request.decision.requestId) {
                return false;
            }
            auto ack = std::make_shared<AgentResponseToClientData>();
            ack->requestId = request.request.requestId;
            ack->kind = AgentResponseKind::ACK;
            ack->terminal = true;
            return navi::EC_NONE == completionPipe->setData(ack) &&
                   navi::EC_NONE == completionPipe->setEof();
        });

    ClientGraphRunner client(clientNavi, false, _clientResponseR);
    client.setClientBizName("client_biz");
    constexpr int64_t entryTimeoutMs = 300;
    auto result = client.schedule(
        makeClientRequest(83, "agent_biz", 256, makeBlockHashes({11})),
        "master_biz", entryTimeoutMs);
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_EQ(1, getReservationState(state).batchCount);
    for (size_t retry = 0;
         retry < 100 &&
         (_clientResponseR->requestCount() != 0 ||
          getReservationState(state).batchCount != 0);
         ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    const auto reservationState = getReservationState(state);
    EXPECT_EQ(0, reservationState.batchCount);
    EXPECT_EQ(0, reservationState.dispatchWaiterCount);
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("agent_biz", snapshot));
    EXPECT_DOUBLE_EQ(0.0, snapshot.load.queueTotalPredictedLatencyMs);
    EXPECT_EQ(0, state->getCacheSyncStats().pendingCacheBatchCount);
}

TEST_F(NaviSchedClusterTest, testMasterTimeoutReturnsBeforeEntryDeadline)
{
    _maxWaitScheduleTimeMs = 1000;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);

    ClientGraphRunner client(clientNavi, true, _clientResponseR);
    client.setClientBizName("client_biz");
    constexpr int64_t entryTimeoutMs = 300;
    const auto rpcTimeoutMs = getNaviSchedRpcTimeoutMs(entryTimeoutMs);
    const auto beginTimeUs = autil::TimeUtility::currentTime();
    auto request =
        makeClientRequest(85, "agent_biz", 256, makeBlockHashes({11}));
    request->timeoutMs = 5000;
    auto result = client.schedule(request, "master_biz", entryTimeoutMs);
    const auto elapsedMs =
        (autil::TimeUtility::currentTime() - beginTimeUs) / 1000.0;

    EXPECT_EQ(1, _masterScheduleR->getSubmitCount());
    EXPECT_EQ(navi::EC_TIMEOUT, result.errorCode);
    EXPECT_EQ("NaviSched request admission timed out", result.errorMessage);
    EXPECT_GE(elapsedMs, rpcTimeoutMs * 0.8);
    EXPECT_LT(elapsedMs, entryTimeoutMs + 50);
    for (size_t retry = 0;
         retry < 100 && (_clientResponseR->requestCount() != 0 ||
                         _masterScheduleR->getPendingRequestCount() != 0); ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    EXPECT_EQ(0, _masterScheduleR->getPendingRequestCount());
}

TEST_F(NaviSchedClusterTest,
       testPartialAdmissionSurvivesPeerRequestTimeout)
{
    _maxWaitScheduleCount = 2;
    _maxWaitScheduleTimeMs = 1000;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    int32_t agentGrpcPort = 0;
    for (const auto &node : _cluster.getClusterGigConfig().nodes) {
        if ("agent_biz" == node._bizName) {
            agentGrpcPort = node._grpcPort;
            break;
        }
    }
    ASSERT_GT(agentGrpcPort, 0);
    auto state = _masterStateR->registerAgent(
        "agent_0", "agent_biz",
        "127.0.0.1:" + std::to_string(agentGrpcPort), 0);
    ASSERT_TRUE(state);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));

    std::atomic<int64_t> scheduleBatchId{0};
    std::atomic<bool> batchIdsMatch{true};
    _agent0RequestR->setCompletionHandler(
        [this, &scheduleBatchId, &batchIdsMatch](
            const NaviSchedForwardRequest &request,
            const navi::AsyncPipePtr &completionPipe) {
            auto expected = int64_t(0);
            if (!scheduleBatchId.compare_exchange_strong(
                    expected, request.decision.batchId) &&
                expected != request.decision.batchId) {
                batchIdsMatch.store(false, std::memory_order_relaxed);
            }
            if (!request.request.blockHashes ||
                request.request.blockHashes->size() != 2) {
                return false;
            }
            const auto sourceRequestId = request.request.blockHashes->back();
            if (84 == sourceRequestId) {
                return bool(completionPipe);
            }
            if (83 != sourceRequestId || !completionPipe ||
                request.request.requestId == 0 ||
                request.request.requestId != request.decision.requestId) {
                return false;
            }

            NaviSchedLocalBatchInfo localBatch;
            localBatch.scheduleBatchId = request.decision.batchId;
            localBatch.localBatchId = 8300;
            localBatch.requestCount = 1;
            localBatch.requestIds = {request.request.requestId};
            auto admitted = std::make_shared<AgentUpdateToMasterData>();
            admitted->event = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED;
            admitted->requestEvent =
                std::make_shared<AgentRequestEventToMasterData>();
            admitted->requestEvent->localBatch = localBatch;
            if (!_masterStateR->reportAgent("agent_0", admitted)) {
                return false;
            }
            auto finished = std::make_shared<AgentUpdateToMasterData>();
            finished->event = NaviSchedAgentEvent::BATCH_FINISHED;
            finished->requestEvent =
                std::make_shared<AgentRequestEventToMasterData>();
            finished->requestEvent->localBatch = localBatch;
            if (!_masterStateR->reportAgent("agent_0", finished)) {
                return false;
            }

            auto ack = std::make_shared<AgentResponseToClientData>();
            ack->requestId = request.request.requestId;
            ack->kind = AgentResponseKind::ACK;
            auto terminal = std::make_shared<AgentResponseToClientData>();
            terminal->requestId = request.request.requestId;
            terminal->kind = AgentResponseKind::OUTPUT;
            terminal->frames = {"done"};
            terminal->terminal = true;
            return navi::EC_NONE == completionPipe->setData(ack) &&
                   navi::EC_NONE == completionPipe->setData(terminal) &&
                   navi::EC_NONE == completionPipe->setEof();
        });

    ClientGraphRunResult admittedResult;
    ClientGraphRunResult timedOutResult;
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    auto run = [&](uint64_t requestId, int64_t timeoutMs,
                   ClientGraphRunResult &result) {
        ClientGraphRunner client(clientNavi, false, _clientResponseR);
        client.setClientBizName("client_biz");
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto request = makeClientRequest(
            requestId, "agent_biz", 512,
            makeBlockHashes({11, requestId}));
        request->timeoutMs = 5000;
        result = client.schedule(request, "master_biz", timeoutMs);
    };
    std::thread admittedThread(run, 83, 1000, std::ref(admittedResult));
    std::thread timedOutThread(run, 84, 100, std::ref(timedOutResult));
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    admittedThread.join();
    timedOutThread.join();

    EXPECT_EQ(2, _masterScheduleR->getSubmitCount());
    EXPECT_EQ(2, _masterStateR->getClientRequestCount());
    EXPECT_EQ(navi::EC_NONE, admittedResult.errorCode)
        << admittedResult.errorMessage;
    EXPECT_EQ(navi::EC_TIMEOUT, timedOutResult.errorCode);
    EXPECT_TRUE(batchIdsMatch.load(std::memory_order_relaxed));
    EXPECT_GT(scheduleBatchId.load(std::memory_order_relaxed), 0);
    for (size_t retry = 0;
         retry < 100 && _clientResponseR->requestCount() != 0; ++retry) {
        usleep(10000);
    }
    EXPECT_EQ(0, _clientResponseR->requestCount());
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("agent_biz", snapshot));
    EXPECT_DOUBLE_EQ(0.0, snapshot.load.queueTotalPredictedLatencyMs);
    auto pendingStats =
        _masterPendingCacheR->getStats(hashString("agent_0"));
    size_t pendingCleanupWaitMs = 0;
    while (pendingStats.scheduledBatchCount != 0 &&
           pendingCleanupWaitMs < 1000) {
        usleep(10000);
        pendingCleanupWaitMs += 10;
        pendingStats =
            _masterPendingCacheR->getStats(hashString("agent_0"));
    }
    EXPECT_EQ(0, pendingStats.scheduledBatchCount)
        << "Master pending cleanup exceeded " << pendingCleanupWaitMs << " ms";
    EXPECT_EQ(0, pendingStats.agentBatchCount);
    EXPECT_EQ(0, pendingStats.batchCount);
    EXPECT_EQ(0, state->getMatchedBlockCount({11}));
    EXPECT_EQ(0, state->getMatchedBlockCount({83}));
    EXPECT_EQ(0, state->getMatchedBlockCount({84}));
}

TEST_F(NaviSchedClusterTest, testMaxCacheHitSchedulerUsesLiveCache)
{
    _schedulerKernelId = MaxCacheHitSchedulerKernel::KERNEL_ID;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto registerAgent = [this](const std::string &agentId, int32_t port,
                                const NaviSchedBlockHashes &hashes) {
        _masterStateR->registerAgent(agentId, "agent_biz",
                                     "127.0.0.1:" + std::to_string(port), 100);
        ASSERT_TRUE(reportReadyAgent(agentId, port, true));
        auto update = std::make_shared<AgentUpdateToMasterData>();
        update->event = NaviSchedAgentEvent::CACHE_UPDATED;
        update->cache = std::make_shared<AgentCacheToMasterData>();
        update->cache->totalBlocks = 100;
        update->cache->revision = 2;
        update->cache->cachedBlockCount = hashes.size();
        update->cache->fullReadyBlockCount = 0;
        update->cache->tailReadyBlockCount = 0;
        for (const auto hash : hashes) {
            update->cache->hashChecksum ^= hash;
        }
        update->cache->addedBlockHashes =
            std::make_shared<const NaviSchedBlockHashes>(hashes);
        ASSERT_TRUE(_masterStateR->reportAgent(agentId, update));
    };
    registerAgent("agent_0", 18000, {11});
    registerAgent("agent_1", 19000, {11, 22});

    auto agent0 = _masterStateR->getNodeState(hashString("agent_0"));
    auto agent1 = _masterStateR->getNodeState(hashString("agent_1"));
    ASSERT_TRUE(agent0);
    ASSERT_TRUE(agent1);
    ASSERT_EQ(size_t(1), agent0->getCacheSyncStats().masterCachedBlockCount);
    ASSERT_EQ(size_t(2), agent1->getCacheSyncStats().masterCachedBlockCount);
    ASSERT_EQ(uint64_t(2), agent0->getCacheSyncStats().agentRevision);
    ASSERT_EQ(uint64_t(2), agent1->getCacheSyncStats().agentRevision);
    ASSERT_EQ(uint64_t(11), agent0->getCacheSyncStats().masterHashChecksum);
    ASSERT_EQ(uint64_t(11 ^ 22), agent1->getCacheSyncStats().masterHashChecksum);
    ASSERT_EQ(1, agent0->getMatchedBlockCount({11, 22, 33}));
    ASSERT_EQ(2, agent1->getMatchedBlockCount({11, 22, 33}));

    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    ClientGraphRunner client(clientNavi, true);
    auto request = makeClientRequest(9, "agent_biz", 768, makeBlockHashes({11, 22, 33}));
    request->collectReplay = true;
    auto result = client.schedule(request, "master_biz", 10000);

    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_TRUE(result.decision);
    EXPECT_EQ("max_cache_hit_selected", result.decision->reason);
    ASSERT_TRUE(result.replay);
    EXPECT_EQ("127.0.0.1:19000", result.decision->target.address)
        << NaviSchedScheduleReplayJson::toJsonString(*result.replay);

    std::set<std::string> coldCacheTargets;
    std::string firstColdCacheTarget;
    for (uint64_t requestId = 100; requestId < 132; ++requestId) {
        auto request =
            makeClientRequest(requestId, "agent_biz", 512,
                              makeBlockHashes({99, 100}));
        const auto scheduleItem = MasterStateR::makeScheduleItem(*request, 0);
        const auto agent0Match =
            agent0->getCacheMatchBreakdown(scheduleItem.request);
        const auto agent1Match =
            agent1->getCacheMatchBreakdown(scheduleItem.request);
        auto coldCacheResult = client.schedule(
            request, "master_biz", 10000);
        ASSERT_EQ(navi::EC_NONE, coldCacheResult.errorCode)
            << coldCacheResult.errorMessage;
        ASSERT_TRUE(coldCacheResult.decision);
        if (firstColdCacheTarget.empty()) {
            firstColdCacheTarget = coldCacheResult.decision->target.address;
        }
        EXPECT_EQ(firstColdCacheTarget, coldCacheResult.decision->target.address)
            << "requestId=" << requestId
            << " agent0Matched=" << agent0Match.matchedBlockCount
            << " agent0Static=" << agent0Match.staticMatchedBlockCount
            << " agent0PendingBatches=" << agent0Match.pendingBatchCount
            << " agent1Matched=" << agent1Match.matchedBlockCount
            << " agent1Static=" << agent1Match.staticMatchedBlockCount
            << " agent1PendingBatches=" << agent1Match.pendingBatchCount;
        coldCacheTargets.emplace(coldCacheResult.decision->target.address);
    }
    EXPECT_EQ(1, coldCacheTargets.size());
}

TEST_F(NaviSchedClusterTest, testRandomSchedulerSelectsAvailableNode)
{
    _schedulerKernelId = RandomSchedulerKernel::KERNEL_ID;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    ClientGraphRunner client(clientNavi, true);

    auto unavailable = client.schedule(makeClientRequest(10, "agent_biz", 256, nullptr),
                                       "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, unavailable.errorCode) << unavailable.errorMessage;
    ASSERT_TRUE(unavailable.decision);
    EXPECT_EQ("no_available_node", unavailable.decision->reason);
    EXPECT_TRUE(unavailable.decision->target.address.empty());

    const std::set<std::string> addresses = {"127.0.0.1:18000", "127.0.0.1:19000"};
    _masterStateR->registerAgent("agent_0", "agent_biz", "127.0.0.1:18000", 0);
    _masterStateR->registerAgent("agent_1", "agent_biz", "127.0.0.1:19000", 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ASSERT_TRUE(reportReadyAgent("agent_1", 19000));
    for (uint64_t requestId = 11; requestId < 27; ++requestId) {
        auto result = client.schedule(
            makeClientRequest(requestId, "agent_biz", 256, nullptr), "master_biz", 10000);
        ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
        ASSERT_TRUE(result.decision);
        EXPECT_EQ("random_selected", result.decision->reason);
        EXPECT_EQ(1, addresses.count(result.decision->target.address));
    }
}

TEST_F(NaviSchedClusterTest, testResidentGraphBatchesClientRequests)
{
    _maxWaitScheduleCount = 5;
    _maxWaitScheduleTimeMs = 30;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    _masterStateR->registerAgent("agent_0", "agent_biz", "127.0.0.1:18000", 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ClientGraphRunner client(clientNavi, true);
    auto warmup = makeClientRequest(3, "agent_biz", 2, nullptr);
    auto warmupResult = client.schedule(warmup, "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, warmupResult.errorCode) << warmupResult.errorMessage;

    const size_t requestCount = 32;
    autil::ThreadCond startCondition;
    size_t readyCount = 0;
    bool start = false;
    std::vector<ClientGraphRunResult> results(requestCount);
    std::vector<std::thread> threads;
    threads.reserve(requestCount);
    for (size_t index = 0; index < requestCount; ++index) {
        threads.emplace_back([&, index]() {
            {
                autil::ScopedLock lock(startCondition);
                ++readyCount;
                startCondition.broadcast();
                while (!start) {
                    startCondition.wait();
                }
            }
            auto request = makeClientRequest(100 + index, "agent_biz", 4, nullptr);
            results[index] = client.schedule(request, "master_biz", 10000);
        });
    }
    {
        autil::ScopedLock lock(startCondition);
        while (readyCount < requestCount) {
            startCondition.wait();
        }
        start = true;
        startCondition.broadcast();
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &result : results) {
        ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
        ASSERT_EQ("random_selected", result.decision->reason);
        ASSERT_GT(result.decision->scheduleId, 0);
        ASSERT_DOUBLE_EQ(0.0, result.decision->scheduleLoss);
        ASSERT_LE(result.decision->expectedBatchSize,
                  static_cast<int64_t>(_maxWaitScheduleCount));
    }
    uint64_t maxScheduleId = 0;
    for (const auto &result : results) {
        maxScheduleId = std::max(maxScheduleId, result.decision->scheduleId);
    }
    auto finalResult = client.schedule(
        makeClientRequest(1000, "agent_biz", 4, nullptr), "master_biz", 10000);
    ASSERT_EQ(navi::EC_NONE, finalResult.errorCode) << finalResult.errorMessage;
    ASSERT_GT(finalResult.decision->scheduleId, maxScheduleId);
    ASSERT_DOUBLE_EQ(0.0, finalResult.decision->scheduleLoss);
    ASSERT_EQ(34, _masterScheduleR->getSubmitCount());
}

TEST_F(NaviSchedClusterTest, testResidentGraphSeparatesTargetBizContexts)
{
    _maxWaitScheduleCount = 2;
    _maxWaitScheduleTimeMs = 1000;
    ASSERT_NO_FATAL_FAILURE(buildPartitionCluster());
    _masterStateR->registerAgent("agent_0", "prefill_biz_0", "127.0.0.1:18000", 0);
    _masterStateR->registerAgent("agent_1", "prefill_biz_1", "127.0.0.1:19000", 0);
    auto reportAgent = [this](const std::string &agentId, int32_t port) {
        ASSERT_TRUE(reportReadyAgent(agentId, port));
    };
    reportAgent("agent_0", 18000);
    reportAgent("agent_1", 19000);
    auto request0 = makeClientRequest(2000, "prefill_biz_0", 4, nullptr);
    auto request1 = makeClientRequest(2001, "prefill_biz_1", 4, nullptr);
    auto responsePipe0 = std::make_shared<ScheduleResponsePipe>();
    auto responsePipe1 = std::make_shared<ScheduleResponsePipe>();

    ASSERT_TRUE(_masterScheduleR->submit(request0, responsePipe0));
    ASSERT_TRUE(_masterScheduleR->submit(request1, responsePipe1));
    auto response0 = responsePipe0->waitResponse();
    auto response1 = responsePipe1->waitResponse();

    ASSERT_TRUE(response0);
    ASSERT_TRUE(response1);
    ASSERT_NE(0, response0->requestId);
    ASSERT_NE(0, response1->requestId);
    ASSERT_NE(response0->requestId, response1->requestId);
    ASSERT_NE(2000, response0->requestId);
    ASSERT_NE(2001, response1->requestId);
    ASSERT_EQ("prefill_biz_0", response0->target.bizName);
    ASSERT_EQ("prefill_biz_1", response1->target.bizName);
    ASSERT_EQ("127.0.0.1:18000", response0->target.address);
    ASSERT_EQ("127.0.0.1:19000", response1->target.address);
}

TEST_F(NaviSchedClusterTest, testResidentGraphDoesNotWakeWhenIdle)
{
    _maxWaitScheduleCount = 20;
    _maxWaitScheduleTimeMs = 10;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    usleep(50000);
    _masterScheduleGraph->stop();
    std::ifstream scheduleVis("./master_schedule_graph.vis", std::ios::binary);
    ASSERT_TRUE(scheduleVis.is_open());
    navi::GraphVisDef scheduleVisProto;
    ASSERT_TRUE(scheduleVisProto.ParseFromIstream(&scheduleVis));
    auto scheduleCount =
        getKernelScheduleCount(scheduleVisProto, RandomSchedulerKernel::KERNEL_ID);
    ASSERT_GE(scheduleCount, 1);
    ASSERT_LE(scheduleCount, 4);
}

TEST_F(NaviSchedClusterTest, testRequestTimeoutForcesImmediateScheduling)
{
    _maxWaitScheduleCount = 20;
    _maxWaitScheduleTimeMs = 2000;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    _masterStateR->registerAgent("agent_0", "agent_biz", "127.0.0.1:18000", 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ClientGraphRunner client(clientNavi, true);
    auto request = makeClientRequest(4, "agent_biz", 2, nullptr);

    auto result = client.schedule(request, "master_biz", 1000);

    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_EQ("random_selected", result.decision->reason);
}

TEST_F(NaviSchedClusterTest, testResidentGraphStopSchedulesWaitingRequest)
{
    _maxWaitScheduleCount = 20;
    _maxWaitScheduleTimeMs = 2000;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    _masterStateR->registerAgent("agent_0", "agent_biz", "127.0.0.1:18000", 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ClientGraphRunner client(clientNavi, true);
    auto warmup = makeClientRequest(5, "agent_biz", 2, nullptr);
    auto warmupResult = client.schedule(warmup, "master_biz", 1000);
    ASSERT_EQ(navi::EC_NONE, warmupResult.errorCode) << warmupResult.errorMessage;
    auto submitCount = _masterScheduleR->getSubmitCount();
    auto request = makeClientRequest(6, "agent_biz", 2, nullptr);
    ClientGraphRunResult result;
    auto beginTimeUs = autil::TimeUtility::monotonicTimeUs();
    std::thread requestThread(
        [&]() { result = client.schedule(request, "master_biz", 5000); });
    for (size_t retry = 0;
         retry < 100 && _masterScheduleR->getSubmitCount() == submitCount; ++retry) {
        usleep(10000);
    }
    ASSERT_GT(_masterScheduleR->getSubmitCount(), submitCount);

    _masterScheduleGraph->stop();
    requestThread.join();

    auto elapsedMs = (autil::TimeUtility::monotonicTimeUs() - beginTimeUs) / 1000;
    ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
    ASSERT_EQ(6, result.decision->requestId);
    ASSERT_LT(elapsedMs, 2000);
}

TEST_F(NaviSchedClusterTest, testResidentGraphStopDrainsMultipleBatches)
{
    _maxWaitScheduleCount = 5;
    _maxWaitScheduleTimeMs = 5000;
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto clientNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(clientNavi);
    _masterStateR->registerAgent("agent_0", "agent_biz", "127.0.0.1:18000", 0);
    ASSERT_TRUE(reportReadyAgent("agent_0", 18000));
    ClientGraphRunner client(clientNavi, true);

    const size_t requestCount = 17;
    std::vector<ClientGraphRunResult> results(requestCount);
    std::vector<std::thread> threads;
    threads.reserve(requestCount);
    for (size_t index = 0; index < requestCount; ++index) {
        threads.emplace_back([&, index]() {
            auto request = makeClientRequest(1000 + index, "agent_biz", 2, nullptr);
            results[index] = client.schedule(request, "master_biz", 10000);
        });
    }
    for (size_t retry = 0;
         retry < 500 && _masterScheduleR->getSubmitCount() < requestCount; ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(requestCount, _masterScheduleR->getSubmitCount());

    _masterScheduleGraph->stop();
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &result : results) {
        ASSERT_EQ(navi::EC_NONE, result.errorCode) << result.errorMessage;
        ASSERT_LE(result.decision->expectedBatchSize,
                  static_cast<int64_t>(_maxWaitScheduleCount));
    }
}

TEST_F(NaviSchedClusterTest, testResidentGraphStopDrainsMultipleBizContexts)
{
    _maxWaitScheduleCount = 20;
    _maxWaitScheduleTimeMs = 5000;
    ASSERT_NO_FATAL_FAILURE(buildPartitionCluster());
    auto registerAgent = [this](const std::string &agentId, const std::string &bizName,
                                int32_t port) {
        _masterStateR->registerAgent(agentId, bizName,
                                     "127.0.0.1:" + std::to_string(port), 0);
        ASSERT_TRUE(reportReadyAgent(agentId, port));
    };
    registerAgent("agent_0", "prefill_biz_0", 18000);
    registerAgent("agent_1", "prefill_biz_1", 19000);
    auto request0 = makeClientRequest(3000, "prefill_biz_0", 4, nullptr);
    auto request1 = makeClientRequest(3001, "prefill_biz_1", 4, nullptr);
    auto responsePipe0 = std::make_shared<ScheduleResponsePipe>();
    auto responsePipe1 = std::make_shared<ScheduleResponsePipe>();
    ASSERT_TRUE(_masterScheduleR->submit(request0, responsePipe0));
    ASSERT_TRUE(_masterScheduleR->submit(request1, responsePipe1));

    _masterScheduleGraph->stop();

    auto response0 = responsePipe0->waitResponse();
    auto response1 = responsePipe1->waitResponse();
    ASSERT_TRUE(response0);
    ASSERT_TRUE(response1);
    ASSERT_NE(0, response0->requestId);
    ASSERT_NE(0, response1->requestId);
    ASSERT_NE(response0->requestId, response1->requestId);
    ASSERT_NE(3000, response0->requestId);
    ASSERT_NE(3001, response1->requestId);
    ASSERT_EQ("prefill_biz_0", response0->target.bizName);
    ASSERT_EQ("prefill_biz_1", response1->target.bizName);
}

TEST_F(NaviSchedClusterTest, testPeriodicAsyncPipeRepeatAndStop)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(masterNavi);
    auto runPeriodicGraph = [&](int64_t repeat, int64_t stopAfterRepeat,
                                size_t expectedOutputCount) {
        SCOPED_TRACE("repeat=" + std::to_string(repeat));
        auto graphDef = std::make_unique<navi::GraphDef>();
        navi::GraphBuilder builder(graphDef.get());
        auto graphId = builder.newSubGraph("master_biz");
        builder.subGraph(graphId).location("master_biz", 1, 0);
        auto attrs = "{\"repeat\":" + std::to_string(repeat) +
                     ",\"stop_after_repeat\":" + std::to_string(stopAfterRepeat) + "}";
        builder.node("periodic_pipe_test")
            .kernel("navi_sched.master_periodic_pipe_test_k")
            .jsonAttrs(attrs)
            .out("tick")
            .asGraphOutput("tick");
        ASSERT_TRUE(builder.ok());
        navi::GraphBuilder::fillHashKey(graphDef.get());
        navi::RunGraphParams params;
        params.setTimeoutMs(5000);
        auto result = masterNavi->runLocalGraph(graphDef.release(), params, {});
        ASSERT_TRUE(result);
        size_t outputCount = 0;
        bool eof = false;
        while (!eof) {
            navi::NaviUserData userData;
            if (result->nextData(userData, eof) && userData.data) {
                ++outputCount;
            }
        }
        auto naviResult = result->getNaviResult();
        ASSERT_TRUE(naviResult);
        ASSERT_FALSE(naviResult->hasError()) << naviResult->getErrorMessage();
        ASSERT_EQ(expectedOutputCount, outputCount);
    };

    runPeriodicGraph(3, 0, 3);
    runPeriodicGraph(100, 0, 100);
    runPeriodicGraph(-1, 3, 3);
    runPeriodicGraph(-2, 3, 3);
    runPeriodicGraph(0, 3, 3);
}

TEST_F(NaviSchedClusterTest, testAutoCollectorsDiscoverAllPrefillBiz)
{
    ASSERT_NO_FATAL_FAILURE(buildPartitionCluster());
    auto masterNavi = _cluster.getNavi("host_0");
    ASSERT_TRUE(masterNavi);

    MasterCollectorManager collectorManager(masterNavi, _masterStateR, "master_biz",
                                            true);
    navi::ResourceMap resources;
    ASSERT_TRUE(masterNavi->createResource("master_biz", 1, 0,
                                           {navi::GigClientR::RESOURCE_ID}, resources));
    auto gigIt = resources.getMap().find(navi::GigClientR::RESOURCE_ID);
    ASSERT_NE(resources.getMap().end(), gigIt);
    auto gigClientR = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    ASSERT_TRUE(gigClientR);
    ASSERT_TRUE(collectorManager.startAutoCollect(gigClientR->getSearchService()));
    for (size_t retry = 0; retry < 100 && (0 == _agent0->getSubscriberCount() ||
                                           0 == _agent1->getSubscriberCount());
         ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(1, _agent0->getSubscriberCount());
    ASSERT_EQ(1, _agent1->getSubscriberCount());
    std::set<std::string> agentMetas;
    for (const auto &bizName : {"prefill_biz_0", "prefill_biz_1"}) {
        auto nodeInfos = gigClientR->getSearchService()->getBizNodeInfos(bizName);
        ASSERT_EQ(1, nodeInfos.size());
        ASSERT_TRUE(nodeInfos.front().metaMap);
        auto metaIt = nodeInfos.front().metaMap->find(AgentInstanceR::RESOURCE_ID);
        ASSERT_NE(nodeInfos.front().metaMap->end(), metaIt);
        ASSERT_FALSE(metaIt->second.empty());
        agentMetas.emplace(metaIt->second);
    }
    ASSERT_EQ(2, agentMetas.size());
    RequestLatencyTracker::Snapshot agent0Inflight;
    agent0Inflight.revision = 2;
    agent0Inflight.requestCount = 1;
    agent0Inflight.totalPredictedLatencyMs = 100.0;
    ASSERT_TRUE(_agent0->setInflightStats(agent0Inflight));
    RequestLatencyTracker::Snapshot agent1Inflight;
    agent1Inflight.revision = 2;
    agent1Inflight.requestCount = 2;
    agent1Inflight.totalPredictedLatencyMs = 200.0;
    ASSERT_TRUE(_agent1->setInflightStats(agent1Inflight));
    _agent0->reportRequestReceived(1, 1, 1, 1, 0, 0, 100);
    _agent1->reportRequestReceived(2, 2, 2, 2, 0, 0, 200);
    std::map<std::string, NaviSchedNodeInfo> nodes;
    for (size_t retry = 0; retry < 100; ++retry) {
        nodes = _masterStateR->getNodes();
        bool foundFirst = false;
        bool foundSecond = false;
        for (const auto &pair : nodes) {
            foundFirst |= 1 == pair.second.inflightRequests;
            foundSecond |= 2 == pair.second.inflightRequests;
        }
        if (foundFirst && foundSecond) {
            break;
        }
        usleep(10000);
    }
    const auto reportsBeforeStop = _masterStateR->getAgentReportCount();
    ASSERT_EQ(1, _agent0->beginStop(300));
    ASSERT_EQ(1, _agent1->beginStop(300));
    auto waitForStopAcks = [](const std::shared_ptr<AgentReportR> &agent) {
        for (size_t retry = 0; retry < 100; ++retry) {
            if (agent->waitForStopAcks(10 * 1000)) {
                return true;
            }
        }
        return false;
    };
    ASSERT_TRUE(waitForStopAcks(_agent0));
    ASSERT_TRUE(waitForStopAcks(_agent1));
    _agent0->finishAll();
    _agent1->finishAll();
    for (size_t retry = 0; retry < 100 && _masterStateR->getAgentReportCount() < 10;
         ++retry) {
        usleep(10000);
    }
    for (size_t retry = 0; retry < 100 && 0 != collectorManager.size(); ++retry) {
        usleep(10000);
    }
    collectorManager.stop();
    ASSERT_EQ(8, reportsBeforeStop);
    const auto reportStats = _masterStateR->getAgentRequestStats();
    ASSERT_EQ(10, reportStats.reportCount);
    ASSERT_EQ(2, reportStats.handshakeCount);
    ASSERT_EQ(2, reportStats.stoppingCount);
    ASSERT_EQ(2, reportStats.cacheUpdateCount);
    ASSERT_EQ(2, reportStats.pendingCacheUpdateCount);
    ASSERT_EQ(2, reportStats.receivedCount);
    ASSERT_EQ(0, _agent0->getSubscriberCount());
    ASSERT_EQ(0, _agent1->getSubscriberCount());
    ASSERT_EQ(0, collectorManager.size());
    ASSERT_EQ(2, nodes.size());
    std::ifstream masterVis("./master_graph.vis", std::ios::binary);
    ASSERT_TRUE(masterVis.is_open());
    navi::GraphVisDef masterVisProto;
    ASSERT_TRUE(masterVisProto.ParseFromIstream(&masterVis));
    ASSERT_GE(masterVisProto.nodes_size(), 2);
    ASSERT_TRUE(containsRemoteKernelMetric(masterVisProto, "prefill_biz_0",
                                           "navi_sched.agent_report_source_k"));
    ASSERT_TRUE(containsRemoteKernelMetric(masterVisProto, "prefill_biz_1",
                                           "navi_sched.agent_report_source_k"));
    auto masterVisString = masterVisProto.SerializeAsString();
    ASSERT_NE(std::string::npos, masterVisString.find(std::to_string(1)));
    ASSERT_NE(std::string::npos, masterVisString.find(std::to_string(2)));
    std::set<int64_t> inflightRequests;
    for (const auto &pair : nodes) {
        const auto &node = pair.second;
        ASSERT_TRUE(node.online);
        ASSERT_EQ(pair.first, node.agentId);
        ASSERT_EQ(hashString(node.agentId), node.agentIdHash);
        ASSERT_EQ(node.agentId, node.collectorAddress);
        if (18000 == node.serverInfo.port) {
            ASSERT_EQ("10.0.0.1:18000", node.serverAddress);
        } else {
            ASSERT_EQ(19000, node.serverInfo.port);
            ASSERT_FALSE(node.serverAddress.empty());
        }
        inflightRequests.emplace(node.inflightRequests);
    }
    ASSERT_EQ(std::set<int64_t>({1, 2}), inflightRequests);
}

TEST_F(NaviSchedClusterTest,
       testAutoCollectorResyncsAfterCacheRevisionGap)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    NaviSchedCacheUpdate cacheUpdate;
    cacheUpdate.addedBlockHashes = makeBlockHashes({11, 22});
    ASSERT_EQ(0, _agent0->reportCacheUpdated(cacheUpdate));

    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(masterNavi);
    MasterCollectorManager collectorManager(masterNavi, _masterStateR,
                                            "master_biz", true);
    navi::ResourceMap resources;
    ASSERT_TRUE(masterNavi->createResource(
        "master_biz", 1, 0, {navi::GigClientR::RESOURCE_ID}, resources));
    auto gigIt = resources.getMap().find(navi::GigClientR::RESOURCE_ID);
    ASSERT_NE(resources.getMap().end(), gigIt);
    auto gigClientR = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    ASSERT_TRUE(gigClientR);
    ASSERT_TRUE(collectorManager.startAutoCollect(gigClientR->getSearchService()));

    auto countHandshakes = [this]() {
        return _masterStateR->getAgentRequestStats().handshakeCount;
    };
    for (size_t retry = 0; retry < 200; ++retry) {
        auto nodes = _masterStateR->getNodes();
        if (1 == nodes.size()) {
            auto node = _masterStateR->getNodeState(
                hashString(nodes.begin()->first));
            if (node && 2 == node->getMatchedBlockCount({11, 22})) {
                break;
            }
        }
        usleep(10000);
    }
    auto nodes = _masterStateR->getNodes();
    ASSERT_EQ(1, nodes.size());
    auto node = _masterStateR->getNodeState(hashString(nodes.begin()->first));
    ASSERT_TRUE(node);
    ASSERT_EQ(2, node->getMatchedBlockCount({11, 22}));
    ASSERT_EQ(1, countHandshakes());

    auto revisionGap = std::make_shared<AgentUpdateToMasterData>();
    revisionGap->event = NaviSchedAgentEvent::CACHE_UPDATED;
    revisionGap->cache = std::make_shared<AgentCacheToMasterData>();
    revisionGap->cache->revision = _agent0->getCacheState().revision + 2;
    revisionGap->cache->cachedBlockCount = 2;
    revisionGap->cache->hashChecksum = 11 ^ 22;
    revisionGap->cache->fullReadyBlockCount = 0;
    revisionGap->cache->tailReadyBlockCount = 0;
    ASSERT_EQ(1, _agent0->publish(revisionGap));

    for (size_t retry = 0; retry < 400 && countHandshakes() < 2; ++retry) {
        usleep(10000);
    }
    ASSERT_GE(countHandshakes(), 2);
    const auto expectedRevision = _agent0->getCacheState().revision;
    for (size_t retry = 0; retry < 400; ++retry) {
        nodes = _masterStateR->getNodes();
        if (1 == nodes.size()) {
            node = _masterStateR->getNodeState(hashString(nodes.begin()->first));
            if (node) {
                const auto stats = node->getCacheSyncStats();
                if (stats.cacheStateReady && stats.agentRevision == expectedRevision) {
                    break;
                }
            }
        }
        usleep(10000);
    }
    ASSERT_EQ(1, nodes.size());
    ASSERT_TRUE(node);
    EXPECT_TRUE(node->getCacheSyncStats().cacheStateReady);
    EXPECT_EQ(expectedRevision, node->getCacheSyncStats().agentRevision);
    EXPECT_EQ(2, node->getMatchedBlockCount({11, 22}));
    EXPECT_EQ(1, _agent0->getSubscriberCount());
    collectorManager.stop();
}

TEST_F(NaviSchedClusterTest,
       testCollectorResyncsOnlyPendingCacheAfterRevisionGap)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    NaviSchedCacheUpdate cacheUpdate;
    cacheUpdate.addedBlockHashes = makeBlockHashes({11, 22});
    ASSERT_EQ(0, _agent0->reportCacheUpdated(cacheUpdate));

    NaviSchedPendingCacheReservation reservation;
    reservation.batchId = 7;
    reservation.scheduleBatchId = 7;
    reservation.fullReadyBlockHashes = {33};
    ASSERT_TRUE(_agent0->reportPendingCacheReady(reservation));

    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(masterNavi);
    MasterCollectorManager collectorManager(masterNavi, _masterStateR,
                                            "master_biz", true);
    ASSERT_TRUE(collectorManager.open("prefill_0", "agent_biz"));
    const auto nodeHash = hashString("prefill_0");
    for (size_t retry = 0; retry < 200; ++retry) {
        auto node = _masterStateR->getNodeState(nodeHash);
        auto pendingStats = _masterPendingCacheR->getStats(nodeHash);
        if (node && node->getInfo().online && pendingStats.agentSnapshotReady &&
            1 == pendingStats.agentBatchCount &&
            3 == node->getMatchedBlockCount({11, 22, 33})) {
            break;
        }
        usleep(10000);
    }
    auto node = _masterStateR->getNodeState(nodeHash);
    ASSERT_TRUE(node);
    ASSERT_TRUE(node->getInfo().online);
    ASSERT_EQ(3, node->getMatchedBlockCount({11, 22, 33}));
    auto countHandshakes = [this]() {
        return _masterStateR->getAgentRequestStats().handshakeCount;
    };
    const auto handshakeCount = countHandshakes();
    const auto realCacheRevision = node->getCacheSyncStats().agentRevision;
    auto initialPendingStats = _masterPendingCacheR->getStats(nodeHash);
    EXPECT_EQ(0, initialPendingStats.agentRevisionGapCount);
    EXPECT_EQ(0, initialPendingStats.agentStateMismatchCount);
    EXPECT_EQ(0,
              node->getCacheSyncStats().pendingCacheResyncTriggerCount);

    auto revisionGap = std::make_shared<AgentUpdateToMasterData>();
    revisionGap->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    revisionGap->pendingCache =
        std::make_shared<AgentPendingCacheToMasterData>();
    revisionGap->pendingCache->revision =
        _agent0->getPendingCacheState().revision + 2;
    revisionGap->pendingCache->removedBatchIds = {7};
    ASSERT_EQ(1, _agent0->publish(revisionGap));

    const auto expectedPendingRevision =
        _agent0->getPendingCacheState().revision;
    for (size_t retry = 0; retry < 200; ++retry) {
        auto stats = _masterPendingCacheR->getStats(nodeHash);
        if (stats.agentSnapshotReady &&
            stats.agentRevision == expectedPendingRevision &&
            1 == stats.agentBatchCount &&
            1 == stats.agentRevisionGapCount) {
            break;
        }
        usleep(10000);
    }
    const auto pendingStats = _masterPendingCacheR->getStats(nodeHash);
    EXPECT_TRUE(pendingStats.agentSnapshotReady);
    EXPECT_EQ(expectedPendingRevision, pendingStats.agentRevision);
    EXPECT_EQ(1, pendingStats.agentBatchCount);
    EXPECT_EQ(1, pendingStats.agentRevisionGapCount);
    EXPECT_EQ(handshakeCount, countHandshakes());
    EXPECT_TRUE(node->getInfo().online);
    EXPECT_TRUE(node->getCacheSyncStats().cacheStateReady);
    EXPECT_EQ(realCacheRevision, node->getCacheSyncStats().agentRevision);
    EXPECT_EQ(3, node->getMatchedBlockCount({11, 22, 33}));
    EXPECT_EQ(1, _agent0->getSubscriberCount());
    EXPECT_EQ(1,
              node->getCacheSyncStats().pendingCacheResyncTriggerCount);

    auto stateMismatch = std::make_shared<AgentUpdateToMasterData>();
    stateMismatch->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    stateMismatch->pendingCache =
        std::make_shared<AgentPendingCacheToMasterData>();
    stateMismatch->pendingCache->revision = expectedPendingRevision + 1;
    stateMismatch->pendingCache->removedBatchIds = {7};
    stateMismatch->pendingCache->checksum = 1;
    ASSERT_EQ(1, _agent0->publish(stateMismatch));

    for (size_t retry = 0; retry < 200; ++retry) {
        auto stats = _masterPendingCacheR->getStats(nodeHash);
        if (stats.agentSnapshotReady &&
            stats.agentRevision == expectedPendingRevision &&
            1 == stats.agentBatchCount &&
            1 == stats.agentStateMismatchCount) {
            break;
        }
        usleep(10000);
    }
    const auto recoveredStats = _masterPendingCacheR->getStats(nodeHash);
    EXPECT_TRUE(recoveredStats.agentSnapshotReady);
    EXPECT_EQ(expectedPendingRevision, recoveredStats.agentRevision);
    EXPECT_EQ(1, recoveredStats.agentBatchCount);
    EXPECT_EQ(1, recoveredStats.agentRevisionGapCount);
    EXPECT_EQ(1, recoveredStats.agentStateMismatchCount);
    EXPECT_EQ(handshakeCount, countHandshakes());
    EXPECT_TRUE(node->getInfo().online);
    const auto recoveredCacheStats = node->getCacheSyncStats();
    EXPECT_TRUE(recoveredCacheStats.cacheStateReady);
    EXPECT_EQ(realCacheRevision, recoveredCacheStats.agentRevision);
    EXPECT_EQ(11 ^ 22, recoveredCacheStats.masterHashChecksum);
    EXPECT_EQ(3, node->getMatchedBlockCount({11, 22, 33}));
    EXPECT_EQ(1, _agent0->getSubscriberCount());
    EXPECT_EQ(2, recoveredCacheStats.pendingCacheResyncTriggerCount);
    collectorManager.stop();

    std::ifstream masterVis("./master_graph.vis", std::ios::binary);
    ASSERT_TRUE(masterVis.is_open());
    navi::GraphVisDef masterVisProto;
    ASSERT_TRUE(masterVisProto.ParseFromIstream(&masterVis));
    EXPECT_TRUE(containsKernel(masterVisProto,
                               "navi_sched.agent_pending_cache_sync_k"));
    EXPECT_GE(countSummarizedDataEvents(
                  masterVisProto,
                  MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID),
              2);
}

TEST_F(NaviSchedClusterTest, testAutoCollectorKeepsEofRoute)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(masterNavi);

    MasterCollectorManager collectorManager(masterNavi, _masterStateR, "master_biz",
                                            true);
    navi::ResourceMap resources;
    ASSERT_TRUE(masterNavi->createResource("master_biz", 1, 0,
                                           {navi::GigClientR::RESOURCE_ID}, resources));
    auto gigIt = resources.getMap().find(navi::GigClientR::RESOURCE_ID);
    ASSERT_NE(resources.getMap().end(), gigIt);
    auto gigClientR = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    ASSERT_TRUE(gigClientR);
    ASSERT_TRUE(collectorManager.startAutoCollect(gigClientR->getSearchService()));
    for (size_t retry = 0; retry < 200 && 0 == _agent0->getSubscriberCount(); ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(1, _agent0->getSubscriberCount());
    ASSERT_EQ(1, _agent0->beginStop(300));
    _agent0->finishAll();
    for (size_t retry = 0; retry < 200 && _masterStateR->getAgentReportCount() < 2;
         ++retry) {
        usleep(10000);
    }
    ASSERT_GE(_masterStateR->getAgentReportCount(), 2);
    for (size_t retry = 0; retry < 100 && 0 != _agent0->getSubscriberCount(); ++retry) {
        usleep(10000);
    }
    ASSERT_EQ(0, _agent0->getSubscriberCount());
    usleep(50000);
    ASSERT_EQ(1, collectorManager.size());
    ASSERT_EQ(0, _agent0->getSubscriberCount());
    collectorManager.stop();
    ASSERT_EQ(0, _agent0->getSubscriberCount());
}

TEST_F(NaviSchedClusterTest,
       testAutoCollectorRestartsUnexpectedEofForOnlineAgent)
{
    ASSERT_NO_FATAL_FAILURE(buildCluster());
    auto masterNavi = _cluster.getNavi("host_1");
    ASSERT_TRUE(masterNavi);

    MasterCollectorManager collectorManager(masterNavi, _masterStateR,
                                            "master_biz", true);
    navi::ResourceMap resources;
    ASSERT_TRUE(masterNavi->createResource(
        "master_biz", 1, 0, {navi::GigClientR::RESOURCE_ID}, resources));
    auto gigIt = resources.getMap().find(navi::GigClientR::RESOURCE_ID);
    ASSERT_NE(resources.getMap().end(), gigIt);
    auto gigClientR = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    ASSERT_TRUE(gigClientR);
    ASSERT_TRUE(collectorManager.startAutoCollect(
        gigClientR->getSearchService()));
    for (size_t retry = 0;
         retry < 200 &&
         (0 == _agent0->getSubscriberCount() ||
          _masterStateR->getNodes().empty());
         ++retry) {
        usleep(10 * 1000);
    }
    ASSERT_EQ(1, _agent0->getSubscriberCount());
    ASSERT_EQ(1, collectorManager.size());
    ASSERT_FALSE(_masterStateR->getNodes().empty());
    const auto initialReportCount = _masterStateR->getAgentReportCount();
    const auto initialStats = _masterStateR->getAgentRequestStats();
    const auto initialNodes = _masterStateR->getNodes();
    const auto initialNode =
        _masterStateR->getNodeState(hashString(initialNodes.begin()->first));
    ASSERT_TRUE(initialNode);

    _agent0->finishAll();
    for (size_t retry = 0;
         retry < 400 &&
         (_agent0->getSubscriberCount() != 1 ||
          _masterStateR->getAgentReportCount() < initialReportCount + 4);
         ++retry) {
        usleep(10 * 1000);
    }
    EXPECT_EQ(1, _agent0->getSubscriberCount());
    EXPECT_GE(_masterStateR->getAgentReportCount(), initialReportCount + 3);
    EXPECT_EQ(1, collectorManager.size());
    const auto nodes = _masterStateR->getNodes();
    ASSERT_FALSE(nodes.empty());
    EXPECT_TRUE(nodes.begin()->second.online);
    const auto replacementStats = _masterStateR->getAgentRequestStats();
    EXPECT_EQ(initialStats.handshakeCount + 1, replacementStats.handshakeCount);
    EXPECT_EQ(initialStats.stoppingCount + 1, replacementStats.stoppingCount);
    EXPECT_NE(initialNode, _masterStateR->getNodeState(hashString(nodes.begin()->first)));
    EXPECT_FALSE(initialNode->getInfo().online);

    collectorManager.stop();
    EXPECT_EQ(0, _agent0->getSubscriberCount());
}
