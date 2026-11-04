#include "rtp_llm/flexlb/navi_sched/graph/MasterCollectorManager.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedAgentEvent.h"
#include "rtp_llm/flexlb/navi_sched/graph/GraphVisRecorder.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentInstanceR.h"
#include "autil/HashAlgorithm.h"
#include "autil/LoopThread.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviUserResult.h"
#include <multi_call/interface/SearchService.h>
#include <set>

namespace navi_sched {
namespace {

const std::string AGENT_SOURCE_NODE = "navi_sched_agent_source";
const std::string MASTER_UPDATE_NODE = "navi_sched_master_update";
const std::string MASTER_GRAPH_VIS_FILE = "./master_graph.vis";
constexpr int64_t COLLECTOR_RETRY_INTERVAL_US = 1000 * 1000;

struct CollectorRoute {
    std::string bizName;
    multi_call::BizNodeInfo node;
    std::string agentMeta;
};

} // namespace

class CollectorResultClosure : public navi::NaviUserResultClosure {
public:
    explicit CollectorResultClosure(const GraphVisRecorderPtr &recorder,
                                    MasterCollectorManager *manager,
                                    bool refreshTopologyOnCompleted)
        : _recorder(recorder)
        , _manager(manager)
        , _refreshTopologyOnCompleted(refreshTopologyOnCompleted)
    {
        waitDestruct = true;
    }

    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        consumeToEof(userResult);
        {
            autil::ScopedLock lock(_cond);
            _userResult = std::move(userResult);
            _completed = true;
            _cond.broadcast();
        }
        if (_refreshTopologyOnCompleted && _manager) {
            _manager->requestRetry();
        }
    }

    bool isCompleted() const
    {
        autil::ScopedLock lock(_cond);
        return _completed;
    }

    bool hasError() const
    {
        autil::ScopedLock lock(_cond);
        auto result = _userResult ? _userResult->getNaviResult() : nullptr;
        return result && result->hasError();
    }

    bool agentStopped() const
    {
        autil::ScopedLock lock(_cond);
        return _agentStopped;
    }

    void waitAndSave() const
    {
        navi::NaviUserResultPtr userResult;
        {
            autil::ScopedLock lock(_cond);
            while (!_completed) {
                _cond.wait();
            }
            userResult = _userResult;
        }
        if (_recorder) {
            _recorder->record(userResult ? userResult->getNaviResult() : nullptr);
        }
    }

private:
    void consumeToEof(const navi::NaviUserResultPtr &userResult)
    {
        if (!userResult) {
            return;
        }
        bool eof = false;
        while (!eof) {
            navi::NaviUserData data;
            userResult->nextData(data, eof);
            auto update =
                std::dynamic_pointer_cast<AgentUpdateToMasterData>(data.data);
            if (update &&
                NaviSchedAgentEvent::AGENT_STOPPING == update->event) {
                _agentStopped = true;
            }
        }
    }

private:
    GraphVisRecorderPtr _recorder;
    MasterCollectorManager *_manager = nullptr;
    bool _refreshTopologyOnCompleted = false;
    mutable autil::ThreadCond _cond;
    navi::NaviUserResultPtr _userResult;
    bool _completed = false;
    bool _agentStopped = false;
};

AUTIL_LOG_SETUP(navi_sched, MasterCollectorManager);

std::string MasterCollectorManager::makeRouteId(const std::string &bizName,
                                                const multi_call::Spec &spec)
{
    return bizName + "@" + spec.getSpecStr(multi_call::MC_PROTOCOL_GRPC_STREAM);
}

bool MasterCollectorManager::isAgentOffline(const MasterStateRPtr &masterStateR,
                                            const std::string &collectorId)
{
    if (!masterStateR || collectorId.empty()) {
        return false;
    }
    std::string nodeId;
    if (!masterStateR->getAgentCollectorNodeId(collectorId, nodeId) ||
        nodeId.empty()) {
        // A graph that failed before its first handshake still needs retries.
        return false;
    }
    auto node = masterStateR->getNodeState(
        autil::HashAlgorithm::hashString64(nodeId.data(), nodeId.size()));
    return !node || !node->getInfo().online;
}

std::unique_ptr<navi::GraphDef> MasterCollectorManager::buildAgentCollectorGraph(
    const std::string &masterBizName, const std::string &agentBizName,
    const std::string &directHost, int32_t grpcStreamPort, const std::string &agentId,
    const std::string &collectorId)
{
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    auto agentGraphId = builder.newSubGraph(agentBizName);
    if (directHost.empty()) {
        builder.subGraph(agentGraphId).location(agentBizName, 1, 0);
    } else {
        builder.subGraph(agentGraphId)
            .directLocation(agentBizName, directHost, grpcStreamPort);
    }
    auto sourceNode = builder.node(AGENT_SOURCE_NODE)
                          .kernel("navi_sched.agent_report_source_k")
                          .attr("collector_id", collectorId);
    auto ackNode = builder.node(agentGraphId, "navi_sched_agent_stop_ack")
                       .kernel("navi_sched.agent_stop_ack_k");
    auto pendingSyncNode =
        builder.node(agentGraphId, "navi_sched_agent_pending_cache_sync")
            .kernel("navi_sched.agent_pending_cache_sync_k");
    auto latencyWarmupNode =
        builder.node(agentGraphId, "navi_sched_agent_latency_model_warmup")
            .kernel("navi_sched.agent_latency_model_warmup_k");
    auto schedulerConfigNode =
        builder.node(agentGraphId, "navi_sched_agent_scheduler_config")
            .kernel("navi_sched.agent_scheduler_config_k");
    auto masterGraphId = builder.newSubGraph(masterBizName);
    builder.subGraph(masterGraphId).location(masterBizName, 1, 0);
    auto updateNode = builder.node(MASTER_UPDATE_NODE)
                          .kernel("navi_sched.master_update_k")
                          .attr("agent_id", agentId)
                          .attr("agent_biz_name", agentBizName)
                          .attr("collector_address", agentId)
                          .attr("collector_id", collectorId);
    sourceNode.out("update").to(updateNode.in("update"));
    updateNode.out("stop_ack").to(ackNode.in("ack"));
    updateNode.out("pending_cache_sync").to(pendingSyncNode.in("request"));
    updateNode.out("latency_warmup").to(latencyWarmupNode.in("warmup"));
    updateNode.out("scheduler_config").to(schedulerConfigNode.in("config"));
    updateNode.out("finished").asGraphOutput("finished");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

void MasterCollectorManager::abortAndWait(
    const navi::NaviUserResultPtr &userResult,
    const std::shared_ptr<CollectorResultClosure> &closure)
{
    if (!closure) {
        return;
    }
    if (userResult && !closure->isCompleted()) {
        userResult->abort(navi::EC_ABORT);
    }
    closure->waitAndSave();
}

MasterCollectorManager::MasterCollectorManager(const navi::NaviPtr &navi,
                                               const MasterStateRPtr &masterStateR,
                                               const std::string &masterBizName,
                                               bool enableVis,
                                               const std::vector<std::string> &taskQueueNames)
    : _navi(navi), _masterStateR(masterStateR), _masterBizName(masterBizName)
    , _taskQueueNames(taskQueueNames)
{
    if (enableVis) {
        _visRecorder = std::make_shared<GraphVisRecorder>(MASTER_GRAPH_VIS_FILE);
    }
}

MasterCollectorManager::~MasterCollectorManager() { stop(); }

bool MasterCollectorManager::open(const std::string &agentId,
                                  const std::string &agentBizName)
{
    return openRoute(agentId, agentId, agentBizName, "", "", -1, "", false);
}

bool MasterCollectorManager::openRoute(const std::string &routeId,
                                       const std::string &agentId,
                                       const std::string &agentBizName,
                                       const std::string &address,
                                       const std::string &directHost,
                                       int32_t grpcStreamPort,
                                       const std::string &agentMeta, bool automatic)
{
    if (!_navi || !_masterStateR || routeId.empty() || agentBizName.empty() ||
        _masterBizName.empty() || (directHost.empty() != (grpcStreamPort <= 0)) ||
        grpcStreamPort > 65535) {
        return false;
    }
    {
        autil::ScopedLock lock(_mutex);
        if (_collectors.end() != _collectors.find(routeId)) {
            return true;
        }
    }
    auto collectorId = makeCollectorId(routeId);
    auto graphDef = buildAgentCollectorGraph(_masterBizName, agentBizName, directHost,
                                             grpcStreamPort, agentId, collectorId);
    if (!graphDef) {
        if (automatic) {
            requestRetry();
        }
        return false;
    }
    navi::RunGraphParams params;
    if (!_taskQueueNames.empty()) {
        const auto hash = autil::HashAlgorithm::hashString64(agentId.data(), agentId.size());
        params.setTaskQueueName(_taskQueueNames[hash % _taskQueueNames.size()]);
    }
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    // Streaming collectors must not retain every log until their graph ends.
    if (_visRecorder) {
        params.setCollectMetric(true);
        params.addParamTopoTreeStr("edge_key=*;edge_query=2");
    }
    auto closure = std::make_shared<CollectorResultClosure>(
        _visRecorder, automatic ? this : nullptr, automatic);
    auto userResult =
        _navi->runLocalGraphAsync(graphDef.release(), params, {}, closure.get());
    if (!userResult) {
        if (automatic) {
            requestRetry();
        }
        return false;
    }
    bool inserted = false;
    {
        autil::ScopedLock lock(_mutex);
        auto result = _collectors.emplace(
            routeId,
            Collector{collectorId, agentId, agentBizName, address, directHost,
                      grpcStreamPort, automatic, agentMeta, closure, userResult});
        inserted = result.second;
    }
    if (!inserted) {
        abortAndWait(userResult, closure);
        return true;
    }
    if (agentId.empty() || automatic) {
        return true;
    }
    return _masterStateR->openAgentCollector(agentId, collectorId);
}

bool MasterCollectorManager::close(const std::string &agentId)
{
    return closeRoute(agentId, true);
}

bool MasterCollectorManager::closeRoute(const std::string &routeId, bool markOffline)
{
    Collector collector;
    navi::NaviUserResultPtr userResult;
    {
        autil::ScopedLock lock(_mutex);
        auto it = _collectors.find(routeId);
        if (_collectors.end() == it) {
            return false;
        }
        collector = it->second;
        _collectors.erase(it);
    }
    abortAndWait(collector.userResult, collector.closure);
    if (_visRecorder) {
        _visRecorder->save();
    }
    std::string nodeId;
    auto bound = _masterStateR->getAgentCollectorNodeId(collector.collectorId, nodeId);
    if (bound) {
        if (markOffline && collector.automatic) {
            _masterStateR->stopAgentCollector(nodeId, collector.collectorId);
        } else {
            _masterStateR->closeAgentCollector(nodeId, collector.collectorId);
        }
    }
    _masterStateR->unbindAgentCollector(collector.collectorId);
    return true;
}

bool MasterCollectorManager::startAutoCollect(
    const multi_call::SearchServicePtr &searchService)
{
    if (!searchService || _searchService) {
        return false;
    }
    _stopping.store(false, std::memory_order_release);
    _searchService = searchService;
    _snapshotChangeListenerId =
        _searchService->addSnapshotChangeListener([this]() { refreshTopology(); });
    if (0 == _snapshotChangeListenerId) {
        _searchService.reset();
        return false;
    }
    _retryThread =
        autil::LoopThread::createLoopThread([this]() { retryFailedCollectors(); },
                                            COLLECTOR_RETRY_INTERVAL_US, "sched_retry");
    if (!_retryThread) {
        _searchService->removeSnapshotChangeListener(_snapshotChangeListenerId);
        _snapshotChangeListenerId = 0;
        _searchService.reset();
        return false;
    }
    refreshTopology();
    return true;
}

void MasterCollectorManager::stop()
{
    _stopping.store(true, std::memory_order_release);
    if (_retryThread) {
        _retryThread->stop();
        _retryThread.reset();
    }
    autil::ScopedLock topologyLock(_topologyLock);
    if (_searchService && 0 != _snapshotChangeListenerId) {
        _searchService->removeSnapshotChangeListener(_snapshotChangeListenerId);
        _snapshotChangeListenerId = 0;
    }
    std::map<std::string, Collector> collectors;
    {
        autil::ScopedLock lock(_mutex);
        collectors.swap(_collectors);
    }
    for (const auto &pair : collectors) {
        abortAndWait(pair.second.userResult, pair.second.closure);
        if (_masterStateR) {
            std::string nodeId;
            auto bound =
                _masterStateR->getAgentCollectorNodeId(pair.second.collectorId, nodeId);
            if (bound) {
                _masterStateR->closeAgentCollector(nodeId,
                                                   pair.second.collectorId);
            }
            _masterStateR->unbindAgentCollector(pair.second.collectorId);
        }
    }
    if (_visRecorder) {
        _visRecorder->save();
    }
    _searchService.reset();
    _retryRequested.store(false, std::memory_order_release);
}

size_t MasterCollectorManager::size() const
{
    autil::ScopedLock lock(_mutex);
    size_t count = 0;
    for (const auto &collector : _collectors) {
        const auto &route = collector.second;
        // Auto-discovered routes remain owned by Gig after a normal EOF.
        if (route.automatic || (route.closure && !route.closure->isCompleted())) {
            ++count;
        }
    }
    return count;
}

std::string MasterCollectorManager::makeCollectorId(const std::string &routeId)
{
    autil::ScopedLock lock(_mutex);
    return routeId + ":" + std::to_string(++_nextCollectorId);
}

void MasterCollectorManager::requestRetry()
{
    if (!_stopping.load(std::memory_order_acquire)) {
        _retryRequested.store(true, std::memory_order_release);
    }
}

void MasterCollectorManager::retryFailedCollectors()
{
    if (_retryRequested.exchange(false, std::memory_order_acq_rel)) {
        refreshTopology();
    }
}

void MasterCollectorManager::refreshTopology()
{
    if (_stopping.load(std::memory_order_acquire)) {
        return;
    }
    autil::ScopedLock topologyLock(_topologyLock);
    if (!_searchService || _stopping.load(std::memory_order_acquire)) {
        return;
    }
    std::map<std::string, CollectorRoute> routes;
    for (const auto &bizName : _searchService->getBizNames()) {
        for (const auto &node : _searchService->getBizNodeInfos(bizName)) {
            auto grpcStreamPort = node.spec.ports[multi_call::MC_PROTOCOL_GRPC_STREAM];
            if (grpcStreamPort > 0) {
                std::string agentMeta;
                if (node.metaMap) {
                    auto metaIt = node.metaMap->find(AgentInstanceR::RESOURCE_ID);
                    if (node.metaMap->end() != metaIt) {
                        agentMeta = metaIt->second;
                    }
                }
                if (agentMeta.empty()) {
                    continue;
                }
                routes.emplace(makeRouteId(bizName, node.spec),
                               CollectorRoute{bizName, node, agentMeta});
            }
        }
    }
    std::map<std::string, std::string> routeMetas;
    for (const auto &route : routes) {
        routeMetas.emplace(route.first, route.second.agentMeta);
    }
    syncRoutes(routeMetas);
    for (const auto &route : routes) {
        const auto &bizName = route.second.bizName;
        const auto &node = route.second.node;
        auto grpcStreamPort = node.spec.ports[multi_call::MC_PROTOCOL_GRPC_STREAM];
        openRoute(route.first, node.spec.getSpecStr(multi_call::MC_PROTOCOL_GRPC_STREAM),
                  bizName, node.spec.getSpecStr(multi_call::MC_PROTOCOL_GRPC_STREAM),
                  node.spec.ip, grpcStreamPort, route.second.agentMeta, true);
    }
}

void MasterCollectorManager::syncRoutes(
    const std::map<std::string, std::string> &routeMetas)
{
    struct RouteToClose {
        std::string routeId;
        std::string agentId;
        bool markOffline = false;
        bool resetInstance = false;
    };
    std::vector<RouteToClose> routesToClose;
    {
        autil::ScopedLock lock(_mutex);
        for (const auto &collector : _collectors) {
            if (!collector.second.automatic) {
                continue;
            }
            auto routeIt = routeMetas.find(collector.first);
            if (routeMetas.end() == routeIt) {
                routesToClose.emplace_back(
                    RouteToClose{collector.first, collector.second.agentId, true, false});
            } else if (!routeIt->second.empty() &&
                       collector.second.agentMeta != routeIt->second) {
                routesToClose.emplace_back(
                    RouteToClose{collector.first, collector.second.agentId, true, true});
            } else if (collector.second.closure &&
                       collector.second.closure->isCompleted() &&
                       !collector.second.closure->agentStopped() &&
                       !isAgentOffline(_masterStateR,
                                       collector.second.collectorId)) {
                if (collector.second.closure->hasError()) {
                    AUTIL_LOG(ERROR,
                              "restart failed automatic collector, routeId [%s], "
                              "collectorId [%s], agentId [%s]",
                              collector.first.c_str(),
                              collector.second.collectorId.c_str(),
                              collector.second.agentId.c_str());
                } else {
                    AUTIL_LOG(WARN,
                              "restart unexpectedly completed automatic collector, "
                              "routeId [%s], collectorId [%s], agentId [%s]",
                              collector.first.c_str(),
                              collector.second.collectorId.c_str(),
                              collector.second.agentId.c_str());
                }
                routesToClose.emplace_back(RouteToClose{
                    collector.first, collector.second.agentId, false, false});
            }
        }
    }
    for (const auto &route : routesToClose) {
        closeRoute(route.routeId, route.markOffline);
        if (route.resetInstance) {
            _masterStateR->resetAgentInstance(route.agentId);
        }
    }
}

} // namespace navi_sched
