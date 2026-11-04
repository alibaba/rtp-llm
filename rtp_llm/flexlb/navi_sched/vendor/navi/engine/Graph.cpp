#include "navi/engine/PartInfoDefOverride.h"
#include "navi/engine/Graph.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "navi/engine/GraphDomainClient.h"
#include "navi/engine/GraphDomainFork.h"
#include "navi/engine/GraphDomainLocal.h"
#include "navi/engine/GraphDomainUser.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/LocalSubGraphInternal.h"
#include "navi/engine/RemoteSubGraphBase.h"
#include "navi/engine/Node.h"
#include "navi/log/TraceAppender.h"
#include "navi/resource/TimeoutCheckerR.h"
#include "navi/resource/QuerySessionR.h"

#include "navi/ops/ResourceKernelDef.h"
#include "navi/util/CommonUtil.h"
#include "autil/HashUtil.h"

using namespace multi_call;

namespace navi {

class TimeoutItem : public NaviNoDropWorkerItem
{
public:
    TimeoutItem(NaviWorkerBase *worker, const CollectFlags &collectFlags, Graph *graph)
        : NaviNoDropWorkerItem(worker, collectFlags)
        , _graph(graph)
    {
    }
public:
    void doProcess(NaviPerfThread *perfThread) override {
        if (_graph->checkTimeout(true)) {
            _worker->decItemCount();
        }
    }
private:
    Graph *_graph;
};

Graph::Graph(GraphParam *param, Graph *parent, bool rootGraph)
    : _logger(this, "graph", param->logger)
    , _partInfoDefGraphMap(std::make_unique<PartInfoDefGraphMap>())
    , _param(param)
    , _parent(parent)
    , _forkNode(nullptr)
    , _errorAsEof(false)
    , _terminated(false)
    , _timerStopped(false)
    , _rootGraph(rootGraph)
    , _timer(nullptr)
{
    atomic_set(&_timerRefreshVersion, 0);
    _metricCollected = false;
}

Graph::~Graph() {
    for (auto subGraph : _subGraphs) {
        delete subGraph;
    }
    DELETE_AND_SET_NULL(_param);
}

ErrorCode Graph::init(GraphDef *graphDef, bool keepGraphDef) {
    auto ec = doInit(graphDef, keepGraphDef);
    if (_param->runParams.getCollectFlags().needCollect()) {
        _param->graphResult->setGraphDef(graphDef);
    }
    return ec;
}

ErrorCode Graph::doInit(GraphDef *graphDef, bool keepGraphDef) {
    NAVI_LOG(SCHEDULE1, "graph def [%s]", graphDef->DebugString().c_str());
    autil::ScopedLock lock(_terminateLock);
    NaviLoggerScope scope(_logger);
    PartInfoDefGraphMapScope partInfoDefGraphMapScope(_partInfoDefGraphMap.get());
    _param->graphResult->setGraph(this);
    auto ec = initGraphDef(graphDef, keepGraphDef);
    if (EC_NONE != ec) {
        return ec;
    }
    if (_copyGraphDef) {
        graphDef = _copyGraphDef;
    }
    initTimeoutChecker();
    if (!preInitGraphDomains(graphDef)) {
        return EC_INIT_GRAPH;
    }
    if (!initBorder()) {
        return EC_INIT_GRAPH;
    }
    if (!postInitGraphDomains()) {
        return EC_INIT_GRAPH;
    }
    ec = createSubGraphs(graphDef);
    if (EC_NONE != ec) {
        return ec;
    }
    NAVI_LOG(SCHEDULE1,
             "after init graph def [%s], graph border [%s]",
             graphDef->DebugString().c_str(),
             autil::StringUtil::toString(*_border).c_str());
    return initNaviUserResult();
}

ErrorCode Graph::initGraphDef(GraphDef *graphDef, bool keepGraphDef) {
    ResourceGraphBuilderPtr resourceGraphBuilder;
    auto subCount = graphDef->sub_graphs_size();
    NAVI_LOG(TRACE3, "sub graph count [%d]", subCount);
    for (int32_t i = 0; i < subCount; i++) {
        auto subGraphDef = graphDef->mutable_sub_graphs(i);
        NAVI_LOG(SCHEDULE2, "sub graph before rewrite: %s", subGraphDef->DebugString().c_str());
        auto ec = rewrite(resourceGraphBuilder, subGraphDef);
        if (EC_NONE != ec) {
            return ec;
        }
        NAVI_LOG(SCHEDULE2, "sub graph after rewrite: %s", subGraphDef->DebugString().c_str());
    }
    if (resourceGraphBuilder && resourceGraphBuilder->needBuild()) {
        auto useGraphDef = graphDef;
        if (keepGraphDef) {
            _copyGraphDef = new navi::GraphDef();
            _copyGraphDef->CopyFrom(*graphDef);
            useGraphDef = _copyGraphDef;
        }
        GraphBuilder builder(useGraphDef);
        if (!resourceGraphBuilder->build(builder)) {
            NAVI_LOG(ERROR, "build resource graph failed");
            return EC_REWRITE_GRAPH;
        }
        _graphInlineBuildMap = resourceGraphBuilder->getGraphInlineBuildMap();
        auto resourceGraph = builder.getSubGraphDef(RS_GRAPH_RESOURCE_GRAPH_ID);
        if (resourceGraph) {
            initThisPartId(resourceGraph);
            if (!prepareRewriteInfo(
                    resourceGraph->graph_id(), resourceGraph, 1, 0)) {
                return EC_REWRITE_GRAPH;
            }
            const auto &borderMap = builder.getBorderMap();
            for (const auto &pair : borderMap) {
                const auto &borderId = pair.first;
                if (RS_GRAPH_RESOURCE_GRAPH_ID == borderId.id ||
                    RS_GRAPH_RESOURCE_GRAPH_ID == borderId.peer) {
                    auto rewriteIt = _rewriteInfoMap.find(borderId.id);
                    if (_rewriteInfoMap.end() != rewriteIt) {
                        rewriteIt->second->needCacheGraph = false;
                    }
                    rewriteIt = _rewriteInfoMap.find(borderId.peer);
                    if (_rewriteInfoMap.end() != rewriteIt) {
                        rewriteIt->second->needCacheGraph = false;
                    }
                }
            }
        }
    }
    for (const auto &pair : _rewriteInfoMap) {
        if (pair.second->needCacheGraph && pair.second->cacheGraph) {
            auto inlineBuildMap = pair.second->cacheGraph->inlineBuildMap;
            if (inlineBuildMap) {
                if (!_graphInlineBuildMap) {
                    _graphInlineBuildMap = std::make_shared<GraphInlineBuildMap>();
                }
                for (const auto &p : *inlineBuildMap) {
                    if (p.first.graphId != pair.first &&
                        p.first.graphId != RS_GRAPH_RESOURCE_GRAPH_ID) {
                        continue;
                    }
                    auto iter = _graphInlineBuildMap->find(p.first);
                    if (_graphInlineBuildMap->end() == iter) {
                        _graphInlineBuildMap->emplace(p);
                    } else {
                        iter->second.insert(p.second.begin(), p.second.end());
                    }
                }
            }
        }
    }
    return EC_NONE;
}

ErrorCode Graph::rewrite(ResourceGraphBuilderPtr &builder, SubGraphDef *subGraphDef) {
    bool isLocalGraph = false;
    NaviPartId partCount = INVALID_NAVI_PART_COUNT;
    NaviPartId partId = INVALID_NAVI_PART_ID;
    if (!tryInitThisPartIdFromPartIds(subGraphDef, isLocalGraph, partCount, partId)) {
        return EC_REWRITE_GRAPH;
    }
    if (!isLocalGraph) {
        NAVI_LOG(TRACE3, "not local sub graph [%d], rewrite skipped", subGraphDef->graph_id());
        return EC_NONE;
    }
    auto rewriteInfo = prepareRewriteInfo(subGraphDef->graph_id(), subGraphDef, partCount, partId);
    if (!rewriteInfo) {
        return EC_REWRITE_GRAPH;
    }
    auto subGraphKey = calcSubGraphKey(*subGraphDef);
    rewriteInfo->graphKey = subGraphKey;
    rewriteInfo->cacheGraph = rewriteInfo->biz->getCacheGraph(subGraphKey);
    rewriteInfo->needCacheGraph = (subGraphKey != 0);
    if (rewriteInfo->cacheGraph && rewriteInfo->cacheGraph->replaceInfoMap) {
        rewriteInfo->replaceInfoMap = rewriteInfo->cacheGraph->replaceInfoMap;
    } else {
        if (subGraphDef && subGraphDef->replace_infos_size() > 0) {
            rewriteInfo->replaceInfoMap = std::make_shared<ReplaceInfoMap>();
            rewriteInfo->replaceInfoMap->init(subGraphDef->replace_infos());
            rewriteInfo->replaceInfoMap->removeDup(rewriteInfo->biz->getReplaceInfoMap());
        }
    }
    if (rewriteInfo->cacheGraph) {
        NAVI_LOG(DEBUG,
                 "subgraph[%d] hit biz[%s] graph cache",
                 subGraphDef->graph_id(),
                 getBizName(*subGraphDef).c_str());
        return EC_NONE;
    }
    rewriteInfo->nodeCountBeforeAddResource = subGraphDef->nodes_size();
    if (!builder) {
        builder = std::make_shared<ResourceGraphBuilder>(_param->runParams);
    }
    auto ec = addResourceNode(subGraphDef, *builder, *rewriteInfo);
    if (EC_NONE != ec) {
        return ec;
    }
    return EC_NONE;
}

size_t Graph::calcSubGraphKey(const SubGraphDef &subGraph) {
    auto hashKey = subGraph.hash_key();
    if (0 == hashKey) {
        return 0;
    }
    autil::HashUtil::combineHash(hashKey, _param->runParams.getResourceBuildMode());
    const auto &buildModeMap = _param->runParams.getResourceBuildModeMap();
    autil::HashUtil::combineHash(hashKey, buildModeMap.size());
    for (const auto &pair : buildModeMap) {
        autil::HashUtil::combineHash(hashKey, pair.first.size());
        autil::HashUtil::combineHash(hashKey, pair.first);
        autil::HashUtil::combineHash(hashKey, pair.second);
    }
    return hashKey;
}

RewriteInfoPtr Graph::prepareRewriteInfo(GraphId graphId, SubGraphDef *subGraphDef,
        NaviPartId partCount, NaviPartId partId)
{
    BizPtr biz;
    if (subGraphDef) {
        biz = getBiz(*subGraphDef);
        if (!biz) {
            NAVI_LOG(ERROR, "get biz failed, subGraph [%d]", graphId);
            return nullptr;
        }
    } else {
        biz = _param->bizManager->getBuildinBiz();
    }
    auto resourceMap = std::make_shared<ResourceMap>();
    if (!validateInputResource(biz)) {
        return nullptr;
    }
    if (!fillInputResource(graphId, resourceMap)) {
        return nullptr;
    }
    auto rewriteInfo = std::make_shared<RewriteInfo>();
    rewriteInfo->biz = biz;
    rewriteInfo->resourceMap = resourceMap;
    rewriteInfo->multiLayerResourceMap.addResourceMap(resourceMap.get());
    if (!biz->getResourceMap(partCount, partId, rewriteInfo->bizResourceMap)) {
        NAVI_LOG(ERROR, "collectResourceMap failed for sub graph [%d]", graphId);
        return nullptr;
    }
    _rewriteInfoMap[graphId] = rewriteInfo;
    return rewriteInfo;
}

ErrorCode Graph::addResourceNode(SubGraphDef *subGraphDef, ResourceGraphBuilder &builder, const RewriteInfo &rewriteInfo) {
    try {
        auto graphId = subGraphDef->graph_id();
        const auto &biz = rewriteInfo.biz;
        const auto &subGraph = *subGraphDef;
        auto nodeCount = subGraph.nodes_size();
        ResourceBuildParam param(rewriteInfo);
        param.graphId = graphId;
        for (auto i = 0; i < nodeCount; i++) {
            const auto &node = subGraph.nodes(i);
            const auto &kernelName = node.kernel_name();
            auto kernelCreator = biz->getKernelCreator(kernelName);
            if (!kernelCreator) {
                if (kernelName.empty()) {
                    NAVI_LOG(ERROR, "kernel creator not found, kernel name not set, node [%s]", node.name().c_str());
                } else {
                    NAVI_LOG(ERROR,
                             "kernel creator not found for kernel [%s] node [%s]",
                             kernelName.c_str(),
                             node.name().c_str());
                }
                return EC_CREATE_KERNEL;
            }
            param.scope = node.scope();
            param.requireNodeId = (void *)&node;
            param.requireNodeIndex = i + 1;
            {
                const auto &dependResourceVec = kernelCreator->getDependResourcesWithoutSelect();
                for (const auto &pair : dependResourceVec) {
                    const auto &resource = pair.first;
                    builder.addResource(param, resource, nullptr, -1);
                    NAVI_LOG(SCHEDULE2,
                             "add depend resource [%s], depend by kernel [%s]",
                             resource.c_str(),
                             kernelName.c_str());
                }
            }
            {
                const auto &dependResourceVec = kernelCreator->getSelectDependResourceVec();
                auto count = dependResourceVec.size();
                for (int32_t i = 0; i < count; i++) {
                    const auto &resource = dependResourceVec[i].first;
                    builder.addResource(param, resource, nullptr, i);
                    NAVI_LOG(SCHEDULE2,
                             "add select depend resource [%s], depend by kernel [%s]",
                             resource.c_str(),
                             kernelName.c_str());
                }
            }
        }
        return EC_NONE;
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_LOG(ERROR, "add resource graph failed, msg [%s]", e.GetMessage().c_str());
        return EC_REWRITE_GRAPH;
    }
}

bool Graph::preInitGraphDomains(GraphDef *graphDef) {
    auto counterInfo = graphDef->mutable_counter_info();
    auto subCount = graphDef->sub_graphs_size();
    for (int32_t i = 0; i < subCount; i++) {
        auto subGraphDef = graphDef->mutable_sub_graphs(i);
        const auto &location = subGraphDef->location();
        auto graphId = subGraphDef->graph_id();
        GraphDomainPtr domain;
        auto it = _graphDomainMap.find(graphId);
        if (_graphDomainMap.end() != it) {
            domain = it->second;
        } else {
            domain = createGraphDomain(graphId, location);
            domain->setCounterInfo(counterInfo);
            _graphDomainMap.emplace(graphId, domain);
        }
        bindGraphDomain(domain, subGraphDef);
        _domainSet.insert(domain);
    }
    for (const auto &domain : _domainSet) {
        if (!domain->preInit()) {
            NAVI_LOG(ERROR, "domain preInit failed");
            return false;
        }
    }
    return true;
}

bool Graph::postInitGraphDomains() {
    for (const auto &pair : _graphDomainMap) {
        if (!pair.second->postInit()) {
            return false;
        }
        NAVI_LOG(SCHEDULE1, "end post init domain [%s]",
                 autil::StringUtil::toString(*pair.second).c_str());
    }
    return true;
}

bool Graph::runGraphDomains() {
    for (const auto &pair : _graphDomainMap) {
        if (!pair.second->run()) {
            return false;
        }
    }
    return true;
}

GraphDomainPtr Graph::createGraphDomain(GraphId graphId, const LocationDef &location) {
    auto domain = getGraphDomain(graphId);
    if (domain) {
        return domain;
    }
    const auto &bizName = location.biz_name();
    if (NAVI_USER_GRAPH_BIZ == bizName) {
        domain = createUserDomain(bizName);
    } else if (_rewriteInfoMap.end() != _rewriteInfoMap.find(graphId)) {
        domain = createLocalDomain(bizName);
    } else {
        domain = createRemoteDomain(bizName);
    }
    return domain;
}

GraphDomainPtr Graph::createUserDomain(const std::string &bizName) {
    return GraphDomainPtr(new GraphDomainUser(this));
}

GraphDomainPtr Graph::createLocalDomain(const std::string &bizName) {
    return GraphDomainPtr(new GraphDomainLocal(this));
}

GraphDomainPtr Graph::createRemoteDomain(const std::string &bizName) {
    return GraphDomainPtr(new GraphDomainClient(this, bizName));
}

void Graph::bindGraphDomain(const GraphDomainPtr &domain,
                            SubGraphDef *subGraphDef)
{
    domain->setSubGraph(subGraphDef);
    _subGraphDomainMap.emplace(subGraphDef, domain);
}

bool Graph::initGraphInlineResource() {
    if (!_graphInlineBuildMap) {
        NAVI_LOG(SCHEDULE1, "inline build map is empty");
        return true;
    }
    ResourceNodeId id;
    id.graphId = RS_GRAPH_RESOURCE_GRAPH_ID;
    auto it = _graphInlineBuildMap->find(id);
    if (_graphInlineBuildMap->end() == it) {
        NAVI_LOG(SCHEDULE1, "no inline build graph resource");
        return true;
    }
    RewriteInfoPtr rewriteInfo;
    {
        auto rewriteIt = _rewriteInfoMap.find(id.graphId);
        if (_rewriteInfoMap.end() == rewriteIt) {
            rewriteInfo = prepareRewriteInfo(RS_GRAPH_RESOURCE_GRAPH_ID, nullptr, 1, 0);
            if (!rewriteInfo) {
                NAVI_LOG(ERROR, "prepareRewriteInfo for buildin biz failed");
                return false;
            }
        } else {
            rewriteInfo = rewriteIt->second;
        }
    }
    auto resourceMap = std::make_shared<ResourceMap>();
    auto multiLayerResourceMap = rewriteInfo->multiLayerResourceMap;
    multiLayerResourceMap.addResourceMap(rewriteInfo->bizResourceMap.get());
    auto namedDataMap = _param->runParams.getSubNamedDataMap(RS_GRAPH_RESOURCE_GRAPH_ID);
    const auto &buildinBiz = _param->bizManager->getBuildinBiz();
    const auto &resourceVec = it->second;
    for (auto namePtr : resourceVec) {
        auto resource = buildinBiz->buildResourceRecur(1,
                                                       0,
                                                       *namePtr,
                                                       RS_GRAPH,
                                                       nullptr,
                                                       namedDataMap,
                                                       multiLayerResourceMap,
                                                       nullptr,
                                                       nullptr,
                                                       *resourceMap);
        NAVI_LOG(SCHEDULE1, "inline build graph resource [%s] [%p]", namePtr->c_str(), resource);
    }
    for (const auto &pair : _rewriteInfoMap) {
        pair.second->multiLayerResourceMap.addResourceMap(resourceMap.get());
    }
    _inlineResourceMap = resourceMap;
    return true;
}

ErrorCode Graph::createSubGraphs(GraphDef *graphDef) {
    auto subCount = graphDef->sub_graphs_size();
    for (int32_t i = 0; i < subCount; i++) {
        auto subGraphDef = graphDef->mutable_sub_graphs(i);
        auto domain = getGraphDomain(subGraphDef->graph_id());
        if (!domain) {
            NAVI_LOG(ERROR, "get graph [%d] domain failed", subGraphDef->graph_id());
            return EC_UNKNOWN;
        }
        auto subGraph = createSubGraph(subGraphDef, domain);
        if (!subGraph) {
            NAVI_LOG(ERROR, "create graph failed");
            return EC_UNKNOWN;
        }
        addSubGraph(subGraph);
        auto ec = subGraph->init();
        if (EC_NONE != ec) {
            return ec;
        }
    }
    return EC_NONE;
}

SubGraphBase *Graph::createSubGraph(SubGraphDef *subGraphDef,
                                    const GraphDomainPtr &domain)
{
    auto graphId = subGraphDef->graph_id();
    auto borderMap = _border->getBorderMap(graphId);
    if (!borderMap) {
        NAVI_LOG(ERROR, "border map not found for sub graph [%d]", graphId);
        return nullptr;
    }
    auto type = domain->getType();
    switch (type) {
    case GDT_LOCAL:
    case GDT_FORK:
    {
        auto it = _rewriteInfoMap.find(graphId);
        if (_rewriteInfoMap.end() == it) {
            NAVI_LOG(ERROR, "rewrite info not found for sub graph [%d]", graphId);
            return nullptr;
        }
        return new LocalSubGraph(subGraphDef, _param, domain, borderMap, this, it->second, _graphInlineBuildMap);
    }
    case GDT_CLIENT:
    case GDT_SERVER:
    case GDT_USER:
        return new RemoteSubGraphBase(subGraphDef, _param, domain, borderMap, this);
    default:
        return nullptr;
    }
}

bool Graph::validateInputResource(const BizPtr &biz) const {
    if (_param->worker->getTestMode() != TM_NONE) {
        return true;
    }
    bool ret = true;
    const auto &map = _inputResourceMap.getMap();
    for (const auto &pair : map) {
        const auto &name = pair.first;
        auto stage = biz->getResourceStage(name);
        if (RS_UNKNOWN != stage && RS_EXTERNAL != stage && RS_RUN_GRAPH_EXTERNAL != stage) {
            NAVI_LOG(ERROR, "invalid input stage [%s], resource [%s]", ResourceStage_Name(stage).c_str(), name.c_str());
            ret = false;
        }
    }
    return ret;
}

bool Graph::tryInitThisPartIdFromPartIds(SubGraphDef *subGraphDef,
                                         bool &isLocalGraph,
                                         NaviPartId &partCount,
                                         NaviPartId &partId) {
    auto graphId = subGraphDef->graph_id();
    auto domain = getGraphDomain(graphId);
    if (domain) {
        auto type = domain->getType();
        if (GDT_SERVER == type) {
            isLocalGraph = false;
            return true;
        } else if (GDT_FORK == type) {
            auto forkDomain = dynamic_cast<GraphDomainFork *>(domain.get());
            if (!forkDomain) {
                NAVI_LOG(ERROR, "get fork domain failed");
                return false;
            }
            isLocalGraph = true;
            partCount = forkDomain->getPartCount();
            partId = forkDomain->getPartId();
            initThisPartId(subGraphDef);
            return true;
        } else {
            NAVI_LOG(ERROR, "invalid domain type [%d]", type);
            return false;
        }
    }
    auto location = subGraphDef->mutable_location();
    auto thisPartId = location->this_part_id();
    auto partInfo = location->mutable_part_info();
    if (INVALID_NAVI_PART_ID == thisPartId) {
        if (1 == partInfo->part_ids_size()) {
            location->set_this_part_id(partInfo->part_ids(0));
        }
    }
    isLocalGraph = isLocal(*location);
    if (isLocalGraph) {
        if (INVALID_NAVI_PART_COUNT == partInfo->part_count()) {
            auto biz = getBiz(*subGraphDef);
            if (!biz) {
                return false;
            }
            partInfo->set_part_count(biz->getPartCount());
        }
        _localBizName = location->biz_name();
        _localPartId = location->this_part_id();
    }
    initThisPartId(subGraphDef);
    partCount = partInfo->part_count();
    partId = location->this_part_id();
    return true;
}

void Graph::initThisPartId(SubGraphDef *subGraphDef) {
    auto location = subGraphDef->mutable_location();
    if (INVALID_NAVI_PART_ID == location->this_part_id()) {
        location->set_this_part_id(0);
    }
}

void Graph::addSubGraph(SubGraphBase *subGraph) {
    _subGraphs.push_back(subGraph);
}

bool Graph::initBorder() {
    GraphBorderPtr border(new GraphBorder(_logger.logger, _param));
    for (const auto &pair : _subGraphDomainMap) {
        if (!border->collectSubGraph(*pair.first, pair.second)) {
            return false;
        }
    }
    if (!border->weave()) {
        return false;
    }
    _border = border;
    return true;
}

ErrorCode Graph::initNaviUserResult() {
    if (getGraphDomain(PARENT_GRAPH_ID)) {
        return EC_NONE;
    }
    const auto &result = _param->worker->getUserResult();
    auto domain = getGraphDomain(USER_GRAPH_ID);
    if (!domain) {
        return EC_NONE;
    }
    if (GDT_SERVER == domain->getType()) {
        return EC_NONE;
    }
    auto userDomain = std::dynamic_pointer_cast<GraphDomainUser>(domain);
    if (!userDomain) {
        NAVI_LOG(ERROR, "invalid domain type, expect GraphDomainUser [%p]", domain.get());
        return EC_INIT_GRAPH;
    }
    if (!result->bindDomain(_param->runParams.getCollectFlags().collectMalloc(),
                            userDomain)) {
        NAVI_LOG(ERROR, "result bind domain failed");
        return EC_INIT_GRAPH;
    }
    return EC_NONE;
}

ErrorCode Graph::run() {
    NaviLoggerScope scope(_logger);
    if (!validOverrideData()) {
        return EC_OVERRIDE_EDGE_DATA;
    }
    if (!initGraphInlineResource()) {
        return EC_INIT_GRAPH;
    }
    if (!initTimer()) {
        return EC_TIMEOUT;
    }
    autil::ScopedLock lock(_terminateLock);
    for (auto subGraph : _subGraphs) {
        auto ec = subGraph->run();
        if (EC_NONE != ec) {
            return ec;
        }
    }
    if (!runGraphDomains()) {
        return EC_INIT_GRAPH;
    }
    return EC_NONE;
}

bool Graph::validOverrideData() const {
    const auto &overrideDatas = _param->runParams.getOverrideDatas();
    if (overrideDatas.empty()) {
        return true;
    }
    std::set<GraphId> graphIdSet;
    for (auto subGraph : _subGraphs) {
        graphIdSet.insert(subGraph->getGraphId());
    }
    auto ret = true;
    for (const auto &overrideData : overrideDatas) {
        auto graphId = overrideData.graphId;
        if (graphIdSet.end() == graphIdSet.find(graphId)) {
            NAVI_LOG(ERROR,
                     "override graph id [%d] not exist, output node [%s], "
                     "output port [%s]",
                     graphId, overrideData.outputNode.c_str(),
                     overrideData.outputPort.c_str());
            ret = false;
        }
    }
    return ret;
}

void Graph::notifyTimeout(bool notifyParent) {
    if (_parent) {
        NAVI_LOG(ERROR, "fork graph timeout, notifyParent: [%d]", notifyParent);
    } else {
        NAVI_LOG(ERROR, "session timeout");
    }
    notifyFinish(getError(EC_TIMEOUT), notifyParent ? GFT_THIS_FINISH : GFT_PARENT_FINISH);
}

bool Graph::checkTimeout(bool notifyParent) {
    constexpr double MS_TO_US = 1000.0;
    auto worker = _param->worker;
    auto evTimer = worker->getEvTimer();
    if (!evTimer) {
        NAVI_KERNEL_LOG(ERROR, "get evTimer from worker failed");
        return true;
    }

    auto checkedRefreshVersion = atomic_read(&_timerRefreshVersion);
    NAVI_MEMORY_BARRIER();
    auto beginTime = _timeoutChecker.beginTime();
    auto globalTimeoutMs = _timeoutChecker.timeoutMs();
    auto endTimeMs = (autil::TimeUtility::currentTime() - beginTime) / MS_TO_US;
    if (endTimeMs >= globalTimeoutMs) {
        notifyTimeout(notifyParent);
        return true;
    }
    int64_t nextTimeoutMs = globalTimeoutMs;
    for (auto subGraph : _subGraphs) {
        auto endTimeMs = (autil::TimeUtility::currentTime() - beginTime) / MS_TO_US;
        auto next = subGraph->checkTimeout(endTimeMs);
        if (next > 0) {
            nextTimeoutMs = std::min(nextTimeoutMs, next);
        }
    }
    if (nextTimeoutMs > 0) {
        double timerTimeoutMs = (nextTimeoutMs + beginTime / MS_TO_US) - autil::TimeUtility::currentTime() / MS_TO_US;
        if (timerTimeoutMs <= 0.0) {
            if (nextTimeoutMs == globalTimeoutMs) {
                notifyTimeout(notifyParent);
                return true;
            } else {
                timerTimeoutMs = 0.0;
            }
        }
        autil::ScopedLock lock(_timerLock);
        if (_timerStopped) {
            return true;
        }
        auto latestRefreshVersion = atomic_read(&_timerRefreshVersion);
        if (checkedRefreshVersion != latestRefreshVersion) {
            timerTimeoutMs = 0.0;
        }
        NAVI_KERNEL_LOG(SCHEDULE1, "add timer with timeout [%f]ms", timerTimeoutMs);

        if (_timer) {
            evTimer->stopTimer(_timer);
        }
        _timer = evTimer->addTimer(timerTimeoutMs / 1000.0, timeoutCallback, this);
        _timerRefreshTriggeredVersion = latestRefreshVersion;
        return false;
    } else {
        notifyTimeout(notifyParent);
        return true;
    }
}

void Graph::timeoutCallback(void *data, bool isTimeout) {
    auto graph = (Graph *)data;
    auto graphParam = graph->getGraphParam();
    auto worker = graphParam->worker;
    if (isTimeout) {
        auto item = new TimeoutItem(worker, graphParam->runParams.getCollectFlags(), graph);
        worker->schedule(item, true);
    } else {
        worker->decItemCount();
    }
}

void Graph::initTimeoutChecker() {
    auto timeoutMs = _param->runParams.getTimeoutMs();
    if (timeoutMs >= DEFAULT_TIMEOUT_MS) {
        NAVI_KERNEL_LOG(SCHEDULE1, "timeoutMs[%ld] not set or too large, skip add timer", timeoutMs);
        return;
    }
    _timeoutChecker.setBeginTime(_param->beginTime > 0 ? _param->beginTime : _param->worker->getSessionBeginTime());
    _timeoutChecker.setTimeoutMs(timeoutMs);
    NAVI_KERNEL_LOG(SCHEDULE1, "set timeout checker with timeout [%ld]ms", timeoutMs);
}

bool Graph::initTimer() {
    auto worker = _param->worker;
    worker->incItemCount();
    if (checkTimeout(false)) {
        worker->decItemCount();
        return false;
    } else {
        return true;
    }
}

bool Graph::isTimeout() const { return _timeoutChecker.timeout(); }

int64_t Graph::getRemainTimeMs() const { return _timeoutChecker.remainTime(); }

bool Graph::updateTimeoutMs(int64_t timeoutMs) {
    auto worker = _param->worker;
    auto evTimer = worker->getEvTimer();
    if (!evTimer) {
        NAVI_KERNEL_LOG(ERROR, "get evTimer from worker failed");
        return false;
    }
    auto remainTimeMs = getRemainTimeMs();
    autil::ScopedLock lock(_timerLock);
    if (_timerStopped) {
        return false;
    }
    if (timeoutMs > remainTimeMs) {
        NAVI_KERNEL_LOG(
            DEBUG, "not allow update a larger timeout value[%ld ms], remain time[%ld ms]", timeoutMs, remainTimeMs);
        return false;
    }
    _param->runParams.setTimeoutMs(timeoutMs);
    _timeoutChecker.setTimeoutMs(timeoutMs);
    if (_timer) {
        evTimer->updateTimer(_timer, timeoutMs / 1000.0);
    } else {
        worker->incItemCount();
        _timer = evTimer->addTimer(timeoutMs / 1000.0, timeoutCallback, this);
    }
    return true;
}

bool Graph::refreshTimeout() {
    atomic_inc(&_timerRefreshVersion);
    auto evTimer = _param->worker->getEvTimer();
    if (!evTimer) {
        NAVI_KERNEL_LOG(ERROR, "get evTimer from worker failed");
        return false;
    }
    autil::ScopedLock lock(_timerLock);
    if (_timerStopped || !_timer) {
        return false;
    }
    auto refreshVersion = atomic_read(&_timerRefreshVersion);
    if (_timerRefreshTriggeredVersion != refreshVersion) {
        evTimer->updateTimer(_timer, 0.0);
        _timerRefreshTriggeredVersion = refreshVersion;
    }
    return true;
}

const TimeoutChecker *Graph::getTimeoutChecker() const { return &_timeoutChecker; }

void Graph::stopTimer() {

    autil::ScopedLock lock(_timerLock);
    _timerStopped = true;
    if (_timer) {
        auto evTimer = _param->worker->getEvTimer();
        evTimer->stopTimer(_timer);
        _timer = nullptr;
        NAVI_LOG(SCHEDULE1, "timer stopped");
    }
}

void Graph::setGraphDomain(GraphId graphId, const GraphDomainPtr &domain) {
    _graphDomainMap.emplace(graphId, domain);
}

void Graph::setForkInfo(GraphId graphId, const GraphDomainPtr &domain,
                        Node *forkNode, bool errorAsEof)
{
    setGraphDomain(graphId, domain);
    _forkNode = forkNode;
    _errorAsEof = errorAsEof;
}

NaviErrorPtr Graph::getError(ErrorCode ec) const {
    if (EC_NONE != ec) {
        return _param->graphResult->makeError(_logger.logger, ec);
    } else {
        return nullptr;
    }
}

void Graph::notifyFinishEc(ErrorCode ec) {
    notifyFinish(getError(ec));
}

void Graph::notifyFinish(const NaviErrorPtr &naviError) {
    notifyFinish(naviError, GFT_THIS_FINISH);
}

void Graph::notifyFinish(const NaviErrorPtr &error, GraphFinishType type) {
    NAVI_LOG(SCHEDULE1, "finish, parent [%p], ec [%s], type [%d]", _parent,
             CommonUtil::getErrorString(error ? error->ec : EC_NONE),
             type);
    {
        bool expect = false;
        if (!_terminated.compare_exchange_weak(expect, true)) {
            return;
        }
        auto worker = _param->worker;
        worker->incItemCount();
        stopTimer();
        {
            auto naviError = _param->graphResult->updateError(error);
            bool hasError = (nullptr != naviError);
            autil::ScopedLock lock(_terminateLock);
            if (!_parent && hasError) {
                worker->setFinish();
            }
            auto thisEc = EC_NONE;
            if (GFT_THIS_FINISH == type && naviError) {
                thisEc = naviError->ec;
            }
            finishResult();
            terminateAll(thisEc);
            if (_parent) {
                auto notifyParent = hasError && (GFT_PARENT_FINISH != type) && !_errorAsEof;
                if (notifyParent) {
                    NAVI_LOG(DEBUG, "notify parent, errorAsEof [%d]",
                             _errorAsEof);
                    if (_forkNode) {
                        _forkNode->notifyFinish(naviError);
                    } else {
                        _parent->notifyFinish(naviError, GFT_SUB_FINISH);
                    }
                }
            }
            if (!_parent) {
                worker->notifyFinish();
            }
        }
        worker->decItemCount();
    }
}

void Graph::finishResult() {
    if (_rootGraph || !_param->runParams.getCollectFlags().collectMalloc()) {
        collectMetric();
    }
    _param->graphResult->setFinish();
}

void Graph::collectMetric() {
    doCollectMetric();
    for (auto subGraph : _subGraphs) {
        subGraph->collectForkGraphMetric();
    }
}

bool Graph::doCollectMetric() {
    autil::ScopedLock lock(_metricCollectLock);
    if (_metricCollected) {
        return false;
    }
    _metricCollected = true;
    auto graphMetric = getGraphMetric();
    if (!needCollect()) {
        graphMetric->end();
        return false;
    }
    for (auto subGraph : _subGraphs) {
        subGraph->collectMetric(graphMetric);
    }
    for (const auto &pair : _graphDomainMap) {
        pair.second->collectMetric(graphMetric);
    }
    graphMetric->end();
    return true;
}

void Graph::terminateAll(ErrorCode ec) {
    for (auto subGraph : _subGraphs) {
        subGraph->terminateThis(ec);
    }
    const auto &bizDomainMap = _graphDomainMap;
    for (const auto &pair : bizDomainMap) {
        pair.second->notifyFinish(ec, false);
    }
}

GraphParam *Graph::getGraphParam() const {
    return _param;
}

void Graph::setResourceMap(const ResourceMap *resourceMap) {
    if (resourceMap) {
        _inputResourceMap.addResource(*resourceMap);
    }
}

const std::string &Graph::getLocalBizName() const {
    if (_parent) {
        return _parent->getLocalBizName();
    }
    return _localBizName;
}

NaviPartId Graph::getLocalPartId() const {
    if (_parent) {
        return _parent->getLocalPartId();
    }
    return _localPartId;
}

bool Graph::fillInputResource(GraphId graphId, const ResourceMapPtr &resourceMap) {
    auto inputSubResourceMap = getSubResourceMap(graphId);
    auto count = _inputResourceMap.size() + 2;
    if (inputSubResourceMap) {
        resourceMap->reserve(count + inputSubResourceMap->size());
        if (!resourceMap->addResource(*inputSubResourceMap)) {
            NAVI_LOG(ERROR, "override sub graph resource failed");
            return false;
        }
    } else {
        resourceMap->reserve(count);
    }
    const auto &resources = _inputResourceMap.getMap();
    for (const auto &pair : resources) {
        if (!resourceMap->addResource(pair.first, pair.second)) {
            NAVI_LOG(ERROR, "resource override [%s]",
                     pair.second->getName().c_str());
            return false;
        }
    }
    resourceMap->addResource(std::make_shared<TimeoutCheckerR>(&_timeoutChecker));
    if (_param->querySessionR) {
        resourceMap->addResource(_param->querySessionR);
    }
    NAVI_LOG(SCHEDULE1, "input resource map [%s]",
             autil::StringUtil::toString(*resourceMap).c_str());
    return true;
}

bool Graph::isLocal(const LocationDef &location) const {
    auto ret = _param->bizManager->isLocal(location);
    NAVI_LOG(SCHEDULE2, "location: %s, isLocal: %d",
             location.DebugString().c_str(), ret);
    return ret;
}

const std::string &Graph::getBizName(const SubGraphDef &subGraphDef) {
    return subGraphDef.location().biz_name();
}

GraphDomainPtr Graph::getGraphDomain(GraphId graphId) const {
    auto it = _graphDomainMap.find(graphId);
    if (_graphDomainMap.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

BizPtr Graph::getBiz(const SubGraphDef &subGraphDef) {
    auto graphId = subGraphDef.graph_id();
    if (PARENT_GRAPH_ID == graphId) {
        auto forkDomain = getForkDomain();
        if (forkDomain){
            return forkDomain->getBiz();
        }
    }
    return doGetBiz(subGraphDef);
}

GraphDomainFork *Graph::getForkDomain() const {
    auto domain = getGraphDomain(PARENT_GRAPH_ID);
    return dynamic_cast<GraphDomainFork *>(domain.get());
}

BizPtr Graph::doGetBiz(const SubGraphDef &subGraphDef) {
    auto bizManager = _param->bizManager;
    return bizManager->getBiz(_param->logger.get(), getBizName(subGraphDef));
}

void Graph::setSubResourceMap(const std::map<GraphId, ResourceMapPtr> &subResourceMaps) {
    _subResourceMaps = subResourceMaps;
}

ResourceMapPtr Graph::getSubResourceMap(GraphId graphId) const {
    auto it = _subResourceMaps.find(graphId);
    if (_subResourceMaps.end() != it) {
        return it->second;
    }
    return nullptr;
}

}
