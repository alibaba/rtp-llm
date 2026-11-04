#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Graph.h"
#include "navi/engine/GraphDomainFork.h"
#include "navi/engine/KernelWorkItem.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/LocalSubGraphInternal.h"
#include "navi/engine/Node.h"
#include "navi/util/CommonUtil.h"
#include <algorithm>
#include <unistd.h>

namespace navi {

thread_local extern size_t current_thread_counter;
thread_local extern size_t current_thread_wait_counter;

LocalSubGraph::LocalSubGraph(SubGraphDef *subGraphDef,
                             GraphParam *param,
                             const GraphDomainPtr &domain,
                             const SubGraphBorderMapPtr &borderMap,
                             Graph *graph,
                             const RewriteInfoPtr &rewriteInfo,
                             const GraphInlineBuildMapPtr &inlineBuildMap)
    : SubGraphBase(subGraphDef, param, domain, borderMap, graph)
    , _internal(nullptr)
    , _biz(rewriteInfo->biz)
    , _rewriteInfo(rewriteInfo)
    , _inlineBuildMap(inlineBuildMap)
    , _finishPortIndex(0)
    , _frozen(true)
    , _inlineMode(false)
    , _needConstructGraph(false)
{
    atomic_set(&_timeoutNodeCount, 0);
    _logger.addPrefix("local");
}

LocalSubGraph::~LocalSubGraph() {
    clearTimeoutNodes();
    if (unlikely(_needConstructGraph && _rewriteInfo->needCacheGraph &&
                 _internal->graphStarted &&
                 !_param->runParams.getCollectFlags().collectMalloc())) {
        auto totalNodeCount = _internal->nodeVec.size();
        for (uint32_t index = 0; index < totalNodeCount; index++) {
            if (_internal->nodeVec[index]->_started) {
                _internal->startedNodes.push_back(index);
            }
        }
        NAVI_LOG(
            DEBUG, "cache graph edgeCount: %lu", _internal->edgeVec.size());
        NaviLoggerScope scope(_logger);
        if (_biz->putCacheGraph(_rewriteInfo->graphKey, _owned)) {
            _owned.reset();
        }
    }
    if (_nodeVec) {
        auto nodeCount = _internal->nodeVec.size();
        for (auto i = 0; i < nodeCount; i++) {
            _nodeVec[i].~Node();
        }
        _nodeVec = nullptr;
    }
    if (_edgeVec) {
        auto edgeCount = _internal->edgeInternalVec.size();
        for (int32_t i = 0; i < edgeCount; ++i) {
            _edgeVec[i].~Edge();
        }
        free(_edgeVec);
        _edgeVec = nullptr;
    }
    if (_scopeInfos) {
        delete[] _scopeInfos;
        _scopeInfos = nullptr;
    }
    if (_graphMem) {
        free(_graphMem);
    }
}

ErrorCode LocalSubGraph::init() {
    if (!doInit()) {
        auto ec = EC_INIT_GRAPH;
        setErrorCode(ec);
        return ec;
    }
    return EC_NONE;
}

bool LocalSubGraph::doInit() {
    NaviLoggerScope scope(_logger);
    if (_rewriteInfo->needCacheGraph) {
        _internal = _rewriteInfo->cacheGraph.get();
    }
    _needConstructGraph = _internal == nullptr;
    if (_needConstructGraph) {
        _owned = std::make_shared<LocalSubGraphInternal>();
        _owned->subGraphDef = new SubGraphDef();
        _owned->subGraphDef->CopyFrom(*_subGraphDef);
        if (_rewriteInfo->nodeCountBeforeAddResource > 0 && !_param->runParams.isLoadGraph()) {
            _owned->subGraphDef->clear_borders();
            _owned->subGraphDef->clear_option();
            _owned->subGraphDef->clear_binary_attrs();
            for (int32_t i = 0; i < _rewriteInfo->nodeCountBeforeAddResource; i++) {
                auto nodeDef = _owned->subGraphDef->mutable_nodes(i);
                nodeDef->clear_json_attrs();
                nodeDef->clear_binary_attrs();
                nodeDef->clear_integer_attrs();
                nodeDef->clear_buildin_attrs();
                nodeDef->clear_integer_array_attrs();
            }
        }
        _owned->inlineBuildMap = _inlineBuildMap;
        _owned->replaceInfoMap = _rewriteInfo->replaceInfoMap;
        _owned->session = _logger.logger->getLoggerId();
        _owned->peerIp = _param->worker->getPeerIp();
        _internal = _owned.get();
    } else {
        _inlineBuildMap = _internal->inlineBuildMap;
    }
    initSubGraphOption();
    if (!initScopeMap(_needConstructGraph)) {
        return false;
    }
    if (_needConstructGraph) {

        if (!createInternalNodesAndEdges()) {
            return false;
        }
        if (!initPortNode()) {
            return false;
        }
        if (!bindPortToNodeInternal()) {
            return false;
        }
        if (!_internal->postInit(_param->creatorManager)) {
            return false;
        }
    }
    createNodes();
    if (!bindPortToNode()) {
        return false;
    }
    if (!createEdges()) {
        return false;
    }
    if (!bindForkDomain()) {
        return false;
    }
    if (!initPortDataType()) {
        return false;
    }
    return true;
}

ErrorCode LocalSubGraph::run() {
    if (!initInlineResource()) {
        return EC_INIT_GRAPH;
    }
    NaviDisableSignalScope disableSignal(true, _param->worker->getThreadPool());
    _frozen = false;
    auto ec = flushEdgeOverride();
    if (EC_NONE != ec) {
        return ec;
    }
    if (_internal->graphStarted) {
        for (const auto &index : _internal->startedNodes) {
            auto node = &_nodeVec[index];
            node->schedule(node);
        }
    } else {
        auto nodeCount = _internal->nodeVec.size();
        for (auto i = 0; i < nodeCount; i++) {
            auto node = &_nodeVec[i];
            bool start = false;
            if (0 == node->outputDegree()) {
                start = true;
            }
            if (node->isBorder()) {
                start = true;
            }
            if (start) {
                node->startSchedule(node, false, true);
            }
        }
        _internal->graphStarted = true;
    }
    return EC_NONE;
}

bool LocalSubGraph::initInlineResource() {
    auto graphId = _internal->subGraphDef->graph_id();
    if (_inlineBuildMap) {
        ResourceNodeId id;
        id.graphId = graphId;
        auto it = _inlineBuildMap->find(id);
        if (_inlineBuildMap->end() != it) {
            initSubGraphResource(it->second);
        } else {
            NAVI_LOG(SCHEDULE1, "no inline build sub graph resource, graphId [%d]", graphId);
        }
    }
    {
        ResourceNodeId scopeId;
        scopeId.graphId = graphId;
        auto count = _internal->scopeInfoMap.size();
        for (int32_t i = 0; i < count; ++i) {
            auto &scopeInfo = _scopeInfos[i];
            scopeId.scope = scopeInfo.internal->scopeId;
            const std::set<const std::string *> *resourceList = nullptr;
            if (_inlineBuildMap) {
                auto it = _inlineBuildMap->find(scopeId);
                if (_inlineBuildMap->end() != it) {
                    resourceList = &it->second;
                } else {
                    NAVI_LOG(SCHEDULE1,
                             "no inline build scope resource, graphId [%d], scopeId: [%d]",
                             graphId,
                             scopeInfo.internal->scopeId);
                }
            }
            initScopeResource(resourceList, scopeInfo);
        }
    }
    return true;
}

void LocalSubGraph::initSubGraphResource(const std::set<const std::string *> &resourceList) {
    if (resourceList.empty()) {
        return;
    }
    _resourceMap = std::make_shared<ResourceMap>();
    auto multiLayerResourceMap = _rewriteInfo->multiLayerResourceMap;
    multiLayerResourceMap.addResourceMap(_rewriteInfo->bizResourceMap.get());
    auto replaceInfoMap = getReplaceInfoMap();
    for (auto namePtr : resourceList) {
        auto resource = buildResourceRecur(
            *namePtr, RS_SUB_GRAPH, nullptr, multiLayerResourceMap, replaceInfoMap, nullptr, *_resourceMap);
        NAVI_LOG(SCHEDULE1,
                 "inline build sub graph resource [%s] [%p], graphId [%d]",
                 namePtr->c_str(),
                 resource,
                 getGraphId());
    }
}

void LocalSubGraph::initScopeResource(const std::set<const std::string *> *resourceList, ScopeInfo &scopeInfo) {
    auto resourceMap = std::make_shared<ResourceMap>();
    {
        auto multiLayerResourceMap = _rewriteInfo->multiLayerResourceMap;
        if (_resourceMap) {
            multiLayerResourceMap.addResourceMap(_resourceMap.get());
        }
        multiLayerResourceMap.collectResource(*resourceMap);
    }
    if (resourceList) {
        auto replaceInfoMap = getReplaceInfoMap();
        MultiLayerResourceMap buildMultiLayerResourceMap;
        buildMultiLayerResourceMap.addResourceMap(_rewriteInfo->bizResourceMap.get());
        for (auto namePtr : *resourceList) {
            auto resource = buildResourceRecur(
                *namePtr, RS_SCOPE, nullptr, buildMultiLayerResourceMap, replaceInfoMap, nullptr, *resourceMap);
            NAVI_LOG(SCHEDULE1,
                     "inline build scope resource [%s] [%p], graphId [%d], scopeId [%d]",
                     namePtr->c_str(),
                     resource,
                     getGraphId(),
                     scopeInfo.internal->scopeId);
        }
    }
    if (resourceMap->size() > 0) {
        scopeInfo.resourceMap = std::move(resourceMap);
    }
}

ErrorCode LocalSubGraph::flushEdgeOverride() {
    const auto &overrideDatas = _param->runParams.getOverrideDatas();
    auto subGraphId = getGraphId();
    auto partId = getPartId();
    for (const auto &overrideData : overrideDatas) {
        if (overrideData.graphId != subGraphId) {
            continue;
        }
        auto overridePartId = overrideData.partId;
        if (INVALID_NAVI_PART_ID != overridePartId && partId != overridePartId) {
            continue;
        }
        auto edgeKeyHash = CommonUtil::getPortKeyHash(overrideData.outputNode,
                                                      overrideData.outputPort);
        auto it = _internal->edges.find(edgeKeyHash);
        if (_internal->edges.end() == it) {
            NAVI_LOG(ERROR,
                     "override edge data failed, graphId [%d] partId [%d] node "
                     "[%s] port[%s] edge[%s %s] not exist",
                     overrideData.graphId,
                     overrideData.partId,
                     overrideData.outputNode.c_str(),
                     overrideData.outputPort.c_str(),
                     overrideData.outputNode.c_str(),
                     overrideData.outputPort.c_str());

            return EC_OVERRIDE_EDGE_DATA;
        }
        if (!transformEdgeFromInternal(it->second)
                 ->overrideData(overrideData.datas)) {
            NAVI_LOG(ERROR,
                     "multiple edge override data, graphId [%d] partId [%d] node [%s] port[%s]",
                     overrideData.graphId,
                     overrideData.partId,
                     overrideData.outputNode.c_str(),
                     overrideData.outputPort.c_str());
            return EC_OVERRIDE_EDGE_DATA;
        }
    }
    return EC_NONE;
}

bool LocalSubGraph::schedule(Node *node) {
    if (_frozen) {
        NAVI_LOG(SCHEDULE1,
                 "graph is frozen, node [%s] kernel [%s] schedule ignored",
                 node->getName().c_str(), node->getKernelName().c_str());
        return false;
    }
    if (terminated()) {
        NAVI_LOG(SCHEDULE1,
                 "graph terminated, node [%s] kernel [%s] schedule ignored",
                 node->getName().c_str(), node->getKernelName().c_str());
        return true;
    }
    auto worker = _param->worker;
    if (unlikely(getTestMode() != TM_NONE)) {
        worker->inlineBegin();
        NAVI_LOG(SCHEDULE1, "inline compute, node [%s] kernel [%s]",
                 node->getName().c_str(), node->getKernelName().c_str());
        {

            auto schedInfo = worker->makeInlineSchedInfo();
            PerfEnableScope perfEnableScope(worker->getNaviPerf(), _param->runParams.getCollectFlags());
            auto computeScope = node->makeComputeScope(*schedInfo, perfEnableScope.getPerfThread());
            node->compute(computeScope);
        }
        worker->inlineEnd();
        return true;
    }
    auto item = node->getKernelWorkItem();
    if (inlineCompute(worker, node)) {
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "inline compute node [%s] kernel [%s]",
                        node->getName().c_str(),
                        node->getKernelName().c_str());
        item->setInlineCompute(true);
    }
    item->initSchedInfo();
    auto ret = worker->schedule(item);
    if (unlikely(!ret)) {
        if (worker->isFinished()) {
            NAVI_LOG(SCHEDULE1, "ignore schedule failure after worker finshed, node[%s]",
                     node->getName().c_str());
        } else if (_graph->isTimeout()) {
            _graph->notifyTimeout(true);
        } else {
            NAVI_LOG(ERROR, "schedule kernel failed, node[%s]",
                     node->getName().c_str());
            auto error =
                _param->graphResult->makeError(_logger.logger, EC_UNKNOWN);
            notifyFinish(node, error);
        }
    }
    return true;
}

bool LocalSubGraph::inlineCompute(NaviWorkerBase * worker,
                                  Node * node)
{
    if (!worker->canInline()) {
        return false;
    }
    if (_inlineMode) {
        return true;
    }
    auto canInline = node->isInline();
    if (!canInline) {
        return false;
    }
    return true;
}

void LocalSubGraph::initSubGraphOption() {
    _inlineMode = _subGraphDef->option().inline_mode();
    _errorHandleStrategy = _subGraphDef->option().error_handle_strategy();
    auto location = _subGraphDef->mutable_location();
    if (location->biz_name().empty()) {
        location->set_biz_name(_biz->getName());
    }
}

bool LocalSubGraph::initScopeMap(bool needConstructGraph) {
    if (needConstructGraph) {
        auto count = _internal->subGraphDef->scopes_size();
        for (int32_t i = 0; i < count; ++i) {
            const auto &scopeDef = _internal->subGraphDef->scopes(i);
            auto id = scopeDef.id();
            auto it = _internal->scopeInfoMap.find(id);
            if (_internal->scopeInfoMap.end() != it) {
                NAVI_KERNEL_LOG(ERROR, "duplicate scope id [%d]", id);
                return false;
            }
            auto index = _internal->scopeInfoMap.size();
            auto scopeInfo = &_internal->scopeInfoMap[id];
            scopeInfo->scopeId = id;
            scopeInfo->index = index;
        }
    }
    auto count = _subGraphDef->scopes_size();
    _scopeInfos = new ScopeInfo[count];
    for (int32_t i = 0; i < count; ++i) {
        const auto &scopeDef = _subGraphDef->scopes(i);
        auto id = scopeDef.id();
        auto it = _internal->scopeInfoMap.find(id);
        if (_internal->scopeInfoMap.end() == it) {
            NAVI_KERNEL_LOG(ERROR, "internl graph missing scope id [%d]", id);
            return false;
        }
        const auto &internal = it->second;
        auto scopeInfo = &_scopeInfos[internal.index];
        scopeInfo->internal = &it->second;
        scopeInfo->def = &scopeDef;
        if (!scopeDef.has_terminator()) {
            scopeInfo->logger = _logger.logger;
        } else {
            scopeInfo->logger =
                std::make_shared<NaviLogger>(*_logger.logger, true);
        }
        scopeInfo->objectLogger.logger = scopeInfo->logger;
    }
    return true;
}

ScopeInfo *LocalSubGraph::getScopeInfo(int32_t scopeIndex) {
    if (scopeIndex >= _internal->scopeInfoMap.size()) {
        NAVI_LOG(ERROR, "scope not exist id [%d]", scopeIndex);
        return nullptr;
    }
    return &_scopeInfos[scopeIndex];
}

void LocalSubGraph::addToScopeInfo(ScopeInfo *scopeInfo, NodeType nodeType,
                                   NodeInternal *node)
{
    node->_scopeIndex = scopeInfo->internal->index;
    if (NT_SCOPE_TERMINATOR == nodeType) {
        scopeInfo->internal->terminator = node;
    } else if (NT_SCOPE_TERMINATOR_SPLIT != nodeType &&
               NT_TERMINATOR != nodeType &&
               NT_RESOURCE != nodeType &&
               NT_SPLIT != nodeType &&
               NT_MERGE != nodeType)
    {
        scopeInfo->internal->nodes.push_back(node);
    }
}

bool LocalSubGraph::createInternalNodesAndEdges() {
    const auto &subGraph = *_internal->subGraphDef;
    auto nodeCount = subGraph.nodes_size();
    KernelMemEstimator estimator;
    estimator.add(sizeof(Node), alignof(Node), nodeCount);
    for (auto i = 0; i < nodeCount; i++) {
        const auto &node = subGraph.nodes(i);
        if (!createNode(i, node, estimator)) {
            NAVI_LOG(ERROR, "create node[%s] fail", node.name().c_str());
            return false;
        }
    }
    _internal->mallocSize = estimator.getBaseAddr();
    auto edgeCount = subGraph.edges_size();
    auto useEdgeIndex = (0 != _rewriteInfo->graphKey);
    _internal->edgeVec.resize(edgeCount);
    for (auto i = 0; i < edgeCount; i++) {
        const auto &edgeDef = subGraph.edges(i);
        if (!createEdge(useEdgeIndex, i, edgeDef)) {
            NAVI_LOG(ERROR,
                     "create edge[%s:%s -> %s:%s] fail",
                     edgeDef.input().node_name().c_str(),
                     edgeDef.input().port_name().c_str(),
                     edgeDef.output().node_name().c_str(),
                     edgeDef.output().port_name().c_str());
            return false;
        }
    }
    return true;
}

void LocalSubGraph::createNodes() {
    const auto &subGraph = *_subGraphDef;
    auto needCollect = _param->runParams.getCollectFlags().needCollect();
    auto currentNodeCount = subGraph.nodes_size();
    auto nodeCount = _internal->nodeVec.size();
    _graphMem = malloc(_internal->mallocSize);
    constexpr size_t align = alignof(Node);
    constexpr size_t size = NAVI_ALIGN(sizeof(Node), align);
    static_assert(size == sizeof(Node));
    char *addr = (char *)NAVI_ALIGN((size_t)_graphMem, align);
    _nodeVec = (Node *)addr;
    for (auto i = 0; i < nodeCount; i++) {
        const NodeDef *nodeDef = nullptr;
        if (i < currentNodeCount) {
            nodeDef = &subGraph.nodes(i);
        } else {
            nodeDef = _internal->nodeVec[i]->_nodeDef;
        }
        new (addr) Node(*nodeDef, this, _internal->nodeVec[i], (char *)_graphMem, needCollect);
        addr = addr + size;
    }
}

bool LocalSubGraph::createEdges() {
    const auto &subGraph = *_subGraphDef;
    auto currentEdgeCount = subGraph.edges_size();
    const auto &edgeInternalVec = _internal->edgeInternalVec;
    auto edgeCount = edgeInternalVec.size();
    Edge *edgeVecMem = (Edge *)malloc(sizeof(Edge) *edgeCount);
    for (size_t i = 0; i < edgeCount; i++) {
        new (&edgeVecMem[i]) Edge(this, edgeInternalVec[i]);
    }
    _edgeVec = edgeVecMem;
    initEdgeTopoParam();
    if (0 != _rewriteInfo->graphKey) {
        const auto &paramEdgeInfo = subGraph.param_edge_info();
        auto count = paramEdgeInfo.edge_params_size();
        for (int32_t i = 0; i < count; ++i) {
            const auto &edgeParamDef = paramEdgeInfo.edge_params(i);
            auto index = edgeParamDef.index();
            if (unlikely(index < 0 || index >= currentEdgeCount)) {
                NAVI_LOG(
                    ERROR, "edge param index [%d] overflow, total edge count [%d]", index, currentEdgeCount);
                return false;
            }
            auto edgeIndex = subGraph.edges(index).edge_index();
            if (unlikely(edgeIndex < 0)) {
                NAVI_LOG(ERROR, "invalid edge index [%d], total edge count [%d]", index, currentEdgeCount);
                return false;
            }
            if (edgeIndex > 0) {
                index = edgeIndex - 1;
            } else {
                // resource edge has no edge index
            }
            if (index >= _internal->edgeVec.size()) {
                NAVI_LOG(ERROR,
                         "edge index [%d] [%d] overflow, total edge count [%lu]",
                         index,
                         edgeIndex,
                         _internal->edgeVec.size());
                return false;
            }
            const auto &edgeOutputInfo = _internal->edgeVec[index];
            auto timeoutMs = edgeParamDef.timeout_ms();
            if (timeoutMs > 0) {
                collectTimeoutEdge(timeoutMs, edgeOutputInfo);
            }
            if (unlikely(edgeParamDef.require())) {
                edgeOutputInfo.setOptional(this, false);
            }
        }
    } else {
        auto count = _internal->edgeVec.size();
        for (int32_t i = 0; i < count; ++i) {
            const auto &edgeOutputInfo = _internal->edgeVec[i];
            auto edgeInternal = edgeOutputInfo.getEdge();
            const EdgeDef *edgeDef = nullptr;
            if (i < currentEdgeCount) {
                edgeDef = &subGraph.edges(i);
            } else {
                edgeDef = edgeInternal->_edgeDef;
            }
            auto timeoutMs = edgeDef->timeout_ms();
            if (timeoutMs > 0) {
                collectTimeoutEdge(timeoutMs, edgeOutputInfo);
            }
            if (unlikely(edgeDef->require())) {
                edgeOutputInfo.setOptional(this, false);
            }
        }
    }
    return true;
}

void LocalSubGraph::initEdgeTopoParam() {
    TopoNodeParam *topoParam = nullptr;
    const auto &paramTopoTreeNode = _param->runParams.getParamTopoTree();
    if (paramTopoTreeNode) {
        topoParam = paramTopoTreeNode->getParam().get();
    }
    if (!topoParam) {
        return;
    }
    const auto &edgeInternalVec = _internal->edgeInternalVec;
    auto edgeCount = edgeInternalVec.size();
    for (size_t i = 0; i < edgeCount; i++) {
        auto edge = &_edgeVec[i];
        auto edgeInternal = edgeInternalVec[i];
        auto edgeDef = edgeInternal->_edgeDef;
        const auto &inputDef = edgeDef->input();
        const auto &inputNodeName = getNodeName(this, inputDef);
        auto edgeKeyHash = CommonUtil::getPortKeyHash(inputNodeName, inputDef.port_name());
        auto query = topoParam->getSummaryQuery(getGraphId(), edgeKeyHash);
        if (query) {
            edge->setSummaryQuery(getGraphId(), inputNodeName, inputDef.port_name(), *query, _param->creatorManager);
        }
    }
}

bool LocalSubGraph::bindForkDomain() {
    if (!_edgeVec) {
        NAVI_LOG(ERROR, "edge is empty");
        return false;
    }
    auto domain = dynamic_cast<GraphDomainFork *>(_domain.get());
    if (!domain) {
        return true;
    }
    return domain->bindGraphEdgeList(_edgeVec, _internal->edgeInternalVec.size());
}

bool LocalSubGraph::isForkDomain() const {
    return nullptr != dynamic_cast<GraphDomainFork *>(_domain.get());
}

bool LocalSubGraph::initPortDataType() {
    const auto &borderMap = *_borderMap;
    for (const auto &pair : borderMap) {
        const auto &borderId = pair.first;
        const auto &border = *pair.second;
        const auto &portVec = border.getPortVec();
        for (auto port : portVec) {
            if (port->isBackPressurePort()) {
                continue;
            }
            if (!setPortType(port, borderId.ioType)) {
                NAVI_LOG(ERROR, "init port data type failed, portKey: %s",
                         port->getPortKey().c_str());
                return false;
            }
        }
    }
    return true;
}

bool LocalSubGraph::setPortType(Port *port, IoType ioType) {
    auto dataType = port->getDataType();
    if (dataType) {
        port->bindDataType(dataType);
        return true;
    }
    const auto &nodeName = port->getNode();
    auto nodeIndex = port->getNodeIndex();
    auto node = getNode(nodeName, nodeIndex);
    if (!node) {
        if (NAVI_FORK_NODE == nodeName) {
            return true;
        }
        return false;
    }
    return bindPortDataType(node, port, ioType);
}

bool LocalSubGraph::bindPortDataType(Node *node, Port *port, IoType ioType) {
    auto typeStr = getNodePortDataTypeStr(node, port->getPort(),
                                          port->getPeerPort(), ioType);
    if (typeStr.empty()) {
        return true;
    }
    auto type = _param->creatorManager->getType(typeStr);
    if (!type) {
        NAVI_LOG(ERROR,
                 "port data type [%s] not registered, node [%s] kernel [%s], "
                 "%s port [%s]",
                 typeStr.c_str(), node->getName().c_str(),
                 node->getKernelName().c_str(), CommonUtil::getIoType(ioType),
                 port->getPort().c_str());
    }
    port->bindDataType(type);
    return true;
}

std::string LocalSubGraph::getNodePortDataTypeStr(
        Node *node,
        const std::string &portName,
        const std::string &peerPortName,
        IoType ioType) const
{
    std::string typeStr;
    if ((0 == portName.find(NODE_CONTROL_INPUT_PORT)) || (0 == peerPortName.find(NODE_CONTROL_INPUT_PORT))) {
        typeStr = CONTROL_DATA_TYPE;
    } else {
        if (!node->getPortTypeStr(portName, ioType, typeStr)) {
            return "";
        }
    }
    return typeStr;
}

Node *LocalSubGraph::getNode(const std::string &nodeName, int32_t portNodeId) const {
    auto nodeInternal = _internal->getNode(nodeName, portNodeId);
    if (!nodeInternal || !_nodeVec) {
        return nullptr;
    }
    return &_nodeVec[nodeInternal->_id];
}

void LocalSubGraph::setScopeError(int32_t scopeIndex,
                                  const NaviErrorPtr &error)
{
    auto scopeInfoPtr = getScopeInfo(scopeIndex);
    if (!scopeInfoPtr) {
        return;
    }
    auto &scopeInfo = *scopeInfoPtr;
    scopeInfo.setError(error);
    if (!scopeInfo.internal->terminator) {
        _param->graphResult->updateError(error);
        return;
    }
}

void LocalSubGraph::notifyFinish(Node *node, const NaviErrorPtr &error) {
    auto def = node->getDef();
    auto nodeType = def->type();
    auto scope = def->scope();
    if (NT_SCOPE_TERMINATOR == nodeType || NT_SCOPE_TERMINATOR_SPLIT == nodeType) {
        terminateScope(node, scope, error);
    } else if (NT_TERMINATOR != nodeType) {
        if (terminateScope(node, scope, error)) {
            return;
        }
    }
    if ((error && EC_NONE != error->ec) || isForkDomain()) {
        if (EHS_ERROR_AS_FATAL == _errorHandleStrategy) {
            _graph->notifyFinish(error);
        } else {
            forceStopNodes(node, _internal->nodeVec);
        }
    }
}

NaviWorkerBase *LocalSubGraph::getWorker() const {
    return _param->worker;
}

const BizPtr &LocalSubGraph::getBiz() const {
    return _biz;
}

bool LocalSubGraph::saveResource(const ResourceMap &resourceMap) {
    ResourceMap fullMap(resourceMap.getMap());
    fullMap.addResource(*_rewriteInfo->resourceMap);
    return _biz->saveResource(getPartCount(), getPartId(), fullMap);
}

bool LocalSubGraph::publishResource() {
    return _biz->publishResource(getPartCount(), getPartId());
}

void LocalSubGraph::collectResourceMap(int32_t scopeIndex, MultiLayerResourceMap &multiLayerResourceMap) const {
    const auto &scopeInfo = _scopeInfos[scopeIndex];
    multiLayerResourceMap.addResourceMap(_rewriteInfo->bizResourceMap.get());
    multiLayerResourceMap.addResourceMap(scopeInfo.resourceMap.get());
}

ResourceStage LocalSubGraph::getResourceStage(const std::string &name) const {
    return _biz->getResourceStage(name);
}

const NaviUnorderedSet<std::string> *LocalSubGraph::getResourceReplacerSet(const std::string &name) const {
    return _biz->getResourceReplacerSet(name);
}

ErrorCode LocalSubGraph::createResource(const std::string &name,
                                        const MultiLayerResourceMap &multiLayerResourceMap,
                                        const ProduceInfo &produceInfo,
                                        bool checkReuse,
                                        Node *requireNode,
                                        bool failAsError,
                                        ResourcePtr &resource) const {
    auto graphId = getGraphId();
    auto minStage = _param->runParams.getResourceStage();
    auto namedDataMap = _param->runParams.getSubNamedDataMap(graphId);
    return _biz->createResource(getPartCount(),
                                getPartId(),
                                name,
                                minStage,
                                multiLayerResourceMap,
                                namedDataMap,
                                produceInfo,
                                checkReuse,
                                requireNode,
                                failAsError,
                                resource);
}

Resource *LocalSubGraph::buildResourceRecur(const std::string &name,
                                            ResourceStage minStage,
                                            NaviConfigContext *inputCtx,
                                            const MultiLayerResourceMap &nodeResourceMap,
                                            const ReplaceInfoMap *replaceInfoMap,
                                            Node *requireNode,
                                            ResourceMap &inputResourceMap)
{
    auto graphId = getGraphId();
    auto namedDataMap = _param->runParams.getSubNamedDataMap(graphId);
    return _biz->buildResourceRecur(getPartCount(),
                                    getPartId(),
                                    name,
                                    minStage,
                                    inputCtx,
                                    namedDataMap,
                                    nodeResourceMap,
                                    replaceInfoMap,
                                    requireNode,
                                    inputResourceMap);
}

NodeInternal *LocalSubGraph::createNode(int32_t id,
                                        const NodeDef &nodeDef,
                                        KernelMemEstimator &estimator)
{
    const auto &nodeName = nodeDef.name();
    if (_internal->nodeMap.end() != _internal->nodeMap.find(nodeName)) {
        NAVI_LOG(ERROR, "duplicate node [%s]", nodeName.c_str());
        return nullptr;
    }
    auto scopeInfo = getScopeInfo(nodeDef.scope());
    if (!scopeInfo) {
        NAVI_LOG(ERROR, "get scope info failed, node [%s]", nodeName.c_str());
        return nullptr;
    }
    auto isResourceNode = (nodeDef.kernel_name() == RESOURCE_CREATE_KERNEL);
    NodeInternal *nodeInternal =
        new NodeInternal(id, nodeDef, this, _biz.get(), isResourceNode);
    if (!nodeInternal->init(nodeDef, estimator)) {
        delete nodeInternal;
        return nullptr;
    }
    _internal->nodeMap.insert(std::make_pair(nodeName, nodeInternal));
    _internal->nodeVec.emplace_back(nodeInternal);
    addToScopeInfo(scopeInfo, nodeDef.type(), nodeInternal);
    return nodeInternal;
}

bool LocalSubGraph::createEdge(bool useEdgeIndex, int32_t id, const EdgeDef &edgeDef) {
    const auto &inputDef = edgeDef.input();
    auto edgeKeyHash =
        CommonUtil::getPortKeyHash(inputDef.node_name(), inputDef.port_name());
    auto &edges = _internal->edges;
    auto &edgeInternalVec = _internal->edgeInternalVec;
    auto it = edges.find(edgeKeyHash);
    EdgeInternal *edge = nullptr;
    if (edges.end() == it) {
        edge = new EdgeInternal(edgeDef, edges.size());
        if (!edge->init(_internal, inputDef)) {
            delete edge;
            return false;
        }
        edges.emplace(edgeKeyHash, edge);
        edgeInternalVec.push_back(edge);
    } else {
        edge = it->second;
    }
    NAVI_KERNEL_LOG(SCHEDULE2, "create edge[%s]", autil::StringUtil::toString(*edge).c_str());
    auto edgeOutputInfo = edge->addOutput(_internal, edgeDef);
    if (!edgeOutputInfo.isValid()) {
        return false;
    } else {
        auto index = id;
        if (useEdgeIndex) {
            auto edgeIndex = edgeDef.edge_index();
            if (unlikely(edgeIndex < 0)) {
                NAVI_KERNEL_LOG(ERROR,
                                "invalid edge index, edge [%s] index [%d]",
                                edgeOutputInfo.getDebugName(this).c_str(),
                                edgeIndex);
                return false;
            } else if (edgeIndex > 0) {
                index = edgeIndex - 1;
            } else {
                // resource edge has no index
            }
            if (index >= _internal->edgeVec.size()) {
                NAVI_KERNEL_LOG(ERROR,
                                "edge index [%d] [%d] overflow, total edge count [%lu]",
                                index,
                                edgeIndex,
                                _internal->edgeVec.size());
                return false;
            }
        }
        auto &edgeOutputInfoRef = _internal->edgeVec[index];
        if (edgeOutputInfoRef.isValid()) {
            NAVI_KERNEL_LOG(
                ERROR, "edge output info override [%s], index [%d]", edgeOutputInfoRef.getDebugName(this).c_str(), index);
            return false;
        }
        edgeOutputInfoRef = edgeOutputInfo;
        return true;
    }
}

void LocalSubGraph::collectTimeoutEdge(int64_t timeoutMs, const EdgeOutputInfo &edgeOutputInfo) {

    if (!_timeoutEdgeMap) {
        _timeoutEdgeMap = std::make_unique<std::map<int64_t, std::vector<EdgeOutputInfo>>>();
    }
    auto it = _timeoutEdgeMap->find(timeoutMs);
    if (_timeoutEdgeMap->end() == it) {
        (*_timeoutEdgeMap)[timeoutMs].push_back(edgeOutputInfo);
    } else {
        it->second.push_back(edgeOutputInfo);
    }
}

int64_t LocalSubGraph::checkTimeout(int64_t endTimeMs) {
    auto nextTimeoutMs = checkTimeoutEdge(endTimeMs);
    if (0 == atomic_read(&_timeoutNodeCount)) {
        return nextTimeoutMs;
    }
    NAVI_MEMORY_BARRIER();
    auto nextPipeTimeoutMs = checkTimeoutAsyncPipe(endTimeMs);
    if (nextPipeTimeoutMs < 0) {
        return nextTimeoutMs;
    }
    return nextTimeoutMs < 0
        ? nextPipeTimeoutMs : std::min(nextTimeoutMs, nextPipeTimeoutMs);
}

int64_t LocalSubGraph::checkTimeoutAsyncPipe(int64_t endTimeMs) {
    int64_t nextTimeoutMs = -1;
    auto timeoutNodes = std::atomic_load(&_timeoutNodes);
    if (!timeoutNodes) {
        return -1;
    }
    for (auto node : *timeoutNodes) {
        auto nextNodeTimeoutMs = node->checkTimeoutAsyncPipes(endTimeMs);
        if (nextNodeTimeoutMs > 0 &&
            (nextTimeoutMs < 0 || nextNodeTimeoutMs < nextTimeoutMs)) {
            nextTimeoutMs = nextNodeTimeoutMs;
        }
    }
    return nextTimeoutMs;
}

int64_t LocalSubGraph::checkTimeoutEdge(int64_t endTimeMs) {
    if (!_timeoutEdgeMap) {
        return -1;
    }
    auto it = _timeoutEdgeMap->begin();
    for (; _timeoutEdgeMap->end() != it;) {
        if (it->first <= endTimeMs) {
            for (const auto &info : it->second) {
                NAVI_LOG(DEBUG, "edge timeout [%s], timeoutMs: %ld,", info.getDebugName(this).c_str(), it->first);
                info.setEof(this, true, nullptr);
            }
            it = _timeoutEdgeMap->erase(it);
        } else {
            break;
        }
    }
    if (_timeoutEdgeMap->end() != it) {
        return it->first;
    } else {
        return -1;
    }
}

bool LocalSubGraph::registerTimeoutNode(Node *node)
{
    if (!node || terminated()) {
        return false;
    }
    {
        autil::ScopedLock lock(_timeoutNodeLock);
        auto currentNodes = std::atomic_load(&_timeoutNodes);
        if (!currentNodes || currentNodes->end() == std::find(
                currentNodes->begin(), currentNodes->end(), node)) {
            auto timeoutNodes = std::make_shared<std::vector<Node *>>(
                currentNodes ? *currentNodes : std::vector<Node *>());
            timeoutNodes->emplace_back(node);
            std::shared_ptr<const std::vector<Node *>> newNodes =
                std::move(timeoutNodes);
            std::atomic_store(&_timeoutNodes, std::move(newNodes));
            NAVI_MEMORY_BARRIER();
            atomic_set(&_timeoutNodeCount, currentNodes
                ? currentNodes->size() + 1 : 1);
        }
    }
    return _graph->refreshTimeout();
}

void LocalSubGraph::unregisterTimeoutNode(Node *node)
{
    autil::ScopedLock lock(_timeoutNodeLock);
    auto currentNodes = std::atomic_load(&_timeoutNodes);
    if (!currentNodes) {
        return;
    }
    auto timeoutNodes = std::make_shared<std::vector<Node *>>(
        *currentNodes);
    auto it = std::find(timeoutNodes->begin(), timeoutNodes->end(), node);
    if (timeoutNodes->end() == it) {
        return;
    }
    if (node->hasActivePeriodicPipe()) {
        node->_timeoutNodeRegistered.store(true);
        return;
    }
    timeoutNodes->erase(it);
    std::shared_ptr<const std::vector<Node *>> newNodes;
    if (timeoutNodes->empty()) {
        newNodes.reset();
    } else {
        newNodes = std::move(timeoutNodes);
    }
    auto nodeCount = newNodes ? newNodes->size() : 0;
    std::atomic_store(&_timeoutNodes, std::move(newNodes));
    NAVI_MEMORY_BARRIER();
    atomic_set(&_timeoutNodeCount, nodeCount);
}

void LocalSubGraph::clearTimeoutNodes()
{
    autil::ScopedLock lock(_timeoutNodeLock);
    std::atomic_store(
        &_timeoutNodes,
        std::shared_ptr<const std::vector<Node *>>());
    NAVI_MEMORY_BARRIER();
    atomic_set(&_timeoutNodeCount, 0);
}

void LocalSubGraph::collectMetric(GraphMetric *graphMetric) {
    if (!_nodeVec) {
        return;
    }
    auto nodeCount = _internal->nodeVec.size();
    for (auto i = 0; i < nodeCount; i++) {
        auto node = &_nodeVec[i];
        if (_edgeVec) {
            node->collectState();
        }
        const auto &metric = node->getMetric();
        if (metric) {
            graphMetric->addKernelMetricPtr(metric);
        }
    }
    if (_edgeVec) {
        auto edgeCount = _internal->edgeInternalVec.size();
        for (size_t i = 0; i < edgeCount; i++) {
            auto edge = &_edgeVec[i];
            if (edge) {
                const auto &metric = edge->getMetric();
                if (metric) {
                    graphMetric->addEdgeMetricPtr(metric);
                }
            }
        }
    }
}

void LocalSubGraph::collectForkGraphMetric() {
    if (!_nodeVec) {
        return;
    }
    auto nodeCount = _internal->nodeVec.size();
    for (auto i = 0; i < nodeCount; i++) {
        auto node = &_nodeVec[i];
        node->collectForkGraphMetric();
    }
}

void LocalSubGraph::terminate(ErrorCode ec) {
    clearTimeoutNodes();
    if (!_nodeVec) {
        return;
    }
    auto nodeCount = _internal->nodeVec.size();
    for (auto i = 0; i < nodeCount; i++) {
        auto node = &_nodeVec[i];
        node->terminate(ec);
    }
}

bool LocalSubGraph::initPortNode() {
    const auto &borderMap = *_borderMap;
    for (const auto &pair : borderMap) {
        const auto &portVec = pair.second->getPortVec();
        for (auto port : portVec) {
            const auto &portNodeName = port->getPortNodeName();
            if (port->isBackPressurePort()) {
                continue;
            }
            auto nodeInternal = _internal->getNode(portNodeName, 0);
            if (!nodeInternal) {
                NAVI_LOG(ERROR, "port node [%s] not exist",
                         portNodeName.c_str());
                return false;
            }
        }
    }
    return true;
}

bool LocalSubGraph::bindPortToNodeInternal() {
    const auto &borderMap = *_borderMap;
    for (const auto &pair : borderMap) {
        const auto &portVec = pair.second->getPortVec();
        for (auto port : portVec) {
            const auto &portNodeName = port->getPortNodeName();
            if (port->isBackPressurePort()) {
                continue;
            }
            auto nodeInternal = _internal->getNode(portNodeName, 0);
            if (!nodeInternal) {
                NAVI_LOG(ERROR, "port node [%s] not exist",
                         portNodeName.c_str());
                return false;
            }
            if (!nodeInternal->bindPort(port)) {
                return false;
            }
        }
    }
    return true;
}

bool LocalSubGraph::bindPortToNode() {
    const auto &borderMap = *_borderMap;
    for (const auto &pair : borderMap) {
        const auto &portVec = pair.second->getPortVec();
        for (auto port : portVec) {
            if (port->isBackPressurePort()) {
                continue;
            }
            const auto &portNodeName = port->getPortNodeName();
            auto portNodeId = port->getPortNodeId();
            auto node = getNode(portNodeName, portNodeId);
            if (!node) {
                NAVI_LOG(ERROR, "port node [%s] not exist",
                         portNodeName.c_str());
                return false;
            }
            if (!node->bindPort(port)) {
                return false;
            }
        }
    }
    return true;
}

bool LocalSubGraph::terminateScope(Node *node, int32_t scopeIndex,
                                   const NaviErrorPtr &error)
{
    auto scopeInfoPtr = getScopeInfo(scopeIndex);
    if (!scopeInfoPtr) {
        return false;
    }
    auto &scopeInfo = *scopeInfoPtr;
    if (!scopeInfo.internal->terminator) {
        return false;
    }
    bool expect = false;
    if (!scopeInfo.terminated.compare_exchange_weak(expect, true)) {
        return true;
    }
    scopeInfo.setError(error);
    NAVI_MEMORY_BARRIER();
    forceStopNodes(node, scopeInfo.internal->nodes);
    return true;
}

void LocalSubGraph::forceStopNodes(Node *stopBy,
                                   const std::vector<NodeInternal *> &nodes) {
    if (!_nodeVec) {
        return;
    }
    for (auto n : nodes) {
        _nodeVec[n->_id].setForceStopFlag(stopBy);
    }
    for (auto n : nodes) {
        _nodeVec[n->_id].forceStop();
    }
}

NaviErrorPtr LocalSubGraph::getScopeError(int32_t scopeIndex) {
    auto scopeInfoPtr = getScopeInfo(scopeIndex);
    if (!scopeInfoPtr) {
        return nullptr;
    }
    return scopeInfoPtr->getError();
}

ErrorHandleStrategy LocalSubGraph::getScopeErrorHandleStrategy(
        int32_t scopeIndex)
{
    auto scopeInfoPtr = getScopeInfo(scopeIndex);
    if (!scopeInfoPtr) {
        return EHS_ERROR_AS_FATAL;
    }
    return scopeInfoPtr->def->error_handle_strategy();
}

const ReplaceInfoMap *LocalSubGraph::getReplaceInfoMap() const {
    return _internal->replaceInfoMap.get();
}

Edge *LocalSubGraph::transformEdgeFromInternal(EdgeInternal *edgeInternal) {
    if (!edgeInternal || !_edgeVec) {
        return nullptr;
    }
    auto id = edgeInternal->id;
    if (id < 0 || id >= _internal->edgeInternalVec.size()) {
        return nullptr;
    }
    return &_edgeVec[id];
}

Node *LocalSubGraph::transformNodeFromInternal(NodeInternal *internal) {
    if (!internal || !_nodeVec) {
        return nullptr;
    }
    auto id = internal->_id;
    if (id < 0 || id >= _internal->nodeVec.size()) {
        return nullptr;
    }
    return &_nodeVec[id];
}

TestMode LocalSubGraph::getTestMode() const { return _param->worker->getTestMode(); }

}
