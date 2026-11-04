#include "navi/builder/GraphBuilder.h"

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "navi/builder/SubGraphBuildInfo.h"
#include "autil/HashUtil.h"

using namespace std;

namespace navi {

static const std::optional<int64_t> &getBackPressureParams() {
    static const auto params = [&]() -> std::optional<int64_t> {
        int64_t queueLow;
        if (autil::EnvUtil::getEnvWithoutDefault("NAVI_BP_PARAMS_QUEUE_LOW", queueLow) && queueLow > 0 ) {
            return queueLow;
        } else {
            return {};
        }
    }();
    return params;
}

GraphBuilder::GraphBuilder(GraphDef *def)
    : _def(def)
    , _counterInfo(nullptr)
    , _backPressureQueueDef(nullptr)
    , _graphId(INVALID_GRAPH_ID)
{
    const auto &backPressureParams = getBackPressureParams();
    if (backPressureParams) {
        enableBackPressure(*backPressureParams, 2 * *backPressureParams);
    }

    _def->set_version(NAVI_VERSION);
    initFromGraphDef();
}

GraphBuilder::GraphBuilder(SubGraphDef *subGraphDef)
    : _def(nullptr)
    , _counterInfo(nullptr)
{
    _graphId = subGraphDef->graph_id();
    addGraphBuildInfo(subGraphDef);
}

GraphBuilder::~GraphBuilder() {
    for (const auto &pair : _graphInfoMap) {
        delete pair.second;
    }
}

void GraphBuilder::initFromGraphDef() {
    _counterInfo = _def->mutable_counter_info();
    _backPressureQueueDef = _def->has_back_pressure_queue() ? _def->mutable_back_pressure_queue() : nullptr;
    auto graphCount = _def->sub_graphs_size();
    for (int i = 0; i < graphCount; i++) {
        addGraphBuildInfo(_def->mutable_sub_graphs(i));
    }
}

GraphId GraphBuilder::newSubGraph(const std::string &bizName) {
    if (!_def) {
        return INVALID_GRAPH_ID;
    }
    auto subGraphDef = _def->add_sub_graphs();
    auto graphId = nextGraphId();
    subGraphDef->set_graph_id(graphId);
    addGraphBuildInfo(subGraphDef);
    subGraph(graphId);
    location(bizName, INVALID_NAVI_PART_COUNT, INVALID_NAVI_PART_ID);
    return graphId;
}

GraphId GraphBuilder::newSubGraph(const std::string &bizName, const SubGraphDef &subGraphDefIn) {
    if (!_def) {
        return INVALID_GRAPH_ID;
    }
    auto subGraphDef = _def->add_sub_graphs();
    subGraphDef->CopyFrom(subGraphDefIn);
    auto graphId = nextGraphId();
    subGraphDef->set_graph_id(graphId);
    addGraphBuildInfo(subGraphDef);
    subGraph(graphId);
    location(bizName, INVALID_NAVI_PART_COUNT, INVALID_NAVI_PART_ID);
    return graphId;
}

ScopeId GraphBuilder::addScope() {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    return graphBuildInfo->addScope();
}

GraphBuilder &GraphBuilder::scope(ScopeId scope) {
    _graphId = scope.graphId;
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->setCurrentScope(scope.scopeIndex);
    return *this;
}

ScopeId GraphBuilder::getCurrentScope() const {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    return graphBuildInfo->getCurrentScope();
}

N GraphBuilder::getScopeTerminator(ScopeId scope) {
    auto graphId = scope.graphId;
    if (INVALID_GRAPH_ID == graphId) {
        graphId = _graphId;
    }
    auto graphBuildInfo = getGraphBuildInfo(graphId);
    assert(graphBuildInfo);
    return graphBuildInfo->createScopeTerminator(scope.scopeIndex);
}

GraphBuilder &GraphBuilder::scopeErrorHandleStrategy(
        ScopeId scope,
        ErrorHandleStrategy strategy)
{
    auto graphId = scope.graphId;
    if (INVALID_GRAPH_ID == graphId) {
        graphId = _graphId;
    }
    auto graphBuildInfo = getGraphBuildInfo(graphId);
    assert(graphBuildInfo);
    graphBuildInfo->scopeErrorHandleStrategy(scope.scopeIndex, strategy);
    return *this;
}

GraphBuilder &GraphBuilder::location(const std::string &bizName,
                                     NaviPartId partCount,
                                     NaviPartId partId)
{
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->location(bizName, partCount, partId);
    return *this;
}

GraphBuilder &GraphBuilder::directLocation(
        const std::string &bizName,
        const std::string &host,
        int32_t grpcStreamPort)
{
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->directLocation(bizName, host, grpcStreamPort);
    return *this;
}

GraphBuilder &GraphBuilder::partIds(const std::vector<NaviPartId> &partIds)
{
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->partIds(partIds);
    return *this;
}

GraphBuilder &GraphBuilder::gigProbe(bool enable) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->gigProbe(enable);
    return *this;
}

GraphBuilder &GraphBuilder::gigTag(const std::string &tag, multi_call::TagMatchType type) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->gigTag(tag, type);
    return *this;
}

GraphBuilder &GraphBuilder::inlineMode(bool inlineMode) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->inlineMode(inlineMode);
    return *this;
}

GraphBuilder &GraphBuilder::errorHandleStrategy(ErrorHandleStrategy strategy) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->errorHandleStrategy(strategy);
    return *this;
}

GraphBuilder &GraphBuilder::subGraphAttr(const std::string &key, const std::string &value)
{
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->subGraphAttr(key, value);
    return *this;
}

GraphBuilder &GraphBuilder::ignoreIsolate(bool ignoreIsolate) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->ignoreIsolate(ignoreIsolate);
    return *this;
}

GraphBuilder &GraphBuilder::subGraphTimeoutMs(int64_t timeoutMs) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->subGraphTimeoutMs(timeoutMs);
    return *this;
}

GraphBuilder &GraphBuilder::replaceR(const std::string &from, const std::string &to) {
    auto graphBuildInfo = getGraphBuildInfo(_graphId);
    assert(graphBuildInfo);
    graphBuildInfo->replaceR(from, to);
    return *this;
}

GraphBuilder &GraphBuilder::enableBackPressure(int64_t queueLow, int64_t queueHigh) {
    auto queueDef = _def->mutable_back_pressure_queue();
    queueDef->set_queue_low(queueLow);
    queueDef->set_queue_high(queueHigh);
    _backPressureQueueDef = queueDef;
    return *this;
}

SubGraphBuildInfo *GraphBuilder::addGraphBuildInfo(SubGraphDef *subGraphDef) {
    auto info = new SubGraphBuildInfo(this, subGraphDef);
    _graphInfoMap.emplace(subGraphDef->graph_id(), info);
    return info;
}

SubGraphBuildInfo *GraphBuilder::getGraphBuildInfo(GraphId graphId) const {
    auto it = _graphInfoMap.find(graphId);
    if (_graphInfoMap.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

GraphBuilder &GraphBuilder::subGraph(GraphId graphId) {
    if (_graphId == graphId) {
        return *this;
    }
    if (_graphInfoMap.end() == _graphInfoMap.find(graphId)) {
        auto msg = "sub graph id [" + autil::StringUtil::toString(graphId) + "] not exist, call newSubGraph first";
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    _graphId = graphId;
    return *this;
}

N GraphBuilder::node(const std::string &name) {
    if (INVALID_GRAPH_ID == _graphId) {
        auto msg = "no sub graph, call newSubGraph first";
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    return nodeInner(_graphId, name, NT_NORMAL);
}

N GraphBuilder::node(GraphId graphId, const std::string &name) {
    if (_graphInfoMap.end() == _graphInfoMap.find(graphId)) {
        auto msg = "sub graph id [" + autil::StringUtil::toString(graphId) + "] not exist, call newSubGraph first";
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    return nodeInner(graphId, name, NT_NORMAL);
}

N GraphBuilder::nodeInner(GraphId graphId, const std::string &name, NodeType nodeType, int32_t scope) {
    auto graphBuildInfo = getGraphBuildInfo(graphId);
    assert(graphBuildInfo);
    return graphBuildInfo->addNode(name, nodeType, scope);
}

N GraphBuilder::resourceNode(const std::string &name) {
    return nodeInner(_graphId, name, NT_RESOURCE);
}

N GraphBuilder::resourceNode(GraphId graphId, const std::string &name) {
    return nodeInner(graphId, name, NT_RESOURCE);
}

N GraphBuilder::getParentNode() {
    const auto parentGraph = PARENT_GRAPH_ID;
    addOutputGraph(parentGraph, NAVI_FORK_BIZ);
    return nodeInner(parentGraph, NAVI_FORK_NODE, NT_FORK);
}

N GraphBuilder::getOutputNode(const std::string &name) {
    const auto userGraph = USER_GRAPH_ID;
    addOutputGraph(userGraph, NAVI_USER_GRAPH_BIZ);
    return nodeInner(userGraph, name, NT_NORMAL);
}

GraphId GraphBuilder::addRSGraphResourceGraph() {
    const auto graphId = RS_GRAPH_RESOURCE_GRAPH_ID;
    const auto &bizName = NAVI_BUILDIN_BIZ;
    addOutputGraph(graphId, bizName);
    subGraph(graphId);
    location(bizName, 1, 0);
    return graphId;
}

EdgeDef *GraphBuilder::linkInternal(const GraphId graphId,
                                    const std::string &inputNode,
                                    const std::string &inputPort,
                                    int32_t inputNodeId,
                                    const std::string &outputNode,
                                    const std::string &outputPort,
                                    int32_t outputNodeId,
                                    int32_t &edgeIndex)
{
    auto inputGraphDef = getSubGraphDef(graphId);
    assert(inputGraphDef);
    edgeIndex = inputGraphDef->edges_size();
    auto edge = inputGraphDef->add_edges();
    fillEdgeInfo(edge,
                 inputNode,
                 inputPort,
                 inputNodeId,
                 outputNode,
                 outputPort,
                 outputNodeId);
    return edge;
}

BorderEdgeDef *GraphBuilder::linkBorder(const BorderId &borderId,
                                        const std::string &inputNode,
                                        const std::string &inputPort,
                                        int32_t inputNodeId,
                                        const std::string &outputNode,
                                        const std::string &outputPort,
                                        int32_t outputNodeId)
{
    auto borderDef = getBorderDef(borderId);
    if (!borderDef) {
        return nullptr;
    }
    auto borderEdge = borderDef->add_edges();
    borderEdge->set_edge_id(nextEdgeId());
    auto edge = borderEdge->mutable_edge();
    fillEdgeInfo(edge, inputNode, inputPort, inputNodeId, outputNode, outputPort, outputNodeId);
    return borderEdge;
}

EdgeParamDef *GraphBuilder::getEdgeParam(GraphId graphId, int32_t edgeIndex) {
    auto graphBuildInfo = getGraphBuildInfo(graphId);
    assert(graphBuildInfo);
    return graphBuildInfo->getEdgeParamDef(edgeIndex);
}

SubGraphDef *GraphBuilder::getSubGraphDef(GraphId graph) const {
    auto it = _graphInfoMap.find(graph);
    if (_graphInfoMap.end() != it) {
        return it->second->getSubGraphDef();
    } else {
        return nullptr;
    }
}

std::vector<N> GraphBuilder::getAllNodes() const {
    std::vector<N> nodes;
    for (const auto &pair : _graphInfoMap) {
        pair.second->collectNodes(nodes);
    }
    std::sort(nodes.begin(), nodes.end(), [](N lhs, N rhs) {
        return lhs.name() < rhs.name();
    });
    return nodes;
}

std::vector<GraphId> GraphBuilder::getAllGraphId() const {
    std::vector<GraphId> ret;
    for (const auto &pair : _graphInfoMap) {
        auto graphId = pair.first;
        if (PARENT_GRAPH_ID != graphId && USER_GRAPH_ID != graphId) {
            ret.push_back(graphId);
        }
    }
    return ret;
}

std::vector<ScopeId> GraphBuilder::getAllScope(GraphId graphId) const {
    auto graphBuildInfo = getGraphBuildInfo(graphId);
    if (!graphBuildInfo) {
        auto msg = "sub graph id [" + autil::StringUtil::toString(graphId) + "] not exist, call newSubGraph first";
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    return graphBuildInfo->getAllScope();
}

bool GraphBuilder::hasSubGraph(GraphId graphId) const {
    return (getSubGraphDef(graphId) != nullptr);
}

BorderDef *GraphBuilder::getBorderDef(const BorderId &borderId) {
    auto it = _borderMap.find(borderId);
    if (_borderMap.end() != it) {
        return it->second;
    } else {
        auto graphDef = getSubGraphDef(borderId.id);
        if (!graphDef) {
            return nullptr;
        }
        auto borderDef = graphDef->add_borders();
        borderDef->set_io_type((int32_t)borderId.ioType);
        auto peerInfo = borderDef->mutable_peer();
        peerInfo->set_graph_id(borderId.peer);
        auto *partInfo = peerInfo->mutable_part_info();
        partInfo->set_part_count(INVALID_NAVI_PART_COUNT);
        _borderMap.emplace(borderId, borderDef);
        return borderDef;
    }
}

void GraphBuilder::addOutputGraph(GraphId graphId,
                                  const std::string &bizName)
{
    auto outputGraph = getSubGraphDef(graphId);
    if (!outputGraph) {
        outputGraph = _def->add_sub_graphs();
        outputGraph->set_graph_id(graphId);
        auto location = outputGraph->mutable_location();
        SubGraphBuildInfo::initLocation(location, bizName,
                INVALID_NAVI_PART_COUNT, INVALID_NAVI_PART_ID, true);
        addGraphBuildInfo(outputGraph);
    }
}

void GraphBuilder::fillEdgeInfo(EdgeDef *edge,
                                const std::string &inputNode,
                                const std::string &inputPort,
                                int32_t inputNodeId,
                                const std::string &outputNode,
                                const std::string &outputPort,
                                int32_t outputNodeId)
{
    auto input = edge->mutable_input();
    input->set_node_name(inputNode);
    input->set_port_name(inputPort);
    input->set_node_id(inputNodeId);
    auto output = edge->mutable_output();
    output->set_node_name(outputNode);
    output->set_port_name(outputPort);
    output->set_node_id(outputNodeId);
}

bool GraphBuilder::hasNode(const std::string &node) const {
    return hasNode(_graphId, node);
}

bool GraphBuilder::hasNode(GraphId graphId, const std::string &node) const {
    auto it = _graphInfoMap.find(graphId);
    if (_graphInfoMap.end() != it) {
        auto graphBuildInfo = it->second;
        return graphBuildInfo->hasNode(node);
    } else {
        return false;
    }
}

GraphId GraphBuilder::currentSubGraph() const {
    return _graphId;
}

bool GraphBuilder::ok() {
    return true;
}

int32_t GraphBuilder::nextGraphId() {
    if (!_counterInfo) {
        auto msg = "can't add new sub graph in single subGraph builder, subGraphId: " +
                   autil::StringUtil::toString(_graphId);
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    auto id = _counterInfo->next_graph_id();
    _counterInfo->set_next_graph_id(id + 1);
    return id;
}

int32_t GraphBuilder::nextEdgeId() {
    if (!_counterInfo) {
        auto msg = "can't add cross graph edge in single subGraph builder, subGraphId: " +
                   autil::StringUtil::toString(_graphId);
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, msg);
    }
    auto id = _counterInfo->next_edge_id();
    _counterInfo->set_next_edge_id(id + 1);
    return id;
}

BackPressureQueueDef *GraphBuilder::getBackPressureQueueDef() {
    return _backPressureQueueDef;
}

void GraphBuilder::fillHashKey(GraphDef *graphDef) {
    if (!graphDef) {
        return;
    }
    auto subGraphCount = graphDef->sub_graphs_size();
    for (int64_t i = 0; i < subGraphCount; i++) {
        auto subGraphDef = graphDef->mutable_sub_graphs(i);
        const auto &subGraph = *subGraphDef;
        auto nodeCount = subGraph.nodes_size();
        size_t seed = 0;
        autil::HashUtil::combineHash(seed, nodeCount);
        for (auto i = 0; i < nodeCount; i++) {
            const auto &node = subGraph.nodes(i);
            autil::HashUtil::combineHash(seed, i);
            autil::HashUtil::combineHash(seed, node.kernel_name().size());
            autil::HashUtil::combineHash(seed, node.kernel_name());
            autil::HashUtil::combineHash(seed, node.scope());
        }
        auto edgeCount = subGraph.edges_size();
        std::vector<std::pair<size_t, int32_t>> edgeVec;
        {
            edgeVec.resize(edgeCount);
            for (auto i = 0; i < edgeCount; i++) {
                size_t edgeHash = 0;
                const auto &edgeDef = subGraph.edges(i);
                const auto &inputDef = edgeDef.input();
                const auto &outputDef = edgeDef.output();
                autil::HashUtil::combineHash(edgeHash, inputDef.node_id());
                autil::HashUtil::combineHash(edgeHash, inputDef.port_name().size());
                autil::HashUtil::combineHash(edgeHash, inputDef.port_name());
                autil::HashUtil::combineHash(edgeHash, outputDef.node_id());
                autil::HashUtil::combineHash(edgeHash, outputDef.port_name().size());
                autil::HashUtil::combineHash(edgeHash, outputDef.port_name());
                edgeVec[i] = std::make_pair(edgeHash, i);
            }
            std::sort(
                edgeVec.begin(), edgeVec.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
        }
        autil::HashUtil::combineHash(seed, edgeCount);
        for (auto i = 0; i < edgeCount; i++) {
            const auto &edgeInfo = edgeVec[i];
            autil::HashUtil::combineHash(seed, edgeInfo.first);
            subGraphDef->mutable_edges(edgeInfo.second)->set_edge_index(i + 1);
        }
        auto replaceCount = subGraph.replace_infos_size();
        autil::HashUtil::combineHash(seed, replaceCount);
        for (int i = 0; i < replaceCount; i++) {
            const auto &replaceInfo = subGraph.replace_infos(i);
            autil::HashUtil::combineHash(seed, replaceInfo.from().size());
            autil::HashUtil::combineHash(seed, replaceInfo.from());
            autil::HashUtil::combineHash(seed, replaceInfo.to().size());
            autil::HashUtil::combineHash(seed, replaceInfo.to());
        }
        autil::HashUtil::combineHash(seed, subGraph.scopes_size());
        if (0 == seed) {
            autil::HashUtil::combineHash(seed, std::string("navi"));
        }
        subGraphDef->set_hash_key(seed);
    }
}

}
