#pragma once

#include "autil/legacy/json.h"
#include "navi/common.h"
#include "navi/builder/BorderId.h"
#include "navi/builder/GraphDesc.h"
#include <multi_call/common/common.h>

namespace navi {

class SubGraphBuildInfo;

class GraphBuilder
{
public:
    GraphBuilder(GraphDef *def);
    GraphBuilder(SubGraphDef *subGraphDef);
    ~GraphBuilder();
private:
    GraphBuilder(const GraphBuilder &);
    GraphBuilder &operator=(const GraphBuilder &);
public:
    GraphId newSubGraph(const std::string &bizName);
    GraphId newSubGraph(const std::string &bizName, const SubGraphDef &subGraphDefIn);
    ScopeId addScope();
    GraphBuilder &scope(ScopeId scope);
    ScopeId getCurrentScope() const;
    N getScopeTerminator(ScopeId scope = ScopeId());
    GraphBuilder &scopeErrorHandleStrategy(ScopeId scope,
                                           ErrorHandleStrategy strategy);
    GraphBuilder &location(const std::string &bizName, NaviPartId partCount,
                           NaviPartId partId);
    GraphBuilder &directLocation(const std::string &bizName,
                                 const std::string &host,
                                 int32_t grpcStreamPort);
    GraphBuilder &partIds(const std::vector<NaviPartId> &partIds);
    GraphBuilder &gigProbe(bool enable);
    GraphBuilder &gigTag(const std::string &tag, multi_call::TagMatchType type);
    GraphBuilder &subGraph(GraphId graphId);
    GraphBuilder &inlineMode(bool inlineMode);
    GraphBuilder &errorHandleStrategy(ErrorHandleStrategy strategy);
    GraphBuilder &subGraphAttr(const std::string &key, const std::string &value);
    GraphBuilder &ignoreIsolate(bool ignoreIsolate);
    GraphBuilder &subGraphTimeoutMs(int64_t timeoutMs);
    GraphBuilder &replaceR(const std::string &from, const std::string &to);
    GraphBuilder &enableBackPressure(int64_t queueLow, int64_t queueHigh);
    N node(const std::string &name);
    N node(GraphId graphId, const std::string &name);
    bool hasNode(GraphId graphId, const std::string &node) const;
    bool hasNode(const std::string &node) const;
    N getOutputNode(const std::string &name);
    std::vector<N> getAllNodes() const;
    GraphId currentSubGraph() const;
    bool hasSubGraph(GraphId graphId) const;
    std::vector<GraphId> getAllGraphId() const;
    std::vector<ScopeId> getAllScope(GraphId graphId) const;
    bool ok();
    const NaviUnorderedMap<BorderId, BorderDef *, BorderIdHash> &
    getBorderMap() const {
        return _borderMap;
    }
public:
    static void fillHashKey(GraphDef *graphDef);

private:
    void initFromGraphDef();
    SubGraphBuildInfo *addGraphBuildInfo(SubGraphDef *subGraphDef);
    SubGraphBuildInfo *getGraphBuildInfo(GraphId graphId) const;
    SubGraphDef *getSubGraphDef(GraphId graph) const;
    N nodeInner(GraphId graphId, const std::string &name, NodeType nodeType, int32_t scope = -1);
    N resourceNode(const std::string &name);
    N resourceNode(GraphId graphId, const std::string &name);
    N getParentNode();
    GraphId addRSGraphResourceGraph();
    void addOutputGraph(GraphId graphId, const std::string &bizName);
    BorderDef *getBorderDef(const BorderId &borderId);
    EdgeDef *linkInternal(const GraphId graphId,
                          const std::string &inputNode,
                          const std::string &inputPort,
                          int32_t inputNodeId,
                          const std::string &outputNode,
                          const std::string &outputPort,
                          int32_t outputNodeId,
                          int32_t &edgeIndex);
    BorderEdgeDef *linkBorder(const BorderId &borderId,
                              const std::string &inputNode,
                              const std::string &inputPort,
                              int32_t inputNodeId,
                              const std::string &outputNode,
                              const std::string &outputPort,
                              int32_t outputNodeId);
    void fillEdgeInfo(EdgeDef *edge,
                      const std::string &inputNode,
                      const std::string &inputPort,
                      int32_t inputNodeId,
                      const std::string &outputNode,
                      const std::string &outputPort,
                      int32_t outputNodeId);
    EdgeParamDef *getEdgeParam(GraphId graphId, int32_t edgeIndex);
    int32_t nextGraphId();
    int32_t nextEdgeId();
    BackPressureQueueDef *getBackPressureQueueDef();
private:
    friend class SubGraphBuildInfo;
    friend class PImpl;
    friend class InterEImpl;
    friend class SingleE;
    friend class BorderEImpl;
    friend class Graph;
    friend class BizManager;
    friend class InitResourceGraphBuilder;
    friend class ResourceGraphBuilder;
    friend class RunResourceGraphBuilder;
private:
    GraphDef *_def;
    GraphCounterInfo *_counterInfo;
    BackPressureQueueDef *_backPressureQueueDef;
    GraphId _graphId;
    NaviUnorderedMap<GraphId, SubGraphBuildInfo *> _graphInfoMap;
    NaviUnorderedMap<BorderId, BorderDef *, BorderIdHash> _borderMap;
};

NAVI_TYPEDEF_PTR(GraphBuilder);

}
