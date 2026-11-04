#pragma once

#include "navi/common.h"
#include "navi/builder/GraphDesc.h"
#include "navi/proto/GraphDef.pb.h"
#include <multi_call/common/common.h>

namespace navi {

class GraphBuilder;
class SubGraphDef;
class NImpl;

class SubGraphBuildInfo {
public:
    SubGraphBuildInfo(GraphBuilder *builder, SubGraphDef *subGraphDef);
    ~SubGraphBuildInfo();
    void location(const std::string &bizName, NaviPartId partCount, NaviPartId partId);
    void directLocation(const std::string &bizName,
                        const std::string &host,
                        int32_t grpcStreamPort);
    void partIds(const std::vector<NaviPartId> &partIds);
    void gigProbe(bool enable);
    void gigTag(const std::string &tag, multi_call::TagMatchType type);
    void subGraphAttr(const std::string &key, const std::string &value);
    void inlineMode(bool inlineMode);
    void errorHandleStrategy(ErrorHandleStrategy strategy);
    void ignoreIsolate(bool ignoreIsolate);
    void subGraphTimeoutMs(int64_t timeoutMs);
    void replaceR(const std::string &from, const std::string &to);
    N addNode(const std::string &name, NodeType nodeType, int32_t scope);
    void collectNodes(std::vector<N> &nodes) const;
    bool hasNode(const std::string &node) const;
    SubGraphDef *getSubGraphDef() const;
    ScopeId addScope();
    ScopeId getCurrentScope() const;
    std::vector<ScopeId> getAllScope() const;
    void setCurrentScope(int32_t index);
    NImplBase *getScopeTerminator(int32_t scopeIndex);
    EdgeParamDef *getEdgeParamDef(int32_t index);
    GraphBuilder *getBuilder() const {
        return _builder;
    }
    GraphId getGraphId() const {
        return _graphId;
    }
    bool isFork() const {
        return _isFork;
    }
    N createScopeTerminator(int32_t scopeIndex);
    void scopeErrorHandleStrategy(int32_t scopeIndex,
                                  ErrorHandleStrategy strategy);
    P getTerminatorInputPort();
public:
    static void initLocation(LocationDef *location,
                             const std::string &bizName,
                             NaviPartId partCount,
                             NaviPartId partId,
                             bool enableProbe);
private:
    void initScope();
    void loadNodeMap();
    void loadEdgeParam();
    N addNodeInner(const std::string &name, NodeType nodeType, int32_t scope);
    NImpl *addNodeFromDef(NodeDef *def, int32_t id);
    int32_t nextScopeId();
    NImplBase *getExistNode(const std::string &nodeName);
    std::string getScopeTerminatorName(int32_t scopeIndex) const;
    void initScopeBorder(int32_t scopeIndex);
private:
    GraphBuilder *_builder;
    SubGraphDef *_subGraphDef;
    GraphId _graphId;
    bool _isFork;
    NaviUnorderedMap<std::string, std::pair<NodeDef *, int32_t>> _nodeDefMap;
    NaviUnorderedMap<std::string, NImplBase *> _nodeMap;
    NaviUnorderedMap<int32_t, EdgeParamDef *> _edgeParamMap;
    std::vector<NImplBase *> _fakeNodeVec;
    P _terminatorInput;
    NaviUnorderedMap<int32_t, SubGraphScopeDef *> _scopeMap;
    ScopeId _currentScope;
};

}
