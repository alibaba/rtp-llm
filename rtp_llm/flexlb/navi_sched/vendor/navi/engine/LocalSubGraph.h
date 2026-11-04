#pragma once

#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Biz.h"
#include "navi/engine/Data.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/engine/Port.h"
#include "navi/engine/SubGraphBase.h"
#include "navi/engine/SubGraphBorder.h"
#include "navi/engine/ResourceGraphBuilder.h"
#include "navi/log/NaviLogger.h"
#include <memory>
#include <vector>

namespace navi {

class NaviWorkerBase;
class Node;
class Edge;
class Resource;
struct NodeInternal;
struct EdgeInternal;
class EdgeOutputInfo;

class ScopeInfo {
public:
    struct Internal {
        int32_t scopeId = -1;
        int32_t index = -1;
        NodeInternal *terminator = nullptr;
        std::vector<NodeInternal *> nodes;
    };

public:
    void setError(const NaviErrorPtr &error) {
        autil::ScopedLock lock(_errorMutex);
        if (!_error) {
            _error = error;
        }
    }
    NaviErrorPtr getError() const {
        autil::ScopedLock lock(_errorMutex);
        return _error;
    }
public:
    Internal *internal = nullptr;
    const SubGraphScopeDef *def = nullptr;
    std::atomic<bool> terminated = false;
    NaviLoggerPtr logger;
    NaviObjectLogger objectLogger;
    ErrorHandleStrategy strategy = EHS_ERROR_AS_FATAL;
    ResourceMapPtr resourceMap;
private:
    mutable autil::ThreadMutex _errorMutex;
    NaviErrorPtr _error;
};

struct LocalSubGraphInternal;

class LocalSubGraph : public SubGraphBase
{
public:
    LocalSubGraph(SubGraphDef *subGraphDef,
                  GraphParam *param,
                  const GraphDomainPtr &domain,
                  const SubGraphBorderMapPtr &borderMap,
                  Graph *graph,
                  const RewriteInfoPtr &rewriteInfo,
                  const GraphInlineBuildMapPtr &inlineBuildMap);
    ~LocalSubGraph();
private:
    LocalSubGraph(const LocalSubGraph &);
    LocalSubGraph& operator=(const LocalSubGraph &);
public:
    ErrorCode init() override;
    ErrorCode run() override;
    void collectMetric(GraphMetric *graphMetric) override;
    void collectForkGraphMetric() override;
    void terminate(ErrorCode ec) override;
public:
    Node *getNode(const std::string &nodeName, int32_t portNodeId) const;
    bool schedule(Node *node);
    void setScopeError(int32_t scopeIndex, const NaviErrorPtr &error);
    void notifyFinish(Node *node, const NaviErrorPtr &error);
    NaviWorkerBase *getWorker() const;
    const BizPtr &getBiz() const;
    bool bindPortDataType(Node *node, Port *port, IoType ioType);
    bool saveResource(const ResourceMap &resourceMap);
    bool publishResource();
    void collectResourceMap(int32_t scopeIndex, MultiLayerResourceMap &multiLayerResourceMap) const;
    ResourceStage getResourceStage(const std::string &name) const;
    const std::vector<std::pair<std::string, bool>> *getResourceDependResourceVec(const std::string &name) const {
        return _biz->getResourceDependResourceVec(name);
    }
    const NaviUnorderedSet<std::string> *getResourceReplacerSet(const std::string &name) const;
    ErrorCode createResource(const std::string &name,
                             const MultiLayerResourceMap &multiLayerResourceMap,
                             const ProduceInfo &produceInfo,
                             bool checkReuse,
                             Node *requireNode,
                             bool failAsError,
                             ResourcePtr &resource) const;
    Resource *buildResourceRecur(const std::string &name,
                                 ResourceStage minStage,
                                 NaviConfigContext *inputCtx,
                                 const MultiLayerResourceMap &nodeResourceMap,
                                 const ReplaceInfoMap *replaceInfoMap,
                                 Node *requireNode,
                                 ResourceMap &inputResourceMap);
    bool terminateScope(Node *node, int32_t scopeIndex,
                        const NaviErrorPtr &error);
    NaviErrorPtr getScopeError(int32_t scopeIndex);
    ErrorHandleStrategy getScopeErrorHandleStrategy(int32_t scopeIndex);
    const ReplaceInfoMap *getReplaceInfoMap() const;
    int64_t checkTimeout(int64_t endTimeMs) override;
    Edge *transformEdgeFromInternal(EdgeInternal *internal);
    Node *transformNodeFromInternal(NodeInternal *internal);
    ScopeInfo *getScopeInfo(int32_t scopeIndex);
public:
    TestMode getTestMode() const;
private:
    bool doInit();
    void initSubGraphOption();
    bool initScopeMap(bool needConstructGraph);
    bool addDependResourceRecur(const std::string &resource, bool require,
                                GraphBuilder &builder);
    bool initPortDataType();
    bool setPortType(Port *port, IoType ioType);
    std::string getNodePortDataTypeStr(Node *node, const std::string &portName,
                                       const std::string &peerPortName,
                                       IoType ioType) const;
    bool createInternalNodesAndEdges();
    bool initPortNode();
    void createNodes();
    void addToScopeInfo(ScopeInfo *scopeInfo, NodeType nodeType, NodeInternal *node);
    bool bindPortToNode();
    bool bindPortToNodeInternal();
    bool createEdges();
    void initEdgeTopoParam();
    bool bindForkDomain();
    bool isForkDomain() const;
    NodeInternal *createNode(int32_t id, const NodeDef &nodeDef, KernelMemEstimator &estimator);
    bool createEdge(bool useEdgeIndex, int32_t id, const EdgeDef &edgeDef);
    void collectTimeoutEdge(int64_t timeoutMs, const EdgeOutputInfo &edgeOutputInfo);
    int64_t checkTimeoutEdge(int64_t endTimeMs);
    int64_t checkTimeoutAsyncPipe(int64_t endTimeMs);
    bool registerTimeoutNode(Node *node);
    void unregisterTimeoutNode(Node *node);
    void clearTimeoutNodes();
    void doSchedule();
    bool initInlineResource();
    void initSubGraphResource(const std::set<const std::string *> &resourceList);
    void initScopeResource(const std::set<const std::string *> *resourceList, ScopeInfo &scopeInfo);
    ErrorCode flushEdgeOverride();
    bool inlineCompute(NaviWorkerBase *worker, Node *node);
    void forceStopNodes(Node *stopBy, const std::vector<NodeInternal *> &nodes);
private:
    LocalSubGraphInternal *_internal;
    std::shared_ptr<LocalSubGraphInternal> _owned;
    BizPtr _biz;
    RewriteInfoPtr _rewriteInfo;
    GraphInlineBuildMapPtr _inlineBuildMap;
    ResourceMapPtr _resourceMap;
    void *_graphMem = nullptr;
    Edge *_edgeVec = nullptr;
    Node *_nodeVec = nullptr;
    IndexType _finishPortIndex;
    bool _frozen;
    bool _inlineMode;
    bool _needConstructGraph;
    ErrorHandleStrategy _errorHandleStrategy = EHS_ERROR_AS_FATAL;
    ScopeInfo *_scopeInfos = nullptr;
    std::unique_ptr<std::map<int64_t, std::vector<EdgeOutputInfo>>> _timeoutEdgeMap;
    mutable autil::ThreadMutex _timeoutNodeLock;
    std::shared_ptr<const std::vector<Node *>> _timeoutNodes;
    atomic64_t _timeoutNodeCount;
private:
    friend class Node;
};

NAVI_TYPEDEF_PTR(LocalSubGraph);

}
