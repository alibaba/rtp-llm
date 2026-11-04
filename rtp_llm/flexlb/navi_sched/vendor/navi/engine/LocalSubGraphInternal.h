#pragma once

#include "navi/engine/Edge.h"

namespace navi {

struct NodeInternal;
struct EdgeInternal;

struct LocalSubGraphInternal {
public:
    LocalSubGraphInternal();
    ~LocalSubGraphInternal();
public:
    size_t getSize() const {
        return size;
    }
    void updateAccessTime() {
        lastAccessTime = autil::TimeUtility::currentTime();
    }
    void fillProto(GraphCacheSummaryDef &summaryDef) const;
public:
    size_t size = 0;
    size_t hashKey = 0;
    size_t mallocSize = 0;
    volatile int64_t lastAccessTime = 0;
    SubGraphDef *subGraphDef;
    NaviUnorderedMap<std::string, NodeInternal *> nodeMap;
    std::vector<NodeInternal *> nodeVec;
    std::vector<EdgeOutputInfo> edgeVec;
    NaviUnorderedMap<uint64_t, EdgeInternal *> edges;
    std::vector<EdgeInternal *> edgeInternalVec;
    NaviUnorderedMap<int32_t, ScopeInfo::Internal> scopeInfoMap;
    GraphInlineBuildMapPtr inlineBuildMap;
    ReplaceInfoMapPtr replaceInfoMap;
    bool graphStarted;
    std::vector<uint32_t> startedNodes;
    SessionId session;
    std::string peerIp;
public:
    NodeInternal *getNode(const std::string &nodeName, int32_t portNodeId) const;
    bool postInit(CreatorManager *creatorManager);
private:
    bool checkEdge(CreatorManager *creatorManager) const;
    void initSize();
};

NAVI_TYPEDEF_PTR(LocalSubGraphInternal);

}

