#pragma once

#include "navi/common.h"
#include "navi/proto/KernelDef.pb.h"
#include "navi/builder/GraphBuilder.h"
#include <vector>

namespace navi {

class ResourceCreator;
class RunGraphParams;
struct RewriteInfo;
class GraphBuilder;
struct ResourceGraphNode;

struct ResourceBuildParam {
public:
    ResourceBuildParam(const RewriteInfo &rewriteInfo_)
        : rewriteInfo(rewriteInfo_)
    {
    }
public:
    const RewriteInfo &rewriteInfo;
    GraphId graphId = INVALID_GRAPH_ID;
    int32_t scope = -1;
    void *requireNodeId = nullptr;
    int32_t requireNodeIndex = 0;
};

struct ResourceNodeId {
public:
    ResourceNodeId() {
    }
    ResourceNodeId(const ResourceBuildParam &param, ResourceStage stage, int32_t resourceId_) {
        switch (stage) {
        case RS_GRAPH:
            graphId = RS_GRAPH_RESOURCE_GRAPH_ID;
            resourceId = resourceId_;
            break;
        case RS_SUB_GRAPH:
            graphId = param.graphId;
            resourceId = resourceId_;
            break;
        case RS_SCOPE:
            graphId = param.graphId;
            scope = param.scope;
            resourceId = resourceId_;
            break;
        case RS_KERNEL:
            graphId = param.graphId;
            scope = param.scope;
            node = param.requireNodeId;
            nodeIndex = param.requireNodeIndex;
            resourceId = resourceId_;
            break;
        case RS_UNKNOWN:
        default:
            graphId = param.graphId;
            node = param.requireNodeId;
            nodeIndex = param.requireNodeIndex;
            break;
        }
    }
    bool operator<(const ResourceNodeId &other)  const {
        if (graphId != other.graphId) {
            return graphId < other.graphId;
        }
        if (scope != other.scope) {
            return scope < other.scope;
        }
        if (node != other.node) {
            return node < other.node;
        }
        if (nodeIndex != other.nodeIndex) {
            return nodeIndex < other.nodeIndex;
        }
        return resourceId < other.resourceId;
    }
public:
    GraphId graphId = INVALID_GRAPH_ID;
    int32_t scope = -2;
    void *node = nullptr;
    int32_t nodeIndex = 0;
    int32_t resourceId = -1;
};

constexpr int32_t MERGE_NODE_THIS = -1;
constexpr int32_t MERGE_NODE_KERNEL = -2;

struct MergeNodeInfo {
private:
    MergeNodeInfo(const MergeNodeInfo &);
    MergeNodeInfo& operator=(const MergeNodeInfo &);
public:
    MergeNodeInfo() {
    }
    ~MergeNodeInfo() {
        DELETE_AND_SET_NULL(selectIndexSet);
    }
public:
    ResourceGraphNode *node = nullptr;
    std::set<int32_t> *selectIndexSet = nullptr;
    bool needBuild = false;
};

struct ResourceGraphNode {
public:
    ResourceGraphNode(const ResourceNodeId &id_, ResourceCreator *creator_, bool parallelBuild_)
        : id(id_)
        , parallelBuild(parallelBuild_)
        , creator(creator_)
    {
    }
public:
    ResourceNodeId id;
    bool updated = false;
    bool mergeUp = false;
    bool parallelBuild = false;
    ResourceCreator *creator;
    const std::string *originName = nullptr;
    std::vector<ResourceGraphNode *> outputs;
    std::map<ResourceGraphNode *, std::set<int32_t>> selectOutputs;
    MergeNodeInfo mergeNodeInfo;
    P in;
    P out;
};

typedef std::map<ResourceNodeId, std::set<const std::string *>> GraphInlineBuildMap;
NAVI_TYPEDEF_PTR(GraphInlineBuildMap);

class ResourceGraphBuilder {
public:
    ResourceGraphBuilder(const RunGraphParams &runParams);
    ~ResourceGraphBuilder();
public:
    ResourceGraphNode *addResource(const ResourceBuildParam &param,
                                   const std::string &inputResource,
                                   ResourceGraphNode *requireNode,
                                   int32_t selectIndex);
    bool needBuild() const {
        return !_nodeMap.empty();
    }
    bool build(GraphBuilder &builder);
    const GraphInlineBuildMapPtr &getGraphInlineBuildMap() const {
        return _graphInlineBuildMap;
    }
private:
    ResourceGraphNode *addKernelNode(const ResourceBuildParam &param);
    ResourceGraphNode *addNode(const ResourceBuildParam &param,
                               const std::string &resource,
                               const std::string &inputResource,
                               ResourceCreator *creator,
                               ResourceGraphNode *requireNode,
                               int32_t selectIndex,
                               bool &isNewNode);
    bool parallelBuild(ResourceCreator *creator) const;
    const std::string &getReplaceR(const ResourceBuildParam &param, const std::string &inputResource) const;
    int32_t getResourceId(const std::string &resource);
    const MergeNodeInfo &updateParallelBuild(ResourceGraphNode *node, GraphBuilder &builder);
    void buildNode(ResourceGraphNode *node, GraphBuilder &builder);
    std::string
    getResourceNodeName(const std::string &resource, ResourceStage stage, int32_t scope, void *requireNode) const;
    void collectInlineGraphNode(ResourceGraphNode *node);
private:
    const RunGraphParams &_runParams;
    NaviUnorderedMap<std::string, int32_t> _resourceIdMap;
    std::map<ResourceNodeId, ResourceGraphNode *> _nodeMap;
    GraphInlineBuildMapPtr _graphInlineBuildMap;
};

NAVI_TYPEDEF_PTR(ResourceGraphBuilder);

}
