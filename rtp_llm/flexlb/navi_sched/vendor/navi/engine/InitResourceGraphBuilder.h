#pragma once

#include "navi/builder/GraphBuilder.h"
#include "navi/engine/CreatorManager.h"

namespace navi {

typedef NaviUnorderedMap<ResourceStage, std::map<std::string, bool> >
    ResourceStageMap;

class InitResourceGraphBuilder {
private:
    struct SubGraphInfo {
        ResourceStage stage = RS_UNKNOWN;
        NaviPartId partId = INVALID_NAVI_PART_ID;
    };
    struct RequireNodeInfo {
        GraphId graphId = INVALID_GRAPH_ID;
        std::string node;
        bool rootNode = false;
    };
public:
    InitResourceGraphBuilder(const std::string &bizName,
                             CreatorManager *creatorManager,
                             const ResourceStageMap &resourceStageMap,
                             NaviPartId partCount,
                             const std::vector<NaviPartId> &partIds,
                             bool ignoreProduceError,
                             GraphBuilder &builder);
    ~InitResourceGraphBuilder();
    InitResourceGraphBuilder(const InitResourceGraphBuilder &) = delete;
    InitResourceGraphBuilder &
    operator=(const InitResourceGraphBuilder &) = delete;
public:
    bool build(GraphId buildinGraphId, GraphId &resultGraphId);
private:
    bool addResourceToGraph(const std::string &name, bool require,
                            const RequireNodeInfo &requireInfo);
    bool addResource(GraphId graphId,
                     const std::string &name,
                     const std::string &producer,
                     bool require,
                     const RequireNodeInfo &requireInfo);
    bool addDependResource(GraphId graphId, const std::string &name);
    bool addReplaceResource(GraphId graphId, const std::string &name);
    GraphId addBuildinSubGraph();
    GraphId addBuildinSave();
    GraphId addBizSubGraph();
    void addBizPartPublish();
    void addBizPartSave();
    GraphId addRunGraphSubGraph(GraphId requireGraph);
    bool addSaveToPublish(GraphId from);
    void addFlushInput(N n);
    SubGraphInfo getGraphInfo(GraphId graphId) const;
    void validate(const ResourceCreator *creator,
                  const RequireNodeInfo &requireInfo);
private:
    const std::string &_bizName;
    bool _isBuildin;
    bool _ignoreProduceError;
    CreatorManager *_creatorManager;
    const ResourceStageMap &_resourceStageMap;
    NaviPartId _partCount;
    std::vector<NaviPartId> _partIds;
    GraphBuilder &_builder;
    GraphId _buildinGraphId;
    GraphId _bizGraphId;
    std::map<NaviPartId, GraphId> _partGraphIds;
    NaviUnorderedMap<GraphId, SubGraphInfo> _graphInfoMap;
    bool _hasError;
};

}

