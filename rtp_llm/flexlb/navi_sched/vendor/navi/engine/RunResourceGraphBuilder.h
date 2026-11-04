#pragma once

#include "navi/builder/GraphBuilder.h"
#include "navi/engine/CreatorManager.h"
#include "navi/engine/MultiLayerResourceMap.h"

namespace navi {

class RunResourceGraphBuilder
{
public:
    RunResourceGraphBuilder(CreatorManager *creatorManager,
                            const MultiLayerResourceMap &multiLayerResourceMap,
                            int32_t scope,
                            const std::string &requireNode,
                            int32_t requireNodeIndex,
                            const NaviUnorderedMap<std::string, std::string> &replaceMap,
                            GraphBuilder &builder);
    ~RunResourceGraphBuilder();
    RunResourceGraphBuilder(const RunResourceGraphBuilder &) = delete;
    RunResourceGraphBuilder &operator=(const RunResourceGraphBuilder &) = delete;
public:
    N build(const std::string &resource);
private:
    const std::string &getReplaceR(const std::string &inputResource) const;
    void addDependResourceRecur(N n, const std::string &inputResource);
    std::string getResourceNodeName(const std::string &resource, ResourceStage stage) const;
private:
    CreatorManager *_creatorManager;
    const MultiLayerResourceMap &_multiLayerResourceMap;
    int32_t _scope;
    const std::string &_requireNode;
    int32_t _requireNodeIndex = 0;
    const NaviUnorderedMap<std::string, std::string> &_replaceMap;
    GraphBuilder &_builder;
    GraphId _graphId;
};

}

