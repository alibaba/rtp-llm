#include "navi/engine/RunResourceGraphBuilder.h"
#include "navi/ops/ResourceKernelDef.h"

namespace navi {

RunResourceGraphBuilder::RunResourceGraphBuilder(
        CreatorManager *creatorManager,
        const MultiLayerResourceMap &multiLayerResourceMap,
        int32_t scope,
        const std::string &requireNode,
        int32_t requireNodeIndex,
        const NaviUnorderedMap<std::string, std::string> &replaceMap,
        GraphBuilder &builder)
    : _creatorManager(creatorManager)
    , _multiLayerResourceMap(multiLayerResourceMap)
    , _scope(scope)
    , _requireNode(requireNode)
    , _requireNodeIndex(requireNodeIndex)
    , _replaceMap(replaceMap)
    , _builder(builder)
    , _graphId(builder.currentSubGraph())
{
}

RunResourceGraphBuilder::~RunResourceGraphBuilder() {
    _builder.subGraph(_graphId);
}

const std::string &RunResourceGraphBuilder::getReplaceR(const std::string &inputResource) const {
    auto it = _replaceMap.find(inputResource);
    if (_replaceMap.end() == it) {
        return inputResource;
    }
    const auto &to = it->second;
    if (!_creatorManager->supportReplace(inputResource, to)) {
        NAVI_KERNEL_LOG(ERROR,
                        "can't replace [%s] by [%s], not enabled, call enableReplaceR first",
                        inputResource.c_str(),
                        to.c_str());
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, "error, can't replace");
    }
    return to;
}

N RunResourceGraphBuilder::build(const std::string &inputResource) {
    const auto &resource = getReplaceR(inputResource);
    auto stage = _creatorManager->getResourceStage(resource);
    auto nodeName = getResourceNodeName(resource, stage);
    if (RS_GRAPH == stage) {
        _builder.addRSGraphResourceGraph();
    }
    if (_builder.hasNode(nodeName)) {
        auto n = _builder.resourceNode(nodeName).integerAttr(RESOURCE_ATTR_REQUIRE, 1);
        return n;
    }
    NAVI_KERNEL_LOG(TRACE3, "add resource create [%s]", resource.c_str());
    auto n = _builder.resourceNode(nodeName)
                 .kernel(RESOURCE_CREATE_KERNEL)
                 .integerAttr(RESOURCE_ATTR_REQUIRE, 1)
                 .attr(RESOURCE_ATTR_NAME, resource);
    if (RS_KERNEL == stage) {
        n.attr(RESOURCE_ATTR_REQUIRE_NODE, _requireNode);
        n.integerAttr(RESOURCE_ATTR_REQUIRE_NODE_INDEX, _requireNodeIndex);
    }
    if (_multiLayerResourceMap.hasResource(resource)) {
        return n;
    }
    addDependResourceRecur(n, resource);
    return n;
}

void RunResourceGraphBuilder::addDependResourceRecur(N n, const std::string &resource) {
    auto dependResourceVecPtr = _creatorManager->getResourceDependResourceVec(resource);
    if (!dependResourceVecPtr) {
        NAVI_KERNEL_LOG(SCHEDULE1, "resource [%s] not registered", resource.c_str());
        return;
    }
    auto inputPort = n.in(NODE_RESOURCE_INPUT_PORT);
    auto curGraphId = _builder.currentSubGraph();
    for (const auto &pair : *dependResourceVecPtr) {
        const auto &dependName = getReplaceR(pair.first);
        if (_multiLayerResourceMap.hasResource(dependName)) {
            NAVI_KERNEL_LOG(SCHEDULE1,
                            "resource [%s] depend [%s] in multiLayerResourceMap, ignore",
                            resource.c_str(),
                            dependName.c_str());
            continue;
        }
        auto stage = _creatorManager->getResourceStage(dependName);
        auto nodeName = getResourceNodeName(dependName, stage);
        if (RS_GRAPH == stage) {
            _builder.addRSGraphResourceGraph();
        } else {
            _builder.subGraph(curGraphId);
        }
        auto nodeExist = _builder.hasNode(nodeName);
        if (!nodeExist) {
            auto dependN = _builder.resourceNode(nodeName)
                               .kernel(RESOURCE_CREATE_KERNEL)
                               .attr(RESOURCE_ATTR_NAME, dependName);
            dependN.out(RESOURCE_CREATE_OUTPUT).to(inputPort);
            if (RS_KERNEL == stage) {
                dependN.attr(RESOURCE_ATTR_REQUIRE_NODE, _requireNode);
                dependN.integerAttr(RESOURCE_ATTR_REQUIRE_NODE_INDEX, _requireNodeIndex);
            }
            NAVI_KERNEL_LOG(SCHEDULE1, "link resource [%s] depend [%s]",
                            resource.c_str(), dependName.c_str());
            addDependResourceRecur(dependN, dependName);
        } else {
            _builder.resourceNode(nodeName).out(RESOURCE_CREATE_OUTPUT).to(inputPort);
            NAVI_KERNEL_LOG(SCHEDULE1, "resource [%s] depend [%s] already exist",
                            resource.c_str(), dependName.c_str());
        }
    }
}

std::string RunResourceGraphBuilder::getResourceNodeName(const std::string &resource, ResourceStage stage) const {
    switch (stage) {
    case RS_SUB_GRAPH:
        return resource;
    case RS_SCOPE:
        return resource + "@scope_" + std::to_string(_scope);
    case RS_KERNEL:
        if (_requireNode.empty()) {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, "error, can't build kernel resource");
        }
        return resource + "@kernel_" + _requireNode;
    default:
        return resource;
    }
}
}

