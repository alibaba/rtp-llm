#include "navi/engine/ResourceGraphBuilder.h"
#include "navi/engine/SubGraphBase.h"
#include "navi/ops/ResourceKernelDef.h"

namespace navi {

ResourceGraphBuilder::ResourceGraphBuilder(const RunGraphParams &runParams)
    : _runParams(runParams)
{
}

ResourceGraphBuilder::~ResourceGraphBuilder() {
    for (const auto &pair : _nodeMap) {
        delete pair.second;
    }
}

ResourceGraphNode *ResourceGraphBuilder::addResource(const ResourceBuildParam &param,
                                                     const std::string &inputResource,
                                                     ResourceGraphNode *requireNode,
                                                     int32_t selectIndex)
{
    const auto &resource = getReplaceR(param, inputResource);
    const auto &rewriteInfo = param.rewriteInfo;
    if (rewriteInfo.multiLayerResourceMap.hasResource(resource)) {
        NAVI_KERNEL_LOG(SCHEDULE1, "resource [%s] exist in resource map", resource.c_str());
        return nullptr;
    }
    auto creator = rewriteInfo.biz->getCreatorManager()->getResourceCreator(resource);
    if (!creator) {
        NAVI_KERNEL_LOG(SCHEDULE1, "resource [%s] not registered", resource.c_str());
        return nullptr;
    }
    if (!requireNode) {
        requireNode = addKernelNode(param);
    }
    bool isNewNode = false;
    auto n = addNode(param, resource, inputResource, creator, requireNode, selectIndex, isNewNode);
    if (isNewNode) {
        bool dependMergeUp = true;
        const auto &dependResourceVec = creator->getDependResources();
        for (const auto &pair : dependResourceVec) {
            auto d = addResource(param, pair.first, n, -1);
            if (d && !d->mergeUp) {
                dependMergeUp = false;
            }
        }
        n->mergeUp = dependMergeUp && !n->parallelBuild && creator->getStage() <= RS_SCOPE;
    }
    return n;
}

ResourceGraphNode *ResourceGraphBuilder::addKernelNode(const ResourceBuildParam &param) {
    ResourceNodeId nodeId(param, RS_UNKNOWN, -1);
    auto it = _nodeMap.find(nodeId);
    if (_nodeMap.end() == it) {
        auto newNode = new ResourceGraphNode(nodeId, nullptr, true);
        newNode->mergeNodeInfo.needBuild = true;
        _nodeMap[nodeId] = newNode;
        return newNode;
    } else {
        return it->second;
    }
}

ResourceGraphNode *ResourceGraphBuilder::addNode(const ResourceBuildParam &param,
                                                 const std::string &resource,
                                                 const std::string &inputResource,
                                                 ResourceCreator *creator,
                                                 ResourceGraphNode *requireNode,
                                                 int32_t selectIndex,
                                                 bool &isNewNode)
{
    auto stage = creator->getStage();
    if (stage <= RS_RUN_GRAPH_EXTERNAL) {
        isNewNode = false;
        return nullptr;
    }
    ResourceNodeId nodeId(param, stage, getResourceId(resource));
    auto it = _nodeMap.find(nodeId);
    if (_nodeMap.end() == it) {
        auto newNode = new ResourceGraphNode(nodeId, creator, parallelBuild(creator));
        newNode->originName = &inputResource;
        _nodeMap[nodeId] = newNode;
        if (selectIndex < 0) {
            newNode->outputs.emplace_back(requireNode);
        } else {
            newNode->selectOutputs[requireNode].insert(selectIndex);
        }
        isNewNode = true;
        return newNode;
    } else {
        if (selectIndex < 0) {
            it->second->outputs.emplace_back(requireNode);
        } else {
            it->second->selectOutputs[requireNode].insert(selectIndex);
        }
        isNewNode = false;
        return it->second;
    }
}

bool ResourceGraphBuilder::parallelBuild(ResourceCreator *creator) const {
    if (!creator || creator->isAsync()) {
        return true;
    }
    switch (_runParams.getResourceBuildMode(creator->getName())) {
    case RBM_INLINE:
        return false;
    case RBM_PARALLEL:
        return true;
    case RBM_AUTO:
        return creator->parallelBuild();
    case RBM_UNKNOWN:
    default: {
        switch (creator->getBuildMode()) {
        case RBM_INLINE:
            return false;
        case RBM_PARALLEL:
            return true;
        case RBM_AUTO:
            return creator->parallelBuild();
        case RBM_UNKNOWN:
        default:
            return false;
        }
    }
    }
}

const std::string &ResourceGraphBuilder::getReplaceR(const ResourceBuildParam &param,
                                                     const std::string &inputResource) const
{
    const auto &rewriteInfo = param.rewriteInfo;
    if (!rewriteInfo.replaceInfoMap) {
        return inputResource;
    }
    const auto &resourceReplaceMap = rewriteInfo.replaceInfoMap->getReplaceFrom2To();
    auto it = resourceReplaceMap.find(inputResource);
    if (resourceReplaceMap.end() == it) {
        return inputResource;
    }
    const auto &to = it->second;
    if (!rewriteInfo.biz->getCreatorManager()->supportReplace(inputResource, to)) {
        NAVI_KERNEL_LOG(ERROR,
                        "can't replace [%s] by [%s], not enabled, call enableReplaceR first",
                        inputResource.c_str(),
                        to.c_str());
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, "error, can't replace");
    }
    return to;
}

int32_t ResourceGraphBuilder::getResourceId(const std::string &resource) {
    auto it = _resourceIdMap.find(resource);
    if (_resourceIdMap.end() != it) {
        return it->second;
    }
    int32_t id = _resourceIdMap.size();
    _resourceIdMap[resource] = id;
    return id;
}

const MergeNodeInfo &ResourceGraphBuilder::updateParallelBuild(ResourceGraphNode *node, GraphBuilder &builder) {
    if (node->updated) {
        return node->mergeNodeInfo;
    }
    bool needBuild = node->mergeNodeInfo.needBuild;
    NaviUnorderedMap<ResourceGraphNode *, std::set<int32_t> *> mergeNodeMap;
    for (auto out : node->outputs) {
        const auto &result = updateParallelBuild(out, builder);
        auto mergeNode = result.node;
        auto selectIndexSet = result.selectIndexSet;
        needBuild = (needBuild || result.needBuild);
        if (!mergeNode) {
            continue;
        }
        auto it = mergeNodeMap.find(mergeNode);
        if (mergeNodeMap.end() == it) {
            if (!selectIndexSet) {
                mergeNodeMap.emplace(mergeNode, nullptr);
            } else {
                mergeNodeMap.emplace(mergeNode, new std::set<int32_t>(*selectIndexSet));
            }
        } else {
            if (!selectIndexSet) {
                if (it->second) {
                    it->second->insert(-1);
                }
            } else {
                if (it->second) {
                    it->second->insert(selectIndexSet->begin(), selectIndexSet->end());
                } else {
                    it->second = new std::set<int32_t>(*selectIndexSet);
                    it->second->insert(-1);
                }
            }
        }
    }
    for (const auto &pair : node->selectOutputs) {
        const auto &result = updateParallelBuild(pair.first, builder);
        auto mergeNode = result.node;
        if (!mergeNode) {
            continue;
        }
        const auto &selectIndexSet = pair.second;
        auto it = mergeNodeMap.find(mergeNode);
        if (mergeNodeMap.end() == it) {
            mergeNodeMap.emplace(mergeNode, new std::set<int32_t>(selectIndexSet));
        } else {
            if (it->second) {
                it->second->insert(selectIndexSet.begin(), selectIndexSet.end());
            } else {
                it->second = new std::set<int32_t>(selectIndexSet);
                it->second->insert(-1);
            }
        }
    }
    auto creator = node->creator;
    node->mergeNodeInfo.needBuild = needBuild;
    if (needBuild && node->mergeUp) {
        node->mergeNodeInfo.node = nullptr;
        node->mergeNodeInfo.selectIndexSet = nullptr;
        collectInlineGraphNode(node);
    } else if (node->parallelBuild || mergeNodeMap.size() > 1u) {
        node->mergeNodeInfo.node = node;
        node->mergeNodeInfo.selectIndexSet = nullptr;
    } else {
        auto it = mergeNodeMap.begin();
        auto mergeNode = it->first;
        if (creator && creator->getStage() <= RS_SCOPE && mergeNode && !mergeNode->creator) {
            node->mergeNodeInfo.node = node;
            node->mergeNodeInfo.selectIndexSet = nullptr;
        } else {
            node->mergeNodeInfo.node = mergeNode;
            node->mergeNodeInfo.selectIndexSet = it->second;
            mergeNodeMap.clear();
        }
    }
    NAVI_KERNEL_LOG(SCHEDULE1,
                    "resource [%s] [%s], needBuild [%d], parallelBuild [%d], mergeUp [%d], "
                    "mergeNodeMap size [%lu]",
                    creator ? creator->getName().c_str() : "",
                    creator ? ResourceStage_Name(creator->getStage()).c_str() : "",
                    needBuild,
                    node->parallelBuild,
                    node->mergeUp,
                    mergeNodeMap.size());
    if (node->mergeNodeInfo.node == node) {
        buildNode(node, builder);
        for (const auto &pair : mergeNodeMap) {
            buildNode(pair.first, builder);
            auto e = node->out.to(pair.first->in.autoNext());
            if (pair.second) {
                e.addSelectIndex(*pair.second);
            }
        }
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "parallel build resource [%s] [%s]",
                        creator ? creator->getName().c_str() : "",
                        creator ? ResourceStage_Name(creator->getStage()).c_str() : "");
    } else {
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "inline build resource [%s] [%s], merge to [%s]",
                        creator ? creator->getName().c_str() : ((NodeDef *)node->id.node)->name().c_str(),
                        creator ? ResourceStage_Name(creator->getStage()).c_str() : "",
                        node->mergeNodeInfo.node
                            ? (node->mergeNodeInfo.node->creator
                                   ? node->mergeNodeInfo.node->creator->getName().c_str()
                                   : ((NodeDef *)node->mergeNodeInfo.node->id.node)->name().c_str())
                            : "");
    }
    for (const auto &pair : mergeNodeMap) {
        if (pair.second) {
            delete pair.second;
        }
    }
    node->updated = true;
    return node->mergeNodeInfo;
}

void ResourceGraphBuilder::buildNode(ResourceGraphNode *node, GraphBuilder &builder) {
    if (node->in.isValid()) {
        return;
    }
    if (!node->creator) {
        builder.subGraph(node->id.graphId);
        auto n = builder.node(((NodeDef *)node->id.node)->name());
        node->in = n.in(NODE_RESOURCE_INPUT_PORT);
    } else {
        auto stage = node->creator->getStage();
        if (!builder.hasSubGraph(node->id.graphId)) {
            if (stage == RS_GRAPH) {
                builder.addRSGraphResourceGraph();
            } else {
                builder.subGraph(node->id.graphId);
            }
        } else {
            builder.subGraph(node->id.graphId);
        }
        const auto &name = node->creator->getName();
        auto n = builder
                 .resourceNode(getResourceNodeName(name, stage, node->id.scope, node->id.node))
                 .kernel(RESOURCE_CREATE_KERNEL)
                 .attr(RESOURCE_ATTR_NAME, name);
        if (RS_KERNEL == stage) {
            n.attr(RESOURCE_ATTR_REQUIRE_NODE, ((NodeDef *)node->id.node)->name());
            n.integerAttr(RESOURCE_ATTR_REQUIRE_NODE_INDEX, node->id.nodeIndex);
        }
        node->in = n.in(NODE_RESOURCE_INPUT_PORT);
        node->out = n.out(RESOURCE_CREATE_OUTPUT);
    }
}

void ResourceGraphBuilder::collectInlineGraphNode(ResourceGraphNode *node) {
    if (!node->originName) {
        return;
    }
    if (!_graphInlineBuildMap) {
        _graphInlineBuildMap = std::make_shared<GraphInlineBuildMap>();
    }
    auto id = node->id;
    id.node = nullptr;
    id.resourceId = -1;
    auto it = _graphInlineBuildMap->find(id);
    if (_graphInlineBuildMap->end() == it) {
        (*_graphInlineBuildMap)[id].insert(node->originName);
    } else {
        it->second.insert(node->originName);
    }
}

std::string ResourceGraphBuilder::getResourceNodeName(const std::string &resource,
                                                      ResourceStage stage,
                                                      int32_t scope,
                                                      void *requireNode) const {
    switch (stage) {
    case RS_SUB_GRAPH:
        return resource;
    case RS_SCOPE:
        return resource + "@scope_" + std::to_string(scope);
    case RS_KERNEL: {
        const auto &requireNodeName = ((NodeDef *)requireNode)->name();
        if (requireNodeName.empty()) {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase, "error, can't build kernel resource");
        }
        return resource + "@kernel_" + requireNodeName;
    }
    default:
        return resource;
    }
}

bool ResourceGraphBuilder::build(GraphBuilder &builder) {
    try {
        for (const auto &pair : _nodeMap) {
            updateParallelBuild(pair.second, builder);
        }
        return true;
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_LOG(ERROR, "add resource graph failed, msg [%s]", e.GetMessage().c_str());
        return false;
    }
}
}
