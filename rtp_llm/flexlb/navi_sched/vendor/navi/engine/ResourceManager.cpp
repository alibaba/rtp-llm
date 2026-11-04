#include "navi/engine/ResourceManager.h"
#include "navi/engine/Biz.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/engine/NamedData.h"
#include "navi/engine/NaviSnapshotSummary.h"
#include "navi/engine/Node.h"
#include "navi/engine/ResourceInitContext.h"
#include "navi/engine/SubGraphBase.h"
#include "navi/perf/NaviSymbolTable.h"
#include "navi/log/NaviLogger.h"
#include "navi/rpc_server/NaviRpcServerR.h"
#include "navi/util/CommonUtil.h"

namespace navi {

ResourceManager::ResourceManager(const Biz *biz,
                                 const std::string &bizName,
                                 const std::string &configPath,
                                 NaviPartId partCount,
                                 NaviPartId partId,
                                 const CreatorManager &creatorManager,
                                 const NaviRegistryConfigMap &configMap)
    : _biz(biz)
    , _bizName(bizName)
    , _configPath(configPath)
    , _partCount(partCount)
    , _partId(partId)
    , _testMode(_biz->getTestMode())
    , _creatorManager(creatorManager)
    , _configMap(configMap)
    , _oldResourceManager(nullptr)
    , _gigRpcServer(nullptr)
{
}

ResourceManager::~ResourceManager() {
    NAVI_KERNEL_LOG(INFO,
                    "resource manager destructed, biz [%s] success, partCount: %d, partId: %d",
                    _bizName.c_str(),
                    _partCount,
                    _partId);
}

bool ResourceManager::preInit(ResourceManager *oldResourceManager) {
    _oldResourceManager = oldResourceManager;
    return true;
}

bool ResourceManager::saveResource(const ResourceMap &resourceMap) {
    auto newResourceMapPtr = std::make_shared<ResourceMap>();
    ResourceMap &newResourceMap = *newResourceMapPtr;
    const auto &map = resourceMap.getMap();
    for (const auto &pair : map) {
        const auto &resource = pair.second;
        if (!resource) {
            NAVI_KERNEL_LOG(ERROR, "save resource failed, resource [%s] is null", pair.first.c_str());
            return false;
        }
        auto stage = _creatorManager.getResourceStage(pair.first);
        if (stage != RS_EXTERNAL && stage >= RS_RUN_GRAPH_EXTERNAL) {
            continue;
        }
        newResourceMap.addResource(resource);
    }
    setResourceMap(newResourceMapPtr);
    if (NAVI_BUILDIN_BIZ == _bizName) {
        auto newMerge = std::make_shared<ResourceMap>();
        newMerge->setResource(newResourceMap);
        setMergedResourceMap(newMerge);
    }
    NAVI_KERNEL_LOG(INFO, "saved resource [%s]", autil::StringUtil::toString(getResourceMap()).c_str());
    return true;
}

bool ResourceManager::collectStaticGraphInfo() {
    auto infoMap = std::make_shared<NaviUnorderedMap<std::string, StaticGraphInfoPtr>>();
    NaviUnorderedMap<std::string, std::pair<std::string, ResourcePtr>> producerMap;
    auto mergedResourceMap = getMergedResourceMap();
    if (mergedResourceMap) {
        auto map = mergedResourceMap->getMap();
        for (const auto &pair : map) {
            const auto &resourceName = pair.first;
            const auto &resource = pair.second;
            if (!resource) {
                NAVI_KERNEL_LOG(ERROR, "collect static graph info failed, resource [%s] is null", resourceName.c_str());
                return false;
            }
            std::vector<StaticGraphInfoPtr> graphInfos;
            if (!resource->getStaticGraphInfo(graphInfos)) {
                continue;
            }
            for (const auto &info : graphInfos) {
                if (!info) {
                    continue;
                }
                const auto &name = info->name;
                if (name.empty()) {
                    NAVI_KERNEL_LOG(ERROR, "empty static graph name, produced by resource [%s]", resourceName.c_str());
                    return false;
                }
                if (infoMap->end() != infoMap->find(name)) {
                    std::string previousProducer;
                    auto it = producerMap.find(name);
                    if (producerMap.end() != it) {
                        if (it->second.second == resource) {
                            continue;
                        } else {
                            previousProducer = it->second.first;
                        }
                    }
                    NAVI_KERNEL_LOG(ERROR,
                                    "graph [%s] produced by two resource, [%s] and [%s]",
                                    name.c_str(),
                                    previousProducer.c_str(),
                                    resourceName.c_str());
                    return false;
                }
                (*infoMap)[name] = info;
                producerMap[name] = pair;
            }
        }
    }
    autil::ScopedLock lock(_publishLock);
    _staticGraphInfoMap = std::move(infoMap);
    return true;
}

bool ResourceManager::publishResource(ResourceManager *buildinResourceManager,
                                      ResourceManager *bizResourceManager,
                                      const ResourceStageMap &resourceStageMap)
{
    if ((buildinResourceManager == this) || (bizResourceManager == this)) {
        NAVI_KERNEL_LOG(ERROR, "buildin or biz resource manager can't publish");
        return false;
    }
    auto newResourceMap = std::make_shared<ResourceMap>();
    {
        auto resourceMap = getResourceMap();
        if (resourceMap) {
            newResourceMap->addResource(*resourceMap);
        }
    }
    if (buildinResourceManager) {
        auto resourceMap = buildinResourceManager->getResourceMap();
        if (resourceMap) {
            newResourceMap->addResource(*resourceMap);
        }
    }
    if (bizResourceManager) {
        auto resourceMap = bizResourceManager->getResourceMap();
        if (resourceMap) {
            newResourceMap->addResource(*resourceMap);
        }
    }
    setMergedResourceMap(newResourceMap);
    if (!validateResource(resourceStageMap)) {
        NAVI_KERNEL_LOG(ERROR, "validate resource failed");
        return false;
    }
    if (NAVI_BUILDIN_BIZ != _bizName && NAVI_BIZ_PART_ID != _partId) {
        if (!collectStaticGraphInfo()) {
            return false;
        }
    }
    if (!updatePublishMeta()) {
        return false;
    }
    NAVI_KERNEL_LOG(INFO, "publish resource success");
    return true;
}

bool ResourceManager::updatePublishMeta() {
    {
        autil::ScopedLock lock(_publishLock);
        if (!_gigRpcServer) {
            NAVI_KERNEL_LOG(INFO, "publish ignored, wait for publish");
            return true;
        }
    }
    std::vector<multi_call::ServerBizTopoInfo> infoVec;
    if (!collectPublish(infoVec)) {
        return false;
    }
    std::vector<multi_call::SignatureTy> signatureVec;
    if (!_gigRpcServer->publish(infoVec, signatureVec)) {
        NAVI_KERNEL_LOG(ERROR, "publish topo info failed");
        return false;
    }
    return true;
}

bool ResourceManager::collectPublish(std::vector<multi_call::ServerBizTopoInfo> &infoVec) {
    autil::ScopedLock lock(_publishLock);
    const auto &publishInfos = _biz->getPublishInfos();
    if (publishInfos.disablePublish) {
        return true;
    }
    const auto &aliasNames = publishInfos.bizAliasNames;
    multi_call::ServerBizTopoInfo info;
    info.partCount = _partCount;
    info.partId = _partId;
    info.version = publishInfos.version;
    info.protocolVersion = 0;
    info.targetWeight = 100;
    info.flowControlConfig = _biz->getFlowControlConfig();
    if (!getPublishMetaMap(info.metas)) {
        return false;
    }
    auto isDefault = _biz->isDefault();
    if (!aliasNames.empty()) {
        for (const auto &publish : aliasNames) {
            info.bizName = publish;
            if (_bizName == publish) {
                info.isDefaultBiz = isDefault;
            } else {
                info.isDefaultBiz = false;
            }
            infoVec.push_back(info);
        }
    } else {
        info.bizName = _bizName;
        info.isDefaultBiz = isDefault;
        infoVec.push_back(info);
    }
     return true;
}

void ResourceManager::startPublish(multi_call::GigRpcServer *gigRpcServer) {
    {
        autil::ScopedLock lock(_publishLock);
        _oldResourceManager = nullptr;
        if (NAVI_BIZ_PART_ID == _partId) {
            return;
        }
        _gigRpcServer = gigRpcServer;
        if (!_gigRpcServer) {
            NAVI_KERNEL_LOG(INFO, "publish disabled, no gigRpcServer");
        }
    }
    updatePublishMeta();
}

bool ResourceManager::getPublishMetaMap(std::map<std::string, std::string> &metaMap) const {
    auto mergedResourceMap = getMergedResourceMap();
    if (mergedResourceMap) {
        auto map = mergedResourceMap->getMap();
        for (const auto &pair : map) {
            const auto &name = pair.first;
            const auto &resource = pair.second;
            if (!resource) {
                NAVI_KERNEL_LOG(ERROR, "get resource meta failed, resource [%s] is null", name.c_str());
                return false;
            }
            std::string meta;
            if (!resource->getMeta(meta)) {
                continue;
            }
            metaMap[name] = meta;
        }
    }
    if (!addStaticGraphMeta(metaMap)) {
        return false;
    }
    return true;
}

bool ResourceManager::addStaticGraphMeta(std::map<std::string, std::string> &metaMap) const {
    if (metaMap.end() != metaMap.find(NAVI_BUILDIN_STATIC_GRAPH_META)) {
        NAVI_KERNEL_LOG(ERROR, "buildin meta key [%s] exist in meta map", NAVI_BUILDIN_STATIC_GRAPH_META.c_str());
        return false;
    }
    if (!_staticGraphInfoMap) {
        return true;
    }
    std::map<std::string, std::string> metaInfoMap;
    for (const auto &pair : *_staticGraphInfoMap) {
        metaInfoMap[pair.first] = pair.second->meta;
    }
    auto metaStr = autil::legacy::FastToJsonString(metaInfoMap);
    metaMap[NAVI_BUILDIN_STATIC_GRAPH_META] = metaStr;
    return true;
}

void ResourceManager::setMergedResourceMap(const ResourceMapPtr &resourceMap) {
    autil::ScopedWriteLock lock(_mergeMapLock);
    _mergedResourceMap = resourceMap;
}

ResourceMapPtr ResourceManager::getMergedResourceMap() const {
    autil::ScopedReadLock lock(_mergeMapLock);
    return _mergedResourceMap;
}

void ResourceManager::setResourceMap(const ResourceMapPtr &resourceMap) {
    autil::ScopedLock lock(_resourceMapLock);
    _resourceMap = resourceMap;
}

ResourceMapPtr ResourceManager::getResourceMap() const {
    autil::ScopedLock lock(_resourceMapLock);
    return _resourceMap;
}

NaviConfigContext *ResourceManager::getConfigCtx(const std::string &name,
                                                 NaviConfigContext *inputCtx,
                                                 ResourceStage stage,
                                                 NaviConfigContext *configCtx,
                                                 Node *requireNode) const {
    auto ctx = inputCtx;
    if (RS_KERNEL == stage) {
        if (!ctx) {
            if (!requireNode) {
                NAVI_KERNEL_LOG(ERROR,
                                "can't build resource [%s] of stage [%s], lack require kernel node, biz: [%s]",
                                name.c_str(),
                                ResourceStage_Name(stage).c_str(),
                                _bizName.c_str());
                return nullptr;
            }
            if (!requireNode->getConfigContext(name, *configCtx)) {
                NAVI_KERNEL_LOG(
                    ERROR,
                    "create resource failed, get require kernel config context failed, node [%s] kernel [%s]",
                    requireNode->getName().c_str(),
                    requireNode->getKernelName().c_str());
                return nullptr;
            }
            ctx = configCtx;
        }
    } else {
        ctx = configCtx;
    }
    return ctx;
}

Resource *ResourceManager::buildResourceRecur(const std::string &resourceName,
                                              ResourceStage minStage,
                                              NaviConfigContext *inputCtx,
                                              const SubNamedDataMap *namedDataMap,
                                              const MultiLayerResourceMap &nodeResourceMap,
                                              const ReplaceInfoMap *replaceInfoMap,
                                              Node *requireNode,
                                              bool failAsError,
                                              ResourceMap &inputResourceMap) const
{
    const std::string *namePtr = replaceInfoMap ? &replaceInfoMap->getReplaceRTo(resourceName) : &resourceName;
    const auto &name = *namePtr;
    bool exist = false;
    auto result = inputResourceMap.getResource(name, exist);
    if (exist) {
        return result;
    }
    result = nodeResourceMap.getResource(name, exist);
    if (exist) {
        return result;
    }
    auto creator = _creatorManager.getResourceCreator(name);
    if (!creator) {
        inputResourceMap.addResource(name, nullptr);
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(
            failAsError, WARN, "resource [%s] not registered, biz: [%s]", name.c_str(), _bizName.c_str());
        return nullptr;
    }
    auto stage = creator->getStage();
    if (RS_EXTERNAL == stage) {
        inputResourceMap.addResource(name, nullptr);
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      DEBUG,
                                      "can't create external resource [%s], stage [%s], biz [%s]",
                                      name.c_str(),
                                      ResourceStage_Name(stage).c_str(),
                                      _bizName.c_str());
        return nullptr;
    }
    if (TM_NONE == _testMode && stage < minStage) {
        inputResourceMap.addResource(name, nullptr);
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      DEBUG,
                                      "can't build resource [%s] of stage [%s], biz: [%s], minStage [%s]",
                                      name.c_str(),
                                      ResourceStage_Name(stage).c_str(),
                                      _bizName.c_str(),
                                      ResourceStage_Name(minStage).c_str());
        return nullptr;
    }
    auto *jsonConfig = _configMap.getConfig(name);
    ResourceConfigContext thisContext(_configPath, jsonConfig, nullptr);
    auto ctx = getConfigCtx(name, inputCtx, stage, &thisContext, requireNode);
    if (!ctx) {
        inputResourceMap.addResource(name, nullptr);
        return nullptr;
    }
    auto resource = newAndConfigResource(name, creator, ctx, failAsError);
    if (!resource) {
        inputResourceMap.addResource(name, nullptr);
        return nullptr;
    }
    auto &dependMap = resource->_dependMap;
    const auto &dependResources = creator->getDependResources();
    for (const auto &pair : dependResources) {
        const auto &dependName = pair.first;
        auto require = pair.second;
        auto dependResource = buildResourceRecur(dependName,
                                                 minStage,
                                                 inputCtx,
                                                 namedDataMap,
                                                 nodeResourceMap,
                                                 replaceInfoMap,
                                                 requireNode,
                                                 failAsError && require,
                                                 inputResourceMap);
        if (!dependResource && require) {
            inputResourceMap.addResource(name, nullptr);
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(
                failAsError, DEBUG, "build resource [%s] failed, biz: [%s]", dependName.c_str(), _bizName.c_str());
            return nullptr;
        }
        dependMap.addResource(dependName, dependResource ? dependResource->shared_from_this() : nullptr);
    }
    auto ec = initResource(resource, namedDataMap, {}, creator, ctx, requireNode, failAsError);
    if (EC_NONE == ec) {
        if (!inputResourceMap.addResource(resource)) {
            NAVI_KERNEL_LOG(ERROR, "add [%s] to resource map failed, biz: [%s]", name.c_str(), _bizName.c_str());
            return nullptr;
        }
        return resource.get();
    } else {
        inputResourceMap.addResource(name, nullptr);
        NAVI_KERNEL_LOG(DEBUG,
                        "build resource [%s] failed, biz: [%s], ec: [%s]",
                        name.c_str(),
                        _bizName.c_str(),
                        CommonUtil::getErrorString(ec));
        return nullptr;
    }
}

ErrorCode ResourceManager::createResource(const std::string &name,
                                          ResourceStage minStage,
                                          const MultiLayerResourceMap &multiLayerResourceMap,
                                          const SubNamedDataMap *namedDataMap,
                                          const ProduceInfo &produceInfo,
                                          bool checkReuse,
                                          Node *requireNode,
                                          bool failAsError,
                                          ResourcePtr &retResource) {
    auto resource = getExistResource(name, multiLayerResourceMap);
    if (resource) {
        if (!checkReuse || validateDepend(resource, multiLayerResourceMap)) {
            NAVI_KERNEL_LOG(TRACE3, "reuse resource [%s] [%p]", name.c_str(), resource);
            retResource = resource->shared_from_this();
            return EC_NONE;
        }
    }
    auto creator = _creatorManager.getResourceCreator(name);
    if (!creator) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(
            failAsError, WARN, "resource [%s] not registered, biz: [%s]", name.c_str(), _bizName.c_str());
        return EC_LACK_REGISTRY;
    }
    auto stage = creator->getStage();
    if (RS_EXTERNAL == stage) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      DEBUG,
                                      "can't create external resource [%s], stage [%s], biz [%s]",
                                      name.c_str(),
                                      ResourceStage_Name(stage).c_str(),
                                      _bizName.c_str());
        return EC_ILLEGAL_RESOURCE_STAGE;
    }
    if (RS_RUN_GRAPH_EXTERNAL == stage) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      DEBUG,
                                      "can't create external resource [%s], stage [%s], biz [%s]",
                                      name.c_str(),
                                      ResourceStage_Name(stage).c_str(),
                                      _bizName.c_str());
        return EC_ILLEGAL_RESOURCE_STAGE;
    }
    if (TM_NONE == _testMode && stage < minStage) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      ERROR,
                                      "can't create resource [%s stage: %s] in stage [%s], biz [%s]",
                                      name.c_str(),
                                      ResourceStage_Name(stage).c_str(),
                                      ResourceStage_Name(minStage).c_str(),
                                      _bizName.c_str());
        return EC_ILLEGAL_RESOURCE_STAGE;
    }
    auto *jsonConfig = _configMap.getConfig(name);
    ResourceConfigContext thisContext(_configPath, jsonConfig, nullptr);
    auto ctx = getConfigCtx(name, nullptr, stage, &thisContext, requireNode);
    if (!ctx) {
        return EC_RESOURCE_CONFIG;
    }
    auto resourcePtr = newAndConfigResource(name, creator, ctx, failAsError);
    if (!resourcePtr) {
        return EC_RESOURCE_CONFIG;
    }
    auto ec = collectDependMap(
        creator, minStage, namedDataMap, multiLayerResourceMap, requireNode, failAsError, resourcePtr->_dependMap);
    if (EC_NONE != ec) {
        return ec;
    }
    ec = initResource(resourcePtr, namedDataMap, &produceInfo, creator, ctx, requireNode, failAsError);
    if (EC_NONE == ec) {
        retResource = resourcePtr;
    }
    return ec;
}

ErrorCode ResourceManager::collectDependMap(const ResourceCreator *creator,
                                            ResourceStage minStage,
                                            const SubNamedDataMap *namedDataMap,
                                            const MultiLayerResourceMap &multiLayerResourceMap,
                                            Node *requireNode,
                                            bool failAsError,
                                            ResourceMap &dependResourceMap)
{
    bool lack = false;
    ResourceStage minLackStage = RS_UNKNOWN;
    const auto &dependResources = creator->getDependResources();
    const auto *replaceInfoMap = requireNode ? requireNode->getReplaceInfoMap() : nullptr;
    for (const auto &pair : dependResources) {
        const auto &dependName = pair.first;
        auto require = pair.second;
        auto dependResource = getExistResource(dependName, multiLayerResourceMap);
        if (!dependResource) {
            dependResource = buildResourceRecur(dependName,
                                                minStage,
                                                nullptr,
                                                namedDataMap,
                                                multiLayerResourceMap,
                                                replaceInfoMap,
                                                requireNode,
                                                failAsError && require,
                                                dependResourceMap);
        }
        if (require && !dependResource) {
            lack = true;
            minLackStage = std::min(minLackStage, _creatorManager.getResourceStage(dependName));
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                          WARN,
                                          "create resource [%s] failed, lack required depend "
                                          "resource [%s]",
                                          creator->getName().c_str(),
                                          dependName.c_str());
            continue;
        }
        dependResourceMap.addResource(dependName, dependResource ? dependResource->shared_from_this() : nullptr);
    }
    if (!lack) {
        return EC_NONE;
    }
    switch (minLackStage) {
    case RS_SNAPSHOT:
        return EC_LACK_RESOURCE_SNAPSHOT;
    case RS_BIZ:
        return EC_LACK_RESOURCE_BIZ;
    case RS_BIZ_PART:
        return EC_LACK_RESOURCE_BIZ_PART;
    case RS_GRAPH:
        return EC_LACK_RESOURCE_GRAPH;
    case RS_SUB_GRAPH:
        return EC_LACK_RESOURCE_SUB_GRAPH;
    case RS_SCOPE:
        return EC_LACK_RESOURCE_SCOPE;
    case RS_KERNEL:
        return EC_LACK_RESOURCE_KERNEL;
    case RS_RUN_GRAPH_EXTERNAL:
    case RS_EXTERNAL:
        return EC_LACK_RESOURCE_EXTERNAL;
    case RS_UNKNOWN:
        return EC_NONE;
    default:
        return EC_UNKNOWN;
    }
}

Resource *ResourceManager::getExistResource(const std::string &name,
                                            const MultiLayerResourceMap &multiLayerResourceMap)
{
    auto resource = multiLayerResourceMap.getResource(name);
    NAVI_KERNEL_LOG(SCHEDULE2, "get resource [%s] from [%s], result [%p]",
                    name.c_str(),
                    autil::StringUtil::toString(multiLayerResourceMap).c_str(),
                    resource);
    if (resource) {
        return resource;
    }
    return getResourceFromOld(name);
}

bool ResourceManager::validateResource(const ResourceStageMap &resourceStageMap) const {
    auto mergedResourceMap = getMergedResourceMap();
    auto resourceMap = getResourceMap();
    if (resourceMap) {
        const auto &map = resourceMap->getMap();
        for (const auto &pair : map) {
            const auto &name = pair.first;
            const auto &resource = pair.second.get();
            auto stage = _creatorManager.getResourceStage(name);
            if (stage >= RS_RUN_GRAPH_EXTERNAL) {
                NAVI_KERNEL_LOG(ERROR,
                                "unknown error, can't cache resource [%s] [%p] stage [%s] in "
                                "biz resource map, biz [%s] partCount [%d] partId [%d]",
                                name.c_str(),
                                resource,
                                ResourceStage_Name(stage).c_str(),
                                _bizName.c_str(),
                                _partCount,
                                _partId);
                return false;
            }
            if (mergedResourceMap) {
                const auto &resourceInMerge = mergedResourceMap->getResource(name);
                if (resourceInMerge != resource) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "unknown error, resource [%s] [%p] stage [%s] is not same in merged "
                                    "resource map [%p], biz [%s] partCount [%d] partId [%d]",
                                    name.c_str(),
                                    resource,
                                    ResourceStage_Name(stage).c_str(),
                                    resourceInMerge,
                                    _bizName.c_str(),
                                    _partCount,
                                    _partId);
                    return false;
                }
            }
        }
    }
    if (mergedResourceMap) {
        const auto &mergedMap = mergedResourceMap->getMap();
        for (const auto &pair : mergedMap) {
            const auto &name = pair.first;
            const auto &resource = pair.second;
            auto stage = _creatorManager.getResourceStage(name);
            if (stage >= RS_RUN_GRAPH_EXTERNAL) {
                NAVI_KERNEL_LOG(ERROR,
                                "unknown error, can't cache resource [%s] [%p] stage [%s] in "
                                "biz merge resource map, biz [%s] partCount [%d] partId [%d]",
                                name.c_str(),
                                resource.get(),
                                ResourceStage_Name(stage).c_str(),
                                _bizName.c_str(),
                                _partCount,
                                _partId);
                return false;
            }
        }
    }
    if (!validateStage(mergedResourceMap, resourceStageMap)) {
        return false;
    }
    return true;
}

bool ResourceManager::validateStage(const ResourceMapPtr &mergedResourceMap, const ResourceStageMap &resourceStageMap) const {
    for (const auto &stagePair : resourceStageMap) {
        auto stage = stagePair.first;
        const auto &stageMap = stagePair.second;
        bool needed = (stage < RS_RUN_GRAPH_EXTERNAL);
        for (const auto &resourcePair : stageMap) {
            const auto &name = resourcePair.first;
            auto require = resourcePair.second;
            bool has = mergedResourceMap && mergedResourceMap->hasResource(name);
            if (!needed) {
                if (has) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "biz [%s] init failed, can't cache stage [%s] "
                                    "resource [%s] in biz resource list, partId [%d]",
                                    _bizName.c_str(),
                                    ResourceStage_Name(stage).c_str(),
                                    name.c_str(),
                                    _partId);
                    return false;
                }
            } else {
                if (require && !has) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "biz [%s] init failed, lack required "
                                    "resource [%s] stage [%s], partId [%d], mergedMap [%s]",
                                    _bizName.c_str(),
                                    name.c_str(),
                                    ResourceStage_Name(stage).c_str(),
                                    _partId,
                                    mergedResourceMap ? autil::StringUtil::toString(mergedResourceMap).c_str() : "");
                    return false;
                }
            }
        }
    }
    return true;
}

bool ResourceManager::validateDepend(
    const Resource *resource,
    const MultiLayerResourceMap &multiLayerResourceMap) const
{
    const auto &dependMap = resource->getDepends().getMap();
    for (const auto &pair : dependMap) {
        const auto &depend = pair.second;
        if (!multiLayerResourceMap.hasResource(depend)) {
            NAVI_KERNEL_LOG(WARN,
                            "biz [%s] partCount [%d] partId [%d], validate resource failed "
                            "[%s] [%p], resourceMap [%s], lack depend [%s]",
                            _bizName.c_str(),
                            _partCount,
                            _partId,
                            pair.first.c_str(),
                            resource,
                            autil::StringUtil::toString(multiLayerResourceMap).c_str(),
                            depend ? depend->getName().c_str() : "");
            return false;
        }
    }
    return true;
}

Resource *ResourceManager::getResourceFromOld(const std::string &name) const {
    if (!_oldResourceManager) {
        return nullptr;
    }
    auto oldMerge = _oldResourceManager->getMergedResourceMap();
    Resource *resource = oldMerge ? oldMerge->getResource(name) : nullptr;
    if (!resource) {
        return nullptr;
    }
    // can reuse?
    return nullptr;
}

ResourcePtr ResourceManager::newAndConfigResource(const std::string &name,
                                                  const ResourceCreator *creator,
                                                  NaviConfigContext *ctx,
                                                  bool failAsError) const
{
    ResourcePtr resource = creator->create();
    assert(resource);
    try {
        if (resource->config(*ctx)) {
            return resource;
        } else {
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                          WARN,
                                          "config resource [%s] failed, config ctx [%s], biz: %s",
                                          name.c_str(),
                                          autil::StringUtil::toString(*ctx).c_str(),
                                          _bizName.c_str());
            return nullptr;
        }
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      WARN,
                                      "config resource [%s] failed, exception [%s], config ctx [%s], biz: %s",
                                      name.c_str(),
                                      e.GetMessage().c_str(),
                                      autil::StringUtil::toString(*ctx).c_str(),
                                      _bizName.c_str());
        NAVI_KERNEL_LOG(
            DEBUG, "exception bt: %s", NAVI_TLS_LOGGER ? NAVI_TLS_LOGGER->logger->getExceptionBt(e).c_str() : "");
        return nullptr;
    }
}

ErrorCode ResourceManager::initResource(const ResourcePtr &resourcePtr,
                                        const SubNamedDataMap *namedDataMap,
                                        const ProduceInfo *produceInfo,
                                        const ResourceCreator *creator,
                                        NaviConfigContext *configCtx,
                                        Node *requireNode,
                                        bool failAsError) const
{
    const auto &bindInfos = creator->getBindInfos();
    const auto &dynamicInfoMap = creator->getDynamicGroupInfoMap();
    auto &resource = *resourcePtr;
    const auto &name = resource.getName();
    if (!bindResource(name, resourcePtr.get(), resource.getCastPtr(), resource._dependMap, dynamicInfoMap,
                      bindInfos, {}, "resource"))
    {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(
            failAsError, WARN, "init resource [%s] failed, bind depend resource failed", name.c_str());
        return EC_BIND_RESOURCE;
    }
    if (!bindNamedData(name, resource, resource.getCastPtr(), namedDataMap, bindInfos, "resource")) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(
            failAsError, WARN, "init resource [%s] failed, bind named data failed", name.c_str());
        return EC_BIND_NAMED_DATA;
    }
    ProduceNotifierPtr notifier;
    if (produceInfo && produceInfo->_produce) {
        notifier = std::make_shared<ProduceNotifier>();
        notifier->_name = name;
        notifier->_info = *produceInfo;
        notifier->_resource = resourcePtr.get();
    }
    auto isRsGraph = (creator->getStage() == RS_GRAPH);
    ResourceInitContext ctx(this,
                            creator,
                            _configPath,
                            _bizName,
                            isRsGraph ? 1 : _partCount,
                            isRsGraph ? 0 : _partId,
                            &resource._dependMap,
                            notifier,
                            *configCtx,
                            namedDataMap,
                            requireNode);
    ErrorCode ec = EC_NONE;
    try {
        ec = resource.init(ctx);
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError,
                                      WARN,
                                      "init resource failed, resource [%s], biz: %s, exception: [%s]",
                                      name.c_str(),
                                      _bizName.c_str(),
                                      e.GetMessage().c_str());
        NAVI_KERNEL_LOG(
            DEBUG, "exception bt: %s", NAVI_TLS_LOGGER ? NAVI_TLS_LOGGER->logger->getExceptionBt(e).c_str() : "");
        ec = EC_ABORT;
    }
    if (EC_NONE != ec) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(
            failAsError, WARN, "init resource failed, resource [%s], biz: %s", name.c_str(), _bizName.c_str());
        return ec;
    }
    NAVI_KERNEL_LOG(TRACE3, "create resource [%s] success: %p, biz: %s",
                    name.c_str(), resourcePtr.get(), _bizName.c_str());
    return EC_NONE;
}

DataPtr ResourceManager::getNamedData(const SubNamedDataMap *namedDataMap,
                                      const std::string &name)
{
    if (!namedDataMap) {
        return nullptr;
    }
    auto it = namedDataMap->find(name);
    if (namedDataMap->end() == it) {
        return nullptr;
    }
    return it->second.data;
}

NaviPartId ResourceManager::getPartCount() const {
    return _partCount;
}

bool ResourceManager::hasConfig(const std::string &name) const {
    return _configMap.hasConfig(name);
}

void ResourceManager::fillSummary(PartSummary &partSummary) const {
    auto resourceMap = getResourceMap();
    if (resourceMap) {
        const auto &map = resourceMap->getMap();
        for (const auto &pair : map) {
            const auto &name = pair.first;
            auto stage = _creatorManager.getResourceStage(name);
            ResourceLoadInfo info;
            info.stage = ResourceStage_Name(stage);
            partSummary.resourceSet.infos.emplace(name, std::move(info));
        }
    }
    std::shared_ptr<NaviUnorderedMap<std::string, StaticGraphInfoPtr>> staticGraphInfoMap;
    {
        autil::ScopedLock lock(_publishLock);
        staticGraphInfoMap = _staticGraphInfoMap;
    }
    if (staticGraphInfoMap) {
        for (const auto &pair : *staticGraphInfoMap) {
            partSummary.staticGraphSummary[pair.first] = pair.second->meta;
        }
    }
}

void ResourceManager::fillProto(StringHashTable &stringHashTable, BizPartSummaryDef &summaryDef) const {
    summaryDef.set_part_id(_partId);
    auto resourceMap = getResourceMap();
    if (resourceMap) {
        const auto &map = resourceMap->getMap();
        auto isBuildin = (_bizName == NAVI_BUILDIN_BIZ);
        for (const auto &pair : map) {
            const auto &name = pair.first;
            auto stage = _creatorManager.getResourceStage(name);
            if (!isBuildin && stage == RS_EXTERNAL && _partId != NAVI_BIZ_PART_ID) {
                continue;
            }
            if (!pair.second) {
                continue;
            }
            const auto &resource = *pair.second;
            auto resourceInfoDef = summaryDef.add_resources();
            resourceInfoDef->set_name_hash(stringHashTable.getHash(name));
            resourceInfoDef->set_stage(stage);
            resourceInfoDef->set_addr((size_t)pair.second.get());
            resourceInfoDef->set_type_name_hash(
                stringHashTable.getHash(CommonUtil::parseSymbol(typeid(resource).name())));
            std::string meta;
            if (resource.getMeta(meta)) {
                resourceInfoDef->set_meta_hash(stringHashTable.getHash(meta));
            }
        }
    }
}
}
