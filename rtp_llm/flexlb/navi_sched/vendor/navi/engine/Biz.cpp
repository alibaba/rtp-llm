#include "navi/engine/Biz.h"
#include "autil/legacy/jsonizable.h"
#include "kmonitor/client/MetricsReporter.h"
#include "navi/engine/BizManager.h"
#include "navi/engine/InitResourceGraphBuilder.h"
#include "navi/engine/Kernel.h"
#include "navi/engine/RunResourceGraphBuilder.h"
#include "navi/engine/StringHashTable.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/engine/NaviSnapshotSummary.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/LocalSubGraphInternal.h"

using namespace kmonitor;

namespace navi {

Biz::Biz(BizManager *bizManager,
         const std::string &bizName)
    : _bizManager(bizManager)
    , _bizName(bizName)
    , _partCount(INVALID_NAVI_PART_COUNT)
    , _graphCache(bizName)
{
}

Biz::~Biz() {
    for (const auto &pair : _graphInfoMap) {
        delete pair.second;
    }
    for (const auto &pair : _resourceManagers) {
        delete pair.second;
    }
}

bool Biz::preInit(bool isRealNavi,
                  const std::string &configPath,
                  const NaviBizConfig &config,
                  const ModuleManagerPtr &moduleManager,
                  const Biz *buildinBiz,
                  const Biz *oldBiz)
{
    initConfigPath(configPath, config);
    if (!initConfigMap(config, buildinBiz)) {
        return false;
    }
    if (!initCreatorManager(isRealNavi, config, moduleManager, buildinBiz)) {
        return false;
    }
    if (!initKernelSet(isRealNavi, config)) {
        return false;
    }
    initKernelCreatorMap();
    if (!initResourceStageMap(config, buildinBiz)) {
        return false;
    }
    if (!initConfig(config, buildinBiz)) {
        return false;
    }
    if (!createResourceManager(config, buildinBiz, oldBiz)) {
        return false;
    }
    if (!loadGraphs(config)) {
        return false;
    }
    if (!initPublishInfos(config)) {
        return false;
    }
    if (!initCache(config)) {
        return false;
    }
    return true;
}

bool Biz::initCreatorManager(bool isRealNavi,
                             const NaviBizConfig &config,
                             const ModuleManagerPtr &moduleManager,
                             const Biz *buildinBiz)
{
    CreatorManagerPtr parent;
    if (buildinBiz) {
        parent = buildinBiz->_creatorManager;
    }
    _replaceInfoMap.init(config.resourceReplaceInfos);
    CreatorManagerPtr creatorManager(new CreatorManager(_bizName, parent));
    if (!creatorManager->init(isRealNavi,
                              _configPath,
                              config.modules,
                              moduleManager,
                              _replaceInfoMap,
                              _resourceConfigMap)) {
        NAVI_KERNEL_LOG(
            ERROR,
            "init biz creator manager failed, biz [%s], configPath: [%s]",
            _bizName.c_str(), _configPath.c_str());
        return false;
    }
    _creatorManager = std::move(creatorManager);
    return true;
}

bool Biz::saveResource(NaviPartId partCount,
                       NaviPartId partId,
                       const ResourceMap &resourceMap)
{
    if (_partCount != partCount) {
        NAVI_KERNEL_LOG(ERROR, "partCount mismatch, expect [%d], got [%d]",
                        _partCount, partCount);
        return false;
    }
    auto resourceManager = getResourceManager(partId);
    if (!resourceManager) {
        NAVI_KERNEL_LOG(ERROR, "partId [%d] not exist", partId);
        return false;
    }
    if (!resourceManager->saveResource(resourceMap)) {
        NAVI_KERNEL_LOG(INFO,
                        "save resource failed, biz [%s] partCount [%d] partId [%d] init resource [%s]",
                        _bizName.c_str(),
                        partCount,
                        partId,
                        autil::StringUtil::toString(resourceMap).c_str());
        return false;
    }
    NAVI_KERNEL_LOG(INFO,
                    "biz [%s] partCount [%d] partId [%d] init resource [%s]",
                    _bizName.c_str(), partCount, partId,
                    autil::StringUtil::toString(resourceMap).c_str());
    return true;
}

bool Biz::collectPublish(std::vector<multi_call::ServerBizTopoInfo> &infoVec) const {
    for (const auto &pair : _resourceManagers) {
        auto partId = pair.first;
        if (partId != NAVI_BIZ_PART_ID) {
            const auto &manager = pair.second;
            if (!manager->collectPublish(infoVec)) {
                return false;
            }
        }
    }
    return true;
}

void Biz::startPublish(multi_call::GigRpcServer *gigRpcServer) {
    for (const auto &pair : _resourceManagers) {
        pair.second->startPublish(gigRpcServer);
    }
}

bool Biz::publishResource(NaviPartId partCount, NaviPartId partId) {
    if (_partCount != partCount) {
        NAVI_KERNEL_LOG(ERROR, "partCount mismatch, expect [%d], got [%d]",
                        _partCount, partCount);
        return false;
    }
    auto resourceManager = getResourceManager(partId);
    if (!resourceManager) {
        NAVI_KERNEL_LOG(ERROR, "partId [%d] not exist", partId);
        return false;
    }
    auto buildinResourceManager =
        _bizManager->getBuildinBiz()->getResourceManager(0);
    auto bizResourceManager = getResourceManager(NAVI_BIZ_PART_ID);
    return resourceManager->publishResource(
        buildinResourceManager, bizResourceManager, _resourceStageMap);
}

bool Biz::initResourceGraph(bool ignoreProduceError,
                            GraphId buildinGraphId,
                            GraphBuilder &builder,
                            GraphId &resultGraphId) const
{
    InitResourceGraphBuilder resourceGraphBuilder(_bizName,
                                                  _creatorManager.get(),
                                                  _resourceStageMap,
                                                  _partCount,
                                                  getPartIdsWithoutBizPart(),
                                                  ignoreProduceError,
                                                  builder);
    return resourceGraphBuilder.build(buildinGraphId, resultGraphId);
}

N Biz::addResourceGraph(const std::string &resource,
                        const MultiLayerResourceMap &multiLayerResourceMap,
                        int32_t scope,
                        const std::string &requireNode,
                        int32_t requireNodeIndex,
                        const NaviUnorderedMap<std::string, std::string> &replaceMap,
                        GraphBuilder &builder) const
{
    RunResourceGraphBuilder resourceGraphBuilder(_creatorManager.get(),
                                                 multiLayerResourceMap,
                                                 scope,
                                                 requireNode,
                                                 requireNodeIndex,
                                                 replaceMap,
                                                 builder);
    return resourceGraphBuilder.build(resource);
}

bool Biz::createResourceManager(const NaviBizConfig &config,
                                const Biz *buildinBiz, const Biz *oldBiz)
{
    auto isBuildin = (!buildinBiz);
    auto partIds = config.partIds;
    if (partIds.empty()) {
        NAVI_KERNEL_LOG(ERROR, "no partId in biz [%s] config [%s]",
                        _bizName.c_str(), FastToJsonString(config).c_str());
        return false;
    }
    if (partIds.end() != partIds.find(NAVI_BIZ_PART_ID)) {
        NAVI_KERNEL_LOG(ERROR, "invalid partId [%d] in biz [%s] config [%s]",
                        NAVI_BIZ_PART_ID, _bizName.c_str(),
                        FastToJsonString(config).c_str());
        return false;
    }
    // biz level resource
    if (!isBuildin) {
        partIds.emplace(NAVI_BIZ_PART_ID);
    }
    auto partCount = config.partCount;
    for (auto partId : partIds) {
        if ((partId < 0 && partId != NAVI_BIZ_PART_ID) || (partId >= partCount)) {
            NAVI_KERNEL_LOG(
                ERROR, "invalid partId: [%d], partCount: [%d], bizName: [%s]",
                partId, partCount, _bizName.c_str());
            return false;
        }
        ResourceManager *oldResourceManager = nullptr;
        if (oldBiz) {
            oldResourceManager = oldBiz->getResourceManager(partId);
        }
        auto manager =
            new ResourceManager(this, _bizName, _configPath, partCount, partId,
                                *_creatorManager, _resourceConfigMap);
        if (!manager->preInit(oldResourceManager)) {
            delete manager;
            return false;
        }
        _resourceManagers.emplace(partId, manager);
    }
    _partCount = partCount;
    return true;
}

ResourceManager *Biz::getResourceManager(NaviPartId partId) const {
    auto it = _resourceManagers.find(partId);
    if (_resourceManagers.end() != it) {
        return it->second;
    }
    return nullptr;
}

bool Biz::loadGraphs(const NaviBizConfig &config) {
    for (const auto &pair : config.graphConfigMap) {
        auto graphInfo = new GraphInfo(_configPath, pair.second);
        if (graphInfo->init()) {
            _graphInfoMap[pair.first] = graphInfo;
        } else {
            delete graphInfo;
            return false;
        }
    }
    return true;
}

const GraphInfo *Biz::getGraphInfo(const std::string &name) const {
    auto it = _graphInfoMap.find(name);
    if (_graphInfoMap.end() == it) {
        return nullptr;
    } else {
        return it->second;
    }
}

bool Biz::initPublishInfos(const NaviBizConfig &config) {
    _publishInfos = config.publishInfos;
    _flowControlConfig = config.flowControlConfig;
    for (const auto &name : _publishInfos.bizAliasNames) {
        if (name.empty()) {
            NAVI_KERNEL_LOG(ERROR, "biz alias name can't be empty string");
            return false;
        }
    }
    return true;
}

bool Biz::initCache(const NaviBizConfig &config) {
    if (!_graphCache.init(config.cacheConfig)) {
        return false;
    }
    return true;
}

bool Biz::getResourceMap(NaviPartId partCount, NaviPartId partId, ResourceMapPtr &resourceMap) const {
    auto resourceManager = getResourceManager(partId);
    if (!resourceManager) {
        NAVI_KERNEL_LOG(ERROR,
                        "resource manager for partId [%d] not exist, partCount [%d], biz [%s]",
                        partId,
                        partCount,
                        _bizName.c_str());
        return false;
    }
    if (partCount != _partCount) {
        NAVI_KERNEL_LOG(
            ERROR,
            "partCount mismatch, expect: [%d], actual: [%d], bizName: %s",
            _partCount, partCount, _bizName.c_str());
        return false;
    }
    resourceMap = resourceManager->getMergedResourceMap();
    return true;
}

ResourceStage Biz::getResourceStage(const std::string &name) const {
    auto creator = _creatorManager->getResourceCreator(name);
    if (!creator) {
        return RS_UNKNOWN;
    }
    return creator->getStage();
}

ErrorCode Biz::createResource(NaviPartId partCount,
                              NaviPartId partId,
                              const std::string &name,
                              ResourceStage minStage,
                              const MultiLayerResourceMap &multiLayerResourceMap,
                              const SubNamedDataMap *namedDataMap,
                              const ProduceInfo &produceInfo,
                              bool checkReuse,
                              Node *requireNode,
                              bool failAsError,
                              ResourcePtr &resource)
{
    auto resourceManager = getResourceManager(partId);
    if (!resourceManager) {
        NAVI_KERNEL_LOG(ERROR,
                        "create resource [%s] failed, partId [%d] not exist",
                        name.c_str(), partId);
        return EC_PART_ID_NOT_EXIST;
    }
    auto expectPartCount = resourceManager->getPartCount();
    if (partCount != expectPartCount) {
        NAVI_KERNEL_LOG(
            ERROR,
            "partCount mismatch, expect: [%d], actual: [%d], bizName: %s",
            expectPartCount, partCount, _bizName.c_str());
        return EC_PART_COUNT_MISMATCH;
    }
    return resourceManager->createResource(name,
                                           minStage,
                                           multiLayerResourceMap,
                                           namedDataMap,
                                           produceInfo,
                                           checkReuse,
                                           requireNode,
                                           failAsError,
                                           resource);
}

Resource *Biz::buildResourceRecur(NaviPartId partCount,
                                  NaviPartId partId,
                                  const std::string &name,
                                  ResourceStage minStage,
                                  NaviConfigContext *inputCtx,
                                  const SubNamedDataMap *namedDataMap,
                                  const MultiLayerResourceMap &nodeResourceMap,
                                  const ReplaceInfoMap *replaceInfoMap,
                                  Node *requireNode,
                                  ResourceMap &inputResourceMap)
{
    auto resourceManager = getResourceManager(partId);
    if (!resourceManager) {
        NAVI_KERNEL_LOG(ERROR, "create resource [%s] failed, partId [%d] not exist", name.c_str(), partId);
        return nullptr;
    }
    auto expectPartCount = resourceManager->getPartCount();
    if (partCount != expectPartCount) {
        NAVI_KERNEL_LOG(ERROR,
                        "partCount mismatch, expect: [%d], actual: [%d], bizName: %s",
                        expectPartCount,
                        partCount,
                        _bizName.c_str());
        return nullptr;
    }
    return resourceManager->buildResourceRecur(
        name, minStage, inputCtx, namedDataMap, nodeResourceMap, replaceInfoMap, requireNode, false, inputResourceMap);
}

const CreatorManagerPtr &Biz::getCreatorManager() const {
    return _creatorManager;
}

const KernelCreator *Biz::getKernelCreator(const std::string &name) const {
    auto it = _kernelCreatorMap.find(name);
    if (unlikely(_kernelCreatorMap.end() == it)) {
        auto creator = _creatorManager->getKernelCreator(name);
        if (!creator) {
            NAVI_KERNEL_LOG(ERROR, "kernel [%s] not registered", name.c_str());
        } else {
            if (_kernelBlacklist.end() != _kernelBlacklist.find(name)) {
                NAVI_KERNEL_LOG(
                    ERROR, "kernel [%s] is in blacklist, check biz [%s] config",
                    name.c_str(), _bizName.c_str());
            } else {
                NAVI_KERNEL_LOG(ERROR,
                                "kernel [%s] registered but not support by biz "
                                "[%s], check biz kernel list",
                                name.c_str(), _bizName.c_str());
            }
        }
        return nullptr;
    }
    return it->second;
}

ResourceCreator *Biz::getResourceCreator(const std::string &name) const {
    return _creatorManager->getResourceCreator(name);
}

bool Biz::initConfigMap(const NaviBizConfig &config, const Biz *buildinBiz) {
    _modules = config.modules;
    _configKernelList = config.kernels;
    _configKernelBlackList = config.kernelBlacklist;
    _configResourcesList = config.resources;
    if (!_kernelConfigMap.init(config.kernelConfigVec)) {
        NAVI_KERNEL_LOG(ERROR, "init kernel config map failed, biz [%s]", _bizName.c_str());
        return false;
    }
    NaviRegistryConfigMap resourceConfigMap;
    if (!resourceConfigMap.init(config.resourceConfigVec)) {
        NAVI_KERNEL_LOG(ERROR, "init resource config map failed, biz [%s]", _bizName.c_str());
        return false;
    }
    for (const auto &info : config.resourceReplaceInfos) {
        auto registry = resourceConfigMap.getRegistryConfig(info.to);
        if (registry) {
            auto config = *registry;
            config.name = info.from;
            resourceConfigMap.setConfig(config);
        }
    }
    _resourceConfigMap = std::move(resourceConfigMap);
    return true;
}

void Biz::initConfigPath(const std::string &configPath,
                         const NaviBizConfig &config)
{
    if (!config.configPath.empty()) {
        _configPath = config.configPath;
    } else {
        _configPath = configPath;
    }
}

bool Biz::initConfig(const NaviBizConfig &config, const Biz *buildinBiz) {
    auto isBuildin = (!buildinBiz);
    const auto &kernelConfigMap = _kernelConfigMap.getConfigMap();
    for (const auto &pair : kernelConfigMap) {
        const auto &name = pair.first;
        if (_kernelSet.end() == _kernelSet.find(name)) {
            NAVI_KERNEL_LOG(ERROR,
                            "kernel [%s] has config but not supported by biz "
                            "[%s], check kernel config or kernel list",
                            name.c_str(), _bizName.c_str());
            return false;
        }
    }
    std::vector<NaviRegistryConfig> bizStageConfigVec;
    const auto &resourceConfigMap = _resourceConfigMap.getConfigMap();
    if (!isBuildin) {
        for (const auto &pair : resourceConfigMap) {
            const auto &name = pair.first;
            auto creator = getResourceCreator(name);
            if (!creator) {
                NAVI_KERNEL_LOG(ERROR, "resource [%s] has config but not registered", name.c_str());
                return false;
            }
            auto stage = creator->getStage();
            if (stage == RS_GRAPH) {
                NAVI_KERNEL_LOG(ERROR,
                                "resource [%s] stage [%s] can't be configured in "
                                "biz config, check biz [%s] config",
                                name.c_str(),
                                ResourceStage_Name(stage).c_str(),
                                _bizName.c_str());
                return false;
            }
        }
    }
    if (buildinBiz) {
        _resourceConfigMap.mergeNotExist(
            buildinBiz->getBizStageResourceConfigMap());
    }
    for (const auto &pair : resourceConfigMap) {
        const auto &name = pair.first;
        auto creator = getResourceCreator(name);
        if (!creator) {
            NAVI_KERNEL_LOG(ERROR, "resource [%s] has config but not registered",
                            name.c_str());
            return false;
        }
        auto stage = creator->getStage();
        if (!isBuildin && stage < RS_BIZ) {
            NAVI_KERNEL_LOG(ERROR,
                            "resource [%s] stage [%s] can't be configured in "
                            "biz config, check biz [%s] config",
                            name.c_str(), ResourceStage_Name(stage).c_str(),
                            _bizName.c_str());
            return false;
        }
        if (isBuildin && stage >= RS_BIZ) {
            bizStageConfigVec.push_back(pair.second);
        }
    }
    if (isBuildin && !_bizStageResourceConfigMap.init(bizStageConfigVec)) {
        NAVI_KERNEL_LOG(ERROR,
                        "init biz stage resource config failed, biz [%s]",
                        _bizName.c_str());
        return false;
    }
    return true;
}

const NaviRegistryConfigMap &Biz::getBizStageResourceConfigMap() const {
    return _bizStageResourceConfigMap;
}

bool Biz::initKernelSet(bool isRealNavi, const NaviBizConfig &config) {
    std::set<std::string> kernelSet;
    std::set<std::string> blacklistSet;
    for (const auto &kernelRegex : config.kernels) {
        bool emptyMatch = true;
        if (!_creatorManager->collectMatchedKernels(kernelRegex, emptyMatch,
                                                    kernelSet))
        {
            NAVI_KERNEL_LOG(ERROR, "invalid kernel list regex [%s], biz [%s]",
                            kernelRegex.c_str(), _bizName.c_str());
            return false;
        }
        if (emptyMatch) {
            NAVI_KERNEL_LOG(
                WARN, "kernel reg: %s, no kernel matched, biz [%s], ignored",
                kernelRegex.c_str(), _bizName.c_str());
        }
    }
    for (const auto &blacklistRegex : config.kernelBlacklist) {
        bool emptyMatch = true;
        if (!_creatorManager->collectMatchedKernels(blacklistRegex, emptyMatch,
                                                    blacklistSet))
        {
            NAVI_KERNEL_LOG(ERROR,
                            "invalid kernel black list regex [%s], biz [%s]",
                            blacklistRegex.c_str(), _bizName.c_str());
            return false;
        }
        if (emptyMatch) {
            NAVI_KERNEL_LOG(WARN,
                            "kernel blacklist reg: %s, no kernel matched, biz "
                            "[%s], ignored",
                            blacklistRegex.c_str(), _bizName.c_str());
        }
    }
    std::set<std::string> remain;
    std::set<std::string> intersect;
    std::set_intersection(kernelSet.begin(), kernelSet.end(),
                          blacklistSet.begin(), blacklistSet.end(),
                          std::inserter(intersect, intersect.begin()));
    std::set_difference(kernelSet.begin(), kernelSet.end(),
                        blacklistSet.begin(), blacklistSet.end(),
                        std::inserter(remain, remain.begin()));
    for (const auto &kernel : intersect) {
        NAVI_KERNEL_LOG(DEBUG, "kernel [%s] disabled by blacklist, biz [%s]",
                        kernel.c_str(), _bizName.c_str());
    }
    if (_bizName != NAVI_BUILDIN_BIZ && remain.empty()) {
        NAVI_KERNEL_LOG(
            WARN,
            "no kernel matched, check kernel list and blacklist, biz [%s]",
            _bizName.c_str());
    }
    _kernelSet = std::move(remain);
    _kernelBlacklist = std::move(intersect);
    addBuildinKernels();
    if (isRealNavi) {
        _kernelSet.erase(NAVI_KERNEL_NAME);
        _kernelSet.erase(NAVI_STATE_KERNEL_NAME);
    }
    return true;
}

void Biz::addBuildinKernels() {
    bool emptyMatch = true;
    _creatorManager->collectMatchedKernels("^navi\\..*", emptyMatch,
                                           _kernelSet);
}

bool Biz::initKernelCreatorMap() {
    for (const auto &name : _kernelSet) {
        auto creator = _creatorManager->getKernelCreator(name);
        if (!creator) {
            NAVI_KERNEL_LOG(ERROR, "kernel [%s] not registered", name.c_str());
            return false;
        }
        _kernelCreatorMap.emplace(name, creator);
    }
    return true;
}

bool Biz::initResourceStageMap(const NaviBizConfig &config,
                               const Biz *buildinBiz)
{
    auto isBuildin = (!buildinBiz);
    for (const auto &kernelName : _kernelSet) {
        auto creator = _creatorManager->getKernelCreator(kernelName);
        if (!creator) {
            NAVI_KERNEL_LOG(ERROR, "kernel [%s] not registered", kernelName.c_str());
            return false;
        }
        const auto &dependResourceVec = creator->getDependResources();
        for (const auto &depend : dependResourceVec) {
            const auto &resourceName = depend.first;
            auto require = depend.second;
            auto resourceCreator =
                _creatorManager->getResourceCreator(resourceName);
            if (!resourceCreator) {
                NAVI_KERNEL_LOG(ERROR,
                                "kernel [%s] depend on not registered resource "
                                "[%s], biz [%s]",
                                kernelName.c_str(), resourceName.c_str(),
                                _bizName.c_str());
                return false;
            }
            auto stage = resourceCreator->getStage();
            if (isBuildin && require) {
                if ((RS_BIZ == stage) || (RS_BIZ_PART == stage)) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "can't create stage [%s] resource [%s] in buildin biz, require kernel [%s]",
                                    ResourceStage_Name(stage).c_str(),
                                    resourceName.c_str(),
                                    kernelName.c_str());
                    return false;
                }
            }
            addToStageMap(stage, resourceName, require);
        }
    }
    for (const auto &resourceName : config.resources) {
        auto resourceCreator =
            _creatorManager->getResourceCreator(resourceName);
        if (!resourceCreator) {
            NAVI_KERNEL_LOG(
                ERROR,
                "resource [%s] not registered, check resource_list in biz [%s]",
                resourceName.c_str(), _bizName.c_str());
            return false;
        }
        auto stage = resourceCreator->getStage();
        if (isBuildin) {
            if ((RS_BIZ == stage) || (RS_BIZ_PART == stage)) {
                NAVI_KERNEL_LOG(ERROR,
                                "can't create stage [%s] resource [%s] in buildin biz, check resource list in config",
                                ResourceStage_Name(stage).c_str(),
                                resourceName.c_str());
                return false;
            }
        }
        addToStageMap(stage, resourceName, true);
    }
    if (buildinBiz) {
        mergeBuildinResource(buildinBiz);
    }
    if (_resourceStageMap.empty()) {
        NAVI_KERNEL_LOG(WARN, "biz [%s] has no update resource",
                        _bizName.c_str());
    }
    return true;
}

void Biz::addToStageMap(ResourceStage stage, const std::string &resource,
                        bool require)
{
    auto &stageMap = _resourceStageMap[stage];
    if (require) {
        stageMap[resource] = true;
    } else {
        stageMap.emplace(resource, require);
    }
}

void Biz::mergeBuildinResource(const Biz *buildinBiz) {
    const auto &buildinMap = buildinBiz->_resourceStageMap;
    for (const auto &stagePair : buildinMap) {
        auto stage = stagePair.first;
        if (stage < RS_BIZ) {
            continue;
        }
        auto &destMap = _resourceStageMap[stage];
        const auto &stageMap = stagePair.second;
        for (const auto &resourcePair : stageMap) {
            destMap[resourcePair.first] = resourcePair.second;
        }
    }
}

const std::set<std::string> &Biz::getKernelSet() const {
    return _kernelSet;
}

const std::set<std::string> &Biz::getKernelBlacklist() const {
    return _kernelBlacklist;
}

const autil::legacy::RapidValue *Biz::getKernelConfig(const std::string &name) const {
    return _kernelConfigMap.getConfig(name);
}

const std::vector<std::pair<std::string, bool>> *Biz::getResourceDependResourceVec(const std::string &name) const {
    return _creatorManager->getResourceDependResourceVec(name);
}

const NaviUnorderedSet<std::string> *Biz::getResourceReplacerSet(const std::string &name) const {
    return _creatorManager->getResourceReplacerSet(name);
}

const std::string &Biz::getConfigPath() const {
    return _configPath;
}

const std::string &Biz::getName() const {
    return _bizName;
}

bool Biz::isSinglePart() const {
    return _partCount == 1;
}

bool Biz::hasPartId(NaviPartId partId) const {
    return _resourceManagers.end() != _resourceManagers.find(partId);
}

NaviPartId Biz::getPartCount() const {
    return _partCount;
}

void Biz::getPartInfo(NaviPartId &partCount, std::vector<NaviPartId> &partIds) const {
    partIds = getPartIdsWithoutBizPart();
    partCount = _partCount;
}

std::vector<NaviPartId> Biz::getPartIdsWithoutBizPart() const {
    std::vector<NaviPartId> partIds;
    partIds.reserve(_resourceManagers.size());
    for (const auto &pair : _resourceManagers) {
        auto partId = pair.first;
        if (partId == NAVI_BIZ_PART_ID) {
            continue;
        }
        partIds.push_back(partId);
    }
    return partIds;
}

void Biz::fillSummary(NaviSnapshotSummary &summary) const {
    BizSummary bizSummary;
    _creatorManager->fillModuleSummary(bizSummary);
    auto partIdVec = getPartIdsWithoutBizPart();
    bizSummary.partIds.insert(partIdVec.begin(), partIdVec.end());
    fillKernelSummary(bizSummary);
    fillResourceSummary(bizSummary);
    summary.bizs.emplace(_bizName, std::move(bizSummary));
}

void Biz::fillKernelSummary(BizSummary &bizSummary) const {
    auto registerKernelSet = _creatorManager->getKernelSet();
    const auto &kernelSet = getKernelSet();
    const auto &blacklist = getKernelBlacklist();
    std::set<std::string> unsupportAndBlacklist;
    std::set_difference(
        registerKernelSet.begin(), registerKernelSet.end(), kernelSet.begin(),
        kernelSet.end(),
        std::inserter(unsupportAndBlacklist, unsupportAndBlacklist.end()));
    std::set<std::string> unsupport;
    std::set_difference(unsupportAndBlacklist.begin(),
                        unsupportAndBlacklist.end(), blacklist.begin(),
                        blacklist.end(),
                        std::inserter(unsupport, unsupport.end()));
    bizSummary.supportKernels.set = kernelSet;
    bizSummary.blacklistKernels.set = blacklist;
    bizSummary.unsupportKernels.set = std::move(unsupport);
}

void Biz::fillResourceSummary(BizSummary &bizSummary) const {
    auto bizResourceManager = getResourceManager(NAVI_BIZ_PART_ID);
    if (bizResourceManager) {
        bizResourceManager->fillSummary(bizSummary.bizResources);
    }
    for (const auto &pair : _resourceManagers) {
        auto partId = pair.first;
        auto manager = pair.second;
        if (partId == NAVI_BIZ_PART_ID) {
            continue;
        }
        auto &partSummary =
            bizSummary.partResources[autil::StringUtil::toString(partId)];
        manager->fillSummary(partSummary);
    }
}

void Biz::fillProto(StringHashTable &stringHashTable, RegistryTableDef &registryTable, BizSummaryDef &bizDef) const {
    bizDef.set_name_hash(stringHashTable.getHash(_bizName));
    bizDef.set_config_path_hash(stringHashTable.getHash(_configPath));
    if (_creatorManager) {
        _creatorManager->fillProto(stringHashTable, registryTable, bizDef);
    }
    bizDef.set_is_default(_isDefault);
    bizDef.set_part_count(_partCount);
    for (const auto &kernel : _kernelSet) {
        bizDef.add_kernels_hash(stringHashTable.getHash(kernel));
    }
    for (const auto &kernel : _kernelBlacklist) {
        bizDef.add_kernel_blacklist_hash(stringHashTable.getHash(kernel));
    }
    if (_publishInfos.disablePublish || !_publishInfos.bizAliasNames.empty()) {
        auto infoDef = bizDef.mutable_publish_infos();
        infoDef->set_disable_publish(_publishInfos.disablePublish);
        for (const auto &name : _publishInfos.bizAliasNames) {
            infoDef->add_biz_alias_names_hash(stringHashTable.getHash(name));
        }
    }
    for (const auto &name : _modules) {
        bizDef.add_modules_hash(stringHashTable.getHash(name));
    }
    for (const auto &name : _configKernelList) {
        bizDef.add_config_kernel_patterns_hash(stringHashTable.getHash(name));
    }
    for (const auto &name : _configKernelBlackList) {
        bizDef.add_config_kernel_blacklist_patterns_hash(stringHashTable.getHash(name));
    }
    for (const auto &name : _configResourcesList) {
        bizDef.add_config_resource_list_hash(stringHashTable.getHash(name));
    }
    for (const auto &pair : _kernelConfigMap.getConfigMap()) {
        const auto &config = pair.second;
        auto configDef = bizDef.add_kernel_config();
        configDef->set_name_hash(stringHashTable.getHash(config.name));
        if (config.config) {
            configDef->set_config_hash(stringHashTable.getHash(autil::legacy::FastToJsonString(*config.config)));
        }
    }
    for (const auto &pair : _resourceConfigMap.getConfigMap()) {
        const auto &config = pair.second;
        auto configDef = bizDef.add_resource_config();
        configDef->set_name_hash(stringHashTable.getHash(config.name));
        if (config.config) {
            configDef->set_config_hash(stringHashTable.getHash(autil::legacy::FastToJsonString(*config.config)));
        }
        configDef->set_build_mode_hash(stringHashTable.getHash(config.buildMode));
    }
    auto bizResourceManager = getResourceManager(NAVI_BIZ_PART_ID);
    if (bizResourceManager) {
        bizResourceManager->fillProto(stringHashTable, *bizDef.mutable_biz_resources());
    }
    for (const auto &pair : _resourceManagers) {
        auto partId = pair.first;
        auto manager = pair.second;
        if (partId == NAVI_BIZ_PART_ID) {
            continue;
        }
        manager->fillProto(stringHashTable, *bizDef.add_parts());
    }
    auto flowControlConfig = getFlowControlConfig();
    if (flowControlConfig) {
        bizDef.set_flow_config_json(stringHashTable.getHash(autil::legacy::FastToJsonString(*flowControlConfig)));
    }
    updateProto(bizDef);
}

void Biz::updateProto(BizSummaryDef &bizDef) const {
    _graphCache.fillProto(*bizDef.mutable_cache_summary());
}

TestMode Biz::getTestMode() const { return _bizManager->getTestMode(); }

} // namespace navi
