#include "navi/engine/BizManager.h"
#include "autil/StringUtil.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviSnapshotSummary.h"
#include "navi/engine/StringHashTable.h"
#include "navi/log/NaviLogger.h"
#include "navi/ops/ResourceKernelDef.h"
#include "navi/proto/NaviSummary.pb.h"
#include "navi/util/CommonUtil.h"
#include <multi_call/rpc/GigRpcServer.h>
#include "autil/StringUtil.h"

namespace navi {

BizManager::BizManager(bool isRealNavi,
                       const std::string &naviName,
                       const ModuleManagerPtr &moduleManager)
    : _isRealNavi(isRealNavi)
    , _naviName(naviName)
    , _moduleManager(moduleManager)
    , _graphDef(nullptr)
{
}

BizManager::~BizManager() {
    if (_gigRpcServer && (!_publishSignatures.empty())) {
        _gigRpcServer->unpublish(_publishSignatures);
        NAVI_KERNEL_LOG(INFO, "unpublish signatures success, navi [%s]", _naviName.c_str());
    }
    _bizMap.clear();
    _bizMapWithAlias.clear();
    _defaultBiz.reset();
    _buildinBiz.reset();
    if (_graphDef) {
        DELETE_AND_SET_NULL(_graphDef);
    }
}

bool BizManager::preInit(const NaviLoggerPtr &logger,
                         BizManager *oldBizManager,
                         const NaviConfig &config,
                         const ResourceMap &rootResourceMapIn)
{
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix("biz manager");
    NaviLoggerScope scope(_logger);
    if (!initModuleManager(oldBizManager)) {
        return false;
    }
    if (!createBuildinBiz(config)) {
        return false;
    }
    ResourceMap rootResourceMap;
    rootResourceMap.addResource(rootResourceMapIn);
    if (!initMemoryPoolR(rootResourceMap)) {
        return false;
    }
    if (!createBizs(config, oldBizManager)) {
        return false;
    }
    if (!initTypeMemoryPool()) {
        return false;
    }
    auto ret = initResourceGraph(config);
    if (!ret) {
        return false;
    }
    initCreatorId();
    return true;
}

GraphDef *BizManager::cloneResourceGraph() const {
    if (!_graphDef) {
        return nullptr;
    }
    auto graphDef = new GraphDef();
    graphDef->CopyFrom(*_graphDef);
    return graphDef;
}

void BizManager::initCreatorId() {
    RegistryIdGenerator idGenerator;
    {
        auto creatorManager = _buildinBiz->getCreatorManager();
        if (creatorManager) {
            creatorManager->postInit(idGenerator);
        }
    }
    for (const auto &pair : _bizMap) {
        auto creatorManager = pair.second->getCreatorManager();
        if (creatorManager) {
            creatorManager->postInit(idGenerator);
        }
    }
}

bool BizManager::postInit() {
    _buildinBiz->getResourceMap(1, 0, _buildinResourceMap);
    _snapshotMultiLayerResourceMap.addResourceMap(_buildinResourceMap.get());
    return true;
}

bool BizManager::start(multi_call::GigRpcServer *gigRpcServer, BizManager *oldBizManager) {
    if (!gigRpcServer) {
        return true;
    }
    _gigRpcServer = gigRpcServer;
    std::vector<multi_call::ServerBizTopoInfo> infoVec;
    for (const auto &pair : _bizMap) {
        if (!pair.second->collectPublish(infoVec)) {
            return false;
        }
    }
    std::vector<multi_call::SignatureTy> signatureVec;
    if (!_gigRpcServer->publish(infoVec, signatureVec)) {
        NAVI_KERNEL_LOG(ERROR, "publish topo info failed");
        return false;
    }
    _publishSignatures = signatureVec;
    if (oldBizManager) {
        oldBizManager->stealPublishSignature(_publishSignatures);
    }
    for (const auto &pair : _bizMap) {
        pair.second->startPublish(gigRpcServer);
    }
    NAVI_KERNEL_LOG(
        INFO, "biz manager start success, topo count: [%lu], biz count [%lu]", infoVec.size(), _bizMap.size());
    return true;
}

void BizManager::stealPublishSignature(const std::vector<multi_call::SignatureTy> &stealList) {
    std::set<multi_call::SignatureTy> oldSet(_publishSignatures.begin(), _publishSignatures.end());
    for (auto sig : stealList) {
        oldSet.erase(sig);
    }
    _publishSignatures = std::vector<multi_call::SignatureTy>(oldSet.begin(), oldSet.end());
}

bool BizManager::initModuleManager(BizManager *oldBizManager) {
    if (_moduleManager) {
        return true;
    }
    if (oldBizManager) {
        _moduleManager = oldBizManager->_moduleManager;
    }
    if (!_moduleManager) {
        _moduleManager = std::make_shared<ModuleManager>();
    }
    return true;
}

bool BizManager::createBuildinBiz(const NaviConfig &config) {
    _buildinConfig.partCount = 1;
    _buildinConfig.partIds.emplace(0);
    _buildinConfig.resourceConfigVec = config.snapshotResourceConfigVec;
    _buildinConfig.modules = config.modules;
    _buildinConfig.resources = config.resources;
    _buildinConfig.kernels = config.kernels;
    initBuildinBizResourceList(config.snapshotResourceConfigVec);
    _buildinBiz = createBiz(NAVI_BUILDIN_BIZ, config.configPath, _buildinConfig,
                            nullptr, nullptr);
    if (!_buildinBiz) {
        return false;
    }
    return true;
}

void BizManager::initBuildinBizResourceList(
    const std::vector<NaviRegistryConfig> &configVec)
{
    for (const auto &conf : configVec) {
        if (conf.name == GIG_CLIENT_RESOURCE_ID) {
            _buildinConfig.resources.insert(GIG_CLIENT_RESOURCE_ID);
            break;
        }
    }
}

bool BizManager::initMemoryPoolR(const ResourceMap &rootResourceMap) {
    MultiLayerResourceMap multiLayerResourceMap;
    multiLayerResourceMap.addResourceMap(&rootResourceMap);
    ResourcePtr memoryPoolR;
    auto ec = _buildinBiz->createResource(1,
                                          0,
                                          MemoryPoolR::RESOURCE_ID,
                                          RS_SNAPSHOT,
                                          multiLayerResourceMap,
                                          nullptr,
                                          {},
                                          true,
                                          nullptr,
                                          true,
                                          memoryPoolR);
    if (EC_NONE != ec) {
        NAVI_KERNEL_LOG(ERROR,
                        "create resource[%s] failed, ec [%s]",
                        MemoryPoolR::RESOURCE_ID.c_str(),
                        CommonUtil::getErrorString(ec));
        return false;
    }
    _memoryPoolRMap.addResource(memoryPoolR);
    _snapshotMultiLayerResourceMap.addResourceMap(&_memoryPoolRMap);
    return _buildinBiz->saveResource(1, 0, _memoryPoolRMap);
}

bool BizManager::initTypeMemoryPool() {
    auto memoryPoolR = dynamic_cast<MemoryPoolR *>(_memoryPoolRMap.getResource(MemoryPoolR::RESOURCE_ID));
    if (!memoryPoolR) {
        NAVI_KERNEL_LOG(ERROR, "null memory pool resource");
        return false;
    }
    auto ptr = std::dynamic_pointer_cast<MemoryPoolR>(memoryPoolR->shared_from_this());
    for (const auto &pair : _bizMap) {
        pair.second->getCreatorManager()->setTypeMemoryPoolR(ptr);
    }
    _buildinBiz->getCreatorManager()->setTypeMemoryPoolR(ptr);
    return true;
}

bool BizManager::createBizs(const NaviConfig &config,
                            BizManager *oldBizManager)
{
    for (const auto &pair : config.bizMap) {
        const auto &bizName = pair.first;
        const auto &bizConfig = pair.second;
        const Biz *oldBiz = nullptr;
        if (oldBizManager) {
            oldBiz = oldBizManager->getBiz(_logger.logger.get(), bizName).get();
        }
        auto biz = createBiz(bizName, config.configPath, bizConfig,
                             _buildinBiz.get(), oldBiz);
        if (!biz) {
            return false;
        }
        _bizMap.emplace(bizName, biz);
        const auto &aliasNames = biz->getPublishInfos().bizAliasNames;
        for (const auto &name : aliasNames) {
            _bizMapWithAlias.emplace(name, biz);
        }
        _bizMapWithAlias.emplace(bizName, biz);
    }
    const auto &defaultBizName = config.defaultBizName;
    if (!defaultBizName.empty()) {
        _defaultBiz = doGetBiz(defaultBizName);
        if (!_defaultBiz) {
            NAVI_KERNEL_LOG(ERROR, "default biz [%s] not configured",
                            defaultBizName.c_str());
            return false;
        }
        NAVI_KERNEL_LOG(INFO, "use biz [%s] as default",
                        defaultBizName.c_str());
    } else if (1u == _bizMap.size()) {
        const auto &pair = *_bizMap.begin();
        _defaultBiz = pair.second;
        NAVI_KERNEL_LOG(INFO, "only one biz [%s], use as default",
                        pair.first.c_str());
    }
    if (_defaultBiz) {
        _defaultBiz->setDefault(true);
    }
    return true;
}

BizPtr BizManager::createBiz(const std::string &bizName,
                             const std::string &configPath,
                             const NaviBizConfig &config,
                             const Biz *buildinBiz,
                             const Biz *oldBiz)
{
    if (bizName.empty()) {
        NAVI_KERNEL_LOG(ERROR, "biz name can't use empty string");
        return nullptr;
    }
    auto biz = std::make_shared<Biz>(this, bizName);
    if (biz->preInit(_isRealNavi, configPath, config, _moduleManager, buildinBiz, oldBiz)) {
        return biz;
    } else {
        return nullptr;
    }
}

bool BizManager::initResourceGraph(const NaviConfig &config) {
    try {
        _graphDef = new GraphDef();
        GraphBuilder builder(_graphDef);
        builder.enableBackPressure(0, 1);
        GraphId buildinGraphId = INVALID_GRAPH_ID;
        if (!_buildinBiz->initResourceGraph(config.ignoreProduceError, INVALID_GRAPH_ID, builder, buildinGraphId)) {
            NAVI_KERNEL_LOG(ERROR, "create init resource graph for buildin biz failed");
            return false;
        }
        for (const auto &pair : _bizMap) {
            const auto &bizName = pair.first;
            const auto &biz = pair.second;
            if (!biz->initResourceGraph(config.ignoreProduceError, buildinGraphId, builder, buildinGraphId)) {
                NAVI_KERNEL_LOG(ERROR, "create init resource graph for biz [%s] failed", bizName.c_str());
                return false;
            }
        }
        addRequireAttr(builder);
        if (_graphDef->sub_graphs_size() <= 0) {
            DELETE_AND_SET_NULL(_graphDef);
        }
        return true;
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_LOG(ERROR, "init resource graph failed, msg [%s]", e.GetMessage().c_str());
        return false;
    }
}

void BizManager::addRequireAttr(GraphBuilder &builder) {
    int64_t count = 0;
    auto nodes = builder.getAllNodes();
    for (const auto &n : nodes) {
        if (!n.isValid()) {
            continue;
        }
        if (NT_RESOURCE != n.getType()) {
            continue;
        }
        addNodeRequireAttr(n, count);
    }
    NAVI_KERNEL_LOG(INFO, "update require node count: %ld", count);
}

void BizManager::addNodeRequireAttr(N n, int64_t &count) {
    if (!needUpdateRequire(n)) {
        return;
    }
    auto portEdgeVec = n.getInputUpstreams(NODE_RESOURCE_INPUT_PORT);
    for (const auto &pair : portEdgeVec) {
        auto p = pair.first;
        auto prev = p.node();
        if (NT_RESOURCE != prev.getType()) {
            continue;
        }
        if (!pair.second.isRequire()) {
            continue;
        }
        auto isProduce = (p.getPortName() == RESOURCE_PRODUCE_OUTPUT);
        if (isProduce) {
            int64_t produceRequire = 0;
            if (!prev.getIntegerAttr(RESOURCE_ATTR_PRODUCE_REQUIRE, produceRequire)) {
                prev.integerAttr(RESOURCE_ATTR_PRODUCE_REQUIRE, 2);
                count++;
                addNodeRequireAttr(prev, count);
            } else if (1 == produceRequire) {
                prev.integerAttr(RESOURCE_ATTR_PRODUCE_REQUIRE, 2);
                count++;
                addNodeRequireAttr(prev, count);
            }
        } else {
            int64_t requireValue = 0;
            if (!prev.getIntegerAttr(RESOURCE_ATTR_REQUIRE, requireValue)) {
                prev.integerAttr(RESOURCE_ATTR_REQUIRE, 2);
                count++;
                addNodeRequireAttr(prev, count);
            } else if (1 == requireValue) {
                prev.integerAttr(RESOURCE_ATTR_REQUIRE, 2);
                count++;
                addNodeRequireAttr(prev, count);
            }
        }
    }
}

bool BizManager::needUpdateRequire(N n) {
    int64_t require = 0;
    if (!n.getIntegerAttr(RESOURCE_ATTR_REQUIRE, require)) {
        require = 0;
    }
    int64_t produceRequire = 0;
    if (!n.getIntegerAttr(RESOURCE_ATTR_PRODUCE_REQUIRE, produceRequire)) {
        produceRequire = 0;
    }
    if (0 == require && 0 == produceRequire) {
        return false;
    }
    if (2 == require || 2 == produceRequire) {
        return false;
    }
    if (1 == require) {
        n.integerAttr(RESOURCE_ATTR_REQUIRE, 2);
    }
    if (1 == produceRequire) {
        n.integerAttr(RESOURCE_ATTR_PRODUCE_REQUIRE, 2);
    }
    return true;
}

GraphDef *BizManager::createResourceGraph(const std::string &bizName,
                                          NaviPartId partCount,
                                          NaviPartId partId,
                                          const std::set<std::string> &resourceSet) const {
    auto biz = doGetBiz(bizName);
    if (!biz) {
        NAVI_KERNEL_LOG(ERROR, "biz [%s] not exist", bizName.c_str());
        return nullptr;
    }
    ResourceMapPtr resourceMap;
    MultiLayerResourceMap multiLayerResourceMap;
    biz->getResourceMap(partCount, partId, resourceMap);
    multiLayerResourceMap.addResourceMap(resourceMap.get());
    NAVI_KERNEL_LOG(SCHEDULE1,
                    "collected resource map [%s], biz [%s] partCount [%d], partId [%d]",
                    autil::StringUtil::toString(multiLayerResourceMap).c_str(),
                    bizName.c_str(),
                    partCount,
                    partId);
    assert(!resourceSet.empty());
    try {
        std::unique_ptr<GraphDef> def(new GraphDef());
        GraphBuilder builder(def.get());
        auto graphId = builder.newSubGraph(bizName);
        builder.subGraph(graphId).location(bizName, partCount, partId);
        for (const auto &resourceName : resourceSet) {
            biz->addResourceGraph(
                resourceName, multiLayerResourceMap, 0, "", 0, {}, builder);
        }
        const auto &subGraph = def->sub_graphs(graphId);
        auto nodeCount = subGraph.nodes_size();
        std::vector<std::string> fullResourceVec;
        for (int32_t i = 0; i < nodeCount; i++) {
            fullResourceVec.emplace_back(subGraph.nodes(i).name());
        }
        for (const auto &resourceName : fullResourceVec) {
            builder.node(graphId, resourceName).out(RESOURCE_CREATE_OUTPUT).asGraphOutput(resourceName);
        }
        return def.release();
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_LOG(ERROR,
                        "create resource graph failed, biz [%s] partCount [%d], partId [%d], msg [%s]",
                        bizName.c_str(),
                        partCount,
                        partId,
                        e.GetMessage().c_str());
        return nullptr;
    }
}

bool BizManager::isLocal(const LocationDef &location) const {
    const auto &bizName = location.biz_name();
    if (bizName == NAVI_BUILDIN_BIZ) {
        return true;
    }
    auto biz = doGetBiz(bizName);
    if (!biz) {
        return false;
    }
    auto thisPartId = location.this_part_id();
    if (INVALID_NAVI_PART_ID == thisPartId) {
        // input graph from user
        if (0 == location.part_info().part_ids_size()) {
            return biz->isSinglePart();
        } else {
            return false;
        }
    } else {
        // internal server graph
        return biz->hasPartId(thisPartId);
    }
}

bool BizManager::getBizPartInfo(const std::string &bizName,
                                NaviPartId &partCount,
                                std::vector<NaviPartId> &partIds) const
{
    auto biz = doGetBiz(bizName);
    if (!biz) {
        return false;
    }
    biz->getPartInfo(partCount, partIds);
    return true;
}

BizPtr BizManager::getBiz(NaviLogger *_logger, const std::string &bizName) const {
    if (bizName == NAVI_BUILDIN_BIZ) {
        return _buildinBiz;
    }
    auto biz = doGetBiz(bizName);
    if (!biz) {
        return nullptr;
    } else {
        return biz;
    }
}

BizPtr BizManager::doGetBiz(const std::string &bizName) const {
    if (bizName.empty()) {
        return _defaultBiz;
    }
    auto it = _bizMapWithAlias.find(bizName);
    if (it != _bizMapWithAlias.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

const BizPtr &BizManager::getBuildinBiz() const {
    return _buildinBiz;
}

const ModuleManagerPtr &BizManager::getModuleManager() const {
    return _moduleManager;
}

const CreatorManagerPtr &BizManager::getBuildinCreatorManager() const {
    return _buildinBiz->getCreatorManager();
}

void BizManager::cleanupModule() {
    if (_moduleManager) {
        _moduleManager->cleanup();
    }
}

Resource *BizManager::getSnapshotResource(const std::string &name) const {
    return _snapshotMultiLayerResourceMap.getResource(name);
}

std::vector<std::string> BizManager::getBizList() const {
    std::vector<std::string> vec;
    for (const auto &pair : _bizMap) {
        vec.push_back(pair.first);
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

void BizManager::reclaimCache() {
    for (const auto &pair : _bizMap) {
        pair.second->reclaimCache();
    }
}

void BizManager::fillSummary(NaviSnapshotSummary &summary) const {
    if (_buildinBiz) {
        _buildinBiz->fillSummary(summary);
    }
    for (const auto &pair : _bizMap) {
        pair.second->fillSummary(summary);
    }
}

void BizManager::fillProto(NaviSummaryDef &summaryDef) const {
    auto &registryTable = *summaryDef.mutable_registry_table();
    StringHashTable stringHashTable;
    if (_buildinBiz) {
        _buildinBiz->fillProto(stringHashTable, registryTable, *summaryDef.add_bizs());
    }
    for (const auto &pair : _bizMap) {
        pair.second->fillProto(stringHashTable, registryTable, *summaryDef.add_bizs());
    }
    stringHashTable.toProto(registryTable.mutable_symbol_table());
}

void BizManager::updateProto(NaviSummaryDef &summaryDef) const {
    if (_bizMap.size() + 1 != summaryDef.bizs_size()) {
        return;
    }
    int i = 0;
    if (_buildinBiz) {
        _buildinBiz->updateProto(*summaryDef.mutable_bizs(i++));
    }
    for (const auto &pair : _bizMap) {
        pair.second->updateProto(*summaryDef.mutable_bizs(i++));
    }
}

}
