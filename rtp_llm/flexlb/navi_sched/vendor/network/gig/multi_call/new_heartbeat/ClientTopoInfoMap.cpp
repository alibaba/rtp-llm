#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ClientTopoInfoMap.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HostHeartbeatInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ServerTopoMap.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceReplica.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MiscUtil.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ClientTopoInfo);
AUTIL_LOG_SETUP(multi_call, ClientTopoInfoMap);

ClientClusterTopoInfo::~ClientClusterTopoInfo() {
    if (provider) {
        provider->setTargetWeight(0, false);
    }
}

ClientTopoInfo::ClientTopoInfo(const std::shared_ptr<HostHeartbeatStats> &hostStats)
    : _hostStats(hostStats)
    , _hasServerAgent(false)
    , _initNetLatencyUs(0)
    , _targetWeight(MAX_WEIGHT) {
}

ClientTopoInfo::~ClientTopoInfo() {
}

bool ClientTopoInfo::init(const HeartbeatSpec &heartbeatSpec, const Spec &spec,
                          const GigMetaEnv &envDef, const BizTopoDef &topoDef, int64_t netLatencyUs,
                          HashInfoParam &param) {
    if (!createTopoNodes(heartbeatSpec, spec, envDef, topoDef, param)) {
        return false;
    }
    _targetWeight = topoDef.target_weight();
    const auto &signature = topoDef.signature();
    _signature.topo = signature.topo();
    _signature.publishId = signature.publish_id();
    initMetas(signature.meta(), topoDef.metas(), param);
    initTags(signature.tag(), topoDef.tags());
    if (topoDef.has_propagation_stat()) {
        _hasServerAgent = true;
        _firstPropagationStatDef.CopyFrom(topoDef.propagation_stat());
        _initNetLatencyUs = netLatencyUs;
    }
    return true;
}

bool ClientTopoInfo::createTopoNodes(const HeartbeatSpec &heartbeatSpec, const Spec &spec,
                                     const GigMetaEnv &envDef, const BizTopoDef &topoDef,
                                     HashInfoParam &param) {
    if (heartbeatSpec.clusterMap.empty()) {
        AUTIL_LOG(ERROR, "BUG! cluster map is empty");
        return false;
    }
    _isDefaultBiz = topoDef.key().is_default_biz();
    auto clusterNameSubType = heartbeatSpec.clusterNameSubType;
    if (CNST_NORMAL == clusterNameSubType) {
        auto clusterInfo = std::make_shared<ClientClusterTopoInfo>();
        auto &topoNode = clusterInfo->topoNode;
        initTopoNode(spec, envDef, topoDef, param, topoNode, _bizNameStr);
        topoNode.bizName = topoNode.serverBizName;
        const auto &firstCluster = *heartbeatSpec.clusterMap.begin();
        topoNode.clusterName = firstCluster.first;
        auto targetWeight = firstCluster.second;
        topoNode.meta.targetWeight = std::min(topoNode.meta.targetWeight, targetWeight);
        if (!topoNode.generateNodeId()) {
            AUTIL_LOG(ERROR, "invalid topo node [%s]", topoNode.nodeId.c_str());
            return false;
        }
        _clusterMap.emplace(topoNode.clusterName, clusterInfo);
    } else if (CNST_CLUSTER_BIZ_SEARCH == clusterNameSubType) {
        TopoNode baseTopoNode;
        initTopoNode(spec, envDef, topoDef, param, baseTopoNode, _bizNameStr);
        for (const auto &pair : heartbeatSpec.clusterMap) {
            const auto &clusterName = pair.first;
            auto targetWeight = pair.second;
            auto clusterInfo = std::make_shared<ClientClusterTopoInfo>();
            auto &topoNode = clusterInfo->topoNode;
            topoNode = baseTopoNode;
            topoNode.bizName = MiscUtil::createBizName(clusterName, topoNode.serverBizName);
            topoNode.clusterName = clusterName;
            topoNode.meta.targetWeight = std::min(topoNode.meta.targetWeight, targetWeight);
            if (!topoNode.generateNodeId()) {
                AUTIL_LOG(ERROR, "invalid topo node [%s], clusterName [%s]",
                          topoNode.nodeId.c_str(), clusterName.c_str());
                return false;
            }
            _clusterMap.emplace(topoNode.clusterName, clusterInfo);
        }
    } else if (CNST_CLUSTER_NAME_AS_BIZ_NAME == clusterNameSubType) {
        TopoNode baseTopoNode;
        initTopoNode(spec, envDef, topoDef, param, baseTopoNode, _bizNameStr);
        bool isValidDefault =
            (_isDefaultBiz && 1 == baseTopoNode.partCnt && 0 == baseTopoNode.partId);
        if (!isValidDefault) {
            return true;
        }
        for (const auto &pair : heartbeatSpec.clusterMap) {
            const auto &clusterName = pair.first;
            auto targetWeight = pair.second;
            auto clusterInfo = std::make_shared<ClientClusterTopoInfo>();
            auto &topoNode = clusterInfo->topoNode;
            topoNode = baseTopoNode;
            topoNode.bizName = clusterName;
            topoNode.clusterName = clusterName;
            topoNode.meta.targetWeight = std::min(topoNode.meta.targetWeight, targetWeight);
            if (!topoNode.generateNodeId()) {
                AUTIL_LOG(ERROR, "invalid topo node [%s], clusterName [%s]",
                          topoNode.nodeId.c_str(), clusterName.c_str());
                return false;
            }
            _clusterMap.emplace(topoNode.clusterName, clusterInfo);
        }
    } else {
        AUTIL_LOG(ERROR, "unknown cluster name sub type");
        return false;
    }
    return true;
}

void ClientTopoInfo::initTopoNode(const Spec &spec, const GigMetaEnv &envDef,
                                  const BizTopoDef &topoDef, HashInfoParam &param,
                                  TopoNode &topoNode, std::shared_ptr<std::string> &bizNameStr) {
    const auto &keyDef = topoDef.key();
    initBizName(keyDef, param, topoNode, bizNameStr);
    topoNode.partCnt = keyDef.part_count();
    topoNode.partId = keyDef.part_id();
    topoNode.version = keyDef.version();
    topoNode.protocalVersion = keyDef.protocol_version();
    topoNode.weight = topoDef.target_weight();
    if (topoDef.target_weight() < MIN_WEIGHT) {
        topoNode.meta.targetWeight = topoDef.target_weight();
    } else {
        topoNode.meta.targetWeight = MAX_WEIGHT;
    }
    topoNode.meta.metaEnv.setFromProto(envDef);
    topoNode.isValid = true;
    topoNode.ssType = ST_HEARTBEAT;
    topoNode.spec = spec;
}

void ClientTopoInfo::initBizName(const BizTopoKeyDef &keyDef, HashInfoParam &param,
                                 TopoNode &topoNode, std::shared_ptr<std::string> &bizNameStr) {
    auto bizNameHash = keyDef.biz_name_hash();
    if (0 != bizNameHash) {
        auto value = param.getStr(bizNameHash);
        if (value) {
            topoNode.serverBizName = *value;
            bizNameStr = value;
        } else {
            return;
        }
    } else {
        topoNode.serverBizName = keyDef.biz_name();
    }
}

bool ClientTopoInfo::initMetas(SignatureTy sig, const BizMetasDef &metas, HashInfoParam &param) {
    auto newMetaMap = std::make_shared<MetaMap>();
    auto count = metas.metas_size();
    for (int32_t i = 0; i < count; i++) {
        const auto &metaDef = metas.metas(i);
        newMetaMap->emplace(metaDef.key(), metaDef.value());
    }
    std::unordered_set<std::shared_ptr<std::string>> newMetaStrSet;
    if (!initHashMetas(metas, newMetaMap, param, newMetaStrSet)) {
        return false;
    }
    size_t flowConfigHash = 0;
    auto flowControlConfig =
        parseFlowControlConfig(newMetaMap, param, flowConfigHash, newMetaStrSet);
    if (newMetaMap->empty()) {
        newMetaMap.reset();
    }
    autil::ScopedReadWriteLock lock(_lock, 'w');
    _signature.meta = sig;
    _metas = std::move(newMetaMap);
    if (flowControlConfig) {
        _flowControlConfig = std::move(flowControlConfig);
        _flowConfigHash = flowConfigHash;
    }
    _metaStrSet = std::move(newMetaStrSet);
    return true;
}

FlowControlConfigPtr
ClientTopoInfo::parseFlowControlConfig(const MetaMapPtr &metaMap, HashInfoParam &param,
                                       uint64_t &hash,
                                       std::unordered_set<std::shared_ptr<std::string>> &strSet) {
    if (!metaMap) {
        return FlowControlConfigPtr();
    }
    auto it = metaMap->find(GIG_HEARTBEAT_FLOW_CONTROL_META_KEY);
    if (metaMap->end() == it) {
        return FlowControlConfigPtr();
    }
    auto configStr = it->second;
    metaMap->erase(it);
    auto config = std::make_shared<FlowControlConfig>();
    try {
        autil::legacy::FromJsonString(*config, configStr);
        std::shared_ptr<std::string> strPtr;
        hash = param.stringTable->addStr(configStr, &strPtr);
        strSet.insert(std::move(strPtr));
        return config;
    } catch (const autil::legacy::ExceptionBase &e) {
        AUTIL_LOG(ERROR, "invalid gig flow control config [%s], error_info[%s]", configStr.c_str(),
                  e.GetMessage().c_str());
        return FlowControlConfigPtr();
    }
}

bool ClientTopoInfo::initHashMetas(const BizMetasDef &metas, const MetaMapPtr &metaMap,
                                   HashInfoParam &param,
                                   std::unordered_set<std::shared_ptr<std::string>> &strSet) {
    auto count = metas.hash_metas_size();
    for (int32_t i = 0; i < count; i++) {
        const auto &metaDef = metas.hash_metas(i);
        auto key = param.getStr(metaDef.key());
        auto value = param.getStr(metaDef.value());
        if (!key || !value) {
            return false;
        }
        metaMap->emplace(*key, *value);
        strSet.insert(key);
        strSet.insert(value);
    }
    return true;
}

FlowControlConfigPtr ClientTopoInfo::getFlowControlConfig(uint64_t &hash) const {
    autil::ScopedReadWriteLock lock(_lock, 'r');
    hash = _flowConfigHash;
    return _flowControlConfig;
}

void ClientTopoInfo::initTags(SignatureTy sig, const BizTagsDef &tags) {
    auto newTagMap = std::make_shared<TagMap>();
    auto count = tags.tags_size();
    for (int32_t i = 0; i < count; i++) {
        const auto &tagDef = tags.tags(i);
        newTagMap->emplace(tagDef.tag(), tagDef.version());
    }
    if (newTagMap->empty()) {
        newTagMap.reset();
    }
    autil::ScopedReadWriteLock lock(_lock, 'w');
    _signature.tag = sig;
    _tags = std::move(newTagMap);
}

bool ClientTopoInfo::update(const BizTopoDef &bizTopoDef, int64_t netLatencyUs,
                            HashInfoParam &param, bool &needNotify) {
    const auto &signature = bizTopoDef.signature();
    bool flush = false;
    if (_signature.meta != signature.meta()) {
        if (initMetas(signature.meta(), bizTopoDef.metas(), param)) {
            needNotify = true;
            flush = true;
        } else {
            return false;
        }
    }
    if (_signature.tag != signature.tag()) {
        initTags(signature.tag(), bizTopoDef.tags());
        flush = true;
    }
    auto oldTargetWeight = _targetWeight;
    if (_targetWeight != bizTopoDef.target_weight()) {
        _targetWeight = bizTopoDef.target_weight();
        flush = true;
    }
    if (bizTopoDef.has_propagation_stat()) {
        updatePropagationStat(bizTopoDef.propagation_stat(), netLatencyUs);
    }
    if (flush) {
        flushUpdate(oldTargetWeight);
    }
    return true;
}

void ClientTopoInfo::flushUpdate(WeightTy oldTargetWeight) {
    auto ignoreNextSkip = ((oldTargetWeight != _targetWeight) && (MIN_WEIGHT == _targetWeight));
    autil::ScopedReadWriteLock lock(_lock, 'w');
    for (const auto &pair : _clusterMap) {
        flushProvider(pair.second->provider);
    }
    _hostStats->ignoreNextSkip = ignoreNextSkip;
}

void ClientTopoInfo::flushProvider(const std::shared_ptr<SearchServiceProvider> &provider) const {
    if (!provider) {
        return;
    }
    auto oldTargetWeight = provider->getTargetWeight();
    if (oldTargetWeight != _targetWeight) {
        provider->setTargetWeight(_targetWeight, false);
    }
    if (_tags) {
        provider->updateTagsFromMap(*_tags);
    } else {
        provider->updateTagsFromMap(TagMap());
    }
    provider->updateHeartbeatMetas(_metas);
}

void ClientTopoInfo::bind(const std::string &clusterName,
                          const std::shared_ptr<SearchServiceReplica> &replica,
                          const std::shared_ptr<SearchServiceProvider> &provider) {
    if (!replica || !provider) {
        return;
    }
    auto it = _clusterMap.find(clusterName);
    if (_clusterMap.end() == it) {
        return;
    }
    if (!bindOne(it->second, replica, provider)) {
        return;
    }
    flushUpdate(_targetWeight);
    if (_firstPropagationStatDef.has_latency() &&
        _firstPropagationStatDef.latency().filter_ready()) {
        updatePropagationStat(_firstPropagationStatDef, _initNetLatencyUs);
        _firstPropagationStatDef.Clear();
    }
}

bool ClientTopoInfo::bindOne(const ClientClusterTopoInfoPtr &clientInfo,
                             const std::shared_ptr<SearchServiceReplica> &replica,
                             const std::shared_ptr<SearchServiceProvider> &provider) {
    auto &thisReplica = clientInfo->replica;
    auto &thisProvider = clientInfo->provider;
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        if (thisReplica == replica && thisProvider == provider) {
            return false;
        }
    }
    {
        autil::ScopedReadWriteLock lock(_lock, 'w');
        if (thisReplica == replica && thisProvider == provider) {
            return false;
        }
        if (thisProvider != provider) {
            thisProvider = provider;
            thisProvider->setHostStats(_hostStats);
            if (_hasServerAgent) {
                thisProvider->setHasServerAgent(true);
            }
        }
        thisReplica = replica;
        return true;
    }
}

void ClientTopoInfo::unBind() {
    autil::ScopedReadWriteLock lock(_lock, 'w');
    for (const auto &pair : _clusterMap) {
        pair.second->unBind();
    }
}

void ClientTopoInfo::updatePropagationStat(const PropagationStatDef &statDef,
                                           int64_t netLatencyUs) {
    for (const auto &pair : _clusterMap) {
        updateOnePropagationStat(pair.second, statDef, netLatencyUs);
    }
}

void ClientTopoInfo::updateOnePropagationStat(const ClientClusterTopoInfoPtr &clientInfo,
                                              const PropagationStatDef &statDef,
                                              int64_t netLatencyUs) {
    std::shared_ptr<SearchServiceProvider> provider;
    std::shared_ptr<SearchServiceReplica> replica;
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        provider = clientInfo->provider;
        replica = clientInfo->replica;
    }
    SearchServiceResource::updateProviderFromHeartbeat(replica, provider, statDef, netLatencyUs);
}

void ClientTopoInfo::fillTopoRequest(TopoHeartbeatRequest &request) const {
    std::vector<std::shared_ptr<SearchServiceReplica>> replicaVec;
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        auto sigDef = request.mutable_signature();
        sigDef->set_topo(_signature.topo);
        sigDef->set_meta(_signature.meta);
        sigDef->set_tag(_signature.tag);
        sigDef->set_publish_id(_signature.publishId);
        if (MIN_WEIGHT == _targetWeight) {
            for (const auto &pair : _clusterMap) {
                if (pair.second->replica) {
                    replicaVec.push_back(pair.second->replica);
                }
            }
        }
    }
    if (!replicaVec.empty()) {
        auto hasStarted = false;
        for (const auto &replica : replicaVec) {
            if (replica->hasStartedProvider()) {
                hasStarted = true;
                break;
            }
        }
        request.set_all_provider_stopped(!hasStarted);
    }
}

size_t ClientTopoInfo::fillBizInfoMap(const ClientTopoInfoPtr &thisInfo,
                                      const std::shared_ptr<ServerId> &serverId,
                                      BizInfoMap &bizInfoMap,
                                      BizFlowControlInfoMap &flowControlMap) const {
    for (const auto &pair : _clusterMap) {
        auto &topoNode = pair.second->topoNode;
        topoNode.clientInfo = thisInfo;
        topoNode.meta.metaEnv = serverId->getMetaEnv();
        bizInfoMap[topoNode.bizName].emplace(topoNode);
        topoNode.clientInfo.reset();
        fillFlowControlInfo(topoNode, flowControlMap);
    }
    return _clusterMap.size();
}

void ClientTopoInfo::fillFlowControlInfo(const TopoNode &topoNode,
                                         BizFlowControlInfoMap &flowControlMap) const {
    uint64_t flowConfigHash = 0;
    auto flowControlConfig = getFlowControlConfig(flowConfigHash);
    if (!flowControlConfig) {
        return;
    }
    const auto &bizName = topoNode.bizName;
    auto it = flowControlMap.find(bizName);
    auto version = topoNode.version;
    if (flowControlMap.end() == it) {
        BizFlowControlInfo flowControlInfo;
        flowControlInfo.version = version;
        flowControlInfo.hash = flowConfigHash;
        flowControlInfo.config = flowControlConfig;
        flowControlMap.emplace(bizName, std::move(flowControlInfo));
    } else {
        auto &controlInfo = it->second;
        bool overrideClient =
            (flowControlConfig->overrideClientConfig && !controlInfo.config->overrideClientConfig);
        if (overrideClient || controlInfo.hash < flowConfigHash) {
            controlInfo.version = version;
            controlInfo.hash = flowConfigHash;
            controlInfo.config = flowControlConfig;
        }
    }
}

void ClientTopoInfo::disableProvider() const {
    autil::ScopedReadWriteLock lock(_lock, 'r');
    for (const auto &pair : _clusterMap) {
        const auto &provider = pair.second->provider;
        if (provider) {
            provider->setWeight(MIN_WEIGHT_FLOAT);
        }
    }
}

void ClientTopoInfo::toStringOne(const ClientClusterTopoInfoPtr &clusterInfo, std::string &debugStr,
                                 MetasSignatureMap &allMetas) const {
    const auto &topoNode = clusterInfo->topoNode;
    std::shared_ptr<SearchServiceProvider> provider;
    std::shared_ptr<SearchServiceReplica> replica;
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        provider = clusterInfo->provider;
        replica = clusterInfo->replica;
    }
    debugStr += "\n    " + topoNode.nodeId;
    debugStr += ", targetWeight: " + autil::StringUtil::toString(topoNode.meta.targetWeight);
    debugStr += ", hasProvider: " + autil::StringUtil::toString(provider != nullptr);
    if (provider) {
        debugStr +=
            ", providerTargetWeight: " + autil::StringUtil::toString(provider->getTargetWeight());
    }
    debugStr += ", hasReplica: " + autil::StringUtil::toString(replica != nullptr);
}

void ClientTopoInfo::toString(std::string &debugStr, MetasSignatureMap &allMetas) const {
    BizTopoSignature signature;
    TagMapPtr tags;
    MetaMapPtr metas;
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        signature = _signature;
        tags = _tags;
        metas = _metas;
    }
    debugStr += "\n  topoSig: " + autil::StringUtil::toString(signature.topo);
    debugStr += ", defaultBiz: " + autil::StringUtil::toString(_isDefaultBiz);
    if (tags) {
        debugStr += ", tagsSig: " + autil::StringUtil::toString(signature.tag);
        debugStr += "(";
        size_t n = tags->size();
        size_t i = 0;
        for (const auto &tagPair : *tags) {
            debugStr += autil::StringUtil::toString(tagPair.first);
            debugStr += ":" + autil::StringUtil::toString(tagPair.second);
            i++;
            if (i < n) {
                debugStr += ", ";
            }
        }
        debugStr += ")";
    }
    if (metas) {
        debugStr += ", metasSig: " + autil::StringUtil::toString(signature.meta);
        if (allMetas.end() == allMetas.find(signature.meta)) {
            allMetas[signature.meta] = metas;
        }
    }
    for (const auto &pair : _clusterMap) {
        toStringOne(pair.second, debugStr, allMetas);
    }
    debugStr += "\n";
}

ClientTopoInfoMap::ClientTopoInfoMap(uint64_t clientId) : _clientId(clientId) {
}

ClientTopoInfoMap::~ClientTopoInfoMap() {
}

bool ClientTopoInfoMap::init(const std::string &ip, const ClientTopoInfoMapPtr &oldMap,
                             const NewHeartbeatResponse &response) {
    _serverRpcVersion = response.rpc_version();
    auto serverSig = response.server_signature();
    if (oldMap) {
        auto oldServerId = oldMap->getServerId();
        if (!oldServerId) {
            AUTIL_LOG(ERROR, "BUG! null server id");
            return false;
        }
        if (oldServerId->getSignature() == serverSig) {
            _serverId = oldServerId;
            return true;
        }
    }
    if (!response.has_server_id()) {
        AUTIL_LOG(ERROR, "heartbeat response has no server_id");
        return false;
    }
    auto serverId = std::make_shared<ServerId>();
    serverId->fromProto(response.server_signature(), ip, response.server_id());
    _serverId = serverId;
    return true;
}

void ClientTopoInfoMap::addInfo(int64_t topoSig, const ClientTopoInfoPtr &info) {
    _topoMap[topoSig] = info;
}

ClientTopoInfoPtr ClientTopoInfoMap::getInfo(SignatureTy topoSig) const {
    auto it = _topoMap.find(topoSig);
    if (_topoMap.end() != it) {
        return it->second;
    }
    return nullptr;
}

bool ClientTopoInfoMap::update(const NewHeartbeatResponse &response, int64_t netLatencyUs,
                               HashInfoParam &param, bool &needNotify) {
    _clientId = response.client_id();
    if (_serverId->getSignature() != response.server_signature()) {
        return false;
    }
    bool isSame = true;
    auto responseCount = response.topo_size();
    for (int32_t i = 0; i < responseCount; i++) {
        const auto &topoDef = response.topo(i);
        const auto &signature = topoDef.signature();
        auto clientInfo = getInfo(signature.topo());
        if (!clientInfo) {
            isSame = false;
            continue;
        }
        if (clientInfo->getPublishId() != signature.publish_id()) {
            isSame = false;
            continue;
        }
        bool notify = false;
        if (!clientInfo->update(topoDef, netLatencyUs, param, notify)) {
            isSame = false;
            continue;
        }
        if (notify) {
            needNotify = true;
        }
    }
    return isSame && ((size_t)responseCount == _topoMap.size());
}

void ClientTopoInfoMap::fillRequest(NewHeartbeatRequest &request) const {
    request.set_client_id(_clientId);
    request.set_server_signature(_serverId->getSignature());
    for (const auto &pair : _topoMap) {
        const auto &clientInfo = *pair.second;
        auto topoRequest = request.add_topo_request();
        clientInfo.fillTopoRequest(*topoRequest);
    }
}

bool ClientTopoInfoMap::fillBizInfoMap(const HeartbeatSpec &spec, BizInfoMap &bizInfoMap,
                                       BizFlowControlInfoMap &flowControlMap) const {
    if (_topoMap.empty()) {
        return true;
    }
    const auto &serverId = getServerId();
    size_t totalCount = 0;
    for (const auto &pair : _topoMap) {
        totalCount +=
            pair.second->fillBizInfoMap(pair.second, serverId, bizInfoMap, flowControlMap);
    }
    if (_serverRpcVersion == 1 && 0 == totalCount) {
        fillNoTopoNode(spec, serverId, bizInfoMap);
    }
    return true;
}

void ClientTopoInfoMap::fillTopoSignatures(multi_call::TopoSignatureSet &signatures) const {
    for (const auto &pair : _topoMap) {
        signatures.insert(pair.first);
    }
}

void ClientTopoInfoMap::fillNoTopoNode(const HeartbeatSpec &spec,
                                       const std::shared_ptr<ServerId> &serverId,
                                       BizInfoMap &bizInfoMap) {
    TopoNode baseTopoNode;
    baseTopoNode.spec = serverId->getSpec();
    baseTopoNode.partCnt = 1;
    baseTopoNode.partId = 0;
    baseTopoNode.version = DEFAULT_VERSION_ID;
    baseTopoNode.protocalVersion = DEFAULT_VERSION_ID;
    baseTopoNode.isValid = true;
    baseTopoNode.ssType = ST_HEARTBEAT_NO_TOPO;
    baseTopoNode.meta.metaEnv = serverId->getMetaEnv();
    for (const auto &pair : spec.clusterMap) {
        auto topoNode = baseTopoNode;
        const auto &clusterName = pair.first;
        auto targetWeight = pair.second;
        topoNode.bizName = clusterName;
        topoNode.clusterName = clusterName;
        topoNode.weight = targetWeight;
        topoNode.meta.targetWeight = targetWeight;
        if (!topoNode.generateNodeId()) {
            AUTIL_LOG(ERROR, "invalid topo node [%s], clusterName [%s]", topoNode.nodeId.c_str(),
                      clusterName.c_str());
            continue;
        }
        bizInfoMap[topoNode.bizName].emplace(topoNode);
    }
}

void ClientTopoInfoMap::disableAllProvider() const {
    for (const auto &pair : _topoMap) {
        pair.second->disableProvider();
    }
}

void ClientTopoInfoMap::toString(std::string &debugStr, MetasSignatureMap &allMetas) const {
    debugStr += ", clientId: " + autil::StringUtil::toString(_clientId);
    debugStr += ", serverSig: " + autil::StringUtil::toString(_serverId->getSignature());
    debugStr += ", topoCount: " + autil::StringUtil::toString(_topoMap.size());
    for (const auto &pair : _topoMap) {
        pair.second->toString(debugStr, allMetas);
    }
}
} // namespace multi_call
