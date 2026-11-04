#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/StringTable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"

namespace multi_call {

class SearchServiceProvider;
class SearchServiceReplica;
class HostHeartbeatStats;
class ServerId;

struct ClientClusterTopoInfo {
    ~ClientClusterTopoInfo();

public:
    void unBind() {
        provider.reset();
        replica.reset();
    }

public:
    TopoNode topoNode;
    std::shared_ptr<SearchServiceProvider> provider;
    std::shared_ptr<SearchServiceReplica> replica;
};

MULTI_CALL_TYPEDEF_PTR(ClientClusterTopoInfo);

typedef std::unordered_set<uint64_t> LackHashSet;
MULTI_CALL_TYPEDEF_PTR(LackHashSet);

class StringTable;
struct HashInfoParam {
public:
    std::shared_ptr<std::string> getStr(uint64_t hash) {
        auto value = stringTable->getStr(hash);
        if (!value) {
            if (!lackHashSet) {
                lackHashSet = std::make_shared<std::unordered_set<uint64_t>>();
            }
            lackHashSet->insert(hash);
        }
        return value;
    }

public:
    std::shared_ptr<StringTable> stringTable;
    LackHashSetPtr lackHashSet;
};

class ClientTopoInfo
{
public:
    ClientTopoInfo(const std::shared_ptr<HostHeartbeatStats> &hostStats);
    ~ClientTopoInfo();

public:
    bool init(const HeartbeatSpec &heartbeatSpec, const Spec &spec, const GigMetaEnv &envDef,
              const BizTopoDef &topoDef, int64_t netLatencyUs, HashInfoParam &param);
    SignatureTy getPublishId() const {
        return _signature.publishId;
    }
    const std::shared_ptr<HostHeartbeatStats> &getHostStats() const {
        return _hostStats;
    }
    bool update(const BizTopoDef &bizTopoDef, int64_t netLatencyUs, HashInfoParam &param,
                bool &needNotify);
    void bind(const std::string &clusterName, const std::shared_ptr<SearchServiceReplica> &replica,
              const std::shared_ptr<SearchServiceProvider> &provider);
    void unBind();
    void fillTopoRequest(TopoHeartbeatRequest &request) const;
    size_t fillBizInfoMap(const std::shared_ptr<ClientTopoInfo> &thisInfo,
                          const std::shared_ptr<ServerId> &serverId, BizInfoMap &bizInfoMap,
                          BizFlowControlInfoMap &flowControlMap) const;
    void disableProvider() const;
    void toString(std::string &debugStr, MetasSignatureMap &allMetas) const;

private:
    bool createTopoNodes(const HeartbeatSpec &heartbeatSpec, const Spec &spec,
                         const GigMetaEnv &envDef, const BizTopoDef &topoDef, HashInfoParam &param);
    bool initMetas(SignatureTy sig, const BizMetasDef &metas, HashInfoParam &param);
    bool initHashMetas(const BizMetasDef &metas, const MetaMapPtr &metaMap, HashInfoParam &param,
                       std::unordered_set<std::shared_ptr<std::string>> &strSet);
    void initTags(SignatureTy sig, const BizTagsDef &tags);
    void updatePropagationStat(const PropagationStatDef &statDef, int64_t netLatencyUs);
    void updateOnePropagationStat(const ClientClusterTopoInfoPtr &clientInfo,
                                  const PropagationStatDef &statDef, int64_t netLatencyUs);
    bool bindOne(const ClientClusterTopoInfoPtr &clientInfo,
                 const std::shared_ptr<SearchServiceReplica> &replica,
                 const std::shared_ptr<SearchServiceProvider> &provider);
    void flushUpdate(WeightTy oldTargetWeight);
    void flushProvider(const std::shared_ptr<SearchServiceProvider> &provider) const;
    void toStringOne(const ClientClusterTopoInfoPtr &clusterInfo, std::string &debugStr,
                     MetasSignatureMap &allMetas) const;
    void fillFlowControlInfo(const TopoNode &topoNode, BizFlowControlInfoMap &flowControlMap) const;
    std::shared_ptr<FlowControlConfig> getFlowControlConfig(uint64_t &hash) const;

private:
    static void initTopoNode(const Spec &spec, const GigMetaEnv &envDef, const BizTopoDef &topoDef,
                             HashInfoParam &param, TopoNode &topoNode,
                             std::shared_ptr<std::string> &bizNameStr);
    static void initBizName(const BizTopoKeyDef &keyDef, HashInfoParam &param, TopoNode &topoNode,
                            std::shared_ptr<std::string> &bizNameStr);
    static std::shared_ptr<FlowControlConfig>
    parseFlowControlConfig(const MetaMapPtr &metaMap, HashInfoParam &param, uint64_t &hash,
                           std::unordered_set<std::shared_ptr<std::string>> &strSet);

private:
    std::shared_ptr<HostHeartbeatStats> _hostStats;
    mutable autil::ReadWriteLock _lock;
    MetaMapPtr _metas;
    std::shared_ptr<FlowControlConfig> _flowControlConfig;
    uint64_t _flowConfigHash = 0;
    std::shared_ptr<std::string> _bizNameStr;
    std::unordered_set<std::shared_ptr<std::string>> _metaStrSet;
    TagMapPtr _tags;
    BizTopoSignature _signature;
    bool _hasServerAgent;
    bool _isDefaultBiz = false;
    PropagationStatDef _firstPropagationStatDef;
    int64_t _initNetLatencyUs;
    WeightTy _targetWeight;
    std::map<std::string, ClientClusterTopoInfoPtr> _clusterMap;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ClientTopoInfo);

class ClientTopoInfoMap;
MULTI_CALL_TYPEDEF_PTR(ClientTopoInfoMap);

class ClientTopoInfoMap
{
public:
    ClientTopoInfoMap(uint64_t clientId);
    ~ClientTopoInfoMap();

public:
    bool init(const std::string &ip, const ClientTopoInfoMapPtr &oldMap,
              const NewHeartbeatResponse &response);
    ClientTopoInfoPtr getInfo(SignatureTy topoSig) const;
    void addInfo(int64_t topoSig, const ClientTopoInfoPtr &info);
    bool update(const NewHeartbeatResponse &response, int64_t netLatencyUs, HashInfoParam &param,
                bool &needNotify);
    void fillRequest(NewHeartbeatRequest &request) const;
    bool fillBizInfoMap(const HeartbeatSpec &spec, BizInfoMap &bizInfoMap,
                        BizFlowControlInfoMap &flowControlMap) const;
    void fillTopoSignatures(multi_call::TopoSignatureSet &signatures) const;
    void disableAllProvider() const;
    void toString(std::string &debugStr, MetasSignatureMap &allMetas) const;
    const std::shared_ptr<ServerId> &getServerId() const {
        return _serverId;
    }

private:
    static void fillNoTopoNode(const HeartbeatSpec &spec, const std::shared_ptr<ServerId> &serverId,
                               BizInfoMap &bizInfoMap);

private:
    uint64_t _clientId;
    int64_t _serverRpcVersion = 0;
    std::shared_ptr<ServerId> _serverId;
    std::map<SignatureTy, ClientTopoInfoPtr> _topoMap;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call
