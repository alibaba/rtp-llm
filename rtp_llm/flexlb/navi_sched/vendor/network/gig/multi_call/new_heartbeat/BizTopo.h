#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/StringTable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"
#include "autil/Lock.h"

namespace multi_call {

class TopoInfoBuilder;

struct BizTopoClientInfo {
    BizTopoSignature sig;
    bool allProviderStopped = false;
};

struct SignatureMap {
    uint64_t clientId;
    int64_t serverSignature;
    int64_t heartbeatId = 0;
    bool onlyDefaultBiz = false;
    StringHashRequestDef request;
    std::map<SignatureTy, BizTopoClientInfo> topoSignatureMap;
};

struct TopoInfoHash {
    uint64_t bizName = 0;
    std::vector<std::pair<uint64_t, uint64_t>> metaVec;
};

MULTI_CALL_TYPEDEF_PTR(SignatureMap);

class BizStat;

class BizTopo
{
public:
    BizTopo(SignatureTy publishId, PublishGroupTy group);
    ~BizTopo();

public:
    bool init(const ServerBizTopoInfo &info, StringTable &stringTable,
              const std::shared_ptr<BizStat> &bizStat);
    void updateVolatileInfo(const BizVolatileInfo &info);
    void start();
    void stop();
    bool isStopped() const;
    void fillBizTopoDiff(const BizTopoSignature &clientSideSignature, bool supportHash,
                         BizTopoDef *topoDef) const;
    void fillBizTopo(bool supportHash, BizTopoDef *topoDef) const;
    void fillPropagationStat(BizTopoDef *topoDef) const;
    void addToBuilder(TopoInfoBuilder &builder) const;
    SignatureTy getTopoSignature() const {
        return _signature.topo;
    }
    SignatureTy getPublishId() const {
        return _signature.publishId;
    }
    PublishGroupTy getGroup() const {
        return _group;
    }
    const std::string &getId() const {
        return _id;
    }
    const ServerBizTopoInfo &getTopoInfo() const {
        return _info;
    }
    void toString(std::string &ret) const;

private:
    void initFlowControl();
    void initTopoSignature();
    void initMetaSignature();
    void initTagSignature();
    void initHash(StringTable &stringTable);
    uint64_t collectString(const std::string &value, StringTable &stringTable);
    void fillSignature(TopoSignatureDef *signatureDef) const;
    void fillTopoKey(bool supportHash, BizTopoKeyDef *keyDef) const;
    void fillMeta(bool supportHash, BizMetasDef *metaDef) const;
    void fillTag(BizTagsDef *tagsDef) const;
    WeightTy getTargetWeight() const;

private:
    PublishGroupTy _group;
    std::string _id;
    std::shared_ptr<BizStat> _bizStat;
    mutable autil::ReadWriteLock _tagLock;
    ServerBizTopoInfo _info;
    volatile bool _stopped = false;
    TopoInfoHash _topoInfoHash;
    BizTopoSignature _signature;
    std::vector<std::shared_ptr<std::string>> _strVec;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(BizTopo);

typedef std::map<SignatureTy, BizTopoPtr> BizTopoMap;
MULTI_CALL_TYPEDEF_PTR(BizTopoMap);

} // namespace multi_call
