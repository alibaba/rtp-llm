#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/MetaEnv.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/Spec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/BizTopo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"

namespace multi_call {

class TopoInfoBuilder;
class StringTable;

class ServerId
{
public:
    void init(const Spec &spec);
    void fillServerIdDef(const SignatureMapPtr &clientSignatureMap,
                         NewHeartbeatResponse &response) const;
    void fromProto(SignatureTy signature, const std::string &ip, const ServerIdDef &def);
    const Spec &getSpec() const {
        return _spec;
    }
    const MetaEnv &getMetaEnv() const {
        return _metaEnv;
    }
    void toString(std::string &ret) const;
    SignatureTy getSignature() const {
        return _signature;
    }

private:
    void initSignature();

private:
    MetaEnv _metaEnv;
    Spec _spec;
    SignatureTy _signature;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ServerId);

class ServerTopoMap
{
public:
    ServerTopoMap();
    ~ServerTopoMap();

private:
    ServerTopoMap(const ServerTopoMap &);
    ServerTopoMap &operator=(const ServerTopoMap &);

public:
    bool setSpec(const Spec &spec);
    std::string getTopoInfoStr() const;
    void setServerId(const ServerIdPtr &serverId);
    ServerIdPtr getServerId() const;
    void setBizTopoMap(const BizTopoMapPtr &newMap, bool recycle = true);
    BizTopoMapPtr getBizTopoMap() const;
    BizTopoMapPtr getBizTopoMap(bool &hasDefaultBiz) const;
    const StringTablePtr &getStringTable() const;
    std::string toString() const;

private:
    std::shared_ptr<StringTable> _stringTable;
    mutable autil::ReadWriteLock _lock;
    ServerIdPtr _serverId;
    bool _hasDefaultBiz = false;
    BizTopoMapPtr _bizTopoMap;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ServerTopoMap);

} // namespace multi_call
