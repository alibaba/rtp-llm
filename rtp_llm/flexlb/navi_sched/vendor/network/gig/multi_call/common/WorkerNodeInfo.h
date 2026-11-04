#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/HbInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/MetaEnv.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/Spec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/StringConvertor.h"
#include "autil/StringUtil.h"
#include "autil/legacy/jsonizable.h"

namespace multi_call {

typedef std::map<std::string, std::string> NodeAttributeMap;

struct TopoNodeMeta : public autil::legacy::Jsonizable {
    TopoNodeMeta();
    ~TopoNodeMeta();

public:
    void init(const std::string &metaStr);
    void setMetaEnv(const MetaEnv &metaEnv);
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const TopoNodeMeta &rhs) const;

private:
    void clear();

public:
    WeightTy targetWeight;
    NodeAttributeMap attributes;
    MetaEnv metaEnv;

private:
    AUTIL_LOG_DECLARE();
};

class ClientTopoInfo;

struct TopoNode {
public:
    TopoNode();

public:
    bool operator==(const TopoNode &rhs) const;
    bool operator<(const TopoNode &rhs) const;
    bool generateNodeId();
    bool validate() const;
    WeightTy getNodeWeight() const;

public:
    friend std::ostream &operator<<(std::ostream &os, const TopoNode &node);

public:
    std::string nodeId;
    std::string bizName;
    std::string serverBizName;
    PartIdTy partCnt;
    PartIdTy partId;
    VersionTy version;
    VersionTy protocalVersion;
    WeightTy weight;
    Spec spec;
    // for log
    std::string clusterName;
    TopoNodeMeta meta;
    // for heartbeat
    HbInfoPtr hbInfo;
    bool supportHeartbeat;
    bool isValid;
    SubscribeType ssType;
    std::shared_ptr<ClientTopoInfo> clientInfo;
};

struct BizFlowControlInfo {
public:
    bool operator==(const BizFlowControlInfo &other) const {
        return version == other.version && hash == other.hash;
    }
public:
    VersionTy version = INVALID_VERSION_ID;
    uint64_t hash = 0;
    std::shared_ptr<FlowControlConfig> config;
};

typedef std::unordered_map<std::string, std::set<TopoNode>> BizInfoMap;
typedef std::unordered_map<std::string, BizFlowControlInfo> BizFlowControlInfoMap;
MULTI_CALL_TYPEDEF_PTR(BizFlowControlInfoMap);
typedef std::vector<TopoNode> TopoNodeVec;
MULTI_CALL_TYPEDEF_PTR(TopoNodeVec);

} // namespace multi_call

