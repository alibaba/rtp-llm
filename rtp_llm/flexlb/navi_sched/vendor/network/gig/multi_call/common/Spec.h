#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"

namespace multi_call {

struct Spec {
    Spec();
    bool operator==(const Spec &rhs) const;
    bool operator!=(const Spec &rhs) const;
    std::string getAnetSpec(ProtocolType type) const;
    std::string getSpecStr(ProtocolType type) const;
    bool hasValidPort() const;

    std::string ip;
    int32_t ports[MC_PROTOCOL_UNKNOWN + 1];

    friend std::ostream &operator<<(std::ostream &os, const Spec &spec);
};

enum ClusterNameSubType {
    CNST_UNKNOWN = 0,
    CNST_NORMAL = 1,
    CNST_CLUSTER_BIZ_SEARCH = 2,
    CNST_CLUSTER_NAME_AS_BIZ_NAME = 3,
};

struct HeartbeatSpec {
public:
    bool operator==(const HeartbeatSpec &rhs) const {
        return spec == rhs.spec && clusterNameSubType == rhs.clusterNameSubType &&
               clusterMap == rhs.clusterMap;
    }
public:
    Spec spec;
    ClusterNameSubType clusterNameSubType = CNST_UNKNOWN;
    std::map<std::string, WeightTy> clusterMap;
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatSpec);

struct HeartbeatSpecMap {
public:
    void add(const Spec &spec, ClusterNameSubType clusterNameSubType, const std::string &clusterName,
             WeightTy targetWeight)
    {
        auto addr = spec.getAnetSpec(MC_PROTOCOL_GRPC_STREAM);
        auto it = specMap.find(addr);
        if (specMap.end() == it) {
            auto specPtr = std::make_shared<HeartbeatSpec>();
            specPtr->spec = spec;
            specPtr->clusterNameSubType = clusterNameSubType;
            specPtr->clusterMap.emplace(clusterName, targetWeight);
            specMap[addr] = specPtr;
        } else {
            it->second->clusterMap.emplace(clusterName, targetWeight);
        }
    }
    void merge(const HeartbeatSpecMap &other) {
        for (const auto &pair : other.specMap) {
            auto it = specMap.find(pair.first);
            if (specMap.end() == it) {
                specMap.emplace(pair);
            } else {
                it->second->clusterMap.insert(pair.second->clusterMap.begin(),
                                              pair.second->clusterMap.end());
            }
        }
    }
    bool empty() const {
        return specMap.empty();
    }
    size_t size() const {
        return specMap.size();
    }
public:
    std::map<std::string, HeartbeatSpecPtr> specMap;
};

} // namespace multi_call

