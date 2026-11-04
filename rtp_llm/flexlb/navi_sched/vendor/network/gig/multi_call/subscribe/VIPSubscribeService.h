#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/SubscribeClustersConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeService.h"
#include "autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VipSubscriptionClient.h"

namespace multi_call {

class VIPSubscribeService : public SubscribeService
{
public:
    VIPSubscribeService(const VipConfig &config, VipSubscriptionClientPtr client);
    ~VIPSubscribeService();

private:
    VIPSubscribeService(const VIPSubscribeService &);
    VIPSubscribeService &operator=(const VIPSubscribeService &);

public:
    bool init() override;
    SubscribeType getType() override {
        return ST_VIP;
    }
    bool clusterInfoNeedUpdate() override;
    bool getClusterInfoMap(TopoNodeVec &topoNodeVec, HeartbeatSpecMap &heartbeatSpecs) override;

    bool addSubscribe(const std::vector<std::string> &names) override;
    bool deleteSubscribe(const std::vector<std::string> &names) override;
    bool addSubscribeDomainConfig(const std::vector<VIPDomainConfig> &vipDomains);
    bool deleteSubscribeDomainConfig(const std::vector<VIPDomainConfig> &vipDomains);
    std::string getServiceKey() const override {
        return _config.serviceKey;
    }

private:
    VipSubscriptionClientPtr _client;
    autil::ReadWriteLock _configLock;
    VipConfig _config;
    std::unordered_map<std::string, VIPDomainConfig> _domainConfigs;

private:
    static uint32_t QUERY_TIMEOUT;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(VIPSubscribeService);

} // namespace multi_call

