#pragma once

#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "autil/legacy/jsonizable.h"

namespace multi_call {

class VIPDomainConfig : public autil::legacy::Jsonizable
{
public:
    VIPDomainConfig(const std::string &domain = "")
        : _vipDomain(domain)
        , _grpcStreamPortDiff(0) {
    }
    ~VIPDomainConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const VIPDomainConfig &rhs) const;

public:
    const std::string &getVipDomain() const {
        return _vipDomain;
    }
    
    
    
    
    
    int32_t getGrpcStreamPortDiff() const {
        return _grpcStreamPortDiff;
    }
    
    

private:
    std::string _vipDomain;
    int32_t _grpcStreamPortDiff;
};

class SubscribeClustersConfig : public autil::legacy::Jsonizable
{
public:
    SubscribeClustersConfig() {
    }
    ~SubscribeClustersConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const SubscribeClustersConfig &rhs) const;

public:
    void addVipDomain(const std::string domain);
    bool checkSub(bool vipSub) const;
    bool hasVipSub() const {
        return !vipDomains.empty();
    }
public:
    std::vector<VIPDomainConfig> vipDomains;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SubscribeClustersConfig);

} // namespace multi_call

