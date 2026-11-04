#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/SubscribeClustersConfig.h"

#include "autil/StringUtil.h"

using namespace std;
using namespace autil;
using namespace autil::legacy;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, SubscribeClustersConfig);

void VIPDomainConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("vip_domain", _vipDomain, _vipDomain);
    json.Jsonize("grpc_stream_port_diff", _grpcStreamPortDiff, _grpcStreamPortDiff);
}

bool VIPDomainConfig::operator==(const VIPDomainConfig &rhs) const {
    return _vipDomain == rhs._vipDomain && _grpcStreamPortDiff == rhs._grpcStreamPortDiff;
}

void SubscribeClustersConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("vip_domains", vipDomains, vipDomains);
}

bool SubscribeClustersConfig::operator==(const SubscribeClustersConfig &rhs) const {
    return vipDomains == rhs.vipDomains;
}

void SubscribeClustersConfig::addVipDomain(const std::string domain) {
    VIPDomainConfig domainConfig(domain);
    vipDomains.push_back(domainConfig);
}

bool SubscribeClustersConfig::checkSub(bool vipSub) const {
    if (!vipSub && hasVipSub()) {
        AUTIL_LOG(ERROR, "vip domains not handled:%s", ToJsonString(vipDomains).c_str());
        return false;
    }
    return true;
}

} // namespace multi_call
