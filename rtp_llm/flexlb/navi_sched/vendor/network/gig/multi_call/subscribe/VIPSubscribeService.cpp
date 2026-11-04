#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VIPSubscribeService.h"

#include <utility>

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, VIPSubscribeService);

uint32_t VIPSubscribeService::QUERY_TIMEOUT = 10 * 1000; // ms

VIPSubscribeService::VIPSubscribeService(const VipConfig &config, VipSubscriptionClientPtr client)
    : _client(std::move(client)), _config(config) {
}

VIPSubscribeService::~VIPSubscribeService() = default;

bool VIPSubscribeService::init() {
    if (!_client) {
        AUTIL_LOG(ERROR, "VIP subscription client is not supplied by the embedding application");
        return false;
    }
    return _client->init(_config.jmenvDom, _config.queryFullApi);
}

bool VIPSubscribeService::clusterInfoNeedUpdate() {
    autil::ScopedReadLock lock(_configLock);
    if (!_config.clusters.empty()) {
        return true;
    }
    return false;
}

bool VIPSubscribeService::getClusterInfoMap(TopoNodeVec &topoNodeVec,
                                            HeartbeatSpecMap &heartbeatSpecs) {
    std::vector<std::string> clusters;
    std::unordered_map<std::string, VIPDomainConfig> domainConfigs;
    {
        // Do not hold the configuration lock while querying the external client.
        autil::ScopedReadLock lock(_configLock);
        clusters = _config.clusters;
        domainConfigs = _domainConfigs;
    }

    for (const auto &clusterName : clusters) {
        std::vector<VipHost> hosts;
        if (!_client->queryAllIp(clusterName, hosts, QUERY_TIMEOUT)) {
            AUTIL_LOG(ERROR, "QueryAllIp failed, domain: [%s]", clusterName.c_str());
            return false;
        }
        for (const auto &host : hosts) {
            TopoNode topoNode;
            topoNode.bizName = clusterName;
            topoNode.partCnt = 1;
            topoNode.partId = 0;
            topoNode.version = DEFAULT_VERSION_ID;
            topoNode.weight = host.weight >= MAX_WEIGHT ? MAX_WEIGHT : host.weight;
            topoNode.meta.targetWeight = topoNode.weight;
            topoNode.clusterName = clusterName;
            topoNode.isValid = host.valid;
            topoNode.ssType = ST_VIP;
            topoNode.spec.ip = string(host.ip);
            auto port = host.port;
            auto iter = domainConfigs.find(clusterName);
            if (iter == domainConfigs.end()) {
                topoNode.spec.ports[MC_PROTOCOL_GRPC_STREAM] =
                    port + (_config.supportHeartbeat ? _config.grpcPortDiff : 0);
            } else {
                const auto &domainConfig = iter->second;
                topoNode.spec.ports[MC_PROTOCOL_GRPC_STREAM] =
                    port + domainConfig.getGrpcStreamPortDiff();
            }
            if (_config.supportHeartbeat) {
                heartbeatSpecs.add(topoNode.spec, CNST_NORMAL, std::string(),
                                   topoNode.meta.targetWeight);
            } else if (topoNode.generateNodeId()) {
                topoNodeVec.push_back(topoNode);
            } else {
                AUTIL_LOG(ERROR, "generateNodeId failed: [%s], domain: [%s]", host.ip.c_str(),
                          clusterName.c_str());
            }
        }
    }
    return true;
}

bool VIPSubscribeService::addSubscribe(const std::vector<std::string> &names) {
    autil::ScopedWriteLock lock(_configLock);
    std::vector<std::string> &clusters = _config.clusters;
    for (const auto &name : names) {
        if (name.empty()) {
            continue;
        }
        if (std::find(clusters.begin(), clusters.end(), name) != clusters.end()) {
            continue;
        }
        clusters.emplace_back(name);
    }
    return true;
}

bool VIPSubscribeService::addSubscribeDomainConfig(const std::vector<VIPDomainConfig> &vipDomains) {
    std::vector<std::string> names;
    {
        autil::ScopedWriteLock lock(_configLock);
        for (const auto &domainConfig : vipDomains) {
            names.push_back(domainConfig.getVipDomain());
            _domainConfigs[domainConfig.getVipDomain()] = domainConfig;
        }
    }
    return addSubscribe(names);
}

bool VIPSubscribeService::deleteSubscribe(const std::vector<std::string> &names) {
    autil::ScopedWriteLock lock(_configLock);
    std::vector<std::string> &clusters = _config.clusters;

    for (const auto &name : names) {
        auto iter = clusters.begin();
        while (iter != clusters.end()) {
            if (*iter == name) {
                clusters.erase(iter);
                break;
            }
            ++iter;
        }
    }
    return true;
}

bool VIPSubscribeService::deleteSubscribeDomainConfig(
    const std::vector<VIPDomainConfig> &vipDomains) {
    std::vector<std::string> names;
    {
        autil::ScopedWriteLock lock(_configLock);
        for (const auto &domainConfig : vipDomains) {
            names.push_back(domainConfig.getVipDomain());
            _domainConfigs.erase(domainConfig.getVipDomain());
        }
    }
    return deleteSubscribe(names);
}

} // namespace multi_call
