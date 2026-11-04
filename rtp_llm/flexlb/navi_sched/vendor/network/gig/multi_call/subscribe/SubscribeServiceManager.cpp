#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceManager.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VIPSubscribeService.h"

using namespace std;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, SubscribeServiceManager);

SubscribeServiceManager::SubscribeServiceManager() {
}

SubscribeServiceManager::~SubscribeServiceManager() {
}

bool SubscribeServiceManager::clusterInfoNeedUpdate() {
    autil::ScopedReadLock lock(_subLock);
    for (const auto &service : _subServiceVec) {
        if (service->clusterInfoNeedUpdate()) {
            return true;
        }
    }
    return false;
}

bool SubscribeServiceManager::getClusterInfoMap(TopoNodeVec &topoNodeVec,
                                                HeartbeatSpecMap &heartbeatSpecs) {
    std::vector<SubscribeServicePtr> updateServiceVec;
    tryGetClusterInfoMapFromUpdateService(topoNodeVec, heartbeatSpecs, updateServiceVec);

    autil::ScopedReadLock lock(_subLock);
    for (const auto &service : _subServiceVec) {
        if (!service->isEnable()) {
            AUTIL_LOG(DEBUG, "disabled subscribe service type:%u", service->getType());
            continue;
        }
        if (std::count(updateServiceVec.begin(), updateServiceVec.end(), service) > 0) {
            AUTIL_LOG(DEBUG,
                      "updated service has called getClusterInfoMap, type: %s, service key: %s",
                      convertSubscribeTypeToStr(service->getType()).c_str(),
                      service->getServiceKey().c_str());
            continue;
        }
        if (!service->getClusterInfoMap(topoNodeVec, heartbeatSpecs)) {
            REPORT_METRICS(reportSubUpdateFailedQps);
            AUTIL_LOG(DEBUG, "Error on service->getClusterInfoMap.");
            return false;
        } else {
            REPORT_METRICS(reportSubUpdateSuccQps);
        }
    }
    return true;
}

bool SubscribeServiceManager::addSubscribeService(const SubscribeConfig &config) {
    bool vipEnabled = false;
    if (!config.validate(vipEnabled)) {
        return false;
    }
    autil::ScopedWriteLock lock(_subLock);
    if (vipEnabled) {
        auto service = SubscribeServiceFactory::createVipSub(config.vipConfig);
        if (!service->init()) {
            AUTIL_LOG(ERROR, "VIP subscription init failed");
            return false;
        }
        _subServiceVec.push_back(service);
    }
    return !_subServiceVec.empty() || config.allowEmptySub;
}

bool SubscribeServiceManager::updateSubscribeService(const SubscribeConfig &config) {
    bool vipEnabled = false;
    if (!config.validate(vipEnabled) || !vipEnabled) {
        return false;
    }
    std::vector<SubscribeServicePtr> services;
    {
        autil::ScopedReadLock lock(_subLock);
        services = _subServiceVec;
    }
    UpdateSubScribeServiceMap updates;
    if (!tryCreateNewSubscribeService(config.vipConfig, services, updates)) {
        return false;
    }
    autil::ScopedWriteLock lock(_updateServiceLock);
    for (const auto &[type, entries] : updates) {
        for (const auto &[key, servicePair] : entries) {
            _updateServiceMap[type][key] = servicePair;
        }
    }
    return true;
}

bool SubscribeServiceManager::addSubscribe(const SubscribeClustersConfig &config) {
    bool handled = false;
    bool result = true;
    autil::ScopedReadLock lock(_subLock);
    for (const auto &service : _subServiceVec) {
        auto vip = std::dynamic_pointer_cast<VIPSubscribeService>(service);
        if (!vip) {
            return false;
        }
        result &= vip->addSubscribeDomainConfig(config.vipDomains);
        handled = true;
    }
    return config.checkSub(handled) && result;
}

bool SubscribeServiceManager::deleteSubscribe(const SubscribeClustersConfig &config) {
    bool handled = false;
    bool result = true;
    autil::ScopedReadLock lock(_subLock);
    for (const auto &service : _subServiceVec) {
        auto vip = std::dynamic_pointer_cast<VIPSubscribeService>(service);
        if (!vip) {
            return false;
        }
        result &= vip->deleteSubscribeDomainConfig(config.vipDomains);
        handled = true;
    }
    return config.checkSub(handled) && result;
}







bool SubscribeServiceManager::tryCreateNewSubscribeService(
    const VipConfig &config, const std::vector<SubscribeServicePtr> &services,
    UpdateSubScribeServiceMap &updates) const {
    if (config.serviceKey.empty()) {
        return false;
    }
    for (const auto &service : services) {
        if (service->getServiceKey() != config.serviceKey) {
            continue;
        }
        auto replacement = SubscribeServiceFactory::createVipSub(config);
        if (!replacement->init()) {
            return false;
        }
        updates[ST_VIP][config.serviceKey] = std::make_pair(service, replacement);
        return true;
    }
    return false;
}

void SubscribeServiceManager::tryGetClusterInfoMapFromUpdateService(
    TopoNodeVec &topoNodeVec, HeartbeatSpecMap &heartbeatSpecMap,
    std::vector<SubscribeServicePtr> &updateServiceVec) {
    {
        // 大多数情况下 map 应该都为空
        autil::ScopedReadLock lock(_updateServiceLock);
        if (_updateServiceMap.empty()) {
            return;
        }
    }
    std::map<SubscribeServicePtr, SubscribeServicePtr> serviceMap;
    {
        autil::ScopedWriteLock lock(_updateServiceLock);
        for (auto &[serviceType, serviceKeyMap] : _updateServiceMap) {
            for (auto it = serviceKeyMap.begin(); it != serviceKeyMap.end();) {
                const auto oldService = it->second.first;
                const auto newService = it->second.second;
                TopoNodeVec tempTopoNodeVec;
                HeartbeatSpecMap tempHeartbeatSpecMap;
                if (!newService->getClusterInfoMap(tempTopoNodeVec, tempHeartbeatSpecMap) ||
                    (tempTopoNodeVec.empty() && tempHeartbeatSpecMap.empty())) {
                    it = std::next(it);
                    continue;
                }
                if (!tempTopoNodeVec.empty()) {
                    topoNodeVec.insert(topoNodeVec.end(), tempTopoNodeVec.begin(),
                                       tempTopoNodeVec.end());
                }
                if (!tempHeartbeatSpecMap.empty()) {
                    heartbeatSpecMap.merge(tempHeartbeatSpecMap);
                }
                serviceMap[oldService] = newService;
                it = serviceKeyMap.erase(it);
            }
        }
    }
    if (serviceMap.empty()) {
        return;
    }
    {
        autil::ScopedWriteLock lock(_subLock);
        for (const auto &[oldService, newService] : serviceMap) {
            auto it = std::find(_subServiceVec.begin(), _subServiceVec.end(), oldService);
            if (it == _subServiceVec.end()) {
                continue;
            }
            // 停止老的 service
            oldService->setEnable(false);
            oldService->stopSubscribe();
            // 替换为新的 service
            *it = newService;
            updateServiceVec.push_back(newService);
        }
    }
}

} // namespace multi_call
