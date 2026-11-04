#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/LoadBalancerContext.h"

namespace multi_call {

VersionSelectorPtr LoadBalancerContext::getVersionSelector(const std::string &bizName) {
    autil::ScopedLock lock(_mutex);
    auto iter = _versionSelectorMap.find(bizName);
    if (iter != _versionSelectorMap.end()) {
        return iter->second;
    }
    // create
    VersionSelectorPtr versionSelector = _versionSelectorFactory->create(bizName);
    if (versionSelector) {
        _versionSelectorMap[bizName] = versionSelector;
    }
    return versionSelector;
}

ProviderSelectorPtr LoadBalancerContext::getProviderSelector(const std::string &bizName) {
    autil::ScopedLock lock(_mutex);
    auto iter = _providerSelectorMap.find(bizName);
    if (iter != _providerSelectorMap.end()) {
        return iter->second;
    }
    // create
    ProviderSelectorPtr providerSelector = _providerSelectorFactory->create(bizName);
    if (providerSelector) {
        _providerSelectorMap[bizName] = providerSelector;
    }
    return providerSelector;
}

} // namespace multi_call
