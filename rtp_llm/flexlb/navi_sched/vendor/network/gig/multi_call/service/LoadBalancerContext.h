#pragma once

#include <map>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ProviderSelectorFactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelectorFactory.h"
#include "autil/Lock.h"

namespace multi_call {

class LoadBalancerContext
{
public:
    LoadBalancerContext(const SearchServicePtr &searchService)
        : _versionSelectorFactory(searchService->getVersionSelectorFactory())
        , _providerSelectorFactory(searchService->getProviderSelectorFactory()) {
    }
    virtual ~LoadBalancerContext() {
    }

public:
    VersionSelectorPtr getVersionSelector(const std::string &bizname);
    ProviderSelectorPtr getProviderSelector(const std::string &bizname);

private:
    autil::ThreadMutex _mutex;
    // bizname -> versionSelector
    std::map<std::string, VersionSelectorPtr> _versionSelectorMap;
    // bizname -> providerSelector
    std::map<std::string, ProviderSelectorPtr> _providerSelectorMap;

    VersionSelectorFactoryPtr _versionSelectorFactory;
    ProviderSelectorFactoryPtr _providerSelectorFactory;
};

MULTI_CALL_TYPEDEF_PTR(LoadBalancerContext);

} // namespace multi_call

