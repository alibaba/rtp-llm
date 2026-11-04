#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceManager.h"

namespace multi_call {

class VersionSelectorFactory
{
public:
    VersionSelectorFactory(const SubscribeServiceManagerPtr &subscribeManager)
        : _subscribeServiceManager(subscribeManager) {
    }
    virtual ~VersionSelectorFactory() = default;

public:
    VersionSelectorPtr create(const std::string &bizName);

private:
    SubscribeServiceManagerPtr _subscribeServiceManager;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(VersionSelectorFactory);

} // namespace multi_call

