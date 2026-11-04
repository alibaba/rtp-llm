#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ProviderSelector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelector.h"

namespace multi_call {

class ProviderSelectorFactory
{
public:
    ProviderSelectorFactory() {
    }
    virtual ~ProviderSelectorFactory() = default;

private:
    ProviderSelectorFactory(const ProviderSelectorFactory &) = delete;
    ProviderSelectorFactory &operator=(const ProviderSelectorFactory &) = delete;

public:
    ProviderSelectorPtr create(const std::string &bizName);

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ProviderSelectorFactory);

} // namespace multi_call

