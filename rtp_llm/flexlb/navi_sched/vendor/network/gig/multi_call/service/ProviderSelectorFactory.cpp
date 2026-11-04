#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ProviderSelectorFactory.h"

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, ProviderSelectorFactory);

ProviderSelectorPtr ProviderSelectorFactory::create(const std::string &bizName) {
    return ProviderSelectorPtr(new ProviderSelector(bizName));
}

} // namespace multi_call
