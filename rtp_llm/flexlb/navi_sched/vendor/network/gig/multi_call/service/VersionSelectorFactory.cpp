#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelectorFactory.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/RandomVersionSelector.h"

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, VersionSelectorFactory);

VersionSelectorPtr VersionSelectorFactory::create(const std::string &bizName) {
    return RandomVersionSelectorPtr(new RandomVersionSelector(bizName, _subscribeServiceManager));
}

} // namespace multi_call
