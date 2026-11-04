#pragma once

#include <functional>
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VipSubscriptionClient.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeService.h"

namespace multi_call {

class SubscribeServiceFactory
{
public:
    using VipClientFactory = std::function<VipSubscriptionClientPtr()>;
    static void setVipClientFactory(VipClientFactory factory);
    static SubscribeServicePtr createVipSub(const VipConfig &vipConfig);
};

} // namespace multi_call

