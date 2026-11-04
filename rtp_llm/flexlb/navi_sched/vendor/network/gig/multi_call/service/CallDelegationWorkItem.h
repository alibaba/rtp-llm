#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Caller.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ChildNodeReply.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/LoadBalancerContext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshotInVersion.h"

namespace multi_call {

class CallDelegationWorkItem
{
public:
    CallDelegationWorkItem(const CallerPtr &caller,
                           const LoadBalancerContextPtr &loadBalancerContext);
    virtual ~CallDelegationWorkItem();

private:
    CallDelegationWorkItem(const CallDelegationWorkItem &);
    CallDelegationWorkItem &operator=(const CallDelegationWorkItem &);

public:
    // return if this query is really terminated
    virtual bool process(int64_t currentTime, const SearchServiceSnapshotPtr &snapshot);

private:
    bool retry(const std::map<std::string, int32_t> &retryBizInfos,
               const SearchServiceSnapshotPtr &snapshot);

private:
    CallerPtr _caller;
    LoadBalancerContextPtr _loadBalancerContext;
    bool _isRetryQueryReported;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(CallDelegationWorkItem);

} // namespace multi_call

