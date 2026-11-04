#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class WarmUpStrategy
{
public:
    WarmUpStrategy(int64_t timeoutInSecond, int64_t queryCountLimit)
        : _timeout(0)
        , _queryCountLimit(0) {
        update(timeoutInSecond, queryCountLimit);
    }
    ~WarmUpStrategy() {
    }

private:
    WarmUpStrategy(const WarmUpStrategy &);
    WarmUpStrategy &operator=(const WarmUpStrategy &);

public:
    void update(int64_t timeoutInSecond, int64_t queryCountLimit) {
        _timeout = timeoutInSecond * FACTOR_S_TO_US;
        _queryCountLimit = queryCountLimit;
    }

public:
    int64_t _timeout;
    int64_t _queryCountLimit;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(WarmUpStrategy);

} // namespace multi_call

