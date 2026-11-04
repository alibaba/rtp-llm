#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/Lock.h"

namespace multi_call {

struct RetryCheckerItem {
    int32_t retryCountPerSecond = 0;
    int64_t retryTimeStamp = 0;
    autil::ReadWriteLock limitLock;
    bool canRetry(int64_t currentTimeInSeconds, int32_t retryCountLimit);

private:
    AUTIL_LOG_DECLARE();
};
class RetryLimitChecker
{
public:
    RetryLimitChecker();
    ~RetryLimitChecker();

private:
    RetryLimitChecker(const RetryLimitChecker &);
    RetryLimitChecker &operator=(const RetryLimitChecker &);

public:
    bool canRetry(const std::string &strategy, int64_t currentTimeInSeconds,
                  int32_t retryCountLimit);

private:
    std::unordered_map<std::string, std::unique_ptr<RetryCheckerItem>> _StrategyRetryChecker;
    autil::ReadWriteLock _checkerLock;
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(RetryLimitChecker);

} // namespace multi_call

