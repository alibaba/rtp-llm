#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/RetryLimitChecker.h"

#include "autil/TimeUtility.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, RetryCheckerItem);
AUTIL_LOG_SETUP(multi_call, RetryLimitChecker);

RetryLimitChecker::RetryLimitChecker() {
}

RetryLimitChecker::~RetryLimitChecker() {
}

bool RetryCheckerItem::canRetry(int64_t currentTimeInSeconds, int32_t retryCountLimit) {
    if (retryCountLimit < 0) {
        return true;
    }
    bool timeStampNeedUpdate = false;
    {
        ScopedReadLock rlock(limitLock);
        if (retryTimeStamp != currentTimeInSeconds) {
            timeStampNeedUpdate = true;
        }
    }

    if (false == timeStampNeedUpdate) {
        if (retryCountPerSecond >= retryCountLimit) {
            return false;
        }
        retryCountPerSecond++;
        if (retryCountPerSecond == retryCountLimit) {
            AUTIL_LOG(WARN, "first reach retry count limit[%d]", retryCountPerSecond);
        }
        return true;
    }

    {
        ScopedWriteLock wlock(limitLock);
        if (retryTimeStamp != currentTimeInSeconds) {
            retryTimeStamp = currentTimeInSeconds;
            retryCountPerSecond = 1;
        }
    }
    return true;
}

bool RetryLimitChecker::canRetry(const string &strategy, int64_t currentTimeInSeconds,
                                 int32_t retryCountLimit) {
    RetryCheckerItem *item = nullptr;
    {
        ScopedReadLock rlock(_checkerLock);
        auto iter = _StrategyRetryChecker.find(strategy);
        if (_StrategyRetryChecker.end() != iter) {
            item = iter->second.get();
        }
    }
    if (item) {
        return item->canRetry(currentTimeInSeconds, retryCountLimit);
    } else {
        ScopedWriteLock wlock(_checkerLock);
        auto iter = _StrategyRetryChecker.find(strategy);
        if (_StrategyRetryChecker.end() != iter) {
            return iter->second->canRetry(currentTimeInSeconds, retryCountLimit);
        } else {
            auto checkerItem = make_unique<RetryCheckerItem>();
            bool ret = checkerItem->canRetry(currentTimeInSeconds, retryCountLimit);
            _StrategyRetryChecker[strategy] = std::move(checkerItem);
            return ret;
        }
    }
}

} // namespace multi_call
