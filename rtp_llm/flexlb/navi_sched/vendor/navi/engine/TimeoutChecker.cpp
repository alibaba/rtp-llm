#include "navi/engine/TimeoutChecker.h"

#include "autil/TimeUtility.h"
#include "navi/log/NaviLogger.h"

namespace navi {

TimeoutChecker::TimeoutChecker(int64_t timeoutMs) {
    _begin = autil::TimeUtility::currentTime();
    setTimeoutMs(timeoutMs);
}

TimeoutChecker::~TimeoutChecker() {
}

void TimeoutChecker::setBeginTime(int64_t beginTime) {
    _begin = beginTime;
}

void TimeoutChecker::setTimeoutMs(int64_t timeoutMs) {
    assert(_begin >= 0);
    _timeoutMs = timeoutMs;
    int64_t timeoutMsLimit = (std::numeric_limits<int64_t>::max() - _begin) / FACTOR_MS_TO_US;
    timeoutMs = std::max(-1L, std::min(timeoutMsLimit, timeoutMs));
    _end = _begin + timeoutMs * FACTOR_MS_TO_US;
    NAVI_KERNEL_LOG(TRACE3, "set timeoutMs[%ld] begin[%ld] end[%ld]", timeoutMs, _begin, _end);
}

int64_t TimeoutChecker::beginTime() const {
    return _begin;
}

int64_t TimeoutChecker::endTime() const {
    return _end;
}

int64_t TimeoutChecker::timeoutMs() const {
    return _timeoutMs;
}

int64_t TimeoutChecker::timeoutUs() const {
    return _timeoutMs * FACTOR_MS_TO_US;
}

int64_t TimeoutChecker::remainTime() const {
    return (_end - autil::TimeUtility::currentTime()) / FACTOR_MS_TO_US;
}

bool TimeoutChecker::timeout() const {
    return autil::TimeUtility::currentTime() > _end;
}

int64_t TimeoutChecker::elapsedTime() const {
    return autil::TimeUtility::currentTime() - _begin;
}

}
