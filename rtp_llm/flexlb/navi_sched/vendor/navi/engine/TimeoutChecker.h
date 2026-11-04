#pragma once

#include "navi/common.h"

namespace navi {

class TimeoutChecker
{
public:
    TimeoutChecker(int64_t timeoutMs = DEFAULT_TIMEOUT_MS);
    ~TimeoutChecker();
private:
    TimeoutChecker(const TimeoutChecker &);
    TimeoutChecker &operator=(const TimeoutChecker &);
public:
    void setBeginTime(int64_t beginTime);
    void setTimeoutMs(int64_t timeoutMs);
    int64_t beginTime() const;
    int64_t endTime() const;
    int64_t timeoutMs() const;
    int64_t timeoutUs() const;
    bool timeout() const;
    int64_t remainTime() const;
    int64_t elapsedTime() const;
private:
    int64_t _timeoutMs;
    int64_t _begin;
    int64_t _end;
};

}
