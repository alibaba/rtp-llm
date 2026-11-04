#pragma once

#include "navi/engine/Resource.h"
#include "navi/engine/TimeoutChecker.h"

namespace navi {

class TimeoutCheckerR : public ExternalResource
{
public:
    TimeoutCheckerR();
    TimeoutCheckerR(TimeoutChecker *timeoutChecker);
    ~TimeoutCheckerR();
    TimeoutCheckerR(const TimeoutCheckerR &) = delete;
    TimeoutCheckerR &operator=(const TimeoutCheckerR &) = delete;
public:
    TimeoutChecker *getTimeoutChecker() const {
        return _timeoutChecker;
    }
public:
    static const std::string RESOURCE_ID;
private:
    TimeoutChecker *_timeoutChecker = nullptr;
};

NAVI_TYPEDEF_PTR(TimeoutCheckerR);

}

