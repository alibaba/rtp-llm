#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerFeedBack.h"

namespace multi_call {

class WarmUpController
{
public:
    WarmUpController() : _isFinished(true) {
    }
    ~WarmUpController() {
    }

private:
    WarmUpController(const WarmUpController &);
    WarmUpController &operator=(const WarmUpController &);

public:
    float update(ControllerFeedBack &feedBack);
    bool needProbe() const {
        return !_isFinished;
    }

private:
    // for ut
    void setFinished(bool finished) {
        _isFinished = finished;
    }

private:
    volatile bool _isFinished;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(WarmUpController);

} // namespace multi_call

