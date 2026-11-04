#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerFeedBack.h"

namespace multi_call {

class ErrorController
{
public:
    ErrorController();
    ~ErrorController();

private:
    ErrorController(const ErrorController &);
    ErrorController &operator=(const ErrorController &);

public:
    float update(ControllerFeedBack &feedBack);

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ErrorController);

} // namespace multi_call

