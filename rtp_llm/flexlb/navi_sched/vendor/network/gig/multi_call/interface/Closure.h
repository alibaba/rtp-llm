#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class Closure
{
public:
    Closure() {
    }
    virtual ~Closure() {
    }

public:
    virtual void Run() = 0;
};

MULTI_CALL_TYPEDEF_PTR(Closure);

typedef Closure SnapshotChangeCallback;

} // namespace multi_call

