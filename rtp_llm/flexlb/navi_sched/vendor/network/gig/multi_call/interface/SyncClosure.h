#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Closure.h"
#include "autil/Lock.h"

namespace multi_call {

class SyncClosure : public Closure
{
public:
    SyncClosure();
    virtual ~SyncClosure();

private:
    SyncClosure(const SyncClosure &);
    SyncClosure &operator=(const SyncClosure &);

public:
    virtual void Run() override;

public:
    virtual int wait();

protected:
    volatile bool _isFinished;
    autil::ThreadCond _cond;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

