#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/Lock.h"

namespace multi_call {

class Closure;

class TerminateNotifier
{
public:
    TerminateNotifier();
    ~TerminateNotifier();

private:
    TerminateNotifier(const TerminateNotifier &);
    void operator=(const TerminateNotifier &);

public:
    void inc(int partIdIndex);
    void dec(int partIdIndex);
    void setClosure(Closure *closure);
    Closure *stealClosure();
    bool canTerminate() const {
        return _partitionCount == 0;
    }

private:
    void incPartitionsByIndex(int index);
    void decPartitionsByIndex(int index);

private:
    volatile int _partitionCount;
    Closure *_closure;
    std::vector<int> _partitions;

    mutable autil::ThreadMutex _mutex;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

