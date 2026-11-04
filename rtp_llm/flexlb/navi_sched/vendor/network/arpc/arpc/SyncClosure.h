#pragma once
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadcond.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class SyncClosure : public RPCClosure {
public:
    SyncClosure() { _needWait = true; }
    virtual ~SyncClosure() {}

public:
    virtual void Run() {
        anet::MutexGuard guard(&_cond);
        _needWait = false;
        _cond.signal();
    }

    void WaitReply() {
        _cond.lock();

        while (_needWait) {
            _cond.wait();
        }

        _cond.unlock();
    }

private:
    anet::ThreadCond _cond;
    bool _needWait;
};

ARPC_END_NAMESPACE(arpc);

