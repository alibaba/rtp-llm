#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SyncClosure.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, SyncClosure);

SyncClosure::SyncClosure() : _isFinished(false) {
}

SyncClosure::~SyncClosure() {
}

void SyncClosure::Run() {
    /**
     * Run may be called before wait. such as in local search situation
     */
    assert(!_isFinished);
    autil::ScopedLock lock(_cond);
    _isFinished = true;
    _cond.signal();
}

int SyncClosure::wait() {
    autil::ScopedLock lock(_cond);
    while (!_isFinished) {
        _cond.wait();
    }
    return 0;
}

} // namespace multi_call
