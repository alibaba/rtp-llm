#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/TerminateNotifier.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Closure.h"

using namespace std;
using namespace autil;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, TerminateNotifier);

TerminateNotifier::TerminateNotifier() : _partitionCount(0), _closure(NULL) {
}

TerminateNotifier::~TerminateNotifier() {
    _closure = NULL;
}

void TerminateNotifier::inc(int partIdIndex) {
    autil::ScopedLock lock(_mutex);
    incPartitionsByIndex(partIdIndex);
}

void TerminateNotifier::dec(int partIdIndex) {
    Closure *closure = NULL;
    {
        autil::ScopedLock lock(_mutex);
        decPartitionsByIndex(partIdIndex);
        if (canTerminate()) {
            if (_closure) {
                closure = _closure;
                _closure = NULL;
            }
        }
    }
    if (closure) {
        closure->Run();
    }
}

void TerminateNotifier::setClosure(Closure *closure) {
    assert(_closure == NULL);
    bool canRun = false;
    {
        autil::ScopedLock lock(_mutex);
        if (canTerminate()) {
            canRun = true;
        } else {
            _closure = closure;
        }
    }
    if (canRun && closure) {
        closure->Run();
    }
}

Closure *TerminateNotifier::stealClosure() {
    Closure *closure = NULL;
    {
        autil::ScopedLock lock(_mutex);
        closure = _closure;
        _closure = NULL;
    }
    return closure;
}

void TerminateNotifier::incPartitionsByIndex(int index) {
    if (index < 0) {
        return;
    }
    int actualSize = _partitions.size();
    int expectSize = index + 1;
    if (actualSize < expectSize) {
        _partitions.resize(expectSize, 0);
    }
    if (0 == _partitions[index]) {
        _partitionCount = _partitionCount + 1;
    }
    ++_partitions[index];
}

void TerminateNotifier::decPartitionsByIndex(int index) {
    if (index < 0 || index >= (int)_partitions.size()) {
        return;
    }
    --_partitions[index];
    if (0 == _partitions[index]) {
        _partitionCount = _partitionCount - 1;
        assert(_partitionCount >= 0);
    }
}

} // namespace multi_call
