#include "navi/util/NaviClosure.h"
#include <assert.h>

namespace navi {

WaitClosure::WaitClosure()
    : _isFinished(false) {}

WaitClosure::~WaitClosure() {}

void WaitClosure::notify() {
    assert(!_isFinished);
    autil::ScopedLock lock(_cond);
    _isFinished = true;
    _cond.broadcast();
}

void WaitClosure::wait() {
    autil::ScopedLock lock(_cond);
    while (!_isFinished) {
        _cond.wait();
    }
}
bool WaitClosure::isFinish() const {
    return _isFinished;
}

}
