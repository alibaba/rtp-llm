#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/DiffCounter.h"

#include "autil/StringUtil.h"

using namespace std;

namespace multi_call {

DiffCounter::DiffCounter() : _snapshot(0) {
    atomic_set(&_counter, 0);
}

DiffCounter::~DiffCounter() {
}

void DiffCounter::inc() {
    atomic_inc(&_counter);
}

int64_t DiffCounter::snapshot() {
    auto currentValue = current();
    auto diff = currentValue - _snapshot;
    _snapshot = currentValue;
    return diff;
}

int64_t DiffCounter::current() const {
    return atomic_read(&_counter);
}

void DiffCounter::setCurrent(int64_t value) {
    atomic_set(&_counter, value);
}

std::string DiffCounter::snapshotStr() {
    auto diff = snapshot();
    return "" + autil::StringUtil::toString(current()) + "(" + autil::StringUtil::toString(diff) +
           ")";
}

} // namespace multi_call
