#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/HbInfo.h"

#include "autil/TimeUtility.h"

using namespace std;

namespace multi_call {

void HbInfo::updateHbMeta(const HbMetaInfoPtr &hbMetaPtr) {
    _lock.lock();
    hbMeta = hbMetaPtr;
    _lock.unlock();

    updateTime();
}

HbMetaInfoPtr HbInfo::getHbMeta() {
    _lock.lock();
    HbMetaInfoPtr tmp = hbMeta; // return a copy
    _lock.unlock();
    return tmp;
}

void HbInfo::updateTime() {
    lastUpdateTime = autil::TimeUtility::currentTime();
}

void HbInfo::updateTime(int64_t time) {
    lastUpdateTime = time;
}

} // namespace multi_call
