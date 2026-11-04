#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
namespace anet {

int64_t TimeUtil::_now = 0;
const int64_t TimeUtil::MIN = (1ll << 63);
const int64_t TimeUtil::MAX = ~(1ll << 63);
const int64_t TimeUtil::PRE_MAX = ~(1ll << 63) - 1;

/*
 * 得到当前时间
 */
int64_t TimeUtil::getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (static_cast<int64_t>(t.tv_sec) * static_cast<int64_t>(1000000) + static_cast<int64_t>(t.tv_usec));
}

/*
 * 设置当前时间
 */
void TimeUtil::setNow() { _now = getTime(); }

/*
 * buf should have at least 26 char in length.
 */
void TimeUtil::getTimeStr(int64_t tval, char *buf) {
    time_t t = tval / 1000000;
    ctime_r(&t, buf);
}

} // namespace anet
