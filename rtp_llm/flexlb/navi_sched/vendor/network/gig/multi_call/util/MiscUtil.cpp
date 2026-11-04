#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MiscUtil.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, MiscUtil);

MiscUtil::MiscUtil() {
}

MiscUtil::~MiscUtil() {
}

std::string MiscUtil::createBizName(const std::string &cluster, const std::string &biz) {
    if (cluster.empty()) {
        return biz;
    } else if (biz.empty()) {
        return cluster;
    } else {
        return biz + GIG_CLUSTER_BIZ_SPLITTER + cluster;
    }
}

float MiscUtil::scaleLinear(uint32_t begin, uint32_t end, float pos) {
    if (pos <= begin || begin > end) {
        return MIN_PERCENT;
    }
    uint32_t range = end - begin;
    if (0 == range) {
        return MAX_PERCENT;
    } else {
        return min(MAX_PERCENT, (pos - begin) / range);
    }
}

} // namespace multi_call
