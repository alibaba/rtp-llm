#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MetricUtil.h"

#include <algorithm>

using namespace std;

namespace multi_call {

string MetricUtil::normalizeTag(const string &tag) {
    string tmp = tag; // copy on write
    ::replace(tmp.begin(), tmp.end(), '@', '-');
    return tmp;
}

string MetricUtil::normalizeEmpty(const std::string &tag) {
    if (!tag.empty()) {
        return tag;
    } else {
        return GIG_UNKNOWN;
    }
}

} // namespace multi_call
