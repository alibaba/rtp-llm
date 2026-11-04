#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelector.h"

#include "autil/StringUtil.h"

using namespace std;
using namespace autil;

namespace multi_call {

bool VersionSelector::needCopy(size_t copyProviderCount, size_t normalProviderCount) {
    if (0 == normalProviderCount) {
        return true;
    }
    auto rangeRandom = _randomGenerator.get() % MAX_WEIGHT;
    auto prob = ((double)copyProviderCount) / normalProviderCount;
    return rangeRandom < prob * MAX_WEIGHT;
}

} // namespace multi_call
