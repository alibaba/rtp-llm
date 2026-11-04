#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomHash.h"

#include <algorithm>
#include <assert.h>

using namespace std;

namespace multi_call {

int32_t RandomHash::get(uint64_t key, const vector<RandomHashNode> &weights) {
    assert(!weights.empty());
    auto back = weights.back();
    assert(back.weightSum > 0);
    uint32_t r = key % back.weightSum;
    auto iter = std::upper_bound(weights.begin(), weights.end(), r);
    return iter->index;
}

} // namespace multi_call
