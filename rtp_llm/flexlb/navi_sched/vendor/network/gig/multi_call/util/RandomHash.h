#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

struct RandomHashNode {
public:
    RandomHashNode(uint32_t weightSum_, uint32_t index_) : weightSum(weightSum_), index(index_) {
    }
    ~RandomHashNode() {
    }
    uint32_t weightSum;
    uint32_t index;
};

inline bool operator<(uint32_t key, const RandomHashNode &rhs) {
    return key < rhs.weightSum;
}

class RandomHash
{
public:
    RandomHash() {
    }
    ~RandomHash() {
    }

private:
    RandomHash(const RandomHash &);
    RandomHash &operator=(const RandomHash &);

public:
    static int32_t get(uint64_t key, const std::vector<RandomHashNode> &weights);
};

} // namespace multi_call

