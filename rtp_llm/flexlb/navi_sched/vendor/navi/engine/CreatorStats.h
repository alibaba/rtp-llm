#pragma once

#include "navi/common.h"
#include <multi_call/util/Filter.h>

namespace navi {

class CreatorStats
{
public:
    CreatorStats();
    ~CreatorStats();
private:
    CreatorStats(const CreatorStats &);
    CreatorStats &operator=(const CreatorStats &);
public:
    void updateCreateLatency(float latencyUs);
    void updateInitLatency(float latencyUs);
    float getCreateAvgLatency();
    float getInitAvgLatency();
private:
    multi_call::Filter _createLatencyFilter;
    multi_call::Filter _initLatencyFilter;
private:
    static constexpr int32_t FILTER_WINDOW_SIZE = 50;
};

}

