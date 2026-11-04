#pragma once

#include "aios/navi_sched/data/NaviSchedLatencySample.h"

#include <memory>

namespace autil {
class DataBuffer;
}

namespace navi_sched {

struct NaviSchedLatencyPrediction {
    bool valid() const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    NaviSchedLatencyFeatures features;
    NaviSchedLatencyFeatures floorFeatures;
    double fixedLatencyMs = 0.0;
    bool hasFloor = false;
    bool recomputable = false;
};

using NaviSchedLatencyPredictionConstPtr =
    std::shared_ptr<const NaviSchedLatencyPrediction>;

}
