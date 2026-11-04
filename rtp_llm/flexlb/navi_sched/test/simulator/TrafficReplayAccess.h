#pragma once

#include "aios/navi_sched/test/simulator/TrafficReplayR.h"
#include <memory>

namespace navi_sched {

class NaviSchedClient;

class TrafficReplayAccess {
public:
    static std::shared_ptr<TrafficReplayAccess> create(
        NaviSchedClient &client, const test::TrafficReplayConfig &trafficConfig);
    ~TrafficReplayAccess();
    void stop();

private:
    TrafficReplayAccess() = default;
    class Impl;
    std::unique_ptr<Impl> _impl;
};

}
