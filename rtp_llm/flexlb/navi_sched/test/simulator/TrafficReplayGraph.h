#pragma once

#include "navi/engine/Navi.h"
#include <memory>
#include <string>

namespace navi_sched::test {

class TrafficReplayR;
class TrafficReplayBridgeR;

class TrafficReplayGraph {
public:
    TrafficReplayGraph(const navi::NaviPtr &navi,
                       const std::shared_ptr<TrafficReplayBridgeR> &trafficReplayBridgeR,
                       const std::string &clientBizName,
                       const std::string &resourceConfig, bool enableVis);
    ~TrafficReplayGraph();

public:
    bool start();
    void stop();

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace navi_sched::test
