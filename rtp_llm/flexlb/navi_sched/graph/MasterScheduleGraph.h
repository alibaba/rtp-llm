#pragma once

#include "autil/Log.h"
#include "navi/engine/Navi.h"
#include <cstdint>
#include <memory>
#include <string>

namespace navi_sched {

class MasterScheduleR;
class SchedulerConfigR;

class MasterScheduleGraph {
public:
    enum class State : uint8_t {
        STOPPED,
        RUNNING,
        RESTARTING,
        STOPPING,
    };

public:
    MasterScheduleGraph(const navi::NaviPtr &navi,
                        const std::shared_ptr<MasterScheduleR> &masterScheduleR,
                        const std::shared_ptr<SchedulerConfigR> &schedulerConfigR,
                        const std::string &masterBizName,
                        const std::string &contextKernelConfig,
                        const std::string &schedulerKernelId,
                        const std::string &schedulerKernelConfig, bool enableVis = false);
    ~MasterScheduleGraph();

public:
    bool start();
    void stop();
    bool running() const;
    State getState() const;
    std::string getStateName() const;
    uint64_t getRestartCount() const;

private:
    static std::unique_ptr<navi::GraphDef> buildGraph(
        const std::string &masterBizName, const std::string &contextKernelConfig,
        const std::string &schedulerKernelId,
        const std::string &schedulerKernelConfig,
        uint64_t requestPipeGeneration);
    class Impl;
    std::unique_ptr<Impl> _impl;

private:
    AUTIL_LOG_DECLARE();
};

NAVI_TYPEDEF_PTR(MasterScheduleGraph);

} // namespace navi_sched
