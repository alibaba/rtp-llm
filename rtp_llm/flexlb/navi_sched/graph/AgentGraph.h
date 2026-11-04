#pragma once

#include "autil/Log.h"
#include "navi/engine/Navi.h"
#include <cstdint>
#include <memory>
#include <string>

class NaviSchedResourceTest;

namespace navi_sched {

class AgentBatchR;
class AgentLatencyInputR;

class AgentGraph {
public:
    enum class State : uint8_t {
        STOPPED,
        RUNNING,
        RESTARTING,
        STOPPING,
    };

public:
    AgentGraph(const navi::NaviPtr &navi, const std::shared_ptr<AgentBatchR> &agentBatchR,
               const std::shared_ptr<AgentLatencyInputR> &latencyInputR,
               const std::string &agentBizName, bool enableVis = false);
    ~AgentGraph();

public:
    bool start();
    void stop();
    bool running() const;
    State getState() const;
    std::string getStateName() const;
    uint64_t getRestartCount() const;

private:
    friend class ::NaviSchedResourceTest;

    bool abortCurrentGraphForTest();
    static std::unique_ptr<navi::GraphDef> buildGraph(
        const std::string &agentBizName, uint64_t batchPipeGeneration,
        uint64_t latencyPipeGeneration);
    class Impl;
    std::unique_ptr<Impl> _impl;

private:
    AUTIL_LOG_DECLARE();
};

NAVI_TYPEDEF_PTR(AgentGraph);

}
