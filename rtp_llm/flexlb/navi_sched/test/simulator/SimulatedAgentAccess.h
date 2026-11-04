#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace navi {
class AsyncPipe;
}

namespace navi_sched {

class NaviSchedAgent;
class NaviSchedAgentImpl;
namespace test {
struct SimulatedExecutionConfig;
}
struct NaviSchedForwardRequest;

class SimulatedAgentAccess {
public:
    static std::shared_ptr<SimulatedAgentAccess> create(
        NaviSchedAgent &agent, const test::SimulatedExecutionConfig &simulateConfig);
    ~SimulatedAgentAccess();
    void stop();
    bool submit(const NaviSchedForwardRequest &request,
                const std::shared_ptr<::navi::AsyncPipe> &completionPipe);
    static bool reportBatch(NaviSchedAgent &agent, uint64_t requestId,
                            int64_t totalTokenCount, int64_t cacheHitTokenCount);

private:
    explicit SimulatedAgentAccess(NaviSchedAgentImpl *impl);
    bool start(const test::SimulatedExecutionConfig &simulateConfig);
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace navi_sched
