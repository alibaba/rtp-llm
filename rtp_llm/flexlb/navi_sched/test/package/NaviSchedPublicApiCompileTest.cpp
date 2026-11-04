#include <navi_sched/NaviSchedAgent.h>

namespace navi_sched {

void verifyNaviSchedPublicApiCompile(const NaviSchedAgentConfig &agentConfig,
                                     const NaviSchedConfig &config)
{
    auto agentInit = &NaviSchedAgent::init;
    auto agentReportCacheInserted = &NaviSchedAgent::reportCacheInserted;
    bool (NaviSchedBatchFinishReporter::*startBatch)(
        const std::vector<uint64_t> &) = &NaviSchedBatchFinishReporter::startBatch;
    bool (NaviSchedBatchFinishReporter::*startBatchAt)(
        const std::vector<uint64_t> &, int64_t) =
        &NaviSchedBatchFinishReporter::startBatch;
    bool (NaviSchedBatchFinishReporter::*finishBatch)(
        const std::vector<NaviSchedRequestStats> &) =
        &NaviSchedBatchFinishReporter::finishBatch;
    bool (NaviSchedBatchFinishReporter::*finishBatchWithExecution)(
        const std::vector<NaviSchedRequestStats> &, double, double, int64_t,
        int64_t, int64_t, int64_t) = &NaviSchedBatchFinishReporter::finishBatch;
    (void)agentInit;
    (void)agentReportCacheInserted;
    (void)startBatch;
    (void)startBatchAt;
    (void)finishBatch;
    (void)finishBatchWithExecution;
    (void)agentConfig;
    (void)config;
}

} // namespace navi_sched
