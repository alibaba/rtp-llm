#pragma once

#include <functional>

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"

#include <limits>

namespace navi_sched {

class MasterPendingCacheR;

struct CostSchedulerPreparedState {
    size_t requestCount = 0;
    size_t nodeCount = 0;
    std::vector<int64_t> requestTokenCounts;
    int64_t totalTokenCount = 0;
    std::vector<double> costs;
    std::vector<double> queueWaits;
    std::vector<double> weights;
    bool parameterGradientsValid = false;
};

class ScheduleReplayRunner {
public:
    template <typename Scheduler>
    static bool runContext(NaviSchedScheduleContext &context, const Scheduler &scheduler,
                           std::vector<NaviSchedDecision> &decisions)
    {
        if (!prepare(context)) {
            decisions.clear();
            return false;
        }
        if (!scheduler.schedule(context) || !collectDecisions(context, decisions)) {
            decisions.clear();
            return false;
        }
        return true;
    }
    template <typename Scheduler>
    static bool run(const NaviSchedScheduleReplay &replay, const Scheduler &scheduler,
                    std::vector<NaviSchedDecision> &decisions)
    {
        NaviSchedScheduleContext context;
        if (!makeContext(replay, context)) {
            decisions.clear();
            return false;
        }
        return runContext(context, scheduler, decisions);
    }
    static void addResponseRequest(NaviSchedScheduleContext &context,
                                   const MasterScheduleRequestDataPtr &request);
    static void setReplay(NaviSchedScheduleContext &context,
                          const std::shared_ptr<NaviSchedScheduleReplay> &replay);
    static bool finish(NaviSchedScheduleContext &context);
    static const autil::legacy::json::JsonMap *getSchedulerInfo(
        const NaviSchedScheduleContext &context);
    static bool decisionsEqual(const std::vector<NaviSchedDecision> &expected,
                               const std::vector<NaviSchedDecision> &actual,
                               const std::string &sharedBizName = "");
    static void initialize(NaviSchedScheduleContext &context, int64_t scheduleTimeUs,
                           std::vector<NaviSchedScheduleItem> requests,
                           std::vector<NaviSchedNodeConstPtr> nodes);
    static int64_t getScheduleTimeUs(const NaviSchedScheduleContext &context);
    static double makeEstimatedQueueWaitMs(int64_t scheduleTimeUs,
                                           double reportedQueueWaitMs,
                                           int64_t reportReceiveTimeUs,
                                           double decayHeadRemainingMs,
                                           double timeCompression = 1.0);
    static double makeUnifiedDecayedLatencyMs(
        int64_t scheduleTimeUs, int64_t reportReceiveTimeUs,
        int64_t reportedBatchCount, double latencyWithReportedWorkloadMs,
        double latencyWithoutReportedWorkloadMs,
        double timeCompression = 1.0,
        bool *usesReportedWorkload = nullptr);
    static bool makeContext(const NaviSchedScheduleReplay &replay,
                            NaviSchedScheduleContext &context,
                            std::shared_ptr<MasterPendingCacheR> pendingCache = nullptr);
    static bool prepareContext(NaviSchedScheduleContext &context);
    static bool collectDecisions(
        NaviSchedScheduleContext &context,
        std::vector<NaviSchedDecision> &decisions);
    static bool initializeCostSchedulerWeights(
        size_t requestCount, size_t nodeCount,
        CostSchedulerPreparedState &preparedState);
    static bool prepareCostScheduler(
        NaviSchedScheduleContext &context,
        CostSchedulerPreparedState &preparedState);
    static bool runCostScheduler(
        const NaviSchedScheduleReplay &replay,
        const NaviSchedScheduleContextPtr &context, double fallbackLambda,
        double fallbackAlpha, double fallbackAlphaDecay,
        double fallbackMinAlpha, int32_t fallbackMaxLoopCount,
        double latencyWeightConcavity,
        std::vector<NaviSchedDecision> &decisions,
        double latencyTokenNormalizationPower =
            std::numeric_limits<double>::quiet_NaN(),
        double latencyWeightConcavityOverride =
            std::numeric_limits<double>::quiet_NaN(),
        const std::function<void(const NaviSchedScheduleContext &)> &inspectInput = {});

private:
    static bool prepare(NaviSchedScheduleContext &context);
    static NaviSchedNodeSnapshot makeNodeSnapshot(const NaviSchedNodeInfo &info);
};

} // namespace navi_sched
