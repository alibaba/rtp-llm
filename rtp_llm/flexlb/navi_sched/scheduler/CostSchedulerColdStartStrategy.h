#pragma once

#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "aios/network/gig/multi_call/util/RandomGenerator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace navi_sched {

class CostSchedulerColdStartStrategy {
public:
    struct Analysis {
        long double trustedBacklogLatencyWeightedSumMs = 0.0L;
        int64_t trustedBacklogInflight = 0;
        int64_t trustedInflight = 0;
        double minimumTrustedBacklogLatencyMs =
            std::numeric_limits<double>::infinity();
        size_t minimumTrustedBacklogNodeIndex = std::numeric_limits<size_t>::max();
        size_t minimumTrustedBacklogNodeCount = 0;
        size_t minimumTrustedBacklogRecoveryRequestCount = 0;
        size_t healthyBacklogRecoveryRequestCount = 0;
        size_t trustedUnknownBusyNodeCount = 0;
        double targetInflight = 0.0;
        size_t trustedNodeCount = 0;
        size_t untrustedNodeCount = 0;
        size_t untrustedCapacity = 0;
        size_t rampingNodeCount = 0;
        size_t unsafeWaitingMarginalNodeCount = 0;
        double minimumRampProgress = 1.0;
        double minimumRampedTargetInflight = 0.0;
    };

    struct Result {
        double targetInflight = 0.0;
        size_t trustedNodeCount = 0;
        size_t untrustedNodeCount = 0;
        size_t trustedRequestCount = 0;
        size_t untrustedRequestCount = 0;
        size_t rejectedRequestCount = 0;
        size_t untrustedCapacity = 0;
        size_t rampingNodeCount = 0;
        size_t unsafeWaitingMarginalNodeCount = 0;
        double minimumRampProgress = 1.0;
        double minimumRampedTargetInflight = 0.0;
        std::vector<NaviSchedColdStartRoutingStats::RandomDecision>
            randomDecisions;
        bool costScheduleRequired = false;
    };

    static bool apply(NaviSchedScheduleContext &context,
                      multi_call::RandomGenerator &randomGenerator,
                      Result &result,
                      const std::function<uint64_t()> &randomDraw = {});
    static Analysis analyze(const NaviSchedScheduleContext &context,
                            double healthyBacklogLatencyLimitMs = 0.0);
    static double calculateTargetInflight(int64_t trustedInflight,
                                          size_t trustedNodeCount);
    static size_t calculateUntrustedCapacity(long double totalPositiveDeficit,
                                             size_t requestCount);
    static double calculateRampProgress(int64_t scheduleTimeUs,
                                        int64_t agentStartTimeUs,
                                        uint64_t sessionSampleCount);
    static bool isRampFinished(int64_t scheduleTimeUs,
                               int64_t agentStartTimeUs,
                               uint64_t sessionSampleCount);
    static double calculateRampedTargetInflight(double targetInflight,
                                                double rampProgress);

private:
    static constexpr int64_t NO_TRUSTED_TARGET_INFLIGHT = 100;
    static constexpr int64_t COLD_START_RAMP_DURATION_US = 60'000'000;
    static constexpr double COLD_START_INITIAL_TARGET_INFLIGHT = 1.0;

    static bool selectUntrusted(size_t trustedNodeCount,
                                size_t untrustedNodeCount,
                                multi_call::RandomGenerator &randomGenerator,
                                uint64_t *sample = nullptr,
                                double *probability = nullptr,
                                const std::function<uint64_t()> &randomDraw = {});
};

}
