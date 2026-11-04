#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace navi_sched {

namespace {

struct TrustedInflightStats {
    int64_t inflight = 0;
    size_t nodeCount = 0;
};

struct UntrustedNodeDeficit {
    size_t nodeIndex = 0;
    double remaining = 0.0;
    bool ramping = false;
};

struct UntrustedNode {
    size_t nodeIndex = 0;
    bool ramping = false;
    double rampProgress = 1.0;
};

struct UntrustedNodeDeficitLess {
    bool operator()(const UntrustedNodeDeficit &left,
                    const UntrustedNodeDeficit &right) const
    {
        if (left.remaining != right.remaining) {
            return left.remaining < right.remaining;
        }
        return left.nodeIndex > right.nodeIndex;
    }
};

struct ColdStartRoutingPlan {
    CostSchedulerColdStartStrategy::Analysis analysis;
    std::vector<size_t> trustedNodes;
    std::vector<UntrustedNodeDeficit> untrustedNodes;
};

bool hasUnsafeWaitingRequestMarginal(
    const NaviSchedScheduleContext &context, size_t nodeIndex)
{
    const auto &node = context.getNode(nodeIndex);
    const auto &parameters = node.getLatencyParameters();
    const auto capacitySplit =
        NonLinearPrefillModelR::usesCapacitySplitParameters(parameters);
    if (!capacitySplit) {
        return false;
    }
    const auto waitingBegin =
        parameters.begin() + NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET;
    const auto waitingEnd =
        waitingBegin + NonLinearPrefillModelR::WAITING_PARAMETER_COUNT;
    if (std::all_of(waitingBegin, waitingEnd,
                    [](double coefficient) { return coefficient >= 0.0; })) {
        return false;
    }
    const auto waitingCountCoefficient =
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET] *
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET);
    const auto waitingHitCoefficient =
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 1] *
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 1);
    const auto waitingMissCoefficient =
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 2] *
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 2);
    const auto waitingMissSquareCoefficient =
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 3] *
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 3);
    const auto waitingHitMissCoefficient =
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 4] *
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 4);
    const auto maxRunningRequestCount =
        std::max(0, node.getMaxRunningRequestCount());
    const auto workloadCapacity = static_cast<size_t>(std::max(
        0, maxRunningRequestCount -
               node.getWorkloadRunningCapacitySlotCount()));
    const auto reservedCapacity = static_cast<size_t>(std::max(
        0, maxRunningRequestCount -
               node.getReservedRunningCapacitySlotCount()));
    const auto firstWaitingRequestIndex =
        std::min(workloadCapacity, reservedCapacity);
    if (context.requestCount() <= firstWaitingRequestIndex) {
        return false;
    }
    for (size_t requestIndex = firstWaitingRequestIndex;
         requestIndex < context.requestCount();
         ++requestIndex) {
        const auto &request = context.getRequest(requestIndex).request;
        constexpr double TOKEN_SCALE = 1024.0;
        const auto cacheHitTokenCount = static_cast<double>(
            context.getCacheHitBlockCount(requestIndex, nodeIndex) *
            request.blockSize);
        const auto cacheMissTokenCount =
            static_cast<double>(request.promptTokenCount) - cacheHitTokenCount;
        const auto waitingCost =
            waitingCountCoefficient +
            waitingHitCoefficient * cacheHitTokenCount / TOKEN_SCALE +
            waitingMissCoefficient * cacheMissTokenCount / TOKEN_SCALE +
            waitingMissSquareCoefficient * cacheMissTokenCount *
                cacheMissTokenCount / (TOKEN_SCALE * TOKEN_SCALE) +
            waitingHitMissCoefficient * cacheHitTokenCount *
                cacheMissTokenCount / (TOKEN_SCALE * TOKEN_SCALE);
        if (!std::isfinite(waitingCost) || waitingCost < -1e-9) {
            return true;
        }
    }
    return false;
}

ColdStartRoutingPlan buildRoutingPlan(
    const NaviSchedScheduleContext &context,
    double healthyBacklogLatencyLimitMs = 0.0)
{
    ColdStartRoutingPlan plan;
    std::vector<UntrustedNode> untrustedNodes;
    std::unordered_map<std::string, TrustedInflightStats> trustedInflightByFingerprint;
    for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
        const auto &node = context.getNode(nodeIndex);
        const auto inflight = std::max<int64_t>(0, node.getInflightRequests());
        const auto rampProgress =
            CostSchedulerColdStartStrategy::calculateRampProgress(
                context.getScheduleTimeUs(), node.getAgentStartTimeUs(),
                node.getLatencySessionSampleCount());
        const auto ramping = !CostSchedulerColdStartStrategy::isRampFinished(
            context.getScheduleTimeUs(), node.getAgentStartTimeUs(),
            node.getLatencySessionSampleCount());
        const auto unsafeWaitingMarginal =
            hasUnsafeWaitingRequestMarginal(context, nodeIndex);
        if (unsafeWaitingMarginal) {
            ++plan.analysis.unsafeWaitingMarginalNodeCount;
        }
        const auto backlogLatencyMs = node.getEstimatedBacklogLatencyMs();
        // Batch counters can be zero while decode still owns execution slots.
        const auto hasUnifiedWorkload =
            node.usesUnifiedLatencyPrediction() &&
            (node.getWorkloadFeatures().getPrecedingRequestCount() > 0 ||
             node.getWorkloadRunningCapacitySlotCount() > 0 ||
             node.getReservedRunningCapacitySlotCount() > 0);
        const auto unknownBusyBacklog =
            backlogLatencyMs <= 0.0 &&
            (inflight > 0 || node.getRunningRequests() > 0 ||
             node.getBatchWaitingRequests() > 0 ||
             node.getQueuedBatchRequests() > 0 || hasUnifiedWorkload);
        if (node.isLatencyModelTrusted() && !ramping &&
            !unsafeWaitingMarginal) {
            plan.trustedNodes.emplace_back(nodeIndex);
            plan.analysis.trustedInflight += inflight;
            if (!unknownBusyBacklog) {
                // Retain the legacy probe even when capacity is full.
                // Reports and reservations overlap: use the largest occupancy,
                // while execution slots and unreported reserved slots are additive.
                const auto occupied = std::max({
                    inflight,
                    static_cast<int64_t>(node.getRunningRequests()) +
                        node.getBatchWaitingRequests() + node.getQueuedBatchRequests(),
                    node.getWorkloadFeatures().getPrecedingRequestCount(),
                    static_cast<int64_t>(node.getWorkloadRunningCapacitySlotCount()) +
                        node.getReservedRunningCapacitySlotCount()});
                const auto capacity = node.getMaxRunningRequestCount();
                const auto available = capacity > 0
                    ? static_cast<size_t>(std::max<int64_t>(1, capacity - occupied))
                    : size_t(1);
                if (capacity > occupied &&
                    backlogLatencyMs <= healthyBacklogLatencyLimitMs) {
                    auto &recovery = plan.analysis.healthyBacklogRecoveryRequestCount;
                    recovery += std::min(context.requestCount() - recovery, available);
                }
                if (backlogLatencyMs < plan.analysis.minimumTrustedBacklogLatencyMs) {
                    plan.analysis.minimumTrustedBacklogLatencyMs = backlogLatencyMs;
                    plan.analysis.minimumTrustedBacklogNodeIndex = nodeIndex;
                    plan.analysis.minimumTrustedBacklogNodeCount = 1;
                    plan.analysis.minimumTrustedBacklogRecoveryRequestCount =
                        std::min(context.requestCount(), available);
                } else if (backlogLatencyMs == plan.analysis.minimumTrustedBacklogLatencyMs) {
                    ++plan.analysis.minimumTrustedBacklogNodeCount;
                    auto &availableCount =
                        plan.analysis.minimumTrustedBacklogRecoveryRequestCount;
                    availableCount += std::min(context.requestCount() - availableCount, available);
                }
                // Unified workload also owns decode requests after prefill finishes.
                const auto backlogRequestCount = node.usesUnifiedLatencyPrediction()
                    ? std::max(inflight, node.getWorkloadFeatures().getPrecedingRequestCount())
                    : inflight;
                plan.analysis.trustedBacklogInflight += backlogRequestCount;
                plan.analysis.trustedBacklogLatencyWeightedSumMs +=
                    static_cast<long double>(backlogRequestCount) * backlogLatencyMs;
            } else {
                ++plan.analysis.trustedUnknownBusyNodeCount;
            }
            const auto &fingerprint = node.getLatencyModelFingerprint();
            if (!fingerprint.empty()) {
                auto &stats = trustedInflightByFingerprint[fingerprint];
                stats.inflight += inflight;
                ++stats.nodeCount;
            }
        } else {
            untrustedNodes.push_back({nodeIndex, ramping, rampProgress});
            if (ramping) {
                ++plan.analysis.rampingNodeCount;
                plan.analysis.minimumRampProgress = std::min(
                    plan.analysis.minimumRampProgress, rampProgress);
            }
        }
    }
    plan.analysis.trustedNodeCount = plan.trustedNodes.size();
    plan.analysis.untrustedNodeCount = untrustedNodes.size();
    plan.analysis.targetInflight =
        CostSchedulerColdStartStrategy::calculateTargetInflight(
            plan.analysis.trustedInflight, plan.analysis.trustedNodeCount);

    long double totalUntrustedDeficit = 0.0L;
    plan.untrustedNodes.reserve(untrustedNodes.size());
    for (const auto &untrustedNode : untrustedNodes) {
        const auto nodeIndex = untrustedNode.nodeIndex;
        const auto &node = context.getNode(nodeIndex);
        auto targetInflight = plan.analysis.targetInflight;
        const auto &fingerprint = node.getLatencyModelFingerprint();
        const auto it = fingerprint.empty()
                            ? trustedInflightByFingerprint.end()
                            : trustedInflightByFingerprint.find(fingerprint);
        if (it != trustedInflightByFingerprint.end()) {
            targetInflight = CostSchedulerColdStartStrategy::calculateTargetInflight(
                it->second.inflight, it->second.nodeCount);
        }
        if (untrustedNode.ramping) {
            targetInflight =
                CostSchedulerColdStartStrategy::calculateRampedTargetInflight(
                    targetInflight, untrustedNode.rampProgress);
            if (plan.analysis.minimumRampedTargetInflight == 0.0) {
                plan.analysis.minimumRampedTargetInflight = targetInflight;
            } else {
                plan.analysis.minimumRampedTargetInflight = std::min(
                    plan.analysis.minimumRampedTargetInflight, targetInflight);
            }
        }
        const auto inflight = std::max<int64_t>(0, node.getInflightRequests());
        const auto deficit = targetInflight - inflight;
        if (deficit > 0.0) {
            plan.untrustedNodes.push_back(
                {nodeIndex, deficit, untrustedNode.ramping});
            totalUntrustedDeficit += deficit;
        }
    }
    plan.analysis.untrustedCapacity =
        CostSchedulerColdStartStrategy::calculateUntrustedCapacity(
            totalUntrustedDeficit, context.requestCount());
    return plan;
}

} // namespace

bool CostSchedulerColdStartStrategy::apply(
    NaviSchedScheduleContext &context,
    multi_call::RandomGenerator &randomGenerator, Result &result,
    const std::function<uint64_t()> &randomDraw)
{
    result = Result();
    if (!context.valid() || context._scheduleViewActive ||
        context.requestCount() == 0 || context.nodeCount() == 0) {
        return false;
    }

    auto routingPlan = buildRoutingPlan(context);
    result.targetInflight = routingPlan.analysis.targetInflight;
    result.trustedNodeCount = routingPlan.analysis.trustedNodeCount;
    result.untrustedNodeCount = routingPlan.analysis.untrustedNodeCount;
    result.untrustedCapacity = routingPlan.analysis.untrustedCapacity;
    result.rampingNodeCount = routingPlan.analysis.rampingNodeCount;
    result.unsafeWaitingMarginalNodeCount =
        routingPlan.analysis.unsafeWaitingMarginalNodeCount;
    result.minimumRampProgress = routingPlan.analysis.minimumRampProgress;
    result.minimumRampedTargetInflight =
        routingPlan.analysis.minimumRampedTargetInflight;

    std::priority_queue<UntrustedNodeDeficit, std::vector<UntrustedNodeDeficit>,
                        UntrustedNodeDeficitLess>
        untrustedQueue(UntrustedNodeDeficitLess(), std::move(routingPlan.untrustedNodes));
    std::vector<size_t> trustedRequests;
    trustedRequests.reserve(context.requestCount());
    size_t remainingUntrustedCapacity = result.untrustedCapacity;
    const auto bootstrapRequestCount =
        context._coldStartBootstrapRequestCount;
    context._coldStartBootstrapRequestCount = 0;
    for (size_t requestIndex = 0; requestIndex < context.requestCount();
         ++requestIndex) {
        const auto forceUntrusted = requestIndex < bootstrapRequestCount;
        bool routeToUntrusted = false;
        if (remainingUntrustedCapacity > 0 && !untrustedQueue.empty()) {
            if (forceUntrusted) {
                routeToUntrusted = true;
            } else if (context.isCollectingReplay()) {
                NaviSchedColdStartRoutingStats::RandomDecision randomDecision;
                randomDecision.requestIndex = requestIndex;
                routeToUntrusted = selectUntrusted(
                    result.trustedNodeCount, result.untrustedNodeCount,
                    randomGenerator,
                    &randomDecision.sample, &randomDecision.probability,
                    randomDraw);
                randomDecision.selected = routeToUntrusted;
                result.randomDecisions.emplace_back(randomDecision);
            } else {
                routeToUntrusted = selectUntrusted(
                    result.trustedNodeCount, result.untrustedNodeCount,
                    randomGenerator, nullptr, nullptr, randomDraw);
            }
        }
        if (routeToUntrusted) {
            auto target = untrustedQueue.top();
            untrustedQueue.pop();
            const auto trainingPhase =
                target.ramping ? NaviSchedLatencyTrainingPhase::RAMP
                               : NaviSchedLatencyTrainingPhase::NORMAL;
            if (!context.selectNode(requestIndex, target.nodeIndex,
                                    "cold_start_inflight_selected",
                                    trainingPhase)) {
                return false;
            }
            target.remaining -= 1.0;
            if (target.remaining > 0.0) {
                untrustedQueue.push(target);
            }
            --remainingUntrustedCapacity;
            ++result.untrustedRequestCount;
            continue;
        }
        if (!routingPlan.trustedNodes.empty()) {
            trustedRequests.emplace_back(requestIndex);
            continue;
        }
        if (!context.reject(requestIndex, "rate_limited")) {
            return false;
        }
        ++result.rejectedRequestCount;
    }

    result.trustedRequestCount = trustedRequests.size();
    result.costScheduleRequired = !trustedRequests.empty();
    const auto success =
        !result.costScheduleRequired ||
        context.activateScheduleView(trustedRequests, routingPlan.trustedNodes);
    if (success) {
        context._coldStartRoutingStats.targetInflight = result.targetInflight;
        context._coldStartRoutingStats.trustedNodeCount =
            result.trustedNodeCount;
        context._coldStartRoutingStats.untrustedNodeCount =
            result.untrustedNodeCount;
        context._coldStartRoutingStats.trustedRequestCount =
            result.trustedRequestCount;
        context._coldStartRoutingStats.untrustedRequestCount =
            result.untrustedRequestCount;
        context._coldStartRoutingStats.rejectedRequestCount =
            result.rejectedRequestCount;
        context._coldStartRoutingStats.untrustedCapacity =
            result.untrustedCapacity;
        context._coldStartRoutingStats.rampingNodeCount =
            result.rampingNodeCount;
        context._coldStartRoutingStats.unsafeWaitingMarginalNodeCount =
            result.unsafeWaitingMarginalNodeCount;
        context._coldStartRoutingStats.minimumRampProgress =
            result.minimumRampProgress;
        context._coldStartRoutingStats.minimumRampedTargetInflight =
            result.minimumRampedTargetInflight;
        if (context.isCollectingReplay()) {
            context._coldStartRoutingStats.randomDecisions =
                result.randomDecisions;
        }
        context._coldStartRoutingStats.evaluated = true;
    }
    return success;
}

CostSchedulerColdStartStrategy::Analysis CostSchedulerColdStartStrategy::analyze(
    const NaviSchedScheduleContext &context, double healthyBacklogLatencyLimitMs)
{
    return buildRoutingPlan(context, healthyBacklogLatencyLimitMs).analysis;
}

double CostSchedulerColdStartStrategy::calculateTargetInflight(
    int64_t trustedInflight, size_t trustedNodeCount)
{
    if (trustedNodeCount == 0) {
        return NO_TRUSTED_TARGET_INFLIGHT;
    }
    return static_cast<double>(std::max<int64_t>(0, trustedInflight)) /
           trustedNodeCount;
}

size_t CostSchedulerColdStartStrategy::calculateUntrustedCapacity(
    long double totalPositiveDeficit, size_t requestCount)
{
    if (!std::isfinite(totalPositiveDeficit) || totalPositiveDeficit < 1.0L ||
        requestCount == 0) {
        return 0;
    }
    return static_cast<size_t>(
        std::min<long double>(requestCount, std::floor(totalPositiveDeficit)));
}

double CostSchedulerColdStartStrategy::calculateRampProgress(
    int64_t scheduleTimeUs, int64_t agentStartTimeUs,
    uint64_t sessionSampleCount)
{
    (void)sessionSampleCount;
    if (agentStartTimeUs <= 0) {
        return 0.0;
    }
    const auto elapsedUs = std::max<int64_t>(0, scheduleTimeUs - agentStartTimeUs);
    return std::min(
        1.0, static_cast<double>(elapsedUs) / COLD_START_RAMP_DURATION_US);
}

bool CostSchedulerColdStartStrategy::isRampFinished(
    int64_t scheduleTimeUs, int64_t agentStartTimeUs,
    uint64_t sessionSampleCount)
{
    (void)sessionSampleCount;
    if (agentStartTimeUs <= 0) {
        return false;
    }
    const auto elapsedUs = std::max<int64_t>(0, scheduleTimeUs - agentStartTimeUs);
    return elapsedUs >= COLD_START_RAMP_DURATION_US;
}

double CostSchedulerColdStartStrategy::calculateRampedTargetInflight(
    double targetInflight, double rampProgress)
{
    if (rampProgress >= 1.0) {
        return std::max(0.0, targetInflight);
    }
    const auto boundedProgress = std::clamp(rampProgress, 0.0, 1.0);
    const auto finalTarget =
        std::max(COLD_START_INITIAL_TARGET_INFLIGHT, targetInflight);
    return COLD_START_INITIAL_TARGET_INFLIGHT +
           (finalTarget - COLD_START_INITIAL_TARGET_INFLIGHT) * boundedProgress;
}

bool CostSchedulerColdStartStrategy::selectUntrusted(
    size_t trustedNodeCount, size_t untrustedNodeCount,
    multi_call::RandomGenerator &randomGenerator, uint64_t *sample,
    double *probability, const std::function<uint64_t()> &randomDraw)
{
    if (sample) {
        *sample = 0;
    }
    if (probability) {
        *probability = 0.0;
    }
    if (untrustedNodeCount == 0) {
        return false;
    }
    if (trustedNodeCount == 0) {
        return true;
    }
    const auto selectedProbability =
        static_cast<double>(untrustedNodeCount) /
        (trustedNodeCount + untrustedNodeCount);
    const auto randomValue = randomDraw ? randomDraw() : randomGenerator.get();
    const auto randomSample = static_cast<double>(randomValue) /
                              std::numeric_limits<uint64_t>::max();
    if (sample) {
        *sample = randomValue;
    }
    if (probability) {
        *probability = selectedProbability;
    }
    return randomSample < selectedProbability;
}

} // namespace navi_sched
