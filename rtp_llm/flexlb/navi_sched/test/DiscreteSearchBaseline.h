#pragma once

// Frozen pre-optimization implementation, used only as the correctness/perf oracle.
// Source SHA256: 81d3843ba3d78d96c8080689197f98ebe145d2ea4d5c777756612af112730844
namespace navi_sched {
namespace discrete_baseline {
using DiscreteForwardWorkspace = CostSchedulerKernelInternal::DiscreteForwardWorkspace;
using DiscreteRefinementStats = CostSchedulerKernelInternal::DiscreteRefinementStats;
inline double pairLoss(const NaviSchedScheduleContext &context, const float *costs,
                       const uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
                       const double *inverseRequestTokenCounts,
                       double tokenNormalizationDenominator, double lambda,
                       double latencyWeightConcavity, DiscreteForwardWorkspace &workspace,
                       size_t firstNode, size_t secondNode)
{
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    if (!costs || !nodeIndexes || !requestTokenCounts || !inverseRequestTokenCounts ||
        requestCount == 0 || nodeCount == 0 || tokenNormalizationDenominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    workspace.requestCostSums.assign(nodeCount, 0.0);
    workspace.requestCostSumsWithoutReportedWorkload.assign(nodeCount, 0.0);
    workspace.totalPromptTokenSums.assign(nodeCount, 0.0);
    workspace.requestWeightSums.assign(nodeCount, 0.0);
    workspace.inverseTokenWeightSums.assign(nodeCount, 0.0);
    workspace.nodeValues.assign(nodeCount, 0.0);
    std::vector<std::vector<size_t>> geometryRequests(nodeCount);
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        auto nodeIndex = static_cast<size_t>(nodeIndexes[requestIndex]);
        if (nodeIndex >= nodeCount) {
            return std::numeric_limits<double>::infinity();
        }
        if (firstNode < nodeCount && nodeIndex != firstNode && nodeIndex != secondNode) {
            continue;
        }
        const auto &node = context.getNode(nodeIndex);
        if (NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            geometryRequests[nodeIndex].emplace_back(requestIndex);
        }
        const auto capacitySplit = NonLinearPrefillModelR::usesCapacitySplitParameters(
            node.getLatencyParameters());
        const auto costIndex = nodeIndex * requestCount + requestIndex;
        const auto waitingCostIndex = nodeCount * requestCount + costIndex;
        const auto assignedRequestCount = workspace.requestWeightSums[nodeIndex];
        const auto usesRunningCapacity =
            !capacitySplit ||
            node.getWorkloadRunningCapacitySlotCount() + assignedRequestCount <
                node.getMaxRunningRequestCount();
        const auto usesReservedRunningCapacity =
            !capacitySplit ||
            node.getReservedRunningCapacitySlotCount() + assignedRequestCount <
                node.getMaxRunningRequestCount();
        if (!NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            workspace.requestCostSums[nodeIndex] +=
                costs[usesRunningCapacity ? costIndex : waitingCostIndex];
            workspace.requestCostSumsWithoutReportedWorkload[nodeIndex] +=
                costs[usesReservedRunningCapacity ? costIndex : waitingCostIndex];
        }
        workspace.totalPromptTokenSums[nodeIndex] += requestTokenCounts[requestIndex];
        workspace.requestWeightSums[nodeIndex] += 1.0;
        workspace.inverseTokenWeightSums[nodeIndex] +=
            inverseRequestTokenCounts[requestIndex];
    }
    auto firstScale = lambda / tokenNormalizationDenominator;
    auto secondScale = (1.0 - lambda) / static_cast<double>(requestCount);
    auto latencyWeightLogScale =
        latencyWeightConcavity > 0.0 ? std::log(latencyWeightConcavity) : 1.0;
    double result = 0.0;
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto requestWeightSum = workspace.requestWeightSums[nodeIndex];
        double unusedDerivative = 0.0;
        double unusedPromptDerivative = 0.0;
        const auto &node = context.getNode(nodeIndex);
        if (requestWeightSum == 0.0) {
            continue;
        }
        double nodeCost = 0.0;
        double existingNodeCost = 0.0;
        if (NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            nodeCost = context.estimateBatchExecutionLatencyMs(
                nodeIndex, geometryRequests[nodeIndex]);
            if (!std::isfinite(nodeCost)) {
                return std::numeric_limits<double>::infinity();
            }
        } else if (node.usesUnifiedLatencyPrediction()) {
            const auto capacitySplit =
                NonLinearPrefillModelR::usesCapacitySplitParameters(
                    node.getLatencyParameters());
            NaviSchedLatencyFeatures floorWorkload;
            if (!node.getUnifiedDecayFloorFeatures(floorWorkload)) {
                return std::numeric_limits<double>::infinity();
            }
            const auto requestCostWithoutReportedWorkload =
                (capacitySplit
                     ? workspace.requestCostSumsWithoutReportedWorkload[nodeIndex]
                     : workspace.requestCostSums[nodeIndex]) +
                NonLinearPrefillModelR::calculateLinearCost(node.getLatencyParameters(),
                                                            floorWorkload);
            const auto promptTokensWithoutReportedWorkload =
                workspace.totalPromptTokenSums[nodeIndex] +
                floorWorkload.getCacheHitTokenCount() +
                floorWorkload.getCacheMissTokenCount();
            auto requestCostWithReportedWorkload =
                capacitySplit
                    ? workspace.requestCostSums[nodeIndex] +
                          NonLinearPrefillModelR::calculateLinearCost(
                              node.getLatencyParameters(), node.getWorkloadFeatures())
                    : requestCostWithoutReportedWorkload +
                          NonLinearPrefillModelR::calculateLinearCost(
                              node.getLatencyParameters(),
                              node.getReportedWorkloadFeatures());
            const auto promptTokensWithReportedWorkload =
                promptTokensWithoutReportedWorkload +
                node.getReportedWorkloadFeatures().getCacheHitTokenCount() +
                node.getReportedWorkloadFeatures().getCacheMissTokenCount();
            const auto precedingRequestsWithoutReportedWorkload =
                floorWorkload.getPrecedingRequestCount();
            const auto precedingRequestsWithReportedWorkload =
                node.getWorkloadFeatures().getPrecedingRequestCount();
            auto costWithReportedWorkload =
                NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(), requestCostWithReportedWorkload,
                    promptTokensWithReportedWorkload,
                    precedingRequestsWithReportedWorkload, unusedDerivative,
                    unusedPromptDerivative);
            costWithReportedWorkload =
                node.applyLatencyResidual(costWithReportedWorkload);
            auto floorCost =
                NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(), requestCostWithoutReportedWorkload,
                    promptTokensWithoutReportedWorkload,
                    precedingRequestsWithoutReportedWorkload, unusedDerivative,
                    unusedPromptDerivative);
            floorCost = node.applyLatencyResidual(floorCost);
            nodeCost = node.decayUnifiedLatency(costWithReportedWorkload, floorCost);
            existingNodeCost = node.getExistingUnifiedLatencyMs();
        } else {
            nodeCost =
                NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(), workspace.requestCostSums[nodeIndex],
                    workspace.totalPromptTokenSums[nodeIndex], 0.0, unusedDerivative,
                    unusedPromptDerivative);
            nodeCost = node.applyLatencyResidual(nodeCost);
        }
        auto inverseTokenWeightSum = workspace.inverseTokenWeightSums[nodeIndex];
        auto latencyWeight =
            latencyWeightConcavity > 0.0
                ? std::log1p(latencyWeightConcavity * inverseTokenWeightSum) /
                      latencyWeightLogScale
                : inverseTokenWeightSum;
        const auto firstTerm = node.usesUnifiedLatencyPrediction()
                                   ? nodeCost - existingNodeCost
                                   : nodeCost * requestWeightSum;
        const auto nodeValue =
            firstScale * firstTerm +
            secondScale *
                (nodeCost + context.getNode(nodeIndex).getEstimatedQueueWaitMs()) *
                latencyWeight;
        workspace.nodeValues[nodeIndex] = nodeValue;
        result += nodeValue;
    }
    return result;
}

inline bool refine(const NaviSchedScheduleContext &context, const float *costs,
                   uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
                   const double *inverseRequestTokenCounts,
                   double tokenNormalizationDenominator, double lambda,
                   double latencyWeightConcavity, double &value, int64_t timeBudgetUs,
                   size_t maxEvaluations, DiscreteRefinementStats &stats)
{
    stats = {};
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    DiscreteForwardWorkspace workspace;
    value = CostSchedulerKernelInternal::forwardDiscrete(
        context, costs, nodeIndexes, requestTokenCounts, inverseRequestTokenCounts,
        tokenNormalizationDenominator, lambda, latencyWeightConcavity, workspace);
    if (!std::isfinite(value)) {
        return false;
    }
    const auto requestCount = context.requestCount();
    const auto nodeCount = context.nodeCount();
    if (nodeCount < 2 || maxEvaluations == 0 || timeBudgetUs == 0) {
        return true;
    }
    const auto originalValue = value;
    const std::vector<uint32_t> original(nodeIndexes, nodeIndexes + requestCount);
    auto nodeValues = workspace.nodeValues;
    const auto exhausted = [&]() {
        return stats.evaluations >= maxEvaluations ||
               (timeBudgetUs > 0 &&
                autil::TimeUtility::monotonicTimeUs() - beginUs >= timeBudgetUs);
    };
    while (!exhausted()) {
        auto bestRequest = requestCount;
        uint32_t bestTarget = 0;
        auto bestDelta = 0.0;
        double bestSourceValue = 0.0;
        double bestTargetValue = 0.0;
        const auto tolerance = 1e-12 * std::max(1.0, std::abs(value));
        for (size_t requestIndex = 0; requestIndex < requestCount && !exhausted();
             ++requestIndex) {
            const auto source = nodeIndexes[requestIndex];
            for (size_t target = 0; target < nodeCount && !exhausted(); ++target) {
                if (target == source) {
                    continue;
                }
                nodeIndexes[requestIndex] = static_cast<uint32_t>(target);
                // Recompute both ordered queues: removing a request can promote
                // another waiting request into a running slot, and vice versa.
                const auto pairValue =
                    pairLoss(context, costs, nodeIndexes, requestTokenCounts,
                             inverseRequestTokenCounts, tokenNormalizationDenominator,
                             lambda, latencyWeightConcavity, workspace, source, target);
                nodeIndexes[requestIndex] = source;
                ++stats.evaluations;
                const auto delta = pairValue - nodeValues[source] - nodeValues[target];
                if (std::isfinite(delta) && delta < bestDelta - tolerance) {
                    bestDelta = delta;
                    bestRequest = requestIndex;
                    bestTarget = static_cast<uint32_t>(target);
                    bestSourceValue = workspace.nodeValues[source];
                    bestTargetValue = workspace.nodeValues[target];
                }
            }
        }
        if (bestRequest == requestCount) {
            break;
        }
        const auto source = nodeIndexes[bestRequest];
        nodeIndexes[bestRequest] = bestTarget;
        nodeValues[source] = bestSourceValue;
        nodeValues[bestTarget] = bestTargetValue;
        value += bestDelta;
        ++stats.moves;
    }
    stats.budgetExhausted = exhausted();
    value = CostSchedulerKernelInternal::forwardDiscrete(
        context, costs, nodeIndexes, requestTokenCounts, inverseRequestTokenCounts,
        tokenNormalizationDenominator, lambda, latencyWeightConcavity, workspace);
    if (!std::isfinite(value) || value > originalValue) {
        std::copy(original.begin(), original.end(), nodeIndexes);
        value = originalValue;
        stats.moves = 0;
    }
    context.recordDiscreteSearch(autil::TimeUtility::monotonicTimeUs() - beginUs,
                                 stats.evaluations, stats.moves);
    return true;
}

} // namespace discrete_baseline
} // namespace navi_sched
