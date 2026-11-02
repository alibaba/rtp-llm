#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/CostSchedulerConfig.h"
#include "autil/TimeUtility.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

namespace navi_sched {
namespace {
constexpr double SELECTION_JITTER_RATIO = 1.0 / 1000.0;

double aggregateCapacitySplitCost(const float *runningCosts,
                                  const float *waitingCosts,
                                  const double *weights,
                                  size_t requestCount,
                                  double runningCapacity)
{
    double cost = 0.0;
    double runningWeight = 0.0;
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        const auto weight = weights[requestIndex];
        const auto requestRunningWeight = std::min(
            weight, std::max(0.0, runningCapacity - runningWeight));
        cost += requestRunningWeight * runningCosts[requestIndex] +
                (weight - requestRunningWeight) * waitingCosts[requestIndex];
        runningWeight += requestRunningWeight;
    }
    return cost;
}

void calculateCapacitySplitMarginalCosts(const float *runningCosts,
                                         const float *waitingCosts,
                                         const double *weights,
                                         size_t requestCount,
                                         double runningCapacity,
                                         double *marginalCosts)
{
    if (runningCapacity <= 0.0) {
        std::copy(waitingCosts, waitingCosts + requestCount, marginalCosts);
        return;
    }
    double totalWeight = 0.0;
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        totalWeight += weights[requestIndex];
    }
    if (totalWeight <= runningCapacity) {
        std::copy(runningCosts, runningCosts + requestCount, marginalCosts);
        return;
    }

    double prefixWeight = 0.0;
    size_t boundaryIndex = requestCount;
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        const auto nextPrefixWeight = prefixWeight + weights[requestIndex];
        if (nextPrefixWeight >= runningCapacity) {
            boundaryIndex = requestIndex;
            break;
        }
        prefixWeight = nextPrefixWeight;
    }
    if (boundaryIndex == requestCount) {
        std::copy(runningCosts, runningCosts + requestCount, marginalCosts);
        return;
    }
    const auto boundaryDelta =
        runningCosts[boundaryIndex] - waitingCosts[boundaryIndex];
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        if (requestIndex < boundaryIndex) {
            marginalCosts[requestIndex] =
                runningCosts[requestIndex] - boundaryDelta;
        } else {
            marginalCosts[requestIndex] = waitingCosts[requestIndex];
        }
    }
}

uint64_t mixSelectionJitter(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double getSelectionJitter(uint64_t randomSeed, size_t requestIndex,
                          size_t nodeIndex,
                          size_t nodeCount)
{
    auto value = randomSeed ^
        (static_cast<uint64_t>(requestIndex) << 32) ^
        static_cast<uint64_t>(nodeIndex);
    constexpr double UINT53_SCALE = 1.0 / 9007199254740992.0;
    auto unit = static_cast<double>(mixSelectionJitter(value) >> 11) *
        UINT53_SCALE;
    return unit * SELECTION_JITTER_RATIO /
        static_cast<double>(nodeCount);
}

void updateSelectionMaximum(
    size_t nodeCount, size_t requestIndex, size_t nodeIndex,
    double projectedWeight, uint64_t randomSeed,
    double &maxWeight, size_t &maxIndex)
{
    if (!std::isfinite(maxWeight)) {
        maxWeight = projectedWeight + getSelectionJitter(
            randomSeed, requestIndex, nodeIndex, nodeCount);
        maxIndex = nodeIndex;
        return;
    }
    auto jitterLimit =
        SELECTION_JITTER_RATIO / static_cast<double>(nodeCount);
    bool jitterApplied = false;
    if (std::abs(projectedWeight - maxWeight) <= jitterLimit) {
        projectedWeight += getSelectionJitter(
            randomSeed, requestIndex, nodeIndex, nodeCount);
        jitterApplied = true;
    }
    if (projectedWeight > maxWeight) {
        maxWeight = projectedWeight +
            (!jitterApplied
                 ? getSelectionJitter(
                       randomSeed, requestIndex, nodeIndex, nodeCount)
                 : 0.0);
        maxIndex = nodeIndex;
    }
}

}

void CostSchedulerKernelInternal::normalizeWeights(
    double *weights, size_t requestCount, size_t nodeCount,
    const std::vector<double> &inverseWeightSums)
{
    std::vector<double> normalizedWeightSums(requestCount, 0.0);
    for (size_t nodeIndex = 0; nodeIndex + 1 < nodeCount; ++nodeIndex) {
        auto weightRow = weights + nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto normalizedWeight =
                weightRow[requestIndex] *
                inverseWeightSums[requestIndex];
            weightRow[requestIndex] = normalizedWeight;
            normalizedWeightSums[requestIndex] += normalizedWeight;
        }
    }
    auto lastWeightRow = weights + (nodeCount - 1) * requestCount;
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        lastWeightRow[requestIndex] =
            1.0 - normalizedWeightSums[requestIndex];
    }
}

bool CostSchedulerKernelInternal::isValidNode(
    const NaviSchedScheduleNode &node)
{
    return node.hasValidLatencyModel();
}

void CostSchedulerKernelInternal::initializeWeights(
    double *weights, size_t requestCount, size_t nodeCount,
    uint64_t randomSeed)
{
    std::mt19937_64 random(randomSeed > 0
        ? randomSeed
        : static_cast<uint64_t>(autil::TimeUtility::currentTime()));
    std::uniform_real_distribution<double> distribution(1.0, 2.0);
    std::vector<double> randomWeightSums(requestCount, 0.0);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto isLastNode = nodeIndex + 1 == nodeCount;
        auto weightRow = weights + nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto &weight = weightRow[requestIndex];
            weight = distribution(random);
            randomWeightSums[requestIndex] += weight;
            if (isLastNode) {
                randomWeightSums[requestIndex] =
                    1.0 / randomWeightSums[requestIndex];
            }
        }
    }
    normalizeWeights(
        weights, requestCount, nodeCount, randomWeightSums);
}

void CostSchedulerKernelInternal::initializeWeights(
    double *weights, const NaviSchedScheduleContext &context,
    const double *randomWeights, size_t randomWeightCount)
{
    if (!randomWeights || 0 == randomWeightCount ||
        0 != (randomWeightCount & (randomWeightCount - 1))) {
        initializeWeights(
            weights, context.requestCount(), context.nodeCount());
        return;
    }
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    auto randomWeightMask = randomWeightCount - 1;
    std::vector<double> randomWeightSums(requestCount, 0.0);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto isLastNode = nodeIndex + 1 == nodeCount;
        auto weightRow = weights + nodeIndex * requestCount;
        auto randomOffset = nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto weight = randomWeights[
                (randomOffset + requestIndex) & randomWeightMask];
            weightRow[requestIndex] = weight;
            randomWeightSums[requestIndex] += weight;
            if (isLastNode) {
                randomWeightSums[requestIndex] =
                    1.0 / randomWeightSums[requestIndex];
            }
        }
    }
    normalizeWeights(
        weights, requestCount, nodeCount, randomWeightSums);
}

bool CostSchedulerKernelInternal::initializeWeightRequestRange(
    double *weights, const NaviSchedScheduleContext &context,
    const double *randomWeights, size_t randomWeightCount,
    size_t requestBegin, size_t requestEnd)
{
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    if (!weights || !randomWeights || 0 == randomWeightCount ||
        0 != (randomWeightCount & (randomWeightCount - 1)) ||
        requestBegin > requestEnd || requestEnd > requestCount ||
        0 == nodeCount) {
        return false;
    }
    auto randomWeightMask = randomWeightCount - 1;
    for (size_t requestIndex = requestBegin;
         requestIndex < requestEnd; ++requestIndex) {
        double randomWeightSum = 0.0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            randomWeightSum += randomWeights[
                (nodeIndex * requestCount + requestIndex) &
                randomWeightMask];
        }
        double normalizedWeightSum = 0.0;
        randomWeightSum = 1.0 / randomWeightSum;
        for (size_t nodeIndex = 0; nodeIndex + 1 < nodeCount;
             ++nodeIndex) {
            auto index = nodeIndex * requestCount + requestIndex;
            auto normalizedWeight = randomWeights[
                index & randomWeightMask] * randomWeightSum;
            weights[index] = normalizedWeight;
            normalizedWeightSum += normalizedWeight;
        }
        weights[(nodeCount - 1) * requestCount + requestIndex] =
            1.0 - normalizedWeightSum;
    }
    return true;
}

bool CostSchedulerKernelInternal::projectWeights(
    double *weights, size_t requestCount, size_t nodeCount)
{
    if (!weights || 0 == requestCount || 0 == nodeCount) {
        return false;
    }
    auto weightCount = requestCount * nodeCount;
    for (size_t index = 0; index < weightCount; ++index) {
        if (!std::isfinite(weights[index])) {
            return false;
        }
    }
    auto tileSize = std::min(
        requestCount, PROJECTION_REQUEST_TILE_SIZE);
    std::vector<double> buffer(nodeCount * tileSize);
    return projectRequestRange(
        weights, requestCount, nodeCount, 0, requestCount, buffer.data());
}

bool CostSchedulerKernelInternal::projectRequestRange(
    double *weights, size_t requestCount, size_t nodeCount,
    size_t requestBegin, size_t requestEnd, double *workspace,
    uint32_t *nodeIndexes, const GradientSpan *gradientSpans,
    size_t gradientSpanCount, double alpha,
    uint64_t selectionRandomSeed)
{
    if (!weights || !workspace || 0 == requestCount || 0 == nodeCount ||
        requestBegin >= requestEnd || requestEnd > requestCount ||
        (nodeIndexes &&
         nodeCount > std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    auto applyGradients = gradientSpans && gradientSpanCount > 0;
    if (applyGradients && (!std::isfinite(alpha) || alpha <= 0.0)) {
        return false;
    }
    if (applyGradients) {
        size_t visitedNodeCount = 0;
        for (size_t spanIndex = 0; spanIndex < gradientSpanCount;
             ++spanIndex) {
            const auto &span = gradientSpans[spanIndex];
            if ((span.nodeBegin != span.nodeEnd && !span.values) ||
                span.nodeBegin != visitedNodeCount ||
                span.nodeEnd < span.nodeBegin ||
                span.nodeEnd > nodeCount) {
                return false;
            }
            visitedNodeCount = span.nodeEnd;
        }
        if (visitedNodeCount != nodeCount) {
            return false;
        }
    }
    for (; requestBegin < requestEnd;
         requestBegin += PROJECTION_REQUEST_TILE_SIZE) {
        auto tileSize = std::min(PROJECTION_REQUEST_TILE_SIZE,
                                 requestEnd - requestBegin);
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> activeSums{};
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> maxValues;
        std::array<size_t, PROJECTION_REQUEST_TILE_SIZE> maxIndexes{};
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> selectionMaxValues;
        std::array<size_t, PROJECTION_REQUEST_TILE_SIZE> selectionMaxIndexes{};
        maxValues.fill(-std::numeric_limits<double>::infinity());
        selectionMaxValues.fill(-std::numeric_limits<double>::infinity());
        if (applyGradients) {
            for (size_t spanIndex = 0; spanIndex < gradientSpanCount;
                 ++spanIndex) {
                const auto &span = gradientSpans[spanIndex];
                for (size_t nodeIndex = span.nodeBegin;
                     nodeIndex < span.nodeEnd; ++nodeIndex) {
                    auto weightRow = weights + nodeIndex * requestCount;
                    auto gradientRow = span.values +
                        (nodeIndex - span.nodeBegin) * requestCount;
                    for (size_t requestOffset = 0;
                         requestOffset < tileSize; ++requestOffset) {
                        auto requestIndex = requestBegin + requestOffset;
                        auto updatedWeight = weightRow[requestIndex] -
                            alpha * gradientRow[requestIndex];
                        workspace[requestOffset * nodeCount + nodeIndex] =
                            updatedWeight;
                        activeSums[requestOffset] += updatedWeight;
                        if (updatedWeight > maxValues[requestOffset]) {
                            maxValues[requestOffset] = updatedWeight;
                            maxIndexes[requestOffset] = nodeIndex;
                        }
                    }
                }
            }
        } else {
            for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
                auto weightRow = weights + nodeIndex * requestCount;
                for (size_t requestOffset = 0;
                     requestOffset < tileSize; ++requestOffset) {
                    auto value =
                        weightRow[requestBegin + requestOffset];
                    workspace[requestOffset * nodeCount + nodeIndex] =
                        value;
                    activeSums[requestOffset] += value;
                    if (value > maxValues[requestOffset]) {
                        maxValues[requestOffset] = value;
                        maxIndexes[requestOffset] = nodeIndex;
                    }
                }
            }
        }
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> thresholds{};
        for (size_t requestOffset = 0;
             requestOffset < tileSize; ++requestOffset) {
            if (!findSimplexThreshold(
                    workspace + requestOffset * nodeCount,
                    nodeCount, activeSums[requestOffset],
                    thresholds[requestOffset])) {
                return false;
            }
        }
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> projectedSums{};
        if (applyGradients) {
            for (size_t spanIndex = 0; spanIndex < gradientSpanCount;
                 ++spanIndex) {
                const auto &span = gradientSpans[spanIndex];
                for (size_t nodeIndex = span.nodeBegin;
                     nodeIndex < span.nodeEnd; ++nodeIndex) {
                    auto weightRow = weights + nodeIndex * requestCount;
                    auto gradientRow = span.values +
                        (nodeIndex - span.nodeBegin) * requestCount;
                    for (size_t requestOffset = 0;
                         requestOffset < tileSize; ++requestOffset) {
                        auto requestIndex = requestBegin + requestOffset;
                        auto projectedWeight = std::max(
                            0.0, weightRow[requestIndex] -
                                     alpha * gradientRow[requestIndex] -
                                     thresholds[requestOffset]);
                        weightRow[requestIndex] = projectedWeight;
                        projectedSums[requestOffset] += projectedWeight;
                        updateSelectionMaximum(
                            nodeCount, requestIndex, nodeIndex,
                            projectedWeight,
                            selectionRandomSeed,
                            selectionMaxValues[requestOffset],
                            selectionMaxIndexes[requestOffset]);
                    }
                }
            }
        } else {
            for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
                auto weightRow = weights + nodeIndex * requestCount;
                for (size_t requestOffset = 0;
                     requestOffset < tileSize; ++requestOffset) {
                    auto requestIndex = requestBegin + requestOffset;
                    auto projectedWeight = std::max(
                        0.0, weightRow[requestIndex] -
                                 thresholds[requestOffset]);
                    weightRow[requestIndex] = projectedWeight;
                    projectedSums[requestOffset] += projectedWeight;
                    updateSelectionMaximum(
                        nodeCount, requestIndex, nodeIndex,
                        projectedWeight, selectionRandomSeed,
                        selectionMaxValues[requestOffset],
                        selectionMaxIndexes[requestOffset]);
                }
            }
        }
        for (size_t requestOffset = 0;
             requestOffset < tileSize; ++requestOffset) {
            weights[maxIndexes[requestOffset] * requestCount +
                    requestBegin + requestOffset] +=
                1.0 - projectedSums[requestOffset];
            if (nodeIndexes) {
                nodeIndexes[requestBegin + requestOffset] =
                    static_cast<uint32_t>(
                        selectionMaxIndexes[requestOffset]);
            }
        }
    }
    return true;
}

bool CostSchedulerKernelInternal::selectWeightRequestRange(
    const double *weights, size_t requestCount, size_t nodeCount,
    size_t requestBegin, size_t requestEnd, uint32_t *nodeIndexes,
    uint64_t selectionRandomSeed)
{
    if (!weights || !nodeIndexes || nodeCount == 0 ||
        nodeCount > std::numeric_limits<uint32_t>::max() ||
        requestBegin > requestEnd || requestEnd > requestCount) {
        return false;
    }
    for (size_t requestIndex = requestBegin; requestIndex < requestEnd;
         ++requestIndex) {
        double maxWeight = -std::numeric_limits<double>::infinity();
        size_t maxIndex = 0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            updateSelectionMaximum(
                nodeCount, requestIndex, nodeIndex,
                weights[nodeIndex * requestCount + requestIndex],
                selectionRandomSeed, maxWeight, maxIndex);
        }
        nodeIndexes[requestIndex] = static_cast<uint32_t>(maxIndex);
    }
    return true;
}

bool CostSchedulerKernelInternal::findSimplexThreshold(
    double *workspace, size_t count, double activeSum,
    double &threshold)
{
    if (!workspace || 0 == count || !std::isfinite(activeSum)) {
        return false;
    }
    auto activeCount = count;
    while (activeCount > 0) {
        threshold =
            (activeSum - 1.0) / static_cast<double>(activeCount);
        double nextSum = 0.0;
        size_t nextCount = 0;
        for (size_t index = 0; index < activeCount; ++index) {
            auto value = workspace[index];
            if (value > threshold) {
                workspace[nextCount] = value;
                nextSum += value;
                ++nextCount;
            }
        }
        if (nextCount == activeCount) {
            return true;
        }
        activeSum = nextSum;
        activeCount = nextCount;
    }
    return false;
}

double CostSchedulerKernelInternal::getStepAlpha(
    double alpha, double alphaDecay, double minAlpha,
    size_t iteration)
{
    auto stepAlpha = std::max(
        minAlpha, alpha * std::pow(alphaDecay, iteration));
    return stepAlpha > 0.0
               ? stepAlpha
               : std::numeric_limits<double>::min();
}

bool CostSchedulerKernelInternal::applyGradient(
    double *weights, size_t weightCount,
    const std::vector<double> &gradients, double alpha)
{
    return applyGradient(
        weights, 0, weightCount, gradients, alpha);
}

bool CostSchedulerKernelInternal::applyGradient(
    double *weights, size_t weightOffset, size_t weightCount,
    const std::vector<double> &gradients, double alpha)
{
    if (!weights || gradients.size() != weightCount ||
        !std::isfinite(alpha) || alpha <= 0.0) {
        return false;
    }
    return applyGradient(weights, weightOffset, weightCount,
                         gradients.data(), alpha);
}

bool CostSchedulerKernelInternal::applyGradient(
    double *weights, size_t weightOffset, size_t weightCount,
    const double *gradients, double alpha)
{
    if (!weights || !gradients || !std::isfinite(alpha) ||
        alpha <= 0.0) {
        return false;
    }
    for (size_t index = 0; index < weightCount; ++index) {
        auto updatedWeight =
            weights[weightOffset + index] - alpha * gradients[index];
        if (!std::isfinite(updatedWeight)) {
            return false;
        }
        weights[weightOffset + index] = updatedWeight;
    }
    return true;
}

bool CostSchedulerKernelInternal::selectNodes(
    NaviSchedScheduleContext &context, const uint32_t *nodeIndexes)
{
    if (!nodeIndexes) {
        return false;
    }
    for (size_t requestIndex = 0;
         requestIndex < context.requestCount(); ++requestIndex) {
        if (!context.selectNode(requestIndex, nodeIndexes[requestIndex],
                                "cost_selected")) {
            return false;
        }
    }
    return true;
}

int64_t CostSchedulerKernelInternal::fillRequestTokenCounts(
    const NaviSchedScheduleContext &context, int64_t *requestTokenCounts,
    double *inverseRequestTokenCounts,
    double latencyTokenNormalizationPower)
{
    int64_t totalTokenCount = 0;
    for (size_t requestIndex = 0; requestIndex < context.requestCount(); ++requestIndex) {
        auto tokenCount = std::max<int64_t>(
            1, context.getRequest(requestIndex).request.promptTokenCount);
        requestTokenCounts[requestIndex] = tokenCount;
        if (inverseRequestTokenCounts) {
            inverseRequestTokenCounts[requestIndex] =
                1.0 / std::pow(static_cast<double>(tokenCount),
                               latencyTokenNormalizationPower);
        }
        totalTokenCount += tokenCount;
    }
    return totalTokenCount;
}

double CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(
    const int64_t *requestTokenCounts, size_t requestCount,
    double latencyTokenNormalizationPower)
{
    if (!requestTokenCounts || requestCount == 0 ||
        !std::isfinite(latencyTokenNormalizationPower)) {
        return 0.0;
    }
    double denominator = 0.0;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        denominator += std::pow(
            static_cast<double>(std::max<int64_t>(
                1, requestTokenCounts[requestIndex])),
            latencyTokenNormalizationPower);
    }
    return std::isfinite(denominator) && denominator > 0.0
               ? denominator
               : 0.0;
}

std::unique_ptr<float[]>
CostSchedulerKernelInternal::prepare(
    NaviSchedScheduleContext &context, const int64_t *requestTokenCounts,
    double lambda, double alpha, double alphaDecay, double minAlpha,
    size_t maxLoopCount, int64_t optimizationTimeBudgetUs,
    ReplayInfo *replayInfo)
{
    auto nodeCount = context.nodeCount();
    auto requestCount = context.requestCount();
    if (nodeCount == 0 || requestCount == 0 ||
        requestCount > std::numeric_limits<size_t>::max() / nodeCount ||
        requestCount * nodeCount >
            std::numeric_limits<size_t>::max() / 2) {
        return nullptr;
    }
    const auto matrixSize = requestCount * nodeCount;
    context.prepareCacheMatchPrefixes();
    std::unique_ptr<float[]> costs(
        new float[matrixSize * 2]);
    if (!fillCostsForNodeRange(
            context, requestTokenCounts, 0, nodeCount, costs.get())) {
        return nullptr;
    }
    if (replayInfo) {
        replayInfo->init(context.isCollectingReplay(), lambda, alpha,
                        alphaDecay, minAlpha, maxLoopCount,
                        optimizationTimeBudgetUs);
    }
    return costs;
}

bool CostSchedulerKernelInternal::fillCostsForNodeRange(
    const NaviSchedScheduleContext &context,
    const int64_t *requestTokenCounts, size_t nodeBegin,
    size_t nodeEnd, float *costs, PreparationTiming *timing)
{
    if (timing) {
        *timing = {};
    }
    auto requestCount = context.requestCount();
    if (!requestTokenCounts || !costs || nodeBegin > nodeEnd ||
        nodeEnd > context.nodeCount()) {
        return false;
    }
    std::vector<int64_t> cacheHitBlockCounts(requestCount);
    const auto matrixSize = requestCount * context.nodeCount();
    for (size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex) {
        const auto &node = context.getNode(nodeIndex);
        if (!isValidNode(node)) {
            return false;
        }
        const auto matchBeginUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
        if (!context.getCacheHitBlockCounts(
                nodeIndex, cacheHitBlockCounts)) {
            return false;
        }
        const auto matchEndUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
        const auto &latencyParameters = node.getLatencyParameters();
        auto costRow = costs + nodeIndex * requestCount;
        auto waitingCostRow = costRow + matrixSize;
        for (size_t requestIndex = 0; requestIndex < requestCount;
            ++requestIndex) {
            const auto &request = context.getRequest(requestIndex).request;
            auto cacheHitBlockCount = cacheHitBlockCounts[requestIndex];
            auto cacheHitTokenCount = cacheHitBlockCount * request.blockSize;
            const auto requestCosts = NonLinearPrefillModelR::calculateRequestLinearCosts(
                latencyParameters, requestTokenCounts[requestIndex], cacheHitTokenCount);
            costRow[requestIndex] = static_cast<float>(requestCosts.first);
            waitingCostRow[requestIndex] = static_cast<float>(requestCosts.second);
        }
        if (timing) {
            timing->matchUs += matchEndUs - matchBeginUs;
            timing->costEvaluationUs += autil::TimeUtility::monotonicTimeUs() - matchEndUs;
            ++timing->completedNodes;
        }
    }
    return true;
}

double CostSchedulerKernelInternal::forwardNodeRange(
    const NaviSchedScheduleContext &context,
    const float *costs, const double *weights,
    const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator,
    double lambda, double latencyWeightConcavity,
    size_t nodeBegin, size_t nodeEnd,
    NodeGradient *nodeGradients, double *firstTerms,
    double *secondTerms)
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    auto requestCount = context.requestCount();
    auto firstScale = lambda / tokenNormalizationDenominator;
    auto secondScale = (1.0 - lambda) / requestCount;
    const auto matrixSize = requestCount * context.nodeCount();
    for (size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex) {
        auto rowOffset = nodeIndex * requestCount;
        auto costRow = costs + rowOffset;
        auto waitingCostRow = costRow + matrixSize;
        auto weightRow = weights + rowOffset;
        double requestCostSum = 0.0;
        double totalPromptTokenSum = 0.0;
        double weightSum = 0.0;
        double inverseTokenWeightSum = 0.0;
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto weight = weightRow[requestIndex];
            totalPromptTokenSum += weight * requestTokenCounts[requestIndex];
            weightSum += weight;
            inverseTokenWeightSum +=
                weight * inverseRequestTokenCounts[requestIndex];
        }
        const auto &node = context.getNode(nodeIndex);
        const auto capacitySplit =
            NonLinearPrefillModelR::usesCapacitySplitParameters(
                node.getLatencyParameters());
        const auto workloadRunningCapacity = capacitySplit
            ? std::max(0, node.getMaxRunningRequestCount() -
                              node.getWorkloadRunningCapacitySlotCount())
            : 0;
        const auto reservedRunningCapacity = capacitySplit
            ? std::max(0, node.getMaxRunningRequestCount() -
                              node.getReservedRunningCapacitySlotCount())
            : 0;
        if (capacitySplit) {
            requestCostSum = aggregateCapacitySplitCost(
                costRow, waitingCostRow, weightRow, requestCount,
                workloadRunningCapacity);
        } else {
            for (size_t requestIndex = 0; requestIndex < requestCount;
                 ++requestIndex) {
                requestCostSum +=
                    weightRow[requestIndex] * costRow[requestIndex];
            }
        }
        auto requestCostWithoutReportedWorkload = requestCostSum;
        auto promptTokensWithoutReportedWorkload = totalPromptTokenSum;
        double precedingRequestsWithoutReportedWorkload = 0.0;
        double precedingRequestsWithReportedWorkload = 0.0;
        double existingNodeCost = 0.0;
        if (node.usesUnifiedLatencyPrediction()) {
            NaviSchedLatencyFeatures floorWorkload;
            if (!node.getUnifiedDecayFloorFeatures(floorWorkload)) {
                return std::numeric_limits<double>::infinity();
            }
            precedingRequestsWithoutReportedWorkload =
                floorWorkload.getPrecedingRequestCount();
            precedingRequestsWithReportedWorkload =
                node.getWorkloadFeatures().getPrecedingRequestCount();
            if (capacitySplit) {
                const auto reservedRequestCost =
                    reservedRunningCapacity == workloadRunningCapacity
                        ? requestCostSum
                        : aggregateCapacitySplitCost(
                              costRow, waitingCostRow, weightRow, requestCount,
                              reservedRunningCapacity);
                requestCostWithoutReportedWorkload =
                    reservedRequestCost +
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(), floorWorkload);
                requestCostSum +=
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(),
                        node.getWorkloadFeatures());
            } else {
                requestCostWithoutReportedWorkload +=
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(), floorWorkload);
                requestCostSum = requestCostWithoutReportedWorkload +
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(),
                        node.getReportedWorkloadFeatures());
            }
            promptTokensWithoutReportedWorkload +=
                floorWorkload.getCacheHitTokenCount() +
                floorWorkload.getCacheMissTokenCount();
            const auto &reportedWorkload = node.getReportedWorkloadFeatures();
            totalPromptTokenSum = promptTokensWithoutReportedWorkload +
                reportedWorkload.getCacheHitTokenCount() +
                reportedWorkload.getCacheMissTokenCount();
            const auto candidateBatchActivation =
                std::min(1.0, std::max(0.0, weightSum));
            const auto batchCost = node.getLatencyParameters()[
                NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
                NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX);
            const auto batchCostAdjustment = batchCost *
                (candidateBatchActivation - 1.0);
            requestCostSum += batchCostAdjustment;
            requestCostWithoutReportedWorkload += batchCostAdjustment;
        }
        double candidateBatchActivationDerivative = 0.0;
        if (node.usesUnifiedLatencyPrediction() && weightSum >= 0.0 &&
            weightSum < 1.0) {
            candidateBatchActivationDerivative = node.getLatencyParameters()[
                NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
                NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX);
        }
        double nodeCostDerivative = 0.0;
        double totalPromptTokenDerivative = 0.0;
        auto nodeCost =
            NonLinearPrefillModelR::
                calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(), requestCostSum,
                    totalPromptTokenSum,
                    precedingRequestsWithReportedWorkload,
                    nodeCostDerivative,
                    totalPromptTokenDerivative);
        double residualDerivativeFactor = 1.0;
        nodeCost = node.applyLatencyResidual(nodeCost,
                                             &residualDerivativeFactor);
        nodeCostDerivative *= residualDerivativeFactor;
        totalPromptTokenDerivative *= residualDerivativeFactor;
        if (node.usesUnifiedLatencyPrediction()) {
            double floorCostDerivative = 0.0;
            double floorPromptDerivative = 0.0;
            auto floorCost = NonLinearPrefillModelR::
                calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(),
                    requestCostWithoutReportedWorkload,
                    promptTokensWithoutReportedWorkload,
                    precedingRequestsWithoutReportedWorkload,
                    floorCostDerivative, floorPromptDerivative);
            double floorResidualDerivativeFactor = 1.0;
            floorCost = node.applyLatencyResidual(
                floorCost, &floorResidualDerivativeFactor);
            floorCostDerivative *= floorResidualDerivativeFactor;
            floorPromptDerivative *= floorResidualDerivativeFactor;
            bool usesReportedWorkload = true;
            nodeCost = node.decayUnifiedLatency(
                nodeCost, floorCost, &usesReportedWorkload);
            if (!usesReportedWorkload) {
                nodeCostDerivative = floorCostDerivative;
                totalPromptTokenDerivative = floorPromptDerivative;
            }
            if (nodeGradients && capacitySplit) {
                nodeGradients[nodeIndex - nodeBegin].runningCapacity =
                    usesReportedWorkload ? workloadRunningCapacity
                                         : reservedRunningCapacity;
            }
            existingNodeCost = node.getExistingUnifiedLatencyMs();
        }
        auto queueCost = nodeCost + node.getEstimatedQueueWaitMs();
        auto firstTerm = node.usesUnifiedLatencyPrediction()
            ? nodeCost - existingNodeCost
            : nodeCost * weightSum;
        auto latencyWeightLogScale = latencyWeightConcavity > 0.0
            ? std::log(latencyWeightConcavity)
            : 1.0;
        auto latencyWeight = latencyWeightConcavity > 0.0
            ? std::log1p(latencyWeightConcavity * inverseTokenWeightSum) /
                  latencyWeightLogScale
            : inverseTokenWeightSum;
        auto secondTerm = queueCost * latencyWeight;
        if (firstTerms) {
            firstTerms[nodeIndex] = firstTerm;
        }
        if (secondTerms) {
            secondTerms[nodeIndex] = secondTerm;
        }
        if (nodeGradients) {
            auto &gradient = nodeGradients[nodeIndex - nodeBegin];
            const auto latencyGradient =
                firstScale *
                    (node.usesUnifiedLatencyPrediction() ? 1.0 : weightSum) +
                secondScale * latencyWeight;
            gradient.cost = nodeCostDerivative * latencyGradient;
            gradient.totalPromptToken = totalPromptTokenDerivative *
                latencyGradient;
            gradient.weight = node.usesUnifiedLatencyPrediction()
                ? 0.0
                : firstScale * nodeCost;
            gradient.weight += gradient.cost *
                candidateBatchActivationDerivative;
            gradient.inverseToken = secondScale * queueCost *
                (latencyWeightConcavity > 0.0
                     ? latencyWeightConcavity /
                           ((1.0 + latencyWeightConcavity *
                             inverseTokenWeightSum) * latencyWeightLogScale)
                     : 1.0);
            gradient.capacitySplit = capacitySplit;
            if (capacitySplit && !node.usesUnifiedLatencyPrediction()) {
                gradient.runningCapacity = workloadRunningCapacity;
            }
        }
        sum1 += firstTerm;
        sum2 += secondTerm;
    }
    return sum1 * firstScale + sum2 * secondScale;
}

double CostSchedulerKernelInternal::forwardDiscrete(
    const NaviSchedScheduleContext &context, const float *costs,
    const uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator, double lambda,
    double latencyWeightConcavity,
    DiscreteForwardWorkspace &workspace)
{
    return forwardDiscreteImpl(
        context, costs, nodeIndexes, requestTokenCounts,
        inverseRequestTokenCounts, tokenNormalizationDenominator,
        lambda, latencyWeightConcavity, workspace,
        context.nodeCount(), context.nodeCount());
}

double CostSchedulerKernelInternal::forwardDiscreteImpl(
    const NaviSchedScheduleContext &context, const float *costs,
    const uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator, double lambda,
    double latencyWeightConcavity, DiscreteForwardWorkspace &workspace,
    size_t firstNode, size_t secondNode,
    const std::vector<std::vector<size_t>> *nodeRequests, size_t movedRequest,
    bool reuseSourceNode)
{
    workspace.nodeCapacityLoads = 0;
    workspace.nodeBaselineLoads = 0;
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    if (!costs || !nodeIndexes || !requestTokenCounts ||
        !inverseRequestTokenCounts ||
        requestCount == 0 || nodeCount == 0 ||
        tokenNormalizationDenominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const bool pairOnly = firstNode < nodeCount;
    reuseSourceNode = reuseSourceNode && pairOnly && nodeRequests;
    if (pairOnly && secondNode >= nodeCount) {
        return std::numeric_limits<double>::infinity();
    }
    if (!pairOnly || workspace.baselineContext != &context ||
        workspace.nodeBaselines.size() != nodeCount) {
        workspace.baselineContext = &context;
        workspace.nodeBaselines.assign(nodeCount, {});
    }
    const auto baselineFor = [&](size_t nodeIndex)
        -> const DiscreteForwardWorkspace::NodeBaseline & {
        auto &baseline = workspace.nodeBaselines[nodeIndex];
        if (baseline.loaded) {
            return baseline;
        }
        ++workspace.nodeBaselineLoads;
        const auto &node = context.getNode(nodeIndex);
        const auto &parameters = node.getLatencyParameters();
        baseline.loaded = true;
        baseline.capacitySplit =
            NonLinearPrefillModelR::usesCapacitySplitParameters(parameters);
        baseline.workloadSlots = node.getWorkloadRunningCapacitySlotCount();
        baseline.reservedSlots = node.getReservedRunningCapacitySlotCount();
        baseline.maxRunning = node.getMaxRunningRequestCount();
        baseline.unified = node.usesUnifiedLatencyPrediction();
        if (baseline.unified) {
            NaviSchedLatencyFeatures floor;
            baseline.floorValid = node.getUnifiedDecayFloorFeatures(floor);
            if (baseline.floorValid) {
                baseline.floorLinearCost =
                    NonLinearPrefillModelR::calculateLinearCost(parameters, floor);
                baseline.floorHitTokens = floor.getCacheHitTokenCount();
                baseline.floorMissTokens = floor.getCacheMissTokenCount();
                baseline.floorPrecedingRequests = floor.getPrecedingRequestCount();
                const auto &reported = node.getReportedWorkloadFeatures();
                baseline.reportedHitTokens = reported.getCacheHitTokenCount();
                baseline.reportedMissTokens = reported.getCacheMissTokenCount();
                baseline.workloadPrecedingRequests =
                    node.getWorkloadFeatures().getPrecedingRequestCount();
                if (baseline.capacitySplit) {
                    baseline.workloadLinearCost = NonLinearPrefillModelR::calculateLinearCost(
                        parameters, node.getWorkloadFeatures());
                } else {
                    baseline.reportedLinearCost =
                        NonLinearPrefillModelR::calculateLinearCost(parameters, reported);
                }
            }
        }
        return baseline;
    };
    const auto reset = [&](std::vector<double> &values) {
        if (pairOnly && values.size() == nodeCount) {
            if (!reuseSourceNode) {
                values[firstNode] = 0.0;
            }
            values[secondNode] = 0.0;
        } else {
            values.assign(nodeCount, 0.0);
        }
    };
    reset(workspace.requestCostSums);
    reset(workspace.requestCostSumsWithoutReportedWorkload);
    reset(workspace.totalPromptTokenSums);
    reset(workspace.requestWeightSums);
    reset(workspace.inverseTokenWeightSums);
    reset(workspace.nodeValues);
    const auto accumulateRequest = [&](size_t nodeIndex, size_t requestIndex) {
        ++workspace.nodeCapacityLoads;
        const auto &baseline = baselineFor(nodeIndex);
        const auto costIndex = nodeIndex * requestCount + requestIndex;
        const auto waitingCostIndex = nodeCount * requestCount + costIndex;
        const auto assignedRequestCount =
            workspace.requestWeightSums[nodeIndex];
        const auto usesRunningCapacity =
            !baseline.capacitySplit ||
            baseline.workloadSlots +
                    assignedRequestCount <
                baseline.maxRunning;
        const auto usesReservedRunningCapacity =
            !baseline.capacitySplit ||
            baseline.reservedSlots +
                    assignedRequestCount <
                baseline.maxRunning;
        workspace.requestCostSums[nodeIndex] +=
            costs[usesRunningCapacity ? costIndex : waitingCostIndex];
        workspace.requestCostSumsWithoutReportedWorkload[nodeIndex] +=
            costs[usesReservedRunningCapacity ? costIndex : waitingCostIndex];
        workspace.totalPromptTokenSums[nodeIndex] +=
            requestTokenCounts[requestIndex];
        workspace.requestWeightSums[nodeIndex] += 1.0;
        workspace.inverseTokenWeightSums[nodeIndex] +=
            inverseRequestTokenCounts[requestIndex];
    };
    if (pairOnly && nodeRequests) {
        const auto accumulateNode = [&](size_t nodeIndex, bool source) {
            ++workspace.nodeCapacityLoads;
            const auto &baseline = baselineFor(nodeIndex);
            const auto capacitySplit = baseline.capacitySplit;
            const auto workloadSlots = baseline.workloadSlots;
            const auto reservedSlots = baseline.reservedSlots;
            const auto maxRunning = baseline.maxRunning;
            const auto runningCosts = costs + nodeIndex * requestCount;
            const auto waitingCosts = runningCosts + nodeCount * requestCount;
            auto requestCost = workspace.requestCostSums[nodeIndex];
            auto floorRequestCost = workspace.requestCostSumsWithoutReportedWorkload[nodeIndex];
            auto totalTokens = workspace.totalPromptTokenSums[nodeIndex];
            auto assignedRequests = workspace.requestWeightSums[nodeIndex];
            auto inverseTokens = workspace.inverseTokenWeightSums[nodeIndex];
            const auto append = [&](size_t requestIndex) {
                requestCost += (!capacitySplit || workloadSlots + assignedRequests < maxRunning)
                    ? runningCosts[requestIndex] : waitingCosts[requestIndex];
                floorRequestCost += (!capacitySplit || reservedSlots + assignedRequests < maxRunning)
                    ? runningCosts[requestIndex] : waitingCosts[requestIndex];
                totalTokens += requestTokenCounts[requestIndex];
                assignedRequests += 1.0;
                inverseTokens += inverseRequestTokenCounts[requestIndex];
            };
            bool inserted = source;
            for (const auto requestIndex : (*nodeRequests)[nodeIndex]) {
                if (source && requestIndex == movedRequest) {
                    continue;
                }
                if (!inserted && movedRequest < requestIndex) {
                    append(movedRequest);
                    inserted = true;
                }
                append(requestIndex);
            }
            if (!inserted) {
                append(movedRequest);
            }
            workspace.requestCostSums[nodeIndex] = requestCost;
            workspace.requestCostSumsWithoutReportedWorkload[nodeIndex] = floorRequestCost;
            workspace.totalPromptTokenSums[nodeIndex] = totalTokens;
            workspace.requestWeightSums[nodeIndex] = assignedRequests;
            workspace.inverseTokenWeightSums[nodeIndex] = inverseTokens;
        };
        if (!reuseSourceNode) {
            accumulateNode(firstNode, true);
        }
        accumulateNode(secondNode, false);
    } else {
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
            const auto nodeIndex = static_cast<size_t>(nodeIndexes[requestIndex]);
            if (nodeIndex >= nodeCount) {
                return std::numeric_limits<double>::infinity();
            }
            if (!pairOnly || nodeIndex == firstNode || nodeIndex == secondNode) {
                accumulateRequest(nodeIndex, requestIndex);
            }
        }
    }
    auto firstScale = lambda / tokenNormalizationDenominator;
    auto secondScale = (1.0 - lambda) /
        static_cast<double>(requestCount);
    auto latencyWeightLogScale = latencyWeightConcavity > 0.0
        ? std::log(latencyWeightConcavity)
        : 1.0;
    double result = reuseSourceNode ? workspace.nodeValues[firstNode] : 0.0;
    const auto evaluatedNodeCount = pairOnly ? size_t{2} : nodeCount;
    for (size_t index = reuseSourceNode ? 1 : 0; index < evaluatedNodeCount; ++index) {
        const auto nodeIndex = pairOnly ? (index == 0 ? firstNode : secondNode) : index;
        auto requestWeightSum =
            workspace.requestWeightSums[nodeIndex];
        const auto &node = context.getNode(nodeIndex);
        if (requestWeightSum == 0.0) {
            continue;
        }
        double nodeCost = 0.0;
        double existingNodeCost = 0.0;
        const auto &baseline = baselineFor(nodeIndex);
        if (baseline.unified) {
            if (!baseline.floorValid) {
                return std::numeric_limits<double>::infinity();
            }
            const auto requestCostWithoutReportedWorkload =
                (baseline.capacitySplit
                     ? workspace.requestCostSumsWithoutReportedWorkload[nodeIndex]
                     : workspace.requestCostSums[nodeIndex]) +
                baseline.floorLinearCost;
            const auto promptTokensWithoutReportedWorkload =
                workspace.totalPromptTokenSums[nodeIndex] +
                baseline.floorHitTokens +
                baseline.floorMissTokens;
            auto requestCostWithReportedWorkload = baseline.capacitySplit
                ? workspace.requestCostSums[nodeIndex] +
                      baseline.workloadLinearCost
                : requestCostWithoutReportedWorkload +
                      baseline.reportedLinearCost;
            const auto promptTokensWithReportedWorkload =
                promptTokensWithoutReportedWorkload +
                baseline.reportedHitTokens +
                baseline.reportedMissTokens;
            const auto precedingRequestsWithoutReportedWorkload =
                baseline.floorPrecedingRequests;
            const auto precedingRequestsWithReportedWorkload =
                baseline.workloadPrecedingRequests;
            auto costWithReportedWorkload = NonLinearPrefillModelR::
                calculateLatencyFromRequestCostSum(
                    node.getLatencyParameters(),
                    requestCostWithReportedWorkload,
                    promptTokensWithReportedWorkload,
                    precedingRequestsWithReportedWorkload);
            costWithReportedWorkload =
                node.applyLatencyResidual(costWithReportedWorkload);
            auto floorCost = NonLinearPrefillModelR::
                calculateLatencyFromRequestCostSum(
                    node.getLatencyParameters(),
                    requestCostWithoutReportedWorkload,
                    promptTokensWithoutReportedWorkload,
                    precedingRequestsWithoutReportedWorkload);
            floorCost = node.applyLatencyResidual(floorCost);
            nodeCost = node.decayUnifiedLatency(costWithReportedWorkload,
                                                floorCost);
            existingNodeCost = node.getExistingUnifiedLatencyMs();
        } else {
            nodeCost = NonLinearPrefillModelR::
                calculateLatencyFromRequestCostSum(
                    node.getLatencyParameters(),
                    workspace.requestCostSums[nodeIndex],
                    workspace.totalPromptTokenSums[nodeIndex], 0.0);
            nodeCost = node.applyLatencyResidual(nodeCost);
        }
        auto inverseTokenWeightSum =
            workspace.inverseTokenWeightSums[nodeIndex];
        auto latencyWeight = latencyWeightConcavity > 0.0
            ? std::log1p(latencyWeightConcavity *
                         inverseTokenWeightSum) /
                  latencyWeightLogScale
            : inverseTokenWeightSum;
        const auto firstTerm = node.usesUnifiedLatencyPrediction()
            ? nodeCost - existingNodeCost
            : nodeCost * requestWeightSum;
        const auto nodeValue = firstScale * firstTerm +
            secondScale *
                (nodeCost +
                 context.getNode(nodeIndex).getEstimatedQueueWaitMs()) *
                latencyWeight;
        workspace.nodeValues[nodeIndex] = nodeValue;
        result += nodeValue;
    }
    return result;
}

int64_t CostSchedulerKernelInternal::discreteRefinementBudgetUs(
    int64_t optimizationTimeBudgetUs, int64_t elapsedUs)
{
    if (optimizationTimeBudgetUs < 0) {
        return -1;
    }
    return std::max<int64_t>(0, optimizationTimeBudgetUs - elapsedUs);
}

bool CostSchedulerKernelInternal::refineDiscreteAssignment(
    const NaviSchedScheduleContext &context, const float *costs,
    uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator, double lambda,
    double latencyWeightConcavity, double &value,
    int64_t timeBudgetUs, size_t maxEvaluations,
    DiscreteRefinementStats &stats)
{
    stats = {};
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    DiscreteForwardWorkspace workspace;
    value = forwardDiscrete(context, costs, nodeIndexes, requestTokenCounts,
                            inverseRequestTokenCounts, tokenNormalizationDenominator,
                            lambda, latencyWeightConcavity, workspace);
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
    std::vector<std::vector<size_t>> nodeRequests(nodeCount);
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        nodeRequests[nodeIndexes[requestIndex]].push_back(requestIndex);
    }
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
        for (size_t requestIndex = 0; requestIndex < requestCount && !exhausted(); ++requestIndex) {
            const auto source = nodeIndexes[requestIndex];
            bool sourceNodeReady = false;
            for (size_t target = 0; target < nodeCount && !exhausted(); ++target) {
                if (target == source) {
                    continue;
                }
                nodeIndexes[requestIndex] = static_cast<uint32_t>(target);
                const auto pairValue = forwardDiscreteImpl(
                    context, costs, nodeIndexes, requestTokenCounts,
                    inverseRequestTokenCounts, tokenNormalizationDenominator,
                    lambda, latencyWeightConcavity, workspace, source, target,
                    &nodeRequests, requestIndex, sourceNodeReady);
                sourceNodeReady = sourceNodeReady || std::isfinite(pairValue);
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
        auto &sourceRequests = nodeRequests[source];
        sourceRequests.erase(std::lower_bound(sourceRequests.begin(), sourceRequests.end(), bestRequest));
        auto &targetRequests = nodeRequests[bestTarget];
        targetRequests.insert(std::lower_bound(targetRequests.begin(), targetRequests.end(), bestRequest),
                              bestRequest);
        nodeIndexes[bestRequest] = bestTarget;
        nodeValues[source] = bestSourceValue;
        nodeValues[bestTarget] = bestTargetValue;
        value += bestDelta;
        ++stats.moves;
    }
    stats.budgetExhausted = exhausted();
    value = forwardDiscrete(context, costs, nodeIndexes, requestTokenCounts,
                            inverseRequestTokenCounts, tokenNormalizationDenominator,
                            lambda, latencyWeightConcavity, workspace);
    if (!std::isfinite(value) || value > originalValue) {
        std::copy(original.begin(), original.end(), nodeIndexes);
        value = originalValue;
        stats.moves = 0;
    }
    context.recordDiscreteSearch(autil::TimeUtility::monotonicTimeUs() - beginUs,
                                 stats.evaluations, stats.moves);
    return true;
}

void CostSchedulerKernelInternal::backwardNodeRange(
    const NaviSchedScheduleContext &context, const float *costs,
    const double *weights, const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts,
    size_t requestCount, size_t nodeBegin, size_t nodeEnd,
    const NodeGradient *nodeGradients, double *gradients)
{
    for (size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex) {
        auto localNodeIndex = nodeIndex - nodeBegin;
        const auto &nodeGradient = nodeGradients[localNodeIndex];
        auto rowOffset = nodeIndex * requestCount;
        auto costRow = costs + rowOffset;
        auto gradientRow = gradients + localNodeIndex * requestCount;
        if (nodeGradient.capacitySplit) {
            const auto waitingCostRow =
                costRow + context.nodeCount() * requestCount;
            calculateCapacitySplitMarginalCosts(
                costRow, waitingCostRow, weights + rowOffset,
                requestCount, nodeGradient.runningCapacity, gradientRow);
        }
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            const auto requestCostDerivative = nodeGradient.capacitySplit
                ? gradientRow[requestIndex]
                : costRow[requestIndex];
            gradientRow[requestIndex] =
                nodeGradient.cost * requestCostDerivative +
                nodeGradient.totalPromptToken *
                    requestTokenCounts[requestIndex] +
                nodeGradient.weight +
                nodeGradient.inverseToken *
                    inverseRequestTokenCounts[requestIndex];
        }
    }
}

bool CostSchedulerKernelInternal::optimize(
    const NaviSchedScheduleContext &context, const float *costs,
    const int64_t *requestTokenCounts, int64_t totalTokenCount,
    double lambda, double alpha, double alphaDecay, double minAlpha,
    size_t maxLoopCount, int64_t optimizationTimeBudgetUs,
    ReplayInfo *replayInfo, double latencyWeightConcavity,
    uint64_t initialWeightSeed,
    double *weights,
    std::vector<uint32_t> &selectedNodeIndexes,
    double &outputForwardValue,
    double latencyTokenNormalizationPower,
    size_t discreteCheckInterval,
    double discreteGapThreshold,
    double discreteRepairAlphaDecay, size_t minLoopCount)
{
    if (replayInfo) {
        replayInfo->minLoopCount = minLoopCount;
    }
    const auto priorCollectionUs = context.getScheduleBeginTimeUs() > 0
        ? 0 : context.getReplayCollectionTimeUs();
    const auto optimizationBeginUs = context.getScheduleBeginTimeUs() > 0
        ? context.getScheduleBeginTimeUs() : autil::TimeUtility::monotonicTimeUs();
    if (context.getScheduleBeginTimeUs() > 0) {
        optimizationTimeBudgetUs = context.getScheduleDeadlineUs() - optimizationBeginUs;
        if (replayInfo) {
            replayInfo->optimizationTimeBudgetUs = optimizationTimeBudgetUs;
        }
    }
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    auto weightCount = requestCount * nodeCount;
    if (!costs || !weights || totalTokenCount <= 0 || 0 == maxLoopCount ||
        nodeCount > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    std::vector<uint32_t> bestNodeIndexes(requestCount);
    std::vector<uint32_t> candidateNodeIndexes(requestCount);
    auto bestLoss = std::numeric_limits<double>::infinity();
    DiscreteForwardWorkspace discreteForwardWorkspace;
    bool discreteRepairActive = false;
    double repairAlpha = alpha;
    std::vector<double> inverseRequestTokenCounts(requestCount);
    for (size_t requestIndex = 0; requestIndex < requestCount;
         ++requestIndex) {
        inverseRequestTokenCounts[requestIndex] =
            1.0 /
            std::pow(static_cast<double>(requestTokenCounts[requestIndex]),
                     latencyTokenNormalizationPower);
    }
    auto tokenNormalizationDenominator =
        calculateTokenNormalizationDenominator(
            requestTokenCounts, requestCount,
            latencyTokenNormalizationPower);
    if (tokenNormalizationDenominator <= 0.0) {
        return false;
    }
    std::vector<NodeGradient> nodeGradients(nodeCount);
    std::vector<double> gradients(weightCount);
    std::vector<double> weightsBeforeProjection;
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        weightsBeforeProjection.resize(weightCount);
        context.addReplayCollectionTimeUs(
            autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
    }
    auto projectionTileSize = std::min(
        requestCount, PROJECTION_REQUEST_TILE_SIZE);
    std::vector<double> projectionBuffer(
        nodeCount * projectionTileSize);
    auto collectTiming = (replayInfo && replayInfo->enabled);
    auto trackLoopTime = collectTiming || optimizationTimeBudgetUs >= 0;

    auto lastLoopEndUs = optimizationBeginUs;
    int64_t lastLoopCollectionUs = 0;
    auto runBeginUs = collectTiming ? optimizationBeginUs : 0;
    size_t actualLoopCount = 0;
    std::string stopReason = "max_loop_count";
    initializeWeights(weights, requestCount, nodeCount,
                      initialWeightSeed);
    if (!selectWeightRequestRange(
            weights, requestCount, nodeCount, 0, requestCount,
            bestNodeIndexes.data(), initialWeightSeed)) {
        return false;
    }
    bestLoss = forwardDiscrete(
        context, costs, bestNodeIndexes.data(), requestTokenCounts,
        inverseRequestTokenCounts.data(), tokenNormalizationDenominator,
        lambda, latencyWeightConcavity, discreteForwardWorkspace);
    if (!std::isfinite(bestLoss)) {
        return false;
    }
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        replayInfo->beginRun(weights, weightCount);
        context.addReplayCollectionTimeUs(
            autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
    }
    for (size_t iteration = 0; iteration < maxLoopCount; ++iteration) {
        bool discreteEvaluated = false;
        double discreteForwardValue = 0.0;
        double discreteGap = 0.0;
        auto stepAlpha = getStepAlpha(alpha, alphaDecay, minAlpha,
                                      iteration);
        auto stepBeginUs = autil::TimeUtility::monotonicTimeUs();
        auto value = forwardNodeRange(
            context, costs, weights, requestTokenCounts,
            inverseRequestTokenCounts.data(),
            tokenNormalizationDenominator,
            lambda, latencyWeightConcavity, 0, nodeCount,
            nodeGradients.data(), nullptr, nullptr);
        auto forwardEndUs = autil::TimeUtility::monotonicTimeUs();
        if (!std::isfinite(value)) {
            return false;
        }
        if (iteration > 0) {
            auto discreteLoss = forwardDiscrete(
                context, costs, candidateNodeIndexes.data(),
                requestTokenCounts,
                inverseRequestTokenCounts.data(),
                tokenNormalizationDenominator,
                lambda, latencyWeightConcavity,
                discreteForwardWorkspace);
            discreteEvaluated = true;
            discreteForwardValue = discreteLoss;
            if (!std::isfinite(discreteLoss)) {
                return false;
            }
            if (discreteLoss < bestLoss) {
                bestLoss = discreteLoss;
                bestNodeIndexes = candidateNodeIndexes;
            }
            auto gap = (discreteLoss - value) /
                std::max(std::abs(value),
                         std::numeric_limits<double>::epsilon());
            discreteGap = gap;
            if ((discreteRepairActive || iteration % discreteCheckInterval == 0) &&
                gap > discreteGapThreshold) {
                repairAlpha = std::max(
                    minAlpha,
                    (discreteRepairActive ? repairAlpha : stepAlpha) *
                        discreteRepairAlphaDecay);
                if (repairAlpha <= 0.0) {
                    repairAlpha = std::numeric_limits<double>::min();
                }
                discreteRepairActive = true;
            }
        }
        if (discreteRepairActive) {
            stepAlpha = repairAlpha;
        }
        const auto backwardBeginUs = autil::TimeUtility::monotonicTimeUs();
        backwardNodeRange(
            context, costs, weights, requestTokenCounts,
            inverseRequestTokenCounts.data(),
            requestCount, 0,
            nodeCount,
            nodeGradients.data(), gradients.data());
        auto backwardEndUs = autil::TimeUtility::monotonicTimeUs();
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->beginStep(
                iteration, stepAlpha, value, gradients.data(),
                weights, weightCount,
                forwardEndUs - stepBeginUs,
                backwardEndUs - forwardEndUs);
            if (discreteEvaluated) {
                replayInfo->recordDiscreteEvaluation(
                    discreteForwardValue, discreteGap,
                    discreteRepairActive);
            }
            std::copy(weights, weights + weightCount,
                      weightsBeforeProjection.begin());
            if (!applyGradient(
                    weightsBeforeProjection.data(), 0, weightCount,
                    gradients.data(), stepAlpha)) {
                return false;
            }
            context.addReplayCollectionTimeUs(
                autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
        }
        auto updateEndUs = collectTiming
                               ? autil::TimeUtility::monotonicTimeUs()
                               : 0;
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->recordBeforeProjection(
                weightsBeforeProjection.data(),
                weightCount, updateEndUs - backwardEndUs);
            context.addReplayCollectionTimeUs(
                autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
        }
        GradientSpan gradientSpan{gradients.data(), 0, nodeCount};
        const auto projectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        if (!projectRequestRange(
                weights, requestCount, nodeCount, 0, requestCount,
                projectionBuffer.data(), candidateNodeIndexes.data(),
                &gradientSpan, 1, stepAlpha, initialWeightSeed)) {
            return false;
        }
        auto projectionEndUs = autil::TimeUtility::monotonicTimeUs();
        context.recordCostGradientIteration(
            forwardEndUs - stepBeginUs + backwardEndUs - backwardBeginUs +
            projectionEndUs - projectionBeginUs);
        auto stepEndUs = trackLoopTime
                             ? autil::TimeUtility::monotonicTimeUs()
                             : 0;
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->recordAfterProjection(
                weights, weightCount,
                projectionEndUs - updateEndUs,
                stepEndUs - projectionEndUs,
                stepEndUs - stepBeginUs);
            context.addReplayCollectionTimeUs(
                autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
        }
        actualLoopCount = iteration + 1;
        const auto budgetEndUs = collectTiming
            ? autil::TimeUtility::monotonicTimeUs() : stepEndUs;
        const auto collectionUs = context.getReplayCollectionTimeUs() - priorCollectionUs;
        auto elapsedUs = std::max<int64_t>(
            0, budgetEndUs - optimizationBeginUs - collectionUs);
        auto lastLoopUs = std::max<int64_t>(
            0, budgetEndUs - lastLoopEndUs -
                   (collectionUs - lastLoopCollectionUs));
        lastLoopEndUs = budgetEndUs;
        lastLoopCollectionUs = collectionUs;
        if (!shouldRunNextLoop(
                actualLoopCount, maxLoopCount,
                optimizationTimeBudgetUs, elapsedUs, lastLoopUs, minLoopCount)) {
            stopReason = actualLoopCount >= maxLoopCount
                             ? "max_loop_count"
                             : "time_budget";
            break;
        }
    }
    context.recordCostOptimizerProgress(
        actualLoopCount, maxLoopCount, stopReason == "time_budget");
    auto finalForwardValue = forwardNodeRange(
        context, costs, weights, requestTokenCounts,
        inverseRequestTokenCounts.data(),
        tokenNormalizationDenominator,
        lambda, latencyWeightConcavity, 0, nodeCount,
        nullptr, nullptr, nullptr);
    if (!std::isfinite(finalForwardValue)) {
        return false;
    }
    auto finalDiscreteLoss = forwardDiscrete(
        context, costs, candidateNodeIndexes.data(),
        requestTokenCounts,
        inverseRequestTokenCounts.data(),
        tokenNormalizationDenominator,
        lambda, latencyWeightConcavity, discreteForwardWorkspace);
    if (!std::isfinite(finalDiscreteLoss)) {
        return false;
    }
    if (finalDiscreteLoss < bestLoss) {
        bestLoss = finalDiscreteLoss;
        bestNodeIndexes = candidateNodeIndexes;
    }
    const auto refinementBudget = discreteRefinementBudgetUs(
        optimizationTimeBudgetUs,
        optimizationTimeBudgetUs > 0
            ? std::max<int64_t>(
                  0, autil::TimeUtility::monotonicTimeUs() - optimizationBeginUs -
                         (context.getReplayCollectionTimeUs() - priorCollectionUs))
            : 0);
    const auto initialDiscreteLoss = bestLoss;
    DiscreteRefinementStats refinementStats;
    if (refinementBudget != 0 &&
        !refineDiscreteAssignment(
            context, costs, bestNodeIndexes.data(), requestTokenCounts,
            inverseRequestTokenCounts.data(), tokenNormalizationDenominator,
            lambda, latencyWeightConcavity, bestLoss, refinementBudget,
            std::numeric_limits<size_t>::max(), refinementStats)) {
        return false;
    }
    context.recordDiscreteSearchLosses(initialDiscreteLoss, bestLoss);
    outputForwardValue = bestLoss;
    selectedNodeIndexes = std::move(bestNodeIndexes);
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        replayInfo->finishRun(
            weights, weightCount, finalForwardValue, bestLoss,
            selectedNodeIndexes.data(), requestCount,
            collectTiming
                ? autil::TimeUtility::monotonicTimeUs() - runBeginUs
                : 0,
            actualLoopCount, stopReason,
            refinementStats.evaluations, refinementStats.moves);
        context.addReplayCollectionTimeUs(
            autil::TimeUtility::monotonicTimeUs() - collectionBeginUs);
    }
    return true;
}

bool CostSchedulerKernelInternal::shouldRunNextLoop(
    size_t completedLoopCount, size_t maxLoopCount,
    int64_t optimizationTimeBudgetUs, int64_t elapsedUs,
    int64_t lastLoopUs, size_t minLoopCount)
{
    if (completedLoopCount >= maxLoopCount) {
        return false;
    }
    if (completedLoopCount < std::min(minLoopCount, maxLoopCount) ||
        optimizationTimeBudgetUs < 0) {
        return true;
    }
    if (optimizationTimeBudgetUs > 0 && elapsedUs >= optimizationTimeBudgetUs) {
        return false;
    }
    auto averageLoopUs =
        (elapsedUs + static_cast<int64_t>(completedLoopCount) - 1) /
        static_cast<int64_t>(completedLoopCount);
    auto nextLoopUs = std::max(lastLoopUs, averageLoopUs);
    return elapsedUs < optimizationTimeBudgetUs &&
           nextLoopUs <= optimizationTimeBudgetUs - elapsedUs;
}

void CostSchedulerKernelInternal::ReplayInfo::OptimizationStep::Jsonize(
    JsonWrapper &json)
{
    json.Jsonize("iteration", iteration);
    json.Jsonize("alpha", alpha, alpha);
    json.Jsonize("forward_value", forwardValue);
    json.Jsonize("elapsed_us", elapsedUs);
    json.Jsonize("forward_us", forwardUs);
    json.Jsonize("backward_us", backwardUs);
    json.Jsonize("update_us", updateUs);
    json.Jsonize("projection_us", projectionUs);
    json.Jsonize("bind_us", bindUs);
    json.Jsonize("discrete_evaluated", discreteEvaluated,
                 discreteEvaluated);
    json.Jsonize("discrete_forward_value", discreteForwardValue,
                 discreteForwardValue);
    json.Jsonize("discrete_gap", discreteGap, discreteGap);
    json.Jsonize("discrete_repair_active", discreteRepairActive,
                 discreteRepairActive);
    json.Jsonize("gradients", gradients);
    json.Jsonize("weights_before_update", weightsBeforeUpdate);
    json.Jsonize("weights_before_projection", weightsBeforeProjection);
    json.Jsonize("weights_after_projection", weightsAfterProjection);
}

void CostSchedulerKernelInternal::ReplayInfo::OptimizationRun::Jsonize(
    JsonWrapper &json)
{
    json.Jsonize("initial_weights", initialWeights);
    json.Jsonize("steps", steps);
    json.Jsonize("final_weights", finalWeights);
    json.Jsonize("final_forward_value", finalForwardValue);
    json.Jsonize("best_forward_value", bestForwardValue,
                 bestForwardValue);
    json.Jsonize("best_node_indexes", bestNodeIndexes,
                 bestNodeIndexes);
    json.Jsonize("elapsed_us", elapsedUs);
    json.Jsonize("actual_loop_count", actualLoopCount,
                 actualLoopCount);
    json.Jsonize("stop_reason", stopReason, stopReason);
    json.Jsonize("refinement_evaluations", refinementEvaluations, refinementEvaluations);
    json.Jsonize("refinement_moves", refinementMoves, refinementMoves);
}

void CostSchedulerKernelInternal::ReplayInfo::Jsonize(JsonWrapper &json)
{
    json.Jsonize("scheduler", scheduler);
    json.Jsonize("lambda", lambda);
    json.Jsonize("alpha", alpha);
    json.Jsonize("alpha_decay", alphaDecay, alphaDecay);
    json.Jsonize("min_alpha", minAlpha, minAlpha);
    json.Jsonize("max_loop_count", maxLoopCount);
    json.Jsonize("min_loop_count", minLoopCount, minLoopCount);
    json.Jsonize("optimization_time_budget_us",
                 optimizationTimeBudgetUs,
                 optimizationTimeBudgetUs);
    json.Jsonize("parallelism", parallelism, parallelism);
    json.Jsonize("latency_weight_concavity", latencyWeightConcavity,
                 latencyWeightConcavity);
    json.Jsonize("latency_token_normalization_power",
                 latencyTokenNormalizationPower,
                 latencyTokenNormalizationPower);
    json.Jsonize("initial_weight_seed", initialWeightSeed,
                 initialWeightSeed);
    json.Jsonize("discrete_check_interval", discreteCheckInterval,
                 discreteCheckInterval);
    json.Jsonize("discrete_gap_threshold", discreteGapThreshold,
                 discreteGapThreshold);
    json.Jsonize("discrete_repair_alpha_decay",
                 discreteRepairAlphaDecay,
                 discreteRepairAlphaDecay);
    json.Jsonize("selection_jitter_ratio", selectionJitterRatio,
                 selectionJitterRatio);
    json.Jsonize("sched_infos", schedInfos);
    json.Jsonize("optimization_runs", optimizationRuns);
}

void CostSchedulerKernelInternal::ReplayInfo::init(
    bool enabledValue, double lambdaValue, double alphaValue,
    double alphaDecayValue, double minAlphaValue,
    size_t maxLoopCountValue, int64_t optimizationTimeBudgetUsValue)
{
    enabled = enabledValue;
    lambda = lambdaValue;
    alpha = alphaValue;
    alphaDecay = alphaDecayValue;
    minAlpha = minAlphaValue;
    maxLoopCount = maxLoopCountValue;
    optimizationTimeBudgetUs = optimizationTimeBudgetUsValue;
}

void CostSchedulerKernelInternal::ReplayInfo::beginRun(
    const double *weights, size_t weightCount)
{
    if (!enabled) {
        return;
    }
    optimizationRuns.emplace_back();
    auto &run = optimizationRuns.back();
    run.steps.reserve(maxLoopCount);
    run.initialWeights.assign(weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::beginStep(
    size_t iterationValue, double alphaValue,
    double forwardValueValue,
    const double *gradientValues,
    const double *weights, size_t weightCount,
    int64_t forwardUsValue,
    int64_t backwardUsValue)
{
    if (!enabled) {
        return;
    }
    auto &steps = optimizationRuns.back().steps;
    steps.emplace_back();
    auto &step = steps.back();
    step.iteration = iterationValue;
    step.alpha = alphaValue;
    step.forwardValue = forwardValueValue;
    step.forwardUs = forwardUsValue;
    step.backwardUs = backwardUsValue;
    step.gradients.assign(gradientValues,
                          gradientValues + weightCount);
    step.weightsBeforeUpdate.assign(weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::recordBeforeProjection(
    const double *weights, size_t weightCount,
    int64_t updateUsValue)
{
    if (!enabled) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.updateUs = updateUsValue;
    step.weightsBeforeProjection.assign(
        weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::recordDiscreteEvaluation(
    double discreteForwardValueValue, double discreteGapValue,
    bool repairActive)
{
    if (!enabled || optimizationRuns.empty() ||
        optimizationRuns.back().steps.empty()) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.discreteEvaluated = true;
    step.discreteForwardValue = discreteForwardValueValue;
    step.discreteGap = discreteGapValue;
    step.discreteRepairActive = repairActive;
}

void CostSchedulerKernelInternal::ReplayInfo::recordAfterProjection(
    const double *weights, size_t weightCount,
    int64_t projectionUsValue,
    int64_t bindUsValue, int64_t elapsedUsValue)
{
    if (!enabled) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.projectionUs = projectionUsValue;
    step.bindUs = bindUsValue;
    step.elapsedUs = elapsedUsValue;
    step.weightsAfterProjection.assign(
        weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::finishRun(
    const double *weights, size_t weightCount, double finalValue,
    double bestValue, const uint32_t *nodeIndexes,
    size_t requestCount, int64_t elapsedUsValue,
    size_t actualLoopCountValue, const std::string &stopReasonValue,
    size_t refinementEvaluationsValue, size_t refinementMovesValue)
{
    if (!enabled) {
        return;
    }
    auto &run = optimizationRuns.back();
    run.finalWeights.assign(weights, weights + weightCount);
    run.finalForwardValue = finalValue;
    run.bestForwardValue = bestValue;
    run.bestNodeIndexes.assign(nodeIndexes, nodeIndexes + requestCount);
    run.elapsedUs = elapsedUsValue;
    run.actualLoopCount = actualLoopCountValue;
    run.stopReason = stopReasonValue;
    run.refinementEvaluations = refinementEvaluationsValue;
    run.refinementMoves = refinementMovesValue;
}

void CostSchedulerKernelInternal::ReplayInfo::finish(
    NaviSchedScheduleContext &context,
    const float *costs, const double *weights,
    size_t requestCount, size_t nodeCount)
{
    if (!enabled) {
        return;
    }
    schedInfos.resize(nodeCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto &nodeSchedInfos = schedInfos[nodeIndex];
        nodeSchedInfos.reserve(requestCount);
        auto rowOffset = nodeIndex * requestCount;
        auto queueWaitMs = static_cast<float>(
            context.getNode(nodeIndex).getEstimatedQueueWaitMs());
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto index = rowOffset + requestIndex;
            SchedInfo schedInfo;
            schedInfo.cost = costs[index];
            schedInfo.waitingCost = costs[nodeCount * requestCount + index];
            schedInfo.queueWaitMs = queueWaitMs;
            schedInfo.weight = weights[index];
            nodeSchedInfos.emplace_back(schedInfo);
        }
    }
    auto replayJson = autil::legacy::ToJson(*this);
    context.setSchedulerInfo(
        autil::legacy::AnyCast<autil::legacy::json::JsonMap>(replayJson));
}

}
