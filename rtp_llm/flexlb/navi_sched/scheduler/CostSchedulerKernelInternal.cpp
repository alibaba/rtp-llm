#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/CostSchedulerConfig.h"
#include "autil/TimeUtility.h"
#include "navi/log/NaviLogger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <iterator>
#include <random>
#include <type_traits>

namespace navi_sched {
namespace {
constexpr double SELECTION_JITTER_RATIO = 1.0 / 1000.0;

double aggregateCapacitySplitCost(const float *runningCosts, const float *waitingCosts,
                                  const double *weights, size_t requestCount,
                                  double runningCapacity)
{
    double cost = 0.0;
    double runningWeight = 0.0;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        const auto weight = weights[requestIndex];
        const auto requestRunningWeight =
            std::min(weight, std::max(0.0, runningCapacity - runningWeight));
        cost += requestRunningWeight * runningCosts[requestIndex] +
                (weight - requestRunningWeight) * waitingCosts[requestIndex];
        runningWeight += requestRunningWeight;
    }
    return cost;
}

void calculateCapacitySplitMarginalCosts(const float *runningCosts,
                                         const float *waitingCosts, const double *weights,
                                         size_t requestCount, double runningCapacity,
                                         double *marginalCosts)
{
    if (runningCapacity <= 0.0) {
        std::copy(waitingCosts, waitingCosts + requestCount, marginalCosts);
        return;
    }
    double totalWeight = 0.0;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        totalWeight += weights[requestIndex];
    }
    if (totalWeight <= runningCapacity) {
        std::copy(runningCosts, runningCosts + requestCount, marginalCosts);
        return;
    }

    double prefixWeight = 0.0;
    size_t boundaryIndex = requestCount;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
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
    const auto boundaryDelta = runningCosts[boundaryIndex] - waitingCosts[boundaryIndex];
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        if (requestIndex < boundaryIndex) {
            marginalCosts[requestIndex] = runningCosts[requestIndex] - boundaryDelta;
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

double getSelectionJitter(uint64_t randomSeed, size_t requestIndex, size_t nodeIndex,
                          size_t nodeCount)
{
    auto value = randomSeed ^ (static_cast<uint64_t>(requestIndex) << 32) ^
                 static_cast<uint64_t>(nodeIndex);
    constexpr double UINT53_SCALE = 1.0 / 9007199254740992.0;
    auto unit = static_cast<double>(mixSelectionJitter(value) >> 11) * UINT53_SCALE;
    return unit * SELECTION_JITTER_RATIO / static_cast<double>(nodeCount);
}

void updateSelectionMaximum(size_t nodeCount, size_t requestIndex, size_t nodeIndex,
                            double projectedWeight, uint64_t randomSeed,
                            double &maxWeight, size_t &maxIndex)
{
    if (!std::isfinite(maxWeight)) {
        maxWeight = projectedWeight +
                    getSelectionJitter(randomSeed, requestIndex, nodeIndex, nodeCount);
        maxIndex = nodeIndex;
        return;
    }
    auto jitterLimit = SELECTION_JITTER_RATIO / static_cast<double>(nodeCount);
    bool jitterApplied = false;
    if (std::abs(projectedWeight - maxWeight) <= jitterLimit) {
        projectedWeight +=
            getSelectionJitter(randomSeed, requestIndex, nodeIndex, nodeCount);
        jitterApplied = true;
    }
    if (projectedWeight > maxWeight) {
        maxWeight = projectedWeight + (!jitterApplied
                                           ? getSelectionJitter(randomSeed, requestIndex,
                                                                nodeIndex, nodeCount)
                                           : 0.0);
        maxIndex = nodeIndex;
    }
}

} // namespace

void CostSchedulerKernelInternal::normalizeWeights(
    double *weights, size_t requestCount, size_t nodeCount,
    const std::vector<double> &inverseWeightSums)
{
    std::vector<double> normalizedWeightSums(requestCount, 0.0);
    for (size_t nodeIndex = 0; nodeIndex + 1 < nodeCount; ++nodeIndex) {
        auto weightRow = weights + nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
            auto normalizedWeight =
                weightRow[requestIndex] * inverseWeightSums[requestIndex];
            weightRow[requestIndex] = normalizedWeight;
            normalizedWeightSums[requestIndex] += normalizedWeight;
        }
    }
    auto lastWeightRow = weights + (nodeCount - 1) * requestCount;
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        lastWeightRow[requestIndex] = 1.0 - normalizedWeightSums[requestIndex];
    }
}

bool CostSchedulerKernelInternal::isValidNode(const NaviSchedScheduleNode &node)
{
    return node.hasValidLatencyModel();
}

void CostSchedulerKernelInternal::initializeWeights(double *weights, size_t requestCount,
                                                    size_t nodeCount, uint64_t randomSeed)
{
    std::mt19937_64 random(
        randomSeed > 0 ? randomSeed
                       : static_cast<uint64_t>(autil::TimeUtility::currentTime()));
    std::uniform_real_distribution<double> distribution(1.0, 2.0);
    std::vector<double> randomWeightSums(requestCount, 0.0);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto isLastNode = nodeIndex + 1 == nodeCount;
        auto weightRow = weights + nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
            auto &weight = weightRow[requestIndex];
            weight = distribution(random);
            randomWeightSums[requestIndex] += weight;
            if (isLastNode) {
                randomWeightSums[requestIndex] = 1.0 / randomWeightSums[requestIndex];
            }
        }
    }
    normalizeWeights(weights, requestCount, nodeCount, randomWeightSums);
}

void CostSchedulerKernelInternal::initializeWeights(
    double *weights, const NaviSchedScheduleContext &context, const double *randomWeights,
    size_t randomWeightCount)
{
    if (!randomWeights || 0 == randomWeightCount ||
        0 != (randomWeightCount & (randomWeightCount - 1))) {
        initializeWeights(weights, context.requestCount(), context.nodeCount());
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
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
            auto weight = randomWeights[(randomOffset + requestIndex) & randomWeightMask];
            weightRow[requestIndex] = weight;
            randomWeightSums[requestIndex] += weight;
            if (isLastNode) {
                randomWeightSums[requestIndex] = 1.0 / randomWeightSums[requestIndex];
            }
        }
    }
    normalizeWeights(weights, requestCount, nodeCount, randomWeightSums);
}

bool CostSchedulerKernelInternal::initializeWeightRequestRange(
    double *weights, const NaviSchedScheduleContext &context, const double *randomWeights,
    size_t randomWeightCount, size_t requestBegin, size_t requestEnd)
{
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    if (!weights || !randomWeights || 0 == randomWeightCount ||
        0 != (randomWeightCount & (randomWeightCount - 1)) || requestBegin > requestEnd ||
        requestEnd > requestCount || 0 == nodeCount) {
        return false;
    }
    auto randomWeightMask = randomWeightCount - 1;
    for (size_t requestIndex = requestBegin; requestIndex < requestEnd; ++requestIndex) {
        double randomWeightSum = 0.0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            randomWeightSum += randomWeights[(nodeIndex * requestCount + requestIndex) &
                                             randomWeightMask];
        }
        double normalizedWeightSum = 0.0;
        randomWeightSum = 1.0 / randomWeightSum;
        for (size_t nodeIndex = 0; nodeIndex + 1 < nodeCount; ++nodeIndex) {
            auto index = nodeIndex * requestCount + requestIndex;
            auto normalizedWeight =
                randomWeights[index & randomWeightMask] * randomWeightSum;
            weights[index] = normalizedWeight;
            normalizedWeightSum += normalizedWeight;
        }
        weights[(nodeCount - 1) * requestCount + requestIndex] =
            1.0 - normalizedWeightSum;
    }
    return true;
}

bool CostSchedulerKernelInternal::projectWeights(double *weights, size_t requestCount,
                                                 size_t nodeCount)
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
    auto tileSize = std::min(requestCount, PROJECTION_REQUEST_TILE_SIZE);
    std::vector<double> buffer(nodeCount * tileSize);
    return projectRequestRange(weights, requestCount, nodeCount, 0, requestCount,
                               buffer.data());
}

bool CostSchedulerKernelInternal::projectRequestRange(
    double *weights, size_t requestCount, size_t nodeCount, size_t requestBegin,
    size_t requestEnd, double *workspace, uint32_t *nodeIndexes,
    const GradientSpan *gradientSpans, size_t gradientSpanCount, double alpha,
    uint64_t selectionRandomSeed)
{
    if (!weights || !workspace || 0 == requestCount || 0 == nodeCount ||
        requestBegin >= requestEnd || requestEnd > requestCount ||
        (nodeIndexes && nodeCount > std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    auto applyGradients = gradientSpans && gradientSpanCount > 0;
    if (applyGradients && (!std::isfinite(alpha) || alpha <= 0.0)) {
        return false;
    }
    if (applyGradients) {
        size_t visitedNodeCount = 0;
        for (size_t spanIndex = 0; spanIndex < gradientSpanCount; ++spanIndex) {
            const auto &span = gradientSpans[spanIndex];
            if ((span.nodeBegin != span.nodeEnd && !span.values) ||
                span.nodeBegin != visitedNodeCount || span.nodeEnd < span.nodeBegin ||
                span.nodeEnd > nodeCount) {
                return false;
            }
            visitedNodeCount = span.nodeEnd;
        }
        if (visitedNodeCount != nodeCount) {
            return false;
        }
    }
    for (; requestBegin < requestEnd; requestBegin += PROJECTION_REQUEST_TILE_SIZE) {
        auto tileSize = std::min(PROJECTION_REQUEST_TILE_SIZE, requestEnd - requestBegin);
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> activeSums{};
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> maxValues;
        std::array<size_t, PROJECTION_REQUEST_TILE_SIZE> maxIndexes{};
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> selectionMaxValues;
        std::array<size_t, PROJECTION_REQUEST_TILE_SIZE> selectionMaxIndexes{};
        maxValues.fill(-std::numeric_limits<double>::infinity());
        selectionMaxValues.fill(-std::numeric_limits<double>::infinity());
        if (applyGradients) {
            for (size_t spanIndex = 0; spanIndex < gradientSpanCount; ++spanIndex) {
                const auto &span = gradientSpans[spanIndex];
                for (size_t nodeIndex = span.nodeBegin; nodeIndex < span.nodeEnd;
                     ++nodeIndex) {
                    auto weightRow = weights + nodeIndex * requestCount;
                    auto gradientRow =
                        span.values + (nodeIndex - span.nodeBegin) * requestCount;
                    for (size_t requestOffset = 0; requestOffset < tileSize;
                         ++requestOffset) {
                        auto requestIndex = requestBegin + requestOffset;
                        auto updatedWeight =
                            weightRow[requestIndex] - alpha * gradientRow[requestIndex];
                        workspace[requestOffset * nodeCount + nodeIndex] = updatedWeight;
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
                for (size_t requestOffset = 0; requestOffset < tileSize;
                     ++requestOffset) {
                    auto value = weightRow[requestBegin + requestOffset];
                    workspace[requestOffset * nodeCount + nodeIndex] = value;
                    activeSums[requestOffset] += value;
                    if (value > maxValues[requestOffset]) {
                        maxValues[requestOffset] = value;
                        maxIndexes[requestOffset] = nodeIndex;
                    }
                }
            }
        }
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> thresholds{};
        for (size_t requestOffset = 0; requestOffset < tileSize; ++requestOffset) {
            if (!findSimplexThreshold(workspace + requestOffset * nodeCount, nodeCount,
                                      activeSums[requestOffset],
                                      thresholds[requestOffset])) {
                return false;
            }
        }
        std::array<double, PROJECTION_REQUEST_TILE_SIZE> projectedSums{};
        if (applyGradients) {
            for (size_t spanIndex = 0; spanIndex < gradientSpanCount; ++spanIndex) {
                const auto &span = gradientSpans[spanIndex];
                for (size_t nodeIndex = span.nodeBegin; nodeIndex < span.nodeEnd;
                     ++nodeIndex) {
                    auto weightRow = weights + nodeIndex * requestCount;
                    auto gradientRow =
                        span.values + (nodeIndex - span.nodeBegin) * requestCount;
                    for (size_t requestOffset = 0; requestOffset < tileSize;
                         ++requestOffset) {
                        auto requestIndex = requestBegin + requestOffset;
                        auto projectedWeight =
                            std::max(0.0, weightRow[requestIndex] -
                                              alpha * gradientRow[requestIndex] -
                                              thresholds[requestOffset]);
                        weightRow[requestIndex] = projectedWeight;
                        projectedSums[requestOffset] += projectedWeight;
                        updateSelectionMaximum(nodeCount, requestIndex, nodeIndex,
                                               projectedWeight, selectionRandomSeed,
                                               selectionMaxValues[requestOffset],
                                               selectionMaxIndexes[requestOffset]);
                    }
                }
            }
        } else {
            for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
                auto weightRow = weights + nodeIndex * requestCount;
                for (size_t requestOffset = 0; requestOffset < tileSize;
                     ++requestOffset) {
                    auto requestIndex = requestBegin + requestOffset;
                    auto projectedWeight = std::max(
                        0.0, weightRow[requestIndex] - thresholds[requestOffset]);
                    weightRow[requestIndex] = projectedWeight;
                    projectedSums[requestOffset] += projectedWeight;
                    updateSelectionMaximum(nodeCount, requestIndex, nodeIndex,
                                           projectedWeight, selectionRandomSeed,
                                           selectionMaxValues[requestOffset],
                                           selectionMaxIndexes[requestOffset]);
                }
            }
        }
        for (size_t requestOffset = 0; requestOffset < tileSize; ++requestOffset) {
            weights[maxIndexes[requestOffset] * requestCount + requestBegin +
                    requestOffset] += 1.0 - projectedSums[requestOffset];
            if (nodeIndexes) {
                nodeIndexes[requestBegin + requestOffset] =
                    static_cast<uint32_t>(selectionMaxIndexes[requestOffset]);
            }
        }
    }
    return true;
}

bool CostSchedulerKernelInternal::selectWeightRequestRange(
    const double *weights, size_t requestCount, size_t nodeCount, size_t requestBegin,
    size_t requestEnd, uint32_t *nodeIndexes, uint64_t selectionRandomSeed)
{
    if (!weights || !nodeIndexes || nodeCount == 0 ||
        nodeCount > std::numeric_limits<uint32_t>::max() || requestBegin > requestEnd ||
        requestEnd > requestCount) {
        return false;
    }
    for (size_t requestIndex = requestBegin; requestIndex < requestEnd; ++requestIndex) {
        double maxWeight = -std::numeric_limits<double>::infinity();
        size_t maxIndex = 0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            updateSelectionMaximum(nodeCount, requestIndex, nodeIndex,
                                   weights[nodeIndex * requestCount + requestIndex],
                                   selectionRandomSeed, maxWeight, maxIndex);
        }
        nodeIndexes[requestIndex] = static_cast<uint32_t>(maxIndex);
    }
    return true;
}

bool CostSchedulerKernelInternal::findSimplexThreshold(double *workspace, size_t count,
                                                       double activeSum,
                                                       double &threshold)
{
    if (!workspace || 0 == count || !std::isfinite(activeSum)) {
        return false;
    }
    auto activeCount = count;
    while (activeCount > 0) {
        threshold = (activeSum - 1.0) / static_cast<double>(activeCount);
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

double CostSchedulerKernelInternal::getStepAlpha(double alpha, double alphaDecay,
                                                 double minAlpha, size_t iteration)
{
    auto stepAlpha = std::max(minAlpha, alpha * std::pow(alphaDecay, iteration));
    return stepAlpha > 0.0 ? stepAlpha : std::numeric_limits<double>::min();
}

bool CostSchedulerKernelInternal::applyGradient(double *weights, size_t weightCount,
                                                const std::vector<double> &gradients,
                                                double alpha)
{
    return applyGradient(weights, 0, weightCount, gradients, alpha);
}

bool CostSchedulerKernelInternal::applyGradient(double *weights, size_t weightOffset,
                                                size_t weightCount,
                                                const std::vector<double> &gradients,
                                                double alpha)
{
    if (!weights || gradients.size() != weightCount || !std::isfinite(alpha) ||
        alpha <= 0.0) {
        return false;
    }
    return applyGradient(weights, weightOffset, weightCount, gradients.data(), alpha);
}

bool CostSchedulerKernelInternal::applyGradient(double *weights, size_t weightOffset,
                                                size_t weightCount,
                                                const double *gradients, double alpha)
{
    if (!weights || !gradients || !std::isfinite(alpha) || alpha <= 0.0) {
        return false;
    }
    for (size_t index = 0; index < weightCount; ++index) {
        auto updatedWeight = weights[weightOffset + index] - alpha * gradients[index];
        if (!std::isfinite(updatedWeight)) {
            return false;
        }
        weights[weightOffset + index] = updatedWeight;
    }
    return true;
}

bool CostSchedulerKernelInternal::selectNodes(NaviSchedScheduleContext &context,
                                              const uint32_t *nodeIndexes)
{
    if (!nodeIndexes) {
        return false;
    }
    for (size_t requestIndex = 0; requestIndex < context.requestCount(); ++requestIndex) {
        if (!context.selectNode(requestIndex, nodeIndexes[requestIndex],
                                "cost_selected")) {
            return false;
        }
    }
    return true;
}

int64_t CostSchedulerKernelInternal::fillRequestTokenCounts(
    const NaviSchedScheduleContext &context, int64_t *requestTokenCounts,
    double *inverseRequestTokenCounts, double latencyTokenNormalizationPower)
{
    int64_t totalTokenCount = 0;
    for (size_t requestIndex = 0; requestIndex < context.requestCount(); ++requestIndex) {
        auto tokenCount = std::max<int64_t>(
            1, context.getRequest(requestIndex).request.promptTokenCount);
        requestTokenCounts[requestIndex] = tokenCount;
        if (inverseRequestTokenCounts) {
            inverseRequestTokenCounts[requestIndex] =
                1.0 /
                std::pow(static_cast<double>(tokenCount), latencyTokenNormalizationPower);
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
            static_cast<double>(std::max<int64_t>(1, requestTokenCounts[requestIndex])),
            latencyTokenNormalizationPower);
    }
    return std::isfinite(denominator) && denominator > 0.0 ? denominator : 0.0;
}

std::unique_ptr<float[]> CostSchedulerKernelInternal::prepare(
    NaviSchedScheduleContext &context, const int64_t *requestTokenCounts, double lambda,
    double alpha, double alphaDecay, double minAlpha, size_t maxLoopCount,
    int64_t optimizationTimeBudgetUs, ReplayInfo *replayInfo)
{
    auto nodeCount = context.nodeCount();
    auto requestCount = context.requestCount();
    if (nodeCount == 0 || requestCount == 0 ||
        requestCount > std::numeric_limits<size_t>::max() / nodeCount ||
        requestCount * nodeCount > std::numeric_limits<size_t>::max() / 2) {
        return nullptr;
    }
    const auto matrixSize = requestCount * nodeCount;
    context.prepareCacheMatchPrefixes();
    std::unique_ptr<float[]> costs(new float[matrixSize * 2]);
    if (!fillCostsForNodeRange(context, requestTokenCounts, 0, nodeCount, costs.get())) {
        return nullptr;
    }
    if (replayInfo) {
        replayInfo->init(context.isCollectingReplay(), lambda, alpha, alphaDecay,
                         minAlpha, maxLoopCount, optimizationTimeBudgetUs);
    }
    return costs;
}

bool CostSchedulerKernelInternal::fillCostsForNodeRange(
    const NaviSchedScheduleContext &context, const int64_t *requestTokenCounts,
    size_t nodeBegin, size_t nodeEnd, float *costs, PreparationTiming *timing)
{
    if (timing) {
        *timing = {};
    }
    auto requestCount = context.requestCount();
    if (!requestTokenCounts || !costs || nodeBegin > nodeEnd ||
        nodeEnd > context.nodeCount()) {
        return false;
    }
    std::vector<int64_t> cacheHitBlockCounts;
    const auto matrixSize = requestCount * context.nodeCount();
    for (size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex) {
        const auto &node = context.getNode(nodeIndex);
        if (!isValidNode(node)) {
            return false;
        }
        auto costRow = costs + nodeIndex * requestCount;
        auto waitingCostRow = costRow + matrixSize;
        if (NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            if (!context.getGeometryOptimizationInputs(nodeIndex)) {
                return false;
            }
            std::fill(costRow, costRow + requestCount, 0.0f);
            std::fill(waitingCostRow, waitingCostRow + requestCount, 0.0f);
            continue;
        }
        const auto matchBeginUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
        if (!context.getCacheHitBlockCounts(nodeIndex, cacheHitBlockCounts)) {
            return false;
        }
        const auto matchEndUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
        const auto &latencyParameters = node.getLatencyParameters();
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
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
            timing->costEvaluationUs +=
                autil::TimeUtility::monotonicTimeUs() - matchEndUs;
            ++timing->completedNodes;
        }
    }
    return true;
}

namespace {
bool forwardContinuousGeometry(const NaviSchedScheduleContext &context, size_t nodeIndex,
                               const double *weights, const int64_t *tokens,
                               const double *inverseTokens, double firstScale,
                               double secondScale, double concavity, double &firstTerm,
                               double &secondTerm,
                               CostSchedulerKernelInternal::NodeGradient *gradient)
{
    const auto &node = context.getNode(nodeIndex);
    const auto &policy = node.getPrefillExecutionPolicy();
    if (!policy) {
        return false;
    }
    const auto &parameters = node.getLatencyParameters();
    const auto inputs = context.getGeometryOptimizationInputs(nodeIndex);
    if (!inputs) {
        return false;
    }
    const bool projectedLookup = NonLinearPrefillModelR::usesProjectedGeometryForVersion(
        node.getLatencyBinaryVersion());
    const bool cacheSource =
        NonLinearPrefillModelR::usesCacheSourceForVersion(node.getLatencyBinaryVersion());
    double linear = 0.0, sharedTokens = 0.0, sharedWeight = 0.0;
    double fixedChunkSum = 0.0;
    double fixedPairs = 0.0, maximumFixedChunks = 0.0;
    size_t maximumFixedChunkRequest = std::numeric_limits<size_t>::max();
    double weightSum = 0.0, totalPrompt = 0.0, inverseSum = 0.0;
    double hitMemberWeight = 0.0, lookupMemberWeight = 0.0;
    double externalReadWeight = 0.0, maximumExternalRead = 0.0;
    size_t maximumExternalReadRequest = std::numeric_limits<size_t>::max();
    auto accumulate = [&](auto lookupEnabled) -> bool {
        for (size_t i = 0; i < context.requestCount(); ++i) {
            const auto *input = context.getGeometryOptimizationInput(*inputs, i);
            if (!input) {
                return false;
            }
            linear += weights[i] * input->linearCost;
            sharedTokens += weights[i] * input->sharedTokens;
            sharedWeight += input->shared ? weights[i] : 0.0;
            fixedPairs += weights[i] * input->fixedAttentionPairs;
            fixedChunkSum += weights[i] * input->fixedChunkCount;
            if (weights[i] * input->fixedChunkCount > maximumFixedChunks) {
                maximumFixedChunks = weights[i] * input->fixedChunkCount;
                maximumFixedChunkRequest = i;
            }
            weightSum += weights[i];
            if constexpr (decltype(lookupEnabled)::value) {
                if (node.hasQueueLookupOpportunity(tokens[i])) {
                    lookupMemberWeight += weights[i];
                }
            }
            if (input->cacheHit) {
                hitMemberWeight += weights[i];
            }
            totalPrompt += weights[i] * tokens[i];
            inverseSum += weights[i] * inverseTokens[i];
            if (cacheSource && input->externalReadTokens > 0.0) {
                externalReadWeight += weights[i];
                const auto read = weights[i] * input->externalReadTokens;
                if (read > maximumExternalRead) {
                    maximumExternalRead = read;
                    maximumExternalReadRequest = i;
                }
            }
        }
        return true;
    };
    if (projectedLookup) {
        if (!accumulate(std::false_type{})) {
            return false;
        }
    } else if (!accumulate(std::true_type{})) {
        return false;
    }
    // Projected geometry already partitions exactly these prompt-only groups.
    if (projectedLookup) {
        lookupMemberWeight = weightSum - sharedWeight;
    }
    // Shared ready requests use a fractional batch activation and token budget.
    // Async lookup groups are independent; their single-group geometry was
    // projected once and is weighted linearly in the cached request input.
    const auto activation = std::min(1.0, sharedWeight);
    const auto activationDerivative = sharedWeight < 1.0 ? 1.0 : 0.0;
    const auto tokenSteps = sharedTokens / policy->maxScheduledTokens;
    const auto usesTokenSteps = tokenSteps > activation;
    const double pairSteps = policy->attentionScheduleCapacity > 0
                                 ? fixedPairs / policy->attentionScheduleCapacity
                                 : 0.0;
    const double baseSteps = usesTokenSteps ? tokenSteps : activation;
    const double inputSteps =
        policy->fixedPrefillChunkPlan
            ? (sharedTokens + fixedChunkSum * policy->draftSlotsPerRequest) /
                  policy->inputTokenBudget()
            : 0.0;
    const bool useFixedChunks =
        policy->fixedPrefillChunkPlan && maximumFixedChunks > baseSteps &&
        maximumFixedChunks >= pairSteps && maximumFixedChunks >= inputSteps;
    const bool usePairSteps = policy->fixedPrefillChunkPlan && pairSteps > baseSteps &&
                              pairSteps > maximumFixedChunks && pairSteps >= inputSteps;
    const bool useInputSteps = policy->fixedPrefillChunkPlan && inputSteps > baseSteps &&
                               inputSteps > maximumFixedChunks && inputSteps > pairSteps;
    // Differentiate the winning continuous bound; final assignments use exact
    // whole-chunk token/input/draft-slot/attention packing.
    const auto steps =
        useFixedChunks
            ? maximumFixedChunks
            : (usePairSteps ? pairSteps : (useInputSteps ? inputSteps : baseSteps));
    const auto stepTokenDerivative =
        (useFixedChunks || usePairSteps)
            ? 0.0
            : (useInputSteps ? 1.0 / policy->inputTokenBudget()
                             : (usesTokenSteps ? 1.0 / policy->maxScheduledTokens : 0.0));
    const auto stepWeightDerivative = (useFixedChunks || usePairSteps || useInputSteps)
                                          ? 0.0
                                          : (usesTokenSteps ? 0.0 : activationDerivative);
    const auto minimumPadding =
        static_cast<double>(policy->pcpSize) * policy->compileSizes.front();
    const auto usesRawTokens = sharedTokens > minimumPadding * steps;
    const auto padding = usesRawTokens ? sharedTokens : minimumPadding * steps;
    const auto paddingTokenDerivative =
        usesRawTokens ? 1.0 : minimumPadding * stepTokenDerivative;
    const auto paddingWeightDerivative =
        usesRawTokens ? 0.0 : minimumPadding * stepWeightDerivative;
    const auto paddingCoefficient =
        parameters[3] * NonLinearPrefillModelR::getParameterScale(3) / 1024.0;
    const auto stepCoefficient =
        parameters[6] * NonLinearPrefillModelR::getParameterScale(6);
    linear += paddingCoefficient * padding + stepCoefficient * steps;
    double linearDerivative = 0.0, promptDerivative = 0.0;
    // The existing formula adds one constant batch term. Replace that term by
    // the fractional step coordinate already included above, without changing
    // the nonlinear prediction function or its parameter scales.
    auto latency =
        NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
            parameters, linear - stepCoefficient, totalPrompt, 0.0, linearDerivative,
            promptDerivative);
    // Continuous, piecewise-linear source coordinates. At every integer
    // assignment these are the true any-read indicator and maximum read size.
    // A tie in the maximum chooses its first valid subgradient deterministically.
    if (cacheSource) {
        latency +=
            parameters[NonLinearPrefillModelR::EXTERNAL_READ_ACTIVATION_PARAMETER_INDEX] *
                std::min(1.0, externalReadWeight) +
            parameters[NonLinearPrefillModelR::EXTERNAL_READ_VOLUME_PARAMETER_INDEX] *
                maximumExternalRead / 1024.0;
    }
    double residualDerivative = 1.0;
    latency = node.applyLatencyResidual(latency, &residualDerivative);
    linearDerivative *= residualDerivative;
    promptDerivative *= residualDerivative;
    double queueHitDerivative = 0.0, queueBatchDerivative = 0.0,
           queueLookupDerivative = 0.0;
    const auto queueCost =
        latency + node.getEstimatedQueueWaitMs(
                      hitMemberWeight, weightSum, lookupMemberWeight, &queueHitDerivative,
                      &queueBatchDerivative, &queueLookupDerivative);
    const auto logScale = concavity > 0.0 ? std::log(concavity) : 1.0;
    const auto latencyWeight =
        concavity > 0.0 ? std::log1p(concavity * inverseSum) / logScale : inverseSum;
    firstTerm = latency * weightSum;
    secondTerm = queueCost * latencyWeight;
    if (!std::isfinite(firstTerm) || !std::isfinite(secondTerm)) {
        return false;
    }
    if (gradient) {
        const auto latencyGradient = firstScale * weightSum + secondScale * latencyWeight;
        gradient->cost = linearDerivative * latencyGradient;
        gradient->totalPromptToken = promptDerivative * latencyGradient;
        gradient->weight =
            firstScale * latency + secondScale * latencyWeight * queueBatchDerivative;
        gradient->queueHitMember = secondScale * latencyWeight * queueHitDerivative;
        gradient->queueLookupMember = secondScale * latencyWeight * queueLookupDerivative;
        gradient->inverseToken =
            secondScale * queueCost *
            (concavity > 0.0 ? concavity / ((1.0 + concavity * inverseSum) * logScale)
                             : 1.0);
        gradient->geometrySharedToken = paddingCoefficient * paddingTokenDerivative +
                                        stepCoefficient * stepTokenDerivative;
        gradient->geometrySharedWeight = paddingCoefficient * paddingWeightDerivative +
                                         stepCoefficient * stepWeightDerivative;
        const auto fixedStepDerivative =
            stepCoefficient + (usesRawTokens ? 0.0 : paddingCoefficient * minimumPadding);
        if (useFixedChunks) {
            gradient->geometryFixedChunks = fixedStepDerivative;
            gradient->geometryFixedChunkRequest = maximumFixedChunkRequest;
        }
        if (useInputSteps) {
            gradient->geometryFixedInputChunks = fixedStepDerivative *
                                                 policy->draftSlotsPerRequest /
                                                 policy->inputTokenBudget();
        }
        if (usePairSteps) {
            gradient->geometryFixedAttentionPairs =
                fixedStepDerivative / policy->attentionScheduleCapacity;
        }
        gradient->capacitySplit = false;
        if (cacheSource) {
            gradient->externalReadMember =
                externalReadWeight < 1.0
                    ? parameters[NonLinearPrefillModelR::
                                     EXTERNAL_READ_ACTIVATION_PARAMETER_INDEX] *
                          residualDerivative * latencyGradient
                    : 0.0;
            gradient->externalReadMaximum =
                parameters[NonLinearPrefillModelR::EXTERNAL_READ_VOLUME_PARAMETER_INDEX] /
                1024.0 * residualDerivative * latencyGradient;
            gradient->externalReadMaximumRequest = maximumExternalReadRequest;
        }
    }
    return true;
}
} // namespace

double CostSchedulerKernelInternal::forwardNodeRange(
    const NaviSchedScheduleContext &context, const float *costs, const double *weights,
    const int64_t *requestTokenCounts, const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator, double lambda, double latencyWeightConcavity,
    size_t nodeBegin, size_t nodeEnd, NodeGradient *nodeGradients, double *firstTerms,
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
        if (nodeGradients) {
            nodeGradients[nodeIndex - nodeBegin] = NodeGradient();
        }
        if (NonLinearPrefillModelR::usesGeometryForVersion(
                context.getNode(nodeIndex).getLatencyBinaryVersion())) {
            double firstTerm = 0.0, secondTerm = 0.0;
            if (!forwardContinuousGeometry(
                    context, nodeIndex, weights + rowOffset, requestTokenCounts,
                    inverseRequestTokenCounts, firstScale, secondScale,
                    latencyWeightConcavity, firstTerm, secondTerm,
                    nodeGradients ? nodeGradients + nodeIndex - nodeBegin : nullptr)) {
                return std::numeric_limits<double>::infinity();
            }
            if (firstTerms) {
                firstTerms[nodeIndex] = firstTerm;
            }
            if (secondTerms) {
                secondTerms[nodeIndex] = secondTerm;
            }
            sum1 += firstTerm;
            sum2 += secondTerm;
            continue;
        }
        auto costRow = costs + rowOffset;
        auto waitingCostRow = costRow + matrixSize;
        auto weightRow = weights + rowOffset;
        double requestCostSum = 0.0;
        double totalPromptTokenSum = 0.0;
        double weightSum = 0.0;
        double inverseTokenWeightSum = 0.0;
        double hitMemberWeight = 0.0, lookupMemberWeight = 0.0;
        const auto &node = context.getNode(nodeIndex);
        const auto queueInputs = node.usesCacheMemberQueuePrediction()
                                     ? context.getCacheHitBlockCountsSnapshot(nodeIndex)
                                     : nullptr;
        if (node.usesCacheMemberQueuePrediction() && !queueInputs) {
            return std::numeric_limits<double>::infinity();
        }
        // Keep policy dispatch outside the request loop.
        auto accumulate = [&](auto lookupEnabled) {
            for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
                auto weight = weightRow[requestIndex];
                totalPromptTokenSum += weight * requestTokenCounts[requestIndex];
                weightSum += weight;
                if constexpr (decltype(lookupEnabled)::value) {
                    if (node.hasQueueLookupOpportunity(
                            requestTokenCounts[requestIndex])) {
                        lookupMemberWeight += weight;
                    }
                }
                if (queueInputs &&
                    context.getCacheHitBlockCount(*queueInputs, requestIndex) > 0) {
                    hitMemberWeight += weight;
                }
                inverseTokenWeightSum += weight * inverseRequestTokenCounts[requestIndex];
            }
        };
        if (node.hasQueueLookupOpportunity(std::numeric_limits<int64_t>::max())) {
            accumulate(std::true_type{});
        } else {
            accumulate(std::false_type{});
        }
        const auto capacitySplit = NonLinearPrefillModelR::usesCapacitySplitParameters(
            node.getLatencyParameters());
        const auto workloadRunningCapacity =
            capacitySplit ? std::max(0, node.getMaxRunningRequestCount() -
                                            node.getWorkloadRunningCapacitySlotCount())
                          : 0;
        const auto reservedRunningCapacity =
            capacitySplit ? std::max(0, node.getMaxRunningRequestCount() -
                                            node.getReservedRunningCapacitySlotCount())
                          : 0;
        if (capacitySplit) {
            requestCostSum =
                aggregateCapacitySplitCost(costRow, waitingCostRow, weightRow,
                                           requestCount, workloadRunningCapacity);
        } else {
            for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
                requestCostSum += weightRow[requestIndex] * costRow[requestIndex];
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
                // Equal capacities use the same immutable rows and weights.
                const auto reservedRequestCost =
                    reservedRunningCapacity == workloadRunningCapacity
                        ? requestCostSum
                        : aggregateCapacitySplitCost(costRow, waitingCostRow, weightRow,
                                                     requestCount,
                                                     reservedRunningCapacity);
                requestCostWithoutReportedWorkload =
                    reservedRequestCost + NonLinearPrefillModelR::calculateLinearCost(
                                              node.getLatencyParameters(), floorWorkload);
                requestCostSum += NonLinearPrefillModelR::calculateLinearCost(
                    node.getLatencyParameters(), node.getWorkloadFeatures());
            } else {
                requestCostWithoutReportedWorkload +=
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(), floorWorkload);
                requestCostSum =
                    requestCostWithoutReportedWorkload +
                    NonLinearPrefillModelR::calculateLinearCost(
                        node.getLatencyParameters(), node.getReportedWorkloadFeatures());
            }
            promptTokensWithoutReportedWorkload += floorWorkload.getCacheHitTokenCount() +
                                                   floorWorkload.getCacheMissTokenCount();
            const auto &reportedWorkload = node.getReportedWorkloadFeatures();
            totalPromptTokenSum = promptTokensWithoutReportedWorkload +
                                  reportedWorkload.getCacheHitTokenCount() +
                                  reportedWorkload.getCacheMissTokenCount();
            const auto candidateBatchActivation = std::min(1.0, std::max(0.0, weightSum));
            const auto batchCost =
                node.getLatencyParameters()
                    [NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
                NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX);
            const auto batchCostAdjustment = batchCost * (candidateBatchActivation - 1.0);
            requestCostSum += batchCostAdjustment;
            requestCostWithoutReportedWorkload += batchCostAdjustment;
        }
        double candidateBatchActivationDerivative = 0.0;
        if (node.usesUnifiedLatencyPrediction() && weightSum >= 0.0 && weightSum < 1.0) {
            candidateBatchActivationDerivative =
                node.getLatencyParameters()
                    [NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
                NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX);
        }
        double nodeCostDerivative = 0.0;
        double totalPromptTokenDerivative = 0.0;
        auto nodeCost =
            NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
                node.getLatencyParameters(), requestCostSum, totalPromptTokenSum,
                precedingRequestsWithReportedWorkload, nodeCostDerivative,
                totalPromptTokenDerivative);
        double residualDerivativeFactor = 1.0;
        nodeCost = node.applyLatencyResidual(nodeCost, &residualDerivativeFactor);
        nodeCostDerivative *= residualDerivativeFactor;
        totalPromptTokenDerivative *= residualDerivativeFactor;
        if (node.usesUnifiedLatencyPrediction()) {
            double floorCostDerivative = 0.0;
            double floorPromptDerivative = 0.0;
            auto floorCost =
                NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
                    node.getLatencyParameters(), requestCostWithoutReportedWorkload,
                    promptTokensWithoutReportedWorkload,
                    precedingRequestsWithoutReportedWorkload, floorCostDerivative,
                    floorPromptDerivative);
            double floorResidualDerivativeFactor = 1.0;
            floorCost =
                node.applyLatencyResidual(floorCost, &floorResidualDerivativeFactor);
            floorCostDerivative *= floorResidualDerivativeFactor;
            floorPromptDerivative *= floorResidualDerivativeFactor;
            bool usesReportedWorkload = true;
            nodeCost =
                node.decayUnifiedLatency(nodeCost, floorCost, &usesReportedWorkload);
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
        double queueHitDerivative = 0.0, queueBatchDerivative = 0.0,
               queueLookupDerivative = 0.0;
        auto queueCost = nodeCost + node.getEstimatedQueueWaitMs(
                                        hitMemberWeight, weightSum, lookupMemberWeight,
                                        &queueHitDerivative, &queueBatchDerivative,
                                        &queueLookupDerivative);
        auto firstTerm = node.usesUnifiedLatencyPrediction() ? nodeCost - existingNodeCost
                                                             : nodeCost * weightSum;
        auto latencyWeightLogScale =
            latencyWeightConcavity > 0.0 ? std::log(latencyWeightConcavity) : 1.0;
        auto latencyWeight =
            latencyWeightConcavity > 0.0
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
                firstScale * (node.usesUnifiedLatencyPrediction() ? 1.0 : weightSum) +
                secondScale * latencyWeight;
            gradient.cost = nodeCostDerivative * latencyGradient;
            gradient.totalPromptToken = totalPromptTokenDerivative * latencyGradient;
            gradient.weight =
                node.usesUnifiedLatencyPrediction() ? 0.0 : firstScale * nodeCost;
            gradient.weight += gradient.cost * candidateBatchActivationDerivative;
            gradient.weight += secondScale * latencyWeight * queueBatchDerivative;
            gradient.queueHitMember = secondScale * latencyWeight * queueHitDerivative;
            gradient.queueLookupMember =
                secondScale * latencyWeight * queueLookupDerivative;
            gradient.inverseToken =
                secondScale * queueCost *
                (latencyWeightConcavity > 0.0
                     ? latencyWeightConcavity /
                           ((1.0 + latencyWeightConcavity * inverseTokenWeightSum) *
                            latencyWeightLogScale)
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
    const double *inverseRequestTokenCounts, double tokenNormalizationDenominator,
    double lambda, double latencyWeightConcavity, DiscreteForwardWorkspace &workspace)
{
    return forwardDiscreteImpl(context, costs, nodeIndexes, requestTokenCounts,
                               inverseRequestTokenCounts, tokenNormalizationDenominator,
                               lambda, latencyWeightConcavity, workspace,
                               context.nodeCount(), context.nodeCount());
}

double CostSchedulerKernelInternal::forwardDiscreteImpl(
    const NaviSchedScheduleContext &context, const float *costs,
    const uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
    const double *inverseRequestTokenCounts, double tokenNormalizationDenominator,
    double lambda, double latencyWeightConcavity, DiscreteForwardWorkspace &workspace,
    size_t firstNode, size_t secondNode,
    const std::vector<std::vector<size_t>> *nodeRequests, size_t movedRequest,
    bool reuseSourceNode)
{
    workspace.nodeCapacityLoads = 0;
    workspace.nodeBaselineLoads = 0;
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    if (!costs || !nodeIndexes || !requestTokenCounts || !inverseRequestTokenCounts ||
        requestCount == 0 || nodeCount == 0 || tokenNormalizationDenominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const bool pairOnly = firstNode < nodeCount;
    reuseSourceNode = reuseSourceNode && pairOnly && nodeRequests && firstNode != secondNode;
    if (pairOnly && secondNode >= nodeCount) {
        return std::numeric_limits<double>::infinity();
    }
    // A full evaluation starts a new immutable snapshot; pair trials reuse it.
    if (!pairOnly || workspace.baselineContext != &context ||
        workspace.nodeBaselines.size() != nodeCount) {
        workspace.baselineContext = &context;
        workspace.nodeBaselines.assign(nodeCount, {});
    }
    const auto baselineFor =
        [&](size_t nodeIndex) -> const DiscreteForwardWorkspace::NodeBaseline & {
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
                    baseline.workloadLinearCost =
                        NonLinearPrefillModelR::calculateLinearCost(
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
    reset(workspace.hitMemberWeightSums);
    reset(workspace.lookupMemberWeightSums);
    if (!pairOnly || workspace.cacheCountSnapshots.size() != nodeCount)
        workspace.cacheCountSnapshots.assign(nodeCount, nullptr);
    auto &geometryRequests = workspace.geometryRequests;
    geometryRequests.resize(nodeCount);
    if (pairOnly) {
        if (!reuseSourceNode) geometryRequests[firstNode].clear();
        geometryRequests[secondNode].clear();
    } else {
        for (auto &requests : geometryRequests) requests.clear();
    }
    const auto accumulateRequest = [&](size_t nodeIndex, size_t requestIndex) -> bool {
        const auto &node = context.getNode(nodeIndex);
        if (NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion()))
            geometryRequests[nodeIndex].emplace_back(requestIndex);

        ++workspace.nodeCapacityLoads;
        const auto &baseline = baselineFor(nodeIndex);
        const auto costIndex = nodeIndex * requestCount + requestIndex;
        const auto waitingCostIndex = nodeCount * requestCount + costIndex;
        const auto assignedRequestCount = workspace.requestWeightSums[nodeIndex];
        const auto usesRunningCapacity =
            !baseline.capacitySplit ||
            baseline.workloadSlots + assignedRequestCount < baseline.maxRunning;
        const auto usesReservedRunningCapacity =
            !baseline.capacitySplit ||
            baseline.reservedSlots + assignedRequestCount < baseline.maxRunning;
        if (!NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion())) {
        workspace.requestCostSums[nodeIndex] +=
            costs[usesRunningCapacity ? costIndex : waitingCostIndex];
        workspace.requestCostSumsWithoutReportedWorkload[nodeIndex] +=
            costs[usesReservedRunningCapacity ? costIndex : waitingCostIndex];
        }
        workspace.totalPromptTokenSums[nodeIndex] += requestTokenCounts[requestIndex];
        workspace.requestWeightSums[nodeIndex] += 1.0;
        workspace.lookupMemberWeightSums[nodeIndex] +=
            node.hasQueueLookupOpportunity(requestTokenCounts[requestIndex]);
        if (node.usesCacheMemberQueuePrediction()) {
            auto &cacheCounts = workspace.cacheCountSnapshots[nodeIndex];
            if (!cacheCounts) {
                cacheCounts = context.getCacheHitBlockCountsSnapshot(nodeIndex);
            }
            if (!cacheCounts) {
                return false;
            }
            workspace.hitMemberWeightSums[nodeIndex] +=
                context.getCacheHitBlockCount(*cacheCounts, requestIndex) > 0;
        }
        workspace.inverseTokenWeightSums[nodeIndex] +=
            inverseRequestTokenCounts[requestIndex];
        return true;
    };
    if (pairOnly && nodeRequests) {
        // Membership is ordered by original request index. Virtually remove
        // the moved request from its source and merge it into its destination.
        // This preserves running-slot promotion and floating-point sum order.
        const auto accumulateNode = [&](size_t nodeIndex, bool source) -> bool {
            const auto &node = context.getNode(nodeIndex);
            const bool geometry = NonLinearPrefillModelR::usesGeometryForVersion(node.getLatencyBinaryVersion());
            auto &cacheCounts = workspace.cacheCountSnapshots[nodeIndex];
            if (node.usesCacheMemberQueuePrediction() && !cacheCounts) cacheCounts = context.getCacheHitBlockCountsSnapshot(nodeIndex);
            if (node.usesCacheMemberQueuePrediction() && !cacheCounts) return false;
            double hitMembers = workspace.hitMemberWeightSums[nodeIndex];
            double lookupMembers = workspace.lookupMemberWeightSums[nodeIndex];
            ++workspace.nodeCapacityLoads;
            const auto &baseline = baselineFor(nodeIndex);
            const auto capacitySplit = baseline.capacitySplit;
            const auto workloadSlots = baseline.workloadSlots;
            const auto reservedSlots = baseline.reservedSlots;
            const auto maxRunning = baseline.maxRunning;
            const auto runningCosts = costs + nodeIndex * requestCount;
            const auto waitingCosts = runningCosts + nodeCount * requestCount;
            auto requestCost = workspace.requestCostSums[nodeIndex];
            auto floorRequestCost =
                workspace.requestCostSumsWithoutReportedWorkload[nodeIndex];
            auto totalTokens = workspace.totalPromptTokenSums[nodeIndex];
            auto assignedRequests = workspace.requestWeightSums[nodeIndex];
            auto inverseTokens = workspace.inverseTokenWeightSums[nodeIndex];
            // Keep sums local so each request does not reload/store five vectors.
            const auto append = [&](size_t requestIndex) {
                if (geometry) geometryRequests[nodeIndex].push_back(requestIndex);
                lookupMembers += node.hasQueueLookupOpportunity(requestTokenCounts[requestIndex]);
                if (cacheCounts) hitMembers += context.getCacheHitBlockCount(*cacheCounts, requestIndex) > 0;
                if (!geometry) {
                requestCost +=
                    (!capacitySplit || workloadSlots + assignedRequests < maxRunning)
                        ? runningCosts[requestIndex]
                        : waitingCosts[requestIndex];
                floorRequestCost +=
                    (!capacitySplit || reservedSlots + assignedRequests < maxRunning)
                        ? runningCosts[requestIndex]
                        : waitingCosts[requestIndex];
                }
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
            workspace.requestCostSumsWithoutReportedWorkload[nodeIndex] =
                floorRequestCost;
            workspace.totalPromptTokenSums[nodeIndex] = totalTokens;
            workspace.requestWeightSums[nodeIndex] = assignedRequests;
            workspace.inverseTokenWeightSums[nodeIndex] = inverseTokens;
            workspace.hitMemberWeightSums[nodeIndex] = hitMembers;
            workspace.lookupMemberWeightSums[nodeIndex] = lookupMembers;
            return true;
        };
        if (!reuseSourceNode) {
            if (!accumulateNode(firstNode, true)) return std::numeric_limits<double>::infinity();
        }
        if (!accumulateNode(secondNode, false)) return std::numeric_limits<double>::infinity();
    } else {
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
            const auto nodeIndex = static_cast<size_t>(nodeIndexes[requestIndex]);
            if (nodeIndex >= nodeCount) {
                return std::numeric_limits<double>::infinity();
            }
            if (!pairOnly || nodeIndex == firstNode || nodeIndex == secondNode) {
                if (!accumulateRequest(nodeIndex, requestIndex)) return std::numeric_limits<double>::infinity();
            }
        }
    }
    auto firstScale = lambda / tokenNormalizationDenominator;
    auto secondScale = (1.0 - lambda) / static_cast<double>(requestCount);
    auto latencyWeightLogScale =
        latencyWeightConcavity > 0.0 ? std::log(latencyWeightConcavity) : 1.0;
    double result = reuseSourceNode ? workspace.nodeValues[firstNode] : 0.0;
    const auto evaluatedNodeCount = pairOnly ? (firstNode == secondNode ? size_t{1} : size_t{2}) : nodeCount;
    for (size_t index = reuseSourceNode ? 1 : 0; index < evaluatedNodeCount; ++index) {
        const auto nodeIndex = pairOnly ? (index == 0 ? firstNode : secondNode) : index;
        auto requestWeightSum = workspace.requestWeightSums[nodeIndex];
        const auto &node = context.getNode(nodeIndex);
        if (requestWeightSum == 0.0) {
            continue;
        }
        double nodeCost = 0.0;
        double existingNodeCost = 0.0;
        const auto &baseline = baselineFor(nodeIndex);
        if (NonLinearPrefillModelR::usesGeometryForVersion(
                node.getLatencyBinaryVersion())) {
            nodeCost = context.estimateBatchExecutionLatencyMs(
                nodeIndex, geometryRequests[nodeIndex]);
            if (!std::isfinite(nodeCost)) {
                return std::numeric_limits<double>::infinity();
            }
        } else if (baseline.unified) {
            if (!baseline.floorValid) {
                return std::numeric_limits<double>::infinity();
            }
            const auto requestCostWithoutReportedWorkload =
                (baseline.capacitySplit
                     ? workspace.requestCostSumsWithoutReportedWorkload[nodeIndex]
                     : workspace.requestCostSums[nodeIndex]) +
                baseline.floorLinearCost;
            const auto promptTokensWithoutReportedWorkload =
                workspace.totalPromptTokenSums[nodeIndex] + baseline.floorHitTokens +
                baseline.floorMissTokens;
            auto requestCostWithReportedWorkload =
                baseline.capacitySplit
                    ? workspace.requestCostSums[nodeIndex] + baseline.workloadLinearCost
                    : requestCostWithoutReportedWorkload + baseline.reportedLinearCost;
            const auto promptTokensWithReportedWorkload =
                promptTokensWithoutReportedWorkload + baseline.reportedHitTokens +
                baseline.reportedMissTokens;
            const auto precedingRequestsWithoutReportedWorkload =
                baseline.floorPrecedingRequests;
            const auto precedingRequestsWithReportedWorkload =
                baseline.workloadPrecedingRequests;
            auto costWithReportedWorkload =
                NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(
                    node.getLatencyParameters(), requestCostWithReportedWorkload,
                    promptTokensWithReportedWorkload,
                    precedingRequestsWithReportedWorkload);
            costWithReportedWorkload =
                node.applyLatencyResidual(costWithReportedWorkload);
            auto floorCost = NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(
                node.getLatencyParameters(), requestCostWithoutReportedWorkload,
                promptTokensWithoutReportedWorkload,
                precedingRequestsWithoutReportedWorkload);
            floorCost = node.applyLatencyResidual(floorCost);
            nodeCost = node.decayUnifiedLatency(costWithReportedWorkload, floorCost);
            existingNodeCost = node.getExistingUnifiedLatencyMs();
        } else {
            nodeCost = NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(
                node.getLatencyParameters(), workspace.requestCostSums[nodeIndex],
                workspace.totalPromptTokenSums[nodeIndex], 0.0);
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
                (nodeCost + context.getNode(nodeIndex).getEstimatedQueueWaitMs(
                                workspace.hitMemberWeightSums[nodeIndex],
                                requestWeightSum,
                                workspace.lookupMemberWeightSums[nodeIndex])) *
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
    const NaviSchedScheduleContext &context, const float *costs, uint32_t *nodeIndexes,
    const int64_t *requestTokenCounts, const double *inverseRequestTokenCounts,
    double tokenNormalizationDenominator, double lambda, double latencyWeightConcavity,
    double &value, int64_t timeBudgetUs, size_t maxEvaluations,
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
        for (size_t requestIndex = 0; requestIndex < requestCount && !exhausted();
             ++requestIndex) {
            const auto source = nodeIndexes[requestIndex];
            // Removing this request has the same source queue for every target.
            // Accepted moves occur only after this entire candidate scan.
            bool sourceNodeReady = false;
            for (size_t target = 0; target < nodeCount && !exhausted(); ++target) {
                if (target == source) {
                    continue;
                }
                nodeIndexes[requestIndex] = static_cast<uint32_t>(target);
                // Preserve ordered queues and running-slot promotion on both
                // sides; only the unchanged source calculation can be reused.
                const auto pairValue = forwardDiscreteImpl(
                    context, costs, nodeIndexes, requestTokenCounts,
                    inverseRequestTokenCounts, tokenNormalizationDenominator, lambda,
                    latencyWeightConcavity, workspace, source, target, &nodeRequests,
                    requestIndex, sourceNodeReady);
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
        sourceRequests.erase(
            std::lower_bound(sourceRequests.begin(), sourceRequests.end(), bestRequest));
        auto &targetRequests = nodeRequests[bestTarget];
        targetRequests.insert(
            std::lower_bound(targetRequests.begin(), targetRequests.end(), bestRequest),
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
    const NaviSchedScheduleContext &context, const float *costs, const double *weights,
    const int64_t *requestTokenCounts, const double *inverseRequestTokenCounts,
    size_t requestCount, size_t nodeBegin, size_t nodeEnd,
    const NodeGradient *nodeGradients, double *gradients)
{
    for (size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex) {
        auto localNodeIndex = nodeIndex - nodeBegin;
        const auto &nodeGradient = nodeGradients[localNodeIndex];
        auto rowOffset = nodeIndex * requestCount;
        auto costRow = costs + rowOffset;
        auto gradientRow = gradients + localNodeIndex * requestCount;
        const auto &node = context.getNode(nodeIndex);
        const auto geometry = NonLinearPrefillModelR::usesGeometryForVersion(
            node.getLatencyBinaryVersion());
        const bool projectedLookup =
            NonLinearPrefillModelR::usesProjectedGeometryForVersion(
                node.getLatencyBinaryVersion());
        const auto inputs =
            geometry ? context.getGeometryOptimizationInputs(nodeIndex) : nullptr;
        const auto queueInputs = !geometry && nodeGradient.queueHitMember != 0.0
                                     ? context.getCacheHitBlockCountsSnapshot(nodeIndex)
                                     : nullptr;
        if (!geometry && nodeGradient.queueHitMember != 0.0 && !queueInputs) {
            std::fill(gradientRow, gradientRow + requestCount,
                      std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        if (nodeGradient.capacitySplit) {
            const auto waitingCostRow = costRow + context.nodeCount() * requestCount;
            calculateCapacitySplitMarginalCosts(
                costRow, waitingCostRow, weights + rowOffset, requestCount,
                nodeGradient.runningCapacity, gradientRow);
        }
        auto calculateGradients = [&](auto lookupEnabled) {
            for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
                bool queueHit = queueInputs && context.getCacheHitBlockCount(
                                                   *queueInputs, requestIndex) > 0;
                auto requestCostDerivative = nodeGradient.capacitySplit
                                                 ? gradientRow[requestIndex]
                                                 : costRow[requestIndex];
                bool lookupMember = false;
                double sourceDerivative = 0.0;
                if (geometry) {
                    const auto *input = inputs ? context.getGeometryOptimizationInput(
                                                     *inputs, requestIndex)
                                               : nullptr;
                    if (!input) {
                        gradientRow[requestIndex] =
                            std::numeric_limits<double>::quiet_NaN();
                        continue;
                    }
                    queueHit = input->cacheHit;
                    if constexpr (decltype(lookupEnabled)::value) {
                        lookupMember = projectedLookup
                                           ? !input->shared
                                           : node.hasQueueLookupOpportunity(
                                                 requestTokenCounts[requestIndex]);
                    }
                    requestCostDerivative =
                        input->linearCost +
                        nodeGradient.geometrySharedToken * input->sharedTokens +
                        (input->shared ? nodeGradient.geometrySharedWeight : 0.0);
                    requestCostDerivative += nodeGradient.geometryFixedAttentionPairs *
                                             input->fixedAttentionPairs;
                    requestCostDerivative +=
                        nodeGradient.geometryFixedInputChunks * input->fixedChunkCount;
                    if (requestIndex == nodeGradient.geometryFixedChunkRequest) {
                        requestCostDerivative +=
                            nodeGradient.geometryFixedChunks * input->fixedChunkCount;
                    }
                    if (input->externalReadTokens > 0.0) {
                        sourceDerivative =
                            nodeGradient.externalReadMember +
                            (requestIndex == nodeGradient.externalReadMaximumRequest
                                 ? nodeGradient.externalReadMaximum *
                                       input->externalReadTokens
                                 : 0.0);
                    }
                } else {
                    if constexpr (decltype(lookupEnabled)::value) {
                        lookupMember = node.hasQueueLookupOpportunity(
                            requestTokenCounts[requestIndex]);
                    }
                }
                gradientRow[requestIndex] =
                    nodeGradient.cost * requestCostDerivative +
                    nodeGradient.totalPromptToken * requestTokenCounts[requestIndex] +
                    nodeGradient.weight + sourceDerivative +
                    (queueHit ? nodeGradient.queueHitMember : 0.0) +
                    (lookupMember ? nodeGradient.queueLookupMember : 0.0) +
                    nodeGradient.inverseToken * inverseRequestTokenCounts[requestIndex];
            }
        };
        if (nodeGradient.queueLookupMember != 0.0) {
            calculateGradients(std::true_type{});
        } else {
            calculateGradients(std::false_type{});
        }
    }
}

bool CostSchedulerKernelInternal::optimize(
    const NaviSchedScheduleContext &context, const float *costs,
    const int64_t *requestTokenCounts, int64_t totalTokenCount, double lambda,
    double alpha, double alphaDecay, double minAlpha, size_t maxLoopCount,
    int64_t optimizationTimeBudgetUs, ReplayInfo *replayInfo,
    double latencyWeightConcavity, uint64_t initialWeightSeed, double *weights,
    std::vector<uint32_t> &selectedNodeIndexes, double &outputForwardValue,
    double latencyTokenNormalizationPower, size_t discreteCheckInterval,
    double discreteGapThreshold, double discreteRepairAlphaDecay, size_t minLoopCount)
{
    if (replayInfo) {
        replayInfo->minLoopCount = minLoopCount;
    }
    const auto optimizationBeginUs = context.getScheduleBeginTimeUs() > 0
                                         ? context.getScheduleBeginTimeUs()
                                         : autil::TimeUtility::monotonicTimeUs();
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
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        inverseRequestTokenCounts[requestIndex] =
            1.0 / std::pow(static_cast<double>(requestTokenCounts[requestIndex]),
                           latencyTokenNormalizationPower);
    }
    auto tokenNormalizationDenominator = calculateTokenNormalizationDenominator(
        requestTokenCounts, requestCount, latencyTokenNormalizationPower);
    if (tokenNormalizationDenominator <= 0.0) {
        return false;
    }
    std::vector<NodeGradient> nodeGradients(nodeCount);
    std::vector<double> gradients(weightCount);
    std::vector<double> weightsBeforeProjection;
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        weightsBeforeProjection.resize(weightCount);
        context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                          collectionBeginUs);
    }
    auto projectionTileSize = std::min(requestCount, PROJECTION_REQUEST_TILE_SIZE);
    std::vector<double> projectionBuffer(nodeCount * projectionTileSize);
    auto collectTiming = (replayInfo && replayInfo->enabled);
    auto trackLoopTime = collectTiming || optimizationTimeBudgetUs >= 0;

    auto lastLoopEndUs = optimizationBeginUs;
    auto runBeginUs = collectTiming ? optimizationBeginUs : 0;
    size_t actualLoopCount = 0;
    std::string stopReason = "max_loop_count";
    initializeWeights(weights, requestCount, nodeCount, initialWeightSeed);
    if (!selectWeightRequestRange(weights, requestCount, nodeCount, 0, requestCount,
                                  bestNodeIndexes.data(), initialWeightSeed)) {
        return false;
    }
    bestLoss =
        forwardDiscrete(context, costs, bestNodeIndexes.data(), requestTokenCounts,
                        inverseRequestTokenCounts.data(), tokenNormalizationDenominator,
                        lambda, latencyWeightConcavity, discreteForwardWorkspace);
    if (!std::isfinite(bestLoss)) {
        return false;
    }
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        replayInfo->beginRun(weights, weightCount);
        context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                          collectionBeginUs);
    }
    for (size_t iteration = 0; iteration < maxLoopCount; ++iteration) {
        bool discreteEvaluated = false;
        double discreteForwardValue = 0.0;
        double discreteGap = 0.0;
        auto stepAlpha = getStepAlpha(alpha, alphaDecay, minAlpha, iteration);
        auto stepBeginUs = autil::TimeUtility::monotonicTimeUs();
        auto value = forwardNodeRange(
            context, costs, weights, requestTokenCounts, inverseRequestTokenCounts.data(),
            tokenNormalizationDenominator, lambda, latencyWeightConcavity, 0, nodeCount,
            nodeGradients.data(), nullptr, nullptr);
        auto forwardEndUs = autil::TimeUtility::monotonicTimeUs();
        if (!std::isfinite(value)) {
            return false;
        }
        if (iteration > 0) {
            auto discreteLoss = forwardDiscrete(
                context, costs, candidateNodeIndexes.data(), requestTokenCounts,
                inverseRequestTokenCounts.data(), tokenNormalizationDenominator, lambda,
                latencyWeightConcavity, discreteForwardWorkspace);
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
                       std::max(std::abs(value), std::numeric_limits<double>::epsilon());
            discreteGap = gap;
            if ((discreteRepairActive || iteration % discreteCheckInterval == 0) &&
                gap > discreteGapThreshold) {
                repairAlpha =
                    std::max(minAlpha, (discreteRepairActive ? repairAlpha : stepAlpha) *
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
        backwardNodeRange(context, costs, weights, requestTokenCounts,
                          inverseRequestTokenCounts.data(), requestCount, 0, nodeCount,
                          nodeGradients.data(), gradients.data());
        auto backwardEndUs = autil::TimeUtility::monotonicTimeUs();
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->beginStep(iteration, stepAlpha, value, gradients.data(), weights,
                                  weightCount, forwardEndUs - stepBeginUs,
                                  backwardEndUs - forwardEndUs);
            if (discreteEvaluated) {
                replayInfo->recordDiscreteEvaluation(discreteForwardValue, discreteGap,
                                                     discreteRepairActive);
            }
            std::copy(weights, weights + weightCount, weightsBeforeProjection.begin());
            if (!applyGradient(weightsBeforeProjection.data(), 0, weightCount,
                               gradients.data(), stepAlpha)) {
                return false;
            }
            context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                              collectionBeginUs);
        }
        auto updateEndUs = collectTiming ? autil::TimeUtility::monotonicTimeUs() : 0;
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->recordBeforeProjection(weightsBeforeProjection.data(),
                                               weightCount, updateEndUs - backwardEndUs);
            context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                              collectionBeginUs);
        }
        GradientSpan gradientSpan{gradients.data(), 0, nodeCount};
        const auto projectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        if (!projectRequestRange(weights, requestCount, nodeCount, 0, requestCount,
                                 projectionBuffer.data(), candidateNodeIndexes.data(),
                                 &gradientSpan, 1, stepAlpha, initialWeightSeed)) {
            return false;
        }
        auto projectionEndUs = autil::TimeUtility::monotonicTimeUs();
        context.recordCostGradientIteration(forwardEndUs - stepBeginUs + backwardEndUs -
                                            backwardBeginUs + projectionEndUs -
                                            projectionBeginUs);
        auto stepEndUs = trackLoopTime ? autil::TimeUtility::monotonicTimeUs() : 0;
        if (replayInfo && replayInfo->enabled) {
            const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
            replayInfo->recordAfterProjection(
                weights, weightCount, projectionEndUs - updateEndUs,
                stepEndUs - projectionEndUs, stepEndUs - stepBeginUs);
            context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                              collectionBeginUs);
        }
        actualLoopCount = iteration + 1;
        const auto budgetEndUs =
            collectTiming ? autil::TimeUtility::monotonicTimeUs() : stepEndUs;
        auto elapsedUs =
            std::max<int64_t>(0, budgetEndUs - optimizationBeginUs);
        auto lastLoopUs = std::max<int64_t>(
            0, budgetEndUs - lastLoopEndUs);
        lastLoopEndUs = budgetEndUs;
        if (!shouldRunNextLoop(actualLoopCount, maxLoopCount, optimizationTimeBudgetUs,
                               elapsedUs, lastLoopUs, minLoopCount)) {
            stopReason =
                actualLoopCount >= maxLoopCount ? "max_loop_count" : "time_budget";
            break;
        }
    }
    context.recordCostOptimizerProgress(actualLoopCount, maxLoopCount,
                                        stopReason == "time_budget");
    auto finalForwardValue = forwardNodeRange(
        context, costs, weights, requestTokenCounts, inverseRequestTokenCounts.data(),
        tokenNormalizationDenominator, lambda, latencyWeightConcavity, 0, nodeCount,
        nullptr, nullptr, nullptr);
    if (!std::isfinite(finalForwardValue)) {
        return false;
    }
    auto finalDiscreteLoss =
        forwardDiscrete(context, costs, candidateNodeIndexes.data(), requestTokenCounts,
                        inverseRequestTokenCounts.data(), tokenNormalizationDenominator,
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
            ? autil::TimeUtility::monotonicTimeUs() - optimizationBeginUs
            : 0);
    const auto initialDiscreteLoss = bestLoss;
    DiscreteRefinementStats refinementStats;
    if (refinementBudget != 0 &&
        !refineDiscreteAssignment(context, costs, bestNodeIndexes.data(),
                                  requestTokenCounts, inverseRequestTokenCounts.data(),
                                  tokenNormalizationDenominator, lambda,
                                  latencyWeightConcavity, bestLoss, refinementBudget,
                                  std::numeric_limits<size_t>::max(), refinementStats)) {
        return false;
    }
    context.recordDiscreteSearchLosses(initialDiscreteLoss, bestLoss);
    outputForwardValue = bestLoss;
    selectedNodeIndexes = std::move(bestNodeIndexes);
    if (replayInfo && replayInfo->enabled) {
        const auto collectionBeginUs = autil::TimeUtility::monotonicTimeUs();
        replayInfo->finishRun(
            weights, weightCount, finalForwardValue, bestLoss, selectedNodeIndexes.data(),
            requestCount,
            collectTiming ? autil::TimeUtility::monotonicTimeUs() - runBeginUs : 0,
            actualLoopCount, stopReason, refinementStats.evaluations,
            refinementStats.moves);
        context.addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                          collectionBeginUs);
    }
    return true;
}

bool CostSchedulerKernelInternal::shouldRunNextLoop(size_t completedLoopCount,
                                                    size_t maxLoopCount,
                                                    int64_t optimizationTimeBudgetUs,
                                                    int64_t elapsedUs, int64_t lastLoopUs,
                                                    size_t minLoopCount)
{
    if (completedLoopCount >= maxLoopCount) {
        return false;
    }
    if (optimizationTimeBudgetUs > 0 && elapsedUs >= optimizationTimeBudgetUs) {
        return false;
    }
    const auto minimumLoopCount = std::min(minLoopCount, maxLoopCount);
    if (completedLoopCount < minimumLoopCount ||
        optimizationTimeBudgetUs < 0) {
        return true;
    }
    if (optimizationTimeBudgetUs > 0 && elapsedUs >= optimizationTimeBudgetUs) {
        return false;
    }
    auto averageLoopUs = (elapsedUs + static_cast<int64_t>(completedLoopCount) - 1) /
                         static_cast<int64_t>(completedLoopCount);
    auto nextLoopUs = std::max(lastLoopUs, averageLoopUs);
    return elapsedUs < optimizationTimeBudgetUs &&
           nextLoopUs <= optimizationTimeBudgetUs - elapsedUs;
}

void CostSchedulerKernelInternal::ReplayInfo::OptimizationStep::Jsonize(JsonWrapper &json)
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
    json.Jsonize("discrete_evaluated", discreteEvaluated, discreteEvaluated);
    json.Jsonize("discrete_forward_value", discreteForwardValue, discreteForwardValue);
    json.Jsonize("discrete_gap", discreteGap, discreteGap);
    json.Jsonize("discrete_repair_active", discreteRepairActive, discreteRepairActive);
    json.Jsonize("gradients", gradients);
    json.Jsonize("weights_before_update", weightsBeforeUpdate);
    json.Jsonize("weights_before_projection", weightsBeforeProjection);
    json.Jsonize("weights_after_projection", weightsAfterProjection);
}

void CostSchedulerKernelInternal::ReplayInfo::OptimizationRun::Jsonize(JsonWrapper &json)
{
    json.Jsonize("initial_weights", initialWeights);
    json.Jsonize("steps", steps);
    json.Jsonize("final_weights", finalWeights);
    json.Jsonize("final_forward_value", finalForwardValue);
    json.Jsonize("best_forward_value", bestForwardValue, bestForwardValue);
    json.Jsonize("best_node_indexes", bestNodeIndexes, bestNodeIndexes);
    json.Jsonize("elapsed_us", elapsedUs);
    json.Jsonize("actual_loop_count", actualLoopCount, actualLoopCount);
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
    json.Jsonize("optimization_time_budget_us", optimizationTimeBudgetUs,
                 optimizationTimeBudgetUs);
    json.Jsonize("parallelism", parallelism, parallelism);
    json.Jsonize("latency_weight_concavity", latencyWeightConcavity,
                 latencyWeightConcavity);
    json.Jsonize("latency_token_normalization_power", latencyTokenNormalizationPower,
                 latencyTokenNormalizationPower);
    json.Jsonize("initial_weight_seed", initialWeightSeed, initialWeightSeed);
    json.Jsonize("discrete_check_interval", discreteCheckInterval, discreteCheckInterval);
    json.Jsonize("discrete_gap_threshold", discreteGapThreshold, discreteGapThreshold);
    json.Jsonize("discrete_repair_alpha_decay", discreteRepairAlphaDecay,
                 discreteRepairAlphaDecay);
    json.Jsonize("selection_jitter_ratio", selectionJitterRatio, selectionJitterRatio);
    json.Jsonize("sched_infos", schedInfos);
    json.Jsonize("optimization_runs", optimizationRuns);
}

void CostSchedulerKernelInternal::ReplayInfo::init(
    bool enabledValue, double lambdaValue, double alphaValue, double alphaDecayValue,
    double minAlphaValue, size_t maxLoopCountValue, int64_t optimizationTimeBudgetUsValue)
{
    enabled = enabledValue;
    lambda = lambdaValue;
    alpha = alphaValue;
    alphaDecay = alphaDecayValue;
    minAlpha = minAlphaValue;
    maxLoopCount = maxLoopCountValue;
    optimizationTimeBudgetUs = optimizationTimeBudgetUsValue;
}

void CostSchedulerKernelInternal::ReplayInfo::beginRun(const double *weights,
                                                       size_t weightCount)
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
    size_t iterationValue, double alphaValue, double forwardValueValue,
    const double *gradientValues, const double *weights, size_t weightCount,
    int64_t forwardUsValue, int64_t backwardUsValue)
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
    step.gradients.assign(gradientValues, gradientValues + weightCount);
    step.weightsBeforeUpdate.assign(weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::recordBeforeProjection(
    const double *weights, size_t weightCount, int64_t updateUsValue)
{
    if (!enabled) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.updateUs = updateUsValue;
    step.weightsBeforeProjection.assign(weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::recordDiscreteEvaluation(
    double discreteForwardValueValue, double discreteGapValue, bool repairActive)
{
    if (!enabled || optimizationRuns.empty() || optimizationRuns.back().steps.empty()) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.discreteEvaluated = true;
    step.discreteForwardValue = discreteForwardValueValue;
    step.discreteGap = discreteGapValue;
    step.discreteRepairActive = repairActive;
}

void CostSchedulerKernelInternal::ReplayInfo::recordAfterProjection(
    const double *weights, size_t weightCount, int64_t projectionUsValue,
    int64_t bindUsValue, int64_t elapsedUsValue)
{
    if (!enabled) {
        return;
    }
    auto &step = optimizationRuns.back().steps.back();
    step.projectionUs = projectionUsValue;
    step.bindUs = bindUsValue;
    step.elapsedUs = elapsedUsValue;
    step.weightsAfterProjection.assign(weights, weights + weightCount);
}

void CostSchedulerKernelInternal::ReplayInfo::finishRun(
    const double *weights, size_t weightCount, double finalValue, double bestValue,
    const uint32_t *nodeIndexes, size_t requestCount, int64_t elapsedUsValue,
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

void CostSchedulerKernelInternal::ReplayInfo::finish(NaviSchedScheduleContext &context,
                                                     const float *costs,
                                                     const double *weights,
                                                     size_t requestCount,
                                                     size_t nodeCount)
{
    if (!enabled) {
        return;
    }
    schedInfos.resize(nodeCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto &nodeSchedInfos = schedInfos[nodeIndex];
        nodeSchedInfos.reserve(requestCount);
        auto rowOffset = nodeIndex * requestCount;
        double batchWeight = 0.0, hitMemberWeight = 0.0, lookupMemberWeight = 0.0;
        const auto queueInputs =
            context.getNode(nodeIndex).usesCacheMemberQueuePrediction()
                ? context.getCacheHitBlockCountsSnapshot(nodeIndex)
                : nullptr;
        for (size_t i = 0; i < requestCount; ++i) {
            batchWeight += weights[rowOffset + i];
            if (context.getNode(nodeIndex).hasQueueLookupOpportunity(
                    context.getRequest(i).request.promptTokenCount)) {
                lookupMemberWeight += weights[rowOffset + i];
            }
            if (queueInputs && context.getCacheHitBlockCount(*queueInputs, i) > 0) {
                hitMemberWeight += weights[rowOffset + i];
            }
        }
        auto queueWaitMs =
            static_cast<float>(context.getNode(nodeIndex).getEstimatedQueueWaitMs(
                hitMemberWeight, batchWeight, lookupMemberWeight));
        for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
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

} // namespace navi_sched
