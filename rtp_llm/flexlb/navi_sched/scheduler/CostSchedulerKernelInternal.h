#pragma once

#include "aios/navi_sched/scheduler/AutoGrad.h"
#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include <autil/legacy/jsonizable.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace navi_sched {

class CostSchedulerKernelInternal {
public:
    static constexpr size_t PROJECTION_REQUEST_TILE_SIZE = 16;
    static constexpr size_t MAX_DISCRETE_REFINEMENT_EVALUATIONS = 4096;
    static constexpr int64_t MAX_DISCRETE_REFINEMENT_US = 2000;

    struct NodeGradient {
        double cost = 0.0;
        double totalPromptToken = 0.0;
        double weight = 0.0;
        double inverseToken = 0.0;
        double queueHitMember = 0.0;
        double queueLookupMember = 0.0;
        double geometrySharedToken = 0.0;
        double geometrySharedWeight = 0.0;
        double geometryFixedChunks = 0.0;
        size_t geometryFixedChunkRequest = std::numeric_limits<size_t>::max();
        double geometryFixedAttentionPairs = 0.0;
        double geometryFixedInputChunks = 0.0;
        double externalReadMember = 0.0;
        double externalReadMaximum = 0.0;
        size_t externalReadMaximumRequest = std::numeric_limits<size_t>::max();
        double runningCapacity = 0.0;
        bool capacitySplit = false;
    };

    struct PreparationTiming {
        int64_t matchUs = 0;
        int64_t costEvaluationUs = 0;
        size_t completedNodes = 0;
    };

    struct GradientSpan {
        const double *values = nullptr;
        size_t nodeBegin = 0;
        size_t nodeEnd = 0;
    };

    struct DiscreteForwardWorkspace {
        struct NodeBaseline {
            bool loaded = false;
            bool capacitySplit = false;
            bool unified = false;
            bool floorValid = true;
            int32_t workloadSlots = 0;
            int32_t reservedSlots = 0;
            int32_t maxRunning = 0;
            double floorLinearCost = 0.0;
            double reportedLinearCost = 0.0;
            double workloadLinearCost = 0.0;
            int64_t floorHitTokens = 0;
            int64_t floorMissTokens = 0;
            int64_t reportedHitTokens = 0;
            int64_t reportedMissTokens = 0;
            int64_t floorPrecedingRequests = 0;
            int64_t workloadPrecedingRequests = 0;
        };

        const NaviSchedScheduleContext *baselineContext = nullptr;
        std::vector<NodeBaseline> nodeBaselines;
        std::vector<double> requestCostSums;
        std::vector<double> requestCostSumsWithoutReportedWorkload;
        std::vector<double> totalPromptTokenSums;
        std::vector<double> requestWeightSums;
        std::vector<double> inverseTokenWeightSums;
        std::vector<double> nodeValues;
        size_t nodeCapacityLoads = 0;
        size_t nodeBaselineLoads = 0;
        std::vector<double> hitMemberWeightSums;
        std::vector<double> lookupMemberWeightSums;
        std::vector<std::shared_ptr<const std::vector<int64_t>>> cacheCountSnapshots;
        std::vector<std::vector<size_t>> geometryRequests;
    };

    struct DiscreteRefinementStats {
        size_t evaluations = 0;
        size_t moves = 0;
        bool budgetExhausted = false;
    };

    struct SchedInfo {
        float cost = 0.0f;
        float waitingCost = 0.0f;
        float queueWaitMs = 0.0f;
        double weight = 0.0;
    };

    struct SchedInfoJson : public autil::legacy::Jsonizable {
        SchedInfoJson() = default;
        explicit SchedInfoJson(const SchedInfo &value) : schedInfo(value) {}

        void Jsonize(JsonWrapper &json) override
        {
            json.Jsonize("cost", schedInfo.cost);
            json.Jsonize("waiting_cost", schedInfo.waitingCost, schedInfo.cost);
            json.Jsonize("queue_wait_ms", schedInfo.queueWaitMs);
            json.Jsonize("weight", schedInfo.weight);
        }

        SchedInfo schedInfo;
    };

    struct ReplayInfo : public autil::legacy::Jsonizable {
        struct OptimizationStep : public autil::legacy::Jsonizable {
            void Jsonize(JsonWrapper &json) override;

            size_t iteration = 0;
            double alpha = 0.0;
            double forwardValue = 0.0;
            int64_t elapsedUs = 0;
            int64_t forwardUs = 0;
            int64_t backwardUs = 0;
            int64_t updateUs = 0;
            int64_t projectionUs = 0;
            int64_t bindUs = 0;
            bool discreteEvaluated = false;
            double discreteForwardValue = 0.0;
            double discreteGap = 0.0;
            bool discreteRepairActive = false;
            std::vector<double> gradients;
            std::vector<double> weightsBeforeUpdate;
            std::vector<double> weightsBeforeProjection;
            std::vector<double> weightsAfterProjection;
        };

        struct OptimizationRun : public autil::legacy::Jsonizable {
            void Jsonize(JsonWrapper &json) override;

            std::vector<double> initialWeights;
            std::vector<OptimizationStep> steps;
            std::vector<double> finalWeights;
            double finalForwardValue = 0.0;
            double bestForwardValue = 0.0;
            std::vector<uint32_t> bestNodeIndexes;
            int64_t elapsedUs = 0;
            size_t actualLoopCount = 0;
            std::string stopReason;
            size_t refinementEvaluations = 0;
            size_t refinementMoves = 0;
        };

        void Jsonize(JsonWrapper &json) override;
        void init(bool enabled, double lambdaValue, double alphaValue,
                  double alphaDecayValue, double minAlphaValue, size_t maxLoopCountValue,
                  int64_t optimizationTimeBudgetUsValue);
        void beginRun(const double *weights, size_t weightCount);
        void beginStep(size_t iteration, double alpha, double forwardValue,
                       const double *gradients, const double *weights, size_t weightCount,
                       int64_t forwardUs, int64_t backwardUs);
        void recordBeforeProjection(const double *weights, size_t weightCount,
                                    int64_t updateUs);
        void recordDiscreteEvaluation(double discreteForwardValue, double discreteGap,
                                      bool repairActive);
        void recordAfterProjection(const double *weights, size_t weightCount,
                                   int64_t projectionUs, int64_t bindUs,
                                   int64_t elapsedUs);
        void finishRun(const double *weights, size_t weightCount,
                       double finalForwardValue, double bestForwardValue,
                       const uint32_t *bestNodeIndexes, size_t requestCount,
                       int64_t elapsedUs, size_t actualLoopCount,
                       const std::string &stopReason, size_t refinementEvaluations = 0,
                       size_t refinementMoves = 0);
        void finish(NaviSchedScheduleContext &context, const float *costs,
                    const double *weights, size_t requestCount, size_t nodeCount);

        bool enabled = false;
        std::string scheduler = "cost";
        double lambda = 0.0;
        double alpha = 0.0;
        double alphaDecay = 1.0;
        double minAlpha = 0.0;
        size_t maxLoopCount = 0;
        size_t minLoopCount = 2;
        int64_t optimizationTimeBudgetUs = 0;
        size_t parallelism = 1;
        double latencyWeightConcavity = 128.0;
        double latencyTokenNormalizationPower = 1.0;
        uint64_t initialWeightSeed = 0;
        size_t discreteCheckInterval = 5;
        double discreteGapThreshold = 0.25;
        double discreteRepairAlphaDecay = 0.7;
        double selectionJitterRatio = 1.0 / 1000.0;
        std::vector<std::vector<SchedInfoJson>> schedInfos;
        std::vector<OptimizationRun> optimizationRuns;
    };

    static int64_t fillRequestTokenCounts(const NaviSchedScheduleContext &context,
                                          int64_t *requestTokenCounts,
                                          double *inverseRequestTokenCounts = nullptr,
                                          double latencyTokenNormalizationPower = 1.0);
    static double calculateTokenNormalizationDenominator(
        const int64_t *requestTokenCounts, size_t requestCount,
        double latencyTokenNormalizationPower);
    static std::unique_ptr<float[]> prepare(
        NaviSchedScheduleContext &context, const int64_t *requestTokenCounts,
        double lambda, double alpha, double alphaDecay, double minAlpha,
        size_t maxLoopCount, int64_t optimizationTimeBudgetUs, ReplayInfo *replayInfo);
    static std::unique_ptr<float[]> prepare(
        NaviSchedScheduleContext &context, const int64_t *requestTokenCounts,
        double lambda, double alpha, double alphaDecay, double minAlpha,
        size_t maxLoopCount, int64_t optimizationTimeBudgetUs, ReplayInfo &replayInfo)
    {
        return prepare(context, requestTokenCounts, lambda, alpha, alphaDecay, minAlpha,
                       maxLoopCount, optimizationTimeBudgetUs, &replayInfo);
    }
    static bool fillCostsForNodeRange(const NaviSchedScheduleContext &context,
                                      const int64_t *requestTokenCounts, size_t nodeBegin,
                                      size_t nodeEnd, float *costs,
                                      PreparationTiming *timing = nullptr);
    static bool optimize(const NaviSchedScheduleContext &context, const float *costs,
                         const int64_t *requestTokenCounts, int64_t totalTokenCount,
                         double lambda, double alpha, double alphaDecay, double minAlpha,
                         size_t maxLoopCount, int64_t optimizationTimeBudgetUs,
                         ReplayInfo *replayInfo, double latencyWeightConcavity,
                         uint64_t initialWeightSeed, double *weights,
                         std::vector<uint32_t> &selectedNodeIndexes,
                         double &outputForwardValue,
                         double latencyTokenNormalizationPower,
                         size_t discreteCheckInterval, double discreteGapThreshold,
                         double discreteRepairAlphaDecay, size_t minLoopCount = 2);
    static bool optimize(const NaviSchedScheduleContext &context, const float *costs,
                         const int64_t *requestTokenCounts, int64_t totalTokenCount,
                         double lambda, double alpha, double alphaDecay, double minAlpha,
                         size_t maxLoopCount, int64_t optimizationTimeBudgetUs,
                         ReplayInfo &replayInfo, double latencyWeightConcavity,
                         uint64_t initialWeightSeed, double *weights,
                         std::vector<uint32_t> &selectedNodeIndexes,
                         double &outputForwardValue,
                         double latencyTokenNormalizationPower = 1.0,
                         size_t discreteCheckInterval = 5,
                         double discreteGapThreshold = 0.25,
                         double discreteRepairAlphaDecay = 0.7, size_t minLoopCount = 2)
    {
        return optimize(
            context, costs, requestTokenCounts, totalTokenCount, lambda, alpha,
            alphaDecay, minAlpha, maxLoopCount, optimizationTimeBudgetUs, &replayInfo,
            latencyWeightConcavity, initialWeightSeed, weights, selectedNodeIndexes,
            outputForwardValue, latencyTokenNormalizationPower, discreteCheckInterval,
            discreteGapThreshold, discreteRepairAlphaDecay, minLoopCount);
    }
    static bool shouldRunNextLoop(size_t completedLoopCount, size_t maxLoopCount,
                                  int64_t optimizationTimeBudgetUs, int64_t elapsedUs,
                                  int64_t lastLoopUs, size_t minLoopCount = 2);
    static double forwardNodeRange(const NaviSchedScheduleContext &context,
                                   const float *costs, const double *weights,
                                   const int64_t *requestTokenCounts,
                                   const double *inverseRequestTokenCounts,
                                   double tokenNormalizationDenominator, double lambda,
                                   double latencyWeightConcavity, size_t nodeBegin,
                                   size_t nodeEnd, NodeGradient *nodeGradients,
                                   double *firstTerms, double *secondTerms);
    static double forwardDiscrete(const NaviSchedScheduleContext &context,
                                  const float *costs, const uint32_t *nodeIndexes,
                                  const int64_t *requestTokenCounts,
                                  const double *inverseRequestTokenCounts,
                                  double tokenNormalizationDenominator, double lambda,
                                  double latencyWeightConcavity,
                                  DiscreteForwardWorkspace &workspace);
    static bool refineDiscreteAssignment(
        const NaviSchedScheduleContext &context, const float *costs,
        uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
        const double *inverseRequestTokenCounts, double tokenNormalizationDenominator,
        double lambda, double latencyWeightConcavity, double &value, int64_t timeBudgetUs,
        size_t maxEvaluations, DiscreteRefinementStats &stats);
    static int64_t discreteRefinementBudgetUs(int64_t optimizationTimeBudgetUs,
                                              int64_t elapsedUs);
    static void backwardNodeRange(const NaviSchedScheduleContext &context,
                                  const float *costs, const double *weights,
                                  const int64_t *requestTokenCounts,
                                  const double *inverseRequestTokenCounts,
                                  size_t requestCount, size_t nodeBegin, size_t nodeEnd,
                                  const NodeGradient *nodeGradients, double *gradients);
    static void initializeWeights(double *weights, size_t requestCount, size_t nodeCount,
                                  uint64_t randomSeed = 0);
    static void initializeWeights(double *weights,
                                  const NaviSchedScheduleContext &context,
                                  const double *randomWeights, size_t randomWeightCount);
    static bool initializeWeightRequestRange(double *weights,
                                             const NaviSchedScheduleContext &context,
                                             const double *randomWeights,
                                             size_t randomWeightCount,
                                             size_t requestBegin, size_t requestEnd);
    static bool projectWeights(double *weights, size_t requestCount, size_t nodeCount);
    static bool applyGradient(double *weights, size_t weightCount,
                              const std::vector<double> &gradients, double alpha);
    static bool applyGradient(double *weights, size_t weightOffset, size_t weightCount,
                              const std::vector<double> &gradients, double alpha);
    static bool applyGradient(double *weights, size_t weightOffset, size_t weightCount,
                              const double *gradients, double alpha);
    static bool projectRequestRange(double *weights, size_t requestCount,
                                    size_t nodeCount, size_t requestBegin,
                                    size_t requestEnd, double *workspace,
                                    uint32_t *nodeIndexes = nullptr,
                                    const GradientSpan *gradientSpans = nullptr,
                                    size_t gradientSpanCount = 0, double alpha = 0.0,
                                    uint64_t selectionRandomSeed = 0);
    static bool selectNodes(NaviSchedScheduleContext &context,
                            const uint32_t *nodeIndexes);
    static bool selectWeightRequestRange(const double *weights, size_t requestCount,
                                         size_t nodeCount, size_t requestBegin,
                                         size_t requestEnd, uint32_t *nodeIndexes,
                                         uint64_t selectionRandomSeed);
    static double getStepAlpha(double alpha, double alphaDecay, double minAlpha,
                               size_t iteration);

private:
    static double forwardDiscreteImpl(
        const NaviSchedScheduleContext &context, const float *costs,
        const uint32_t *nodeIndexes, const int64_t *requestTokenCounts,
        const double *inverseRequestTokenCounts, double tokenNormalizationDenominator,
        double lambda, double latencyWeightConcavity, DiscreteForwardWorkspace &workspace,
        size_t firstNode, size_t secondNode,
        const std::vector<std::vector<size_t>> *nodeRequests = nullptr,
        size_t movedRequest = 0, bool reuseSourceNode = false);
    static void normalizeWeights(double *weights, size_t requestCount, size_t nodeCount,
                                 const std::vector<double> &inverseWeightSums);
    static bool isValidNode(const NaviSchedScheduleNode &node);
    static bool findSimplexThreshold(double *workspace, size_t count, double activeSum,
                                     double &threshold);
};

}
