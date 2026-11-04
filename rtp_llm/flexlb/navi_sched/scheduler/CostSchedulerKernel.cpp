#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "autil/Scope.h"
#include "autil/TimeUtility.h"

namespace navi_sched {

const std::string CostSchedulerKernel::KERNEL_ID = "navi_sched.cost_scheduler_k";

CostSchedulerKernel::CostSchedulerKernel()
    : _randomGenerator(autil::TimeUtility::currentTimeInMicroSeconds())
{
    (void)_randomGenerator.get();
}

void CostSchedulerKernel::def(navi::KernelDefBuilder &builder) const
{
    SchedulerBaseKernel::def(builder, KERNEL_ID);
}

bool CostSchedulerKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "lambda", _config.lambda, _config.lambda);
    NAVI_JSONIZE(ctx, "alpha", _config.alpha, _config.alpha);
    NAVI_JSONIZE(ctx, "alpha_decay", _config.alphaDecay,
                 _config.alphaDecay);
    NAVI_JSONIZE(ctx, "min_alpha", _config.minAlpha,
                 _config.minAlpha);
    NAVI_JSONIZE(ctx, "min_loop_count", _config.minLoopCount, _config.minLoopCount);
    NAVI_JSONIZE(ctx, "max_loop_count", _config.maxLoopCount,
                 _config.maxLoopCount);
    NAVI_JSONIZE(ctx, "optimization_time_budget_us",
                 _config.optimizationTimeBudgetUs,
                 _config.optimizationTimeBudgetUs);
    NAVI_JSONIZE(ctx, "parallelism", _config.parallelism,
                 _config.parallelism);
    NAVI_JSONIZE(ctx, "latency_weight_concavity",
                 _config.latencyWeightConcavity,
                 _config.latencyWeightConcavity);
    NAVI_JSONIZE(ctx, "latency_token_normalization_power",
                 _config.latencyTokenNormalizationPower,
                 _config.latencyTokenNormalizationPower);
    NAVI_JSONIZE(ctx, "initial_weight_seed", _config.initialWeightSeed,
                 _config.initialWeightSeed);
    NAVI_JSONIZE(ctx, "discrete_check_interval",
                 _config.discreteCheckInterval,
                 _config.discreteCheckInterval);
    NAVI_JSONIZE(ctx, "discrete_gap_threshold",
                 _config.discreteGapThreshold,
                 _config.discreteGapThreshold);
    NAVI_JSONIZE(ctx, "discrete_repair_alpha_decay",
                 _config.discreteRepairAlphaDecay,
                 _config.discreteRepairAlphaDecay);
    if (!_config.valid()) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid cost scheduler config: lambda [%.9g], "
                        "alpha [%.9g], alpha_decay [%.9g], min_alpha [%.9g], "
                        "max_loop_count [%d], time_budget_us [%ld], "
                        "parallelism [%d], latency_weight_concavity [%.9g], "
                        "latency_token_normalization_power [%.9g]",
                        _config.lambda, _config.alpha, _config.alphaDecay,
                        _config.minAlpha,
                        _config.maxLoopCount,
                        _config.optimizationTimeBudgetUs,
                        _config.parallelism,
                        _config.latencyWeightConcavity,
                        _config.latencyTokenNormalizationPower);
        return false;
    }
    return true;
}

bool CostSchedulerKernel::schedule(NaviSchedScheduleContext &context) const
{
    CostSchedulerColdStartStrategy::Result coldStartResult;
    if (!CostSchedulerColdStartStrategy::apply(
            context, _randomGenerator, coldStartResult)) {
        return false;
    }
    NAVI_KERNEL_LOG(
        DEBUG,
        "cold-start routing evaluated, targetInflight [%.3f], "
        "trustedNodes [%lu], untrustedNodes [%lu], trustedRequests [%lu], "
        "untrustedRequests [%lu], rejectedRequests [%lu], "
        "untrustedCapacity [%lu], rampingNodes [%lu], "
        "unsafeWaitingMarginalNodes [%lu], "
        "minimumRampProgress [%.6f], minimumRampedTargetInflight [%.3f]",
        coldStartResult.targetInflight, coldStartResult.trustedNodeCount,
        coldStartResult.untrustedNodeCount,
        coldStartResult.trustedRequestCount,
        coldStartResult.untrustedRequestCount,
        coldStartResult.rejectedRequestCount,
        coldStartResult.untrustedCapacity, coldStartResult.rampingNodeCount,
        coldStartResult.unsafeWaitingMarginalNodeCount,
        coldStartResult.minimumRampProgress,
        coldStartResult.minimumRampedTargetInflight);
    if (!coldStartResult.costScheduleRequired) {
        return context.setScheduleLoss(0.0);
    }
    autil::ScopeGuard clearScheduleView(
        [&context]() { context.clearScheduleView(); });
    const auto config = _schedulerConfigR
        ? _schedulerConfigR->get().config.cost
        : _config;
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    auto schedInfoCount = requestCount * nodeCount;
    auto requestTokenCounts = std::make_unique<int64_t[]>(requestCount);
    auto totalTokenCount =
        CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, requestTokenCounts.get());
    std::unique_ptr<CostSchedulerKernelInternal::ReplayInfo> replayInfo;
    if (context.isCollectingReplay()) {
        replayInfo = std::make_unique<CostSchedulerKernelInternal::ReplayInfo>();
    }
    auto costs = CostSchedulerKernelInternal::prepare(
        context, requestTokenCounts.get(), config.lambda, config.alpha,
        config.alphaDecay, config.minAlpha,
        static_cast<size_t>(config.maxLoopCount),
        config.optimizationTimeBudgetUs, replayInfo.get());
    if (!costs) {
        return false;
    }
    if (replayInfo) {
        replayInfo->minLoopCount = config.minLoopCount;
        replayInfo->latencyWeightConcavity = config.latencyWeightConcavity;
        replayInfo->latencyTokenNormalizationPower =
            config.latencyTokenNormalizationPower;
        replayInfo->initialWeightSeed = config.initialWeightSeed;
        replayInfo->discreteCheckInterval =
            static_cast<size_t>(config.discreteCheckInterval);
        replayInfo->discreteGapThreshold = config.discreteGapThreshold;
        replayInfo->discreteRepairAlphaDecay =
            config.discreteRepairAlphaDecay;
    }
    auto weights = std::make_unique<double[]>(schedInfoCount);
    std::vector<uint32_t> selectedNodeIndexes;
    double forwardValue = 0.0;
    if (!CostSchedulerKernelInternal::optimize(
            context, costs.get(), requestTokenCounts.get(), totalTokenCount,
            config.lambda, config.alpha, config.alphaDecay,
            config.minAlpha,
            static_cast<size_t>(config.maxLoopCount),
            config.optimizationTimeBudgetUs, replayInfo.get(),
            config.latencyWeightConcavity,
            config.initialWeightSeed,
            weights.get(), selectedNodeIndexes, forwardValue,
            config.latencyTokenNormalizationPower,
            static_cast<size_t>(config.discreteCheckInterval),
            config.discreteGapThreshold,
            config.discreteRepairAlphaDecay, config.minLoopCount) ||
        !CostSchedulerKernelInternal::selectNodes(
            context, selectedNodeIndexes.data())) {
        return false;
    }
    if (!context.setScheduleLoss(forwardValue)) {
        return false;
    }
    if (replayInfo) {
        replayInfo->finish(context, costs.get(), weights.get(),
                           requestCount, nodeCount);
    }
    return true;
}

REGISTER_KERNEL(CostSchedulerKernel);

} // namespace navi_sched
