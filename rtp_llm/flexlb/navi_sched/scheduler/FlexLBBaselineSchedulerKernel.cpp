#include "aios/navi_sched/scheduler/FlexLBBaselineSchedulerKernel.h"

#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace navi_sched {
namespace {

struct Candidate {
    size_t nodeIndex = 0;
    int64_t score = 0;
    double endpointWaitMs = 0.0;
    int64_t pendingCount = 0;
};

int64_t estimateSingleRequestScore(const NaviSchedScheduleContext &context,
                                   size_t requestIndex, size_t nodeIndex,
                                   const std::string &predictorType,
                                   double computeTokenCost,
                                   double cacheHitTokenCost)
{
    const auto &node = context.getNode(nodeIndex);
    const auto &request = context.getRequest(requestIndex).request;
    auto cacheHitTokens =
        context.getCacheHitBlockCount(requestIndex, nodeIndex) * request.blockSize;
    if (cacheHitTokens >= request.promptTokenCount) {
        cacheHitTokens = std::max<int64_t>(0, request.promptTokenCount - request.blockSize);
    }
    cacheHitTokens = std::clamp<int64_t>(cacheHitTokens, 0, request.promptTokenCount);
    if (predictorType == "flexlb_formula") {
        return static_cast<int64_t>(
            (request.promptTokenCount - cacheHitTokens) * computeTokenCost +
            cacheHitTokens * cacheHitTokenCost);
    }
    if (!node.hasLatencyModel() ||
        NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL != node.getLatencyFormula() ||
        0 != node.getLatencyBinaryVersion()) {
        return 0.0;
    }
    NaviSchedLatencyFeatures features;
    features.appendRequest(cacheHitTokens, request.promptTokenCount - cacheHitTokens);
    const auto latencyMs =
        NonLinearPrefillModelR::predict(features, node.getLatencyParameters());
    return std::isfinite(latencyMs) && latencyMs > 0.0
               ? static_cast<int64_t>(latencyMs)
               : 0;
}

}

const std::string FlexLBBaselineSchedulerKernel::KERNEL_ID =
    "navi_sched.flexlb_baseline_scheduler_k";

void FlexLBBaselineSchedulerKernel::def(navi::KernelDefBuilder &builder) const
{
    SchedulerBaseKernel::def(builder, KERNEL_ID);
}

bool FlexLBBaselineSchedulerKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "predictor_type", _predictorType, _predictorType);
    NAVI_JSONIZE(ctx, "compute_token_cost", _computeTokenCost,
                 _computeTokenCost);
    NAVI_JSONIZE(ctx, "cache_hit_token_cost", _cacheHitTokenCost,
                 _cacheHitTokenCost);
    NAVI_JSONIZE(ctx, "use_reported_load", _useReportedLoad,
                 _useReportedLoad);
    NAVI_JSONIZE(ctx, "score_tie_random_enabled", _scoreTieRandomEnabled,
                 _scoreTieRandomEnabled);
    NAVI_JSONIZE(ctx, "score_tie_threshold_pct", _scoreTieThresholdPct,
                 _scoreTieThresholdPct);
    NAVI_JSONIZE(ctx, "score_tie_threshold_ms", _scoreTieThresholdMs,
                 _scoreTieThresholdMs);
    NAVI_JSONIZE(ctx, "hotspot_multiplier", _hotspotMultiplier,
                 _hotspotMultiplier);
    NAVI_JSONIZE(ctx, "imbalance_multiplier", _imbalanceMultiplier,
                 _imbalanceMultiplier);
    return (_predictorType == "flexlb_formula" ||
            _predictorType == "navi_latency_model") &&
           std::isfinite(_computeTokenCost) && _computeTokenCost >= 0.0 &&
           std::isfinite(_cacheHitTokenCost) &&
           _cacheHitTokenCost >= 0.0 &&
           std::isfinite(_scoreTieThresholdPct) && _scoreTieThresholdPct >= 0.0 &&
           std::isfinite(_scoreTieThresholdMs) && _scoreTieThresholdMs >= 0.0 &&
           std::isfinite(_hotspotMultiplier) && _hotspotMultiplier >= 0.0 &&
           std::isfinite(_imbalanceMultiplier) && _imbalanceMultiplier >= 0.0;
}

bool FlexLBBaselineSchedulerKernel::schedule(
    NaviSchedScheduleContext &context) const
{
    if (context.isCollectingReplay()) {
        context.setSchedulerInfo({
            {"predictor_type", _predictorType},
            {"compute_token_cost", _computeTokenCost},
            {"cache_hit_token_cost", _cacheHitTokenCost},
            {"use_reported_load", _useReportedLoad},
            {"score_tie_random_enabled", _scoreTieRandomEnabled},
            {"score_tie_threshold_pct", _scoreTieThresholdPct},
            {"score_tie_threshold_ms", _scoreTieThresholdMs},
            {"hotspot_multiplier", _hotspotMultiplier},
            {"imbalance_multiplier", _imbalanceMultiplier},
        });
    }
    const auto nodeCount = context.nodeCount();
    std::vector<Candidate> candidates;
    candidates.reserve(nodeCount);
    for (size_t requestIndex = 0; requestIndex < context.requestCount();
         ++requestIndex) {
        candidates.clear();
        double sumWaitMs = 0.0;
        int64_t sumPendingCount = 0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            const auto singleRequestScore =
                estimateSingleRequestScore(
                    context, requestIndex, nodeIndex, _predictorType,
                    _computeTokenCost, _cacheHitTokenCost);
            if (singleRequestScore <= 0) {
                continue;
            }
            const auto &node = context.getNode(nodeIndex);
            const auto endpointWaitMs =
                _useReportedLoad ? node.getEstimatedQueueWaitMs() : 0.0;
            const auto pendingCount = _useReportedLoad
                                          ? static_cast<int64_t>(
                                                node.getInflightRequests())
                                          : 0;
            candidates.push_back({
                nodeIndex,
                singleRequestScore + static_cast<int64_t>(endpointWaitMs),
                endpointWaitMs,
                pendingCount,
            });
            sumWaitMs += endpointWaitMs;
            sumPendingCount += pendingCount;
        }
        if (candidates.empty()) {
            return context.reject(requestIndex, "flexlb_no_predictor");
        }

        const auto averageWaitMs = sumWaitMs / candidates.size();
        const auto averagePendingCount =
            static_cast<double>(sumPendingCount) / candidates.size();
        std::vector<const Candidate *> survivors;
        survivors.reserve(candidates.size());
        const Candidate *leastLoaded = nullptr;
        for (const auto &candidate : candidates) {
            if (!leastLoaded ||
                candidate.endpointWaitMs < leastLoaded->endpointWaitMs) {
                leastLoaded = &candidate;
            }
            if (_hotspotMultiplier > 0.0 && averagePendingCount > 0.0 &&
                candidate.pendingCount > averagePendingCount * _hotspotMultiplier) {
                continue;
            }
            if (_imbalanceMultiplier > 0.0 && averageWaitMs > 0.0 &&
                candidate.endpointWaitMs > averageWaitMs * _imbalanceMultiplier) {
                continue;
            }
            survivors.push_back(&candidate);
        }
        if (survivors.empty()) {
            survivors.push_back(leastLoaded);
        }

        auto minimumScore = std::numeric_limits<int64_t>::max();
        for (const auto *candidate : survivors) {
            minimumScore = std::min(minimumScore, candidate->score);
        }
        const auto tieThreshold = _scoreTieRandomEnabled
                                      ? std::max<int64_t>(
                                            static_cast<int64_t>(
                                                minimumScore *
                                                _scoreTieThresholdPct),
                                            static_cast<int64_t>(
                                                _scoreTieThresholdMs))
                                      : 0;
        const Candidate *selected = nullptr;
        uint64_t selectedRank = 0;
        const auto requestId = context.getRequest(requestIndex).request.requestId;
        for (const auto *candidate : survivors) {
            if (candidate->score > minimumScore + tieThreshold) {
                continue;
            }
            const auto rank = getNodeRank(
                requestId, context.getNode(candidate->nodeIndex).getAgentIdHash());
            if (!selected || rank > selectedRank) {
                selected = candidate;
                selectedRank = rank;
            }
        }
        if (!selected ||
            !context.selectNode(requestIndex, selected->nodeIndex,
                                "flexlb_baseline_selected")) {
            return false;
        }
    }
    return true;
}

uint64_t FlexLBBaselineSchedulerKernel::getNodeRank(
    uint64_t requestId, uint64_t agentIdHash)
{
    auto value = requestId ^
                 (agentIdHash + 0x9e3779b97f4a7c15ULL +
                  (requestId << 6) + (requestId >> 2));
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

REGISTER_KERNEL(FlexLBBaselineSchedulerKernel);

}
