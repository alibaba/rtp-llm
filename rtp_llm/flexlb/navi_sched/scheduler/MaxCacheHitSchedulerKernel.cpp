#include "aios/navi_sched/scheduler/MaxCacheHitSchedulerKernel.h"

#include "autil/HashUtil.h"

namespace navi_sched {

const std::string MaxCacheHitSchedulerKernel::KERNEL_ID =
    "navi_sched.max_cache_hit_scheduler_k";

void MaxCacheHitSchedulerKernel::def(navi::KernelDefBuilder &builder) const
{
    SchedulerBaseKernel::def(builder, KERNEL_ID);
}

bool MaxCacheHitSchedulerKernel::schedule(NaviSchedScheduleContext &context) const
{
    const auto requestCount = context.requestCount();
    const auto nodeCount = context.nodeCount();
    std::vector<int64_t> cacheHitBlockCounts;
    std::vector<int64_t> maxCacheHitBlockCounts(requestCount);
    std::vector<uint64_t> selectedRanks(requestCount);
    std::vector<size_t> selectedNodeIndexes(requestCount);
    if (!context.getCacheHitBlockCounts(0, cacheHitBlockCounts)) {
        return false;
    }
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        const auto requestId = context.getRequest(requestIndex).request.requestId;
        maxCacheHitBlockCounts[requestIndex] =
            cacheHitBlockCounts[requestIndex];
        selectedRanks[requestIndex] = getNodeRank(
            requestId, context.getNode(0).getAgentIdHash());
    }
    for (size_t nodeIndex = 1; nodeIndex < nodeCount; ++nodeIndex) {
        if (!context.getCacheHitBlockCounts(
                nodeIndex, cacheHitBlockCounts)) {
            return false;
        }
        auto agentIdHash = context.getNode(nodeIndex).getAgentIdHash();
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            const auto requestId =
                context.getRequest(requestIndex).request.requestId;
            auto cacheHitBlockCount = cacheHitBlockCounts[requestIndex];
            auto rank =
                getNodeRank(requestId, agentIdHash);
            if (cacheHitBlockCount > maxCacheHitBlockCounts[requestIndex] ||
                (cacheHitBlockCount == maxCacheHitBlockCounts[requestIndex] &&
                 rank > selectedRanks[requestIndex])) {
                maxCacheHitBlockCounts[requestIndex] = cacheHitBlockCount;
                selectedNodeIndexes[requestIndex] = nodeIndex;
                selectedRanks[requestIndex] = rank;
            }
        }
    }
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        if (!context.selectNode(requestIndex, selectedNodeIndexes[requestIndex],
                                "max_cache_hit_selected")) {
            return false;
        }
    }
    return true;
}

uint64_t MaxCacheHitSchedulerKernel::getNodeRank(uint64_t requestId, uint64_t agentIdHash)
{
    uint64_t rank = requestId;
    autil::HashUtil::combineHash(rank, agentIdHash);
    return rank;
}

REGISTER_KERNEL(MaxCacheHitSchedulerKernel);

}
