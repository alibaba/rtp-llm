package org.flexlb.balance.dp;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;

/**
 * Batch-level cache-aware group selector. For each candidate DP group, sums
 * the estimated prefill time across all requests in the batch — the group with
 * the lowest aggregate prefill cost (= best cache affinity) wins.
 *
 * <p>Falls back to round-robin when no request carries {@code blockCacheKeys}.
 */
@Component("cacheAwareGroupSelector")
public class CacheAwareGroupSelector implements GroupSelector {

    public static final String NAME = "CACHE_AWARE";

    private final CacheAwareService cacheAwareService;

    public CacheAwareGroupSelector(CacheAwareService cacheAwareService) {
        this.cacheAwareService = cacheAwareService;
    }

    @Override
    public WorkerStatus select(List<WorkerStatus> candidates, DispatchContext ctx) {
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        if (candidates.size() == 1) {
            return candidates.get(0);
        }

        List<QueuedRequest> requests = ctx.requests();
        boolean anyCacheKeys = requests != null && requests.stream()
                .anyMatch(qr -> qr.ctx() != null && qr.ctx().getRequest() != null
                        && qr.ctx().getRequest().getBlockCacheKeys() != null
                        && !qr.ctx().getRequest().getBlockCacheKeys().isEmpty());

        if (!anyCacheKeys) {
            return leastLoadedFallback(candidates);
        }

        WorkerStatus best = null;
        long bestScore = Long.MAX_VALUE;

        for (WorkerStatus candidate : candidates) {
            long groupScore = scoreCandidateForBatch(candidate, requests);
            if (groupScore < bestScore) {
                bestScore = groupScore;
                best = candidate;
            }
        }

        return best != null ? best : leastLoadedFallback(candidates);
    }

    @Override
    public String name() {
        return NAME;
    }

    private long scoreCandidateForBatch(WorkerStatus candidate, List<QueuedRequest> requests) {
        String ipPort = candidate.getIpPort();
        long blockSize = getBlockSize(candidate);
        long score = candidate.getRunningQueueTime().get();

        for (QueuedRequest qr : requests) {
            if (qr.ctx() == null || qr.ctx().getRequest() == null) {
                continue;
            }
            long seqLen = qr.ctx().getRequest().getSeqLen();
            List<Long> cacheKeys = qr.ctx().getRequest().getBlockCacheKeys();

            long cacheMatchedTokens = 0;
            if (cacheKeys != null && !cacheKeys.isEmpty()) {
                int prefixBlocks = cacheAwareService.findMatchingPrefixLength(ipPort, cacheKeys);
                cacheMatchedTokens = prefixBlocks * blockSize;
            }

            score += TaskInfo.estimatePrefillTimeMs(seqLen, cacheMatchedTokens);
        }
        return score;
    }

    private WorkerStatus leastLoadedFallback(List<WorkerStatus> candidates) {
        return candidates.stream()
                .min(Comparator.comparingLong(w -> w.getRunningQueueTime().get()))
                .orElse(null);
    }

    private static long getBlockSize(WorkerStatus w) {
        if (w.getCacheStatus() != null && w.getCacheStatus().getBlockSize() > 0) {
            return w.getCacheStatus().getBlockSize();
        }
        return 1;
    }
}
