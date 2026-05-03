package org.flexlb.balance.dp;

import org.flexlb.cache.service.CacheAwareService;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Longest Processing Time first — assigns requests to dp_ranks so that
 * the total effective compute across ranks is as balanced as possible.
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Compute effective compute per request (seqLen minus cache-aware
 *       prefix hit tokens).</li>
 *   <li>Sort requests descending by effective compute.</li>
 *   <li>Greedily assign each request to the rank with the lowest current
 *       cumulative load (min-heap).</li>
 * </ol>
 *
 * <p>When per-rank cache info is available via
 * {@link CacheAwareService#findMatchingRanksInGroup}, the effective compute
 * accounts for the best rank's cache hit. Otherwise falls back to raw seqLen.
 */
@Component("lptDpAssign")
public class LptDpAssignStrategy implements DpAssignStrategy {

    public static final String NAME = "LPT";

    private final CacheAwareService cacheAwareService;

    public LptDpAssignStrategy(CacheAwareService cacheAwareService) {
        this.cacheAwareService = cacheAwareService;
    }

    @Override
    public List<RankAssignment> assign(PrefillBatch batch) {
        int dpSize = batch.dpSize();
        if (dpSize <= 0) {
            throw new IllegalArgumentException("dpSize must be > 0, got " + dpSize);
        }

        List<PendingRequest> requests = batch.requests();
        if (requests.isEmpty()) {
            return List.of();
        }

        String groupIpPort = batch.prefillTarget().getServerIp() + ":"
                + batch.prefillTarget().getHttpPort();
        long blockSize = batch.blockSize();

        List<RequestWithCompute> sorted = new ArrayList<>(requests.size());
        for (PendingRequest req : requests) {
            long seqLen = 0;
            List<Long> cacheKeys = null;
            if (req.ctx() != null && req.ctx().getRequest() != null) {
                seqLen = req.ctx().getRequest().getSeqLen();
                cacheKeys = req.ctx().getRequest().getBlockCacheKeys();
            }
            long effectiveCompute = seqLen;
            if (cacheKeys != null && !cacheKeys.isEmpty()) {
                Map<Integer, Integer> perRank = cacheAwareService.findMatchingRanksInGroup(
                        groupIpPort, cacheKeys);
                if (!perRank.isEmpty()) {
                    int bestHit = perRank.values().stream().mapToInt(Integer::intValue).max().orElse(0);
                    effectiveCompute = Math.max(0, seqLen - bestHit * blockSize);
                }
            }
            sorted.add(new RequestWithCompute(req, effectiveCompute));
        }
        sorted.sort(Comparator.comparingLong(RequestWithCompute::effectiveCompute).reversed());

        PriorityQueue<long[]> rankHeap = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        for (int rank = 0; rank < dpSize; rank++) {
            rankHeap.offer(new long[]{0, rank});
        }

        List<RankAssignment> result = new ArrayList<>(sorted.size());
        for (RequestWithCompute rwc : sorted) {
            long[] leastLoaded = rankHeap.poll();
            int rank = (int) leastLoaded[1];
            leastLoaded[0] += rwc.effectiveCompute;
            rankHeap.offer(leastLoaded);
            result.add(new RankAssignment(rwc.request, rank));
        }

        return result;
    }

    @Override
    public String name() {
        return NAME;
    }

    private record RequestWithCompute(PendingRequest request, long effectiveCompute) {}
}
