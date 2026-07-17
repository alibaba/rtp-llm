package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Cache-affinity-first routing with bounded queue spillover.
 *
 * <p>Cold requests have no cache lead and therefore follow the shortest TTFT. Each local
 * assignment immediately increases that worker's estimated queue, so subsequent cold requests
 * naturally spread to other workers. A cache-poor worker that remains idle receives a bounded
 * probe request so it can build the shared prefix instead of starving indefinitely. Once cache
 * locality appears, the cache leader is preferred until its additional queue work exceeds the
 * configured multiple of the saved prefill work.
 */
@Component("cacheAffinityFirstStrategy")
public class CacheAffinityFirstStrategy extends ShortestTTFTStrategy {

    public CacheAffinityFirstStrategy(
            EngineWorkerStatus engineWorkerStatus,
            EngineHealthReporter engineHealthReporter,
            CacheAwareService cacheAwareService,
            ResourceMeasureFactory resourceMeasureFactory) {
        super(
                engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST);
    }

    @Override
    protected ScoredWorker selectBestWorker(
            List<ScoredWorker> scoredWorkers,
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            FlexlbConfig config) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        List<ScoredWorker> sortedWorkers = sortByTTFT(scoredWorkers);
        ScoredWorker shortestTtftWorker = sortedWorkers.getFirst();
        ScoredWorker cacheLeader = sortedWorkers.stream()
                .min(Comparator.comparingLong(ScoredWorker::hitCacheTokens)
                        .reversed()
                        .thenComparingLong(ScoredWorker::ttft)
                        .thenComparingLong(ScoredWorker::lastSelectedTime))
                .orElse(shortestTtftWorker);

        ScoredWorker coldWorker = findColdWorkerForProbe(
                sortedWorkers, cacheLeader, seqLen, config);
        if (coldWorker != null) {
            Logger.debug(
                    "Cache affinity first cold worker probe - cacheLeader: {}, cacheLeaderHitTokens: {}, coldWorker: {}, coldWorkerHitTokens: {}, idleTimeUs: {}",
                    cacheLeader.worker().getIpPort(),
                    cacheLeader.hitCacheTokens(),
                    coldWorker.worker().getIpPort(),
                    coldWorker.hitCacheTokens(),
                    idleTimeUs(coldWorker));
            return selectAndRecordDecision(
                    coldWorker,
                    sortedWorkers,
                    shortestTtftWorker,
                    balanceContext,
                    roleType,
                    group,
                    seqLen,
                    config,
                    "COLD_WORKER_PROBE");
        }

        long blockSize = cacheLeader.worker().getCacheStatus() == null
                ? 0
                : cacheLeader.worker().getCacheStatus().getBlockSize();
        long cacheLeadTokens = Math.max(
                0, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens());
        long minimumCacheLeadTokens = blockSize
                * Math.max(0, config.getPrefillCachePreferenceMinBlockGap());
        long additionalQueueWork = Math.max(
                0,
                estimateQueueWork(cacheLeader, seqLen, config.getPrefillCacheHitDiscount())
                        - estimateQueueWork(
                                shortestTtftWorker,
                                seqLen,
                                config.getPrefillCacheHitDiscount()));
        double toleratedQueueWork = cacheLeadTokens
                * Math.max(0.0, config.getPrefillCacheHitDiscount())
                * Math.max(0.0, config.getCacheAffinityFirstQueueToleranceFactor());

        boolean cacheLeaderWithinLoadGuard = cacheLeader.equals(shortestTtftWorker)
                || (blockSize > 0
                        && cacheLeadTokens >= minimumCacheLeadTokens
                        && additionalQueueWork <= toleratedQueueWork);
        ScoredWorker preferredWorker = cacheLeaderWithinLoadGuard
                ? cacheLeader
                : shortestTtftWorker;

        Logger.debug(
                "Cache affinity first - shortest: {}, cacheLeader: {}, cacheLeadTokens: {}, additionalQueueWork: {}, toleratedQueueWork: {}, preferred: {}",
                shortestTtftWorker.worker().getIpPort(),
                cacheLeader.worker().getIpPort(),
                cacheLeadTokens,
                additionalQueueWork,
                toleratedQueueWork,
                preferredWorker.worker().getIpPort());

        String selectionReason = preferredWorker.equals(cacheLeader)
                        && !preferredWorker.equals(shortestTtftWorker)
                ? "CACHE_LEADER"
                : "SHORTEST_TTFT";
        return selectAndRecordDecision(
                preferredWorker,
                sortedWorkers,
                shortestTtftWorker,
                balanceContext,
                roleType,
                group,
                seqLen,
                config,
                selectionReason);
    }

    private ScoredWorker selectAndRecordDecision(
            ScoredWorker preferredWorker,
            List<ScoredWorker> sortedWorkers,
            ScoredWorker shortestTtftWorker,
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            FlexlbConfig config,
            String selectionReason) {
        ScoredWorker selectedWorker = claimPreferredWorker(
                preferredWorker, sortedWorkers, shortestTtftWorker);
        String effectiveReason = selectedWorker.equals(preferredWorker)
                ? selectionReason
                : "CONCURRENT_FALLBACK";
        recordDecisionSnapshot(
                balanceContext,
                selectedWorker,
                sortedWorkers,
                sortedWorkers,
                List.of(),
                shortestTtftWorker.ttft(),
                0,
                roleType,
                group,
                seqLen,
                config.getPrefillCacheHitDiscount(),
                effectiveReason);
        return selectedWorker;
    }

    private ScoredWorker findColdWorkerForProbe(
            List<ScoredWorker> workers,
            ScoredWorker cacheLeader,
            long seqLen,
            FlexlbConfig config) {
        long probeIntervalMs = config.getCacheAffinityFirstColdWorkerProbeIntervalMs();
        if (probeIntervalMs <= 0 || workers.size() < 2) {
            return null;
        }

        long blockSize = cacheLeader.worker().getCacheStatus() == null
                ? 0
                : cacheLeader.worker().getCacheStatus().getBlockSize();
        if (blockSize <= 0) {
            return null;
        }

        long minimumCacheGap = blockSize
                * Math.max(0, config.getPrefillCachePreferenceMinBlockGap());
        long probeIntervalUs = TimeUnit.MILLISECONDS.toMicros(probeIntervalMs);
        long nowUs = System.nanoTime() / 1000;
        return workers.stream()
                .filter(worker -> !worker.equals(cacheLeader))
                .filter(worker -> {
                    long cacheGap = cacheLeader.hitCacheTokens() - worker.hitCacheTokens();
                    return cacheGap > 0 && cacheGap >= minimumCacheGap;
                })
                .filter(worker -> estimateQueueWork(
                        worker, seqLen, config.getPrefillCacheHitDiscount()) == 0)
                .filter(worker -> worker.lastSelectedTime() < 0
                        || nowUs - worker.lastSelectedTime() >= probeIntervalUs)
                .min(Comparator.comparingLong(ScoredWorker::lastSelectedTime)
                        .thenComparingLong(ScoredWorker::ttft))
                .orElse(null);
    }

    private long idleTimeUs(ScoredWorker worker) {
        if (worker.lastSelectedTime() < 0) {
            return -1;
        }
        return Math.max(0, System.nanoTime() / 1000 - worker.lastSelectedTime());
    }

    private long estimateQueueWork(
            ScoredWorker worker, long seqLen, double cacheHitDiscount) {
        long requestPrefillWork = TaskInfo.estimatePrefillTimeMs(
                seqLen, worker.hitCacheTokens(), cacheHitDiscount);
        return Math.max(0, worker.ttft() - requestPrefillWork);
    }
}
