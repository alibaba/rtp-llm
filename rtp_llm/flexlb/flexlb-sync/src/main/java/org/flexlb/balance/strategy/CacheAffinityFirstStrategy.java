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

/**
 * Cache-affinity-first routing with bounded queue spillover.
 *
 * <p>Cold requests have no cache lead and therefore follow the shortest TTFT. Each local
 * assignment immediately increases that worker's estimated queue, so subsequent cold requests
 * naturally spread to other workers. Once cache locality appears, the cache leader is preferred
 * until its additional queue work exceeds the configured multiple of the saved prefill work.
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

        return claimPreferredWorker(preferredWorker, sortedWorkers, shortestTtftWorker);
    }

    private long estimateQueueWork(
            ScoredWorker worker, long seqLen, double cacheHitDiscount) {
        long requestPrefillWork = TaskInfo.estimatePrefillTimeMs(
                seqLen, worker.hitCacheTokens(), cacheHitDiscount);
        return Math.max(0, worker.ttft() - requestPrefillWork);
    }
}
