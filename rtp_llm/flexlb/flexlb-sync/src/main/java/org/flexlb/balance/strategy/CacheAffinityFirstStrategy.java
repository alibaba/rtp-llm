package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.pv.ShortestTtftDecision.CacheAffinityDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.StrategySelectionReason;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Cache-affinity-first routing with bounded additional prefill work.
 *
 * <p>Cold requests have no cache lead and therefore follow the shortest TTFT. Each local
 * assignment immediately increases that worker's estimated queue, so subsequent cold requests
 * naturally spread to other workers. A cache leader may have a bounded higher TTFT when its cache
 * lead exists and the additional work stays within a fixed bound.
 */
@Component("cacheAffinityFirstStrategy")
public class CacheAffinityFirstStrategy extends ShortestTTFTStrategy {

    // Highest cache hit wins; ties prefer lower TTFT, then the worker selected least recently.
    private static final Comparator<ScoredWorker> CACHE_LEADER_ORDER =
            Comparator.comparingLong(ScoredWorker::hitCacheTokens).reversed()
                    .thenComparingLong(ScoredWorker::ttft)
                    .thenComparingLong(ScoredWorker::lastSelectedTime);

    public CacheAffinityFirstStrategy(EngineWorkerStatus engineWorkerStatus,
                                      EngineHealthReporter engineHealthReporter, CacheAwareService cacheAwareService,
                                      ResourceMeasureFactory resourceMeasureFactory) {
        super(
                engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST);
    }

    @Override
    protected ScoredWorker selectBestWorker(List<ScoredWorker> scoredWorkers, BalanceContext balanceContext,
                                            RoleType roleType, String group, long seqLen, FlexlbConfig config) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        // Keep every scored worker visible in the decision snapshot, but apply the
        // outstanding-work threshold before choosing a target.
        List<ScoredWorker> workersByTtft = sortByTTFT(scoredWorkers);
        List<ScoredWorker> eligibleWorkers = filterByOutstandingUncachedTokens(
                workersByTtft, roleType, seqLen, config);
        ScoredWorker cacheLeader = findCacheLeader(workersByTtft);
        if (eligibleWorkers.isEmpty()) {
            return selectShortestTtftWhenAllWorkersExceedOutstandingThreshold(
                    balanceContext, workersByTtft, cacheLeader, roleType, group, seqLen, config);
        }
        ScoredWorker shortestTtftWorker = eligibleWorkers.getFirst();

        // The global cache leader is checked before fallback so an overloaded cache
        // leader falls directly back to the shortest eligible TTFT worker.
        CacheLeaderDecision decision;
        if (!cacheLeaderMeetsMinimumHitRate(cacheLeader, seqLen, config)) {
            decision = rejectCacheLeader(
                    cacheLeader,
                    shortestTtftWorker,
                    config,
                    StrategySelectionReason.SHORTEST_TTFT_LOW_CACHE_HIT,
                    false);
        } else if (eligibleWorkers.contains(cacheLeader)) {
            decision = evaluateCacheLeader(cacheLeader, shortestTtftWorker, config);
        } else {
            decision = rejectCacheLeader(
                    cacheLeader,
                    shortestTtftWorker,
                    config,
                    StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD,
                    true);
        }

        // Low effective cache hit makes cache affinity meaningless, so preserve TTFT order
        // even if the original shortest worker was selected concurrently.
        ScoredWorker selectedWorker = decision.cacheAffinityEnabled()
                ? selectWorkerByCacheAffinity(
                        decision.preferredWorker(), eligibleWorkers, shortestTtftWorker, config)
                : selectFirstWorkerWithoutConcurrentConflict(eligibleWorkers, shortestTtftWorker);
        StrategySelectionReason selectionReason = selectedWorker.equals(decision.preferredWorker())
                ? decision.selectionReason()
                : decision.cacheAffinityEnabled()
                                && satisfiesCacheAffinityTolerance(selectedWorker, shortestTtftWorker, config)
                        ? StrategySelectionReason.CACHE_AFFINITY_FALLBACK
                        : StrategySelectionReason.SHORTEST_TTFT_FALLBACK;

        reportCacheAffinityDecision(roleType, selectedWorker.worker().getIp(), selectionReason.name());

        // Preserve the decision path in the request PV snapshot, including a concurrent fallback.
        recordDecisionSnapshot(balanceContext, selectedWorker, workersByTtft, eligibleWorkers, List.of(),
                shortestTtftWorker.ttft(), 0, roleType, group, seqLen, selectionReason.name(),
                new CacheAffinityDecision(
                        cacheLeader.worker().getLogicalIpPort(),
                        shortestTtftWorker.worker().getLogicalIpPort(),
                        decision.cacheLeadTokens(),
                        decision.extraWorkTokens(),
                        decision.toleratedExtraWorkTokens(),
                        configuredOutstandingUncachedTokensThreshold(config),
                        eligibleWorkers.contains(cacheLeader)),
                outstandingUncachedTokensThresholdForSnapshot(roleType, config));
        return selectedWorker;
    }

    private ScoredWorker findCacheLeader(List<ScoredWorker> workers) {
        return workers.stream().min(CACHE_LEADER_ORDER).orElseThrow();
    }

    private ScoredWorker selectShortestTtftWhenAllWorkersExceedOutstandingThreshold(BalanceContext balanceContext,
                                                                                    List<ScoredWorker> workersByTtft,
                                                                                    ScoredWorker cacheLeader,
                                                                                    RoleType roleType,
                                                                                    String group,
                                                                                    long seqLen,
                                                                                    FlexlbConfig config) {
        ScoredWorker shortestTtftWorker = workersByTtft.getFirst();
        ScoredWorker selectedWorker = selectFirstWorkerWithoutConcurrentConflict(workersByTtft, shortestTtftWorker);
        StrategySelectionReason selectionReason = StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK;

        reportCacheAffinityDecision(roleType, selectedWorker.worker().getIp(), selectionReason.name());
        recordDecisionSnapshot(balanceContext, selectedWorker, workersByTtft, List.of(), List.of(),
                shortestTtftWorker.ttft(), 0, roleType, group, seqLen, selectionReason.name(),
                new CacheAffinityDecision(
                        cacheLeader.worker().getLogicalIpPort(),
                        shortestTtftWorker.worker().getLogicalIpPort(),
                        Math.max(0, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens()),
                        cacheLeader.ttft() - shortestTtftWorker.ttft(),
                        configuredMaxExtraWorkTokens(config),
                        configuredOutstandingUncachedTokensThreshold(config),
                        false),
                outstandingUncachedTokensThresholdForSnapshot(roleType, config));
        return selectedWorker;
    }

    private CacheLeaderDecision rejectCacheLeader(ScoredWorker cacheLeader,
                                                   ScoredWorker shortestTtftWorker,
                                                   FlexlbConfig config,
                                                   StrategySelectionReason selectionReason,
                                                   boolean cacheAffinityEnabled) {
        long cacheLeadTokens = Math.max(0, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens());
        long extraWorkTokens = cacheLeader.ttft() - shortestTtftWorker.ttft();
        return new CacheLeaderDecision(
                shortestTtftWorker,
                cacheLeadTokens,
                extraWorkTokens,
                configuredMaxExtraWorkTokens(config),
                selectionReason,
                cacheAffinityEnabled);
    }

    /**
     * Compare cache lead and prefill work separately. The score already includes the worker queue
     * and this request's uncached prefill work. Cache lead determines whether affinity applies,
     * while the configured hard bound determines how much additional work is tolerated.
     */
    private CacheLeaderDecision evaluateCacheLeader(ScoredWorker cacheLeader, ScoredWorker shortestTtftWorker,
                                                    FlexlbConfig config) {
        // Cache lead is independent of queue state and measures the extra prefix reuse from caching.
        long cacheLeadTokens = Math.max(0, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens());

        // Both scores already include prior queue work and this request's uncached Prefill work.
        long extraWorkTokens = cacheLeader.ttft() - shortestTtftWorker.ttft();

        // Cache affinity may add at most the configured additional uncached work.
        long toleratedExtraWorkTokens = configuredMaxExtraWorkTokens(config);

        // Prefer cache only when its final work stays within the configured tolerance.
        if (extraWorkTokens <= toleratedExtraWorkTokens) {
            return new CacheLeaderDecision(
                    cacheLeader,
                    cacheLeadTokens,
                    extraWorkTokens,
                    toleratedExtraWorkTokens,
                    StrategySelectionReason.CACHE_LEADER,
                    true);
        }
        return new CacheLeaderDecision(
                shortestTtftWorker,
                cacheLeadTokens,
                extraWorkTokens,
                toleratedExtraWorkTokens,
                StrategySelectionReason.SHORTEST_TTFT,
                true);
    }

    /**
     * Keep cache affinity when the preferred worker was selected concurrently. Every cache-first
     * fallback must satisfy the same work tolerance as the original cache leader.
     */
    private ScoredWorker selectWorkerByCacheAffinity(ScoredWorker preferredWorker, List<ScoredWorker> workersByTtft,
                                                     ScoredWorker shortestTtftWorker, FlexlbConfig config) {
        List<ScoredWorker> selectionOrder = new ArrayList<>(workersByTtft.size());

        // Preserve the cache/TTFT decision made above as the first choice.
        selectionOrder.add(preferredWorker);

        // If it conflicts, retain cache affinity among workers that independently pass the guard.
        workersByTtft.stream()
                .filter(worker -> !worker.equals(preferredWorker))
                .filter(worker -> satisfiesCacheAffinityTolerance(worker, shortestTtftWorker, config))
                .sorted(CACHE_LEADER_ORDER)
                .forEach(selectionOrder::add);

        // Once cache-affinity candidates are exhausted, fall back to the minimum-TTFT worker.
        if (!selectionOrder.contains(shortestTtftWorker)) {
            selectionOrder.add(shortestTtftWorker);
        }

        // The remaining workers are already in ascending TTFT order.
        workersByTtft.stream()
                .filter(worker -> !selectionOrder.contains(worker))
                .forEach(selectionOrder::add);

        return selectFirstWorkerWithoutConcurrentConflict(selectionOrder, shortestTtftWorker);
    }

    private boolean satisfiesCacheAffinityTolerance(ScoredWorker worker, ScoredWorker shortestTtftWorker,
                                                    FlexlbConfig config) {
        // Workers without a cache lead belong to the final TTFT fallback, not the affinity group.
        long cacheLeadTokens = worker.hitCacheTokens() - shortestTtftWorker.hitCacheTokens();
        if (cacheLeadTokens <= 0) {
            return false;
        }

        // Apply the same cache-lead versus work-cost rule used for the preferred worker.
        long extraWorkTokens = worker.ttft() - shortestTtftWorker.ttft();
        return extraWorkTokens <= configuredMaxExtraWorkTokens(config);
    }

    private ScoredWorker selectFirstWorkerWithoutConcurrentConflict(List<ScoredWorker> selectionOrder,
                                                                    ScoredWorker fallbackWorker) {
        // A changed timestamp means another scheduler has selected this worker after scoring.
        long now = System.nanoTime() / 1000;
        for (ScoredWorker candidate : selectionOrder) {
            // CAS lets only one scheduler select a worker from the same status snapshot.
            if (candidate.worker().getLastSelectedTime().compareAndSet(candidate.lastSelectedTime(), now)) {
                return candidate;
            }
        }

        // All snapshots changed concurrently; preserve the original minimum-TTFT fallback.
        return fallbackWorker;
    }

    private long configuredMaxExtraWorkTokens(FlexlbConfig config) {
        return Math.max(0L, config.getCacheAffinityFirstMaxExtraWorkTokens());
    }

    private boolean cacheLeaderMeetsMinimumHitRate(ScoredWorker cacheLeader,
                                                    long seqLen,
                                                    FlexlbConfig config) {
        double minimumHitRatePct = configuredMinimumHitRatePct(config);
        return minimumHitRatePct <= 0 || (seqLen > 0
                && cacheLeader.hitCacheTokens() * 100.0 / seqLen >= minimumHitRatePct);
    }

    private double configuredMinimumHitRatePct(FlexlbConfig config) {
        return Math.max(0, config.getCacheAffinityFirstMinHitRate());
    }

    private record CacheLeaderDecision(ScoredWorker preferredWorker,
                                       long cacheLeadTokens,
                                       long extraWorkTokens,
                                       long toleratedExtraWorkTokens,
                                       StrategySelectionReason selectionReason,
                                       boolean cacheAffinityEnabled) {}
}
