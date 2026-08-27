package org.flexlb.balance.strategy;

import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Selects the shortest projected-TTFT pool and uses live LRU as its final
 * fairness tie-breaker. A lost endpoint CAS causes one complete rescan.
 */
@Component
public class ShortestTTFTStrategy extends CostBasedPrefillStrategy {

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory,
                                EngineHealthReporter engineHealthReporter) {
        super(engineWorkerStatus,
                cacheAwareService,
                resourceMeasureFactory,
                engineHealthReporter);
    }

    @Override
    public boolean supports(
            RoleType role,
            RoutingConfig.EndpointSelectorConfig configured) {
        return (role == RoleType.PREFILL || role == RoleType.PDFUSION)
                && configured
                instanceof RoutingConfig.EstimatedTtftSelectorConfig estimated
                && estimated.getCandidateChoice()
                instanceof RoutingConfig.LeastRecentlyUsedInPoolConfig;
    }

    @Override
    protected int selectBestCandidate(CandidateSet survivors,
                                      long minProjectedTtftMs,
                                      BalanceContext balanceContext,
                                      RoleType roleType,
                                      String group,
                                      long seqLen,
                                      FlexlbConfig config) {
        if (survivors.size() == 0) {
            return -1;
        }

        List<Integer> baselinePool = shortestCandidateIndexes(
                survivors,
                config.shortestTtftCandidateCount(survivors.size()));
        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        CacheAffinityPolicy.Decision affinity = null;
        if (cacheAffinity != null) {
            boolean hasAvailable = survivors.hasAvailable();
            long referenceHitTokens = 0L;
            for (int i = 0; i < survivors.size(); i++) {
                if ((!hasAvailable || survivors.resourceAvailable(i))
                        && survivors.projectedTtftMs(i) == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(
                            referenceHitTokens, survivors.cacheHit(i));
                }
            }
            affinity = CacheAffinityPolicy.evaluate(
                    survivors.size(),
                    survivors::projectedTtftMs,
                    survivors::cacheHit,
                    minProjectedTtftMs,
                    referenceHitTokens,
                    seqLen,
                    remainingCacheAffinityBudgetMs(
                            balanceContext, cacheAffinity.getMaxExtraTtftMs()),
                    cacheAffinity.getMinPrefixHitPercent());
        }

        ClaimResult result = claimCandidate(survivors, affinity, baselinePool);
        if (result == null) {
            return -1;
        }

        if (affinity != null) {
            String reason = result.preferred()
                    ? CacheAffinityPolicy.Reason.CACHE_LEADER.name()
                    : affinity.hasPreference()
                            ? "CACHE_AFFINITY_FALLBACK"
                            : affinity.reason().name();
            reportCacheAffinityDecision(
                    roleType, survivors.endpoint(result.index()).getIp(), reason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "ShortestTtft cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minTtftMs: {}, selectedTtftMs: {}, "
                                + "ttftCutoffMs: {}, hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddress(result.index()),
                        affinity.minProjectedTtftMs(),
                        survivors.projectedTtftMs(result.index()),
                        affinity.projectedTtftCutoffMs(),
                        survivors.cacheHit(result.index()),
                        reason);
            }
        }
        return result.index();
    }

    /** Shortest-TTFT selection never consumes suffix drain. */
    @Override
    protected RouteProjection.Demand projectionDemand(FlexlbConfig config) {
        return RouteProjection.Demand.TTFT_ONLY;
    }

    /**
     * Keep only indexes for the configured shortest-TTFT pool. Production does
     * not materialize endpoint score objects; the list only carries indexes
     * into the common projection result.
     */
    private static List<Integer> shortestCandidateIndexes(
            CandidateSet candidates, int configuredCount) {
        boolean hasAvailable = candidates.hasAvailable();
        List<Integer> indexes = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            if (!hasAvailable || candidates.resourceAvailable(i)) {
                indexes.add(i);
            }
        }
        indexes.sort(Comparator
                .comparingLong((Integer index) -> candidates.projectedTtftMs(index))
                .thenComparingInt(Integer::intValue));
        int count = Math.min(Math.max(1, configuredCount), indexes.size());
        return indexes.subList(0, count);
    }

    /**
     * Scan the preferred pool (when present), otherwise the baseline pool.
     * A CAS conflict invalidates the live LRU observation, so the final round
     * starts again from a complete scan and publishes to that fresh target
     * atomically. The cold publish prevents ordinary contention from turning a
     * non-empty candidate pool into a false "no available worker" result.
     */
    private static ClaimResult claimCandidate(
            CandidateSet candidates,
            CacheAffinityPolicy.Decision affinity,
            List<Integer> baselinePool) {
        LiveCandidate target = findTarget(candidates, affinity, baselinePool);
        if (target == null) {
            return null;
        }
        if (claim(target.clock(), target.expected())) {
            return new ClaimResult(target.index(), target.preferred());
        }

        target = findTarget(candidates, affinity, baselinePool);
        if (target == null) {
            return null;
        }
        publishMonotonically(target.clock());
        return new ClaimResult(target.index(), target.preferred());
    }

    private static LiveCandidate findTarget(
            CandidateSet candidates,
            CacheAffinityPolicy.Decision affinity,
            List<Integer> baselinePool) {
        LiveCandidate target = affinity != null && affinity.hasPreference()
                ? findLiveLru(candidates, affinity)
                : null;
        if (target != null) {
            return target.asPreferred();
        }
        return findLiveLru(candidates, baselinePool);
    }

    private static LiveCandidate findLiveLru(
            CandidateSet candidates,
            CacheAffinityPolicy.Decision affinity) {
        int first = affinity.preferredIndex(0);
        long bestHit = candidates.cacheHit(first);
        long bestTtftMs = candidates.projectedTtftMs(first);
        LiveCandidate selected = null;
        for (int i = 0; i < affinity.preferredCount(); i++) {
            int index = affinity.preferredIndex(i);
            if (candidates.cacheHit(index) != bestHit
                    || candidates.projectedTtftMs(index) != bestTtftMs) {
                break;
            }
            selected = chooseLiveLru(
                    candidates, index, selected);
        }
        return selected;
    }

    private static LiveCandidate findLiveLru(
            CandidateSet candidates, List<Integer> indexes) {
        LiveCandidate selected = null;
        for (int index : indexes) {
            selected = chooseLiveLru(candidates, index, selected);
        }
        return selected;
    }

    private static LiveCandidate chooseLiveLru(
            CandidateSet candidates, int index, LiveCandidate selected) {
        AtomicLong clock = candidates.endpoint(index).getLastSelectedTime();
        long live = clock.get();
        if (live == Long.MAX_VALUE) {
            return selected;
        }
        if (selected == null
                || live < selected.expected()
                || live == selected.expected()
                        && (candidates.projectedTtftMs(index)
                                < candidates.projectedTtftMs(selected.index())
                        || candidates.projectedTtftMs(index)
                                == candidates.projectedTtftMs(selected.index())
                                && index < selected.index())) {
            return new LiveCandidate(index, clock, live);
        }
        return selected;
    }

    private static boolean claim(AtomicLong clock, long expected) {
        long nowMicros = System.nanoTime() / 1_000;
        long claimedAt = Math.max(nowMicros, expected + 1L);
        return clock.compareAndSet(expected, claimedAt);
    }

    private static void publishMonotonically(AtomicLong clock) {
        long nowMicros = System.nanoTime() / 1_000;
        clock.updateAndGet(current -> current == Long.MAX_VALUE
                ? Long.MAX_VALUE
                : Math.max(nowMicros, current + 1L));
    }

    private record LiveCandidate(
            int index, AtomicLong clock, long expected, boolean preferred) {
        private LiveCandidate(int index, AtomicLong clock, long expected) {
            this(index, clock, expected, false);
        }

        private LiveCandidate asPreferred() {
            return new LiveCandidate(index, clock, expected, true);
        }
    }

    private record ClaimResult(int index, boolean preferred) {}
}
