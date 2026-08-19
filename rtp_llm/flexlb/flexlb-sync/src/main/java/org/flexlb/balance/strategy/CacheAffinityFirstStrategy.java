package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.StrategyConfigs;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Selects the global cache leader when doing so adds only bounded estimated prefill work.
 *
 * <p>The strategy reuses {@link ShortestTTFTStrategy}'s availability filtering, cache lookup,
 * scoring, local task accounting, and response construction. Requests without a meaningful cache
 * advantage follow the original shortest-TTFT candidate-pool and fairness behavior exactly.
 */
@Component("cacheAffinityFirstStrategy")
public class CacheAffinityFirstStrategy extends ShortestTTFTStrategy {

    private static final String CACHE_LEADER = "CACHE_LEADER";
    private static final String NO_CACHE_LEAD = "NO_CACHE_LEAD";
    private static final String LOW_CACHE_HIT = "LOW_CACHE_HIT";
    private static final String OVER_CAP = "OVER_CAP";
    private static final String CACHE_AFFINITY_FALLBACK = "CACHE_AFFINITY_FALLBACK";
    private static final String SHORTEST_TTFT_FALLBACK = "SHORTEST_TTFT_FALLBACK";

    // Highest cache hit wins. Equal cache hits prefer lower work, then the least recently selected
    // worker, matching the deterministic tie breakers used by shortest TTFT.
    private static final Comparator<ScoredWorker> CACHE_LEADER_ORDER =
            Comparator.comparingLong(ScoredWorker::hitCacheTokens).reversed()
                    .thenComparingLong(ScoredWorker::ttft)
                    .thenComparingLong(ScoredWorker::lastSelectedTime);

    private final EngineHealthReporter engineHealthReporter;

    public CacheAffinityFirstStrategy(EngineWorkerStatus engineWorkerStatus,
                                      EngineHealthReporter engineHealthReporter,
                                      CacheAwareService cacheAwareService,
                                      ResourceMeasureFactory resourceMeasureFactory,
                                      ConfigService configService) {
        super(engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                configService,
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST);
        this.engineHealthReporter = engineHealthReporter;
    }

    @Override
    protected ScoredWorker selectBestWorker(List<ScoredWorker> scoredWorkers,
                                            BalanceContext balanceContext,
                                            RoleType roleType,
                                            String group,
                                            long seqLen,
                                            FlexlbConfig config,
                                            StrategyConfigs.CandidatePoolConfig candidatePoolConfig) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        List<ScoredWorker> workersByTtft = sortByTTFT(scoredWorkers);
        ScoredWorker shortestTtftWorker = workersByTtft.getFirst();
        ScoredWorker cacheLeader = findCacheLeader(scoredWorkers);

        if (seqLen <= 0) {
            return selectBaseline(scoredWorkers, balanceContext, roleType, group, seqLen, config,
                    candidatePoolConfig, NO_CACHE_LEAD, cacheLeader, shortestTtftWorker);
        }

        // A cold request, or a request for which every worker has the same cache prefix, has no
        // affinity signal. Preserve the complete legacy candidate-pool/fairness decision.
        if (!hasRealCacheLead(cacheLeader, shortestTtftWorker, scoredWorkers)) {
            return selectBaseline(scoredWorkers, balanceContext, roleType, group, seqLen, config,
                    candidatePoolConfig, NO_CACHE_LEAD, cacheLeader, shortestTtftWorker);
        }

        if (!meetsMinimumHitRate(cacheLeader, seqLen, config)) {
            return selectBaseline(scoredWorkers, balanceContext, roleType, group, seqLen, config,
                    candidatePoolConfig, LOW_CACHE_HIT, cacheLeader, shortestTtftWorker);
        }

        if (!isWithinExtraWorkCap(cacheLeader, shortestTtftWorker, config)) {
            return selectBaseline(scoredWorkers, balanceContext, roleType, group, seqLen, config,
                    candidatePoolConfig, OVER_CAP, cacheLeader, shortestTtftWorker);
        }

        List<ScoredWorker> selectionOrder = buildCacheAffinitySelectionOrder(
                cacheLeader, shortestTtftWorker, workersByTtft, seqLen, config);
        ScoredWorker selectedWorker = selectFirstWorkerWithoutConcurrentConflict(
                selectionOrder, shortestTtftWorker);

        String decision;
        if (isSameWorker(selectedWorker, cacheLeader)) {
            decision = CACHE_LEADER;
        } else if (isCacheAffinityCandidate(
                selectedWorker, cacheLeader, shortestTtftWorker, seqLen, config)) {
            decision = CACHE_AFFINITY_FALLBACK;
        } else {
            decision = SHORTEST_TTFT_FALLBACK;
        }
        reportDecision(roleType, selectedWorker, decision, cacheLeader, shortestTtftWorker, config);
        return selectedWorker;
    }

    private ScoredWorker selectBaseline(List<ScoredWorker> scoredWorkers,
                                        BalanceContext balanceContext,
                                        RoleType roleType,
                                        String group,
                                        long seqLen,
                                        FlexlbConfig config,
                                        StrategyConfigs.CandidatePoolConfig candidatePoolConfig,
                                        String decision,
                                        ScoredWorker cacheLeader,
                                        ScoredWorker shortestTtftWorker) {
        ScoredWorker selectedWorker = super.selectBestWorker(
                scoredWorkers, balanceContext, roleType, group, seqLen, config, candidatePoolConfig);
        reportDecision(roleType, selectedWorker, decision, cacheLeader, shortestTtftWorker, config);
        return selectedWorker;
    }

    private ScoredWorker findCacheLeader(List<ScoredWorker> workers) {
        return workers.stream().min(CACHE_LEADER_ORDER).orElseThrow();
    }

    private boolean hasRealCacheLead(ScoredWorker cacheLeader,
                                     ScoredWorker shortestTtftWorker,
                                     List<ScoredWorker> workers) {
        if (!isSameWorker(cacheLeader, shortestTtftWorker)) {
            return cacheLeader.hitCacheTokens() > shortestTtftWorker.hitCacheTokens();
        }

        // The shortest worker may itself be the cache leader. It is still a real affinity result
        // only when its prefix match is strictly better than at least one other available worker.
        return workers.stream()
                .anyMatch(worker -> worker.hitCacheTokens() < cacheLeader.hitCacheTokens());
    }

    private List<ScoredWorker> buildCacheAffinitySelectionOrder(ScoredWorker cacheLeader,
                                                                 ScoredWorker shortestTtftWorker,
                                                                 List<ScoredWorker> workersByTtft,
                                                                 long seqLen,
                                                                 FlexlbConfig config) {
        List<ScoredWorker> selectionOrder = new ArrayList<>(workersByTtft.size());
        selectionOrder.add(cacheLeader);

        // If the leader was claimed concurrently, retain affinity only for workers that pass the
        // same strict cache-lead, hit-rate, and extra-work gates.
        workersByTtft.stream()
                .filter(worker -> !isSameWorker(worker, cacheLeader))
                .filter(worker -> isCacheAffinityCandidate(
                        worker, cacheLeader, shortestTtftWorker, seqLen, config))
                .sorted(CACHE_LEADER_ORDER)
                .forEach(worker -> addIfAbsent(selectionOrder, worker));

        addIfAbsent(selectionOrder, shortestTtftWorker);
        workersByTtft.forEach(worker -> addIfAbsent(selectionOrder, worker));
        return selectionOrder;
    }

    private boolean isCacheAffinityCandidate(ScoredWorker worker,
                                             ScoredWorker cacheLeader,
                                             ScoredWorker shortestTtftWorker,
                                             long seqLen,
                                             FlexlbConfig config) {
        boolean hasCacheLead = worker.hitCacheTokens() > shortestTtftWorker.hitCacheTokens();
        if (isSameWorker(cacheLeader, shortestTtftWorker)) {
            // When the shortest worker is the global leader, an equal-max-hit peer is still a
            // valid affinity fallback after a concurrent claim of the leader.
            hasCacheLead = worker.hitCacheTokens() == cacheLeader.hitCacheTokens();
        }
        return hasCacheLead
                && meetsMinimumHitRate(worker, seqLen, config)
                && isWithinExtraWorkCap(worker, shortestTtftWorker, config);
    }

    private boolean meetsMinimumHitRate(ScoredWorker worker, long seqLen, FlexlbConfig config) {
        double configuredMinimumHitRate = config.getCacheAffinityFirstMinHitRate();
        if (Double.isNaN(configuredMinimumHitRate) || seqLen <= 0) {
            return false;
        }
        // Clamp infinities and out-of-range values to a finite percentage.
        double minimumHitRate = Math.min(100.0, Math.max(0.0, configuredMinimumHitRate));
        if (minimumHitRate == 0.0) {
            return true;
        }
        return worker.hitCacheTokens() * 100.0 / seqLen >= minimumHitRate;
    }

    private boolean isWithinExtraWorkCap(ScoredWorker worker,
                                         ScoredWorker shortestTtftWorker,
                                         FlexlbConfig config) {
        long maxExtraWork = configuredMaxExtraWork(config);
        long maximumAllowedWork = saturatingAdd(shortestTtftWorker.ttft(), maxExtraWork);
        return worker.ttft() <= maximumAllowedWork;
    }

    private long configuredMaxExtraWork(FlexlbConfig config) {
        return Math.max(0L, config.getCacheAffinityFirstMaxExtraWorkTokens());
    }

    private long saturatingAdd(long value, long nonNegativeIncrement) {
        if (value > Long.MAX_VALUE - nonNegativeIncrement) {
            return Long.MAX_VALUE;
        }
        return value + nonNegativeIncrement;
    }

    private long saturatingSubtract(long left, long right) {
        try {
            return Math.subtractExact(left, right);
        } catch (ArithmeticException ignored) {
            return left >= right ? Long.MAX_VALUE : Long.MIN_VALUE;
        }
    }

    private void addIfAbsent(List<ScoredWorker> workers, ScoredWorker candidate) {
        if (workers.stream().noneMatch(worker -> isSameWorker(worker, candidate))) {
            workers.add(candidate);
        }
    }

    private boolean isSameWorker(ScoredWorker left, ScoredWorker right) {
        return left != null && right != null && left.worker() == right.worker();
    }

    private void reportDecision(RoleType roleType,
                                ScoredWorker selectedWorker,
                                String decision,
                                ScoredWorker cacheLeader,
                                ScoredWorker shortestTtftWorker,
                                FlexlbConfig config) {
        String selectedIp = selectedWorker == null ? "" : selectedWorker.worker().getIp();
        engineHealthReporter.reportCacheAffinityDecision(roleType, selectedIp, decision);
        Logger.debug(
                "Cache affinity decision - role: {}, decision: {}, selected: {}, cacheLeader: {}, shortestTtft: {}, cacheHitTokens: {}, cacheLeadTokens: {}, extraWork: {}, maxExtraWork: {}",
                roleType,
                decision,
                selectedWorker == null ? "" : selectedWorker.worker().getIpPort(),
                cacheLeader.worker().getIpPort(),
                shortestTtftWorker.worker().getIpPort(),
                cacheLeader.hitCacheTokens(),
                Math.max(0L, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens()),
                saturatingSubtract(cacheLeader.ttft(), shortestTtftWorker.ttft()),
                configuredMaxExtraWork(config));
    }
}
