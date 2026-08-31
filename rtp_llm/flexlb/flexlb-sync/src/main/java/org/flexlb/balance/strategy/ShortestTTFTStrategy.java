package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The feat/dsv4_on_dev shortest-TTFT candidate-pool policy: sort by score,
 * keep the configured pool, then use endpoint LRU with a single CAS claim.
 */
@Component
public class ShortestTTFTStrategy
        extends CostBasedPrefillStrategy {

    public ShortestTTFTStrategy(
            WorkerDirectory workerDirectory,
            CacheAwareService cacheAwareService,
            ResourceMeasureFactory resourceMeasureFactory,
            EngineHealthReporter engineHealthReporter) {
        super(
                workerDirectory,
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
    protected int selectBestCandidate(
            CandidateSet candidates,
            long minimumScore,
            BalanceContext context,
            RoleType role,
            String group,
            long sequenceLength,
            FlexlbConfig config) {
        if (candidates.size() == 0) {
            return -1;
        }

        List<Integer> baseline = sortedByScore(candidates);
        int poolSize = Math.min(
                config.shortestTtftCandidateCount(
                        candidates.size()),
                candidates.size());
        baseline = new ArrayList<>(
                baseline.subList(0, Math.max(1, poolSize)));

        RoutingConfig.CacheAffinityConfig affinityConfig =
                config.getRouter().getRoles()
                        .getPrefill().getCacheAffinity();
        CacheAffinityPolicy.Decision affinity = null;
        if (affinityConfig != null) {
            long referenceHit = 0L;
            for (int index = 0;
                    index < candidates.size();
                    index++) {
                if (candidates.scoreMs(index) == minimumScore) {
                    referenceHit = Math.max(
                            referenceHit,
                            candidates.cacheHit(index));
                }
            }
            affinity = CacheAffinityPolicy.evaluate(
                    candidates.size(),
                    candidates::scoreMs,
                    candidates::cacheHit,
                    minimumScore,
                    referenceHit,
                    sequenceLength,
                    affinityConfig.getMaxExtraTtftMs(),
                    affinityConfig.getMinPrefixHitPercent());
        }

        int selected;
        String reason = null;
        if (affinity != null && affinity.hasPreference()) {
            List<Integer> preferred =
                    new ArrayList<>(affinity.preferredCount());
            for (int index = 0;
                    index < affinity.preferredCount();
                    index++) {
                preferred.add(affinity.preferredIndex(index));
            }
            selected = claimInOrder(candidates, preferred);
            if (selected < 0) {
                selected = selectFromBaseline(
                        candidates, baseline);
                reason = "CACHE_AFFINITY_FALLBACK";
            } else {
                reason = selected == preferred.getFirst()
                        ? CacheAffinityPolicy.Reason
                                .CACHE_LEADER.name()
                        : "CACHE_AFFINITY_FALLBACK";
            }
        } else {
            selected = selectFromBaseline(candidates, baseline);
            if (affinity != null) {
                reason = affinity.reason().name();
            }
        }

        if (selected >= 0 && reason != null) {
            reportCacheAffinityDecision(
                    role,
                    candidates.endpoint(selected).getIp(),
                    reason);
        }
        return selected;
    }

    private static List<Integer> sortedByScore(
            CandidateSet candidates) {
        List<Integer> indexes =
                new ArrayList<>(candidates.size());
        for (int index = 0;
                index < candidates.size();
                index++) {
            indexes.add(index);
        }
        indexes.sort(
                Comparator.comparingLong(
                        (Integer index) ->
                                candidates.scoreMs(index))
                        .thenComparingLong(index ->
                                candidates.lastSelectedTime(index)));
        return indexes;
    }

    private static int selectFromBaseline(
            CandidateSet candidates,
            List<Integer> baseline) {
        List<Integer> fairnessOrder =
                new ArrayList<>(baseline);
        fairnessOrder.sort(
                Comparator.comparingLong(index ->
                        candidates.lastSelectedTime(index)));
        int selected = claimInOrder(
                candidates, fairnessOrder);
        return selected >= 0
                ? selected
                : baseline.getFirst();
    }

    private static int claimInOrder(
            CandidateSet candidates,
            List<Integer> order) {
        long nowMicros = System.nanoTime() / 1_000L;
        for (int index : order) {
            AtomicLong clock = candidates.endpoint(index)
                    .getLastSelectedTime();
            long expected = candidates.lastSelectedTime(index);
            if (expected == Long.MAX_VALUE) {
                continue;
            }
            long selectedAt = Math.max(
                    nowMicros, expected + 1L);
            if (clock.compareAndSet(
                    expected, selectedAt)) {
                return index;
            }
        }
        return -1;
    }
}
