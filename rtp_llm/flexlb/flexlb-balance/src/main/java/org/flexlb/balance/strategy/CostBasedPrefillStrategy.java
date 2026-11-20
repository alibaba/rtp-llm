package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.util.formula.PrefillTimePredictor;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Prefers cache hits among admissible workers, then compares the configured compute cost.
 */
public class CostBasedPrefillStrategy extends PrefillSelectionStrategy {
    private final PrefillTimePredictor predictor;
    private final AtomicLong nextWorker = new AtomicLong();
    private final boolean cacheFirst;
    private final double minimumHitPercent;

    public CostBasedPrefillStrategy(
            ConfigService config,
            EndpointRegistry endpoints,
            CacheAwareService cache,
            CacheMetricsReporter metrics) {
        super(config, endpoints, cache, metrics);
        var selection =
                config.loadBalanceConfig().getRouter().getRoles().getPrefill().getSelection();
        predictor = new PrefillTimePredictor(selection.getCostExpression());
        cacheFirst = selection.getCacheAffinity() != null;
        minimumHitPercent = cacheFirst ? selection.getCacheAffinity().getMinPrefixHitPercent() : 0;
    }

    @Override
    public java.util.Optional<Selection> chooseBest(
            Request request, String group, List<PrefillEndpoint> inputs, CacheMatchResult matches) {
        int groupSize = inputs.size();
        if (group != null) {
            groupSize = 0;
            for (var input : inputs) if (group.equals(input.getStatus().getGroup())) groupSize++;
        }
        if (groupSize == 0) return java.util.Optional.empty();
        int offset = Math.floorMod(nextWorker.getAndIncrement(), groupSize);
        if (group != null) {
            int remaining = offset;
            for (int index = 0; index < inputs.size(); index++) {
                if (group.equals(inputs.get(index).getStatus().getGroup()) && remaining-- == 0) {
                    offset = index;
                    break;
                }
            }
        }
        Candidate best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        long bestCacheRank = -1L;
        long maximumRoutingHit = 0L;
        boolean noCacheMatches = matches == null || matches.hostMatches().isEmpty();
        for (int index = 0; index < inputs.size(); index++) {
            var input = inputs.get((offset + index) % inputs.size());
            if (group != null && !group.equals(input.getStatus().getGroup())) continue;
            var snapshot = input.selectionSnapshot();
            if (snapshot.admissionBlocked()) continue;
            var estimate = estimate(request, input, matches);
            long inputTokens = estimate.inputTokens();
            long routingHit = estimate.routingHitTokens();
            long cacheHit = estimate.hitTokens();
            long computeTokens = estimate.computeTokens();
            maximumRoutingHit = Math.max(maximumRoutingHit, routingHit);
            long work = saturatedAdd(snapshot.computeTokens(), computeTokens);
            double cost = predictor.score(work);
            long cacheRank =
                    cacheFirst
                                    && inputTokens > 0
                                    && cacheHit * RoutingConfig.PERCENTAGE_SCALE / inputTokens
                                            >= minimumHitPercent
                            ? cacheHit
                            : 0L;
            if (best == null
                    || cacheRank > bestCacheRank
                    || cacheRank == bestCacheRank && cost < bestCost) {
                best = new Candidate(input, snapshot, computeTokens, cacheHit, routingHit);
                bestCacheRank = cacheRank;
                bestCost = cost;
            }
            // With no cache hits, idle work reaches the lowest possible default score.
            if (noCacheMatches && predictor.usesDefaultFormula() && snapshot.computeTokens() == 0L)
                break;
        }
        if (best == null) return java.util.Optional.empty();
        var snapshot = best.snapshot();
        var selected =
                new Selection(
                        best.input().ipPort(),
                        snapshot.generationId(),
                        best.computeTokens(),
                        best.cacheHit(),
                        best.routingHit(),
                        maximumRoutingHit,
                        cacheFirst ? "CACHE_FIRST" : "MIN_PREFILL_COST",
                        best.input(),
                        matches,
                        snapshot);
        return java.util.Optional.of(selected);
    }

    private record Candidate(
            PrefillEndpoint input,
            PrefillEndpoint.SelectionSnapshot snapshot,
            long computeTokens,
            long cacheHit,
            long routingHit) {}
}
