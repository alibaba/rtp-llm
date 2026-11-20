package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Keeps hot requests on cache leaders and balances cold requests by outstanding token work.
 * Each selection call uses one cache view; a later scheduling retry queries cache matches again.
 */
public final class TokenBalancedPrefillStrategy extends PrefillSelectionStrategy {
    private final double minimumHitPercent;

    private final AtomicLong nextWorker = new AtomicLong();

    public TokenBalancedPrefillStrategy(
            ConfigService config,
            EndpointRegistry endpoints,
            CacheAwareService cache,
            CacheMetricsReporter metrics) {
        super(config, endpoints, cache, metrics);
        minimumHitPercent =
                config.loadBalanceConfig()
                        .getRouter()
                        .getRoles()
                        .getPrefill()
                        .getSelection()
                        .getMinPrefixHitPercent();
    }

    @Override
    public Optional<Selection> chooseBest(
            Request request, String group, List<PrefillEndpoint> inputs, CacheMatchResult matches) {
        return chooseBest(
                request, group, inputs, matches, cacheCandidates(request, group, inputs, matches));
    }

    @Override
    protected CacheCandidates cacheCandidates(
            Request request,
            String group,
            Iterable<PrefillEndpoint> inputs,
            CacheMatchResult matches) {
        long maximum = 0L;
        var generations = new java.util.HashSet<Long>();
        for (var candidate : inputs) {
            if (!candidate.acceptsNewRequests()
                    || group != null && !group.equals(candidate.getStatus().getGroup())) continue;
            long hit = estimate(request, candidate, matches).hitTokens();
            if (hit > maximum) {
                maximum = hit;
                generations.clear();
            }
            if (hit == maximum && hit > 0) generations.add(candidate.getStatus().getGenerationId());
        }
        return new CacheCandidates(maximum, generations);
    }

    @Override
    protected CacheEstimate estimate(
            Request request, PrefillEndpoint endpoint, CacheMatchResult matches) {
        var estimate = super.estimate(request, endpoint, matches);
        if (estimate.inputTokens() == 0
                || 100.0 * estimate.hitTokens() / estimate.inputTokens() <= minimumHitPercent) {
            return new CacheEstimate(
                    estimate.inputTokens(), estimate.inputTokens(), 0, estimate.routingHitTokens());
        }
        return estimate;
    }

    @Override
    protected Optional<Selection> chooseBest(
            Request request,
            String group,
            List<PrefillEndpoint> inputs,
            CacheMatchResult matches,
            CacheCandidates candidates) {
        if (inputs.isEmpty()) return Optional.empty();
        long inputTokens = Math.max(0L, request.getSeqLen());
        boolean hot = !candidates.generations().isEmpty();
        int offset = Math.floorMod(nextWorker.getAndIncrement(), inputs.size());
        Selection best = null;
        long bestWork = Long.MAX_VALUE;
        for (int i = 0; i < inputs.size(); i++) {
            var endpoint = inputs.get((offset + i) % inputs.size());
            if (group != null && !group.equals(endpoint.getStatus().getGroup())) continue;
            // Hot candidates passed availability when ranked; registration checks it again.
            if (hot
                    ? !candidates.generations().contains(endpoint.getStatus().getGenerationId())
                    : !endpoint.acceptsNewRequests()) continue;
            var snapshot = endpoint.selectionSnapshot();
            if (snapshot.admissionBlocked()) continue;
            var estimate = estimate(request, endpoint, matches);
            long work = hot ? 0 : saturatedAdd(snapshot.computeTokens(), inputTokens);
            if (best != null && work >= bestWork) continue;
            bestWork = work;
            best =
                    new Selection(
                            endpoint.ipPort(),
                            snapshot.generationId(),
                            estimate.computeTokens(),
                            estimate.hitTokens(),
                            estimate.routingHitTokens(),
                            candidates.maximumHit(),
                            hot ? "TOKEN_BALANCE_CACHE_LEADER" : "TOKEN_BALANCE_COLD",
                            endpoint,
                            matches,
                            snapshot);
        }
        return Optional.ofNullable(best);
    }
}
