package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.formula.PrefillTimePredictor;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Shares Prefill cache estimates, exact registration and rollback across node-ranking policies.
 * Candidate ranking is supplied by each policy; a successful result owns one registration.
 */
public abstract class PrefillSelectionStrategy
        extends AbstractWorkerSelectionStrategy<
                PrefillEndpoint, PrefillSelectionStrategy.Selection> {
    private static final CacheCandidates UNRESTRICTED = new CacheCandidates(0L, java.util.Set.of());

    private final EndpointRegistry endpoints;
    private final CacheAwareService cacheService;
    private final CacheMetricsReporter cacheReporter;

    private final double remoteDiscount;

    protected PrefillSelectionStrategy(
            ConfigService configService,
            EndpointRegistry endpoints,
            CacheAwareService cacheService,
            CacheMetricsReporter cacheReporter) {
        this.endpoints = Objects.requireNonNull(endpoints);
        this.cacheService = Objects.requireNonNull(cacheService);
        this.cacheReporter = Objects.requireNonNull(cacheReporter);
        var prefill = configService.loadBalanceConfig().getRouter().getRoles().getPrefill();
        remoteDiscount = prefill.getRemoteDiscount();
    }

    @Override
    public SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
            RequestContext context, String group) {
        return selectPrefill(context, RoleType.PREFILL, group);
    }

    public abstract java.util.Optional<Selection> chooseBest(
            Request request, String group, List<PrefillEndpoint> inputs, CacheMatchResult matches);

    public final OptionalLong estimateQueuedTokens(Request request, RoleType role, String group) {
        var candidates = endpoints.<PrefillEndpoint>endpoints(role, null);
        var matches =
                cacheService.findMatchingEngines(
                        new org.flexlb.cache.CacheMatchQuery(
                                request.getRequestId(),
                                request.getBlockCacheKeys(),
                                request.getBlockSize(),
                                role,
                                group));
        long best = Long.MAX_VALUE;
        boolean found = false;
        for (var candidate : candidates) {
            if (!candidate.acceptsNewRequests()
                    || group != null && !group.equals(candidate.getStatus().getGroup())) continue;
            best = Math.min(best, estimate(request, candidate, matches).computeTokens());
            found = true;
        }
        return found ? OptionalLong.of(best) : OptionalLong.empty();
    }

    protected java.util.Optional<Selection> chooseBest(
            Request request,
            String group,
            List<PrefillEndpoint> inputs,
            CacheMatchResult matches,
            CacheCandidates candidates) {
        return chooseBest(request, group, inputs, matches);
    }

    protected CacheEstimate estimate(
            Request request, PrefillEndpoint endpoint, CacheMatchResult matches) {
        var match =
                matches == null
                        ? null
                        : matches.hostMatch(
                                endpoint.ipPort(),
                                endpoint.getStatus().getPhysicalIpPort(),
                                endpoint.getStatus().getMultiEngineNum());
        double blocks =
                match == null
                        ? 0.0
                        : Math.max(0, match.localMatchBlocks())
                                + Math.max(0, match.globalMatchBlocks() - match.localMatchBlocks())
                                        * remoteDiscount;
        long input = Math.max(0L, request.getSeqLen());
        long blockSize = matchingBlockSize(request, endpoint, matches);
        long routingHit = CacheMatchResult.matchedTokens(blocks, blockSize, input);
        long hit = routingHit >= input ? Math.max(0L, input - blockSize) : routingHit;
        return new CacheEstimate(input, input - hit, hit, routingHit);
    }

    protected CacheCandidates cacheCandidates(
            Request request,
            String group,
            Iterable<PrefillEndpoint> inputs,
            CacheMatchResult matches) {
        return UNRESTRICTED;
    }

    @Override
    protected WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context,
            RoleType role,
            String group,
            PrefillSelectionStrategy.Selection selected) {
        var request = context.getRequest();
        var candidate = endpoints.getEndpoint(role, selected.address());
        if (!(candidate instanceof PrefillEndpoint endpoint)
                || !endpoint.acceptsNewRequests()
                || endpoint.getStatus().getGenerationId() != selected.generationId())
            return WorkerEndpoint.Registration.rejected(
                    WorkerEndpoint.RegistrationStatus.UNAVAILABLE);
        var accepted =
                endpoint.tryRegisterSelection(
                        context.getRequestId(),
                        selected.selectionSnapshot(),
                        selected.computeTokens(),
                        PrefillTimePredictor.estimateDurationMs(
                                request.getSeqLen(), selected.cacheHitTokens()));
        if (accepted.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
            return WorkerEndpoint.Registration.rejected(accepted.status());
        try {
            var status = endpoint.getStatus();
            var result =
                    RoleSelection.prefill(
                            endpoint,
                            status.topologySnapshot(),
                            selected.cacheHitTokens(),
                            selected.computeTokens(),
                            accepted.value());
            reportCacheAffinity(role, endpoint.getIp(), selected.affinityReason());
            context.recordCacheSelection(role, endpoint.getIp(), selected.cacheHitTokens());
            reportSelection(role, request.getSeqLen(), selected);
            if (selected.cacheMatches() != null) {
                try {
                    cacheService.trackRoutingPrediction(
                            request.getRequestId(),
                            role,
                            group,
                            status,
                            request.getSeqLen(),
                            selected.cacheHitTokens(),
                            selected.cacheMatches());
                } catch (RuntimeException failure) {
                    Logger.warn("Cache prediction observation failed", failure);
                }
            }
            return new WorkerEndpoint.Registration<>(
                    WorkerEndpoint.RegistrationStatus.REGISTERED, result);
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(null, () -> endpoint.rollbackSelection(accepted.value())));
            throw failure;
        }
    }

    @Override
    protected PrefillEndpoint candidate(Selection selected) {
        return selected.input();
    }

    SelectionResult<RoleSelection, CapacityKey> selectAndRegisterFusion(
            RequestContext context, String group) {
        return selectPrefill(context, RoleType.PDFUSION, group);
    }

    private SelectionResult<RoleSelection, CapacityKey> selectPrefill(
            RequestContext context, RoleType role, String group) {
        var candidates = endpoints.<PrefillEndpoint>endpoints(role, null);
        var request = context.getRequest();
        var matches =
                candidates.isEmpty()
                        ? null
                        : cacheService.findMatchingEngines(
                                new org.flexlb.cache.CacheMatchQuery(
                                        request.getRequestId(),
                                        request.getBlockCacheKeys(),
                                        request.getBlockSize(),
                                        role,
                                        group));
        var scope = new CapacityKey(role, group, null);
        var registration = new RegistrationCandidates<>(candidates);
        var result =
                selectCandidates(
                        context,
                        scope,
                        registration,
                        current -> {
                            var leaders =
                                    cacheCandidates(
                                            request, group, registration.includingFull(), matches);
                            var selected = chooseBest(request, group, current, matches, leaders);
                            return selected.isPresent()
                                    ? SelectionResult.success(selected.get())
                                    : SelectionResult.<Selection, CapacityKey>blocked(
                                                    scope,
                                                    Response.error(StrategyErrorType.forRole(role)),
                                                    null)
                                            .withCandidates(leaders.generations());
                        });
        if (!registration.onlyFullWorkersRemain()) return result;
        var leaders = cacheCandidates(request, group, registration.includingFull(), matches);
        return leaders.generations().isEmpty()
                ? result
                : result.withCandidates(leaders.generations());
    }

    private static long matchingBlockSize(
            Request request, PrefillEndpoint input, CacheMatchResult matches) {
        if (matches != null && matches.blockSize() > 0) return matches.blockSize();
        if (request.getBlockSize() > 0) return request.getBlockSize();
        if (request.getCacheKeyBlockSize() > 0) return request.getCacheKeyBlockSize();
        var cache = input.getStatus().getCacheStatus();
        return cache == null ? 0L : Math.max(0L, cache.getBlockSize());
    }

    private void reportCacheAffinity(RoleType role, String ip, String reason) {
        if (reason == null) return;
        try {
            cacheReporter.reportCacheAffinityDecision(role, ip, reason);
        } catch (Throwable telemetryFailure) {
            Logger.warn("Cache affinity telemetry failed", telemetryFailure);
        }
    }

    private void reportSelection(
            RoleType role, long seqLen, PrefillSelectionStrategy.Selection selected) {
        try {
            cacheReporter.reportSelectionCache(
                    role,
                    seqLen,
                    selected.cacheHitTokens(),
                    selected.routingCacheMatchTokens(),
                    selected.maximumRoutingCacheMatchTokens());
        } catch (Throwable telemetryFailure) {
            Logger.warn("Prefill cache telemetry failed", telemetryFailure);
        }
    }

    /**
     * Captures cache candidates before admission filtering for one selection call.
     * An empty generation set leaves node ranking unrestricted by cache affinity.
     */
    protected record CacheCandidates(long maximumHit, java.util.Set<Long> generations) {
        protected CacheCandidates {
            generations = java.util.Set.copyOf(generations);
        }
    }

    protected record CacheEstimate(
            long inputTokens, long computeTokens, long hitTokens, long routingHitTokens) {}

    /**
     * Frozen scores; input must be the original candidate object, never a reconstructed identity.
     * The selection snapshot supplies the generation and capacity version checked during registration.
     */
    public record Selection(
            String address,
            long generationId,
            long computeTokens,
            long cacheHitTokens,
            long routingCacheMatchTokens,
            long maximumRoutingCacheMatchTokens,
            String affinityReason,
            PrefillEndpoint input,
            org.flexlb.cache.CacheMatchResult cacheMatches,
            PrefillEndpoint.SelectionSnapshot selectionSnapshot) {}
}
