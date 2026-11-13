package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.formula.PrefillTimePredictor;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.DebugInfo;
import org.flexlb.worker.status.ServerStatus;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Selects and registers Prefill workers using cache affinity and the configured work formula.
 * Cache-first selection prefers hits within its absolute compute-token cap; full workers are skipped.
 */
public class CostBasedPrefillStrategy
        extends AbstractWorkerSelectionStrategy<
                PrefillEndpoint, CostBasedPrefillStrategy.Selection> {
    private final EndpointRegistry endpoints;
    private final CacheAwareService cacheService;
    private final CacheMetricsReporter cacheReporter;
    private final PrefillTimePredictor predictor;
    private final AtomicLong nextWorker = new AtomicLong();

    private final boolean cacheFirst;
    private final long cacheMaxComputeTokens;
    private final double minimumHitPercent;
    private final double remoteDiscount;

    public CostBasedPrefillStrategy(
            ConfigService configService,
            EndpointRegistry endpoints,
            CacheAwareService cacheService,
            CacheMetricsReporter cacheReporter) {
        this.endpoints = Objects.requireNonNull(endpoints);
        this.cacheService = Objects.requireNonNull(cacheService);
        this.cacheReporter = Objects.requireNonNull(cacheReporter);
        var prefill = configService.loadBalanceConfig().getRouter().getRoles().getPrefill();
        predictor = new PrefillTimePredictor(prefill.getCostExpression());
        var affinity = prefill.getCacheAffinity();
        cacheFirst = affinity != null;
        cacheMaxComputeTokens =
                affinity == null || affinity.getMaxComputeTokens() == null
                        ? prefill.getMaxComputeTokens()
                        : affinity.getMaxComputeTokens();
        minimumHitPercent = affinity == null ? 0.0 : affinity.getMinPrefixHitPercent();
        remoteDiscount = affinity == null ? 0.2 : affinity.getRemoteDiscount();
    }

    public RoutingResult<RoleSelection, CapacityKey> select(
            RequestContext context, RoleType role, String group) {
        var candidates = endpoints.<PrefillEndpoint>endpoints(role, group);
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
        return selectCandidates(
                context,
                scope,
                candidates,
                current ->
                        this.chooseBest(request, current, matches)
                                .map(
                                        selected ->
                                                RoutingResult
                                                        .<CostBasedPrefillStrategy.Selection,
                                                                CapacityKey>
                                                                success(selected))
                                .orElseGet(
                                        () ->
                                                RoutingResult.blocked(
                                                        scope,
                                                        Response.error(
                                                                org.flexlb.worker.endpoint
                                                                        .StrategyErrorType.forRole(
                                                                        role)),
                                                        null)));
    }

    public java.util.Optional<Selection> chooseBest(
            Request request, List<PrefillEndpoint> inputs, CacheMatchResult matches) {
        int offset =
                inputs.isEmpty() ? 0 : Math.floorMod(nextWorker.getAndIncrement(), inputs.size());
        Candidate best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        long bestCacheRank = -1L;
        long maximumRoutingHit = 0L;
        for (int index = 0; index < inputs.size(); index++) {
            var input = inputs.get((offset + index) % inputs.size());
            var snapshot = input.selectionSnapshot();
            var match =
                    matches == null
                            ? null
                            : matches.hostMatch(
                                    input.ipPort(),
                                    input.getStatus().getPhysicalIpPort(),
                                    input.getStatus().getMultiEngineNum());
            double blocks =
                    match == null
                            ? 0.0
                            : Math.max(0, match.localMatchBlocks())
                                    + Math.max(
                                                    0,
                                                    match.globalMatchBlocks()
                                                            - match.localMatchBlocks())
                                            * remoteDiscount;
            long blockSize = matchingBlockSize(request, input, matches);
            long inputTokens = Math.max(0L, request.getSeqLen());
            long routingHit = CacheMatchResult.matchedTokens(blocks, blockSize, inputTokens);
            maximumRoutingHit = Math.max(maximumRoutingHit, routingHit);
            long cacheHit =
                    routingHit >= inputTokens ? Math.max(0L, inputTokens - blockSize) : routingHit;
            long computeTokens = predictor.computeTokens(inputTokens, cacheHit);
            long work = saturatedAdd(snapshot.computeTokens(), computeTokens);
            if (work > snapshot.maxComputeTokens() || cacheFirst && work > cacheMaxComputeTokens)
                continue;
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

    @Override
    protected WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context,
            RoleType role,
            String group,
            CostBasedPrefillStrategy.Selection selected) {
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
                        selected.computeTokens());
        if (accepted.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
            return WorkerEndpoint.Registration.rejected(accepted.status());
        try {
            var status = endpoint.getStatus();
            var assignment =
                    ServerStatus.forWorker(
                            role,
                            request.getRequestId(),
                            status.topologySnapshot(),
                            status.committedEngineObservation());
            var debug = new DebugInfo();
            debug.setHitCacheLen(selected.cacheHitTokens());
            assignment.setDebugInfo(debug);
            var result =
                    RoleSelection.prefill(
                            endpoint, assignment, selected.computeTokens(), accepted.value());
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

    @Override
    protected PrefillEndpoint refresh(PrefillEndpoint candidate) {
        return candidate;
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
            RoleType role, long seqLen, CostBasedPrefillStrategy.Selection selected) {
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

    private record Candidate(
            PrefillEndpoint input,
            PrefillEndpoint.SelectionSnapshot snapshot,
            long computeTokens,
            long cacheHit,
            long routingHit) {}

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
