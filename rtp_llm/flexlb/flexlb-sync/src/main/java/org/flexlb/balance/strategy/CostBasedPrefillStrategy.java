package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.CapacityKey;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;

import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Ranks Prefill candidates using predicted cost and cache affinity, then registers the selected candidate.
 */
public class CostBasedPrefillStrategy implements PrefillSelectionStrategy {
    private static final int MAX_CACHED_WORKERS = 4096;

    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final CacheAwareService cacheAwareService;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();

    private final boolean priorityOrdering;
    private final boolean queueScheduling;

    private final Long cacheAffinityExtraTtftMs;

    private final double cacheAffinityMinHitRate;
    private final double remoteDiscount;

    public CostBasedPrefillStrategy(
            CacheAwareService cacheAwareService, ConfigService configService) {
        this.cacheAwareService = cacheAwareService;
        var config = configService.loadBalanceConfig();
        priorityOrdering = config.isPriorityOrdering();
        queueScheduling = config.isQueue();
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        remoteDiscount = affinity == null ? 0.2 : affinity.getRemoteDiscount();
        cacheAffinityExtraTtftMs =
                affinity == null ? null : Math.max(0L, affinity.getMaxExtraTtftMs());
        cacheAffinityMinHitRate =
                affinity == null ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent());
    }

    @Override
    public <I extends CandidateInput> RoutingResult<Selection<I>, CapacityKey> selectAndRegister(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group,
            List<I> inputs,
            LongSupplier clock,
            java.util.function.Function<Selection<I>, RoutingResult.Registration> register) {
        PrefillCandidateSet cached = CANDIDATES.get();
        PrefillCandidateSet candidates = cached.inUse ? new PrefillCandidateSet() : cached;
        candidates.inUse = true;
        try {
            candidates.capture(inputs, group);
            Map<String, Integer> rejections =
                    candidates.registeredCount == 0 ? Map.of() : new java.util.HashMap<>();
            org.flexlb.cache.domain.CacheMatchResult cacheMatches = null;
            if (candidates.registeredCount != 0) {
                cacheMatches =
                        cacheAwareService.findMatchingEngines(
                                new org.flexlb.cache.domain.CacheMatchQuery(
                                        request.getRequestId(),
                                        request.getBlockCacheKeys(),
                                        request.getBlockSize(),
                                        roleType,
                                        group));
                evaluateCandidates(
                        request, metadata, cacheMatches, rejections, candidates, clock.getAsLong());
            }
            if (candidates.size == 0) {
                return observe(
                        blockedWithDiagnostics(
                                Response.buildErrorResponse(
                                        roleType.getErrorType(),
                                        candidates.registeredCount == 0
                                                ? "No Prefill candidate in the requested routing group"
                                                : "No Prefill candidate could be scored: "
                                                        + rejections),
                                new CapacityKey(roleType, group, null),
                                candidates.registeredCount,
                                rejections),
                        candidates,
                        -1,
                        roleType,
                        group,
                        rejections,
                        cacheMatches);
            }
            for (int attempt = 0; attempt < 8; attempt++) {
                int selected =
                        selectBestCandidate(
                                candidates,
                                candidates.minimumTtftRank,
                                roleType,
                                group,
                                request.getSeqLen());
                RoutingResult<Selection<I>, CapacityKey> result =
                        selected < 0
                                ? blockedWithDiagnostics(
                                        Response.buildErrorResponse(
                                                roleType.getErrorType(),
                                                candidates.size == 0
                                                        ? "All captured Prefill candidates became unavailable before registration"
                                                        : "No Prefill candidate satisfies the routing policy"),
                                        new CapacityKey(roleType, group, null),
                                        candidates.registeredCount,
                                        Map.of(
                                                "reason",
                                                candidates.size == 0
                                                        ? "CANDIDATES_UNAVAILABLE"
                                                        : "NO_POLICY_MATCH"))
                                : selection(candidates, selected, roleType, metadata, cacheMatches);
                if (result.status() != RoutingResult.Status.SUCCESS)
                    return observe(
                            result,
                            candidates,
                            selected,
                            roleType,
                            group,
                            rejections,
                            cacheMatches);
                var registered = register.apply(result.value());
                if (registered == RoutingResult.Registration.REGISTERED)
                    return observe(
                            result,
                            candidates,
                            selected,
                            roleType,
                            group,
                            rejections,
                            cacheMatches);
                if (registered == RoutingResult.Registration.BLOCKED)
                    return observe(
                            RoutingResult.blocked(
                                    CapacityKey.exact(
                                            roleType,
                                            result.value().input().group(),
                                            result.value().address()),
                                    Response.buildErrorResponse(
                                            StrategyErrorType.RESOURCE_EXHAUSTED,
                                            "Prefill request capacity rejected registration at "
                                                    + result.value().address()),
                                    Map.of(
                                            "cause",
                                            "Prefill request capacity rejected registration",
                                            "endpoint",
                                            result.value().address())),
                            candidates,
                            selected,
                            roleType,
                            group,
                            rejections,
                            cacheMatches);
                var input = candidates.capturedSources[selected];
                long hit = candidates.cacheHitTokens[selected];
                long routingHit = candidates.routingCacheMatchTokens[selected];
                candidates.remove(selected);
                if (registered == RoutingResult.Registration.CONFLICT)
                    evaluateCandidate(
                            request,
                            metadata,
                            rejections,
                            candidates,
                            clock.getAsLong(),
                            input,
                            hit,
                            routingHit);
            }
            return RoutingResult.retry(
                    Response.buildErrorResponse(
                            StrategyErrorType.SCHEDULER_PLAN_CONFLICT,
                            "Prefill selection could not register a current candidate after 8 attempts"));
        } finally {
            candidates.clear();
            candidates.inUse = false;
            if (candidates == cached && candidates.capturedSources.length > MAX_CACHED_WORKERS) {
                CANDIDATES.remove();
            }
        }
    }

    /** Select from candidates that already passed the common hard filters. */
    int selectBestCandidate(
            PrefillCandidateSet survivors,
            long minProjectedTtftMs,
            RoleType roleType,
            String group,
            long seqLen) {
        if (survivors.size == 0) {
            return -1;
        }

        String previousReason = survivors.affinityReason;
        boolean affinity = cacheAffinityExtraTtftMs != null && survivors.hasKnownTtft;
        int leader = affinity ? findCacheLeader(survivors, minProjectedTtftMs, seqLen) : -1;
        int selected =
                selectTiedCandidate(
                        survivors,
                        leader < 0 ? minProjectedTtftMs : survivors.ttftRank(leader),
                        leader < 0 ? Long.MIN_VALUE : survivors.cacheHitTokens[leader],
                        roleType,
                        group);
        if (selected < 0) survivors.affinityReason = previousReason;
        else if (affinity) logAffinity(survivors, selected, minProjectedTtftMs, roleType, group);
        return selected;
    }

    private <I extends CandidateInput> RoutingResult<Selection<I>, CapacityKey> selection(
            PrefillCandidateSet candidates,
            int selected,
            RoleType roleType,
            SchedulingMetadata metadata,
            org.flexlb.cache.domain.CacheMatchResult cacheMatches) {
        @SuppressWarnings("unchecked")
        I winner = (I) candidates.capturedSources[selected];
        if (!queueScheduling && !winner.requestCapacityAvailable()) {
            var reason = classifyWorkerCapacity(winner.admissionSummary(metadata.priority()));
            var error =
                    switch (reason) {
                        case HIGHER_PRIORITY_AHEAD, SAME_PRIORITY_AHEAD ->
                                StrategyErrorType.PRIORITY_ADMISSION_REJECTED;
                        case UNSPECIFIED -> StrategyErrorType.ADMISSION_UNAVAILABLE;
                        default -> StrategyErrorType.RESOURCE_EXHAUSTED;
                    };
            return blockedWithDiagnostics(
                    Response.error(
                            error,
                            reason,
                            "Prefill request capacity unavailable at " + winner.address()),
                    CapacityKey.exact(roleType, winner.group(), winner.address()),
                    candidates.registeredCount,
                    Map.of(
                            "reason",
                            "PREFILL_REQUEST_CAPACITY",
                            "selectedEndpoint",
                            winner.address(),
                            "cacheHitTokens",
                            candidates.cacheHitTokens[selected],
                            "affinityReason",
                            candidates.affinityReason == null ? "" : candidates.affinityReason));
        }
        return RoutingResult.success(
                new Selection<>(
                        candidates.endpointAddresses[selected],
                        winner.generationId(),
                        candidates.ownershipVersions[selected],
                        candidates.projectedTtftMs[selected],
                        candidates.incomingPrefillMs[selected],
                        candidates.cacheHitTokens[selected],
                        candidates.routingCacheMatchTokens[selected],
                        candidates.maximumRoutingCacheMatchTokens,
                        candidates.affinityReason,
                        candidates.registeredCount,
                        winner,
                        cacheMatches,
                        candidates.inputs[selected]));
    }

    private static AdmissionRejectReason classifyWorkerCapacity(AdmissionSummary summary) {
        long residual = Math.max(0L, summary.requestsToRelease() - summary.lowerPriorityRequests());
        long protectedRequests = summary.higherPriorityRequests() + summary.samePriorityRequests();
        if (residual > protectedRequests && summary.unknownPriorityRequests() > 0L) {
            return AdmissionRejectReason.UNSPECIFIED;
        }
        if (residual > 0L && protectedRequests >= residual) {
            return summary.higherPriorityRequests() > 0L
                    ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    : AdmissionRejectReason.SAME_PRIORITY_AHEAD;
        }
        return AdmissionRejectReason.RESOURCE_EXHAUSTED;
    }

    private static <I extends CandidateInput>
            RoutingResult<Selection<I>, CapacityKey> blockedWithDiagnostics(
                    Response failure, CapacityKey key, int workerCount, Map<String, ?> details) {
        return RoutingResult.blocked(
                key,
                failure,
                Map.of(
                        "cause",
                        failure.getErrorMessage(),
                        "role",
                        key.role().name(),
                        "workers",
                        workerCount,
                        "observedAtMs",
                        System.currentTimeMillis(),
                        "details",
                        Map.copyOf(details)));
    }

    private static <I extends CandidateInput> RoutingResult<Selection<I>, CapacityKey> observe(
            RoutingResult<Selection<I>, CapacityKey> result,
            PrefillCandidateSet candidates,
            int selected,
            RoleType role,
            String group,
            Map<String, Integer> rejections,
            org.flexlb.cache.domain.CacheMatchResult cacheMatches) {
        var snapshot = new java.util.ArrayList<org.flexlb.dao.pv.RoutingDecision.Candidate>(5);
        if (selected >= 0) snapshot.add(snapshotCandidate(candidates, selected, true));
        for (int index = 0; index < candidates.size && snapshot.size() < 5; index++) {
            if (index != selected) snapshot.add(snapshotCandidate(candidates, index, false));
        }
        var details = new java.util.HashMap<String, Object>();
        if (result.diagnostics() != null) details.putAll(result.diagnostics());
        if (result.failure() != null) details.put("cause", result.failure().getErrorMessage());
        details.put(
                "routingDecision",
                new org.flexlb.dao.pv.RoutingDecision(
                        role,
                        group,
                        "CostBasedPrefill",
                        candidates.affinityReason == null
                                ? "COST_BASED"
                                : candidates.affinityReason,
                        System.currentTimeMillis(),
                        0,
                        selected < 0 ? null : candidates.endpointAddresses[selected],
                        candidates.registeredCount,
                        candidates.size,
                        candidates.size > snapshot.size(),
                        rejections,
                        snapshot,
                        null));
        if (cacheMatches != null) details.put("cacheQuery", cacheMatches);
        return new RoutingResult<>(
                result.status(), result.value(), result.failure(), result.blocker(), details);
    }

    private static org.flexlb.dao.pv.RoutingDecision.Candidate snapshotCandidate(
            PrefillCandidateSet candidates, int index, boolean selected) {
        long ttft = candidates.projectedTtftMs[index];
        return new org.flexlb.dao.pv.RoutingDecision.Candidate(
                candidates.endpointAddresses[index],
                selected,
                ttft < 0 ? null : ttft,
                null,
                candidates.incomingPrefillMs[index],
                candidates.cacheHitTokens[index],
                candidates.routingCacheMatchTokens[index],
                0L,
                null,
                null,
                null,
                "MODELED",
                candidates.ownershipVersions[index]);
    }

    /** Cache affinity may spend only the configured time above the fastest eligible candidate. */
    private int findCacheLeader(PrefillCandidateSet candidates, long minimumTtft, long seqLen) {
        long referenceHit = 0L;
        for (int i = 0; i < candidates.size; i++) {
            if (candidates.ttftRank(i) == minimumTtft)
                referenceHit = Math.max(referenceHit, candidates.cacheHitTokens[i]);
        }
        candidates.affinityReason = "NO_CACHE_LEAD";
        if (candidates.maximumCacheHit <= referenceHit) return -1;
        long cutoff = saturatedAdd(minimumTtft, cacheAffinityExtraTtftMs);
        int best = -1;
        boolean minimumHitRateMet = false;
        for (int i = 0; i < candidates.size; i++) {
            long hit = candidates.cacheHitTokens[i];
            if (hit <= referenceHit
                    || cacheAffinityMinHitRate > 0.0
                            && (seqLen <= 0L
                                    || hit * RoutingConfig.PERCENTAGE_SCALE / seqLen
                                            < cacheAffinityMinHitRate)) continue;
            minimumHitRateMet = true;
            long ttft = candidates.ttftRank(i);
            if (candidates.projectedTtftMs[i] < 0L || ttft > cutoff) continue;
            if (best < 0
                    || hit > candidates.cacheHitTokens[best]
                    || hit == candidates.cacheHitTokens[best] && ttft < candidates.ttftRank(best))
                best = i;
        }
        candidates.affinityReason =
                best >= 0 ? "CACHE_LEADER" : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
        return best;
    }

    private void logAffinity(
            PrefillCandidateSet candidates,
            int selected,
            long minimumTtft,
            RoleType role,
            String group) {
        if (!Logger.isDebugEnabled()) return;
        Logger.debug(
                "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                        + "selected: {}, minProjectedTtftMs: {}, "
                        + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                        + "hitTokens: {}, reason: {}",
                role,
                group,
                candidates.endpointAddresses[selected],
                minimumTtft,
                candidates.ttftRank(selected),
                saturatedAdd(minimumTtft, cacheAffinityExtraTtftMs),
                candidates.cacheHitTokens[selected],
                candidates.affinityReason);
    }

    private static double normalizedHitRate(double configuredRate) {
        return Double.isNaN(configuredRate)
                ? RoutingConfig.PERCENTAGE_SCALE
                : Math.clamp(configuredRate, 0.0, RoutingConfig.PERCENTAGE_SCALE);
    }

    private int selectTiedCandidate(
            PrefillCandidateSet candidates, long ttft, long hit, RoleType role, String group) {
        int count = 0;
        for (int i = 0; i < candidates.size; i++) {
            if (candidates.ttftRank(i) == ttft
                    && (!candidates.hasKnownTtft || candidates.projectedTtftMs[i] >= 0L)
                    && (hit == Long.MIN_VALUE || candidates.cacheHitTokens[i] == hit)) {
                candidates.tieIndexes[count++] = i;
            }
        }
        return tieRotation.next(
                role, group, candidates.endpointAddresses, candidates.tieIndexes, count);
    }

    private void evaluateCandidates(
            Request request,
            SchedulingMetadata metadata,
            org.flexlb.cache.domain.CacheMatchResult cacheMatchResults,
            Map<String, Integer> rejections,
            PrefillCandidateSet candidates,
            long planningAtMs) {
        var estimator = RequestTimeEstimation.current();
        long seqLen = request.getSeqLen();
        // Only overwrite visited slots; the unvisited fleet remains intact after rejections.
        for (int i = 0; i < candidates.registeredCount; i++) {
            var input = candidates.capturedSources[i];
            String endpointAddress = candidates.endpointAddresses[i];
            var match =
                    cacheMatchResults == null
                            ? null
                            : cacheMatchResults.hostMatch(
                                    endpointAddress, input.physicalAddress(), input.engineCount());
            double blocks =
                    match == null
                            ? 0
                            : Math.max(0, match.localMatchBlocks())
                                    + Math.max(
                                                    0,
                                                    match.globalMatchBlocks()
                                                            - match.localMatchBlocks())
                                            * remoteDiscount;
            long blockSize = matchingBlockSize(request, input, cacheMatchResults);
            long routingCacheMatchTokens =
                    seqLen <= 0L
                            ? 0L
                            : org.flexlb.cache.domain.CacheMatchResult.matchedTokens(
                                    blocks, blockSize, seqLen);
            long cacheHit =
                    routingCacheMatchTokens >= seqLen
                            ? Math.max(0L, seqLen - blockSize)
                            : routingCacheMatchTokens;
            evaluateCandidate(
                    request,
                    metadata,
                    rejections,
                    candidates,
                    planningAtMs,
                    input,
                    cacheHit,
                    routingCacheMatchTokens);
        }
    }

    private static long matchingBlockSize(
            Request request,
            CandidateInput input,
            org.flexlb.cache.domain.CacheMatchResult matches) {
        if (matches != null && matches.blockSize() > 0) return matches.blockSize();
        if (request.getBlockSize() > 0) return request.getBlockSize();
        if (request.getCacheKeyBlockSize() > 0) return request.getCacheKeyBlockSize();
        return Math.max(0, input.cacheBlockSize());
    }

    private void evaluateCandidate(
            Request request,
            SchedulingMetadata metadata,
            Map<String, Integer> rejections,
            PrefillCandidateSet candidates,
            long planningAtMs,
            CandidateInput input,
            long cacheHit,
            long routingCacheMatchTokens) {
        var estimator = RequestTimeEstimation.current();
        long seqLen = request.getSeqLen();
        String endpointAddress = input.address();
        RequestTimeEstimation.Inputs projectionInputs = input.projectionInputs();
        RequestTimeEstimation.CandidateView projection =
                estimator.projectCandidate(
                        projectionInputs,
                        request.getRequestId(),
                        metadata.priority(),
                        planningAtMs,
                        // ExpirationTimer drives request deadlines through the owning
                        // scheduler.
                        // Selection scores endpoint work only; an expiry race
                        // must not be reported as a capacity blocker.
                        Long.MAX_VALUE,
                        seqLen,
                        cacheHit,
                        routingCacheMatchTokens,
                        planningAtMs);
        if (!projection.selectable()) {
            rejections.merge(
                    "PROJECTION_" + projection.state().name() + "_" + projection.detail(),
                    1,
                    Integer::sum);
            return;
        }

        candidates.addCandidate(
                endpointAddress, input, projection, projectionInputs.ownershipVersion());
        candidates.inputs[candidates.size - 1] = projectionInputs;
        logProjection(endpointAddress, cacheHit, projection);
    }

    private void logProjection(
            String address, long cacheHit, RequestTimeEstimation.CandidateView projection) {
        if (!Logger.isTraceEnabled()) return;
        Logger.trace(
                "Prefill projection - ip: {}, order: {}, hitCache: {}, ttftMs: {}",
                address,
                priorityOrdering ? "PRIORITY" : "FIFO",
                cacheHit,
                projection.projectedTtftMsValue());
    }
}
