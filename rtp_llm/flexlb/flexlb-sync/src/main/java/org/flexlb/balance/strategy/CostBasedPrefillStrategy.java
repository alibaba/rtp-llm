package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.service.CacheAwareService;
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
import static com.google.common.math.LongMath.saturatedMultiply;

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

    public CostBasedPrefillStrategy(
            CacheAwareService cacheAwareService, ConfigService configService) {
        this.cacheAwareService = cacheAwareService;
        var config = configService.loadBalanceConfig();
        priorityOrdering = config.isPriorityOrdering();
        queueScheduling = config.isQueue();
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        cacheAffinityExtraTtftMs =
                affinity == null ? null : Math.max(0L, affinity.getMaxExtraTtftMs());
        cacheAffinityMinHitRate =
                affinity == null ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent());
    }

    @Override
    public <I extends CandidateInput> PlacementResult<Selection<I>, PlacementKey> select(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group,
            List<I> inputs,
            LongSupplier clock) {
        PrefillCandidateSet cached = CANDIDATES.get();
        PrefillCandidateSet candidates = cached.inUse ? new PrefillCandidateSet() : cached;
        candidates.inUse = true;
        try {
            candidates.capture(inputs, group);
            Map<String, Integer> rejections =
                    candidates.registeredCount == 0 ? Map.of() : new java.util.HashMap<>();
            if (candidates.registeredCount != 0) {
                var cacheMatches =
                        cacheAwareService.findMatchingEngines(
                                request.getBlockCacheKeys(), roleType, candidates.addresses);
                evaluateCandidates(
                        request, metadata, cacheMatches, rejections, candidates, clock.getAsLong());
            }
            if (candidates.size == 0) {
                return blockedWithDiagnostics(
                        Response.error(
                                candidates.registeredCount == 0
                                        ? roleType.getErrorType()
                                        : StrategyErrorType.RESOURCE_EXHAUSTED),
                        new PlacementKey(roleType, group, null),
                        candidates.registeredCount,
                        rejections);
            }
            int selected =
                    selectBestCandidate(
                            candidates,
                            candidates.minimumTtftRank,
                            roleType,
                            group,
                            request.getSeqLen());
            return selected < 0
                    ? PlacementResult.blocked(new PlacementKey(roleType, group, null))
                    : selection(candidates, selected, roleType, metadata);
        } finally {
            candidates.clear();
            candidates.inUse = false;
            if (candidates == cached && candidates.capturedSources.length > MAX_CACHED_WORKERS) {
                CANDIDATES.remove();
            }
        }
    }

    private <I extends CandidateInput> PlacementResult<Selection<I>, PlacementKey> selection(
            PrefillCandidateSet candidates,
            int selected,
            RoleType roleType,
            SchedulingMetadata metadata) {
        @SuppressWarnings("unchecked")
        I winner = (I) candidates.capturedSources[selected];
        String block = candidates.admissionBlocks[selected];
        StrategyErrorType error = StrategyErrorType.RESOURCE_EXHAUSTED;
        AdmissionRejectReason reason = AdmissionRejectReason.RESOURCE_EXHAUSTED;
        if (block == null && !queueScheduling && !winner.requestCapacityAvailable()) {
            block = "PREFILL_CAPACITY";
            reason = classifyWorkerCapacity(winner.admissionSummary(metadata.priority()));
            error =
                    switch (reason) {
                        case HIGHER_PRIORITY_AHEAD, SAME_PRIORITY_AHEAD ->
                                StrategyErrorType.PRIORITY_ADMISSION_REJECTED;
                        case UNSPECIFIED -> StrategyErrorType.ADMISSION_UNAVAILABLE;
                        default -> StrategyErrorType.RESOURCE_EXHAUSTED;
                    };
        }
        if (block != null) {
            return blockedWithDiagnostics(
                    Response.error(error, reason),
                    PlacementKey.exact(roleType, winner.group(), winner.address()),
                    candidates.registeredCount,
                    Map.of(
                            "reason",
                            block,
                            "selectedEndpoint",
                            winner.address(),
                            "cacheHitTokens",
                            candidates.cacheHitTokens[selected],
                            "affinityReason",
                            candidates.affinityReason == null ? "" : candidates.affinityReason));
        }
        return PlacementResult.success(
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
                        winner));
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
            PlacementResult<Selection<I>, PlacementKey> blockedWithDiagnostics(
                    Response failure, PlacementKey key, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(
                key,
                failure,
                Map.of(
                        "role",
                        key.role().name(),
                        "workers",
                        workerCount,
                        "observedAtMs",
                        System.currentTimeMillis(),
                        "details",
                        Map.copyOf(details)));
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
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections,
            PrefillCandidateSet candidates,
            long planningAtMs) {
        var estimator = RequestTimeEstimation.current();
        long seqLen = request.getSeqLen();
        // Only overwrite visited slots; the unvisited fleet remains intact after rejections.
        for (int i = 0; i < candidates.registeredCount; i++) {
            var input = candidates.capturedSources[i];
            String endpointAddress = candidates.endpointAddresses[i];
            Integer prefix =
                    cacheMatchResults == null ? null : cacheMatchResults.get(endpointAddress);
            long blockSize = matchingBlockSize(request, input, prefix);
            long rawHit = blockSize <= 0L ? 0L : saturatedMultiply(blockSize, prefix.longValue());
            long routingCacheMatchTokens = Math.min(rawHit, Math.max(0L, seqLen));
            long cacheHit = rawHit >= seqLen ? Math.max(0L, seqLen - blockSize) : rawHit;
            RequestTimeEstimation.Inputs projectionInputs = input.projectionInputs();
            RequestTimeEstimation.CandidateView projection =
                    estimator.projectRankingView(
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
            if (!projection.selectable() && !projection.engineWorkUnmodeled()) {
                rejections.merge(
                        "PROJECTION_" + projection.state().name() + "_" + projection.detail(),
                        1,
                        Integer::sum);
                continue;
            }

            candidates.addCandidate(
                    endpointAddress, input, projection, projectionInputs.ownershipVersion());
            var admission =
                    RequestTimeEstimation.applyAdmissionPolicy(
                            projectionInputs.queue(), projection);
            if (!admission.selectable() && !admission.engineWorkUnmodeled()) {
                candidates.admissionBlocks[candidates.size - 1] = admission.detail();
            }
            logProjection(endpointAddress, cacheHit, projection);
        }
    }

    private static long matchingBlockSize(Request request, CandidateInput input, Integer prefix) {
        if (request.getSeqLen() <= 0L || prefix == null || prefix <= 0) return 0L;
        long size = request.getCacheKeyBlockSize();
        return size > 0L ? size : Math.max(0L, input.cacheBlockSize());
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
