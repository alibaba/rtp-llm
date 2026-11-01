package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
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
    private final Long cacheAffinityExtraTtftMs;
    private final double cacheAffinityMinHitRate;

    public CostBasedPrefillStrategy(
            CacheAwareService cacheAwareService, ConfigService configService) {
        this.cacheAwareService = cacheAwareService;
        var config = configService.loadBalanceConfig();
        priorityOrdering = config.isPriorityOrdering();
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
            if (candidates.registeredCount == 0) {
                return blockedWithDiagnostics(
                        Response.error(roleType.getErrorType()),
                        new PlacementKey(roleType, group, null),
                        0,
                        Map.of());
            }
            Map<String, Integer> cacheMatches =
                    cacheAwareService.findMatchingEngines(
                            request.getBlockCacheKeys(), roleType, candidates.addresses);
            Map<String, Integer> rejections = new java.util.HashMap<>();
            evaluateCandidates(
                    request, metadata, cacheMatches, rejections, candidates, clock.getAsLong());
            if (candidates.size == 0) {
                return blockedWithDiagnostics(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
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
            if (selected < 0) {
                return PlacementResult.blocked(new PlacementKey(roleType, group, null));
            }
            // This invocation populated the scratch exclusively from List<I>.
            @SuppressWarnings("unchecked")
            I winner = (I) candidates.capturedSources[selected];
            String block = candidates.admissionBlocks[selected];
            if (block != null) {
                return blockedWithDiagnostics(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
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
                                candidates.affinityReason == null
                                        ? ""
                                        : candidates.affinityReason));
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
        } finally {
            candidates.clear();
            candidates.inUse = false;
            if (candidates == cached && candidates.capturedSources.length > MAX_CACHED_WORKERS) {
                CANDIDATES.remove();
            }
        }
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

        long bestHit = Long.MIN_VALUE;
        long bestTtft = Long.MAX_VALUE;
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinityExtraTtftMs != null && survivors.hasKnownTtft) {
            long referenceHitTokens = 0L;
            long maxHitTokens = survivors.maximumCacheHit;
            affinityCutoffMs = saturatedAdd(minProjectedTtftMs, cacheAffinityExtraTtftMs);
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size; i++) {
                if (survivors.ttftRank(i) == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(referenceHitTokens, survivors.cacheHitTokens[i]);
                }
            }
            if (maxHitTokens > referenceHitTokens) {
                boolean minimumHitRateMet = false;
                double minimumHitRate = cacheAffinityMinHitRate;
                for (int i = 0; i < survivors.size; i++) {
                    long hitTokens = survivors.cacheHitTokens[i];
                    if (hitTokens <= referenceHitTokens
                            || minimumHitRate > 0.0
                                    && (seqLen <= 0L
                                            || hitTokens * RoutingConfig.PERCENTAGE_SCALE / seqLen
                                                    < minimumHitRate)) {
                        continue;
                    }
                    minimumHitRateMet = true;
                    if (survivors.projectedTtftMs[i] >= 0L
                            && survivors.ttftRank(i) <= affinityCutoffMs) {
                        long ttft = survivors.ttftRank(i);
                        if (hitTokens > bestHit || hitTokens == bestHit && ttft < bestTtft) {
                            bestHit = hitTokens;
                            bestTtft = ttft;
                        }
                    }
                }
                affinityReason =
                        bestHit != Long.MIN_VALUE
                                ? "CACHE_LEADER"
                                : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        int selectedIndex =
                selectTiedCandidate(
                        survivors,
                        bestHit != Long.MIN_VALUE ? bestTtft : minProjectedTtftMs,
                        bestHit,
                        roleType,
                        group);

        if (selectedIndex >= 0 && affinityReason != null) {
            survivors.affinityReason = affinityReason;
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minProjectedTtftMs: {}, "
                                + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                                + "hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddresses[selectedIndex],
                        minProjectedTtftMs,
                        survivors.ttftRank(selectedIndex),
                        affinityCutoffMs,
                        survivors.cacheHitTokens[selectedIndex],
                        affinityReason);
            }
        }
        return selectedIndex;
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
            long cacheHit = 0L;
            long routingCacheMatchTokens = 0L;
            Integer prefix =
                    cacheMatchResults == null ? null : cacheMatchResults.get(endpointAddress);
            if (seqLen > 0L && prefix != null && prefix > 0) {
                long blockSize = request.getCacheKeyBlockSize();
                if (blockSize <= 0L) {
                    blockSize = input.cacheBlockSize();
                }
                if (blockSize > 0L) {
                    long rawHit = saturatedMultiply(blockSize, prefix.longValue());
                    routingCacheMatchTokens = Math.min(rawHit, seqLen);
                    cacheHit = rawHit >= seqLen ? Math.max(0L, seqLen - blockSize) : rawHit;
                }
            }
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
            if (Logger.isTraceEnabled()) {
                Logger.trace(
                        "Prefill projection - ip: {}, order: {}, hitCache: {}, ttftMs: {}",
                        endpointAddress,
                        priorityOrdering ? "PRIORITY" : "FIFO",
                        cacheHit,
                        projection.projectedTtftMsValue());
            }
        }
    }
}
