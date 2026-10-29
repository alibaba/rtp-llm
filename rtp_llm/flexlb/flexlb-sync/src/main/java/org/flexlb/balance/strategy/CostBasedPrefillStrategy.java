package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Map;

import static com.google.common.math.LongMath.saturatedAdd;
import static com.google.common.math.LongMath.saturatedMultiply;

@Component
public class CostBasedPrefillStrategy {

    private static final int MAX_CACHED_WORKERS = 4096;
    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final EndpointRegistry endpointRegistry;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();

    private final boolean priorityOrdering;
    private final boolean queueScheduling;
    private final Long cacheAffinityExtraTtftMs;
    private final double cacheAffinityMinHitRate;
    private final String deliveryMode;

    public CostBasedPrefillStrategy(
            EndpointRegistry endpointRegistry,
            CacheAwareService cacheAwareService,
            EngineHealthReporter engineHealthReporter,
            CacheMetricsReporter cacheMetricsReporter,
            ConfigService configService) {
        this.endpointRegistry = endpointRegistry;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheMetricsReporter = cacheMetricsReporter;
        var config = configService.loadBalanceConfig();
        priorityOrdering = config.isPriorityOrdering();
        queueScheduling = config.isQueue();
        deliveryMode = config.getDispatcher().typeName();
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        cacheAffinityExtraTtftMs =
                affinity == null ? null : Math.max(0L, affinity.getMaxExtraTtftMs());
        cacheAffinityMinHitRate =
                affinity == null ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent());
    }

    public PlacementResult<ServerStatus, PlacementKey> select(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group,
            RequestRoute route) {
        PrefillCandidateSet cached = CANDIDATES.get();
        PrefillCandidateSet candidates = cached.inUse ? new PrefillCandidateSet() : cached;
        candidates.inUse = true;
        try {
            candidates.capture(endpointRegistry.prefillRoutingSnapshot(roleType), group);
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
            evaluateCandidates(request, metadata, cacheMatches, rejections, candidates);
            if (candidates.size() == 0) {
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
            return selected < 0
                    ? PlacementResult.blocked(new PlacementKey(roleType, group, null))
                    : acceptCandidate(
                            request, metadata, roleType, group, route, candidates, selected);
        } finally {
            candidates.clear();
            candidates.inUse = false;
            if (candidates == cached && candidates.capacity() > MAX_CACHED_WORKERS) {
                CANDIDATES.remove();
            }
        }
    }

    private PlacementResult<ServerStatus, PlacementKey> acceptCandidate(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            RequestRoute route,
            PrefillCandidateSet candidates,
            int selected) {
        PrefillEndpoint endpoint = candidates.endpoint(selected);
        String address = candidates.endpointAddress(selected);
        WorkerEndpoint.GenerationPin pin = endpointRegistry.capture(role, address);
        try {
            if (pin == null || pin.endpoint() != endpoint) {
                return PlacementResult.blocked(new PlacementKey(role, group, null));
            }
            String admissionBlock = candidates.admissionBlock(selected);
            if (admissionBlock != null
                    || !queueScheduling
                            && !endpoint.ownership.requestCapacityAvailable(
                                    endpoint.requestLimit, 1)) {
                Response failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                if (admissionBlock == null) {
                    var reason =
                            classifyWorkerCapacity(
                                    endpoint.ownership.admissionSummary(
                                            metadata.priority(), endpoint.requestLimit));
                    failure =
                            reason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                                            || reason == AdmissionRejectReason.SAME_PRIORITY_AHEAD
                                    ? Response.error(
                                            StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason)
                                    : reason == AdmissionRejectReason.UNSPECIFIED
                                            ? Response.error(
                                                    StrategyErrorType.ADMISSION_UNAVAILABLE)
                                            : failure;
                }
                return blockedWithDiagnostics(
                        failure,
                        PlacementKey.exact(
                                role, endpoint.getStatus().topologySnapshot().group(), address),
                        candidates.registeredCount,
                        Map.of(
                                "reason",
                                admissionBlock == null ? "PREFILL_CAPACITY" : admissionBlock,
                                "selectedEndpoint",
                                address,
                                "cacheHitTokens",
                                candidates.cacheHit(selected)));
            }
            WorkerStatus status = endpoint.getStatus();
            ServerStatus assignment =
                    ServerStatus.forWorker(
                            role,
                            request.getRequestId(),
                            status.topologySnapshot(),
                            status.committedEngineObservation());
            long ttft = candidates.projectedTtftMs(selected);
            if (ttft >= 0L) {
                assignment.setPrefillTime(ttft);
            }
            DebugInfo debug = new DebugInfo();
            debug.setHitCacheLen(candidates.cacheHit(selected));
            assignment.setDebugInfo(debug);
            // Route.add consumes the pin even when validation fails.
            WorkerEndpoint.GenerationPin ownedPin = pin;
            pin = null;
            route.add(
                    ownedPin,
                    assignment,
                    candidates.prefillMs(selected),
                    candidates.ownershipVersion(selected));
            reportSelection(role, request.getSeqLen(), candidates, selected);
            return PlacementResult.success(assignment);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    private static AdmissionRejectReason classifyWorkerCapacity(
            PrefillState.AdmissionSummary summary) {
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

    private static PlacementResult<ServerStatus, PlacementKey> blockedWithDiagnostics(
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
        if (survivors.size() == 0) {
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
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.ttftRank(i) == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(referenceHitTokens, survivors.cacheHit(i));
                }
            }
            if (maxHitTokens > referenceHitTokens) {
                boolean minimumHitRateMet = false;
                double minimumHitRate = cacheAffinityMinHitRate;
                for (int i = 0; i < survivors.size(); i++) {
                    long hitTokens = survivors.cacheHit(i);
                    if (hitTokens <= referenceHitTokens
                            || minimumHitRate > 0.0
                                    && (seqLen <= 0L
                                            || hitTokens * RoutingConfig.PERCENTAGE_SCALE / seqLen
                                                    < minimumHitRate)) {
                        continue;
                    }
                    minimumHitRateMet = true;
                    if (survivors.projectedTtftMs(i) >= 0L
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
            cacheMetricsReporter.reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex).getIp(), affinityReason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minProjectedTtftMs: {}, "
                                + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                                + "hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddress(selectedIndex),
                        minProjectedTtftMs,
                        survivors.ttftRank(selectedIndex),
                        affinityCutoffMs,
                        survivors.cacheHit(selectedIndex),
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
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.ttftRank(i) == ttft
                    && (!candidates.hasKnownTtft || candidates.projectedTtftMs(i) >= 0L)
                    && (hit == Long.MIN_VALUE || candidates.cacheHit(i) == hit)) {
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
            PrefillCandidateSet candidates) {
        long planningAtMs = System.currentTimeMillis();
        var estimator = RequestTimeEstimation.current();
        long seqLen = request.getSeqLen();
        // Only overwrite visited slots; the unvisited fleet remains intact after rejections.
        for (int i = 0; i < candidates.registeredCount; i++) {
            PrefillEndpoint ep = candidates.endpoint(i);
            String endpointAddress = candidates.endpointAddress(i);
            long cacheHit = 0L;
            long routingCacheMatchTokens = 0L;
            Integer prefix =
                    cacheMatchResults == null ? null : cacheMatchResults.get(endpointAddress);
            if (seqLen > 0L && prefix != null && prefix > 0) {
                long blockSize = request.getCacheKeyBlockSize();
                if (blockSize <= 0L) {
                    CacheStatus cache = ep.getStatus().getCacheStatus();
                    blockSize = cache == null ? 0L : cache.getBlockSize();
                }
                if (blockSize > 0L) {
                    long rawHit = saturatedMultiply(blockSize, prefix.longValue());
                    routingCacheMatchTokens = Math.min(rawHit, seqLen);
                    cacheHit = rawHit >= seqLen ? Math.max(0L, seqLen - blockSize) : rawHit;
                }
            }
            RequestTimeEstimation.Inputs projectionInputs = ep.captureRouteProjectionInputs();
            PrefillTimePredictor predictor = ep.predictor;
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
                            predictor == null ? null : predictor.evaluator(),
                            ep.deliveryMode,
                            planningAtMs);
            if (!projection.selectable() && !projection.engineWorkUnmodeled()) {
                rejections.merge(
                        "PROJECTION_" + projection.state().name() + "_" + projection.detail(),
                        1,
                        Integer::sum);
                continue;
            }

            candidates.addCandidate(
                    endpointAddress, ep, projection, projectionInputs.ownershipVersion());
            var admission =
                    RequestTimeEstimation.applyAdmissionPolicy(
                            projectionInputs.queue(), projection);
            if (!admission.selectable() && !admission.engineWorkUnmodeled()) {
                candidates.admissionBlock(admission.detail());
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

    private void reportSelection(
            RoleType role, long seqLen, PrefillCandidateSet candidates, int selected) {
        long ttft = candidates.projectedTtftMs(selected);
        if (ttft >= 0L) {
            PrefillEndpoint endpoint = candidates.endpoint(selected);
            try {
                engineHealthReporter.reportPrefillSelectedEstimates(
                        role, endpoint.getIp(), deliveryMode, ttft, candidates.prefillMs(selected));
            } catch (RuntimeException telemetryFailure) {
                Logger.warn(
                        "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                        endpoint.ipPort(),
                        deliveryMode,
                        telemetryFailure);
            }
        }
        long hit = candidates.cacheHit(selected);
        cacheMetricsReporter.reportCacheHitMetrics(
                role, hit, seqLen > 0L ? hit / (double) seqLen : 0.0);
        cacheMetricsReporter.reportRoutingSelectedCacheMatchMetrics(
                role, candidates.routingCacheMatchTokens(selected), seqLen);
        cacheMetricsReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                role, candidates.maximumRoutingCacheMatchTokens);
    }
}
