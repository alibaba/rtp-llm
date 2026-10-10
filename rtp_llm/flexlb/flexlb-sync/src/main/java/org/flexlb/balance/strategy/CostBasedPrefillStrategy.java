package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.RouteTimelineProjector;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;
import static com.google.common.math.LongMath.saturatedMultiply;

@Component
public class CostBasedPrefillStrategy {

    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final EndpointRegistry endpointRegistry;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();

    public CostBasedPrefillStrategy(EndpointRegistry endpointRegistry,
                                    CacheAwareService cacheAwareService,
                                    EngineHealthReporter engineHealthReporter,
                                    CacheMetricsReporter cacheMetricsReporter) {
        this.endpointRegistry = endpointRegistry;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheMetricsReporter = cacheMetricsReporter;
    }

    public PlacementResult<WorkerAssignment, RoleType> select(
            RequestRequirements request,
            FlexlbConfig config,
            RoleType roleType,
            String group) {
        long requestId = request.requestId();
        long seqLen = request.seqLen();

        EndpointDiscovery discovery = discoverAvailableEndpoints(request, config, roleType, group);
        if (discovery.candidates().isEmpty()) {
            Logger.debug("Prefill select failed: no admission capacity, request_id={}",
                    requestId);
            return classifyAdmissionFailure(
                    discovery.directory(), request.priority(), roleType, group);
        }
        Map<String, Integer> cacheMatchResults =
                cacheAwareService.findMatchingEngines(request.blockCacheKeys(), roleType, discovery.addresses());
        Map<String, Integer> rejections = new java.util.HashMap<>();
        Map<RoleType, Integer> poolWideBlockers =
                new EnumMap<>(RoleType.class);
        PrefillCandidateSet survivors = evaluateCandidates(
                discovery,
                request,
                config,
                cacheMatchResults,
                rejections,
                poolWideBlockers);
        if (survivors.size() == 0) {
            Logger.debug(
                    "Prefill select failed: no available endpoints, request_id={},"
                        + " rejections={}",
                    requestId,
                    rejections);
            RoleType poolWideBlocker = provenPoolWideBlocker(
                    poolWideBlockers,
                    discovery.registeredCount());
            if (poolWideBlocker != null) {
                return blockedWithDiagnostics(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                        poolWideBlocker, discovery.registeredCount(), rejections);
            }
            return blockedWithDiagnostics(
                    Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                    roleType, discovery.registeredCount(), rejections);
        }

        int selectedIndex = selectBestCandidate(
                survivors, survivors.minimumTtftRank,
                roleType, group, seqLen, config);

        if (selectedIndex < 0) {
            Logger.debug(
                    "Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId,
                    rejections);
            return PlacementResult.blocked(roleType);
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        long selectedPrefillMs = survivors.prefillMs(selectedIndex);
        WorkerEndpoint.GenerationPin selectedPin =
                endpointRegistry.capture(
                        roleType,
                        survivors.endpointAddress(selectedIndex));
        if (selectedPin == null || selectedPin.endpoint() != best) {
            if (selectedPin != null) {
                selectedPin.close();
            }
            // The planning endpoint retired or was replaced after the
            // full-fleet snapshot. Re-enter the queue rather than
            // transferring ownership for a stale generation.
            return PlacementResult.blocked(roleType);
        }
        long selectedTtft = survivors.projectedTtftMs(selectedIndex);
        WorkerAssignment assignment = buildWorkerAssignment(
                best,
                roleType,
                requestId,
                selectedTtft,
                selectedPrefillMs,
                bestCacheHit,
                survivors.ownershipVersion(selectedIndex),
                selectedPin);
        try {
            if (selectedTtft >= 0L) { reportSelectedEstimates(
                    roleType,
                    best,
                    config,
                    selectedTtft,
                    selectedPrefillMs); }
            cacheMetricsReporter.reportCacheHitMetrics(roleType, bestCacheHit,
                    seqLen > 0 ? bestCacheHit / (double) seqLen : 0.0);
            cacheMetricsReporter.reportRoutingSelectedCacheMatchMetrics(
                    roleType, survivors.routingCacheMatchTokens(selectedIndex), seqLen);
            cacheMetricsReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                    roleType, survivors.maximumRoutingCacheMatchTokens);
            return PlacementResult.success(assignment);
        } catch (Throwable failure) {
            try (assignment) {
                throw failure;
            }
        }
    }

    private static PlacementResult<WorkerAssignment, RoleType> classifyAdmissionFailure(
            List<EndpointRegistry.PrefillRoutingEntry> directory,
            int priority, RoleType role, String group) {
        AdmissionRejectReason reason = null;
        int workerCount = 0;
        long outstandingRequestCount = 0;
        for (var entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            workerCount++;
            var summary = endpoint.admissionSummary(priority);
            outstandingRequestCount += summary.occupiedRequests();
            AdmissionRejectReason workerReason = classifyWorkerCapacity(summary);
            if (reason == null) {
                reason = workerReason;
            } else if (reason == AdmissionRejectReason.UNSPECIFIED
                    || workerReason == AdmissionRejectReason.UNSPECIFIED) {
                reason = AdmissionRejectReason.UNSPECIFIED;
            } else if (reason != workerReason) {
                reason = AdmissionRejectReason.RESOURCE_EXHAUSTED;
            }
        }
        Response failure = reason == null ? Response.error(role.getErrorType()) : switch (reason) {
            case HIGHER_PRIORITY_AHEAD, SAME_PRIORITY_AHEAD ->
                    Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason);
            case RESOURCE_EXHAUSTED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
            case UNSPECIFIED -> Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        };
        return blockedWithDiagnostics(failure, role, workerCount, Map.of("observedRequests", outstandingRequestCount));
    }

    private static AdmissionRejectReason classifyWorkerCapacity(PrefillState.AdmissionSummary summary) {
        long residual = Math.max(0L, summary.requestsToRelease() - summary.lowerPriorityRequests());
        long protectedRequests = summary.higherPriorityRequests() + summary.samePriorityRequests();
        if (residual > protectedRequests && summary.unknownPriorityRequests() > 0L) {
            return AdmissionRejectReason.UNSPECIFIED;
        }
        if (residual > 0L && protectedRequests >= residual) {
            return summary.higherPriorityRequests() > 0L ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    : AdmissionRejectReason.SAME_PRIORITY_AHEAD;
        }
        return AdmissionRejectReason.RESOURCE_EXHAUSTED;
    }

    private static PlacementResult<WorkerAssignment, RoleType> blockedWithDiagnostics(
            Response failure, RoleType role, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(role, failure, Map.of("role", role.name(), "workers", workerCount,
                "observedAtMs", System.currentTimeMillis(), "details", Map.copyOf(details)));
    }

    /** Select from candidates that already passed the common hard filters. */
    int selectBestCandidate(PrefillCandidateSet survivors,
                            long minProjectedTtftMs,
                            RoleType roleType,
                            String group,
                            long seqLen,
                            FlexlbConfig config) {
        if (survivors.size() == 0) {
            return -1;
        }

        var cacheAffinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null && survivors.hasKnownTtft) {
            long referenceHitTokens = 0L;
            long maxHitTokens = survivors.maximumCacheHit;
            affinityCutoffMs = saturatedAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.ttftRank(i)
                        == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(
                            referenceHitTokens,
                            survivors.cacheHit(i));
                }
            }
            if (maxHitTokens > referenceHitTokens) {
                boolean minimumHitRateMet = false;
                double minimumHitRate = normalizedHitRate(
                        cacheAffinity.getMinPrefixHitPercent());
                for (int i = 0; i < survivors.size(); i++) {
                    long hitTokens = survivors.cacheHit(i);
                    if (hitTokens <= referenceHitTokens
                            || minimumHitRate > 0.0
                                    && (seqLen <= 0L
                                        || hitTokens
                                                * RoutingConfig.PERCENTAGE_SCALE
                                                / seqLen
                                                < minimumHitRate)) {
                        continue;
                    }
                    minimumHitRateMet = true;
                    if (survivors.projectedTtftMs(i) >= 0L && survivors.ttftRank(i) <= affinityCutoffMs) {
                        preferredCandidates.set(i);
                    }
                }
                affinityReason = !preferredCandidates.isEmpty()
                        ? "CACHE_LEADER"
                        : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        int selectedIndex;
        if (!preferredCandidates.isEmpty()) {
            selectedIndex = selectCacheLeader(survivors, preferredCandidates, roleType, group);
        } else {
            selectedIndex = selectBaselineCandidate(
                    survivors, minProjectedTtftMs, roleType, group);
        }

        if (selectedIndex >= 0 && affinityReason != null) {
            cacheMetricsReporter.reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex).getIp(),
                    affinityReason);
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

    private int selectBaselineCandidate(PrefillCandidateSet survivors, long minimumTtftMs,
                                        RoleType role, String group) {
        return tieRotation.next(role, group, survivors.size(),
                i -> survivors.ttftRank(i) == minimumTtftMs
                        && (!survivors.hasKnownTtft || survivors.projectedTtftMs(i) >= 0L),
                survivors::endpointAddress);
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            PrefillCandidateSet survivors, BitSet preferredCandidates, RoleType role, String group) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!preferredCandidates.get(candidateIndex)) {
                continue;
            }
            long hit = survivors.cacheHit(candidateIndex);
            long projectedTtftMs = survivors.ttftRank(candidateIndex);
            if (hit > bestHit
                    || hit == bestHit && projectedTtftMs < bestProjectedTtftMs) {
                bestHit = hit;
                bestProjectedTtftMs = projectedTtftMs;
            }
        }
        long selectedHit = bestHit;
        long selectedTtft = bestProjectedTtftMs;
        return tieRotation.next(role, group, survivors.size(),
                i -> preferredCandidates.get(i) && survivors.cacheHit(i) == selectedHit
                        && survivors.ttftRank(i) == selectedTtft,
                survivors::endpointAddress);
    }

    private record EndpointDiscovery(
            List<EndpointRegistry.PrefillRoutingEntry> directory,
            List<EndpointRegistry.PrefillRoutingEntry> candidates,
            int registeredCount) {

        private EndpointDiscovery {
            candidates = java.util.Objects.requireNonNull(
                    candidates, "candidates");
        }

        /** Zero-copy address view over the exact fleet used by this decision. */
        private List<String> addresses() {
            return new AbstractList<>() {
                @Override
                public String get(int index) {
                    return candidates.get(index).address();
                }

                @Override
                public int size() {
                    return candidates.size();
                }
            };
        }
    }

    private PrefillCandidateSet evaluateCandidates(
            EndpointDiscovery discovery,
            RequestRequirements request,
            FlexlbConfig config,
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections,
            Map<RoleType, Integer> poolWideBlockers) {
        int eligibleSize = discovery.candidates().size();
        PrefillCandidateSet candidates = CANDIDATES.get();
        candidates.reset(eligibleSize);
        long planningAtMs = System.currentTimeMillis();
        var projector = RouteTimelineProjector.current();

        // Use one endpoint snapshot for both service prediction and decision-group planning.
        for (int i = 0; i < discovery.candidates().size(); i++) {
            EndpointRegistry.PrefillRoutingEntry routingEntry =
                    discovery.candidates().get(i);
            PrefillEndpoint ep = routingEntry.endpoint();
            String endpointAddress = routingEntry.address();
            CacheTokenMatch cacheMatch =
                    calculateCacheMatch(ep, endpointAddress, cacheMatchResults, request);
            long cacheHit = cacheMatch.effectiveHitTokens();
            long routingCacheMatchTokens = cacheMatch.routingHitTokens();
            RouteProjection.Inputs projectionInputs =
                    ep.captureRouteProjectionInputs();
            PrefillTimePredictor predictor = ep.getPredictor();
            RouteProjection.CandidateView projection =
                    projector.projectView(
                            projectionInputs,
                            request.requestId(),
                            request.priority(),
                            planningAtMs,
                            // ExpirationTimer drives request deadlines through the owning scheduler.
                            // Selection scores endpoint work only; an expiry race
                            // must not be reported as a capacity blocker.
                            Long.MAX_VALUE,
                            request.seqLen(),
                            cacheHit,
                            routingCacheMatchTokens,
                            predictor == null
                                    ? null : predictor.evaluator(),
                            ep.deliveryProjection(),
                            planningAtMs);
            if (!projection.selectable() && !projection.engineWorkUnmodeled()) {
                RoleType blockerRole = projection.blockerRole();
                if (blockerRole != null) {
                    poolWideBlockers.merge(
                            blockerRole, 1, Integer::sum);
                }
                rejections.merge(
                        "PROJECTION_"
                                + projection.state().name()
                                + "_"
                                + projection.detail(),
                        1,
                        Integer::sum);
                continue;
            }

            candidates.addCandidate(
                    endpointAddress, ep, projection,
                    projectionInputs.ownershipVersion());
            if (Logger.isTraceEnabled()) {
                Logger.trace("Prefill projection - ip: {}, order: {}, hitCache: {}, ttftMs: {}",
                        endpointAddress, config.isPriorityOrdering() ? "PRIORITY" : "FIFO",
                        cacheHit, projection.projectedTtftMsValue());
            }
        }

        return candidates;
    }

    static RoleType provenPoolWideBlocker(
            Map<RoleType, Integer> blockers,
            int registeredEndpoints) {
        if (registeredEndpoints <= 0) {
            return null;
        }
        for (Map.Entry<RoleType, Integer> blocker : blockers.entrySet()) {
            if (blocker.getValue() >= registeredEndpoints) {
                return blocker.getKey();
            }
        }
        return null;
    }

    private EndpointDiscovery discoverAvailableEndpoints(
            RequestRequirements request,
            FlexlbConfig config,
            RoleType roleType,
            String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                endpointRegistry.prefillRoutingSnapshot(roleType);
        List<EndpointRegistry.PrefillRoutingEntry> matching = new ArrayList<>();
        int registered = 0;
        boolean preemptQueued = config.allowsPreemption(VictimStage.PREFILL_QUEUED);
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            registered++;
            if (endpoint.canAcceptRequest()
                    || preemptQueued && endpoint.canPreemptQueuedRequest(request.priority())) {
                matching.add(entry);
            }
        }
        return new EndpointDiscovery(directory, matching, registered);
    }

    private record CacheTokenMatch(
            long effectiveHitTokens,
            long routingHitTokens) {
        private static final CacheTokenMatch NONE =
                new CacheTokenMatch(0L, 0L);
    }

    private CacheTokenMatch calculateCacheMatch(
            PrefillEndpoint ep,
            String endpointAddress,
            Map<String, Integer> cacheMatchResults,
            RequestRequirements request) {
        if (cacheMatchResults == null || cacheMatchResults.isEmpty() || request == null) {
            return CacheTokenMatch.NONE;
        }
        long seqLen = request.seqLen();
        if (seqLen <= 0L) {
            return CacheTokenMatch.NONE;
        }
        Integer prefixMatchLength = cacheMatchResults.get(endpointAddress);
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = request.cacheKeyBlockSize();
        WorkerStatus status = ep.getStatus();
        CacheStatus cacheStatus = status == null ? null : status.getCacheStatus();
        if (blockSize <= 0L && cacheStatus != null) {
            blockSize = cacheStatus.getBlockSize();
        }
        if (blockSize <= 0L) {
            return CacheTokenMatch.NONE;
        }
        long rawHit = saturatedMultiply(blockSize, prefixMatchLength.longValue());
        long routingHit = Math.clamp(rawHit, 0L, seqLen);
        long effectiveHit = rawHit >= seqLen
                ? Math.max(0L, seqLen - blockSize)
                : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportSelectedEstimates(
            RoleType roleType,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            long projectedTtftMs,
            long executionTimeMs) {
        String deliveryMode = config.getDispatcher().typeName();
        try {
            engineHealthReporter.reportPrefillSelectedEstimates(
                    roleType,
                    endpoint.getIp(),
                    deliveryMode,
                    projectedTtftMs,
                    executionTimeMs);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    endpoint.ipPort(), deliveryMode, telemetryFailure);
        }
    }

    private WorkerAssignment buildWorkerAssignment(
            PrefillEndpoint ep,
            RoleType roleType,
            long requestId,
            long projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin) {
        try {
            // Populate DebugInfo so RequestRoute.hitCache() can read
            // hitCacheLen for batch metrics.
            DebugInfo debugInfo = new DebugInfo();
            debugInfo.setHitCacheLen(bestCacheHit);

            checkState(selectedPin != null && selectedPin.endpoint() == ep,
                    "selected Prefill endpoint generation changed before handoff");
            WorkerStatus workerStatus = ep.getStatus();
            WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
            WorkerStatus.EngineObservation status = workerStatus.committedEngineObservation();
            ServerStatus result = WorkerAssignment.workerMetadata(roleType, requestId, topology, status);
            if (projectedTtftMs >= 0L) { result.setPrefillTime(projectedTtftMs); }
            result.setDebugInfo(debugInfo);
            WorkerEndpoint.GenerationPin ownedPin = selectedPin;
            selectedPin = null;
            return WorkerAssignment.prefill(
                    ownedPin, result, selectedPrefillMs, placementVersion);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

}
