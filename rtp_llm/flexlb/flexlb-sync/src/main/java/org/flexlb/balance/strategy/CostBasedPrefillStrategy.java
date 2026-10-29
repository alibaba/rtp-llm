package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RouteTimelineProjector;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.balance.scheduler.RequestData;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.RoutingDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

@Component
public class CostBasedPrefillStrategy {

    private static final int MAX_CACHED_WORKERS = 4096;
    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final EndpointRegistry workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();

    public CostBasedPrefillStrategy(
            EndpointRegistry workerDirectory,
            CacheAwareService cacheAwareService,
            EngineHealthReporter engineHealthReporter) {
        this.workerDirectory = workerDirectory;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
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
            return selectCandidate(request, metadata, roleType, group, route, candidates);
        } finally {
            candidates.clear();
            candidates.inUse = false;
            if (candidates == cached && candidates.capacity() > MAX_CACHED_WORKERS) {
                CANDIDATES.remove();
            }
        }
    }

    private PlacementResult<ServerStatus, PlacementKey> selectCandidate(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group,
            RequestRoute route,
            PrefillCandidateSet candidates) {
        RequestData balanceContext = route.ctx().data();
        balanceContext.beginRoutingAttempt(roleType);
        if (balanceContext.getSchedulingMetadata() == null) {
            balanceContext.setSchedulingMetadata(metadata);
        }
        FlexlbConfig config = balanceContext.getConfig();
        String requestId = request.getRequestId();

        long seqLen = request.getSeqLen();

        EndpointDiscovery discovery = discoverAvailableEndpoints(balanceContext, roleType, group);
        if (discovery.candidates().isEmpty()) {
            Logger.debug("Prefill select failed: no admission capacity, request_id={}", requestId);
            balanceContext.recordSelectionReason(roleType, "NO_ADMISSION_CAPACITY");
            return blockedWithDiagnostics(
                    Response.error(roleType.getErrorType()),
                    new PlacementKey(roleType, group, null),
                    0,
                    Map.of());
        }
        CacheMatchResult cacheMatchResult = getCacheMatchResult(balanceContext, roleType, group);
        Map<String, Integer> rejections = new java.util.HashMap<>();
        Map<RoleType, Integer> poolWideBlockers = new EnumMap<>(RoleType.class);
        PrefillCandidateSet survivors =
                evaluateCandidates(
                        discovery,
                        balanceContext,
                        cacheMatchResult,
                        rejections,
                        poolWideBlockers,
                        candidates);

        if (survivors.size() == 0) {
            Logger.debug(
                    "Prefill select failed: no available endpoints, request_id={},"
                            + " rejections={}",
                    requestId,
                    rejections);
            balanceContext.recordSelectionReason(roleType, "NO_AVAILABLE_CANDIDATES");
            recordDecision(
                    balanceContext,
                    roleType,
                    group,
                    discovery.registeredCount(),
                    survivors,
                    -1,
                    "NO_AVAILABLE_CANDIDATES",
                    rejections);
            RoleType poolWideBlocker =
                    provenPoolWideBlocker(poolWideBlockers, discovery.registeredCount());
            if (poolWideBlocker != null) {
                return blockedWithDiagnostics(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                        new PlacementKey(poolWideBlocker, group, null),
                        discovery.registeredCount(),
                        rejections);
            }
            return blockedWithDiagnostics(
                    Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                    new PlacementKey(roleType, group, null),
                    discovery.registeredCount(),
                    rejections);
        }

        int selectedIndex =
                selectBestCandidate(
                        survivors, survivors.minimumTtftRank, roleType, group, seqLen, config);

        if (selectedIndex < 0) {
            Logger.debug(
                    "Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId,
                    rejections);
            balanceContext.recordSelectionReason(roleType, "NO_SELECTED_CANDIDATE");
            recordDecision(
                    balanceContext,
                    roleType,
                    group,
                    discovery.registeredCount(),
                    survivors,
                    -1,
                    "NO_SELECTED_CANDIDATE",
                    rejections);
            return PlacementResult.blocked(new PlacementKey(roleType, group, null));
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        long selectedPrefillMs = survivors.prefillMs(selectedIndex);
        WorkerEndpoint.GenerationPin selectedPin =
                workerDirectory.capture(roleType, survivors.endpointAddress(selectedIndex));
        if (selectedPin == null || selectedPin.endpoint() != best) {
            recordDecision(
                    balanceContext,
                    roleType,
                    group,
                    discovery.registeredCount(),
                    survivors,
                    -1,
                    "ENDPOINT_GENERATION_CHANGED",
                    rejections);
            if (selectedPin != null) {
                selectedPin.close();
            }
            // The planning endpoint retired or was replaced after the
            // full-fleet snapshot. Re-enter the queue rather than
            // transferring ownership for a stale generation.
            return PlacementResult.blocked(new PlacementKey(roleType, group, null));
        }
        String admissionBlock = survivors.admissionBlock(selectedIndex);
        if (admissionBlock != null
                || !config.isQueue()
                        && !best.ownership.requestCapacityAvailable(best.requestLimit, 1)) {
            selectedPin.close();
            Response failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
            if (admissionBlock == null) {
                var reason =
                        classifyWorkerCapacity(
                                best.ownership.admissionSummary(
                                        metadata.priority(), best.requestLimit));
                failure =
                        reason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                                        || reason == AdmissionRejectReason.SAME_PRIORITY_AHEAD
                                ? Response.error(
                                        StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason)
                                : reason == AdmissionRejectReason.UNSPECIFIED
                                        ? Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE)
                                        : failure;
            }
            return blockedWithDiagnostics(
                    failure,
                    PlacementKey.exact(
                            roleType,
                            best.getStatus().topologySnapshot().group(),
                            survivors.endpointAddress(selectedIndex)),
                    discovery.registeredCount(),
                    Map.of(
                            "reason",
                            admissionBlock == null ? "PREFILL_CAPACITY" : admissionBlock,
                            "selectedEndpoint",
                            survivors.endpointAddress(selectedIndex),
                            "cacheHitTokens",
                            bestCacheHit));
        }
        long selectedTtft = survivors.projectedTtftMs(selectedIndex);
        ServerStatus selectedRole =
                buildSelectedRole(
                        best,
                        roleType,
                        requestId,
                        selectedTtft,
                        selectedPrefillMs,
                        bestCacheHit,
                        survivors.ownershipVersion(selectedIndex),
                        selectedPin,
                        route);
        {
            if (balanceContext.selectionReason(roleType) == null) {
                balanceContext.recordSelectionReason(roleType, "COST_BASED");
            }
            cacheAwareService.trackRoutingPrediction(
                    requestId,
                    roleType,
                    group,
                    best.getStatus(),
                    seqLen,
                    bestCacheHit,
                    cacheMatchResult);
            HostCacheMatch selectedMatch = cacheMatchResult.hostMatch(best.getStatus());
            if (cacheMatchResult.source() == CacheMatchSource.KVCM
                    && cacheMatchResult.querySucceeded()) {
                engineHealthReporter.reportKvcmSelectedMatch(
                        roleType,
                        best.getStatus().getMetricIpPort(),
                        selectedMatch == null
                                ? 0
                                : CacheMatchResult.matchedTokens(
                                        selectedMatch.localMatchBlocks(),
                                        cacheMatchResult.blockSize(),
                                        seqLen),
                        selectedMatch == null
                                ? 0
                                : CacheMatchResult.matchedTokens(
                                        selectedMatch.globalMatchBlocks(),
                                        cacheMatchResult.blockSize(),
                                        seqLen),
                        seqLen);
            }
            balanceContext.recordCacheSelection(roleType, best.getIp(), bestCacheHit);
            recordDecision(
                    balanceContext,
                    roleType,
                    group,
                    discovery.registeredCount(),
                    survivors,
                    selectedIndex,
                    balanceContext.selectionReason(roleType),
                    rejections);
            if (selectedTtft >= 0L) {
                reportSelectedEstimates(roleType, best, config, selectedTtft, selectedPrefillMs);
            }
            reportCacheHitMetrics(
                    roleType, best.getStatus().getMetricIpPort(), bestCacheHit, seqLen);
            reportRoutingCacheMatchMetrics(
                    roleType,
                    survivors.routingCacheMatchTokens(selectedIndex),
                    survivors.maximumRoutingCacheMatchTokens,
                    seqLen);
        }
        return PlacementResult.success(selectedRole);
    }

    private void recordDecision(
            RequestData context,
            RoleType role,
            String group,
            int totalWorkers,
            PrefillCandidateSet candidates,
            int selectedIndex,
            String reason,
            Map<String, Integer> rejections) {
        List<RoutingDecision.Candidate> snapshot = new ArrayList<>();
        if (selectedIndex >= 0) {
            snapshot.add(snapshotCandidate(candidates, selectedIndex, true));
        }
        for (int index = 0; index < candidates.size() && snapshot.size() < 5; index++) {
            if (index != selectedIndex) {
                snapshot.add(snapshotCandidate(candidates, index, false));
            }
        }
        context.recordRoutingDecision(
                new RoutingDecision(
                        role,
                        group,
                        "CostBasedPrefill",
                        reason == null ? "COST_BASED" : reason,
                        System.currentTimeMillis(),
                        context.routingAttempt(role),
                        selectedIndex < 0 ? null : candidates.endpointAddress(selectedIndex),
                        totalWorkers,
                        candidates.size(),
                        candidates.size() > snapshot.size(),
                        rejections,
                        snapshot,
                        null));
    }

    private static RoutingDecision.Candidate snapshotCandidate(
            PrefillCandidateSet candidates, int index, boolean selected) {
        long projectedTtftMs = candidates.projectedTtftMs(index);
        return new RoutingDecision.Candidate(
                candidates.endpointAddress(index),
                selected,
                projectedTtftMs < 0L ? null : projectedTtftMs,
                null,
                candidates.prefillMs(index),
                candidates.cacheHit(index),
                candidates.routingCacheMatchTokens(index),
                0L,
                null,
                null,
                null,
                projectedTtftMs < 0L ? "UNMODELED_ENGINE_WORK" : "MODELED",
                candidates.ownershipVersion(index));
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

    /**
     * Select from candidates that already passed the common hard filters.
     */
    int selectBestCandidate(
            PrefillCandidateSet survivors,
            long minProjectedTtftMs,
            RoleType roleType,
            String group,
            long seqLen,
            FlexlbConfig config) {
        if (survivors.size() == 0) {
            return -1;
        }

        RoutingConfig.CacheAffinityConfig cacheAffinity =
                config.getRouter().getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null && survivors.hasKnownTtft) {
            long referenceHitTokens = 0L;
            long maxHitTokens = survivors.maximumCacheHit;
            affinityCutoffMs =
                    saturatingAdd(
                            minProjectedTtftMs, Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.ttftRank(i) == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(referenceHitTokens, survivors.cacheHit(i));
                }
            }
            if (maxHitTokens > referenceHitTokens) {
                boolean minimumHitRateMet = false;
                double minimumHitRate = normalizedHitRate(cacheAffinity.getMinPrefixHitPercent());
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
                        preferredCandidates.set(i);
                    }
                }
                affinityReason =
                        !preferredCandidates.isEmpty()
                                ? "CACHE_LEADER"
                                : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        int selectedIndex;
        if (!preferredCandidates.isEmpty()) {
            selectedIndex = selectCacheLeader(survivors, preferredCandidates, roleType, group);
        } else {
            selectedIndex = selectBaselineCandidate(survivors, minProjectedTtftMs, roleType, group);
        }

        if (selectedIndex >= 0 && affinityReason != null) {
            reportCacheAffinityDecision(
                    roleType,
                    survivors.endpoint(selectedIndex).getStatus().getMetricIpPort(),
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
        if (Double.isNaN(configuredRate) || configuredRate == Double.POSITIVE_INFINITY) {
            return RoutingConfig.PERCENTAGE_SCALE;
        }
        if (configuredRate == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        return Math.min(RoutingConfig.PERCENTAGE_SCALE, Math.max(0.0, configuredRate));
    }

    private int selectBaselineCandidate(
            PrefillCandidateSet survivors, long minimumTtftMs, RoleType role, String group) {
        return tieRotation.next(
                role,
                group,
                survivors.size(),
                i ->
                        survivors.ttftRank(i) == minimumTtftMs
                                && (!survivors.hasKnownTtft || survivors.projectedTtftMs(i) >= 0L),
                survivors::endpointAddress);
    }

    /**
     * Fairly rotate only endpoints with the same best cache hit and projected TTFT.
     */
    private int selectCacheLeader(
            PrefillCandidateSet survivors,
            BitSet preferredCandidates,
            RoleType role,
            String group) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        int tiedCount = 0;
        for (int candidateIndex = 0; candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)) {
                continue;
            }
            long hit = survivors.cacheHit(candidateIndex);
            long projectedTtftMs = survivors.ttftRank(candidateIndex);
            if (hit > bestHit || hit == bestHit && projectedTtftMs < bestProjectedTtftMs) {
                bestHit = hit;
                bestProjectedTtftMs = projectedTtftMs;
                tiedCount = 1;
            } else if (hit == bestHit && projectedTtftMs == bestProjectedTtftMs) {
                tiedCount++;
            }
        }
        if (tiedCount == 0) {
            return -1;
        }
        long selectedHit = bestHit;
        long selectedTtft = bestProjectedTtftMs;
        return tieRotation.next(
                role,
                group,
                survivors.size(),
                i ->
                        contains(preferredCandidates, i)
                                && survivors.cacheHit(i) == selectedHit
                                && survivors.ttftRank(i) == selectedTtft,
                survivors::endpointAddress);
    }

    private static boolean contains(BitSet candidates, int index) {
        return candidates.get(index);
    }

    private record EndpointDiscovery(
            List<EndpointRegistry.PrefillRoutingEntry> directory,
            List<EndpointRegistry.PrefillRoutingEntry> candidates,
            int registeredCount) {

        private EndpointDiscovery {
            candidates = java.util.Objects.requireNonNull(candidates, "candidates");
        }

        /**
         * Zero-copy address view over the exact fleet used by this decision.
         */
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
            RequestData balanceContext,
            CacheMatchResult cacheMatchResult,
            Map<String, Integer> rejections,
            Map<RoleType, Integer> poolWideBlockers,
            PrefillCandidateSet candidates) {
        Request request = balanceContext.getRequest();

        int eligibleSize = discovery.candidates().size();
        candidates.reset(eligibleSize);
        long planningAtMs = System.currentTimeMillis();
        var projectionSession = RouteTimelineProjector.current();

        // Use one endpoint snapshot for both service prediction and decision-group planning.
        for (int i = 0; i < discovery.candidates().size(); i++) {
            EndpointRegistry.PrefillRoutingEntry routingEntry = discovery.candidates().get(i);
            PrefillEndpoint ep = routingEntry.endpoint();
            String endpointAddress = routingEntry.address();
            CacheTokenMatch cacheMatch =
                    calculateCacheMatch(ep, cacheMatchResult, request, balanceContext.getConfig());
            long cacheHit = cacheMatch.effectiveHitTokens();
            long routingCacheMatchTokens = cacheMatch.routingHitTokens();
            RequestTimeEstimation.Inputs projectionInputs = ep.captureRouteProjectionInputs();
            PrefillTimePredictor predictor = ep.predictor;
            RequestTimeEstimation.CandidateView projection =
                    projectionSession.projectRankingView(
                            projectionInputs,
                            request.getRequestId(),
                            balanceContext.getPriority(),
                            planningAtMs,
                            // RequestRepository owns terminal deadlines.
                            // Selection scores endpoint work only; an expiry race
                            // must not be reported as a capacity blocker.
                            Long.MAX_VALUE,
                            request.getSeqLen(),
                            cacheHit,
                            routingCacheMatchTokens,
                            predictor == null ? null : predictor.evaluator(),
                            ep.deliveryProjection,
                            planningAtMs);
            if (!projection.selectable() && !projection.engineWorkUnmodeled()) {
                RoleType blockerRole = projection.blockerRole();
                if (blockerRole != null) {
                    poolWideBlockers.merge(blockerRole, 1, Integer::sum);
                }
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
                        "Prefill projection - ip: "
                                + endpointAddress
                                + ", order: "
                                + (balanceContext.getConfig().isPriorityOrdering()
                                        ? "PRIORITY"
                                        : "FIFO")
                                + ", hitCache: "
                                + cacheHit
                                + ", ttftMs: "
                                + projection.projectedTtftMsValue());
            }
        }

        return candidates;
    }

    static RoleType provenPoolWideBlocker(
            Map<RoleType, Integer> blockers, int registeredEndpoints) {
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
            RequestData context, RoleType roleType, String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                workerDirectory.prefillRoutingSnapshot(roleType);
        if (group == null) {
            return new EndpointDiscovery(directory, directory, directory.size());
        }
        List<EndpointRegistry.PrefillRoutingEntry> matching = List.of();
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            if (!group.equals(entry.endpoint().getStatus().topologySnapshot().group())) {
                continue;
            }
            if (matching.isEmpty()) {
                matching = new ArrayList<>();
            }
            matching.add(entry);
        }
        return new EndpointDiscovery(directory, matching, matching.size());
    }

    private CacheMatchResult getCacheMatchResult(
            RequestData balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long blockSize =
                request.getBlockSize() > 0L
                        ? request.getBlockSize()
                        : request.getCacheKeyBlockSize();
        CacheMatchResult result =
                cacheAwareService.findMatchingEngines(
                        new CacheMatchQuery(
                                balanceContext.getRequestId(),
                                request.getBlockCacheKeys(),
                                blockSize,
                                roleType,
                                group));
        CacheMatchResult observed =
                result == null ? CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC) : result;
        balanceContext.recordCacheQuery(observed.source().name(), observed.queryTimeUs());
        return observed;
    }

    private record CacheTokenMatch(long effectiveHitTokens, long routingHitTokens) {
        private static final CacheTokenMatch NONE = new CacheTokenMatch(0L, 0L);
    }

    private CacheTokenMatch calculateCacheMatch(
            PrefillEndpoint ep,
            CacheMatchResult cacheMatchResult,
            Request request,
            FlexlbConfig config) {
        if (cacheMatchResult == null || request == null) {
            return CacheTokenMatch.NONE;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return CacheTokenMatch.NONE;
        }
        HostCacheMatch match = cacheMatchResult.hostMatch(ep.getStatus());
        if (match == null) {
            return CacheTokenMatch.NONE;
        }
        long localMatchBlocks = Math.max(0L, match.localMatchBlocks());
        long remoteMatchBlocks = Math.max(0L, match.globalMatchBlocks() - localMatchBlocks);
        RoutingConfig.CacheAffinityConfig affinity =
                config.getRouter().getRoles().getPrefill().getCacheAffinity();
        double remoteDiscount =
                affinity == null ? 0.2 : Math.max(0.0, affinity.getRemoteDiscount());
        double effectiveMatchBlocks = localMatchBlocks + remoteMatchBlocks * remoteDiscount;
        if (effectiveMatchBlocks <= 0.0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = cacheMatchResult.blockSize();
        if (blockSize <= 0L) {
            blockSize =
                    request.getBlockSize() > 0L
                            ? request.getBlockSize()
                            : request.getCacheKeyBlockSize();
        }
        WorkerStatus status = ep.getStatus();
        CacheStatus cacheStatus = status == null ? null : status.getCacheStatus();
        if (blockSize <= 0L && cacheStatus != null) {
            blockSize = cacheStatus.getBlockSize();
        }
        if (blockSize <= 0L) {
            return CacheTokenMatch.NONE;
        }
        double rawMatchTokens = blockSize * effectiveMatchBlocks;
        long rawHit =
                rawMatchTokens >= Long.MAX_VALUE
                        ? Long.MAX_VALUE
                        : Math.max(0L, Math.round(rawMatchTokens));
        long routingHit = Math.min(seqLen, Math.max(0L, rawHit));
        long effectiveHit = rawHit >= seqLen ? Math.max(0L, seqLen - blockSize) : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportCacheHitMetrics(
            RoleType roleType, String ipIndex, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(
                roleType, ipIndex, hitCacheTokens, seqLen, hitRate);
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
                    endpoint.getStatus().getMetricIpPort(),
                    deliveryMode,
                    projectedTtftMs,
                    executionTimeMs);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    endpoint.ipPort(),
                    deliveryMode,
                    telemetryFailure);
        }
    }

    private void reportRoutingCacheMatchMetrics(
            RoleType roleType,
            long selectedHitTokens,
            long candidateMaxHitTokens,
            long totalTokens) {
        engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(roleType, selectedHitTokens);
        engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, candidateMaxHitTokens);
    }

    private ServerStatus buildSelectedRole(
            PrefillEndpoint ep,
            RoleType roleType,
            String requestId,
            long projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin,
            RequestRoute route) {
        try {
            // Populate DebugInfo so RequestRoute.hitCache() can read
            // hitCacheLen for batch metrics.
            DebugInfo debugInfo = new DebugInfo();
            debugInfo.setHitCacheLen(bestCacheHit);

            if (selectedPin == null || selectedPin.endpoint() != ep) {
                throw new IllegalStateException(
                        "selected Prefill endpoint generation changed before handoff");
            }
            WorkerStatus workerStatus = ep.getStatus();
            WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
            WorkerStatus.EngineObservation status = workerStatus.committedEngineObservation();
            ServerStatus result = new ServerStatus();
            result.setRole(roleType);
            result.setRequestId(requestId);
            if (projectedTtftMs >= 0L) {
                result.setPrefillTime(projectedTtftMs);
            }
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setDebugInfo(debugInfo);
            result.setSuccess(true);
            result.setSelectedEngineIndex(topology.engineIndex(), topology.multiEngineNum());
            WorkerEndpoint.GenerationPin ownedPin = selectedPin;
            selectedPin = null;
            return route.add(ownedPin, result, selectedPrefillMs, placementVersion);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

    private void reportCacheAffinityDecision(RoleType roleType, String engineIp, String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }
}
