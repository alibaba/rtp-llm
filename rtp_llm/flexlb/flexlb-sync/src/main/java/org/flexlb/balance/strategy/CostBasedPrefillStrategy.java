package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
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
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;

@Component
public class CostBasedPrefillStrategy {

    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();

    public CostBasedPrefillStrategy(WorkerDirectory workerDirectory,
                                    CacheAwareService cacheAwareService,
                                    EngineHealthReporter engineHealthReporter) {
        this.workerDirectory = workerDirectory;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
    }

    public PlacementResult<SelectedRole, PlacementKey> select(
            BalanceContext balanceContext,
            RoleType roleType,
            String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        // Cache matching sees every live Prefill, including busy or blocked workers.
        EndpointDiscovery discovery = discoverEndpoints(roleType, group);
        Map<String, Integer> cacheMatchResults = discovery.directory().isEmpty() ? Map.of()
                : getCacheMatchResults(balanceContext, roleType, discovery);
        if (discovery.candidates().isEmpty()) {
            return blockedWithDiagnostics(Response.error(roleType.getErrorType()),
                    new PlacementKey(roleType, group), 0, Map.of());
        }
        Map<String, Integer> rejections = new java.util.HashMap<>();
        PrefillCandidateSet survivors = evaluateCandidates(
                discovery, balanceContext, cacheMatchResults, rejections);
        if (survivors.size() == 0) {
            return blockedWithDiagnostics(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                    new PlacementKey(roleType, group), discovery.candidates().size(), rejections);
        }

        // Rank known work and cache hits without excluding admission-blocked candidates.
        int selectedIndex = selectBestCandidate(
                survivors, survivors.minimumProjectedTtftMs,
                roleType, group, seqLen, config);

        if (selectedIndex < 0) {
            Logger.debug(
                    "Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId,
                    rejections);
            return PlacementResult.blocked(new PlacementKey(roleType, group));
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        long selectedPrefillMs = survivors.prefillMs(selectedIndex);
        // Check only the winner: temporary pressure must not send B to a colder Prefill.
        String admissionBlock = survivors.admissionBlock(selectedIndex);
        boolean directCapacityFull = admissionBlock == null
                && config.isDirect() && !best.canAcceptRequest();
        WorkerEndpoint.GenerationPin selectedPin =
                workerDirectory.captureEndpoint(
                        roleType,
                        survivors.endpointAddress(selectedIndex));
        if (selectedPin == null || selectedPin.endpoint() != best) {
            if (selectedPin != null) {
                selectedPin.close();
            }
            // The planning endpoint retired or was replaced after the
            // full-fleet snapshot. Re-enter the queue rather than
            // transferring ownership for a stale generation.
            return PlacementResult.blocked(new PlacementKey(roleType, group));
        }
        if (admissionBlock != null || directCapacityFull) {
            try (selectedPin) {
                Response failure = admissionBlock == null
                        ? admissionFailure(best, balanceContext.getPriority())
                        : Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                // The selected P owns queue progress, including head removal and Decode wakeups.
                return PlacementResult.blocked(PlacementKey.exact(roleType,
                        best.getStatus().topologySnapshot().group(), survivors.endpointAddress(selectedIndex)), failure, Map.of(
                        "role", roleType.name(), "workers", discovery.candidates().size(),
                        "observedAtMs", System.currentTimeMillis(),
                        "selectedEndpoint", survivors.endpointAddress(selectedIndex),
                        "cacheHitTokens", bestCacheHit,
                        "projectedTtftMs", RouteProjection.Candidate.UNKNOWN,
                        "reason", admissionBlock == null ? "PREFILL_CAPACITY" : admissionBlock));
            }
        }
        long selectedTtft = survivors.projectedTtftMs(selectedIndex);
        SelectedRole selectedRole = buildSelectedRole(
                best,
                roleType,
                requestId,
                selectedTtft,
                selectedPrefillMs,
                bestCacheHit,
                survivors.ownershipVersion(selectedIndex),
                selectedPin);
        try {
            reportSelectedEstimates(
                    roleType,
                    best,
                    config,
                    selectedTtft,
                    selectedPrefillMs);
            reportCacheHitMetrics(roleType, bestCacheHit, seqLen);
            reportRoutingCacheMatchMetrics(
                    roleType,
                    survivors.routingCacheMatchTokens(selectedIndex),
                    survivors.maximumRoutingCacheMatchTokens,
                    seqLen);
            return PlacementResult.success(selectedRole);
        } catch (RuntimeException | Error failure) {
            try {
                selectedRole.close();
            } catch (RuntimeException | Error cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    private static Response admissionFailure(PrefillEndpoint endpoint, int priority) {
        AdmissionRejectReason reason = endpoint.admissionRejectReason(priority);
        return switch (reason) {
            case HIGHER_PRIORITY_AHEAD, SAME_PRIORITY_AHEAD ->
                    Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason);
            case RESOURCE_EXHAUSTED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
            case UNSPECIFIED -> Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        };
    }

    private static PlacementResult<SelectedRole, PlacementKey> blockedWithDiagnostics(
            Response failure, PlacementKey key, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(key, failure, Map.of("role", key.role().name(), "workers", workerCount,
                "observedAtMs", System.currentTimeMillis(), "details", Map.copyOf(details)));
    }

    /** Rank modeled work and cache hits before checking the selected endpoint admission. */
    int selectBestCandidate(PrefillCandidateSet survivors,
                            long minProjectedTtftMs,
                            RoleType roleType,
                            String group,
                            long seqLen,
                            FlexlbConfig config) {
        if (survivors.size() == 0) {
            return -1;
        }

        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter()
                .getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null) {
            long referenceHitTokens = 0L;
            long maxHitTokens = survivors.maximumCacheHit;
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.projectedTtftMs(i)
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
                    if (survivors.projectedTtftMs(i) <= affinityCutoffMs) {
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
                    survivors, minProjectedTtftMs, roleType, group,
                    cacheAffinity != null);
        }

        if (selectedIndex >= 0 && affinityReason != null) {
            reportCacheAffinityDecision(
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
                        survivors.projectedTtftMs(selectedIndex),
                        affinityCutoffMs,
                        survivors.cacheHit(selectedIndex),
                        affinityReason);
            }
        }
        return selectedIndex;
    }

    private static double normalizedHitRate(double configuredRate) {
        if (Double.isNaN(configuredRate)
                || configuredRate == Double.POSITIVE_INFINITY) {
            return RoutingConfig.PERCENTAGE_SCALE;
        }
        if (configuredRate == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        return Math.min(
                RoutingConfig.PERCENTAGE_SCALE,
                Math.max(0.0, configuredRate));
    }

    /** Lowest TTFT first; when affinity is enabled, highest cache hit breaks TTFT ties. */
    private int selectBaselineCandidate(PrefillCandidateSet survivors, long minimumTtftMs,
                                        RoleType role, String group, boolean preferCache) {
        long bestCacheHit = 0L;
        if (preferCache) {
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.projectedTtftMs(i) == minimumTtftMs) {
                    bestCacheHit = Math.max(bestCacheHit, survivors.cacheHit(i));
                }
            }
        }
        long selectedCacheHit = bestCacheHit;
        return tieRotation.next(role, group, survivors.size(),
                i -> survivors.projectedTtftMs(i) == minimumTtftMs
                        && (!preferCache || survivors.cacheHit(i) == selectedCacheHit),
                survivors::endpointAddress);
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            PrefillCandidateSet survivors, BitSet preferredCandidates, RoleType role, String group) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        int tiedCount = 0;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)) {
                continue;
            }
            long hit = survivors.cacheHit(candidateIndex);
            long projectedTtftMs = survivors.projectedTtftMs(candidateIndex);
            if (hit > bestHit
                    || hit == bestHit && projectedTtftMs < bestProjectedTtftMs) {
                bestHit = hit;
                bestProjectedTtftMs = projectedTtftMs;
                tiedCount = 1;
            } else if (hit == bestHit
                    && projectedTtftMs == bestProjectedTtftMs) {
                tiedCount++;
            }
        }
        if (tiedCount == 0) {
            return -1;
        }
        long selectedHit = bestHit;
        long selectedTtft = bestProjectedTtftMs;
        return tieRotation.next(role, group, survivors.size(),
                i -> contains(preferredCandidates, i) && survivors.cacheHit(i) == selectedHit
                        && survivors.projectedTtftMs(i) == selectedTtft,
                survivors::endpointAddress);
    }

    private static boolean contains(BitSet candidates, int index) {
        return candidates.get(index);
    }

    private record EndpointDiscovery(
            List<EndpointRegistry.PrefillRoutingEntry> directory,
            List<EndpointRegistry.PrefillRoutingEntry> candidates) {

        private EndpointDiscovery {
            candidates = java.util.Objects.requireNonNull(
                    candidates, "candidates");
        }

        /** All live endpoints participate in cache matching, including other routing groups. */
        private List<String> addresses() {
            return new AbstractList<>() {
                @Override
                public String get(int index) {
                    return directory.get(index).address();
                }

                @Override
                public int size() {
                    return directory.size();
                }
            };
        }
    }

    private PrefillCandidateSet evaluateCandidates(
            EndpointDiscovery discovery,
            BalanceContext balanceContext,
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections) {
        Request request = balanceContext.getRequest();
        int eligibleSize = discovery.candidates().size();
        PrefillCandidateSet candidates = CANDIDATES.get();
        candidates.reset(eligibleSize);
        long planningAtMs = System.currentTimeMillis();
        long enqueueSeq = balanceContext.getWorkerEnqueueSequence();
        long enqueuedAtMs = enqueueSeq == 0L
                ? planningAtMs : balanceContext.getFirstWorkerEnqueueTime();
        if (enqueueSeq == 0L) {
            enqueueSeq = Long.MAX_VALUE;
        }
        RouteProjection.Session projectionSession = RouteProjection.session();
        GroupPlanner.Item admissionProbe = null;

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
                    projectionSession.projectView(
                            projectionInputs,
                            request.getRequestId(),
                            balanceContext.getPriority(),
                            enqueueSeq,
                            enqueuedAtMs,
                            // RequestRegistry owns terminal deadlines.
                            // Selection scores endpoint work only; an expiry race
                            // must not be reported as a capacity blocker.
                            Long.MAX_VALUE,
                            request.getSeqLen(),
                            cacheHit,
                            routingCacheMatchTokens,
                            predictor == null
                                    ? null : predictor.evaluator(),
                            ep.deliveryProjection(),
                            planningAtMs);
            if (!projection.hasPrediction()) {
                rejections.merge(
                        "PROJECTION_" + projection.detail(),
                        1,
                        Integer::sum);
                continue;
            }

            String admissionBlock = null;
            if (projectionInputs.queue().admissionBlock() != null) {
                // Ordering is cache-independent. Reuse one probe across blocked endpoints.
                if (admissionProbe == null) {
                    admissionProbe = new GroupPlanner.Item(request.getRequestId(),
                            balanceContext.getPriority(), enqueueSeq, enqueuedAtMs,
                            Long.MAX_VALUE, request.getSeqLen(), 0L);
                }
                admissionBlock = RouteProjection.admissionBlockReason(
                        projectionInputs.queue(), projection, admissionProbe, planningAtMs);
            }
            candidates.addCandidate(endpointAddress, ep, projection, admissionBlock,
                    projectionInputs.ownershipVersion());
            if (Logger.isTraceEnabled()) {
                Logger.trace("Prefill ranking - ip: {}, order: {}, hitCache: {}, modeledWorkMs: {}",
                        endpointAddress, balanceContext.getConfig().isPriorityOrdering() ? "PRIORITY" : "FIFO",
                        cacheHit, projection.projectedTtftMsValue());
            }
        }

        return candidates;
    }

    private EndpointDiscovery discoverEndpoints(
            RoleType roleType,
            String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                workerDirectory.prefillRoutingSnapshot(roleType);
        List<EndpointRegistry.PrefillRoutingEntry> matching = new ArrayList<>();
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            matching.add(entry);
        }
        return new EndpointDiscovery(directory, matching);
    }

    private Map<String, Integer> getCacheMatchResults(
            BalanceContext balanceContext,
            RoleType roleType,
            EndpointDiscovery discovery) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(
                blockCacheKeys, roleType, discovery.addresses());
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
            Request request) {
        if (cacheMatchResults == null || cacheMatchResults.isEmpty() || request == null) {
            return CacheTokenMatch.NONE;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return CacheTokenMatch.NONE;
        }
        Integer prefixMatchLength = cacheMatchResults.get(endpointAddress);
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = request.getCacheKeyBlockSize();
        WorkerStatus status = ep.getStatus();
        CacheStatus cacheStatus = status == null ? null : status.getCacheStatus();
        if (blockSize <= 0L && cacheStatus != null) {
            blockSize = cacheStatus.getBlockSize();
        }
        if (blockSize <= 0L) {
            return CacheTokenMatch.NONE;
        }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(
                    blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            rawHit = seqLen;
        }
        long routingHit = Math.min(seqLen, Math.max(0L, rawHit));
        long effectiveHit = rawHit >= seqLen
                ? Math.max(0L, seqLen - blockSize)
                : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportCacheHitMetrics(RoleType roleType, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, hitCacheTokens, hitRate);
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

    private void reportRoutingCacheMatchMetrics(
            RoleType roleType,
            long selectedHitTokens,
            long candidateMaxHitTokens,
            long totalTokens) {
        engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(
                roleType, selectedHitTokens, totalTokens);
        engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, candidateMaxHitTokens);
    }

    private SelectedRole buildSelectedRole(
            PrefillEndpoint ep,
            RoleType roleType,
            long requestId,
            long projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin) {
        try {
            // Populate DebugInfo so ScheduledRequest.hitCache() can read
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
            result.setPrefillTime(projectedTtftMs);
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setDebugInfo(debugInfo);
            result.setSuccess(true);
            WorkerEndpoint.GenerationPin ownedPin = selectedPin;
            selectedPin = null;
            return SelectedRole.prefill(
                    ownedPin, result, selectedPrefillMs, placementVersion);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

    private void reportCacheAffinityDecision(
            RoleType roleType, String engineIp, String decision) {
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
