package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
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
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashMap;
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

    public PlacementResult<SelectedRole, RoleType> select(BalanceContext balanceContext,
                                                          RoleType roleType,
                                                          String group) {
        balanceContext.beginRoutingAttempt(roleType);
        String requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();
        long inflightRequestLimit = config.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH
                ? config.getDispatcher().getMaxInflightPerPrefillWorker() : 0L;

        EndpointDiscovery discovery = discoverAvailableEndpoints(roleType, group);
        if (discovery.candidates().isEmpty()) {
            Logger.debug("Prefill select failed: no registered endpoint matches group, request_id={}",
                    requestId);
            balanceContext.recordSelectionReason(roleType, "NO_MATCHING_ENDPOINTS");
            return classifyAdmissionFailure(discovery, balanceContext.getPriority(), roleType, group, inflightRequestLimit);
        }

        CacheMatchResult cacheMatchResult = getCacheMatchResult(balanceContext, roleType, group);

        PrefillCandidateSet survivors = evaluateCandidates(discovery, balanceContext, cacheMatchResult);

        int selectedIndex = selectBestCandidate(survivors, survivors.minimumTtftRank, roleType, group, seqLen, config);

        if ("NO_CACHE_ANYWHERE".equals(survivors.affinityDecision())) {
            if (seqLen < cacheMatchResult.blockSize()) {
                survivors.recordAffinityDecision("NO_CACHE_ANYWHERE_BELOW_BLOCK_SIZE");
            } else if (cacheMatchResult.source() == CacheMatchSource.KVCM && cacheMatchResult.querySucceeded()) {
                boolean kvcmHasHit = cacheMatchResult.hostMatches().values().stream()
                        .anyMatch(match -> match.globalMatchBlocks() > 0L);
                survivors.recordAffinityDecision(kvcmHasHit
                        ? "NO_CACHE_ANYWHERE_BUT_KVCM_HAS_HIT"
                        : "NO_CACHE_ANYWHERE_KVCM");
            }
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        long selectedPrefillMs = survivors.prefillMs(selectedIndex);
        WorkerEndpoint.GenerationPin selectedPin =
                workerDirectory.captureEndpoint(
                        roleType,
                        survivors.endpointAddress(selectedIndex));
        if (selectedPin == null || selectedPin.endpoint() != best) {
            recordDecision(balanceContext, roleType, group, discovery.registeredCount(), survivors,
                    -1, "ENDPOINT_GENERATION_CHANGED", cacheMatchResult);
            if (selectedPin != null) {
                selectedPin.close();
            }
            // The planning endpoint retired or was replaced after the
            // full-fleet snapshot. Re-enter the queue rather than
            // transferring ownership for a stale generation.
            return PlacementResult.blocked(roleType);
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
        if (balanceContext.selectionReason(roleType) == null) {
            balanceContext.recordSelectionReason(roleType, "COST_BASED");
        }
        cacheAwareService.trackRoutingPrediction(
                requestId, roleType, group, best.getStatus(),
                seqLen, bestCacheHit, cacheMatchResult);
        balanceContext.recordCacheSelection(roleType, best.getIp(), bestCacheHit);
        recordDecision(balanceContext, roleType, group, discovery.registeredCount(), survivors,
                selectedIndex, balanceContext.selectionReason(roleType), cacheMatchResult);
        if (selectedTtft >= 0L) { reportSelectedEstimates(
                roleType,
                best,
                config,
                selectedTtft,
                selectedPrefillMs); }
        return PlacementResult.success(selectedRole.onCommit(commitTelemetry(
                roleType, best, cacheMatchResult, bestCacheHit, seqLen,
                survivors.routingCacheMatchTokens(selectedIndex),
                survivors.maximumRoutingCacheMatchTokens,
                survivors.affinityDecision(), survivors.cacheAffinity())));
    }

    /**
     * Cache counters for the selected route, captured now and reported once the route commits.
     */
    private Runnable commitTelemetry(
            RoleType roleType,
            PrefillEndpoint best,
            CacheMatchResult cacheMatchResult,
            long bestCacheHit,
            long seqLen,
            long selectedRoutingHitTokens,
            long candidateMaxRoutingHitTokens,
            String affinityDecision,
            RoutingDecision.CacheAffinity cacheAffinity) {
        String metricIpPort = best.getStatus().getMetricIpPort();
        HostCacheMatch selectedMatch = cacheMatchResult.hostMatch(best.getStatus());
        boolean kvcmQueried = cacheMatchResult.source() == CacheMatchSource.KVCM
                && cacheMatchResult.querySucceeded();
        long localMatchTokens = selectedMatch == null ? 0 : CacheMatchResult.matchedTokens(
                selectedMatch.localMatchBlocks(), cacheMatchResult.blockSize(), seqLen);
        long globalMatchTokens = selectedMatch == null ? 0 : CacheMatchResult.matchedTokens(
                selectedMatch.globalMatchBlocks(), cacheMatchResult.blockSize(), seqLen);
        return () -> {
            if (kvcmQueried) {
                engineHealthReporter.reportKvcmSelectedMatch(
                        roleType, metricIpPort, localMatchTokens, globalMatchTokens, seqLen);
            }
            reportCacheHitMetrics(roleType, metricIpPort, bestCacheHit, seqLen);
            reportRoutingCacheMatchMetrics(
                    roleType, selectedRoutingHitTokens, candidateMaxRoutingHitTokens, seqLen);
            if (affinityDecision != null) {
                reportCacheAffinityDecision(roleType, metricIpPort, affinityDecision);
            }
            if ("OVER_CAP".equals(affinityDecision) && cacheAffinity.overCapMs() != null) {
                engineHealthReporter.reportCacheAffinityOverCap(
                        roleType, metricIpPort, cacheAffinity.overCapMs());
            }
        };
    }

    private void recordDecision(BalanceContext context,
                                RoleType role,
                                String group,
                                int totalWorkers,
                                PrefillCandidateSet candidates,
                                int selectedIndex,
                                String reason,
                                CacheMatchResult cacheMatchResult) {
        List<RoutingDecision.Candidate> snapshot = new ArrayList<>();
        if (selectedIndex >= 0) {
            snapshot.add(snapshotCandidate(candidates, selectedIndex, true));
        }
        for (int index = 0; index < candidates.size() && snapshot.size() < 5; index++) {
            if (index != selectedIndex) {
                snapshot.add(snapshotCandidate(candidates, index, false));
            }
        }
        context.recordRoutingDecision(new RoutingDecision(
                role,
                group,
                "CostBasedPrefill",
                reason == null ? "COST_BASED" : reason,
                selectedIndex < 0 ? null : candidates.affinityDecision(),
                candidates.cacheAffinity(),
                kvcmMatches(cacheMatchResult, context.getRequest().getSeqLen(),
                        selectedIndex < 0 ? null : candidates.endpoint(selectedIndex).getStatus()),
                System.currentTimeMillis(),
                context.routingAttempt(role),
                selectedIndex < 0 ? null : candidates.endpointAddress(selectedIndex),
                totalWorkers,
                candidates.size(),
                candidates.size() > snapshot.size(),
                Map.of(),
                snapshot,
                null));
    }

    private static RoutingDecision.KvcmMatches kvcmMatches(CacheMatchResult result,
                                                           long inputTokens,
                                                           WorkerStatus selectedWorker) {
        if (result.source() != CacheMatchSource.KVCM || !result.querySucceeded()) {
            return null;
        }
        List<Map.Entry<String, Long>> largestHits = new ArrayList<>(3);
        for (Map.Entry<String, HostCacheMatch> match : result.hostMatches().entrySet()) {
            long hitTokens = CacheMatchResult.matchedTokens(
                    match.getValue().globalMatchBlocks(), result.blockSize(), inputTokens);
            if (hitTokens == 0L) {
                continue;
            }
            int position = 0;
            while (position < largestHits.size()
                    && (largestHits.get(position).getValue() > hitTokens
                        || largestHits.get(position).getValue() == hitTokens
                            && largestHits.get(position).getKey().compareTo(match.getKey()) < 0)) {
                position++;
            }
            if (position < 3) {
                largestHits.add(position, Map.entry(match.getKey(), hitTokens));
                if (largestHits.size() > 3) {
                    largestHits.remove(3);
                }
            }
        }
        Map<String, Long> topHits = new LinkedHashMap<>();
        for (Map.Entry<String, Long> hit : largestHits) {
            topHits.put(hit.getKey(), hit.getValue());
        }
        HostCacheMatch selectedMatch = selectedWorker == null ? null : result.hostMatch(selectedWorker);
        Long selectedHitTokens = null;
        if (selectedWorker != null) {
            selectedHitTokens = selectedMatch == null ? 0L
                    : CacheMatchResult.matchedTokens(selectedMatch.globalMatchBlocks(), result.blockSize(), inputTokens);
        }
        return new RoutingDecision.KvcmMatches(result.hostMatches().size(), topHits, selectedHitTokens);
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

    private static PlacementResult<SelectedRole, RoleType> classifyAdmissionFailure(EndpointDiscovery discovery,
                                                                                    int priority,
                                                                                    RoleType role,
                                                                                    String group,
                                                                                    long inflightRequestLimit) {
        AdmissionRejectReason reason = null;
        int workerCount = 0;
        long observedRequestCount = 0;
        for (var entry : discovery.directory()) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            workerCount++;
            observedRequestCount += endpoint.observedRequestCount();
            AdmissionRejectReason workerReason = endpoint.admissionRejectReason(priority, inflightRequestLimit);
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
        return blockedWithDiagnostics(failure, role, workerCount, Map.of("observedRequests", observedRequestCount));
    }

    private static PlacementResult<SelectedRole, RoleType> blockedWithDiagnostics(
            Response failure, RoleType role, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(role, failure, Map.of("role", role.name(), "workers", workerCount,
                "observedAtMs", System.currentTimeMillis(), "details", Map.copyOf(details)));
    }

    /**
     * Select one Prefill endpoint from the registered target-group candidates.
     */
    int selectBestCandidate(PrefillCandidateSet survivors,
                            long minProjectedTtftMs,
                            RoleType roleType,
                            String group,
                            long seqLen,
                            FlexlbConfig config) {
        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter()
                .getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null && survivors.hasKnownTtft) {
            long referenceHitTokens = 0L;
            long maxHitTokens = survivors.maximumCacheHit;
            long maxExtraTtftMs = Math.max(0L, cacheAffinity.getMaxExtraTtftMs());
            double minimumHitRate = normalizedHitRate(cacheAffinity.getMinPrefixHitPercent());
            int fastestHigherHitIndex = -1;
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    maxExtraTtftMs);
            affinityReason = maxHitTokens > 0L ? "FASTEST_IS_CACHE_BEST" : "NO_CACHE_ANYWHERE";
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
                    if (survivors.projectedTtftMs(i) >= 0L
                            && (fastestHigherHitIndex < 0
                                || survivors.ttftRank(i) < survivors.ttftRank(fastestHigherHitIndex))) {
                        fastestHigherHitIndex = i;
                    }
                    if (survivors.projectedTtftMs(i) >= 0L && survivors.ttftRank(i) <= affinityCutoffMs) {
                        preferredCandidates.set(i);
                    }
                }
                affinityReason = !preferredCandidates.isEmpty()
                        ? "CACHE_LEADER"
                        : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
            Long fastestHigherHitTtftMs = fastestHigherHitIndex < 0
                    ? null : survivors.projectedTtftMs(fastestHigherHitIndex);
            survivors.recordCacheAffinity(new RoutingDecision.CacheAffinity(
                    minProjectedTtftMs, maxExtraTtftMs, minimumHitRate, affinityCutoffMs,
                    fastestHigherHitIndex < 0 ? null : survivors.endpointAddress(fastestHigherHitIndex),
                    fastestHigherHitTtftMs,
                    fastestHigherHitIndex < 0 ? null : survivors.cacheHit(fastestHigherHitIndex),
                    "OVER_CAP".equals(affinityReason) && fastestHigherHitTtftMs != null
                            ? fastestHigherHitTtftMs - affinityCutoffMs : null));
        }

        int selectedIndex;
        if (!preferredCandidates.isEmpty()) {
            selectedIndex = selectCacheLeader(survivors, preferredCandidates, roleType, group);
        } else {
            selectedIndex = selectBaselineCandidate(
                    survivors, minProjectedTtftMs, roleType, group);
        }

        if (affinityReason != null) {
            survivors.recordAffinityDecision(affinityReason);
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

    private int selectBaselineCandidate(PrefillCandidateSet survivors, long minimumTtftMs,
                                        RoleType role, String group) {
        return tieRotation.next(role, group, survivors.size(),
                i -> survivors.ttftRank(i) == minimumTtftMs
                        && (!survivors.hasKnownTtft || survivors.projectedTtftMs(i) >= 0L),
                survivors::endpointAddress);
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(PrefillCandidateSet survivors,
                                  BitSet preferredCandidates,
                                  RoleType role,
                                  String group) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)) {
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
                i -> contains(preferredCandidates, i) && survivors.cacheHit(i) == selectedHit
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

    private PrefillCandidateSet evaluateCandidates(EndpointDiscovery discovery,
                                                   BalanceContext balanceContext,
                                                   CacheMatchResult cacheMatchResult) {
        Request request = balanceContext.getRequest();
        int eligibleSize = discovery.candidates().size();
        PrefillCandidateSet candidates = CANDIDATES.get();
        candidates.reset(eligibleSize);
        long planningAtMs = System.currentTimeMillis();
        RouteProjection.Session projectionSession = RouteProjection.session();

        // Use one endpoint snapshot for both service prediction and decision-group planning.
        for (int i = 0; i < discovery.candidates().size(); i++) {
            EndpointRegistry.PrefillRoutingEntry routingEntry = discovery.candidates().get(i);

            PrefillEndpoint ep = routingEntry.endpoint();
            String endpointAddress = routingEntry.address();

            CacheTokenMatch cacheMatch = calculateCacheMatch(ep, cacheMatchResult, request, balanceContext.getConfig());
            long cacheHit = cacheMatch.effectiveHitTokens();
            long routingCacheMatchTokens = cacheMatch.routingHitTokens();
            RouteProjection.Inputs projectionInputs = ep.captureRouteProjectionInputs();
            RouteProjection.CandidateView projection =
                    projectionSession.projectViewForSelection(
                            projectionInputs,
                            request.getRequestId(),
                            balanceContext.getPriority(),
                            planningAtMs,
                            // RequestRegistry owns terminal deadlines.
                            // Selection scores endpoint work only; an expiry race
                            // must not be reported as a capacity blocker.
                            Long.MAX_VALUE,
                            request.getSeqLen(),
                            cacheHit,
                            routingCacheMatchTokens,
                            ep.getPredictor().evaluator(),
                            ep.deliveryProjection(),
                            planningAtMs);
            candidates.addCandidate(endpointAddress, ep, projection, projectionInputs.ownershipVersion());
            if (Logger.isTraceEnabled()) {
                Logger.trace("Prefill projection - ip: " + endpointAddress
                        + ", order: "
                        + (balanceContext.getConfig().isPriorityOrdering() ? "PRIORITY" : "FIFO")
                        + ", hitCache: " + cacheHit
                        + ", ttftMs: " + projection.projectedTtftMsValue());
            }
        }

        return candidates;
    }

    private EndpointDiscovery discoverAvailableEndpoints(RoleType roleType, String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                workerDirectory.prefillRoutingSnapshot(roleType);
        List<EndpointRegistry.PrefillRoutingEntry> matching = new ArrayList<>();
        int registered = 0;
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            registered++;
            matching.add(entry);
        }
        return new EndpointDiscovery(directory, matching, registered);
    }

    private CacheMatchResult getCacheMatchResult(
            BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long blockSize = request.getBlockSize() > 0L
                ? request.getBlockSize()
                : request.getCacheKeyBlockSize();
        CacheMatchResult result = cacheAwareService.findMatchingEngines(
                new CacheMatchQuery(
                        balanceContext.getRequestId(),
                        request.getBlockCacheKeys(),
                        blockSize,
                        roleType,
                        group));
        CacheMatchResult observed = result == null
                ? CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC)
                : result;
        balanceContext.recordCacheQuery(observed.source().name(), observed.queryTimeUs());
        return observed;
    }

    private record CacheTokenMatch(
            long effectiveHitTokens,
            long routingHitTokens) {
        private static final CacheTokenMatch NONE =
                new CacheTokenMatch(0L, 0L);
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
        long remoteMatchBlocks = Math.max(
                0L, match.globalMatchBlocks() - localMatchBlocks);
        RoutingConfig.CacheAffinityConfig affinity = config.getRouter()
                .getRoles().getPrefill().getCacheAffinity();
        double remoteDiscount = affinity == null
                ? 0.2
                : Math.max(0.0, affinity.getRemoteDiscount());
        double effectiveMatchBlocks = localMatchBlocks
                + remoteMatchBlocks * remoteDiscount;
        if (effectiveMatchBlocks <= 0.0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = cacheMatchResult.blockSize();
        if (blockSize <= 0L) {
            blockSize = request.getBlockSize() > 0L
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
        long rawHit = rawMatchTokens >= Long.MAX_VALUE
                ? Long.MAX_VALUE
                : Math.max(0L, Math.round(rawMatchTokens));
        long routingHit = Math.min(seqLen, Math.max(0L, rawHit));
        long effectiveHit = rawHit >= seqLen
                ? Math.max(0L, seqLen - blockSize)
                : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportCacheHitMetrics(
            RoleType roleType, String ipIndex, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ipIndex, hitCacheTokens, seqLen, hitRate);
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
                    endpoint.ipPort(), deliveryMode, telemetryFailure);
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

    private SelectedRole buildSelectedRole(
            PrefillEndpoint ep,
            RoleType roleType,
            String requestId,
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
            if (projectedTtftMs >= 0L) { result.setPrefillTime(projectedTtftMs); }
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setDebugInfo(debugInfo);
            result.setSuccess(true);
            result.setSelectedEngineIndex(
                    topology.engineIndex(), topology.multiEngineNum());
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
