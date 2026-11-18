package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.RouteTimelineProjector;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
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

import java.util.AbstractList;
import java.util.ArrayList;
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

    private final boolean priorityOrdering;
    private final boolean preemptQueued;
    private final RoutingConfig.CacheAffinityConfig cacheAffinity;
    private final double cacheAffinityMinHitRate;
    private final String deliveryMode;
    private final int maxInflightBatches;
    private final RoutingConfig.TokenBalanceConfig tokenBalance;
    private long lastRescueNanos = Long.MIN_VALUE;
    private final Object tokenSelectionLock = new Object();

    public CostBasedPrefillStrategy(EndpointRegistry endpointRegistry,
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
        preemptQueued = config.allowsPreemption(VictimStage.PREFILL_QUEUED);
        deliveryMode = config.getDispatcher().typeName();
        maxInflightBatches = config.getDispatcher().requiresGenerateInput() ? config.getDispatcher().getMaxInflightPerPrefillWorker() : 0;
        tokenBalance = config.getRouter().getRoles().getPrefill().getTokenBalance();
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        cacheAffinity = affinity;
        cacheAffinityMinHitRate = affinity == null ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent());
    }

    /** Payload-free token estimate for benefit ordering; no time predictor or capacity reservation. */
    public record TokenEstimate(long inputTokens, long hitTokens) {
        public TokenEstimate {
            if (inputTokens < 0 || hitTokens < 0 || hitTokens > inputTokens) {
                throw new IllegalArgumentException("invalid Prefill token estimate");
            }
        }
        public long computeTokens() { return inputTokens - hitTokens; }
    }

    public TokenEstimate estimateQueuedTokens(Request request, RoleType role, String group) {
        var fleet = endpointRegistry.prefillRoutingSnapshot(role).stream()
                .filter(entry -> group == null || group.equals(entry.endpoint().getStatus().topologySnapshot().group()))
                .toList();
        if (fleet.isEmpty()) { return null; }
        var matches = cacheAwareService.findMatchingEngines(request.getBlockCacheKeys(), role,
                fleet.stream().map(EndpointRegistry.PrefillRoutingEntry::address).toList());
        long hit = 0L;
        for (var entry : fleet) {
            hit = Math.max(hit, calculateCacheMatch(entry.endpoint(), entry.address(), matches, request).effectiveHitTokens());
        }
        if (tokenBalance != null && !PrefillTokenBalance.cacheFirst(
                hit, request.getSeqLen(), tokenBalance.getMinPrefixHitPercent())) { hit = 0L; }
        return new TokenEstimate(request.getSeqLen(), hit);
    }

    public PlacementResult<ServerStatus, RoleType> select(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group, RequestRoute route) {
        PrefillCandidateSet cached = CANDIDATES.get();
        PrefillCandidateSet candidates = cached.inUse ? new PrefillCandidateSet() : cached;
        candidates.inUse = true;
        try {
            return selectCandidate(request, metadata, roleType, group, route, candidates);
        } finally {
            candidates.reset(0);
            candidates.inUse = false;
        }
    }

    private PlacementResult<ServerStatus, RoleType> selectCandidate(
            Request request, SchedulingMetadata metadata, RoleType roleType,
            String group, RequestRoute route, PrefillCandidateSet candidates) {
        long requestId = request.getRequestId();
        long seqLen = request.getSeqLen();

        EndpointDiscovery discovery = discoverAvailableEndpoints(request, metadata, roleType, group);
        if (tokenBalance == null && discovery.candidates().isEmpty()) {
            return classifyAdmissionFailure(discovery.directory(), metadata.priority(), roleType, group);
        }
        Map<String, Integer> cacheMatchResults =
                cacheAwareService.findMatchingEngines(request.getBlockCacheKeys(), roleType, discovery.addresses());
        long fleetMaxHit = 0L;
        var cacheWorkers = new java.util.HashSet<String>();
        if (tokenBalance != null) {
            for (var entry : discovery.cacheCandidates()) {
                long hit = calculateCacheMatch(entry.endpoint(), entry.address(), cacheMatchResults, request).effectiveHitTokens();
                if (hit > fleetMaxHit) { cacheWorkers.clear(); fleetMaxHit = hit; }
                if (hit == fleetMaxHit) { cacheWorkers.add(entry.address()); }
            }
        }
        boolean hot = tokenBalance != null && PrefillTokenBalance.cacheFirst(
                fleetMaxHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        // Cache eligibility survives capacity filtering: a failure on one hot
        // worker is not evidence that requests cached elsewhere cannot run.
        route.setPrefillCacheWorkers(hot ? cacheWorkers : java.util.Set.of());
        if (discovery.candidates().isEmpty()) {
            return classifyAdmissionFailure(discovery.directory(), metadata.priority(), roleType, group);
        }
        Map<String, Integer> rejections = new java.util.HashMap<>();
        Map<RoleType, Integer> poolWideBlockers =
                new EnumMap<>(RoleType.class);
        PrefillCandidateSet survivors = evaluateCandidates(
                discovery, request, metadata,
                cacheMatchResults, rejections, poolWideBlockers, candidates, tokenBalance != null && !hot);
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

        boolean probe = hot && tokenBalance.getHotspotRescue() > 0 && route.ctx().isHotspotProbe();
        if (hot && survivors.maximumCacheHit < fleetMaxHit && !probe) {
            return PlacementResult.blocked(roleType);
        }
        route.setColdTokenBalance(tokenBalance != null && !hot);
        int selectedIndex;
        if (tokenBalance == null) {
            selectedIndex = selectBestCandidate(survivors, survivors.minimumTtftRank, roleType, group, seqLen);
        } else {
            // Parallel planners must see earlier choices before asynchronous admission commits.
            // Only scoring + soft reservation are serialized; cache lookup/projection stay parallel.
            synchronized (tokenSelectionLock) {
                selectedIndex = selectTokenBalancedCandidate(survivors, roleType, group, seqLen, fleetMaxHit);
                if (selectedIndex < 0 && probe) {
                    selectedIndex = selectRescueTarget(discovery, cacheWorkers, survivors, seqLen, fleetMaxHit);
                    if (selectedIndex >= 0) {
                        if (route.ctx().claimHotspotRescue()) {
                            lastRescueNanos = System.nanoTime();
                        } else {
                            selectedIndex = -1;
                        }
                    }
                }
                if (selectedIndex >= 0) {
                    route.reserveTokenPlan(survivors.endpoint(selectedIndex).ownership.planTokenWork(
                            Math.max(0L, seqLen - survivors.cacheHit(selectedIndex))));
                }
            }
        }

        if (selectedIndex < 0) {
            Logger.debug(
                    "Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId,
                    rejections);
            return PlacementResult.blocked(roleType);
        }

        if (probe && survivors.cacheHit(selectedIndex) < fleetMaxHit) {
            Logger.info("Prefill hotspot rescue plan request={} leaders={} target={} input={} leaderHit={} targetHit={}",
                    requestId, cacheWorkers, survivors.endpointAddress(selectedIndex), seqLen, fleetMaxHit,
                    survivors.cacheHit(selectedIndex));
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
        ServerStatus assignment = buildServerStatus(
                best, roleType, requestId,
                selectedTtft, selectedPrefillMs, bestCacheHit,
                survivors.ownershipVersion(selectedIndex), selectedPin, route);
        if (selectedTtft >= 0L) { reportSelectedEstimates(
                roleType,
                best,
                selectedTtft,
                selectedPrefillMs); }
        cacheMetricsReporter.reportCacheHitMetrics(roleType, bestCacheHit,
                seqLen > 0 ? bestCacheHit / (double) seqLen : 0.0);
        cacheMetricsReporter.reportRoutingSelectedCacheMatchMetrics(
                roleType, survivors.routingCacheMatchTokens(selectedIndex), seqLen);
        cacheMetricsReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, survivors.maximumRoutingCacheMatchTokens);
        return PlacementResult.success(assignment);
    }

    private static PlacementResult<ServerStatus, RoleType> classifyAdmissionFailure(
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
            var summary = endpoint.ownership.admissionSummary(priority, endpoint.requestLimit);
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

    private static PlacementResult<ServerStatus, RoleType> blockedWithDiagnostics(
            Response failure, RoleType role, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(role, failure, Map.of("role", role.name(), "workers", workerCount,
                "observedAtMs", System.currentTimeMillis(), "details", Map.copyOf(details)));
    }

    /** Select from candidates that already passed the common hard filters. */
    int selectBestCandidate(PrefillCandidateSet survivors,
                            long minProjectedTtftMs,
                            RoleType roleType,
                            String group,
                            long seqLen) {
        if (survivors.size() == 0) {
            return -1;
        }

        if (tokenBalance != null) {
            return selectTokenBalancedCandidate(survivors, roleType, group, seqLen);
        }

        long bestHit = Long.MIN_VALUE;
        long bestTtft = Long.MAX_VALUE;
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
                double minimumHitRate = cacheAffinityMinHitRate;
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
                        long ttft = survivors.ttftRank(i);
                        if (hitTokens > bestHit || hitTokens == bestHit && ttft < bestTtft) {
                            bestHit = hitTokens;
                            bestTtft = ttft;
                        }
                    }
                }
                affinityReason = bestHit != Long.MIN_VALUE
                        ? "CACHE_LEADER"
                        : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        int selectedIndex;
        if (bestHit != Long.MIN_VALUE) {
            long selectedHit = bestHit;
            long selectedTtft = bestTtft;
            selectedIndex = tieRotation.next(roleType, group, survivors.size(),
                    i -> survivors.projectedTtftMs(i) >= 0L && survivors.cacheHit(i) == selectedHit
                            && survivors.ttftRank(i) == selectedTtft,
                    survivors::endpointAddress);
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

    private int selectTokenBalancedCandidate(PrefillCandidateSet candidates, RoleType role, String group, long seqLen) {
        return selectTokenBalancedCandidate(candidates, role, group, seqLen, candidates.maximumCacheHit);
    }

    private int selectTokenBalancedCandidate(PrefillCandidateSet candidates, RoleType role, String group, long seqLen, long maxHit) {
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(maxHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        double[] scores = new double[candidates.size()];
        java.util.Arrays.fill(scores, Double.POSITIVE_INFINITY);
        double minimum = Double.POSITIVE_INFINITY;
        for (int i = 0; i < candidates.size(); i++) {
            if (!PrefillTokenBalance.eligible(candidates.cacheHit(i), maxHit, cacheFirst)) { continue; }
            var work = candidates.endpoint(i).ownership.tokenWorkload();
            // Preserve cache eligibility while waiting for Engine queue capacity.
            if (candidates.endpoint(i).engineQueueAdmissionBlocked()) { continue; }
            scores[i] = cacheFirst ? 0.0 : work.score() + Math.max(0L, seqLen);
            minimum = Math.min(minimum, scores[i]);
        }
        if (!Double.isFinite(minimum)) { return -1; }
        double selectedScore = minimum;
        return tieRotation.next(role, group, candidates.size(), i -> scores[i] == selectedScore,
                candidates::endpointAddress);
    }

    /** Called under tokenSelectionLock; a failed later admission still consumes the probe budget. */
    private int selectRescueTarget(EndpointDiscovery discovery, java.util.Set<String> leaders,
                                   PrefillCandidateSet candidates, long input, long leaderHit) {
        long nowNanos = System.nanoTime();
        if (lastRescueNanos != Long.MIN_VALUE && nowNanos - lastRescueNanos < tokenBalance.hotspotRescueIntervalNanos()) {
            return -1;
        }
        long now = nowNanos / 1_000_000L;
        long lastRescueMs = lastRescueNanos == Long.MIN_VALUE ? Long.MIN_VALUE : lastRescueNanos / 1_000_000L;
        var groups = new java.util.HashSet<String>();
        for (var entry : discovery.cacheCandidates()) {
            if (!leaders.contains(entry.address())) { continue; }
            if (!entry.endpoint().permitsHotspotRescue(true, now, lastRescueMs)) { return -1; }
            groups.add(entry.endpoint().getStatus().topologySnapshot().group());
        }
        int best = -1;
        for (int i = 0; i < candidates.size(); i++) {
            long hit = candidates.cacheHit(i);
            // Bound extra compute to one quarter of the original uncached work.
            if (hit >= leaderHit || leaderHit - hit > (input - leaderHit) / 4L) { continue; }
            var ep = candidates.endpoint(i);
            if (!groups.contains(ep.getStatus().topologySnapshot().group())
                    || !ep.permitsHotspotRescue(false, now, lastRescueMs)) { continue; }
            if (best < 0 || hit > candidates.cacheHit(best)
                    || hit == candidates.cacheHit(best) && candidates.ttftRank(i) < candidates.ttftRank(best)) {
                best = i;
            }
        }
        return best;
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

    private record EndpointDiscovery(
            List<EndpointRegistry.PrefillRoutingEntry> directory,
            List<EndpointRegistry.PrefillRoutingEntry> candidates,
            List<EndpointRegistry.PrefillRoutingEntry> cacheCandidates,
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
                    return cacheCandidates.get(index).address();
                }

                @Override
                public int size() {
                    return cacheCandidates.size();
                }
            };
        }
    }

    private PrefillCandidateSet evaluateCandidates(
            EndpointDiscovery discovery,
            Request request,
            SchedulingMetadata metadata,
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections,
            Map<RoleType, Integer> poolWideBlockers, PrefillCandidateSet candidates, boolean ignoreCache) {
        int eligibleSize = discovery.candidates().size();
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
            long cacheHit = ignoreCache ? 0L : cacheMatch.effectiveHitTokens();
            long routingCacheMatchTokens = cacheMatch.routingHitTokens();
            RouteProjection.Inputs projectionInputs =
                    ep.captureRouteProjectionInputs();
            PrefillTimePredictor predictor = ep.predictor;
            RouteProjection.CandidateView projection =
                    projector.projectView(
                            projectionInputs,
                            request.getRequestId(),
                            metadata.priority(),
                            planningAtMs,
                            // ExpirationTimer drives request deadlines through the owning scheduler.
                            // Selection scores endpoint work only; an expiry race
                            // must not be reported as a capacity blocker.
                            Long.MAX_VALUE,
                            request.getSeqLen(),
                            cacheHit,
                            routingCacheMatchTokens,
                            predictor == null
                                    ? null : predictor.evaluator(),
                            ep.deliveryProjection,
                            planningAtMs);
            if (!projection.selectable() && !projection.engineWorkUnmodeled()
                    && ep.engineQueueAllowsAdmission()
                    && (projection.blockerRole() == null || projection.blockerRole().supportsPrefill())) {
                projection = new RouteProjection.Candidate(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                        RouteProjection.Candidate.UNKNOWN, projection.incomingPrefillMs(), projection.initialHeadDisposition(),
                        "Engine queue accepts Prefill", null, cacheHit, routingCacheMatchTokens);
            }
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
                        endpointAddress, priorityOrdering ? "PRIORITY" : "FIFO",
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

    private boolean hasInflightCapacity(PrefillEndpoint endpoint) {
        return !endpoint.engineQueueAdmissionBlocked() && (endpoint.requestLimit == 0L || endpoint.ownership.progress().outstandingRequests() < endpoint.requestLimit)
                && (maxInflightBatches == 0 || endpoint.ownership.batchCapacityAvailable(maxInflightBatches));
    }

    private EndpointDiscovery discoverAvailableEndpoints(
            Request request,
            SchedulingMetadata metadata,
            RoleType roleType,
            String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                endpointRegistry.prefillRoutingSnapshot(roleType);
        var matching = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
        var cacheCandidates = tokenBalance == null ? matching : new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
        int registered = 0;
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (group != null && !group.equals(endpoint.getStatus().topologySnapshot().group())) {
                continue;
            }
            registered++;
            // Capacity can remove dispatch candidates, but must not redefine cache eligibility.
            if (tokenBalance != null) { cacheCandidates.add(entry); }
            if (endpoint.engineQueueAdmissionBlocked()) { continue; }
            if (tokenBalance != null && maxInflightBatches > 0
                    && !endpoint.ownership.batchCapacityAvailable(maxInflightBatches)) { continue; }
            if ((endpoint.requestLimit == 0L || endpoint.ownership.progress().outstandingRequests() < endpoint.requestLimit)
                    || preemptQueued && endpoint.ownership.canPreemptQueuedRequest(metadata.priority(), endpoint.requestLimit)) {
                if (group == null && matching.isEmpty()) { matching.ensureCapacity(directory.size()); }
                matching.add(entry);
            }
        }
        return new EndpointDiscovery(directory, matching, cacheCandidates, registered);
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
            long projectedTtftMs,
            long executionTimeMs) {
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

    private ServerStatus buildServerStatus(
            PrefillEndpoint ep,
            RoleType roleType,
            long requestId,
            long projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin, RequestRoute route) {
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
            ServerStatus result = ServerStatus.forWorker(roleType, requestId, topology, status);
            if (projectedTtftMs >= 0L) { result.setPrefillTime(projectedTtftMs); }
            result.setDebugInfo(debugInfo);
            WorkerEndpoint.GenerationPin ownedPin = selectedPin;
            selectedPin = null;
            return route.add(ownedPin, result, selectedPrefillMs, placementVersion);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

}
