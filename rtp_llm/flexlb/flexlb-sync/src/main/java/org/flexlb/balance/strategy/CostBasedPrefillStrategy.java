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
import org.flexlb.balance.scheduler.PrefillQueueShedding;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PrefillQueueSheddingConfig;
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
    private final double minAcceptingWorkerPercent;
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
        var shedding = config.isQueue() ? config.queueScheduler().getPrefillQueueShedding() : null;
        minAcceptingWorkerPercent = shedding == null
                ? PrefillQueueSheddingConfig.DEFAULT_MIN_ACCEPTING_WORKER_PERCENT : shedding.getMinAcceptingWorkerPercent();
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        cacheAffinity = affinity;
        cacheAffinityMinHitRate = affinity == null ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent());
    }

    /** Advisory overload estimate includes busy workers, before admission-capacity filtering. */
    public PrefillQueueShedding.Estimate estimateQueuedWork(
            Request request, RoleType role, String group) {
        var fleet = endpointRegistry.prefillRoutingSnapshot(role).stream()
                .filter(entry -> group == null || group.equals(entry.endpoint().getStatus().topologySnapshot().group()))
                .toList();
        if (fleet.isEmpty()) { return null; }
        var matches = cacheAwareService.findMatchingEngines(request.getBlockCacheKeys(), role,
                fleet.stream().map(EndpointRegistry.PrefillRoutingEntry::address).toList());
        long bestWork = Long.MAX_VALUE;
        long bestHit = -1L;
        for (var entry : fleet) {
            var endpoint = entry.endpoint();
            if (endpoint.predictor == null) { return null; }
            long hit = calculateCacheMatch(endpoint, entry.address(), matches, request).effectiveHitTokens();
            long work = PrefillPredictionBoundary.predictSingleRequestMs(
                    endpoint.predictor.evaluator(), request.getSeqLen(), hit);
            if (hit > bestHit || hit == bestHit && work < bestWork) {
                bestHit = hit;
                bestWork = work;
            }
        }
        if (tokenBalance != null && !PrefillTokenBalance.cacheFirst(
                bestHit, request.getSeqLen(), tokenBalance.getMinPrefixHitPercent())) {
            bestHit = 0L;
            bestWork = Long.MAX_VALUE;
            for (var entry : fleet) {
                bestWork = Math.min(bestWork, PrefillPredictionBoundary.predictSingleRequestMs(
                        entry.endpoint().predictor.evaluator(), request.getSeqLen(), 0L));
            }
        }
        return new PrefillQueueShedding.Estimate(
                request.getSeqLen(), bestHit, bestWork, fleet.size());
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
        if (discovery.candidates().isEmpty()) {
            Logger.debug("Prefill select failed: no admission capacity, request_id={}",
                    requestId);
            return classifyAdmissionFailure(
                    discovery.directory(), metadata.priority(), roleType, group);
        }
        Map<String, Integer> cacheMatchResults =
                cacheAwareService.findMatchingEngines(request.getBlockCacheKeys(), roleType, discovery.addresses());
        long fleetMaxHit = 0L;
        if (tokenBalance != null) {
            for (var entry : discovery.cacheCandidates()) {
                fleetMaxHit = Math.max(fleetMaxHit,
                        calculateCacheMatch(entry.endpoint(), entry.address(), cacheMatchResults, request).effectiveHitTokens());
            }
        }
        boolean hot = tokenBalance != null && PrefillTokenBalance.cacheFirst(
                fleetMaxHit, seqLen, tokenBalance.getMinPrefixHitPercent());
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

        boolean rescue = false;
        if (tokenBalance != null) {
            var escape = tokenBalance.getHotspotRescue();
            long createdAt = route.ctx().snapshot().createdAtMs();
            boolean agedShort = escape != null && createdAt > 0L
                    && System.currentTimeMillis() - createdAt >= escape.getMinWaitMs()
                    && seqLen <= escape.getMaxInputTokens();
            if (agedShort && fleetMaxHit > 0L) {
                rescue = true;
                for (var entry : discovery.cacheCandidates()) {
                    long hit = calculateCacheMatch(entry.endpoint(), entry.address(), cacheMatchResults, request).effectiveHitTokens();
                    if (hit == fleetMaxHit && entry.endpoint().ownership.tokenWorkload()
                            .pendingTokens() < tokenBalance.getMaxPendingComputeTokens()) {
                        rescue = false;
                        break;
                    }
                }
            }
            rescue |= escape != null && route.ctx().prefillMigrationSource() != null;
            if (PrefillTokenBalance.cacheFirst(fleetMaxHit, seqLen, tokenBalance.getMinPrefixHitPercent())
                    && survivors.maximumCacheHit < fleetMaxHit) {
                return PlacementResult.blocked(roleType);
            }
        }

        route.setColdTokenBalance(tokenBalance != null && !hot);
        boolean migrating = tokenBalance != null && tokenBalance.getHotspotRescue() != null
                && route.ctx().prefillMigrationSource() != null;
        // A migrated hot request must recheck target load and exclude its old worker,
        // while retaining the same maximum-cache-hit eligibility as normal routing.
        rescue = migrating || (rescue && !hot);
        int selectedIndex;
        if (tokenBalance == null) {
            selectedIndex = selectBestCandidate(survivors, survivors.minimumTtftRank, roleType, group, seqLen);
        } else {
            // Parallel planners must see earlier choices before asynchronous admission commits.
            // Only scoring + soft reservation are serialized; cache lookup/projection stay parallel.
            synchronized (tokenSelectionLock) {
                selectedIndex = rescue ? selectRescueCandidate(survivors, roleType, group, seqLen, fleetMaxHit, route.ctx().prefillMigrationSource())
                        : selectTokenBalancedCandidate(survivors, roleType, group, seqLen, fleetMaxHit);
                if (selectedIndex < 0 && rescue && !migrating && hasEngineQueueAdmission(survivors)) {
                    selectedIndex = selectTokenBalancedCandidate(survivors, roleType, group, seqLen, fleetMaxHit);
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

    public int currentPrefillWorkerCount(String group) {
        return (int) endpointRegistry.prefillRoutingSnapshot(RoleType.PREFILL).stream()
                .filter(entry -> group == null || group.equals(entry.endpoint().getStatus().topologySnapshot().group()))
                .count();
    }

    private int migrationSourceCursor;

    public java.util.Set<org.flexlb.balance.scheduler.PlacementKey> hotspotRescueDomains() {
        if (tokenBalance == null || tokenBalance.getHotspotRescue() == null) { return java.util.Set.of(); }
        var domains = new java.util.HashSet<org.flexlb.balance.scheduler.PlacementKey>();
        for (var role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            for (var entry : endpointRegistry.prefillRoutingSnapshot(role)) {
                var endpoint = entry.endpoint();
                var work = endpoint.ownership.tokenWorkload();
                if (work.pendingTokens() + work.plannedTokens() <= tokenBalance.getHotspotRescue().getMaxIdlePendingTokens()) {
                    domains.add(new org.flexlb.balance.scheduler.PlacementKey(role, endpoint.getStatus().topologySnapshot().group(), null));
                    domains.add(org.flexlb.balance.scheduler.PlacementKey.anyGroup(role));
                }
            }
        }
        return domains;
    }

    /** Advisory capture only. Context/delivery ownership is rechecked atomically before removal. */
    public java.util.List<RequestRoute> hotspotMigrationCandidates(int limit) {
        if (limit <= 0 || tokenBalance == null || tokenBalance.getHotspotRescue() == null) { return List.of(); }
        var rescue = tokenBalance.getHotspotRescue();
        var fleet = endpointRegistry.prefillRoutingSnapshot(RoleType.PREFILL);
        if (fleet.isEmpty()) { return List.of(); }
        long wallMs = System.currentTimeMillis();
        var idleGroups = new java.util.HashSet<String>();
        for (var entry : fleet) {
            var work = entry.endpoint().ownership.tokenWorkload();
            if (work.pendingTokens() + work.plannedTokens() <= rescue.getMaxIdlePendingTokens()) {
                idleGroups.add(entry.endpoint().getStatus().topologySnapshot().group());
            }
        }
        if (idleGroups.isEmpty()) { return List.of(); }
        var order = java.util.Comparator.comparingLong(RequestRoute::seqLen).thenComparingLong(RequestRoute::enqueueSeq);
        var candidates = new java.util.PriorityQueue<RequestRoute>(Math.min(limit, 64), order.reversed());
        int budget = (int) Math.min(Integer.MAX_VALUE, Math.max(64L, 64L * limit));
        int sourceLimit = (int) Math.min(fleet.size(), Math.max(1L, 2L * limit));
        for (int examined = 0; examined < sourceLimit && budget > 0; examined++) {
            migrationSourceCursor %= fleet.size();
            var endpoint = fleet.get(migrationSourceCursor++).endpoint();
            if (!idleGroups.contains(endpoint.getStatus().topologySnapshot().group())
                    || endpoint.ownership.tokenWorkload().pendingTokens() < tokenBalance.getMaxPendingComputeTokens()) { continue; }
            for (var route : endpoint.ownership.captureQueue(Math.min(64, budget)).items()) {
                if (budget-- <= 0) { break; }
                if (route.seqLen() > rescue.getMaxInputTokens() || route.decodeEp() == null
                        || route.ctx().prefillMigrationSource() != null
                        || wallMs - route.ctx().snapshot().createdAtMs() < rescue.getMinWaitMs()) { continue; }
                if (candidates.size() < limit) { candidates.add(route); }
                else if (order.compare(route, candidates.peek()) < 0) { candidates.poll(); candidates.add(route); }
            }
        }
        var result = new ArrayList<>(candidates);
        result.sort(order);
        // Validate at most the bounded shortlist. Final placement rechecks the current cache fleet.
        result.removeIf(route -> !hasCacheEligibleMigrationTarget(route, fleet));
        return result;
    }

    private boolean hasCacheEligibleMigrationTarget(RequestRoute route, List<EndpointRegistry.PrefillRoutingEntry> fleet) {
        var request = route.ctx().data().getRequest();
        var source = route.prefillEp();
        if (request == null || source == null) { return false; }
        String group = source.getStatus().topologySnapshot().group();
        var pool = fleet.stream().filter(entry -> java.util.Objects.equals(group,
                entry.endpoint().getStatus().topologySnapshot().group())).toList();
        var matches = cacheAwareService.findMatchingEngines(request.getBlockCacheKeys(), RoleType.PREFILL,
                pool.stream().map(EndpointRegistry.PrefillRoutingEntry::address).toList());
        long[] hits = new long[pool.size()];
        long maxHit = 0L;
        for (int i = 0; i < pool.size(); i++) {
            var entry = pool.get(i);
            hits[i] = calculateCacheMatch(entry.endpoint(), entry.address(), matches, request).effectiveHitTokens();
            maxHit = Math.max(maxHit, hits[i]);
        }
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(maxHit, request.getSeqLen(), tokenBalance.getMinPrefixHitPercent());
        for (int i = 0; i < pool.size(); i++) {
            var endpoint = pool.get(i).endpoint();
            if (endpoint == source || endpoint.isGenerationRetiringOrRetired()
                    || !PrefillTokenBalance.eligible(hits[i], maxHit, cacheFirst)) { continue; }
            var work = endpoint.ownership.tokenWorkload();
            if (work.pendingTokens() + work.plannedTokens() <= tokenBalance.getHotspotRescue().getMaxIdlePendingTokens()) {
                return true;
            }
        }
        return false;
    }

    /** Rescue remains within the same cache-eligible set as ordinary routing. */
    private int selectRescueCandidate(PrefillCandidateSet candidates, RoleType role, String group, long seqLen, long maxHit, PrefillEndpoint excluded) {
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(maxHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        double[] scores = new double[candidates.size()];
        java.util.Arrays.fill(scores, Double.POSITIVE_INFINITY);
        double minimum = Double.POSITIVE_INFINITY;
        for (int i = 0; i < candidates.size(); i++) {
            if (!PrefillTokenBalance.eligible(candidates.cacheHit(i), maxHit, cacheFirst)) { continue; }
            var work = candidates.endpoint(i).ownership.tokenWorkload();
            if (candidates.endpoint(i) == excluded || work.pendingTokens() + work.plannedTokens() > tokenBalance.getHotspotRescue().getMaxIdlePendingTokens()) { continue; }
            scores[i] = cacheFirst ? 0.0 : work.score() + Math.max(0L, seqLen);
            minimum = Math.min(minimum, scores[i]);
        }
        if (!Double.isFinite(minimum)) { return -1; }
        double chosen = minimum;
        return tieRotation.next(role, group, candidates.size(), i -> scores[i] == chosen, candidates::endpointAddress);
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
            // Preserve a busy cache leader by parking globally, where existing HIT_RATE shedding applies.
            if (!PrefillTokenBalance.hasCapacity(work.pendingTokens(), tokenBalance.getMaxPendingComputeTokens())
                    && !candidates.endpoint(i).engineQueueAllowsAdmission()) { continue; }
            scores[i] = cacheFirst ? 0.0 : work.score() + Math.max(0L, seqLen);
            minimum = Math.min(minimum, scores[i]);
        }
        if (!Double.isFinite(minimum)) { return -1; }
        double selectedScore = minimum;
        return tieRotation.next(role, group, candidates.size(), i -> scores[i] == selectedScore,
                candidates::endpointAddress);
    }

    private static boolean hasEngineQueueAdmission(PrefillCandidateSet candidates) {
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.endpoint(i).engineQueueAllowsAdmission()) { return true; }
        }
        return false;
    }

    /** Pool-wide shedding relief needs the same minimum fraction as the per-request cold veto. */
    public boolean hasEngineQueueAdmission(RoleType role, String group) {
        int workers = 0;
        int accepting = 0;
        for (var entry : endpointRegistry.prefillRoutingSnapshot(role)) {
            if (group != null && !group.equals(entry.endpoint().getStatus().topologySnapshot().group())) { continue; }
            workers++;
            if (entry.endpoint().engineQueueAllowsAdmission()) { accepting++; }
        }
        return enoughAcceptingWorkers(accepting, workers);
    }

    private boolean enoughAcceptingWorkers(int accepting, int workers) {
        return workers > 0 && accepting > 0 && accepting * 100.0 >= workers * minAcceptingWorkerPercent;
    }

    public enum ColdAdmission { AVAILABLE, INSUFFICIENT_CAPACITY, UNKNOWN }

    /** Read-only veto for cold shedding. Use the routing admission filters, never a TTFT threshold.
     * No cache affinity, route pin, soft token reservation or round-robin state is acquired here.
     */
    public ColdAdmission coldAdmission(Request request, SchedulingMetadata metadata, RoleType role, String group) {
        if (tokenBalance == null) { return ColdAdmission.UNKNOWN; }
        try {
            var discovery = discoverAvailableEndpoints(request, metadata, role, group);
            if (discovery.registeredCount() == 0) { return ColdAdmission.UNKNOWN; }
            if (discovery.candidates().isEmpty()) { return ColdAdmission.INSUFFICIENT_CAPACITY; }
            var blockers = new EnumMap<RoleType, Integer>(RoleType.class);
            var candidates = new PrefillCandidateSet();
            evaluateCandidates(discovery, request, metadata, Map.of(), new java.util.HashMap<>(), blockers,
                    candidates, true);
            // A different role or an indeterminate projection is not proof of Prefill overload.
            if (candidates.size() == 0) { return ColdAdmission.UNKNOWN; }
            int accepting = 0;
            for (int i = 0; i < candidates.size(); i++) {
                var endpoint = candidates.endpoint(i);
                var work = endpoint.ownership.tokenWorkload();
                if (PrefillTokenBalance.hasCapacity(work.pendingTokens(), tokenBalance.getMaxPendingComputeTokens())
                        || endpoint.engineQueueAllowsAdmission()) {
                    accepting++;
                    // Count the whole matching pool, including request-count and token-full workers.
                    if (enoughAcceptingWorkers(accepting, discovery.registeredCount())) {
                        return ColdAdmission.AVAILABLE;
                    }
                }
            }
            // Unresolved projections cannot prove that the accepting fraction is below the threshold.
            if (candidates.size() < discovery.candidates().size()) { return ColdAdmission.UNKNOWN; }
            return ColdAdmission.INSUFFICIENT_CAPACITY;
        } catch (RuntimeException failure) {
            // Missing/invalid telemetry must not become evidence authorizing early rejection.
            Logger.debug("Cold Prefill admission check unavailable, request_id={}", request.getRequestId(), failure);
            return ColdAdmission.UNKNOWN;
        }
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
