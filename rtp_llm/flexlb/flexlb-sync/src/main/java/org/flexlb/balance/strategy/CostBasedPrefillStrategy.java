package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.balance.scheduler.PrefillQueueShedding;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
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
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Prefill 池的候选排序与端点选择：默认按建模 TTFT，可选按未缓存 token 工作量均衡。
 * 热点逃逸开关在两种排序下都能释放被钉住的缓存亲和，避免单个缓存热点持续过载。
 * 同时向队列丢弃策略提供池占用快照、过载证据与单请求工作量估计。
 */
@Component
public class CostBasedPrefillStrategy {

    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();
    /** Pool-wide sustained-pressure evidence for global shedding admission. */
    private final PrefillQueuePressure queuePressure = new PrefillQueuePressure();
    /** Hotspot governor window; one selector per window advances the whole pool. */
    private static final long HOTSPOT_GOVERN_INTERVAL_MS = 100L;
    private final AtomicLong hotspotGovernAtMs = new AtomicLong();

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
        RoutingConfig.TokenBalanceConfig tokenBalance = config.getRouter()
                .getRoles().getPrefill().getTokenBalance();
        boolean hotspotRelease = config.getRouter().getRoles()
                .getPrefill().getHotspotRelease().isEnabled();
        if (hotspotRelease) {
            governHotspotRelease(discovery.candidates(), config, System.currentTimeMillis());
        }
        int selectedIndex = tokenBalance != null
                ? selectTokenBalancedCandidate(
                        survivors, roleType, group, seqLen, tokenBalance, hotspotRelease)
                : selectBestCandidate(
                        survivors, roleType, group, seqLen, config, hotspotRelease,
                        balanceContext.getRequestExpiresAtMs() - System.currentTimeMillis());

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
        // Reserve outstanding uncached work before handing the pin to the route
        // owner; blocked and stale-generation paths return before this point.
        PrefillTokenWorkload.Reservation tokenReservation =
                tokenBalance == null ? null : best.planTokenWork(
                        Math.max(0L, seqLen - bestCacheHit));
        SelectedRole selectedRole = buildSelectedRole(
                best,
                roleType,
                requestId,
                selectedTtft,
                selectedPrefillMs,
                bestCacheHit,
                survivors.routingCacheMatchTokens(selectedIndex),
                survivors.ownershipVersion(selectedIndex),
                selectedPin,
                tokenReservation);
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

    /**
     * Rank outstanding uncached token work instead of modeled TTFT. Cold
     * requests go to the least-loaded worker; hot requests only rotate among
     * cache leaders so their prefix reuse is not discarded. Candidates with a
     * blocked Engine queue are skipped while waiting for capacity. With
     * hotspot release enabled, a leader whose integral line climbs above this
     * request's hit rate stops pinning it: every worker then competes by
     * outstanding work, counting each candidate's own hit as relief so the
     * drift prefers nodes that can keep part of the prefix, while returning to
     * a released leader stays a legal outcome of the same ranking.
     */
    private int selectTokenBalancedCandidate(PrefillCandidateSet survivors,
            RoleType roleType, String group, long seqLen,
            RoutingConfig.TokenBalanceConfig tokenBalance, boolean hotspotRelease) {
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(
                survivors.maximumCacheHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        boolean pinnedLeader = false;
        if (cacheFirst) {
            for (int index = 0; index < survivors.size(); index++) {
                if (survivors.cacheHit(index) == survivors.maximumCacheHit
                        && !PrefillTokenBalance.hotspotReleases(
                                survivors.endpoint(index).hotspotReleaseLine(),
                                survivors.cacheHit(index), seqLen)) {
                    pinnedLeader = true;
                    break;
                }
            }
        }
        double[] scores = new double[survivors.size()];
        double minimumScore = Double.MAX_VALUE;
        int eligibleCount = 0;
        long longestEligiblePrefix = 0L;
        for (int index = 0; index < survivors.size(); index++) {
            long cacheHit = survivors.cacheHit(index);
            PrefillEndpoint endpoint = survivors.endpoint(index);
            // While a pinned leader exists only leaders compete; once every
            // leader released, the whole pool competes by balanced work.
            boolean pinEligible = !pinnedLeader
                    || cacheHit == survivors.maximumCacheHit
                            && !PrefillTokenBalance.hotspotReleases(
                                    endpoint.hotspotReleaseLine(), cacheHit, seqLen);
            if (!pinEligible || endpoint.engineQueueAdmissionBlocked()) {
                scores[index] = Double.MAX_VALUE;
                continue;
            }
            long plannedTokens = hotspotRelease
                    ? Math.max(0L, seqLen - cacheHit) : seqLen;
            double score = cacheFirst && pinnedLeader
                    ? 0.0
                    : endpoint.tokenWorkloadScore() + plannedTokens;
            scores[index] = score;
            minimumScore = Math.min(minimumScore, score);
            if (pinnedLeader) {
                longestEligiblePrefix = Math.max(longestEligiblePrefix, survivors.routingCacheMatchTokens(index));
            }
            eligibleCount++;
        }
        if (eligibleCount == 0) {
            return -1;
        }
        double selectedScore = minimumScore;
        long requiredPrefix = pinnedLeader ? longestEligiblePrefix : 0L;
        return tieRotation.next(roleType, group, survivors.size(),
                index -> scores[index] == selectedScore && survivors.routingCacheMatchTokens(index) >= requiredPrefix,
                survivors::endpointAddress);
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

    /** Occupancy snapshot of one Prefill pool, used by queue shedding decisions. */
    public record PrefillPoolSnapshot(int workers, int accepting, long pendingTokens) { }

    /** Queue shedding measures Engine admission independently of local delivery slots. */
    public PrefillPoolSnapshot prefillPoolSnapshot(RoleType roleType, String group) {
        List<EndpointRegistry.PrefillRoutingEntry> candidates =
                discoverEndpoints(roleType, group).candidates();
        if (candidates.isEmpty()) {
            return null;
        }
        int accepting = 0;
        long pendingTokens = 0L;
        for (EndpointRegistry.PrefillRoutingEntry routingEntry : candidates) {
            PrefillEndpoint endpoint = routingEntry.endpoint();
            if (!endpoint.engineQueueAdmissionBlocked()) {
                accepting++;
            }
            pendingTokens += Math.max(0L, endpoint.pendingTokenWork());
        }
        return new PrefillPoolSnapshot(candidates.size(), accepting, pendingTokens);
    }

    /** True when more than half of the pool reports deep Engine waiting queues for a sustained period. */
    public boolean isPrefillClusterOverloaded(RoleType roleType, String group, long nowMs) {
        List<EndpointRegistry.PrefillRoutingEntry> candidates =
                discoverEndpoints(roleType, group).candidates();
        int waiting = 0;
        for (EndpointRegistry.PrefillRoutingEntry routingEntry : candidates) {
            if (routingEntry.endpoint().getStatus()
                    .committedEngineObservation().waitingQueryLen() > 10) {
                waiting++;
            }
        }
        return queuePressure.observe(roleType, group, candidates.size(), waiting, nowMs);
    }

    /** Compare resident prefixes on the same cache-index snapshot after waiting for Decode. */
    public boolean hasBetterResidentPrefix(BalanceContext context, RoleType role, String group,
                                           PrefillEndpoint provisional) {
        Request request = context.getRequest();
        if (request.getBlockCacheKeys() == null || request.getBlockCacheKeys().isEmpty()) { return false; }
        EndpointDiscovery discovery = discoverEndpoints(role, group);
        if (discovery.candidates().size() < 2 || discovery.directory().isEmpty()) { return false; }
        Map<String, Integer> matches = getCacheMatchResults(context, role, discovery);
        long current = calculateCacheMatch(provisional, provisional.ipPort(), matches, request).routingHitTokens();
        for (var candidate : discovery.candidates()) {
            if (candidate.endpoint() != provisional && calculateCacheMatch(candidate.endpoint(), candidate.address(),
                    matches, request).routingHitTokens() > current) {
                return true;
            }
        }
        return false;
    }

    /**
     * Advisory fleet estimate for one queued request: the best reachable cache
     * hit, the best modeled service time across the pool, and the pool size.
     * Returns null when the pool is empty or no estimator produced a value.
     */
    public PrefillQueueShedding.Estimate estimateQueuedWork(
            BalanceContext balanceContext, RoleType roleType, String group) {
        EndpointDiscovery discovery = discoverEndpoints(roleType, group);
        int workers = discovery.candidates().size();
        if (workers <= 0) {
            return null;
        }
        Request request = balanceContext.getRequest();
        long seqLen = Math.max(0L, request.getSeqLen());
        Map<String, Integer> cacheMatchResults = discovery.directory().isEmpty() ? Map.of()
                : getCacheMatchResults(balanceContext, roleType, discovery);
        long bestHit = 0L;
        long bestWorkMs = -1L;
        for (EndpointRegistry.PrefillRoutingEntry routingEntry : discovery.candidates()) {
            PrefillEndpoint endpoint = routingEntry.endpoint();
            CacheTokenMatch cacheMatch = calculateCacheMatch(
                    endpoint, routingEntry.address(), cacheMatchResults, request);
            long hit = Math.max(0L, cacheMatch.effectiveHitTokens());
            bestHit = Math.max(bestHit, hit);
            PrefillTimePredictor predictor = endpoint.getPredictor();
            if (predictor == null) {
                continue;
            }
            try {
                long workMs = PrefillPredictionBoundary.predictSingleRequestMs(
                        predictor.evaluator(), seqLen, hit);
                if (bestWorkMs < 0L || workMs < bestWorkMs) {
                    bestWorkMs = workMs;
                }
            } catch (RuntimeException predictionFailure) {
                // A broken estimator only removes its own candidate from the estimate.
            }
        }
        if (bestWorkMs < 0L) {
            return null;
        }
        return new PrefillQueueShedding.Estimate(seqLen, bestHit, bestWorkMs, workers);
    }

    private static PlacementResult<SelectedRole, PlacementKey> blockedWithDiagnostics(
            Response failure, PlacementKey key, int workerCount, Map<String, ?> details) {
        return PlacementResult.blocked(key, failure, Map.of("role", key.role().name(), "workers", workerCount,
                "observedAtMs", System.currentTimeMillis(), "details", Map.copyOf(details)));
    }

    /**
     * Rank modeled work and cache hits before checking the selected endpoint
     * admission. With hotspot release enabled, a cache leader whose integral
     * release line climbed past a request's hit rate no longer pins it: the
     * request falls back to plain TTFT ranking, where remaining partial hits
     * still shorten modeled work.
     */
    int selectBestCandidate(PrefillCandidateSet survivors,
                            RoleType roleType,
                            String group,
                            long seqLen,
                            FlexlbConfig config,
                            boolean hotspotRelease,
                            long remainingQueueMs) {
        long minProjectedTtftMs = survivors.minimumProjectedTtftMs;
        if (survivors.size() == 0) {
            return -1;
        }

        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter()
                .getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        int baselineCacheLeader = -1;
        if (cacheAffinity != null) {
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.projectedTtftMs(i) == minProjectedTtftMs
                        && (baselineCacheLeader < 0 || survivors.compareCacheMatch(i, baselineCacheLeader) > 0)) {
                    baselineCacheLeader = i;
                }
            }
            boolean strongerCache = false;
            boolean minimumHitRateMet = false;
            double minimumHitRate = normalizedHitRate(cacheAffinity.getMinPrefixHitPercent());
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.compareCacheMatch(i, baselineCacheLeader) <= 0) {
                    continue;
                }
                strongerCache = true;
                long hitTokens = survivors.cacheHit(i);
                if (minimumHitRate > 0.0 && (seqLen <= 0L
                        || hitTokens * RoutingConfig.PERCENTAGE_SCALE / seqLen < minimumHitRate)) {
                    continue;
                }
                minimumHitRateMet = true;
                boolean withinBudget = hotspotRelease
                        ? !PrefillTokenBalance.hotspotReleases(survivors.endpoint(i).hotspotReleaseLine(), hitTokens, seqLen)
                        : survivors.projectedTtftMs(i) <= affinityCutoffMs;
                if (withinBudget) {
                    preferredCandidates.set(i);
                }
            }
            if (!preferredCandidates.isEmpty()) {
                affinityReason = "CACHE_LEADER";
            } else if (minimumHitRateMet) {
                affinityReason = hotspotRelease ? "HOTSPOT_RELEASE" : "OVER_CAP";
            } else if (strongerCache) {
                affinityReason = "LOW_CACHE_HIT";
            }
        }

        int selectedIndex;
        if (!preferredCandidates.isEmpty()) {
            selectedIndex = selectCacheLeader(survivors, preferredCandidates, roleType, group);
            if (hotspotRelease && config.isSingleDecision()
                    && config.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH) {
                int idle = selectIdleAlternativeBeforeDeadline(survivors, roleType, group, selectedIndex,
                        remainingQueueMs, config.getRouter().getRoles().getPrefill().getHotspotRelease().getSetpointFloor());
                if (idle >= 0) {
                    selectedIndex = idle;
                    affinityReason = "QUEUE_DEADLINE";
                }
            }
        } else {
            int reference = baselineCacheLeader;
            selectedIndex = tieRotation.next(roleType, group, survivors.size(),
                    i -> survivors.projectedTtftMs(i) == minProjectedTtftMs
                            && (reference < 0 || survivors.compareCacheMatch(i, reference) == 0),
                    survivors::endpointAddress);
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

    /** A short burst can fill the pinned queue before the sustained-pressure window elapses. */
    private int selectIdleAlternativeBeforeDeadline(PrefillCandidateSet candidates, RoleType role, String group,
                                                   int pinned, long remainingQueueMs, long depthFloor) {
        if (remainingQueueMs <= 0L || candidates.projectedWaitMs(pinned) < remainingQueueMs
                || candidates.endpoint(pinned).queuedRequestCount() <= depthFloor) {
            return -1;
        }
        BitSet idle = new BitSet(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.admissionBlock(i) == null && candidates.projectedWaitMs(i) < remainingQueueMs
                    && candidates.projectedTtftMs(i) < candidates.projectedTtftMs(pinned)
                    && candidates.endpoint(i).observedRequestCount() == 0L) {
                idle.set(i);
            }
        }
        return idle.isEmpty() ? -1 : selectCacheLeader(candidates, idle, role, group);
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            PrefillCandidateSet survivors, BitSet preferredCandidates, RoleType role, String group) {
        int best = -1;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)) {
                continue;
            }
            int cacheOrder = best < 0 ? 1 : survivors.compareCacheMatch(candidateIndex, best);
            if (cacheOrder > 0 || cacheOrder == 0
                    && survivors.projectedTtftMs(candidateIndex) < survivors.projectedTtftMs(best)) {
                best = candidateIndex;
            }
        }
        if (best < 0) {
            return -1;
        }
        int leader = best;
        return tieRotation.next(role, group, survivors.size(),
                i -> contains(preferredCandidates, i) && survivors.compareCacheMatch(i, leader) == 0
                        && survivors.projectedTtftMs(i) == survivors.projectedTtftMs(leader),
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
        long enqueuedAtMs = balanceContext.getFirstWorkerEnqueueTime();
        if (enqueuedAtMs == 0L) {
            enqueuedAtMs = planningAtMs;
        }
        if (enqueueSeq == 0L) {
            enqueueSeq = Long.MAX_VALUE;
        }
        RouteProjection.Session projectionSession = RouteProjection.session();

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
                GroupPlanner.Item admissionProbe = new GroupPlanner.Item(request.getRequestId(),
                        balanceContext.getPriority(), enqueueSeq, enqueuedAtMs,
                        Long.MAX_VALUE, request.getSeqLen(), cacheHit);
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

    /**
     * Advance every Prefill endpoint's hotspot release line at most once per
     * window: one concurrent selector wins the CAS and governs the whole pool
     * snapshot in a single pass over the fleet's router and Engine waiting depths —
     * every endpoint integrates against the shared fleet-relative setpoint.
     * A pool with any fully idle worker counts as having spare capacity, so
     * congestion there is load imbalance, not fleet saturation, and the line
     * is still allowed to climb.
     */
    void governHotspotRelease(
            List<EndpointRegistry.PrefillRoutingEntry> candidates,
            FlexlbConfig config,
            long nowMs) {
        if (candidates.isEmpty()) {
            return;
        }
        long lastMs = hotspotGovernAtMs.get();
        if (lastMs >= 0L && nowMs - lastMs < HOTSPOT_GOVERN_INTERVAL_MS) {
            return;
        }
        if (!hotspotGovernAtMs.compareAndSet(lastMs, nowMs)) {
            return;
        }
        long[] waitingDepths = new long[candidates.size()];
        boolean poolSpare = false;
        for (int i = 0; i < candidates.size(); i++) {
            PrefillEndpoint endpoint = candidates.get(i).endpoint();
            // NON_BATCH can hold the backlog at the router while the Engine queue stays shallow.
            waitingDepths[i] = saturatingAdd(endpoint.queuedRequestCount(),
                    Math.max(0L, endpoint.getStatus().committedEngineObservation().waitingQueryLen()));
            if (waitingDepths[i] == 0L
                    && endpoint.getInflightBatchCount() == 0) {
                poolSpare = true;
            }
        }
        long setpoint = hotspotReleaseSetpoint(
                config.getRouter().getRoles().getPrefill().getHotspotRelease(),
                waitingDepths);
        for (int i = 0; i < candidates.size(); i++) {
            candidates.get(i).endpoint().governHotspotRelease(waitingDepths[i], setpoint, poolSpare, nowMs);
        }
    }

    /**
     * Fleet-relative congestion zero point: max(floor, outlierFactor x fleet
     * median waiting depth). A large-batch engine whose healthy queue is deep
     * stays below its own median-scaled setpoint, while a shallow-queue
     * hotspot clears the floor. Package-private so the scenario tests drive
     * the exact production formula.
     */
    static long hotspotReleaseSetpoint(
            RoutingConfig.HotspotReleaseConfig release, long[] waitingDepths) {
        long[] sorted = waitingDepths.clone();
        Arrays.sort(sorted);
        double median = sorted.length % 2 == 1
                ? sorted[sorted.length / 2]
                : 0.5 * sorted[sorted.length / 2 - 1] + 0.5 * sorted[sorted.length / 2];
        double floor = Math.max(1L, release.getSetpointFloor());
        double outlierFactor = Math.max(1.0, release.getOutlierFactor());
        return (long) Math.max(floor, outlierFactor * median);
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
            long residentPrefixTokens,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin,
            PrefillTokenWorkload.Reservation tokenReservation) {
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
            PrefillTokenWorkload.Reservation ownedReservation = tokenReservation;
            tokenReservation = null;
            return SelectedRole.prefill(
                    ownedPin, result, selectedPrefillMs, placementVersion, residentPrefixTokens, ownedReservation);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
            if (tokenReservation != null) {
                tokenReservation.close();
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
