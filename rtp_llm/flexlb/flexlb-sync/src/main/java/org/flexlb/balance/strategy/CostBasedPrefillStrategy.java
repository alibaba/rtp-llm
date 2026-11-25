package org.flexlb.balance.strategy;

import org.flexlb.balance.AdmissionVersion;
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
import org.flexlb.config.FlexlbConfig;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Prefill 池的候选排序与端点选择：默认按建模 TTFT，可选按未缓存 token 工作量均衡。
 * 热点逃逸开关在两种排序下都能释放被钉住的缓存亲和，避免单个缓存热点持续过载。
 * 同时向队列丢弃策略提供池占用快照、过载证据与单请求工作量估计。
 */
@Component
public class CostBasedPrefillStrategy {

    private static final ThreadLocal<PrefillCandidateSet> CANDIDATES =
            ThreadLocal.withInitial(PrefillCandidateSet::new);

    private final HotspotRedirectLimiter redirectLimiter;
    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final PrefillCandidateRanking ranking;
    /** Pool-wide sustained-pressure evidence for global shedding admission. */
    private final PrefillQueuePressure queuePressure = new PrefillQueuePressure();
    private final PrefillHotspots hotspots;

    @Autowired
    public CostBasedPrefillStrategy(WorkerDirectory workerDirectory,
                                    CacheAwareService cacheAwareService,
                                    EngineHealthReporter engineHealthReporter) {
        this(workerDirectory, cacheAwareService, engineHealthReporter,
                new HotspotRedirectLimiter(), new PrefillHotspots(System::nanoTime));
    }

    CostBasedPrefillStrategy(WorkerDirectory workerDirectory,
                             CacheAwareService cacheAwareService,
                             EngineHealthReporter engineHealthReporter,
                             HotspotRedirectLimiter redirectLimiter, PrefillHotspots hotspots) {
        this.hotspots = hotspots;
        this.redirectLimiter = redirectLimiter;
        this.ranking = new PrefillCandidateRanking(engineHealthReporter);
        this.workerDirectory = workerDirectory;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
    }

    public PlacementResult<SelectedRole, PlacementKey> select(
            BalanceContext balanceContext,
            RoleType roleType,
            String group) {
        // Cache matching sees every live Prefill, including busy or blocked workers.
        EndpointDiscovery discovery = discoverEndpoints(roleType, group);
        Map<String, Integer> cacheMatchResults = discovery.directory().isEmpty() ? Map.of()
                : getCacheMatchResults(balanceContext, roleType, discovery);
        if (discovery.candidates().isEmpty()) {
            return blockedWithDiagnostics(Response.error(roleType.getErrorType()),
                    new PlacementKey(roleType, group), 0, Map.of());
        }
        var release = balanceContext.getConfig().getRouter().getRoles().getPrefill().getHotspotRelease();
        return AdmissionVersion.retryConflicts(() -> {
            Map<String, Integer> rejections = new java.util.HashMap<>();
            PrefillCandidateSet candidates = evaluateCandidates(discovery, balanceContext, cacheMatchResults, rejections);
            if (candidates.size() == 0) {
                return blockedWithDiagnostics(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                        new PlacementKey(roleType, group), discovery.candidates().size(), rejections);
            }
            if (release.isEnabled()) {
                redirectLimiter.recordPlacement(balanceContext, candidates.maximumCacheHit);
                candidates.setHotspots(hotspots.sample(roleType, group, discovery.candidates(), release));
            }
            var choice = ranking.select(candidates, roleType, group, balanceContext);
            return registerChoice(balanceContext, roleType, group, candidates, choice, discovery.candidates().size());
        }, attempt -> attempt.status() == PlacementResult.Status.REPLAN);
    }

    private PlacementResult<SelectedRole, PlacementKey> registerChoice(
            BalanceContext context, RoleType role, String group, PrefillCandidateSet candidates,
            PrefillCandidateRanking.Choice choice, int workers) {
        long seqLen = context.getRequest().getSeqLen();
        FlexlbConfig config = context.getConfig();
        while (choice.index() >= 0) {
            int index = choice.index();
            PrefillEndpoint endpoint = candidates.endpoint(index);
            String address = candidates.endpointAddress(index);
            String block = candidates.admissionBlock(index);
            boolean directCapacityFull = block == null && config.isDirect() && !endpoint.canAcceptRequest();
            WorkerEndpoint.GenerationPin pin = workerDirectory.captureEndpoint(role, address);
            if (pin == null || pin.endpoint() != endpoint) {
                if (pin != null) { pin.close(); }
                return PlacementResult.blocked(new PlacementKey(role, group));
            }
            PrefillTokenWorkload.Reservation work = null;
            HotspotRedirectLimiter.Ticket budget = null;
            boolean transferred = false;
            try {
                if (block != null || directCapacityFull) {
                    Response failure = block == null ? admissionFailure(endpoint, context.getPriority())
                            : Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                    return PlacementResult.blocked(PlacementKey.exact(role,
                            endpoint.getStatus().topologySnapshot().group(), address), failure, Map.of(
                            "role", role.name(), "workers", workers, "observedAtMs", System.currentTimeMillis(),
                            "selectedEndpoint", address, "cacheHitTokens", candidates.cacheHit(index),
                            "projectedTtftMs", RouteProjection.Candidate.UNKNOWN,
                            "reason", block == null ? "PREFILL_CAPACITY" : block));
                }

                long hit = candidates.cacheHit(index);
                long duration = candidates.prefillMs(index);
                work = endpoint.tryPlanTokenWork(choice.expectedVersion(candidates),
                        Math.max(0L, seqLen - hit), duration);
                if (work == null) { return PlacementResult.replan(); }
                if (choice.redirect()) {
                    var release = config.getRouter().getRoles().getPrefill().getHotspotRelease();
                    budget = redirectLimiter.tryAcquire(candidates.maximumCacheHit - hit, release.getMaxExtraComputePermille());
                    if (budget == null) {
                        choice = choice.rateLimited();
                        continue;
                    }
                }

                long ttft = candidates.projectedTtftMs(index);
                var status = prefillStatus(endpoint, role, context.getRequestId(), ttft, hit);
                transferred = true; // The factory consumes all three resources, including on failure.
                SelectedRole selected = SelectedRole.prefill(pin, status, duration, candidates.ownershipVersion(index),
                        candidates.routingCacheMatchTokens(index), work, budget);
                try {
                    ranking.report(candidates, role, group, index, choice.reason());
                    reportSelectedEstimates(role, endpoint, config, ttft, duration);
                    reportCacheHitMetrics(role, hit, seqLen);
                    reportRoutingCacheMatchMetrics(role, candidates.routingCacheMatchTokens(index),
                            candidates.maximumRoutingCacheMatchTokens, seqLen);
                    return PlacementResult.success(selected);
                } catch (RuntimeException | Error failure) {
                    try (selected) { throw failure; }
                }
            } finally {
                if (!transferred) {
                    try (var ownedPin = pin; var ownedWork = work; var ownedBudget = budget) {
                        // Release this exact attempt before retrying or selecting the fallback.
                    }
                }
            }
        }
        return PlacementResult.blocked(new PlacementKey(role, group));
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
            var workload = ep.tokenWorkloadSnapshot();
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
                    projectionInputs.ownershipVersion(), workload);
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

    private static ServerStatus prefillStatus(PrefillEndpoint ep, RoleType roleType, long requestId,
                                              long projectedTtftMs, long bestCacheHit) {
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);
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
        return result;
    }
}
