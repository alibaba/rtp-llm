package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Load balancing strategy based on shortest Time-To-First-Token (TTFT).
 *
 * <p>This strategy selects the optimal worker by considering the following factors:
 * 1. KV-Cache hit rate: Prioritize workers with higher cache hit rates
 * 2. Queue time: Consider the current task queue status of workers
 * 3. Scheduling fairness: Achieve load balancing among workers with similar performance
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Score all eligible endpoints by TTFT = prefillTime + queueTime</li>
 *   <li>Sort by TTFT ascending, take top-N as the candidate pool</li>
 *   <li>Within the pool, use CAS on {@code lastSelectedTime} to pick the
 *       least-recently-selected worker, ensuring concurrent requests spread
 *       across different workers</li>
 *   <li>If all CAS attempts fail, fall back to the lowest-TTFT candidate</li>
 * </ol>
 *
 * <p>Supports DIRECT and QUEUE scheduling with either dispatcher. Common
 * scheduler paths include the worker-batcher wait while their inflight
 * lifecycle remains owned by {@code PriorityScheduler}; DIRECT and FIFO
 * non-batch queue routing reserve locally.
 */
@Component("shortestTtftStrategy")
public class ShortestTTFTStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;
    private final SessionPlacementStore sessionPlacementStore;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory,
                                EngineHealthReporter engineHealthReporter,
                                SessionPlacementStore sessionPlacementStore) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        this.sessionPlacementStore = sessionPlacementStore;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("{} select failed", LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(), e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(WorkerEndpoint ep, long requestId) {
        // Release non-batch prefill inflight reservation on routing failure.
        // Batch path inflight is managed by PriorityScheduler — no-op here.
        if (ep instanceof PrefillEndpoint pe) {
            pe.releaseBatch(requestId);
        }
    }

    /** Internal record holding TTFT score and cache hit for a single endpoint. */
    protected record ScoredEndpoint(PrefillEndpoint ep,
                                    long ttft,
                                    long hitCache,
                                    long prefillMs,
                                    long lastSelectedTime) {}

    // ==================== Core Selection ====================

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<PrefillEndpoint> eligible = getAvailableEndpoints(
                roleType,
                group,
                config.resourceMeasureFor(roleType),
                balanceContext.getExcludedPrefillIpPort());
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.debug("ShortestTTFT select failed: no available endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        // Score all eligible endpoints by TTFT
        List<ScoredEndpoint> scoredEndpoints = scoreEndpoints(
                eligible, cacheMatchResults, balanceContext);
        if (scoredEndpoints.isEmpty()) {
            Logger.debug("ShortestTTFT select failed: no scored endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        long candidateMaxHitTokens = scoredEndpoints.stream()
                .mapToLong(scored -> calculateRoutingCacheMatchTokens(
                        scored.ep(), cacheMatchResults, balanceContext.getRequest()))
                .max()
                .orElse(0L);

        // Sort by TTFT ascending; secondary sort by the selection-time snapshot for determinism.
        scoredEndpoints.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));

        ScoredEndpoint selected = selectBestEndpoint(
                scoredEndpoints, balanceContext, roleType, group, seqLen, config);
        if (selected == null) {
            Logger.debug("{} select failed: no selectable endpoint, request_id={}",
                    LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(), requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Logger.debug("{} selected endpoint - ip: {}, port: {}, ttft: {}, hitCache: {}",
                LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(),
                selected.ep().getIp(), selected.ep().getHttpPort(),
                selected.ttft(), selected.hitCache());

        reportSelectedEstimates(roleType, selected, config);
        reportCacheHitMetrics(roleType, selected.hitCache(), seqLen);
        reportRoutingCacheMatchMetrics(
                roleType,
                calculateRoutingCacheMatchTokens(
                        selected.ep(), cacheMatchResults, balanceContext.getRequest()),
                candidateMaxHitTokens,
                seqLen);

        return buildServerStatus(selected, roleType, requestId, balanceContext);
    }

    /**
     * Select the final endpoint from a list already sorted by TTFT.
     * Subclasses may override this decision while reusing filtering, cache lookup,
     * scoring, metrics, and response construction.
     */
    protected ScoredEndpoint selectBestEndpoint(List<ScoredEndpoint> scoredEndpoints,
                                                BalanceContext balanceContext,
                                                RoleType roleType,
                                                String group,
                                                long seqLen,
                                                FlexlbConfig config) {
        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        CacheAffinityPolicy.Decision affinity = null;
        long minTtftMs = scoredEndpoints.getFirst().ttft();
        if (cacheAffinity != null) {
            long referenceHitTokens = scoredEndpoints.stream()
                    .filter(candidate -> candidate.ttft() == minTtftMs)
                    .mapToLong(ScoredEndpoint::hitCache)
                    .max()
                    .orElse(0L);
            affinity = CacheAffinityPolicy.evaluate(
                    scoredEndpoints.size(),
                    index -> scoredEndpoints.get(index).ttft(),
                    index -> scoredEndpoints.get(index).hitCache(),
                    minTtftMs,
                    referenceHitTokens,
                    seqLen,
                    cacheAffinity.getMaxExtraTtftMs(),
                    cacheAffinity.getMinPrefixHitPercent());
        }

        if (affinity != null && affinity.hasPreference()) {
            List<ScoredEndpoint> preferenceOrder = new ArrayList<>(affinity.preferredCount());
            for (int i = 0; i < affinity.preferredCount(); i++) {
                preferenceOrder.add(scoredEndpoints.get(affinity.preferredIndex(i)));
            }
            ScoredEndpoint selected = selectFirstWithoutConcurrentConflict(preferenceOrder);
            if (selected == null) {
                selected = selectBaselineEndpoint(refreshSelectionSnapshots(scoredEndpoints), config);
                if (selected != null) {
                    reportCacheAffinityDecision(
                            roleType, selected.ep().getIp(), "CACHE_AFFINITY_FALLBACK");
                    SessionAffinityPolicy.reportDecision(balanceContext, roleType,
                            engineHealthReporter,
                            SessionAffinityPolicy.Reason.CACHE_AFFINITY_PRECEDENCE);
                }
                return selected;
            }
            String reason = selected.equals(preferenceOrder.getFirst())
                    ? CacheAffinityPolicy.Reason.CACHE_LEADER.name()
                    : "CACHE_AFFINITY_FALLBACK";
            reportCacheAffinityDecision(roleType, selected.ep().getIp(), reason);
            SessionAffinityPolicy.reportDecision(balanceContext, roleType,
                    engineHealthReporter,
                    SessionAffinityPolicy.Reason.CACHE_AFFINITY_PRECEDENCE);
            return selected;
        }

        RoutingConfig.SessionAffinityConfig sessionConfig = config.getRouter().getRoles()
                .getPrefill().getSessionAffinity();
        SessionAffinityPolicy.Decision sessionAffinity = SessionAffinityPolicy.evaluate(
                balanceContext.getRequest(), sessionConfig, sessionPlacementStore,
                scoredEndpoints.size(),
                index -> scoredEndpoints.get(index).ep().ipPort(),
                index -> scoredEndpoints.get(index).ttft(), minTtftMs);
        if (sessionAffinity.hasPreference()) {
            ScoredEndpoint preferred = scoredEndpoints.get(sessionAffinity.preferredIndex());
            ScoredEndpoint selected = selectFirstWithoutConcurrentConflict(List.of(preferred));
            if (selected != null) {
                SessionAffinityPolicy.reportDecision(
                        balanceContext, roleType, engineHealthReporter, sessionAffinity.reason());
                return selected;
            }
            SessionAffinityPolicy.reportDecision(balanceContext, roleType,
                    engineHealthReporter,
                    SessionAffinityPolicy.Reason.SESSION_AFFINITY_CAS_FALLBACK);
            return selectBaselineEndpoint(refreshSelectionSnapshots(scoredEndpoints), config);
        }
        SessionAffinityPolicy.reportDecision(
                balanceContext, roleType, engineHealthReporter, sessionAffinity.reason());

        ScoredEndpoint selected = selectBaselineEndpoint(scoredEndpoints, config);
        if (selected != null && affinity != null) {
            reportCacheAffinityDecision(roleType, selected.ep().getIp(), affinity.reason().name());
        }
        return selected;
    }

    /** Refresh CAS snapshots after an affinity claim loses to concurrent schedulers. */
    private List<ScoredEndpoint> refreshSelectionSnapshots(
            List<ScoredEndpoint> scoredEndpoints) {
        List<ScoredEndpoint> refreshed = new ArrayList<>(scoredEndpoints.size());
        for (ScoredEndpoint scored : scoredEndpoints) {
            refreshed.add(new ScoredEndpoint(
                    scored.ep(),
                    scored.ttft(),
                    scored.hitCache(),
                    scored.prefillMs(),
                    scored.ep().getLastSelectedTime().get()));
        }
        refreshed.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));
        return refreshed;
    }

    /** Preserve the original candidate-pool and CAS fairness behavior. */
    private ScoredEndpoint selectBaselineEndpoint(
            List<ScoredEndpoint> scoredEndpoints, FlexlbConfig config) {
        int candidateCount = config.shortestTtftCandidateCount(scoredEndpoints.size());
        List<ScoredEndpoint> candidates = scoredEndpoints.subList(
                0, Math.min(candidateCount, scoredEndpoints.size()));

        ScoredEndpoint selected = selectByFairness(candidates);
        if (selected != null) {
            return selected;
        }

        ScoredEndpoint fallback = candidates.getFirst();
        Logger.debug("ShortestTtft: all CAS claims failed, falling back to lowest-TTFT endpoint, ip={}",
                fallback.ep().getIp());
        return fallback;
    }

    /**
     * Select worker based on scheduling fairness.
     *
     * <p>Among the candidate pool, prefer the least-recently-selected worker.
     * CAS on {@code lastSelectedTime} ensures concurrent requests are spread
     * across different workers rather than all landing on the same one.
     *
     * @param candidates candidate pool (already sorted by TTFT ascending)
     * @return selected endpoint, or {@code null} if all CAS attempts failed
     */
    protected ScoredEndpoint selectByFairness(List<ScoredEndpoint> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }

        // Sort ascending by lastSelectedTime so the least recently used worker is tried first
        List<ScoredEndpoint> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingLong(ScoredEndpoint::lastSelectedTime));
        return selectFirstWithoutConcurrentConflict(sorted);
    }

    /**
     * Claim the first endpoint whose selection timestamp still matches the scoring snapshot.
     * A failed claim means another scheduler made a decision from the same snapshot.
     */
    protected ScoredEndpoint selectFirstWithoutConcurrentConflict(List<ScoredEndpoint> selectionOrder) {
        long now = System.nanoTime() / 1000;
        for (ScoredEndpoint candidate : selectionOrder) {
            long expected = candidate.lastSelectedTime();
            if (expected == Long.MAX_VALUE) {
                continue;
            }
            long claimedAt = Math.max(now, expected + 1L);
            if (candidate.ep().getLastSelectedTime().compareAndSet(expected, claimedAt)) {
                return candidate;
            }
        }
        return null;
    }

    protected void reportCacheAffinityDecision(RoleType roleType,
                                               String engineIp,
                                               String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    // ==================== Scoring ====================

    /**
     * Calculate TTFT scores for all eligible endpoints.
     *
     * <p>TTFT = predicted prefill time + estimated queue wait time.
     * Endpoints without a predictor are skipped.
     *
     * @param endpoints eligible endpoint list
     * @param cacheMatchResults cache match results from {@link CacheAwareService}
     * @param balanceContext request and scheduling context
     * @return list of scored endpoints
     */
    private List<ScoredEndpoint> scoreEndpoints(List<PrefillEndpoint> endpoints,
                                                Map<String, Integer> cacheMatchResults,
                                                BalanceContext balanceContext) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        List<ScoredEndpoint> result = new ArrayList<>(endpoints.size());
        for (PrefillEndpoint ep : endpoints) {
            PrefillTimePredictor predictor = ep.getPredictor();
            if (predictor == null) {
                Logger.debug("ShortestTTFT: skipping endpoint without predictor, ip={}", ep.getIp());
                continue;
            }
            long cacheHit = calculateCacheHit(ep, cacheMatchResults, request);
            long prefillMs = Math.max(0L, predictor.estimateMs(seqLen, cacheHit));
            long queueMs = estimatedQueueWaitMs(ep, balanceContext);
            if (queueMs == Long.MAX_VALUE) {
                Logger.debug("ShortestTTFT: skipping endpoint with unavailable wait estimate, ip={}",
                        ep.getIp());
                continue;
            }
            long ttft = saturatingAdd(prefillMs, queueMs);
            Logger.debug("ShortestTTFT score - ip: {}, hitCache: {}, prefillMs: {}, queueMs: {}, ttft: {}",
                    ep.getIp(), cacheHit, prefillMs, queueMs, ttft);
            result.add(new ScoredEndpoint(
                    ep, ttft, cacheHit, prefillMs, ep.getLastSelectedTime().get()));
        }
        return result;
    }

    /**
     * Estimate all work already ahead of a request. Queue-scheduler-owned paths include
     * both dispatched inflight work and the per-worker batcher queue; otherwise only the
     * inflight ledger is relevant. Long.MAX_VALUE remains an unavailable sentinel.
     */
    protected long estimatedQueueWaitMs(PrefillEndpoint ep, BalanceContext balanceContext) {
        long inflightWaitMs = ep.realWaitTimeMs();
        if (inflightWaitMs == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        inflightWaitMs = Math.max(0L, inflightWaitMs);
        if (!queueSchedulerOwnsRequest(balanceContext)) {
            return inflightWaitMs;
        }

        FlexlbConfig config = balanceContext.getConfig();
        long batcherWaitMs = config.isPriorityOrdering()
                ? ep.batcherEstimatedWaitMs(
                        balanceContext.getPriority(),
                        balanceContext.getRequestId())
                : ep.batcherWaitMs();
        batcherWaitMs = Math.max(0L, batcherWaitMs);
        return saturatingAdd(inflightWaitMs, batcherWaitMs);
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

    // ==================== Endpoint Filtering (mirrors CostBasedPrefillStrategy) ====================

    private List<PrefillEndpoint> getAvailableEndpoints(RoleType roleType,
                                                        String group,
                                                        ResourceMeasureIndicatorEnum indicator,
                                                        String excludedIpPort) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new ArrayList<>();
        }
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new ArrayList<>();
        }
        List<PrefillEndpoint> result = new ArrayList<>();
        PrefillEndpoint excludedEligible = null;
        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!(ep instanceof PrefillEndpoint pe)) {
                continue;
            }
            if (!pe.getStatus().isAlive()) {
                continue;
            }
            if (!measure.isResourceAvailable(pe)) {
                continue;
            }
            if (excludedIpPort != null && excludedIpPort.equals(pe.ipPort())) {
                excludedEligible = pe;
                continue;
            }
            result.add(pe);
        }
        if (result.isEmpty() && excludedEligible != null) {
            result.add(excludedEligible);
        }
        return result;
    }

    private Map<String, Integer> getCacheMatchResults(BalanceContext balanceContext, RoleType roleType, String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    private long calculateCacheHit(PrefillEndpoint ep,
                                   Map<String, Integer> cacheMatchResults,
                                   Request request) {
        if (cacheMatchResults == null || request == null) {
            return 0L;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return 0L;
        }
        long blockSize = request.getCacheKeyBlockSize();
        if (blockSize <= 0L && ep.getStatus().getCacheStatus() != null) {
            blockSize = ep.getStatus().getCacheStatus().getBlockSize();
        }
        if (blockSize <= 0L) {
            return 0L;
        }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            rawHit = seqLen;
        }
        if (rawHit >= seqLen) {
            return Math.max(0L, seqLen - blockSize);
        }
        return Math.max(0L, rawHit);
    }

    private long calculateRoutingCacheMatchTokens(PrefillEndpoint ep,
                                                  Map<String, Integer> cacheMatchResults,
                                                  Request request) {
        if (ep == null || cacheMatchResults == null || request == null || request.getSeqLen() <= 0L) {
            return 0L;
        }

        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return 0L;
        }

        long blockSize = request.getCacheKeyBlockSize();
        if (blockSize <= 0L && ep.getStatus().getCacheStatus() != null) {
            blockSize = ep.getStatus().getCacheStatus().getBlockSize();
        }
        if (blockSize <= 0L) {
            return 0L;
        }

        long hitTokens;
        try {
            hitTokens = Math.multiplyExact(blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            return request.getSeqLen();
        }
        return Math.min(request.getSeqLen(), hitTokens);
    }

    // ==================== Metrics & ServerStatus (mirrors CostBasedPrefillStrategy) ====================

    private void reportCacheHitMetrics(RoleType roleType, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, hitCacheTokens, hitRate);
    }

    private void reportSelectedEstimates(RoleType roleType,
                                         ScoredEndpoint selected,
                                         FlexlbConfig config) {
        String deliveryMode = config.isBatchDispatch() ? "BATCH" : "NON_BATCH";
        try {
            engineHealthReporter.reportPrefillSelectedEstimates(
                    roleType,
                    selected.ep().getIp(),
                    deliveryMode,
                    selected.ttft(),
                    selected.prefillMs());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    selected.ep().ipPort(), deliveryMode, telemetryFailure);
        }
    }

    private void reportRoutingCacheMatchMetrics(RoleType roleType,
                                                long selectedHitTokens,
                                                long candidateMaxHitTokens,
                                                long totalTokens) {
        engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(
                roleType, selectedHitTokens, totalTokens);
        engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, candidateMaxHitTokens);
    }

    private ServerStatus buildServerStatus(ScoredEndpoint selected,
                                           RoleType roleType,
                                           long requestId,
                                           BalanceContext balanceContext) {
        PrefillEndpoint ep = selected.ep();
        long ttft = selected.ttft();
        long bestCacheHit = selected.hitCache();

        // DIRECT owns its reservation here; QUEUE owns reservations in the scheduler.
        if (strategyOwnsInflightTracking(balanceContext)) {
            ep.commitBatch(requestId, selected.prefillMs(), Collections.emptyList());
        }

        // Populate DebugInfo so BatchItem.hitCache() can read hitCacheLen for batch metrics
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);

        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(ttft);
        result.setGroup(ep.getStatus().getGroup());
        result.setServerIp(ep.getIp());
        result.setHttpPort(ep.getHttpPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(ep.getHttpPort()));
        result.setDpRank(ep.getStatus().getDpRank());
        result.setDebugInfo(debugInfo);
        return result;
    }

    /** Whether this request is owned by the common queue/batch scheduler. */
    private static boolean queueSchedulerOwnsRequest(BalanceContext balanceContext) {
        if (balanceContext == null || balanceContext.getConfig() == null) {
            return false;
        }
        FlexlbConfig config = balanceContext.getConfig();
        return config.isQueue();
    }

    /** Whether this strategy, rather than PriorityScheduler, owns Prefill accounting. */
    private static boolean strategyOwnsInflightTracking(BalanceContext balanceContext) {
        return !queueSchedulerOwnsRequest(balanceContext);
    }
}
