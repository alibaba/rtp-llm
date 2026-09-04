package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.ScheduleModeEnum;
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
 * <p>Intended for the non-batch routing path (Direct/Queue).
 * Batch path inflight is managed by {@code FlexlbBatchScheduler}.
 */
@Component("shortestTtftStrategy")
public class ShortestTTFTStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory,
                                EngineHealthReporter engineHealthReporter) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("ShortestTTFTStrategy select failed", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(WorkerEndpoint ep, long requestId) {
        // Release non-batch prefill inflight reservation on routing failure.
        // Batch path inflight is managed by FlexlbBatchScheduler — no-op here.
        if (ep instanceof PrefillEndpoint pe) {
            pe.releaseBatch(requestId);
        }
    }

    /** Selection-time snapshot for one endpoint. */
    private record ScoredEndpoint(PrefillEndpoint ep,
                                  long ttft,
                                  long hitCache,
                                  long prefillMs,
                                  long lastSelectedTime) {}

    // ==================== Core Selection ====================

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<PrefillEndpoint> eligible = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.debug("ShortestTTFT select failed: no available endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        // Score all eligible endpoints by TTFT.
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

        // Sort by TTFT ascending; the snapshot keeps concurrent fairness decisions coherent.
        scoredEndpoints.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));

        ScoredEndpoint selected = selectBestEndpoint(
                scoredEndpoints, roleType, group, seqLen, config);
        if (selected == null) {
            Logger.debug("ShortestTTFT select failed: no selectable endpoint, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Logger.debug("ShortestTTFT selected endpoint - ip: {}, port: {}, ttft: {}, hitCache: {}",
                selected.ep().getIp(), selected.ep().getHttpPort(), selected.ttft(), selected.hitCache());

        reportSelectedEstimates(roleType, selected, balanceContext);
        reportCacheHitMetrics(roleType, selected.hitCache(), seqLen);
        reportRoutingCacheMatchMetrics(
                roleType,
                calculateRoutingCacheMatchTokens(
                        selected.ep(), cacheMatchResults, balanceContext.getRequest()),
                candidateMaxHitTokens,
                seqLen);

        return buildServerStatus(selected, roleType, requestId, balanceContext);
    }

    private ScoredEndpoint selectBestEndpoint(List<ScoredEndpoint> scoredEndpoints,
                                               RoleType roleType,
                                               String group,
                                               long seqLen,
                                               FlexlbConfig config) {
        if (!config.isCacheAffinityEnabled()) {
            return selectBaselineEndpoint(scoredEndpoints, config);
        }

        long minTtftMs = scoredEndpoints.getFirst().ttft();
        long referenceHitTokens = scoredEndpoints.stream()
                .filter(candidate -> candidate.ttft() == minTtftMs)
                .mapToLong(ScoredEndpoint::hitCache)
                .max()
                .orElse(0L);
        CacheAffinityPolicy.Decision affinity = CacheAffinityPolicy.evaluate(
                scoredEndpoints.size(),
                index -> scoredEndpoints.get(index).ttft(),
                index -> scoredEndpoints.get(index).hitCache(),
                minTtftMs,
                referenceHitTokens,
                seqLen,
                config.getCacheAffinityMaxExtraTtftMs(),
                config.getCacheAffinityMinHitRate());

        ScoredEndpoint selected;
        String reason;
        if (affinity.hasPreference()) {
            List<ScoredEndpoint> preferenceOrder = new ArrayList<>(affinity.preferredCount());
            for (int i = 0; i < affinity.preferredCount(); i++) {
                preferenceOrder.add(scoredEndpoints.get(affinity.preferredIndex(i)));
            }
            selected = selectFirstWithoutConcurrentConflict(preferenceOrder);
            if (selected == null) {
                selected = selectBaselineEndpoint(
                        refreshSelectionSnapshots(scoredEndpoints), config);
                reason = "CACHE_AFFINITY_FALLBACK";
            } else {
                reason = selected.equals(preferenceOrder.getFirst())
                        ? CacheAffinityPolicy.Reason.CACHE_LEADER.name()
                        : "CACHE_AFFINITY_FALLBACK";
            }
        } else {
            selected = selectBaselineEndpoint(scoredEndpoints, config);
            reason = affinity.reason().name();
        }

        if (selected != null) {
            reportCacheAffinityDecision(roleType, selected.ep(), reason);
            Logger.debug(
                    "ShortestTTFT cache-affinity decision - role: {}, group: {}, selected: {}, "
                            + "minTtftMs: {}, selectedTtftMs: {}, ttftCutoffMs: {}, "
                            + "hitTokens: {}, reason: {}",
                    roleType, group, selected.ep().ipPort(), affinity.minScoreMs(),
                    selected.ttft(), affinity.scoreCutoffMs(), selected.hitCache(), reason);
        }
        return selected;
    }

    private List<ScoredEndpoint> refreshSelectionSnapshots(
            List<ScoredEndpoint> scoredEndpoints) {
        List<ScoredEndpoint> refreshed = new ArrayList<>(scoredEndpoints.size());
        for (ScoredEndpoint scored : scoredEndpoints) {
            refreshed.add(new ScoredEndpoint(
                    scored.ep(), scored.ttft(), scored.hitCache(), scored.prefillMs(),
                    scored.ep().getLastSelectedTime().get()));
        }
        refreshed.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));
        return refreshed;
    }

    /** Preserve the original candidate-pool and CAS fairness behavior. */
    private ScoredEndpoint selectBaselineEndpoint(
            List<ScoredEndpoint> scoredEndpoints, FlexlbConfig config) {
        int candidateCount = config.resolveShortestTtftCandidateCount(scoredEndpoints.size());
        List<ScoredEndpoint> candidates = scoredEndpoints.subList(
                0, Math.min(candidateCount, scoredEndpoints.size()));
        ScoredEndpoint selected = selectByFairness(candidates);
        if (selected != null) {
            return selected;
        }
        ScoredEndpoint fallback = candidates.getFirst();
        Logger.debug("ShortestTTFT: all CAS failed, falling back to lowest-TTFT endpoint, ip={}",
                fallback.ep().getIp());
        return fallback;
    }

    /** Select the least-recently-used candidate using its selection-time CAS snapshot. */
    private ScoredEndpoint selectByFairness(List<ScoredEndpoint> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }
        List<ScoredEndpoint> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingLong(ScoredEndpoint::lastSelectedTime));
        return selectFirstWithoutConcurrentConflict(sorted);
    }

    private ScoredEndpoint selectFirstWithoutConcurrentConflict(
            List<ScoredEndpoint> selectionOrder) {
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

    // ==================== Scoring ====================

    /**
     * Calculate TTFT scores for all eligible endpoints.
     *
     * <p>TTFT = predicted prefill time + estimated queue wait time.
     * Endpoints without a predictor are skipped.
     *
     * @param endpoints eligible endpoint list
     * @param cacheMatchResults cache match results from {@link CacheAwareService}
     * @param seqLen request sequence length
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
            long ttft = saturatingAdd(prefillMs, queueMs);
            Logger.debug("ShortestTTFT score - ip: {}, hitCache: {}, prefillMs: {}, queueMs: {}, ttft: {}",
                    ep.getIp(), cacheHit, prefillMs, queueMs, ttft);
            result.add(new ScoredEndpoint(
                    ep, ttft, cacheHit, prefillMs, ep.getLastSelectedTime().get()));
        }
        return result;
    }

    private long estimatedQueueWaitMs(PrefillEndpoint ep, BalanceContext balanceContext) {
        long inflightWaitMs = Math.max(0L, ep.realWaitTimeMs());
        if (!isBatchPath(balanceContext)) {
            return inflightWaitMs;
        }
        FlexlbConfig config = balanceContext.getConfig();
        long batcherWaitMs = config != null && config.isAutoTpmEnabled()
                ? ep.batcherEstimatedWaitMs(
                        balanceContext.getPriority(),
                        balanceContext.getDeadlineMs(),
                        balanceContext.getRequestId())
                : ep.batcherWaitMs();
        return saturatingAdd(inflightWaitMs, Math.max(0L, batcherWaitMs));
    }

    // ==================== Endpoint Filtering (mirrors CostBasedPrefillStrategy) ====================

    private List<PrefillEndpoint> getAvailableEndpoints(RoleType roleType, String group,
                                                        ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new ArrayList<>();
        }
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new ArrayList<>();
        }
        List<PrefillEndpoint> result = new ArrayList<>();
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
            result.add(pe);
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
        if (cacheMatchResults == null || request == null || request.getSeqLen() <= 0L) {
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
            rawHit = request.getSeqLen();
        }
        if (rawHit >= request.getSeqLen()) {
            return Math.max(0L, request.getSeqLen() - blockSize);
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

        long hitTokens = blockSize * prefixMatchLength;
        if (hitTokens < 0L) {
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
                                         BalanceContext balanceContext) {
        String deliveryMode = isBatchPath(balanceContext) ? "BATCH" : "NON_BATCH";
        try {
            engineHealthReporter.reportPrefillSelectedEstimates(
                    roleType, selected.ep().getIp(), deliveryMode,
                    selected.ttft(), selected.prefillMs());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    selected.ep().ipPort(), deliveryMode, telemetryFailure);
        }
    }

    private void reportCacheAffinityDecision(RoleType roleType,
                                             PrefillEndpoint endpoint,
                                             String decision) {
        try {
            engineHealthReporter.reportCacheAffinityDecision(
                    roleType, endpoint.getIp(), decision);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("Cache-affinity metric failed: engine={}, decision={}",
                    endpoint.ipPort(), decision, telemetryFailure);
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

        // Non-batch path: reserve only execution work; queue wait is not new work.
        if (!isBatchPath(balanceContext)) {
            ep.commitBatch(requestId, selected.prefillMs(), Collections.emptyList());
        }

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

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }

    /** Request-level mode is authoritative because configured BATCH may downgrade to DIRECT. */
    private static boolean isBatchPath(BalanceContext balanceContext) {
        return balanceContext != null
                && balanceContext.getScheduleMode() == ScheduleModeEnum.BATCH;
    }
}
