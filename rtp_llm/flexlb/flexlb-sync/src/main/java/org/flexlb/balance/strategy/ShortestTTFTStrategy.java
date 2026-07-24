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

    /** Internal record holding TTFT score and cache hit for a single endpoint. */
    private record ScoredEndpoint(PrefillEndpoint ep, long ttft, long hitCache) {}

    // ==================== Core Selection ====================

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<PrefillEndpoint> eligible = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.warn("ShortestTTFT select failed: no available endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        // Score all eligible endpoints by TTFT
        List<ScoredEndpoint> scoredEndpoints = scoreEndpoints(eligible, cacheMatchResults, seqLen);
        if (scoredEndpoints.isEmpty()) {
            Logger.warn("ShortestTTFT select failed: no scored endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Sort by TTFT ascending; secondary sort by lastSelectedTime for determinism
        scoredEndpoints.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(se -> se.ep().getLastSelectedTime().get()));

        // Select candidate pool (top-N by TTFT)
        int candidateCount = config.resolveShortestTtftCandidateCount(scoredEndpoints.size());
        List<ScoredEndpoint> candidates = scoredEndpoints.subList(0, Math.min(candidateCount, scoredEndpoints.size()));

        // CAS fairness: prefer the least-recently-selected candidate
        ScoredEndpoint selected = selectByFairness(candidates);
        if (selected == null) {
            // All CAS attempts failed; fall back to the lowest-TTFT candidate
            selected = candidates.getFirst();
            Logger.debug("ShortestTTFT: all CAS failed, falling back to lowest-TTFT endpoint, ip={}", selected.ep().getIp());
        }

        Logger.debug("ShortestTTFT selected endpoint - ip: {}, port: {}, ttft: {}, hitCache: {}",
                selected.ep().getIp(), selected.ep().getHttpPort(), selected.ttft(), selected.hitCache());

        reportCacheHitMetrics(roleType, selected.ep().getIp(), selected.ep().ipPort(), selected.hitCache(), seqLen);

        return buildServerStatus(selected, roleType, requestId, config);
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
    private ScoredEndpoint selectByFairness(List<ScoredEndpoint> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }
        if (candidates.size() == 1) {
            // Single candidate — still update lastSelectedTime for future fairness
            long now = System.nanoTime() / 1000;
            candidates.getFirst().ep().getLastSelectedTime().set(now);
            return candidates.getFirst();
        }

        // Sort ascending by lastSelectedTime so the least recently used worker is tried first
        List<ScoredEndpoint> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingLong(se -> se.ep().getLastSelectedTime().get()));

        long now = System.nanoTime() / 1000;
        for (ScoredEndpoint candidate : sorted) {
            long expected = candidate.ep().getLastSelectedTime().get();
            // CAS: claim this worker only if lastSelectedTime hasn't changed since we read it.
            // A failed CAS means another concurrent request already claimed this worker.
            if (candidate.ep().getLastSelectedTime().compareAndSet(expected, now)) {
                return candidate;
            }
            // Another request claimed this worker; try the next candidate
        }

        // All candidates were claimed concurrently
        return null;
    }

    // ==================== Scoring ====================

    /**
     * Calculate TTFT scores for all eligible endpoints.
     *
     * <p>TTFT = predicted prefill time + estimated queue wait time.
     *
     * @param endpoints eligible endpoint list
     * @param cacheMatchResults cache match results from {@link CacheAwareService}
     * @param seqLen request sequence length
     * @return list of scored endpoints
     */
    private List<ScoredEndpoint> scoreEndpoints(List<PrefillEndpoint> endpoints,
                                                Map<String, Integer> cacheMatchResults,
                                                long seqLen) {
        List<ScoredEndpoint> result = new ArrayList<>(endpoints.size());
        for (PrefillEndpoint ep : endpoints) {
            PrefillTimePredictor predictor = ep.getPredictor();
            long cacheHit = calculateCacheHit(ep, cacheMatchResults, seqLen);
            long prefillMs = predictor.estimateMs(seqLen, cacheHit);
            long queueMs = ep.realWaitTimeMs();
            long ttft = prefillMs + queueMs;
            Logger.debug("ShortestTTFT score - ip: {}, hitCache: {}, prefillMs: {}, queueMs: {}, ttft: {}",
                    ep.getIp(), cacheHit, prefillMs, queueMs, ttft);
            result.add(new ScoredEndpoint(ep, ttft, cacheHit));
        }
        return result;
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

    private long calculateCacheHit(PrefillEndpoint ep, Map<String, Integer> cacheMatchResults, long seqLen) {
        if (ep.getStatus().getCacheStatus() == null || cacheMatchResults == null) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null) {
            return 0L;
        }
        long blockSize = ep.getStatus().getCacheStatus().getBlockSize();
        long rawHit = blockSize * prefixMatchLength;
        if (rawHit >= seqLen) {
            return Math.max(0L, seqLen - blockSize);
        }
        return Math.max(0L, rawHit);
    }

    // ==================== Metrics & ServerStatus (mirrors CostBasedPrefillStrategy) ====================

    private void reportCacheHitMetrics(RoleType roleType, String ip, String engineIpPort, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, engineIpPort, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(ScoredEndpoint selected, RoleType roleType, long requestId,
                                           FlexlbConfig config) {
        PrefillEndpoint ep = selected.ep();
        long ttft = selected.ttft();
        long bestCacheHit = selected.hitCache();

        // Non-batch path: reserve prefill inflight for load-aware scoring.
        // Batch path uses FlexlbBatchScheduler.commitBatch() instead — skip here to avoid double-counting.
        if (isNonBatchPath(config)) {
            ep.commitBatch(requestId, ttft, Collections.emptyList());
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

    /**
     * Whether batch dispatching is globally disabled.
     * <p>When batch is enabled, FlexlbBatchScheduler handles all inflight tracking;
     * placeholders are only needed when batch is fully off ({@code flexlbBatchEnabled=false}).
     */
    private static boolean isNonBatchPath(FlexlbConfig config) {
        return !config.isFlexlbBatchEnabled();
    }
}
