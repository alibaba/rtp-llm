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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

@Component("costBasedPrefillStrategy")
public class CostBasedPrefillStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;

    public CostBasedPrefillStrategy(EngineWorkerStatus engineWorkerStatus,
                                    CacheAwareService cacheAwareService,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EngineHealthReporter engineHealthReporter) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.COST_BASED_PREFILL, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("CostBasedPrefillStrategy select failed", e);
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

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        EndpointFilterResult filterResult = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        List<PrefillEndpoint> eligible = filterResult.endpoints();
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.warn("Prefill select failed: no available endpoints, request_id={}, rejections={}",
                    requestId, filterResult.rejections());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        FilterResult hardFilterResult = applyHardFilters(eligible, seqLen, config, cacheMatchResults);
        List<PrefillEndpoint> survivors = hardFilterResult.endpoints();

        PrefillEndpoint best = null;
        long bestScore = Long.MAX_VALUE;
        long bestCacheHit = 0;

        // First pass: compute scores and cache results, find minScore
        int survivorsSize = survivors.size();
        long[] scores = new long[survivorsSize];
        long[] cacheHits = new long[survivorsSize];
        long minScore = Long.MAX_VALUE;
        for (int i = 0; i < survivorsSize; i++) {
            PrefillEndpoint ep = survivors.get(i);
            long cacheHit = calculateCacheHit(ep, cacheMatchResults, seqLen);
            long score = computeScore(ep, cacheHit, seqLen);
            cacheHits[i] = cacheHit;
            scores[i] = score;
            if (score < minScore) {
                minScore = score;
            }
        }

        // Second pass: collect tied endpoints using cached scores (no re-computation)
        if (minScore != Long.MAX_VALUE) {
            long tieThreshold = 0;
            if (config.isScoreTieRandomEnabled()) {
                tieThreshold = Math.max((long) (minScore * config.getScoreTieThresholdPct()), config.getScoreTieThresholdMs());
            }
            long scoreCutoff = minScore + tieThreshold;
            int[] tiedIndices = new int[survivorsSize];
            int tiedCount = 0;
            for (int i = 0; i < survivorsSize; i++) {
                if (scores[i] <= scoreCutoff) {
                    tiedIndices[tiedCount++] = i;
                }
            }

            // Random selection among threshold-eligible endpoints to avoid deterministic bias
            int selectedIdx = tiedIndices[ThreadLocalRandom.current().nextInt(tiedCount)];
            best = survivors.get(selectedIdx);
            bestScore = minScore;
            bestCacheHit = cacheHits[selectedIdx];
        }

        if (best == null) {
            Map<String, Integer> merged = new java.util.HashMap<>(filterResult.rejections());
            hardFilterResult.rejections().forEach((k, v) -> merged.merge(k, v, Integer::sum));
            Logger.warn("Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId, merged);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        reportCacheHitMetrics(roleType, best.getIp(), best.ipPort(), bestCacheHit, seqLen);

        return buildServerStatus(best, roleType, requestId, bestScore, config, balanceContext, bestCacheHit);
    }

    private record EndpointFilterResult(List<PrefillEndpoint> endpoints, Map<String, Integer> rejections) {}
    private record FilterResult(List<PrefillEndpoint> endpoints, Map<String, Integer> rejections) {}

    private FilterResult applyHardFilters(List<PrefillEndpoint> eligible, long seqLen,
                                                FlexlbConfig config, Map<String, Integer> cacheMatchResults) {
        long sloMs = config.resolveSloMs(seqLen);
        long sloRiskMarginMs = config.getCostSloRiskMarginMs();
        boolean sloFilterEnabled = config.isCostSloFilterEnabled();
        double hotspotMultiplier = config.getCostHotspotMultiplier();
        double imbalanceMultiplier = config.getCostImbalanceMultiplier();

        int eligibleSize = eligible.size();
        List<PrefillEndpoint> feasible = new ArrayList<>(eligibleSize);
        long[] feasibleWaitMs = new long[eligibleSize];
        long[] feasiblePendingCount = new long[eligibleSize];
        Map<String, Integer> rejections = new java.util.HashMap<>();
        long sumWaitMs = 0;
        long sumPendingCount = 0;

        // Round 1: SLO filter + cache wait time / pending count for feasible endpoints
        for (PrefillEndpoint ep : eligible) {
            PrefillTimePredictor predictor = ep.getPredictor();
            if (predictor == null) {
                rejections.merge("PREDICTOR_MISSING", 1, Integer::sum);
                continue;
            }

            long cacheHit = calculateCacheHit(ep, cacheMatchResults, seqLen);
            long singlePrefillMs = predictor.estimateMs(seqLen, cacheHit);

            long endpointWaitMs = ep.realWaitTimeMs();

            if (sloFilterEnabled && endpointWaitMs + singlePrefillMs > sloMs - sloRiskMarginMs) {
                rejections.merge("SLO_VIOLATION", 1, Integer::sum);
                continue;
            }

            long pendingCount = ep.realPendingCount();
            int idx = feasible.size();
            feasible.add(ep);
            feasibleWaitMs[idx] = endpointWaitMs;
            feasiblePendingCount[idx] = pendingCount;
            sumWaitMs += endpointWaitMs;
            sumPendingCount += pendingCount;
        }

        if (feasible.isEmpty()) {
            return new FilterResult(feasible, rejections);
        }

        long avgWaitMs = sumWaitMs / feasible.size();
        long avgPendingCount = sumPendingCount / feasible.size();

        // Round 2: hotspot / imbalance filter using cached values (no re-computation)
        List<PrefillEndpoint> survivors = new ArrayList<>(feasible.size());
        for (int i = 0; i < feasible.size(); i++) {
            PrefillEndpoint ep = feasible.get(i);
            long endpointWaitMs = feasibleWaitMs[i];
            long pendingCount = feasiblePendingCount[i];

            if (hotspotMultiplier > 0 && avgPendingCount > 0 && pendingCount > avgPendingCount * hotspotMultiplier) {
                rejections.merge("HOTSPOT_FILTERED", 1, Integer::sum);
                continue;
            }
            if (imbalanceMultiplier > 0 && avgWaitMs > 0 && endpointWaitMs > avgWaitMs * imbalanceMultiplier) {
                rejections.merge("IMBALANCE_FILTERED", 1, Integer::sum);
                continue;
            }

            survivors.add(ep);
        }

        if (survivors.isEmpty()) {
            int minIdx = -1;
            long minWait = Long.MAX_VALUE;
            for (int i = 0; i < feasible.size(); i++) {
                if (feasibleWaitMs[i] < minWait) {
                    minWait = feasibleWaitMs[i];
                    minIdx = i;
                }
            }
            if (minIdx >= 0) {
                survivors.add(feasible.get(minIdx));
            }
        }

        return new FilterResult(survivors, rejections);
    }

    private long computeScore(PrefillEndpoint ep, long cacheHit, long seqLen) {
        PrefillTimePredictor predictor = ep.getPredictor();
        long prefillMs = predictor.estimateMs(seqLen, cacheHit);
        return prefillMs + ep.batcherWaitMs() + ep.realWaitTimeMs();
    }

    private EndpointFilterResult getAvailableEndpoints(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new EndpointFilterResult(new ArrayList<>(), Map.of("NO_REGISTERED", 1));
        }
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult(new ArrayList<>(), Map.of("NO_REGISTERED", 1));
        }
        List<PrefillEndpoint> result = new ArrayList<>();
        Map<String, Integer> rejections = new java.util.HashMap<>();

        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!(ep instanceof PrefillEndpoint pe)) {
                continue;
            }
            if (!pe.getStatus().isAlive()) {
                rejections.merge("NOT_ALIVE", 1, Integer::sum);
                continue;
            }
            if (!measure.isResourceAvailable(pe)) {
                rejections.merge("RESOURCE_UNAVAILABLE", 1, Integer::sum);
                continue;
            }
            result.add(pe);
        }
        return new EndpointFilterResult(result, rejections);
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

    private void reportCacheHitMetrics(RoleType roleType, String ip, String engineIpPort, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, engineIpPort, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(PrefillEndpoint ep, RoleType roleType, long requestId, long score,
                                            FlexlbConfig config, BalanceContext balanceContext,
                                            long bestCacheHit) {
        // Non-batch path: reserve prefill inflight for load-aware scoring.
        // Batch path uses FlexlbBatchScheduler.commitBatch() instead — skip here to avoid double-counting.
        if (isNonBatchPath(config, balanceContext)) {
            ep.commitBatch(requestId, score, Collections.emptyList());
        }

        // Populate DebugInfo so BatchItem.hitCache() can read hitCacheLen for batch metrics
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);

        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(score);
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
    private static boolean isNonBatchPath(FlexlbConfig config, BalanceContext ctx) {
        return !config.isFlexlbBatchEnabled();
    }
}
