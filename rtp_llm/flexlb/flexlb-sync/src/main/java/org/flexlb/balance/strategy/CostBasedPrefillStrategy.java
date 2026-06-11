package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

@Component("costBasedPrefillStrategy")
public class CostBasedPrefillStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;
    private final EndpointRegistry endpointRegistry;

    public CostBasedPrefillStrategy(EngineWorkerStatus engineWorkerStatus,
                                    CacheAwareService cacheAwareService,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EngineHealthReporter engineHealthReporter,
                                    EndpointRegistry endpointRegistry) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        this.endpointRegistry = endpointRegistry;
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
    public void rollBack(String ipPort, long requestId) {
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<PrefillEndpoint> eligible = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(eligible)) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);
        PrefillTimePredictor predictor = getPredictor(eligible, config);

        List<PrefillEndpoint> survivors = applyHardFilters(eligible, seqLen, config, predictor);

        PrefillEndpoint best = null;
        long bestScore = Long.MAX_VALUE;
        long bestCacheHit = 0;

        for (PrefillEndpoint ep : survivors) {
            long cacheHit = calculateCacheHit(ep, cacheMatchResults);
            long score = computeScore(ep, seqLen, cacheHit, config, predictor);

            if (score < bestScore) {
                bestScore = score;
                best = ep;
                bestCacheHit = cacheHit;
            }
        }

        if (best == null) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        reportCacheHitMetrics(roleType, best.getIp(), bestCacheHit, seqLen);

        return buildServerStatus(best, roleType, requestId, bestScore);
    }

    private List<PrefillEndpoint> applyHardFilters(List<PrefillEndpoint> eligible, long seqLen,
                                                FlexlbConfig config, PrefillTimePredictor predictor) {
        long sloMs = config.resolveSloMs(seqLen);
        long sloRiskMarginMs = config.getCostSloRiskMarginMs();
        double hotspotMultiplier = config.getCostHotspotMultiplier();
        double imbalanceMultiplier = config.getCostImbalanceMultiplier();

        long sumWaitMs = 0;
        long sumPendingCount = 0;
        for (PrefillEndpoint ep : eligible) {
            sumWaitMs += ep.getEstimatedWaitingTimeMs();
            sumPendingCount += ep.getBatcherQueueSize() + ep.getInflightRequestCount();
        }
        long avgWaitMs = sumWaitMs / eligible.size();
        long avgPendingCount = sumPendingCount / eligible.size();

        long singlePrefillMs = predictor.predictBatchMs(
                List.of(new BatchRequest(0, seqLen, 0)));

        List<PrefillEndpoint> survivors = new ArrayList<>(eligible.size());
        for (PrefillEndpoint ep : eligible) {
            long endpointWaitMs = ep.getEstimatedWaitingTimeMs();
            long pendingCount = ep.getBatcherQueueSize() + ep.getInflightRequestCount();

            if (endpointWaitMs + singlePrefillMs > sloMs - sloRiskMarginMs) {
                continue;
            }
            if (hotspotMultiplier > 0 && avgPendingCount > 0 && pendingCount > avgPendingCount * hotspotMultiplier) {
                continue;
            }
            if (imbalanceMultiplier > 0 && avgWaitMs > 0 && endpointWaitMs > avgWaitMs * imbalanceMultiplier) {
                continue;
            }

            survivors.add(ep);
        }

        if (survivors.isEmpty()) {
            PrefillEndpoint leastLoaded = eligible.stream()
                    .min(Comparator.comparingLong(PrefillEndpoint::getEstimatedWaitingTimeMs))
                    .orElse(null);
            if (leastLoaded != null) {
                survivors.add(leastLoaded);
            }
        }

        return survivors;
    }

    private long computeScore(PrefillEndpoint ep, long seqLen, long cacheHit,
                              FlexlbConfig config, PrefillTimePredictor predictor) {
        BatcherSnapshot snap = ep.getBatcherSnapshot();

        long batcherWaitMs;
        if (snap.queueSize() == 0) {
            long predMs = predictor.estimateMs(seqLen, cacheHit);
            batcherWaitMs = Math.max(0, config.resolveSloMs(seqLen) - predMs - config.getCostSloRiskMarginMs());
        } else {
            batcherWaitMs = Math.max(0, snap.headDeadlineMs() - System.currentTimeMillis() - config.getCostSloRiskMarginMs());
        }
        if (snap.queueSize() + 1 >= config.getFlexlbBatchSizeMax()) {
            batcherWaitMs = 0;
        }

        long endpointWaitMs = ep.getEstimatedWaitingTimeMs();

        return batcherWaitMs + endpointWaitMs;
    }

    private PrefillTimePredictor getPredictor(List<PrefillEndpoint> eligible, FlexlbConfig config) {
        for (PrefillEndpoint ep : eligible) {
            PrefillTimePredictor p = ep.getPredictor();
            if (p != null) {
                return p;
            }
        }
        return new PrefillTimePredictor(
                config.getCostAlpha0(), config.getCostAlpha1(), config.getCostAlpha2(),
                config.getCostAlpha3(), config.getCostAlpha4(), config.getCostAlpha5());
    }

    private List<PrefillEndpoint> getAvailableEndpoints(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new ArrayList<>();
        }
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            return new ArrayList<>();
        }
        List<PrefillEndpoint> result = new ArrayList<>();
        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!ep.isAlive() || !resourceMeasure.isResourceAvailable(ep)) {
                continue;
            }
            if (ep instanceof PrefillEndpoint pe && pe.isAvailable()) {
                result.add(pe);
            }
        }
        return result;
    }

    private Map<String, Integer> getCacheMatchResults(BalanceContext balanceContext, RoleType roleType, String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    private long calculateCacheHit(PrefillEndpoint ep, Map<String, Integer> cacheMatchResults) {
        if (ep.getCacheStatus() == null || cacheMatchResults == null) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null) {
            return 0L;
        }
        return ep.getCacheStatus().getBlockSize() * prefixMatchLength;
    }

    private void reportCacheHitMetrics(RoleType roleType, String ip, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(PrefillEndpoint ep, RoleType roleType, long requestId, long score) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(score);
        result.setGroup(ep.getGroup());
        result.setServerIp(ep.getIp());
        result.setHttpPort(ep.getHttpPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(ep.getHttpPort()));
        result.setDpRank(ep.getDpRank());
        return result;
    }
}
