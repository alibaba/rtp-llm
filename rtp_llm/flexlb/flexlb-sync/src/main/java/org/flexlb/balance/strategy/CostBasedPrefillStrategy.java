package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
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
    private final FlexlbBatchScheduler batchScheduler;
    private final EndpointRegistry endpointRegistry;

    public CostBasedPrefillStrategy(EngineWorkerStatus engineWorkerStatus,
                                    CacheAwareService cacheAwareService,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EngineHealthReporter engineHealthReporter,
                                    FlexlbBatchScheduler batchScheduler,
                                    EndpointRegistry endpointRegistry) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        this.batchScheduler = batchScheduler;
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
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        WorkerStatus workerStatus = workerStatusMap != null ? workerStatusMap.get(ipPort) : null;
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        String model = balanceContext.getRequest().getModel();
        FlexlbConfig config = balanceContext.getConfig();

        List<WorkerStatus> eligible = getAvailableWorkers(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(eligible)) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);
        PrefillTimePredictor predictor = createPredictor(config);

        List<WorkerStatus> survivors = applyHardFilters(eligible, model, seqLen, config, predictor);

        WorkerStatus best = null;
        long bestScore = Long.MAX_VALUE;
        long bestCacheHit = 0;

        for (WorkerStatus w : survivors) {
            long cacheHit = calculateCacheHit(w, cacheMatchResults);
            long score = computeScore(w, model, seqLen, cacheHit, config, predictor);

            if (score < bestScore) {
                bestScore = score;
                best = w;
                bestCacheHit = cacheHit;
            }
        }

        if (best == null) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        reportCacheHitMetrics(roleType, best.getIp(), bestCacheHit, seqLen);
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(seqLen);
        task.setPrefixLength(bestCacheHit);
        best.putLocalTask(requestId, task);

        return buildServerStatus(best, roleType, requestId, bestScore);
    }

    private List<WorkerStatus> applyHardFilters(List<WorkerStatus> eligible, String model, long seqLen,
                                                FlexlbConfig config, PrefillTimePredictor predictor) {
        long sloMs = config.resolveSloMs(seqLen);
        long sloRiskMarginMs = config.getCostSloRiskMarginMs();
        double hotspotMultiplier = config.getCostHotspotMultiplier();
        double imbalanceMultiplier = config.getCostImbalanceMultiplier();

        long sumQueueTime = 0;
        long sumBatcherSize = 0;
        for (WorkerStatus w : eligible) {
            sumQueueTime += w.getRunningQueueTime().get();
            sumBatcherSize += batchScheduler.snapshotForWorker(model, w.getIp(), w.getPort()).queueSize();
        }
        long avgQueueTime = sumQueueTime / eligible.size();
        long avgBatcherSize = sumBatcherSize / eligible.size();

        long singlePrefillMs = predictor.predictBatchMs(
                List.of(new RequestProfile(seqLen, 0)));

        List<WorkerStatus> survivors = new ArrayList<>(eligible.size());
        for (WorkerStatus w : eligible) {
            long queueMs = w.getRunningQueueTime().get();
            BatcherSnapshot snap = batchScheduler.snapshotForWorker(model, w.getIp(), w.getPort());

            if (queueMs + singlePrefillMs > sloMs - sloRiskMarginMs) {
                continue;
            }
            if (hotspotMultiplier > 0 && avgBatcherSize > 0 && snap.queueSize() > avgBatcherSize * hotspotMultiplier) {
                continue;
            }
            if (imbalanceMultiplier > 0 && avgQueueTime > 0 && queueMs > avgQueueTime * imbalanceMultiplier) {
                continue;
            }

            survivors.add(w);
        }

        if (survivors.isEmpty()) {
            WorkerStatus leastLoaded = eligible.stream()
                    .min(Comparator.comparingLong(w -> w.getRunningQueueTime().get()))
                    .orElse(null);
            if (leastLoaded != null) {
                survivors.add(leastLoaded);
            }
        }

        return survivors;
    }

    private long computeScore(WorkerStatus w, String model, long seqLen, long cacheHit,
                              FlexlbConfig config, PrefillTimePredictor predictor) {
        BatcherSnapshot snap = batchScheduler.snapshotForWorker(model, w.getIp(), w.getPort());

        long waitMs;
        if (snap.queueSize() == 0) {
            long predMs = predictor.estimateMs(seqLen, cacheHit);
            waitMs = Math.max(0, config.resolveSloMs(seqLen) - predMs - config.getCostSloRiskMarginMs());
        } else {
            waitMs = Math.max(0, snap.headDeadlineMs() - System.currentTimeMillis() - config.getCostSloRiskMarginMs());
        }
        if (snap.queueSize() + 1 >= config.getFlexlbBatchSizeMax()) {
            waitMs = 0;
        }

        WorkerEndpoint ep = endpointRegistry.get(w.getIpPort());
        long workerWaitMs = ep != null ? ep.getEstimatedWaitingTimeMs() : 0;

        // delta_prefill_ms: marginal cost of adding this request to the batcher batch
        List<RequestProfile> oldBatch = snap.requests();
        List<RequestProfile> newBatch = new ArrayList<>(oldBatch);
        newBatch.add(new RequestProfile(seqLen, cacheHit));
        long deltaPrefillMs = predictor.predictBatchMs(newBatch) - predictor.predictBatchMs(oldBatch);

        return waitMs + workerWaitMs + deltaPrefillMs;
    }

    private PrefillTimePredictor createPredictor(FlexlbConfig config) {
        return new PrefillTimePredictor(
                config.getCostAlpha0(), config.getCostAlpha1(), config.getCostAlpha2(),
                config.getCostAlpha3(), config.getCostAlpha4(), config.getCostAlpha5());
    }

    private List<WorkerStatus> getAvailableWorkers(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            return new ArrayList<>();
        }
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            return new ArrayList<>();
        }
        return workerStatusMap.values().stream()
                .filter(WorkerStatus::isAlive)
                .filter(resourceMeasure::isResourceAvailable)
                .toList();
    }

    private Map<String, Integer> getCacheMatchResults(BalanceContext balanceContext, RoleType roleType, String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    private long calculateCacheHit(WorkerStatus workerStatus, Map<String, Integer> cacheMatchResults) {
        if (workerStatus.getCacheStatus() == null || cacheMatchResults == null) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(workerStatus.getIpPort());
        if (prefixMatchLength == null) {
            return 0L;
        }
        return workerStatus.getCacheStatus().getBlockSize() * prefixMatchLength;
    }

    private void reportCacheHitMetrics(RoleType roleType, String ip, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, long requestId, long score) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(score);
        result.setGroup(worker.getGroup());
        result.setServerIp(worker.getIp());
        result.setHttpPort(worker.getPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
        result.setDpRank(worker.getDpRank());
        return result;
    }
}
