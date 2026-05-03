package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author saichen.sm
 * description: Weighted random load balancing strategy based on normalized cache usage
 * Performs weighted random selection by normalizing cache usage across all workers
 * date: 2025/3/21
 */
@Component("weightedCacheStrategy")
public class WeightedCacheLoadBalancer implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final double decayFactor;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final CacheAwareService cacheAwareService;

    public WeightedCacheLoadBalancer(ConfigService configService,
                                     EngineWorkerStatus engineWorkerStatus,
                                     ResourceMeasureFactory resourceMeasureFactory,
                                     CacheAwareService cacheAwareService) {
        this.engineWorkerStatus = engineWorkerStatus;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.decayFactor = config.getWeightedCacheDecayFactor();
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.cacheAwareService = cacheAwareService;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, this);
    }

    private record WeightedWorker(WorkerStatus worker, long normalizedCacheUsed, double weight) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        Map<String/*ip*/, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            Logger.warn("select ROLE: {} failed, workerStatusMap is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        FlexlbConfig config = balanceContext.getConfig();
        ResourceMeasureIndicatorEnum indicator = config.getResourceMeasureIndicator(roleType);
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            Logger.warn("No ResourceMeasure registered for indicator: {}, roleType: {}", indicator, roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        // DP-enabled pods are scored too. The weighting metric
        // (usedKvCacheTokens) is summed across ranks by DP0, which is a
        // reasonable load proxy at group level. Per-rank cache-hit precision
        // (calcPrefixMatchLength below) still uses the union view from the
        // outer CacheStatus.cachedKeys — this slightly over-estimates DP
        // hits, but the strategy is randomized weighted selection rather
        // than a hard-comparison TTFT score, so the impact is bounded.
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)
                .filter(resourceMeasure::isResourceAvailable)
                .toList();
        if (CollectionUtils.isEmpty(workerStatusList)) {
            Logger.warn("select ROLE: {} failed, workerStatusList is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Implement weighted random selection algorithm
        WorkerStatus selectedWorker = weightedRandomSelection(workerStatusList);

        if (selectedWorker != null) {
            long prefixLength = calcPrefixMatchLength(selectedWorker, balanceContext.getRequest().getBlockCacheKeys());
            // Update local task state
            return buildServerStatus(selectedWorker, seqLen, prefixLength, roleType, balanceContext.getRequestId());
        }

        // Return failure if no suitable worker found
        Logger.warn("Failed to select worker, no suitable worker available");
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    /**
     * Release local cached tasks on the specified worker
     *
     * @param ipPort Worker IP address
     * @param requestId Request ID
     */
    @Override
    public void rollBack(String ipPort, long requestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);
        Logger.debug("Decode rollBack - ip: {}, requestId: {}",
                ipPort, requestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    /**
     * Token-level prefix match length for the chosen worker. Routes through
     * {@link CacheAwareService#findMatchingPrefixLength} so DP engines get
     * the per-rank MAX (not the misleading union of {@code cacheStatus.cachedKeys}).
     * For non-DP engines, the result equals the legacy union-based number.
     *
     * <p>Result feeds {@link TaskInfo#setPrefixLength} which downstream
     * accounting ({@code WorkerStatus.putLocalTask}) uses to estimate
     * {@code needNewKvCacheLen} and prefill time. An honest prefix here
     * prevents DP workers from looking artificially less loaded than they
     * really are after a request lands on them.
     */
    private long calcPrefixMatchLength(WorkerStatus worker, List<Long> promptCacheKeys) {
        if (worker == null || worker.getCacheStatus() == null || promptCacheKeys == null) {
            return 0;
        }
        long blockSize = worker.getCacheStatus().getBlockSize();
        int prefixBlocks = cacheAwareService.findMatchingPrefixLength(worker.getIpPort(), promptCacheKeys);
        return blockSize * prefixBlocks;
    }

    /**
     * Weighted random selection algorithm: performs weighted random selection based on normalized cache usage
     *
     * @param candidateWorkers Candidate worker list
     * @return Selected WorkerStatus, or null if no suitable worker found
     */
    private WorkerStatus weightedRandomSelection(List<WorkerStatus> candidateWorkers) {
        int workerCount = candidateWorkers.size();
        if (workerCount == 0) {
            return null;
        }

        // 1. Calculate sum and average of cacheUsed
        long totalCacheUsed = 0;
        for (WorkerStatus worker : candidateWorkers) {
            totalCacheUsed += worker.getUsedKvCacheTokens().get();
        }
        double avgCacheUsed = (double) totalCacheUsed / workerCount;

        // 2. Normalize cacheUsed and calculate weights
        List<WeightedWorker> weightedWorkers = new ArrayList<>();
        boolean allSameUsage = true;
        double totalWeight = 0;
        Long firstCacheUsed = null;

        for (WorkerStatus worker : candidateWorkers) {
            long cacheUsed = worker.getUsedKvCacheTokens().get();
            double normalizedValue = cacheUsed - avgCacheUsed;

            if (firstCacheUsed == null) {
                firstCacheUsed = cacheUsed;
            } else if (cacheUsed != firstCacheUsed) {
                allSameUsage = false;
            }

            double weight = Math.exp(-decayFactor * normalizedValue);

            weightedWorkers.add(new WeightedWorker(worker, (long) normalizedValue, weight));
            totalWeight += weight;
        }

        // Check if total weight is valid
        if (totalWeight <= 0) {
            Logger.warn("Total weight is zero or negative: {}, using uniform random selection", totalWeight);
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        // If all workers have same cache usage, use uniform random
        if (allSameUsage) {
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        // 3. Perform weighted random selection using roulette wheel algorithm
        double randomValue = ThreadLocalRandom.current().nextDouble() * totalWeight;
        double cumulativeWeight = 0;

        for (WeightedWorker weightedWorker : weightedWorkers) {
            cumulativeWeight += weightedWorker.weight;
            if (Double.compare(randomValue, cumulativeWeight) <= 0) {
                return weightedWorker.worker;
            }
        }

        // Fallback: select worker with minimum cacheUsed
        return weightedWorkers.stream()
                .min(Comparator.comparingLong(w -> w.worker.getUsedKvCacheTokens().get()))
                .map(w -> w.worker)
                .orElse(null);
    }

    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, long requestId) {
        ServerStatus result = new ServerStatus();
        try {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setPrefillTime(0);
            taskInfo.setWaitingTime(0);
            taskInfo.setInputLength(seqLen);
            taskInfo.setPrefixLength(prefixLength);
            taskInfo.setRequestId(requestId);

            // Update local task state
            optimalWorker.putLocalTask(requestId, taskInfo);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalWorker.getIp());
            result.setHttpPort(optimalWorker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalWorker.getPort()));
            result.setGroup(optimalWorker.getGroup());
            result.setRequestId(requestId);
        } catch (Exception e) {
            Logger.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }
}
