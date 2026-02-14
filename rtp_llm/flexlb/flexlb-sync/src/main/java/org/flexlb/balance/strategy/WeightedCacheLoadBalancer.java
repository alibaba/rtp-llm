package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

    public WeightedCacheLoadBalancer(ConfigService configService,
                                     EngineWorkerStatus engineWorkerStatus,
                                     ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.decayFactor = configService.loadBalanceConfig().getWeightedCacheDecayFactor();
        this.resourceMeasureFactory = resourceMeasureFactory;
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
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(roleType.getResourceMeasureIndicator());
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)                   // Check if resource is available
                .filter(resourceMeasure::isResourceAvailable)    // Check if worker has available resources
                .toList();
        if (CollectionUtils.isEmpty(workerStatusList)) {
            Logger.warn("select ROLE: {} failed, workerStatusList is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Implement weighted random selection algorithm
        WorkerStatus selectedWorker = weightedRandomSelection(workerStatusList);

        if (selectedWorker != null) {
            long prefixLength = calcPrefixMatchLength(selectedWorker.getCacheStatus(), balanceContext.getRequest().getBlockCacheKeys());
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
     * @param interRequestId Internal request ID
     */
    @Override
    public void rollBack(String ipPort, String interRequestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);
        Logger.debug("Decode rollBack - ip: {}, interRequestId: {}",
                ipPort, interRequestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(interRequestId);
        }
    }

    private long calcPrefixMatchLength(CacheStatus cacheStatus, List<Long> promptCacheKeys) {

        if (cacheStatus == null || promptCacheKeys == null) {
            return 0;
        }
        long blockSize = cacheStatus.getBlockSize();
        Set<Long> cachePrefixHash = cacheStatus.getCachedKeys();
        if (cachePrefixHash == null) {
            return 0;
        }
        
        // Iterate from beginning to find first mismatch position
        for (int index = 0; index < promptCacheKeys.size(); index++) {
            long hash = promptCacheKeys.get(index);
            if (!cachePrefixHash.contains(hash)) {
                // Return matching prefix length (matched block count * block size)
                return blockSize * index;
            }
        }

        // Return total length if all match
        return blockSize * promptCacheKeys.size();
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

    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, String interRequestId) {
        ServerStatus result = new ServerStatus();
        try {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setPrefillTime(0);
            taskInfo.setWaitingTime(0);
            taskInfo.setInputLength(seqLen);
            taskInfo.setPrefixLength(prefixLength);
            taskInfo.setInterRequestId(interRequestId);

            // Update local task state
            optimalWorker.putLocalTask(interRequestId, taskInfo);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalWorker.getIp());
            result.setHttpPort(optimalWorker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalWorker.getPort()));
            result.setGroup(optimalWorker.getGroup());
            result.setInterRequestId(interRequestId);
        } catch (Exception e) {
            Logger.error("buildServerStatus error", e);
            result.setSuccess(false);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
        }
        return result;
    }
}
