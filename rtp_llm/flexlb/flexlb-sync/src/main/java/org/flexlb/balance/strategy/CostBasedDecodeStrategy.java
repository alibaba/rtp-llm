package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
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
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

@Component("costBasedDecodeStrategy")
public class CostBasedDecodeStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final double decayFactor;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EndpointRegistry endpointRegistry;

    public CostBasedDecodeStrategy(ConfigService configService,
                                    EngineWorkerStatus engineWorkerStatus,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EndpointRegistry endpointRegistry) {
        this.engineWorkerStatus = engineWorkerStatus;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.decayFactor = config.getWeightedCacheDecayFactor();
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.endpointRegistry = endpointRegistry;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.COST_BASED_DECODE, this);
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
        List<WorkerStatus> workerStatusList = new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)
                .filter(resourceMeasure::isResourceAvailable)
                .toList();
        if (CollectionUtils.isEmpty(workerStatusList)) {
            Logger.warn("select ROLE: {} failed, workerStatusList is empty", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        List<WorkerStatus> survivors = applyHardFilters(workerStatusList, seqLen, config);

        WorkerStatus selectedWorker = weightedRandomSelection(survivors);

        if (selectedWorker != null) {
            long prefixLength = calcPrefixMatchLength(selectedWorker.getCacheStatus(), balanceContext.getRequest().getBlockCacheKeys());
            return buildServerStatus(selectedWorker, seqLen, prefixLength, roleType, balanceContext.getRequestId());
        }

        Logger.warn("Failed to select worker, no suitable worker available");
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        Logger.debug("Decode rollBack - ip: {}, requestId: {}", ipPort, requestId);

        DecodeEndpoint ep = endpointRegistry.getDecode(ipPort);
        if (ep != null) {
            ep.release(requestId);
        }
    }

    private List<WorkerStatus> applyHardFilters(List<WorkerStatus> eligible, long seqLen, FlexlbConfig config) {
        double hotspotMultiplier = config.getDecodeHotspotMultiplier();
        double imbalanceMultiplier = config.getDecodeImbalanceMultiplier();

        long sumLoad = 0;
        long sumCacheUsed = 0;
        for (WorkerStatus w : eligible) {
            DecodeEndpoint ep = endpointRegistry.getDecode(w.getIpPort());
            sumLoad += ep != null ? ep.getTotalLoad() : 0;
            sumCacheUsed += w.getUsedKvCacheTokens().get();
        }
        long avgLoad = sumLoad / eligible.size();
        long avgCacheUsed = sumCacheUsed / eligible.size();

        List<WorkerStatus> survivors = new ArrayList<>(eligible.size());
        for (WorkerStatus w : eligible) {
            DecodeEndpoint ep = endpointRegistry.getDecode(w.getIpPort());
            long availableKv = ep != null ? ep.getAvailableKvTokens() : w.getAvailableKvCacheTokens().get();
            long totalKv = w.getUsedKvCacheTokens().get() + w.getAvailableKvCacheTokens().get();
            if (totalKv > 0 && availableKv < seqLen) {
                continue;
            }
            long load = ep != null ? ep.getTotalLoad() : 0;
            if (hotspotMultiplier > 0 && avgLoad > 0
                    && load > avgLoad * hotspotMultiplier) {
                continue;
            }
            if (imbalanceMultiplier > 0 && avgCacheUsed > 0
                    && w.getUsedKvCacheTokens().get() > avgCacheUsed * imbalanceMultiplier) {
                continue;
            }
            survivors.add(w);
        }

        if (survivors.isEmpty()) {
            WorkerStatus leastUsed = eligible.stream()
                    .min(Comparator.comparingLong(w -> w.getUsedKvCacheTokens().get()))
                    .orElse(null);
            if (leastUsed != null) {
                survivors.add(leastUsed);
            }
        }

        return survivors;
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

        for (int index = 0; index < promptCacheKeys.size(); index++) {
            long hash = promptCacheKeys.get(index);
            if (!cachePrefixHash.contains(hash)) {
                return blockSize * index;
            }
        }

        return blockSize * promptCacheKeys.size();
    }

    private WorkerStatus weightedRandomSelection(List<WorkerStatus> candidateWorkers) {
        int workerCount = candidateWorkers.size();
        if (workerCount == 0) {
            return null;
        }

        long totalCacheUsed = 0;
        for (WorkerStatus worker : candidateWorkers) {
            totalCacheUsed += worker.getUsedKvCacheTokens().get();
        }
        double avgCacheUsed = (double) totalCacheUsed / workerCount;

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

        if (totalWeight <= 0) {
            Logger.warn("Total weight is zero or negative: {}, using uniform random selection", totalWeight);
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        if (allSameUsage) {
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateWorkers.get(randomIndex);
        }

        double randomValue = ThreadLocalRandom.current().nextDouble() * totalWeight;
        double cumulativeWeight = 0;

        for (WeightedWorker weightedWorker : weightedWorkers) {
            cumulativeWeight += weightedWorker.weight;
            if (Double.compare(randomValue, cumulativeWeight) <= 0) {
                return weightedWorker.worker;
            }
        }

        return weightedWorkers.stream()
                .min(Comparator.comparingLong(w -> w.worker.getUsedKvCacheTokens().get()))
                .map(w -> w.worker)
                .orElse(null);
    }

    private ServerStatus buildServerStatus(WorkerStatus optimalWorker, long seqLen, long prefixLength, RoleType roleType, long requestId) {
        ServerStatus result = new ServerStatus();
        try {
            DecodeEndpoint ep = endpointRegistry.getDecode(optimalWorker.getIpPort());
            if (ep != null) {
                ep.reserve(requestId, seqLen);
            }

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalWorker.getIp());
            result.setHttpPort(optimalWorker.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalWorker.getPort()));
            result.setDpRank(optimalWorker.getDpRank());
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
