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
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.CacheStatus;
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

    private record WeightedWorker(DecodeEndpoint endpoint, long normalizedCacheUsed, double weight) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<DecodeEndpoint> eligible = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.warn("select ROLE: {} failed, no available endpoints", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        List<DecodeEndpoint> survivors = applyHardFilters(eligible, seqLen, config);

        DecodeEndpoint selectedEndpoint = weightedRandomSelection(survivors);

        if (selectedEndpoint != null) {
            long prefixLength = calcPrefixMatchLength(selectedEndpoint.getStatus().getCacheStatus(), balanceContext.getRequest().getBlockCacheKeys());
            return buildServerStatus(selectedEndpoint, seqLen, prefixLength, roleType, balanceContext.getRequestId());
        }

        Logger.warn("Failed to select worker, no suitable worker available");
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    private List<DecodeEndpoint> getAvailableEndpoints(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new ArrayList<>();
        }
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            return new ArrayList<>();
        }
        List<DecodeEndpoint> result = new ArrayList<>();
        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!ep.getStatus().isAlive() || !resourceMeasure.isResourceAvailable(ep)) {
                continue;
            }
            if (ep instanceof DecodeEndpoint de && de.isAvailable()) {
                result.add(de);
            }
        }
        return result;
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        Logger.debug("Decode rollBack - ip: {}, requestId: {}", ipPort, requestId);

        DecodeEndpoint ep = endpointRegistry.getDecode(ipPort);
        if (ep != null) {
            ep.release(requestId);
        }
    }

    private List<DecodeEndpoint> applyHardFilters(List<DecodeEndpoint> eligible, long seqLen, FlexlbConfig config) {
        double hotspotMultiplier = config.getDecodeHotspotMultiplier();
        double imbalanceMultiplier = config.getDecodeImbalanceMultiplier();

        long sumLoad = 0;
        long sumCacheUsed = 0;
        for (DecodeEndpoint ep : eligible) {
            sumLoad += ep.getTotalLoad();
            sumCacheUsed += ep.getStatus().getUsedKvCacheTokens().get();
        }
        long avgLoad = sumLoad / eligible.size();
        long avgCacheUsed = sumCacheUsed / eligible.size();

        List<DecodeEndpoint> survivors = new ArrayList<>(eligible.size());
        for (DecodeEndpoint ep : eligible) {
            long availableKv = ep.getAvailableKvTokens();
            long totalKv = ep.getStatus().getUsedKvCacheTokens().get() + ep.getStatus().getAvailableKvCacheTokens().get();
            if (totalKv > 0 && availableKv < seqLen) {
                continue;
            }
            long load = ep.getTotalLoad();
            if (hotspotMultiplier > 0 && avgLoad > 0
                    && load > avgLoad * hotspotMultiplier) {
                continue;
            }
            if (imbalanceMultiplier > 0 && avgCacheUsed > 0
                    && ep.getStatus().getUsedKvCacheTokens().get() > avgCacheUsed * imbalanceMultiplier) {
                continue;
            }
            survivors.add(ep);
        }

        if (survivors.isEmpty()) {
            DecodeEndpoint leastUsed = eligible.stream()
                    .min(Comparator.comparingLong(ep -> ep.getStatus().getUsedKvCacheTokens().get()))
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

    private DecodeEndpoint weightedRandomSelection(List<DecodeEndpoint> candidateEndpoints) {
        int workerCount = candidateEndpoints.size();
        if (workerCount == 0) {
            return null;
        }

        long totalCacheUsed = 0;
        for (DecodeEndpoint ep : candidateEndpoints) {
            totalCacheUsed += ep.getStatus().getUsedKvCacheTokens().get();
        }
        double avgCacheUsed = (double) totalCacheUsed / workerCount;

        List<WeightedWorker> weightedEndpoints = new ArrayList<>();
        boolean allSameUsage = true;
        double totalWeight = 0;
        Long firstCacheUsed = null;

        for (DecodeEndpoint ep : candidateEndpoints) {
            long cacheUsed = ep.getStatus().getUsedKvCacheTokens().get();
            double normalizedValue = cacheUsed - avgCacheUsed;

            if (firstCacheUsed == null) {
                firstCacheUsed = cacheUsed;
            } else if (cacheUsed != firstCacheUsed) {
                allSameUsage = false;
            }

            double weight = Math.exp(-decayFactor * normalizedValue);

            weightedEndpoints.add(new WeightedWorker(ep, (long) normalizedValue, weight));
            totalWeight += weight;
        }

        if (totalWeight <= 0) {
            Logger.warn("Total weight is zero or negative: {}, using uniform random selection", totalWeight);
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateEndpoints.get(randomIndex);
        }

        if (allSameUsage) {
            int randomIndex = ThreadLocalRandom.current().nextInt(workerCount);
            return candidateEndpoints.get(randomIndex);
        }

        double randomValue = ThreadLocalRandom.current().nextDouble() * totalWeight;
        double cumulativeWeight = 0;

        for (WeightedWorker weightedEndpoint : weightedEndpoints) {
            cumulativeWeight += weightedEndpoint.weight;
            if (Double.compare(randomValue, cumulativeWeight) <= 0) {
                return weightedEndpoint.endpoint;
            }
        }

        return weightedEndpoints.stream()
                .min(Comparator.comparingLong(w -> w.endpoint.getStatus().getUsedKvCacheTokens().get()))
                .map(w -> w.endpoint)
                .orElse(null);
    }

    private ServerStatus buildServerStatus(DecodeEndpoint optimalEndpoint, long seqLen, long prefixLength, RoleType roleType, long requestId) {
        ServerStatus result = new ServerStatus();
        try {
            optimalEndpoint.reserve(requestId, seqLen);

            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(optimalEndpoint.getIp());
            result.setHttpPort(optimalEndpoint.getHttpPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(optimalEndpoint.getHttpPort()));
            result.setDpRank(optimalEndpoint.getStatus().getDpRank());
            result.setGroup(optimalEndpoint.getStatus().getGroup());
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
