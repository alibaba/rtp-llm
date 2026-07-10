package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.resource.DecodeResourceMeasure;
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
import org.flexlb.enums.ScheduleModeEnum;
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
public class CostBasedDecodeStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final double decayFactor;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public CostBasedDecodeStrategy(ConfigService configService,
                                    EngineWorkerStatus engineWorkerStatus,
                                    ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.decayFactor = config.getWeightedCacheDecayFactor();
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.COST_BASED_DECODE, this);
    }

    private record WeightedWorker(DecodeEndpoint endpoint, long normalizedCacheUsed, double weight) {
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        EndpointFilterResult filterResult = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        List<DecodeEndpoint> eligible = filterResult.endpoints();
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.warn("Decode select failed: no available endpoints, request_id={}, rejections={}",
                    balanceContext.getRequestId(), filterResult.rejections());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        FilterResult hardFilterResult = applyHardFilters(eligible, seqLen, config);
        List<DecodeEndpoint> survivors = hardFilterResult.endpoints();

        DecodeEndpoint selectedEndpoint = weightedRandomSelection(survivors);

        if (selectedEndpoint != null) {
            long prefixLength = calcPrefixMatchLength(selectedEndpoint.getStatus().getCacheStatus(), balanceContext.getRequest().getBlockCacheKeys());
            return buildServerStatus(selectedEndpoint, seqLen, prefixLength, roleType, balanceContext.getRequestId(), balanceContext.getScheduleMode());
        }

        Map<String, Integer> merged = new java.util.HashMap<>(filterResult.rejections());
        hardFilterResult.rejections().forEach((k, v) -> merged.merge(k, v, Integer::sum));
        Logger.warn("Decode select failed: all filtered out, request_id={}, rejections={}",
                balanceContext.getRequestId(), merged);
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }

    private record EndpointFilterResult(List<DecodeEndpoint> endpoints, Map<String, Integer> rejections) {}
    private record FilterResult(List<DecodeEndpoint> endpoints, Map<String, Integer> rejections) {}

    private EndpointFilterResult getAvailableEndpoints(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new EndpointFilterResult(new ArrayList<>(), Map.of("NO_REGISTERED", 1));
        }
        DecodeResourceMeasure measure = (DecodeResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult(new ArrayList<>(), Map.of("NO_REGISTERED", 1));
        }
        List<DecodeEndpoint> result = new ArrayList<>();
        Map<String, Integer> rejections = new java.util.HashMap<>();
        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!(ep instanceof DecodeEndpoint de)) {
                continue;
            }
            if (!de.getStatus().isAlive()) {
                rejections.merge("NOT_ALIVE", 1, Integer::sum);
                continue;
            }
            if (!measure.isResourceAvailable(de)) {
                rejections.merge("RESOURCE_UNAVAILABLE", 1, Integer::sum);
                continue;
            }
            result.add(de);
        }
        return new EndpointFilterResult(result, rejections);
    }

    @Override
    public void rollBack(WorkerEndpoint ep, long requestId) {
        Logger.debug("Decode rollBack - ip: {}, requestId: {}", ep.ipPort(), requestId);

        if (ep instanceof DecodeEndpoint de) {
            de.release(requestId);
        }
    }

    private FilterResult applyHardFilters(List<DecodeEndpoint> eligible, long seqLen, FlexlbConfig config) {
        double hotspotMultiplier = config.getDecodeHotspotMultiplier();
        double imbalanceMultiplier = config.getDecodeImbalanceMultiplier();

        long sumLoad = 0;
        long sumCacheUsed = 0;
        for (DecodeEndpoint ep : eligible) {
            sumLoad += ep.getTotalLoad();
            sumCacheUsed += ep.realKvUsed();
        }
        long avgLoad = sumLoad / eligible.size();
        long avgCacheUsed = sumCacheUsed / eligible.size();

        List<DecodeEndpoint> survivors = new ArrayList<>(eligible.size());
        Map<String, Integer> rejections = new java.util.HashMap<>();
        for (DecodeEndpoint ep : eligible) {
            long availableKv = ep.realKvAvailable();
            long totalKv = ep.realKvTotal();
            if (totalKv > 0 && availableKv < seqLen) {
                rejections.merge("KV_CAPACITY", 1, Integer::sum);
                continue;
            }
            long load = ep.getTotalLoad();
            if (hotspotMultiplier > 0 && avgLoad > 0
                    && load > avgLoad * hotspotMultiplier) {
                rejections.merge("HOTSPOT_FILTERED", 1, Integer::sum);
                continue;
            }
            if (imbalanceMultiplier > 0 && avgCacheUsed > 0
                    && ep.realKvUsed() > avgCacheUsed * imbalanceMultiplier) {
                rejections.merge("IMBALANCE_FILTERED", 1, Integer::sum);
                continue;
            }
            survivors.add(ep);
        }

        if (survivors.isEmpty()) {
            DecodeEndpoint leastUsed = eligible.stream()
                    .min(Comparator.comparingLong(DecodeEndpoint::realKvUsed))
                    .orElse(null);
            if (leastUsed != null) {
                survivors.add(leastUsed);
            }
        }
        return new FilterResult(survivors, rejections);
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
            totalCacheUsed += ep.realKvUsed();
        }
        double avgCacheUsed = (double) totalCacheUsed / workerCount;

        List<WeightedWorker> weightedEndpoints = new ArrayList<>();
        boolean allSameUsage = true;
        double totalWeight = 0;
        Long firstCacheUsed = null;

        for (DecodeEndpoint ep : candidateEndpoints) {
            long cacheUsed = ep.realKvUsed();
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
                .min(Comparator.comparingLong(w -> w.endpoint.realKvUsed()))
                .map(w -> w.endpoint)
                .orElse(null);
    }

    private ServerStatus buildServerStatus(DecodeEndpoint optimalEndpoint, long seqLen, long prefixLength, RoleType roleType, long requestId, ScheduleModeEnum scheduleMode) {
        ServerStatus result = new ServerStatus();
        try {
            // Skip KV reservation for DIRECT and QUEUE paths.
            //
            // These paths do not track request lifecycle after routing succeeds:
            //   - DIRECT: Master returns the response immediately, no inflight tracking.
            //   - QUEUE:  RequestScheduler completes the future after routing, no release() on success/cancel.
            //
            // Reserved KV would only be cleaned by calibrate (~20ms cycle) or TTL eviction (300s).
            // When the engine is overloaded and calibrate stalls, reservations pile up and drive
            // available KV to zero, causing NO_AVAILABLE_WORKER cascade failures.
            // The engine's own KV admission control serves as backstop.
            //
            // BATCH path keeps reserve because FlexlbBatchScheduler tracks lifecycle and calls
            // de.release() on cancel/failure/expiry.
            boolean skipReserve = scheduleMode == ScheduleModeEnum.DIRECT
                    || scheduleMode == ScheduleModeEnum.QUEUE;
            if (!skipReserve) {
                optimalEndpoint.reserve(requestId, seqLen);
            }

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
