package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Decode role resource measure
 * Availability criteria: KV cache usage percentage below threshold and decode concurrency below limit
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class DecodeResourceMeasure implements ResourceMeasure {
    private final long availableThreshold;
    private final long hysteresisBiasPercent;
    private final long fullSpeedThreshold;
    private final long stopThreshold;
    private final long concurrencyLimit;
    private final EndpointRegistry endpointRegistry;

    public DecodeResourceMeasure(ConfigService configService, EndpointRegistry endpointRegistry) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.availableThreshold = config.getDecodeAvailableMemoryThreshold();
        this.hysteresisBiasPercent = config.getHysteresisBiasPercent();
        this.fullSpeedThreshold = config.getDecodeFullSpeedThreshold();
        this.stopThreshold = config.getDecodeStopThreshold();
        this.concurrencyLimit = config.getDecodeConcurrencyLimit();
        this.endpointRegistry = endpointRegistry;
    }

    @Override
    public boolean isResourceAvailable(WorkerEndpoint endpoint) {
        if (endpoint instanceof DecodeEndpoint) {
            return isResourceAvailable((DecodeEndpoint) endpoint);
        }
        return ResourceMeasure.super.isResourceAvailable(endpoint);
    }

    public boolean isResourceAvailable(DecodeEndpoint endpoint) {
        if (endpoint == null || !endpoint.getStatus().isAlive()) {
            return false;
        }
        long totalLoad = endpoint.decodeTotalLoad();
        if (concurrencyLimit > 0 && totalLoad >= concurrencyLimit) {
            Logger.warn("Decode worker {} resource unavailable: totalLoad={}, limit={}",
                    endpoint.ipPort(), totalLoad, concurrencyLimit);
            return false;
        }
        long used = endpoint.decodeRealKvUsed();
        long total = endpoint.decodeKvTotal();
        if (total == 0) {
            endpoint.getStatus().getResourceAvailable().set(true);
            return true;
        }
        long usagePercentage = (long) ((used * 100.0) / total);
        boolean available = endpoint.getStatus().updateResourceAvailabilityWithHysteresis(usagePercentage, availableThreshold, hysteresisBiasPercent);
        if (!available) {
            Logger.warn("Decode worker {} resource unavailable: kvUsage={}%, threshold={}%, used={}, total={}",
                    endpoint.ipPort(), usagePercentage, availableThreshold, used, total);
        }
        return available;
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
    }

    @Override
    public double calculateAverageWaterLevel(Map<String, WorkerStatus> workerStatusMap) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return 0.0;
        }

        double totalWaterLevel = 0;
        int count = 0;

        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            double waterLevel = calculateWaterLevel(entry.getKey(), entry.getValue());
            totalWaterLevel += waterLevel;
            count++;
        }

        return count > 0 ? totalWaterLevel / count : 0.0;
    }

    /**
     * Water level for a single worker, preferring the {@link DecodeEndpoint} real*() views
     * (engine-reported state + local inflight reservations) over raw {@link WorkerStatus}
     * fields. The real view is inflight-aware, so the resulting water level is more
     * conservative (never lower) than the raw view. Thresholds are unchanged — only the
     * data source differs.
     *
     * <p>Falls back to raw WorkerStatus fields when no DecodeEndpoint is registered for
     * the key (e.g., this measure is configured for a non-decode role, or the endpoint
     * has not been created yet).
     */
    private double calculateWaterLevel(String ipPort, WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return 0.0;
        }

        DecodeEndpoint endpoint = endpointRegistry != null ? endpointRegistry.getDecode(ipPort) : null;
        if (endpoint != null) {
            return Math.max(
                    calculateKvCacheWaterLevel(endpoint.decodeRealKvUsed(), endpoint.decodeKvTotal()),
                    calculateConcurrencyWaterLevel(endpoint.decodeTotalLoad()));
        }

        long total = workerStatus.getTotalKvCacheTokens().get();
        long used = total - workerStatus.getAvailableKvCacheTokens().get();
        return Math.max(
                calculateKvCacheWaterLevel(used, total),
                calculateConcurrencyWaterLevel(calculateDecodeConcurrency(workerStatus)));
    }

    private double calculateKvCacheWaterLevel(long used, long total) {
        if (total == 0) {
            return 0.0;
        }

        double usedPercentage = (used * 100.0) / total;

        if (usedPercentage <= fullSpeedThreshold) {
            return 0.0;
        } else if (usedPercentage >= stopThreshold) {
            return 100.0;
        } else {
            return (usedPercentage - fullSpeedThreshold) /
                    (stopThreshold - fullSpeedThreshold) * 100.0;
        }
    }

    private double calculateConcurrencyWaterLevel(long currentConcurrency) {
        if (concurrencyLimit <= 0) {
            return 0.0;
        }

        if (currentConcurrency <= 0) {
            return 0.0;
        }
        return Math.min(100.0, currentConcurrency * 100.0 / concurrencyLimit);
    }

    private long calculateDecodeConcurrency(WorkerStatus workerStatus) {
        if (MapUtils.isNotEmpty(workerStatus.getRunningTaskList())) {
            return workerStatus.getRunningTaskList().size();
        }
        return 0;
    }
}
