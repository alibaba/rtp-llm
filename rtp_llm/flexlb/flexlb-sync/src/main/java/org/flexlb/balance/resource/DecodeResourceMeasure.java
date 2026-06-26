package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
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

    public DecodeResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.availableThreshold = config.getDecodeAvailableMemoryThreshold();
        this.hysteresisBiasPercent = config.getHysteresisBiasPercent();
        this.fullSpeedThreshold = config.getDecodeFullSpeedThreshold();
        this.stopThreshold = config.getDecodeStopThreshold();
        this.concurrencyLimit = config.getDecodeConcurrencyLimit();
    }

    public boolean isResourceAvailable(DecodeEndpoint endpoint) {
        if (endpoint == null || !endpoint.getStatus().isAlive()) {
            return false;
        }
        long concurrency = calculateDecodeConcurrency(endpoint.getStatus());
        if (concurrencyLimit > 0 && concurrency >= concurrencyLimit) {
            Logger.warn("Decode worker {} resource unavailable: concurrency={}, limit={}",
                    endpoint.getIp(), concurrency, concurrencyLimit);
            return false;
        }
        long used = endpoint.realKvUsed();
        long total = endpoint.realKvTotal();
        if (total == 0) {
            endpoint.getStatus().getResourceAvailable().set(true);
            return true;
        }
        long usagePercentage = (long) ((used * 100.0) / total);
        boolean available = endpoint.getStatus().updateResourceAvailabilityWithHysteresis(usagePercentage, availableThreshold, hysteresisBiasPercent);
        if (!available) {
            Logger.warn("Decode worker {} resource unavailable: kvUsage={}%, threshold={}%, used={}, total={}",
                    endpoint.getIp(), usagePercentage, availableThreshold, used, total);
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

        for (WorkerStatus worker : workerStatusMap.values()) {
            double waterLevel = calculateWaterLevel(worker);
            totalWaterLevel += waterLevel;
            count++;
        }

        return count > 0 ? totalWaterLevel / count : 0.0;
    }

    private double calculateWaterLevel(WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return 0.0;
        }

        return Math.max(calculateKvCacheWaterLevel(workerStatus), calculateConcurrencyWaterLevel(workerStatus));
    }

    private double calculateKvCacheWaterLevel(WorkerStatus workerStatus) {
        long total = workerStatus.getTotalKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long used = total - available;

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

    private double calculateConcurrencyWaterLevel(WorkerStatus workerStatus) {
        if (concurrencyLimit <= 0) {
            return 0.0;
        }

        long currentConcurrency = calculateDecodeConcurrency(workerStatus);
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
