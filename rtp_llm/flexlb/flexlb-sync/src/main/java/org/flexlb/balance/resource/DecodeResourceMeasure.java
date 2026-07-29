package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

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

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        if (isConcurrencyLimitReached(workerStatus)) {
            return false;
        }

        long used = workerStatus.getUsedKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long total = used + available;

        if (total == 0) {
            workerStatus.getResourceAvailable().set(true);
            return true;
        }

        long usagePercentage = (long) ((used * 100.0) / total);
        return workerStatus.updateResourceAvailabilityWithHysteresis(usagePercentage, availableThreshold, hysteresisBiasPercent);
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
        long used = workerStatus.getUsedKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long total = used + available;

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

    private boolean isConcurrencyLimitReached(WorkerStatus workerStatus) {
        return concurrencyLimit > 0 && calculateDecodeConcurrency(workerStatus) >= concurrencyLimit;
    }

    private long calculateDecodeConcurrency(WorkerStatus workerStatus) {
        Set<String> requestIds = new HashSet<>();
        if (MapUtils.isNotEmpty(workerStatus.getWaitingTaskList())) {
            requestIds.addAll(workerStatus.getWaitingTaskList().keySet());
        }
        if (MapUtils.isNotEmpty(workerStatus.getRunningTaskList())) {
            requestIds.addAll(workerStatus.getRunningTaskList().keySet());
        }
        if (MapUtils.isNotEmpty(workerStatus.getLocalTaskMap())) {
            workerStatus.getLocalTaskMap().keySet().stream()
                    .map(String::valueOf)
                    .forEach(requestIds::add);
        }
        return requestIds.size();
    }
}
