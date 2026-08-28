package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
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
    private final long fullSpeedThreshold;
    private final long stopThreshold;
    private final long concurrencyLimit;

    public DecodeResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.availableThreshold = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxKvUsagePercent();
        this.fullSpeedThreshold = config.getInternalRuntime()
                .getDecodeFullSpeedBelowKvUsagePercent();
        this.stopThreshold = config.getInternalRuntime()
                .getDecodeSaturatedAtKvUsagePercent();
        Long configuredLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        this.concurrencyLimit = configuredLimit == null ? 0 : configuredLimit;
    }

    /** Pure availability decision over one caller-owned routing snapshot. */
    public boolean isResourceAvailable(
            DecodeEndpoint.DecodeRoutingView view) {
        if (view == null) {
            return false;
        }
        // The concurrency gate is Engine-facing: reservations parked in a
        // Prefill queue must not saturate it. The placement KV view below stays
        // conservative and still retains queued prompt/expected demand.
        return isAvailable(
                view.engineLoad(), view.realKvUsed(), view.totalKv());
    }

    /**
     * Transient delivery preference for non-preemptive queues. Prefill-queued
     * shadows are deliberately excluded; only Engine-facing ownership and
     * dispatch permits can make this tier unavailable.
     */
    public boolean isEngineDispatchAvailable(
            DecodeEndpoint.DecodeRoutingView view) {
        if (view == null) {
            return false;
        }
        return isAvailable(
                view.engineCapacityUsed(),
                view.engineFacingKvUsed(),
                view.totalKv());
    }

    private boolean isAvailable(long engineLoad, long used, long total) {
        if (concurrencyLimit > 0 && engineLoad >= concurrencyLimit) {
            return false;
        }
        if (total == 0) {
            return true;
        }
        double usagePercentage = used * 100.0 / total;
        return usagePercentage < availableThreshold;
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
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        return Math.max(calculateKvCacheWaterLevel(status),
                calculateConcurrencyWaterLevel(status));
    }

    private double calculateKvCacheWaterLevel(
            WorkerStatus.EngineObservation status) {
        long total = status.totalKvCacheTokens();
        long available = status.availableKvCacheTokens();
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

    private double calculateConcurrencyWaterLevel(
            WorkerStatus.EngineObservation status) {
        if (concurrencyLimit <= 0) {
            return 0.0;
        }

        long currentConcurrency = calculateDecodeConcurrency(status);
        if (currentConcurrency <= 0) {
            return 0.0;
        }
        return Math.min(100.0, currentConcurrency * 100.0 / concurrencyLimit);
    }

    private long calculateDecodeConcurrency(
            WorkerStatus.EngineObservation status) {
        if (MapUtils.isNotEmpty(status.runningTaskList())) {
            return status.runningTaskList().size();
        }
        return 0;
    }
}
