package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Decode role resource measure
 * Availability criteria: KV cache usage percentage below threshold
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class DecodeResourceMeasure implements ResourceMeasure {
    private final ConfigService configService;

    public DecodeResourceMeasure(ConfigService configService) {
        this.configService = configService;
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        long used = workerStatus.getUsedKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long total = used + available;

        if (total == 0) {
            workerStatus.getResourceAvailable().set(true);
            return true;
        }

        FlexlbConfig config = configService.loadBalanceConfig();
        long usagePercentage = (long) ((used * 100.0) / total);
        return workerStatus.updateResourceAvailabilityWithHysteresis(
                usagePercentage, config.getDecodeAvailableMemoryThreshold(), config.getHysteresisBiasPercent());
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
            double waterLevel = calculateWorkerWaterLevel(worker);
            totalWaterLevel += waterLevel;
            count++;
        }

        return count > 0 ? totalWaterLevel / count : 0.0;
    }

    @Override
    public double calculateWorkerWaterLevel(WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return 0.0;
        }

        long used = workerStatus.getUsedKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long total = used + available;

        if (total == 0) {
            return 0.0;
        }

        double usedPercentage = (used * 100.0) / total;

        FlexlbConfig config = configService.loadBalanceConfig();
        long fullSpeedThreshold = config.getDecodeFullSpeedThreshold();
        long stopThreshold = config.getDecodeStopThreshold();

        if (usedPercentage <= fullSpeedThreshold) {
            return 0.0;
        } else if (usedPercentage >= stopThreshold) {
            return 100.0;
        } else {
            return (usedPercentage - fullSpeedThreshold) /
                    (stopThreshold - fullSpeedThreshold) * 100.0;
        }
    }
}
