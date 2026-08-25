package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Prefill role resource measure
 * Availability criteria: effective pending task count below threshold
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class PrefillResourceMeasure implements ResourceMeasure {
    private final ConfigService configService;

    public PrefillResourceMeasure(ConfigService configService) {
        this.configService = configService;
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        return workerStatus.getInTransitAndWaitingTaskCount() < currentThreshold();
    }

    private long currentThreshold() {
        return configService.loadBalanceConfig().getPrefillQueueSizeThreshold();
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.WAIT_TIME;
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

        return waterLevel(workerStatus.getInTransitAndWaitingTaskCount(), currentThreshold());
    }

    private double waterLevel(long value, long threshold) {
        if (value <= 0) {
            return 0.0;
        }
        if (threshold <= 0 || value >= threshold) {
            return 100.0;
        }
        return (value * 100.0) / threshold;
    }

}
