package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Prefill role resource measure
 * Availability criteria: queue wait time below threshold
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class PrefillResourceMeasure implements ResourceMeasure {
    private final long queueSizeThreshold;
    private final long hysteresisBiasPercent;
    private final long maxQueueSize;

    public PrefillResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.queueSizeThreshold = config.getPrefillQueueSizeThreshold();
        this.hysteresisBiasPercent = config.getHysteresisBiasPercent();
        this.maxQueueSize = config.getMaxPrefillQueueSize();
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        long queueSize = workerStatus.getWaitingTaskList() == null ? 0 : workerStatus.getWaitingTaskList().size();
        return workerStatus.updateResourceAvailabilityWithHysteresis(queueSize, queueSizeThreshold, hysteresisBiasPercent);
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

        long queueSize = workerStatus.getWaitingTaskList() == null ? 0 : workerStatus.getWaitingTaskList().size();

        if (queueSize <= 0) {
            return 0.0;
        } else if (queueSize >= maxQueueSize) {
            return 100.0;
        } else {
            return (queueSize * 100.0) / maxQueueSize;
        }
    }
}