package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
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
    private final long queueSizeThreshold;

    public PrefillResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.queueSizeThreshold = config.getPrefillQueueSizeThreshold();
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        long queueSize = effectiveQueueSize(workerStatus);
        return queueSize < queueSizeThreshold;
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

        long queueSize = effectiveQueueSize(workerStatus);

        if (queueSize <= 0) {
            return 0.0;
        } else if (queueSize >= queueSizeThreshold) {
            return 100.0;
        } else {
            return (queueSize * 100.0) / queueSizeThreshold;
        }
    }

    private long effectiveQueueSize(WorkerStatus workerStatus) {
        long engineWaitingTaskCount = workerStatus.getWaitingTaskList() == null
                ? 0 : workerStatus.getWaitingTaskList().size();
        long localOutstandingTaskCount = workerStatus.getLocalTaskMap() == null
                ? 0 : workerStatus.getLocalTaskMap().size();

        // The two collections overlap, so use the larger count instead of double-counting tasks.
        // Local tasks are recorded immediately and cover gaps between engine status snapshots.
        return Math.max(engineWaitingTaskCount, localOutstandingTaskCount);
    }
}
