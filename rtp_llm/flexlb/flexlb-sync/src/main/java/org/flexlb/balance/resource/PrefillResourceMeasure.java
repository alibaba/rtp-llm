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
    private final long maxBatchTokens;
    private final int waitingUncachedTokenBatchCount;

    public PrefillResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.queueSizeThreshold = config.getPrefillQueueSizeThreshold();
        this.maxBatchTokens = config.getPrefillMaxBatchTokens();
        this.waitingUncachedTokenBatchCount = config.getPrefillWaitingUncachedTokenBatchCount();
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        if (workerStatus.getInTransitAndWaitingTaskCount() >= queueSizeThreshold) {
            return false;
        }
        return !isWaitingUncachedTokenLimitEnabled()
                || workerStatus.getInTransitAndWaitingUncachedTokens() < waitingUncachedTokenThreshold();
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

        double queueWaterLevel = waterLevel(workerStatus.getInTransitAndWaitingTaskCount(), queueSizeThreshold);
        if (!isWaitingUncachedTokenLimitEnabled()) {
            return queueWaterLevel;
        }
        return Math.max(queueWaterLevel,
                waterLevel(workerStatus.getInTransitAndWaitingUncachedTokens(), waitingUncachedTokenThreshold()));
    }

    private long waitingUncachedTokenThreshold() {
        return maxBatchTokens * waitingUncachedTokenBatchCount;
    }

    private boolean isWaitingUncachedTokenLimitEnabled() {
        return maxBatchTokens > 0 && waitingUncachedTokenBatchCount > 0;
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
