package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.TaskPhase;
import org.springframework.stereotype.Component;

import org.flexlb.util.Logger;

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
    public boolean isResourceAvailable(WorkerEndpoint endpoint) {
        if (endpoint instanceof PrefillEndpoint) {
            return isResourceAvailable((PrefillEndpoint) endpoint);
        }
        return ResourceMeasure.super.isResourceAvailable(endpoint);
    }

    public boolean isResourceAvailable(PrefillEndpoint endpoint) {
        if (endpoint == null || !endpoint.getStatus().isAlive()) {
            return false;
        }
        long queueSize = endpoint.realPendingCount();
        boolean available = endpoint.getStatus().updateResourceAvailabilityWithHysteresis(queueSize, queueSizeThreshold, hysteresisBiasPercent);
        if (!available) {
            Logger.warn("Prefill worker {} resource unavailable: queueSize={}, threshold={}, alive={}",
                    endpoint.getIp(), queueSize, queueSizeThreshold, endpoint.getStatus().isAlive());
        }
        return available;
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

        long queueSize = countWaitingTasks(workerStatus);

        if (queueSize <= 0) {
            return 0.0;
        } else if (queueSize >= maxQueueSize) {
            return 100.0;
        } else {
            return (queueSize * 100.0) / maxQueueSize;
        }
    }

    private static long countWaitingTasks(WorkerStatus workerStatus) {
        if (MapUtils.isEmpty(workerStatus.getRunningTaskList())) {
            return 0;
        }
        return workerStatus.getRunningTaskList().values().stream()
                .filter(t -> t.getPhase() != TaskPhase.RUNNING).count();
    }

}