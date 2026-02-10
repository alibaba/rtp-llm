package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
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
    private final EngineWorkerStatus engineWorkerStatus;

    public DecodeResourceMeasure(ConfigService configService, EngineWorkerStatus engineWorkerStatus) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long threshold = config.getDecodeAvailableMemoryThreshold();

        long used = workerStatus.getUsedKvCacheTokens().get();
        long available = workerStatus.getAvailableKvCacheTokens().get();
        long total = used + available;

        if (total == 0) {
            workerStatus.getResourceAvailable().set(true);
            return true;
        }

        long usagePercentage = (long) ((used * 100.0) / total);
        return workerStatus.updateResourceAvailabilityWithHysteresis(usagePercentage, threshold, config.getHysteresisBiasPercent());
    }

    @Override
    public boolean hasResourceAvailableWorker(RoleType roleType, String group) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long threshold = config.getDecodeAvailableMemoryThreshold();

        return workerStatusMap.values().stream()
                .anyMatch(ws -> {
                    if (!ws.isAlive()) {
                        return false;
                    }
                    long used = ws.getUsedKvCacheTokens().get();
                    long available = ws.getAvailableKvCacheTokens().get();
                    long total = used + available;
                    if (total == 0) {
                        ws.getResourceAvailable().set(true);
                        return true;
                    }
                    long usagePercentage = (long) ((used * 100.0) / total);
                    return ws.updateResourceAvailabilityWithHysteresis(usagePercentage, threshold, config.getHysteresisBiasPercent());
                });
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

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long fullSpeedThreshold = config.getDecodeFullSpeedThreshold();
        long stopThreshold = config.getDecodeStopThreshold();

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
}