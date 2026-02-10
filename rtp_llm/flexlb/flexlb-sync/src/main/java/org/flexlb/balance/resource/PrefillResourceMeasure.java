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
 * Prefill角色资源度量器
 * 判断标准: 排队时间是否低于阈值
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class PrefillResourceMeasure implements ResourceMeasure {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;

    public PrefillResourceMeasure(ConfigService configService, EngineWorkerStatus engineWorkerStatus) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
    }

    @Override
    public boolean isResourceAvailable(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long threshold = config.getPrefillQueueSizeThreshold();

        long queueSize = workerStatus.getWaitingTaskList() == null ? 0 : workerStatus.getWaitingTaskList().size();
        return workerStatus.updateResourceAvailabilityWithHysteresis(queueSize, threshold, config.getHysteresisBiasPercent());
    }

    @Override
    public boolean hasResourceAvailableWorker(RoleType roleType, String group) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long threshold = config.getPrefillQueueSizeThreshold();
        long hysteresisBias = config.getHysteresisBiasPercent();

        return workerStatusMap.values().stream()
                .anyMatch(ws -> {
                    if (!ws.isAlive()) {
                        return false;
                    }
                    long queueSize = ws.getWaitingTaskList() == null ? 0 : ws.getWaitingTaskList().size();
                    return ws.updateResourceAvailabilityWithHysteresis(queueSize, threshold, hysteresisBias);
                });
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

        WhaleMasterConfig config = configService.loadBalanceConfig();
        int maxQueueSize = config.getMaxPrefillQueueSize();

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