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

        // Prefill role: check if queue size is below threshold (with hysteresis)
        long queueSize = workerStatus.getWaitingTaskList() == null ? 0 : workerStatus.getWaitingTaskList().size();
        return workerStatus.updateResourceAvailabilityWithHysteresis(queueSize, threshold, config.getHysteresisBiasPercent());
    }

    @Override
    public boolean hasResourceAvailableWorker(String modelName, RoleType roleType, String group) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long threshold = config.getPrefillQueueSizeThreshold();

        // At least one worker with queue size below threshold (with hysteresis)
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
}
