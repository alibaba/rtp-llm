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
        long queueTimeThreshold = config.getPrefillQueueTimeThresholdMs();

        // Prefill角色: 检查排队时间是否低于阈值
        return workerStatus.getRunningQueueTime().get() <= queueTimeThreshold;
    }

    @Override
    public boolean hasResourceAvailableWorker(String modelName, String group) {

        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.PREFILL, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long queueTimeThreshold = config.getPrefillQueueTimeThresholdMs();

        // 至少有一个worker的排队时间低于阈值
        return workerStatusMap.values().stream()
                .anyMatch(ws -> ws.isAlive() && ws.getRunningQueueTime().get() <= queueTimeThreshold);
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.WAIT_TIME;
    }
}
