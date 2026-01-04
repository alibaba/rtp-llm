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
 * Decode角色资源度量器
 * 判断标准: 剩余显存是否高于阈值
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
        long availableMemoryThreshold = config.getDecodeAvailableMemoryThreshold();

        // Decode角色: 检查剩余显存是否高于阈值
        return workerStatus.getAvailableKvCacheTokens().get() >= availableMemoryThreshold;
    }

    @Override
    public boolean hasResourceAvailableWorker(String modelName, String group) {

        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.DECODE, group);

        if (MapUtils.isEmpty(workerStatusMap)) {
            return false;
        }

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long availableMemoryThreshold = config.getDecodeAvailableMemoryThreshold();

        // 至少有一个worker的剩余显存高于阈值
        return workerStatusMap.values().stream()
                .anyMatch(ws -> ws.isAlive() && ws.getAvailableKvCacheTokens().get() >= availableMemoryThreshold);
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
    }
}
