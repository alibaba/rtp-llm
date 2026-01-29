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
 * 判断标准: KV cache使用率是否低于阈值(百分比)
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

        // Decode角色: 检查KV cache使用率是否低于阈值
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
    public boolean hasResourceAvailableWorker(String modelName, RoleType roleType, String group) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(modelName, roleType, group);

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
}
