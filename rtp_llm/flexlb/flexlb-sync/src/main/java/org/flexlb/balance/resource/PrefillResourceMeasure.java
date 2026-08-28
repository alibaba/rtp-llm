package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.TaskPhase;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Prefill availability derived from coherent pending-request ownership.
 *
 * <p>The hard gate is the configured pending request count, not an estimated
 * millisecond duration.
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class PrefillResourceMeasure implements ResourceMeasure {
    private final long maxPendingRequests;
    private final long prefillSaturatedAtPendingRequests;

    public PrefillResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.maxPendingRequests = config.getRouter().getRoles().getPrefill()
                .getAvailability().getMaxPendingRequests();
        this.prefillSaturatedAtPendingRequests = config.getInternalRuntime()
                .getPrefillSaturatedAtPendingRequests();
    }

    /**
     * Evaluate availability from one caller-owned canonical pending-count
     * snapshot. The configured hysteresis remains loadable for configuration
     * compatibility, but routing readers do not persist availability state.
     */
    public boolean isResourceAvailable(long pendingRequests) {
        if (pendingRequests < 0L) {
            throw new IllegalArgumentException("pendingRequests must be non-negative");
        }
        return pendingRequests < maxPendingRequests;
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.PREFILL_PENDING_REQUESTS;
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

        long pendingRequests = countWaitingTasks(workerStatus);

        if (pendingRequests <= 0) {
            return 0.0;
        } else if (pendingRequests >= prefillSaturatedAtPendingRequests) {
            return 100.0;
        } else {
            return (pendingRequests * 100.0) / prefillSaturatedAtPendingRequests;
        }
    }

    private static long countWaitingTasks(WorkerStatus workerStatus) {
        Map<String, WorkerStatus.TaskObservation> runningTasks =
                workerStatus.committedEngineObservation().runningTaskList();
        if (MapUtils.isEmpty(runningTasks)) {
            return 0;
        }
        return runningTasks.values().stream()
                .filter(t -> t.phase() != TaskPhase.RUNNING).count();
    }

}
