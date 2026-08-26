package org.flexlb.balance.resource;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import java.util.Map;

/**
 * Resource telemetry contract shared by role-specific measures.
 *
 * <p>Routing availability is intentionally exposed only by the concrete role
 * measure because Prefill and Decode consume different immutable snapshot
 * types. This interface owns no cross-request availability state.
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
public interface ResourceMeasure {

    /**
     * Get resource evaluation indicator
     *
     * @return Resource measure indicator
     */
    ResourceMeasureIndicatorEnum getResourceMeasureIndicator();

    /**
     * Calculate average water level for the role (0-100)
     *
     * @param workerStatusMap Worker status map
     * @return Global average water level percentage (0-100)
     */
    double calculateAverageWaterLevel(Map<String, WorkerStatus> workerStatusMap);
}
