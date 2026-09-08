package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import java.util.Map;

/**
 * Resource measure interface
 * Uses different resource availability logic based on RoleType
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
public interface ResourceMeasure {

    /**
     * Check if specified endpoint has available resources.
     * Concrete implementations should override with type-specific logic.
     */
    default boolean isResourceAvailable(WorkerEndpoint endpoint) {
        if (endpoint == null) {
            return false;
        }
        return endpoint.getStatus().isAlive();
    }

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
