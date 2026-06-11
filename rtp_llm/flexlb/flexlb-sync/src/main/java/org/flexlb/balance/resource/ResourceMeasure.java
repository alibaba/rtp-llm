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
     * Check if specified worker has available resources
     *
     * @param workerStatus Individual worker status
     * @return true if worker has available resources, false otherwise
     */
    boolean isResourceAvailable(WorkerStatus workerStatus);

    /**
     * Check if specified endpoint has available resources.
     * Default implementation unwraps the internal {@link WorkerStatus}.
     */
    default boolean isResourceAvailable(WorkerEndpoint endpoint) {
        if (endpoint == null) {
            return false;
        }
        return isResourceAvailable(endpoint.getStatus());
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
