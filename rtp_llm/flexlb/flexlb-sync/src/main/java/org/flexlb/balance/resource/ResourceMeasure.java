package org.flexlb.balance.resource;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
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
     * Check if at least one worker in the group has available resources
     *
     * @param roleType Worker role type
     * @param group    Worker group (null means no group restriction)
     * @return true if at least one worker is available, false otherwise
     */
    boolean hasResourceAvailableWorker(RoleType roleType, String group);

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
