package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;

import java.util.Collection;
import java.util.List;

/**
 * Worker status provider interface
 *
 * @author FlexLB
 */
public interface WorkerStatusProvider {

    /**
     * Get all worker IP:Port addresses
     *
     * @param roleType Worker role type to query
     * @param group    Worker group to query
     * @return Worker IP:Port list
     */
    default List<String> getWorkerIpPorts(RoleType roleType, String group) {
        return getWorkerStatuses(roleType, group).stream()
                .map(WorkerStatus::getIpPort)
                .toList();
    }

    /**
     * Get in-memory worker statuses for cache metadata and block-hash configuration.
     */
    Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group);
}
