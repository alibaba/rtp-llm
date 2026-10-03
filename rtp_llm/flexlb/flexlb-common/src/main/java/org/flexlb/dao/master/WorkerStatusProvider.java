package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;

import java.util.Collection;

/**
 * Worker status provider interface
 *
 * @author FlexLB
 */
public interface WorkerStatusProvider {

    /**
     * Get in-memory worker statuses for a role and group.
     */
    Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group);
}
