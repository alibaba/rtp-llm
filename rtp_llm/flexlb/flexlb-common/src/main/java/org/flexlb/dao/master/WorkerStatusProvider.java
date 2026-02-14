package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;

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
    List<String> getWorkerIpPorts(RoleType roleType, String group);
}