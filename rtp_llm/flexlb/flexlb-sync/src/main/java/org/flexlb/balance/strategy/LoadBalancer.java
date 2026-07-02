package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

public interface LoadBalancer {

    ServerStatus select(BalanceContext context, RoleType roleType, String group);

    /**
     * Release local state associated with a previously-selected worker.
     *
     * @param ep        the endpoint to rollback (non-null)
     * @param requestId the request identifier
     */
    void rollBack(WorkerEndpoint ep, long requestId);
}
