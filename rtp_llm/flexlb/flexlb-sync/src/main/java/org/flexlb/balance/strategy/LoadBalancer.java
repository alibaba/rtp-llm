package org.flexlb.balance.strategy;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

public interface LoadBalancer {

    ServerStatus select(BalanceContext context, RoleType roleType, String group);

    /**
     * Rolls back request-local accounting for one logical worker.
     *
     * @param ipPort logical worker identity in {@code ip:port@engineIndex} format; the index
     *               identifies one independently routable engine behind the physical frontend
     * @param requestId request whose local accounting must be rolled back
     */
    void rollBack(String ipPort, String requestId);
}
