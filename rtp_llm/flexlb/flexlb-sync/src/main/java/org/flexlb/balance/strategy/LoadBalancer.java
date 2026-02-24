package org.flexlb.balance.strategy;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

public interface LoadBalancer {

    ServerStatus select(BalanceContext context, RoleType roleType, String group);

    void rollBack(String ipPort, long interRequestId);
}
