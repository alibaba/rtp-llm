package org.flexlb.balance.strategy;

import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.BalanceContext;

public interface LoadBalancer {

    ServerStatus select(BalanceContext context, RoleType roleType, String group);

    void rollBack(String modelName, String ipPort, Long interRequestId);
}
