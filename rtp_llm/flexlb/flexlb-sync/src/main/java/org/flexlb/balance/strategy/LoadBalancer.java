package org.flexlb.balance.strategy;

import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;

/**
 * @author zjw
 * description:
 * date: 2025/3/12
 */
public interface LoadBalancer {

    ServerStatus select(BalanceContext context, RoleType roleType, String group);

    boolean releaseLocalCache(String modelName, String ip, Long interRequestId);
}
