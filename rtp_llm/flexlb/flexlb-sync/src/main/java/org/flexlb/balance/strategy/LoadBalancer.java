package org.flexlb.balance.strategy;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import reactor.core.publisher.Mono;

public interface LoadBalancer {

    Mono<ServerStatus> select(BalanceContext context, RoleType roleType, String group);

    void rollBack(String ipPort, String requestId);
}
