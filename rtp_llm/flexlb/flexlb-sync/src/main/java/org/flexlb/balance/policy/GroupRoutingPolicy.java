package org.flexlb.balance.policy;

import org.flexlb.dao.BalanceContext;

public interface GroupRoutingPolicy {

    GroupRoutingDecision route(BalanceContext balanceContext);
}
