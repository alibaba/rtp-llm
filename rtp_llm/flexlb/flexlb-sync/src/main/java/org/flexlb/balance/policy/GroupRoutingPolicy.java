package org.flexlb.balance.policy;

import org.flexlb.dao.BalanceContext;

/** Resolves one immutable routing group decision from request metadata. */
@FunctionalInterface
public interface GroupRoutingPolicy {

    GroupRoutingDecision route(BalanceContext balanceContext);
}
