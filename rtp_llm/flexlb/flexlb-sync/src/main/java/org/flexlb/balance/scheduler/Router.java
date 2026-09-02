package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

/** Selects endpoint generations without owning request lifecycle state. */
public interface Router {

    /** Select and immediately publish a direct-routing response. */
    Response routeDirect(BalanceContext balanceContext);

    /** Prepare queue ownership or return the resource that currently blocks it. */
    QueueRoutingResult routeForQueue(BalanceContext balanceContext);
}
