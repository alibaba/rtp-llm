package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;

/**
 * Router interface - responsible for selecting appropriate worker nodes based on load balancing context.
 * <p>
 * This interface defines the core routing logic for the load balancing scheduler,
 * selecting optimal worker nodes for request processing based on current load conditions,
 * cache status, and scheduling strategies.
 * </p>
 *
 * @author saichen.sm
 * @since 1.0
 */
public interface Router {

    /**
     * Route requests based on load balancing strategy and select appropriate worker nodes.
     *
     * @param balanceContext Load balancing context containing request information and available worker list
     * @return RouteResult containing direct endpoint references and selected server statuses
     */
    RouteResult route(BalanceContext balanceContext);

}
