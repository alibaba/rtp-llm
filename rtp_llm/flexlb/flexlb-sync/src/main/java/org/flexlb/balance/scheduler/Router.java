package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import reactor.core.publisher.Mono;

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
     * Routes a request according to the configured load-balancing strategies.
     *
     * @param balanceContext load-balancing context containing request information and available workers
     * @return a publisher that emits the selected workers or a routing error response
     */
    Mono<Response> route(BalanceContext balanceContext);

    /**
     * Releases worker reservations represented by a route response that can no longer be used.
     *
     * @param balanceContext load-balancing context for the route
     * @param response successful response whose selected workers must be released
     */
    default void rollBack(BalanceContext balanceContext, Response response) {
    }

}
