package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Base class for all scheduler implementations in the v2 redesign.
 *
 * <p>Subclasses implement {@link #submit(BalanceContext)} to route and
 * dispatch requests. The default implementations of {@link #cancel(String)}
 * and {@link #removeBatchInflight(String)} are no-ops, suitable for
 * schedulers that do not track inflight state (e.g. a pure direct
 * scheduler).
 */
public abstract class AbstractScheduler {

    /**
     * Submit a request for scheduling and dispatch.
     *
     * @param ctx the request context carrying the {@link BalanceContext}
     * @return a future that will be completed with the routing {@link Response}
     */
    public abstract CompletableFuture<Response> submit(BalanceContext ctx);

    /**
     * Cancel an inflight request by its string-form request ID.
     *
     * <p>Default implementation returns {@code false} (no inflight tracking).
     * Subclasses that maintain an {@link InflightStore} override this to
     * look up and terminate the item.
     *
     * @param requestId string-form request ID (see {@link InflightItem#requestId()})
     * @return {@code true} if the request was found and cancelled
     */
    public boolean cancel(String requestId) {
        return false;
    }

    /**
     * Remove a batch inflight entry after all items have reached terminal state.
     *
     * <p>Default implementation is a no-op. Subclasses that track batch
     * inflight state override this to clean up.
     *
     * @param batchId string-form batch ID
     */
    public void removeBatchInflight(String batchId) {
        // default: no batch inflight tracking
    }

    /**
     * Report path-specific metrics for this scheduler.
     *
     * <p>Default implementation is a no-op. Subclasses override this to
     * report their own metrics (e.g. inflight size, queue length).
     * Called periodically by {@code RouteService.triggerSchedulerMetrics()}.
     */
    public void reportMetrics() {
        // default: no scheduler-specific metrics
    }
}
