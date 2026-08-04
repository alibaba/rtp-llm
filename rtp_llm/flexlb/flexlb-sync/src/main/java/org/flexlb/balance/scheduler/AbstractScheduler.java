package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.FlexlbMetricHelper;

import java.util.concurrent.CompletableFuture;

/**
 * Base class for all scheduler implementations in the v2 redesign.
 *
 * <p>Owns the shared scheduling infrastructure:
 * <ul>
 *   <li>{@link InflightStore} registration via {@link #register} — atomic
 *       duplicate detection, active-count accounting, and terminal-state
 *       wiring on the result future</li>
 *   <li>{@link #cancel(String)} — CAS-guarded cancel through the store</li>
 *   <li>{@link #reportMetrics()} — periodic path-specific metrics hook</li>
 * </ul>
 *
 * <p>Subclasses implement {@link #submit(BalanceContext)} to route and
 * dispatch requests.
 */
public abstract class AbstractScheduler {

    protected final InflightStore globalStore;
    protected final FlexlbMetricHelper metricHelper;

    protected AbstractScheduler(InflightStore globalStore, FlexlbMetricHelper metricHelper) {
        this.globalStore = globalStore;
        this.metricHelper = metricHelper;
    }

    /**
     * Submit a request for scheduling and dispatch.
     *
     * @param ctx the request context carrying the {@link BalanceContext}
     * @return a future that will be completed with the routing {@link Response}
     */
    public abstract CompletableFuture<Response> submit(BalanceContext ctx);

    /**
     * Atomically register the request in the global {@link InflightStore}
     * and wire the terminal transition onto the given result future.
     *
     * <p>Uses {@link InflightStore#putIfAbsent} — duplicate request IDs
     * (active or tombstone within TTL) are rejected without a check-then-act
     * window. On successful insert the item's terminal state is driven by
     * the future's completion (success → {@link InflightItem#complete},
     * failure → {@link InflightItem#fail}); the item then remains in the
     * store as a tombstone until the TTL evictor removes it.
     *
     * @param ctx    the request context
     * @param future the result future whose completion drives the terminal state
     * @return {@code null} if the item was registered, or the existing item
     *         when the request ID is a duplicate
     */
    protected InflightItem register(BalanceContext ctx, CompletableFuture<Response> future) {
        InflightItem item = new InflightItem(ctx, future, this);
        item.setMetricHelper(metricHelper);
        InflightItem existing = globalStore.putIfAbsent(item.requestId(), item);
        if (existing == null) {
            future.whenComplete((response, throwable) -> {
                if (throwable != null) {
                    item.fail(throwable);
                } else if (response != null) {
                    item.complete(response);
                } else {
                    item.fail(new IllegalStateException("null routing response"));
                }
            });
        }
        return existing;
    }

    /**
     * Cancel an inflight request by its string-form request ID.
     *
     * <p>Looks up the {@link InflightItem} in the global store and atomically
     * cancels it via CAS. Returns {@code false} if the request was not found
     * (already evicted or never tracked) or is already terminal.
     *
     * @param requestId string-form request ID (see {@link InflightItem#requestId()})
     * @return {@code true} if the request was found and cancelled
     */
    public boolean cancel(String requestId) {
        InflightItem item = globalStore.get(requestId);
        if (item == null) {
            return false;
        }
        return item.cancel();
    }

    /**
     * Error type delivered when a request tracked by this scheduler is timed
     * out by the {@link InflightStore} TTL safety net
     * ({@link InflightItem#timeoutWithError()}).
     *
     * <p>Default: {@link StrategyErrorType#INFLIGHT_TTL_EXPIRED}. The batch
     * path overrides this to keep its SLO-expiry error semantics.
     */
    protected StrategyErrorType ttlExpiryErrorType() {
        return StrategyErrorType.INFLIGHT_TTL_EXPIRED;
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
