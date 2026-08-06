package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.FlexlbMetricHelper;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Base class for all scheduler implementations in the v2 redesign.
 *
 * <p>Owns the shared scheduling infrastructure:
 * <ul>
 *   <li>{@link InflightStore} registration via {@link #register} — atomic
 *       duplicate detection, active-count accounting, and terminal-state
 *       wiring on the result future</li>
 *   <li>{@link #onCancel(InflightItem)} — protected hook for path-specific
 *       cancel cleanup, driven from {@code RouteService}</li>
 *   <li>{@link #reportMetrics()} — periodic path-specific metrics hook</li>
 * </ul>
 *
 * <p>Subclasses implement {@link #submit(BalanceContext)} to route and
 * dispatch requests.
 */
public abstract class AbstractScheduler implements DiagnosticsProvider {

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
     * Hook invoked after a cancel wins the terminal CAS, letting the owning
     * scheduler release path-specific resources (e.g. a queue slot) with
     * best-effort semantics — the request may already have left the
     * scheduler's structures.
     *
     * <p>Protected: the cancel cascade is driven from {@code RouteService}
     * via {@link InflightItem#fireOnCancel()}, which is in the same package
     * and can access this protected method. Subclasses override to add
     * path-specific cleanup (e.g. {@link QueueScheduler} removes the queued
     * request).
     *
     * <p>Default implementation is a no-op.
     */
    protected void onCancel(InflightItem item) {
        // default: no path-specific cancel cleanup
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

    /**
     * Start any background resources owned by this scheduler (e.g. worker
     * pool, queue consumer). Called once at startup by
     * {@code RouteService.start()}.
     *
     * <p>Default implementation is a no-op. Subclasses with background
     * resources override this (e.g. {@link QueueScheduler} starts its
     * {@code QueueingComponent} worker pool).
     */
    public void start() {
        // default: no background resources to start
    }

    /**
     * Shut down any background resources owned by this scheduler.
     * Called once at shutdown by {@code RouteService.shutdown()}.
     *
     * <p>Default implementation is a no-op. Subclasses with background
     * resources override this (e.g. {@link QueueScheduler} shuts down its
     * {@code QueueingComponent} worker pool).
     */
    public void shutdown() {
        // default: no background resources to shut down
    }

    /**
     * {@inheritDoc}
     *
     * <p>Default implementation returns an empty map. Subclasses with
     * diagnostics to report (e.g. queue length) override this.
     */
    @Override
    public Map<String, Object> getDiagnostics() {
        return Map.of();
    }
}
