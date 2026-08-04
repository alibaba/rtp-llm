package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Thin-wrapper scheduler for QUEUE mode.
 *
 * <p>Delegates request submission to the existing {@link QueueManager} while
 * overlaying an {@link InflightItem} tracking layer for unified cancel
 * support. The future used for tracking is the {@code Mono.toFuture()} result
 * (which includes timeout and error-handling from the reactive pipeline),
 * ensuring the InflightItem's terminal state fires even on queue timeout.
 *
 * <p>Note: {@code ctx.getFuture()} is set inside {@link QueueManager#tryRouteAsync}
 * to the raw CompletableFuture used by worker threads. The InflightItem uses
 * the Mono-derived future so that timeout-driven terminal transitions are
 * captured. The two futures share the same underlying completion — the worker
 * completes the raw future, which propagates through the Mono pipeline.
 *
 * <p>On terminal transition the item is NOT removed from the global store —
 * it remains as a tombstone for late cancel detection. The {@link InflightStore}
 * TTL evictor cleans up tombstones after the safety TTL.
 */
public class QueueScheduler extends AbstractScheduler {

    private final QueueManager delegate;
    private final InflightStore globalStore;
    private final FlexlbMetricHelper metricHelper;

    public QueueScheduler(QueueManager delegate, InflightStore globalStore,
                          FlexlbMetricHelper metricHelper) {
        this.delegate = delegate;
        this.globalStore = globalStore;
        this.metricHelper = metricHelper;
    }

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = delegate.tryRouteAsync(ctx).toFuture();

        InflightItem item = new InflightItem(ctx, future, this);
        item.setMetricHelper(metricHelper);
        globalStore.put(item.requestId(), item);

        future.whenComplete((response, throwable) -> {
            if (throwable == null) {
                item.complete(response);
            } else {
                item.fail(throwable);
            }
        });

        return future;
    }

    @Override
    public boolean cancel(String requestId) {
        InflightItem item = globalStore.get(requestId);
        if (item == null) {
            return false;
        }
        return item.cancel();
    }

    /**
     * Report QUEUE-specific metrics: the current queue length (requests
     * waiting in the routing queue for a worker to become available).
     */
    @Override
    public void reportMetrics() {
        metricHelper.reportInflightSize("PREFILL", "scheduler", delegate.queueSize());
    }
}
