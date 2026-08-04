package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Thin-wrapper scheduler for BATCH mode.
 *
 * <p>Delegates request submission to the existing {@link FlexlbBatchScheduler},
 * which registers the {@link InflightItem} in the global {@link InflightStore}
 * atomically inside {@code submit()}. This wrapper wires the metric helper and
 * the terminal transition onto that item. The InflightItem shares the same
 * {@code CompletableFuture} as the delegate, so CAS-guarded terminal state
 * ensures cancel and normal completion never conflict.
 *
 * <p>EP references are {@code null} in the InflightItem because the delegate
 * owns the full EP lifecycle (route, commit, rollback, release). The
 * InflightItem is purely for cancel lookup via {@link InflightStore}.
 *
 * <p>On terminal transition the item is NOT removed from the global store —
 * it remains as a tombstone (terminal {@link InflightState}) for late
 * cancel detection. The {@link InflightStore} TTL evictor cleans up
 * tombstones after the safety TTL.
 */
public class BatchScheduler extends AbstractScheduler {

    private final FlexlbBatchScheduler delegate;
    private final InflightStore globalStore;
    private final FlexlbMetricHelper metricHelper;

    public BatchScheduler(FlexlbBatchScheduler delegate, InflightStore globalStore,
                          FlexlbMetricHelper metricHelper) {
        this.delegate = delegate;
        this.globalStore = globalStore;
        this.metricHelper = metricHelper;
    }

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = delegate.submit(ctx);
        ctx.setFuture(future);

        // The delegate registers the InflightItem atomically inside submit()
        // (InflightStore.putIfAbsent) — no separate registration here. Retrieve
        // it for metric wiring and terminal transition; the future identity
        // guard skips items owned by an earlier submit (duplicate request ID).
        InflightItem item = globalStore.get(String.valueOf(ctx.getRequestId()));
        if (item != null && item.future() == future) {
            item.setMetricHelper(metricHelper);
            future.whenComplete((response, throwable) -> {
                if (throwable == null) {
                    item.complete(response);
                } else {
                    item.fail(throwable);
                }
            });
        }

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
     * Report BATCH-specific metrics: the delegate's internal inflight size
     * (requests currently in the BATCH dispatch pipeline).
     */
    @Override
    public void reportMetrics() {
        metricHelper.reportInflightSize("PREFILL", "scheduler", delegate.getInflightSize());
    }
}
