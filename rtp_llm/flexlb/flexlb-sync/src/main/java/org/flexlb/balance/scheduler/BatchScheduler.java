package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Thin-wrapper scheduler for BATCH mode.
 *
 * <p>Delegates request submission to the existing {@link FlexlbBatchScheduler}
 * while overlaying an {@link InflightItem} tracking layer for unified cancel
 * support. The InflightItem shares the same {@code CompletableFuture} as the
 * delegate, so CAS-guarded terminal state ensures cancel and normal
 * completion never conflict.
 *
 * <p>EP references are {@code null} in the InflightItem because the delegate
 * owns the full EP lifecycle (route, commit, rollback, release). The
 * InflightItem is purely for cancel lookup via {@link InflightStore}.
 *
 * <p>On terminal transition the item is NOT removed from the global store —
 * it remains as a tombstone (terminated=true + terminalReason) for late
 * cancel detection. The {@link InflightStore} TTL evictor cleans up
 * tombstones after the safety TTL.
 */
public class BatchScheduler extends AbstractScheduler {

    private final FlexlbBatchScheduler delegate;
    private final InflightStore globalStore;

    public BatchScheduler(FlexlbBatchScheduler delegate, InflightStore globalStore) {
        this.delegate = delegate;
        this.globalStore = globalStore;
    }

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = delegate.submit(ctx);
        ctx.setFuture(future);

        InflightItem item = new InflightItem(ctx, future, this);
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
}
